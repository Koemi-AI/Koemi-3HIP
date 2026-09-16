from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from numbers import Integral


__all__ = ["BatchingMetrics", "BatchingMode", "MicrobatchPlan"]


def _validate_integer(value: object, name: str, minimum: int | None = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer_value = int(value)
    if minimum is not None and integer_value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer_value


@dataclass(frozen=True)
class MicrobatchPlan:
    """Describe one immutable microbatch without materializing its samples.

    `sample_indices` identifies the original dataset positions in execution order.
    `max_length` is the deterministic padding length for the microbatch.
    `max_tokens` is enforced against `padded_token_count`, not only real tokens.
    `is_optimizer_step_boundary` is true for every complete accumulation group and
    for the final partial group.
    """

    sample_indices: tuple[int, ...]
    sample_lengths: tuple[int, ...]
    max_length: int
    real_token_count: int
    padded_token_count: int
    padding_token_count: int
    microbatch_index: int
    optimizer_step_index: int
    accumulation_position: int
    is_optimizer_step_boundary: bool

    def __post_init__(self) -> None:
        if not isinstance(self.sample_indices, tuple) or not isinstance(self.sample_lengths, tuple):
            raise TypeError("sample_indices and sample_lengths must be tuples")
        if not self.sample_indices:
            raise ValueError("a microbatch must contain at least one sample")
        if len(self.sample_indices) != len(self.sample_lengths):
            raise ValueError("sample_indices and sample_lengths must have the same shape")
        validated_indices = tuple(
            _validate_integer(index, f"sample_indices[{position}]")
            for position, index in enumerate(self.sample_indices)
        )
        if len(set(validated_indices)) != len(validated_indices):
            raise ValueError("sample_indices must be unique within a microbatch")
        if any(index < 0 for index in validated_indices):
            raise ValueError("sample_indices must be non-negative")
        validated_lengths = tuple(
            _validate_integer(length, f"sample_lengths[{position}]", minimum=1)
            for position, length in enumerate(self.sample_lengths)
        )
        if self.max_length != max(validated_lengths):
            raise ValueError("max_length must equal the longest sample length")
        real_token_count = sum(validated_lengths)
        padded_token_count = len(validated_lengths) * self.max_length
        if self.real_token_count != real_token_count:
            raise ValueError("real_token_count does not match sample_lengths")
        if self.padded_token_count != padded_token_count:
            raise ValueError("padded_token_count does not match sample_lengths and max_length")
        if self.padding_token_count != padded_token_count - real_token_count:
            raise ValueError("padding_token_count does not match the planned padding")
        _validate_integer(self.max_length, "max_length", minimum=1)
        _validate_integer(self.real_token_count, "real_token_count", minimum=1)
        _validate_integer(self.padded_token_count, "padded_token_count", minimum=1)
        _validate_integer(self.padding_token_count, "padding_token_count", minimum=0)
        _validate_integer(self.microbatch_index, "microbatch_index", minimum=0)
        _validate_integer(self.optimizer_step_index, "optimizer_step_index", minimum=0)
        _validate_integer(self.accumulation_position, "accumulation_position", minimum=1)
        if not isinstance(self.is_optimizer_step_boundary, bool):
            raise TypeError("is_optimizer_step_boundary must be a boolean")

    @property
    def batch_size(self) -> int:
        return len(self.sample_indices)


@dataclass(frozen=True)
class BatchingMetrics:
    """Aggregate integer token and optimizer-step metrics for one plan."""

    sample_count: int
    microbatch_count: int
    real_token_count: int
    padded_token_count: int
    padding_token_count: int
    optimizer_step_count: int

    def __post_init__(self) -> None:
        metric_names = (
            "sample_count",
            "microbatch_count",
            "real_token_count",
            "padded_token_count",
            "padding_token_count",
            "optimizer_step_count",
        )
        for name in metric_names:
            _validate_integer(getattr(self, name), name, minimum=0)
        if self.padding_token_count != self.padded_token_count - self.real_token_count:
            raise ValueError("padding_token_count does not match aggregate token counts")
        if self.microbatch_count == 0 and self.optimizer_step_count != 0:
            raise ValueError("an empty plan cannot contain optimizer steps")
        if self.microbatch_count > 0 and self.optimizer_step_count == 0:
            raise ValueError("a non-empty plan must contain an optimizer step")

    @property
    def padding_fraction(self) -> float:
        if self.padded_token_count == 0:
            return 0.0
        return self.padding_token_count / self.padded_token_count


class BatchingMode:
    """Create deterministic, length-aware microbatch plans for a future trainer.

    `sample_lengths` is a one-dimensional iterable of positive sequence lengths.
    `max_batch_size` caps samples per microbatch and `max_tokens` optionally caps
    padded tokens per microbatch. `bucket_size` is a length interval; samples do
    not cross that interval when bucketing is active. `preserve_order=True` keeps
    dataset order and may leave batches underfilled at bucket boundaries. With
    `preserve_order=False`, batches are stably sorted by length and optionally
    shuffled as complete microbatches by `seed`; the resulting permutation and its
    inverse are exposed. No tensors or sample objects are copied.

    Empty `sample_lengths` produces an empty plan. Invalid shapes, lengths, limits,
    or a sample longer than `max_tokens` raise `TypeError` or `ValueError`.
    `gradient_accumulation_steps` annotates every microbatch and retains a final
    partial optimizer group instead of dropping it.
    """

    def __init__(
        self,
        sample_lengths: Iterable[int],
        *,
        max_batch_size: int,
        max_tokens: int | None = None,
        bucket_size: int | None = None,
        preserve_order: bool = True,
        seed: int | None = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self._sample_lengths = self._validate_sample_lengths(sample_lengths)
        self._max_batch_size = _validate_integer(max_batch_size, "max_batch_size", minimum=1)
        self._max_tokens = (
            None if max_tokens is None else _validate_integer(max_tokens, "max_tokens", minimum=1)
        )
        self._bucket_size = (
            None if bucket_size is None else _validate_integer(bucket_size, "bucket_size", minimum=1)
        )
        if not isinstance(preserve_order, bool):
            raise TypeError("preserve_order must be a boolean")
        self._preserve_order = preserve_order
        if seed is not None:
            seed = _validate_integer(seed, "seed", minimum=None)
        if preserve_order and seed is not None:
            raise ValueError("seed requires preserve_order=False")
        self._seed = seed
        self._gradient_accumulation_steps = _validate_integer(
            gradient_accumulation_steps, "gradient_accumulation_steps", minimum=1
        )
        if self.max_tokens is not None and self.sample_lengths:
            longest_length = max(self.sample_lengths)
            if longest_length > self.max_tokens:
                raise ValueError("max_tokens must accommodate every individual sample")

        batch_indices = self._create_batch_indices()
        self._microbatches = self._create_microbatch_plans(batch_indices)
        self._permutation = tuple(
            sample_index
            for microbatch in self._microbatches
            for sample_index in microbatch.sample_indices
        )
        inverse_permutation = [0] * len(self._permutation)
        for planned_position, sample_index in enumerate(self._permutation):
            inverse_permutation[sample_index] = planned_position
        self._inverse_permutation = tuple(inverse_permutation)
        self._metrics = BatchingMetrics(
            sample_count=len(self.sample_lengths),
            microbatch_count=len(self._microbatches),
            real_token_count=sum(microbatch.real_token_count for microbatch in self._microbatches),
            padded_token_count=sum(
                microbatch.padded_token_count for microbatch in self._microbatches
            ),
            padding_token_count=sum(
                microbatch.padding_token_count for microbatch in self._microbatches
            ),
            optimizer_step_count=self.optimizer_step_count,
        )

    @staticmethod
    def _validate_sample_lengths(sample_lengths: Iterable[int]) -> tuple[int, ...]:
        if isinstance(sample_lengths, (str, bytes)):
            raise TypeError("sample_lengths must be an iterable of integer lengths")
        try:
            raw_lengths = tuple(sample_lengths)
        except TypeError as error:
            raise TypeError("sample_lengths must be an iterable of integer lengths") from error
        return tuple(
            _validate_integer(length, f"sample_lengths[{index}]", minimum=1)
            for index, length in enumerate(raw_lengths)
        )

    @property
    def sample_lengths(self) -> tuple[int, ...]:
        return self._sample_lengths

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    @property
    def bucket_size(self) -> int | None:
        return self._bucket_size

    @property
    def preserve_order(self) -> bool:
        return self._preserve_order

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def gradient_accumulation_steps(self) -> int:
        return self._gradient_accumulation_steps

    def _bucket_key(self, sample_length: int) -> int:
        if self.bucket_size is None:
            return 0
        return (sample_length - 1) // self.bucket_size

    def _would_exceed_limits(self, batch_indices: list[int], sample_index: int) -> bool:
        if len(batch_indices) >= self.max_batch_size:
            return True
        if self.max_tokens is None:
            return False
        proposed_max_length = max(
            self.sample_lengths[index] for index in (*batch_indices, sample_index)
        )
        proposed_padded_tokens = (len(batch_indices) + 1) * proposed_max_length
        return proposed_padded_tokens > self.max_tokens

    def _split_indices(self, ordered_indices: Iterable[int]) -> list[tuple[int, ...]]:
        batches: list[tuple[int, ...]] = []
        current_batch: list[int] = []
        for sample_index in ordered_indices:
            if current_batch and self._would_exceed_limits(current_batch, sample_index):
                batches.append(tuple(current_batch))
                current_batch = []
            current_batch.append(sample_index)
        if current_batch:
            batches.append(tuple(current_batch))
        return batches

    def _create_preserved_batches(self) -> list[tuple[int, ...]]:
        batches: list[tuple[int, ...]] = []
        current_batch: list[int] = []
        current_bucket_key: int | None = None
        for sample_index, sample_length in enumerate(self.sample_lengths):
            sample_bucket_key = self._bucket_key(sample_length)
            bucket_changed = current_batch and sample_bucket_key != current_bucket_key
            limit_reached = current_batch and self._would_exceed_limits(current_batch, sample_index)
            if bucket_changed or limit_reached:
                batches.append(tuple(current_batch))
                current_batch = []
            if not current_batch:
                current_bucket_key = sample_bucket_key
            current_batch.append(sample_index)
        if current_batch:
            batches.append(tuple(current_batch))
        return batches

    def _create_reordered_batches(self) -> list[tuple[int, ...]]:
        buckets: dict[int, list[int]] = {}
        for sample_index, sample_length in enumerate(self.sample_lengths):
            buckets.setdefault(self._bucket_key(sample_length), []).append(sample_index)
        batches: list[tuple[int, ...]] = []
        for bucket_key in sorted(buckets):
            bucket_indices = sorted(
                buckets[bucket_key], key=lambda index: (self.sample_lengths[index], index)
            )
            batches.extend(self._split_indices(bucket_indices))
        if self.seed is not None:
            random.Random(self.seed).shuffle(batches)
        return batches

    def _create_batch_indices(self) -> tuple[tuple[int, ...], ...]:
        if self.preserve_order:
            return tuple(self._create_preserved_batches())
        return tuple(self._create_reordered_batches())

    def _create_microbatch_plans(
        self, batch_indices: tuple[tuple[int, ...], ...]
    ) -> tuple[MicrobatchPlan, ...]:
        microbatch_plans: list[MicrobatchPlan] = []
        for microbatch_index, sample_indices in enumerate(batch_indices):
            sample_lengths = tuple(self.sample_lengths[index] for index in sample_indices)
            max_length = max(sample_lengths)
            real_token_count = sum(sample_lengths)
            padded_token_count = len(sample_indices) * max_length
            accumulation_position = (
                microbatch_index % self.gradient_accumulation_steps
            ) + 1
            is_optimizer_step_boundary = (
                accumulation_position == self.gradient_accumulation_steps
                or microbatch_index == len(batch_indices) - 1
            )
            microbatch_plans.append(
                MicrobatchPlan(
                    sample_indices=sample_indices,
                    sample_lengths=sample_lengths,
                    max_length=max_length,
                    real_token_count=real_token_count,
                    padded_token_count=padded_token_count,
                    padding_token_count=padded_token_count - real_token_count,
                    microbatch_index=microbatch_index,
                    optimizer_step_index=microbatch_index // self.gradient_accumulation_steps,
                    accumulation_position=accumulation_position,
                    is_optimizer_step_boundary=is_optimizer_step_boundary,
                )
            )
        return tuple(microbatch_plans)

    @property
    def permutation(self) -> tuple[int, ...]:
        """Return planned-position to original-sample indices."""

        return self._permutation

    @property
    def inverse_permutation(self) -> tuple[int, ...]:
        """Return original-sample index to planned-position mapping."""

        return self._inverse_permutation

    @property
    def metrics(self) -> BatchingMetrics:
        """Return aggregate real-token, padding, batch, and step metrics."""

        return self._metrics

    @property
    def optimizer_step_count(self) -> int:
        """Return optimizer updates, including a final partial accumulation group."""

        if not self._microbatches:
            return 0
        return (len(self._microbatches) + self.gradient_accumulation_steps - 1) // (
            self.gradient_accumulation_steps
        )

    def __iter__(self) -> Iterator[MicrobatchPlan]:
        return iter(self._microbatches)

    def __len__(self) -> int:
        return len(self._microbatches)
