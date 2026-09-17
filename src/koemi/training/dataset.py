from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from koemi.configuration.settings import PAD_TOKEN_ID
from koemi.data.contracts import DatasetRecord, DatasetValidationError
from koemi.data.serialization import serialize_record
from koemi.training.batching_mode import BatchingMode


IGNORE_TARGET_ID = -100


@dataclass(frozen=True)
class CausalChunk:
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    thinking_mask: tuple[bool, ...]


class CausalByteDataset(Dataset[CausalChunk]):
    def __init__(self, records: tuple[DatasetRecord, ...], sequence_length: int) -> None:
        self.chunks = self.create_chunks(records, sequence_length)
        if not self.chunks:
            raise DatasetValidationError("dataset does not contain a trainable causal sequence")
        if not any(target_id != IGNORE_TARGET_ID for chunk in self.chunks for target_id in chunk.target_ids):
            raise DatasetValidationError("dataset does not contain supervised target tokens")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> CausalChunk:
        return self.chunks[index]

    def create_chunks(self, records: tuple[DatasetRecord, ...], sequence_length: int) -> tuple[CausalChunk, ...]:
        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        chunks: list[CausalChunk] = []
        for record in records:
            serialized_record = serialize_record(record)
            token_ids = tuple(serialized_record.token_bytes)
            if len(token_ids) < 2:
                continue
            for start_index in range(0, len(token_ids) - 1, sequence_length):
                end_index = min(start_index + sequence_length, len(token_ids) - 1)
                input_ids = token_ids[start_index:end_index]
                raw_target_ids = token_ids[start_index + 1 : end_index + 1]
                target_positions = serialized_record.supervised_positions[start_index + 1 : end_index + 1]
                thinking_positions = serialized_record.thinking_positions[start_index + 1 : end_index + 1]
                target_ids = tuple(
                    token_id if is_supervised else IGNORE_TARGET_ID
                    for token_id, is_supervised in zip(raw_target_ids, target_positions, strict=True)
                )
                chunks.append(CausalChunk(input_ids, target_ids, thinking_positions))
        return tuple(chunks)


class _LengthAwareBatchSampler(Sampler[list[int]]):
    """Rebuild a deterministic length-aware plan at the start of each epoch."""

    def __init__(
        self,
        sample_lengths: tuple[int, ...],
        max_batch_size: int,
        max_batch_tokens: int | None,
        length_bucket_size: int | None,
        shuffle: bool,
        seed: int,
    ) -> None:
        self._sample_lengths = sample_lengths
        self._max_batch_size = max_batch_size
        self._max_batch_tokens = max_batch_tokens
        self._length_bucket_size = length_bucket_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0
        self._batch_count = len(self._build_plan(None))

    def _build_plan(self, epoch: int | None) -> BatchingMode:
        seed = None
        if self._shuffle:
            seed = self._seed if epoch is None else self._seed + epoch
        return BatchingMode(
            self._sample_lengths,
            max_batch_size=self._max_batch_size,
            max_tokens=self._max_batch_tokens,
            bucket_size=self._length_bucket_size,
            preserve_order=not self._shuffle,
            seed=seed,
        )

    def __iter__(self) -> Iterator[list[int]]:
        plan = self._build_plan(self._epoch)
        self._epoch += 1
        for microbatch in plan:
            yield list(microbatch.sample_indices)

    def __len__(self) -> int:
        return self._batch_count


def create_training_loader(
    dataset: CausalByteDataset,
    batch_size: int,
    generator: torch.Generator | None = None,
    *,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    max_batch_tokens: int | None = None,
    length_bucket_size: int | None = None,
) -> DataLoader[CausalChunk]:
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be at least 1")
    worker_options = {}
    if num_workers > 0:
        worker_options = {"prefetch_factor": prefetch_factor, "persistent_workers": True}
    if max_batch_tokens is not None or length_bucket_size is not None:
        seed = (
            generator.initial_seed()
            if generator is not None
            else random.SystemRandom().randrange(0, 2**63)
        )
        batch_sampler = _LengthAwareBatchSampler(
            tuple(len(chunk.input_ids) for chunk in dataset.chunks),
            batch_size,
            max_batch_tokens,
            length_bucket_size,
            bool(shuffle),
            seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_chunks,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **worker_options,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_chunks,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **worker_options,
    )


def collate_chunks(chunks: list[CausalChunk]) -> dict[str, Tensor]:
    maximum_length = max(len(chunk.input_ids) for chunk in chunks)
    input_ids = torch.full((len(chunks), maximum_length), PAD_TOKEN_ID, dtype=torch.long)
    target_ids = torch.full((len(chunks), maximum_length), IGNORE_TARGET_ID, dtype=torch.long)
    thinking_mask = torch.zeros((len(chunks), maximum_length), dtype=torch.bool)
    for row_index, chunk in enumerate(chunks):
        chunk_length = len(chunk.input_ids)
        input_ids[row_index, :chunk_length] = torch.tensor(chunk.input_ids, dtype=torch.long)
        target_ids[row_index, :chunk_length] = torch.tensor(chunk.target_ids, dtype=torch.long)
        thinking_mask[row_index, :chunk_length] = torch.tensor(chunk.thinking_mask, dtype=torch.bool)
    return {"input_ids": input_ids, "target_ids": target_ids, "thinking_mask": thinking_mask}
