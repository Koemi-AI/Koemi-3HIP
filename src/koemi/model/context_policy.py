from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor
from torch.nn import functional


class AdmissionReason(IntEnum):
    ADMITTED = 0
    REPLACED = 1
    DUPLICATE = 2
    FUTURE = 3
    INVALID = 4
    CAPACITY = 5
    EVICTED = 6
    DISABLED = 7


ADMISSION_REASON_COUNT = len(AdmissionReason)


def _count_reasons(reason_codes: Tensor) -> Tensor:
    reason_counts = torch.zeros(
        (reason_codes.shape[0], ADMISSION_REASON_COUNT),
        device=reason_codes.device,
        dtype=torch.long,
    )
    return reason_counts.scatter_add_(1, reason_codes, torch.ones_like(reason_codes))


@dataclass(frozen=True)
class ContextSelection:
    """Result of one bounded, causal context-selection pass.

    ``selected_indices`` is ``[batch, capacity]`` and uses ``-1`` for empty
    slots. ``selected_mask`` is the final ``[batch, candidates]`` membership
    mask. ``priority_scores`` and all counters stay on the input device.
    ``admitted_mask`` records every admission event, including entries later
    evicted by a higher-priority candidate. ``reason_codes`` describes the
    final status of each candidate using :class:`AdmissionReason` values.
    """

    selected_indices: Tensor
    selected_mask: Tensor
    selected_priority_scores: Tensor
    selected_positions: Tensor
    priority_scores: Tensor
    admitted_mask: Tensor
    reason_codes: Tensor
    reason_counts: Tensor

    @property
    def admission_counts(self) -> Tensor:
        return self.admitted_mask.sum(dim=1)

    @property
    def selected_counts(self) -> Tensor:
        return self.selected_mask.sum(dim=1)

    @property
    def reasons(self) -> Tensor:
        return self.reason_codes

    @property
    def counts(self) -> Tensor:
        return self.reason_counts


class ContextPolicy:
    """Select a fixed-size causal context using surprise, recency, and novelty.

    Candidates must be supplied in observation order. ``current_position``
    defines the causal horizon; candidates beyond it are never admitted or
    used to normalize a score. When ``enabled`` is false, the policy emits no
    admissions so callers can retain their existing memory path.

    The hard selection is intentionally detached from autograd. Scores and
    feature tensors are read-only inputs; the returned decisions are not a
    differentiable surrogate for them. Selection uses one pass over candidates
    and bounded scans over the fixed capacity, with no host scalar extraction.
    """

    def __init__(
        self,
        capacity: int,
        *,
        enabled: bool = False,
        surprise_weight: float = 0.5,
        recency_weight: float = 0.3,
        diversity_weight: float = 0.2,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("context capacity must be a positive integer")
        if not isinstance(enabled, bool):
            raise TypeError("context policy enabled must be a boolean")
        weights = (surprise_weight, recency_weight, diversity_weight)
        if any(isinstance(weight, bool) or not isinstance(weight, (int, float)) for weight in weights):
            raise TypeError("context policy weights must be real numbers")
        if any(not math.isfinite(float(weight)) or float(weight) < 0.0 for weight in weights):
            raise ValueError("context policy weights must be finite and non-negative")
        total_weight = sum(float(weight) for weight in weights)
        if total_weight <= 0.0:
            raise ValueError("context policy requires at least one positive weight")

        self.capacity = capacity
        self.enabled = enabled
        self.surprise_weight = float(surprise_weight) / total_weight
        self.recency_weight = float(recency_weight) / total_weight
        self.diversity_weight = float(diversity_weight) / total_weight

    def select(
        self,
        causal_scores: Tensor,
        observed_positions: Tensor,
        *,
        current_position: int | Tensor | None = None,
        valid_mask: Tensor | None = None,
        entry_ids: Tensor | None = None,
        features: Tensor | None = None,
        diversity_scores: Tensor | None = None,
    ) -> ContextSelection:
        """Select observed token or block candidates without future access.

        ``causal_scores`` and ``observed_positions`` have shape ``[batch,
        candidates]``. ``current_position`` is a scalar or ``[batch]`` causal
        horizon; when omitted, the greatest valid observed position is used.
        ``features`` has shape ``[batch, candidates, width]`` and supplies
        cosine-based novelty. It is mutually exclusive with precomputed
        ``diversity_scores`` of shape ``[batch, candidates]``. ``entry_ids``
        enables bounded deduplication against currently retained entries.

        Returns a :class:`ContextSelection`; invalid shapes, non-finite
        floating inputs, incompatible devices, and non-positive capacity are
        rejected with an exception.
        """

        valid_mask, entry_ids, current_positions = self._validate_inputs(
            causal_scores,
            observed_positions,
            current_position,
            valid_mask,
            entry_ids,
            features,
            diversity_scores,
        )
        batch_size, candidate_count = causal_scores.shape
        decision_dtype = self._decision_dtype(causal_scores.dtype)
        detached_scores = causal_scores.detach().to(decision_dtype)
        if not self.enabled:
            return self._disabled_selection(
                batch_size,
                candidate_count,
                causal_scores,
                decision_dtype,
                observed_positions.dtype,
            )
        if candidate_count == 0:
            return self._empty_selection(
                batch_size,
                causal_scores.device,
                decision_dtype,
                observed_positions.dtype,
            )

        eligible_mask = valid_mask & (
            observed_positions <= current_positions.unsqueeze(1)
        )
        future_mask = valid_mask & (
            observed_positions > current_positions.unsqueeze(1)
        )
        normalized_surprise = self._normalize_component(detached_scores, eligible_mask)
        normalized_recency = self._normalize_component(
            observed_positions.to(decision_dtype), eligible_mask
        )
        detached_features = (
            functional.normalize(features.detach().to(decision_dtype), dim=-1)
            if features is not None
            else None
        )
        detached_diversity = (
            diversity_scores.detach().to(decision_dtype)
            if diversity_scores is not None
            else None
        )

        return self._select_observed_candidates(
            valid_mask,
            eligible_mask,
            future_mask,
            entry_ids,
            observed_positions,
            normalized_surprise,
            normalized_recency,
            detached_features,
            detached_diversity,
            batch_size,
            candidate_count,
            decision_dtype,
            observed_positions.dtype,
        )

    def _select_observed_candidates(
        self,
        valid_mask: Tensor,
        eligible_mask: Tensor,
        future_mask: Tensor,
        entry_ids: Tensor,
        observed_positions: Tensor,
        normalized_surprise: Tensor,
        normalized_recency: Tensor,
        detached_features: Tensor | None,
        detached_diversity: Tensor | None,
        batch_size: int,
        candidate_count: int,
        decision_dtype: torch.dtype,
        position_dtype: torch.dtype,
    ) -> ContextSelection:
        device = eligible_mask.device
        selected_indices = torch.full(
            (batch_size, self.capacity), -1, device=device, dtype=torch.long
        )
        selected_valid = torch.zeros(
            (batch_size, self.capacity), device=device, dtype=torch.bool
        )
        selected_priority_scores = torch.full(
            (batch_size, self.capacity),
            float("-inf"),
            device=device,
            dtype=decision_dtype,
        )
        selected_positions = torch.zeros(
            (batch_size, self.capacity), device=device, dtype=position_dtype
        )
        selected_orders = torch.full(
            (batch_size, self.capacity), -1, device=device, dtype=torch.long
        )
        selected_entry_ids = torch.zeros(
            (batch_size, self.capacity), device=device, dtype=entry_ids.dtype
        )
        selected_features = (
            torch.zeros(
                (batch_size, self.capacity, detached_features.shape[-1]),
                device=device,
                dtype=decision_dtype,
            )
            if detached_features is not None
            else None
        )
        priority_scores = torch.zeros(
            (batch_size, candidate_count), device=device, dtype=decision_dtype
        )
        admitted_mask = torch.zeros(
            (batch_size, candidate_count), device=device, dtype=torch.bool
        )
        reason_codes = torch.full(
            (batch_size, candidate_count),
            int(AdmissionReason.INVALID),
            device=device,
            dtype=torch.long,
        )
        selected_mask = torch.zeros(
            (batch_size, candidate_count), device=device, dtype=torch.bool
        )
        candidate_index = torch.empty((batch_size,), device=device, dtype=torch.long)

        for candidate_position in range(candidate_count):
            candidate_index.fill_(candidate_position)
            candidate_position_value = observed_positions[:, candidate_position]
            candidate_observed = eligible_mask[:, candidate_position]
            candidate_identity = entry_ids[:, candidate_position]
            duplicate_mask = candidate_observed & (
                selected_valid
                & selected_entry_ids.eq(candidate_identity.unsqueeze(1))
            ).any(dim=1)

            candidate_features: Tensor | None = None
            if detached_features is None:
                if detached_diversity is None:
                    novelty = torch.ones(
                        (batch_size,), device=device, dtype=decision_dtype
                    )
                else:
                    novelty = detached_diversity[:, candidate_position].clamp(0.0, 1.0)
            else:
                candidate_features = detached_features[:, candidate_position]
                normalized_candidate = candidate_features
                normalized_selected = selected_features
                similarity = torch.einsum(
                    "bcd,bd->bc", normalized_selected, normalized_candidate
                )
                similarity = similarity.masked_fill(~selected_valid, -1.0)
                highest_similarity = similarity.amax(dim=1)
                has_selected = selected_valid.any(dim=1)
                novelty = torch.where(
                    has_selected,
                    1.0 - highest_similarity,
                    torch.ones_like(highest_similarity),
                ).clamp(0.0, 1.0)

            candidate_priority = (
                self.surprise_weight * normalized_surprise[:, candidate_position]
                + self.recency_weight * normalized_recency[:, candidate_position]
                + self.diversity_weight * novelty
            )
            candidate_priority = torch.where(
                candidate_observed,
                candidate_priority,
                torch.zeros_like(candidate_priority),
            )
            priority_scores = priority_scores.scatter(
                1, candidate_index.unsqueeze(1), candidate_priority.unsqueeze(1)
            )

            has_free_slot = (~selected_valid).any(dim=1)
            first_free_slot = (~selected_valid).to(torch.long).argmax(dim=1)
            worst_slot = torch.zeros((batch_size,), device=device, dtype=torch.long)
            for slot_index in range(1, self.capacity):
                slot_priority = selected_priority_scores[:, slot_index]
                current_worst_priority = selected_priority_scores.gather(
                    1, worst_slot.unsqueeze(1)
                ).squeeze(1)
                slot_position = selected_positions[:, slot_index]
                current_worst_position = selected_positions.gather(
                    1, worst_slot.unsqueeze(1)
                ).squeeze(1)
                slot_order = selected_orders[:, slot_index]
                current_worst_order = selected_orders.gather(
                    1, worst_slot.unsqueeze(1)
                ).squeeze(1)
                equal_priority = slot_priority == current_worst_priority
                slot_is_worse = (slot_priority < current_worst_priority) | (
                    equal_priority
                    & (
                        (slot_position < current_worst_position)
                        | (
                            (slot_position == current_worst_position)
                            & (slot_order > current_worst_order)
                        )
                    )
                )
                worst_slot = torch.where(
                    slot_is_worse,
                    slot_index,
                    worst_slot,
                )
            worst_priority = selected_priority_scores.gather(
                1, worst_slot.unsqueeze(1)
            ).squeeze(1)
            worst_position = selected_positions.gather(
                1, worst_slot.unsqueeze(1)
            ).squeeze(1)
            worst_order = selected_orders.gather(
                1, worst_slot.unsqueeze(1)
            ).squeeze(1)
            priority_tie = candidate_priority == worst_priority
            candidate_wins_tie = (
                candidate_position_value > worst_position
            ) | (
                (candidate_position_value == worst_position)
                & (candidate_index < worst_order)
            )
            candidate_beats_worst = (candidate_priority > worst_priority) | (
                priority_tie & candidate_wins_tie
            )
            candidate_allowed = candidate_observed & ~duplicate_mask
            candidate_admitted = candidate_allowed & (
                has_free_slot | candidate_beats_worst
            )
            candidate_replaces = candidate_admitted & ~has_free_slot

            evicted_index = selected_indices.gather(
                1, worst_slot.unsqueeze(1)
            ).squeeze(1)
            evicted_target = evicted_index.clamp_min(0).unsqueeze(1)
            evicted_reason = reason_codes.gather(1, evicted_target).squeeze(1)
            evicted_reason = torch.where(
                candidate_replaces,
                int(AdmissionReason.EVICTED),
                evicted_reason,
            )
            reason_codes = reason_codes.scatter(1, evicted_target, evicted_reason.unsqueeze(1))
            candidate_reason = torch.full(
                (batch_size,),
                int(AdmissionReason.CAPACITY),
                device=device,
                dtype=torch.long,
            )
            candidate_reason = torch.where(
                ~valid_mask[:, candidate_position],
                int(AdmissionReason.INVALID),
                candidate_reason,
            )
            candidate_reason = torch.where(
                future_mask[:, candidate_position],
                int(AdmissionReason.FUTURE),
                candidate_reason,
            )
            candidate_reason = torch.where(
                duplicate_mask,
                int(AdmissionReason.DUPLICATE),
                candidate_reason,
            )
            candidate_reason = torch.where(
                candidate_replaces,
                int(AdmissionReason.REPLACED),
                candidate_reason,
            )
            candidate_reason = torch.where(
                candidate_admitted & ~candidate_replaces,
                int(AdmissionReason.ADMITTED),
                candidate_reason,
            )
            reason_codes = reason_codes.scatter(
                1, candidate_index.unsqueeze(1), candidate_reason.unsqueeze(1)
            )
            admitted_mask = admitted_mask.scatter(
                1, candidate_index.unsqueeze(1), candidate_admitted.unsqueeze(1)
            )
            retained_before_update = selected_mask.gather(1, evicted_target).squeeze(1)
            retained_after_eviction = torch.where(
                candidate_replaces,
                torch.zeros_like(retained_before_update),
                retained_before_update,
            )
            selected_mask = selected_mask.scatter(
                1, evicted_target, retained_after_eviction.unsqueeze(1)
            )
            selected_mask = selected_mask.scatter(
                1, candidate_index.unsqueeze(1), candidate_admitted.unsqueeze(1)
            )

            update_slot = torch.where(has_free_slot, first_free_slot, worst_slot)
            slot_mask = functional.one_hot(
                update_slot, num_classes=self.capacity
            ).to(torch.bool)
            slot_mask = slot_mask & candidate_admitted.unsqueeze(1)
            selected_indices = torch.where(
                slot_mask,
                candidate_index.unsqueeze(1),
                selected_indices,
            )
            selected_valid = torch.where(
                slot_mask,
                True,
                selected_valid,
            )
            selected_priority_scores = torch.where(
                slot_mask,
                candidate_priority.unsqueeze(1),
                selected_priority_scores,
            )
            selected_positions = torch.where(
                slot_mask,
                candidate_position_value.unsqueeze(1),
                selected_positions,
            )
            selected_orders = torch.where(
                slot_mask,
                candidate_index.unsqueeze(1),
                selected_orders,
            )
            selected_entry_ids = torch.where(
                slot_mask,
                candidate_identity.unsqueeze(1),
                selected_entry_ids,
            )
            if selected_features is not None:
                selected_features = torch.where(
                    slot_mask.unsqueeze(-1),
                    candidate_features.unsqueeze(1),
                    selected_features,
                )

        _, temporal_order = torch.sort(selected_positions, dim=1, stable=True)
        temporal_valid = selected_valid.gather(1, temporal_order)
        _, validity_order = torch.sort(
            (~temporal_valid).to(torch.long), dim=1, stable=True
        )
        slot_order = temporal_order.gather(1, validity_order)
        ordered_indices = selected_indices.gather(1, slot_order)
        ordered_valid = selected_valid.gather(1, slot_order)
        ordered_priority_scores = selected_priority_scores.gather(1, slot_order)
        ordered_positions = selected_positions.gather(1, slot_order)
        ordered_priority_scores = torch.where(
            ordered_valid,
            ordered_priority_scores,
            torch.zeros_like(ordered_priority_scores),
        )
        ordered_indices = torch.where(
            ordered_valid,
            ordered_indices,
            torch.full_like(ordered_indices, -1),
        )
        ordered_positions = torch.where(
            ordered_valid,
            ordered_positions,
            torch.zeros_like(ordered_positions),
        )
        reason_counts = _count_reasons(reason_codes)
        return ContextSelection(
            selected_indices=ordered_indices,
            selected_mask=selected_mask,
            selected_priority_scores=ordered_priority_scores,
            selected_positions=ordered_positions,
            priority_scores=priority_scores,
            admitted_mask=admitted_mask,
            reason_codes=reason_codes,
            reason_counts=reason_counts,
        )

    def _validate_inputs(
        self,
        causal_scores: Tensor,
        observed_positions: Tensor,
        current_position: int | Tensor | None,
        valid_mask: Tensor | None,
        entry_ids: Tensor | None,
        features: Tensor | None,
        diversity_scores: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(causal_scores, Tensor) or causal_scores.ndim != 2:
            raise ValueError("causal_scores must have shape [batch, candidates]")
        if not torch.is_floating_point(causal_scores) or torch.is_complex(causal_scores):
            raise TypeError("causal_scores must be a real floating tensor")
        if not isinstance(observed_positions, Tensor) or observed_positions.shape != causal_scores.shape:
            raise ValueError("observed_positions must match causal_scores shape")
        if (
            observed_positions.device != causal_scores.device
            or observed_positions.dtype == torch.bool
            or torch.is_floating_point(observed_positions)
            or torch.is_complex(observed_positions)
        ):
            raise TypeError("observed_positions must be an integer tensor on the score device")
        self._assert_finite(causal_scores, "causal_scores")

        if valid_mask is None:
            valid_mask = torch.ones_like(causal_scores, dtype=torch.bool)
        elif not isinstance(valid_mask, Tensor):
            raise TypeError("valid_mask must be a tensor")
        elif valid_mask.shape != causal_scores.shape or valid_mask.device != causal_scores.device:
            raise ValueError("valid_mask must match causal_scores shape and device")
        elif valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")

        if entry_ids is None:
            entry_ids = observed_positions
        elif not isinstance(entry_ids, Tensor):
            raise TypeError("entry_ids must be a tensor")
        elif entry_ids.shape != causal_scores.shape or entry_ids.device != causal_scores.device:
            raise ValueError("entry_ids must match causal_scores shape and device")
        elif (
            entry_ids.dtype == torch.bool
            or torch.is_floating_point(entry_ids)
            or torch.is_complex(entry_ids)
        ):
            raise TypeError("entry_ids must be an integer tensor")

        if features is not None and diversity_scores is not None:
            raise ValueError("features and diversity_scores are mutually exclusive")
        if features is not None:
            if not isinstance(features, Tensor):
                raise TypeError("features must be a tensor")
            if (
                features.ndim != 3
                or features.shape[:2] != causal_scores.shape
                or features.shape[2] < 1
                or features.device != causal_scores.device
                or not torch.is_floating_point(features)
                or torch.is_complex(features)
            ):
                raise ValueError(
                    "features must have shape [batch, candidates, width] on the score device"
                )
            self._assert_finite(features, "features")
        if diversity_scores is not None:
            if not isinstance(diversity_scores, Tensor):
                raise TypeError("diversity_scores must be a tensor")
            if (
                diversity_scores.shape != causal_scores.shape
                or diversity_scores.device != causal_scores.device
                or not torch.is_floating_point(diversity_scores)
                or torch.is_complex(diversity_scores)
            ):
                raise ValueError("diversity_scores must match causal_scores as a real tensor")
            self._assert_finite(diversity_scores, "diversity_scores")

        current_positions = self._resolve_current_positions(
            observed_positions, valid_mask, current_position
        )
        return valid_mask, entry_ids, current_positions

    @staticmethod
    def _resolve_current_positions(
        observed_positions: Tensor,
        valid_mask: Tensor,
        current_position: int | Tensor | None,
    ) -> Tensor:
        batch_size, candidate_count = observed_positions.shape
        if current_position is None:
            if candidate_count == 0:
                return torch.zeros(
                    (batch_size,), device=observed_positions.device, dtype=observed_positions.dtype
                )
            minimum_position = torch.iinfo(observed_positions.dtype).min
            masked_positions = torch.where(
                valid_mask,
                observed_positions,
                torch.full_like(observed_positions, minimum_position),
            )
            return masked_positions.amax(dim=1)
        if isinstance(current_position, bool):
            raise TypeError("current_position must be an integer")
        if isinstance(current_position, int):
            return torch.full(
                (batch_size,),
                current_position,
                device=observed_positions.device,
                dtype=observed_positions.dtype,
            )
        if not isinstance(current_position, Tensor):
            raise TypeError("current_position must be an integer or integer tensor")
        if (
            current_position.device != observed_positions.device
            or current_position.dtype == torch.bool
            or torch.is_floating_point(current_position)
            or torch.is_complex(current_position)
        ):
            raise TypeError("current_position must be an integer tensor on the score device")
        if current_position.ndim == 0:
            return current_position.expand(batch_size)
        if current_position.ndim == 1 and current_position.shape[0] == batch_size:
            return current_position
        if current_position.ndim == 2 and current_position.shape == (batch_size, 1):
            return current_position.squeeze(1)
        raise ValueError("current_position must be scalar, [batch], or [batch, 1]")

    @staticmethod
    def _assert_finite(values: Tensor, name: str) -> None:
        torch._assert_async(torch.isfinite(values).all(), f"{name} must be finite")

    @staticmethod
    def _decision_dtype(dtype: torch.dtype) -> torch.dtype:
        return torch.float64 if dtype == torch.float64 else torch.float32

    @staticmethod
    def _normalize_component(values: Tensor, eligible_mask: Tensor) -> Tensor:
        maximum_finite = torch.finfo(values.dtype).max
        minimum_finite = torch.finfo(values.dtype).min
        minimum = torch.where(
            eligible_mask,
            values,
            torch.full_like(values, maximum_finite),
        ).amin(dim=1)
        maximum = torch.where(
            eligible_mask,
            values,
            torch.full_like(values, minimum_finite),
        ).amax(dim=1)
        span = maximum - minimum
        normalized = (values - minimum.unsqueeze(1)) / span.unsqueeze(1).clamp_min(
            torch.finfo(values.dtype).eps
        )
        normalized = torch.where(
            span.unsqueeze(1) > 0.0,
            normalized,
            torch.ones_like(normalized),
        )
        return torch.where(
            eligible_mask,
            normalized.clamp(0.0, 1.0),
            torch.zeros_like(normalized),
        )

    def _disabled_selection(
        self,
        batch_size: int,
        candidate_count: int,
        causal_scores: Tensor,
        decision_dtype: torch.dtype,
        position_dtype: torch.dtype,
    ) -> ContextSelection:
        device = causal_scores.device
        reason_codes = torch.full(
            (batch_size, candidate_count),
            int(AdmissionReason.DISABLED),
            device=device,
            dtype=torch.long,
        )
        reason_counts = _count_reasons(reason_codes)
        return ContextSelection(
            selected_indices=torch.full(
                (batch_size, self.capacity), -1, device=device, dtype=torch.long
            ),
            selected_mask=torch.zeros(
                (batch_size, candidate_count), device=device, dtype=torch.bool
            ),
            selected_priority_scores=torch.zeros(
                (batch_size, self.capacity), device=device, dtype=decision_dtype
            ),
            selected_positions=torch.zeros(
                (batch_size, self.capacity), device=device, dtype=position_dtype
            ),
            priority_scores=causal_scores.detach().to(decision_dtype),
            admitted_mask=torch.zeros(
                (batch_size, candidate_count), device=device, dtype=torch.bool
            ),
            reason_codes=reason_codes,
            reason_counts=reason_counts,
        )

    def _empty_selection(
        self,
        batch_size: int,
        device: torch.device,
        decision_dtype: torch.dtype,
        position_dtype: torch.dtype,
    ) -> ContextSelection:
        reason_codes = torch.empty(
            (batch_size, 0), device=device, dtype=torch.long
        )
        return ContextSelection(
            selected_indices=torch.full(
                (batch_size, self.capacity), -1, device=device, dtype=torch.long
            ),
            selected_mask=torch.empty(
                (batch_size, 0), device=device, dtype=torch.bool
            ),
            selected_priority_scores=torch.zeros(
                (batch_size, self.capacity), device=device, dtype=decision_dtype
            ),
            selected_positions=torch.zeros(
                (batch_size, self.capacity), device=device, dtype=position_dtype
            ),
            priority_scores=torch.empty(
                (batch_size, 0), device=device, dtype=decision_dtype
            ),
            admitted_mask=torch.empty(
                (batch_size, 0), device=device, dtype=torch.bool
            ),
            reason_codes=reason_codes,
            reason_counts=torch.zeros(
                (batch_size, ADMISSION_REASON_COUNT), device=device, dtype=torch.long
            ),
        )


__all__ = ["ADMISSION_REASON_COUNT", "AdmissionReason", "ContextPolicy", "ContextSelection"]
