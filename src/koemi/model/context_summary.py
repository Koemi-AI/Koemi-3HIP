from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn


DEFAULT_CONFIDENCE_SCALE = 4.0
DEFAULT_MAX_DECAY = 1.0 - 2.0**-12
DEFAULT_MAX_BATCH_SIZE = 4096
DEFAULT_MAX_STATE_ELEMENTS = 16_777_216
MAX_STEP_INDEX = 2**63 - 1
SERIALIZED_FORMAT_VERSION = 1
SERIALIZABLE_FLOAT_DTYPES = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
}
DESERIALIZABLE_FLOAT_DTYPES = {name: dtype for dtype, name in SERIALIZABLE_FLOAT_DTYPES.items()}
SERIALIZED_KEYS = frozenset(
    {"format_version", "shape", "dtype", "slots", "evidence_shape", "evidence", "step_index"}
)


@dataclass(frozen=True)
class ContextSummaryState:
    slots: Tensor
    evidence: Tensor
    step_index: int


@dataclass(frozen=True)
class ContextSummaryRead:
    value: Tensor
    confidence: Tensor
    evidence: Tensor
    mask: Tensor


@dataclass(frozen=True)
class ContextSummaryCost:
    update_calls: int
    active_slot_updates: int
    estimated_element_updates: int


@dataclass(frozen=True)
class SurpriseMemoryState:
    value: Tensor
    momentum: Tensor
    evidence: Tensor
    step_index: int


@dataclass(frozen=True)
class SurpriseMemoryRead:
    value: Tensor
    confidence: Tensor
    evidence: Tensor
    mask: Tensor


def _assert_tensor_condition(condition: Tensor, message: str) -> None:
    if condition.device.type == "cpu":
        if not bool(condition):
            raise ValueError(message)
        return
    torch._assert_async(condition, message)


class ContextSummary(nn.Module):
    """Finite, opt-in hierarchical EMA summaries for one isolated batch state."""

    def __init__(
        self,
        slots: int,
        width: int,
        decays: float | Sequence[float] | None = None,
        max_evidence: int = 1024,
        confidence_scale: float = DEFAULT_CONFIDENCE_SCALE,
        trainable: bool = False,
        measure_cost: bool = False,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_state_elements: int = DEFAULT_MAX_STATE_ELEMENTS,
    ) -> None:
        super().__init__()
        self.slot_count = self._require_positive_int(slots, "slots")
        self.width = self._require_positive_int(width, "width")
        self.max_evidence = self._require_positive_int(max_evidence, "max_evidence")
        if self.max_evidence > torch.iinfo(torch.int64).max - 1:
            raise ValueError("max_evidence is too large for int64 evidence counters")
        self.confidence_scale = self._require_positive_float(confidence_scale, "confidence_scale")
        self.max_batch_size = self._require_positive_int(max_batch_size, "max_batch_size")
        self.max_state_elements = self._require_positive_int(max_state_elements, "max_state_elements")
        if type(trainable) is not bool:
            raise TypeError("trainable must be a bool")
        if type(measure_cost) is not bool:
            raise TypeError("measure_cost must be a bool")
        self.measure_cost = measure_cost
        decay_values = self._resolve_decays(decays)
        if trainable:
            decay_logits = torch.tensor(
                [math.log(decay / (1.0 - decay)) for decay in decay_values],
                dtype=torch.get_default_dtype(),
            )
            self.decay_logits = nn.Parameter(decay_logits)
        else:
            self.register_buffer("decay_values", torch.tensor(decay_values, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "_cost_active_slot_updates",
            torch.zeros((), dtype=torch.int64),
            persistent=False,
        )
        self._cost = ContextSummaryCost(0, 0, 0)

    def create_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ContextSummaryState:
        """Create zero summaries on the module device, with bounded batch and state size."""
        batch_size = self._require_positive_int(batch_size, "batch_size")
        self._validate_state_size(batch_size)
        target_device = self._resolve_device(device)
        if target_device != self._decays().device:
            raise ValueError("context summary state device must match the module device")
        target_dtype = self._resolve_dtype(self._decays().dtype if dtype is None else dtype)
        state = ContextSummaryState(
            slots=torch.zeros(
                batch_size,
                self.slot_count,
                self.width,
                device=target_device,
                dtype=target_dtype,
            ),
            evidence=torch.zeros(batch_size, self.slot_count, device=target_device, dtype=torch.int64),
            step_index=0,
        )
        self._validate_state(state)
        return state

    def empty(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ContextSummaryState:
        """Alias for create_state for callers that model the empty memory explicitly."""
        return self.create_state(batch_size, device=device, dtype=dtype)

    def update(
        self,
        state: ContextSummaryState,
        values: Tensor,
        valid_mask: Tensor | None = None,
        slot_mask: Tensor | None = None,
        *,
        detach: bool = False,
    ) -> ContextSummaryState:
        """Apply one causal [batch, width] update and return a new bounded state."""
        self._validate_state(state)
        self._validate_module_device(state)
        self._validate_detach(detach)
        self._validate_values(values, state)
        valid_mask = self._validate_mask(valid_mask, (values.shape[0],), state.slots.device, "valid_mask")
        slot_mask = self._validate_mask(
            slot_mask,
            (values.shape[0], self.slot_count),
            state.slots.device,
            "slot_mask",
        )
        if valid_mask is None:
            active_slots = slot_mask
        elif slot_mask is None:
            active_slots = valid_mask.unsqueeze(1).expand(-1, self.slot_count)
        else:
            active_slots = valid_mask.unsqueeze(1) & slot_mask
        if state.step_index >= MAX_STEP_INDEX:
            raise ValueError("context summary step index reached its limit")

        update_rate = (1.0 - self._decays()).to(dtype=state.slots.dtype).view(1, self.slot_count, 1)
        bounded_values = torch.tanh(values).unsqueeze(1)
        candidate_slots = state.slots + update_rate * (bounded_values - state.slots)
        if active_slots is None:
            next_slots = candidate_slots.clamp(-1.0, 1.0)
            next_evidence = (state.evidence + 1).clamp_max(self.max_evidence)
        else:
            next_slots = torch.where(
                active_slots.unsqueeze(-1), candidate_slots, state.slots
            ).clamp(-1.0, 1.0)
            next_evidence = (
                state.evidence + active_slots.to(dtype=torch.int64)
            ).clamp_max(self.max_evidence)
        if detach:
            next_slots = next_slots.detach()
            next_evidence = next_evidence.detach()
        next_state = ContextSummaryState(next_slots, next_evidence, state.step_index + 1)
        if self.measure_cost:
            self._record_cost(active_slots, values.shape[0])
        return next_state

    def read(self, state: ContextSummaryState, *, detach: bool = False) -> ContextSummaryRead | None:
        """Read evidence-weighted slots; return None when the batch is empty.

        This compatibility method preserves the nullable result contract and may
        synchronize when the evidence lives on CUDA. Use :meth:`read_device` in
        a device-hot path when an empty read can be represented by zero confidence.
        """
        self._validate_state(state)
        self._validate_detach(detach)
        has_evidence = bool(torch.any(state.evidence > 0).detach().cpu().item())
        if not has_evidence:
            return None
        return self._read_validated(state, detach)

    def read_device(self, state: ContextSummaryState, *, detach: bool = False) -> ContextSummaryRead:
        """Read without a host scalar check; an empty state returns zero confidence."""
        self._validate_state(state)
        self._validate_detach(detach)
        return self._read_validated(state, detach)

    def _read_validated(self, state: ContextSummaryState, detach: bool) -> ContextSummaryRead:
        evidence_dtype = torch.float64 if state.slots.dtype == torch.float64 else torch.float32
        evidence = state.evidence.to(dtype=evidence_dtype)
        total_evidence = evidence.sum(dim=1)
        normalized_evidence = evidence / total_evidence.clamp_min(1.0).unsqueeze(1)
        slot_weights = normalized_evidence.to(dtype=state.slots.dtype)
        aggregated_value = torch.sum(slot_weights.unsqueeze(-1) * state.slots, dim=1)
        confidence = (1.0 - torch.exp(-total_evidence / self.confidence_scale)).clamp(0.0, 1.0)
        confidence = confidence.to(dtype=state.slots.dtype)
        value = aggregated_value * confidence.unsqueeze(-1)
        result = ContextSummaryRead(
            value=value,
            confidence=confidence,
            evidence=state.evidence,
            mask=state.evidence > 0,
        )
        if detach:
            return ContextSummaryRead(
                value=result.value.detach(),
                confidence=result.confidence.detach(),
                evidence=result.evidence.detach(),
                mask=result.mask.detach(),
            )
        return result

    def reset(self, state: ContextSummaryState) -> ContextSummaryState:
        """Return an empty state preserving the input batch, device, and dtype."""
        self._validate_state(state)
        return self.create_state(
            state.slots.shape[0],
            device=state.slots.device,
            dtype=state.slots.dtype,
        )

    def serialize(self, state: ContextSummaryState) -> dict[str, object]:
        """Return only validated primitive data and detached CPU lists."""
        self._validate_state(state)
        return {
            "format_version": SERIALIZED_FORMAT_VERSION,
            "shape": [int(size) for size in state.slots.shape],
            "dtype": SERIALIZABLE_FLOAT_DTYPES[state.slots.dtype],
            "slots": state.slots.detach().cpu().tolist(),
            "evidence_shape": [int(size) for size in state.evidence.shape],
            "evidence": state.evidence.detach().cpu().tolist(),
            "step_index": state.step_index,
        }

    def deserialize(
        self,
        payload: Any,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ContextSummaryState:
        """Validate a primitive payload and rebuild state on the requested device and dtype."""
        self._validate_payload_container(payload)
        shape = self._validate_shape_list(payload["shape"], "shape", 3)
        evidence_shape = self._validate_shape_list(payload["evidence_shape"], "evidence_shape", 2)
        if shape[1:] != [self.slot_count, self.width] or evidence_shape != [shape[0], self.slot_count]:
            raise ValueError("context summary payload shapes do not match this module")
        self._validate_state_size(shape[0])
        serialized_dtype = payload["dtype"]
        if type(serialized_dtype) is not str or serialized_dtype not in DESERIALIZABLE_FLOAT_DTYPES:
            raise ValueError("context summary payload dtype is invalid")
        serialized_state_dtype = DESERIALIZABLE_FLOAT_DTYPES[serialized_dtype]
        target_dtype = serialized_state_dtype if dtype is None else self._resolve_dtype(dtype)
        target_device = self._resolve_device(device)
        if target_device != self._decays().device:
            raise ValueError("context summary state device must match the module device")
        self._validate_nested_values(payload["slots"], shape, "slots", self._validate_float_value)
        self._validate_nested_values(
            payload["evidence"],
            evidence_shape,
            "evidence",
            self._validate_evidence_value,
        )
        step_index = payload["step_index"]
        if type(step_index) is not int or not 0 <= step_index <= MAX_STEP_INDEX:
            raise ValueError("context summary payload step index is invalid")
        slots = torch.tensor(payload["slots"], dtype=target_dtype, device=target_device)
        evidence = torch.tensor(payload["evidence"], dtype=torch.int64, device=target_device)
        state = ContextSummaryState(slots, evidence, step_index)
        self._validate_state(state)
        return state

    def cost_statistics(self) -> ContextSummaryCost:
        """Return aggregate update instrumentation without exposing tensors or session data."""
        return ContextSummaryCost(
            update_calls=self._cost.update_calls,
            active_slot_updates=int(self._cost_active_slot_updates.detach().item()),
            estimated_element_updates=self._cost.estimated_element_updates,
        )

    def reset_cost_statistics(self) -> None:
        """Reset aggregate update instrumentation."""
        self._cost_active_slot_updates.zero_()
        self._cost = ContextSummaryCost(0, 0, 0)

    def _decays(self) -> Tensor:
        if hasattr(self, "decay_logits"):
            return torch.sigmoid(self.decay_logits)
        return self.decay_values

    def _record_cost(self, active_slots: Tensor | None, batch_size: int) -> None:
        if active_slots is None:
            self._cost_active_slot_updates.add_(batch_size * self.slot_count)
        else:
            self._cost_active_slot_updates.add_(active_slots.sum(dtype=torch.int64))
        self._cost = ContextSummaryCost(
            update_calls=self._cost.update_calls + 1,
            active_slot_updates=0,
            estimated_element_updates=self._cost.estimated_element_updates
            + batch_size * self.slot_count * self.width,
        )

    def _validate_module_device(self, state: ContextSummaryState) -> None:
        if state.slots.device != self._decays().device:
            raise ValueError("context summary module and state must use the same device")

    def _validate_state(self, state: ContextSummaryState) -> None:
        if not isinstance(state, ContextSummaryState):
            raise TypeError("state must be a ContextSummaryState")
        if not isinstance(state.slots, Tensor) or not isinstance(state.evidence, Tensor):
            raise TypeError("context summary state fields must be tensors")
        if state.slots.ndim != 3 or tuple(state.slots.shape[1:]) != (self.slot_count, self.width):
            raise ValueError("context summary slots must have shape [batch, slots, width]")
        if state.evidence.shape != (state.slots.shape[0], self.slot_count):
            raise ValueError("context summary evidence shape is invalid")
        self._validate_state_size(state.slots.shape[0])
        self._validate_floating_dtype(state.slots.dtype, "state slots")
        if state.evidence.dtype != torch.int64:
            raise TypeError("context summary evidence must use int64")
        if state.slots.device != state.evidence.device:
            raise ValueError("context summary state tensors must use the same device")
        if type(state.step_index) is not int or not 0 <= state.step_index <= MAX_STEP_INDEX:
            raise ValueError("context summary step index is invalid")
        _assert_tensor_condition(
            torch.isfinite(state.slots).all(),
            "context summary slots must be finite",
        )
        _assert_tensor_condition(
            state.slots.abs().le(1.0).all(),
            "context summary slots must stay within [-1, 1]",
        )
        _assert_tensor_condition(
            state.evidence.ge(0).logical_and(state.evidence.le(self.max_evidence)).all(),
            "context summary evidence is outside its limit",
        )

    def _validate_values(self, values: Tensor, state: ContextSummaryState) -> None:
        if not isinstance(values, Tensor):
            raise TypeError("values must be a tensor")
        if values.ndim != 2 or tuple(values.shape) != (state.slots.shape[0], self.width):
            raise ValueError("values must have shape [batch, width]")
        self._validate_floating_dtype(values.dtype, "values")
        if values.dtype != state.slots.dtype:
            raise TypeError("values and state slots must use the same dtype")
        if values.device != state.slots.device:
            raise ValueError("values and state slots must use the same device")
        _assert_tensor_condition(torch.isfinite(values).all(), "values must be finite")

    @staticmethod
    def _validate_mask(
        mask: Tensor | None,
        expected_shape: tuple[int, ...],
        device: torch.device,
        name: str,
    ) -> Tensor | None:
        if mask is None:
            return None
        if not isinstance(mask, Tensor):
            raise TypeError(f"{name} must be a bool tensor")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must use bool dtype")
        if tuple(mask.shape) != expected_shape:
            raise ValueError(f"{name} has an invalid shape")
        if mask.device != device:
            raise ValueError(f"{name} and state must use the same device")
        return mask

    @staticmethod
    def _validate_detach(detach: bool) -> None:
        if type(detach) is not bool:
            raise TypeError("detach must be a bool")

    def _validate_state_size(self, batch_size: int) -> None:
        batch_size = self._require_positive_int(batch_size, "batch_size")
        if batch_size > self.max_batch_size:
            raise ValueError("context summary batch size exceeds its limit")
        if batch_size * self.slot_count * self.width > self.max_state_elements:
            raise ValueError("context summary state exceeds its element limit")

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        module_device = self._decays().device
        if device is None:
            target_device = module_device
        elif isinstance(device, (torch.device, str)):
            try:
                target_device = torch.device(device)
            except (RuntimeError, TypeError) as error:
                raise ValueError("context summary device is invalid") from error
        else:
            raise TypeError("device must be a torch.device or string")
        if target_device.type == "meta":
            raise ValueError("meta device is not supported for context summary state")
        return target_device

    @staticmethod
    def _resolve_dtype(dtype: torch.dtype | None) -> torch.dtype:
        target_dtype = torch.get_default_dtype() if dtype is None else dtype
        ContextSummary._validate_floating_dtype(target_dtype, "state")
        return target_dtype

    @staticmethod
    def _validate_floating_dtype(dtype: torch.dtype, name: str) -> None:
        if dtype not in SERIALIZABLE_FLOAT_DTYPES:
            raise TypeError(f"{name} must use a supported real floating dtype")

    @staticmethod
    def _require_positive_int(value: int, name: str) -> int:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive int")
        return value

    @staticmethod
    def _require_positive_float(value: float, name: str) -> float:
        if type(value) not in (int, float) or not math.isfinite(float(value)) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    def _resolve_decays(self, decays: float | Sequence[float] | None) -> list[float]:
        if decays is None:
            resolved_decays = [
                min(math.exp(-math.ldexp(1.0, -index)), DEFAULT_MAX_DECAY)
                for index in range(self.slot_count)
            ]
        elif type(decays) in (int, float):
            resolved_decays = [float(decays)] * self.slot_count
        elif isinstance(decays, (list, tuple)):
            if len(decays) != self.slot_count:
                raise ValueError("decays must contain one value per slot")
            resolved_decays = [float(decay) if type(decay) in (int, float) else math.nan for decay in decays]
        else:
            raise TypeError("decays must be a number, a list, a tuple, or None")
        if any(not math.isfinite(decay) or not 0.0 < decay < 1.0 for decay in resolved_decays):
            raise ValueError("each decay must be strictly between 0 and 1")
        return resolved_decays

    @staticmethod
    def _validate_payload_container(payload: Any) -> None:
        if type(payload) is not dict or set(payload) != SERIALIZED_KEYS:
            raise ValueError("context summary payload must be an exact data dictionary")
        if type(payload["format_version"]) is not int or payload["format_version"] != SERIALIZED_FORMAT_VERSION:
            raise ValueError("context summary payload format is invalid")

    @staticmethod
    def _validate_shape_list(value: Any, name: str, dimensions: int) -> list[int]:
        if type(value) is not list or len(value) != dimensions or any(type(size) is not int for size in value):
            raise ValueError(f"context summary payload {name} is invalid")
        if any(size < 1 for size in value):
            raise ValueError(f"context summary payload {name} contains an invalid limit")
        return value

    @classmethod
    def _validate_nested_values(
        cls,
        value: Any,
        shape: list[int],
        name: str,
        leaf_validator: Any,
    ) -> None:
        if type(value) is not list or len(value) != shape[0]:
            raise ValueError(f"context summary payload {name} shape is invalid")
        if len(shape) == 1:
            for leaf in value:
                leaf_validator(leaf, name)
            return
        for child in value:
            cls._validate_nested_values(child, shape[1:], name, leaf_validator)

    @staticmethod
    def _validate_float_value(value: Any, name: str) -> None:
        if type(value) not in (int, float) or not math.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0:
            raise ValueError(f"context summary payload {name} contains an invalid value")

    def _validate_evidence_value(self, value: Any, name: str) -> None:
        if type(value) is not int or not 0 <= value <= self.max_evidence:
            raise ValueError(f"context summary payload {name} contains an invalid count")


class SurpriseMemory(nn.Module):
    """Small opt-in causal EMA memory gated by a bounded surprise signal."""

    def __init__(
        self,
        width: int,
        *,
        memory_decay: float = 0.95,
        momentum_decay: float = 0.90,
        confidence_scale: float = DEFAULT_CONFIDENCE_SCALE,
        max_evidence: int = 1024,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_state_elements: int = DEFAULT_MAX_STATE_ELEMENTS,
    ) -> None:
        super().__init__()
        self.width = ContextSummary._require_positive_int(width, "width")
        self.memory_decay = self._require_decay(memory_decay, "memory_decay")
        self.momentum_decay = self._require_decay(momentum_decay, "momentum_decay")
        self.confidence_scale = ContextSummary._require_positive_float(
            confidence_scale, "confidence_scale"
        )
        self.max_evidence = ContextSummary._require_positive_int(max_evidence, "max_evidence")
        self.max_batch_size = ContextSummary._require_positive_int(max_batch_size, "max_batch_size")
        self.max_state_elements = ContextSummary._require_positive_int(
            max_state_elements, "max_state_elements"
        )
        self.register_buffer(
            "_device_anchor",
            torch.empty((), dtype=torch.get_default_dtype()),
            persistent=False,
        )

    def create_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> SurpriseMemoryState:
        """Create a bounded zero state on the module device."""
        batch_size = ContextSummary._require_positive_int(batch_size, "batch_size")
        self._validate_state_size(batch_size)
        target_device = self._resolve_device(device)
        target_dtype = ContextSummary._resolve_dtype(dtype)
        state = SurpriseMemoryState(
            value=torch.zeros(batch_size, self.width, device=target_device, dtype=target_dtype),
            momentum=torch.zeros(batch_size, self.width, device=target_device, dtype=target_dtype),
            evidence=torch.zeros(batch_size, device=target_device, dtype=torch.int64),
            step_index=0,
        )
        self._validate_state(state)
        return state

    def update(
        self,
        state: SurpriseMemoryState,
        values: Tensor,
        surprise: Tensor,
        valid_mask: Tensor | None = None,
        *,
        detach: bool = False,
    ) -> SurpriseMemoryState:
        """Apply one causal surprise-gated EMA and momentum update."""
        self._validate_state(state)
        self._validate_module_device(state)
        ContextSummary._validate_detach(detach)
        self._validate_values(values, state)
        self._validate_surprise(surprise, values)
        valid_mask = ContextSummary._validate_mask(
            valid_mask,
            (values.shape[0],),
            state.value.device,
            "valid_mask",
        )
        if state.step_index >= MAX_STEP_INDEX:
            raise ValueError("surprise memory step index reached its limit")

        bounded_values = torch.tanh(values)
        bounded_surprise = surprise.clamp(0.0, 1.0).unsqueeze(1)
        candidate_momentum = (
            self.momentum_decay * state.momentum
            + (1.0 - self.momentum_decay) * bounded_values * bounded_surprise
        ).clamp(-1.0, 1.0)
        candidate_value = (
            self.memory_decay * state.value
            + (1.0 - self.memory_decay) * candidate_momentum
        ).clamp(-1.0, 1.0)
        if valid_mask is None:
            next_value = candidate_value
            next_momentum = candidate_momentum
            next_evidence = (state.evidence + 1).clamp_max(self.max_evidence)
        else:
            next_value = torch.where(valid_mask.unsqueeze(1), candidate_value, state.value)
            next_momentum = torch.where(
                valid_mask.unsqueeze(1), candidate_momentum, state.momentum
            )
            next_evidence = (
                state.evidence + valid_mask.to(dtype=torch.int64)
            ).clamp_max(self.max_evidence)
        if detach:
            next_value = next_value.detach()
            next_momentum = next_momentum.detach()
            next_evidence = next_evidence.detach()
        return SurpriseMemoryState(
            value=next_value,
            momentum=next_momentum,
            evidence=next_evidence,
            step_index=state.step_index + 1,
        )

    def read(self, state: SurpriseMemoryState, *, detach: bool = False) -> SurpriseMemoryRead:
        """Read bounded memory with device-local confidence weighting."""
        self._validate_state(state)
        self._validate_module_device(state)
        ContextSummary._validate_detach(detach)
        evidence = state.evidence.to(dtype=state.value.dtype)
        confidence = (
            1.0 - torch.exp(-evidence / self.confidence_scale)
        ).clamp(0.0, 1.0)
        value = state.value * confidence.unsqueeze(1)
        result = SurpriseMemoryRead(
            value=value,
            confidence=confidence,
            evidence=state.evidence,
            mask=state.evidence > 0,
        )
        if detach:
            return SurpriseMemoryRead(
                value=result.value.detach(),
                confidence=result.confidence.detach(),
                evidence=result.evidence.detach(),
                mask=result.mask.detach(),
            )
        return result

    def reset(self, state: SurpriseMemoryState) -> SurpriseMemoryState:
        """Return an empty state preserving the input batch, device, and dtype."""
        self._validate_state(state)
        self._validate_module_device(state)
        return self.create_state(
            state.value.shape[0],
            device=state.value.device,
            dtype=state.value.dtype,
        )

    def _validate_module_device(self, state: SurpriseMemoryState) -> None:
        if state.value.device != self._device_anchor.device:
            raise ValueError("surprise memory module and state must use the same device")

    def _validate_state(self, state: SurpriseMemoryState) -> None:
        if not isinstance(state, SurpriseMemoryState):
            raise TypeError("state must be a SurpriseMemoryState")
        if not isinstance(state.value, Tensor) or not isinstance(state.momentum, Tensor):
            raise TypeError("surprise memory state fields must be tensors")
        if state.value.ndim != 2 or state.value.shape[1] != self.width:
            raise ValueError("surprise memory value must have shape [batch, width]")
        if state.momentum.shape != state.value.shape:
            raise ValueError("surprise memory momentum shape is invalid")
        if state.evidence.shape != (state.value.shape[0],):
            raise ValueError("surprise memory evidence shape is invalid")
        self._validate_state_size(state.value.shape[0])
        ContextSummary._validate_floating_dtype(state.value.dtype, "surprise memory value")
        if state.momentum.dtype != state.value.dtype:
            raise TypeError("surprise memory momentum must use the value dtype")
        if state.evidence.dtype != torch.int64:
            raise TypeError("surprise memory evidence must use int64")
        if state.value.device != state.momentum.device or state.value.device != state.evidence.device:
            raise ValueError("surprise memory state tensors must use the same device")
        if type(state.step_index) is not int or not 0 <= state.step_index <= MAX_STEP_INDEX:
            raise ValueError("surprise memory step index is invalid")
        _assert_tensor_condition(
            torch.isfinite(state.value).all(),
            "surprise memory value must be finite",
        )
        _assert_tensor_condition(
            state.value.abs().le(1.0).all(),
            "surprise memory value must stay within [-1, 1]",
        )
        _assert_tensor_condition(
            torch.isfinite(state.momentum).all(),
            "surprise memory momentum must be finite",
        )
        _assert_tensor_condition(
            state.momentum.abs().le(1.0).all(),
            "surprise memory momentum must stay within [-1, 1]",
        )
        _assert_tensor_condition(
            state.evidence.ge(0).logical_and(state.evidence.le(self.max_evidence)).all(),
            "surprise memory evidence is outside its limit",
        )

    def _validate_values(self, values: Tensor, state: SurpriseMemoryState) -> None:
        if not isinstance(values, Tensor):
            raise TypeError("values must be a tensor")
        if values.shape != state.value.shape:
            raise ValueError("values must have shape [batch, width]")
        ContextSummary._validate_floating_dtype(values.dtype, "surprise memory values")
        if values.dtype != state.value.dtype:
            raise TypeError("surprise memory values and state must use the same dtype")
        if values.device != state.value.device:
            raise ValueError("surprise memory values and state must use the same device")
        _assert_tensor_condition(torch.isfinite(values).all(), "surprise memory values must be finite")

    @staticmethod
    def _validate_surprise(surprise: Tensor, values: Tensor) -> None:
        if not isinstance(surprise, Tensor):
            raise TypeError("surprise must be a tensor")
        if surprise.ndim != 1 or surprise.shape[0] != values.shape[0]:
            raise ValueError("surprise must have shape [batch]")
        if (
            not torch.is_floating_point(surprise)
            or torch.is_complex(surprise)
            or surprise.dtype != values.dtype
        ):
            raise TypeError("surprise must be a real tensor with the values dtype")
        if surprise.device != values.device:
            raise ValueError("surprise and values must use the same device")
        _assert_tensor_condition(torch.isfinite(surprise).all(), "surprise must be finite")

    def _validate_state_size(self, batch_size: int) -> None:
        batch_size = ContextSummary._require_positive_int(batch_size, "batch_size")
        if batch_size > self.max_batch_size:
            raise ValueError("surprise memory batch size exceeds its limit")
        if batch_size * self.width * 2 > self.max_state_elements:
            raise ValueError("surprise memory state exceeds its element limit")

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is None:
            return self._device_anchor.device
        if not isinstance(device, (torch.device, str)):
            raise TypeError("device must be a torch.device or string")
        try:
            target_device = torch.device(device)
        except (RuntimeError, TypeError) as error:
            raise ValueError("surprise memory device is invalid") from error
        if target_device.type == "meta":
            raise ValueError("meta device is not supported for surprise memory state")
        if target_device != self._device_anchor.device:
            raise ValueError("surprise memory state device must match the module device")
        return target_device

    @staticmethod
    def _require_decay(value: float, name: str) -> float:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
        return float(value)


__all__ = [
    "ContextSummary",
    "ContextSummaryCost",
    "ContextSummaryRead",
    "ContextSummaryState",
    "SurpriseMemory",
    "SurpriseMemoryRead",
    "SurpriseMemoryState",
]
