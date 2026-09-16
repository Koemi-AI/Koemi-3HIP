from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Iterator, Sequence

import torch
from torch import Tensor


ResetValue = bool | int | float | complex
Shape = tuple[int, ...]
_UNSET = object()


@dataclass(frozen=True)
class StateBufferSpec:
    shape: Shape
    dtype: torch.dtype
    device: torch.device
    reset_value: ResetValue


@dataclass(frozen=True)
class StateBufferStatistics:
    allocations: int
    reuses: int
    resets: int
    resizes: int
    active_buffers: int
    active_bytes: int
    timing_enabled: bool
    allocation_seconds: float | None
    reset_seconds: float | None
    resize_seconds: float | None


@dataclass
class _StateBufferRecord:
    spec: StateBufferSpec
    tensor: Tensor


class StateBufferPool:
    """Owns fixed-layout tensors for opt-in HERM state staging.

    The pool is bound to one CPU or CUDA device. Each named buffer keeps exact
    shape, dtype, device, and reset-value metadata. It does not update
    ``KoemiState`` or implement HERM transitions.

    ``stream`` scopes allocation and reset work on a matching CUDA stream. No
    synchronization is performed; callers must order compute that reuses a
    buffer on the same stream or provide an external dependency.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        *,
        default_dtype: torch.dtype = torch.float32,
        measure_timings: bool = False,
    ) -> None:
        self.device = _normalize_device(device)
        self.default_dtype = _validate_dtype(default_dtype)
        if not isinstance(measure_timings, bool):
            raise TypeError("measure_timings must be a bool")
        self.measure_timings = measure_timings
        self._bound_device = (
            self.device
            if self.device.type == "cpu" or self.device.index is not None
            else None
        )
        self._buffers: dict[str, _StateBufferRecord] = {}
        self._allocations = 0
        self._reuses = 0
        self._resets = 0
        self._resizes = 0
        self._allocation_seconds = 0.0
        self._reset_seconds = 0.0
        self._resize_seconds = 0.0

    def acquire(
        self,
        name: str,
        shape: Sequence[int],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        reset_value: ResetValue | object = _UNSET,
        reset: bool = False,
        stream: torch.cuda.Stream | None = None,
    ) -> Tensor:
        """Return a reusable buffer, rejecting incompatible metadata.

        A first request allocates and initializes the tensor. Later requests
        must match the stored shape, dtype, device, and reset value exactly;
        call :meth:`resize` for an explicit shape change. ``reset=True``
        fills a reused buffer in-place before returning it.
        """
        _validate_name(name)
        if not isinstance(reset, bool):
            raise TypeError("reset must be a bool")
        normalized_shape = _normalize_shape(shape)
        record = self._buffers.get(name)
        if record is None:
            resolved_dtype = self.default_dtype if dtype is None else _validate_dtype(dtype)
            resolved_device = self._resolve_device(device, stream)
            resolved_reset_value = 0 if reset_value is _UNSET else _validate_reset_value(reset_value)
            requested_spec = StateBufferSpec(
                normalized_shape,
                resolved_dtype,
                resolved_device,
                resolved_reset_value,
            )
            tensor = self._allocate(requested_spec, stream=stream, is_resize=False)
            self._buffers[name] = _StateBufferRecord(
                StateBufferSpec(
                    normalized_shape,
                    resolved_dtype,
                    tensor.device,
                    resolved_reset_value,
                ),
                tensor,
            )
            return tensor

        resolved_dtype = record.spec.dtype if dtype is None else _validate_dtype(dtype)
        resolved_device = self._resolve_device(
            record.spec.device if device is None else device,
            stream,
        )
        resolved_reset_value = (
            record.spec.reset_value
            if reset_value is _UNSET
            else _validate_reset_value(reset_value)
        )
        mismatches: list[str] = []
        if normalized_shape != record.spec.shape:
            mismatches.append(f"shape {normalized_shape} != {record.spec.shape}")
        if resolved_dtype != record.spec.dtype:
            mismatches.append(f"dtype {resolved_dtype} != {record.spec.dtype}")
        if resolved_device != record.spec.device:
            mismatches.append(f"device {resolved_device} != {record.spec.device}")
        if resolved_reset_value != record.spec.reset_value:
            mismatches.append("reset_value differs")
        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(
                f"buffer '{name}' is incompatible: {details}; use resize for shape changes "
                "or a different name for dtype, device, or reset-value changes"
            )

        self._validate_stream_for_device(stream, record.spec.device)
        self._reuses += 1
        if reset:
            self._reset_records((record,), stream)
        return record.tensor

    def acquire_like(
        self,
        name: str,
        reference: Tensor,
        *,
        reset: bool = False,
        stream: torch.cuda.Stream | None = None,
    ) -> Tensor:
        """Acquire a buffer with a tensor's metadata without copying it."""
        if not isinstance(reference, Tensor):
            raise TypeError("reference must be a torch.Tensor")
        return self.acquire(
            name,
            tuple(reference.shape),
            dtype=reference.dtype,
            device=reference.device,
            reset=reset,
            stream=stream,
        )

    def resize(
        self,
        name: str,
        shape: Sequence[int],
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> Tensor:
        """Replace a named buffer with an explicit shape, keeping its other policies."""
        _validate_name(name)
        record = self._buffers.get(name)
        if record is None:
            raise KeyError(f"unknown state buffer '{name}'")
        normalized_shape = _normalize_shape(shape)
        self._validate_stream_for_device(stream, record.spec.device)
        if normalized_shape == record.spec.shape:
            return record.tensor

        requested_spec = StateBufferSpec(
            normalized_shape,
            record.spec.dtype,
            record.spec.device,
            record.spec.reset_value,
        )
        tensor = self._allocate(requested_spec, stream=stream, is_resize=True)
        self._buffers[name] = _StateBufferRecord(
            StateBufferSpec(
                normalized_shape,
                record.spec.dtype,
                tensor.device,
                record.spec.reset_value,
            ),
            tensor,
        )
        return tensor

    def reset(
        self,
        name: str | None = None,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Fill one or all buffers with their declared reset values in-place."""
        if name is None:
            records = tuple(self._buffers.values())
            if not records:
                if stream is not None:
                    self._resolve_device(None, stream)
                return
        else:
            _validate_name(name)
            record = self._buffers.get(name)
            if record is None:
                raise KeyError(f"unknown state buffer '{name}'")
            records = (record,)
        self._reset_records(records, stream)

    def specification(self, name: str) -> StateBufferSpec:
        """Return immutable metadata for a named buffer."""
        _validate_name(name)
        record = self._buffers.get(name)
        if record is None:
            raise KeyError(f"unknown state buffer '{name}'")
        return record.spec

    def statistics(self, *, include_timing: bool = False) -> StateBufferStatistics:
        """Return allocation counters and optional host enqueue timings."""
        if not isinstance(include_timing, bool):
            raise TypeError("include_timing must be a bool")
        allocation_seconds = (
            self._allocation_seconds if include_timing and self.measure_timings else None
        )
        reset_seconds = self._reset_seconds if include_timing and self.measure_timings else None
        resize_seconds = self._resize_seconds if include_timing and self.measure_timings else None
        active_bytes = sum(
            record.tensor.numel() * record.tensor.element_size()
            for record in self._buffers.values()
        )
        return StateBufferStatistics(
            allocations=self._allocations,
            reuses=self._reuses,
            resets=self._resets,
            resizes=self._resizes,
            active_buffers=len(self._buffers),
            active_bytes=active_bytes,
            timing_enabled=self.measure_timings,
            allocation_seconds=allocation_seconds,
            reset_seconds=reset_seconds,
            resize_seconds=resize_seconds,
        )

    def _allocate(
        self,
        spec: StateBufferSpec,
        *,
        stream: torch.cuda.Stream | None,
        is_resize: bool,
    ) -> Tensor:
        started_at = time.perf_counter() if self.measure_timings else None
        with _stream_scope(stream):
            tensor = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            tensor.fill_(spec.reset_value)
        self._bind_device(tensor.device)
        self._allocations += 1
        if started_at is not None:
            elapsed_seconds = time.perf_counter() - started_at
            self._allocation_seconds += elapsed_seconds
            if is_resize:
                self._resize_seconds += elapsed_seconds
        if is_resize:
            self._resizes += 1
        return tensor

    def _reset_records(
        self,
        records: Sequence[_StateBufferRecord],
        stream: torch.cuda.Stream | None,
    ) -> None:
        if not records:
            return
        self._validate_stream_for_device(stream, records[0].spec.device)
        started_at = time.perf_counter() if self.measure_timings else None
        with _stream_scope(stream):
            for record in records:
                record.tensor.fill_(record.spec.reset_value)
        self._resets += len(records)
        if started_at is not None:
            self._reset_seconds += time.perf_counter() - started_at

    def _resolve_device(
        self,
        device: torch.device | str | None,
        stream: torch.cuda.Stream | None,
    ) -> torch.device:
        if device is None:
            requested_device = self._bound_device or self.device
        else:
            requested_device = _normalize_device(device)
        stream_device = _stream_device(stream) if stream is not None else None
        if stream_device is not None:
            if requested_device.index is None:
                requested_device = stream_device
            elif requested_device != stream_device:
                raise ValueError(
                    f"stream device {stream_device} does not match buffer device {requested_device}"
                )
        self._validate_pool_device(requested_device)
        return requested_device

    def _validate_pool_device(self, requested_device: torch.device) -> None:
        if requested_device.type != self.device.type:
            raise ValueError(
                f"buffer device {requested_device} does not match pool device {self.device}; no implicit copy"
            )
        if self.device.index is not None and requested_device != self.device:
            raise ValueError(
                f"buffer device {requested_device} does not match pool device {self.device}; no implicit copy"
            )
        if self._bound_device is not None and requested_device != self._bound_device:
            raise ValueError(
                f"buffer device {requested_device} does not match bound pool device {self._bound_device}; no implicit copy"
            )

    def _validate_stream_for_device(
        self,
        stream: torch.cuda.Stream | None,
        device: torch.device,
    ) -> None:
        if stream is None:
            return
        stream_device = _stream_device(stream)
        if device.type != "cuda":
            raise ValueError("a CUDA stream cannot be used with a CPU state buffer")
        if stream_device != device:
            raise ValueError(f"stream device {stream_device} does not match buffer device {device}")

    def _bind_device(self, actual_device: torch.device) -> None:
        if self.device.type != "cuda":
            return
        if self._bound_device is None:
            self._bound_device = actual_device
            return
        if actual_device != self._bound_device:
            raise RuntimeError(
                f"allocated device {actual_device} changed the pool binding from {self._bound_device}"
            )


@contextmanager
def _stream_scope(stream: torch.cuda.Stream | None) -> Iterator[None]:
    if stream is None:
        yield
        return
    with torch.cuda.stream(stream):
        yield


def _stream_device(stream: torch.cuda.Stream) -> torch.device:
    if not isinstance(stream, torch.cuda.Stream):
        raise TypeError("stream must be a torch.cuda.Stream")
    return torch.device(stream.device)


def _normalize_device(device: torch.device | str) -> torch.device:
    try:
        normalized_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid device {device!r}") from error
    if normalized_device.type not in {"cpu", "cuda"}:
        raise ValueError("state buffers support only CPU and CUDA devices")
    return normalized_device


def _validate_dtype(dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    return dtype


def _normalize_shape(shape: Sequence[int]) -> Shape:
    try:
        normalized_shape = tuple(shape)
    except TypeError as error:
        raise TypeError("shape must be a sequence of integers") from error
    for dimension in normalized_shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("shape dimensions must be integers")
        if dimension < 0:
            raise ValueError("shape dimensions must be non-negative")
    return normalized_shape


def _validate_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("buffer name must be a string")
    if not name.strip():
        raise ValueError("buffer name must not be empty")


def _validate_reset_value(reset_value: object) -> ResetValue:
    if not isinstance(reset_value, (bool, int, float, complex)):
        raise TypeError("reset_value must be a Python scalar")
    return reset_value


__all__ = ["StateBufferPool", "StateBufferSpec", "StateBufferStatistics"]
