from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor


__all__ = [
    "CudaScanDiagnostics",
    "CudaScanUnavailableError",
    "CudaScanValidationError",
    "cuda_affine_scan",
    "diagnose_cuda_affine_scan",
]


_SUPPORTED_CUDA_DTYPES: tuple[torch.dtype, ...] = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


class CudaScanUnavailableError(RuntimeError):
    """Raised when the opt-in scan cannot run because CUDA is unavailable."""


class CudaScanValidationError(ValueError):
    """Raised when scan tensors or options violate the CUDA scan contract."""


@dataclass(frozen=True)
class CudaScanDiagnostics:
    """Result and measurements returned by the CUDA scan diagnostic API."""

    states: Tensor
    elapsed_seconds: float
    token_count: int
    chunk_count: int


def cuda_affine_scan(
    retention: Tensor,
    increment: Tensor,
    initial: Tensor | None = None,
    *,
    chunk_size: int | None = None,
) -> Tensor:
    """Run the affine scan using PyTorch operations on CUDA tensors only.

    Parameters are CUDA tensors with shape `[batch, sequence, ...]`; `retention`
    may broadcast over the state dimensions, and `initial` has shape
    `[batch, ...]`. Omitted `initial` means an all-zero initial state.
    `chunk_size` limits each parallel scan segment while carrying its final
    state into the next segment.

    Returns the scanned states with the same shape, device, and dtype as
    `increment`. The function never moves tensors or falls back to CPU.

    Raises `CudaScanValidationError` for invalid tensors or options, and
    `CudaScanUnavailableError` when CUDA is unavailable.
    """

    sequence_length, _, _ = _validate_inputs(retention, increment, initial, chunk_size)
    return _run_scan(retention, increment, initial, chunk_size, sequence_length)


def diagnose_cuda_affine_scan(
    retention: Tensor,
    increment: Tensor,
    initial: Tensor | None = None,
    *,
    chunk_size: int | None = None,
) -> CudaScanDiagnostics:
    """Run the CUDA scan and report synchronized elapsed time and work counts.

    The returned `token_count` is `batch * sequence`, and `chunk_count` is the
    number of scan segments selected by `chunk_size`. Timing synchronizes the
    CUDA device before and after the scan so asynchronous launches are counted.
    Raises the same validation and availability errors as `cuda_affine_scan`.
    """

    sequence_length, chunk_count, device = _validate_inputs(retention, increment, initial, chunk_size)
    torch.cuda.synchronize(device)
    started_at = perf_counter()
    states = _run_scan(retention, increment, initial, chunk_size, sequence_length)
    torch.cuda.synchronize(device)
    elapsed_seconds = perf_counter() - started_at
    return CudaScanDiagnostics(
        states=states,
        elapsed_seconds=elapsed_seconds,
        token_count=retention.shape[0] * sequence_length,
        chunk_count=chunk_count,
    )


def _validate_inputs(
    retention: Tensor,
    increment: Tensor,
    initial: Tensor | None,
    chunk_size: int | None,
) -> tuple[int, int, torch.device]:
    _validate_tensor("retention", retention)
    _validate_tensor("increment", increment)
    if initial is not None:
        _validate_tensor("initial", initial)

    if retention.ndim < 2 or increment.ndim < 2:
        raise CudaScanValidationError("retention and increment must have at least two dimensions")
    if retention.ndim != increment.ndim:
        raise CudaScanValidationError("retention and increment must have the same rank")
    if retention.shape[:2] != increment.shape[:2]:
        raise CudaScanValidationError("retention and increment must have the same batch and sequence dimensions")
    try:
        broadcast_shape = torch.broadcast_shapes(retention.shape, increment.shape)
    except RuntimeError as error:
        raise CudaScanValidationError("retention must broadcast to increment's shape") from error
    if broadcast_shape != increment.shape:
        raise CudaScanValidationError("retention must broadcast to increment's shape")

    tensor_values = (retention, increment) if initial is None else (retention, increment, initial)
    tensor_dtypes = {value.dtype for value in tensor_values}
    unsupported_dtypes = tensor_dtypes.difference(_SUPPORTED_CUDA_DTYPES)
    if unsupported_dtypes:
        names = ", ".join(str(dtype) for dtype in sorted(unsupported_dtypes, key=str))
        raise CudaScanValidationError(f"unsupported scan dtype(s): {names}")
    if len(tensor_dtypes) != 1:
        raise CudaScanValidationError("retention, increment and initial must have the same dtype")

    if initial is not None:
        expected_initial_shape = (increment.shape[0],) + tuple(increment.shape[2:])
        if initial.shape != expected_initial_shape:
            raise CudaScanValidationError(
                f"initial must have shape {expected_initial_shape}, got {tuple(initial.shape)}"
            )

    if chunk_size is not None and (isinstance(chunk_size, bool) or not isinstance(chunk_size, int)):
        raise CudaScanValidationError("chunk_size must be a positive integer or None")
    if chunk_size is not None and chunk_size < 1:
        raise CudaScanValidationError("chunk_size must be a positive integer or None")

    if not torch.cuda.is_available():
        raise CudaScanUnavailableError("CUDA affine scan is unavailable: torch.cuda.is_available() is False")

    scan_tensors = (retention, increment) if initial is None else (retention, increment, initial)
    non_cuda_tensors = [value for value in scan_tensors if value.device.type != "cuda"]
    if non_cuda_tensors:
        devices = ", ".join(str(value.device) for value in non_cuda_tensors)
        raise CudaScanValidationError(f"all scan tensors must be on CUDA; found {devices}")
    scan_devices = {value.device for value in scan_tensors}
    if len(scan_devices) != 1:
        devices = ", ".join(sorted(str(device) for device in scan_devices))
        raise CudaScanValidationError(f"all scan tensors must use one CUDA device; found {devices}")

    sequence_length = increment.shape[1]
    resolved_chunk_size = sequence_length if chunk_size is None else min(chunk_size, sequence_length)
    chunk_count = 0 if sequence_length == 0 else (sequence_length + resolved_chunk_size - 1) // resolved_chunk_size
    return sequence_length, chunk_count, increment.device


def _validate_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _run_scan(
    retention: Tensor,
    increment: Tensor,
    initial: Tensor | None,
    chunk_size: int | None,
    sequence_length: int,
) -> Tensor:
    if sequence_length == 0:
        return increment.new_empty(increment.shape)

    resolved_chunk_size = sequence_length if chunk_size is None else min(chunk_size, sequence_length)
    current_state = initial
    scanned_chunks: list[Tensor] = []
    for start in range(0, sequence_length, resolved_chunk_size):
        end = min(start + resolved_chunk_size, sequence_length)
        chunk_states = _scan_chunk(retention[:, start:end], increment[:, start:end], current_state)
        scanned_chunks.append(chunk_states)
        current_state = chunk_states[:, -1]
    if len(scanned_chunks) == 1:
        return scanned_chunks[0]
    return torch.cat(scanned_chunks, dim=1)


def _scan_chunk(retention: Tensor, increment: Tensor, initial: Tensor | None) -> Tensor:
    sequence_length = increment.shape[1]
    coefficient = retention
    value = increment
    offset = 1
    while offset < sequence_length:
        shifted_coefficient = _shift_with_fill(coefficient, offset, 1.0)
        shifted_value = _shift_with_fill(value, offset, 0.0)
        value = coefficient * shifted_value + value
        coefficient = coefficient * shifted_coefficient
        offset *= 2
    if initial is None:
        return value
    return value + coefficient * initial.unsqueeze(1)


def _shift_with_fill(values: Tensor, offset: int, fill_value: float) -> Tensor:
    length = values.shape[1]
    if offset >= length:
        return torch.full_like(values, fill_value)
    leading = torch.full_like(values[:, :offset], fill_value)
    return torch.cat((leading, values[:, : length - offset]), dim=1)
