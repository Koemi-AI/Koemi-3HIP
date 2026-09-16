from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor, nn


PrecisionRequest: TypeAlias = Literal["auto", "fp32", "bf16", "fp16"]
ResolvedPrecision: TypeAlias = Literal["fp32", "bf16", "fp16"]
TensorSource: TypeAlias = Tensor | Iterable[Tensor] | Mapping[object, Tensor]
GradientSource: TypeAlias = (
    Mapping[object, Tensor | None] | Iterable[Tensor | None] | Tensor | None
)

_VALID_PRECISION_REQUESTS = frozenset(("auto", "fp32", "bf16", "fp16"))
_MIXED_PRECISION_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass(frozen=True)
class PrecisionPolicy:
    device: torch.device
    requested: PrecisionRequest
    precision: ResolvedPrecision
    autocast_dtype: torch.dtype | None
    use_grad_scaler: bool
    enable_tf32: bool | None

    @property
    def resolved(self) -> ResolvedPrecision:
        return self.precision

    @property
    def autocast_enabled(self) -> bool:
        return self.autocast_dtype is not None


@dataclass(frozen=True)
class PrecisionHealth:
    device: torch.device | None
    tensor_count: int
    finite_tensor_count: int
    non_finite_tensor_count: int
    non_finite_value_count: int
    max_abs_value: float | None
    gradient_tensor_count: int | None
    finite_gradient_tensor_count: int | None
    non_finite_gradient_tensor_count: int | None
    non_finite_gradient_value_count: int | None
    max_abs_gradient: float | None
    memory_allocated_bytes: int | None
    memory_reserved_bytes: int | None
    max_memory_allocated_bytes: int | None
    max_memory_reserved_bytes: int | None

    @property
    def all_finite(self) -> bool:
        return self.non_finite_tensor_count == 0

    @property
    def gradients_finite(self) -> bool | None:
        if self.gradient_tensor_count is None:
            return None
        return self.non_finite_gradient_tensor_count == 0


@dataclass(frozen=True)
class PrecisionComparison:
    output_allclose: bool
    output_max_abs_error: float
    output_max_relative_error: float
    gradient_allclose: bool | None
    gradient_max_abs_error: float | None
    gradient_max_relative_error: float | None
    failure_reason: str | None = None

    @property
    def passed(self) -> bool:
        return self.output_allclose and self.gradient_allclose is not False


@dataclass(frozen=True)
class _TensorSummary:
    tensor_count: int
    finite_tensor_count: int
    non_finite_value_count: int
    max_abs_value: float | None


def resolve_precision(
    device: torch.device | str,
    requested: str = "auto",
    *,
    enable_tf32: bool | None = None,
) -> PrecisionPolicy:
    """Resolve a validated precision policy without changing global runtime state.

    `device` accepts CPU or CUDA devices. `requested` accepts `auto`, `fp32`,
    `bf16` and `fp16`; `auto` selects FP32 on CPU and BF16 or FP16 on CUDA.
    `enable_tf32` is a CUDA-only opt-in and is applied only while the returned
    policy's context is active. This function performs no model or tensor move.
    """
    normalized_device = _normalize_device(device)
    normalized_request = _validate_precision_request(requested)
    normalized_tf32 = _validate_tf32_setting(normalized_device, enable_tf32)
    _validate_requested_support(normalized_device, normalized_request)
    resolved_precision = _resolve_requested_precision(normalized_device, normalized_request)
    return PrecisionPolicy(
        normalized_device,
        normalized_request,
        resolved_precision,
        _MIXED_PRECISION_DTYPES.get(resolved_precision),
        normalized_device.type == "cuda" and resolved_precision == "fp16",
        normalized_tf32,
    )


def validate_precision_support(
    device: torch.device | str,
    requested: str,
    *,
    enable_tf32: bool | None = None,
) -> None:
    """Validate a device and precision request, raising before model execution.

    CPU accepts only `auto` and `fp32`, with `auto` resolving to FP32. CUDA
    must be available, explicit BF16 requires CUDA BF16 support, and TF32
    requires a CUDA device with compute capability 8.0 or newer.
    """
    normalized_device = _normalize_device(device)
    normalized_request = _validate_precision_request(requested)
    _validate_tf32_setting(normalized_device, enable_tf32)
    _validate_requested_support(normalized_device, normalized_request)


@contextmanager
def autocast_context(policy: PrecisionPolicy):
    """Yield an AMP/TF32 context described by `policy` and restore backend flags.

    FP32 policies yield a no-op context. CUDA TF32 flags are changed only when
    `policy.enable_tf32` is explicitly boolean and are restored on exit. The
    context never issues a device-wide synchronization barrier.
    """
    if not isinstance(policy, PrecisionPolicy):
        raise TypeError("policy must be a PrecisionPolicy")

    previous_tf32_flags: tuple[bool, bool] | None = None
    if policy.device.type == "cuda" and policy.enable_tf32 is not None:
        previous_tf32_flags = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        torch.backends.cuda.matmul.allow_tf32 = policy.enable_tf32
        torch.backends.cudnn.allow_tf32 = policy.enable_tf32

    try:
        autocast_manager = (
            nullcontext()
            if policy.autocast_dtype is None
            else torch.autocast(device_type=policy.device.type, dtype=policy.autocast_dtype)
        )
        with autocast_manager:
            yield
    finally:
        if previous_tf32_flags is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32_flags[0]
            torch.backends.cudnn.allow_tf32 = previous_tf32_flags[1]


def collect_precision_health(
    tensors: TensorSource = (),
    parameters: nn.Module | Tensor | Iterable[Tensor] | None = None,
    *,
    device: torch.device | str | None = None,
) -> PrecisionHealth:
    """Aggregate finiteness, gradients and allocator counters for one probe.

    `tensors` may be a tensor, nested tuple/list/mapping or tensor iterable.
    `parameters` may be a module, one tensor or tensor iterable; only present
    gradients are measured. CUDA allocator queries are read-only and no explicit
    device-wide synchronization barrier is issued. CPU memory fields are `None`.
    """
    output_tensors = _flatten_tensors(tensors, "tensors")
    parameter_tensors = _flatten_parameters(parameters)
    gradient_tensors = tuple(
        parameter.grad for parameter in parameter_tensors if parameter.grad is not None
    )
    report_device = _resolve_report_device(
        device,
        output_tensors,
        parameter_tensors,
    )
    output_summary = _summarize_tensors(output_tensors)
    gradient_summary = _summarize_tensors(gradient_tensors) if parameters is not None else None
    memory_values = _read_cuda_memory(report_device)
    return PrecisionHealth(
        report_device,
        output_summary.tensor_count,
        output_summary.finite_tensor_count,
        output_summary.tensor_count - output_summary.finite_tensor_count,
        output_summary.non_finite_value_count,
        output_summary.max_abs_value,
        None if gradient_summary is None else gradient_summary.tensor_count,
        None if gradient_summary is None else gradient_summary.finite_tensor_count,
        None if gradient_summary is None else gradient_summary.tensor_count - gradient_summary.finite_tensor_count,
        None if gradient_summary is None else gradient_summary.non_finite_value_count,
        None if gradient_summary is None else gradient_summary.max_abs_value,
        memory_values[0],
        memory_values[1],
        memory_values[2],
        memory_values[3],
    )


def compare_against_fp32(
    fp32_output: TensorSource,
    candidate_output: TensorSource,
    fp32_gradients: GradientSource = None,
    candidate_gradients: GradientSource = None,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> PrecisionComparison:
    """Compare mixed-precision output and optional gradients to an FP32 run.

    The caller executes both runs and supplies detached tensors; gradient
    mappings may include `None` to detect a missing gradient. Shapes and device
    placement must match. Non-finite values always fail the comparison.
    """
    _validate_tolerance(rtol, "rtol")
    _validate_tolerance(atol, "atol")
    output_reference = _flatten_tensors(fp32_output, "fp32_output")
    output_candidate = _flatten_tensors(candidate_output, "candidate_output")
    output_result = _compare_tensor_sequences(
        output_reference,
        output_candidate,
        rtol,
        atol,
        "output",
    )
    gradient_result = _compare_gradient_sources(
        fp32_gradients,
        candidate_gradients,
        rtol,
        atol,
    )
    reasons = [reason for reason in (output_result[3], gradient_result[3]) if reason]
    return PrecisionComparison(
        output_result[0],
        output_result[1],
        output_result[2],
        None if gradient_result[0] is None else gradient_result[0],
        None if gradient_result[1] is None else gradient_result[1],
        None if gradient_result[2] is None else gradient_result[2],
        "; ".join(reasons) if reasons else None,
    )


def _normalize_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device):
        normalized_device = device
    elif isinstance(device, str):
        try:
            normalized_device = torch.device(device)
        except (RuntimeError, TypeError) as error:
            raise ValueError("device must be a valid CPU or CUDA device") from error
    else:
        raise TypeError("device must be a torch.device or string")
    if normalized_device.type not in {"cpu", "cuda"}:
        raise ValueError("device type must be CPU or CUDA")
    if normalized_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA precision was requested but CUDA is unavailable")
        if normalized_device.index is not None:
            device_count = torch.cuda.device_count()
            if not 0 <= normalized_device.index < device_count:
                raise ValueError(
                    f"CUDA device index {normalized_device.index} is outside the available range"
                )
    return normalized_device


def _validate_precision_request(requested: str) -> PrecisionRequest:
    if not isinstance(requested, str) or requested not in _VALID_PRECISION_REQUESTS:
        allowed_values = ", ".join(sorted(_VALID_PRECISION_REQUESTS))
        raise ValueError(f"requested precision must be one of: {allowed_values}")
    return cast(PrecisionRequest, requested)


def _validate_tf32_setting(device: torch.device, enable_tf32: bool | None) -> bool | None:
    if enable_tf32 is not None and not isinstance(enable_tf32, bool):
        raise ValueError("enable_tf32 must be true, false or none")
    if enable_tf32 is True:
        if device.type != "cuda":
            raise ValueError("TF32 requires CUDA; CPU remains in FP32")
        capability_major, _ = torch.cuda.get_device_capability(device)
        if capability_major < 8:
            raise RuntimeError("TF32 requires CUDA compute capability 8.0 or newer")
    return enable_tf32


def _validate_requested_support(device: torch.device, requested: PrecisionRequest) -> None:
    if device.type == "cpu" and requested in {"bf16", "fp16"}:
        raise ValueError(f"{requested} precision requires CUDA; CPU remains in FP32")
    if device.type == "cuda" and requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 precision is unsupported by the selected CUDA device")


def _resolve_requested_precision(
    device: torch.device,
    requested: PrecisionRequest,
) -> ResolvedPrecision:
    if requested != "auto":
        return requested
    if device.type == "cpu":
        return "fp32"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def _flatten_tensors(value: object, name: str) -> tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        return (value,)
    if isinstance(value, Mapping):
        flattened: list[Tensor] = []
        for nested_value in value.values():
            flattened.extend(_flatten_tensors(nested_value, name))
        return tuple(flattened)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        flattened = []
        for nested_value in value:
            flattened.extend(_flatten_tensors(nested_value, name))
        return tuple(flattened)
    raise TypeError(f"{name} must contain only tensors")


def _flatten_parameters(
    parameters: nn.Module | Tensor | Iterable[Tensor] | None,
) -> tuple[Tensor, ...]:
    if parameters is None:
        return ()
    if isinstance(parameters, nn.Module):
        parameter_values = tuple(parameters.parameters())
    elif isinstance(parameters, Tensor):
        parameter_values = (parameters,)
    else:
        parameter_values = tuple(parameters)
    if not all(isinstance(parameter, Tensor) for parameter in parameter_values):
        raise TypeError("parameters must contain only tensors or an nn.Module")
    return parameter_values


def _resolve_report_device(
    requested_device: torch.device | str | None,
    output_tensors: tuple[Tensor, ...],
    parameter_tensors: tuple[Tensor, ...],
) -> torch.device | None:
    if requested_device is not None:
        return _normalize_device(requested_device)
    tensor_devices = tuple(tensor.device for tensor in output_tensors + parameter_tensors)
    for tensor_device in tensor_devices:
        if tensor_device.type == "cuda":
            return tensor_device
    return tensor_devices[0] if tensor_devices else None


def _summarize_tensors(tensors: tuple[Tensor, ...]) -> _TensorSummary:
    grouped_reductions: dict[torch.device, list[tuple[Tensor, Tensor, Tensor]]] = {}
    finite_empty_tensor_count = 0
    for tensor in tensors:
        if tensor.layout != torch.strided:
            raise ValueError("precision diagnostics support dense strided tensors only")
        if tensor.numel() == 0:
            finite_empty_tensor_count += 1
            continue
        detached_tensor = tensor.detach()
        finite_mask = torch.isfinite(detached_tensor)
        non_finite_value_count = (~finite_mask).sum(dtype=torch.int64)
        finite_tensor_flag = finite_mask.all().to(dtype=torch.int64)
        absolute_values = detached_tensor.abs().to(dtype=torch.float32)
        finite_absolute_values = torch.where(
            finite_mask,
            absolute_values,
            torch.zeros_like(absolute_values),
        )
        maximum_absolute_value = finite_absolute_values.amax()
        grouped_reductions.setdefault(tensor.device, []).append(
            (finite_tensor_flag, non_finite_value_count, maximum_absolute_value)
        )

    finite_tensor_count = finite_empty_tensor_count
    non_finite_value_count = 0
    maximum_values: list[float] = []
    for reductions in grouped_reductions.values():
        finite_tensor_count += int(
            torch.stack([reduction[0] for reduction in reductions]).sum().item()
        )
        non_finite_value_count += int(
            torch.stack([reduction[1] for reduction in reductions]).sum().item()
        )
        maximum_values.append(
            float(torch.stack([reduction[2] for reduction in reductions]).amax().item())
        )

    max_abs_value = None if not maximum_values else max(maximum_values)
    if non_finite_value_count > 0:
        max_abs_value = math.inf
    return _TensorSummary(
        len(tensors),
        finite_tensor_count,
        non_finite_value_count,
        max_abs_value,
    )


def _read_cuda_memory(
    device: torch.device | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    if device is None or device.type != "cuda":
        return None, None, None, None
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.memory_reserved(device)),
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _validate_tolerance(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _compare_tensor_sequences(
    references: tuple[Tensor, ...],
    candidates: tuple[Tensor, ...],
    rtol: float,
    atol: float,
    label: str,
) -> tuple[bool, float, float, str | None]:
    if len(references) != len(candidates):
        return False, math.inf, math.inf, f"{label} tensor count differs"

    allclose = True
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    failure_reason: str | None = None
    for index, (reference, candidate) in enumerate(zip(references, candidates)):
        if reference.shape != candidate.shape:
            allclose = False
            failure_reason = f"{label} shape differs at index {index}"
            maximum_absolute_error = math.inf
            maximum_relative_error = math.inf
            continue
        if reference.device != candidate.device:
            allclose = False
            failure_reason = f"{label} device differs at index {index}"
            maximum_absolute_error = math.inf
            maximum_relative_error = math.inf
            continue
        reference_values = reference.detach().to(dtype=torch.float32)
        candidate_values = candidate.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(reference_values).all().item()) or not bool(
            torch.isfinite(candidate_values).all().item()
        ):
            allclose = False
            failure_reason = f"{label} contains non-finite values at index {index}"
            maximum_absolute_error = math.inf
            maximum_relative_error = math.inf
            continue
        if reference_values.numel() == 0:
            continue
        absolute_error = (candidate_values - reference_values).abs()
        denominator = reference_values.abs()
        relative_error = torch.where(
            denominator > torch.finfo(torch.float32).tiny,
            absolute_error / denominator.clamp_min(torch.finfo(torch.float32).tiny),
            absolute_error,
        )
        maximum_absolute_error = max(maximum_absolute_error, float(absolute_error.amax().item()))
        maximum_relative_error = max(maximum_relative_error, float(relative_error.amax().item()))
        tensor_allclose = bool(
            torch.allclose(reference_values, candidate_values, rtol=rtol, atol=atol)
        )
        allclose = allclose and tensor_allclose
        if not tensor_allclose and failure_reason is None:
            failure_reason = f"{label} exceeds tolerance at index {index}"
    return allclose, maximum_absolute_error, maximum_relative_error, failure_reason


def _normalize_gradient_source(
    source: GradientSource,
) -> tuple[str, tuple[object, Tensor | None]] | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return "mapping", tuple(source.items())
    if isinstance(source, Tensor):
        return "sequence", ((0, source),)
    if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        return "sequence", tuple(enumerate(source))
    raise TypeError("gradients must be a mapping, tensor iterable or None")


def _compare_gradient_sources(
    reference_source: GradientSource,
    candidate_source: GradientSource,
    rtol: float,
    atol: float,
) -> tuple[bool | None, float | None, float | None, str | None]:
    reference = _normalize_gradient_source(reference_source)
    candidate = _normalize_gradient_source(candidate_source)
    if reference is None and candidate is None:
        return None, None, None, None
    if reference is None or candidate is None:
        return False, math.inf, math.inf, "gradient collection is missing from one run"
    if reference[0] != candidate[0]:
        return False, math.inf, math.inf, "gradient collection kinds differ"

    if reference[0] == "mapping":
        reference_values = dict(reference[1])
        candidate_values = dict(candidate[1])
        if set(reference_values) != set(candidate_values):
            return False, math.inf, math.inf, "gradient parameter names differ"
        gradient_pairs = tuple(
            (name, reference_values[name], candidate_values[name]) for name in reference_values
        )
    else:
        if len(reference[1]) != len(candidate[1]):
            return False, math.inf, math.inf, "gradient tensor count differs"
        gradient_pairs = tuple(
            (index, reference_value, candidate_value)
            for index, ((_, reference_value), (_, candidate_value)) in enumerate(
                zip(reference[1], candidate[1])
            )
        )

    allclose = True
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    failure_reason: str | None = None
    for name, reference_value, candidate_value in gradient_pairs:
        if reference_value is None or candidate_value is None:
            if reference_value is not None or candidate_value is not None:
                allclose = False
                failure_reason = f"gradient presence differs for {name}"
            continue
        result = _compare_tensor_sequences(
            (reference_value,),
            (candidate_value,),
            rtol,
            atol,
            f"gradient {name}",
        )
        allclose = allclose and result[0]
        maximum_absolute_error = max(maximum_absolute_error, result[1])
        maximum_relative_error = max(maximum_relative_error, result[2])
        if failure_reason is None and result[3] is not None:
            failure_reason = result[3]
    return allclose, maximum_absolute_error, maximum_relative_error, failure_reason


__all__ = [
    "PrecisionComparison",
    "PrecisionHealth",
    "PrecisionPolicy",
    "autocast_context",
    "collect_precision_health",
    "compare_against_fp32",
    "resolve_precision",
    "validate_precision_support",
]
