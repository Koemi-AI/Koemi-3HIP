from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from koemi.configuration.settings import PAD_TOKEN_ID
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.cache import CachedPrefixState, DiskMappingCache, WarmTokenCache
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel
from koemi.model.state import KoemiState
from koemi.runtime.bulk_prefix_cache import BulkPrefixCache


_VARIABLE_LENGTH_STATE_FIELDS = frozenset(
    {
        "local_keys",
        "local_values",
        "local_valid",
        "salient_keys",
        "salient_values",
        "salient_valid",
    }
)


@dataclass(frozen=True)
class PromptEvaluation:
    last_logits: Tensor
    state: KoemiState
    processed_tokens: int
    reused_prefix_tokens: int


@dataclass(frozen=True)
class PrefillRequest:
    request_id: str
    input_ids: Tensor
    initial_state: KoemiState | None = None

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        normalized_input_ids = _normalize_prompt_input(self.input_ids)
        object.__setattr__(self, "input_ids", normalized_input_ids)
        if self.initial_state is not None:
            _validate_single_state(self.initial_state, normalized_input_ids.device, "initial_state")


@dataclass(frozen=True)
class PrefillResult:
    request_id: str
    last_logits: Tensor
    state: KoemiState
    processed_tokens: int
    reused_prefix_tokens: int


@dataclass(frozen=True)
class DecodeRequest:
    request_id: str
    input_ids: Tensor
    state: KoemiState

    def __post_init__(self) -> None:
        _validate_request_id(self.request_id)
        normalized_input_ids = _normalize_decode_input(self.input_ids)
        object.__setattr__(self, "input_ids", normalized_input_ids)
        _validate_single_state(self.state, normalized_input_ids.device, "state")


@dataclass(frozen=True)
class DecodeResult:
    request_id: str
    last_logits: Tensor
    state: KoemiState


def evaluate_prompt_state(
    model: KoemiModel,
    input_ids: Tensor,
    warm_cache: WarmTokenCache | None,
    prefix_cache: DiskMappingCache | None,
    bulk_prefix_cache: BulkPrefixCache | None = None,
    *,
    initial_state: KoemiState | None = None,
) -> PromptEvaluation:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("prompt evaluation requires input_ids with shape [1, sequence]")
    if prefix_cache is not None and bulk_prefix_cache is not None:
        raise ValueError("mapping cache and bulk prefix cache are mutually exclusive")
    active_prefix_cache = (
        bulk_prefix_cache if bulk_prefix_cache is not None else prefix_cache
    )
    if initial_state is not None:
        if active_prefix_cache is not None:
            raise ValueError("initial state and prefix cache are mutually exclusive")
        _validate_single_state(initial_state, input_ids.device, "initial_state")
    cached_prefix = (
        active_prefix_cache.get_longest_prefix(input_ids, input_ids.device)
        if active_prefix_cache is not None
        else None
    )
    if cached_prefix is None:
        prefix_length = 0
        current_state = initial_state
    else:
        prefix_length, cached_state = cached_prefix
        current_state = cached_state.state
        if prefix_length == input_ids.shape[1]:
            return PromptEvaluation(cached_state.last_logits, current_state, 0, prefix_length)

    processed_tokens = 0
    last_logits = None
    chunk_size = (
        bulk_prefix_cache.block_size
        if bulk_prefix_cache is not None
        else model.settings.scan_chunk
    )
    for end in range(prefix_length + chunk_size, input_ids.shape[1], chunk_size):
        output = model(input_ids[:, prefix_length:end], current_state, warm_cache=warm_cache)
        processed_tokens += end - prefix_length
        prefix_length = end
        current_state = output.state
        last_logits = output.logits[:, -1]
        if active_prefix_cache is not None:
            active_prefix_cache.put_prefix(
                input_ids[:, :end],
                CachedPrefixState(last_logits, current_state),
            )
    if prefix_length < input_ids.shape[1]:
        output = model(input_ids[:, prefix_length:], current_state, warm_cache=warm_cache)
        processed_tokens += input_ids.shape[1] - prefix_length
        prefix_length = input_ids.shape[1]
        current_state = output.state
        last_logits = output.logits[:, -1]
        if active_prefix_cache is not None:
            active_prefix_cache.put_prefix(
                input_ids,
                CachedPrefixState(last_logits, current_state),
            )
    if last_logits is None or current_state is None:
        raise RuntimeError("prompt evaluation did not produce model state")
    return PromptEvaluation(last_logits, current_state, processed_tokens, prefix_length - processed_tokens)


def prefill_batch(
    model: KoemiModel,
    requests: Sequence[PrefillRequest],
    warm_cache: WarmTokenCache | None = None,
    prefix_cache: DiskMappingCache | None = None,
    bulk_prefix_cache: BulkPrefixCache | None = None,
    *,
    max_padded_tokens: int | None = None,
    length_bucket_size: int | None = None,
) -> tuple[PrefillResult, ...]:
    request_list = _validate_prefill_requests(requests)
    if not request_list:
        return ()
    _validate_prefill_limits(request_list, max_padded_tokens, length_bucket_size)
    devices = {request.input_ids.device for request in request_list}
    if len(devices) != 1:
        raise ValueError("prefill requests must use one device")
    if prefix_cache is not None or bulk_prefix_cache is not None:
        if any(request.initial_state is not None for request in request_list):
            raise ValueError("prefix caches cannot be combined with initial states")
        evaluations = tuple(
            evaluate_prompt_state(
                model,
                request.input_ids,
                warm_cache,
                prefix_cache,
                bulk_prefix_cache,
            )
            for request in request_list
        )
        return tuple(
            PrefillResult(
                request_id=request.request_id,
                last_logits=evaluation.last_logits,
                state=evaluation.state,
                processed_tokens=evaluation.processed_tokens,
                reused_prefix_tokens=evaluation.reused_prefix_tokens,
            )
            for request, evaluation in zip(request_list, evaluations, strict=True)
        )

    initial_states = tuple(request.initial_state for request in request_list)
    has_initial_state = any(state is not None for state in initial_states)
    if has_initial_state and not all(state is not None for state in initial_states):
        raise ValueError("prefill requests must either all provide a state or all start empty")
    maximum_length = max(request.input_ids.shape[1] for request in request_list)
    device = request_list[0].input_ids.device
    padded_input_ids = torch.full(
        (len(request_list), maximum_length),
        PAD_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    for row_index, request in enumerate(request_list):
        length = request.input_ids.shape[1]
        start = maximum_length - length
        padded_input_ids[row_index, start : start + length].copy_(request.input_ids[0])
    batched_state = (
        _stack_states(tuple(state for state in initial_states if state is not None))
        if has_initial_state
        else None
    )
    output = model(padded_input_ids, batched_state, warm_cache=warm_cache)
    results: list[PrefillResult] = []
    for row_index, request in enumerate(request_list):
        length = request.input_ids.shape[1]
        start = maximum_length - length
        initial_step = 0 if request.initial_state is None else request.initial_state.step_index
        results.append(
            PrefillResult(
                request_id=request.request_id,
                last_logits=output.logits[row_index : row_index + 1, start + length - 1],
                state=_slice_state(output.state, row_index, initial_step + length),
                processed_tokens=length,
                reused_prefix_tokens=0,
            )
        )
    return tuple(results)


def decode_batch(
    model: KoemiModel,
    requests: Sequence[DecodeRequest],
    warm_cache: WarmTokenCache | None = None,
) -> tuple[DecodeResult, ...]:
    request_list = _validate_decode_requests(requests)
    if not request_list:
        return ()
    devices = {request.input_ids.device for request in request_list}
    if len(devices) != 1:
        raise ValueError("decode requests must use one device")
    input_ids = torch.cat([request.input_ids for request in request_list], dim=0)
    state = _stack_states(tuple(request.state for request in request_list))
    output = model(
        input_ids,
        state,
        execution_mode=ExecutionMode.PARALLEL,
        warm_cache=warm_cache,
    )
    return tuple(
        DecodeResult(
            request_id=request.request_id,
            last_logits=output.logits[row_index : row_index + 1, -1],
            state=_slice_state(output.state, row_index, request.state.step_index + 1),
        )
        for row_index, request in enumerate(request_list)
    )


def decode_step(
    model: KoemiModel,
    request: DecodeRequest,
    warm_cache: WarmTokenCache | None = None,
) -> DecodeResult:
    return decode_batch(model, (request,), warm_cache=warm_cache)[0]


def generate_text(
    model: KoemiModel,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_bytes: int,
    temperature: float,
    device: str,
    warm_cache: WarmTokenCache | None = None,
    mapping_cache: DiskMappingCache | None = None,
    bulk_prefix_cache: BulkPrefixCache | None = None,
) -> str:
    if not prompt:
        raise ValueError("prompt must not be empty")
    if max_new_bytes < 1:
        raise ValueError("max_new_bytes must be at least 1")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    generated_ids = list(prompt_ids)
    with torch.no_grad():
        prompt_evaluation = prefill_batch(
            model,
            (PrefillRequest("generation", input_ids),),
            warm_cache=warm_cache,
            prefix_cache=mapping_cache,
            bulk_prefix_cache=bulk_prefix_cache,
        )[0]
        next_logits = prompt_evaluation.last_logits.clone()
        current_state = prompt_evaluation.state
        for _ in range(max_new_bytes):
            next_logits[:, PAD_TOKEN_ID] = float("-inf")
            probabilities = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token_id = int(next_token.item())
            generated_ids.append(next_token_id)
            decode_result = decode_step(
                model,
                DecodeRequest("generation", next_token, current_state),
                warm_cache=warm_cache,
            )
            next_logits = decode_result.last_logits.clone()
            current_state = decode_result.state
    return tokenizer.decode(generated_ids)


def _validate_request_id(request_id: object) -> None:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")


def _normalize_prompt_input(input_ids: Tensor) -> Tensor:
    if not isinstance(input_ids, Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("prefill input_ids must have shape [1, sequence]")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must use torch.long")
    if bool(torch.any(input_ids < 0).item()) or bool(torch.any(input_ids >= PAD_TOKEN_ID).item()):
        raise ValueError("input_ids must contain byte token IDs without padding")
    return input_ids.detach()


def _normalize_decode_input(input_ids: Tensor) -> Tensor:
    if not isinstance(input_ids, Tensor):
        raise TypeError("decode input_ids must be a torch.Tensor")
    if input_ids.ndim == 1 and input_ids.shape[0] == 1:
        input_ids = input_ids.reshape(1, 1)
    if input_ids.ndim != 2 or input_ids.shape != (1, 1):
        raise ValueError("decode input_ids must contain exactly one token")
    return _normalize_prompt_input(input_ids)


def _validate_single_state(state: KoemiState, device: torch.device, name: str) -> None:
    if not isinstance(state, KoemiState):
        raise TypeError(f"{name} must be a KoemiState")
    if isinstance(state.step_index, bool) or not isinstance(state.step_index, int) or state.step_index < 0:
        raise ValueError(f"{name}.step_index must be a non-negative integer")
    for field_name in (
        "working_state",
        "memory_basis",
        "memory_normalizer",
        "refine_basis",
        "refine_normalizer",
        "local_keys",
        "local_values",
        "local_valid",
        "salient_keys",
        "salient_values",
        "salient_valid",
        "last_token_ids",
    ):
        value = getattr(state, field_name)
        if not isinstance(value, Tensor):
            raise TypeError(f"{name}.{field_name} must be a tensor")
        if value.ndim == 0 or value.shape[0] != 1:
            raise ValueError(f"{name}.{field_name} must have batch dimension 1")
        if value.device != device:
            raise ValueError(f"{name}.{field_name} device must match input_ids")


def _stack_states(states: tuple[KoemiState, ...]) -> KoemiState:
    if not states:
        raise ValueError("at least one state is required")
    field_names = (
        "working_state",
        "memory_basis",
        "memory_normalizer",
        "refine_basis",
        "refine_normalizer",
        "local_keys",
        "local_values",
        "local_valid",
        "salient_keys",
        "salient_values",
        "salient_valid",
        "last_token_ids",
    )
    stacked_fields: dict[str, Tensor] = {}
    for field_name in field_names:
        values = tuple(getattr(state, field_name) for state in states)
        if field_name in _VARIABLE_LENGTH_STATE_FIELDS:
            reference_shape = values[0].shape[2:]
            if any(value.shape[2:] != reference_shape for value in values[1:]):
                raise ValueError(f"state field {field_name} shapes cannot be batched")
            maximum_length = max(value.shape[1] for value in values)
            values = tuple(
                _left_pad_state_field(value, maximum_length) for value in values
            )
        else:
            reference_shape = values[0].shape[1:]
            if any(value.shape[1:] != reference_shape for value in values[1:]):
                raise ValueError(f"state field {field_name} shapes cannot be batched")
        stacked_fields[field_name] = torch.cat(values, dim=0)
    return KoemiState(**stacked_fields, step_index=states[0].step_index)


def _left_pad_state_field(value: Tensor, maximum_length: int) -> Tensor:
    padding_length = maximum_length - value.shape[1]
    if padding_length == 0:
        return value
    padding = value.new_zeros((value.shape[0], padding_length, *value.shape[2:]))
    return torch.cat((padding, value), dim=1)


def _slice_state(state: KoemiState, row_index: int, step_index: int) -> KoemiState:
    return KoemiState(
        working_state=state.working_state[row_index : row_index + 1],
        memory_basis=state.memory_basis[row_index : row_index + 1],
        memory_normalizer=state.memory_normalizer[row_index : row_index + 1],
        refine_basis=state.refine_basis[row_index : row_index + 1],
        refine_normalizer=state.refine_normalizer[row_index : row_index + 1],
        local_keys=state.local_keys[row_index : row_index + 1],
        local_values=state.local_values[row_index : row_index + 1],
        local_valid=state.local_valid[row_index : row_index + 1],
        salient_keys=state.salient_keys[row_index : row_index + 1],
        salient_values=state.salient_values[row_index : row_index + 1],
        salient_valid=state.salient_valid[row_index : row_index + 1],
        last_token_ids=state.last_token_ids[row_index : row_index + 1],
        step_index=step_index,
    )


def _validate_prefill_requests(requests: Sequence[PrefillRequest]) -> tuple[PrefillRequest, ...]:
    if isinstance(requests, (str, bytes, bytearray)):
        raise TypeError("prefill requests must be a sequence of PrefillRequest")
    try:
        request_list = tuple(requests)
    except TypeError as error:
        raise TypeError("prefill requests must be a sequence of PrefillRequest") from error
    if not all(isinstance(request, PrefillRequest) for request in request_list):
        raise TypeError("prefill requests must contain only PrefillRequest values")
    request_ids = tuple(request.request_id for request in request_list)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("prefill request IDs must be unique")
    return request_list


def _validate_decode_requests(requests: Sequence[DecodeRequest]) -> tuple[DecodeRequest, ...]:
    if isinstance(requests, (str, bytes, bytearray)):
        raise TypeError("decode requests must be a sequence of DecodeRequest")
    try:
        request_list = tuple(requests)
    except TypeError as error:
        raise TypeError("decode requests must be a sequence of DecodeRequest") from error
    if not all(isinstance(request, DecodeRequest) for request in request_list):
        raise TypeError("decode requests must contain only DecodeRequest values")
    request_ids = tuple(request.request_id for request in request_list)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("decode request IDs must be unique")
    return request_list


def _validate_prefill_limits(
    requests: tuple[PrefillRequest, ...],
    max_padded_tokens: int | None,
    length_bucket_size: int | None,
) -> None:
    if max_padded_tokens is not None:
        if isinstance(max_padded_tokens, bool) or not isinstance(max_padded_tokens, int) or max_padded_tokens < 1:
            raise ValueError("max_padded_tokens must be a positive integer")
        maximum_length = max(request.input_ids.shape[1] for request in requests)
        if len(requests) * maximum_length > max_padded_tokens:
            raise ValueError("prefill padded token count exceeds max_padded_tokens")
    if length_bucket_size is not None:
        if isinstance(length_bucket_size, bool) or not isinstance(length_bucket_size, int) or length_bucket_size < 1:
            raise ValueError("length_bucket_size must be a positive integer")
        bucket_keys = {
            (request.input_ids.shape[1] - 1) // length_bucket_size for request in requests
        }
        if len(bucket_keys) != 1:
            raise ValueError("prefill requests must belong to one length bucket")


__all__ = [
    "DecodeRequest",
    "DecodeResult",
    "PrefillRequest",
    "PrefillResult",
    "PromptEvaluation",
    "decode_batch",
    "decode_step",
    "evaluate_prompt_state",
    "generate_text",
    "prefill_batch",
]
