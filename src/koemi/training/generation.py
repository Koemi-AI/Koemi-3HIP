from __future__ import annotations

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


@dataclass(frozen=True)
class PromptEvaluation:
    last_logits: Tensor
    state: KoemiState
    processed_tokens: int
    reused_prefix_tokens: int


def evaluate_prompt_state(
    model: KoemiModel,
    input_ids: Tensor,
    warm_cache: WarmTokenCache | None,
    prefix_cache: DiskMappingCache | None,
    bulk_prefix_cache: BulkPrefixCache | None = None,
) -> PromptEvaluation:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("prompt evaluation requires input_ids with shape [1, sequence]")
    if prefix_cache is not None and bulk_prefix_cache is not None:
        raise ValueError("mapping cache and bulk prefix cache are mutually exclusive")
    active_prefix_cache = (
        bulk_prefix_cache if bulk_prefix_cache is not None else prefix_cache
    )
    cached_prefix = (
        active_prefix_cache.get_longest_prefix(input_ids, input_ids.device)
        if active_prefix_cache is not None
        else None
    )
    if cached_prefix is None:
        prefix_length = 0
        current_state = None
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
        prompt_evaluation = evaluate_prompt_state(
            model,
            input_ids,
            warm_cache,
            mapping_cache,
            bulk_prefix_cache,
        )
        next_logits = prompt_evaluation.last_logits.clone()
        current_state = prompt_evaluation.state
        for _ in range(max_new_bytes):
            next_logits[:, PAD_TOKEN_ID] = float("-inf")
            probabilities = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            next_token_id = int(next_token.item())
            generated_ids.append(next_token_id)
            output = model(
                next_token,
                current_state,
                execution_mode=ExecutionMode.PARALLEL,
                warm_cache=warm_cache,
            )
            next_logits = output.logits[:, -1, :].clone()
            current_state = output.state
    return tokenizer.decode(generated_ids)
