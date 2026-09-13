from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.cache import CachedMapping, DiskMappingCache, WarmTokenCache
from koemi.model.execution import ExecutionMode
from koemi.model.experts import DeterministicExpertMixture
from koemi.model.layers import RootMeanSquareNorm
from koemi.model.memory import (
    BoundedRecurrentState,
    HierarchicalAssociativeMemory,
    LocalKeyValueMemory,
    MemoryWriteTerms,
)
from koemi.model.scan import affine_scan, previous_states
from koemi.model.state import KoemiState


@dataclass(frozen=True)
class KoemiOutput:
    logits: Tensor
    state: KoemiState
    surprise_values: Tensor
    expert_indices: Tensor
    valid_positions: Tensor
    token_count: int
    cache_hits: int
    cache_misses: int
    expert_count: int
    active_expert_indices: Tensor | None = None

    @property
    def expert_activation_counts(self) -> tuple[int, ...]:
        if self.expert_indices.numel() == 0:
            return ()
        assignments = self.active_expert_indices if self.active_expert_indices is not None else self.expert_indices.unsqueeze(-1)
        return tuple(int((assignments == index).sum()) for index in range(self.expert_count))


class KoemiModel(nn.Module):
    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        self.settings = settings
        embedding_size = settings.embedding_size
        self.embedding = nn.Embedding(settings.vocabulary_size, embedding_size, padding_idx=PAD_TOKEN_ID)
        self.input_normalizer = RootMeanSquareNorm(embedding_size)
        self.recurrent_state = BoundedRecurrentState(embedding_size)
        self.associative_memory = HierarchicalAssociativeMemory(
            embedding_size,
            settings.memory_features,
            settings.refine_decay_rate,
        )
        self.local_memory = LocalKeyValueMemory(embedding_size, settings.local_memory_size)
        self.memory_refine_gate = (
            None if settings.ablation == "no_refine" else nn.Linear(embedding_size * 4, 1)
        )
        self.fusion_projection = nn.Linear(embedding_size * 4, embedding_size)
        self.fusion_normalizer = RootMeanSquareNorm(embedding_size)
        self.experts = DeterministicExpertMixture(embedding_size, settings.expert_count, settings.expert_top_k)
        self.token_predictor = nn.Linear(embedding_size, settings.vocabulary_size)

    def forward(
        self,
        input_ids: Tensor,
        state: KoemiState | None = None,
        execution_mode: ExecutionMode = ExecutionMode.PARALLEL,
        warm_cache: WarmTokenCache | None = None,
        mapping_cache: DiskMappingCache | None = None,
    ) -> KoemiOutput:
        self.validate_input_ids(input_ids)
        if (warm_cache is not None or mapping_cache is not None) and self.training:
            raise RuntimeError("inference caches are only valid while the model is in evaluation mode")
        root_forward = state is None
        if mapping_cache is not None:
            if input_ids.shape[0] != 1:
                raise ValueError("disk mapping cache requires batch size 1")
            if root_forward:
                cached_mapping = mapping_cache.get(input_ids, input_ids.device)
                if cached_mapping is not None:
                    return self.output_from_cached_mapping(cached_mapping, input_ids)
        if execution_mode is ExecutionMode.SEQUENTIAL:
            output = self.forward_sequential(input_ids, state, warm_cache)
        else:
            output = self.forward_parallel(input_ids, state, warm_cache)
        if mapping_cache is not None and root_forward:
            mapping_cache.put(
                input_ids,
                CachedMapping(
                    logits=output.logits,
                    state=output.state,
                    surprise_values=output.surprise_values,
                    expert_indices=output.expert_indices,
                    active_expert_indices=output.active_expert_indices,
                    valid_positions=output.valid_positions,
                    token_count=output.token_count,
                ),
            )
        return output

    def initial_state(self, batch_size: int, device: torch.device) -> KoemiState:
        return KoemiState.create(batch_size, self.settings.embedding_size, self.settings.memory_features, device)

    def forward_parallel(
        self,
        input_ids: Tensor,
        state: KoemiState | None,
        warm_cache: WarmTokenCache | None,
    ) -> KoemiOutput:
        batch_size, length = input_ids.shape
        current_state = state or self.initial_state(batch_size, input_ids.device)
        window = self.settings.scan_chunk
        if window >= length:
            return self.forward_window(input_ids, current_state, warm_cache)
        windows: list[KoemiOutput] = []
        for start in range(0, length, window):
            piece = self.forward_window(input_ids[:, start : start + window], current_state, warm_cache)
            windows.append(piece)
            current_state = piece.state
        return concatenate_outputs(windows)

    def forward_window(
        self,
        input_ids: Tensor,
        current_state: KoemiState,
        warm_cache: WarmTokenCache | None,
    ) -> KoemiOutput:
        _, length = input_ids.shape
        valid_mask = input_ids != PAD_TOKEN_ID
        input_state, cache_hits, cache_misses = self.embed_inputs(input_ids, warm_cache)
        retention, increment = self.recurrent_state.gates(input_state)
        retention = torch.where(valid_mask.unsqueeze(-1), retention, torch.ones_like(retention))
        increment = torch.where(valid_mask.unsqueeze(-1), increment, torch.zeros_like(increment))
        working_states = affine_scan(retention, increment, current_state.working_state)
        if self.settings.ablation == "affine":
            return self.forward_affine_window(input_ids, current_state, working_states, cache_hits, cache_misses)
        prior_working_states = previous_states(working_states, current_state.working_state)
        surprise = (
            torch.zeros_like(input_ids, dtype=working_states.dtype)
            if self.settings.ablation == "no_surprise"
            else self.calculate_surprise(prior_working_states, input_ids, valid_mask)
        )

        local_keys, local_values, local_valid = self.local_memory.entries(working_states, valid_mask)
        local_value, novelty = self.local_memory.read_window(
            current_state.local_keys,
            current_state.local_values,
            current_state.local_valid,
            local_keys,
            local_values,
            local_valid,
            working_states,
        )

        projection = self.associative_memory.project(working_states)
        fast_terms = self.mask_write_terms(
            self.associative_memory.fast_write_terms(projection, surprise),
            valid_mask,
        )
        fast_memory, next_basis, next_normalizer = self.associative_memory.scan_and_read(
            current_state.memory_basis,
            current_state.memory_normalizer,
            fast_terms,
            working_states,
        )
        salience_admission = valid_mask & (surprise > self.settings.salience_threshold)
        salient_value = self.local_memory.read_salient_window(
            current_state.salient_keys,
            current_state.salient_values,
            current_state.salient_valid,
            local_keys,
            local_values,
            salience_admission,
            working_states,
        )

        reconstruction_error = projection.value - fast_memory
        if self.settings.ablation == "no_refine":
            next_refine_basis = current_state.refine_basis
            next_refine_normalizer = current_state.refine_normalizer
            refine_memory = torch.zeros_like(fast_memory)
        else:
            refine_terms = self.mask_write_terms(
                self.associative_memory.refine_write_terms(
                    projection,
                    reconstruction_error,
                    surprise,
                    novelty,
                ),
                valid_mask,
            )
            refine_memory, next_refine_basis, next_refine_normalizer = self.associative_memory.scan_and_read(
                current_state.refine_basis,
                current_state.refine_normalizer,
                refine_terms,
                working_states,
            )

        memory_value = (
            fast_memory
            if self.settings.ablation == "no_refine"
            else self.refine_memory(working_states, fast_memory, refine_memory, local_value)
        )
        fused_context = self.fuse(working_states, memory_value, local_value, salient_value)
        previous_token_ids = self.previous_token_ids(input_ids, valid_mask, current_state.last_token_ids)
        final_context, expert_indices, active_expert_indices = self.experts(
            fused_context,
            input_ids,
            previous_token_ids,
            valid_mask,
        )
        logits = self.predict_tokens(final_context)
        next_local_keys, next_local_values, next_local_valid = self.local_memory.tail(
            current_state.local_keys,
            current_state.local_values,
            current_state.local_valid,
            local_keys,
            local_values,
            local_valid,
        )
        next_salient_keys, next_salient_values, next_salient_valid = self.local_memory.salient_tail(
            current_state.salient_keys,
            current_state.salient_values,
            current_state.salient_valid,
            local_keys,
            local_values,
            salience_admission,
            self.settings.salience_memory_size,
        )
        next_state = KoemiState(
            working_state=working_states[:, -1],
            memory_basis=next_basis,
            memory_normalizer=next_normalizer,
            refine_basis=next_refine_basis,
            refine_normalizer=next_refine_normalizer,
            local_keys=next_local_keys,
            local_values=next_local_values,
            local_valid=next_local_valid,
            salient_keys=next_salient_keys,
            salient_values=next_salient_values,
            salient_valid=next_salient_valid,
            last_token_ids=self.last_valid_token_ids(input_ids, valid_mask, current_state.last_token_ids),
            step_index=current_state.step_index + length,
        )
        return KoemiOutput(
            logits=logits,
            state=next_state,
            surprise_values=surprise,
            expert_indices=expert_indices,
            valid_positions=valid_mask,
            token_count=int(valid_mask.sum()),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            expert_count=self.settings.expert_count,
            active_expert_indices=active_expert_indices,
        )

    def forward_affine_window(
        self,
        input_ids: Tensor,
        current_state: KoemiState,
        working_states: Tensor,
        cache_hits: int,
        cache_misses: int,
    ) -> KoemiOutput:
        valid_mask = input_ids != PAD_TOKEN_ID
        next_state = KoemiState(
            working_state=working_states[:, -1],
            memory_basis=current_state.memory_basis,
            memory_normalizer=current_state.memory_normalizer,
            refine_basis=current_state.refine_basis,
            refine_normalizer=current_state.refine_normalizer,
            local_keys=current_state.local_keys,
            local_values=current_state.local_values,
            local_valid=current_state.local_valid,
            salient_keys=current_state.salient_keys,
            salient_values=current_state.salient_values,
            salient_valid=current_state.salient_valid,
            last_token_ids=self.last_valid_token_ids(input_ids, valid_mask, current_state.last_token_ids),
            step_index=current_state.step_index + input_ids.shape[1],
        )
        return KoemiOutput(
            logits=self.predict_tokens(working_states),
            state=next_state,
            surprise_values=torch.zeros_like(working_states[..., 0]),
            expert_indices=torch.full_like(input_ids, -1),
            valid_positions=valid_mask,
            token_count=int(valid_mask.sum()),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            expert_count=0,
            active_expert_indices=None,
        )

    def forward_sequential(
        self,
        input_ids: Tensor,
        state: KoemiState | None,
        warm_cache: WarmTokenCache | None,
    ) -> KoemiOutput:
        batch_size, length = input_ids.shape
        current_state = state or self.initial_state(batch_size, input_ids.device)
        logits_by_position: list[Tensor] = []
        surprise_by_position: list[Tensor] = []
        expert_indices_by_position: list[Tensor] = []
        active_expert_indices_by_position: list[Tensor] = []
        valid_by_position: list[Tensor] = []
        cache_hits = 0
        cache_misses = 0
        for position in range(length):
            token_ids = input_ids[:, position]
            valid_mask = token_ids != PAD_TOKEN_ID
            input_state, position_hits, position_misses = self.embed_inputs(token_ids.unsqueeze(1), warm_cache)
            cache_hits += position_hits
            cache_misses += position_misses
            surprise = (
                torch.zeros_like(token_ids, dtype=input_state.dtype)
                if self.settings.ablation == "no_surprise"
                else self.calculate_surprise(
                    current_state.working_state.unsqueeze(1),
                    token_ids.unsqueeze(1),
                    valid_mask.unsqueeze(1),
                ).squeeze(1)
            )
            working_state = torch.where(
                valid_mask.unsqueeze(-1),
                self.recurrent_state(current_state.working_state, input_state.squeeze(1)),
                current_state.working_state,
            )
            if self.settings.ablation == "affine":
                logits_by_position.append(self.predict_tokens(working_state))
                surprise_by_position.append(torch.zeros_like(working_state[:, 0]))
                expert_indices_by_position.append(torch.full_like(token_ids, -1))
                valid_by_position.append(valid_mask)
                current_state = KoemiState(
                    working_state=working_state,
                    memory_basis=current_state.memory_basis,
                    memory_normalizer=current_state.memory_normalizer,
                    refine_basis=current_state.refine_basis,
                    refine_normalizer=current_state.refine_normalizer,
                    local_keys=current_state.local_keys,
                    local_values=current_state.local_values,
                    local_valid=current_state.local_valid,
                    salient_keys=current_state.salient_keys,
                    salient_values=current_state.salient_values,
                    salient_valid=current_state.salient_valid,
                    last_token_ids=torch.where(valid_mask, token_ids, current_state.last_token_ids),
                    step_index=current_state.step_index + 1,
                )
                continue
            local_value, novelty = self.local_memory.read(
                current_state.local_keys,
                current_state.local_values,
                current_state.local_valid,
                working_state,
            )
            salient_value, _ = self.local_memory.read(
                current_state.salient_keys,
                current_state.salient_values,
                current_state.salient_valid,
                working_state,
            )
            projection = self.associative_memory.project(working_state)
            fast_memory, _ = self.associative_memory.read(
                current_state.memory_basis,
                current_state.memory_normalizer,
                working_state,
            )
            if self.settings.ablation == "no_refine":
                refine_memory = torch.zeros_like(fast_memory)
            else:
                refine_memory, _ = self.associative_memory.read(
                    current_state.refine_basis,
                    current_state.refine_normalizer,
                    working_state,
                )
            memory_value = (
                fast_memory
                if self.settings.ablation == "no_refine"
                else self.refine_memory(working_state, fast_memory, refine_memory, local_value)
            )
            fused_context = self.fuse(working_state, memory_value, local_value, salient_value)
            final_context, expert_indices, active_expert_indices = self.experts(
                fused_context.unsqueeze(1),
                token_ids.unsqueeze(1),
                current_state.last_token_ids.unsqueeze(1),
                valid_mask.unsqueeze(1),
            )
            logits_by_position.append(self.predict_tokens(final_context[:, 0]))
            surprise_by_position.append(surprise)
            expert_indices_by_position.append(expert_indices[:, 0])
            active_expert_indices_by_position.append(active_expert_indices[:, 0])
            valid_by_position.append(valid_mask)

            fast_terms = self.associative_memory.fast_write_terms(projection, surprise)
            next_basis, next_normalizer = self.associative_memory.update(
                current_state.memory_basis,
                current_state.memory_normalizer,
                fast_terms,
            )
            if self.settings.ablation == "no_refine":
                next_refine_basis, next_refine_normalizer = (
                    current_state.refine_basis,
                    current_state.refine_normalizer,
                )
            else:
                refine_terms = self.associative_memory.refine_write_terms(
                    projection,
                    projection.value - fast_memory,
                    surprise,
                    novelty,
                )
                next_refine_basis, next_refine_normalizer = self.associative_memory.update(
                    current_state.refine_basis,
                    current_state.refine_normalizer,
                    refine_terms,
                )
            written_key, written_value, written_valid = self.local_memory.entries(working_state, valid_mask)
            next_local_keys, next_local_values, next_local_valid = self.local_memory.append(
                current_state.local_keys,
                current_state.local_values,
                current_state.local_valid,
                written_key,
                written_value,
                written_valid,
            )
            salience_admission = valid_mask & (surprise > self.settings.salience_threshold)
            next_salient_keys, next_salient_values, next_salient_valid = self.local_memory.salient_tail(
                current_state.salient_keys,
                current_state.salient_values,
                current_state.salient_valid,
                written_key.unsqueeze(1),
                written_value.unsqueeze(1),
                salience_admission.unsqueeze(1),
                self.settings.salience_memory_size,
            )
            current_state = KoemiState(
                working_state=working_state,
                memory_basis=torch.where(
                    valid_mask.unsqueeze(-1).unsqueeze(-1),
                    next_basis,
                    current_state.memory_basis,
                ),
                memory_normalizer=torch.where(
                    valid_mask.unsqueeze(-1),
                    next_normalizer,
                    current_state.memory_normalizer,
                ),
                refine_basis=torch.where(
                    valid_mask.unsqueeze(-1).unsqueeze(-1),
                    next_refine_basis,
                    current_state.refine_basis,
                ),
                refine_normalizer=torch.where(
                    valid_mask.unsqueeze(-1),
                    next_refine_normalizer,
                    current_state.refine_normalizer,
                ),
                local_keys=next_local_keys,
                local_values=next_local_values,
                local_valid=next_local_valid,
                salient_keys=next_salient_keys,
                salient_values=next_salient_values,
                salient_valid=next_salient_valid,
                last_token_ids=torch.where(valid_mask, token_ids, current_state.last_token_ids),
                step_index=current_state.step_index + 1,
            )
        valid_positions = torch.stack(valid_by_position, dim=1)
        expert_indices = torch.stack(expert_indices_by_position, dim=1)
        return KoemiOutput(
            logits=torch.stack(logits_by_position, dim=1),
            state=current_state,
            surprise_values=torch.stack(surprise_by_position, dim=1),
            expert_indices=expert_indices,
            valid_positions=valid_positions,
            token_count=int(valid_positions.sum()),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            expert_count=self.settings.expert_count,
            active_expert_indices=torch.stack(active_expert_indices_by_position, dim=1),
        )

    def calculate_surprise(self, prior_states: Tensor, input_ids: Tensor, valid_mask: Tensor) -> Tensor:
        content_logits = self.predict_tokens(prior_states)[..., :PAD_TOKEN_ID]
        safe_input_ids = input_ids.clamp(max=PAD_TOKEN_ID - 1)
        token_nll = functional.cross_entropy(
            content_logits.reshape(-1, PAD_TOKEN_ID),
            safe_input_ids.reshape(-1),
            reduction="none",
        ).view_as(input_ids)
        normalized_nll = token_nll / math.log(PAD_TOKEN_ID)
        surprise = 1.0 - torch.exp(-normalized_nll)
        return torch.where(valid_mask, surprise, torch.zeros_like(surprise))

    def predict_tokens(self, context: Tensor) -> Tensor:
        return self.token_predictor(context)

    def embed_inputs(self, input_ids: Tensor, warm_cache: WarmTokenCache | None) -> tuple[Tensor, int, int]:
        if warm_cache is None:
            return self.input_normalizer(self.embedding(input_ids)), 0, 0
        embeddings, cache_hits, cache_misses = warm_cache.embeddings(input_ids, self.embedding)
        return self.input_normalizer(embeddings), cache_hits, cache_misses

    def refine_memory(
        self,
        working_state: Tensor,
        fast_memory: Tensor,
        refine_memory: Tensor,
        local_value: Tensor,
    ) -> Tensor:
        if self.memory_refine_gate is None:
            raise RuntimeError("refine memory is disabled by the model settings")
        refine_gate = torch.sigmoid(
            self.memory_refine_gate(torch.cat((working_state, fast_memory, refine_memory, local_value), dim=-1))
        )
        return fast_memory + refine_gate * refine_memory

    def fuse(
        self,
        working_state: Tensor,
        memory_value: Tensor,
        local_value: Tensor,
        salient_value: Tensor,
    ) -> Tensor:
        return self.fusion_normalizer(
            self.fusion_projection(torch.cat((working_state, memory_value, local_value, salient_value), dim=-1))
        )

    def mask_write_terms(self, terms: MemoryWriteTerms, valid_mask: Tensor) -> MemoryWriteTerms:
        return MemoryWriteTerms(
            decay=torch.where(valid_mask.unsqueeze(-1), terms.decay, torch.ones_like(terms.decay)),
            value=terms.value,
            features=terms.features,
            write_weight=torch.where(valid_mask, terms.write_weight, torch.zeros_like(terms.write_weight)),
        )

    def previous_token_ids(self, input_ids: Tensor, valid_mask: Tensor, carried_token_ids: Tensor) -> Tensor:
        _, length = input_ids.shape
        indices = torch.arange(length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        valid_indices = torch.where(valid_mask, indices, torch.full_like(indices, -1))
        latest_indices = torch.cummax(valid_indices, dim=1).values
        previous_indices = torch.cat((torch.full_like(latest_indices[:, :1], -1), latest_indices[:, :-1]), dim=1)
        gathered_tokens = input_ids.gather(1, previous_indices.clamp_min(0))
        return torch.where(previous_indices >= 0, gathered_tokens, carried_token_ids.unsqueeze(1))

    def last_valid_token_ids(
        self,
        input_ids: Tensor,
        valid_mask: Tensor,
        carried_token_ids: Tensor,
    ) -> Tensor:
        _, length = input_ids.shape
        indices = torch.arange(length, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        last_indices = torch.where(valid_mask, indices, torch.full_like(indices, -1)).amax(dim=1)
        gathered_tokens = input_ids.gather(1, last_indices.clamp_min(0).unsqueeze(1)).squeeze(1)
        return torch.where(last_indices >= 0, gathered_tokens, carried_token_ids)

    def validate_input_ids(self, input_ids: Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= self.settings.vocabulary_size:
            raise ValueError("input_ids contain values outside the model vocabulary")

    def output_from_cached_mapping(self, mapping: CachedMapping, input_ids: Tensor) -> KoemiOutput:
        device = input_ids.device
        if mapping.valid_positions.shape != input_ids.shape:
            raise RuntimeError("disk cache entry sequence shape does not match the request")
        state = KoemiState(
            working_state=mapping.state.working_state.to(device),
            memory_basis=mapping.state.memory_basis.to(device),
            memory_normalizer=mapping.state.memory_normalizer.to(device),
            refine_basis=mapping.state.refine_basis.to(device),
            refine_normalizer=mapping.state.refine_normalizer.to(device),
            local_keys=mapping.state.local_keys.to(device),
            local_values=mapping.state.local_values.to(device),
            local_valid=mapping.state.local_valid.to(device),
            salient_keys=mapping.state.salient_keys.to(device),
            salient_values=mapping.state.salient_values.to(device),
            salient_valid=mapping.state.salient_valid.to(device),
            last_token_ids=mapping.state.last_token_ids.to(device),
            step_index=mapping.state.step_index,
        )
        return KoemiOutput(
            logits=mapping.logits.to(device),
            state=state,
            surprise_values=mapping.surprise_values.to(device),
            expert_indices=mapping.expert_indices.to(device),
            active_expert_indices=(
                mapping.active_expert_indices.to(device)
                if mapping.active_expert_indices is not None
                else None
            ),
            valid_positions=mapping.valid_positions.to(device),
            token_count=mapping.token_count,
            cache_hits=0,
            cache_misses=0,
            expert_count=self.settings.expert_count,
        )


def concatenate_outputs(windows: list[KoemiOutput]) -> KoemiOutput:
    return KoemiOutput(
        logits=torch.cat([window.logits for window in windows], dim=1),
        state=windows[-1].state,
        surprise_values=torch.cat([window.surprise_values for window in windows], dim=1),
        expert_indices=torch.cat([window.expert_indices for window in windows], dim=1),
        valid_positions=torch.cat([window.valid_positions for window in windows], dim=1),
        token_count=sum(window.token_count for window in windows),
        cache_hits=sum(window.cache_hits for window in windows),
        cache_misses=sum(window.cache_misses for window in windows),
        expert_count=windows[0].expert_count,
        active_expert_indices=(
            torch.cat([window.active_expert_indices for window in windows], dim=1)
            if windows[0].active_expert_indices is not None
            else None
        ),
    )
