from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional


NEGATIVE_INFINITY = float("-inf")


class BoundedRecurrentState(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.retention_projection = nn.Linear(embedding_size, embedding_size)
        self.candidate_projection = nn.Linear(embedding_size, embedding_size)
        self.minimum_retention = 2.0**-8
        self.maximum_retention = 1.0 - 2.0**-8

    def gates(self, input_state: Tensor) -> tuple[Tensor, Tensor]:
        retention = torch.sigmoid(self.retention_projection(input_state))
        bounded_retention = self.minimum_retention + (self.maximum_retention - self.minimum_retention) * retention
        candidate_state = torch.tanh(self.candidate_projection(input_state))
        return bounded_retention, (1.0 - bounded_retention) * candidate_state

    def forward(self, previous_state: Tensor, input_state: Tensor) -> Tensor:
        retention, increment = self.gates(input_state)
        return retention * previous_state + increment


@dataclass(frozen=True)
class MemoryProjection:
    value: Tensor
    features: Tensor
    decay: Tensor
    write_weight: Tensor


@dataclass(frozen=True)
class MemoryWriteTerms:
    decay: Tensor
    value: Tensor
    features: Tensor
    write_weight: Tensor

    @property
    def basis_increment(self) -> Tensor:
        return self.write_weight.unsqueeze(-1).unsqueeze(-1) * (
            self.value.unsqueeze(-1) @ self.features.unsqueeze(-2)
        )

    @property
    def normalizer_increment(self) -> Tensor:
        return self.write_weight.unsqueeze(-1) * self.features

    def at(self, position: int) -> MemoryWriteTerms:
        return MemoryWriteTerms(
            self.decay[:, position],
            self.value[:, position],
            self.features[:, position],
            self.write_weight[:, position],
        )


class HierarchicalAssociativeMemory(nn.Module):
    def __init__(self, embedding_size: int, memory_features: int, refine_decay_rate: float) -> None:
        super().__init__()
        self.key_projection = nn.Linear(embedding_size, embedding_size)
        self.query_projection = nn.Linear(embedding_size, embedding_size)
        self.value_projection = nn.Linear(embedding_size, embedding_size)
        self.feature_projection = nn.Linear(embedding_size, memory_features)
        self.decay_projection = nn.Linear(embedding_size, 1)
        self.write_projection = nn.Linear(embedding_size, 1)
        self.refine_decay_rate = refine_decay_rate
        self.minimum_decay = 2.0**-12
        self.maximum_decay = 1.0 - 2.0**-12
        self.maximum_refine_decay = 1.0 - 2.0**-16
        self.epsilon = 2.0**-8

    def project(self, write_source: Tensor) -> MemoryProjection:
        key = self.key_projection(write_source)
        features = torch.softmax(self.feature_projection(key), dim=-1)
        value = torch.tanh(self.value_projection(write_source))
        decay = torch.sigmoid(self.decay_projection(write_source))
        bounded_decay = self.minimum_decay + (self.maximum_decay - self.minimum_decay) * decay
        write_weight = torch.sigmoid(self.write_projection(write_source)).squeeze(-1)
        return MemoryProjection(value, features, bounded_decay, write_weight)

    def read(self, memory_basis: Tensor, memory_normalizer: Tensor, query_source: Tensor) -> tuple[Tensor, Tensor]:
        query_features = self.query_features(query_source)
        numerator = (memory_basis @ query_features.unsqueeze(-1)).squeeze(-1)
        denominator = (memory_normalizer * query_features).sum(dim=-1, keepdim=True)
        return self.confidence_weighted_read(numerator, denominator), query_features

    def scan_and_read(
        self,
        initial_basis: Tensor,
        initial_normalizer: Tensor,
        terms: MemoryWriteTerms,
        query_source: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query_features = self.query_features(query_source)
        decay = terms.decay.squeeze(-1)
        cumulative_log_decay = torch.cumsum(torch.log(decay.float()), dim=1)
        prior_log_decay = cumulative_log_decay - torch.log(decay.float())
        initial_factors = torch.exp(prior_log_decay).to(dtype=terms.value.dtype)
        length = terms.value.shape[1]
        positions = torch.arange(length, device=terms.value.device)
        past_writes = positions.unsqueeze(1) > positions.unsqueeze(0)
        pair_log_decay = prior_log_decay.unsqueeze(-1) - cumulative_log_decay.unsqueeze(1)
        pair_decay = torch.exp(pair_log_decay.masked_fill(~past_writes, float("-inf"))).to(
            dtype=terms.value.dtype
        )
        feature_similarity = torch.einsum("bsm,btm->bts", terms.features, query_features)
        write_influence = pair_decay * feature_similarity * terms.write_weight.unsqueeze(1)
        initial_numerator = torch.einsum("bdm,btm->btd", initial_basis, query_features)
        numerator = initial_factors.unsqueeze(-1) * initial_numerator
        numerator = numerator + torch.einsum("bts,bsd->btd", write_influence, terms.value)
        initial_denominator = torch.einsum("bm,btm->bt", initial_normalizer, query_features)
        denominator = initial_factors * initial_denominator + write_influence.sum(dim=-1)
        read = self.confidence_weighted_read(numerator, denominator.unsqueeze(-1))

        total_log_decay = cumulative_log_decay[:, -1]
        initial_final_factor = torch.exp(total_log_decay).to(dtype=terms.value.dtype)
        final_write_decay = torch.exp(total_log_decay.unsqueeze(1) - cumulative_log_decay).to(
            dtype=terms.value.dtype
        )
        final_write_weight = final_write_decay * terms.write_weight
        final_basis = initial_final_factor.unsqueeze(-1).unsqueeze(-1) * initial_basis
        final_basis = final_basis + torch.einsum(
            "bs,bsd,bsm->bdm",
            final_write_weight,
            terms.value,
            terms.features,
        )
        final_normalizer = initial_final_factor.unsqueeze(-1) * initial_normalizer
        final_normalizer = final_normalizer + torch.einsum(
            "bs,bsm->bm",
            final_write_weight,
            terms.features,
        )
        return read, final_basis, final_normalizer

    def query_features(self, query_source: Tensor) -> Tensor:
        query = self.query_projection(query_source)
        return torch.softmax(self.feature_projection(query), dim=-1)

    def confidence_weighted_read(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        regularized_denominator = denominator + self.epsilon
        confidence = denominator / regularized_denominator
        return confidence * numerator / regularized_denominator

    def fast_write_terms(self, projection: MemoryProjection, surprise: Tensor) -> MemoryWriteTerms:
        write_weight = projection.write_weight * (0.25 + 0.75 * surprise.clamp(0.0, 1.0))
        return self.write_terms(projection.decay, projection.value, projection.features, write_weight)

    def refine_write_terms(
        self,
        projection: MemoryProjection,
        reconstruction_error: Tensor,
        surprise: Tensor,
        novelty: Tensor,
    ) -> MemoryWriteTerms:
        refine_decay = 1.0 - (1.0 - projection.decay) * self.refine_decay_rate
        refine_decay = refine_decay.clamp(max=self.maximum_refine_decay)
        write_weight = projection.write_weight * surprise.clamp(0.0, 1.0) * novelty.clamp(0.0, 1.0)
        bounded_error = reconstruction_error.clamp(-2.0, 2.0)
        return self.write_terms(refine_decay, bounded_error, projection.features, write_weight)

    def write_terms(
        self,
        decay: Tensor,
        value: Tensor,
        features: Tensor,
        write_weight: Tensor,
    ) -> MemoryWriteTerms:
        return MemoryWriteTerms(decay, value, features, write_weight)

    def update(
        self,
        memory_basis: Tensor,
        memory_normalizer: Tensor,
        terms: MemoryWriteTerms,
    ) -> tuple[Tensor, Tensor]:
        next_basis = terms.decay.unsqueeze(-1) * memory_basis + terms.basis_increment
        next_normalizer = terms.decay * memory_normalizer + terms.normalizer_increment
        return next_basis, next_normalizer


class LocalKeyValueMemory(nn.Module):
    def __init__(self, embedding_size: int, local_memory_size: int) -> None:
        super().__init__()
        self.local_memory_size = local_memory_size
        self.local_key_projection = nn.Linear(embedding_size, embedding_size)
        self.local_value_projection = nn.Linear(embedding_size, embedding_size)

    def entries(self, write_source: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        key = self.local_key_projection(write_source)
        value = self.local_value_projection(write_source)
        keep = valid_mask.unsqueeze(-1)
        return (
            torch.where(keep, key, torch.zeros_like(key)),
            torch.where(keep, value, torch.zeros_like(value)),
            valid_mask,
        )

    def read(
        self,
        local_keys: Tensor,
        local_values: Tensor,
        local_valid: Tensor,
        query: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if local_keys.shape[1] == 0:
            empty_value = torch.zeros_like(query)
            empty_novelty = torch.ones(query.shape[0], device=query.device, dtype=query.dtype)
            return empty_value, empty_novelty
        attention_scores = torch.bmm(local_keys, query.unsqueeze(-1)).squeeze(-1) / math.sqrt(query.shape[-1])
        masked_scores = attention_scores.masked_fill(~local_valid, NEGATIVE_INFINITY)
        attention_weights = torch.softmax(masked_scores, dim=-1)
        any_slot = local_valid.any(dim=-1, keepdim=True)
        attention_weights = torch.where(any_slot, attention_weights, torch.zeros_like(attention_weights))
        attention_weights = torch.where(local_valid, attention_weights, torch.zeros_like(attention_weights))
        local_value = torch.bmm(attention_weights.unsqueeze(1), local_values).squeeze(1)
        normalized_query = functional.normalize(query, dim=-1)
        normalized_keys = functional.normalize(local_keys, dim=-1)
        similarity = torch.bmm(normalized_keys, normalized_query.unsqueeze(-1)).squeeze(-1)
        similarity = similarity.masked_fill(~local_valid, NEGATIVE_INFINITY)
        novelty = torch.where(any_slot.squeeze(-1), 1.0 - similarity.amax(dim=-1), torch.ones_like(query[:, 0]))
        return local_value, novelty.clamp(0.0, 1.0)

    def read_window(
        self,
        carried_keys: Tensor,
        carried_values: Tensor,
        carried_valid: Tensor,
        keys: Tensor,
        values: Tensor,
        valid_mask: Tensor,
        queries: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, length, width = queries.shape
        window = self.local_memory_size
        carried_length = carried_keys.shape[1]
        all_keys = torch.cat((carried_keys, keys), dim=1)
        all_values = torch.cat((carried_values, values), dim=1)
        all_valid = torch.cat((carried_valid, valid_mask), dim=1)
        leading_keys = all_keys.new_zeros(batch_size, window, width)
        leading_values = all_values.new_zeros(batch_size, window, width)
        leading_valid = torch.zeros(batch_size, window, dtype=torch.bool, device=queries.device)
        padded_keys = torch.cat((leading_keys, all_keys), dim=1)
        padded_values = torch.cat((leading_values, all_values), dim=1)
        padded_valid = torch.cat((leading_valid, all_valid), dim=1)
        start = carried_length
        key_windows = padded_keys.unfold(1, window, 1)[:, start : start + length]
        value_windows = padded_values.unfold(1, window, 1)[:, start : start + length]
        valid_windows = padded_valid.unfold(1, window, 1)[:, start : start + length]
        scores = torch.einsum("btdw,btd->btw", key_windows, queries) / math.sqrt(width)
        masked_scores = scores.masked_fill(~valid_windows, NEGATIVE_INFINITY)
        any_slot = valid_windows.any(dim=-1, keepdim=True)
        weights = torch.softmax(masked_scores, dim=-1)
        weights = torch.where(any_slot, weights, torch.zeros_like(weights))
        local_value = torch.einsum("btdw,btw->btd", value_windows, weights)
        normalized_queries = functional.normalize(queries, dim=-1)
        normalized_keys = functional.normalize(key_windows, dim=2)
        similarity = torch.einsum("btdw,btd->btw", normalized_keys, normalized_queries)
        similarity = similarity.masked_fill(~valid_windows, NEGATIVE_INFINITY)
        highest = similarity.amax(dim=-1)
        novelty = torch.where(any_slot.squeeze(-1), 1.0 - highest, torch.ones_like(highest))
        return local_value, novelty.clamp(0.0, 1.0)

    def tail(
        self,
        carried_keys: Tensor,
        carried_values: Tensor,
        carried_valid: Tensor,
        keys: Tensor,
        values: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        all_keys = torch.cat((carried_keys, keys), dim=1)
        all_values = torch.cat((carried_values, values), dim=1)
        all_valid = torch.cat((carried_valid, valid_mask), dim=1)
        if all_keys.shape[1] <= self.local_memory_size:
            return all_keys, all_values, all_valid
        return (
            all_keys[:, -self.local_memory_size :],
            all_values[:, -self.local_memory_size :],
            all_valid[:, -self.local_memory_size :],
        )

    def read_salient_window(
        self,
        carried_keys: Tensor,
        carried_values: Tensor,
        carried_valid: Tensor,
        keys: Tensor,
        values: Tensor,
        admission_mask: Tensor,
        queries: Tensor,
    ) -> Tensor:
        batch_size, length, width = queries.shape
        carried_length = carried_keys.shape[1]
        all_keys = torch.cat((carried_keys, keys), dim=1)
        all_values = torch.cat((carried_values, values), dim=1)
        carried_eligible = carried_valid.unsqueeze(1).expand(batch_size, length, carried_length)
        positions = torch.arange(length, device=queries.device)
        prior_positions = positions.unsqueeze(1) > positions.unsqueeze(0)
        current_eligible = admission_mask.unsqueeze(1) & prior_positions.unsqueeze(0)
        eligible = torch.cat((carried_eligible, current_eligible), dim=-1)
        scores = torch.einsum("bsd,btd->bts", all_keys, queries) / math.sqrt(width)
        masked_scores = scores.masked_fill(~eligible, NEGATIVE_INFINITY)
        any_slot = eligible.any(dim=-1, keepdim=True)
        weights = torch.softmax(masked_scores, dim=-1)
        weights = torch.where(any_slot, weights, torch.zeros_like(weights))
        return torch.einsum("bts,bsd->btd", weights, all_values)

    def salient_tail(
        self,
        carried_keys: Tensor,
        carried_values: Tensor,
        carried_valid: Tensor,
        keys: Tensor,
        values: Tensor,
        admission_mask: Tensor,
        capacity: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        all_keys = torch.cat((carried_keys, keys), dim=1)
        all_values = torch.cat((carried_values, values), dim=1)
        all_valid = torch.cat((carried_valid, admission_mask), dim=1)
        if all_keys.shape[1] <= capacity:
            return all_keys, all_values, all_valid
        temporal_indices = torch.arange(all_keys.shape[1], device=all_keys.device).unsqueeze(0).expand_as(all_valid)
        ranked_indices = torch.where(all_valid, temporal_indices, torch.full_like(temporal_indices, -1))
        selected_indices = ranked_indices.topk(capacity, dim=1, largest=True).indices.sort(dim=1).values
        selected_valid = all_valid.gather(1, selected_indices)
        expanded_indices = selected_indices.unsqueeze(-1).expand(-1, -1, all_keys.shape[-1])
        selected_keys = all_keys.gather(1, expanded_indices)
        selected_values = all_values.gather(1, expanded_indices)
        selected_keys = torch.where(selected_valid.unsqueeze(-1), selected_keys, torch.zeros_like(selected_keys))
        selected_values = torch.where(selected_valid.unsqueeze(-1), selected_values, torch.zeros_like(selected_values))
        return selected_keys, selected_values, selected_valid

    def append(
        self,
        local_keys: Tensor,
        local_values: Tensor,
        local_valid: Tensor,
        key: Tensor,
        value: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        next_keys = torch.cat((local_keys, key.unsqueeze(1)), dim=1)
        next_values = torch.cat((local_values, value.unsqueeze(1)), dim=1)
        next_valid = torch.cat((local_valid, valid.unsqueeze(1)), dim=1)
        if next_keys.shape[1] <= self.local_memory_size:
            return next_keys, next_values, next_valid
        return (
            next_keys[:, -self.local_memory_size :],
            next_values[:, -self.local_memory_size :],
            next_valid[:, -self.local_memory_size :],
        )
