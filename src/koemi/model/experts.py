from __future__ import annotations

import torch
from torch import Tensor, nn

from koemi.model.layers import GatedFeedForward, RootMeanSquareNorm


TOKEN_HASH_FACTOR = 1_000_003
PREVIOUS_TOKEN_HASH_FACTOR = 97_409
HASH_WIDTH_MASK = 0xFFFFFFFF
HASH_MIX_MULTIPLIER = 0x45D9F3B
HASH_MIX_SHIFT = 16
UNASSIGNED_EXPERT = -1


def content_dispatch_hash(token_ids: Tensor, previous_token_ids: Tensor) -> Tensor:
    value = (
        token_ids * TOKEN_HASH_FACTOR + previous_token_ids * PREVIOUS_TOKEN_HASH_FACTOR
    ) & HASH_WIDTH_MASK
    value = value ^ (value >> HASH_MIX_SHIFT)
    value = (value * HASH_MIX_MULTIPLIER) & HASH_WIDTH_MASK
    value = value ^ (value >> HASH_MIX_SHIFT)
    value = (value * HASH_MIX_MULTIPLIER) & HASH_WIDTH_MASK
    return value ^ (value >> HASH_MIX_SHIFT)


class DeterministicExpertMixture(nn.Module):
    def __init__(self, embedding_size: int, expert_count: int, top_k: int = 1) -> None:
        super().__init__()
        self.expert_count = expert_count
        self.top_k = top_k
        if expert_count < 0 or top_k < 1 or (expert_count > 0 and top_k > expert_count):
            raise ValueError("top_k must be between one and expert_count")
        self.experts = nn.ModuleList(GatedFeedForward(embedding_size) for _ in range(expert_count))
        self.output_normalizer = RootMeanSquareNorm(embedding_size)

    def forward(
        self,
        context: Tensor,
        token_ids: Tensor,
        previous_token_ids: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.expert_count == 0:
            empty = torch.full_like(token_ids, UNASSIGNED_EXPERT)
            return context, empty, empty.unsqueeze(-1)
        assignments = self.assign_top_k(token_ids, previous_token_ids, valid_mask)
        flattened_context = context.reshape(-1, context.shape[-1])
        flattened_assignments = assignments.reshape(-1, self.top_k)
        mixed_context = flattened_context.clone()
        expert_updates = torch.zeros_like(flattened_context)
        for expert_index, expert in enumerate(self.experts):
            row_indices = torch.nonzero((flattened_assignments == expert_index).any(dim=1), as_tuple=False).squeeze(-1)
            if row_indices.numel() == 0:
                continue
            expert_context = expert(flattened_context.index_select(0, row_indices))
            expert_updates.index_add_(0, row_indices, expert_context / self.top_k)
        valid_rows = valid_mask.reshape(-1)
        updated_context = self.output_normalizer(flattened_context + expert_updates)
        mixed_context = torch.where(valid_rows.unsqueeze(-1), updated_context, flattened_context)
        return mixed_context.reshape_as(context), assignments[:, :, 0], assignments

    def assign(self, token_ids: Tensor, previous_token_ids: Tensor, valid_mask: Tensor) -> Tensor:
        return self.assign_top_k(token_ids, previous_token_ids, valid_mask)[:, :, 0]

    def assign_top_k(self, token_ids: Tensor, previous_token_ids: Tensor, valid_mask: Tensor) -> Tensor:
        if self.expert_count == 0:
            return torch.full((*token_ids.shape, self.top_k), UNASSIGNED_EXPERT, dtype=torch.long, device=token_ids.device)
        context_hash = content_dispatch_hash(token_ids, previous_token_ids)
        offsets = torch.arange(self.top_k, device=token_ids.device, dtype=context_hash.dtype)
        assignments = (context_hash.unsqueeze(-1) + offsets * 0x9E3779B9).remainder(self.expert_count)
        return assignments.masked_fill(~valid_mask.unsqueeze(-1), UNASSIGNED_EXPERT)
