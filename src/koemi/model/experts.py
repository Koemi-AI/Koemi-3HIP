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
    def __init__(self, embedding_size: int, expert_count: int) -> None:
        super().__init__()
        self.expert_count = expert_count
        self.experts = nn.ModuleList(GatedFeedForward(embedding_size) for _ in range(expert_count))
        self.output_normalizer = RootMeanSquareNorm(embedding_size)

    def forward(
        self,
        context: Tensor,
        token_ids: Tensor,
        previous_token_ids: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.expert_count == 0:
            return context, torch.full_like(token_ids, UNASSIGNED_EXPERT)
        assignment = self.assign(token_ids, previous_token_ids, valid_mask)
        flattened_context = context.reshape(-1, context.shape[-1])
        flattened_assignment = assignment.reshape(-1)
        mixed_context = context.reshape(-1, context.shape[-1]).clone()
        for expert_index, expert in enumerate(self.experts):
            row_indices = torch.nonzero(flattened_assignment == expert_index, as_tuple=False).squeeze(-1)
            if row_indices.numel() == 0:
                continue
            expert_context = expert(flattened_context.index_select(0, row_indices))
            updated_context = self.output_normalizer(
                flattened_context.index_select(0, row_indices) + expert_context
            )
            mixed_context.index_copy_(0, row_indices, updated_context)
        return mixed_context.reshape_as(context), assignment

    def assign(self, token_ids: Tensor, previous_token_ids: Tensor, valid_mask: Tensor) -> Tensor:
        context_hash = content_dispatch_hash(token_ids, previous_token_ids)
        return context_hash.remainder(self.expert_count).masked_fill(~valid_mask, UNASSIGNED_EXPERT)
