from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from koemi.configuration.settings import PAD_TOKEN_ID


@dataclass(frozen=True)
class KoemiState:
    working_state: Tensor
    memory_basis: Tensor
    memory_normalizer: Tensor
    refine_basis: Tensor
    refine_normalizer: Tensor
    local_keys: Tensor
    local_values: Tensor
    local_valid: Tensor
    salient_keys: Tensor
    salient_values: Tensor
    salient_valid: Tensor
    last_token_ids: Tensor
    step_index: int

    @classmethod
    def create(cls, batch_size: int, embedding_size: int, memory_features: int, device: torch.device) -> KoemiState:
        return cls(
            working_state=torch.zeros(batch_size, embedding_size, device=device),
            memory_basis=torch.zeros(batch_size, embedding_size, memory_features, device=device),
            memory_normalizer=torch.zeros(batch_size, memory_features, device=device),
            refine_basis=torch.zeros(batch_size, embedding_size, memory_features, device=device),
            refine_normalizer=torch.zeros(batch_size, memory_features, device=device),
            local_keys=torch.empty(batch_size, 0, embedding_size, device=device),
            local_values=torch.empty(batch_size, 0, embedding_size, device=device),
            local_valid=torch.empty(batch_size, 0, dtype=torch.bool, device=device),
            salient_keys=torch.empty(batch_size, 0, embedding_size, device=device),
            salient_values=torch.empty(batch_size, 0, embedding_size, device=device),
            salient_valid=torch.empty(batch_size, 0, dtype=torch.bool, device=device),
            last_token_ids=torch.full((batch_size,), PAD_TOKEN_ID, dtype=torch.long, device=device),
            step_index=0,
        )
