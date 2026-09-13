from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.experts import UNASSIGNED_EXPERT, DeterministicExpertMixture, content_dispatch_hash
from koemi.model.network import KoemiModel


def build_mixture(expert_count: int) -> DeterministicExpertMixture:
    torch.manual_seed(3)
    return DeterministicExpertMixture(16, expert_count)


def assignment_for(mixture: DeterministicExpertMixture, tokens: list[int], previous: list[int]) -> list[int]:
    token_ids = torch.tensor([tokens], dtype=torch.long)
    previous_ids = torch.tensor([previous], dtype=torch.long)
    valid = token_ids != PAD_TOKEN_ID
    return mixture.assign(token_ids, previous_ids, valid)[0].tolist()


class ContentDispatchTests(unittest.TestCase):
    def test_top_k_assignment_returns_six_distinct_experts(self) -> None:
        mixture = DeterministicExpertMixture(16, 128, top_k=6)
        token_ids = torch.tensor([[97, 98, 99]], dtype=torch.long)
        previous_ids = torch.tensor([[32, 97, 98]], dtype=torch.long)
        assignments = mixture.assign_top_k(token_ids, previous_ids, torch.ones_like(token_ids, dtype=torch.bool))
        self.assertEqual((1, 3, 6), tuple(assignments.shape))
        self.assertTrue(bool((assignments >= 0).all()))
        self.assertEqual(6, assignments.unique(dim=-1).shape[-1])

    def test_the_same_bigram_reaches_the_same_expert_at_every_position(self) -> None:
        mixture = build_mixture(64)
        tokens = [97, 98, 97, 98, 97, 98]
        previous = [32, 97, 98, 97, 98, 97]
        assignment = assignment_for(mixture, tokens, previous)
        self.assertEqual(assignment[1], assignment[3])
        self.assertEqual(assignment[3], assignment[5])
        self.assertEqual(assignment[2], assignment[4])

    def test_the_assignment_ignores_absolute_position(self) -> None:
        mixture = build_mixture(64)
        early = assignment_for(mixture, [97] * 4, [32] * 4)
        self.assertEqual(1, len(set(early)))

    def test_consecutive_bytes_reach_most_of_the_bank(self) -> None:
        mixture = build_mixture(64)
        tokens = list(range(1, 256))
        previous = [token - 1 for token in tokens]
        reachable = set(assignment_for(mixture, tokens, previous))
        self.assertGreaterEqual(len(reachable), 48)

    def test_padding_stays_unassigned(self) -> None:
        mixture = build_mixture(8)
        assignment = assignment_for(mixture, [65, PAD_TOKEN_ID, 66], [32, 65, PAD_TOKEN_ID])
        self.assertEqual(UNASSIGNED_EXPERT, assignment[1])
        self.assertNotEqual(UNASSIGNED_EXPERT, assignment[0])
        self.assertNotEqual(UNASSIGNED_EXPERT, assignment[2])

    def test_the_hash_uses_more_than_the_low_bits(self) -> None:
        tokens = torch.arange(1, 256, dtype=torch.long)
        previous = tokens - 1
        mixed = content_dispatch_hash(tokens, previous)
        affine = tokens * 1_000_003 + previous * 97_409
        for expert_count in (4, 64):
            with self.subTest(expert_count=expert_count):
                mixed_buckets = len(set(mixed.remainder(expert_count).tolist()))
                affine_buckets = len(set(affine.remainder(expert_count).tolist()))
                self.assertGreater(mixed_buckets, affine_buckets)

    def test_the_hash_is_deterministic(self) -> None:
        tokens = torch.tensor([[10, 200, 255]], dtype=torch.long)
        previous = torch.tensor([[1, 99, 128]], dtype=torch.long)
        self.assertTrue(
            torch.equal(content_dispatch_hash(tokens, previous), content_dispatch_hash(tokens, previous))
        )

    def test_the_hash_stays_non_negative(self) -> None:
        tokens = torch.arange(0, 257, dtype=torch.long)
        previous = torch.full_like(tokens, PAD_TOKEN_ID)
        self.assertTrue(bool((content_dispatch_hash(tokens, previous) >= 0).all()))


class DispatchThroughTheModelTests(unittest.TestCase):
    def build_model(self, expert_count: int) -> KoemiModel:
        torch.manual_seed(0)
        return KoemiModel(
            ModelSettings(
                embedding_size=16,
                memory_features=4,
                local_memory_size=4,
                expert_count=expert_count,
            )
        )

    def test_every_valid_token_reaches_one_expert(self) -> None:
        model = self.build_model(8)
        input_ids = torch.tensor([[72, 101, 108, 108, 111, PAD_TOKEN_ID]], dtype=torch.long)
        output = model(input_ids)
        self.assertEqual(5, sum(output.expert_activation_counts))
        self.assertEqual(1, int((output.expert_indices == UNASSIGNED_EXPERT).sum()))

    def test_a_repeated_bigram_keeps_its_expert_across_a_longer_sequence(self) -> None:
        model = self.build_model(16)
        pattern = [97, 98] * 16
        output = model(torch.tensor([pattern], dtype=torch.long))
        assignment = output.expert_indices[0].tolist()
        self.assertEqual(1, len(set(assignment[3::2])))
        self.assertEqual(1, len(set(assignment[2::2])))

    def test_six_experts_are_active_per_valid_token_with_a_128_expert_bank(self) -> None:
        model = KoemiModel(
            ModelSettings(
                embedding_size=16,
                memory_features=4,
                local_memory_size=4,
                expert_count=128,
                expert_top_k=6,
            )
        )
        output = model(torch.tensor([[72, 101, 108, 108, 111]], dtype=torch.long))
        self.assertEqual((1, 5, 6), tuple(output.active_expert_indices.shape))
        self.assertEqual(30, sum(output.expert_activation_counts))
