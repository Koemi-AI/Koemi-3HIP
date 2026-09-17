from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.experts import UNASSIGNED_EXPERT, DeterministicExpertMixture, content_dispatch_hash
from koemi.model.network import KoemiModel


def build_mixture(expert_count: int, top_k: int = 1) -> DeterministicExpertMixture:
    torch.manual_seed(3)
    return DeterministicExpertMixture(16, expert_count, top_k=top_k)


def assignment_for(mixture: DeterministicExpertMixture, tokens: list[int], previous: list[int]) -> list[int]:
    token_ids = torch.tensor([tokens], dtype=torch.long)
    previous_ids = torch.tensor([previous], dtype=torch.long)
    valid = token_ids != PAD_TOKEN_ID
    return mixture.assign(token_ids, previous_ids, valid)[0].tolist()


def reference_dispatch(
    mixture: DeterministicExpertMixture,
    context: torch.Tensor,
    token_ids: torch.Tensor,
    previous_token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assignments = mixture.assign_top_k(token_ids, previous_token_ids, valid_mask)
    flattened_context = context.reshape(-1, context.shape[-1])
    flattened_assignments = assignments.reshape(-1, mixture.top_k)
    expert_updates = torch.zeros_like(flattened_context)
    for expert_index, expert in enumerate(mixture.experts):
        row_indices = torch.nonzero(
            (flattened_assignments == expert_index).any(dim=1), as_tuple=False
        ).squeeze(-1)
        if row_indices.numel() == 0:
            continue
        expert_context = expert(flattened_context.index_select(0, row_indices))
        expert_updates.index_add_(0, row_indices, expert_context / mixture.top_k)
    valid_rows = valid_mask.reshape(-1)
    updated_context = mixture.output_normalizer(flattened_context + expert_updates)
    mixed_context = torch.where(valid_rows.unsqueeze(-1), updated_context, flattened_context)
    return mixed_context.reshape_as(context), assignments[:, :, 0], assignments


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

    def test_static_dispatch_matches_the_reference_for_top_k_one(self) -> None:
        mixture = build_mixture(4, top_k=1)
        context = torch.randn(2, 4, 16, dtype=torch.float64)
        mixture = mixture.to(dtype=context.dtype)
        token_ids = torch.tensor(
            [[65, 66, PAD_TOKEN_ID, 67], [68, PAD_TOKEN_ID, 69, 70]], dtype=torch.long
        )
        previous_ids = torch.tensor(
            [[32, 65, 66, PAD_TOKEN_ID], [32, 68, PAD_TOKEN_ID, 69]], dtype=torch.long
        )
        valid_mask = token_ids != PAD_TOKEN_ID
        with torch.no_grad():
            expected = reference_dispatch(mixture, context, token_ids, previous_ids, valid_mask)
            actual = mixture(context, token_ids, previous_ids, valid_mask)
        for expected_value, actual_value in zip(expected, actual):
            self.assertEqual(expected_value.shape, actual_value.shape)
            self.assertEqual(expected_value.dtype, actual_value.dtype)
            self.assertEqual(expected_value.device, actual_value.device)
            self.assertTrue(torch.allclose(expected_value, actual_value, atol=1e-6, rtol=1e-6))

    def test_static_dispatch_matches_the_reference_for_top_k_greater_than_one(self) -> None:
        mixture = build_mixture(4, top_k=3)
        context = torch.randn(2, 3, 16)
        token_ids = torch.tensor([[65, 66, PAD_TOKEN_ID], [67, 68, 69]], dtype=torch.long)
        previous_ids = torch.tensor([[32, 65, 66], [32, 67, 68]], dtype=torch.long)
        valid_mask = token_ids != PAD_TOKEN_ID
        with torch.no_grad():
            expected = reference_dispatch(mixture, context, token_ids, previous_ids, valid_mask)
            actual = mixture(context, token_ids, previous_ids, valid_mask)
        self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_invalid_tokens_remain_unassigned_across_padded_batch_lengths(self) -> None:
        mixture = build_mixture(8, top_k=2)
        context = torch.randn(2, 5, 16)
        token_ids = torch.tensor(
            [
                [65, 66, 67, PAD_TOKEN_ID, PAD_TOKEN_ID],
                [68, PAD_TOKEN_ID, 69, 70, PAD_TOKEN_ID],
            ],
            dtype=torch.long,
        )
        previous_ids = torch.tensor(
            [
                [32, 65, 66, 67, PAD_TOKEN_ID],
                [32, 68, PAD_TOKEN_ID, 69, 70],
            ],
            dtype=torch.long,
        )
        valid_mask = token_ids != PAD_TOKEN_ID
        with torch.no_grad():
            mixed_context, primary_assignments, assignments = mixture(
                context, token_ids, previous_ids, valid_mask
            )
        self.assertTrue(torch.equal(mixed_context[~valid_mask], context[~valid_mask]))
        self.assertTrue(
            torch.equal(
                primary_assignments[~valid_mask],
                torch.full_like(primary_assignments[~valid_mask], UNASSIGNED_EXPERT),
            )
        )
        self.assertTrue(
            torch.equal(
                assignments[~valid_mask],
                torch.full_like(assignments[~valid_mask], UNASSIGNED_EXPERT),
            )
        )

    def test_static_dispatch_preserves_parameter_and_context_gradients(self) -> None:
        mixture = build_mixture(4, top_k=2)
        token_ids = torch.tensor([[65, 66, 67, PAD_TOKEN_ID]], dtype=torch.long)
        previous_ids = torch.tensor([[32, 65, 66, 67]], dtype=torch.long)
        valid_mask = token_ids != PAD_TOKEN_ID
        context = torch.randn(1, 4, 16, requires_grad=True)
        assignments = mixture.assign_top_k(token_ids, previous_ids, valid_mask)
        actual_context, _, _ = mixture._forward_with_static_dispatch(
            context, assignments, valid_mask
        )
        actual_context.square().sum().backward()
        actual_context_gradient = context.grad.detach().clone()
        actual_parameter_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in mixture.named_parameters()
        }

        mixture.zero_grad(set_to_none=True)
        reference_context = context.detach().clone().requires_grad_()
        expected_context, _, _ = reference_dispatch(
            mixture, reference_context, token_ids, previous_ids, valid_mask
        )
        expected_context.square().sum().backward()
        self.assertTrue(
            torch.allclose(actual_context_gradient, reference_context.grad, atol=1e-6, rtol=1e-6)
        )
        for name, parameter in mixture.named_parameters():
            with self.subTest(parameter=name):
                expected_gradient = (
                    torch.zeros_like(parameter)
                    if parameter.grad is None
                    else parameter.grad
                )
                self.assertTrue(
                    torch.allclose(
                        actual_parameter_gradients[name],
                        expected_gradient,
                        atol=1e-6,
                        rtol=1e-6,
                    )
                )

    def test_module_dispatch_accepts_autocast_expert_outputs(self) -> None:
        mixture = build_mixture(4, top_k=1)
        context = torch.randn(1, 4, 16)
        token_ids = torch.tensor([[65, 66, 67, PAD_TOKEN_ID]], dtype=torch.long)
        previous_ids = torch.tensor([[32, 65, 66, 67]], dtype=torch.long)
        valid_mask = token_ids != PAD_TOKEN_ID

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            mixed_context, _, _ = mixture(context, token_ids, previous_ids, valid_mask)

        self.assertEqual(context.dtype, mixed_context.dtype)
        self.assertTrue(bool(torch.isfinite(mixed_context).all()))


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
        with torch.no_grad():
            output = model(input_ids)
        self.assertEqual(5, sum(output.expert_activation_counts))
        self.assertEqual(1, int((output.expert_indices == UNASSIGNED_EXPERT).sum()))

    def test_a_repeated_bigram_keeps_its_expert_across_a_longer_sequence(self) -> None:
        model = self.build_model(16)
        pattern = [97, 98] * 16
        with torch.no_grad():
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
        with torch.no_grad():
            output = model(torch.tensor([[72, 101, 108, 108, 111]], dtype=torch.long))
        self.assertEqual((1, 5, 6), tuple(output.active_expert_indices.shape))
        self.assertEqual(30, sum(output.expert_activation_counts))
