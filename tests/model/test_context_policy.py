from __future__ import annotations

import unittest

import torch

from koemi.model.context_policy import AdmissionReason, ContextPolicy


class ContextPolicyTests(unittest.TestCase):
    def test_selection_is_causal_at_an_explicit_horizon(self) -> None:
        policy = ContextPolicy(
            2,
            enabled=True,
            surprise_weight=1.0,
            recency_weight=0.0,
            diversity_weight=0.0,
        )
        positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        first_scores = torch.tensor([[0.2, 0.8, 0.1, 100.0]])
        second_scores = torch.tensor([[0.2, 0.8, 0.1, -100.0]])
        first = policy.select(first_scores, positions, current_position=2)
        second = policy.select(second_scores, positions, current_position=2)

        self.assertTrue(torch.equal(first.selected_indices, second.selected_indices))
        self.assertTrue(torch.equal(first.priority_scores[:, :3], second.priority_scores[:, :3]))
        self.assertEqual(int(first.reason_codes[0, 3]), int(AdmissionReason.FUTURE))

    def test_capacity_and_tie_break_keep_newer_positions(self) -> None:
        policy = ContextPolicy(
            2,
            enabled=True,
            surprise_weight=1.0,
            recency_weight=0.0,
            diversity_weight=0.0,
        )
        result = policy.select(
            torch.ones(1, 4),
            torch.arange(4, dtype=torch.long).unsqueeze(0),
        )

        self.assertTrue(torch.equal(result.selected_indices, torch.tensor([[2, 3]])))
        self.assertEqual(int(result.selected_counts[0]), 2)
        self.assertEqual(int(result.admission_counts[0]), 4)
        self.assertEqual(int(result.reason_codes[0, 0]), int(AdmissionReason.EVICTED))
        self.assertEqual(int(result.reason_codes[0, 1]), int(AdmissionReason.EVICTED))
        self.assertEqual(int(result.reason_codes[0, 2]), int(AdmissionReason.REPLACED))
        self.assertTrue(bool(torch.isfinite(result.selected_priority_scores).all()))

    def test_surprise_and_recency_are_independent_components(self) -> None:
        positions = torch.tensor([[10, 11, 12]], dtype=torch.long)
        scores = torch.tensor([[0.9, 0.1, 0.2]])
        surprise_policy = ContextPolicy(
            1,
            enabled=True,
            surprise_weight=1.0,
            recency_weight=0.0,
            diversity_weight=0.0,
        )
        recency_policy = ContextPolicy(
            1,
            enabled=True,
            surprise_weight=0.0,
            recency_weight=1.0,
            diversity_weight=0.0,
        )

        surprise_result = surprise_policy.select(scores, positions)
        recency_result = recency_policy.select(scores, positions)

        self.assertTrue(torch.equal(surprise_result.selected_indices, torch.tensor([[0]])))
        self.assertTrue(torch.equal(recency_result.selected_indices, torch.tensor([[2]])))

    def test_equal_position_ties_keep_the_lower_input_index(self) -> None:
        policy = ContextPolicy(
            1,
            enabled=True,
            surprise_weight=1.0,
            recency_weight=0.0,
            diversity_weight=0.0,
        )
        result = policy.select(
            torch.ones(1, 2),
            torch.tensor([[5, 5]], dtype=torch.long),
            entry_ids=torch.tensor([[10, 11]], dtype=torch.long),
        )

        self.assertTrue(torch.equal(result.selected_indices, torch.tensor([[0]])))
        self.assertEqual(int(result.reason_codes[0, 1]), int(AdmissionReason.CAPACITY))

    def test_diversity_replaces_a_redundant_entry(self) -> None:
        policy = ContextPolicy(
            2,
            enabled=True,
            surprise_weight=0.0,
            recency_weight=0.0,
            diversity_weight=1.0,
        )
        features = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
        result = policy.select(
            torch.ones(1, 3),
            torch.arange(3, dtype=torch.long).unsqueeze(0),
            features=features,
        )

        self.assertTrue(torch.equal(result.selected_indices, torch.tensor([[0, 2]])))
        self.assertEqual(int(result.reason_codes[0, 1]), int(AdmissionReason.EVICTED))

    def test_masks_deduplication_and_reasons_are_exposed(self) -> None:
        policy = ContextPolicy(
            3,
            enabled=True,
            surprise_weight=1.0,
            recency_weight=0.0,
            diversity_weight=0.0,
        )
        result = policy.select(
            torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.7]]),
            torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
            current_position=3,
            valid_mask=torch.tensor([[True, False, True, True, True]]),
            entry_ids=torch.tensor([[4, 5, 4, 6, 7]], dtype=torch.long),
        )

        self.assertTrue(torch.equal(result.selected_indices, torch.tensor([[0, 3, -1]])))
        self.assertFalse(bool(result.selected_mask[0, 1]))
        self.assertEqual(int(result.reason_codes[0, 1]), int(AdmissionReason.INVALID))
        self.assertEqual(int(result.reason_codes[0, 2]), int(AdmissionReason.DUPLICATE))
        self.assertEqual(int(result.reason_codes[0, 4]), int(AdmissionReason.FUTURE))
        self.assertEqual(int(result.reason_counts[0].sum()), 5)

    def test_disabled_policy_emits_no_admissions(self) -> None:
        policy = ContextPolicy(2)
        result = policy.select(
            torch.ones(1, 3),
            torch.arange(3, dtype=torch.long).unsqueeze(0),
        )

        self.assertTrue(torch.equal(result.selected_indices, torch.tensor([[-1, -1]])))
        self.assertFalse(bool(result.admitted_mask.any()))
        self.assertTrue(torch.equal(result.reason_counts[0], torch.tensor([0, 0, 0, 0, 0, 0, 0, 3])))

    def test_selection_is_deterministic_and_preserves_score_gradients(self) -> None:
        policy = ContextPolicy(
            2,
            enabled=True,
            surprise_weight=1.0,
            recency_weight=0.0,
            diversity_weight=0.0,
        )
        scores = torch.tensor([[0.2, 0.8, 0.4]], requires_grad=True)
        positions = torch.tensor([[0, 1, 2]], dtype=torch.long)
        first = policy.select(scores, positions)
        second = policy.select(scores, positions)

        self.assertTrue(torch.equal(first.selected_indices, second.selected_indices))
        self.assertTrue(torch.equal(first.reason_codes, second.reason_codes))
        self.assertFalse(first.priority_scores.requires_grad)
        self.assertTrue(torch.equal(scores, torch.tensor([[0.2, 0.8, 0.4]])))
        scores.sum().backward()
        self.assertTrue(torch.equal(scores.grad, torch.ones_like(scores)))

    def test_batch_horizons_and_empty_candidate_sets_remain_bounded(self) -> None:
        policy = ContextPolicy(
            2,
            enabled=True,
            surprise_weight=0.0,
            recency_weight=1.0,
            diversity_weight=0.0,
        )
        scores = torch.ones(2, 3)
        positions = torch.tensor([[0, 1, 2], [10, 11, 12]], dtype=torch.long)
        horizons = torch.tensor([1, 11], dtype=torch.long)
        result = policy.select(scores, positions, current_position=horizons)
        empty = policy.select(
            torch.empty(2, 0),
            torch.empty(2, 0, dtype=torch.long),
        )
        maximum_position = torch.iinfo(torch.long).max
        maximum_position_result = policy.select(
            torch.ones(1, 1),
            torch.tensor([[maximum_position]], dtype=torch.long),
        )

        self.assertTrue(torch.equal(result.selected_indices, torch.tensor([[0, 1], [0, 1]])))
        self.assertEqual(int(result.reason_codes[0, 2]), int(AdmissionReason.FUTURE))
        self.assertEqual(int(result.reason_codes[1, 2]), int(AdmissionReason.FUTURE))
        self.assertTrue(torch.equal(empty.selected_indices, torch.full((2, 2), -1, dtype=torch.long)))
        self.assertEqual(tuple(empty.reason_counts.shape), (2, 8))
        self.assertTrue(torch.equal(maximum_position_result.selected_indices, torch.tensor([[0, -1]])))

    def test_invalid_capacity_shape_and_non_finite_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ContextPolicy(0)
        policy = ContextPolicy(1, enabled=True)
        with self.assertRaises(ValueError):
            policy.select(torch.ones(1, 2), torch.ones(1, 3, dtype=torch.long))
        with self.assertRaises(RuntimeError):
            policy.select(
                torch.tensor([[float("nan")]]),
                torch.zeros(1, 1, dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()
