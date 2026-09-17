from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from koemi.model.context_summary import (
    ContextSummary,
    ContextSummaryState,
    SurpriseMemory,
)


class ContextSummaryTests(unittest.TestCase):
    def test_empty_read_is_none_and_reset_clears_evidence(self) -> None:
        summary = ContextSummary(3, 4)
        state = summary.create_state(2)

        self.assertIsNone(summary.read(state))
        updated = summary.update(state, torch.ones(2, 4))
        self.assertIsNotNone(summary.read(updated))
        reset = summary.reset(updated)

        self.assertEqual(0, reset.step_index)
        self.assertTrue(torch.equal(reset.slots, torch.zeros_like(reset.slots)))
        self.assertTrue(torch.equal(reset.evidence, torch.zeros_like(reset.evidence)))
        self.assertIsNone(summary.read(reset))

    def test_updates_are_causal_and_do_not_mutate_the_prior_state(self) -> None:
        summary = ContextSummary(2, 3, decays=[0.5, 0.75])
        empty = summary.create_state(1)
        prefix_state = summary.update(empty, torch.tensor([[0.2, -0.4, 0.6]]))
        prefix_read = summary.read(prefix_state)
        prefix_slots = prefix_state.slots.clone()
        prefix_evidence = prefix_state.evidence.clone()
        future_a = summary.update(prefix_state, torch.tensor([[0.9, 0.9, 0.9]]))
        future_b = summary.update(prefix_state, torch.tensor([[-0.9, -0.9, -0.9]]))

        self.assertIsNotNone(prefix_read)
        self.assertTrue(torch.equal(prefix_slots, prefix_state.slots))
        self.assertTrue(torch.equal(prefix_evidence, prefix_state.evidence))
        self.assertTrue(torch.equal(prefix_read.value, summary.read(prefix_state).value))
        self.assertEqual(1, prefix_state.step_index)
        self.assertTrue(torch.equal(future_a.evidence, prefix_state.evidence + 1))
        self.assertTrue(torch.equal(future_b.evidence, prefix_state.evidence + 1))
        self.assertFalse(torch.equal(future_a.slots, future_b.slots))
        self.assertFalse(torch.equal(prefix_state.slots, empty.slots))

    def test_slots_reads_and_confidence_are_bounded(self) -> None:
        summary = ContextSummary(4, 3, max_evidence=2)
        state = summary.create_state(1)
        for _ in range(8):
            state = summary.update(state, torch.tensor([[1.0e20, -1.0e20, 1.0e20]]))
        result = summary.read(state)

        self.assertIsNotNone(result)
        self.assertLessEqual(float(state.slots.abs().max()), 1.0)
        self.assertLessEqual(float(result.value.abs().max()), 1.0)
        self.assertGreaterEqual(float(result.confidence.min()), 0.0)
        self.assertLessEqual(float(result.confidence.max()), 1.0)
        self.assertTrue(torch.all(state.evidence == 2))

    def test_confidence_only_gates_a_convex_read(self) -> None:
        summary = ContextSummary(2, 2, decays=[0.5, 0.5], confidence_scale=10.0)
        state = summary.update(summary.create_state(1), torch.tensor([[0.8, -0.4]]))
        result = summary.read(state)

        self.assertIsNotNone(result)
        self.assertGreater(float(result.confidence.item()), 0.0)
        self.assertLessEqual(float(result.confidence.item()), 1.0)
        raw_value = state.slots.mean(dim=1)
        self.assertLessEqual(float(result.value.norm()), float(raw_value.norm()) + 1.0e-6)

    def test_float16_read_keeps_large_evidence_counts_finite(self) -> None:
        summary = ContextSummary(128, 1, max_evidence=1024)
        state = ContextSummaryState(
            slots=torch.ones(1, 128, 1, dtype=torch.float16),
            evidence=torch.full((1, 128), 1024, dtype=torch.int64),
            step_index=1,
        )

        result = summary.read(state)

        self.assertIsNotNone(result)
        self.assertTrue(torch.isfinite(result.value).all())
        self.assertTrue(torch.isfinite(result.confidence).all())

    def test_trainable_decay_and_values_receive_gradients(self) -> None:
        summary = ContextSummary(2, 3, trainable=True)
        values = torch.tensor([[0.25, -0.5, 0.75]], requires_grad=True)
        state = summary.update(summary.create_state(1), values)
        result = summary.read(state)

        self.assertIsNotNone(result)
        result.value.square().sum().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in summary.parameters()))

    def test_detach_separates_the_next_state_from_training_graph(self) -> None:
        summary = ContextSummary(2, 2, trainable=True)
        values = torch.ones(1, 2, requires_grad=True)
        state = summary.update(summary.create_state(1), values, detach=True)

        self.assertFalse(state.slots.requires_grad)
        self.assertFalse(state.evidence.requires_grad)

    def test_masks_control_updates_and_evidence_counts(self) -> None:
        summary = ContextSummary(2, 2, measure_cost=True)
        state = summary.create_state(2)
        state = summary.update(
            state,
            torch.ones(2, 2),
            valid_mask=torch.tensor([True, False]),
            slot_mask=torch.tensor([[True, False], [True, True]]),
        )

        self.assertTrue(torch.equal(state.evidence, torch.tensor([[1, 0], [0, 0]], dtype=torch.int64)))
        self.assertEqual(1, summary.cost_statistics().update_calls)
        self.assertEqual(1, summary.cost_statistics().active_slot_updates)
        self.assertEqual(8, summary.cost_statistics().estimated_element_updates)

    def test_hot_update_and_device_read_do_not_extract_host_scalars(self) -> None:
        summary = ContextSummary(2, 2, measure_cost=True)
        state = summary.create_state(1)

        with patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar read")), patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("host tensor copy")
        ):
            updated = summary.update(state, torch.ones(1, 2))
            result = summary.read_device(updated)

        self.assertTrue(torch.isfinite(result.value).all())
        self.assertEqual(2, summary.cost_statistics().active_slot_updates)

    def test_surprise_memory_is_causal_bounded_and_opt_in(self) -> None:
        memory = SurpriseMemory(3, max_evidence=2)
        empty = memory.create_state(1)
        values = torch.tensor([[0.2, -0.4, 0.6]])
        first = memory.update(empty, values, torch.tensor([1.0]))
        first_value = first.value.clone()
        future_a = memory.update(first, torch.full_like(values, 1.0e20), torch.tensor([2.0]))
        future_b = memory.update(first, torch.full_like(values, -1.0e20), torch.tensor([-2.0]))
        limited = memory.update(future_a, values, torch.tensor([1.0]))
        result = memory.read(limited)

        self.assertTrue(torch.equal(first.value, first_value))
        self.assertFalse(torch.equal(future_a.value, future_b.value))
        self.assertTrue(torch.isfinite(result.value).all())
        self.assertTrue(torch.isfinite(result.confidence).all())
        self.assertLessEqual(float(limited.value.abs().max()), 1.0)
        self.assertLessEqual(float(limited.momentum.abs().max()), 1.0)
        self.assertTrue(torch.all(limited.evidence == 2))

    def test_surprise_memory_masks_invalid_steps_and_rejects_non_finite_input(self) -> None:
        memory = SurpriseMemory(2)
        state = memory.create_state(1)
        masked = memory.update(
            state,
            torch.ones(1, 2),
            torch.ones(1),
            valid_mask=torch.tensor([False]),
        )

        self.assertTrue(torch.equal(masked.value, state.value))
        self.assertTrue(torch.equal(masked.momentum, state.momentum))
        self.assertTrue(torch.equal(masked.evidence, state.evidence))
        self.assertEqual(1, masked.step_index)
        with self.assertRaises(ValueError):
            memory.update(state, torch.tensor([[float("nan"), 0.0]]), torch.ones(1))

    def test_shape_dtype_device_and_limit_validation(self) -> None:
        summary = ContextSummary(2, 3, max_batch_size=1)
        state = summary.create_state(1)

        with self.assertRaises(ValueError):
            summary.update(state, torch.ones(1, 2))
        with self.assertRaises(TypeError):
            summary.update(state, torch.ones(1, 3, dtype=torch.int64))
        with self.assertRaises(TypeError):
            summary.update(state, torch.ones(1, 3), valid_mask=torch.ones(1, dtype=torch.int64))
        with self.assertRaises(ValueError):
            summary.create_state(2)
        with self.assertRaises(ValueError):
            summary.create_state(1, device="meta")
        with self.assertRaises(ValueError):
            summary.update(
                state,
                torch.ones(1, 3),
                slot_mask=torch.ones(1, 1, dtype=torch.bool),
            )

    def test_serialization_is_primitive_data_and_rejects_invalid_payloads(self) -> None:
        summary = ContextSummary(2, 3)
        state = summary.update(summary.create_state(1), torch.tensor([[0.2, 0.3, -0.4]]))
        payload = summary.serialize(state)

        self.assertTrue(all(not isinstance(value, torch.Tensor) for value in payload.values()))
        restored = summary.deserialize(payload)
        self.assertTrue(torch.equal(state.slots, restored.slots))
        self.assertTrue(torch.equal(state.evidence, restored.evidence))
        self.assertEqual(state.step_index, restored.step_index)

        invalid_payloads = []
        tensor_payload = copy.deepcopy(payload)
        tensor_payload["slots"] = torch.zeros(1, 2, 3)
        invalid_payloads.append(tensor_payload)
        object_payload = copy.deepcopy(payload)
        object_payload["evidence"][0][0] = object()
        invalid_payloads.append(object_payload)
        shape_payload = copy.deepcopy(payload)
        shape_payload["shape"] = [1, 2, 4]
        invalid_payloads.append(shape_payload)
        extra_key_payload = copy.deepcopy(payload)
        extra_key_payload["session_id"] = "private"
        invalid_payloads.append(extra_key_payload)

        for invalid_payload in invalid_payloads:
            with self.subTest(payload=invalid_payload):
                with self.assertRaises((TypeError, ValueError)):
                    summary.deserialize(invalid_payload)

    def test_default_forward_path_does_not_construct_context_summary(self) -> None:
        from koemi.model.network import KoemiModel
        from koemi.configuration.settings import ModelSettings

        model = KoemiModel(ModelSettings(embedding_size=8, memory_features=2))

        self.assertFalse(
            any(isinstance(module, (ContextSummary, SurpriseMemory)) for module in model.modules())
        )


if __name__ == "__main__":
    unittest.main()
