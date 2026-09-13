from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel


def build_model(**overrides) -> KoemiModel:
    settings = dict(
        embedding_size=24,
        memory_features=6,
        local_memory_size=5,
        expert_count=2,
        scan_chunk=7,
    )
    settings.update(overrides)
    return KoemiModel(ModelSettings(**settings))


class ExecutionEquivalenceTests(unittest.TestCase):
    def assert_paths_agree(self, model: KoemiModel, input_ids: torch.Tensor) -> None:
        with torch.no_grad():
            parallel = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
            sequential = model(input_ids, execution_mode=ExecutionMode.SEQUENTIAL)
        self.assertTrue(
            torch.allclose(parallel.logits, sequential.logits, atol=1e-4),
            f"logits differ by {float((parallel.logits - sequential.logits).abs().max())}",
        )
        self.assertTrue(torch.allclose(parallel.surprise_values, sequential.surprise_values, atol=1e-4))
        self.assertTrue(torch.equal(parallel.expert_indices, sequential.expert_indices))
        self.assertEqual(parallel.token_count, sequential.token_count)
        self.assertTrue(torch.allclose(parallel.state.working_state, sequential.state.working_state, atol=1e-4))
        self.assertTrue(torch.allclose(parallel.state.memory_basis, sequential.state.memory_basis, atol=1e-4))
        self.assertTrue(torch.allclose(parallel.state.memory_normalizer, sequential.state.memory_normalizer, atol=1e-4))
        self.assertTrue(torch.allclose(parallel.state.refine_basis, sequential.state.refine_basis, atol=1e-4))
        self.assertTrue(torch.allclose(parallel.state.refine_normalizer, sequential.state.refine_normalizer, atol=1e-4))
        self.assertTrue(torch.allclose(parallel.state.local_keys, sequential.state.local_keys, atol=1e-4))
        self.assertTrue(torch.equal(parallel.state.local_valid, sequential.state.local_valid))
        self.assertTrue(torch.allclose(parallel.state.salient_keys, sequential.state.salient_keys, atol=1e-4))
        self.assertTrue(torch.equal(parallel.state.salient_valid, sequential.state.salient_valid))
        self.assertTrue(torch.equal(parallel.state.last_token_ids, sequential.state.last_token_ids))
        self.assertEqual(parallel.state.step_index, sequential.state.step_index)

    def test_paths_agree_for_a_long_sequence(self) -> None:
        torch.manual_seed(0)
        model = build_model()
        model.eval()
        input_ids = torch.randint(0, 255, (3, 40), dtype=torch.long)
        self.assert_paths_agree(model, input_ids)

    def test_paths_agree_with_padded_rows(self) -> None:
        torch.manual_seed(1)
        model = build_model()
        model.eval()
        input_ids = torch.tensor(
            [[65, 66, 67, 68, 69, 70], [71, 72, 73, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]],
            dtype=torch.long,
        )
        self.assert_paths_agree(model, input_ids)

    def test_paths_agree_when_a_state_is_carried_in(self) -> None:
        torch.manual_seed(2)
        model = build_model()
        model.eval()
        prefix = torch.randint(0, 255, (2, 12), dtype=torch.long)
        suffix = torch.randint(0, 255, (2, 9), dtype=torch.long)
        with torch.no_grad():
            carried = model(prefix).state
        self.assert_paths_agree_with_state(model, suffix, carried)

    def assert_paths_agree_with_state(self, model: KoemiModel, input_ids: torch.Tensor, state) -> None:
        with torch.no_grad():
            parallel = model(input_ids, state, execution_mode=ExecutionMode.PARALLEL)
            sequential = model(input_ids, state, execution_mode=ExecutionMode.SEQUENTIAL)
        self.assertTrue(torch.allclose(parallel.logits, sequential.logits, atol=1e-4))
        self.assertTrue(torch.allclose(parallel.state.refine_basis, sequential.state.refine_basis, atol=1e-4))
        self.assertTrue(torch.allclose(parallel.state.local_keys, sequential.state.local_keys, atol=1e-4))
        self.assertTrue(torch.equal(parallel.state.local_valid, sequential.state.local_valid))
        self.assertTrue(torch.allclose(parallel.state.salient_keys, sequential.state.salient_keys, atol=1e-4))
        self.assertTrue(torch.equal(parallel.state.salient_valid, sequential.state.salient_valid))
        self.assertTrue(torch.equal(parallel.state.last_token_ids, sequential.state.last_token_ids))

    def test_one_pass_agrees_with_step_by_step_decoding(self) -> None:
        torch.manual_seed(3)
        model = build_model()
        model.eval()
        input_ids = torch.randint(0, 255, (2, 11), dtype=torch.long)
        with torch.no_grad():
            whole = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
            state = None
            stepped = []
            for position in range(input_ids.shape[1]):
                output = model(input_ids[:, position : position + 1], state, execution_mode=ExecutionMode.PARALLEL)
                stepped.append(output.logits)
                state = output.state
        self.assertTrue(torch.allclose(whole.logits, torch.cat(stepped, dim=1), atol=1e-4))

    def test_gradients_agree_between_paths(self) -> None:
        for ablation in ("no_refine", "herm"):
            with self.subTest(ablation=ablation):
                torch.manual_seed(4)
                model = build_model(expert_count=0, ablation=ablation)
                input_ids = torch.randint(0, 255, (2, 16), dtype=torch.long)
                gradients = {}
                for mode in (ExecutionMode.PARALLEL, ExecutionMode.SEQUENTIAL):
                    model.zero_grad(set_to_none=True)
                    model(input_ids, execution_mode=mode).logits.square().mean().backward()
                    gradients[mode] = {
                        name: None if parameter.grad is None else parameter.grad.detach().clone()
                        for name, parameter in model.named_parameters()
                    }
                for name, sequential_gradient in gradients[ExecutionMode.SEQUENTIAL].items():
                    parallel_gradient = gradients[ExecutionMode.PARALLEL][name]
                    self.assertEqual(sequential_gradient is None, parallel_gradient is None, name)
                    if sequential_gradient is None or parallel_gradient is None:
                        continue
                    difference = float((parallel_gradient - sequential_gradient).abs().max())
                    scale = float(sequential_gradient.abs().max())
                    self.assertLess(difference, max(1e-5, scale * 1e-3), name)
