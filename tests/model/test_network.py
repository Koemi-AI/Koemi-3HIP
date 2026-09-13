from __future__ import annotations

import unittest
import tempfile
import os
import time
from pathlib import Path

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.model.cache import DiskMappingCache, WarmTokenCache
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel


class KoemiModelTests(unittest.TestCase):
    def build_model(self, **overrides) -> KoemiModel:
        settings = dict(embedding_size=16, memory_features=4, local_memory_size=4)
        settings.update(overrides)
        return KoemiModel(ModelSettings(**settings))

    def test_produces_finite_logits_without_a_router(self) -> None:
        model = self.build_model()
        input_ids = torch.tensor([[65, 66, 67], [68, 69, 70]], dtype=torch.long)
        output = model(input_ids)
        self.assertEqual((2, 3, model.settings.vocabulary_size), tuple(output.logits.shape))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertFalse(hasattr(model, "router"))
        self.assertEqual(6, output.token_count)
        self.assertEqual((), output.expert_activation_counts)
        self.assertTrue(torch.equal(output.expert_indices, torch.full_like(input_ids, -1)))

    def test_affine_ablation_is_a_memory_free_control(self) -> None:
        model = self.build_model(ablation="affine", expert_count=4)
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        output = model(input_ids)
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertEqual((), output.expert_activation_counts)
        self.assertTrue(torch.equal(output.state.memory_basis, torch.zeros_like(output.state.memory_basis)))
        self.assertTrue(torch.equal(output.surprise_values, torch.zeros_like(output.surprise_values)))

    def test_ablation_parallel_and_sequential_paths_agree(self) -> None:
        model = self.build_model(ablation="no_refine")
        input_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
        parallel = model(input_ids)
        sequential = model(input_ids, execution_mode=ExecutionMode.SEQUENTIAL)
        self.assertTrue(torch.allclose(parallel.logits, sequential.logits, atol=1e-5))
        self.assertTrue(torch.allclose(parallel.state.memory_basis, sequential.state.memory_basis, atol=1e-5))

    def test_backpropagates_through_output_and_refine_memory(self) -> None:
        model = self.build_model(ablation="herm")
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        output = model(input_ids)
        output.logits.sum().backward()
        self.assertIsNotNone(model.token_predictor.weight.grad)
        self.assertIsNotNone(model.memory_refine_gate.weight.grad)
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
        self.assertGreater(float(output.state.refine_basis.detach().abs().sum()), 0.0)

    def test_surprise_is_the_causal_error_of_the_observed_token(self) -> None:
        model = self.build_model()
        with torch.no_grad():
            model.token_predictor.weight.zero_()
            model.token_predictor.bias.zero_()
            model.token_predictor.bias[65] = 12.0
        prior_states = torch.zeros(1, 2, model.settings.embedding_size)
        input_ids = torch.tensor([[65, 66]], dtype=torch.long)
        surprise = model.calculate_surprise(prior_states, input_ids, torch.ones_like(input_ids, dtype=torch.bool))
        self.assertLess(float(surprise[0, 0].detach()), float(surprise[0, 1].detach()))

    def test_refine_memory_retains_more_slowly_than_fast_memory(self) -> None:
        model = self.build_model(ablation="herm")
        projection = model.associative_memory.project(torch.randn(2, model.settings.embedding_size))
        surprise = torch.full((2,), 0.75)
        fast_terms = model.associative_memory.fast_write_terms(projection, surprise)
        refine_terms = model.associative_memory.refine_write_terms(
            projection,
            torch.randn_like(projection.value),
            surprise,
            torch.ones_like(surprise),
        )
        self.assertTrue(torch.all(refine_terms.decay >= fast_terms.decay))
        self.assertTrue(torch.isfinite(refine_terms.basis_increment).all())

    def test_padding_is_not_written_as_a_valid_local_entry(self) -> None:
        torch.manual_seed(0)
        model = self.build_model()
        input_ids = torch.tensor([[65, 66, 67], [68, 69, PAD_TOKEN_ID]], dtype=torch.long)
        output = model(input_ids)
        self.assertFalse(bool(output.state.local_valid[1, -1]))
        self.assertTrue(torch.equal(output.state.local_keys[1, -1], torch.zeros_like(output.state.local_keys[1, -1])))
        self.assertTrue(bool(output.state.local_valid[0, -1]))

    def test_deterministic_moe_assigns_one_expert_to_each_valid_token(self) -> None:
        model = self.build_model(expert_count=4)
        input_ids = torch.tensor([[65, 66, 67, PAD_TOKEN_ID]], dtype=torch.long)
        output = model(input_ids)
        self.assertEqual([2, 1, 0, 0], list(output.expert_activation_counts))
        self.assertEqual([1, 0, 0, -1], output.expert_indices[0].tolist())
        self.assertEqual(3, sum(output.expert_activation_counts))

    def test_warm_cache_preserves_logits_and_reports_reuse(self) -> None:
        torch.manual_seed(0)
        model = self.build_model()
        model.eval()
        input_ids = torch.tensor([[65, 66, 65]], dtype=torch.long)
        cache = WarmTokenCache(capacity=2)
        uncached = model(input_ids)
        first = model(input_ids, warm_cache=cache)
        second = model(input_ids, warm_cache=cache)
        self.assertTrue(torch.allclose(uncached.logits, first.logits, atol=1e-6))
        self.assertTrue(torch.allclose(first.logits, second.logits, atol=1e-6))
        self.assertEqual(0, first.cache_hits)
        self.assertEqual(2, first.cache_misses)
        self.assertEqual(2, second.cache_hits)
        self.assertEqual(0, second.cache_misses)

    def test_warm_cache_is_rejected_during_training(self) -> None:
        model = self.build_model()
        with self.assertRaisesRegex(RuntimeError, "evaluation mode"):
            model(torch.tensor([[65, 66]], dtype=torch.long), warm_cache=WarmTokenCache(2))

    def test_disk_mapping_cache_reuses_an_exact_sequence(self) -> None:
        torch.manual_seed(0)
        model = self.build_model()
        model.eval()
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = DiskMappingCache(Path(temporary_directory), capacity=1, namespace="test")
            first = model(input_ids, mapping_cache=cache)
            second = model(input_ids, mapping_cache=cache)
            statistics = cache.statistics()
        self.assertTrue(torch.allclose(first.logits, second.logits, atol=1e-6))
        self.assertTrue(torch.allclose(first.state.memory_basis, second.state.memory_basis, atol=1e-6))
        self.assertEqual(1, statistics.misses)
        self.assertEqual(1, statistics.hits)

    def test_disk_mapping_cache_eviction_keeps_foreign_files(self) -> None:
        torch.manual_seed(0)
        model = self.build_model()
        model.eval()
        first_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        second_ids = torch.tensor([[68, 69, 70]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            foreign_file = cache_directory / "foreign.pt"
            foreign_file.write_bytes(b"keep")
            cache = DiskMappingCache(cache_directory, capacity=1, namespace="test")
            model(first_ids, mapping_cache=cache)
            model(second_ids, mapping_cache=cache)
            self.assertTrue(foreign_file.exists())
            self.assertEqual(1, cache.statistics().evictions)

    def test_disk_mapping_cache_capacity_is_isolated_by_namespace(self) -> None:
        torch.manual_seed(0)
        model = self.build_model()
        model.eval()
        first_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        second_ids = torch.tensor([[68, 69, 70]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            first_cache = DiskMappingCache(cache_directory, capacity=1, namespace="checkpoint-a")
            second_cache = DiskMappingCache(cache_directory, capacity=1, namespace="checkpoint-b")
            model(first_ids, mapping_cache=first_cache)
            model(second_ids, mapping_cache=second_cache)
            self.assertTrue(first_cache.path_for(first_ids).exists())
            self.assertTrue(second_cache.path_for(second_ids).exists())

    def test_disk_mapping_cache_rejects_an_entry_over_the_size_limit(self) -> None:
        model = self.build_model()
        model.eval()
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache = DiskMappingCache(Path(temporary_directory), namespace="test", max_entry_bytes=1)
            model(input_ids, mapping_cache=cache)
            self.assertFalse(cache.path_for(input_ids).exists())

    def test_disk_mapping_cache_requires_isolation_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "namespace"):
                DiskMappingCache(Path(temporary_directory))

    def test_disk_mapping_cache_expires_and_deletes_only_its_namespace(self) -> None:
        model = self.build_model()
        model.eval()
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            cache = DiskMappingCache(cache_directory, namespace="tenant-a", ttl_seconds=1.0)
            other = DiskMappingCache(cache_directory, namespace="tenant-b", ttl_seconds=1.0)
            model(input_ids, mapping_cache=cache)
            model(input_ids, mapping_cache=other)
            old_time = time.time() - 10.0
            os.utime(cache.path_for(input_ids), (old_time, old_time))
            self.assertIsNone(cache.get(input_ids, torch.device("cpu")))
            self.assertEqual(1, cache.statistics().expirations)
            self.assertEqual(0, cache.clear())
            self.assertTrue(other.path_for(input_ids).exists())
            self.assertTrue(other.delete(input_ids))
            self.assertEqual(1, other.statistics().deletions)
