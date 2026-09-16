from __future__ import annotations

import tempfile
import os
import time
import unittest
from pathlib import Path

import torch

from koemi.configuration.settings import ModelSettings
from koemi.model.cache import DiskMappingCache
from koemi.model.network import KoemiModel
from koemi.runtime.bulk_prefix_cache import BulkPrefixCache
from koemi.training.generation import evaluate_prompt_state


class PrefixLedgerTests(unittest.TestCase):
    def test_snapshot_tensor_bytes_do_not_grow_with_prefix_length(self) -> None:
        torch.manual_seed(5)
        model = KoemiModel(
            ModelSettings(
                embedding_size=16,
                memory_features=4,
                local_memory_size=3,
                salience_memory_size=3,
                salience_threshold=0.0,
            )
        ).eval()
        short = torch.randint(0, 255, (1, 16), dtype=torch.long)
        long = torch.randint(0, 255, (1, 64), dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            cache = DiskMappingCache(Path(directory), namespace="ledger", capacity=16)
            evaluate_prompt_state(model, short, None, cache)
            short_entry = cache.get_longest_prefix(short, torch.device("cpu"))
            evaluate_prompt_state(model, long, None, cache)
            long_entry = cache.get_longest_prefix(long, torch.device("cpu"))
            self.assertIsNotNone(short_entry)
            self.assertIsNotNone(long_entry)
            short_bytes = cache.estimate_prefix_bytes(short_entry[1])
            long_bytes = cache.estimate_prefix_bytes(long_entry[1])
        self.assertEqual(short_bytes, long_bytes)

    def test_an_extended_prompt_only_processes_the_uncached_suffix(self) -> None:
        torch.manual_seed(7)
        model = KoemiModel(
            ModelSettings(embedding_size=16, memory_features=4, local_memory_size=3, scan_chunk=2)
        ).eval()
        prefix = torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long)
        extension = torch.tensor([[65, 66, 67, 68, 69, 70, 71]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            cache = DiskMappingCache(Path(directory), namespace="ledger", capacity=16)
            first = evaluate_prompt_state(model, prefix, None, cache)
            second = evaluate_prompt_state(model, extension, None, cache)
            exact = evaluate_prompt_state(model, extension, None, cache)
            oracle = model(extension)

        self.assertEqual(5, first.processed_tokens)
        self.assertEqual(2, second.processed_tokens)
        self.assertEqual(5, second.reused_prefix_tokens)
        self.assertEqual(0, exact.processed_tokens)
        self.assertEqual(7, exact.reused_prefix_tokens)
        self.assertTrue(torch.allclose(second.last_logits, oracle.logits[:, -1], atol=1e-6))
        self.assertTrue(torch.allclose(second.state.memory_basis, oracle.state.memory_basis, atol=1e-5))

    def test_a_corrupt_prefix_entry_is_a_miss(self) -> None:
        input_ids = torch.tensor([[65, 66]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory:
            cache = DiskMappingCache(Path(directory), namespace="ledger")
            cache.prefix_path_for(input_ids).write_bytes(b"not a checkpoint")
            self.assertIsNone(cache.get_longest_prefix(input_ids, torch.device("cpu")))

    def test_an_oversized_prefix_entry_is_not_loaded(self) -> None:
        input_ids = torch.tensor([[65, 66]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory:
            cache = DiskMappingCache(Path(directory), namespace="ledger", max_entry_bytes=8)
            cache.prefix_path_for(input_ids).write_bytes(b"larger than eight bytes")
            self.assertIsNone(cache.get_longest_prefix(input_ids, torch.device("cpu")))

    def test_prefix_entries_expire_and_stay_inside_their_namespace(self) -> None:
        model = KoemiModel(ModelSettings(embedding_size=16, memory_features=4, local_memory_size=3)).eval()
        input_ids = torch.tensor([[65, 66, 67]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            first = DiskMappingCache(Path(directory), namespace="first", ttl_seconds=1.0)
            second = DiskMappingCache(Path(directory), namespace="second", ttl_seconds=1.0)
            evaluate_prompt_state(model, input_ids, None, first)
            first_path = first.prefix_path_for(input_ids)
            self.assertTrue(first_path.exists())
            self.assertIsNone(second.get_longest_prefix(input_ids, torch.device("cpu")))
            old_time = time.time() - 10.0
            os.utime(first_path, (old_time, old_time))
            self.assertIsNone(first.get_longest_prefix(input_ids, torch.device("cpu")))
            self.assertFalse(first_path.exists())
            self.assertEqual(1, first.statistics().expirations)

    def test_bulk_prefix_cache_reuses_blocks_and_only_processes_the_new_suffix(self) -> None:
        torch.manual_seed(11)
        model = KoemiModel(
            ModelSettings(embedding_size=16, memory_features=4, local_memory_size=3, scan_chunk=2)
        ).eval()
        prefix = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
        extension = torch.tensor([[65, 66, 67, 68, 69, 70]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            cache = BulkPrefixCache(
                Path(directory),
                namespace="checkpoint-v1",
                block_size=2,
                ram_capacity=16,
                disk_capacity=16,
            )
            first = evaluate_prompt_state(model, prefix, None, None, cache)
            second = evaluate_prompt_state(model, extension, None, None, cache)
            exact = evaluate_prompt_state(model, extension, None, None, cache)
            oracle = model(extension)
            statistics = cache.statistics()

        self.assertEqual(4, first.processed_tokens)
        self.assertEqual(2, second.processed_tokens)
        self.assertEqual(4, second.reused_prefix_tokens)
        self.assertEqual(0, exact.processed_tokens)
        self.assertEqual(6, exact.reused_prefix_tokens)
        self.assertTrue(torch.allclose(second.last_logits, oracle.logits[:, -1], atol=1e-6))
        self.assertTrue(torch.allclose(second.state.memory_basis, oracle.state.memory_basis, atol=1e-5))
        self.assertEqual(2, statistics.hits)
        self.assertEqual(3, statistics.misses)

    def test_bulk_prefix_cache_does_not_reuse_a_block_after_history_changes(self) -> None:
        model = KoemiModel(
            ModelSettings(embedding_size=16, memory_features=4, local_memory_size=3, scan_chunk=2)
        ).eval()
        original = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
        changed = torch.tensor([[90, 66, 67, 68]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            cache = BulkPrefixCache(
                Path(directory),
                namespace="checkpoint-v1",
                block_size=2,
                ram_capacity=16,
                disk_capacity=16,
            )
            evaluate_prompt_state(model, original, None, None, cache)
            evaluation = evaluate_prompt_state(model, changed, None, None, cache)

        self.assertEqual(4, evaluation.processed_tokens)
        self.assertEqual(0, evaluation.reused_prefix_tokens)

    def test_bulk_prefix_cache_restores_a_disk_block_in_a_new_instance(self) -> None:
        model = KoemiModel(
            ModelSettings(embedding_size=16, memory_features=4, local_memory_size=3, scan_chunk=2)
        ).eval()
        input_ids = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            writer = BulkPrefixCache(
                Path(directory),
                namespace="checkpoint-v1",
                block_size=2,
                ram_capacity=16,
                disk_capacity=16,
            )
            evaluate_prompt_state(model, input_ids, None, None, writer)
            reader = BulkPrefixCache(
                Path(directory),
                namespace="checkpoint-v1",
                block_size=2,
                ram_capacity=16,
                disk_capacity=16,
            )
            evaluation = evaluate_prompt_state(model, input_ids, None, None, reader)
            statistics = reader.statistics()

        self.assertEqual(0, evaluation.processed_tokens)
        self.assertEqual(4, evaluation.reused_prefix_tokens)
        self.assertEqual(1, statistics.disk_hits)


if __name__ == "__main__":
    unittest.main()
