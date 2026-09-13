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


if __name__ == "__main__":
    unittest.main()
