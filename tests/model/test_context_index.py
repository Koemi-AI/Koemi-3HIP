from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import unittest
from unittest.mock import patch

import torch

from koemi.model import context_index
from koemi.model.context_index import ContextCandidate, ContextIndex


class FakeClock:
    def __init__(self) -> None:
        self.current_time = 0.0

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class ContextIndexTests(unittest.TestCase):
    def test_exact_hit_and_miss_keep_the_payload_namespace_safe(self) -> None:
        index = ContextIndex(capacity=4, ttl_seconds=60.0)
        payload = {"state": [1, 2]}
        index.put("model-a", [10, 20], payload)

        hit = index.get_longest_prefix("model-a", [10, 20])
        miss = index.get_longest_prefix("model-a", [10, 21])
        foreign_namespace = index.get_longest_prefix("model-b", [10, 20])

        self.assertIsNotNone(hit)
        self.assertIs(payload, hit.payload)
        self.assertIsNone(miss)
        self.assertIsNone(foreign_namespace)
        statistics = index.statistics()
        self.assertEqual(1, statistics.hits)
        self.assertEqual(2, statistics.misses)
        self.assertEqual(1, statistics.candidates)

    def test_longest_prefix_wins_over_a_shorter_stored_prefix(self) -> None:
        index = ContextIndex(capacity=4, ttl_seconds=60.0)
        short_payload = {"kind": 1}
        long_payload = {"kind": 2}
        index.put("model", [1, 2], short_payload)
        index.put("model", [1, 2, 3], long_payload)

        candidate = index.get_longest_prefix("model", [1, 2, 3, 4])

        self.assertIsNotNone(candidate)
        self.assertEqual((1, 2, 3), candidate.sequence)
        self.assertIs(long_payload, candidate.payload)

    def test_cpu_tensor_sequences_are_normalized_to_exact_token_ids(self) -> None:
        index = ContextIndex(capacity=2, ttl_seconds=60.0)
        input_ids = torch.tensor([[17, 18, 19]], dtype=torch.long)
        index.put("model", input_ids[:, :2], {"state": [1]})

        candidate = index.get_longest_prefix("model", torch.tensor([[17, 18, 20]], dtype=torch.long))

        self.assertIsNotNone(candidate)
        self.assertEqual((17, 18), candidate.sequence)

    def test_hash_collision_still_requires_exact_sequence_equality(self) -> None:
        constant_digest = "a" * 64
        constant_chain = lambda sequence: tuple(constant_digest for _ in sequence)
        with patch.object(context_index, "_sequence_digest", return_value=constant_digest), patch.object(
            context_index, "_build_digest_chain", side_effect=constant_chain
        ):
            index = ContextIndex(capacity=4, ttl_seconds=60.0)
            index.put("model", [1, 2], 1)
            index.put("model", [1, 3], 2)
            candidate = index.get_longest_prefix("model", [1, 3])

        self.assertIsNotNone(candidate)
        self.assertEqual((1, 3), candidate.sequence)
        self.assertEqual(2, candidate.payload)
        self.assertEqual(2, index.statistics().candidates)

    def test_invalid_serialized_hash_is_rejected_before_indexing(self) -> None:
        candidate = ContextIndex(capacity=2, ttl_seconds=60.0).put("model", [4, 5], {"state": [6]})
        record = candidate.to_record()
        record["sequence_digest"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "hash is invalid"):
            ContextCandidate.from_record(record)

    def test_ttl_removes_an_expired_candidate_and_records_the_expiration(self) -> None:
        clock = FakeClock()
        index = ContextIndex(capacity=2, ttl_seconds=5.0, clock=clock)
        index.put("model", [7], {"state": [7]})
        clock.advance(5.0)

        self.assertIsNone(index.get_longest_prefix("model", [7]))
        statistics = index.statistics()
        self.assertEqual(1, statistics.expirations)
        self.assertEqual(1, statistics.misses)
        self.assertEqual(0, statistics.entries)

    def test_namespace_clear_does_not_remove_another_namespace(self) -> None:
        index = ContextIndex(capacity=4, ttl_seconds=60.0)
        index.put("first", [8], {"state": [1]})
        index.put("second", [8], {"state": [2]})

        self.assertEqual(1, index.clear_namespace("first"))
        self.assertIsNone(index.get_longest_prefix("first", [8]))
        remaining = index.get_longest_prefix("second", [8])

        self.assertIsNotNone(remaining)
        self.assertEqual({"state": [2]}, remaining.payload)
        self.assertEqual(1, len(index))

    def test_capacity_evicts_the_least_recently_used_candidate(self) -> None:
        index = ContextIndex(capacity=2, ttl_seconds=60.0)
        index.put("model", [11], 11)
        index.put("model", [12], 12)
        self.assertEqual(11, index.get_longest_prefix("model", [11]).payload)
        index.put("model", [13], 13)

        self.assertIsNone(index.get_longest_prefix("model", [12]))
        self.assertEqual(11, index.get_longest_prefix("model", [11]).payload)
        self.assertEqual(13, index.get_longest_prefix("model", [13]).payload)
        self.assertEqual(1, index.statistics().evictions)

    def test_raw_text_payload_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "raw text"):
            ContextIndex(capacity=2, ttl_seconds=60.0).put("model", [16], "prompt")

    def test_candidate_json_round_trip_uses_explicit_data(self) -> None:
        candidate = ContextIndex(capacity=2, ttl_seconds=60.0).put("model", [14, 15], {"logits": [0.1, 0.2]})
        serialized = candidate.to_json()
        restored = ContextCandidate.from_json(serialized)

        self.assertEqual(candidate.namespace, restored.namespace)
        self.assertEqual(candidate.sequence, restored.sequence)
        self.assertEqual(candidate.sequence_digest, restored.sequence_digest)
        self.assertEqual(candidate.payload, restored.payload)
        self.assertNotIn("pickle", serialized.decode("utf-8"))
        self.assertEqual(1, json.loads(serialized)["format_version"])

    def test_concurrent_put_and_lookup_preserve_exact_candidates(self) -> None:
        index = ContextIndex(capacity=64, ttl_seconds=60.0)

        def write_and_read(indexed_value: int) -> tuple[str, int, tuple[int, ...]]:
            namespace = "model-a" if indexed_value % 2 == 0 else "model-b"
            sequence = [indexed_value, indexed_value + 100]
            index.put(namespace, sequence, indexed_value)
            candidate = index.get_longest_prefix(namespace, sequence)
            if candidate is None:
                raise AssertionError("concurrent lookup missed its own exact candidate")
            return candidate.namespace, candidate.payload, candidate.sequence

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(write_and_read, range(32)))

        self.assertEqual(32, len(results))
        for indexed_value, result in enumerate(results):
            expected_namespace = "model-a" if indexed_value % 2 == 0 else "model-b"
            self.assertEqual((expected_namespace, indexed_value, (indexed_value, indexed_value + 100)), result)
        self.assertLessEqual(len(index), 64)


if __name__ == "__main__":
    unittest.main()
