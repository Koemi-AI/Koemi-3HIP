from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from koemi.runtime import bulk_blocks
from koemi.runtime.bulk_blocks import BulkBlockStore


class FakeClock:
    def __init__(self) -> None:
        self.current_time = 0.0

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class BulkBlockStoreTests(unittest.TestCase):
    def test_exact_hit_returns_a_candidate_that_requires_validation(self) -> None:
        store = BulkBlockStore(block_size=2, ram_capacity=4)
        stored = store.put("model", 0, [10, 11], b"state")

        self.assertIsNotNone(stored)
        candidate = store.get("model", 0, [10, 11])
        self.assertIsNotNone(candidate)
        self.assertFalse(hasattr(candidate, "payload"))
        validated = store.validate_candidate(candidate, "model", 0, [10, 11])

        self.assertEqual(b"state", validated.payload)
        statistics = store.statistics()
        self.assertEqual(1, statistics.hits)
        self.assertEqual(0, statistics.misses)
        self.assertEqual(1, statistics.ram_hits)

    def test_sequence_miss_namespace_miss_and_offset_miss_are_exact(self) -> None:
        store = BulkBlockStore(block_size=2, ram_capacity=4)
        store.put("model-a", 4, [20, 21], b"state")

        self.assertIsNone(store.get("model-a", 4, [20, 22]))
        self.assertIsNone(store.get("model-b", 4, [20, 21]))
        self.assertIsNone(store.get("model-a", 5, [20, 21]))
        statistics = store.statistics()

        self.assertEqual(0, statistics.hits)
        self.assertEqual(3, statistics.misses)

    def test_digest_collision_still_requires_the_full_sequence(self) -> None:
        constant_digest = "a" * 64

        def constant_chain(namespace: str, offset: int, sequence: tuple[int, ...]) -> tuple[str, ...]:
            return tuple(constant_digest for _ in sequence)

        with patch.object(bulk_blocks, "_build_digest_chain", side_effect=constant_chain):
            store = BulkBlockStore(block_size=2, ram_capacity=4)
            first = store.put("model", 0, [1, 2], b"first")
            second = store.put("model", 0, [1, 3], b"second")

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertEqual(first.key, second.key)
            first_candidate = store.get("model", 0, [1, 2])
            second_candidate = store.get("model", 0, [1, 3])
            self.assertEqual(
                b"first",
                store.validate_candidate(first_candidate, "model", 0, [1, 2]).payload,
            )
            self.assertEqual(
                b"second",
                store.validate_candidate(second_candidate, "model", 0, [1, 3]).payload,
            )

    def test_ttl_removes_an_expired_ram_block_and_records_the_expiration(self) -> None:
        clock = FakeClock()
        store = BulkBlockStore(block_size=1, ttl_seconds=5.0, clock=clock)
        store.put("model", 0, [7], b"state")
        clock.advance(5.0)

        self.assertIsNone(store.get("model", 0, [7]))
        statistics = store.statistics()
        self.assertEqual(1, statistics.expirations)
        self.assertEqual(1, statistics.misses)
        self.assertEqual(0, statistics.ram_entries)

    def test_entry_limit_rejects_without_allocating_a_ram_block(self) -> None:
        store = BulkBlockStore(block_size=1, max_entry_bytes=1)

        self.assertIsNone(store.put("model", 0, [7], b"state"))
        statistics = store.statistics()
        self.assertEqual(1, statistics.rejected_entries)
        self.assertEqual(0, statistics.ram_entries)
        self.assertEqual(0, statistics.ram_bytes)

    def test_ram_eviction_is_lru_and_deterministic(self) -> None:
        store = BulkBlockStore(block_size=1, ram_capacity=2)
        store.put("model", 0, [10], b"zero")
        store.put("model", 1, [11], b"one")
        self.assertIsNotNone(store.get("model", 0, [10]))
        store.put("model", 2, [12], b"two")

        self.assertIsNone(store.get("model", 1, [11]))
        self.assertIsNotNone(store.get("model", 0, [10]))
        self.assertIsNotNone(store.get("model", 2, [12]))
        statistics = store.statistics()
        self.assertEqual(1, statistics.ram_evictions)
        self.assertEqual(1, statistics.evictions)

    def test_json_data_and_explicit_encoder_are_supported_but_raw_text_is_not(self) -> None:
        store = BulkBlockStore(block_size=1, ram_capacity=4)

        with self.assertRaisesRegex(TypeError, "explicit payload encoder"):
            store.put("model", 0, [1], "plain text")

        store.put("model", 1, [2], {"valid": True, "values": [1, 2]})
        json_candidate = store.get("model", 1, [2])
        self.assertEqual(
            {"valid": True, "values": [1, 2]},
            store.validate_candidate(json_candidate, "model", 1, [2]).payload,
        )

        store.put(
            "model",
            2,
            [3],
            "encoded text",
            payload_encoder=lambda value: {"text": value},
        )
        encoded_candidate = store.get("model", 2, [3])
        self.assertEqual(
            {"text": "encoded text"},
            store.validate_candidate(encoded_candidate, "model", 2, [3]).payload,
        )

        with self.assertRaises(TypeError):
            store.put("model", 3, [4], object())

    def test_candidate_validation_rejects_a_different_sequence(self) -> None:
        store = BulkBlockStore(block_size=2)
        store.put("model", 0, [1, 2], b"state")
        candidate = store.get("model", 0, [1, 2])

        with self.assertRaisesRegex(ValueError, "not exact"):
            store.validate_candidate(candidate, "model", 0, [1, 3])

    def test_ssd_hit_promotes_to_ram_and_metrics_count_disk_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = BulkBlockStore(
                block_size=1,
                ram_capacity=1,
                disk_directory=directory,
                disk_capacity=4,
            )
            first = store.put("model", 0, [1], b"first")
            first_path = store.path_for(first)
            first_size = first_path.stat().st_size
            second = store.put("model", 1, [2], b"second")
            second_path = store.path_for(second)
            second_size = second_path.stat().st_size

            candidate = store.get("model", 0, [1])
            validated = store.validate_candidate(candidate, "model", 0, [1])
            statistics = store.statistics()

            self.assertEqual(b"first", validated.payload)
            self.assertEqual("disk", candidate.source)
            self.assertEqual(1, statistics.disk_hits)
            self.assertEqual(0, statistics.ram_hits)
            self.assertEqual(first_size, statistics.bytes_read)
            self.assertEqual(first_size + second_size, statistics.bytes_written)
            self.assertEqual(2, statistics.disk_entries)
            self.assertEqual(first_size + second_size, statistics.disk_bytes)

    def test_ssd_record_survives_a_new_store_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = BulkBlockStore(block_size=1, disk_directory=directory)
            writer.put("model", 0, [1], b"persisted")

            reader = BulkBlockStore(block_size=1, disk_directory=directory)
            candidate = reader.get("model", 0, [1])

            self.assertEqual("disk", candidate.source)
            self.assertEqual(
                b"persisted",
                reader.validate_candidate(candidate, "model", 0, [1]).payload,
            )
            self.assertEqual(1, reader.statistics().disk_hits)

    def test_ssd_capacity_uses_fifo_and_keeps_only_allowed_files(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            store = BulkBlockStore(
                block_size=1,
                ram_capacity=1,
                disk_directory=directory,
                disk_capacity=2,
                clock=clock,
            )
            first = store.put("model", 0, [1], b"first")
            second = store.put("model", 1, [2], b"second")
            third = store.put("model", 2, [3], b"third")

            first_path = store.path_for(first)
            second_path = store.path_for(second)
            third_path = store.path_for(third)
            statistics = store.statistics()

            self.assertFalse(first_path.exists())
            self.assertTrue(second_path.exists())
            self.assertTrue(third_path.exists())
            self.assertEqual(1, statistics.disk_evictions)
            self.assertEqual(2, statistics.disk_entries)
            self.assertTrue(all(path.parent == Path(directory).resolve() for path in Path(directory).glob("*")))

    def test_atomic_write_leaves_no_temporary_file_and_preserves_old_entry_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = BulkBlockStore(block_size=1, disk_directory=directory)
            candidate = store.put("model", 0, [1], b"old")
            path = store.path_for(candidate)
            original_bytes = path.read_bytes()

            with patch.object(bulk_blocks.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    store.put("model", 0, [1], b"new")

            self.assertEqual(original_bytes, path.read_bytes())
            self.assertEqual([], list(Path(directory).glob("*.tmp")))

    def test_corrupt_and_temporary_files_never_become_hits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = BulkBlockStore(block_size=1, ram_capacity=1, disk_directory=directory)
            candidate = store.put("model", 0, [1], b"state")
            path = store.path_for(candidate)
            store.put("model", 1, [2], b"other")
            record = json.loads(path.read_bytes())
            record["payload"] = "dGFtcGVyZWQ="
            path.write_bytes(json.dumps(record).encode("utf-8"))
            temporary_path = path.with_suffix(".tmp")
            temporary_path.write_bytes(b'{"not": "an entry"}')

            self.assertIsNone(store.get("model", 0, [1]))
            statistics = store.statistics()

            self.assertTrue(temporary_path.exists())
            self.assertFalse(path.exists())
            self.assertEqual(0, statistics.hits)
            self.assertEqual(1, statistics.misses)
            self.assertEqual(1, statistics.corruptions)

    def test_disk_record_is_expired_before_a_disk_hit(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            store = BulkBlockStore(
                block_size=1,
                ram_capacity=1,
                disk_directory=directory,
                ttl_seconds=5.0,
                clock=clock,
            )
            candidate = store.put("model", 0, [1], b"state")
            store.put("model", 1, [2], b"other")
            clock.advance(5.0)

            self.assertIsNone(store.get("model", 0, [1]))
            self.assertFalse(store.path_for(candidate).exists())
            statistics = store.statistics()
            self.assertEqual(3, statistics.expirations)
            self.assertEqual(1, statistics.misses)

    def test_path_for_namespace_is_contained_and_payload_is_explicit_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = BulkBlockStore(block_size=1, disk_directory=directory)
            candidate = store.put("..\\..\\outside", 0, [1], b"binary")
            path = store.path_for(candidate)
            record = json.loads(path.read_bytes())

            self.assertEqual(Path(directory).resolve(), path.resolve().parent)
            self.assertEqual("bytes", record["payload_encoding"])
            self.assertNotIn("binary", path.read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                store.path_for_values("model", 0, [1], sequence_digest="../outside")

    def test_statistics_are_serializable(self) -> None:
        store = BulkBlockStore(block_size=1)
        store.put("model", 0, [1], {"value": 1})

        statistics = store.statistics()
        serialized = json.dumps(statistics.to_dict(), sort_keys=True)

        self.assertIn("ram_bytes", serialized)


if __name__ == "__main__":
    unittest.main()
