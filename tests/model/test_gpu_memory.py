from __future__ import annotations

import inspect
import unittest

import torch

from koemi.model import gpu_memory
from koemi.model.gpu_memory import StateBufferPool


class StateBufferPoolTests(unittest.TestCase):
    def test_same_layout_reuses_the_preallocated_tensor(self) -> None:
        pool = StateBufferPool(device="cpu")

        first = pool.acquire("memory_basis", (2, 16, 4), dtype=torch.float32)
        second = pool.acquire("memory_basis", (2, 16, 4), dtype=torch.float32)

        self.assertIs(first, second)
        self.assertEqual(torch.device("cpu"), first.device)
        statistics = pool.statistics()
        self.assertEqual(1, statistics.allocations)
        self.assertEqual(1, statistics.reuses)
        self.assertEqual(1, statistics.active_buffers)

    def test_reset_reuses_herm_local_and_salience_storage(self) -> None:
        pool = StateBufferPool(device="cpu")
        memory_basis = pool.acquire("memory_basis", (2, 16, 4))
        local_keys = pool.acquire("local_keys", (2, 5, 16))
        local_valid = pool.acquire("local_valid", (2, 5), dtype=torch.bool)
        salient_valid = pool.acquire(
            "salient_valid",
            (2, 3),
            dtype=torch.bool,
            reset_value=False,
        )

        memory_basis.fill_(3.0)
        local_keys.fill_(4.0)
        local_valid.fill_(True)
        salient_valid.fill_(True)
        pool.reset()

        self.assertTrue(torch.equal(memory_basis, torch.zeros_like(memory_basis)))
        self.assertTrue(torch.equal(local_keys, torch.zeros_like(local_keys)))
        self.assertFalse(bool(local_valid.any()))
        self.assertFalse(bool(salient_valid.any()))
        self.assertIs(memory_basis, pool.acquire("memory_basis", (2, 16, 4)))
        self.assertIs(local_valid, pool.acquire("local_valid", (2, 5), dtype=torch.bool))
        self.assertEqual(4, pool.statistics().resets)

    def test_reset_value_is_preserved_for_integer_state(self) -> None:
        pool = StateBufferPool(device="cpu")
        token_ids = pool.acquire(
            "last_token_ids",
            (2,),
            dtype=torch.long,
            reset_value=-1,
        )
        token_ids.fill_(17)

        pool.reset("last_token_ids")

        self.assertTrue(torch.equal(token_ids, torch.full_like(token_ids, -1)))
        self.assertEqual(-1, pool.specification("last_token_ids").reset_value)

    def test_shape_mismatch_requires_explicit_resize(self) -> None:
        pool = StateBufferPool(device="cpu")
        original = pool.acquire("local_keys", (2, 5, 16))

        with self.assertRaisesRegex(ValueError, "shape"):
            pool.acquire("local_keys", (2, 7, 16))

        resized = pool.resize("local_keys", (2, 7, 16))

        self.assertIsNot(original, resized)
        self.assertEqual((2, 7, 16), tuple(resized.shape))
        statistics = pool.statistics()
        self.assertEqual(2, statistics.allocations)
        self.assertEqual(1, statistics.resizes)

    def test_dtype_and_device_mismatch_are_rejected_without_copy(self) -> None:
        pool = StateBufferPool(device="cpu")
        reference = pool.acquire("working_state", (2, 16), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "dtype"):
            pool.acquire("working_state", (2, 16), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "device"):
            pool.acquire("working_state", (2, 16), device="cuda")
        with self.assertRaisesRegex(ValueError, "device"):
            pool.acquire_like("working_state", torch.empty_like(reference, device="meta"))

        self.assertIs(reference, pool.acquire_like("working_state", reference))
        self.assertEqual(1, pool.statistics().allocations)

    def test_timing_is_collected_and_exposed_only_when_requested(self) -> None:
        pool = StateBufferPool(device="cpu", measure_timings=True)
        pool.acquire("working_state", (2, 16))
        pool.reset("working_state")
        pool.resize("working_state", (3, 16))

        hidden = pool.statistics()
        visible = pool.statistics(include_timing=True)

        self.assertTrue(hidden.timing_enabled)
        self.assertIsNone(hidden.allocation_seconds)
        self.assertIsNotNone(visible.allocation_seconds)
        self.assertIsNotNone(visible.reset_seconds)
        self.assertIsNotNone(visible.resize_seconds)

    def test_layer_has_no_per_token_host_transfer_or_scalar_sync(self) -> None:
        source = inspect.getsource(gpu_memory)

        for forbidden_operation in (".item(", ".tolist(", ".cpu(", ".to(", "synchronize("):
            self.assertNotIn(forbidden_operation, source)

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA unavailable: this environment uses a CPU-only PyTorch wheel",
    )
    def test_cuda_stream_and_device_contract(self) -> None:
        device = torch.device("cuda")
        stream = torch.cuda.Stream(device=device)
        pool = StateBufferPool(device=device)

        with torch.cuda.stream(stream):
            first = pool.acquire(
                "memory_basis",
                (2, 16, 4),
                dtype=torch.float16,
                stream=stream,
            )
            second = pool.acquire_like("memory_basis", first, stream=stream)
            first.fill_(1.0)
            pool.reset("memory_basis", stream=stream)
        stream.synchronize()

        self.assertIs(first, second)
        self.assertEqual(device.type, first.device.type)
        self.assertTrue(torch.equal(first, torch.zeros_like(first)))
        self.assertEqual(1, pool.statistics().allocations)
        self.assertEqual(1, pool.statistics().reuses)


if __name__ == "__main__":
    unittest.main()
