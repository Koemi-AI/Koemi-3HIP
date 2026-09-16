from __future__ import annotations

import threading
import time
import unittest

import torch

from koemi.runtime.bulk_executor import (
    BulkBackpressureError,
    BulkClosedError,
    BulkExecutor,
    BulkResult,
    BulkValidationError,
)


class BulkExecutorTests(unittest.TestCase):
    def test_synchronous_execution_returns_isolated_result_and_metrics(self) -> None:
        with BulkExecutor(lambda value: {"value": value * 2}) as executor:
            future = executor.submit(4)
            result = future.result()

            self.assertIsInstance(result, BulkResult)
            self.assertEqual({"value": 8}, result.value)
            self.assertEqual(torch.device("cpu"), result.device)
            self.assertIsNone(result.completion_event)
            self.assertEqual({"value": 8}, result.wait())

        metrics = executor.metrics
        self.assertEqual(1, metrics.submitted)
        self.assertEqual(1, metrics.completed)
        self.assertEqual(0, metrics.failed)
        self.assertEqual(0, metrics.cancelled)
        self.assertEqual(0, metrics.in_flight)
        self.assertGreaterEqual(metrics.prepare_seconds, 0.0)
        self.assertGreaterEqual(metrics.total_seconds, metrics.snapshot_seconds)

    def test_thread_pool_never_exceeds_the_configured_worker_bound(self) -> None:
        state_lock = threading.Lock()
        first_workers_ready = threading.Barrier(2)
        active_workers = 0
        maximum_active_workers = 0
        prepared_by_threads: set[int] = set()
        prepared_count = 0

        def prepare(value: int) -> int:
            nonlocal active_workers, maximum_active_workers, prepared_count
            with state_lock:
                prepared_count += 1
                current_count = prepared_count
                active_workers += 1
                maximum_active_workers = max(maximum_active_workers, active_workers)
                prepared_by_threads.add(threading.get_ident())
            if current_count <= 2:
                first_workers_ready.wait(timeout=2.0)
            with state_lock:
                active_workers -= 1
            return value * 2

        with BulkExecutor(
            prepare,
            max_workers=2,
            max_in_flight=2,
            preserve_order=True,
        ) as executor:
            futures = executor.submit_many(range(4))
            values = executor.collect(futures)

        self.assertEqual((0, 2, 4, 6), values)
        self.assertEqual(2, maximum_active_workers)
        self.assertEqual(2, len(prepared_by_threads))
        self.assertEqual(4, prepared_count)
        self.assertEqual(4, executor.metrics.completed)

    def test_requested_submission_order_is_restored_after_out_of_order_completion(self) -> None:
        def prepare(value: int) -> int:
            time.sleep((4 - value) * 0.01)
            return value

        with BulkExecutor(
            prepare,
            max_workers=3,
            max_in_flight=3,
            preserve_order=True,
        ) as executor:
            futures = executor.submit_many(range(4))
            self.assertEqual((0, 1, 2, 3), executor.collect(futures))

    def test_backpressure_blocks_until_a_slot_is_released(self) -> None:
        started = threading.Event()
        release = threading.Event()
        producer_started = threading.Event()
        second_submitted = threading.Event()
        submission_errors: list[BaseException] = []

        def prepare(value: int) -> int:
            started.set()
            release.wait(timeout=2.0)
            return value

        executor = BulkExecutor(prepare, max_workers=1, max_in_flight=1)
        first_future = executor.submit(1)
        self.assertTrue(started.wait(timeout=2.0))

        def submit_second() -> None:
            producer_started.set()
            try:
                executor.submit(2, timeout=2.0)
            except BaseException as error:
                submission_errors.append(error)
            else:
                second_submitted.set()

        producer = threading.Thread(target=submit_second)
        producer.start()
        self.assertTrue(producer_started.wait(timeout=2.0))
        self.assertFalse(second_submitted.wait(timeout=0.05))
        self.assertEqual(1, executor.metrics.in_flight)

        release.set()
        self.assertEqual(1, first_future.result().value)
        producer.join(timeout=2.0)
        self.assertFalse(producer.is_alive())
        self.assertEqual([], submission_errors)
        self.assertTrue(second_submitted.is_set())
        executor.close()
        self.assertEqual(0, executor.metrics.in_flight)
        self.assertGreater(executor.metrics.backpressure_seconds, 0.0)

    def test_backpressure_timeout_is_explicit(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def prepare(value: int) -> int:
            started.set()
            release.wait(timeout=2.0)
            return value

        executor = BulkExecutor(prepare, max_workers=1, max_in_flight=1)
        first_future = executor.submit(1)
        self.assertTrue(started.wait(timeout=2.0))
        with self.assertRaises(BulkBackpressureError):
            executor.submit(2, timeout=0.01)
        release.set()
        first_future.result()
        executor.close()

    def test_prepare_exception_reaches_the_future_and_metrics(self) -> None:
        def prepare(value: int) -> int:
            raise ValueError(f"bad item {value}")

        with BulkExecutor(prepare, max_workers=1) as executor:
            future = executor.submit(7)
            with self.assertRaisesRegex(ValueError, "bad item 7"):
                future.result()
            self.assertEqual(1, executor.metrics.failed)
            self.assertEqual(0, executor.metrics.in_flight)

    def test_mutable_inputs_are_not_shared_between_requests(self) -> None:
        source = {"values": [1], "tensor": torch.tensor([2])}
        with BulkExecutor() as executor:
            first = executor.submit(source).result().value
            second = executor.submit(source).result().value

            first["values"].append(3)
            first["tensor"].fill_(9)

            self.assertEqual([1], second["values"])
            self.assertTrue(torch.equal(torch.tensor([2]), second["tensor"]))
            self.assertEqual([1], source["values"])
            self.assertTrue(torch.equal(torch.tensor([2]), source["tensor"]))

    def test_shape_and_device_contract_is_checked_after_transfer(self) -> None:
        def wrong_transfer(value: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
            return {"input_ids": value["input_ids"].to("meta")}

        with BulkExecutor(
            device="cpu",
            expected_shapes={"input_ids": (2, None)},
            transfer=wrong_transfer,
        ) as executor:
            wrong_shape = executor.submit({"input_ids": torch.zeros(3, 4)})
            with self.assertRaisesRegex(BulkValidationError, "input_ids"):
                wrong_shape.result()

            wrong_device = executor.submit({"input_ids": torch.zeros(2, 4)})
            with self.assertRaisesRegex(BulkValidationError, "expected cpu"):
                wrong_device.result()

        self.assertEqual(2, executor.metrics.failed)
        self.assertEqual(0, executor.metrics.in_flight)

    def test_cancel_pending_future_is_counted_and_close_leaves_no_pending_work(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def prepare(value: int) -> int:
            if value == 1:
                started.set()
                release.wait(timeout=2.0)
            return value

        executor = BulkExecutor(prepare, max_workers=1, max_in_flight=2)
        running = executor.submit(1)
        self.assertTrue(started.wait(timeout=2.0))
        queued = executor.submit(2)
        self.assertTrue(queued.cancel())
        self.assertTrue(queued.cancelled())
        release.set()
        self.assertEqual(1, running.result().value)
        executor.close()

        self.assertTrue(executor.closed)
        self.assertTrue(running.done())
        self.assertTrue(queued.done())
        self.assertEqual(1, executor.metrics.completed)
        self.assertEqual(1, executor.metrics.cancelled)
        self.assertEqual(0, executor.metrics.in_flight)

    def test_close_can_cancel_queued_work_and_reject_new_submissions(self) -> None:
        started = threading.Event()
        release = threading.Event()
        close_started = threading.Event()
        close_finished = threading.Event()

        def prepare(value: int) -> int:
            if value == 1:
                started.set()
                release.wait(timeout=2.0)
            return value

        executor = BulkExecutor(prepare, max_workers=1, max_in_flight=2)
        running = executor.submit(1)
        self.assertTrue(started.wait(timeout=2.0))
        queued = executor.submit(2)

        def close_executor() -> None:
            close_started.set()
            executor.close(cancel_pending=True)
            close_finished.set()

        closer = threading.Thread(target=close_executor)
        closer.start()
        self.assertTrue(close_started.wait(timeout=2.0))
        with self.assertRaises(BulkClosedError):
            executor.submit(3)
        self.assertFalse(close_finished.wait(timeout=0.05))
        release.set()
        closer.join(timeout=2.0)

        self.assertFalse(closer.is_alive())
        self.assertTrue(close_finished.is_set())
        self.assertEqual(1, running.result().value)
        self.assertTrue(queued.cancelled())
        self.assertEqual(1, executor.metrics.completed)
        self.assertEqual(1, executor.metrics.cancelled)
        self.assertEqual(0, executor.metrics.in_flight)

    def test_closed_executor_rejects_submission(self) -> None:
        executor = BulkExecutor()
        executor.close()
        with self.assertRaises(BulkClosedError):
            executor.submit(1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: bulk executor CUDA contract")
class BulkExecutorCudaTests(unittest.TestCase):
    def test_cuda_enqueue_records_dedicated_stream_event_and_supports_stream_ordering(self) -> None:
        device = torch.device("cuda")
        with BulkExecutor(
            device=device,
            max_workers=1,
            max_in_flight=1,
            expected_shapes={"input_ids": (2, 4)},
        ) as executor:
            self.assertIsNotNone(executor.cuda_stream)
            self.assertNotEqual(executor.cuda_stream, torch.cuda.current_stream(device))
            future = executor.submit({"input_ids": torch.ones(2, 4, dtype=torch.float32)})
            result = future.result()
            self.assertIsNotNone(result.completion_event)
            self.assertEqual(device, result.device)
            self.assertEqual(device, result.value["input_ids"].device)
            result.wait_for_stream(torch.cuda.current_stream(device))
            self.assertTrue(
                torch.equal(result.wait()["input_ids"], torch.ones(2, 4, device=device))
            )

        self.assertEqual(1, executor.metrics.completed)
        self.assertEqual(0, executor.metrics.in_flight)


if __name__ == "__main__":
    unittest.main()
