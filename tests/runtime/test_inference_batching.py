from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
import threading
import unittest

import torch

from koemi.configuration.settings import PAD_TOKEN_ID
from koemi.model.state import KoemiState
from koemi.runtime.inference_batching import (
    BatchContract,
    InferenceBatchScheduler,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.lock = threading.Lock()

    def __call__(self) -> float:
        with self.lock:
            return self.value

    def advance(self, seconds: float) -> None:
        with self.lock:
            self.value += seconds


def make_contract(namespace: str = "tenant-a", model_id: str = "herm-test") -> BatchContract:
    return BatchContract(model_id, torch.device("cpu"), torch.float32, namespace)


def make_state(value: float) -> KoemiState:
    state = KoemiState.create(1, embedding_size=4, memory_features=2, device=torch.device("cpu"))
    state.working_state.fill_(value)
    return state


class InferenceBatchSchedulerTests(unittest.TestCase):
    def test_poll_on_an_empty_queue_returns_no_batches(self) -> None:
        scheduler = InferenceBatchScheduler(max_wait_seconds=1.0)

        self.assertEqual((), scheduler.poll(now=0.0))
        self.assertEqual(0, scheduler.metrics(now=0.0).pending_requests)

    def test_wait_window_groups_compatible_requests_and_pads_without_losing_ids(self) -> None:
        scheduler = InferenceBatchScheduler(
            max_wait_seconds=1.0,
            max_batch_items=4,
            max_batch_tokens=10,
            clock=lambda: 0.0,
        )
        first = scheduler.submit("first", torch.tensor([1, 2]), make_contract())
        second = scheduler.submit("second", torch.tensor([3, 4, 5]), make_contract())
        foreign = scheduler.submit(
            "foreign",
            torch.tensor([6]),
            make_contract(namespace="tenant-b"),
        )

        self.assertEqual((), scheduler.poll(now=0.5))
        batches = scheduler.poll(now=1.0)

        self.assertEqual(2, len(batches))
        compatible = next(batch for batch in batches if batch.request_ids == ("first", "second"))
        isolated = next(batch for batch in batches if batch.request_ids == ("foreign",))
        self.assertIs(compatible.handles[0], first)
        self.assertIs(compatible.handles[1], second)
        self.assertIs(isolated.handles[0], foreign)
        self.assertEqual((2, 3), compatible.sequence_lengths)
        self.assertEqual((2, 3), compatible.input_ids.shape)
        self.assertTrue(
            torch.equal(
                compatible.input_ids,
                torch.tensor([[1, 2, PAD_TOKEN_ID], [3, 4, 5]]),
            )
        )
        self.assertTrue(
            torch.equal(
                compatible.attention_mask,
                torch.tensor([[True, True, False], [True, True, True]]),
            )
        )
        self.assertEqual("tenant-a", compatible.contract.namespace)
        self.assertEqual("tenant-b", isolated.contract.namespace)

    def test_fifo_is_preserved_inside_a_contract_group(self) -> None:
        scheduler = InferenceBatchScheduler(
            max_wait_seconds=100.0,
            max_batch_items=2,
            max_batch_tokens=20,
            clock=lambda: 0.0,
        )
        handles = [
            scheduler.submit(f"request-{index}", torch.tensor([index]), make_contract())
            for index in range(3)
        ]

        first_batch = scheduler.poll(now=0.0)
        self.assertEqual(("request-2",), scheduler.queued_request_ids())
        second_batch = scheduler.flush(now=0.0)

        self.assertEqual(("request-0", "request-1"), first_batch[0].request_ids)
        self.assertEqual(("request-2",), second_batch[0].request_ids)
        self.assertEqual(handles[0].request_id, first_batch[0].handles[0].request_id)

    def test_item_and_real_token_limits_bound_each_emitted_batch(self) -> None:
        scheduler = InferenceBatchScheduler(
            max_wait_seconds=100.0,
            max_batch_items=3,
            max_batch_tokens=5,
            clock=lambda: 0.0,
        )
        scheduler.submit("one", torch.tensor([1, 2]), make_contract())
        scheduler.submit("two", torch.tensor([3, 4, 5]), make_contract())
        scheduler.submit("three", torch.tensor([6, 7]), make_contract())

        first_batch = scheduler.poll(now=0.0)
        remaining_batch = scheduler.flush(now=0.0)

        self.assertEqual(1, len(first_batch))
        self.assertEqual(("one", "two"), first_batch[0].request_ids)
        self.assertLessEqual(len(first_batch[0].request_ids), 3)
        self.assertLessEqual(first_batch[0].token_count, 5)
        self.assertEqual(("three",), remaining_batch[0].request_ids)

    def test_states_and_results_remain_attached_to_their_request_ids(self) -> None:
        scheduler = InferenceBatchScheduler(
            max_wait_seconds=0.0,
            max_batch_items=2,
            clock=lambda: 0.0,
        )
        first_state = make_state(1.0)
        second_state = make_state(2.0)
        first = scheduler.submit(
            "first",
            torch.tensor([10]),
            make_contract(),
            initial_state=first_state,
        )
        second = scheduler.submit(
            "second",
            torch.tensor([11]),
            make_contract(),
            initial_state=second_state,
        )

        batch = scheduler.poll(now=0.0)[0]
        first_final = make_state(3.0)
        second_final = make_state(4.0)
        scheduler.complete_batch(
            batch,
            outputs={"second": "second-output", "first": "first-output"},
            final_states={"first": first_final, "second": second_final},
            completed_at=1.0,
        )

        self.assertIs(batch.initial_states[0], first_state)
        self.assertIs(batch.initial_states[1], second_state)
        self.assertIs(first.result().final_state, first_final)
        self.assertIs(second.result().final_state, second_final)
        self.assertEqual("first-output", first.result().output)
        self.assertEqual("second-output", second.result().output)
        self.assertEqual((1.0, 1.0), (first.result().latency_seconds, second.result().latency_seconds))

    def test_cancelled_request_is_removed_without_affecting_other_requests(self) -> None:
        scheduler = InferenceBatchScheduler(
            max_wait_seconds=100.0,
            max_batch_items=4,
            clock=lambda: 0.0,
        )
        cancelled = scheduler.submit("cancelled", torch.tensor([1]), make_contract())
        kept = scheduler.submit("kept", torch.tensor([2]), make_contract())

        self.assertTrue(cancelled.cancel())
        self.assertFalse(cancelled.cancel())
        batch = scheduler.flush(now=0.0)[0]
        scheduler.complete_batch(batch, {"kept": "ok"}, {"kept": None}, completed_at=0.0)

        self.assertEqual(("kept",), batch.request_ids)
        self.assertTrue(cancelled.cancelled())
        with self.assertRaises(CancelledError):
            cancelled.result()
        self.assertEqual("ok", kept.result().output)
        metrics = scheduler.metrics(now=0.0)
        self.assertEqual(1, metrics.cancelled_requests)
        self.assertEqual(1, metrics.completed_requests)
        self.assertEqual(0, metrics.pending_requests)

    def test_cancelled_emitted_request_is_skipped_when_the_batch_is_completed(self) -> None:
        scheduler = InferenceBatchScheduler(
            max_wait_seconds=0.0,
            max_batch_items=2,
            clock=lambda: 0.0,
        )
        cancelled = scheduler.submit("cancelled", torch.tensor([1]), make_contract())
        kept = scheduler.submit("kept", torch.tensor([2]), make_contract())
        batch = scheduler.poll(now=0.0)[0]

        self.assertTrue(cancelled.cancel())
        results = scheduler.complete_batch(
            batch,
            {"kept": "ok"},
            {"kept": None},
            completed_at=0.0,
        )

        self.assertEqual(("kept",), tuple(result.request_id for result in results))
        self.assertEqual("ok", kept.result().output)
        with self.assertRaises(CancelledError):
            cancelled.result()
        self.assertEqual(1, scheduler.metrics(now=0.0).cancelled_requests)

    def test_timeout_fails_a_pending_request_explicitly(self) -> None:
        scheduler = InferenceBatchScheduler(max_wait_seconds=100.0, clock=lambda: 0.0)
        handle = scheduler.submit(
            "expiring",
            torch.tensor([1]),
            make_contract(),
            timeout_seconds=2.0,
        )

        self.assertEqual(0, scheduler.expire(now=1.99))
        self.assertEqual(1, scheduler.expire(now=2.0))
        with self.assertRaises(TimeoutError):
            handle.result()
        metrics = scheduler.metrics(now=2.0)
        self.assertEqual(1, metrics.timed_out_requests)
        self.assertEqual(0, metrics.pending_requests)
        self.assertEqual((), scheduler.flush(now=2.0))

    def test_active_timeout_is_not_resolved_by_late_model_completion(self) -> None:
        scheduler = InferenceBatchScheduler(max_wait_seconds=0.0, clock=lambda: 0.0)
        handle = scheduler.submit(
            "expiring",
            torch.tensor([1]),
            make_contract(),
            timeout_seconds=1.0,
        )
        batch = scheduler.poll(now=0.0)[0]

        self.assertEqual((), scheduler.complete_batch(batch, {}, {}, completed_at=2.0))
        with self.assertRaises(TimeoutError):
            handle.result()
        self.assertEqual(0, scheduler.metrics(now=2.0).in_flight_requests)

    def test_metrics_report_queue_wait_latency_and_padding_cost(self) -> None:
        clock = FakeClock()
        scheduler = InferenceBatchScheduler(
            max_wait_seconds=100.0,
            max_batch_items=2,
            clock=clock,
        )
        first = scheduler.submit("first", torch.tensor([1]), make_contract())
        second = scheduler.submit("second", torch.tensor([2, 3, 4]), make_contract())
        clock.advance(0.5)
        batch = scheduler.flush()[0]
        clock.advance(0.75)
        scheduler.complete_batch(
            batch,
            {"first": "one", "second": "two"},
            {"first": None, "second": None},
        )

        metrics = scheduler.metrics()
        self.assertEqual(2, metrics.submitted_requests)
        self.assertEqual(2, metrics.dispatched_requests)
        self.assertEqual(2, metrics.completed_requests)
        self.assertEqual(4, metrics.dispatched_tokens)
        self.assertEqual(6, metrics.dispatched_padded_tokens)
        self.assertAlmostEqual(1.0, metrics.total_queue_wait_seconds)
        self.assertAlmostEqual(2.5, metrics.total_latency_seconds)
        self.assertAlmostEqual(0.5, metrics.mean_queue_wait_seconds)
        self.assertAlmostEqual(1.25, metrics.mean_latency_seconds)
        self.assertEqual("one", first.result().output)
        self.assertEqual("two", second.result().output)

    def test_concurrent_submission_does_not_lose_or_duplicate_requests(self) -> None:
        scheduler = InferenceBatchScheduler(
            max_wait_seconds=100.0,
            max_batch_items=7,
            max_batch_tokens=100,
        )
        total_requests = 100

        def submit_request(index: int) -> str:
            scheduler.submit(
                f"request-{index}",
                torch.tensor([index % 255]),
                make_contract(),
            )
            return f"request-{index}"

        with ThreadPoolExecutor(max_workers=8) as executor:
            submitted_ids = tuple(executor.map(submit_request, range(total_requests)))

        batches = scheduler.flush(now=0.0)
        batched_ids = [request_id for batch in batches for request_id in batch.request_ids]
        for batch in batches:
            scheduler.complete_batch(
                batch,
                {request_id: request_id for request_id in batch.request_ids},
                {request_id: None for request_id in batch.request_ids},
                completed_at=0.0,
            )

        self.assertEqual(total_requests, len(batched_ids))
        self.assertEqual(set(submitted_ids), set(batched_ids))
        self.assertEqual(total_requests, len(set(batched_ids)))
        metrics = scheduler.metrics(now=0.0)
        self.assertEqual(total_requests, metrics.completed_requests)
        self.assertEqual(0, metrics.pending_requests)
        self.assertEqual(0, metrics.in_flight_requests)

    def test_contract_groups_by_model_dtype_device_and_namespace(self) -> None:
        scheduler = InferenceBatchScheduler(
            max_wait_seconds=0.0,
            max_batch_items=4,
            clock=lambda: 0.0,
        )
        scheduler.submit("first", torch.tensor([1]), make_contract(namespace="one"))
        scheduler.submit("second", torch.tensor([2]), make_contract(namespace="one", model_id="other"))
        scheduler.submit("third", torch.tensor([3]), BatchContract("herm-test", "cpu", torch.float64, "one"))
        scheduler.submit("fourth", torch.tensor([4]), make_contract(namespace="two"))

        batches = scheduler.poll(now=0.0)

        self.assertEqual(4, len(batches))
        self.assertEqual(
            {("herm-test", "cpu", "torch.float32", "one"),
             ("other", "cpu", "torch.float32", "one"),
             ("herm-test", "cpu", "torch.float64", "one"),
             ("herm-test", "cpu", "torch.float32", "two")},
            {batch.contract.key for batch in batches},
        )
        self.assertEqual(
            {"one", "two"},
            {batch.contract.namespace for batch in batches},
        )


if __name__ == "__main__":
    unittest.main()
