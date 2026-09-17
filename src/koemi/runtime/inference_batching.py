from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import torch
from torch import Tensor

from koemi.configuration.settings import PAD_TOKEN_ID
from koemi.model.state import KoemiState


DEFAULT_MAX_WAIT_SECONDS = 0.005
DEFAULT_MAX_BATCH_ITEMS = 8
DEFAULT_MAX_BATCH_TOKENS = 2048
PREFILL_PHASE = "prefill"
DECODE_PHASE = "decode"
_VALID_PHASES = frozenset((PREFILL_PHASE, DECODE_PHASE))

_STATE_TENSOR_FIELDS = (
    "working_state",
    "memory_basis",
    "memory_normalizer",
    "refine_basis",
    "refine_normalizer",
    "local_keys",
    "local_values",
    "local_valid",
    "salient_keys",
    "salient_values",
    "salient_valid",
    "last_token_ids",
)
_STATE_FLOAT_FIELDS = (
    "working_state",
    "memory_basis",
    "memory_normalizer",
    "refine_basis",
    "refine_normalizer",
    "local_keys",
    "local_values",
    "salient_keys",
    "salient_values",
)


@dataclass(frozen=True)
class BatchContract:
    """Identifies requests that can share one inference phase and batch."""

    model_id: str
    device: torch.device | str
    dtype: torch.dtype
    namespace: str
    phase: str = PREFILL_PHASE

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if not isinstance(self.phase, str) or self.phase not in _VALID_PHASES:
            raise ValueError(f"phase must be one of {sorted(_VALID_PHASES)}")
        normalized_device = _normalize_device(self.device)
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        object.__setattr__(self, "device", normalized_device)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.model_id, str(self.device), str(self.dtype), self.namespace

    @property
    def scheduling_key(self) -> tuple[str, str, str, str, str]:
        return (*self.key, self.phase)


@dataclass(frozen=True)
class BatchingMetrics:
    """Snapshot of queue, dispatch, cancellation, and latency counters."""

    submitted_requests: int
    dispatched_requests: int
    completed_requests: int
    cancelled_requests: int
    timed_out_requests: int
    emitted_batches: int
    pending_requests: int
    in_flight_requests: int
    peak_queue_depth: int
    queued_tokens: int
    dispatched_tokens: int
    dispatched_padded_tokens: int
    total_queue_wait_seconds: float
    total_latency_seconds: float
    oldest_pending_wait_seconds: float

    @property
    def mean_queue_wait_seconds(self) -> float:
        if self.dispatched_requests == 0:
            return 0.0
        return self.total_queue_wait_seconds / self.dispatched_requests

    @property
    def mean_latency_seconds(self) -> float:
        if self.completed_requests == 0:
            return 0.0
        return self.total_latency_seconds / self.completed_requests


@dataclass(frozen=True)
class InferenceResult:
    """Result delivered to one request handle after caller-owned execution."""

    request_id: str
    output: object
    final_state: KoemiState | None
    queue_wait_seconds: float
    latency_seconds: float


@dataclass(frozen=True)
class InferenceBatch:
    """Padded request batch and the per-request state handles it represents.

    `padded_token_count` is the scheduler budget unit. `pinned_host_memory` and
    `non_blocking_transfer` describe the optional CPU-to-CUDA staging that
    produced the tensors; they are false for the CPU path.
    """

    batch_id: int
    contract: BatchContract
    request_ids: tuple[str, ...]
    input_ids: Tensor
    attention_mask: Tensor
    initial_states: tuple[KoemiState | None, ...]
    sequence_lengths: tuple[int, ...]
    handles: tuple[InferenceRequestHandle, ...]
    dispatched_at: float
    pinned_host_memory: bool = False
    non_blocking_transfer: bool = False

    @property
    def token_count(self) -> int:
        return sum(self.sequence_lengths)

    @property
    def padded_token_count(self) -> int:
        return int(self.input_ids.numel())


class InferenceRequestHandle:
    """Future-like control handle for one queued inference request."""

    def __init__(self, scheduler: InferenceBatchScheduler, request_id: str, future: Future[InferenceResult]) -> None:
        self._scheduler = scheduler
        self._request_id = request_id
        self._future = future

    @property
    def request_id(self) -> str:
        return self._request_id

    def cancel(self) -> bool:
        """Cancel the request if it has not reached a terminal state."""

        return self._scheduler.cancel(self.request_id)

    def result(self, timeout: float | None = None) -> InferenceResult:
        """Wait for the caller to complete this request and return its result."""

        return self._future.result(timeout)

    def done(self) -> bool:
        return self._future.done()

    def cancelled(self) -> bool:
        return self._future.cancelled()


@dataclass
class _QueuedRequest:
    request_id: str
    input_ids: Tensor
    contract: BatchContract
    initial_state: KoemiState | None
    enqueued_at: float
    deadline: float | None
    handle: InferenceRequestHandle
    future: Future[InferenceResult]
    status: str = "queued"
    batch_id: int | None = None


@dataclass(frozen=True)
class _ActiveBatch:
    batch: InferenceBatch
    requests: tuple[_QueuedRequest, ...]


class InferenceBatchScheduler:
    """Thread-safe opt-in scheduler that never invokes a model itself."""

    def __init__(
        self,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
        max_batch_items: int = DEFAULT_MAX_BATCH_ITEMS,
        max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
        clock: Callable[[], float] | None = None,
        length_bucket_size: int | None = None,
        pin_memory: bool = True,
    ) -> None:
        """Create an opt-in queue with a padded-token budget.

        Requests are partitioned by contract phase and, when configured, by
        length bucket. The scheduler does not invoke a model or own execution.
        """
        self._max_wait_seconds = _validate_non_negative_float(max_wait_seconds, "max_wait_seconds")
        self._max_batch_items = _validate_positive_int(max_batch_items, "max_batch_items")
        self._max_batch_tokens = _validate_positive_int(max_batch_tokens, "max_batch_tokens")
        self._length_bucket_size = (
            None
            if length_bucket_size is None
            else _validate_positive_int(length_bucket_size, "length_bucket_size")
        )
        if not isinstance(pin_memory, bool):
            raise TypeError("pin_memory must be a boolean")
        self._pin_memory = pin_memory
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._pending_by_key: dict[tuple[str, str, str, str, str, int | None], deque[_QueuedRequest]] = {}
        self._group_order: deque[tuple[str, str, str, str, str, int | None]] = deque()
        self._requests: dict[str, _QueuedRequest] = {}
        self._active_batches: dict[int, _ActiveBatch] = {}
        self._next_batch_id = 1
        self._pending_requests = 0
        self._pending_tokens = 0
        self._submitted_requests = 0
        self._dispatched_requests = 0
        self._completed_requests = 0
        self._cancelled_requests = 0
        self._timed_out_requests = 0
        self._emitted_batches = 0
        self._peak_queue_depth = 0
        self._dispatched_tokens = 0
        self._dispatched_padded_tokens = 0
        self._total_queue_wait_seconds = 0.0
        self._total_latency_seconds = 0.0

    def submit(
        self,
        request_id: str,
        input_ids: Tensor,
        contract: BatchContract,
        initial_state: KoemiState | None = None,
        timeout_seconds: float | None = None,
    ) -> InferenceRequestHandle:
        """Queue one validated request without running model code."""

        _validate_request_id(request_id)
        if not isinstance(contract, BatchContract):
            raise TypeError("contract must be a BatchContract")
        normalized_input_ids = _normalize_input_ids(input_ids, contract)
        token_count = int(normalized_input_ids.numel())
        if contract.phase == DECODE_PHASE and token_count != 1:
            raise ValueError("decode requests must contain exactly one token")
        if token_count > self._max_batch_tokens:
            raise ValueError("request padded token count exceeds max_batch_tokens")
        if contract.phase == DECODE_PHASE and initial_state is None:
            raise ValueError("decode requests require an initial recurrent state")
        _validate_state(initial_state, contract, "initial_state")
        timeout = _validate_timeout(timeout_seconds)
        enqueued_at = _validate_time(self._clock(), "clock")
        deadline = enqueued_at + timeout if timeout is not None else None
        future: Future[InferenceResult] = Future()
        handle = InferenceRequestHandle(self, request_id, future)
        request = _QueuedRequest(
            request_id=request_id,
            input_ids=normalized_input_ids,
            contract=contract,
            initial_state=initial_state,
            enqueued_at=enqueued_at,
            deadline=deadline,
            handle=handle,
            future=future,
        )
        group_key = self._group_key(contract, token_count)
        with self._lock:
            if request_id in self._requests:
                raise ValueError(f"request_id is already active: {request_id}")
            queue = self._pending_by_key.get(group_key)
            if queue is None:
                queue = deque()
                self._pending_by_key[group_key] = queue
                self._group_order.append(group_key)
            queue.append(request)
            self._requests[request_id] = request
            self._pending_requests += 1
            self._pending_tokens += token_count
            self._submitted_requests += 1
            self._peak_queue_depth = max(self._peak_queue_depth, self._pending_requests)
        return handle

    def poll(self, now: float | None = None) -> tuple[InferenceBatch, ...]:
        """Emit every group that reached its wait or capacity threshold."""

        current_time = self._resolve_time(now)
        return self._dispatch_ready(current_time, force=False)

    def flush(self, now: float | None = None) -> tuple[InferenceBatch, ...]:
        """Emit all non-expired pending requests, respecting batch limits."""

        current_time = self._resolve_time(now)
        return self._dispatch_ready(current_time, force=True)

    def expire(self, now: float | None = None) -> int:
        """Fail queued or in-flight requests whose explicit deadline has passed."""

        current_time = self._resolve_time(now)
        with self._lock:
            timed_out_futures = self._expire_locked(current_time)
        for future in timed_out_futures:
            future.set_exception(TimeoutError("inference request timed out"))
        return len(timed_out_futures)

    def cancel(self, request_id: str) -> bool:
        """Cancel a queued or emitted request and prevent its future from resolving."""

        _validate_request_id(request_id)
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.status in {"cancelled", "timed_out", "completed"}:
                return False
            if request.status == "queued":
                self._remove_queued_request_locked(request)
                self._requests.pop(request_id, None)
            request.status = "cancelled"
            self._cancelled_requests += 1
            future = request.future
        future.cancel()
        return True

    def complete_batch(
        self,
        batch: InferenceBatch,
        outputs: Mapping[str, object] | Sequence[object],
        final_states: Mapping[str, KoemiState | None] | Sequence[KoemiState | None],
        completed_at: float | None = None,
    ) -> tuple[InferenceResult, ...]:
        """Resolve emitted handles from caller-provided outputs and isolated states."""

        if not isinstance(batch, InferenceBatch):
            raise TypeError("batch must be an InferenceBatch")
        current_time = self._resolve_time(completed_at)
        with self._lock:
            active = self._active_batches.get(batch.batch_id)
            if active is None or active.batch is not batch:
                raise ValueError("batch is not active in this scheduler")
        self.expire(current_time)
        output_by_id = _values_by_request_id(outputs, batch.request_ids, "outputs")
        state_by_id = _values_by_request_id(final_states, batch.request_ids, "final_states")
        with self._lock:
            active = self._active_batches.get(batch.batch_id)
            if active is None or active.batch is not batch:
                raise ValueError("batch is not active in this scheduler")
            for request in active.requests:
                if request.status == "dispatched" and request.future.cancelled():
                    request.status = "cancelled"
                    self._cancelled_requests += 1
            active_requests = tuple(
                request for request in active.requests if request.status == "dispatched"
            )
            missing_outputs = [
                request.request_id
                for request in active_requests
                if request.request_id not in output_by_id
            ]
            if missing_outputs:
                raise ValueError(f"outputs missing request IDs: {missing_outputs}")
            missing_states = [
                request.request_id
                for request in active_requests
                if request.request_id not in state_by_id
            ]
            if missing_states:
                raise ValueError(f"final_states missing request IDs: {missing_states}")
            for request in active_requests:
                final_state = state_by_id[request.request_id]
                if request.contract.phase == DECODE_PHASE and final_state is None:
                    raise ValueError("decode requests require a recurrent final state")
                _validate_state(
                    final_state,
                    request.contract,
                    "final_state",
                )
            results = []
            for request in active_requests:
                queue_wait = max(0.0, batch.dispatched_at - request.enqueued_at)
                latency = max(0.0, current_time - request.enqueued_at)
                results.append(
                    InferenceResult(
                        request_id=request.request_id,
                        output=output_by_id[request.request_id],
                        final_state=state_by_id[request.request_id],
                        queue_wait_seconds=queue_wait,
                        latency_seconds=latency,
                    )
                )
                request.status = "completed"
                self._completed_requests += 1
                self._total_latency_seconds += latency
            for request in active.requests:
                self._requests.pop(request.request_id, None)
            self._active_batches.pop(batch.batch_id, None)
        requests_by_id = {request.request_id: request for request in active.requests}
        for result in results:
            requests_by_id[result.request_id].future.set_result(result)
        return tuple(results)

    def metrics(self, now: float | None = None) -> BatchingMetrics:
        """Return a consistent snapshot without expiring requests implicitly."""

        current_time = self._resolve_time(now)
        with self._lock:
            oldest_wait = 0.0
            for request in self._requests.values():
                if request.status != "queued":
                    continue
                oldest_wait = max(oldest_wait, max(0.0, current_time - request.enqueued_at))
            in_flight_requests = sum(
                request.status in {"dispatched", "cancelled", "timed_out"}
                for active in self._active_batches.values()
                for request in active.requests
            )
            return BatchingMetrics(
                submitted_requests=self._submitted_requests,
                dispatched_requests=self._dispatched_requests,
                completed_requests=self._completed_requests,
                cancelled_requests=self._cancelled_requests,
                timed_out_requests=self._timed_out_requests,
                emitted_batches=self._emitted_batches,
                pending_requests=self._pending_requests,
                in_flight_requests=in_flight_requests,
                peak_queue_depth=self._peak_queue_depth,
                queued_tokens=self._pending_tokens,
                dispatched_tokens=self._dispatched_tokens,
                dispatched_padded_tokens=self._dispatched_padded_tokens,
                total_queue_wait_seconds=self._total_queue_wait_seconds,
                total_latency_seconds=self._total_latency_seconds,
                oldest_pending_wait_seconds=oldest_wait,
            )

    def queued_request_ids(self) -> tuple[str, ...]:
        """Return pending IDs in scheduler group order and per-group FIFO order."""

        with self._lock:
            return tuple(
                request.request_id
                for group_key in self._group_order
                for request in self._pending_by_key.get(group_key, ())
            )

    def _resolve_time(self, value: float | None) -> float:
        return _validate_time(self._clock() if value is None else value, "time")

    def _group_key(
        self,
        contract: BatchContract,
        sequence_length: int,
    ) -> tuple[str, str, str, str, str, int | None]:
        bucket_key = (
            None
            if self._length_bucket_size is None
            else (sequence_length - 1) // self._length_bucket_size
        )
        return (*contract.scheduling_key, bucket_key)

    def _dispatch_ready(self, current_time: float, force: bool) -> tuple[InferenceBatch, ...]:
        with self._lock:
            timed_out_futures = self._expire_locked(current_time)
            batches: list[InferenceBatch] = []
            for group_key in tuple(self._group_order):
                while True:
                    queue = self._pending_by_key.get(group_key)
                    if not queue or not self._group_is_ready(queue, current_time, force):
                        break
                    requests = self._select_requests(queue)
                    batch = self._create_batch_locked(requests, current_time)
                    for request in requests:
                        queue.popleft()
                        request.status = "dispatched"
                        request.batch_id = batch.batch_id
                    self._pending_requests -= len(requests)
                    self._pending_tokens -= sum(request.input_ids.numel() for request in requests)
                    self._active_batches[batch.batch_id] = _ActiveBatch(batch, tuple(requests))
                    self._dispatched_requests += len(requests)
                    self._emitted_batches += 1
                    self._dispatched_tokens += batch.token_count
                    self._dispatched_padded_tokens += batch.padded_token_count
                    self._total_queue_wait_seconds += sum(
                        max(0.0, current_time - request.enqueued_at) for request in requests
                    )
                    batches.append(batch)
                    if not queue:
                        self._remove_empty_group_locked(group_key)
        for future in timed_out_futures:
            future.set_exception(TimeoutError("inference request timed out"))
        return tuple(batches)

    def _group_is_ready(
        self,
        queue: deque[_QueuedRequest],
        current_time: float,
        force: bool,
    ) -> bool:
        if force:
            return True
        if len(queue) >= self._max_batch_items:
            return True
        selected = self._select_requests(queue)
        if self._padded_token_count(selected) >= self._max_batch_tokens:
            return True
        return current_time - queue[0].enqueued_at >= self._max_wait_seconds

    def _select_requests(self, queue: deque[_QueuedRequest]) -> tuple[_QueuedRequest, ...]:
        selected: list[_QueuedRequest] = []
        selected_max_length = 0
        for request in queue:
            if len(selected) >= self._max_batch_items:
                break
            request_tokens = int(request.input_ids.numel())
            proposed_max_length = max(selected_max_length, request_tokens)
            proposed_padded_tokens = (len(selected) + 1) * proposed_max_length
            if selected and proposed_padded_tokens > self._max_batch_tokens:
                break
            selected.append(request)
            selected_max_length = proposed_max_length
        if not selected:
            raise RuntimeError("ready queue did not yield a batch")
        return tuple(selected)

    @staticmethod
    def _padded_token_count(requests: Sequence[_QueuedRequest]) -> int:
        if not requests:
            return 0
        return len(requests) * max(int(request.input_ids.numel()) for request in requests)

    def _create_batch_locked(
        self,
        requests: tuple[_QueuedRequest, ...],
        dispatched_at: float,
    ) -> InferenceBatch:
        contract = requests[0].contract
        maximum_length = max(request.input_ids.numel() for request in requests)
        input_ids, attention_mask, pinned_host_memory, non_blocking_transfer = self._prepare_batch_tensors(
            requests,
            maximum_length,
            contract,
        )
        batch_id = self._next_batch_id
        self._next_batch_id += 1
        return InferenceBatch(
            batch_id=batch_id,
            contract=contract,
            request_ids=tuple(request.request_id for request in requests),
            input_ids=input_ids,
            attention_mask=attention_mask,
            initial_states=tuple(request.initial_state for request in requests),
            sequence_lengths=tuple(int(request.input_ids.numel()) for request in requests),
            handles=tuple(request.handle for request in requests),
            dispatched_at=dispatched_at,
            pinned_host_memory=pinned_host_memory,
            non_blocking_transfer=non_blocking_transfer,
        )

    def _prepare_batch_tensors(
        self,
        requests: tuple[_QueuedRequest, ...],
        maximum_length: int,
        contract: BatchContract,
    ) -> tuple[Tensor, Tensor, bool, bool]:
        shape = (len(requests), maximum_length)
        destination_device = contract.device
        if destination_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA batch contract cannot be dispatched without CUDA")
        can_stage_cpu_inputs = (
            destination_device.type == "cuda"
            and all(request.input_ids.device.type == "cpu" for request in requests)
        )
        if can_stage_cpu_inputs:
            pinned_host_memory = self._pin_memory
            host_input_ids = torch.full(
                shape,
                PAD_TOKEN_ID,
                dtype=torch.long,
                device="cpu",
                pin_memory=pinned_host_memory,
            )
            host_attention_mask = torch.zeros(
                shape,
                dtype=torch.bool,
                device="cpu",
                pin_memory=pinned_host_memory,
            )
            for row_index, request in enumerate(requests):
                length = request.input_ids.numel()
                host_input_ids[row_index, :length].copy_(request.input_ids)
                host_attention_mask[row_index, :length] = True
            return (
                host_input_ids.to(destination_device, non_blocking=pinned_host_memory),
                host_attention_mask.to(destination_device, non_blocking=pinned_host_memory),
                pinned_host_memory,
                pinned_host_memory,
            )
        input_ids = torch.full(
            shape,
            PAD_TOKEN_ID,
            dtype=torch.long,
            device=destination_device,
        )
        attention_mask = torch.zeros(shape, dtype=torch.bool, device=destination_device)
        for row_index, request in enumerate(requests):
            length = request.input_ids.numel()
            source = request.input_ids
            if source.device != destination_device:
                source = source.to(destination_device, non_blocking=False)
            input_ids[row_index, :length].copy_(source)
            attention_mask[row_index, :length] = True
        return input_ids, attention_mask, False, False

    def _expire_locked(self, current_time: float) -> list[Future[InferenceResult]]:
        timed_out_futures: list[Future[InferenceResult]] = []
        for group_key in tuple(self._group_order):
            queue = self._pending_by_key.get(group_key)
            if queue is None:
                continue
            remaining = deque()
            while queue:
                request = queue.popleft()
                if request.deadline is not None and current_time >= request.deadline:
                    request.status = "timed_out"
                    self._requests.pop(request.request_id, None)
                    self._pending_requests -= 1
                    self._pending_tokens -= request.input_ids.numel()
                    self._timed_out_requests += 1
                    timed_out_futures.append(request.future)
                else:
                    remaining.append(request)
            self._pending_by_key[group_key] = remaining
            if not remaining:
                self._remove_empty_group_locked(group_key)
        for active in self._active_batches.values():
            for request in active.requests:
                if (
                    request.status == "dispatched"
                    and request.deadline is not None
                    and current_time >= request.deadline
                ):
                    request.status = "timed_out"
                    self._timed_out_requests += 1
                    timed_out_futures.append(request.future)
        return timed_out_futures

    def _remove_queued_request_locked(self, request: _QueuedRequest) -> None:
        group_key = self._group_key(request.contract, int(request.input_ids.numel()))
        queue = self._pending_by_key[group_key]
        queue.remove(request)
        self._pending_requests -= 1
        self._pending_tokens -= request.input_ids.numel()
        if not queue:
            self._remove_empty_group_locked(group_key)

    def _remove_empty_group_locked(
        self,
        group_key: tuple[str, str, str, str, str, int | None],
    ) -> None:
        queue = self._pending_by_key.get(group_key)
        if queue:
            return
        self._pending_by_key.pop(group_key, None)
        try:
            self._group_order.remove(group_key)
        except ValueError:
            return


def _validate_request_id(request_id: str) -> None:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_non_negative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _validate_timeout(value: float | None) -> float | None:
    if value is None:
        return None
    return _validate_non_negative_float(value, "timeout_seconds")


def _validate_time(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must return a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _normalize_device(device: torch.device | str) -> torch.device:
    try:
        normalized_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must be a valid torch device") from error
    if normalized_device.type == "cuda" and normalized_device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return normalized_device


def _normalize_input_ids(input_ids: Tensor, contract: BatchContract) -> Tensor:
    if not isinstance(input_ids, Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids.reshape(-1)
    if input_ids.ndim != 1 or input_ids.numel() == 0:
        raise ValueError("input_ids must have shape [sequence] or [1, sequence] and be non-empty")
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must use torch.long")
    if input_ids.device != contract.device:
        raise ValueError("input_ids device must match the batch contract")
    if bool(torch.any(input_ids < 0).item()) or bool(torch.any(input_ids >= PAD_TOKEN_ID).item()):
        raise ValueError("input_ids must contain byte token IDs without padding")
    return input_ids.detach().clone()


def _validate_state(
    state: KoemiState | None,
    contract: BatchContract,
    name: str,
) -> None:
    if state is None:
        return
    if not isinstance(state, KoemiState):
        raise TypeError(f"{name} must be KoemiState or None")
    if isinstance(state.step_index, bool) or not isinstance(state.step_index, int) or state.step_index < 0:
        raise ValueError(f"{name}.step_index must be a non-negative integer")
    for field_name in _STATE_TENSOR_FIELDS:
        value = getattr(state, field_name)
        if not isinstance(value, Tensor):
            raise TypeError(f"{name}.{field_name} must be a tensor")
        if value.ndim == 0 or value.shape[0] != 1:
            raise ValueError(f"{name}.{field_name} must have batch dimension 1")
        if value.device != contract.device:
            raise ValueError(f"{name}.{field_name} device must match the batch contract")
    for field_name in _STATE_FLOAT_FIELDS:
        value = getattr(state, field_name)
        if value.dtype != contract.dtype:
            raise ValueError(f"{name}.{field_name} dtype must match the batch contract")


def _values_by_request_id(
    values: Mapping[str, Any] | Sequence[Any],
    request_ids: tuple[str, ...],
    name: str,
) -> dict[str, Any]:
    if isinstance(values, Mapping):
        unknown_ids = [key for key in values if not isinstance(key, str) or key not in request_ids]
        if unknown_ids:
            raise ValueError(f"{name} contains unknown request IDs: {unknown_ids}")
        return dict(values)
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a mapping or a sequence in batch order")
    if len(values) != len(request_ids):
        raise ValueError(f"{name} must contain one value per request")
    return dict(zip(request_ids, values, strict=True))


__all__ = [
    "BatchContract",
    "BatchingMetrics",
    "DECODE_PHASE",
    "InferenceBatch",
    "InferenceBatchScheduler",
    "InferenceRequestHandle",
    "InferenceResult",
    "PREFILL_PHASE",
]
