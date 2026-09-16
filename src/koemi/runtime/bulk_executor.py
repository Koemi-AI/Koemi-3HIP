from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
Shape = tuple[int | None, ...]


class BulkExecutorError(RuntimeError):
    """Base error for the bounded bulk execution seam."""


class BulkClosedError(BulkExecutorError):
    """Raised when work is submitted after the executor started closing."""


class BulkBackpressureError(BulkExecutorError):
    """Raised when a bounded submission wait reaches its timeout."""


class BulkValidationError(ValueError):
    """Raised when a prepared payload violates its tensor-tree contract."""


@dataclass(frozen=True)
class BulkMetrics:
    submitted: int
    completed: int
    failed: int
    cancelled: int
    in_flight: int
    snapshot_seconds: float
    prepare_seconds: float
    enqueue_seconds: float
    total_seconds: float
    backpressure_seconds: float

    @property
    def cpu_prepare_seconds(self) -> float:
        return self.prepare_seconds

    @property
    def device_enqueue_seconds(self) -> float:
        return self.enqueue_seconds


@dataclass(frozen=True)
class BulkResult(Generic[OutputType]):
    sequence: int
    value: OutputType
    device: torch.device
    completion_event: torch.cuda.Event | None = None

    def wait(self) -> OutputType:
        """Wait for this result's CUDA event, when one exists, and return its value."""
        if self.completion_event is not None:
            self.completion_event.synchronize()
        return self.value

    def wait_for_stream(self, stream: torch.cuda.Stream) -> OutputType:
        """Make a consumer CUDA stream wait for this result without a host sync."""
        if not isinstance(stream, torch.cuda.Stream):
            raise TypeError("stream must be a torch.cuda.Stream")
        stream_device = torch.device(stream.device)
        if stream_device != self.device:
            raise ValueError(
                f"stream device {stream_device} does not match result device {self.device}"
            )
        if self.completion_event is not None:
            stream.wait_event(self.completion_event)
        return self.value


@dataclass
class _TaskState:
    sequence: int
    snapshot_seconds: float
    prepare_seconds: float = 0.0
    enqueue_seconds: float = 0.0
    worker_seconds: float = 0.0
    terminal: bool = False


class BulkExecutor(Generic[InputType, OutputType]):
    """Run caller-owned items through bounded CPU preparation and optional CUDA enqueue.

    `prepare` runs once per submitted item and never receives the caller's mutable
    object directly. `transfer`, when supplied, runs inside the executor's CUDA
    stream context and receives `(prepared_value, destination_device)`. The default
    transfer recursively moves tensors in mappings, lists and tuples.

    `max_workers=0` executes synchronously in the submitting thread. A positive
    value creates a non-daemon `ThreadPoolExecutor`; `max_in_flight` bounds both
    running and queued work, so `submit` applies backpressure. A CUDA destination
    creates one dedicated stream and records one event per successfully enqueued
    item. Futures complete after CPU preparation and device enqueue; consumers use
    `BulkResult.wait()` or `wait_for_stream()` when device readiness is required.

    `close` rejects new work, drains or cancels pending futures, shuts down the
    owned CPU executor, and synchronizes the dedicated CUDA stream once. It does
    not execute a model or retain state between requests.
    """

    def __init__(
        self,
        prepare: Callable[[InputType], OutputType] | None = None,
        *,
        device: torch.device | str = "cpu",
        max_workers: int = 0,
        max_in_flight: int | None = None,
        preserve_order: bool = False,
        expected_shapes: Mapping[str, Iterable[int | None]] | None = None,
        transfer: Callable[[OutputType, torch.device], OutputType] | None = None,
        non_blocking: bool = True,
    ) -> None:
        self._device = _normalize_device(device)
        self._max_workers = _validate_non_negative_int(max_workers, "max_workers")
        default_in_flight = self._max_workers if self._max_workers > 0 else 1
        self._max_in_flight = _validate_positive_int(
            default_in_flight if max_in_flight is None else max_in_flight,
            "max_in_flight",
        )
        _validate_bool(preserve_order, "preserve_order")
        _validate_bool(non_blocking, "non_blocking")
        if prepare is not None and not callable(prepare):
            raise TypeError("prepare must be callable or none")
        if transfer is not None and not callable(transfer):
            raise TypeError("transfer must be callable or none")

        self._prepare = prepare if prepare is not None else _identity
        self._transfer = transfer
        self._non_blocking = non_blocking
        self._preserve_order = preserve_order
        self._expected_shapes = _normalize_expected_shapes(expected_shapes)
        self._condition = threading.Condition()
        self._cuda_enqueue_lock = threading.Lock()
        self._active_futures: dict[Future[BulkResult[OutputType]], _TaskState] = {}
        self._in_flight = 0
        self._next_sequence = 0
        self._closed = False
        self._close_finished = False
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._snapshot_seconds = 0.0
        self._prepare_seconds = 0.0
        self._enqueue_seconds = 0.0
        self._total_seconds = 0.0
        self._backpressure_seconds = 0.0
        self._cuda_stream = (
            torch.cuda.Stream(device=self._device) if self._device.type == "cuda" else None
        )
        self._executor = (
            ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="koemi-bulk",
            )
            if self._max_workers > 0
            else None
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def max_in_flight(self) -> int:
        return self._max_in_flight

    @property
    def cuda_stream(self) -> torch.cuda.Stream | None:
        return self._cuda_stream

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def metrics(self) -> BulkMetrics:
        with self._condition:
            return BulkMetrics(
                self._submitted,
                self._completed,
                self._failed,
                self._cancelled,
                self._in_flight,
                self._snapshot_seconds,
                self._prepare_seconds,
                self._enqueue_seconds,
                self._total_seconds,
                self._backpressure_seconds,
            )

    def submit(
        self,
        item: InputType,
        *,
        timeout: float | None = None,
    ) -> Future[BulkResult[OutputType]]:
        """Submit one item, blocking at the configured bound until a slot opens.

        `timeout` limits only the backpressure wait. The returned future propagates
        preparation, validation and transfer exceptions from `result()` and can be
        cancelled before its CPU task starts.
        """
        normalized_timeout = _validate_timeout(timeout)
        with self._condition:
            self._wait_for_slot(normalized_timeout)
            self._in_flight += 1
            self._submitted += 1
            sequence = self._next_sequence
            self._next_sequence += 1
            snapshot_started = time.perf_counter()
            try:
                payload = _snapshot_payload(item)
            except BaseException as error:
                snapshot_seconds = time.perf_counter() - snapshot_started
                self._snapshot_seconds += snapshot_seconds
                self._total_seconds += snapshot_seconds
                self._in_flight -= 1
                self._failed += 1
                self._condition.notify_all()
                failed_future: Future[BulkResult[OutputType]] = Future()
                failed_future.set_exception(error)
                return failed_future
            snapshot_seconds = time.perf_counter() - snapshot_started
            self._snapshot_seconds += snapshot_seconds
            state = _TaskState(sequence, snapshot_seconds)

            if self._executor is None:
                future: Future[BulkResult[OutputType]] = Future()
                self._active_futures[future] = state
                future.add_done_callback(
                    lambda completed_future, task_state=state: self._finalize(
                        completed_future,
                        task_state,
                    )
                )
                try:
                    result = self._execute(state, payload)
                except BaseException as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)
                return future

            try:
                future = self._executor.submit(self._execute, state, payload)
            except BaseException as error:
                self._in_flight -= 1
                self._failed += 1
                self._total_seconds += snapshot_seconds
                self._condition.notify_all()
                failed_future = Future()
                failed_future.set_exception(error)
                return failed_future
            self._active_futures[future] = state
            future.add_done_callback(
                lambda completed_future, task_state=state: self._finalize(
                    completed_future,
                    task_state,
                )
            )
            return future

    def submit_many(
        self,
        items: Iterable[InputType],
        *,
        timeout: float | None = None,
    ) -> tuple[Future[BulkResult[OutputType]], ...]:
        """Submit an iterable in caller order while preserving bounded backpressure."""
        return tuple(self.submit(item, timeout=timeout) for item in items)

    def collect(
        self,
        futures: Iterable[Future[BulkResult[OutputType]]],
        *,
        preserve_order: bool | None = None,
        wait_for_device: bool = False,
    ) -> tuple[OutputType, ...]:
        """Collect future values in completion or submission order.

        `wait_for_device=True` explicitly waits each returned CUDA event. The
        default collects enqueue-complete values without a per-item host sync.
        """
        if preserve_order is None:
            preserve_order = self._preserve_order
        _validate_bool(preserve_order, "preserve_order")
        _validate_bool(wait_for_device, "wait_for_device")
        future_values = tuple(futures)
        if preserve_order:
            results = [self._result_from_future(future) for future in future_values]
            results.sort(key=lambda result: result.sequence)
        else:
            results = [
                self._result_from_future(future) for future in as_completed(future_values)
            ]
        if wait_for_device:
            return tuple(result.wait() for result in results)
        return tuple(result.value for result in results)

    def map(
        self,
        items: Iterable[InputType],
        *,
        preserve_order: bool | None = None,
        wait_for_device: bool = False,
        timeout: float | None = None,
    ) -> tuple[OutputType, ...]:
        """Submit and collect caller-produced items using the configured policy."""
        futures = self.submit_many(items, timeout=timeout)
        return self.collect(
            futures,
            preserve_order=preserve_order,
            wait_for_device=wait_for_device,
        )

    def cancel_pending(self) -> int:
        """Cancel queued futures and return the number cancelled successfully."""
        with self._condition:
            futures = tuple(self._active_futures)
            return sum(future.cancel() for future in futures)

    def close(self, *, cancel_pending: bool = False) -> None:
        """Close deterministically, draining work or cancelling queued futures."""
        _validate_bool(cancel_pending, "cancel_pending")
        with self._condition:
            if self._close_finished:
                return
            if self._closed:
                while not self._close_finished:
                    self._condition.wait()
                return
            self._closed = True
            futures = tuple(self._active_futures)
            if cancel_pending:
                for future in futures:
                    future.cancel()
            executor = self._executor
            cuda_stream = self._cuda_stream
            self._condition.notify_all()

        try:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=cancel_pending)
            with self._condition:
                while self._in_flight:
                    self._condition.wait()
            if cuda_stream is not None:
                cuda_stream.synchronize()
        finally:
            with self._condition:
                self._close_finished = True
                self._condition.notify_all()

    def __enter__(self) -> BulkExecutor[InputType, OutputType]:
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> bool:
        self.close(cancel_pending=exception_type is not None)
        return False

    def _wait_for_slot(self, timeout: float | None) -> None:
        started = time.perf_counter()
        deadline = None if timeout is None else started + timeout
        while self._in_flight >= self._max_in_flight:
            if self._closed:
                raise BulkClosedError("bulk executor is closed")
            if deadline is None:
                self._condition.wait()
                continue
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self._backpressure_seconds += time.perf_counter() - started
                raise BulkBackpressureError("bulk executor in-flight limit reached")
            self._condition.wait(remaining)
        if self._closed:
            raise BulkClosedError("bulk executor is closed")
        self._backpressure_seconds += time.perf_counter() - started

    def _execute(self, state: _TaskState, payload: object) -> BulkResult[OutputType]:
        worker_started = time.perf_counter()
        try:
            prepare_started = time.perf_counter()
            prepared = self._prepare(payload)
            state.prepare_seconds = time.perf_counter() - prepare_started
            self._validate_payload(prepared, require_device=False)
            enqueue_started = time.perf_counter()
            transferred, completion_event = self._enqueue(prepared)
            state.enqueue_seconds = time.perf_counter() - enqueue_started
            return BulkResult(
                state.sequence,
                transferred,
                self._device,
                completion_event,
            )
        finally:
            state.worker_seconds = time.perf_counter() - worker_started

    def _enqueue(self, prepared: OutputType) -> tuple[OutputType, torch.cuda.Event | None]:
        if self._cuda_stream is None:
            transferred = self._transfer_or_move(prepared)
            self._validate_payload(transferred, require_device=True)
            return transferred, None
        with self._cuda_enqueue_lock:
            with torch.cuda.stream(self._cuda_stream):
                transferred = self._transfer_or_move(prepared)
                self._validate_payload(transferred, require_device=True)
                completion_event = torch.cuda.Event(enable_timing=False)
                completion_event.record(self._cuda_stream)
        return transferred, completion_event

    def _transfer_or_move(self, prepared: OutputType) -> OutputType:
        if self._transfer is not None:
            return self._transfer(prepared, self._device)
        return _move_payload(prepared, self._device, self._non_blocking)

    def _validate_payload(self, payload: object, *, require_device: bool) -> None:
        tensor_items = tuple(_iter_tensors(payload))
        actual_shapes = {path: tuple(tensor.shape) for path, tensor in tensor_items}
        for path, expected_shape in self._expected_shapes.items():
            actual_shape = actual_shapes.get(path)
            if actual_shape is None:
                raise BulkValidationError(f"expected tensor at payload path {path!r}")
            if not _shape_matches(actual_shape, expected_shape):
                raise BulkValidationError(
                    f"payload tensor {path!r} has shape {actual_shape}, "
                    f"expected {expected_shape}"
                )
        if require_device:
            for path, tensor in tensor_items:
                if not _same_device(tensor.device, self._device):
                    raise BulkValidationError(
                        f"payload tensor {path!r} is on {tensor.device}, "
                        f"expected {self._device}"
                    )

    @staticmethod
    def _result_from_future(future: Future[BulkResult[OutputType]]) -> BulkResult[OutputType]:
        result = future.result()
        if not isinstance(result, BulkResult):
            raise BulkExecutorError("future did not return a BulkResult from this executor")
        return result

    def _finalize(self, future: Future[BulkResult[OutputType]], state: _TaskState) -> None:
        try:
            future.result()
        except CancelledError:
            terminal_status = "cancelled"
        except BaseException:
            terminal_status = "failed"
        else:
            terminal_status = "completed"
        with self._condition:
            if state.terminal:
                return
            state.terminal = True
            self._active_futures.pop(future, None)
            self._in_flight -= 1
            self._total_seconds += state.snapshot_seconds + state.worker_seconds
            self._prepare_seconds += state.prepare_seconds
            self._enqueue_seconds += state.enqueue_seconds
            if terminal_status == "completed":
                self._completed += 1
            elif terminal_status == "failed":
                self._failed += 1
            else:
                self._cancelled += 1
            self._condition.notify_all()


def _identity(value: InputType) -> InputType:
    return value


def _normalize_device(device: torch.device | str) -> torch.device:
    try:
        normalized_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must be a valid CPU or CUDA device") from error
    if normalized_device.type not in {"cpu", "cuda"}:
        raise ValueError("bulk executor supports only CPU and CUDA devices")
    if normalized_device.type == "cpu":
        if normalized_device.index not in {None, 0}:
            raise ValueError("CPU device index must be zero or omitted")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA bulk execution was requested but CUDA is unavailable")
    device_count = torch.cuda.device_count()
    device_index = (
        torch.cuda.current_device() if normalized_device.index is None else normalized_device.index
    )
    if device_index < 0 or device_index >= device_count:
        raise ValueError(f"CUDA device index {device_index} is outside the available range")
    return torch.device("cuda", device_index)


def _normalize_expected_shapes(
    expected_shapes: Mapping[str, Iterable[int | None]] | None,
) -> dict[str, Shape]:
    if expected_shapes is None:
        return {}
    if not isinstance(expected_shapes, Mapping):
        raise TypeError("expected_shapes must be a mapping from paths to shapes")
    normalized: dict[str, Shape] = {}
    for path, shape in expected_shapes.items():
        if not isinstance(path, str) or not path:
            raise TypeError("expected shape paths must be non-empty strings")
        try:
            dimensions = tuple(shape)
        except TypeError as error:
            raise TypeError(f"expected shape for {path!r} must be iterable") from error
        for dimension in dimensions:
            if dimension is not None and (
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            ):
                raise ValueError(
                    f"expected shape dimensions for {path!r} must be non-negative integers or none"
                )
        normalized[path] = dimensions
    return normalized


def _snapshot_payload(value: object) -> object:
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, Mapping):
        return {
            copy.deepcopy(key): _snapshot_payload(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_snapshot_payload(nested_value) for nested_value in value]
    if isinstance(value, tuple):
        nested_values = tuple(_snapshot_payload(nested_value) for nested_value in value)
        if hasattr(value, "_fields"):
            return type(value)(*nested_values)
        return nested_values
    return copy.deepcopy(value)


def _iter_tensors(value: object, path: str = "$") -> Iterable[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            nested_path = str(key) if path == "$" else f"{path}.{key}"
            yield from _iter_tensors(nested_value, nested_path)
        return
    if isinstance(value, list | tuple):
        for index, nested_value in enumerate(value):
            yield from _iter_tensors(nested_value, f"{path}[{index}]")
        return
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return
    raise BulkValidationError(
        "payload must be a tensor, an immutable scalar, or a nested mapping/list/tuple"
    )


def _move_payload(value: OutputType, device: torch.device, non_blocking: bool) -> OutputType:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=non_blocking and device.type == "cuda")
    if isinstance(value, Mapping):
        return {
            key: _move_payload(nested_value, device, non_blocking)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_move_payload(nested_value, device, non_blocking) for nested_value in value]
    if isinstance(value, tuple):
        nested_values = tuple(
            _move_payload(nested_value, device, non_blocking) for nested_value in value
        )
        if hasattr(value, "_fields"):
            return type(value)(*nested_values)
        return nested_values
    return value


def _shape_matches(actual_shape: tuple[int, ...], expected_shape: Shape) -> bool:
    return len(actual_shape) == len(expected_shape) and all(
        expected_dimension is None or actual_dimension == expected_dimension
        for actual_dimension, expected_dimension in zip(actual_shape, expected_shape, strict=True)
    )


def _same_device(actual: torch.device, expected: torch.device) -> bool:
    return actual.type == expected.type and (
        actual.type != "cuda" or actual.index == expected.index
    )


def _validate_non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a finite non-negative number or none")
    normalized_timeout = float(timeout)
    if not math.isfinite(normalized_timeout) or normalized_timeout < 0:
        raise ValueError("timeout must be a finite non-negative number or none")
    return normalized_timeout


def _validate_bool(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


__all__ = [
    "BulkBackpressureError",
    "BulkClosedError",
    "BulkExecutor",
    "BulkExecutorError",
    "BulkMetrics",
    "BulkResult",
    "BulkValidationError",
]
