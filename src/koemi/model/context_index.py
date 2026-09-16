from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import operator
import threading
import time
from typing import Any


_DIGEST_SIZE_BYTES = 32
_TOKEN_WIDTH_BYTES = 8
_MAX_TOKEN_ID = (1 << (_TOKEN_WIDTH_BYTES * 8)) - 1
_HASH_DOMAIN = b"koemi-context-index/v1\x00"
_CANDIDATE_FIELDS = frozenset(
    {"format_version", "namespace", "sequence_length", "sequence_digest", "sequence", "payload"}
)


def _normalize_namespace(namespace: str) -> str:
    if not isinstance(namespace, str):
        raise TypeError("context index namespace must be a string")
    if not namespace.strip():
        raise ValueError("context index namespace must not be empty")
    return namespace


def _tensor_values(sequence: object) -> Iterable[Any] | None:
    if not hasattr(sequence, "detach") or not hasattr(sequence, "reshape"):
        return None
    detached_sequence = sequence.detach().cpu()
    dimensions = getattr(detached_sequence, "ndim", None)
    shape = getattr(detached_sequence, "shape", None)
    if dimensions == 2:
        if shape is None or shape[0] != 1:
            raise ValueError("context index tensor sequence must have shape [1, sequence]")
        detached_sequence = detached_sequence.reshape(-1)
    elif dimensions != 1:
        raise ValueError("context index tensor sequence must have one dimension")
    return detached_sequence.tolist()


def _normalize_sequence(sequence: object) -> tuple[int, ...]:
    if isinstance(sequence, (str, bytes, bytearray, Mapping)):
        raise TypeError("context index sequence must contain token IDs, not text or a mapping")
    tensor_values = _tensor_values(sequence)
    values = tensor_values if tensor_values is not None else sequence
    try:
        normalized_values = tuple(values)
    except TypeError as error:
        raise TypeError("context index sequence must be iterable") from error
    if not normalized_values:
        raise ValueError("context index sequence must not be empty")
    normalized_sequence: list[int] = []
    for value in normalized_values:
        if isinstance(value, bool):
            raise TypeError("context index token IDs must be integers")
        try:
            token_id = operator.index(value)
        except TypeError as error:
            raise TypeError("context index token IDs must be integers") from error
        if token_id < 0 or token_id > _MAX_TOKEN_ID:
            raise ValueError("context index token ID is outside the supported range")
        normalized_sequence.append(token_id)
    return tuple(normalized_sequence)


def _build_digest_chain(sequence: tuple[int, ...]) -> tuple[str, ...]:
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE_BYTES)
    digest.update(_HASH_DOMAIN)
    chain: list[str] = []
    for position, token_id in enumerate(sequence):
        digest.update(position.to_bytes(_TOKEN_WIDTH_BYTES, "big"))
        digest.update(token_id.to_bytes(_TOKEN_WIDTH_BYTES, "big"))
        chain.append(digest.copy().hexdigest())
    return tuple(chain)


def _sequence_digest(sequence: tuple[int, ...]) -> str:
    return _build_digest_chain(sequence)[-1]


def _encode_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        encoded_mapping: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            encoded_mapping[key] = _encode_json_value(nested_value, f"{path}.{key}")
        return encoded_mapping
    if isinstance(value, (list, tuple)):
        return [_encode_json_value(nested_value, f"{path}[{index}]") for index, nested_value in enumerate(value)]
    raise TypeError(f"{path} requires an explicit JSON payload encoder")


def _reject_raw_text(value: Any, path: str) -> None:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{path} must not contain raw text")
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            _reject_raw_text(nested_value, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            _reject_raw_text(nested_value, f"{path}[{index}]")


@dataclass(frozen=True)
class ContextCandidate:
    """Exact token-prefix identity and caller-owned model-state payload."""

    namespace: str
    sequence_length: int
    sequence_digest: str
    sequence: tuple[int, ...]
    payload: Any

    def __post_init__(self) -> None:
        normalized_namespace = _normalize_namespace(self.namespace)
        normalized_sequence = _normalize_sequence(self.sequence)
        _reject_raw_text(self.payload, "payload")
        if isinstance(self.sequence_length, bool) or not isinstance(self.sequence_length, int):
            raise TypeError("context index sequence length must be an integer")
        if self.sequence_length != len(normalized_sequence):
            raise ValueError("context index sequence length does not match the sequence")
        if not isinstance(self.sequence_digest, str) or self.sequence_digest != _sequence_digest(normalized_sequence):
            raise ValueError("context index sequence hash is invalid")
        object.__setattr__(self, "namespace", normalized_namespace)
        object.__setattr__(self, "sequence", normalized_sequence)

    def to_record(self, payload_encoder: Callable[[Any], Any] | None = None) -> dict[str, Any]:
        """Return an explicit JSON-compatible record without using object deserialization."""
        if payload_encoder is not None and not callable(payload_encoder):
            raise TypeError("context index payload encoder must be callable")
        encoded_payload = self.payload if payload_encoder is None else payload_encoder(self.payload)
        _reject_raw_text(encoded_payload, "payload")
        return {
            "format_version": 1,
            "namespace": self.namespace,
            "sequence_length": self.sequence_length,
            "sequence_digest": self.sequence_digest,
            "sequence": list(self.sequence),
            "payload": _encode_json_value(encoded_payload, "payload"),
        }

    def to_json(self, payload_encoder: Callable[[Any], Any] | None = None) -> bytes:
        """Serialize this candidate as UTF-8 JSON with an explicit data schema."""
        return json.dumps(
            self.to_record(payload_encoder),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, Any],
        payload_decoder: Callable[[Any], Any] | None = None,
    ) -> ContextCandidate:
        """Validate and construct a candidate from explicit data."""
        if not isinstance(record, Mapping) or set(record) != _CANDIDATE_FIELDS:
            raise ValueError("context index candidate record fields are invalid")
        if isinstance(record["format_version"], bool) or record["format_version"] != 1:
            raise ValueError("context index candidate format is invalid")
        if not isinstance(record["sequence"], list):
            raise ValueError("context index candidate sequence must be a list")
        encoded_payload = _encode_json_value(record["payload"], "payload")
        if payload_decoder is not None and not callable(payload_decoder):
            raise TypeError("context index payload decoder must be callable")
        payload = encoded_payload if payload_decoder is None else payload_decoder(encoded_payload)
        return cls(
            namespace=record["namespace"],
            sequence_length=record["sequence_length"],
            sequence_digest=record["sequence_digest"],
            sequence=tuple(record["sequence"]),
            payload=payload,
        )

    @classmethod
    def from_json(
        cls,
        serialized: str | bytes | bytearray,
        payload_decoder: Callable[[Any], Any] | None = None,
    ) -> ContextCandidate:
        """Deserialize and validate a candidate from UTF-8 JSON only."""
        if not isinstance(serialized, (str, bytes, bytearray)):
            raise TypeError("context index candidate JSON must be text or bytes")
        try:
            record = json.loads(serialized)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("context index candidate JSON is invalid") from error
        return cls.from_record(record, payload_decoder)


@dataclass(frozen=True)
class ContextIndexStatistics:
    """Immutable counters collected by a context index."""

    hits: int
    misses: int
    candidates: int
    evictions: int
    expirations: int
    entries: int


@dataclass(frozen=True)
class _StoredContext:
    candidate: ContextCandidate
    expires_at: float


class ContextIndex:
    """Bounded, exact, namespace-aware in-memory prefix index."""

    def __init__(
        self,
        capacity: int = 128,
        ttl_seconds: float = 3600.0,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("context index capacity must be at least 1")
        if not isinstance(ttl_seconds, (int, float)) or isinstance(ttl_seconds, bool):
            raise TypeError("context index TTL must be a number")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0.0:
            raise ValueError("context index TTL must be positive and finite")
        if clock is not None and not callable(clock):
            raise TypeError("context index clock must be callable")
        self.capacity = capacity
        self.ttl_seconds = float(ttl_seconds)
        self._clock = time.monotonic if clock is None else clock
        self._entries: OrderedDict[int, _StoredContext] = OrderedDict()
        self._buckets: dict[tuple[str, int, str], list[int]] = {}
        self._next_entry_id = 0
        self._hits = 0
        self._misses = 0
        self._candidates = 0
        self._evictions = 0
        self._expirations = 0
        self._lock = threading.RLock()

    def put(self, namespace: str, sequence: Iterable[int], payload: Any) -> ContextCandidate:
        """Index an exact token sequence and return its validated candidate."""
        normalized_namespace = _normalize_namespace(namespace)
        normalized_sequence = _normalize_sequence(sequence)
        candidate = ContextCandidate(
            namespace=normalized_namespace,
            sequence_length=len(normalized_sequence),
            sequence_digest=_sequence_digest(normalized_sequence),
            sequence=normalized_sequence,
            payload=payload,
        )
        return self.put_candidate(candidate)

    def put_candidate(self, candidate: ContextCandidate) -> ContextCandidate:
        """Index a candidate after rechecking its namespace, length, and hash."""
        if not isinstance(candidate, ContextCandidate):
            raise TypeError("context index candidate has an invalid type")
        validated_candidate = ContextCandidate(
            namespace=candidate.namespace,
            sequence_length=candidate.sequence_length,
            sequence_digest=candidate.sequence_digest,
            sequence=candidate.sequence,
            payload=candidate.payload,
        )
        with self._lock:
            now = self._now()
            self._purge_expired_locked(now)
            bucket_key = self._bucket_key(validated_candidate)
            bucket = self._buckets.setdefault(bucket_key, [])
            for entry_id in bucket:
                stored_context = self._entries.get(entry_id)
                if stored_context is not None and stored_context.candidate.sequence == validated_candidate.sequence:
                    self._entries[entry_id] = _StoredContext(
                        candidate=validated_candidate,
                        expires_at=now + self.ttl_seconds,
                    )
                    self._entries.move_to_end(entry_id)
                    return validated_candidate
            entry_id = self._next_entry_id
            self._next_entry_id += 1
            self._entries[entry_id] = _StoredContext(
                candidate=validated_candidate,
                expires_at=now + self.ttl_seconds,
            )
            bucket.append(entry_id)
            self._evict_oldest_locked()
        return validated_candidate

    def get_longest_prefix(self, namespace: str, sequence: Iterable[int]) -> ContextCandidate | None:
        """Return the longest stored prefix proven equal to the supplied token IDs."""
        normalized_namespace = _normalize_namespace(namespace)
        normalized_sequence = _normalize_sequence(sequence)
        digest_chain = _build_digest_chain(normalized_sequence)
        with self._lock:
            self._purge_expired_locked(self._now())
            for prefix_length in range(len(normalized_sequence), 0, -1):
                digest = digest_chain[prefix_length - 1]
                bucket_key = (normalized_namespace, prefix_length, digest)
                for entry_id in tuple(self._buckets.get(bucket_key, ())):
                    stored_context = self._entries.get(entry_id)
                    if stored_context is None:
                        continue
                    self._candidates += 1
                    candidate = stored_context.candidate
                    if (
                        candidate.namespace == normalized_namespace
                        and candidate.sequence_length == prefix_length
                        and candidate.sequence_digest == digest
                        and candidate.sequence == normalized_sequence[:prefix_length]
                    ):
                        self._entries.move_to_end(entry_id)
                        self._hits += 1
                        return candidate
            self._misses += 1
        return None

    def lookup(self, namespace: str, sequence: Iterable[int]) -> ContextCandidate | None:
        """Return the longest exact prefix candidate for the supplied token IDs."""
        return self.get_longest_prefix(namespace, sequence)

    def clear_namespace(self, namespace: str) -> int:
        """Remove only entries owned by the supplied namespace and return its count."""
        normalized_namespace = _normalize_namespace(namespace)
        with self._lock:
            self._purge_expired_locked(self._now())
            entry_ids = [
                entry_id
                for entry_id, stored_context in self._entries.items()
                if stored_context.candidate.namespace == normalized_namespace
            ]
            for entry_id in entry_ids:
                self._remove_entry_locked(entry_id)
            return len(entry_ids)

    def statistics(self) -> ContextIndexStatistics:
        """Return a snapshot of lookup, candidate, TTL, and eviction metrics."""
        with self._lock:
            self._purge_expired_locked(self._now())
            return ContextIndexStatistics(
                hits=self._hits,
                misses=self._misses,
                candidates=self._candidates,
                evictions=self._evictions,
                expirations=self._expirations,
                entries=len(self._entries),
            )

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired_locked(self._now())
            return len(self._entries)

    def _now(self) -> float:
        current_time = float(self._clock())
        if not math.isfinite(current_time):
            raise RuntimeError("context index clock returned a non-finite value")
        return current_time

    @staticmethod
    def _bucket_key(candidate: ContextCandidate) -> tuple[str, int, str]:
        return candidate.namespace, candidate.sequence_length, candidate.sequence_digest

    def _purge_expired_locked(self, current_time: float) -> None:
        expired_ids = [
            entry_id
            for entry_id, stored_context in self._entries.items()
            if stored_context.expires_at <= current_time
        ]
        for entry_id in expired_ids:
            self._remove_entry_locked(entry_id)
            self._expirations += 1

    def _evict_oldest_locked(self) -> None:
        while len(self._entries) > self.capacity:
            oldest_entry_id = next(iter(self._entries))
            self._remove_entry_locked(oldest_entry_id)
            self._evictions += 1

    def _remove_entry_locked(self, entry_id: int) -> None:
        stored_context = self._entries.pop(entry_id, None)
        if stored_context is None:
            return
        bucket_key = self._bucket_key(stored_context.candidate)
        bucket = self._buckets.get(bucket_key)
        if bucket is None:
            return
        bucket.remove(entry_id)
        if not bucket:
            del self._buckets[bucket_key]
