from __future__ import annotations

from base64 import b64decode, b64encode
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any


_DIGEST_SIZE_BYTES = 32
_TOKEN_WIDTH_BYTES = 8
_MAX_TOKEN_ID = (1 << (_TOKEN_WIDTH_BYTES * 8)) - 1
_HASH_DOMAIN = b"koemi-bulk-block/v1\x00"
_STORAGE_DOMAIN = b"koemi-bulk-storage/v1\x00"
_PAYLOAD_DOMAIN = b"koemi-bulk-payload/v1\x00"
_FORMAT_VERSION = 1
_ENTRY_PREFIX = "koemi-bulk-block-"
_ENTRY_SUFFIX = ".json"
_ENTRY_FIELDS = frozenset(
    {
        "format_version",
        "namespace",
        "offset",
        "block_size",
        "sequence_length",
        "sequence_digest",
        "sequence",
        "payload_encoding",
        "payload",
        "payload_digest",
        "stored_at",
        "expires_at",
    }
)


def _normalize_namespace(namespace: str) -> str:
    if not isinstance(namespace, str):
        raise TypeError("bulk block namespace must be a string")
    if not namespace.strip():
        raise ValueError("bulk block namespace must not be empty")
    return namespace


def _normalize_offset(offset: int) -> int:
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("bulk block offset must be an integer")
    if offset < 0 or offset > _MAX_TOKEN_ID:
        raise ValueError("bulk block offset is outside the supported range")
    return offset


def _sequence_values(sequence: object) -> Iterable[Any]:
    if isinstance(sequence, (str, bytes, bytearray, Mapping)):
        raise TypeError("bulk block sequence must contain token IDs, not text or a mapping")
    if hasattr(sequence, "detach") and hasattr(sequence, "reshape"):
        detached_sequence = sequence.detach().cpu()
        dimensions = getattr(detached_sequence, "ndim", None)
        shape = getattr(detached_sequence, "shape", None)
        if dimensions == 2:
            if shape is None or shape[0] != 1:
                raise ValueError("bulk block tensor sequence must have shape [1, sequence]")
            detached_sequence = detached_sequence.reshape(-1)
        elif dimensions != 1:
            raise ValueError("bulk block tensor sequence must have one dimension")
        return detached_sequence.tolist()
    return sequence


def _normalize_sequence(sequence: object, block_size: int) -> tuple[int, ...]:
    try:
        values = tuple(_sequence_values(sequence))
    except TypeError as error:
        raise TypeError("bulk block sequence must be iterable") from error
    if len(values) != block_size:
        raise ValueError(f"bulk block sequence must contain exactly {block_size} token IDs")
    normalized_sequence: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("bulk block token IDs must be integers")
        try:
            token_id = operator.index(value)
        except TypeError as error:
            raise TypeError("bulk block token IDs must be integers") from error
        if token_id < 0 or token_id > _MAX_TOKEN_ID:
            raise ValueError("bulk block token ID is outside the supported range")
        normalized_sequence.append(token_id)
    return tuple(normalized_sequence)


def _build_digest_chain(
    namespace: str,
    offset: int,
    sequence: tuple[int, ...],
) -> tuple[str, ...]:
    namespace_bytes = namespace.encode("utf-8")
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE_BYTES)
    digest.update(_HASH_DOMAIN)
    digest.update(len(namespace_bytes).to_bytes(_TOKEN_WIDTH_BYTES, "big"))
    digest.update(namespace_bytes)
    digest.update(offset.to_bytes(_TOKEN_WIDTH_BYTES, "big"))
    chain: list[str] = []
    for position, token_id in enumerate(sequence):
        digest.update(position.to_bytes(_TOKEN_WIDTH_BYTES, "big"))
        digest.update(token_id.to_bytes(_TOKEN_WIDTH_BYTES, "big"))
        chain.append(digest.copy().hexdigest())
    return tuple(chain)


def _sequence_digest(namespace: str, offset: int, sequence: tuple[int, ...]) -> str:
    return _build_digest_chain(namespace, offset, sequence)[-1]


def _storage_digest(namespace: str, offset: int, sequence: tuple[int, ...]) -> str:
    namespace_bytes = namespace.encode("utf-8")
    storage_key = bytearray()
    storage_key.extend(_STORAGE_DOMAIN)
    storage_key.extend(len(namespace_bytes).to_bytes(_TOKEN_WIDTH_BYTES, "big"))
    storage_key.extend(namespace_bytes)
    storage_key.extend(offset.to_bytes(_TOKEN_WIDTH_BYTES, "big"))
    for token_id in sequence:
        storage_key.extend(token_id.to_bytes(_TOKEN_WIDTH_BYTES, "big"))
    return hashlib.blake2b(bytes(storage_key), digest_size=_DIGEST_SIZE_BYTES).hexdigest()


def _payload_digest(payload_encoding: str, payload_bytes: bytes) -> str:
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE_BYTES)
    digest.update(_PAYLOAD_DOMAIN)
    digest.update(payload_encoding.encode("ascii"))
    digest.update(b"\x00")
    digest.update(payload_bytes)
    return digest.hexdigest()


def _validate_json_value(value: Any, path: str, allow_text: bool) -> Any:
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, str):
        if not allow_text:
            raise TypeError(f"{path} raw text requires an explicit payload encoder")
        return value
    if isinstance(value, Mapping):
        encoded_mapping: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            encoded_mapping[key] = _validate_json_value(nested_value, f"{path}.{key}", allow_text)
        return encoded_mapping
    if isinstance(value, (list, tuple)):
        return [
            _validate_json_value(nested_value, f"{path}[{index}]", allow_text)
            for index, nested_value in enumerate(value)
        ]
    raise TypeError(f"{path} requires bytes, JSON data, or an explicit payload encoder")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _encode_payload(
    payload: Any,
    payload_encoder: Callable[[Any], Any] | None,
) -> tuple[str, Any, bytes]:
    if payload_encoder is not None and not callable(payload_encoder):
        raise TypeError("bulk block payload encoder must be callable")
    encoded_payload = payload if payload_encoder is None else payload_encoder(payload)
    if isinstance(encoded_payload, (bytes, bytearray, memoryview)):
        payload_bytes = bytes(encoded_payload)
        return "bytes", b64encode(payload_bytes).decode("ascii"), payload_bytes
    normalized_payload = _validate_json_value(
        encoded_payload,
        "payload",
        allow_text=payload_encoder is not None,
    )
    return "json", normalized_payload, _canonical_json_bytes(normalized_payload)


def _decode_payload(payload_encoding: str, payload_bytes: bytes) -> Any:
    if payload_encoding == "bytes":
        return bytes(payload_bytes)
    if payload_encoding == "json":
        return json.loads(payload_bytes.decode("utf-8"))
    raise ValueError("bulk block payload encoding is invalid")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"bulk block {name} is invalid")
    normalized_value = float(value)
    if not math.isfinite(normalized_value):
        raise ValueError(f"bulk block {name} is invalid")
    return normalized_value


@dataclass(frozen=True)
class BulkBlockKey:
    """Exact identity fields used to route a fixed token block."""

    namespace: str
    offset: int
    sequence_length: int
    sequence_digest: str


@dataclass(frozen=True)
class BulkBlockCandidate:
    """Unapplied block metadata returned after an exact store lookup."""

    namespace: str
    offset: int
    block_size: int
    sequence_length: int
    sequence_digest: str
    sequence: tuple[int, ...]
    expires_at: float
    entry_bytes: int
    source: str
    _payload_encoding: str = field(repr=False, compare=False)
    _payload_bytes: bytes = field(repr=False, compare=False)
    _record_bytes: bytes = field(repr=False, compare=False)

    @property
    def key(self) -> BulkBlockKey:
        return BulkBlockKey(
            namespace=self.namespace,
            offset=self.offset,
            sequence_length=self.sequence_length,
            sequence_digest=self.sequence_digest,
        )


@dataclass(frozen=True)
class ValidatedBulkBlock:
    """Payload released only after the caller proves the supplied sequence matches."""

    candidate: BulkBlockCandidate
    payload: Any


@dataclass(frozen=True)
class BulkBlockStatistics:
    """Immutable hit, storage, capacity, and integrity counters."""

    hits: int
    misses: int
    ram_hits: int
    disk_hits: int
    bytes_read: int
    bytes_written: int
    ram_bytes: int
    disk_bytes: int
    evictions: int
    ram_evictions: int
    disk_evictions: int
    expirations: int
    corruptions: int
    rejected_entries: int
    ram_entries: int
    disk_entries: int

    def to_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "ram_hits": self.ram_hits,
            "disk_hits": self.disk_hits,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "ram_bytes": self.ram_bytes,
            "disk_bytes": self.disk_bytes,
            "evictions": self.evictions,
            "ram_evictions": self.ram_evictions,
            "disk_evictions": self.disk_evictions,
            "expirations": self.expirations,
            "corruptions": self.corruptions,
            "rejected_entries": self.rejected_entries,
            "ram_entries": self.ram_entries,
            "disk_entries": self.disk_entries,
        }


@dataclass(frozen=True)
class _StoredRamBlock:
    candidate: BulkBlockCandidate
    record_bytes: bytes


class BulkBlockStore:
    """Bounded exact fixed-block storage with optional RAM and SSD tiers.

    SSD records are integrity-checked but not encrypted; callers must decide
    whether a payload is suitable for unencrypted local persistence.
    """

    def __init__(
        self,
        block_size: int,
        ram_capacity: int = 128,
        *,
        disk_directory: str | Path | None = None,
        disk_capacity: int = 128,
        ttl_seconds: float = 3600.0,
        max_entry_bytes: int = 64 * 1024 * 1024,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.block_size = self._positive_integer(block_size, "block size")
        self.ram_capacity = self._positive_integer(ram_capacity, "RAM capacity")
        self.disk_capacity = self._positive_integer(disk_capacity, "SSD capacity")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
            raise TypeError("bulk block TTL must be a number")
        if not math.isfinite(float(ttl_seconds)) or ttl_seconds <= 0.0:
            raise ValueError("bulk block TTL must be positive and finite")
        self.ttl_seconds = float(ttl_seconds)
        self.max_entry_bytes = self._positive_integer(max_entry_bytes, "max entry bytes")
        if clock is not None and not callable(clock):
            raise TypeError("bulk block clock must be callable")
        self._clock = time.time if clock is None else clock
        self.disk_directory = self._prepare_disk_directory(disk_directory)
        self._ram_entries: OrderedDict[int, _StoredRamBlock] = OrderedDict()
        self._ram_buckets: dict[BulkBlockKey, list[int]] = {}
        self._next_ram_entry_id = 0
        self._ram_bytes = 0
        self._hits = 0
        self._misses = 0
        self._ram_hits = 0
        self._disk_hits = 0
        self._bytes_read = 0
        self._bytes_written = 0
        self._evictions = 0
        self._ram_evictions = 0
        self._disk_evictions = 0
        self._expirations = 0
        self._corruptions = 0
        self._rejected_entries = 0
        self._lock = threading.RLock()

    @staticmethod
    def _positive_integer(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"bulk block {name} must be at least 1")
        return value

    @staticmethod
    def _prepare_disk_directory(directory: str | Path | None) -> Path | None:
        if directory is None:
            return None
        resolved_directory = Path(directory).expanduser().resolve()
        if resolved_directory.exists() and not resolved_directory.is_dir():
            raise ValueError("bulk block SSD directory must be a directory")
        resolved_directory.mkdir(parents=True, exist_ok=True)
        os.chmod(resolved_directory, 0o700)
        return resolved_directory

    def put(
        self,
        namespace: str,
        offset: int,
        sequence: Iterable[int],
        payload: Any,
        *,
        payload_encoder: Callable[[Any], Any] | None = None,
    ) -> BulkBlockCandidate | None:
        """Store one fixed block and return its candidate, or None if too large.

        The payload is bytes or explicit JSON data; arbitrary values require
        payload_encoder, and an unencoded text payload is rejected.
        """
        normalized_namespace = _normalize_namespace(namespace)
        normalized_offset = _normalize_offset(offset)
        normalized_sequence = _normalize_sequence(sequence, self.block_size)
        payload_encoding, payload_value, payload_bytes = _encode_payload(payload, payload_encoder)
        now = self._now()
        expires_at = now + self.ttl_seconds
        if not math.isfinite(expires_at):
            raise RuntimeError("bulk block expiration time is not finite")
        sequence_digest = _sequence_digest(
            normalized_namespace,
            normalized_offset,
            normalized_sequence,
        )
        record = {
            "format_version": _FORMAT_VERSION,
            "namespace": normalized_namespace,
            "offset": normalized_offset,
            "block_size": self.block_size,
            "sequence_length": self.block_size,
            "sequence_digest": sequence_digest,
            "sequence": list(normalized_sequence),
            "payload_encoding": payload_encoding,
            "payload": payload_value,
            "payload_digest": _payload_digest(payload_encoding, payload_bytes),
            "stored_at": now,
            "expires_at": expires_at,
        }
        record_bytes = _canonical_json_bytes(record)
        with self._lock:
            if len(record_bytes) > self.max_entry_bytes:
                self._rejected_entries += 1
                return None
            self._purge_expired_ram_locked(now)
            self._purge_expired_disk_locked(now)
            candidate = self._candidate_from_record(
                record,
                len(record_bytes),
                source="ram",
                expected_sequence=normalized_sequence,
                record_bytes=record_bytes,
            )
            if self.disk_directory is not None:
                disk_path = self.path_for(candidate)
                self._atomic_write(disk_path, record_bytes)
                self._bytes_written += len(record_bytes)
                self._evict_disk_locked()
            self._store_ram_locked(candidate, record_bytes)
            self._evict_ram_locked()
            return candidate

    def get(
        self,
        namespace: str,
        offset: int,
        sequence: Iterable[int],
    ) -> BulkBlockCandidate | None:
        """Return a candidate only when namespace, offset, digest, and sequence match."""
        normalized_namespace = _normalize_namespace(namespace)
        normalized_offset = _normalize_offset(offset)
        normalized_sequence = _normalize_sequence(sequence, self.block_size)
        key = self._key_for(normalized_namespace, normalized_offset, normalized_sequence)
        with self._lock:
            now = self._now()
            self._purge_expired_ram_locked(now)
            self._purge_expired_disk_locked(now)
            ram_block = self._find_ram_block_locked(key, normalized_sequence)
            if ram_block is not None:
                self._hits += 1
                self._ram_hits += 1
                return ram_block.candidate
            if self.disk_directory is not None:
                disk_path = self.path_for_values(
                    normalized_namespace,
                    normalized_offset,
                    normalized_sequence,
                    key.sequence_digest,
                )
                disk_candidate = self._read_disk_candidate_locked(
                    disk_path,
                    key,
                    normalized_sequence,
                )
                if disk_candidate is not None:
                    self._hits += 1
                    self._disk_hits += 1
                    self._bytes_read += disk_candidate.entry_bytes
                    ram_candidate = self._candidate_with_source(disk_candidate, "ram")
                    self._store_ram_locked(ram_candidate, disk_candidate._record_bytes)
                    self._evict_ram_locked()
                    return disk_candidate
            self._misses += 1
            return None

    def validate_candidate(
        self,
        candidate: BulkBlockCandidate,
        namespace: str,
        offset: int,
        sequence: Iterable[int],
    ) -> ValidatedBulkBlock:
        """Release the encoded payload only after an exact caller-side sequence check."""
        if not isinstance(candidate, BulkBlockCandidate):
            raise TypeError("bulk block candidate has an invalid type")
        normalized_namespace = _normalize_namespace(namespace)
        normalized_offset = _normalize_offset(offset)
        normalized_sequence = _normalize_sequence(sequence, self.block_size)
        expected_key = self._key_for(normalized_namespace, normalized_offset, normalized_sequence)
        with self._lock:
            if candidate.expires_at <= self._now():
                raise ValueError("bulk block candidate has expired")
        if (
            candidate.block_size != self.block_size
            or candidate.key != expected_key
            or candidate.sequence != normalized_sequence
            or candidate.sequence_length != len(normalized_sequence)
        ):
            raise ValueError("bulk block candidate sequence is not exact")
        return ValidatedBulkBlock(
            candidate,
            _decode_payload(candidate._payload_encoding, candidate._payload_bytes),
        )

    def path_for(self, candidate: BulkBlockCandidate) -> Path:
        """Return the generated SSD path after checking it remains inside the tier directory."""
        if not isinstance(candidate, BulkBlockCandidate):
            raise TypeError("bulk block path candidate has an invalid type")
        return self.path_for_values(
            candidate.namespace,
            candidate.offset,
            candidate.sequence,
            candidate.sequence_digest,
        )

    def path_for_values(
        self,
        namespace: str,
        offset: int,
        sequence: Iterable[int],
        sequence_digest: str | None = None,
    ) -> Path:
        """Return a safe deterministic SSD path for normalized key material."""
        if self.disk_directory is None:
            raise RuntimeError("bulk block SSD tier is disabled")
        normalized_namespace = _normalize_namespace(namespace)
        normalized_offset = _normalize_offset(offset)
        normalized_sequence = _normalize_sequence(sequence, self.block_size)
        computed_digest = _sequence_digest(
            normalized_namespace,
            normalized_offset,
            normalized_sequence,
        )
        digest = computed_digest if sequence_digest is None else sequence_digest
        if digest != computed_digest:
            raise ValueError("bulk block sequence digest is invalid")
        namespace_digest = hashlib.blake2b(
            normalized_namespace.encode("utf-8"),
            digest_size=8,
        ).hexdigest()
        storage_digest = _storage_digest(
            normalized_namespace,
            normalized_offset,
            normalized_sequence,
        )
        path = self.disk_directory / (
            f"{_ENTRY_PREFIX}{namespace_digest}-{normalized_offset:016x}-"
            f"{digest}-{storage_digest}{_ENTRY_SUFFIX}"
        )
        return self._allowed_disk_path(path)

    def statistics(self) -> BulkBlockStatistics:
        """Return coherent counters and current bytes for both tiers."""
        with self._lock:
            now = self._now()
            self._purge_expired_ram_locked(now)
            self._purge_expired_disk_locked(now)
            disk_paths = self._disk_paths_locked()
            disk_bytes = sum(path.stat().st_size for path in disk_paths)
            return BulkBlockStatistics(
                hits=self._hits,
                misses=self._misses,
                ram_hits=self._ram_hits,
                disk_hits=self._disk_hits,
                bytes_read=self._bytes_read,
                bytes_written=self._bytes_written,
                ram_bytes=self._ram_bytes,
                disk_bytes=disk_bytes,
                evictions=self._evictions,
                ram_evictions=self._ram_evictions,
                disk_evictions=self._disk_evictions,
                expirations=self._expirations,
                corruptions=self._corruptions,
                rejected_entries=self._rejected_entries,
                ram_entries=len(self._ram_entries),
                disk_entries=len(disk_paths),
            )

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired_ram_locked(self._now())
            return len(self._ram_entries)

    def _now(self) -> float:
        current_time = float(self._clock())
        if not math.isfinite(current_time):
            raise RuntimeError("bulk block clock returned a non-finite value")
        return current_time

    def _key_for(
        self,
        namespace: str,
        offset: int,
        sequence: tuple[int, ...],
    ) -> BulkBlockKey:
        return BulkBlockKey(
            namespace=namespace,
            offset=offset,
            sequence_length=len(sequence),
            sequence_digest=_sequence_digest(namespace, offset, sequence),
        )

    def _candidate_from_record(
        self,
        record: Mapping[str, Any],
        entry_bytes: int,
        *,
        source: str,
        expected_sequence: tuple[int, ...] | None = None,
        record_bytes: bytes,
    ) -> BulkBlockCandidate:
        if not isinstance(record, Mapping) or set(record) != _ENTRY_FIELDS:
            raise ValueError("bulk block record fields are invalid")
        if isinstance(record["format_version"], bool) or record["format_version"] != _FORMAT_VERSION:
            raise ValueError("bulk block record format is invalid")
        namespace = _normalize_namespace(record["namespace"])
        offset = _normalize_offset(record["offset"])
        if (
            isinstance(record["block_size"], bool)
            or not isinstance(record["block_size"], int)
            or record["block_size"] != self.block_size
        ):
            raise ValueError("bulk block record block size is invalid")
        if isinstance(record["sequence"], (str, bytes, bytearray)) or not isinstance(
            record["sequence"], list
        ):
            raise ValueError("bulk block record sequence is invalid")
        sequence = _normalize_sequence(record["sequence"], self.block_size)
        if (
            isinstance(record["sequence_length"], bool)
            or not isinstance(record["sequence_length"], int)
            or record["sequence_length"] != len(sequence)
        ):
            raise ValueError("bulk block record sequence length is invalid")
        sequence_digest = record["sequence_digest"]
        if not isinstance(sequence_digest, str) or sequence_digest != _sequence_digest(
            namespace,
            offset,
            sequence,
        ):
            raise ValueError("bulk block record digest is invalid")
        if expected_sequence is not None and sequence != expected_sequence:
            raise ValueError("bulk block record sequence does not match the lookup")
        payload_encoding = record["payload_encoding"]
        if payload_encoding not in {"bytes", "json"}:
            raise ValueError("bulk block payload encoding is invalid")
        if payload_encoding == "bytes":
            encoded_payload = record["payload"]
            if not isinstance(encoded_payload, str):
                raise ValueError("bulk block bytes payload is invalid")
            try:
                payload_bytes = b64decode(encoded_payload.encode("ascii"), validate=True)
            except (UnicodeEncodeError, ValueError) as error:
                raise ValueError("bulk block bytes payload is invalid") from error
        else:
            normalized_payload = _validate_json_value(record["payload"], "payload", allow_text=True)
            payload_bytes = _canonical_json_bytes(normalized_payload)
        payload_digest = record["payload_digest"]
        if not isinstance(payload_digest, str) or payload_digest != _payload_digest(
            payload_encoding,
            payload_bytes,
        ):
            raise ValueError("bulk block payload digest is invalid")
        _finite_number(record["stored_at"], "stored time")
        expires_at = _finite_number(record["expires_at"], "expiration time")
        if entry_bytes < 1:
            raise ValueError("bulk block record size is invalid")
        return BulkBlockCandidate(
            namespace=namespace,
            offset=offset,
            block_size=self.block_size,
            sequence_length=len(sequence),
            sequence_digest=sequence_digest,
            sequence=sequence,
            expires_at=expires_at,
            entry_bytes=entry_bytes,
            source=source,
            _payload_encoding=payload_encoding,
            _payload_bytes=payload_bytes,
            _record_bytes=record_bytes,
        )

    @staticmethod
    def _candidate_with_source(candidate: BulkBlockCandidate, source: str) -> BulkBlockCandidate:
        return BulkBlockCandidate(
            namespace=candidate.namespace,
            offset=candidate.offset,
            block_size=candidate.block_size,
            sequence_length=candidate.sequence_length,
            sequence_digest=candidate.sequence_digest,
            sequence=candidate.sequence,
            expires_at=candidate.expires_at,
            entry_bytes=candidate.entry_bytes,
            source=source,
            _payload_encoding=candidate._payload_encoding,
            _payload_bytes=candidate._payload_bytes,
            _record_bytes=candidate._record_bytes,
        )

    def _store_ram_locked(self, candidate: BulkBlockCandidate, record_bytes: bytes) -> None:
        key = candidate.key
        bucket = self._ram_buckets.setdefault(key, [])
        for entry_id in bucket:
            stored_block = self._ram_entries.get(entry_id)
            if stored_block is None or stored_block.candidate.sequence != candidate.sequence:
                continue
            self._ram_bytes -= len(stored_block.record_bytes)
            replacement = _StoredRamBlock(candidate, record_bytes)
            self._ram_entries[entry_id] = replacement
            self._ram_entries.move_to_end(entry_id)
            self._ram_bytes += len(record_bytes)
            return
        entry_id = self._next_ram_entry_id
        self._next_ram_entry_id += 1
        self._ram_entries[entry_id] = _StoredRamBlock(candidate, record_bytes)
        bucket.append(entry_id)
        self._ram_bytes += len(record_bytes)

    def _find_ram_block_locked(
        self,
        key: BulkBlockKey,
        sequence: tuple[int, ...],
    ) -> _StoredRamBlock | None:
        for entry_id in tuple(self._ram_buckets.get(key, ())):
            stored_block = self._ram_entries.get(entry_id)
            if stored_block is None:
                continue
            if stored_block.candidate.key == key and stored_block.candidate.sequence == sequence:
                self._ram_entries.move_to_end(entry_id)
                return stored_block
        return None

    def _evict_ram_locked(self) -> None:
        while len(self._ram_entries) > self.ram_capacity:
            oldest_entry_id = next(iter(self._ram_entries))
            self._remove_ram_entry_locked(oldest_entry_id)
            self._evictions += 1
            self._ram_evictions += 1

    def _remove_ram_entry_locked(self, entry_id: int) -> None:
        stored_block = self._ram_entries.pop(entry_id, None)
        if stored_block is None:
            return
        self._ram_bytes -= len(stored_block.record_bytes)
        bucket = self._ram_buckets.get(stored_block.candidate.key)
        if bucket is None:
            return
        bucket.remove(entry_id)
        if not bucket:
            del self._ram_buckets[stored_block.candidate.key]

    def _purge_expired_ram_locked(self, current_time: float) -> None:
        expired_ids = [
            entry_id
            for entry_id, stored_block in self._ram_entries.items()
            if stored_block.candidate.expires_at <= current_time
        ]
        for entry_id in expired_ids:
            self._remove_ram_entry_locked(entry_id)
            self._expirations += 1

    def _allowed_disk_path(self, path: Path) -> Path:
        if self.disk_directory is None:
            raise RuntimeError("bulk block SSD tier is disabled")
        if path.is_symlink():
            raise RuntimeError("bulk block SSD path must not be a symlink")
        resolved_path = path.resolve(strict=False)
        try:
            relative_path = resolved_path.relative_to(self.disk_directory)
        except ValueError as error:
            raise RuntimeError("bulk block SSD path escapes its directory") from error
        if len(relative_path.parts) != 1:
            raise RuntimeError("bulk block SSD path must be a direct child")
        return path

    def _disk_paths_locked(self) -> tuple[Path, ...]:
        if self.disk_directory is None:
            return ()
        paths: list[Path] = []
        for path in self.disk_directory.glob(f"{_ENTRY_PREFIX}*{_ENTRY_SUFFIX}"):
            if path.is_symlink() or not path.is_file():
                continue
            self._allowed_disk_path(path)
            paths.append(path)
        return tuple(paths)

    def _atomic_write(self, path: Path, record_bytes: bytes) -> None:
        if self.disk_directory is None:
            raise RuntimeError("bulk block SSD tier is disabled")
        self._allowed_disk_path(path)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f"{path.name}-",
                suffix=".tmp",
                dir=self.disk_directory,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                self._allowed_disk_path(temporary_path)
                os.chmod(temporary_path, 0o600)
                temporary_file.write(record_bytes)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                self._allowed_disk_path(temporary_path)
                temporary_path.unlink()

    def _read_disk_candidate_locked(
        self,
        path: Path,
        expected_key: BulkBlockKey,
        expected_sequence: tuple[int, ...],
    ) -> BulkBlockCandidate | None:
        self._allowed_disk_path(path)
        if path.is_symlink() or not path.exists() or not path.is_file():
            return None
        try:
            entry_bytes = path.stat().st_size
            if entry_bytes > self.max_entry_bytes:
                raise ValueError("bulk block disk entry exceeds its limit")
            record_bytes = path.read_bytes()
            record = json.loads(record_bytes)
            candidate = self._candidate_from_record(
                record,
                entry_bytes,
                source="disk",
                expected_sequence=expected_sequence,
                record_bytes=record_bytes,
            )
            if candidate.key != expected_key:
                raise ValueError("bulk block disk key does not match the lookup")
            if candidate.expires_at <= self._now():
                self._remove_disk_file(path)
                self._expirations += 1
                return None
            return candidate
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            self._corruptions += 1
            self._remove_disk_file(path)
            return None

    def _remove_disk_file(self, path: Path) -> None:
        self._allowed_disk_path(path)
        if path.exists() and not path.is_symlink():
            path.unlink()

    def _purge_expired_disk_locked(self, current_time: float) -> None:
        for path in self._disk_paths_locked():
            try:
                record = json.loads(path.read_bytes())
                expires_at = _finite_number(record["expires_at"], "expiration time")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue
            if expires_at <= current_time:
                self._remove_disk_file(path)
                self._expirations += 1

    def _evict_disk_locked(self) -> None:
        if self.disk_directory is None:
            return
        disk_paths = list(self._disk_paths_locked())
        disk_paths.sort(key=self._disk_order_key)
        while len(disk_paths) > self.disk_capacity:
            oldest_path = disk_paths.pop(0)
            self._remove_disk_file(oldest_path)
            self._evictions += 1
            self._disk_evictions += 1

    @staticmethod
    def _disk_order_key(path: Path) -> tuple[int, float, str]:
        try:
            record = json.loads(path.read_bytes())
            stored_at = _finite_number(record["stored_at"], "stored time")
            return 1, stored_at, path.name
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            return 0, 0.0, path.name


__all__ = [
    "BulkBlockCandidate",
    "BulkBlockKey",
    "BulkBlockStatistics",
    "BulkBlockStore",
    "ValidatedBulkBlock",
]
