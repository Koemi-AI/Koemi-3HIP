from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from koemi.model.state import KoemiState


@dataclass(frozen=True)
class CacheStatistics:
    hits: int
    misses: int
    evictions: int
    expirations: int = 0
    deletions: int = 0
    prefix_hits: int = 0
    prefix_misses: int = 0
    prefix_tokens_reused: int = 0


@dataclass(frozen=True)
class CachedMapping:
    logits: Tensor
    state: KoemiState
    surprise_values: Tensor
    expert_indices: Tensor
    active_expert_indices: Tensor | None
    valid_positions: Tensor
    token_count: int


@dataclass(frozen=True)
class CachedPrefixState:
    last_logits: Tensor
    state: KoemiState


class WarmTokenCache:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("cache capacity must be at least 1")
        self.capacity = capacity
        self._entries: OrderedDict[int, Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def lookup(self, token_id: int, device: torch.device, dtype: torch.dtype) -> Tensor | None:
        cached_embedding = self._entries.get(token_id)
        if cached_embedding is None or cached_embedding.device != device or cached_embedding.dtype != dtype:
            self._misses += 1
            return None
        self._entries.move_to_end(token_id)
        self._hits += 1
        return cached_embedding

    def store(self, token_id: int, embedding: Tensor) -> None:
        self._entries[token_id] = embedding.detach()
        self._entries.move_to_end(token_id)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
            self._evictions += 1

    def embeddings(self, token_ids: Tensor, embedding_table: torch.nn.Embedding) -> tuple[Tensor, int, int]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if embedding_table.training:
            raise RuntimeError("warm token cache is only valid while the model is in evaluation mode")
        device = token_ids.device
        dtype = embedding_table.weight.dtype
        flattened_ids = token_ids.reshape(-1)
        unique_ids = torch.unique(flattened_ids).tolist()
        resolved_embeddings: dict[int, Tensor] = {}
        hits_before = self._hits
        misses_before = self._misses
        missing_ids: list[int] = []
        for raw_token_id in unique_ids:
            token_id = int(raw_token_id)
            cached_embedding = self.lookup(token_id, device, dtype)
            if cached_embedding is None:
                missing_ids.append(token_id)
            else:
                resolved_embeddings[token_id] = cached_embedding
        if missing_ids:
            missing_tensor = torch.tensor(missing_ids, device=device, dtype=torch.long)
            missing_embeddings = embedding_table(missing_tensor)
            for index, token_id in enumerate(missing_ids):
                embedding = missing_embeddings[index].detach()
                self.store(token_id, embedding)
                resolved_embeddings[token_id] = embedding
        stacked_embeddings = torch.stack(
            [resolved_embeddings[int(token_id)] for token_id in flattened_ids.tolist()], dim=0
        )
        return stacked_embeddings.reshape(*token_ids.shape, -1), self._hits - hits_before, self._misses - misses_before

    def clear(self) -> None:
        self._entries.clear()

    def statistics(self) -> CacheStatistics:
        return CacheStatistics(self._hits, self._misses, self._evictions)

    def __len__(self) -> int:
        return len(self._entries)


class DiskMappingCache:
    def __init__(
        self,
        directory: str | Path,
        capacity: int = 128,
        namespace: str | None = None,
        max_entry_bytes: int = 64 * 1024 * 1024,
        ttl_seconds: float = 3600.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("disk cache capacity must be at least 1")
        if max_entry_bytes < 1:
            raise ValueError("disk cache max entry bytes must be at least 1")
        if namespace is None or not namespace.strip():
            raise ValueError("disk cache namespace is required")
        if ttl_seconds <= 0.0:
            raise ValueError("disk cache TTL must be positive")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.namespace = namespace
        self._namespace_digest = hashlib.blake2b(namespace.encode("utf-8"), digest_size=8).hexdigest()
        self.max_entry_bytes = max_entry_bytes
        self.ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        self._deletions = 0
        self._prefix_hits = 0
        self._prefix_misses = 0
        self._prefix_tokens_reused = 0

    def get(self, input_ids: Tensor, device: torch.device) -> CachedMapping | None:
        cache_path = self.path_for(input_ids)
        if not cache_path.exists() or not cache_path.is_file():
            self._misses += 1
            return None
        if self.is_expired(cache_path):
            cache_path.unlink(missing_ok=True)
            self._expirations += 1
            self._misses += 1
            return None
        if cache_path.stat().st_size > self.max_entry_bytes:
            self._misses += 1
            return None
        try:
            payload = torch.load(cache_path, map_location=device, weights_only=True)
            cached_mapping = self.validate_payload(payload)
            os.utime(cache_path, None)
        except (OSError, RuntimeError, ValueError, TypeError, EOFError, IndexError, KeyError, pickle.UnpicklingError):
            self._misses += 1
            return None
        self._hits += 1
        return cached_mapping

    def put(self, input_ids: Tensor, mapping: CachedMapping) -> None:
        if self.estimate_mapping_bytes(mapping) > self.max_entry_bytes:
            return
        cache_path = self.path_for(input_ids)
        payload = {
            "format_version": 2,
            "logits": mapping.logits.detach().cpu(),
            "working_state": mapping.state.working_state.detach().cpu(),
            "memory_basis": mapping.state.memory_basis.detach().cpu(),
            "memory_normalizer": mapping.state.memory_normalizer.detach().cpu(),
            "refine_basis": mapping.state.refine_basis.detach().cpu(),
            "refine_normalizer": mapping.state.refine_normalizer.detach().cpu(),
            "local_keys": mapping.state.local_keys.detach().cpu(),
            "local_values": mapping.state.local_values.detach().cpu(),
            "local_valid": mapping.state.local_valid.detach().cpu(),
            "salient_keys": mapping.state.salient_keys.detach().cpu(),
            "salient_values": mapping.state.salient_values.detach().cpu(),
            "salient_valid": mapping.state.salient_valid.detach().cpu(),
            "last_token_ids": mapping.state.last_token_ids.detach().cpu(),
            "step_index": mapping.state.step_index,
            "surprise_values": mapping.surprise_values.detach().cpu(),
            "expert_indices": mapping.expert_indices.detach().cpu(),
            "active_expert_indices": (
                mapping.active_expert_indices.detach().cpu()
                if mapping.active_expert_indices is not None
                else None
            ),
            "valid_positions": mapping.valid_positions.detach().cpu(),
            "token_count": mapping.token_count,
        }
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f"{cache_path.stem}-", suffix=".tmp", dir=self.directory, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            torch.save(payload, temporary_path)
            if temporary_path.stat().st_size > self.max_entry_bytes:
                return
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self.purge_expired()
        self.evict_old_entries()

    def estimate_mapping_bytes(self, mapping: CachedMapping) -> int:
        tensors = (
            mapping.logits,
            mapping.state.working_state,
            mapping.state.memory_basis,
            mapping.state.memory_normalizer,
            mapping.state.refine_basis,
            mapping.state.refine_normalizer,
            mapping.state.local_keys,
            mapping.state.local_values,
            mapping.state.local_valid,
            mapping.state.salient_keys,
            mapping.state.salient_values,
            mapping.state.salient_valid,
            mapping.state.last_token_ids,
            mapping.surprise_values,
            mapping.expert_indices,
            mapping.valid_positions,
        )
        if mapping.active_expert_indices is not None:
            tensors = (*tensors, mapping.active_expert_indices)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def path_for(self, input_ids: Tensor) -> Path:
        token_bytes = repr(
            (self.namespace, tuple(int(token_id) for token_id in input_ids.detach().cpu().reshape(-1).tolist()))
        ).encode()
        digest = hashlib.blake2b(token_bytes, digest_size=20).hexdigest()
        return self.directory / f"koemi-mapping-{self._namespace_digest}-{digest}.pt"

    def prefix_path_for(self, input_ids: Tensor) -> Path:
        prefix_paths = self.prefix_paths(input_ids)
        if not prefix_paths:
            raise ValueError("prefix cache input must not be empty")
        return prefix_paths[-1]

    def get_longest_prefix(
        self,
        input_ids: Tensor,
        device: torch.device,
    ) -> tuple[int, CachedPrefixState] | None:
        prefix_paths = self.prefix_paths(input_ids)
        for prefix_length in range(len(prefix_paths), 0, -1):
            cache_path = prefix_paths[prefix_length - 1]
            if not cache_path.exists() or not cache_path.is_file():
                continue
            if self.is_expired(cache_path):
                cache_path.unlink(missing_ok=True)
                self._expirations += 1
                continue
            if cache_path.stat().st_size > self.max_entry_bytes:
                continue
            try:
                payload = torch.load(cache_path, map_location=device, weights_only=True)
                cached_prefix = self.validate_prefix_payload(payload, prefix_length)
                os.utime(cache_path, None)
            except (OSError, RuntimeError, ValueError, TypeError, EOFError, IndexError, KeyError, pickle.UnpicklingError):
                continue
            self._prefix_hits += 1
            self._prefix_tokens_reused += prefix_length
            return prefix_length, cached_prefix
        self._prefix_misses += 1
        return None

    def put_prefix(self, input_ids: Tensor, cached_prefix: CachedPrefixState) -> None:
        if self.estimate_prefix_bytes(cached_prefix) > self.max_entry_bytes:
            return
        cache_path = self.prefix_path_for(input_ids)
        payload = {
            "format_version": 1,
            "entry_type": "prefix_state",
            "prefix_length": input_ids.shape[1],
            "last_logits": cached_prefix.last_logits.detach().cpu(),
            "working_state": cached_prefix.state.working_state.detach().cpu(),
            "memory_basis": cached_prefix.state.memory_basis.detach().cpu(),
            "memory_normalizer": cached_prefix.state.memory_normalizer.detach().cpu(),
            "refine_basis": cached_prefix.state.refine_basis.detach().cpu(),
            "refine_normalizer": cached_prefix.state.refine_normalizer.detach().cpu(),
            "local_keys": cached_prefix.state.local_keys.detach().cpu(),
            "local_values": cached_prefix.state.local_values.detach().cpu(),
            "local_valid": cached_prefix.state.local_valid.detach().cpu(),
            "salient_keys": cached_prefix.state.salient_keys.detach().cpu(),
            "salient_values": cached_prefix.state.salient_values.detach().cpu(),
            "salient_valid": cached_prefix.state.salient_valid.detach().cpu(),
            "last_token_ids": cached_prefix.state.last_token_ids.detach().cpu(),
            "step_index": cached_prefix.state.step_index,
        }
        self.write_payload(cache_path, payload)
        self.purge_expired()
        self.evict_old_entries()

    def estimate_prefix_bytes(self, cached_prefix: CachedPrefixState) -> int:
        tensors = (
            cached_prefix.last_logits,
            cached_prefix.state.working_state,
            cached_prefix.state.memory_basis,
            cached_prefix.state.memory_normalizer,
            cached_prefix.state.refine_basis,
            cached_prefix.state.refine_normalizer,
            cached_prefix.state.local_keys,
            cached_prefix.state.local_values,
            cached_prefix.state.local_valid,
            cached_prefix.state.salient_keys,
            cached_prefix.state.salient_values,
            cached_prefix.state.salient_valid,
            cached_prefix.state.last_token_ids,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def prefix_paths(self, input_ids: Tensor) -> tuple[Path, ...]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("prefix cache requires input_ids with shape [1, sequence]")
        digest = hashlib.blake2b(self.namespace.encode("utf-8"), digest_size=20).digest()
        paths = []
        for prefix_length, raw_token_id in enumerate(input_ids.detach().cpu().reshape(-1).tolist(), start=1):
            token_id = int(raw_token_id)
            if not 0 <= token_id <= 0xFFFF:
                raise ValueError("prefix cache token id is outside the supported range")
            digest = hashlib.blake2b(digest + token_id.to_bytes(2, "little"), digest_size=20).digest()
            paths.append(
                self.directory
                / f"koemi-prefix-{self._namespace_digest}-{prefix_length}-{digest.hex()}.pt"
            )
        return tuple(paths)

    def evict_old_entries(self) -> None:
        cache_files = sorted(
            self.namespace_files(),
            key=lambda path: path.stat().st_atime,
        )
        while len(cache_files) > self.capacity:
            oldest_path = cache_files.pop(0)
            oldest_path.unlink()
            self._evictions += 1

    def is_expired(self, cache_path: Path, current_time: float | None = None) -> bool:
        now = time.time() if current_time is None else current_time
        return now - cache_path.stat().st_mtime > self.ttl_seconds

    def purge_expired(self) -> int:
        expired_count = 0
        for cache_path in self.namespace_files():
            if self.is_expired(cache_path):
                cache_path.unlink(missing_ok=True)
                expired_count += 1
        self._expirations += expired_count
        return expired_count

    def delete(self, input_ids: Tensor) -> bool:
        cache_path = self.path_for(input_ids)
        if not cache_path.exists():
            return False
        cache_path.unlink()
        self._deletions += 1
        return True

    def delete_prefix(self, input_ids: Tensor) -> bool:
        cache_path = self.prefix_path_for(input_ids)
        if not cache_path.exists():
            return False
        cache_path.unlink()
        self._deletions += 1
        return True

    def clear(self) -> int:
        deletion_count = 0
        for cache_path in self.namespace_files():
            cache_path.unlink(missing_ok=True)
            deletion_count += 1
        self._deletions += deletion_count
        return deletion_count

    def namespace_files(self) -> tuple[Path, ...]:
        mapping_files = self.directory.glob(f"koemi-mapping-{self._namespace_digest}-*.pt")
        prefix_files = self.directory.glob(f"koemi-prefix-{self._namespace_digest}-*.pt")
        return tuple(path for path in (*mapping_files, *prefix_files) if path.is_file())

    def validate_payload(self, payload: Any) -> CachedMapping:
        if not isinstance(payload, dict) or payload.get("format_version") not in {1, 2}:
            raise ValueError("disk cache entry format is invalid")
        tensor_names = (
            "logits",
            "working_state",
            "memory_basis",
            "memory_normalizer",
            "refine_basis",
            "refine_normalizer",
            "local_keys",
            "local_values",
            "local_valid",
            "last_token_ids",
            "surprise_values",
            "expert_indices",
            "valid_positions",
        )
        if not all(isinstance(payload.get(name), Tensor) for name in tensor_names):
            raise ValueError("disk cache entry tensors are invalid")
        if not isinstance(payload.get("step_index"), int) or not isinstance(payload.get("token_count"), int):
            raise ValueError("disk cache entry counters are invalid")
        if payload["logits"].ndim != 3 or payload["valid_positions"].ndim != 2:
            raise ValueError("disk cache entry output shapes are invalid")
        output_shape = payload["valid_positions"].shape
        if payload["logits"].shape[:2] != output_shape:
            raise ValueError("disk cache entry output lengths do not match")
        if payload["surprise_values"].shape != output_shape or payload["expert_indices"].shape != output_shape:
            raise ValueError("disk cache entry metrics do not match output")
        if payload["working_state"].ndim != 2 or payload["working_state"].shape[0] != output_shape[0]:
            raise ValueError("disk cache entry working state is invalid")
        if payload["memory_basis"].ndim != 3 or payload["memory_normalizer"].ndim != 2:
            raise ValueError("disk cache entry associative state is invalid")
        if payload["refine_basis"].shape != payload["memory_basis"].shape:
            raise ValueError("disk cache entry refine basis is invalid")
        if payload["refine_normalizer"].shape != payload["memory_normalizer"].shape:
            raise ValueError("disk cache entry refine normalizer is invalid")
        if payload["local_keys"].ndim != 3 or payload["local_values"].shape != payload["local_keys"].shape:
            raise ValueError("disk cache entry local state is invalid")
        if payload["local_valid"].shape != payload["local_keys"].shape[:2]:
            raise ValueError("disk cache entry local mask is invalid")
        if payload["last_token_ids"].shape != (output_shape[0],):
            raise ValueError("disk cache entry context tokens are invalid")
        if payload["token_count"] < 0 or payload["token_count"] > int(payload["valid_positions"].sum()):
            raise ValueError("disk cache entry token count is invalid")
        if payload["format_version"] == 1:
            salient_keys = payload["local_keys"].new_empty(payload["local_keys"].shape[0], 0, payload["local_keys"].shape[2])
            salient_values = salient_keys.clone()
            salient_valid = payload["local_valid"].new_empty(payload["local_valid"].shape[0], 0)
        else:
            if not all(isinstance(payload.get(name), Tensor) for name in ("salient_keys", "salient_values", "salient_valid")):
                raise ValueError("disk cache entry salient state is invalid")
            salient_keys = payload["salient_keys"]
            salient_values = payload["salient_values"]
            salient_valid = payload["salient_valid"]
            if salient_keys.ndim != 3 or salient_values.shape != salient_keys.shape:
                raise ValueError("disk cache entry salient values are invalid")
            if salient_valid.shape != salient_keys.shape[:2]:
                raise ValueError("disk cache entry salient mask is invalid")
        active_expert_indices = payload.get("active_expert_indices")
        if active_expert_indices is not None and not isinstance(active_expert_indices, Tensor):
            raise ValueError("disk cache expert assignments are invalid")
        if active_expert_indices is not None and active_expert_indices.shape[:2] != output_shape:
            raise ValueError("disk cache expert assignment shape is invalid")
        state = KoemiState(
            working_state=payload["working_state"],
            memory_basis=payload["memory_basis"],
            memory_normalizer=payload["memory_normalizer"],
            refine_basis=payload["refine_basis"],
            refine_normalizer=payload["refine_normalizer"],
            local_keys=payload["local_keys"],
            local_values=payload["local_values"],
            local_valid=payload["local_valid"].to(dtype=torch.bool),
            salient_keys=salient_keys,
            salient_values=salient_values,
            salient_valid=salient_valid.to(dtype=torch.bool),
            last_token_ids=payload["last_token_ids"].to(dtype=torch.long),
            step_index=int(payload["step_index"]),
        )
        return CachedMapping(
            logits=payload["logits"],
            state=state,
            surprise_values=payload["surprise_values"],
            expert_indices=payload["expert_indices"],
            valid_positions=payload["valid_positions"].to(dtype=torch.bool),
            token_count=int(payload["token_count"]),
            active_expert_indices=active_expert_indices,
        )

    def validate_prefix_payload(self, payload: Any, expected_prefix_length: int) -> CachedPrefixState:
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("prefix cache entry format is invalid")
        if payload.get("entry_type") != "prefix_state" or payload.get("prefix_length") != expected_prefix_length:
            raise ValueError("prefix cache entry identity is invalid")
        tensor_names = (
            "last_logits",
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
        if not all(isinstance(payload.get(name), Tensor) for name in tensor_names):
            raise ValueError("prefix cache entry tensors are invalid")
        if payload["last_logits"].ndim != 2 or payload["last_logits"].shape[0] != 1:
            raise ValueError("prefix cache logits are invalid")
        if not isinstance(payload.get("step_index"), int) or payload["step_index"] != expected_prefix_length:
            raise ValueError("prefix cache step index is invalid")
        batch_size = payload["last_logits"].shape[0]
        if payload["working_state"].ndim != 2 or payload["working_state"].shape[0] != batch_size:
            raise ValueError("prefix cache working state is invalid")
        if payload["memory_basis"].ndim != 3 or payload["memory_basis"].shape[0] != batch_size:
            raise ValueError("prefix cache associative basis is invalid")
        if payload["memory_normalizer"].shape != (batch_size, payload["memory_basis"].shape[2]):
            raise ValueError("prefix cache associative normalizer is invalid")
        if payload["refine_basis"].shape != payload["memory_basis"].shape:
            raise ValueError("prefix cache refine basis is invalid")
        if payload["refine_normalizer"].shape != payload["memory_normalizer"].shape:
            raise ValueError("prefix cache refine normalizer is invalid")
        for key_name, value_name, valid_name in (
            ("local_keys", "local_values", "local_valid"),
            ("salient_keys", "salient_values", "salient_valid"),
        ):
            if payload[key_name].ndim != 3 or payload[value_name].shape != payload[key_name].shape:
                raise ValueError(f"prefix cache {key_name} are invalid")
            if payload[valid_name].shape != payload[key_name].shape[:2]:
                raise ValueError(f"prefix cache {valid_name} is invalid")
        if payload["last_token_ids"].shape != (batch_size,):
            raise ValueError("prefix cache context tokens are invalid")
        state = KoemiState(
            working_state=payload["working_state"],
            memory_basis=payload["memory_basis"],
            memory_normalizer=payload["memory_normalizer"],
            refine_basis=payload["refine_basis"],
            refine_normalizer=payload["refine_normalizer"],
            local_keys=payload["local_keys"],
            local_values=payload["local_values"],
            local_valid=payload["local_valid"].to(dtype=torch.bool),
            salient_keys=payload["salient_keys"],
            salient_values=payload["salient_values"],
            salient_valid=payload["salient_valid"].to(dtype=torch.bool),
            last_token_ids=payload["last_token_ids"].to(dtype=torch.long),
            step_index=payload["step_index"],
        )
        return CachedPrefixState(payload["last_logits"], state)

    def write_payload(self, cache_path: Path, payload: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f"{cache_path.stem}-", suffix=".tmp", dir=self.directory, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            torch.save(payload, temporary_path)
            if temporary_path.stat().st_size <= self.max_entry_bytes:
                os.replace(temporary_path, cache_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def statistics(self) -> CacheStatistics:
        return CacheStatistics(
            self._hits,
            self._misses,
            self._evictions,
            self._expirations,
            self._deletions,
            self._prefix_hits,
            self._prefix_misses,
            self._prefix_tokens_reused,
        )
