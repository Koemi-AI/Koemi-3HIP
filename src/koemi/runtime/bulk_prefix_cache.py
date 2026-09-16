from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
import hashlib
import operator
import pickle
from pathlib import Path

import torch
from torch import Tensor

from koemi.model.cache import (
    CachedPrefixState,
    build_prefix_state_payload,
    validate_prefix_state_payload,
)
from koemi.runtime.bulk_blocks import BulkBlockStatistics, BulkBlockStore


_PREFIX_DIGEST_DOMAIN = b"koemi-bulk-prefix/v1\x00"
_DIGEST_SIZE_BYTES = 32
_TOKEN_WIDTH_BYTES = 8
_MAX_TOKEN_ID = (1 << (_TOKEN_WIDTH_BYTES * 8)) - 1


class BulkPrefixCache:
    """Reuse exact generation prefixes through fixed-token RAM/SSD blocks.

    The namespace must identify the model checkpoint and caller isolation scope.
    A hit returns a validated recurrent state after the longest cached block;
    the generation caller remains responsible for evaluating the uncached suffix.
    """

    def __init__(
        self,
        directory: str | Path | None,
        *,
        namespace: str,
        block_size: int,
        ram_capacity: int = 128,
        disk_capacity: int = 128,
        max_entry_bytes: int = 64 * 1024 * 1024,
        ttl_seconds: float = 3600.0,
    ) -> None:
        if not isinstance(namespace, str):
            raise TypeError("bulk prefix namespace must be a string")
        if not namespace.strip():
            raise ValueError("bulk prefix namespace must not be empty")
        self.namespace = namespace
        self.store = BulkBlockStore(
            block_size,
            ram_capacity=ram_capacity,
            disk_directory=directory,
            disk_capacity=disk_capacity,
            max_entry_bytes=max_entry_bytes,
            ttl_seconds=ttl_seconds,
        )

    @property
    def block_size(self) -> int:
        return self.store.block_size

    def get_longest_prefix(
        self,
        input_ids: Tensor,
        device: torch.device,
    ) -> tuple[int, CachedPrefixState] | None:
        """Return the longest exact cached prefix, or ``None`` on a miss."""
        token_ids = self._token_ids(input_ids)
        maximum_prefix_length = len(token_ids) - len(token_ids) % self.block_size
        if maximum_prefix_length == 0:
            return None
        block_namespaces = self._block_namespaces(token_ids, maximum_prefix_length)
        for block_offset in range(
            maximum_prefix_length - self.block_size,
            -1,
            -self.block_size,
        ):
            block_end = block_offset + self.block_size
            block_namespace = block_namespaces[block_offset // self.block_size]
            block_sequence = token_ids[block_offset:block_end]
            candidate = self.store.get(block_namespace, block_offset, block_sequence)
            if candidate is None:
                continue
            try:
                encoded_payload = self.store.validate_candidate(
                    candidate,
                    block_namespace,
                    block_offset,
                    block_sequence,
                ).payload
                if not isinstance(encoded_payload, bytes):
                    raise ValueError("bulk prefix payload encoding is invalid")
                payload = torch.load(
                    BytesIO(encoded_payload),
                    map_location=device,
                    weights_only=True,
                )
                cached_prefix = validate_prefix_state_payload(payload, block_end)
            except (
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
                EOFError,
                IndexError,
                KeyError,
                pickle.UnpicklingError,
            ):
                continue
            return block_end, cached_prefix
        return None

    def put_prefix(self, input_ids: Tensor, cached_prefix: CachedPrefixState) -> None:
        """Store a validated-size prefix state when the prefix ends on a block."""
        token_ids = self._token_ids(input_ids)
        prefix_length = len(token_ids)
        if prefix_length == 0 or prefix_length % self.block_size:
            return
        block_offset = prefix_length - self.block_size
        block_namespace = self._block_namespaces(token_ids, prefix_length)[-1]
        payload_buffer = BytesIO()
        torch.save(
            build_prefix_state_payload(prefix_length, cached_prefix),
            payload_buffer,
        )
        self.store.put(
            block_namespace,
            block_offset,
            token_ids[block_offset:prefix_length],
            payload_buffer.getvalue(),
        )

    def statistics(self) -> BulkBlockStatistics:
        """Return hit, miss, capacity and integrity counters for the block store."""
        return self.store.statistics()

    def __len__(self) -> int:
        return len(self.store)

    def _token_ids(self, input_ids: Tensor) -> tuple[int, ...]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("bulk prefix cache requires input_ids with shape [1, sequence]")
        raw_token_ids: Iterable[object] = input_ids.detach().cpu().reshape(-1).tolist()
        token_ids: list[int] = []
        for raw_token_id in raw_token_ids:
            try:
                token_id = operator.index(raw_token_id)
            except TypeError as error:
                raise TypeError("bulk prefix cache token IDs must be integers") from error
            if token_id < 0 or token_id > _MAX_TOKEN_ID:
                raise ValueError("bulk prefix cache token ID is outside the supported range")
            token_ids.append(token_id)
        return tuple(token_ids)

    def _block_namespaces(
        self,
        token_ids: tuple[int, ...],
        prefix_length: int,
    ) -> tuple[str, ...]:
        digest = self._initial_prefix_digest()
        namespaces: list[str] = []
        for block_offset in range(0, prefix_length, self.block_size):
            namespaces.append(f"{self.namespace}:{digest.hex()}")
            for token_offset, token_id in enumerate(
                token_ids[block_offset : block_offset + self.block_size],
                start=block_offset,
            ):
                digest = self._advance_prefix_digest(digest, token_offset, token_id)
        return tuple(namespaces)

    def _initial_prefix_digest(self) -> bytes:
        namespace_bytes = self.namespace.encode("utf-8")
        digest = hashlib.blake2b(digest_size=_DIGEST_SIZE_BYTES)
        digest.update(_PREFIX_DIGEST_DOMAIN)
        digest.update(len(namespace_bytes).to_bytes(_TOKEN_WIDTH_BYTES, "big"))
        digest.update(namespace_bytes)
        return digest.digest()

    @staticmethod
    def _advance_prefix_digest(digest: bytes, offset: int, token_id: int) -> bytes:
        next_digest = hashlib.blake2b(digest_size=_DIGEST_SIZE_BYTES)
        next_digest.update(digest)
        next_digest.update(offset.to_bytes(_TOKEN_WIDTH_BYTES, "big"))
        next_digest.update(token_id.to_bytes(_TOKEN_WIDTH_BYTES, "big"))
        return next_digest.digest()


__all__ = ["BulkPrefixCache"]
