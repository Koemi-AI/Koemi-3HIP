from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import tempfile
import time
import uuid
from array import array
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID
from koemi.data.contracts import DatasetRecord, DatasetValidationError
from koemi.data.serialization import serialize_record
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel
from koemi.training.checkpoints import CheckpointStore
from koemi.training.dataset import CausalByteDataset, IGNORE_TARGET_ID
from koemi.training.generation import generate_text
from koemi.training.objective import calculate_training_objective, token_cross_entropy


RUN_FORMAT_VERSION = 1
CORPUS_FORMAT_VERSION = 1
CHECKPOINT_FORMAT_VERSION = 1
MINIMUM_A100_MEMORY_BYTES = 70 * 2**30
CODE_SYSTEM_PROMPT = (
    "You are a precise English software engineer. Diagnose errors, explain the cause, "
    "provide a minimal safe change or command, and state how to verify it."
)
MATH_SYSTEM_PROMPT = (
    "Solve in English. Keep the algebra explicit, check each inference, and present "
    "the final answer separately."
)
PRIORITY_CODE_TERMS = (
    "python",
    "typescript",
    "javascript",
    "node",
    "npm",
    "bash",
    "powershell",
    "shell",
    "terminal",
    "command line",
    "cli",
    "git",
    "commit",
    "rebase",
    "merge",
    "debug",
    "bug",
    "error",
    "exception",
    "traceback",
    "test",
    "pytest",
)
SOURCE_REVISIONS = {
    "opencode": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
    "codefeedback": "a08c213a9748c66c15d0225814be80a2e77adf4a",
    "magicoder": "b0079beaa0361d82412520b873715bee59cc7dd4",
    "openr1_math": "e4e141ec9dea9f8326f4d347be56105859b2bd68",
}


class SourceRowRejected(DatasetValidationError):
    pass


@dataclass(frozen=True)
class CorpusQuotas:
    opencode_priority: int
    opencode_general: int
    codefeedback: int
    magicoder: int
    openr1_math: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RunConfiguration:
    results_directory: Path
    session_seconds: int
    data_seed: int
    model_seed: int
    sequence_length: int
    quotas: CorpusQuotas
    opencode_scan_limit: int
    source_scan_limit: int
    shuffle_buffer_size: int
    num_workers: int
    checkpoint_interval_seconds: int
    log_interval_steps: int
    evaluation_batches: int


@dataclass(frozen=True)
class CheckpointLoadResult:
    payload: dict[str, Any] | None
    recovery_messages: tuple[str, ...]


class MaterializedCausalByteDataset(Dataset[tuple[bytes, bytes, bytes, bytes]]):
    def __init__(self, records: Sequence[DatasetRecord], sequence_length: int) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.sequence_length = sequence_length
        self.token_streams: list[bytes] = []
        self.supervised_streams: list[bytes] = []
        self.thinking_streams: list[bytes] = []
        self.record_indices = array("I")
        self.offsets = array("I")
        supervised_token_count = 0
        for record_index, record in enumerate(records):
            serialized = serialize_record(record)
            token_stream = serialized.token_bytes
            if len(token_stream) < 2:
                continue
            supervised_stream = bytes(serialized.supervised_positions)
            thinking_stream = bytes(serialized.thinking_positions)
            storage_index = len(self.token_streams)
            self.token_streams.append(token_stream)
            self.supervised_streams.append(supervised_stream)
            self.thinking_streams.append(thinking_stream)
            for offset in range(0, len(token_stream) - 1, sequence_length):
                end_offset = min(offset + sequence_length, len(token_stream) - 1)
                chunk_supervised_token_count = sum(supervised_stream[offset + 1 : end_offset + 1])
                if chunk_supervised_token_count == 0:
                    continue
                self.record_indices.append(storage_index)
                self.offsets.append(offset)
                supervised_token_count += chunk_supervised_token_count
        if not self.record_indices:
            raise DatasetValidationError("dataset does not contain a trainable causal sequence")
        if supervised_token_count == 0:
            raise DatasetValidationError("dataset does not contain supervised target tokens")
        self.source_record_count = len(records)

    def __len__(self) -> int:
        return len(self.record_indices)

    def __getitem__(self, index: int) -> tuple[bytes, bytes, bytes, bytes]:
        stream_index = self.record_indices[index]
        offset = self.offsets[index]
        token_stream = self.token_streams[stream_index]
        end_offset = min(offset + self.sequence_length, len(token_stream) - 1)
        return (
            token_stream[offset:end_offset],
            token_stream[offset + 1 : end_offset + 1],
            self.supervised_streams[stream_index][offset + 1 : end_offset + 1],
            self.thinking_streams[stream_index][offset + 1 : end_offset + 1],
        )


def collate_materialized_chunks(chunks: list[tuple[bytes, bytes, bytes, bytes]]) -> dict[str, Tensor]:
    if not chunks:
        raise ValueError("cannot collate an empty batch")
    maximum_length = max(len(input_bytes) for input_bytes, _, _, _ in chunks)
    input_ids = torch.full((len(chunks), maximum_length), PAD_TOKEN_ID, dtype=torch.long)
    target_ids = torch.full((len(chunks), maximum_length), IGNORE_TARGET_ID, dtype=torch.long)
    thinking_mask = torch.zeros((len(chunks), maximum_length), dtype=torch.bool)
    for row_index, (input_bytes, target_bytes, supervised_bytes, thinking_bytes) in enumerate(chunks):
        length = len(input_bytes)
        input_ids[row_index, :length] = torch.tensor(bytearray(input_bytes), dtype=torch.long)
        target_values = torch.tensor(bytearray(target_bytes), dtype=torch.long)
        supervised_values = torch.tensor(bytearray(supervised_bytes), dtype=torch.bool)
        target_values = torch.where(
            supervised_values,
            target_values,
            torch.full_like(target_values, IGNORE_TARGET_ID),
        )
        target_ids[row_index, :length] = target_values
        thinking_mask[row_index, :length] = torch.tensor(bytearray(thinking_bytes), dtype=torch.bool)
    return {"input_ids": input_ids, "target_ids": target_ids, "thinking_mask": thinking_mask}


class DeterministicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset_length: int,
        batch_size: int,
        data_seed: int,
        epoch_index: int,
        start_batch_index: int,
    ) -> None:
        if dataset_length < 1:
            raise ValueError("dataset_length must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset_length = dataset_length
        self.batch_size = batch_size
        self.data_seed = data_seed
        self.epoch_index = epoch_index
        self.start_batch_index = start_batch_index
        self.batch_count = math.ceil(dataset_length / batch_size)
        if not 0 <= start_batch_index <= self.batch_count:
            raise ValueError("start_batch_index is outside this epoch")

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.data_seed + self.epoch_index)
        order = torch.randperm(self.dataset_length, generator=generator).tolist()
        for batch_index in range(self.start_batch_index, self.batch_count):
            start_offset = batch_index * self.batch_size
            yield order[start_offset : start_offset + self.batch_size]

    def __len__(self) -> int:
        return self.batch_count - self.start_batch_index


class RotatingCheckpointStore:
    def __init__(self, directory: Path, run_signature: str) -> None:
        self.directory = directory
        self.run_signature = run_signature
        self.manifest_path = directory / "checkpoint_manifest.json"

    @property
    def slot_paths(self) -> tuple[Path, Path]:
        return self.directory / "checkpoint_a.pt", self.directory / "checkpoint_b.pt"

    def save(self, payload: dict[str, Any], optimizer_step: int) -> Path:
        if optimizer_step < 0:
            raise ValueError("optimizer_step must be non-negative")
        self.directory.mkdir(parents=True, exist_ok=True)
        slot_path = self.slot_paths[optimizer_step % len(self.slot_paths)]
        saved_payload = dict(payload)
        saved_payload["checkpoint_format_version"] = CHECKPOINT_FORMAT_VERSION
        saved_payload["run_signature"] = self.run_signature
        saved_payload["optimizer_step"] = optimizer_step
        atomic_torch_save(slot_path, saved_payload)
        atomic_write_json(
            self.manifest_path,
            {
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "run_signature": self.run_signature,
                "active_slot": slot_path.name,
                "optimizer_step": optimizer_step,
            },
        )
        return slot_path

    def load_latest(self) -> CheckpointLoadResult:
        recovery_messages: list[str] = []
        if self.manifest_path.exists():
            manifest = read_json_object(self.manifest_path)
            if manifest.get("run_signature") != self.run_signature:
                raise ValueError("checkpoint manifest belongs to a different run signature")
            active_name = manifest.get("active_slot")
            if not isinstance(active_name, str):
                raise ValueError("checkpoint manifest has no active slot")
            candidates = [self.directory / active_name]
            candidates.extend(path for path in self.slot_paths if path not in candidates)
        else:
            candidates = list(self.slot_paths)
        valid_payloads: list[tuple[int, dict[str, Any]]] = []
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                optimizer_step = validate_checkpoint_payload(payload, self.run_signature)
                valid_payloads.append((optimizer_step, payload))
            except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
                recovery_messages.append(f"ignored invalid checkpoint {path.name}: {type(error).__name__}: {error}")
        if valid_payloads:
            optimizer_step, payload = max(valid_payloads, key=lambda item: item[0])
            if recovery_messages:
                recovery_messages.append(f"resumed checkpoint at optimizer step {optimizer_step}")
            return CheckpointLoadResult(payload, tuple(recovery_messages))
        if recovery_messages:
            raise RuntimeError("no valid checkpoint slot remains: " + " | ".join(recovery_messages))
        return CheckpointLoadResult(None, ())


class RunningMetrics:
    def __init__(self, device: torch.device, expert_count: int) -> None:
        self.device = device
        self.loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        self.answer_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        self.thinking_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        self.loss_square_sum = torch.zeros((), device=device, dtype=torch.float64)
        self.supervised_count = torch.zeros((), device=device, dtype=torch.long)
        self.answer_count = torch.zeros((), device=device, dtype=torch.long)
        self.thinking_count = torch.zeros((), device=device, dtype=torch.long)
        self.expert_counts = torch.zeros(expert_count, device=device, dtype=torch.long)
        self.valid_token_count = 0

    def add(
        self,
        output,
        token_losses: Tensor,
        target_ids: Tensor,
        thinking_mask: Tensor,
    ) -> None:
        supervised_mask = target_ids != IGNORE_TARGET_ID
        answer_mask = supervised_mask & ~thinking_mask
        thinking_positions = supervised_mask & thinking_mask
        self.loss_sum += (token_losses * supervised_mask).sum().detach().to(torch.float64)
        self.answer_loss_sum += (token_losses * answer_mask).sum().detach().to(torch.float64)
        self.thinking_loss_sum += (token_losses * thinking_positions).sum().detach().to(torch.float64)
        self.loss_square_sum += (token_losses.square() * supervised_mask).sum().detach().to(torch.float64)
        self.supervised_count += supervised_mask.sum().detach()
        self.answer_count += answer_mask.sum().detach()
        self.thinking_count += thinking_positions.sum().detach()
        self.valid_token_count += output.token_count
        if output.active_expert_indices is not None:
            assignments = output.active_expert_indices[output.valid_positions].reshape(-1)
            self.expert_counts += torch.bincount(assignments, minlength=len(self.expert_counts))

    def as_dict(self, elapsed_seconds: float) -> dict[str, Any]:
        supervised_count = int(self.supervised_count.item())
        answer_count = int(self.answer_count.item())
        thinking_count = int(self.thinking_count.item())
        if supervised_count == 0:
            raise RuntimeError("metrics contain no supervised targets")
        loss_sum = float(self.loss_sum.item())
        answer_loss_sum = float(self.answer_loss_sum.item())
        thinking_loss_sum = float(self.thinking_loss_sum.item())
        mean_loss = loss_sum / supervised_count
        variance = max(0.0, float(self.loss_square_sum.item()) / supervised_count - mean_loss * mean_loss)
        expert_counts = self.expert_counts.cpu().tolist()
        return {
            "loss_nats": mean_loss,
            "bpb": mean_loss / math.log(2),
            "perplexity": math.exp(min(mean_loss, 80.0)),
            "answer_loss_nats": answer_loss_sum / answer_count if answer_count else None,
            "answer_bpb": answer_loss_sum / answer_count / math.log(2) if answer_count else None,
            "thinking_loss_nats": thinking_loss_sum / thinking_count if thinking_count else None,
            "thinking_bpb": thinking_loss_sum / thinking_count / math.log(2) if thinking_count else None,
            "loss_standard_error": math.sqrt(variance / supervised_count),
            "supervised_tokens": supervised_count,
            "answer_tokens": answer_count,
            "thinking_tokens": thinking_count,
            "valid_input_tokens": self.valid_token_count,
            "supervised_tokens_per_second": supervised_count / max(elapsed_seconds, 1e-9),
            "expert_load": expert_load_summary(expert_counts),
        }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_checkpoint_payload(payload: Any, run_signature: str) -> int:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a dictionary")
    if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint format version does not match")
    if payload.get("run_signature") != run_signature:
        raise ValueError("checkpoint run signature does not match")
    optimizer_step = payload.get("optimizer_step")
    if not isinstance(optimizer_step, int) or optimizer_step < 0:
        raise ValueError("checkpoint optimizer step is invalid")
    return optimizer_step


def require_text(raw_value: Any, field_name: str) -> str:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise SourceRowRejected(f"field '{field_name}' must be a non-empty string")
    return raw_value.strip()


def require_bounded_text(raw_value: Any, field_name: str, maximum_bytes: int) -> str:
    value = require_text(raw_value, field_name)
    if len(value.encode("utf-8")) > maximum_bytes:
        raise SourceRowRejected(f"field '{field_name}' exceeds {maximum_bytes} UTF-8 bytes")
    return value


def parse_full_test_score(raw_value: Any) -> None:
    if not isinstance(raw_value, str):
        raise SourceRowRejected("average_test_score must be a string")
    try:
        score = float(raw_value)
    except ValueError as error:
        raise SourceRowRejected("average_test_score is not a number") from error
    if score != 1.0:
        raise SourceRowRejected("average_test_score is not exactly 1.0")


def require_all_test_statuses_pass(raw_value: Any) -> None:
    if not isinstance(raw_value, str):
        raise SourceRowRejected("tests_execution_status must be a JSON string")
    try:
        statuses = json.loads(raw_value)
    except json.JSONDecodeError as error:
        raise SourceRowRejected("tests_execution_status is not valid JSON") from error
    if not isinstance(statuses, list) or not statuses or any(status != "pass" for status in statuses):
        raise SourceRowRejected("tests_execution_status contains a non-passing test")


def bounded_record(
    identifier: str,
    input_text: str,
    output_text: str,
    metadata: dict[str, Any],
    system_text: str,
    thinking_text: str | None = None,
) -> DatasetRecord:
    if len(input_text.encode("utf-8")) > 8_192:
        raise SourceRowRejected("input exceeds 8192 UTF-8 bytes")
    if len(output_text.encode("utf-8")) > 12_288:
        raise SourceRowRejected("output exceeds 12288 UTF-8 bytes")
    if thinking_text is not None and len(thinking_text.encode("utf-8")) > 12_288:
        raise SourceRowRejected("thinking exceeds 12288 UTF-8 bytes")
    return DatasetRecord(identifier, input_text, thinking_text, output_text, metadata, system_text)


def adapt_opencode_row(raw_row: Mapping[str, Any], fallback_identifier: str) -> DatasetRecord:
    parse_full_test_score(raw_row.get("average_test_score"))
    require_all_test_statuses_pass(raw_row.get("tests_execution_status"))
    identifier = require_text(raw_row.get("id", fallback_identifier), "id")
    input_text = require_bounded_text(raw_row.get("input"), "input", 8_192)
    output_text = require_bounded_text(raw_row.get("output"), "output", 12_288)
    domain = require_text(raw_row.get("domain", "unknown"), "domain")
    return bounded_record(
        f"opencode:{identifier}",
        input_text,
        output_text,
        {"source": "nvidia/OpenCodeInstruct", "revision": SOURCE_REVISIONS["opencode"], "domain": domain},
        CODE_SYSTEM_PROMPT,
    )


def adapt_codefeedback_row(raw_row: Mapping[str, Any], fallback_identifier: str) -> DatasetRecord:
    query = require_bounded_text(raw_row.get("query"), "query", 8_192)
    answer = require_bounded_text(raw_row.get("answer"), "answer", 12_288)
    language = require_text(raw_row.get("lang", "unknown"), "lang")
    return bounded_record(
        f"codefeedback:{fallback_identifier}",
        query,
        answer,
        {"source": "m-a-p/CodeFeedback-Filtered-Instruction", "revision": SOURCE_REVISIONS["codefeedback"], "language": language},
        CODE_SYSTEM_PROMPT,
    )


def adapt_magicoder_row(raw_row: Mapping[str, Any], fallback_identifier: str) -> DatasetRecord:
    instruction = require_bounded_text(raw_row.get("instruction"), "instruction", 8_192)
    response = require_bounded_text(raw_row.get("response"), "response", 12_288)
    return bounded_record(
        f"magicoder:{fallback_identifier}",
        instruction,
        response,
        {"source": "ise-uiuc/Magicoder-Evol-Instruct-110K", "revision": SOURCE_REVISIONS["magicoder"]},
        CODE_SYSTEM_PROMPT,
    )


def extract_reasoning_span(generation: str) -> str:
    stripped = generation.strip()
    if "<think>" not in stripped:
        return stripped
    opening = stripped.find("<think>") + len("<think>")
    closing = stripped.find("</think>", opening)
    if closing < 0:
        raise SourceRowRejected("verified math generation has an unclosed think span")
    reasoning = stripped[opening:closing].strip()
    if not reasoning:
        raise SourceRowRejected("verified math generation has an empty think span")
    return reasoning


def adapt_openr1_math_row(raw_row: Mapping[str, Any], fallback_identifier: str) -> DatasetRecord:
    problem = require_bounded_text(raw_row.get("problem"), "problem", 8_192)
    answer = require_bounded_text(raw_row.get("answer"), "answer", 12_288)
    generations = raw_row.get("generations")
    correctness = raw_row.get("correctness_math_verify")
    completeness = raw_row.get("is_reasoning_complete")
    if not isinstance(generations, Sequence) or isinstance(generations, (str, bytes)):
        raise SourceRowRejected("generations must be an array")
    if not isinstance(correctness, Sequence) or isinstance(correctness, (str, bytes)):
        raise SourceRowRejected("correctness_math_verify must be an array")
    if not isinstance(completeness, Sequence) or isinstance(completeness, (str, bytes)):
        raise SourceRowRejected("is_reasoning_complete must be an array")
    if not (len(generations) == len(correctness) == len(completeness)):
        raise SourceRowRejected("math verification arrays do not have matching lengths")
    verified_reasoning = [
        extract_reasoning_span(generation)
        for generation, is_correct, is_complete in zip(generations, correctness, completeness, strict=True)
        if isinstance(generation, str) and is_correct is True and is_complete is True
    ]
    if not verified_reasoning:
        raise SourceRowRejected("row has no complete Math-Verify-correct reasoning generation")
    identifier = require_text(raw_row.get("uuid", fallback_identifier), "uuid")
    reasoning = min(verified_reasoning, key=len)
    return bounded_record(
        f"openr1_math:{identifier}",
        problem,
        answer,
        {
            "source": "open-r1/OpenR1-Math-220k",
            "revision": SOURCE_REVISIONS["openr1_math"],
            "verification": "math_verify_and_complete",
        },
        MATH_SYSTEM_PROMPT,
        reasoning,
    )


def is_priority_code_record(record: DatasetRecord) -> bool:
    candidate_text = f"{record.input_text}\n{record.output_text}".lower()
    return any(term in candidate_text for term in PRIORITY_CODE_TERMS)


def load_streaming_dataset(dataset_name: str, revision: str, config_name: str | None, seed: int, buffer_size: int):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError("install datasets==3.6.0 before running the A100 trainer") from error
    dataset = load_dataset(dataset_name, name=config_name, split="train", streaming=True, revision=revision)
    return dataset.shuffle(seed=seed, buffer_size=buffer_size)


def rejection_key(error: Exception) -> str:
    message = str(error).split(":", maxsplit=1)[0]
    return f"{type(error).__name__}: {message}"[:180]


def collect_source_records(
    stream,
    target_count: int,
    scan_limit: int,
    adapter: Callable[[Mapping[str, Any], str], DatasetRecord],
    source_name: str,
) -> tuple[list[DatasetRecord], dict[str, Any]]:
    records: list[DatasetRecord] = []
    rejections: Counter[str] = Counter()
    scanned = 0
    for raw_row in stream:
        scanned += 1
        if scanned > scan_limit:
            break
        if not isinstance(raw_row, Mapping):
            rejections["SourceRowRejected: row is not an object"] += 1
            continue
        try:
            records.append(adapter(raw_row, str(scanned - 1)))
        except (DatasetValidationError, SourceRowRejected, TypeError, ValueError, json.JSONDecodeError) as error:
            rejections[rejection_key(error)] += 1
        if len(records) == target_count:
            return records, {"source": source_name, "selected": len(records), "scanned": scanned, "rejections": dict(rejections)}
    raise RuntimeError(
        f"{source_name} produced {len(records)} valid records, below target {target_count}, after {scanned} scanned rows; rejections={dict(rejections)}"
    )


def collect_opencode_records(stream, quotas: CorpusQuotas, scan_limit: int) -> tuple[list[DatasetRecord], dict[str, Any]]:
    priority_records: list[DatasetRecord] = []
    general_records: list[DatasetRecord] = []
    rejections: Counter[str] = Counter()
    scanned = 0
    for raw_row in stream:
        scanned += 1
        if scanned > scan_limit:
            break
        if not isinstance(raw_row, Mapping):
            rejections["SourceRowRejected: row is not an object"] += 1
            continue
        try:
            record = adapt_opencode_row(raw_row, str(scanned - 1))
        except (DatasetValidationError, SourceRowRejected, TypeError, ValueError, json.JSONDecodeError) as error:
            rejections[rejection_key(error)] += 1
            continue
        if is_priority_code_record(record) and len(priority_records) < quotas.opencode_priority:
            priority_records.append(record)
        elif len(general_records) < quotas.opencode_general:
            general_records.append(record)
        else:
            rejections["SourceRowRejected: quota bucket already full"] += 1
        if len(priority_records) == quotas.opencode_priority and len(general_records) == quotas.opencode_general:
            return priority_records + general_records, {
                "source": "nvidia/OpenCodeInstruct",
                "selected": len(priority_records) + len(general_records),
                "priority_selected": len(priority_records),
                "general_selected": len(general_records),
                "scanned": scanned,
                "rejections": dict(rejections),
            }
    raise RuntimeError(
        "nvidia/OpenCodeInstruct produced "
        f"priority={len(priority_records)}/{quotas.opencode_priority} and "
        f"general={len(general_records)}/{quotas.opencode_general} after {scanned} scanned rows; "
        f"rejections={dict(rejections)}"
    )


def record_to_json(record: DatasetRecord) -> dict[str, Any]:
    return {
        "id": record.identifier,
        "input": record.input_text,
        "thinking": record.thinking_text,
        "output": record.output_text,
        "metadata": record.metadata,
        "system": record.system_text,
    }


def record_from_json(value: Mapping[str, Any]) -> DatasetRecord:
    identifier = require_text(value.get("id"), "id")
    input_text = require_text(value.get("input"), "input")
    output_text = require_text(value.get("output"), "output")
    thinking_text = value.get("thinking")
    system_text = value.get("system")
    metadata = value.get("metadata")
    if thinking_text is not None and not isinstance(thinking_text, str):
        raise DatasetValidationError("field 'thinking' must be a string or null")
    if system_text is not None and not isinstance(system_text, str):
        raise DatasetValidationError("field 'system' must be a string or null")
    if not isinstance(metadata, dict):
        raise DatasetValidationError("field 'metadata' must be an object")
    return DatasetRecord(identifier, input_text, thinking_text, output_text, metadata, system_text)


def write_corpus(path: Path, records: Sequence[DatasetRecord]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as corpus_file:
            for record in records:
                corpus_file.write(json.dumps(record_to_json(record), ensure_ascii=False, sort_keys=True))
                corpus_file.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def read_corpus(path: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    with path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                raise DatasetValidationError(f"corpus line {line_number} is empty")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise DatasetValidationError(f"corpus line {line_number} is not an object")
            records.append(record_from_json(value))
    if not records:
        raise DatasetValidationError("corpus is empty")
    identifiers = [record.identifier for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise DatasetValidationError("corpus contains duplicate record identifiers")
    return records


def build_or_load_corpus(configuration: RunConfiguration) -> tuple[list[DatasetRecord], dict[str, Any]]:
    corpus_path = configuration.results_directory / "corpus.jsonl"
    manifest_path = configuration.results_directory / "corpus_manifest.json"
    corpus_exists = corpus_path.exists()
    manifest_exists = manifest_path.exists()
    requested_contract = {
        "corpus_format_version": CORPUS_FORMAT_VERSION,
        "source_revisions": SOURCE_REVISIONS,
        "quotas": configuration.quotas.to_dict(),
        "data_seed": configuration.data_seed,
        "shuffle_buffer_size": configuration.shuffle_buffer_size,
    }
    if corpus_exists and manifest_exists:
        manifest = read_json_object(manifest_path)
        if manifest.get("requested_contract") != requested_contract:
            raise ValueError("existing corpus contract does not match this A100 run")
        actual_hash = sha256_file(corpus_path)
        if manifest.get("sha256") != actual_hash:
            raise ValueError("existing corpus SHA-256 does not match its manifest")
        records = read_corpus(corpus_path)
        if manifest.get("record_count") != len(records):
            raise ValueError("existing corpus record count does not match its manifest")
        return records, manifest
    recovered_incomplete_artifact = corpus_exists != manifest_exists

    opencode_stream = load_streaming_dataset(
        "nvidia/OpenCodeInstruct",
        SOURCE_REVISIONS["opencode"],
        "train",
        configuration.data_seed,
        configuration.shuffle_buffer_size,
    )
    opencode_records, opencode_report = collect_opencode_records(
        opencode_stream,
        configuration.quotas,
        configuration.opencode_scan_limit,
    )
    codefeedback_stream = load_streaming_dataset(
        "m-a-p/CodeFeedback-Filtered-Instruction",
        SOURCE_REVISIONS["codefeedback"],
        None,
        configuration.data_seed + 1,
        configuration.shuffle_buffer_size,
    )
    codefeedback_records, codefeedback_report = collect_source_records(
        codefeedback_stream,
        configuration.quotas.codefeedback,
        configuration.source_scan_limit,
        adapt_codefeedback_row,
        "m-a-p/CodeFeedback-Filtered-Instruction",
    )
    magicoder_stream = load_streaming_dataset(
        "ise-uiuc/Magicoder-Evol-Instruct-110K",
        SOURCE_REVISIONS["magicoder"],
        None,
        configuration.data_seed + 2,
        configuration.shuffle_buffer_size,
    )
    magicoder_records, magicoder_report = collect_source_records(
        magicoder_stream,
        configuration.quotas.magicoder,
        configuration.source_scan_limit,
        adapt_magicoder_row,
        "ise-uiuc/Magicoder-Evol-Instruct-110K",
    )
    math_stream = load_streaming_dataset(
        "open-r1/OpenR1-Math-220k",
        SOURCE_REVISIONS["openr1_math"],
        "default",
        configuration.data_seed + 3,
        configuration.shuffle_buffer_size,
    )
    math_records, math_report = collect_source_records(
        math_stream,
        configuration.quotas.openr1_math,
        configuration.source_scan_limit,
        adapt_openr1_math_row,
        "open-r1/OpenR1-Math-220k",
    )
    records = opencode_records + codefeedback_records + magicoder_records + math_records
    identifiers = [record.identifier for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise DatasetValidationError("selected corpus contains duplicate record identifiers")
    write_corpus(corpus_path, records)
    manifest = {
        "requested_contract": requested_contract,
        "record_count": len(records),
        "sha256": sha256_file(corpus_path),
        "sources": [opencode_report, codefeedback_report, magicoder_report, math_report],
        "recovered_incomplete_artifact": recovered_incomplete_artifact,
    }
    atomic_write_json(manifest_path, manifest)
    return records, manifest


def split_records(records: Sequence[DatasetRecord], data_seed: int, validation_fraction: float = 0.03) -> tuple[list[DatasetRecord], list[DatasetRecord]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    training_records: list[DatasetRecord] = []
    validation_records: list[DatasetRecord] = []
    threshold = int(validation_fraction * 10_000)
    for record in records:
        digest = hashlib.sha256(f"{data_seed}:{record.identifier}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % 10_000
        (validation_records if bucket < threshold else training_records).append(record)
    if not training_records or not validation_records:
        raise DatasetValidationError("deterministic split produced an empty partition")
    return training_records, validation_records


def model_settings() -> ModelSettings:
    return ModelSettings(
        embedding_size=512,
        memory_features=16,
        local_memory_size=16,
        salience_memory_size=16,
        salience_threshold=0.75,
        expert_count=128,
        expert_top_k=6,
        cache_capacity=256,
        scan_chunk=128,
        refine_decay_rate=0.0625,
        ablation="no_refine",
    )


def configure_a100() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("this notebook requires a CUDA A100 runtime")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    gpu_name = torch.cuda.get_device_name(device)
    if "A100" not in gpu_name.upper():
        raise RuntimeError(f"this run requires an A100, received '{gpu_name}'")
    if properties.total_memory < MINIMUM_A100_MEMORY_BYTES:
        raise RuntimeError(f"this run requires at least 70 GiB VRAM, received {properties.total_memory / 2**30:.2f} GiB")
    if properties.major < 8:
        raise RuntimeError(f"this run requires compute capability >= 8.0, received {properties.major}.{properties.minor}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("this A100 runtime does not expose CUDA BF16 support")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device, {
        "gpu": gpu_name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16": True,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def state_max_abs_difference(left_state, right_state) -> dict[str, float]:
    fields = (
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
    differences: dict[str, float] = {}
    for field_name in fields:
        left_value = getattr(left_state, field_name)
        right_value = getattr(right_state, field_name)
        if left_value.dtype in {torch.bool, torch.long}:
            differences[field_name] = 0.0 if torch.equal(left_value, right_value) else float("inf")
        else:
            differences[field_name] = float((left_value - right_value).abs().max().cpu())
    differences["step_index"] = 0.0 if left_state.step_index == right_state.step_index else float("inf")
    return differences


def run_cuda_preflight(model: KoemiModel, device: torch.device) -> dict[str, Any]:
    model.eval()
    contract_input = torch.randint(0, 256, (2, 32), device=device)
    with torch.inference_mode():
        parallel = model(contract_input, execution_mode=ExecutionMode.PARALLEL)
        sequential = model(contract_input, execution_mode=ExecutionMode.SEQUENTIAL)
    logits_error = float((parallel.logits - sequential.logits).abs().max().cpu())
    state_error = state_max_abs_difference(parallel.state, sequential.state)
    if logits_error > 1e-3 or any(not math.isfinite(value) or value > 1e-3 for value in state_error.values()):
        raise AssertionError(f"parallel/sequential HERM contract failed: logits={logits_error}, state={state_error}")
    if not torch.equal(parallel.active_expert_indices, sequential.active_expert_indices):
        raise AssertionError("parallel/sequential expert assignments differ")
    numerator = torch.ones(1, model.settings.embedding_size, device=device)
    denominator = torch.zeros(1, 1, device=device)
    empty_read = model.associative_memory.confidence_weighted_read(numerator, denominator)
    if not torch.equal(empty_read, torch.zeros_like(empty_read)):
        raise AssertionError("zero-evidence associative read is not zero")
    model.train()
    return {
        "parallel_sequential_logits_max_abs": logits_error,
        "parallel_sequential_state_max_abs": state_error,
        "zero_evidence_read_max_abs": float(empty_read.abs().max().cpu()),
        "expert_assignments_equal": True,
    }


def move_batch(batch: dict[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        batch["input_ids"].to(device, non_blocking=True),
        batch["target_ids"].to(device, non_blocking=True),
        batch["thinking_mask"].to(device, non_blocking=True),
    )


def make_probe_batch(dataset: MaterializedCausalByteDataset, batch_size: int) -> dict[str, Tensor]:
    chunk_count = min(len(dataset), batch_size)
    chunks = [dataset[index % chunk_count] for index in range(batch_size)]
    return collate_materialized_chunks(chunks)


def benchmark_batch_size(
    settings: ModelSettings,
    dataset: MaterializedCausalByteDataset,
    device: torch.device,
    model_seed: int,
    candidate_batch_size: int,
) -> dict[str, Any]:
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    candidate_model = KoemiModel(settings).to(device)
    optimizer = torch.optim.AdamW(candidate_model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = make_probe_batch(dataset, candidate_batch_size)
    input_ids, target_ids, thinking_mask = move_batch(batch, device)
    try:
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = candidate_model(input_ids, execution_mode=ExecutionMode.PARALLEL)
                objective = calculate_training_objective(output, target_ids, thinking_mask, 0.5)
            scaler.scale(objective.total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started_at = time.perf_counter()
        supervised_tokens = 0
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = candidate_model(input_ids, execution_mode=ExecutionMode.PARALLEL)
                objective = calculate_training_objective(output, target_ids, thinking_mask, 0.5)
            if not bool(torch.isfinite(objective.total_loss)):
                raise FloatingPointError("non-finite objective in A100 batch calibration")
            scaler.scale(objective.total_loss).backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(candidate_model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("non-finite gradient in A100 batch calibration")
            scaler.step(optimizer)
            scaler.update()
            supervised_tokens += int((target_ids != IGNORE_TARGET_ID).sum().item())
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started_at
        return {
            "batch_size": candidate_batch_size,
            "status": "ok",
            "supervised_tokens_per_second": supervised_tokens / elapsed_seconds,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
            "elapsed_seconds": elapsed_seconds,
        }
    except torch.cuda.OutOfMemoryError as error:
        return {"batch_size": candidate_batch_size, "status": "oom", "error": str(error).splitlines()[0]}
    finally:
        del candidate_model, optimizer, scaler, batch, input_ids, target_ids, thinking_mask
        torch.cuda.empty_cache()


def calibrate_batch_size(
    settings: ModelSettings,
    dataset: MaterializedCausalByteDataset,
    device: torch.device,
    model_seed: int,
) -> tuple[int, list[dict[str, Any]]]:
    candidates = (4, 8, 12, 16, 24, 32)
    reports = [benchmark_batch_size(settings, dataset, device, model_seed, candidate) for candidate in candidates]
    safe_reports = [
        report
        for report in reports
        if report["status"] == "ok" and report["peak_memory_bytes"] <= int(MINIMUM_A100_MEMORY_BYTES * 0.85)
    ]
    if not safe_reports:
        raise RuntimeError(f"no calibrated batch fits the A100 safety budget: {reports}")
    selected = max(safe_reports, key=lambda report: report["supervised_tokens_per_second"])
    return int(selected["batch_size"]), reports


def create_loader(
    dataset: MaterializedCausalByteDataset,
    batch_size: int,
    data_seed: int,
    epoch_index: int,
    start_batch_index: int,
    num_workers: int,
) -> DataLoader[dict[str, Tensor]]:
    sampler = DeterministicBatchSampler(len(dataset), batch_size, data_seed, epoch_index, start_batch_index)
    options: dict[str, Any] = {}
    if num_workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_materialized_chunks,
        num_workers=num_workers,
        pin_memory=True,
        **options,
    )


def schedule_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def training_payload(
    model: KoemiModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    training_state: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "training_state": dict(training_state),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": torch.cuda.get_rng_state_all(),
    }


def restore_training_payload(
    payload: Mapping[str, Any],
    model: KoemiModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> dict[str, int]:
    required = ("model_state", "optimizer_state", "scheduler_state", "scaler_state", "training_state")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"checkpoint is missing fields: {missing}")
    training_state = payload["training_state"]
    if not isinstance(training_state, Mapping):
        raise ValueError("checkpoint training_state is invalid")
    restored_state = {name: int(training_state[name]) for name in ("epoch_index", "next_batch_index", "tokens_seen")}
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["scaler_state"])
    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
    if "torch_random_state" in payload:
        torch.set_rng_state(payload["torch_random_state"])
    if "cuda_random_states" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_random_states"])
    return restored_state


def append_json_line(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as log_file:
        log_file.write(json.dumps(value, sort_keys=True))
        log_file.write("\n")


def evaluate(
    model: KoemiModel,
    dataset: MaterializedCausalByteDataset,
    batch_size: int,
    device: torch.device,
    maximum_batches: int,
    num_workers: int,
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if num_workers > 0:
        options["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_materialized_chunks,
        num_workers=num_workers,
        pin_memory=True,
        **options,
    )
    metrics = RunningMetrics(device, model.settings.expert_count)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= maximum_batches:
                break
            input_ids, target_ids, thinking_mask = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
                token_losses = token_cross_entropy(output.logits.float(), target_ids)
            if not bool(torch.isfinite(token_losses[target_ids != IGNORE_TARGET_ID]).all()):
                raise FloatingPointError("non-finite validation token loss")
            metrics.add(output, token_losses, target_ids, thinking_mask)
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at
    report = metrics.as_dict(elapsed_seconds)
    report["batches"] = min(maximum_batches, math.ceil(len(dataset) / batch_size))
    report["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    model.train()
    return report


def expert_load_summary(counts: Sequence[int]) -> dict[str, Any]:
    total = sum(counts)
    if not counts:
        return {"counts": [], "total_assignments": 0, "occupied_experts": 0, "entropy_nats": 0.0, "normalized_entropy": 0.0, "load_gini": 0.0}
    probabilities = [count / total for count in counts if count > 0] if total else []
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    mean = total / len(counts) if counts else 0.0
    absolute_difference_total = sum(abs(left - right) for left in counts for right in counts)
    gini = absolute_difference_total / (2 * len(counts) * total) if total else 0.0
    return {
        "counts": list(counts),
        "total_assignments": total,
        "occupied_experts": sum(count > 0 for count in counts),
        "entropy_nats": entropy,
        "normalized_entropy": entropy / math.log(len(counts)) if len(counts) > 1 else 0.0,
        "load_min": min(counts),
        "load_max": max(counts),
        "load_mean": mean,
        "load_gini": gini,
    }


def run_training(
    configuration: RunConfiguration,
    corpus_manifest: Mapping[str, Any],
    training_dataset: MaterializedCausalByteDataset,
    validation_dataset: MaterializedCausalByteDataset,
    device: torch.device,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    settings = model_settings()
    run_manifest_path = configuration.results_directory / "run_manifest.json"
    checkpoint_root = configuration.results_directory / "checkpoints"
    if run_manifest_path.exists():
        run_manifest = read_json_object(run_manifest_path)
        if run_manifest.get("corpus_sha256") != corpus_manifest.get("sha256"):
            raise ValueError("run manifest corpus hash does not match the selected corpus")
        if run_manifest.get("model_settings") != settings.to_dict():
            raise ValueError("run manifest model settings do not match")
        selected_batch_size = run_manifest.get("selected_batch_size")
        if not isinstance(selected_batch_size, int) or selected_batch_size < 1:
            raise ValueError("run manifest selected batch size is invalid")
        calibration = run_manifest.get("batch_calibration")
        if not isinstance(calibration, list):
            raise ValueError("run manifest batch calibration is invalid")
    else:
        selected_batch_size, calibration = calibrate_batch_size(settings, training_dataset, device, configuration.model_seed)
        run_manifest = {
            "run_format_version": RUN_FORMAT_VERSION,
            "corpus_sha256": corpus_manifest["sha256"],
            "model_settings": settings.to_dict(),
            "selected_batch_size": selected_batch_size,
            "target_effective_batch_size": 64,
            "batch_calibration": calibration,
            "environment": dict(environment),
            "source_revisions": SOURCE_REVISIONS,
            "sequence_length": configuration.sequence_length,
            "thinking_loss_weight": 0.5,
        }
        atomic_write_json(run_manifest_path, run_manifest)
    gradient_accumulation_steps = math.ceil(run_manifest["target_effective_batch_size"] / selected_batch_size)
    run_contract = {
        "run_format_version": RUN_FORMAT_VERSION,
        "corpus_sha256": corpus_manifest["sha256"],
        "model_settings": settings.to_dict(),
        "selected_batch_size": selected_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "thinking_loss_weight": 0.5,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "warmup_steps": 1_000,
        "schedule_steps": 100_000,
        "precision": "bf16",
    }
    run_signature = fingerprint(run_contract)
    checkpoint_store = RotatingCheckpointStore(checkpoint_root, run_signature)
    torch.manual_seed(configuration.model_seed)
    torch.cuda.manual_seed_all(configuration.model_seed)
    model = KoemiModel(settings).to(device)
    preflight_report = run_cuda_preflight(model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_lambda(step, 1_000, 100_000))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint_result = checkpoint_store.load_latest()
    training_state = {"epoch_index": 0, "next_batch_index": 0, "tokens_seen": 0}
    if checkpoint_result.payload is not None:
        training_state = restore_training_payload(checkpoint_result.payload, model, optimizer, scheduler, scaler)
    step_log_path = configuration.results_directory / "training_steps.jsonl"
    started_at = time.perf_counter()
    last_checkpoint_at = started_at
    optimizer_step = int(checkpoint_result.payload["optimizer_step"]) if checkpoint_result.payload is not None else 0
    session_step_count = 0
    stopped_for_budget = False
    while time.perf_counter() - started_at < configuration.session_seconds:
        loader = create_loader(
            training_dataset,
            selected_batch_size,
            configuration.data_seed,
            training_state["epoch_index"],
            training_state["next_batch_index"],
            configuration.num_workers,
        )
        if len(loader) == 0:
            training_state["epoch_index"] += 1
            training_state["next_batch_index"] = 0
            continue
        model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics = RunningMetrics(device, settings.expert_count)
        optimizer_window_started_at = time.perf_counter()
        accumulated_batches = 0
        starting_batch_index = training_state["next_batch_index"]
        last_consumed_batch = training_state["next_batch_index"]
        for relative_batch_index, batch in enumerate(loader):
            if time.perf_counter() - started_at >= configuration.session_seconds:
                stopped_for_budget = True
                break
            absolute_batch_index = starting_batch_index + relative_batch_index
            input_ids, target_ids, thinking_mask = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(input_ids, execution_mode=ExecutionMode.PARALLEL)
                objective = calculate_training_objective(output, target_ids, thinking_mask, 0.5)
                scaled_loss = objective.total_loss / gradient_accumulation_steps
            if not bool(torch.isfinite(objective.total_loss)):
                raise FloatingPointError(f"non-finite training objective at optimizer step {optimizer_step + 1}")
            scaler.scale(scaled_loss).backward()
            token_losses = token_cross_entropy(output.logits.float(), target_ids)
            metrics.add(output, token_losses, target_ids, thinking_mask)
            accumulated_batches += 1
            last_consumed_batch = absolute_batch_index + 1
            if accumulated_batches < gradient_accumulation_steps:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f"non-finite gradient at optimizer step {optimizer_step + 1}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            optimizer_step += 1
            session_step_count += 1
            training_state["next_batch_index"] = last_consumed_batch
            step_elapsed_seconds = time.perf_counter() - optimizer_window_started_at
            step_report = metrics.as_dict(step_elapsed_seconds)
            step_report.update(
                {
                    "optimizer_step": optimizer_step,
                    "epoch_index": training_state["epoch_index"],
                    "next_batch_index": training_state["next_batch_index"],
                    "tokens_seen": training_state["tokens_seen"] + step_report["supervised_tokens"],
                    "gradient_norm": float(gradient_norm.cpu()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gpu_allocated_bytes": torch.cuda.memory_allocated(device),
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                }
            )
            training_state["tokens_seen"] = int(step_report["tokens_seen"])
            if optimizer_step % configuration.log_interval_steps == 0:
                append_json_line(step_log_path, step_report)
                print(json.dumps(step_report, sort_keys=True))
            now = time.perf_counter()
            if now - last_checkpoint_at >= configuration.checkpoint_interval_seconds:
                checkpoint_store.save(training_payload(model, optimizer, scheduler, scaler, training_state), optimizer_step)
                last_checkpoint_at = now
            metrics = RunningMetrics(device, settings.expert_count)
            optimizer_window_started_at = time.perf_counter()
            accumulated_batches = 0
        if stopped_for_budget:
            optimizer.zero_grad(set_to_none=True)
            break
        if accumulated_batches:
            optimizer.zero_grad(set_to_none=True)
        training_state["epoch_index"] += 1
        training_state["next_batch_index"] = 0
    final_checkpoint_path = checkpoint_store.save(training_payload(model, optimizer, scheduler, scaler, training_state), optimizer_step)
    validation_report = evaluate(
        model,
        validation_dataset,
        selected_batch_size,
        device,
        configuration.evaluation_batches,
        configuration.num_workers,
    )
    model_checkpoint_path = configuration.results_directory / "koemi-3hip-a100-code-reasoning-model.pt"
    CheckpointStore().save(model_checkpoint_path, model, overwrite=True)
    generated_text = generate_text(
        model,
        ByteTokenizer(),
        "Diagnose this Python traceback and state a testable fix:\n",
        max_new_bytes=256,
        temperature=0.7,
        device=device,
    )
    sample_path = configuration.results_directory / "sample_generation.txt"
    atomic_write_text(sample_path, generated_text)
    session_report = {
        "run_signature": run_signature,
        "run_contract": run_contract,
        "preflight": preflight_report,
        "checkpoint_recovery": list(checkpoint_result.recovery_messages),
        "training_state": training_state,
        "optimizer_step": optimizer_step,
        "session_optimizer_steps": session_step_count,
        "session_wall_seconds": time.perf_counter() - started_at,
        "stopped_for_budget": stopped_for_budget,
        "checkpoint": str(final_checkpoint_path),
        "model_checkpoint": str(model_checkpoint_path),
        "sample_generation": str(sample_path),
        "validation": validation_report,
    }
    atomic_write_json(configuration.results_directory / "latest_session.json", session_report)
    return session_report


def run_internal_contract_tests() -> dict[str, Any]:
    opencode = adapt_opencode_row(
        {
            "id": "ok",
            "input": "Fix this Python traceback.",
            "output": "Check the exception and add a test.",
            "domain": "debugging",
            "average_test_score": "1.0",
            "tests_execution_status": '["pass", "pass"]',
        },
        "0",
    )
    if not is_priority_code_record(opencode):
        raise AssertionError("priority code classification rejected a Python traceback")
    math_record = adapt_openr1_math_row(
        {
            "uuid": "math-ok",
            "problem": "What is 1 + 1?",
            "answer": "2",
            "generations": ["<think>Add one and one to get two.</think>\\n2"],
            "correctness_math_verify": [True],
            "is_reasoning_complete": [True],
        },
        "0",
    )
    reference_dataset = CausalByteDataset((opencode, math_record), sequence_length=16)
    materialized_dataset = MaterializedCausalByteDataset((opencode, math_record), sequence_length=16)
    reference_chunks = [
        chunk
        for chunk in reference_dataset.chunks
        if any(target != IGNORE_TARGET_ID for target in chunk.target_ids)
    ]
    if len(reference_chunks) != len(materialized_dataset):
        raise AssertionError("materialized dataset chunk count differs from repository dataset")
    for index, reference_chunk in enumerate(reference_chunks):
        input_bytes, target_bytes, supervised_bytes, thinking_bytes = materialized_dataset[index]
        expected_targets = tuple(
            target if supervised else IGNORE_TARGET_ID
            for target, supervised in zip(target_bytes, supervised_bytes, strict=True)
        )
        if reference_chunk.input_ids != tuple(input_bytes) or reference_chunk.target_ids != expected_targets:
            raise AssertionError("materialized dataset does not match repository causal chunks")
        if reference_chunk.thinking_mask != tuple(bool(value) for value in thinking_bytes):
            raise AssertionError("materialized thinking mask does not match repository causal chunks")
    sampler = DeterministicBatchSampler(11, 3, 7, 2, 1)
    full_sampler = DeterministicBatchSampler(11, 3, 7, 2, 0)
    if list(sampler) != list(full_sampler)[1:]:
        raise AssertionError("resumed deterministic sampler does not match the epoch suffix")
    with tempfile.TemporaryDirectory() as temporary_directory:
        store = RotatingCheckpointStore(Path(temporary_directory), "contract-test")
        store.save({"training_state": {}}, 1)
        store.save({"training_state": {}}, 2)
        newer_slot = store.slot_paths[2 % 2]
        newer_slot.write_bytes(b"corrupt")
        recovered = store.load_latest()
        if recovered.payload is None or recovered.payload["optimizer_step"] != 1:
            raise AssertionError("rotating checkpoint store did not recover the older valid slot")
    return {"status": "passed", "dataset_chunks": len(materialized_dataset), "checkpoint_recovery": "passed"}


def parse_arguments() -> RunConfiguration:
    parser = argparse.ArgumentParser(description="Run the Koemi-3HIP A100 code and reasoning experiment")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--session-hours", type=float, default=9.0)
    parser.add_argument("--data-seed", type=int, default=20260913)
    parser.add_argument("--model-seed", type=int, default=1337)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--opencode-priority-records", type=int, default=60_000)
    parser.add_argument("--opencode-general-records", type=int, default=120_000)
    parser.add_argument("--codefeedback-records", type=int, default=80_000)
    parser.add_argument("--magicoder-records", type=int, default=70_000)
    parser.add_argument("--math-records", type=int, default=35_000)
    parser.add_argument("--opencode-scan-limit", type=int, default=2_000_000)
    parser.add_argument("--source-scan-limit", type=int, default=200_000)
    parser.add_argument("--shuffle-buffer-size", type=int, default=20_000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-minutes", type=int, default=15)
    parser.add_argument("--log-interval-steps", type=int, default=10)
    parser.add_argument("--evaluation-batches", type=int, default=128)
    arguments = parser.parse_args()
    if arguments.session_hours <= 0:
        raise ValueError("session-hours must be positive")
    if arguments.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    positive_values = (
        arguments.sequence_length,
        arguments.opencode_priority_records,
        arguments.opencode_general_records,
        arguments.codefeedback_records,
        arguments.magicoder_records,
        arguments.math_records,
        arguments.opencode_scan_limit,
        arguments.source_scan_limit,
        arguments.shuffle_buffer_size,
        arguments.checkpoint_minutes,
        arguments.log_interval_steps,
        arguments.evaluation_batches,
    )
    if any(value < 1 for value in positive_values):
        raise ValueError("all counts and intervals must be positive")
    return RunConfiguration(
        results_directory=Path(arguments.results_dir).expanduser().resolve(),
        session_seconds=round(arguments.session_hours * 60 * 60),
        data_seed=arguments.data_seed,
        model_seed=arguments.model_seed,
        sequence_length=arguments.sequence_length,
        quotas=CorpusQuotas(
            arguments.opencode_priority_records,
            arguments.opencode_general_records,
            arguments.codefeedback_records,
            arguments.magicoder_records,
            arguments.math_records,
        ),
        opencode_scan_limit=arguments.opencode_scan_limit,
        source_scan_limit=arguments.source_scan_limit,
        shuffle_buffer_size=arguments.shuffle_buffer_size,
        num_workers=arguments.num_workers,
        checkpoint_interval_seconds=arguments.checkpoint_minutes * 60,
        log_interval_steps=arguments.log_interval_steps,
        evaluation_batches=arguments.evaluation_batches,
    )


def main() -> None:
    configuration = parse_arguments()
    internal_tests = run_internal_contract_tests()
    device, environment = configure_a100()
    configuration.results_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(configuration.results_directory / "environment.json", environment)
    atomic_write_json(configuration.results_directory / "internal_contract_tests.json", internal_tests)
    records, corpus_manifest = build_or_load_corpus(configuration)
    training_records, validation_records = split_records(records, configuration.data_seed)
    training_dataset = MaterializedCausalByteDataset(training_records, configuration.sequence_length)
    validation_dataset = MaterializedCausalByteDataset(validation_records, configuration.sequence_length)
    dataset_report = {
        "records": len(records),
        "training_records": len(training_records),
        "validation_records": len(validation_records),
        "training_chunks": len(training_dataset),
        "validation_chunks": len(validation_dataset),
        "sequence_length": configuration.sequence_length,
        "corpus_manifest": corpus_manifest,
    }
    atomic_write_json(configuration.results_directory / "dataset_report.json", dataset_report)
    session_report = run_training(
        configuration,
        corpus_manifest,
        training_dataset,
        validation_dataset,
        device,
        environment,
    )
    final_report = {"environment": environment, "dataset": dataset_report, "session": session_report}
    atomic_write_json(configuration.results_directory / "final_report.json", final_report)
    print(json.dumps(final_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
