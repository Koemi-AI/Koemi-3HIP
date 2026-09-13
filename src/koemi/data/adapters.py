from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from koemi.data.contracts import DatasetRecord, DatasetValidationError


CANONICAL_FORMAT = "canonical"
ALPACA_FORMAT = "alpaca"
SHAREGPT_FORMAT = "sharegpt"
AUTO_FORMAT = "auto"
SUPPORTED_DATASET_FORMATS = (AUTO_FORMAT, CANONICAL_FORMAT, ALPACA_FORMAT, SHAREGPT_FORMAT)


class RecordAdapter(Protocol):
    name: str

    def matches(self, raw_record: Mapping[str, Any]) -> bool:
        ...

    def adapt(self, raw_record: Mapping[str, Any], fallback_identifier: str) -> DatasetRecord:
        ...


def require_string(raw_value: Any, field_name: str, allow_empty: bool = True) -> str:
    if not isinstance(raw_value, str):
        raise DatasetValidationError(f"field '{field_name}' must be a string")
    if not allow_empty and not raw_value.strip():
        raise DatasetValidationError(f"field '{field_name}' must not be empty")
    return raw_value


def optional_string(raw_value: Any, field_name: str) -> str | None:
    if raw_value is None:
        return None
    return require_string(raw_value, field_name)


def read_identifier(raw_record: Mapping[str, Any], fallback_identifier: str, required: bool) -> str:
    if "id" not in raw_record:
        if required:
            raise DatasetValidationError("field 'id' is required")
        return fallback_identifier
    return require_string(raw_record["id"], "id", allow_empty=False)


def read_metadata(raw_record: Mapping[str, Any]) -> dict[str, Any]:
    raw_metadata = raw_record.get("metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise DatasetValidationError("field 'metadata' must be an object")
    if not all(isinstance(key, str) for key in raw_metadata):
        raise DatasetValidationError("field 'metadata' must use string keys")
    return dict(raw_metadata)


@dataclass(frozen=True)
class CanonicalRecordAdapter:
    name: str = CANONICAL_FORMAT

    def matches(self, raw_record: Mapping[str, Any]) -> bool:
        return (
            "input" in raw_record
            and ("output" in raw_record or "thinking" in raw_record)
            and "instruction" not in raw_record
            and "conversations" not in raw_record
        )

    def adapt(self, raw_record: Mapping[str, Any], fallback_identifier: str) -> DatasetRecord:
        identifier = read_identifier(raw_record, fallback_identifier, required=True)
        input_text = require_string(raw_record.get("input"), "input")
        thinking_text = optional_string(raw_record.get("thinking"), "thinking")
        output_text = optional_string(raw_record.get("output"), "output")
        if output_text is None and thinking_text is not None:
            raise DatasetValidationError("field 'thinking' requires a string 'output'")
        if not input_text and output_text is None:
            raise DatasetValidationError("record must contain input text or output text")
        return DatasetRecord(
            identifier,
            input_text,
            thinking_text,
            output_text,
            read_metadata(raw_record),
            optional_string(raw_record.get("system"), "system"),
        )


@dataclass(frozen=True)
class AlpacaRecordAdapter:
    name: str = ALPACA_FORMAT

    def matches(self, raw_record: Mapping[str, Any]) -> bool:
        return "instruction" in raw_record and "output" in raw_record

    def adapt(self, raw_record: Mapping[str, Any], fallback_identifier: str) -> DatasetRecord:
        instruction = require_string(raw_record.get("instruction"), "instruction", allow_empty=False)
        input_suffix = optional_string(raw_record.get("input", ""), "input") or ""
        output_text = require_string(raw_record.get("output"), "output")
        input_text = instruction if not input_suffix else f"{instruction}\n\n{input_suffix}"
        return DatasetRecord(
            read_identifier(raw_record, fallback_identifier, required=False),
            input_text,
            optional_string(raw_record.get("thinking"), "thinking"),
            output_text,
            read_metadata(raw_record),
            optional_string(raw_record.get("system"), "system"),
        )


@dataclass(frozen=True)
class ShareGptRecordAdapter:
    name: str = SHAREGPT_FORMAT

    def matches(self, raw_record: Mapping[str, Any]) -> bool:
        return "conversations" in raw_record

    def adapt(self, raw_record: Mapping[str, Any], fallback_identifier: str) -> DatasetRecord:
        raw_conversations = raw_record.get("conversations")
        if not isinstance(raw_conversations, Sequence) or isinstance(raw_conversations, (str, bytes)):
            raise DatasetValidationError("field 'conversations' must be an array")
        turns = [self.read_turn(turn, index) for index, turn in enumerate(raw_conversations)]
        system_turns = [turn.content for turn in turns if turn.role == "system"]
        conversations = [turn for turn in turns if turn.role != "system"]
        assistant_positions = [index for index, turn in enumerate(conversations) if turn.role == "assistant"]
        if not assistant_positions:
            raise DatasetValidationError("field 'conversations' must contain an assistant response")
        response_index = assistant_positions[-1]
        output_text = conversations[response_index].content
        input_turns = conversations[:response_index]
        input_text = "\n".join(f"{turn.role.title()}: {turn.content}" for turn in input_turns)
        system_text = (
            "\n".join(system_turns)
            if system_turns
            else optional_string(raw_record.get("system"), "system")
        )
        if not input_text and system_text is None:
            raise DatasetValidationError("field 'conversations' must contain context before the assistant response")
        return DatasetRecord(
            read_identifier(raw_record, fallback_identifier, required=False),
            input_text,
            optional_string(raw_record.get("thinking"), "thinking"),
            output_text,
            read_metadata(raw_record),
            system_text,
        )

    def read_turn(self, raw_turn: Any, index: int) -> ConversationTurn:
        if not isinstance(raw_turn, Mapping):
            raise DatasetValidationError(f"conversation turn {index} must be an object")
        raw_role = raw_turn.get("role", raw_turn.get("from"))
        raw_content = raw_turn.get("content", raw_turn.get("value"))
        role_aliases = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
        }
        if not isinstance(raw_role, str) or raw_role not in role_aliases:
            raise DatasetValidationError(f"conversation turn {index} has an unsupported role")
        return ConversationTurn(role_aliases[raw_role], require_string(raw_content, f"conversations[{index}] content"))


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str


ADAPTERS: tuple[RecordAdapter, ...] = (
    CanonicalRecordAdapter(),
    AlpacaRecordAdapter(),
    ShareGptRecordAdapter(),
)


def adapt_record(raw_record: Mapping[str, Any], dataset_format: str, fallback_identifier: str) -> tuple[DatasetRecord, str]:
    selected_adapters = ADAPTERS if dataset_format == AUTO_FORMAT else tuple(
        adapter for adapter in ADAPTERS if adapter.name == dataset_format
    )
    if not selected_adapters:
        raise DatasetValidationError(f"unsupported dataset format '{dataset_format}'")
    matching_adapters = [adapter for adapter in selected_adapters if adapter.matches(raw_record)]
    if len(matching_adapters) != 1:
        raise DatasetValidationError("record does not match exactly one supported dataset format")
    adapter = matching_adapters[0]
    return adapter.adapt(raw_record, fallback_identifier), adapter.name
