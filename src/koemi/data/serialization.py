from __future__ import annotations

from dataclasses import dataclass

from koemi.data.contracts import (
    INPUT_TAG,
    OUTPUT_TAG,
    SYSTEM_TAG,
    THINKING_TAG,
    DatasetRecord,
    reject_reserved_tags,
)


SYSTEM_MARKER = f"{SYSTEM_TAG}\n"
INPUT_MARKER = f"{INPUT_TAG}\n"
THINKING_MARKER = f"\n{THINKING_TAG}\n"
OUTPUT_MARKER = f"\n{OUTPUT_TAG}\n"


@dataclass(frozen=True)
class SerializedRecord:
    token_bytes: bytes
    supervised_positions: tuple[bool, ...]
    thinking_positions: tuple[bool, ...]


def reject_prompt_tags(system_text: str | None, user_text: str) -> None:
    reject_reserved_tags("system", system_text)
    reject_reserved_tags("prompt", user_text)


def system_prefix(system_text: str | None) -> str:
    if system_text is None:
        return ""
    return f"{SYSTEM_MARKER}{system_text}\n"


def build_answer_prompt(system_text: str | None, user_text: str) -> str:
    reject_prompt_tags(system_text, user_text)
    return f"{system_prefix(system_text)}{INPUT_MARKER}{user_text}{OUTPUT_MARKER}"


def build_thinking_prompt(system_text: str | None, user_text: str) -> str:
    reject_prompt_tags(system_text, user_text)
    return f"{system_prefix(system_text)}{INPUT_MARKER}{user_text}{THINKING_MARKER}"


def strip_prompt(generated_text: str, prompt: str) -> str:
    if not generated_text.startswith(prompt):
        raise ValueError("the generated text does not start with the prompt it was conditioned on")
    return generated_text[len(prompt) :]


def serialize_record(record: DatasetRecord) -> SerializedRecord:
    prefix = system_prefix(record.system_text).encode("utf-8")
    if record.output_text is None:
        body = record.input_text.encode("utf-8")
        return SerializedRecord(
            prefix + body,
            tuple(False for _ in prefix) + tuple(True for _ in body),
            tuple(False for _ in prefix + body),
        )
    segments: list[tuple[bytes, bool, bool]] = [
        (prefix, False, False),
        (INPUT_MARKER.encode("utf-8"), False, False),
        (record.input_text.encode("utf-8"), False, False),
    ]
    if record.thinking_text is not None:
        segments.extend(
            [
                (THINKING_MARKER.encode("utf-8"), False, False),
                (record.thinking_text.encode("utf-8"), True, True),
            ]
        )
    segments.extend(
        [
            (OUTPUT_MARKER.encode("utf-8"), False, False),
            (record.output_text.encode("utf-8"), True, False),
        ]
    )
    token_bytes = b"".join(segment for segment, _, _ in segments)
    supervised_positions = tuple(
        is_supervised for segment, is_supervised, _ in segments for _ in segment
    )
    thinking_positions = tuple(
        is_thinking for segment, _, is_thinking in segments for _ in segment
    )
    return SerializedRecord(token_bytes, supervised_positions, thinking_positions)


def supervised_prefix_bytes(record: DatasetRecord) -> bytes:
    serialized = serialize_record(record)
    for position, is_supervised in enumerate(serialized.supervised_positions):
        if is_supervised:
            return serialized.token_bytes[:position]
    return serialized.token_bytes
