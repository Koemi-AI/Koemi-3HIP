from __future__ import annotations

from dataclasses import dataclass

from koemi.data.contracts import DatasetRecord


SYSTEM_MARKER = "<|system|>\n"
INPUT_MARKER = "<|input|>\n"
THINKING_MARKER = "\n<|thinking|>\n"
OUTPUT_MARKER = "\n<|output|>\n"


@dataclass(frozen=True)
class SerializedRecord:
    token_bytes: bytes
    supervised_positions: tuple[bool, ...]
    thinking_positions: tuple[bool, ...]


def system_prefix(system_text: str | None) -> str:
    if system_text is None:
        return ""
    return f"{SYSTEM_MARKER}{system_text}\n"


def build_answer_prompt(system_text: str | None, user_text: str) -> str:
    return f"{system_prefix(system_text)}{INPUT_MARKER}{user_text}{OUTPUT_MARKER}"


def build_thinking_prompt(system_text: str | None, user_text: str) -> str:
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
