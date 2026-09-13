from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SYSTEM_TAG = "<|system|>"
INPUT_TAG = "<|input|>"
THINKING_TAG = "<|thinking|>"
OUTPUT_TAG = "<|output|>"
RESERVED_TAGS = (SYSTEM_TAG, INPUT_TAG, THINKING_TAG, OUTPUT_TAG)


class DatasetValidationError(ValueError):
    pass


def reject_reserved_tags(field_name: str, value: str | None) -> None:
    if value is None:
        return
    for tag in RESERVED_TAGS:
        if tag in value:
            raise DatasetValidationError(
                f"field '{field_name}' contains the reserved span marker {tag}"
            )


@dataclass(frozen=True)
class DatasetRecord:
    identifier: str
    input_text: str
    thinking_text: str | None
    output_text: str | None
    metadata: dict[str, Any]
    system_text: str | None = None

    def __post_init__(self) -> None:
        reject_reserved_tags("input", self.input_text)
        reject_reserved_tags("thinking", self.thinking_text)
        reject_reserved_tags("output", self.output_text)
        reject_reserved_tags("system", self.system_text)
