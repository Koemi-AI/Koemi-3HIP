from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetRecord:
    identifier: str
    input_text: str
    thinking_text: str | None
    output_text: str | None
    metadata: dict[str, Any]
    system_text: str | None = None
