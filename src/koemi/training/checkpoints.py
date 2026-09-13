from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from koemi.configuration.settings import ModelSettings
from koemi.model.network import KoemiModel


CHECKPOINT_FORMAT_VERSION = 6


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: KoemiModel
    model_settings: ModelSettings


class CheckpointStore:
    def save(self, checkpoint_path: str | Path, model: KoemiModel, overwrite: bool = False) -> Path:
        target_path = Path(checkpoint_path).expanduser().resolve()
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"checkpoint already exists: {target_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_settings": model.settings.to_dict(),
            "model_state": model.state_dict(),
        }
        torch.save(payload, target_path)
        return target_path

    def load(self, checkpoint_path: str | Path, device: str = "cpu") -> LoadedCheckpoint:
        source_path = Path(checkpoint_path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"checkpoint file does not exist: {source_path}")
        raw_checkpoint = torch.load(source_path, map_location=device, weights_only=True)
        checkpoint = self.validate_checkpoint(raw_checkpoint)
        model_settings = ModelSettings.from_dict(checkpoint["model_settings"])
        model = KoemiModel(model_settings).to(device)
        model.load_state_dict(checkpoint["model_state"])
        return LoadedCheckpoint(model, model_settings)

    def validate_checkpoint(self, raw_checkpoint: Any) -> dict[str, Any]:
        if not isinstance(raw_checkpoint, dict):
            raise ValueError("checkpoint must contain a dictionary payload")
        if raw_checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("checkpoint format version is not supported")
        if not isinstance(raw_checkpoint.get("model_settings"), dict):
            raise ValueError("checkpoint model settings are invalid")
        if not isinstance(raw_checkpoint.get("model_state"), dict):
            raise ValueError("checkpoint model state is invalid")
        return raw_checkpoint
