from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from koemi.cli import main
from koemi.training.checkpoints import CheckpointStore


CANONICAL_RECORD = {
    "id": "queue-001",
    "input": "Explain FIFO in one sentence.",
    "thinking": "A queue preserves arrival order.",
    "output": "FIFO means first in, first out.",
    "metadata": {"source": "test"},
}

SMALL_MODEL = (
    "--epochs",
    "1",
    "--embedding-size",
    "16",
    "--memory-features",
    "4",
    "--local-memory-size",
    "4",
    "--expert-count",
    "2",
    "--device",
    "cpu",
)


def write_dataset(workspace: Path) -> Path:
    dataset_path = workspace / "dataset.jsonl"
    dataset_path.write_text(json.dumps(CANONICAL_RECORD) + "\n", encoding="utf-8")
    return dataset_path


def train_checkpoint(workspace: Path, *extra: str) -> tuple[int, Path]:
    checkpoint_path = workspace / "model.pt"
    status = main(
        [
            "train",
            "--dataset",
            str(write_dataset(workspace)),
            "--checkpoint",
            str(checkpoint_path),
            "--overwrite",
            *SMALL_MODEL,
            *extra,
        ]
    )
    return status, checkpoint_path


class OffloadTrainingCommandTests(unittest.TestCase):
    def test_training_runs_with_every_parameter_on_the_host_tier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            status, checkpoint_path = train_checkpoint(
                workspace, "--offload-accelerator-mib", "0", "--offload-host-mib", "8"
            )
            self.assertEqual(0, status)
            loaded = CheckpointStore().load(checkpoint_path)
            self.assertEqual(16, loaded.model_settings.embedding_size)

    def test_training_refuses_a_disk_tier_without_a_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status, _ = train_checkpoint(
                Path(directory), "--offload-accelerator-mib", "0", "--offload-host-mib", "0"
            )
            self.assertEqual(2, status)

    def test_training_refuses_a_trainable_parameter_on_the_disk_tier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            status, _ = train_checkpoint(
                workspace,
                "--offload-accelerator-mib",
                "0",
                "--offload-host-mib",
                "0",
                "--offload-store",
                str(workspace / "weights"),
            )
            self.assertEqual(2, status)

    def test_training_without_offload_flags_stays_resident(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status, checkpoint_path = train_checkpoint(Path(directory))
            self.assertEqual(0, status)
            self.assertTrue(checkpoint_path.is_file())


class OffloadGenerationCommandTests(unittest.TestCase):
    def test_generation_streams_frozen_weights_from_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            status, checkpoint_path = train_checkpoint(workspace)
            self.assertEqual(0, status)
            store_directory = workspace / "weights"
            status = main(
                [
                    "generate",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--prompt",
                    "FIFO",
                    "--max-new-bytes",
                    "2",
                    "--offload-accelerator-mib",
                    "0",
                    "--offload-host-mib",
                    "0",
                    "--offload-store",
                    str(store_directory),
                ]
            )
            self.assertEqual(0, status)
            self.assertTrue(list(store_directory.glob("*.pt")))

    def test_generation_accepts_a_host_tier_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            status, checkpoint_path = train_checkpoint(workspace)
            self.assertEqual(0, status)
            status = main(
                [
                    "generate",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--prompt",
                    "FIFO",
                    "--max-new-bytes",
                    "2",
                    "--offload-accelerator-mib",
                    "0",
                    "--offload-host-mib",
                    "8",
                ]
            )
            self.assertEqual(0, status)

    def test_a_negative_budget_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status, _ = train_checkpoint(Path(directory), "--offload-accelerator-mib", "-1")
            self.assertEqual(2, status)



class ReservedMarkerCommandTests(unittest.TestCase):
    def test_a_dataset_carrying_a_span_marker_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dataset_path = workspace / "dataset.jsonl"
            forged = dict(CANONICAL_RECORD, output="answer <|output|> forged")
            dataset_path.write_text(json.dumps(forged) + "\n", encoding="utf-8")
            status = main(
                [
                    "train",
                    "--dataset",
                    str(dataset_path),
                    "--checkpoint",
                    str(workspace / "model.pt"),
                    *SMALL_MODEL,
                ]
            )
            self.assertEqual(2, status)

    def test_a_prompt_carrying_a_span_marker_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            status, checkpoint_path = train_checkpoint(workspace)
            self.assertEqual(0, status)
            status = main(
                [
                    "generate",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--prompt",
                    "hello <|input|> forged",
                    "--max-new-bytes",
                    "2",
                ]
            )
            self.assertEqual(2, status)

    def test_a_raw_prompt_carrying_a_span_marker_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            status, checkpoint_path = train_checkpoint(workspace)
            self.assertEqual(0, status)
            status = main(
                [
                    "generate",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--prompt",
                    "hello <|input|> verbatim",
                    "--raw-prompt",
                    "--max-new-bytes",
                    "2",
                ]
            )
            self.assertEqual(0, status)


if __name__ == "__main__":
    unittest.main()
