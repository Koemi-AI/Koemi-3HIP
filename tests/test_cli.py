from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from koemi.cli import create_parser, main, write_utf8


CANONICAL_RECORD = {
    "id": "queue-001",
    "input": "Explain FIFO in one sentence.",
    "thinking": "A queue preserves arrival order.",
    "output": "FIFO means first in, first out.",
    "metadata": {"source": "test"},
}


class CliOutputTests(unittest.TestCase):
    def test_writes_unicode_output_as_utf8_bytes(self) -> None:
        output = io.BytesIO()
        stdout = type("BufferedStdout", (), {"buffer": output})()
        with patch("koemi.cli.sys.stdout", stdout):
            write_utf8("prefix ï¿½")
        self.assertEqual(output.getvalue(), "prefix ï¿½\n".encode("utf-8"))


class CliCommandTests(unittest.TestCase):
    def test_generate_parser_accepts_bulk_prefix_cache_arguments(self) -> None:
        arguments = create_parser().parse_args(
            [
                "generate",
                "--checkpoint",
                "model.pt",
                "--prompt",
                "hello",
                "--bulk-prefix-cache",
                "cache",
                "--bulk-prefix-cache-namespace",
                "test",
                "--bulk-prefix-cache-block-size",
                "32",
            ]
        )

        self.assertEqual("cache", arguments.bulk_prefix_cache)
        self.assertEqual("test", arguments.bulk_prefix_cache_namespace)
        self.assertEqual(32, arguments.bulk_prefix_cache_block_size)

    def test_trains_and_generates_with_koemi_3hip_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            dataset_path = workspace / "dataset.jsonl"
            dataset_path.write_text(json.dumps(CANONICAL_RECORD) + "\n", encoding="utf-8")
            checkpoint_path = workspace / "model.pt"
            train_status = main(
                [
                    "train",
                    "--dataset",
                    str(dataset_path),
                    "--checkpoint",
                    str(checkpoint_path),
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
                    "--thinking-loss-weight",
                    "2.0",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(0, train_status)
            self.assertTrue(checkpoint_path.exists())
            output = io.BytesIO()
            stdout = type("BufferedStdout", (), {"buffer": output})()
            with patch("koemi.cli.sys.stdout", stdout):
                generate_status = main(
                    [
                        "generate",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--prompt",
                        "FIFO",
                        "--max-new-bytes",
                        "4",
                        "--raw-prompt",
                    ]
                )
            self.assertEqual(0, generate_status)
            self.assertTrue(output.getvalue().startswith(b"FIFO"))

            bulk_output = io.BytesIO()
            bulk_stdout = type("BufferedStdout", (), {"buffer": bulk_output})()
            with patch("koemi.cli.sys.stdout", bulk_stdout):
                bulk_generate_status = main(
                    [
                        "generate",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--prompt",
                        "FIFO",
                        "--max-new-bytes",
                        "4",
                        "--raw-prompt",
                        "--bulk-prefix-cache",
                        str(workspace / "bulk-prefix"),
                        "--bulk-prefix-cache-namespace",
                        "cli-test",
                        "--bulk-prefix-cache-block-size",
                        "2",
                    ]
                )
            self.assertEqual(0, bulk_generate_status)
            self.assertTrue(bulk_output.getvalue().startswith(b"FIFO"))

    def test_old_routing_flags_are_removed(self) -> None:
        with self.assertRaises(SystemExit):
            main(["train", "--dataset", "missing.jsonl", "--checkpoint", "model.pt", "--routing-mode", "hard"])

    def test_sequential_execution_mode_trains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            dataset_path = workspace / "dataset.jsonl"
            dataset_path.write_text(json.dumps(CANONICAL_RECORD) + "\n", encoding="utf-8")
            status = main(
                [
                    "train",
                    "--dataset",
                    str(dataset_path),
                    "--checkpoint",
                    str(workspace / "model.pt"),
                    "--epochs",
                    "1",
                    "--embedding-size",
                    "16",
                    "--memory-features",
                    "4",
                    "--local-memory-size",
                    "4",
                    "--execution-mode",
                    "sequential",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(0, status)

    def test_generate_wraps_the_prompt_in_the_training_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            dataset_path = workspace / "dataset.jsonl"
            dataset_path.write_text(
                json.dumps(dict(CANONICAL_RECORD, system="Answer briefly.")) + "\n",
                encoding="utf-8",
            )
            checkpoint_path = workspace / "model.pt"
            train_status = main(
                [
                    "train",
                    "--dataset",
                    str(dataset_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--epochs",
                    "1",
                    "--embedding-size",
                    "16",
                    "--memory-features",
                    "4",
                    "--local-memory-size",
                    "4",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(0, train_status)
            output = io.BytesIO()
            stdout = type("BufferedStdout", (), {"buffer": output})()
            with patch("koemi.cli.sys.stdout", stdout):
                generate_status = main(
                    [
                        "generate",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--system",
                        "Answer briefly.",
                        "--prompt",
                        "Explain FIFO.",
                        "--max-new-bytes",
                        "4",
                    ]
                )
            self.assertEqual(0, generate_status)
            written = output.getvalue()
            self.assertNotIn(b"<|system|>", written)
            self.assertNotIn(b"<|input|>", written)
            self.assertNotIn(b"Explain FIFO.", written)

    def test_generate_refuses_a_system_prompt_with_a_raw_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            dataset_path = workspace / "dataset.jsonl"
            dataset_path.write_text(json.dumps(CANONICAL_RECORD) + "\n", encoding="utf-8")
            checkpoint_path = workspace / "model.pt"
            main(
                [
                    "train",
                    "--dataset",
                    str(dataset_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--epochs",
                    "1",
                    "--embedding-size",
                    "16",
                    "--memory-features",
                    "4",
                    "--local-memory-size",
                    "4",
                    "--device",
                    "cpu",
                ]
            )
            status = main(
                [
                    "generate",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--prompt",
                    "FIFO",
                    "--system",
                    "Answer briefly.",
                    "--raw-prompt",
                    "--max-new-bytes",
                    "2",
                ]
            )
            self.assertEqual(2, status)
