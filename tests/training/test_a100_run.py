from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from koemi.configuration.settings import ModelSettings
from koemi.data.contracts import DatasetRecord, DatasetValidationError
from koemi.model.network import KoemiModel
from koemi.training.a100_run import (
    DeterministicBatchSampler,
    MaterializedCausalByteDataset,
    RotatingCheckpointStore,
    adapt_codefeedback_row,
    adapt_opencode_row,
    adapt_openr1_math_row,
    build_or_load_corpus,
    collate_materialized_chunks,
    CorpusQuotas,
    RunConfiguration,
    parse_arguments,
    restore_training_payload,
    training_payload,
)
from koemi.training.dataset import CausalByteDataset, IGNORE_TARGET_ID


class A100RunTests(unittest.TestCase):
    def test_opencode_requires_all_passing_tests_and_preserves_source_metadata(self) -> None:
        record = adapt_opencode_row(
            {
                "id": "source-id",
                "input": "Debug this Python traceback.",
                "output": "Inspect the stack trace and write a regression test.",
                "domain": "debugging",
                "average_test_score": "1.0",
                "tests_execution_status": '["pass", "pass"]',
            },
            "fallback",
        )
        self.assertEqual("opencode:source-id", record.identifier)
        self.assertEqual("debugging", record.metadata["domain"])
        with self.assertRaisesRegex(DatasetValidationError, "not exactly 1.0"):
            adapt_opencode_row(
                {
                    "id": "partial",
                    "input": "Task",
                    "output": "Code",
                    "domain": "generic",
                    "average_test_score": "0.9",
                    "tests_execution_status": '["pass"]',
                },
                "fallback",
            )

    def test_math_rows_require_complete_verified_reasoning(self) -> None:
        record = adapt_openr1_math_row(
            {
                "uuid": "math-id",
                "problem": "What is 2 + 2?",
                "answer": "4",
                "generations": ["<think>Two plus two is four.</think>\n4"],
                "correctness_math_verify": [True],
                "is_reasoning_complete": [True],
            },
            "fallback",
        )
        self.assertEqual("Two plus two is four.", record.thinking_text)
        self.assertEqual("4", record.output_text)
        with self.assertRaisesRegex(DatasetValidationError, "no complete Math-Verify-correct"):
            adapt_openr1_math_row(
                {
                    "uuid": "unverified",
                    "problem": "What is 2 + 2?",
                    "answer": "4",
                    "generations": ["<think>Two plus two is four.</think>\n4"],
                    "correctness_math_verify": [False],
                    "is_reasoning_complete": [True],
                },
                "fallback",
            )

    def test_materialized_dataset_matches_repository_chunk_and_target_contract(self) -> None:
        records = (
            DatasetRecord("first", "Prompt", "Reason", "Answer", {}),
            adapt_codefeedback_row(
                {"query": "Fix this Bash error.", "answer": "Check the quoted variable.", "lang": "shell"},
                "second",
            ),
        )
        reference = CausalByteDataset(records, sequence_length=7)
        materialized = MaterializedCausalByteDataset(records, sequence_length=7)
        reference_chunks = [
            chunk
            for chunk in reference.chunks
            if any(target != IGNORE_TARGET_ID for target in chunk.target_ids)
        ]
        self.assertEqual(len(reference_chunks), len(materialized))
        for index, reference_chunk in enumerate(reference_chunks):
            input_bytes, target_bytes, supervised_bytes, thinking_bytes = materialized[index]
            expected_targets = tuple(
                target if supervised else IGNORE_TARGET_ID
                for target, supervised in zip(target_bytes, supervised_bytes, strict=True)
            )
            self.assertEqual(reference_chunk.input_ids, tuple(input_bytes))
            self.assertEqual(reference_chunk.target_ids, expected_targets)
            self.assertEqual(reference_chunk.thinking_mask, tuple(bool(value) for value in thinking_bytes))
        batch = collate_materialized_chunks([materialized[0], materialized[1]])
        self.assertEqual((2, 7), tuple(batch["input_ids"].shape))
        self.assertTrue(bool((batch["target_ids"] != IGNORE_TARGET_ID).any()))

    def test_resumed_sampler_is_the_exact_epoch_suffix(self) -> None:
        full_batches = list(DeterministicBatchSampler(19, 4, data_seed=9, epoch_index=3, start_batch_index=0))
        resumed_batches = list(DeterministicBatchSampler(19, 4, data_seed=9, epoch_index=3, start_batch_index=2))
        self.assertEqual(full_batches[2:], resumed_batches)

    def test_checkpoint_store_recovers_previous_slot_after_newest_slot_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = RotatingCheckpointStore(Path(temporary_directory), "run-signature")
            store.save({"training_state": {"epoch_index": 0}}, optimizer_step=1)
            store.save({"training_state": {"epoch_index": 0}}, optimizer_step=2)
            store.slot_paths[0].write_bytes(b"corrupt")
            loaded = store.load_latest()
        self.assertIsNotNone(loaded.payload)
        assert loaded.payload is not None
        self.assertEqual(1, loaded.payload["optimizer_step"])
        self.assertTrue(loaded.recovery_messages)

    def test_checkpoint_store_accepts_the_full_weights_only_training_payload(self) -> None:
        settings = ModelSettings(
            embedding_size=8,
            memory_features=2,
            local_memory_size=1,
            salience_memory_size=1,
            expert_count=1,
            expert_top_k=1,
        )
        model = KoemiModel(settings)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        payload = training_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            {"epoch_index": 1, "next_batch_index": 2, "tokens_seen": 3},
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = RotatingCheckpointStore(Path(temporary_directory), "payload-signature")
            store.save(payload, optimizer_step=4)
            loaded = store.load_latest()
        self.assertIsNotNone(loaded.payload)
        assert loaded.payload is not None
        restored_model = KoemiModel(settings)
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1.0)
        restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
        restored_state = restore_training_payload(
            loaded.payload,
            restored_model,
            restored_optimizer,
            restored_scheduler,
            restored_scaler,
        )
        self.assertEqual({"epoch_index": 1, "next_batch_index": 2, "tokens_seen": 3}, restored_state)

    def test_incomplete_corpus_artifact_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_directory = Path(temporary_directory)
            (results_directory / "corpus.jsonl").write_text("stale\n", encoding="utf-8")
            configuration = RunConfiguration(
                results_directory=results_directory,
                session_seconds=1,
                data_seed=7,
                model_seed=11,
                sequence_length=16,
                quotas=CorpusQuotas(1, 1, 1, 1, 1),
                opencode_scan_limit=8,
                source_scan_limit=8,
                shuffle_buffer_size=2,
                num_workers=0,
                checkpoint_interval_seconds=1,
                log_interval_steps=1,
                evaluation_batches=1,
            )
            streams = {
                "nvidia/OpenCodeInstruct": [
                    {
                        "id": "priority",
                        "input": "Fix this Python traceback.",
                        "output": "Add a regression test.",
                        "domain": "debugging",
                        "average_test_score": "1.0",
                        "tests_execution_status": '["pass"]',
                    },
                    {
                        "id": "general",
                        "input": "Explain a sorting algorithm.",
                        "output": "Compare the invariants.",
                        "domain": "generic",
                        "average_test_score": "1.0",
                        "tests_execution_status": '["pass"]',
                    },
                ],
                "m-a-p/CodeFeedback-Filtered-Instruction": [
                    {"query": "Fix the shell command.", "answer": "Quote the variable.", "lang": "shell"}
                ],
                "ise-uiuc/Magicoder-Evol-Instruct-110K": [
                    {"instruction": "Write a Python function.", "response": "Return the value."}
                ],
                "open-r1/OpenR1-Math-220k": [
                    {
                        "uuid": "math",
                        "problem": "What is 1 + 1?",
                        "answer": "2",
                        "generations": ["<think>Add one and one.</think>\\n2"],
                        "correctness_math_verify": [True],
                        "is_reasoning_complete": [True],
                    }
                ],
            }

            def fake_stream(dataset_name: str, *_arguments):
                return iter(streams[dataset_name])

            with patch("koemi.training.a100_run.load_streaming_dataset", side_effect=fake_stream):
                records, manifest = build_or_load_corpus(configuration)

        self.assertEqual(5, len(records))
        self.assertTrue(manifest["recovered_incomplete_artifact"])

    def test_cli_default_session_is_nine_hours(self) -> None:
        with patch("sys.argv", ["a100_run", "--results-dir", "results"]):
            configuration = parse_arguments()
        self.assertEqual(9 * 60 * 60, configuration.session_seconds)
