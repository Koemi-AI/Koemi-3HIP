from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import torch

from koemi.configuration.settings import ModelSettings, PAD_TOKEN_ID, TrainingSettings
from koemi.data.contracts import DatasetRecord
from koemi.data.tokenizer import ByteTokenizer
from koemi.model.network import KoemiModel
from koemi.training.checkpoints import CheckpointStore
from koemi.training.dataset import CausalByteDataset, IGNORE_TARGET_ID, create_training_loader
from koemi.training.generation import generate_text
from koemi.training.trainer import Trainer


def build_records() -> tuple[DatasetRecord, ...]:
    return (
        DatasetRecord("one", "Complete the sequence", "The answer repeats the word.", "alpha alpha", {}),
        DatasetRecord("two", "Complete the sequence", None, "beta beta", {}),
    )


def build_model(**overrides) -> KoemiModel:
    settings = dict(embedding_size=16, memory_features=4, local_memory_size=4, expert_count=1)
    settings.update(overrides)
    return KoemiModel(ModelSettings(**settings))


class TrainingTests(unittest.TestCase):
    def test_trains_saves_loads_and_generates_with_moe_and_thinking(self) -> None:
        torch.manual_seed(0)
        tokenizer = ByteTokenizer()
        dataset = CausalByteDataset(build_records(), sequence_length=64)
        loader = create_training_loader(dataset, batch_size=2)
        model = build_model()
        training_settings = TrainingSettings(
            sequence_length=64,
            batch_size=2,
            epochs=1,
            learning_rate=0.001,
            device="cpu",
            thinking_loss_weight=2.0,
        )
        result = Trainer(logging.getLogger("koemi-test")).train(model, loader, training_settings)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"
            store = CheckpointStore()
            store.save(checkpoint_path, model)
            loaded_checkpoint = store.load(checkpoint_path)
            generated_text = generate_text(
                loaded_checkpoint.model,
                tokenizer,
                "A",
                max_new_bytes=2,
                temperature=1.0,
                device="cpu",
            )
        self.assertGreater(result.supervised_token_count, 0)
        self.assertGreater(result.token_count, result.supervised_token_count)
        self.assertEqual((result.token_count,), result.expert_activation_counts)
        self.assertGreater(result.mean_thinking_loss, 0.0)
        self.assertTrue(generated_text.startswith("A"))

    def test_sequential_training_uses_the_same_objective_contract(self) -> None:
        torch.manual_seed(0)
        dataset = CausalByteDataset(build_records(), sequence_length=64)
        loader = create_training_loader(dataset, batch_size=2)
        result = Trainer(logging.getLogger("koemi-test")).train(
            build_model(expert_count=0),
            loader,
            TrainingSettings(sequence_length=64, batch_size=2, epochs=1, device="cpu", execution_mode="sequential"),
        )
        self.assertGreater(result.mean_loss, 0.0)
        self.assertEqual((), result.expert_activation_counts)

    def test_training_settings_reject_negative_thinking_weight(self) -> None:
        with self.assertRaises(ValueError):
            TrainingSettings(thinking_loss_weight=-1.0)

    def test_training_settings_reject_invalid_batching_limits(self) -> None:
        with self.assertRaises(ValueError):
            TrainingSettings(max_batch_tokens=0)
        with self.assertRaises(ValueError):
            TrainingSettings(max_batch_tokens=1.5)
        with self.assertRaises(ValueError):
            TrainingSettings(length_bucket_size=False)

    def test_length_aware_loader_enforces_padded_budget_and_buckets(self) -> None:
        dataset = CausalByteDataset(build_records(), sequence_length=8)
        loader = create_training_loader(
            dataset,
            batch_size=3,
            shuffle=False,
            max_batch_tokens=16,
            length_bucket_size=4,
        )

        for batch in loader:
            lengths = [
                int((row != PAD_TOKEN_ID).sum())
                for row in batch["input_ids"]
            ]
            self.assertLessEqual(batch["input_ids"].numel(), 16)
            self.assertLess(max(lengths) - min(lengths), 4)

    def test_accumulation_scheduler_validation_and_precision_metrics(self) -> None:
        torch.manual_seed(0)
        dataset = CausalByteDataset(build_records(), sequence_length=32)
        counting_loader = create_training_loader(dataset, batch_size=1, shuffle=False)
        supervised_batches = sum(
            int((batch["target_ids"] != IGNORE_TARGET_ID).any()) for batch in counting_loader
        )
        training_loader = create_training_loader(dataset, batch_size=1, generator=torch.Generator().manual_seed(0))
        validation_loader = create_training_loader(dataset, batch_size=2, shuffle=False)
        result = Trainer(logging.getLogger("koemi-test")).train(
            build_model(expert_count=0),
            training_loader,
            TrainingSettings(
                sequence_length=32,
                batch_size=1,
                epochs=1,
                device="cpu",
                precision="auto",
                gradient_accumulation_steps=2,
                warmup_steps=1,
                label_smoothing=0.05,
            ),
            validation_loader,
        )
        self.assertEqual("fp32", result.precision)
        self.assertEqual((supervised_batches + 1) // 2, result.optimizer_steps)
        self.assertIsNotNone(result.validation_loss)
        self.assertIsNotNone(result.validation_perplexity)
        self.assertGreater(result.tokens_per_second, 0.0)

    def test_fp16_training_rejects_cpu(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            Trainer.resolve_precision(torch.device("cpu"), "fp16")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_auto_precision_training_path(self) -> None:
        dataset = CausalByteDataset(build_records(), sequence_length=32)
        loader = create_training_loader(dataset, batch_size=2, pin_memory=True)
        result = Trainer(logging.getLogger("koemi-test")).train(
            build_model(expert_count=0),
            loader,
            TrainingSettings(sequence_length=32, batch_size=2, epochs=1, device="cuda", precision="auto"),
        )
        self.assertIn(result.precision, {"bf16", "fp16"})
        self.assertGreater(result.optimizer_steps, 0)
