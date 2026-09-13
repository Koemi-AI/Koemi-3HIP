from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings
from koemi.model.network import KoemiModel
from koemi.training.dataset import IGNORE_TARGET_ID
from koemi.training.objective import calculate_training_objective


class Koemi3HIPObjectiveTests(unittest.TestCase):
    def build_model(self) -> KoemiModel:
        return KoemiModel(ModelSettings(embedding_size=16, memory_features=4, local_memory_size=4))

    def test_thinking_weight_changes_only_the_weighted_objective(self) -> None:
        model = self.build_model()
        output = model(torch.tensor([[65, 66, 67, 68]], dtype=torch.long))
        target_ids = torch.tensor([[66, 67, 68, 69]], dtype=torch.long)
        thinking_mask = torch.tensor([[False, True, True, False]])
        ordinary = calculate_training_objective(output, target_ids, thinking_mask, 1.0)
        emphasized = calculate_training_objective(output, target_ids, thinking_mask, 3.0)
        self.assertAlmostEqual(float(ordinary.task_loss.detach()), float(emphasized.task_loss.detach()), places=6)
        self.assertAlmostEqual(float(ordinary.thinking_loss.detach()), float(emphasized.thinking_loss.detach()), places=6)
        self.assertNotAlmostEqual(float(ordinary.total_loss.detach()), float(emphasized.total_loss.detach()), places=6)

    def test_no_thinking_keeps_the_total_loss_equal_to_task_loss(self) -> None:
        model = self.build_model()
        output = model(torch.tensor([[65, 66, 67]], dtype=torch.long))
        target_ids = torch.tensor([[66, 67, 68]], dtype=torch.long)
        thinking_mask = torch.zeros_like(target_ids, dtype=torch.bool)
        objective = calculate_training_objective(output, target_ids, thinking_mask, 4.0)
        self.assertAlmostEqual(float(objective.total_loss.detach()), float(objective.task_loss.detach()), places=6)
        self.assertEqual(0.0, float(objective.thinking_loss))

    def test_objective_ignores_unsupervised_positions(self) -> None:
        model = self.build_model()
        output = model(torch.tensor([[65, 66, 67, 68]], dtype=torch.long))
        target_ids = torch.tensor([[66, IGNORE_TARGET_ID, IGNORE_TARGET_ID, 69]], dtype=torch.long)
        thinking_mask = torch.tensor([[False, True, True, False]])
        objective = calculate_training_objective(output, target_ids, thinking_mask, 1.0)
        self.assertTrue(torch.isfinite(objective.total_loss))

    def test_objective_rejects_a_fully_masked_batch(self) -> None:
        model = self.build_model()
        output = model(torch.tensor([[65, 66]], dtype=torch.long))
        target_ids = torch.full((1, 2), IGNORE_TARGET_ID, dtype=torch.long)
        with self.assertRaises(ValueError):
            calculate_training_objective(output, target_ids, torch.zeros_like(target_ids, dtype=torch.bool), 1.0)
