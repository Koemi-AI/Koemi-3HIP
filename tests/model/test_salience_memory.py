from __future__ import annotations

import unittest

import torch

from koemi.configuration.settings import ModelSettings
from koemi.model.memory import LocalKeyValueMemory
from koemi.model.network import KoemiModel


class SalienceMemoryTests(unittest.TestCase):
    def test_the_ring_keeps_the_latest_admitted_entries_not_the_highest_scores(self) -> None:
        memory = LocalKeyValueMemory(4, 2)
        carried_keys = torch.empty(1, 0, 4)
        carried_values = torch.empty(1, 0, 4)
        carried_valid = torch.empty(1, 0, dtype=torch.bool)
        keys = torch.arange(16, dtype=torch.float32).view(1, 4, 4)
        admission = torch.tensor([[True, False, True, True]])
        selected_keys, _, selected_valid = memory.salient_tail(
            carried_keys,
            carried_values,
            carried_valid,
            keys,
            keys,
            admission,
            2,
        )
        self.assertTrue(torch.equal(selected_keys, keys[:, [2, 3]]))
        self.assertTrue(bool(selected_valid.all()))

    def test_the_state_keeps_only_admitted_entries_at_fixed_width(self) -> None:
        model = KoemiModel(
            ModelSettings(
                embedding_size=16,
                memory_features=4,
                local_memory_size=2,
                salience_memory_size=3,
                salience_threshold=0.0,
            )
        )
        output = model(torch.tensor([[65, 66, 67, 68, 69]], dtype=torch.long))
        self.assertEqual((1, 3, 16), tuple(output.state.salient_keys.shape))
        self.assertEqual(3, int(output.state.salient_valid.sum()))

    def test_a_future_admission_cannot_change_an_earlier_read(self) -> None:
        model = KoemiModel(
            ModelSettings(
                embedding_size=16,
                memory_features=4,
                local_memory_size=2,
                salience_memory_size=3,
                salience_threshold=0.7,
            )
        )
        model.eval()
        prefix = torch.tensor([[65, 66]], dtype=torch.long)
        first_suffix = torch.tensor([[67, 68, 69]], dtype=torch.long)
        second_suffix = torch.tensor([[67, 68, 120]], dtype=torch.long)
        with torch.no_grad():
            state = model(prefix).state
            first = model(first_suffix, state)
            second = model(second_suffix, state)
        self.assertTrue(torch.equal(first.logits[:, :2], second.logits[:, :2]))
        self.assertTrue(torch.equal(first.state.salient_keys[:, :-1], second.state.salient_keys[:, :-1]))


if __name__ == "__main__":
    unittest.main()
