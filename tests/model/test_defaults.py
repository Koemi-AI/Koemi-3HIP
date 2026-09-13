from __future__ import annotations

import unittest

from koemi.configuration.settings import ModelSettings
from koemi.model.network import KoemiModel


class ModelDefaultTests(unittest.TestCase):
    def test_the_default_skips_refine_and_uses_the_measured_chunk(self) -> None:
        settings = ModelSettings()
        model = KoemiModel(settings)
        self.assertEqual("no_refine", settings.ablation)
        self.assertEqual(128, settings.scan_chunk)
        self.assertIsNone(model.memory_refine_gate)

    def test_salience_settings_reject_invalid_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "salience_memory_size"):
            ModelSettings(salience_memory_size=0)
        for threshold in (-0.01, 1.01):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "salience_threshold"):
                ModelSettings(salience_threshold=threshold)


if __name__ == "__main__":
    unittest.main()
