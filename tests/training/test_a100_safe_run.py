from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from koemi.training.a100_safe_run import (
    COST_PER_HOUR_USD,
    SafeA100Plan,
    build_run_configuration,
    main,
    parse_arguments,
    reserve_budget,
    settle_budget,
)


class SafeA100RunTests(unittest.TestCase):
    def test_default_plan_is_small_and_budgeted(self) -> None:
        plan = SafeA100Plan(Path("results"))

        self.assertEqual(45_000, plan.target_record_count)
        self.assertEqual(7.5, plan.session_hours)
        self.assertEqual(25, plan.maximum_sessions)
        self.assertEqual(47.48, round(plan.requested_cost_usd, 2))
        self.assertEqual(1190.04, round(plan.budget_cost_usd, 2))
        self.assertEqual(256, plan.sequence_length)

    def test_plan_maps_to_the_existing_checkpointed_runner(self) -> None:
        plan = SafeA100Plan(Path("results"))
        configuration = build_run_configuration(plan)

        self.assertEqual(round(7.5 * 60 * 60), configuration.session_seconds)
        self.assertEqual(plan.quotas, configuration.quotas)
        self.assertEqual(10 * 60, configuration.checkpoint_interval_seconds)

    def test_train_parser_requires_an_explicit_budget_confirmation_value(self) -> None:
        mode, plan, confirmation = parse_arguments(
            ["--mode", "train", "--budget-hours", "188", "--session-hours", "7.5"]
        )

        self.assertEqual("train", mode)
        self.assertEqual(188.0, plan.budget_hours)
        self.assertIsNone(confirmation)

    def test_train_gate_stops_before_any_paid_work(self) -> None:
        with self.assertRaisesRegex(ValueError, "confirm-budget-hours"):
            main(["--mode", "train", "--results-dir", "unused-safe-results"])

    def test_budget_reservation_is_conservative_and_settlement_is_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            plan = SafeA100Plan(Path(temporary_directory), budget_hours=8.0, session_hours=7.5)
            reserve_budget(plan)
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                reserve_budget(plan)
            state = settle_budget(plan, 1.25, "failed")
            state_path = plan.results_directory / "safe_budget_state.json"
            persisted = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(0.0, state["reserved_hours"])
        self.assertEqual(1.25, state["consumed_hours"])
        self.assertEqual(1, len(state["sessions"]))
        self.assertEqual(7.91, state["sessions"][0]["cost_usd"])
        self.assertEqual(persisted, state)
        self.assertEqual(6.33, COST_PER_HOUR_USD)


if __name__ == "__main__":
    unittest.main()
