from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from koemi.configuration.settings import ModelSettings
from koemi.model.network import KoemiModel
from koemi.runtime.offload import (
    ACCELERATOR_TIER,
    DISK_TIER,
    HOST_TIER,
    DiskParameterStore,
    ModuleTraffic,
    OffloadEngine,
    OffloadError,
    TierBudget,
    calibrate_traffic,
    plan_placement,
)


INPUT_IDS = torch.tensor([[72, 101, 108, 108, 111, 33, 72, 105]], dtype=torch.long)


def build_model(**overrides) -> KoemiModel:
    torch.manual_seed(5)
    settings = dict(embedding_size=16, memory_features=4, local_memory_size=4, expert_count=4)
    settings.update(overrides)
    return KoemiModel(ModelSettings(**settings))


def traffic_of(model: KoemiModel) -> tuple[ModuleTraffic, ...]:
    return calibrate_traffic(model, lambda: model(INPUT_IDS))


def traffic_by_name(model: KoemiModel) -> dict[str, ModuleTraffic]:
    return {item.name: item for item in traffic_of(model)}


class TrafficCalibrationTests(unittest.TestCase):
    def test_calibration_records_one_row_per_processed_token(self) -> None:
        model = build_model(expert_count=0)
        measured = traffic_by_name(model)
        token_count = INPUT_IDS.numel()
        self.assertEqual(token_count, measured["fusion_projection"].rows)
        self.assertEqual(token_count, measured["recurrent_state.retention_projection"].rows)

    def test_the_token_predictor_is_reached_twice_per_forward(self) -> None:
        measured = traffic_by_name(build_model(expert_count=0))
        self.assertEqual(2, measured["token_predictor"].calls)
        self.assertEqual(2 * INPUT_IDS.numel(), measured["token_predictor"].rows)

    def test_the_embedding_table_carries_no_arithmetic(self) -> None:
        measured = traffic_by_name(build_model(expert_count=0))
        self.assertEqual(0, measured["embedding"].flops)
        self.assertEqual(0.0, measured["embedding"].intensity)
        self.assertGreater(measured["embedding"].parameter_bytes, 0)

    def test_experts_see_a_fraction_of_the_tokens(self) -> None:
        measured = traffic_by_name(build_model(expert_count=4))
        expert_rows = sum(
            item.rows for name, item in measured.items() if name.startswith("experts.experts.0.")
        )
        self.assertGreater(expert_rows, 0)
        self.assertLess(expert_rows, 3 * INPUT_IDS.numel())

    def test_the_token_predictor_outranks_every_expert(self) -> None:
        measured = traffic_by_name(build_model(expert_count=4))
        predictor = measured["token_predictor"].intensity
        expert_intensities = [
            item.intensity for name, item in measured.items() if name.startswith("experts.experts.")
        ]
        self.assertTrue(expert_intensities)
        self.assertGreater(predictor, max(expert_intensities))

    def test_calibration_removes_its_hooks(self) -> None:
        model = build_model(expert_count=0)
        traffic_of(model)
        self.assertEqual(0, len(model.token_predictor._forward_pre_hooks))


class PlacementPlanTests(unittest.TestCase):
    def test_an_unlimited_budget_keeps_everything_on_the_accelerator(self) -> None:
        plan = plan_placement(traffic_of(build_model()), TierBudget())
        self.assertEqual({ACCELERATOR_TIER}, {placement.tier for placement in plan.placements})
        self.assertEqual(0, plan.bytes_by_tier()[DISK_TIER])

    def test_a_zero_accelerator_budget_pushes_everything_to_the_host(self) -> None:
        plan = plan_placement(traffic_of(build_model()), TierBudget(accelerator_bytes=0))
        self.assertEqual({HOST_TIER}, {placement.tier for placement in plan.placements})

    def test_exhausted_budgets_fall_through_to_disk(self) -> None:
        plan = plan_placement(
            traffic_of(build_model()), TierBudget(accelerator_bytes=0, host_bytes=0)
        )
        self.assertEqual({DISK_TIER}, {placement.tier for placement in plan.placements})

    def test_the_accelerator_budget_is_filled_by_descending_intensity(self) -> None:
        measured = traffic_of(build_model())
        ranked = sorted(measured, key=lambda item: (-item.intensity, -item.parameter_bytes, item.name))
        budget = TierBudget(accelerator_bytes=ranked[0].parameter_bytes, host_bytes=0)
        plan = plan_placement(measured, budget)
        self.assertEqual((ranked[0].name,), plan.names_by_tier(ACCELERATOR_TIER))
        self.assertNotIn(ranked[0].name, plan.names_by_tier(DISK_TIER))

    def test_the_plan_never_exceeds_a_tier_budget(self) -> None:
        measured = traffic_of(build_model())
        total = sum(item.parameter_bytes for item in measured)
        budget = TierBudget(accelerator_bytes=total // 4, host_bytes=total // 4)
        plan = plan_placement(measured, budget)
        totals = plan.bytes_by_tier()
        self.assertLessEqual(totals[ACCELERATOR_TIER], total // 4)
        self.assertLessEqual(totals[HOST_TIER], total // 4)
        self.assertEqual(total, sum(totals.values()))

    def test_the_plan_reports_every_owning_module_once(self) -> None:
        measured = traffic_of(build_model())
        plan = plan_placement(measured, TierBudget())
        names = [placement.name for placement in plan.placements]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual({item.name for item in measured}, set(names))

    def test_tier_of_rejects_an_unknown_module(self) -> None:
        plan = plan_placement(traffic_of(build_model()), TierBudget())
        with self.assertRaises(KeyError):
            plan.tier_of("not_a_module")


class DiskParameterStoreTests(unittest.TestCase):
    def test_a_stored_tensor_returns_bit_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DiskParameterStore(directory)
            original = torch.randn(7, 3)
            store.write("block.weight", original)
            recovered = store.read("block.weight", torch.device("cpu"))
            self.assertTrue(torch.equal(original, recovered))
            self.assertEqual(original.numel() * original.element_size(), store.read_bytes)
            self.assertEqual(1, store.reads)

    def test_a_name_that_escapes_the_directory_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DiskParameterStore(directory)
            for unsafe in ("../weight", "nested/weight", "weight name", "weight*"):
                with self.subTest(name=unsafe):
                    with self.assertRaises(OffloadError):
                        store.path_for(unsafe)

    def test_reading_a_missing_entry_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DiskParameterStore(directory)
            with self.assertRaisesRegex(OffloadError, "store entry is missing"):
                store.read("absent.weight", torch.device("cpu"))

    def test_clear_removes_every_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DiskParameterStore(directory)
            store.write("one.weight", torch.zeros(2))
            store.write("two.weight", torch.zeros(2))
            self.assertEqual(2, store.clear())
            self.assertEqual(0, len(list(Path(directory).glob("*.pt"))))


class OffloadEquivalenceTests(unittest.TestCase):
    def resident_logits(self, model: KoemiModel) -> torch.Tensor:
        with torch.no_grad():
            return model(INPUT_IDS).logits.clone()

    def test_host_tier_offload_keeps_logits_bit_exact(self) -> None:
        model = build_model()
        expected = self.resident_logits(model)
        plan = plan_placement(traffic_of(model), TierBudget(accelerator_bytes=0))
        engine = OffloadEngine(model, plan, "cpu")
        engine.attach()
        try:
            with torch.no_grad():
                measured = model(INPUT_IDS).logits
            self.assertTrue(torch.equal(expected, measured))
            self.assertGreater(engine.statistics.host_materializations, 0)
        finally:
            engine.detach()

    def test_host_tier_offload_keeps_gradients_bit_exact(self) -> None:
        reference = build_model()
        reference(INPUT_IDS).logits.square().sum().backward()
        expected = {name: parameter.grad.clone() for name, parameter in reference.named_parameters()}

        model = build_model()
        plan = plan_placement(traffic_of(model), TierBudget(accelerator_bytes=0))
        engine = OffloadEngine(model, plan, "cpu")
        engine.attach()
        try:
            model(INPUT_IDS).logits.square().sum().backward()
        finally:
            engine.detach()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(expected[name], parameter.grad))

    def test_disk_tier_offload_keeps_logits_bit_exact_for_frozen_weights(self) -> None:
        model = build_model()
        expected = self.resident_logits(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with tempfile.TemporaryDirectory() as directory:
            store = DiskParameterStore(directory)
            plan = plan_placement(
                traffic_of(model), TierBudget(accelerator_bytes=0, host_bytes=0)
            )
            engine = OffloadEngine(model, plan, "cpu", store)
            engine.attach()
            try:
                self.assertEqual(0, model.token_predictor.weight.numel())
                with torch.no_grad():
                    measured = model(INPUT_IDS).logits
                self.assertTrue(torch.equal(expected, measured))
                statistics = engine.refresh_statistics()
                self.assertGreater(statistics.disk_materializations, 0)
                self.assertGreater(statistics.disk_read_bytes, 0)
            finally:
                engine.detach()
            self.assertEqual(
                expected.shape[-1] * model.settings.embedding_size,
                model.token_predictor.weight.numel(),
            )

    def test_the_disk_tier_refuses_a_trainable_parameter(self) -> None:
        model = build_model()
        with tempfile.TemporaryDirectory() as directory:
            plan = plan_placement(
                traffic_of(model), TierBudget(accelerator_bytes=0, host_bytes=0)
            )
            engine = OffloadEngine(model, plan, "cpu", DiskParameterStore(directory))
            with self.assertRaisesRegex(OffloadError, "trainable"):
                engine.attach()

    def test_a_disk_placement_without_a_store_is_refused(self) -> None:
        model = build_model()
        plan = plan_placement(traffic_of(model), TierBudget(accelerator_bytes=0, host_bytes=0))
        with self.assertRaisesRegex(OffloadError, "requires a parameter store"):
            OffloadEngine(model, plan, "cpu").attach()

    def test_attaching_twice_is_refused(self) -> None:
        model = build_model()
        engine = OffloadEngine(model, plan_placement(traffic_of(model), TierBudget()), "cpu")
        engine.attach()
        try:
            with self.assertRaisesRegex(OffloadError, "already attached"):
                engine.attach()
        finally:
            engine.detach()

    def test_detach_leaves_no_hook_behind(self) -> None:
        model = build_model()
        plan = plan_placement(traffic_of(model), TierBudget(accelerator_bytes=0))
        engine = OffloadEngine(model, plan, "cpu")
        engine.attach()
        engine.detach()
        self.assertEqual(0, len(model.token_predictor._forward_pre_hooks))
        self.assertEqual(0, len(model.token_predictor._forward_hooks))
        self.assertIn("weight", model.token_predictor._parameters)


if __name__ == "__main__":
    unittest.main()
