from __future__ import annotations

import unittest

import torch
from torch import nn

from koemi.model.gpu_precision import (
    autocast_context,
    collect_precision_health,
    compare_against_fp32,
    resolve_precision,
    validate_precision_support,
)


class CpuPrecisionContractTests(unittest.TestCase):
    def test_auto_resolves_to_fp32_without_autocast_or_scaler(self) -> None:
        policy = resolve_precision(torch.device("cpu"), "auto")

        self.assertEqual(torch.device("cpu"), policy.device)
        self.assertEqual("auto", policy.requested)
        self.assertEqual("fp32", policy.precision)
        self.assertIsNone(policy.autocast_dtype)
        self.assertFalse(policy.autocast_enabled)
        self.assertFalse(policy.use_grad_scaler)

    def test_fp32_cpu_context_keeps_float32(self) -> None:
        policy = resolve_precision("cpu", "fp32")

        with autocast_context(policy):
            result = torch.matmul(torch.ones(2, 2), torch.ones(2, 2))

        self.assertEqual(torch.float32, result.dtype)

    def test_cpu_rejects_mixed_precision_requests(self) -> None:
        for requested in ("bf16", "fp16"):
            with self.subTest(requested=requested):
                with self.assertRaisesRegex(ValueError, "requires CUDA"):
                    resolve_precision("cpu", requested)

    def test_invalid_precision_device_and_tf32_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "one of"):
            resolve_precision("cpu", "fp8")
        with self.assertRaisesRegex(ValueError, "CPU or CUDA"):
            resolve_precision("meta", "fp32")
        with self.assertRaisesRegex(ValueError, "TF32 requires CUDA"):
            resolve_precision("cpu", "fp32", enable_tf32=True)
        with self.assertRaisesRegex(ValueError, "true, false or none"):
            validate_precision_support("cpu", "fp32", enable_tf32=1)

    def test_health_aggregates_finite_output_gradient_and_cpu_memory_contract(self) -> None:
        parameter = nn.Parameter(torch.tensor([2.0, -3.0]))
        output = parameter.square()
        output.sum().backward()

        health = collect_precision_health(output, parameter)

        self.assertTrue(health.all_finite)
        self.assertTrue(health.gradients_finite)
        self.assertEqual(1, health.tensor_count)
        self.assertEqual(1, health.finite_tensor_count)
        self.assertEqual(1, health.gradient_tensor_count)
        self.assertEqual(1, health.finite_gradient_tensor_count)
        self.assertEqual(6.0, health.max_abs_gradient)
        self.assertIsNone(health.memory_allocated_bytes)
        self.assertIsNone(health.memory_reserved_bytes)

    def test_health_detects_non_finite_outputs_and_gradients(self) -> None:
        parameter = nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("nan")])
        health = collect_precision_health(torch.tensor([1.0, float("inf")]), parameter)

        self.assertFalse(health.all_finite)
        self.assertFalse(health.gradients_finite)
        self.assertEqual(1, health.non_finite_tensor_count)
        self.assertEqual(1, health.non_finite_value_count)
        self.assertEqual(1, health.non_finite_gradient_tensor_count)
        self.assertEqual(1, health.non_finite_gradient_value_count)

    def test_fp32_comparison_reports_output_and_gradient_errors(self) -> None:
        comparison = compare_against_fp32(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[1.0001, 1.9999]]),
            {"weight": torch.tensor([2.0, -1.0])},
            {"weight": torch.tensor([2.0001, -1.0001])},
            rtol=1e-3,
            atol=1e-3,
        )

        self.assertTrue(comparison.passed)
        self.assertTrue(comparison.output_allclose)
        self.assertTrue(comparison.gradient_allclose)
        self.assertIsNone(comparison.failure_reason)

    def test_comparison_rejects_non_finite_candidate(self) -> None:
        comparison = compare_against_fp32(
            torch.tensor([1.0]),
            torch.tensor([float("nan")]),
        )

        self.assertFalse(comparison.passed)
        self.assertFalse(comparison.output_allclose)
        self.assertEqual(float("inf"), comparison.output_max_abs_error)
        self.assertIn("non-finite", comparison.failure_reason or "")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable: CUDA precision contract")
class CudaPrecisionContractTests(unittest.TestCase):
    def test_auto_resolves_to_supported_cuda_precision(self) -> None:
        policy = resolve_precision(torch.device("cuda"), "auto")

        self.assertIn(policy.precision, {"bf16", "fp16"})
        self.assertTrue(policy.autocast_enabled)
        self.assertEqual(policy.device.type, "cuda")

    def test_cuda_output_and_gradient_comparison_against_fp32(self) -> None:
        device = torch.device("cuda")
        torch.manual_seed(5)
        fp32_model = nn.Linear(8, 4, device=device)
        mixed_model = nn.Linear(8, 4, device=device)
        mixed_model.load_state_dict(fp32_model.state_dict())
        input_values = torch.randn(3, 8, device=device)

        fp32_policy = resolve_precision(device, "fp32")
        mixed_policy = resolve_precision(device, "auto")

        fp32_model.zero_grad(set_to_none=True)
        with autocast_context(fp32_policy):
            fp32_output = fp32_model(input_values)
            fp32_output.square().mean().backward()
        fp32_gradients = {
            name: parameter.grad.detach().clone()
            if parameter.grad is not None
            else None
            for name, parameter in fp32_model.named_parameters()
        }

        mixed_model.zero_grad(set_to_none=True)
        with autocast_context(mixed_policy):
            mixed_output = mixed_model(input_values)
            mixed_output.square().mean().backward()
        mixed_gradients = {
            name: parameter.grad.detach().clone()
            if parameter.grad is not None
            else None
            for name, parameter in mixed_model.named_parameters()
        }

        comparison = compare_against_fp32(
            fp32_output.detach(),
            mixed_output.detach(),
            fp32_gradients,
            mixed_gradients,
            rtol=5e-2,
            atol=5e-3,
        )

        self.assertTrue(comparison.output_allclose, comparison.failure_reason)
        self.assertTrue(comparison.gradient_allclose, comparison.failure_reason)

    def test_cuda_health_reads_allocator_without_changing_contract(self) -> None:
        device = torch.device("cuda")
        value = torch.ones(2, device=device)

        health = collect_precision_health(value, device=device)

        self.assertEqual(device, health.device)
        self.assertTrue(health.all_finite)
        self.assertIsInstance(health.memory_allocated_bytes, int)
        self.assertIsInstance(health.memory_reserved_bytes, int)
