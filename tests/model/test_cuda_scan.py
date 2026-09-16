from __future__ import annotations

import unittest

import torch

from koemi.model.cuda_scan import (
    CudaScanDiagnostics,
    CudaScanUnavailableError,
    CudaScanValidationError,
    cuda_affine_scan,
    diagnose_cuda_affine_scan,
)


CUDA_SKIP_REASON = "CUDA unavailable: torch.cuda.is_available() is False"


def sequential_affine_scan(
    retention: torch.Tensor,
    increment: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    state = initial
    states: list[torch.Tensor] = []
    for position in range(increment.shape[1]):
        state = retention[:, position] * state + increment[:, position]
        states.append(state)
    if not states:
        return increment.new_empty(increment.shape)
    return torch.stack(states, dim=1)


class CudaAffineScanContractTests(unittest.TestCase):
    def test_valid_cpu_tensors_are_rejected_instead_of_running_as_cuda(self) -> None:
        retention = torch.ones(1, 2, 3)
        increment = torch.ones(1, 2, 3)
        if torch.cuda.is_available():
            with self.assertRaisesRegex(CudaScanValidationError, "all scan tensors must be on CUDA"):
                cuda_affine_scan(retention, increment)
        else:
            with self.assertRaisesRegex(CudaScanUnavailableError, r"torch.cuda.is_available\(\) is False"):
                cuda_affine_scan(retention, increment)

    def test_shape_mismatch_is_rejected_before_device_dispatch(self) -> None:
        retention = torch.ones(1, 2, 3)
        increment = torch.ones(1, 3, 3)
        with self.assertRaisesRegex(CudaScanValidationError, "same batch and sequence dimensions"):
            cuda_affine_scan(retention, increment)

    def test_dtype_mismatch_is_rejected_before_device_dispatch(self) -> None:
        retention = torch.ones(1, 2, 3, dtype=torch.float32)
        increment = torch.ones(1, 2, 3, dtype=torch.float64)
        with self.assertRaisesRegex(CudaScanValidationError, "same dtype"):
            cuda_affine_scan(retention, increment)

    def test_invalid_chunk_size_is_rejected_before_device_dispatch(self) -> None:
        retention = torch.ones(1, 2, 3)
        increment = torch.ones(1, 2, 3)
        with self.assertRaisesRegex(CudaScanValidationError, "positive integer"):
            cuda_affine_scan(retention, increment, chunk_size=0)

    def test_invalid_initial_shape_is_rejected_before_device_dispatch(self) -> None:
        retention = torch.ones(1, 2, 3)
        increment = torch.ones(1, 2, 3)
        initial = torch.ones(1, 2)
        with self.assertRaisesRegex(CudaScanValidationError, "initial must have shape"):
            cuda_affine_scan(retention, increment, initial)


@unittest.skipUnless(torch.cuda.is_available(), CUDA_SKIP_REASON)
class CudaAffineScanExecutionTests(unittest.TestCase):
    device = torch.device("cuda:0")

    def test_empty_sequence_returns_an_empty_cuda_result(self) -> None:
        retention = torch.empty((2, 0, 4), device=self.device, dtype=torch.float32)
        increment = torch.empty((2, 0, 4), device=self.device, dtype=torch.float32)
        initial = torch.zeros((2, 4), device=self.device, dtype=torch.float32)
        states = cuda_affine_scan(retention, increment, initial, chunk_size=3)
        self.assertEqual((2, 0, 4), tuple(states.shape))
        self.assertEqual(self.device, states.device)
        self.assertEqual(torch.float32, states.dtype)

    def test_single_chunk_matches_the_sequential_reference(self) -> None:
        torch.manual_seed(1)
        retention = torch.rand((2, 7, 5), device=self.device) * 0.99 + 0.004
        increment = torch.randn((2, 7, 5), device=self.device)
        initial = torch.randn((2, 5), device=self.device)
        expected = sequential_affine_scan(retention, increment, initial)
        obtained = cuda_affine_scan(retention, increment, initial, chunk_size=7)
        self.assertTrue(torch.allclose(obtained, expected, atol=1e-5, rtol=1e-5))

    def test_multiple_chunks_match_the_sequential_reference_for_matrix_states(self) -> None:
        torch.manual_seed(2)
        retention = torch.rand((2, 19, 1, 1), device=self.device) * 0.99 + 0.004
        increment = torch.randn((2, 19, 4, 3), device=self.device)
        initial = torch.randn((2, 4, 3), device=self.device)
        expected = sequential_affine_scan(retention, increment, initial)
        obtained = cuda_affine_scan(retention, increment, initial, chunk_size=4)
        self.assertTrue(torch.allclose(obtained, expected, atol=1e-5, rtol=1e-5))

    def test_omitted_initial_state_matches_a_zero_initial_reference(self) -> None:
        torch.manual_seed(3)
        retention = torch.rand((2, 11, 3), device=self.device) * 0.99 + 0.004
        increment = torch.randn((2, 11, 3), device=self.device)
        expected = sequential_affine_scan(retention, increment, torch.zeros((2, 3), device=self.device))
        obtained = cuda_affine_scan(retention, increment, chunk_size=3)
        self.assertTrue(torch.allclose(obtained, expected, atol=1e-5, rtol=1e-5))

    def test_float_dtypes_match_the_sequential_reference(self) -> None:
        supported_dtypes = [torch.float16, torch.float32]
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            supported_dtypes.append(torch.bfloat16)
        for dtype in supported_dtypes:
            with self.subTest(dtype=dtype):
                torch.manual_seed(4)
                retention = torch.rand((1, 6, 3), device=self.device, dtype=dtype) * 0.9 + 0.05
                increment = torch.randn((1, 6, 3), device=self.device, dtype=dtype)
                initial = torch.randn((1, 3), device=self.device, dtype=dtype)
                expected = sequential_affine_scan(retention, increment, initial)
                obtained = cuda_affine_scan(retention, increment, initial, chunk_size=2)
                tolerance = 5e-3 if dtype != torch.float32 else 1e-5
                self.assertTrue(torch.allclose(obtained, expected, atol=tolerance, rtol=tolerance))

    def test_gradients_match_the_sequential_reference(self) -> None:
        torch.manual_seed(5)
        retention_values = torch.rand((1, 9, 3), device=self.device) * 0.9 + 0.05
        increment_values = torch.randn((1, 9, 3), device=self.device)
        initial_values = torch.randn((1, 3), device=self.device)

        parallel_retention = retention_values.detach().requires_grad_()
        parallel_increment = increment_values.detach().requires_grad_()
        parallel_initial = initial_values.detach().requires_grad_()
        parallel_states = cuda_affine_scan(
            parallel_retention,
            parallel_increment,
            parallel_initial,
            chunk_size=3,
        )
        parallel_states.square().sum().backward()
        parallel_gradients = (
            parallel_retention.grad.detach().clone(),
            parallel_increment.grad.detach().clone(),
            parallel_initial.grad.detach().clone(),
        )

        sequential_retention = retention_values.detach().requires_grad_()
        sequential_increment = increment_values.detach().requires_grad_()
        sequential_initial = initial_values.detach().requires_grad_()
        sequential_states = sequential_affine_scan(sequential_retention, sequential_increment, sequential_initial)
        sequential_states.square().sum().backward()
        sequential_gradients = (
            sequential_retention.grad.detach(),
            sequential_increment.grad.detach(),
            sequential_initial.grad.detach(),
        )

        self.assertTrue(torch.allclose(parallel_states, sequential_states, atol=1e-5, rtol=1e-5))
        for obtained, expected in zip(parallel_gradients, sequential_gradients):
            self.assertTrue(torch.allclose(obtained, expected, atol=1e-5, rtol=1e-5))

    def test_diagnostic_reports_synchronized_work_counts(self) -> None:
        retention = torch.full((2, 9, 3), 0.75, device=self.device)
        increment = torch.ones((2, 9, 3), device=self.device)
        initial = torch.zeros((2, 3), device=self.device)
        diagnostic = diagnose_cuda_affine_scan(retention, increment, initial, chunk_size=4)
        self.assertIsInstance(diagnostic, CudaScanDiagnostics)
        self.assertEqual(18, diagnostic.token_count)
        self.assertEqual(3, diagnostic.chunk_count)
        self.assertGreaterEqual(diagnostic.elapsed_seconds, 0.0)
        self.assertEqual(self.device, diagnostic.states.device)


if __name__ == "__main__":
    unittest.main()
