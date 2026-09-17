from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from koemi.model.memory import HierarchicalAssociativeMemory


class AssociativeReadTests(unittest.TestCase):
    def build_memory(self) -> HierarchicalAssociativeMemory:
        torch.manual_seed(11)
        return HierarchicalAssociativeMemory(8, 4, 0.0625)

    def test_chunk_read_matches_the_sequential_oracle(self) -> None:
        memory = self.build_memory()
        write_source = torch.randn(2, 9, 8)
        query_source = torch.randn(2, 9, 8)
        surprise = torch.rand(2, 9)
        initial_basis = torch.randn(2, 8, 4)
        initial_normalizer = torch.rand(2, 4)
        terms = memory.fast_write_terms(memory.project(write_source), surprise)

        actual_read, final_basis, final_normalizer = memory.scan_and_read(
            initial_basis,
            initial_normalizer,
            terms,
            query_source,
        )

        basis = initial_basis
        normalizer = initial_normalizer
        expected_reads = []
        for position in range(write_source.shape[1]):
            expected_read, _ = memory.read(basis, normalizer, query_source[:, position])
            expected_reads.append(expected_read)
            basis, normalizer = memory.update(basis, normalizer, terms.at(position))

        self.assertTrue(torch.allclose(actual_read, torch.stack(expected_reads, dim=1), atol=1e-5))
        self.assertTrue(torch.allclose(final_basis, basis, atol=1e-5))
        self.assertTrue(torch.allclose(final_normalizer, normalizer, atol=1e-5))

    def test_microblock_read_matches_oracle_state_and_gradients(self) -> None:
        torch.manual_seed(29)
        memory = self.build_memory()
        write_source = torch.randn(2, 11, 8, requires_grad=True)
        query_source = torch.randn(2, 11, 8, requires_grad=True)
        surprise = torch.rand(2, 11, requires_grad=True)
        initial_basis = torch.randn(2, 8, 4, requires_grad=True)
        initial_normalizer = torch.rand(2, 4, requires_grad=True)
        terms = memory.fast_write_terms(memory.project(write_source), surprise)

        oracle = memory.scan_and_read(initial_basis, initial_normalizer, terms, query_source)
        microblock = memory.scan_and_read_microblocks(
            initial_basis,
            initial_normalizer,
            terms,
            query_source,
            microblock_size=3,
        )

        for actual, expected in zip(microblock, oracle):
            self.assertTrue(torch.allclose(actual, expected, atol=3e-5, rtol=3e-5))
            self.assertTrue(torch.isfinite(actual).all())

        gradient_targets = (
            write_source,
            query_source,
            surprise,
            initial_basis,
            initial_normalizer,
            *tuple(memory.parameters()),
        )
        oracle_loss = sum(value.square().mean() for value in oracle)
        microblock_loss = sum(value.square().mean() for value in microblock)
        oracle_gradients = torch.autograd.grad(oracle_loss, gradient_targets, retain_graph=True)
        microblock_gradients = torch.autograd.grad(microblock_loss, gradient_targets)
        for actual, expected in zip(microblock_gradients, oracle_gradients):
            self.assertTrue(torch.allclose(actual, expected, atol=5e-5, rtol=5e-5))

    def test_microblock_read_does_not_materialize_the_full_pairwise_matrix(self) -> None:
        memory = self.build_memory()
        batch_size = 2
        length = 11
        microblock_size = 3
        write_source = torch.randn(batch_size, length, 8)
        query_source = torch.randn(batch_size, length, 8)
        terms = memory.fast_write_terms(memory.project(write_source), torch.rand(batch_size, length))
        initial_basis = torch.randn(batch_size, 8, 4)
        initial_normalizer = torch.rand(batch_size, 4)
        exponential_input_shapes = []
        pairwise_einsum_output_shapes = []
        original_exp = torch.exp
        original_einsum = torch.einsum

        def record_exp(input_tensor, *arguments, **keyword_arguments):
            exponential_input_shapes.append(tuple(input_tensor.shape))
            return original_exp(input_tensor, *arguments, **keyword_arguments)

        def record_einsum(equation, *operands, **keyword_arguments):
            result = original_einsum(equation, *operands, **keyword_arguments)
            if equation == "bsm,btm->bts":
                pairwise_einsum_output_shapes.append(tuple(result.shape))
            return result

        with patch.object(torch, "exp", side_effect=record_exp):
            with patch.object(torch, "einsum", side_effect=record_einsum):
                memory.scan_and_read_microblocks(
                    initial_basis,
                    initial_normalizer,
                    terms,
                    query_source,
                    microblock_size=microblock_size,
                )

        pairwise_decay_shapes = [shape for shape in exponential_input_shapes if len(shape) == 3]
        for pairwise_shapes in (pairwise_decay_shapes, pairwise_einsum_output_shapes):
            self.assertTrue(pairwise_shapes)
            self.assertNotIn((batch_size, length, length), pairwise_shapes)
            for shape in pairwise_shapes:
                self.assertLessEqual(shape[1], microblock_size)
                self.assertLessEqual(shape[2], microblock_size)

    def test_microblock_read_is_causal(self) -> None:
        torch.manual_seed(37)
        memory = self.build_memory()
        length = 11
        cutoff = 5
        write_source = torch.randn(2, length, 8)
        query_source = torch.randn(2, length, 8)
        terms = memory.fast_write_terms(memory.project(write_source), torch.rand(2, length))
        changed_terms = memory.write_terms(
            torch.cat((terms.decay[:, :cutoff], torch.full_like(terms.decay[:, cutoff:], 0.75)), dim=1),
            torch.cat((terms.value[:, :cutoff], torch.randn_like(terms.value[:, cutoff:])), dim=1),
            torch.cat((terms.features[:, :cutoff], torch.randn_like(terms.features[:, cutoff:])), dim=1),
            torch.cat((terms.write_weight[:, :cutoff], torch.ones_like(terms.write_weight[:, cutoff:])), dim=1),
        )
        initial_basis = torch.randn(2, 8, 4)
        initial_normalizer = torch.rand(2, 4)
        baseline = memory.scan_and_read_microblocks(
            initial_basis,
            initial_normalizer,
            terms,
            query_source,
            microblock_size=3,
        )
        changed = memory.scan_and_read_microblocks(
            initial_basis,
            initial_normalizer,
            changed_terms,
            query_source,
            microblock_size=3,
        )
        self.assertTrue(torch.allclose(baseline[0][:, :cutoff], changed[0][:, :cutoff], atol=3e-5, rtol=3e-5))

    def test_write_terms_keep_the_rank_one_factors(self) -> None:
        memory = self.build_memory()
        terms = memory.fast_write_terms(memory.project(torch.randn(2, 7, 8)), torch.rand(2, 7))
        self.assertEqual((2, 7, 8), tuple(terms.value.shape))
        self.assertEqual((2, 7, 4), tuple(terms.features.shape))
        self.assertEqual((2, 7), tuple(terms.write_weight.shape))

    def test_zero_evidence_suppresses_an_inconsistent_basis(self) -> None:
        memory = self.build_memory()
        with torch.no_grad():
            memory.query_projection.weight.zero_()
            memory.query_projection.bias.zero_()
            memory.feature_projection.weight.zero_()
            memory.feature_projection.bias.zero_()
        basis = torch.ones(1, 8, 4)
        normalizer = torch.zeros(1, 4)
        read, _ = memory.read(basis, normalizer, torch.zeros(1, 8))
        self.assertTrue(torch.equal(read, torch.zeros_like(read)))


if __name__ == "__main__":
    unittest.main()
