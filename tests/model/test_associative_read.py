from __future__ import annotations

import unittest

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
