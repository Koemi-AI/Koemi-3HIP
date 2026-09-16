from __future__ import annotations

import unittest

from koemi.training.batching_mode import BatchingMode, MicrobatchPlan


class BatchingModeTests(unittest.TestCase):
    def test_reordered_length_buckets_keep_padding_deterministic(self) -> None:
        batching_mode = BatchingMode(
            [1, 4, 2, 8, 7, 5, 3],
            max_batch_size=2,
            bucket_size=4,
            preserve_order=False,
        )

        microbatches = list(batching_mode)
        self.assertEqual(
            [(0, 2), (6, 1), (5, 4), (3,)],
            [microbatch.sample_indices for microbatch in microbatches],
        )
        self.assertEqual(
            [(1, 2), (3, 4), (5, 7), (8,)],
            [microbatch.sample_lengths for microbatch in microbatches],
        )
        for microbatch in microbatches:
            self.assertEqual(
                len(microbatch.sample_indices) * microbatch.max_length,
                microbatch.padded_token_count,
            )
            self.assertEqual(
                microbatch.padded_token_count - microbatch.real_token_count,
                microbatch.padding_token_count,
            )
            self.assertLessEqual(max(microbatch.sample_lengths) - min(microbatch.sample_lengths), 3)

    def test_preserve_order_keeps_the_original_permutation(self) -> None:
        batching_mode = BatchingMode(
            [5, 1, 4, 2],
            max_batch_size=2,
            preserve_order=True,
        )

        self.assertEqual(
            ((0, 1), (2, 3)),
            tuple(microbatch.sample_indices for microbatch in batching_mode),
        )
        self.assertEqual((0, 1, 2, 3), batching_mode.permutation)
        self.assertEqual((0, 1, 2, 3), batching_mode.inverse_permutation)

    def test_seeded_order_is_repeatable_and_reversible(self) -> None:
        first_mode = BatchingMode(
            [4, 4, 4, 4, 4, 4, 4, 4],
            max_batch_size=2,
            preserve_order=False,
            seed=17,
        )
        second_mode = BatchingMode(
            [4, 4, 4, 4, 4, 4, 4, 4],
            max_batch_size=2,
            preserve_order=False,
            seed=17,
        )

        first_batches = tuple(microbatch.sample_indices for microbatch in first_mode)
        second_batches = tuple(microbatch.sample_indices for microbatch in second_mode)
        self.assertEqual(first_batches, second_batches)
        self.assertEqual(tuple(range(8)), tuple(sorted(first_mode.permutation)))
        for original_index, planned_position in enumerate(first_mode.inverse_permutation):
            self.assertEqual(original_index, first_mode.permutation[planned_position])

    def test_token_budget_and_batch_limit_are_enforced(self) -> None:
        batching_mode = BatchingMode(
            [2, 3, 4, 5],
            max_batch_size=2,
            max_tokens=8,
            preserve_order=True,
        )

        self.assertEqual(
            ((0, 1), (2,), (3,)),
            tuple(microbatch.sample_indices for microbatch in batching_mode),
        )
        for microbatch in batching_mode:
            self.assertLessEqual(microbatch.batch_size, 2)
            self.assertLessEqual(microbatch.padded_token_count, 8)
        self.assertEqual(14, batching_mode.metrics.real_token_count)
        self.assertEqual(15, batching_mode.metrics.padded_token_count)
        self.assertEqual(1, batching_mode.metrics.padding_token_count)

    def test_aggregate_metrics_are_integer_token_counts(self) -> None:
        batching_mode = BatchingMode([2, 4, 3], max_batch_size=2, preserve_order=True)

        self.assertEqual(3, batching_mode.metrics.sample_count)
        self.assertEqual(2, batching_mode.metrics.microbatch_count)
        self.assertEqual(9, batching_mode.metrics.real_token_count)
        self.assertEqual(11, batching_mode.metrics.padded_token_count)
        self.assertEqual(2, batching_mode.metrics.padding_token_count)
        self.assertAlmostEqual(2 / 11, batching_mode.metrics.padding_fraction)

    def test_empty_input_produces_no_batches_or_steps(self) -> None:
        batching_mode = BatchingMode([], max_batch_size=4, gradient_accumulation_steps=3)

        self.assertEqual((), tuple(batching_mode))
        self.assertEqual(0, len(batching_mode))
        self.assertEqual(0, batching_mode.optimizer_step_count)
        self.assertEqual((), batching_mode.permutation)
        self.assertEqual((), batching_mode.inverse_permutation)
        self.assertEqual(0, batching_mode.metrics.padding_fraction)

    def test_final_partial_accumulation_group_is_retained(self) -> None:
        batching_mode = BatchingMode(
            [2, 2, 2, 2, 2],
            max_batch_size=1,
            gradient_accumulation_steps=2,
        )

        microbatches = list(batching_mode)
        self.assertEqual(5, len(microbatches))
        self.assertEqual(3, batching_mode.optimizer_step_count)
        self.assertEqual(
            [1, 2, 1, 2, 1],
            [microbatch.accumulation_position for microbatch in microbatches],
        )
        self.assertEqual(
            [False, True, False, True, True],
            [microbatch.is_optimizer_step_boundary for microbatch in microbatches],
        )
        self.assertEqual([0, 0, 1, 1, 2], [microbatch.optimizer_step_index for microbatch in microbatches])
        self.assertEqual((0, 1, 2, 3, 4), batching_mode.permutation)

    def test_invalid_lengths_shapes_and_limits_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            BatchingMode([[2]], max_batch_size=1)
        with self.assertRaises(ValueError):
            BatchingMode([0], max_batch_size=1)
        with self.assertRaises(ValueError):
            BatchingMode([3], max_batch_size=1, max_tokens=2)
        with self.assertRaises(ValueError):
            BatchingMode([1], max_batch_size=0)
        with self.assertRaises(ValueError):
            BatchingMode([1], max_batch_size=1, bucket_size=0)
        with self.assertRaises(ValueError):
            BatchingMode([1], max_batch_size=1, gradient_accumulation_steps=0)
        with self.assertRaises(ValueError):
            BatchingMode([1], max_batch_size=1, preserve_order=True, seed=9)

    def test_microbatch_shape_contract_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            MicrobatchPlan(
                sample_indices=(0, 1),
                sample_lengths=(3,),
                max_length=3,
                real_token_count=3,
                padded_token_count=6,
                padding_token_count=3,
                microbatch_index=0,
                optimizer_step_index=0,
                accumulation_position=1,
                is_optimizer_step_boundary=True,
            )


if __name__ == "__main__":
    unittest.main()
