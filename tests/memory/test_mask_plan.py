import unittest
from unittest import mock

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    SAMPLE_MODE_CONTIGUOUS_SEGMENT,
)


class _FakeRawDataset:
    """Small in-memory stand-in exposing only the production sampler contract."""

    def __init__(
        self,
        name="fake",
        trajectory_ids=(10, 11, 12),
        trajectory_lengths=(9, 42, 31),
    ):
        self.dataset_name = name
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
        self.trajectory_lengths = np.asarray(trajectory_lengths, dtype=np.int64)
        self.delta_indices = {
            "video.primary": np.arange(3, dtype=np.int64),
            "action.delta": np.arange(2, dtype=np.int64),
            "language.task": np.asarray([0], dtype=np.int64),
        }
        self.modality_keys = {
            "video": ["video.primary"],
            "action": ["action.delta"],
            "language": ["language.task"],
            "state": [],
        }
        self.transforms = lambda data: data

    def get_step_data(self, trajectory_id, base_index):
        video = np.full((3, 4, 4, 3), base_index % 255, dtype=np.uint8)
        action = np.arange(base_index, base_index + 2, dtype=np.float32)[:, None]
        return {
            "video.primary": video,
            "action.delta": action,
            "language.task": [f"episode-{trajectory_id}"],
        }


def _make_mixture(mask_rate=0.0, mask_cap=1, ramp_samples=0, **overrides):
    kwargs = {
        "data_mixture": [(_FakeRawDataset(), 1.0)],
        "mode": "train",
        "balance_dataset_weights": False,
        "balance_trajectory_weights": False,
        "with_state": False,
        "resolution_size": 4,
        "video_resolution_size": 4,
        "seed": 17,
        "sample_mode": SAMPLE_MODE_CONTIGUOUS_SEGMENT,
        "segment_length": 3,
        "burn_in_max_decisions": 3,
        "segment_stride": 4,
    }
    kwargs.update(overrides)
    with mock.patch.object(LeRobotMixtureDataset, "update_metadata", return_value=None):
        mixture = LeRobotMixtureDataset(**kwargs)
    # The trainer pushes masking knobs as plain attributes before dataloader
    # workers fork (VLAMTrainer._configure_mask_schedule); mirror that here.
    mixture.memory_mask_rate = mask_rate
    mixture.memory_mask_max_per_segment = mask_cap
    mixture.memory_mask_ramp_samples = ramp_samples
    return mixture


def _signature(sample):
    return (
        sample["dataset_id"],
        sample["episode_id"],
        tuple(sample["base_indices"].tolist()),
    )


class MaskPlanTest(unittest.TestCase):
    def test_defaults_are_memv1_inert(self):
        self.assertEqual(LeRobotMixtureDataset.memory_mask_rate, 0.0)
        self.assertEqual(LeRobotMixtureDataset.memory_mask_max_per_segment, 1)
        self.assertEqual(LeRobotMixtureDataset.memory_mask_ramp_samples, 0)

        sample = _make_mixture().sample_segment(5)
        self.assertNotIn("mask_plan", sample)
        for step in sample["steps"]:
            if step is not None:
                self.assertNotIn("video_clean", step)

    def test_plan_is_deterministic_per_epoch_index_and_seed(self):
        first = _make_mixture(mask_rate=0.6)
        second = _make_mixture(mask_rate=0.6)
        plans_a = [first._sample_mask_plan(i).tolist() for i in range(64)]
        plans_b = [second._sample_mask_plan(i).tolist() for i in range(64)]
        self.assertEqual(plans_a, plans_b)

        first.set_epoch(1)
        plans_epoch_one = [first._sample_mask_plan(i).tolist() for i in range(64)]
        self.assertNotEqual(plans_a, plans_epoch_one)

        evaluation = _make_mixture(mask_rate=0.6, mode="val")
        val_zero = [evaluation._sample_mask_plan(i).tolist() for i in range(16)]
        evaluation.set_epoch(9)
        val_later = [evaluation._sample_mask_plan(i).tolist() for i in range(16)]
        self.assertEqual(val_zero, val_later)

    def test_first_supervised_decision_is_never_masked(self):
        mixture = _make_mixture(mask_rate=1.0, mask_cap=3)
        for index in range(64):
            plan = mixture._sample_mask_plan(index)
            self.assertEqual(plan.shape, (3,))
            self.assertFalse(bool(plan[0]))
            self.assertTrue(bool(plan[1]) and bool(plan[2]))

    def test_max_per_segment_cap_is_enforced(self):
        mixture = _make_mixture(mask_rate=1.0, mask_cap=1)
        for index in range(64):
            plan = mixture._sample_mask_plan(index)
            self.assertEqual(int(plan.sum()), 1)
            self.assertTrue(bool(plan[1]))

    def test_mask_rate_marginals_over_1000_draws(self):
        mixture = _make_mixture(mask_rate=0.25, mask_cap=3)
        plans = np.stack([mixture._sample_mask_plan(i) for i in range(1000)])
        frequencies = plans.mean(axis=0)
        self.assertEqual(frequencies[0], 0.0)
        for position in (1, 2):
            self.assertAlmostEqual(frequencies[position], 0.25, delta=0.05)

    def test_segment_selection_is_invariant_to_masking_config(self):
        plain = _make_mixture(mask_rate=0.0)
        masked = _make_mixture(mask_rate=0.9, mask_cap=3)
        self.assertEqual(
            [_signature(plain.sample_segment(i)) for i in range(48)],
            [_signature(masked.sample_segment(i)) for i in range(48)],
        )

        reference, sample = plain.sample_segment(7), masked.sample_segment(7)
        for key in ("base_indices", "loss_mask", "update_mask", "sequence_valid"):
            np.testing.assert_array_equal(reference[key], sample[key])
        for plain_step, masked_step in zip(reference["steps"], sample["steps"]):
            if plain_step is not None:
                # The dataloader only flags; the observation itself is untouched.
                np.testing.assert_array_equal(plain_step["video"], masked_step["video"])

    def test_video_clean_rides_exactly_the_masked_decisions(self):
        mixture = _make_mixture(mask_rate=1.0, mask_cap=2)
        sample = mixture.sample_segment(3)
        plan = sample["mask_plan"]
        self.assertEqual(plan.dtype, np.bool_)
        self.assertEqual(plan.tolist(), [False, True, True])
        burn_in = 3
        for offset, masked in enumerate(plan):
            step = sample["steps"][burn_in + offset]
            if masked:
                np.testing.assert_array_equal(step["video_clean"], step["video"])
                self.assertFalse(np.shares_memory(step["video_clean"], step["video"]))
            else:
                self.assertNotIn("video_clean", step)
        for step in sample["steps"][:burn_in]:
            if step is not None:
                self.assertNotIn("video_clean", step)

    def test_run_masking_emits_one_contiguous_run_sparing_position_zero(self):
        mixture = _make_mixture(mask_rate=1.0, mask_cap=2, segment_length=4)
        mixture.memory_mask_run_len = 2
        starts = set()
        for index in range(128):
            plan = mixture._sample_mask_plan(index)
            self.assertEqual(plan.shape, (4,))
            self.assertFalse(bool(plan[0]))
            self.assertEqual(int(plan.sum()), 2)
            run = np.flatnonzero(plan)
            self.assertEqual(int(run[1] - run[0]), 1)
            starts.add(int(run[0]))
        self.assertEqual(starts, {1, 2})

    def test_run_masking_respects_rate(self):
        mixture = _make_mixture(mask_rate=0.25, mask_cap=2, segment_length=4)
        mixture.memory_mask_run_len = 2
        masked = sum(
            bool(mixture._sample_mask_plan(i).any()) for i in range(1000)
        )
        self.assertAlmostEqual(masked / 1000.0, 0.25, delta=0.05)

    def test_run_masking_is_deterministic(self):
        first = _make_mixture(mask_rate=0.5, mask_cap=2, segment_length=4)
        second = _make_mixture(mask_rate=0.5, mask_cap=2, segment_length=4)
        first.memory_mask_run_len = 2
        second.memory_mask_run_len = 2
        self.assertEqual(
            [first._sample_mask_plan(i).tolist() for i in range(64)],
            [second._sample_mask_plan(i).tolist() for i in range(64)],
        )

    def test_run_longer_than_maskable_window_never_masks(self):
        mixture = _make_mixture(mask_rate=1.0, mask_cap=4, segment_length=4)
        mixture.memory_mask_run_len = 4
        for index in range(32):
            self.assertFalse(bool(mixture._sample_mask_plan(index).any()))

    def test_run_masking_default_is_legacy_singleton_path(self):
        self.assertEqual(LeRobotMixtureDataset.memory_mask_run_len, 1)
        mixture = _make_mixture(mask_rate=1.0, mask_cap=1)
        for index in range(32):
            self.assertEqual(int(mixture._sample_mask_plan(index).sum()), 1)

    def test_video_clean_rides_both_positions_of_a_run(self):
        mixture = _make_mixture(mask_rate=1.0, mask_cap=2, segment_length=4)
        mixture.memory_mask_run_len = 2
        sample = mixture.sample_segment(3)
        plan = sample["mask_plan"]
        self.assertEqual(int(plan.sum()), 2)
        burn_in = 3
        for offset, masked in enumerate(plan):
            step = sample["steps"][burn_in + offset]
            if masked:
                np.testing.assert_array_equal(step["video_clean"], step["video"])
            else:
                self.assertNotIn("video_clean", step)

    def test_linear_ramp_scales_rate_with_sample_ordinal(self):
        ramped = _make_mixture(mask_rate=0.9, mask_cap=3, ramp_samples=40)
        unramped = _make_mixture(mask_rate=0.9, mask_cap=3)

        self.assertEqual(int(ramped._sample_mask_plan(0).sum()), 0)
        for index in range(1, 40):
            subset = ramped._sample_mask_plan(index) & ~unramped._sample_mask_plan(index)
            self.assertFalse(bool(subset.any()))
        for index in range(40, 80):
            np.testing.assert_array_equal(
                ramped._sample_mask_plan(index), unramped._sample_mask_plan(index)
            )
        ramped_totals = sum(int(ramped._sample_mask_plan(i).sum()) for i in range(40))
        unramped_totals = sum(int(unramped._sample_mask_plan(i).sum()) for i in range(40))
        self.assertLess(ramped_totals, unramped_totals)


if __name__ == "__main__":
    unittest.main()
