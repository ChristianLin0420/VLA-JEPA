import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    SAMPLE_MODE_CONTIGUOUS_SEGMENT,
    SAMPLE_MODE_SINGLE_STEP,
)
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


class _FakeRawDataset:
    """Small in-memory stand-in exposing only the production sampler contract."""

    def __init__(
        self,
        name="fake",
        trajectory_ids=(10, 11, 12),
        trajectory_lengths=(9, 42, 31),
        delta_indices=None,
    ):
        self.dataset_name = name
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
        self.trajectory_lengths = np.asarray(trajectory_lengths, dtype=np.int64)
        self.delta_indices = delta_indices or {
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
        self._all_steps_accesses = 0
        self._all_steps = [
            (int(episode_id), base_index)
            for episode_id, length in zip(
                self.trajectory_ids, self.trajectory_lengths
            )
            for base_index in range(int(length))
        ]

    def __len__(self):
        return len(self._all_steps)

    def __str__(self):
        return self.dataset_name

    @property
    def all_steps(self):
        self._all_steps_accesses += 1
        return self._all_steps

    def get_step_data(self, trajectory_id, base_index):
        if trajectory_id not in self.trajectory_ids:
            raise AssertionError(f"unknown trajectory {trajectory_id}")
        video = np.full((3, 4, 4, 3), base_index % 255, dtype=np.uint8)
        action = np.arange(base_index, base_index + 2, dtype=np.float32)[:, None]
        return {
            "video.primary": video,
            "action.delta": action,
            "language.task": [f"episode-{trajectory_id}"],
        }


def _make_mixture(dataset, **overrides):
    kwargs = {
        "data_mixture": [(dataset, 1.0)],
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
        return LeRobotMixtureDataset(**kwargs)


def _signature(sample):
    return (
        sample["dataset_id"],
        sample["episode_id"],
        tuple(sample["base_indices"].tolist()),
    )


class ContiguousSegmentSamplingTest(unittest.TestCase):
    def test_default_mode_preserves_single_step_sampling(self):
        dataset = _FakeRawDataset()
        with mock.patch.object(LeRobotMixtureDataset, "update_metadata", return_value=None):
            mixture = LeRobotMixtureDataset(
                [(dataset, 1.0)],
                mode="train",
                balance_dataset_weights=False,
                balance_trajectory_weights=False,
                resolution_size=4,
                video_resolution_size=4,
            )

        self.assertEqual(mixture.sample_mode, SAMPLE_MODE_SINGLE_STEP)
        selected_dataset, episode_id, base_index = mixture.sample_step(3)
        self.assertIs(selected_dataset, dataset)
        self.assertIn((int(episode_id), int(base_index)), dataset._all_steps)
        self.assertGreater(dataset._all_steps_accesses, 0)

    def test_segment_mode_never_uses_filtered_all_steps(self):
        dataset = _FakeRawDataset()
        mixture = _make_mixture(dataset)
        with mock.patch(
            "starVLA.dataloader.gr00t_lerobot.datasets.random.randint",
            side_effect=AssertionError("global random must not be used"),
        ):
            sample = mixture[7]

        self.assertEqual(dataset._all_steps_accesses, 0)
        self.assertEqual(len(sample["steps"]), 6)
        self.assertEqual(sample["base_indices"].dtype, np.int64)
        for key in (
            "segment_start",
            "is_first",
            "is_last",
            "sequence_valid",
            "loss_mask",
            "update_mask",
        ):
            self.assertEqual(sample[key].dtype, np.bool_)
            self.assertEqual(sample[key].shape, (6,))

    def test_segment_decode_failure_is_fail_fast_not_randomly_substituted(self):
        dataset = _FakeRawDataset()
        mixture = _make_mixture(dataset)
        with mock.patch.object(
            dataset, "get_step_data", side_effect=RuntimeError("decode failed")
        ), mock.patch(
            "starVLA.dataloader.gr00t_lerobot.datasets.random.randint"
        ) as global_retry:
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                mixture[2]
        global_retry.assert_not_called()

    def test_selection_is_deterministic_by_epoch_index_and_seed(self):
        first = _make_mixture(_FakeRawDataset())
        second = _make_mixture(_FakeRawDataset())

        epoch_zero_a = [_signature(first.sample_segment(i)) for i in range(32)]
        epoch_zero_b = [_signature(second.sample_segment(i)) for i in range(32)]
        self.assertEqual(epoch_zero_a, epoch_zero_b)

        first.set_epoch(1)
        epoch_one = [_signature(first.sample_segment(i)) for i in range(32)]
        self.assertTrue(any(a != b for a, b in zip(epoch_zero_a, epoch_one)))

        evaluation = _make_mixture(_FakeRawDataset(), mode="val")
        val_zero = [_signature(evaluation.sample_segment(i)) for i in range(8)]
        evaluation.set_epoch(99)
        val_later = [_signature(evaluation.sample_segment(i)) for i in range(8)]
        self.assertEqual(val_zero, val_later)

    def test_supervised_k_is_fully_valid_and_on_raw_stride(self):
        dataset = _FakeRawDataset()
        mixture = _make_mixture(dataset)
        lengths = dict(
            zip(dataset.trajectory_ids.tolist(), dataset.trajectory_lengths.tolist())
        )

        for index in range(40):
            sample = mixture.sample_segment(index)
            supervised = sample["base_indices"][3:]
            self.assertTrue(np.all(np.diff(supervised) == 4))
            self.assertTrue(np.all(supervised >= 0))
            self.assertLess(int(supervised[-1]) + 2, lengths[sample["episode_id"]])
            self.assertTrue(np.all(sample["sequence_valid"][3:]))
            self.assertTrue(np.all(sample["loss_mask"][3:]))
            self.assertTrue(np.all(sample["update_mask"][3:]))
            self.assertEqual(int(np.count_nonzero(sample["segment_start"])), 1)
            self.assertTrue(bool(sample["segment_start"][3]))
            # The nine-frame episode cannot contain K=3 decisions at stride 4
            # plus max modality delta 2, and must never be selected.
            self.assertNotEqual(sample["episode_id"], 10)
            for base_index, step in zip(
                sample["base_indices"][sample["sequence_valid"]],
                [step for step in sample["steps"] if step is not None],
            ):
                self.assertEqual(float(step["action"][0, 0]), float(base_index))

    def test_burn_in_is_bounded_fixed_and_left_padded_with_none(self):
        mixture = _make_mixture(_FakeRawDataset())
        early = None
        late = None
        for index in range(200):
            candidate = mixture.sample_segment(index)
            start = int(candidate["base_indices"][3])
            if start == 0 and early is None:
                early = candidate
            if start >= 12 and late is None:
                late = candidate
            if early is not None and late is not None:
                break

        self.assertIsNotNone(early)
        self.assertEqual(early["base_indices"][:3].tolist(), [-1, -1, -1])
        self.assertEqual(early["steps"][:3], [None, None, None])
        self.assertEqual(early["sequence_valid"][:3].tolist(), [False] * 3)
        self.assertEqual(early["update_mask"][:3].tolist(), [False] * 3)
        self.assertTrue(bool(early["is_first"][3]))

        self.assertIsNotNone(late)
        start = int(late["base_indices"][3])
        self.assertEqual(
            late["base_indices"][:3].tolist(),
            [start - 12, start - 8, start - 4],
        )
        self.assertTrue(all(step is not None for step in late["steps"][:3]))
        self.assertEqual(late["loss_mask"][:3].tolist(), [False] * 3)
        self.assertEqual(late["update_mask"][:3].tolist(), [True] * 3)

    def test_negative_deltas_shift_first_fully_valid_lattice_point(self):
        dataset = _FakeRawDataset(
            trajectory_ids=(4,),
            trajectory_lengths=(40,),
            delta_indices={
                "video.primary": np.asarray([-1, 0, 2]),
                "action.delta": np.asarray([0, 1]),
                "language.task": np.asarray([0]),
            },
        )
        mixture = _make_mixture(dataset)
        for index in range(20):
            sample = mixture.sample_segment(index)
            real = sample["base_indices"][sample["sequence_valid"]]
            self.assertGreaterEqual(int(real[0]), 4)
            self.assertGreaterEqual(int(real[0]) - 1, 0)
            self.assertLess(int(real[-1]) + 2, 40)
            self.assertFalse(np.any(sample["is_first"]))

    def test_no_dataset_with_a_valid_segment_fails_loudly(self):
        dataset = _FakeRawDataset(
            trajectory_ids=(1, 2), trajectory_lengths=(3, 4)
        )
        with self.assertRaisesRegex(ValueError, "No valid datasets"):
            _make_mixture(dataset)

    def test_segment_configuration_validation(self):
        dataset = _FakeRawDataset()
        for override, message in (
            ({"segment_length": 0}, "segment_length"),
            ({"burn_in_max_decisions": -1}, "burn_in_max_decisions"),
            ({"segment_stride": 0}, "segment_stride"),
            ({"sample_mode": "mystery"}, "sample_mode"),
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    _make_mixture(dataset, **override)

    def test_data_config_is_plumbed_to_single_and_mixture_datasets(self):
        config = OmegaConf.create(
            {
                "data_root_dir": "/tmp/data",
                "data_mix": "libero_goal",
                "with_state": False,
                "resolution_size": 12,
                "video_resolution_size": 16,
                "delete_pause_frame": False,
                "sample_mode": "contiguous_segment",
                "segment_length": 5,
                "burn_in_max_decisions": 2,
                "segment_stride": 6,
            }
        )
        sentinel_single = object()
        sentinel_mixture = object()
        with mock.patch(
            "starVLA.dataloader.lerobot_datasets.make_LeRobotSingleDataset",
            return_value=sentinel_single,
        ) as make_single, mock.patch(
            "starVLA.dataloader.lerobot_datasets.LeRobotMixtureDataset",
            return_value=sentinel_mixture,
        ) as make_mixture:
            result = get_vla_dataset(
                config,
                action_horizon=7,
                video_horizon=8,
            )

        self.assertIs(result, sentinel_mixture)
        single_kwargs = make_single.call_args.kwargs
        self.assertFalse(single_kwargs["delete_pause_frame"])
        self.assertEqual(single_kwargs["sample_mode"], "contiguous_segment")
        self.assertEqual(single_kwargs["action_horizon"], 7)
        self.assertEqual(single_kwargs["video_horizon"], 8)
        mixture_kwargs = make_mixture.call_args.kwargs
        self.assertEqual(mixture_kwargs["sample_mode"], "contiguous_segment")
        self.assertEqual(mixture_kwargs["segment_length"], 5)
        self.assertEqual(mixture_kwargs["burn_in_max_decisions"], 2)
        self.assertEqual(mixture_kwargs["segment_stride"], 6)


if __name__ == "__main__":
    unittest.main()
