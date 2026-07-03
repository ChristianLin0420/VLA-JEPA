import unittest
from unittest import mock

import numpy as np

from scripts.offline_eval.lattice import (
    dataset_delta_bounds,
    decision_lattice,
    first_lattice_base,
    segment_base_indices,
    valid_segment_start_count,
)
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

    def __len__(self):
        return int(self.trajectory_lengths.sum())

    def get_step_data(self, trajectory_id, base_index):
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


class FirstLatticeBaseTest(unittest.TestCase):
    def test_hand_computed_cases(self):
        for min_delta, stride, expected in (
            (0, 7, 0),
            (5, 7, 0),
            (-1, 7, 7),
            (-7, 7, 7),
            (-8, 7, 14),
            (0, 4, 0),
            (-1, 4, 4),
        ):
            with self.subTest(min_delta=min_delta, stride=stride):
                self.assertEqual(first_lattice_base(min_delta, stride), expected)

    def test_stride_validation(self):
        with self.assertRaisesRegex(ValueError, "stride"):
            first_lattice_base(0, 0)


class DecisionLatticeTest(unittest.TestCase):
    def test_training_config_cases(self):
        # Training deltas: video [0..7], action [0..6] -> min 0, max 7; stride 7.
        cases = {
            85: list(range(0, 78, 7)),  # 12 decisions, last base 77
            29: [0, 7, 14, 21],
            8: [0],
            7: [],
            0: [],
        }
        for length, expected in cases.items():
            with self.subTest(length=length):
                self.assertEqual(
                    decision_lattice(length, stride=7, min_delta=0, max_delta=7).tolist(),
                    expected,
                )

    def test_negative_delta_shifts_first_base(self):
        bases = decision_lattice(40, stride=4, min_delta=-1, max_delta=2)
        self.assertEqual(int(bases[0]), 4)
        self.assertEqual(int(bases[-1]), 36)


class SegmentStartCountTest(unittest.TestCase):
    def test_training_config_cases(self):
        # K=4 supervised decisions at stride 7, max_delta 7 -> minimum length 29.
        for length, expected in ((28, 0), (29, 1), (35, 1), (36, 2), (85, 9), (505, 69)):
            with self.subTest(length=length):
                self.assertEqual(
                    valid_segment_start_count(
                        length, stride=7, segment_length=4, min_delta=0, max_delta=7
                    ),
                    expected,
                )

    def test_negative_delta_case(self):
        self.assertEqual(
            valid_segment_start_count(
                40, stride=4, segment_length=3, min_delta=-1, max_delta=2
            ),
            7,
        )


class SegmentBaseIndicesTest(unittest.TestCase):
    def test_left_padding_matches_sampler(self):
        kwargs = {"stride": 4, "segment_length": 3, "burn_in": 3}
        for start, expected in (
            (0, [-1, -1, -1, 0, 4, 8]),
            (4, [-1, -1, 0, 4, 8, 12]),
            (12, [0, 4, 8, 12, 16, 20]),
            (16, [4, 8, 12, 16, 20, 24]),
        ):
            with self.subTest(start=start):
                self.assertEqual(segment_base_indices(start, **kwargs).tolist(), expected)

    def test_negative_delta_raises_lattice_floor(self):
        kwargs = {"stride": 4, "segment_length": 3, "burn_in": 3, "min_delta": -1}
        self.assertEqual(
            segment_base_indices(4, **kwargs).tolist(), [-1, -1, -1, 4, 8, 12]
        )
        self.assertEqual(
            segment_base_indices(12, **kwargs).tolist(), [-1, 4, 8, 12, 16, 20]
        )

    def test_off_lattice_start_rejected(self):
        with self.assertRaisesRegex(ValueError, "lattice"):
            segment_base_indices(5, stride=4, segment_length=3, burn_in=3)
        with self.assertRaisesRegex(ValueError, "lattice"):
            segment_base_indices(0, stride=4, segment_length=3, burn_in=3, min_delta=-1)


class ProductionSamplerParityTest(unittest.TestCase):
    """The lattice util must reproduce the training sampler bit-for-bit."""

    def test_catalog_parity(self):
        dataset = _FakeRawDataset()
        mixture = _make_mixture(dataset)
        catalog = mixture._segment_catalogs[0]
        min_delta, max_delta = dataset_delta_bounds(dataset.delta_indices)
        self.assertEqual(min_delta, catalog["min_delta"])
        self.assertEqual(max_delta, catalog["max_delta"])
        self.assertEqual(first_lattice_base(min_delta, 4), catalog["first_base"])
        for length, expected in zip(
            catalog["trajectory_lengths"], catalog["valid_start_counts"]
        ):
            self.assertEqual(
                valid_segment_start_count(
                    int(length),
                    stride=4,
                    segment_length=3,
                    min_delta=min_delta,
                    max_delta=max_delta,
                ),
                int(expected),
            )

    def test_sampled_segments_lie_on_the_util_lattice(self):
        dataset = _FakeRawDataset()
        mixture = _make_mixture(dataset)
        catalog = mixture._segment_catalogs[0]
        lengths = dict(
            zip(catalog["trajectory_ids"].tolist(), catalog["trajectory_lengths"].tolist())
        )
        for index in range(40):
            sample = mixture.sample_segment(index)
            supervised_start = int(sample["base_indices"][3])
            expected = segment_base_indices(
                supervised_start,
                stride=4,
                segment_length=3,
                burn_in=3,
                min_delta=catalog["min_delta"],
            )
            self.assertEqual(sample["base_indices"].tolist(), expected.tolist())
            lattice = set(
                decision_lattice(
                    lengths[sample["episode_id"]],
                    stride=4,
                    min_delta=catalog["min_delta"],
                    max_delta=catalog["max_delta"],
                ).tolist()
            )
            real = sample["base_indices"][sample["sequence_valid"]]
            self.assertTrue(set(real.tolist()) <= lattice)


if __name__ == "__main__":
    unittest.main()
