import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from torch.utils.data import BatchSampler, SequentialSampler
from accelerate.data_loader import BatchSamplerShard

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "enumerate_consumed_episodes",
    REPO_ROOT / "scripts" / "data" / "enumerate_consumed_episodes.py",
)
enumerator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(enumerator)

SEGMENT_LENGTH = 3
SEGMENT_STRIDE = 4
MIN_DELTA, MAX_DELTA = 0, 2
SEED = 17

# Third dataset has no fully valid K=3 segment at stride 4 with max delta 2
# (needs length >= 11) and must be dropped identically by both paths.
REGISTRY = {
    "synthetic_a": [(10, 9), (11, 42), (12, 31), (13, 200), (14, 5)],
    "synthetic_b": [(0, 64), (1, 29), (2, 28), (3, 300), (4, 11)],
    "synthetic_short": [(0, 10), (1, 8)],
}


class _FakeRawDataset:
    """In-memory stand-in exposing only the production sampler contract."""

    def __init__(self, name, trajectory_ids, trajectory_lengths):
        self.dataset_name = name
        self.trajectory_ids = np.asarray(trajectory_ids, dtype=np.int64)
        self.trajectory_lengths = np.asarray(trajectory_lengths, dtype=np.int64)
        self.delta_indices = {
            "video.primary": np.arange(MAX_DELTA + 1, dtype=np.int64),
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


def _write_registry(root: Path) -> None:
    for name, episodes in REGISTRY.items():
        meta_dir = root / name / "meta"
        meta_dir.mkdir(parents=True)
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for episode_index, length in episodes:
                f.write(
                    json.dumps(
                        {"episode_index": episode_index, "tasks": ["t"], "length": length}
                    )
                    + "\n"
                )


def _make_real_mixture() -> LeRobotMixtureDataset:
    datasets = [
        (_FakeRawDataset(name, *zip(*episodes)), 1.0)
        for name, episodes in REGISTRY.items()
    ]
    with mock.patch.object(LeRobotMixtureDataset, "update_metadata", return_value=None):
        return LeRobotMixtureDataset(
            datasets,
            mode="train",
            balance_dataset_weights=False,
            balance_trajectory_weights=False,
            with_state=False,
            resolution_size=4,
            video_resolution_size=4,
            seed=SEED,
            sample_mode="contiguous_segment",
            segment_length=SEGMENT_LENGTH,
            burn_in_max_decisions=3,
            segment_stride=SEGMENT_STRIDE,
        )


def _make_replay(registry_root: Path) -> "enumerator.SamplerReplay":
    names, catalogs = [], []
    for name in REGISTRY:
        ids, lengths = enumerator.read_episode_metadata(registry_root / name)
        catalogs.append(
            enumerator.build_catalog(
                ids, lengths, MIN_DELTA, MAX_DELTA, SEGMENT_LENGTH, SEGMENT_STRIDE
            )
        )
        names.append(name)
    return enumerator.SamplerReplay(
        names, catalogs, [1.0] * len(names), SEGMENT_STRIDE, SEED
    )


class ConsumedEnumeratorEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        _write_registry(root)
        cls.mixture = _make_real_mixture()
        cls.replay = _make_replay(root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_replay_matches_real_sampler_draw_for_draw(self):
        for epoch in (0, 1, 5):
            self.mixture.set_epoch(epoch)
            for index in range(300):
                dataset, _, _, episode_id, supervised_start = (
                    self.mixture._sample_segment_location(index)
                )
                replay_dataset_index, replay_episode, replay_start = (
                    self.replay.sample_location(epoch, index)
                )
                self.assertEqual(
                    (dataset.dataset_name, int(episode_id), int(supervised_start)),
                    (
                        self.replay.dataset_names[replay_dataset_index],
                        replay_episode,
                        replay_start,
                    ),
                    msg=f"divergence at epoch={epoch} index={index}",
                )

    def test_consumed_episode_sets_are_identical(self):
        real_consumed, replay_consumed = {}, {}
        self.mixture.set_epoch(0)
        for index in range(400):
            dataset, _, _, episode_id, _ = self.mixture._sample_segment_location(index)
            real_consumed.setdefault(dataset.dataset_name, set()).add(int(episode_id))
            dataset_index, episode, _ = self.replay.sample_location(0, index)
            replay_consumed.setdefault(
                self.replay.dataset_names[dataset_index], set()
            ).add(episode)
        self.assertEqual(real_consumed, replay_consumed)
        # The zero-segment dataset must be dropped by both paths.
        self.assertNotIn("synthetic_short", self.replay.dataset_names)
        self.assertEqual(len(self.mixture.datasets), 2)

    def test_epoch_length_matches_real_mixture_len(self):
        self.assertEqual(self.replay.epoch_length(), len(self.mixture))

    def test_index_schedule_matches_accelerate_batch_sampler_shard(self):
        epoch_length, world_size = 53, 4  # 53 % 4 != 0 exercises the padded group
        shards = [
            list(
                BatchSamplerShard(
                    BatchSampler(
                        SequentialSampler(range(epoch_length)), batch_size=1, drop_last=False
                    ),
                    num_processes=world_size,
                    process_index=rank,
                )
            )
            for rank in range(world_size)
        ]
        per_rank = len(shards[0])
        self.assertEqual(enumerator.batches_per_epoch(epoch_length, world_size), per_rank)
        steps = 2 * per_rank + 3  # spans one full epoch plus a partial second
        expected = [
            shards[rank][step % per_rank][0]
            for step in range(steps)
            for rank in range(world_size)
        ]
        actual = [
            index
            for _, index in enumerator.iter_consumed_indices(
                steps, world_size, epoch_length
            )
        ]
        self.assertEqual(actual, expected)
        epochs = [
            epoch
            for epoch, _ in enumerator.iter_consumed_indices(steps, world_size, epoch_length)
        ]
        per_epoch = world_size * per_rank
        self.assertEqual(epochs[:per_epoch], [0] * per_epoch)
        self.assertEqual(epochs[per_epoch : 2 * per_epoch], [1] * per_epoch)
        self.assertEqual(epochs[2 * per_epoch :], [2] * (3 * world_size))

    def test_production_robot_types_have_expected_delta_bounds(self):
        for robot_type in ("libero_franka", "droid_libero", "oxe_bridge", "oxe_rt1"):
            self.assertEqual(
                enumerator.delta_bounds(robot_type, action_horizon=7, video_horizon=8),
                (0, 7),
                msg=robot_type,
            )

    def test_summary_counts_unseen_at_decision_thresholds(self):
        consumed = {"synthetic_a": {11, 13}, "synthetic_b": {0, 1, 2, 3, 4}}
        summary, unseen_records = enumerator.summarize(
            self.replay, consumed, SEGMENT_STRIDE
        )
        # synthetic_a keeps episodes {10..14}; unseen = {10, 12, 14} with
        # lengths {9, 31, 5} -> decisions {2, 7, 1} at stride 4.
        entry = summary["synthetic_a"]
        self.assertEqual(entry["episodes"], 5)
        self.assertEqual(entry["consumed"], 2)
        self.assertEqual(entry["unseen"], 3)
        self.assertEqual(entry["unseen_ge100"], 0)
        self.assertEqual(summary["synthetic_b"]["unseen"], 0)
        unseen_a = {r["episode_index"]: r for r in unseen_records if r["dataset"] == "synthetic_a"}
        self.assertEqual(set(unseen_a), {10, 12, 14})
        self.assertEqual(unseen_a[12]["length"], 31)
        self.assertEqual(unseen_a[12]["num_decisions"], 31 // SEGMENT_STRIDE)


if __name__ == "__main__":
    unittest.main()
