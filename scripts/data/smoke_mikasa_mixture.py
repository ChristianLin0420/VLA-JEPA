"""Standalone dataloader smoke for the all_robot_mikasa mixture.

Builds the exact vla_data dataset the trainer would build from
scripts/config/vlajepa_memv1_mikasa_ft.yaml (no GPUs, no slurm), draws
segments until at least the requested number of MIKASA samples has been seen,
and checks shapes, dtypes, masks, action ranges, and the per-tag merged
normalization statistics. Exercises the same code path as
starVLA/dataloader/__init__.py:build_dataloader ('lerobot_datasets').

Run from the repo root:
  python scripts/data/smoke_mikasa_mixture.py \
      --config scripts/config/vlajepa_memv1_mikasa_ft.yaml --num-samples 20
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starVLA.dataloader.lerobot_datasets import get_vla_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="scripts/config/vlajepa_memv1_mikasa_ft.yaml"
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--min-mikasa", type=int, default=3)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
        delete_pause_frame=cfg.datasets.vla_data.get("delete_pause_frame", True),
        sample_mode=cfg.datasets.vla_data.get("sample_mode", "single_step"),
        segment_length=cfg.datasets.vla_data.get("segment_length", 4),
        burn_in_max_decisions=cfg.datasets.vla_data.get("burn_in_max_decisions", 8),
        segment_stride=cfg.datasets.vla_data.get(
            "segment_stride", cfg.framework.action_model.action_horizon
        ),
    )

    names = [d.dataset_name for d in dataset.datasets]
    weights = dataset.dataset_sampling_weights
    mikasa_mask = np.array([n.startswith("mikasa_") for n in names])
    print(f"datasets in mixture: {len(names)} ({int(mikasa_mask.sum())} mikasa)")
    print(f"mikasa draw probability: {weights[mikasa_mask].sum():.4f} (target ~0.20)")
    for name, w, n_starts in zip(names, weights, dataset.dataset_lengths):
        print(f"  {name:55s} p={w:.4f} starts/steps={n_starts}")

    horizon = int(cfg.framework.action_model.action_horizon)
    n_frames = int(cfg.framework.vj2_model.num_frames)
    vres = int(cfg.datasets.vla_data.video_resolution_size)
    ires = int(cfg.datasets.vla_data.resolution_size)
    seg_len = int(cfg.datasets.vla_data.burn_in_max_decisions) + int(
        cfg.datasets.vla_data.segment_length
    )

    counts: collections.Counter = collections.Counter()
    mikasa_seen = 0
    rng = np.random.default_rng(0)
    index = 0
    while (
        sum(counts.values()) < args.num_samples or mikasa_seen < args.min_mikasa
    ):
        # deterministic path first, then random probing until enough mikasa draws
        idx = index if index < args.num_samples else int(rng.integers(len(dataset)))
        index += 1
        sample = dataset[idx]
        counts[sample["dataset_id"]] += 1
        is_mikasa = sample["dataset_id"].startswith("mikasa_")
        mikasa_seen += int(is_mikasa)

        assert len(sample["steps"]) == seg_len, len(sample["steps"])
        assert sample["loss_mask"].sum() == int(cfg.datasets.vla_data.segment_length)
        supervised = [s for s in sample["steps"] if s is not None]
        assert len(supervised) == int(sample["sequence_valid"].sum())
        step = supervised[-1]
        video = step["video"]
        assert video.shape == (2, n_frames, vres, vres, 3), video.shape
        assert step["action"].shape == (horizon, 7), step["action"].shape
        # single-view corpora (bridge/fractal) legitimately emit 1 image and
        # droid may carry empty instructions; strict checks are mikasa-only.
        assert all(im.size == (ires, ires) for im in step["image"])
        if is_mikasa:
            assert len(step["image"]) == 2, len(step["image"])
            assert isinstance(step["lang"], str) and step["lang"]
            a = np.asarray(step["action"], dtype=np.float32)
            assert np.abs(a).max() <= 1.0 + 1e-3, (a.min(), a.max())
            print(
                f"mikasa sample: {sample['dataset_id']} ep={sample['episode_id']} "
                f"bases={sample['base_indices'].tolist()} lang={step['lang'][:60]!r} "
                f"action[min,max]=[{a.min():.3f},{a.max():.3f}]"
            )

    print("\ndraw counts:", dict(counts))

    for tag, meta in dataset.merged_metadata.items():
        stats = meta.statistics.action["x"]
        print(
            f"tag={tag:16s} action.x min={stats.min[0]:+.4f} max={stats.max[0]:+.4f} "
            f"q01={stats.q01[0]:+.4f} q99={stats.q99[0]:+.4f}"
        )
    assert "new_embodiment" in dataset.merged_metadata, "mikasa stats block missing"
    dataset.save_dataset_statistics("/tmp/mikasa_smoke_dataset_statistics.json")
    print("wrote /tmp/mikasa_smoke_dataset_statistics.json")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
