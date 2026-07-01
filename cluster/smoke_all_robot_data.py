#!/usr/bin/env python3
"""Decode representative samples from every dataset used by full co-training."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


EXPECTED_ALL_ROBOT_DATASETS = (
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
    "droid_lerobot",
    "bridge_orig_1.0.0_lerobot",
    "fractal20220817_data_0.1.0_lerobot",
)
SSV2_EXPECTED_VIDEOS = 220_847
SSV2_EXPECTED_LABELS = 193_690


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.dataloader.video_datasets import VideoFolderDataset

    cfg = OmegaConf.load(args.config)
    mixture = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        action_horizon=int(cfg.framework.action_model.action_horizon),
        video_horizon=int(cfg.framework.vj2_model.num_frames),
        seed=int(cfg.seed),
    )

    if str(cfg.datasets.vla_data.data_mix) != "all_robot":
        raise RuntimeError(
            f"smoke config must use data_mix=all_robot, got {cfg.datasets.vla_data.data_mix!r}"
        )
    actual_dataset_names = tuple(dataset.dataset_name for dataset in mixture.datasets)
    if actual_dataset_names != EXPECTED_ALL_ROBOT_DATASETS:
        raise RuntimeError(
            "all_robot mixture membership/order mismatch: "
            f"actual={actual_dataset_names}, expected={EXPECTED_ALL_ROBOT_DATASETS}"
        )

    robot_results = []
    original_sample_step = mixture.sample_step
    try:
        for dataset in mixture.datasets:
            step_index = len(dataset) // 2
            trajectory_id, base_index = dataset.all_steps[step_index]

            # Exercise the real mixture-level transform/resize/packing path while
            # pinning sampling to this exact dataset and step.  The pin also makes
            # mixture retry attempts target the same sample instead of hiding a
            # bad decode behind a random replacement.
            def fixed_sample_step(_index, selected=dataset, selected_step=step_index):
                trajectory, base = selected.all_steps[selected_step]
                return selected, trajectory, base

            mixture.sample_step = fixed_sample_step
            item = mixture[step_index]

            action = np.asarray(item["action"])
            expected_action_shape = (
                int(cfg.framework.action_model.action_horizon),
                int(cfg.framework.action_model.action_dim),
            )
            if action.shape != expected_action_shape or not np.isfinite(action).all():
                raise RuntimeError(
                    f"{dataset.dataset_name} action is invalid: "
                    f"shape={action.shape}, expected={expected_action_shape}, "
                    f"finite={bool(np.isfinite(action).all())}"
                )

            video = np.asarray(item["video"])
            expected_video_shape = (
                2,
                int(cfg.framework.vj2_model.num_frames),
                int(cfg.datasets.vla_data.video_resolution_size),
                int(cfg.datasets.vla_data.video_resolution_size),
                3,
            )
            if video.shape != expected_video_shape or not np.isfinite(video).all():
                raise RuntimeError(
                    f"{dataset.dataset_name} video is invalid: "
                    f"shape={video.shape}, expected={expected_video_shape}, "
                    f"finite={bool(np.isfinite(video).all())}"
                )

            language = item.get("lang")
            if not isinstance(language, str) or not language.strip():
                raise RuntimeError(
                    f"{dataset.dataset_name} has invalid language value {language!r}"
                )

            images = item.get("image", [])
            if not isinstance(images, list) or not images:
                raise RuntimeError(f"{dataset.dataset_name} has no decoded images")
            expected_image_shape = (
                int(cfg.datasets.vla_data.resolution_size),
                int(cfg.datasets.vla_data.resolution_size),
                3,
            )
            image_shapes = []
            for image in images:
                image_array = np.asarray(image)
                image_shapes.append(list(image_array.shape))
                if image_array.shape != expected_image_shape or not np.isfinite(image_array).all():
                    raise RuntimeError(
                        f"{dataset.dataset_name} image is invalid: "
                        f"shape={image_array.shape}, expected={expected_image_shape}, "
                        f"finite={bool(np.isfinite(image_array).all())}"
                    )

            sample = {
                "step_index": step_index,
                "trajectory_id": int(trajectory_id),
                "base_index": int(base_index),
                "action_shape": list(action.shape),
                "video_shape": list(video.shape),
                "image_shapes": image_shapes,
                "language": True,
            }
            robot_results.append(
                {"dataset": dataset.dataset_name, "steps": len(dataset), "sample": sample}
            )
            print(f"[smoke] {dataset.dataset_name}: mixture transform sample passed", flush=True)
    finally:
        mixture.sample_step = original_sample_step

    video_dataset = VideoFolderDataset(
        video_dir=str(cfg.datasets.video_data.video_dir),
        text_file=str(cfg.datasets.video_data.text_file),
        n_frames=int(cfg.framework.vj2_model.num_frames),
        extensions=tuple(cfg.datasets.video_data.extensions),
        crop_h_size=int(cfg.datasets.video_data.video_resolution_size),
        crop_w_size=int(cfg.datasets.video_data.video_resolution_size),
        max_retry=10,
        expected_video_count=SSV2_EXPECTED_VIDEOS,
        expected_label_count=SSV2_EXPECTED_LABELS,
    )
    if len(video_dataset) != SSV2_EXPECTED_VIDEOS:
        raise RuntimeError(
            f"SSV2 video count mismatch: {len(video_dataset)}, "
            f"expected {SSV2_EXPECTED_VIDEOS}"
        )
    if video_dataset.label_count != SSV2_EXPECTED_LABELS:
        raise RuntimeError(
            f"SSV2 label count mismatch: {video_dataset.label_count}, "
            f"expected {SSV2_EXPECTED_LABELS}"
        )

    ssv2_samples = []
    for index in (0, len(video_dataset) // 2, len(video_dataset) - 1):
        random.seed(int(cfg.seed) + index)
        video, text = video_dataset._load_video(index)
        if tuple(video.shape) != (
            int(cfg.framework.vj2_model.num_frames),
            int(cfg.datasets.video_data.video_resolution_size),
            int(cfg.datasets.video_data.video_resolution_size),
            3,
        ) or not np.isfinite(video).all():
            raise RuntimeError(
                f"SSV2[{index}] invalid video: shape={video.shape}, "
                f"finite={bool(np.isfinite(video).all())}"
            )
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"SSV2[{index}] has invalid text {text!r}")
        ssv2_samples.append(
            {
                "index": index,
                "filename": video_dataset.video_files[index],
                "shape": list(video.shape),
                "text": True,
            }
        )
    print(f"[smoke] SSV2: decoded {len(ssv2_samples)} samples", flush=True)

    result = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "robot": robot_results,
        "ssv2_count": len(video_dataset),
        "ssv2_samples": ssv2_samples,
    }
    write_json_atomic(args.state_dir / "data_smoke_complete.json", result)
    print("[smoke] all_robot + SSV2 decode smoke PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
