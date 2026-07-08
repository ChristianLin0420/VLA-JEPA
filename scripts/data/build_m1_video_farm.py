"""Build the memv3 M1 robot-video farm: SSv2-style symlink dir + labels file.

VideoFolderDataset expects integer filenames and an ``id;text`` label CSV, so
robot episode mp4s (main camera only) are symlinked as ``<id>.mp4``.  Episodes
shorter than the M1 sampling span are skipped by the loader's retry logic, not
here.  Idempotent: wipes and rebuilds the farm dir.

  python scripts/data/build_m1_video_farm.py \
      --out /lustre/fsw/portfolios/edgeai/users/chrislin/memexp_stage/m1_video_farm
"""

import argparse
import csv
from pathlib import Path

DATA_ROOT = Path("/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot")
# Vanilla LIBERO + libero_mem + the 14 C0-certified MIKASA anchors.
DATASETS = [
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
    "libero_mem_1.0.0_lerobot",
    "mikasa_shell_game_shuffle_color_lamp_touch_vla_v0",
    "mikasa_gather_and_recall_5_vla_v0",
    "mikasa_batteries_checker_easy_3_vla_v0",
    "mikasa_gather_and_recall_3_vla_v0",
    "mikasa_shell_game_shuffle_touch_vla_v0",
    "mikasa_trace_shape_medium_vla_v0",
    "mikasa_chain_of_colors_7_vla_v0",
    "mikasa_bunch_of_colors_5_vla_v0",
    "mikasa_blink_count_button_press_medium_vla_v0",
    "mikasa_seq_of_colors_5_vla_v0",
    "mikasa_chain_of_colors_5_vla_v0",
    "mikasa_chain_of_colors_3_vla_v0",
    "mikasa_take_it_back_vla_v0",
    "mikasa_timed_transfer_medium_vla_v0",
]
WRIST_MARKERS = ("wrist", "hand")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        for stale in out.glob("*"):
            stale.unlink()
    out.mkdir(parents=True, exist_ok=True)

    rows, next_id = [], 0
    for dataset in DATASETS:
        video_root = DATA_ROOT / dataset / "videos"
        cameras = sorted(
            camera
            for chunk in sorted(video_root.glob("chunk-*"))
            for camera in chunk.iterdir()
            if camera.is_dir()
        )
        main_cameras = [
            camera for camera in {c.name for c in cameras}
            if not any(marker in camera.lower() for marker in WRIST_MARKERS)
        ]
        if not main_cameras:
            raise RuntimeError(f"no main camera found for {dataset}")
        camera_name = sorted(main_cameras)[0]
        episodes = sorted(video_root.glob(f"chunk-*/{camera_name}/episode_*.mp4"))
        if not episodes:
            raise RuntimeError(f"no episodes for {dataset}/{camera_name}")
        for episode in episodes:
            (out / f"{next_id}.mp4").symlink_to(episode)
            rows.append((next_id, f"robot manipulation episode ({dataset})"))
            next_id += 1
        print(f"{dataset}: {len(episodes)} episodes via {camera_name}")

    with open(out / "labels.csv", "w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerows(rows)
    print(f"farm complete: {next_id} episodes -> {out}")


if __name__ == "__main__":
    main()
