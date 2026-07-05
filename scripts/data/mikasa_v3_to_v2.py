"""Convert MIKASA-Robo-VLA LeRobot v3.0 datasets to the v2.1 per-episode layout.

Why a converter and not a loader adapter: the gr00t loader
(`starVLA/dataloader/gr00t_lerobot/datasets.py`) assumes one parquet and one
mp4 *per episode* throughout (path patterns keyed by ``episode_index``,
per-episode timestamp-relative video decode, per-episode DataFrame caching).
LeRobot v3.0 concatenates episodes into shared ``file-XXX`` parquets/videos
with per-episode row/timestamp offsets in ``meta/episodes``.  Splitting the
files offline is ~200 lines here; teaching the loader about offsets would
touch data, video, language, integrity, and caching paths.

Emitted layout (mirrors the existing LIBERO v2.1 corpora):
  data/chunk-XXX/episode_XXXXXX.parquet   (row index reset to 0..L-1;
                                           get_language indexes by label)
  videos/chunk-XXX/{video_key}/episode_XXXXXX.mp4  (h264 re-encode; source av1)
  meta/{episodes.jsonl,tasks.jsonl,info.json,modality.json}

Stats (meta/stats_gr00t.json) are intentionally NOT written: the loader
computes them from the per-episode parquets on first init, exactly how the
existing corpora got theirs.

Usage:
  python scripts/data/mikasa_v3_to_v2.py \
      --src /lustre/.../memexp_stage/mikasa-data \
      --dst /lustre/.../memexp_stage/mikasa-lerobot \
      --tasks take_it_back_vla_v0 [...] [--limit-episodes 2] [--workers 8]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd

CHUNK_SIZE = 1000

MODALITY_JSON = {
    "state": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "action": {
        "x": {"start": 0, "end": 1},
        "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "primary_image": {"original_key": "observation.images.top"},
        "wrist_image": {"original_key": "observation.images.wrist"},
    },
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}


def _episode_chunk(ep: int) -> int:
    return ep // CHUNK_SIZE


def _write_episode_video(frames: np.ndarray, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out_path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames.shape[1], frames.shape[2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for frame_array in frames:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _split_video_file(
    src_video: Path,
    episodes: pd.DataFrame,
    video_key: str,
    dst_dir: Path,
    fps: int,
) -> None:
    """Decode one concatenated v3 video and write per-episode v2 mp4s."""
    lengths = episodes["length"].to_numpy()
    boundaries = np.concatenate(([0], np.cumsum(lengths)))
    # Episodes are stored back to back: from_timestamp must equal the
    # cumulative frame count / fps, else our split would be misaligned.
    from_ts = episodes[f"videos/{video_key}/from_timestamp"].to_numpy()
    np.testing.assert_allclose(from_ts, boundaries[:-1] / fps, atol=1.0 / (2 * fps))

    needed = int(boundaries[-1])  # < file total when --limit-episodes is set
    frames = []
    with av.open(str(src_video)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == needed:
                break
    frames = np.stack(frames)
    if len(frames) != needed:
        raise ValueError(
            f"{src_video}: decoded {len(frames)} frames, meta says {needed}"
        )

    for row, start, stop in zip(
        episodes.itertuples(), boundaries[:-1], boundaries[1:]
    ):
        ep = int(row.episode_index)
        out_path = (
            dst_dir
            / f"videos/chunk-{_episode_chunk(ep):03d}/{video_key}/episode_{ep:06d}.mp4"
        )
        _write_episode_video(frames[start:stop], out_path, fps)


def convert_task(src_dir: Path, dst_dir: Path, limit_episodes: int | None) -> str:
    # meta/info.json is written last, so its presence marks a completed task
    # and makes reruns (e.g. after a timeout) idempotent.
    if (dst_dir / "meta/info.json").exists():
        return f"{src_dir.name}: already converted, skipping"
    info = json.loads((src_dir / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0", src_dir
    fps = int(info["fps"])
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

    episode_files = sorted((src_dir / "meta/episodes").glob("chunk-*/file-*.parquet"))
    episodes = pd.concat([pd.read_parquet(p) for p in episode_files])
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    if limit_episodes is not None:
        episodes = episodes.iloc[:limit_episodes]

    dst_dir.mkdir(parents=True, exist_ok=True)

    # --- data: split concatenated parquets into per-episode files -----------
    for (chunk_idx, file_idx), group in episodes.groupby(
        ["data/chunk_index", "data/file_index"]
    ):
        data = pd.read_parquet(
            src_dir / info["data_path"].format(chunk_index=chunk_idx, file_index=file_idx)
        )
        for row in group.itertuples():
            ep = int(row.episode_index)
            rows = data.iloc[row.dataset_from_index : row.dataset_to_index]
            if len(rows) != int(row.length):
                raise ValueError(f"{src_dir} ep{ep}: row count != meta length")
            assert (rows["episode_index"] == ep).all()
            out = (
                dst_dir
                / f"data/chunk-{_episode_chunk(ep):03d}/episode_{ep:06d}.parquet"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            rows.reset_index(drop=True).to_parquet(out)

    # --- videos --------------------------------------------------------------
    for video_key in video_keys:
        for (chunk_idx, file_idx), group in episodes.groupby(
            [f"videos/{video_key}/chunk_index", f"videos/{video_key}/file_index"]
        ):
            src_video = src_dir / info["video_path"].format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            _split_video_file(src_video, group, video_key, dst_dir, fps)

    # --- meta ----------------------------------------------------------------
    meta_dir = dst_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for row in episodes.itertuples():
            f.write(
                json.dumps(
                    {
                        "episode_index": int(row.episode_index),
                        "tasks": list(row.tasks),
                        "length": int(row.length),
                    }
                )
                + "\n"
            )

    tasks = pd.read_parquet(src_dir / "meta/tasks.parquet")
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task_str, row in tasks.iterrows():
            f.write(
                json.dumps({"task_index": int(row["task_index"]), "task": task_str})
                + "\n"
            )

    features = json.loads(json.dumps(info["features"]))  # deep copy
    for key in video_keys:
        features[key]["info"]["video.codec"] = "h264"
    total_frames = int(episodes["length"].sum())
    v2_info = {
        "codebase_version": "v2.1",
        "robot_type": info["robot_type"],
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(video_keys),
        "total_chunks": _episode_chunk(int(episodes["episode_index"].max())) + 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(v2_info, indent=4))
    (meta_dir / "modality.json").write_text(json.dumps(MODALITY_JSON, indent=4))
    return f"{src_dir.name}: {len(episodes)} episodes, {total_frames} frames"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--prefix", default="mikasa_", help="prefix for the output dataset dir names"
    )
    args = parser.parse_args()

    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in args.tasks:
            src_dir = args.src / task
            if not src_dir.is_dir():
                raise FileNotFoundError(src_dir)
            dst_dir = args.dst / f"{args.prefix}{task}"
            jobs.append(pool.submit(convert_task, src_dir, dst_dir, args.limit_episodes))
        for job in jobs:
            print(job.result(), flush=True)


if __name__ == "__main__":
    main()
