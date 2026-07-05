"""LIBERO-Mem robosuite HDF5 demos -> LeRobot v2.1 dataset (GR00T layout our trainer loads).

Input: `<TASK_NAME>_demo.hdf5` files from hf.co/datasets/libero-mem/LIBERO-Mem
(standard LIBERO demo layout: data/demo_<i>/{actions,obs/{agentview_rgb,
eye_in_hand_rgb,ee_states,gripper_states,...}}; keyboard-teleop human demos,
actions (T,7) robosuite delta-EE with gripper -1=open/+1=close).

Output mirrors our staged LIBERO conversions
(vlajepa_stage/datasets/lerobot/libero_*_no_noops_1.0.0_lerobot), verified
against episode_000000.parquet of libero_object_no_noops_1.0.0_lerobot:
  observation.state f32[8] = [ee_states(6: xyz + axis-angle), gripper_states(2)]
  action            f32[7] = [raw delta xyz+rot(6), gripper mapped (1-g)/2]
                             (reference gripper convention: 1=open, 0=close)
  videos: agentview_rgb -> observation.images.image,
          eye_in_hand_rgb -> observation.images.wrist_image  (256x256, fps 20)
  meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl,
        stats_gr00t.json,modality.json}

Deliberate conventions:
  - Language replicates libero.benchmark.grab_language_from_filename on the
    task name, KEEPING the leading task number ("1 pick up the bowl and place
    it back on the plate") so training text matches eval-time instructions.
  - No no-op filtering: keyboard demos pause a lot, but the loader's
    delete_pause_frame already skips zero-translation/zero-gripper-change
    steps at sample time, and dropping frames would corrupt the memory
    (counting) structure of the benchmark.
  - Split: last N (default 20) demos of each task by numeric demo id -> val,
    remainder -> train ("first-100 train / last-20 val" adapted: the released
    set has 82-100 demos/task, 961 total, not 120/task). Episodes are written
    train-block-first so info.json splits stay contiguous index ranges.
  - stats_gr00t.json is computed over ALL episodes (train+val), matching how
    the reference corpora ship a single whole-dataset stats file.

Run in the VLA_JEPA conda env:
  .../envs/VLA_JEPA/bin/python scripts/data/libero_mem_to_lerobot.py \
      --src /lustre/.../memexp_stage/libero-mem-data/LIBERO-Mem \
      --out /lustre/.../memexp_stage/libero-mem-lerobot/libero_mem_1.0.0_lerobot \
      --max-episodes-per-task 1   # unit smoke
"""

import argparse
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

FRONT_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist_image"
CHUNK_SIZE = 1000
FPS = 20

MODALITY = {
    "state": {
        "x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3}, "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5}, "yaw": {"start": 5, "end": 6},
        "pad": {"start": 6, "end": 7}, "gripper": {"start": 7, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2},
        "z": {"start": 2, "end": 3}, "roll": {"start": 3, "end": 4},
        "pitch": {"start": 4, "end": 5}, "yaw": {"start": 5, "end": 6},
        "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "primary_image": {"original_key": FRONT_KEY},
        "wrist_image": {"original_key": WRIST_KEY},
    },
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}

HF_SCHEMA_METADATA = json.dumps({"info": {"features": {
    "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 8, "_type": "Sequence"},
    "action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 7, "_type": "Sequence"},
    "timestamp": {"dtype": "float32", "_type": "Value"},
    "frame_index": {"dtype": "int64", "_type": "Value"},
    "episode_index": {"dtype": "int64", "_type": "Value"},
    "index": {"dtype": "int64", "_type": "Value"},
    "task_index": {"dtype": "int64", "_type": "Value"},
}}})


def grab_language_from_filename(x: str) -> str:
    """Verbatim port of libero.libero.benchmark.grab_language_from_filename.

    For LIBERO-Mem task names this KEEPS the leading task number
    ("KITCHEN_SCENE1_1_pick_up_..." -> "1 pick up ..."), which is exactly the
    instruction the benchmark serves at eval time.
    """
    if x[0].isupper():  # LIBERO-100 style
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8:].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7:].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]


def task_language(h5_path: Path) -> str:
    stem = h5_path.name
    if not stem.endswith("_demo.hdf5"):
        raise ValueError(f"unexpected file name: {stem}")
    return grab_language_from_filename(stem[: -len("_demo.hdf5")] + ".bddl")


def task_number(h5_path: Path) -> int:
    m = re.search(r"SCENE\d+_(\d+)_", h5_path.name)
    if not m:
        raise ValueError(f"cannot parse task number from {h5_path.name}")
    return int(m.group(1))


def natural_sorted_demos(keys) -> list[str]:
    pat = re.compile(r"^demo_(\d+)$")
    pairs = [(int(m.group(1)), k) for k in keys if (m := pat.match(k))]
    return [k for _, k in sorted(pairs)]


def vec_stats(arr: np.ndarray) -> dict:
    a = arr.reshape(len(arr), -1).astype(np.float64)
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "count": [len(a)],
    }


def image_stats(frames: np.ndarray, sample: int = 100) -> dict:
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
    a = frames[idx].astype(np.float64) / 255.0
    per_c = a.reshape(-1, a.shape[-1])
    fmt = lambda v: [[[float(x)]] for x in v]  # noqa: E731 - (3,1,1) nesting like reference
    return {
        "min": fmt(per_c.min(0)), "max": fmt(per_c.max(0)),
        "mean": fmt(per_c.mean(0)), "std": fmt(per_c.std(0)),
        "count": [len(idx)],
    }


def convert_episode(job: dict) -> dict:
    """Worker: one demo -> parquet + two mp4s; returns small arrays + stats."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import imageio

    out = Path(job["out"])
    episode_index, index_offset = job["episode_index"], job["index_offset"]
    with h5py.File(job["h5"], "r") as f:
        demo = f["data"][job["demo"]]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        ee = np.asarray(demo["obs"]["ee_states"], dtype=np.float32)
        grip = np.asarray(demo["obs"]["gripper_states"], dtype=np.float32)
        front = np.asarray(demo["obs"]["agentview_rgb"], dtype=np.uint8)
        wrist = np.asarray(demo["obs"]["eye_in_hand_rgb"], dtype=np.uint8)

    raw_gripper = actions[:, 6].copy()
    action = actions.copy()
    action[:, 6] = (1.0 - raw_gripper) / 2.0  # -1(open)->1, +1(close)->0
    state = np.concatenate([ee, grip], axis=1).astype(np.float32)
    n = len(action)
    if not (len(state) == len(front) == len(wrist) == n):
        raise ValueError(f"length mismatch in {job['h5']}:{job['demo']}")

    chunk = episode_index // CHUNK_SIZE
    table = pa.table({
        "observation.state": pa.FixedSizeListArray.from_arrays(
            pa.array(state.reshape(-1), type=pa.float32()), 8),
        "action": pa.FixedSizeListArray.from_arrays(
            pa.array(action.reshape(-1), type=pa.float32()), 7),
        "timestamp": pa.array((np.arange(n) / FPS).astype(np.float32), type=pa.float32()),
        "frame_index": pa.array(np.arange(n), type=pa.int64()),
        "episode_index": pa.array(np.full(n, episode_index), type=pa.int64()),
        "index": pa.array(np.arange(index_offset, index_offset + n), type=pa.int64()),
        "task_index": pa.array(np.full(n, job["task_index"]), type=pa.int64()),
    }).replace_schema_metadata({"huggingface": HF_SCHEMA_METADATA})
    pq_dir = out / "data" / f"chunk-{chunk:03d}"
    pq_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, pq_dir / f"episode_{episode_index:06d}.parquet")

    for key, frames in ((FRONT_KEY, front), (WRIST_KEY, wrist)):
        path = out / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(path, list(frames), fps=FPS, codec=job["codec"],
                         pixelformat="yuv420p", macro_block_size=1)

    return {
        "episode_index": episode_index,
        "state": state,
        "action": action,
        "raw_gripper_unique": np.unique(raw_gripper).tolist(),
        "stats": {
            WRIST_KEY: image_stats(wrist),
            FRONT_KEY: image_stats(front),
            "observation.state": vec_stats(state),
            "action": vec_stats(action),
            "timestamp": vec_stats((np.arange(n) / FPS)[:, None]),
            "frame_index": vec_stats(np.arange(n)[:, None]),
            "episode_index": vec_stats(np.full((n, 1), episode_index)),
            "index": vec_stats(np.arange(index_offset, index_offset + n)[:, None]),
            "task_index": vec_stats(np.full((n, 1), job["task_index"])),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dir with <TASK>_demo.hdf5 files")
    ap.add_argument("--out", required=True, help="output LeRobot dataset dir")
    ap.add_argument("--val-per-task", type=int, default=20,
                    help="last N demos per task (by demo id) held out as val")
    ap.add_argument("--max-episodes-per-task", type=int, default=0,
                    help="cap demos per task (0 = all; small values for smokes)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--codec", default="libx264",
                    help="reference corpora use av1; libx264 decodes identically via torchvision_av")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    h5_files = sorted(src.glob("*_demo.hdf5"), key=task_number)
    if not h5_files:
        raise SystemExit(f"no *_demo.hdf5 under {src}")

    # Plan pass: task registry, per-episode lengths (metadata-only reads),
    # train/val assignment, deterministic ordering train-block then val-block.
    tasks: dict[str, int] = {}
    train_plan, val_plan = [], []
    for h5_path in h5_files:
        language = task_language(h5_path)
        task_index = tasks.setdefault(language, len(tasks))
        with h5py.File(h5_path, "r") as f:
            demos = natural_sorted_demos(f["data"].keys())
            if args.max_episodes_per_task:
                demos = demos[: args.max_episodes_per_task]
            lengths = {d: int(f["data"][d]["actions"].shape[0]) for d in demos}
        n_val = min(args.val_per_task, max(len(demos) - 1, 0))
        split_at = len(demos) - n_val
        for i, demo in enumerate(demos):
            record = {
                "h5": str(h5_path), "demo": demo, "task_index": task_index,
                "language": language, "length": lengths[demo],
                "split": "train" if i < split_at else "val",
            }
            (train_plan if i < split_at else val_plan).append(record)
        print(f"[plan] {h5_path.name}: {len(demos)} demos "
              f"({split_at} train / {len(demos) - split_at} val) task={language!r}")

    plan = train_plan + val_plan
    index_offset = 0
    for episode_index, record in enumerate(plan):
        record.update(episode_index=episode_index, index_offset=index_offset,
                      out=str(out), codec=args.codec)
        index_offset += record["length"]
    total_frames = index_offset
    n_train = len(train_plan)
    print(f"[plan] {len(plan)} episodes / {total_frames} frames / {len(tasks)} tasks "
          f"(train 0:{n_train}, val {n_train}:{len(plan)})")

    results: list[dict | None] = [None] * len(plan)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, result in enumerate(pool.map(convert_episode, plan, chunksize=1), 1):
            results[result["episode_index"]] = result
            if done % 25 == 0 or done == len(plan):
                print(f"[convert] {done}/{len(plan)} episodes", flush=True)

    gripper_values = sorted({v for r in results for v in r["raw_gripper_unique"]})
    print(f"[convert] raw gripper action values across dataset: {gripper_values}")
    if not set(gripper_values) <= {-1.0, 1.0}:
        print("[convert] WARNING: gripper values outside {-1,1}; (1-g)/2 mapping "
              "produced non-binary outputs — inspect before training")

    episodes_meta = [{
        "episode_index": r["episode_index"], "tasks": [p["language"]],
        "length": p["length"], "split": p["split"],
        "source_h5": Path(p["h5"]).name, "source_demo": p["demo"],
    } for p, r in zip(plan, results)]
    episodes_stats = [{"episode_index": r["episode_index"], "stats": r["stats"]}
                      for r in results]

    def agg(arrs_list):
        a = np.concatenate(arrs_list).astype(np.float64)
        a = a.reshape(len(a), -1)
        return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                "min": a.min(0).tolist(), "max": a.max(0).tolist(),
                "q01": np.quantile(a, 0.01, axis=0).tolist(),
                "q99": np.quantile(a, 0.99, axis=0).tolist()}

    lengths = [p["length"] for p in plan]
    ep_idx_col = np.concatenate([np.full(n, p["episode_index"]) for n, p in zip(lengths, plan)])
    task_idx_col = np.concatenate([np.full(n, p["task_index"]) for n, p in zip(lengths, plan)])
    ts_col = np.concatenate([np.arange(n) / FPS for n in lengths])
    fr_col = np.concatenate([np.arange(n) for n in lengths])
    stats_gr00t = {
        "observation.state": agg([r["state"] for r in results]),
        "action": agg([r["action"] for r in results]),
        "timestamp": agg([ts_col[:, None]]), "frame_index": agg([fr_col[:, None]]),
        "episode_index": agg([ep_idx_col[:, None]]),
        "index": agg([np.arange(total_frames)[:, None]]),
        "task_index": agg([task_idx_col[:, None]]),
    }

    video_info = {
        "dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "rgb"],
        "info": {"video.height": 256, "video.width": 256, "video.codec": args.codec,
                 "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                 "video.fps": FPS, "video.channels": 3, "has_audio": False},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": len(plan),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": 2 * len(plan),
        "total_chunks": (len(plan) + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{n_train}", "val": f"{n_train}:{len(plan)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            WRIST_KEY: video_info, FRONT_KEY: video_info,
            "observation.state": {
                "dtype": "float32", "shape": [8],
                "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2",
                                     "axis_angle3", "gripper", "gripper"]},
            },
            "action": {
                "dtype": "float32", "shape": [7],
                "names": {"motors": ["x", "y", "z", "axis_angle1", "axis_angle2",
                                     "axis_angle3", "gripper"]},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        # Provenance (ignored by loaders, load-bearing for humans):
        "source": "hf.co/datasets/libero-mem/LIBERO-Mem (AAAI 2026, arXiv 2511.11478)",
        "conversion": {
            "script": "scripts/data/libero_mem_to_lerobot.py",
            "gripper_mapping": "action[6] = (1 - raw)/2 (robosuite -1=open -> 1=open, matches libero_*_no_noops corpora)",
            "language": "grab_language_from_filename incl. leading task number (matches eval-time instruction)",
            "noop_filtering": "none (loader delete_pause_frame handles pauses; frames kept for memory structure)",
            "split_rule": f"last {args.val_per_task} demos per task by demo id -> val "
                          "(adapted from 'first-100 train / last-20 val'; released set has 82-100 demos/task)",
            "stats_scope": "all episodes (train+val), matching reference corpora",
        },
    }

    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (out / "meta" / "modality.json").write_text(json.dumps(MODALITY, indent=4))
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        f.writelines(json.dumps({"task_index": i, "task": t}) + "\n"
                     for t, i in sorted(tasks.items(), key=lambda kv: kv[1]))
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in episodes_meta)
    with open(out / "meta" / "episodes_stats.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in episodes_stats)
    (out / "meta" / "stats_gr00t.json").write_text(json.dumps(stats_gr00t, indent=4))
    print(f"[convert] DONE: {len(plan)} episodes / {total_frames} frames / "
          f"{len(tasks)} tasks -> {out}")


if __name__ == "__main__":
    main()
