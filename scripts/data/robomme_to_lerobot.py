"""RoboMME HDF5 demos -> LeRobot v2.1 dataset (GR00T layout our trainer loads).

Input: `record_dataset_<Task>.h5` files from hf.co/datasets/Yinpei/robomme_data_h5
(100 scripted demos per task, 16 tasks; format per robomme doc/h5_data_format.md:
episode_<i>/setup + episode_<i>/timestep_<k>/{obs,action,info}).

Output: one dataset dir mirroring our staged LIBERO conversions
(vlajepa_stage/datasets/lerobot/libero_*_lerobot), i.e.:
  data/chunk-XXX/episode_XXXXXX.parquet   observation.state f32[8], action f32[7],
                                          timestamp f32, frame/episode/index/task_index i64
  videos/chunk-XXX/observation.images.image/episode_XXXXXX.mp4        (front_rgb)
  videos/chunk-XXX/observation.images.wrist_image/episode_XXXXXX.mp4  (wrist_rgb)
  meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl,
        stats_gr00t.json,modality.json}

Mapping decisions (all overridable):
  state  = [eef_state(6: xyz + extrinsic-XYZ rpy, unwrapped), gripper_state(2)]
           (--state-rot axis_angle converts rpy->axis-angle to match the LIBERO
           franka convention; default rpy — a robomme-specific unnorm_key gets
           fresh stats either way)
  action = eef_action(7): ABSOLUTE world-frame [x,y,z,r,p,y,gripper] with
           gripper -1=close/+1=open (RoboMME convention, opposite of LIBERO)
  conditioning-video frames (info/is_video_demo=True) are KEPT by default —
  they are the memory cue; --drop-video-demo removes them.
  fps: RoboMME h5 has no timestamps; default --fps 20 (our LIBERO datasets'
  rate) is an assumption, documented in info.json via the flag you pass.

Runs in the VLA_JEPA conda env (h5py + pyarrow + imageio-ffmpeg present):
  /lustre/.../envs/VLA_JEPA/bin/python scripts/data/robomme_to_lerobot.py \
      --h5 /abs/record_dataset_StopCube.h5 --out /abs/robomme_stopcube_lerobot \
      --max-episodes 2   # unit smoke
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np

FRONT_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist_image"
CHUNK_SIZE = 1000

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


def natural_sorted(keys, prefix):
    pat = re.compile(rf"^{prefix}_(\d+)$")
    pairs = [(int(m.group(1)), k) for k in keys if (m := pat.match(k))]
    return [k for _, k in sorted(pairs)]


def rpy_to_axis_angle(rpy: np.ndarray) -> np.ndarray:
    """Extrinsic-XYZ rpy -> axis-angle 3-vector (pure numpy, no scipy)."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry_ = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rot = rz @ ry_ @ rx  # extrinsic X, then Y, then Z
    angle = np.arccos(np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    axis = axis / (2 * np.sin(angle))
    return (axis * angle).astype(np.float32)


def episode_arrays(ep_group, state_rot: str, drop_video_demo: bool):
    """One h5 episode -> dict of stacked arrays (+ per-frame front/wrist rgb)."""
    steps = natural_sorted(ep_group.keys(), "timestep")
    if not steps:
        raise ValueError("episode has no timesteps")
    states, actions, fronts, wrists, is_demo = [], [], [], [], []
    for name in steps:
        ts = ep_group[name]
        demo = bool(ts["info"]["is_video_demo"][()]) if "is_video_demo" in ts["info"] else False
        if drop_video_demo and demo:
            continue
        eef = np.asarray(ts["obs"]["eef_state"], dtype=np.float32).reshape(-1)[:6]
        grip = np.asarray(ts["obs"]["gripper_state"], dtype=np.float32).reshape(-1)[:2]
        rot = rpy_to_axis_angle(eef[3:6]) if state_rot == "axis_angle" else eef[3:6]
        states.append(np.concatenate([eef[:3], rot, grip]).astype(np.float32))
        actions.append(np.asarray(ts["action"]["eef_action"], dtype=np.float32).reshape(-1)[:7])
        fronts.append(np.asarray(ts["obs"]["front_rgb"], dtype=np.uint8))
        wrists.append(np.asarray(ts["obs"]["wrist_rgb"], dtype=np.uint8))
        is_demo.append(demo)
    return {
        "state": np.stack(states), "action": np.stack(actions),
        "front": np.stack(fronts), "wrist": np.stack(wrists),
        "n_video_demo": int(sum(is_demo)),
    }


def decode_task_goal(setup) -> str:
    goals = setup["task_goal"]
    raw = goals[0] if getattr(goals, "shape", ()) not in ((), None) else goals[()]
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def vec_stats(arr: np.ndarray) -> dict:
    a = arr.reshape(len(arr), -1).astype(np.float64)
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "count": [len(a)],
    }


def image_stats(frames: np.ndarray, sample: int = 100) -> dict:
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
    a = frames[idx].astype(np.float64) / 255.0  # (N,H,W,3) -> per-channel
    per_c = a.reshape(-1, a.shape[-1])
    fmt = lambda v: [[[float(x)]] for x in v]  # noqa: E731 - (3,1,1) nesting like reference
    return {
        "min": fmt(per_c.min(0)), "max": fmt(per_c.max(0)),
        "mean": fmt(per_c.mean(0)), "std": fmt(per_c.std(0)),
        "count": [len(idx)],
    }


def write_parquet(path: Path, state, action, episode_index, index_offset, fps):
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(state)
    table = pa.table({
        "observation.state": pa.FixedSizeListArray.from_arrays(
            pa.array(state.reshape(-1), type=pa.float32()), 8),
        "action": pa.FixedSizeListArray.from_arrays(
            pa.array(action.reshape(-1), type=pa.float32()), 7),
        "timestamp": pa.array((np.arange(n) / fps).astype(np.float32), type=pa.float32()),
        "frame_index": pa.array(np.arange(n), type=pa.int64()),
        "episode_index": pa.array(np.full(n, episode_index), type=pa.int64()),
        "index": pa.array(np.arange(index_offset, index_offset + n), type=pa.int64()),
        "task_index": pa.array(np.full(n, 0), type=pa.int64()),  # patched by caller
    })
    return table, pq, path


def write_video(path: Path, frames: np.ndarray, fps: int, codec: str):
    import imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, list(frames), fps=fps, codec=codec,
                     pixelformat="yuv420p", macro_block_size=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", nargs="+", required=True,
                    help="record_dataset_<Task>.h5 files (converted in order)")
    ap.add_argument("--out", required=True, help="output LeRobot dataset dir")
    ap.add_argument("--fps", type=int, default=20,
                    help="assumed control rate (h5 carries no timestamps)")
    ap.add_argument("--state-rot", choices=["rpy", "axis_angle"], default="rpy")
    ap.add_argument("--drop-video-demo", action="store_true",
                    help="drop conditioning-video frames (default: keep — they are the memory cue)")
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="cap episodes per h5 file (0 = all; use small values for smokes)")
    ap.add_argument("--codec", default="libx264",
                    help="video codec (reference datasets use av1; libx264 decodes everywhere)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    tasks: dict[str, int] = {}
    episodes_meta, episodes_stats = [], []
    all_state, all_action = [], []
    episode_index, global_index, total_videos = 0, 0, 0

    for h5_path in args.h5:
        with h5py.File(h5_path, "r") as f:
            ep_names = natural_sorted(f.keys(), "episode")
            if args.max_episodes:
                ep_names = ep_names[: args.max_episodes]
            print(f"[convert] {h5_path}: {len(ep_names)} episodes")
            for ep_name in ep_names:
                ep = f[ep_name]
                goal = decode_task_goal(ep["setup"])
                task_index = tasks.setdefault(goal, len(tasks))
                arrs = episode_arrays(ep, args.state_rot, args.drop_video_demo)
                n = len(arrs["state"])
                chunk = episode_index // CHUNK_SIZE

                table, _, _ = write_parquet(
                    out, arrs["state"], arrs["action"], episode_index, global_index, args.fps)
                table = table.set_column(
                    table.schema.get_field_index("task_index"), "task_index",
                    pa.array(np.full(n, task_index), type=pa.int64()))
                table = table.replace_schema_metadata({"huggingface": HF_SCHEMA_METADATA})
                pq_dir = out / "data" / f"chunk-{chunk:03d}"
                pq_dir.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, pq_dir / f"episode_{episode_index:06d}.parquet")

                for key, frames in ((FRONT_KEY, arrs["front"]), (WRIST_KEY, arrs["wrist"])):
                    write_video(
                        out / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4",
                        frames, args.fps, args.codec)
                    total_videos += 1

                episodes_meta.append({
                    "episode_index": episode_index, "tasks": [goal], "length": n,
                    "source_h5": Path(h5_path).name, "source_episode": ep_name,
                    "video_demo_frames": arrs["n_video_demo"],
                })
                episodes_stats.append({"episode_index": episode_index, "stats": {
                    WRIST_KEY: image_stats(arrs["wrist"]),
                    FRONT_KEY: image_stats(arrs["front"]),
                    "observation.state": vec_stats(arrs["state"]),
                    "action": vec_stats(arrs["action"]),
                    "timestamp": vec_stats((np.arange(n) / args.fps)[:, None]),
                    "frame_index": vec_stats(np.arange(n)[:, None]),
                    "episode_index": vec_stats(np.full((n, 1), episode_index)),
                    "index": vec_stats(np.arange(global_index, global_index + n)[:, None]),
                    "task_index": vec_stats(np.full((n, 1), task_index)),
                }})
                all_state.append(arrs["state"])
                all_action.append(arrs["action"])
                global_index += n
                episode_index += 1
                print(f"[convert]   {ep_name}: {n} frames "
                      f"(video_demo={arrs['n_video_demo']}) goal={goal!r}")

    total_frames = global_index
    total_chunks = (episode_index + CHUNK_SIZE - 1) // CHUNK_SIZE

    def agg(arrs_list):
        a = np.concatenate(arrs_list).astype(np.float64)
        a = a.reshape(len(a), -1)
        return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                "min": a.min(0).tolist(), "max": a.max(0).tolist(),
                "q01": np.quantile(a, 0.01, axis=0).tolist(),
                "q99": np.quantile(a, 0.99, axis=0).tolist()}

    lengths = [m["length"] for m in episodes_meta]
    ep_idx_col = np.concatenate([np.full(l, m["episode_index"]) for l, m in zip(lengths, episodes_meta)])
    task_idx_col = np.concatenate([np.full(l, tasks[m["tasks"][0]]) for l, m in zip(lengths, episodes_meta)])
    ts_col = np.concatenate([np.arange(l) / args.fps for l in lengths])
    fr_col = np.concatenate([np.arange(l) for l in lengths])
    stats_gr00t = {
        "observation.state": agg(all_state), "action": agg(all_action),
        "timestamp": agg([ts_col[:, None]]), "frame_index": agg([fr_col[:, None]]),
        "episode_index": agg([ep_idx_col[:, None]]),
        "index": agg([np.arange(total_frames)[:, None]]),
        "task_index": agg([task_idx_col[:, None]]),
    }

    video_info = {
        "dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "rgb"],
        "info": {"video.height": 256, "video.width": 256, "video.codec": args.codec,
                 "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                 "video.fps": args.fps, "video.channels": 3, "has_audio": False},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": episode_index,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": args.fps,
        "splits": {"train": f"0:{episode_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            WRIST_KEY: video_info, FRONT_KEY: video_info,
            "observation.state": {
                "dtype": "float32", "shape": [8],
                "names": {"motors": ["x", "y", "z",
                                     *(["axis_angle1", "axis_angle2", "axis_angle3"]
                                       if args.state_rot == "axis_angle" else ["roll", "pitch", "yaw"]),
                                     "gripper", "gripper"]},
            },
            "action": {
                "dtype": "float32", "shape": [7],
                "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        # Provenance (ignored by loaders, load-bearing for humans):
        "source": "hf.co/datasets/Yinpei/robomme_data_h5 (RoboMME, arXiv 2603.04639)",
        "conversion": {
            "script": "scripts/data/robomme_to_lerobot.py",
            "action_source": "eef_action (ABSOLUTE world-frame; gripper -1=close/+1=open)",
            "state_rot": args.state_rot,
            "drop_video_demo": args.drop_video_demo,
            "fps_assumed": True,
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
    print(f"[convert] DONE: {episode_index} episodes / {total_frames} frames / "
          f"{len(tasks)} tasks -> {out}")


if __name__ == "__main__":
    main()
