"""RoboMME headless env-creation smoke (Track D, one task per suite category).

Runs inside the robomme uv venv (memexp_stage/robomme/.venv). For each task it
builds the benchmark env for a fixed test episode, resets (this generates the
conditioning/demo batch), executes a few no-op-ish joint actions, and checks
the observation contract we care about:
  - front_rgb / wrist_rgb 256x256x3 uint8 (matches our 2-view obs spec)
  - eef_state 6-D [x,y,z,r,p,y], joint_state 7-D, gripper_state 2-D
  - info: task_goal list[str], status in {ongoing,success,fail,timeout,error}

Writes a first-frame PNG per task plus a PASS/FAIL line per task to stdout;
exits non-zero if any task fails. Default battery = one task per suite:
  Counting=StopCube, Permanence=ButtonUnmask, Reference=PickHighlight,
  Imitation=MoveCube.

Usage (from the wt-memexp worktree, via sbatch_vj_bench_robomme_smoke.sh):
  <robomme-venv-python> cluster/robomme_headless_smoke.py --out-dir /abs/dir
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

DEFAULT_TASKS = ["StopCube", "ButtonUnmask", "PickHighlight", "MoveCube"]
SUITE_OF = {
    "BinFill": "Counting", "PickXtimes": "Counting", "SwingXtimes": "Counting",
    "StopCube": "Counting",
    "VideoUnmask": "Permanence", "VideoUnmaskSwap": "Permanence",
    "ButtonUnmask": "Permanence", "ButtonUnmaskSwap": "Permanence",
    "PickHighlight": "Reference", "VideoRepick": "Reference",
    "VideoPlaceButton": "Reference", "VideoPlaceOrder": "Reference",
    "MoveCube": "Imitation", "InsertPeg": "Imitation",
    "PatternLock": "Imitation", "RouteStick": "Imitation",
}
# Neutral joint_angle action (their DummyModel base action + gripper open).
NEUTRAL_JOINT_ACTION = np.array(
    [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0], dtype=np.float32
)


def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def check_task(task_id: str, dataset: str, episode_idx: int, num_steps: int,
               out_dir: Path) -> dict:
    import robomme.robomme_env  # noqa: F401  (registers env ids with ManiSkill)
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id=task_id, dataset=dataset, action_space="joint_angle",
        gui_render=False, max_steps=num_steps + 5,
    )
    n_eps = builder.get_episode_num()
    env = builder.make_env_for_episode(episode_idx)
    try:
        obs, info = env.reset()
        front0 = to_numpy(obs["front_rgb_list"][-1])
        wrist0 = to_numpy(obs["wrist_rgb_list"][-1])
        eef = to_numpy(obs["eef_state_list"][-1]).reshape(-1)
        joints = to_numpy(obs["joint_state_list"][-1]).reshape(-1)
        grip = to_numpy(obs["gripper_state_list"][-1]).reshape(-1)
        task_goal = info["task_goal"][0]
        n_conditioning = len(obs["front_rgb_list"]) - 1

        assert front0.shape == (256, 256, 3), f"front_rgb {front0.shape}"
        assert wrist0.shape == (256, 256, 3), f"wrist_rgb {wrist0.shape}"
        assert eef.shape == (6,), f"eef_state {eef.shape}"
        assert joints.shape == (7,), f"joint_state {joints.shape}"
        assert grip.shape == (2,), f"gripper_state {grip.shape}"
        assert isinstance(task_goal, str) and task_goal, "empty task_goal"
        # Non-degenerate render (same criterion as the Track B Vulkan smoke).
        frac_lit = float((front0.astype(np.float32).mean(-1) > 10).mean())
        assert frac_lit > 0.5, f"front frame looks black (lit={frac_lit:.3f})"

        status = "ongoing"
        for _ in range(num_steps):
            obs, _, terminated, truncated, info = env.step(NEUTRAL_JOINT_ACTION)
            status = info.get("status", "unknown")
            if status == "error":
                raise RuntimeError(f"step error: {info.get('error_message')}")
            if to_numpy(terminated).any() or to_numpy(truncated).any():
                break

        import imageio
        frame = np.hstack([front0, wrist0]).astype(np.uint8)
        png = out_dir / f"frame_{task_id}.png"
        imageio.imwrite(png, frame)
        return {
            "task": task_id, "suite": SUITE_OF.get(task_id, "?"), "ok": True,
            "episodes_in_split": n_eps, "task_goal": task_goal,
            "conditioning_frames": n_conditioning, "front_lit_frac": frac_lit,
            "status_after_steps": status, "png": str(png),
        }
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--dataset", default="test", choices=["train", "val", "test"])
    ap.add_argument("--episode-idx", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for task_id in args.tasks:
        try:
            result = check_task(task_id, args.dataset, args.episode_idx,
                                args.num_steps, out_dir)
            print(f"[SMOKE PASS] {result}", flush=True)
        except Exception as exc:  # noqa: BLE001 - smoke must report and continue
            failures += 1
            traceback.print_exc()
            print(f"[SMOKE FAIL] task={task_id}: {exc}", flush=True)
    print(f"[SMOKE {'PASS' if failures == 0 else 'FAIL'}] "
          f"{len(args.tasks) - failures}/{len(args.tasks)} tasks ok", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
