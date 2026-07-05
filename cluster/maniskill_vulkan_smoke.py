#!/usr/bin/env python
"""Track B Vulkan/SAPIEN headless gate for MIKASA-Robo-VLA + RoboMME (both ManiSkill3).

Creates PickCube-v1 (obs_mode='rgb', no human render camera), steps 10x with
random actions, saves one sensor-camera RGB frame to PNG, and asserts the frame
is not (near-)black. Exit code 0 == PASS for the *current* env-var configuration;
the sbatch driver tries several Vulkan ICD configurations and records which one
passed.

Run: python maniskill_vulkan_smoke.py --out /path/frame.png
"""
import argparse
import os
import sys
import traceback


def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def dump_env() -> None:
    for k in (
        "VK_ICD_FILENAMES",
        "VK_DRIVER_FILES",
        "DISPLAY",
        "SAPIEN_HEADLESS",
        "MS_ASSET_DIR",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
    ):
        log(f"env {k}={os.environ.get(k)!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--sim-backend", default="auto")
    args = ap.parse_args()

    dump_env()

    import numpy as np

    try:
        import sapien

        log(f"sapien {sapien.__version__}")
    except Exception:
        traceback.print_exc()
        return 2

    try:
        import gymnasium as gym
        import mani_skill  # noqa: F401  (registers envs)
        import mani_skill.envs  # noqa: F401

        log(f"mani_skill {mani_skill.__version__}")
    except Exception:
        traceback.print_exc()
        return 2

    try:
        env = gym.make(
            args.env_id,
            obs_mode="rgb",
            render_mode=None,  # human render camera OFF
            sim_backend=args.sim_backend,
            num_envs=1,
        )
        obs, _ = env.reset(seed=0)
        frame = None
        for i in range(args.steps):
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
        # obs['sensor_data'][<cam>]['rgb'] : (num_envs, H, W, 3) uint8 tensor
        cams = list(obs["sensor_data"].keys())
        log(f"sensor cameras: {cams}")
        rgb = obs["sensor_data"][cams[0]]["rgb"]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        frame = np.asarray(rgb)[0].astype(np.uint8)
        env.close()
    except Exception:
        traceback.print_exc()
        return 3

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(frame).save(args.out)
    except Exception:
        # PIL missing or save failed; fall back to raw npy so evidence survives
        np.save(args.out + ".npy", frame)
        traceback.print_exc()

    fmax, fmean = int(frame.max()), float(frame.mean())
    nonzero_frac = float((frame > 10).mean())
    log(f"frame shape={frame.shape} max={fmax} mean={fmean:.2f} frac(>10)={nonzero_frac:.4f}")
    if fmax < 20 or nonzero_frac < 0.01:
        log("FAIL: frame is (near-)black")
        return 4
    log(f"PASS: non-black frame written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
