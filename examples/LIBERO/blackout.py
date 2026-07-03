"""Pure client-side helpers for the memory-experiment LIBERO evals (H6).

Blackout scheduling / frame corruption plus episodes.jsonl / decisions.jsonl
record construction. Deliberately simulator-free so unit tests run without
LIBERO installed.
"""

import os
import subprocess

import numpy as np

BLACKOUT_FILLS = ("black", "freeze")
BLACKOUT_VIEWS = ("agentview", "both")

# Server-side memory knobs mirrored into every episode record so each JSONL
# line is self-describing (values kept as raw env strings).
MEMORY_PARAM_ENV_KEYS = (
    "MEMORY_RESET_K",
    "MEMORY_FREEZE_K",
    "MEMORY_WRITE_EVERY",
    "MEMORY_GATE_SCALE",
    "MEMORY_DONOR_DIR",
    "MEMORY_STATE_DUMP_DIR",
    "MEMORY_COUNTERFACTUAL",
)

EPISODE_RECORD_KEYS = (
    "suite",
    "task_id",
    "task_description",
    "episode_idx",
    "memory_mode",
    "memory_params",
    "episode_seed",
    "success",
    "num_env_steps",
    "num_decisions",
    "ckpt",
    "git_sha",
)


def in_blackout(decision_idx: int, start_decision: int, num_decisions: int) -> bool:
    """True when decision_idx lies in [start_decision, start_decision + num_decisions)."""
    if start_decision < 0 or num_decisions <= 0:
        return False
    return start_decision <= decision_idx < start_decision + num_decisions


def corrupt_frame(frame: np.ndarray, fill: str, last_clean: np.ndarray | None) -> np.ndarray:
    """Blackout replacement for one frame.

    ``black`` zeroes the frame; ``freeze`` repeats the last pre-blackout frame,
    falling back to black when the blackout starts at decision 0.
    """
    if fill not in BLACKOUT_FILLS:
        raise ValueError(f"unsupported blackout fill: {fill}")
    if fill == "freeze" and last_clean is not None:
        return last_clean.copy()
    return np.zeros_like(frame)


def corrupt_views(
    img: np.ndarray,
    wrist_img: np.ndarray,
    fill: str,
    views: str,
    last_clean_img: np.ndarray | None,
    last_clean_wrist: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply blackout corruption to the selected camera views."""
    if views not in BLACKOUT_VIEWS:
        raise ValueError(f"unsupported blackout views: {views}")
    img = corrupt_frame(img, fill, last_clean_img)
    if views == "both":
        wrist_img = corrupt_frame(wrist_img, fill, last_clean_wrist)
    return img, wrist_img


def memory_params_from_env(env=None) -> dict:
    env = os.environ if env is None else env
    return {key: env[key] for key in MEMORY_PARAM_ENV_KEYS if key in env}


def episode_record(
    *,
    suite,
    task_id,
    task_description,
    episode_idx,
    memory_mode,
    memory_params,
    episode_seed,
    success,
    num_env_steps,
    num_decisions,
    ckpt,
    git_sha,
    extras: dict | None = None,
) -> dict:
    """One episodes.jsonl line (JSON-native types only)."""
    record = {
        "suite": str(suite),
        "task_id": int(task_id),
        "task_description": str(task_description),
        "episode_idx": int(episode_idx),
        "memory_mode": str(memory_mode),
        "memory_params": dict(memory_params),
        "episode_seed": int(episode_seed),
        "success": bool(success),
        "num_env_steps": int(num_env_steps),
        "num_decisions": int(num_decisions),
        "ckpt": str(ckpt),
        "git_sha": git_sha,
    }
    if extras:
        record.update(extras)
    return record


def decision_record(
    *,
    episode_idx,
    d,
    memory_extras: dict | None = None,
    blackout_active: bool = False,
    extras: dict | None = None,
) -> dict:
    """One decisions.jsonl line: {episode_idx, d, ...memory_extras, blackout_active}."""
    record = {"episode_idx": int(episode_idx), "d": int(d)}
    if memory_extras:
        record.update(memory_extras)
    record["blackout_active"] = bool(blackout_active)
    if extras:
        record.update(extras)
    return record


def read_git_sha(repo_dir) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None
