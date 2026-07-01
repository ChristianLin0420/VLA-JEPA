# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Runtime utilities for robust, requeue-friendly training:

* ``RichWandbLogger``  -- rank-0 W&B online logging with namespaced panels, config
  capture, scalar + figure + media logging, and resume support (stable run id).
* ``GracefulStopper``  -- SIGUSR1/SIGTERM handlers PLUS a wall-clock deadline so each
  rank self-stops a safe margin before the SLURM time limit. The stop decision is
  all-reduced across ranks so every rank checkpoints at the same step (required for
  collective DeepSpeed save_state).
* checkpoint helpers   -- ``save_full_state`` / ``find_latest_checkpoint`` use
  ``accelerator.save_state``/``load_state`` (model + optimizer + scheduler + RNG +
  DeepSpeed ZeRO shards on ALL ranks) and maintain a ``latest`` pointer + a JSON
  sidecar with ``completed_steps`` and W&B run id.

These are intentionally framework-agnostic helpers shared by the VLA-JEPA trainers.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist


# ============================================================================ W&B
def sanitize_run_id(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "-", str(s))[:120]


class RichWandbLogger:
    """Thin, rank-0-only wrapper around W&B with rich panel organization.

    Panels are grouped by the ``foo/`` prefix of each metric key (W&B convention):
      loss/*  jepa/*  jepa_view/*  opt/*  time/*  media/*  eval/*
    """

    def __init__(
        self,
        cfg,
        accelerator,
        *,
        project: str,
        entity: Optional[str],
        run_name: str,
        run_id: str,
        mode: str = "online",
        config_dict: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
    ):
        self.enabled = accelerator.is_main_process and mode != "disabled"
        self.run = None
        self._wandb = None
        if not self.enabled:
            return
        import wandb

        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            id=sanitize_run_id(run_id),
            resume="allow",
            mode=mode,
            dir=os.path.join(cfg.output_dir, "wandb"),
            config=config_dict or {},
            tags=tags or [],
            group=group,
            settings=wandb.Settings(start_method="thread"),
        )
        # x-axis = global step; let every metric share it.
        wandb.define_metric("train/step")
        wandb.define_metric("*", step_metric="train/step")

    def log(self, data: Dict[str, float], step: int):
        if not self.enabled or not data:
            return
        payload = {"train/step": int(step)}
        payload.update({k: v for k, v in data.items() if v is not None})
        self._wandb.log(payload, step=int(step))

    def log_figures(self, figures: Dict[str, "object"], step: int, prefix: str = "media/"):
        if not self.enabled or not figures:
            return
        import matplotlib.pyplot as plt

        payload = {"train/step": int(step)}
        for name, fig in figures.items():
            try:
                payload[f"{prefix}{name}"] = self._wandb.Image(fig)
            except Exception:
                pass
            finally:
                try:
                    plt.close(fig)
                except Exception:
                    pass
        self._wandb.log(payload, step=int(step))

    def log_images(self, images: Dict[str, "object"], step: int, prefix: str = "media/"):
        """images: {name: (PIL.Image|np.ndarray, caption)} or {name: PIL.Image}."""
        if not self.enabled or not images:
            return
        payload = {"train/step": int(step)}
        for name, val in images.items():
            try:
                if isinstance(val, tuple):
                    img, cap = val
                    payload[f"{prefix}{name}"] = self._wandb.Image(img, caption=str(cap))
                else:
                    payload[f"{prefix}{name}"] = self._wandb.Image(val)
            except Exception:
                pass
        self._wandb.log(payload, step=int(step))

    def summary(self, **kwargs):
        if self.enabled and self.run is not None:
            for k, v in kwargs.items():
                self.run.summary[k] = v

    def finish(self):
        if self.enabled and self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


# ============================================================ graceful stop / deadline
class GracefulStopper:
    """Decide when to stop early so we can checkpoint before SLURM kills the job.

    Primary signal: a wall-clock deadline shared by all ranks via the
    ``TRAIN_DEADLINE_EPOCH`` env var (absolute unix time set by the sbatch script).
    Secondary: SIGUSR1 / SIGTERM handlers set a flag. The decision is all-reduced
    (max) across ranks so every rank stops on the same iteration.
    """

    def __init__(self, grace_seconds: int = 300):
        self._flagged = False
        self.grace_seconds = grace_seconds
        self.reason = None
        self.start_time = time.time()

        deadline = os.environ.get("TRAIN_DEADLINE_EPOCH")
        budget = os.environ.get("TRAIN_TIME_BUDGET_SECONDS")
        if deadline:
            self.deadline = float(deadline)
        elif budget:
            self.deadline = self.start_time + float(budget)
        else:
            self.deadline = None

        for sig in (signal.SIGUSR1, signal.SIGTERM):
            try:
                signal.signal(sig, self._handler)
            except Exception:
                pass

    def _handler(self, signum, frame):
        self._flagged = True
        self.reason = f"signal {signum}"

    def _local_should_stop(self) -> bool:
        if self._flagged:
            return True
        if self.deadline is not None and time.time() >= (self.deadline - self.grace_seconds):
            if self.reason is None:
                self.reason = "wall-clock deadline"
            return True
        return False

    def should_stop(self, accelerator) -> bool:
        local = 1.0 if self._local_should_stop() else 0.0
        if dist.is_available() and dist.is_initialized() and accelerator.num_processes > 1:
            t = torch.tensor([local], device=accelerator.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            agreed = bool(t.item() > 0.5)
            if agreed and self.reason is None:
                self.reason = "peer rank requested stop"
            return agreed
        return bool(local > 0.5)

    def seconds_left(self) -> Optional[float]:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())


# ============================================================ checkpoint save / load
STATE_SIDECAR = "training_state.json"
LATEST_POINTER = "latest.txt"


def _step_dirs(ckpt_root: str) -> List[str]:
    if not os.path.isdir(ckpt_root):
        return []
    dirs = []
    for d in os.listdir(ckpt_root):
        if re.fullmatch(r"step_\d+", d) and os.path.isdir(os.path.join(ckpt_root, d)):
            dirs.append(d)
    return sorted(dirs, key=lambda x: int(x.split("_")[1]))


def save_full_state(
    accelerator,
    ckpt_root: str,
    completed_steps: int,
    extra: Optional[dict] = None,
    keep_last: int = 2,
    tag: Optional[str] = None,
) -> str:
    """Collective full-state save (model+optimizer+scheduler+RNG+ZeRO shards).

    MUST be called on ALL ranks (DeepSpeed save is collective). Rank-0 then writes
    the JSON sidecar + ``latest.txt`` pointer and prunes old step dirs.
    """
    os.makedirs(ckpt_root, exist_ok=True)
    name = tag or f"step_{completed_steps}"
    ckpt_dir = os.path.join(ckpt_root, name)
    accelerator.save_state(ckpt_dir)  # collective

    if accelerator.is_main_process:
        state = {"completed_steps": int(completed_steps)}
        if extra:
            state.update(extra)
        with open(os.path.join(ckpt_dir, STATE_SIDECAR), "w") as f:
            json.dump(state, f, indent=2)
        with open(os.path.join(ckpt_root, LATEST_POINTER), "w") as f:
            f.write(name)
        # prune old step_* dirs (never touch the one we just wrote)
        keep = set(_step_dirs(ckpt_root)[-keep_last:]) | {name}
        for d in _step_dirs(ckpt_root):
            if d not in keep:
                shutil.rmtree(os.path.join(ckpt_root, d), ignore_errors=True)
    accelerator.wait_for_everyone()
    return ckpt_dir


def find_latest_checkpoint(ckpt_root: str):
    """Return (ckpt_dir, state_dict) for the latest checkpoint, or (None, None)."""
    pointer = os.path.join(ckpt_root, LATEST_POINTER)
    name = None
    if os.path.isfile(pointer):
        with open(pointer) as f:
            name = f.read().strip()
    if not name:
        steps = _step_dirs(ckpt_root)
        name = steps[-1] if steps else None
    if not name:
        return None, None
    ckpt_dir = os.path.join(ckpt_root, name)
    if not os.path.isdir(ckpt_dir):
        return None, None
    state = {}
    sidecar = os.path.join(ckpt_dir, STATE_SIDECAR)
    if os.path.isfile(sidecar):
        try:
            with open(sidecar) as f:
                state = json.load(f)
        except Exception:
            state = {}
    return ckpt_dir, state


COMPLETE_MARKER = ".training_complete"


def mark_training_complete(output_dir: str):
    try:
        with open(os.path.join(output_dir, COMPLETE_MARKER), "w") as f:
            f.write(str(int(time.time())))
    except Exception:
        pass


def is_training_complete(output_dir: str) -> bool:
    return os.path.isfile(os.path.join(output_dir, COMPLETE_MARKER))
