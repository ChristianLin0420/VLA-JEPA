# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
#
# VLA-JEPA cotrain (robot VLA + human video) trainer, extended with:
#   * rich W&B online logging + JEPA latent representation analysis
#   * robust requeue-friendly resume (collective DeepSpeed full-state save/load)
#   * graceful stop on SIGUSR1 / wall-clock deadline for self-requeue.
"""
StarVLA cotrain loop: per step, one VLA (action + world-model) pass and one
human-video (world-model only) pass. Native PyTorch + Accelerate + DeepSpeed.
"""
import warnings

warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.model.modules.memory import MemoryState
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)
from starVLA.training.trainer_utils import jepa_analysis
from starVLA.training.trainer_utils.run_utils import (
    GracefulStopper,
    RichWandbLogger,
    find_latest_checkpoint,
    mark_training_complete,
    sanitize_run_id,
    save_full_state,
)

# Plain DDP (no DeepSpeed) -- avoids DeepSpeed's eager Triton-inference import,
# whose JIT compile (-lcuda) fails on the compute nodes. The model fits on 8xH100.
# find_unused_parameters=True: the frozen V-JEPA2 teacher (no_grad) and the action
# head (unused on the per-step video pass) would otherwise trip DDP.
accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
accelerator.print(accelerator.state)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)


def _cfg_get(cfg, dotted, default=None):
    cur = cfg
    for k in dotted.split("."):
        if cur is None:
            return default
        try:
            cur = cur[k] if k in cur else default
        except Exception:
            cur = getattr(cur, k, default)
        if cur is default:
            return default
    return cur


def _migration_missing_prefixes(cfg):
    migration = _cfg_get(cfg, "trainer.checkpoint_migration", None)
    if not migration or not bool(migration.get("enabled", False)):
        return ()
    unexpected = list(migration.get("allow_unexpected_prefixes", []))
    if unexpected:
        raise ValueError(
            "checkpoint migration does not permit unexpected keys; "
            f"got allow_unexpected_prefixes={unexpected}"
        )
    return tuple(migration.get("allow_missing_prefixes", []))


def _stable_id(value) -> int:
    """Process-stable 63-bit hash for cross-rank metadata comparison."""
    digest = hashlib.sha256(repr(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _all_gather_list(tensor):
    """Every rank's copy of ``tensor`` (equal shapes required), rank-ordered."""
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return gathered


class EpisodicNCEQueue:
    """Cross-rank FIFO of detached InfoNCE targets with episode/task metadata.

    Current positives (detached) are gathered across ranks and enqueued
    before the loss, so same-step cross-rank targets act as negatives.
    Same-task different-episode entries are the hard negatives and enter the
    denominator automatically; entries from the anchor's own episode are
    masked out as false negatives.
    """

    def __init__(self, size: int = 256, temperature: float = 0.07):
        self.size = int(size)
        self.temperature = float(temperature)
        self.embeddings = None
        self.tasks = None
        self.episodes = None

    def _enqueue(self, embeddings, tasks, episodes):
        if self.embeddings is None:
            self.embeddings, self.tasks, self.episodes = embeddings, tasks, episodes
        else:
            self.embeddings = torch.cat((self.embeddings, embeddings))
            self.tasks = torch.cat((self.tasks, tasks))
            self.episodes = torch.cat((self.episodes, episodes))
        if self.embeddings.shape[0] > self.size:
            self.embeddings = self.embeddings[-self.size:]
            self.tasks = self.tasks[-self.size:]
            self.episodes = self.episodes[-self.size:]

    def loss(self, anchors, positives, task_ids, episode_ids):
        """Return (InfoNCE loss, scalar diagnostics) for [N, D] unit anchors."""
        if anchors.ndim != 2 or anchors.shape != positives.shape:
            raise ValueError("anchors and positives must both have shape [N, D]")
        candidates = positives.detach()
        candidate_tasks, candidate_episodes = task_ids, episode_ids
        if dist.is_initialized() and dist.get_world_size() > 1:
            candidates = torch.cat(_all_gather_list(candidates))
            candidate_tasks = torch.cat(_all_gather_list(task_ids))
            candidate_episodes = torch.cat(_all_gather_list(episode_ids))
        self._enqueue(candidates, candidate_tasks, candidate_episodes)

        positive_logit = (anchors * positives).sum(dim=-1, keepdim=True)
        negative_logits = anchors @ self.embeddings.t()
        false_negative = self.episodes[None, :] == episode_ids[:, None]
        negative_logits = negative_logits.masked_fill(false_negative, float("-inf"))
        logits = torch.cat((positive_logit, negative_logits), dim=1) / self.temperature
        targets = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
        loss = F.cross_entropy(logits, targets)

        diagnostics = {}
        with torch.no_grad():
            rows = (~false_negative).any(dim=1)
            if bool(rows.any()):
                hardest = negative_logits.max(dim=1).values
                diagnostics["nce/acc"] = float(
                    (positive_logit.squeeze(1) > hardest)[rows].float().mean()
                )
            same_task = ~false_negative & (self.tasks[None, :] == task_ids[:, None])
            rows_same = same_task.any(dim=1)
            if bool(rows_same.any()):
                hardest_same = negative_logits.masked_fill(
                    ~same_task, float("-inf")
                ).max(dim=1).values
                diagnostics["nce/same_task_acc"] = float(
                    (positive_logit.squeeze(1) > hardest_same)[rows_same].float().mean()
                )
        return loss, diagnostics


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        os.makedirs(output_dir / "wandb", exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            json.dump(yaml.safe_load(f_yaml), f_json, indent=2)
    return output_dir


def prepare_data(cfg, accelerator, output_dir):
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    robot_only = bool(_cfg_get(cfg, "trainer.robot_only", False))
    video_train_dataloader = None
    if not robot_only:
        video_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.video_data.dataset_py)
    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader, video_train_dataloader


def setup_optimizer_and_scheduler(model, cfg):
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")
    # One outer cotrain step performs one VLA optimizer update and one human-video
    # optimizer update.  Express the LR schedule in optimizer-update units so a
    # 100k outer-step run does not finish (and then repeat) its cosine at 50k.
    optimizer_steps_per_training_step = int(
        _cfg_get(cfg, "trainer.optimizer_steps_per_training_step", 2)
    )
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=(
            cfg.trainer.num_warmup_steps * optimizer_steps_per_training_step
        ),
        num_training_steps=(
            cfg.trainer.max_train_steps * optimizer_steps_per_training_step
        ),
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )
    return optimizer, lr_scheduler


class VLAMTrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, video_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.video_train_dataloader = video_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.run_root_dir, cfg.run_id, "tensorboard"))

        self.completed_steps = 0
        self.vla_epoch_count = 0
        self.vlm_epoch_count = 0
        self.total_batch_size = self._calculate_total_batch_size()
        segment_length = int(_cfg_get(cfg, "datasets.vla_data.segment_length", 1))
        self.decisions_per_step = self.total_batch_size * (
            segment_length
            if _cfg_get(cfg, "datasets.vla_data.sample_mode", "single_step") == "contiguous_segment"
            else 1
        )
        self.processed_decisions = 0

        self.segment_length = segment_length
        self.rec_loss_weight = float(_cfg_get(cfg, "trainer.rec_loss_weight", 0.0))
        self.nce_loss_weight = float(_cfg_get(cfg, "trainer.nce_loss_weight", 0.0))
        self.mask_ramp_steps = int(_cfg_get(cfg, "trainer.mask_ramp_steps", 0))
        self.memory_mask_rate = float(
            _cfg_get(cfg, "datasets.vla_data.memory_mask_rate", 0.0)
        )
        self._nce_queue = EpisodicNCEQueue() if self.nce_loss_weight else None
        self._meters_due = False

        self.jepa_log_interval = int(_cfg_get(cfg, "trainer.jepa_log_interval", 25))
        self.jepa_figure_interval = int(_cfg_get(cfg, "trainer.jepa_figure_interval", 250))
        self.ckpt_interval = int(_cfg_get(cfg, "trainer.ckpt_interval", _cfg_get(cfg, "trainer.save_interval", 1000)))
        self.keep_last_ckpts = int(_cfg_get(cfg, "trainer.keep_last_checkpoints", 2))
        self.resume_from_latest = bool(_cfg_get(cfg, "trainer.resume_from_latest", False))
        self.grace_seconds = int(_cfg_get(cfg, "trainer.grace_seconds", 300))
        self.eval_interval = int(_cfg_get(cfg, "trainer.eval_interval", 0))
        self.wandb_logger = None
        self.stopper = None
        self._exit_for_requeue = False
        self._unwrapped = None

    def _calculate_total_batch_size(self):
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    # ----------------------------------------------------------------- setup
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        set_seed((self.config.seed if hasattr(self.config, "seed") else 3047) + rank)

        if _cfg_get(self.config, "trainer.pretrained_checkpoint", None):
            self.model = self.load_pretrained_backbones(
                self.model,
                self.config.trainer.pretrained_checkpoint,
                reload_modules=_cfg_get(self.config, "trainer.reload_modules", None),
                allowed_missing_prefixes=_migration_missing_prefixes(self.config),
            )
        self.model = self.freeze_backbones(self.model, freeze_modules=_cfg_get(self.config, "trainer.freeze_modules", None))
        self.print_trainable_parameters(self.model)

        if self.video_train_dataloader is None:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator, self.model, self.optimizer, self.vla_train_dataloader
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader, self.video_train_dataloader = (
                self.setup_distributed_training(
                    self.accelerator,
                    self.model,
                    self.optimizer,
                    self.vla_train_dataloader,
                    self.video_train_dataloader,
                )
            )
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        self._configure_mask_schedule()

        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._init_wandb()
        self.stopper = GracefulStopper(grace_seconds=self.grace_seconds)
        self._maybe_resume()

    def _configure_mask_schedule(self):
        """Push blind-decision masking knobs to the segment dataset.

        Worker-state inspection: the VLA dataloader forks non-persistent
        workers at each epoch's ``iter()``, and every worker snapshots the
        dataset object at fork — the same mechanism ``set_epoch`` relies on.
        A per-step mutable attribute would therefore go stale inside an
        epoch, so every schedule input set here is static and
        ``sample_segment`` derives the linearly ramped rate itself from the
        (epoch, index) sample ordinal, which also keeps mask plans
        deterministic under requeue/resume.
        """
        if self.memory_mask_rate <= 0.0:
            return
        dataset = getattr(self.vla_train_dataloader, "dataset", None)
        if dataset is None or not callable(getattr(dataset, "sample_segment", None)):
            raise ValueError(
                "datasets.vla_data.memory_mask_rate requires the contiguous-segment VLA dataset"
            )
        dataset.memory_mask_rate = self.memory_mask_rate
        dataset.memory_mask_max_per_segment = int(
            _cfg_get(self.config, "datasets.vla_data.memory_mask_max_per_segment", 1)
        )
        dataset.memory_mask_run_len = int(
            _cfg_get(self.config, "datasets.vla_data.memory_mask_run_len", 1)
        )
        if dataset.memory_mask_run_len > dataset.memory_mask_max_per_segment:
            raise ValueError(
                "memory_mask_run_len exceeds memory_mask_max_per_segment; "
                "raise the cap to match the run length"
            )
        # One outer step consumes total_batch_size segments across all ranks.
        dataset.memory_mask_ramp_samples = self.mask_ramp_steps * self.total_batch_size

    def _wandb_run_id(self):
        exp = _cfg_get(self.config, "experiment_id", self.config.run_id)
        return sanitize_run_id(f"{self.config.run_id}-{exp}")

    def _init_wandb(self):
        try:
            config_dict = OmegaConf.to_container(self.config, resolve=True)
        except Exception:
            config_dict = {}
        self.wandb_logger = RichWandbLogger(
            self.config,
            self.accelerator,
            project=_cfg_get(self.config, "wandb.project", "vla-jepa"),
            entity=_cfg_get(self.config, "wandb.entity", None),
            run_name=f"{self.config.run_id}-{_cfg_get(self.config, 'experiment_id', '')}",
            run_id=self._wandb_run_id(),
            mode=_cfg_get(self.config, "wandb.mode", "online"),
            config_dict=config_dict,
            tags=["vla-jepa", "cotrain"],
            group=_cfg_get(self.config, "wandb.group", "vlajepa-cotrain"),
        )
        if self.wandb_logger.enabled:
            n_params, n_train = self.print_trainable_parameters(self.model) or (0, 0)
            self.wandb_logger.summary(total_params_M=n_params, trainable_params_M=n_train,
                                      total_batch_size=self.total_batch_size)

    def _maybe_resume(self):
        if not self.resume_from_latest:
            return
        ckpt_dir, state = find_latest_checkpoint(self.checkpoint_dir)
        if ckpt_dir is None:
            self.accelerator.print(f"[resume] no checkpoint under {self.checkpoint_dir}; starting fresh")
            return
        self.accelerator.print(f"[resume] loading full state from {ckpt_dir}")
        self.accelerator.load_state(ckpt_dir)
        self.completed_steps = int(state.get("completed_steps", 0))
        self.vla_epoch_count = int(state.get("vla_epoch_count", 0))
        self.vlm_epoch_count = int(state.get("vlm_epoch_count", 0))
        self.processed_decisions = int(
            state.get("processed_decisions", self.completed_steps * self.decisions_per_step)
        )
        self.accelerator.print(f"[resume] resumed at step {self.completed_steps}")
        self.accelerator.wait_for_everyone()

    def _save_full(self, tag=None):
        save_full_state(
            self.accelerator,
            self.checkpoint_dir,
            self.completed_steps,
            extra={
                "vla_epoch_count": self.vla_epoch_count,
                "vlm_epoch_count": self.vlm_epoch_count,
                "wandb_run_id": self._wandb_run_id(),
                "memory_schema_version": int(
                    _cfg_get(self.config, "framework.memory.schema_version", 0)
                    if _cfg_get(self.config, "framework.memory.enabled", False)
                    else 0
                ),
                "processed_decisions": self.processed_decisions,
            },
            keep_last=self.keep_last_ckpts,
            tag=tag,
        )
        if self.accelerator.is_main_process:
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps({"steps": self.completed_steps, "time": int(time.time())}) + "\n")
            self.accelerator.print(f"✅ full-state checkpoint @ step {self.completed_steps}")

    # ----------------------------------------------------------------- data iter
    @staticmethod
    def _set_dataloader_epoch(dataloader, epoch):
        if callable(getattr(dataloader, "set_epoch", None)):
            dataloader.set_epoch(epoch)
        elif callable(getattr(getattr(dataloader, "sampler", None), "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch)
        elif callable(getattr(getattr(dataloader, "dataset", None), "set_epoch", None)):
            dataloader.dataset.set_epoch(epoch)

    def _resume_data_iterator(self, dataloader, label):
        batches_per_epoch = len(dataloader)
        if batches_per_epoch <= 0:
            raise RuntimeError(f"{label} dataloader is empty")
        epoch, offset = divmod(self.completed_steps, batches_per_epoch)
        self._set_dataloader_epoch(dataloader, epoch)
        iterable = (
            self.accelerator.skip_first_batches(dataloader, num_batches=offset)
            if offset
            else dataloader
        )
        # ``skip_first_batches`` constructs a new DataLoaderShard whose iteration
        # counter starts at zero.  Apply the epoch to that wrapper as well so its
        # first ``__iter__`` does not reset the dataset/sampler epoch on resume.
        self._set_dataloader_epoch(iterable, epoch)
        self.accelerator.print(
            f"[data-resume] {label}: completed={self.completed_steps}, "
            f"batches/epoch={batches_per_epoch}, epoch={epoch}, offset={offset}"
        )
        return iter(iterable), epoch

    def _next_epoch_iterator(self, dataloader, epoch):
        epoch += 1
        self._set_dataloader_epoch(dataloader, epoch)
        return iter(dataloader), epoch

    def _create_data_iterators(self):
        self.vla_iter, self.vla_epoch_count = self._resume_data_iterator(
            self.vla_train_dataloader, "vla"
        )
        self.vlm_iter = None
        if self.video_train_dataloader is not None:
            self.vlm_iter, self.vlm_epoch_count = self._resume_data_iterator(
                self.video_train_dataloader, "video"
            )

    def _get_next_batch(self):
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_iter, self.vla_epoch_count = self._next_epoch_iterator(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)
        batch_vlm = None
        if self.video_train_dataloader is not None:
            try:
                batch_vlm = next(self.vlm_iter)
            except StopIteration:
                self.vlm_iter, self.vlm_epoch_count = self._next_epoch_iterator(
                    self.video_train_dataloader, self.vlm_epoch_count
                )
                batch_vlm = next(self.vlm_iter)
        return batch_vla, batch_vlm

    # ----------------------------------------------------------------- train
    def train(self):
        self._log_training_config()
        self._create_data_iterators()
        self._unwrapped = self.accelerator.unwrap_model(self.model)
        self._unwrapped.jepa_num_views = 2

        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            step_to_complete = self.completed_steps + 1
            do_jepa = bool(self.wandb_logger.enabled and step_to_complete % self.jepa_log_interval == 0)
            do_fig = bool(self.wandb_logger.enabled and step_to_complete % self.jepa_figure_interval == 0)
            self._capture = do_jepa or do_fig
            # Meter cadence is static config + step count, so every rank
            # agrees and the cross-rank state exchange inside cannot skew.
            self._meters_due = bool(
                self.memory_mask_rate > 0.0
                and step_to_complete % int(self.config.trainer.logging_frequency) == 0
            )

            t0 = time.perf_counter()
            batch_vla, batch_vlm = self._get_next_batch()
            t1 = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_vlm)
            t2 = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
                self.processed_decisions += self.decisions_per_step

            step_metrics["time/data"] = t1 - t0
            step_metrics["time/model"] = t2 - t1
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix({"data": f"{t1-t0:.2f}", "model": f"{t2-t1:.2f}"})

            if self.eval_interval and self.completed_steps % self.eval_interval == 0 and self.completed_steps > 0:
                self._safe_eval(step_metrics)

            self._log_metrics(step_metrics)
            if do_jepa or do_fig:
                self._log_jepa(do_fig)

            if self.completed_steps % self.ckpt_interval == 0 and self.completed_steps > 0:
                self._save_full()

            if self.stopper.should_stop(self.accelerator):
                self.accelerator.print(f"[stop] {self.stopper.reason} @ step {self.completed_steps}; checkpointing for requeue")
                self._save_full()
                self._exit_for_requeue = True
                break

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _train_step(self, batch_vla, batch_vlm):
        log_dict = {}
        with self.accelerator.accumulate(self.model):
            # --- VLA pass (action + world-model); capture JEPA tensors here ---
            self.optimizer.zero_grad()
            if self._unwrapped is not None:
                self._unwrapped.capture_jepa = bool(getattr(self, "_capture", False))
                # Toggle both memory capture flags together so fusion's
                # injection_ratio (a GPU sync) is only computed on capture
                # steps; memory/injection_ratio is therefore cadence-sampled.
                for module_name in ("memory_module", "policy_memory_fusion"):
                    module = getattr(self._unwrapped, module_name, None)
                    if module is not None:
                        module.capture_diagnostics = bool(getattr(self, "_capture", False))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model(batch_vla)  # __call__ -> DDP grad sync hooks
            nce_anchor = output_dict.pop("nce_anchor", None)
            nce_positive = output_dict.pop("nce_positive", None)
            total_loss, nce_loss, nce_metrics = self._aggregate_vla_losses(
                output_dict, nce_anchor, nce_positive, batch_vla
            )
            self.accelerator.backward(total_loss)
            if self._unwrapped is not None:
                self._unwrapped.capture_jepa = False  # don't let the video pass overwrite
            if self.config.trainer.gradient_clipping is not None:
                gn = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                try:
                    log_dict["opt/grad_norm_vla"] = float(gn)
                except Exception:
                    pass
            # Cache robot-pass memory scalars now: the meter passes and the
            # video pass below call forward(), which resets the diagnostics.
            memory_metrics = self._collect_memory_metrics()
            if self._meters_due:
                # Counterfactual passes must see pre-update weights.
                memory_metrics.update(self._memv2_meters(batch_vla, output_dict))
            self.optimizer.step()
            self.lr_scheduler.step()

            vlm_output = None
            vlm_loss = None
            if batch_vlm is not None:
                # --- human-video pass (world-model only, independent state) ---
                self.optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vlm_output = self.model(batch_vlm)  # __call__ -> DDP grad sync hooks
                    vlm_loss = sum(vlm_output.values())
                self.accelerator.backward(vlm_loss)
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                self.optimizer.step()
                self.lr_scheduler.step()

        for k, v in output_dict.items():
            log_dict[f"loss/vla_{k}"] = v.item()
        if nce_loss is not None:
            log_dict["loss/vla_nce_loss"] = float(nce_loss.item())
            log_dict.update(nce_metrics)
        if vlm_output is not None:
            for k, v in vlm_output.items():
                log_dict[f"loss/vlm_{k}"] = v.item()
        log_dict["loss/vla_total"] = float(total_loss.item())
        if vlm_loss is not None:
            log_dict["loss/vlm_total"] = float(vlm_loss.item())
        log_dict.update(memory_metrics)
        return log_dict

    def _collect_memory_metrics(self):
        """Robot-pass memory scalars; must be read before the video pass runs."""
        metrics = {}
        model = self._unwrapped
        if model is None:
            return metrics
        diagnostics = getattr(model, "last_memory_diagnostics", None)
        if diagnostics:
            for key, value in diagnostics.items():
                try:
                    metrics[f"memory/{key}"] = float(value.float().mean().item())
                except Exception:
                    pass
        fusion_diagnostics = getattr(
            getattr(model, "policy_memory_fusion", None), "last_fusion_diagnostics", None
        )
        if fusion_diagnostics:
            metrics["memory/injection_ratio"] = float(fusion_diagnostics["injection_ratio"])
        write_diagnostics = getattr(
            getattr(model, "memory_module", None), "last_write_diagnostics", None
        )
        if write_diagnostics:
            for key, value in write_diagnostics.items():
                if isinstance(value, float):
                    metrics[f"memory/{key}"] = value
                elif key == "per_slot_delta_norm":
                    metrics[f"memory/{key}"] = float(value.float().mean().item())
        return metrics

    def _aggregate_vla_losses(self, losses, nce_anchor, nce_positive, batch_vla):
        """memv2 loss assembly.

        With the default-zero weights this reduces exactly to memv1's plain
        sum over the model's loss dict; ``rec_loss`` is trainer-weighted and
        the InfoNCE term is computed here from the model's per-step
        anchor/positive tensors.
        """
        total = sum(value for key, value in losses.items() if key != "rec_loss")
        if "rec_loss" in losses and self.rec_loss_weight:
            total = total + self.rec_loss_weight * losses["rec_loss"]
        nce_loss, nce_metrics = None, {}
        if self.nce_loss_weight and nce_anchor is not None and nce_positive is not None:
            anchors = nce_anchor.reshape(-1, nce_anchor.shape[-1]).float()
            positives = nce_positive.reshape(-1, nce_positive.shape[-1]).float()
            task_ids, episode_ids = self._segment_metadata_ids(
                batch_vla, anchors.shape[0], anchors.device
            )
            nce_loss, nce_metrics = self._nce_queue.loss(
                anchors, positives, task_ids, episode_ids
            )
            total = total + self.nce_loss_weight * nce_loss
        return total, nce_loss, nce_metrics

    @staticmethod
    def _segment_metadata_ids(batch_vla, count, device):
        """Per-anchor (task, episode) ids, tiled to match step-major anchors."""
        if not batch_vla or count % len(batch_vla) != 0:
            raise ValueError(
                f"cannot align {count} NCE anchors with {len(batch_vla)} segments"
            )
        tasks, episodes = [], []
        for segment in batch_vla:
            dataset_id = str(segment.get("dataset_id", ""))
            first_step = next(step for step in segment["steps"] if step is not None)
            tasks.append(_stable_id((dataset_id, str(first_step.get("lang", "")))))
            episodes.append(_stable_id((dataset_id, int(segment.get("episode_id", -1)))))
        repeats = count // len(batch_vla)
        task_ids = torch.tensor(tasks, dtype=torch.int64, device=device)
        episode_ids = torch.tensor(episodes, dtype=torch.int64, device=device)
        return task_ids.repeat(repeats), episode_ids.repeat(repeats)

    def _memv2_meters(self, batch_vla, live_losses):
        """Δbypass/Δforeign no-backward counterfactual passes (design §5).

        Reruns the unwrapped model on the live batch before the optimizer
        step: once with the fusion forced to bypass, once with the states
        seen by the last K (= segment_length) memory reads replaced by a
        maturity-matched foreign episode's — the neighbouring rank's under
        DDP, the neighbouring batch row when world_size == 1, skipped when
        neither exists.  The foreign swap is applied at BOTH consumers of the
        pre-write state: memory.read and the fusion (the schema-2 fusion takes
        the state directly, so a read-only swap never reaches the policy or
        rec conditioning — the memv2 stage-1 delta_foreign_rec ≡ 0 bug).
        Diagnostics capture is forced off so the live-pass scalars collected
        just before are not clobbered; the local write chain is untouched
        (foreign content enters at read time only).
        """
        model = self._unwrapped
        memory = getattr(model, "memory_module", None)
        fusion = getattr(model, "policy_memory_fusion", None)
        if memory is None or fusion is None:
            return {}
        memory.capture_diagnostics = False
        fusion.capture_diagnostics = False

        meters = {}
        recorded = []
        bound_read = memory.read
        bound_fusion = fusion.forward

        def recording_read(source_tokens, state, **kwargs):
            recorded.append(state.detach())
            return bound_read(source_tokens, state, **kwargs)

        def bypass_forward(*args, **kwargs):
            kwargs["bypass"] = True
            return bound_fusion(*args, **kwargs)

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        with torch.no_grad():
            memory.read = recording_read
            fusion.forward = bypass_forward
            try:
                with autocast:
                    bypassed = model(batch_vla)
            finally:
                del memory.read
                del fusion.forward
            self._meter_deltas(meters, "bypass", live_losses, bypassed)

            foreign = self._foreign_read_states(recorded)
            if foreign is None:
                return meters
            foreign_working, foreign_keys = foreign
            first_foreign_call = len(recorded) - foreign_working.shape[0]
            calls = {"seen": 0}
            current = {"state": None}

            def foreign_read(source_tokens, state, **kwargs):
                position = calls["seen"] - first_foreign_call
                calls["seen"] += 1
                if 0 <= position < foreign_working.shape[0]:
                    state = dataclasses.replace(
                        state,
                        working=foreign_working[position],
                        keys=(
                            foreign_keys[position]
                            if foreign_keys is not None
                            else state.keys
                        ),
                    )
                current["state"] = state
                return bound_read(source_tokens, state, **kwargs)

            def foreign_fusion(consumer_tokens, memory_arg, *args, **kwargs):
                # The schema-2 fusion consumes the pre-write MemoryState
                # directly, not the read's output tokens, so the swap made
                # inside memory.read never reaches it: re-apply it here.
                # Read and fusion run in lockstep once per decision, so the
                # latest read's state is exactly this call's state.  Schema-1
                # fusion receives read tokens (a Tensor) and is untouched.
                if isinstance(memory_arg, MemoryState) and current["state"] is not None:
                    memory_arg = current["state"]
                return bound_fusion(consumer_tokens, memory_arg, *args, **kwargs)

            memory.read = foreign_read
            fusion.forward = foreign_fusion
            try:
                with autocast:
                    foreigned = model(batch_vla)
            finally:
                del memory.read
                del fusion.forward
            self._meter_deltas(meters, "foreign", live_losses, foreigned)
        return meters

    def _foreign_read_states(self, recorded):
        """Foreign (working, keys) for the last K supervised reads, or None."""
        if len(recorded) < self.segment_length:
            return None
        supervised = recorded[-self.segment_length:]
        working = torch.stack([state.working for state in supervised])
        keys = (
            torch.stack([state.keys for state in supervised])
            if supervised[0].keys is not None
            else None
        )
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            neighbor = (dist.get_rank() + 1) % world_size
            working = _all_gather_list(working)[neighbor]
            keys = _all_gather_list(keys)[neighbor] if keys is not None else None
        elif working.shape[1] > 1:
            working = working.roll(1, dims=1)
            keys = keys.roll(1, dims=1) if keys is not None else None
        else:
            return None
        return working, keys

    @staticmethod
    def _meter_deltas(meters, mode, live, counterfactual):
        for key, name in (("rec_loss", "rec"), ("action_loss", "act")):
            if key in live and key in counterfactual:
                meters[f"meters/delta_{mode}_{name}"] = float(counterfactual[key]) - float(live[key])

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return
        if not self.accelerator.is_main_process:
            return
        metrics["opt/learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        denom = max(1, len(self.vla_train_dataloader))
        metrics["opt/epoch"] = round(self.completed_steps / denom, 4)
        metrics["data/processed_decisions"] = self.processed_decisions
        if self.stopper is not None and self.stopper.seconds_left() is not None:
            metrics["time/seconds_to_deadline"] = self.stopper.seconds_left()
        for k, v in metrics.items():
            try:
                self.writer.add_scalar(k.replace("/", "_"), v, self.completed_steps)
            except Exception:
                pass
        self.wandb_logger.log(metrics, step=self.completed_steps)
        logger.info(f"Step {self.completed_steps} | {metrics}")

    def _log_jepa(self, do_fig):
        if not self.accelerator.is_main_process:
            return
        t = getattr(self._unwrapped, "last_jepa_tensors", None)
        if not t:
            return
        try:
            stats = jepa_analysis.compute_jepa_scalar_stats(
                t["predicted"], t["gt"], t.get("input"), t.get("action_tokens"),
                num_views=getattr(self._unwrapped, "jepa_num_views", 2),
            )
            self.wandb_logger.log(stats, step=self.completed_steps)
            for k, v in stats.items():
                try:
                    self.writer.add_scalar(k.replace("/", "_"), v, self.completed_steps)
                except Exception:
                    pass
            if do_fig:
                figs = jepa_analysis.make_jepa_figures(t["predicted"], t["gt"], t.get("input"))
                self.wandb_logger.log_figures(figs, step=self.completed_steps, prefix="jepa_media/")
        except Exception as e:
            logger.info(f"[jepa-analysis] skipped: {e}")
        finally:
            self._unwrapped.last_jepa_tensors = None

    def _safe_eval(self, step_metrics):
        """Synchronize at eval intervals without advancing live training data.

        The former best-effort probe fetched through ``_get_next_batch`` on rank 0,
        permanently shifting that rank's VLA and SSV2 iterators relative to its
        peers and making exact resume offsets impossible.  Full LIBERO evaluation
        is already run by the dependent evaluation job, so keep this interval as a
        cheap synchronization/logging point and leave both training streams intact.
        """
        if self.accelerator.is_main_process:
            logger.info(
                f"[eval] step {self.completed_steps}: in-training probe disabled "
                "to preserve deterministic dataloader position"
            )
        if dist.is_initialized():
            dist.barrier()

    def _log_training_config(self):
        if self.accelerator.is_main_process:
            logger.info("***** Cotrain Configuration *****")
            updates_per_step = int(
                _cfg_get(self.config, "trainer.optimizer_steps_per_training_step", 2)
            )
            logger.info(f"  Total outer training steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Optimizer updates per training step = {updates_per_step}")
            logger.info(
                f"  Total optimizer updates = "
                f"{self.config.trainer.max_train_steps * updates_per_step}"
            )
            logger.info(f"  Per device batch size (vla) = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(f"  Supervised robot decisions per outer step = {self.decisions_per_step}")
            logger.info(f"  Resume-from-latest = {self.resume_from_latest}; start step = {self.completed_steps}")
            if self.stopper and self.stopper.deadline:
                logger.info(f"  Deadline in {self.stopper.seconds_left():.0f}s (grace {self.grace_seconds}s)")

    def _finalize_training(self):
        if self._exit_for_requeue:
            self.wandb_logger.finish()
            self.accelerator.wait_for_everyone()
            self.accelerator.print("[exit] graceful stop complete; exiting 0 for requeue")
            sys.exit(0)

        self._save_full(tag=f"step_{self.completed_steps}")
        if self.accelerator.is_main_process:
            final_dir = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            torch.save(self.accelerator.get_state_dict(self.model), os.path.join(final_dir, "pytorch_model.pt"))
            mark_training_complete(self.config.output_dir)
            logger.info(f"Training complete. Final model saved at {final_dir}")
        self.wandb_logger.finish()
        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA-JEPA Cotrain :: Warming Up")
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader, video_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)
    trainer = VLAMTrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        video_train_dataloader=video_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_cotrain_oxe.yaml")
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(clipargs))
    cfg = OmegaConf.merge(cfg, cli_cfg)
    main(cfg)
