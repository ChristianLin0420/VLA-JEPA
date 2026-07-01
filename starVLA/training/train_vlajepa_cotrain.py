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
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
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
            )
        self.model = self.freeze_backbones(self.model, freeze_modules=_cfg_get(self.config, "trainer.freeze_modules", None))
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader, self.video_train_dataloader = (
            self.setup_distributed_training(
                self.accelerator, self.model, self.optimizer, self.vla_train_dataloader, self.video_train_dataloader
            )
        )
        self.accelerator.register_for_checkpointing(self.lr_scheduler)

        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._init_wandb()
        self.stopper = GracefulStopper(grace_seconds=self.grace_seconds)
        self._maybe_resume()

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

            t0 = time.perf_counter()
            batch_vla, batch_vlm = self._get_next_batch()
            t1 = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_vlm)
            t2 = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model(batch_vla)  # __call__ -> DDP grad sync hooks
                total_loss = sum(output_dict.values())
            self.accelerator.backward(total_loss)
            if self._unwrapped is not None:
                self._unwrapped.capture_jepa = False  # don't let the video pass overwrite
            if self.config.trainer.gradient_clipping is not None:
                gn = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
                try:
                    log_dict["opt/grad_norm_vla"] = float(gn)
                except Exception:
                    pass
            self.optimizer.step()
            self.lr_scheduler.step()

            # --- human-video pass (world-model only) ---
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
        for k, v in vlm_output.items():
            log_dict[f"loss/vlm_{k}"] = v.item()
        log_dict["loss/vla_total"] = float(total_loss.item())
        log_dict["loss/vlm_total"] = float(vlm_loss.item())
        return log_dict

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return
        if not self.accelerator.is_main_process:
            return
        metrics["opt/learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        denom = max(1, len(self.vla_train_dataloader))
        metrics["opt/epoch"] = round(self.completed_steps / denom, 4)
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
