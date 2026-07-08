# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].
#
# Video-only JEPA pretraining trainer, extended with:
#   * rich W&B online logging + JEPA latent representation analysis
#   * robust requeue-friendly resume (collective DeepSpeed full-state save/load,
#     scheduler registered for checkpointing, completed_steps persisted)
#   * graceful stop on SIGUSR1 / wall-clock deadline so the run can self-requeue
#     across the SLURM time limit.
"""
StarVLA video JEPA pretrain loop (native PyTorch + Accelerate + DeepSpeed).
"""
import warnings

warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

# Standard Library
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

# Local Modules
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
    is_training_complete,
    mark_training_complete,
    sanitize_run_id,
    save_full_state,
)

# Plain DDP (no DeepSpeed): a ~2.5B-param model fits on 8xH100, and this avoids
# DeepSpeed eagerly importing Triton inference ops (whose JIT gcc/-lcuda compile
# fails on the compute nodes). find_unused_parameters=True because the frozen
# V-JEPA2 teacher (no_grad) and, on the video path, the action head are unused.
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
    logger.info(f"Creating Video Dataset `{cfg.datasets.video_data.dataset_py}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.video_data.dataset_py)
    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


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
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )
    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.run_root_dir, cfg.run_id, "tensorboard"))

        self.completed_steps = 0
        self.vla_epoch_count = 0
        self.total_batch_size = self._calculate_total_batch_size()

        # logging / resume / stop configuration (all have safe defaults)
        self.jepa_log_interval = int(_cfg_get(cfg, "trainer.jepa_log_interval", 25))
        self.jepa_figure_interval = int(_cfg_get(cfg, "trainer.jepa_figure_interval", 250))
        self.ckpt_interval = int(_cfg_get(cfg, "trainer.ckpt_interval", _cfg_get(cfg, "trainer.save_interval", 1000)))
        self.keep_last_ckpts = int(_cfg_get(cfg, "trainer.keep_last_checkpoints", 2))
        self.resume_from_latest = bool(_cfg_get(cfg, "trainer.resume_from_latest", False))
        self.grace_seconds = int(_cfg_get(cfg, "trainer.grace_seconds", 300))
        self.wandb_logger = None
        self.stopper = None
        self._exit_for_requeue = False

    def _calculate_total_batch_size(self):
        return (
            self.config.datasets.video_data.per_device_batch_size
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

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator, self.model, self.optimizer, self.vla_train_dataloader
        )
        # Register the (manually-stepped) scheduler so accelerator.save_state/load_state
        # persists its state for resume, WITHOUT changing LR-stepping semantics.
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
        mode = _cfg_get(self.config, "wandb.mode", "online")
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
            mode=mode,
            config_dict=config_dict,
            tags=["vla-jepa", "video-pretrain"],
            group=_cfg_get(self.config, "wandb.group", "vlajepa-video"),
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
        self.accelerator.load_state(ckpt_dir)  # collective
        self.completed_steps = int(state.get("completed_steps", 0))
        self.vla_epoch_count = int(state.get("vla_epoch_count", 0))
        self.accelerator.print(f"[resume] resumed at step {self.completed_steps}")
        self.accelerator.wait_for_everyone()

    # ----------------------------------------------------------------- checkpoint
    def _save_full(self, tag=None):
        save_full_state(
            self.accelerator,
            self.checkpoint_dir,
            self.completed_steps,
            extra={
                "vla_epoch_count": self.vla_epoch_count,
                "wandb_run_id": self._wandb_run_id(),
                "memory_schema_version": int(
                    _cfg_get(self.config, "framework.memory.schema_version", 0)
                    if _cfg_get(self.config, "framework.memory.enabled", False)
                    else 0
                ),
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

    def _resume_data_iterator(self):
        batches_per_epoch = len(self.vla_train_dataloader)
        if batches_per_epoch <= 0:
            raise RuntimeError("video dataloader is empty")
        epoch, offset = divmod(self.completed_steps, batches_per_epoch)
        self._set_dataloader_epoch(self.vla_train_dataloader, epoch)
        iterable = (
            self.accelerator.skip_first_batches(
                self.vla_train_dataloader, num_batches=offset
            )
            if offset
            else self.vla_train_dataloader
        )
        self._set_dataloader_epoch(iterable, epoch)
        self.accelerator.print(
            f"[data-resume] video: completed={self.completed_steps}, "
            f"batches/epoch={batches_per_epoch}, epoch={epoch}, offset={offset}"
        )
        return iter(iterable), epoch

    def _create_data_iterators(self):
        self.vla_iter, self.vla_epoch_count = self._resume_data_iterator()

    def _get_next_batch(self):
        try:
            return next(self.vla_iter)
        except StopIteration:
            self.vla_epoch_count += 1
            self._set_dataloader_epoch(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            self.vla_iter = iter(self.vla_train_dataloader)
            return next(self.vla_iter)

    # ----------------------------------------------------------------- train
    def train(self):
        self._log_training_config()
        self._create_data_iterators()
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.jepa_num_views = 2

        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            step_to_complete = self.completed_steps + 1
            do_jepa = bool(
                self.wandb_logger.enabled
                and self.jepa_log_interval > 0
                and step_to_complete % self.jepa_log_interval == 0
            )
            do_fig = bool(
                self.wandb_logger.enabled
                and self.jepa_figure_interval > 0
                and step_to_complete % self.jepa_figure_interval == 0
            )
            unwrapped.capture_jepa = do_jepa or do_fig

            t0 = time.perf_counter()
            batch_vla = self._get_next_batch()
            t1 = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t2 = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            step_metrics["time/data"] = t1 - t0
            step_metrics["time/model"] = t2 - t1
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix({"data": f"{t1-t0:.2f}", "model": f"{t2-t1:.2f}"})

            self._log_metrics(step_metrics)
            if do_jepa or do_fig:
                self._log_jepa(unwrapped, do_fig)

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

    def _train_step(self, batch_vla, batch_vlm=None):
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model(batch_vla)  # __call__ -> DDP grad sync hooks
                total_loss = sum(output_dict.values())
            self.accelerator.backward(total_loss)
            grad_norm = None
            if self.config.trainer.gradient_clipping is not None:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )
            self.optimizer.step()
            self.lr_scheduler.step()

        result = {f"loss/{k}": v.item() for k, v in output_dict.items()}
        result["loss/total"] = float(sum(output_dict.values()).item())
        if grad_norm is not None:
            try:
                result["opt/grad_norm"] = float(grad_norm)
            except Exception:
                pass
        # memv3 retro diagnostics (pick_acc, prior_gap, ...) ride the same
        # scalar channel as losses; the model refreshes them every forward.
        diagnostics = getattr(
            self.accelerator.unwrap_model(self.model), "last_memory_diagnostics", None
        )
        if diagnostics:
            for key, value in diagnostics.items():
                try:
                    result[f"memory/{key}"] = float(value)
                except Exception:
                    pass
        return result

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
            tag = k.replace("/", "_")
            try:
                self.writer.add_scalar(tag, v, self.completed_steps)
            except Exception:
                pass
        self.wandb_logger.log(metrics, step=self.completed_steps)
        logger.info(f"Step {self.completed_steps} | {metrics}")

    def _log_jepa(self, unwrapped, do_fig):
        if not self.accelerator.is_main_process:
            return
        t = getattr(unwrapped, "last_jepa_tensors", None)
        if not t:
            return
        try:
            stats = jepa_analysis.compute_jepa_scalar_stats(
                t["predicted"], t["gt"], t.get("input"), t.get("action_tokens"),
                num_views=getattr(unwrapped, "jepa_num_views", 2),
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
            unwrapped.last_jepa_tensors = None

    def _log_training_config(self):
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.video_data.per_device_batch_size}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(f"  Resume-from-latest = {self.resume_from_latest}; start step = {self.completed_steps}")
            if self.stopper and self.stopper.deadline:
                logger.info(f"  Deadline in {self.stopper.seconds_left():.0f}s (grace {self.grace_seconds}s)")

    def _finalize_training(self):
        if self._exit_for_requeue:
            # Cut short by deadline/signal: do NOT mark complete; let the sbatch requeue.
            self.wandb_logger.finish()
            self.accelerator.wait_for_everyone()
            self.accelerator.print("[exit] graceful stop complete; exiting 0 for requeue")
            sys.exit(0)

        # Natural completion: save final weights (inference-friendly) + full state + marker.
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
    logger.info("VLA-JEPA Video Pretrain :: Warming Up")
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
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
