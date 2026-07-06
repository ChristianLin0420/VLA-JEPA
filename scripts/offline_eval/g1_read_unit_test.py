"""G1 read unit test + leak audit for the memv2 sparse-key read (design gate G1).

(a) State-replay read pretrain: build a schema-2 model from a config, freeze
everything except ``policy_memory_fusion``, ``mem_cond_adapter``, and
``wm_mask_token``, and train ONLY the masked-reconstruction path on
teacher-forced state traces (the frozen memv2 write circuit replayed over the
cached Qwen action tokens, so schema-2 content keys exist; conventions from
``scripts/offline_eval/replay_engine.py``).  The fusion consumer is a zero
token block — the blind-step stand-in: cached embodied tokens are sighted and
would leak the target through the consumer channel, defeating both (a) and
(b).  PASS: held-out L_rec under a maturity-matched foreign state exceeds live
L_rec (one-sided paired bootstrap p < 0.05) — the new read extracts stored
episode content.

(b) Leak audit: retrain the same path from the same initialization with the
fusion bypassed; held-out L_rec must sit at the unconditional floor (L1 loss
of the best constant prediction, the train-set latent median).  PASS:
L_rec(bypass-trained) >= (1 - tolerance) * floor.

Teacher targets are frozen ``vj_encoder`` latents of each decision's clean
clip, computed exactly as ``_compute_world_loss`` (``VLA_JEPA.py``).  No
grad-scale on ``policy_tokens`` here: with no BC term the recon gradient is
the only gradient, so alpha would only rescale the learning rate.  Single GPU,
<1h at defaults; ``--smoke`` runs the full plumbing on tiny synthetic modules
on CPU (no checkpoint, no dataset).
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from starVLA.model.modules.memory.state import MemoryState

try:
    from scripts.offline_eval.replay_engine import (
        pick_donor_index,
        stable_seed,
        teacher_forced_states,
    )
except ImportError:  # executed as a plain file instead of a module
    from replay_engine import pick_donor_index, stable_seed, teacher_forced_states

DEFAULT_DATA_ROOT = "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot"


class ReconPath(NamedTuple):
    """The trainable recon circuit plus the frozen decoder it drives."""

    fusion: nn.Module
    adapter: nn.Module
    mask_token: nn.Parameter
    predictor: nn.Module
    num_consumer_tokens: int
    consumer_dim: int
    device: torch.device

    def trainable_parameters(self) -> List[nn.Parameter]:
        params = list(self.fusion.parameters()) + list(self.adapter.parameters())
        return params + [self.mask_token]


class Decision(NamedTuple):
    """One (episode, decision) with its live/foreign read states and teacher target."""

    live: MemoryState
    foreign: MemoryState
    target: torch.Tensor  # [T, D] fp16, CPU


def stack_states(states: List[MemoryState]) -> MemoryState:
    return MemoryState(
        working=torch.cat([state.working for state in states]),
        episodic=None,
        steps=torch.cat([state.steps for state in states]),
        valid=torch.cat([state.valid for state in states]),
        keys=torch.cat([state.keys for state in states]) if states[0].keys is not None else None,
    )


def recon_losses(path: ReconPath, state: MemoryState, targets: torch.Tensor,
                 *, bypass: bool = False) -> torch.Tensor:
    """Per-decision L1 recon loss [B] through the policy read (contract arithmetic)."""

    batch_size, num_latents = targets.shape[0], targets.shape[1]
    consumer = torch.zeros(
        batch_size, path.num_consumer_tokens, path.consumer_dim,
        dtype=torch.float32, device=path.device,
    )
    autocast = (
        torch.autocast("cuda", torch.bfloat16) if path.device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with autocast:
        policy_tokens = path.fusion(consumer, state, bypass=bypass)
        conditioning = path.adapter(policy_tokens)
        mask_tokens = (
            path.mask_token.to(torch.float32)[None, None, :].expand(batch_size, num_latents, -1)
        )
        predicted = path.predictor(mask_tokens, conditioning)
    return (predicted.to(torch.float32) - targets.to(torch.float32)).abs().mean(dim=(1, 2))


def train_arm(path: ReconPath, pool: List[Decision], *, steps: int, batch_size: int,
              lr: float, seed: int, bypass: bool, log_every: int = 50) -> None:
    """Train the recon path in place on the training pool (one arm)."""

    optimizer = torch.optim.AdamW(
        [param for param in path.trainable_parameters() if param.requires_grad], lr=lr
    )
    rng = np.random.default_rng(stable_seed("g1-train-v1", seed, bypass))
    label = "bypass" if bypass else "live"
    for step in range(steps):
        picks = rng.choice(len(pool), size=min(batch_size, len(pool)), replace=False)
        state = stack_states([pool[i].live for i in picks]).to(device=path.device)
        targets = torch.stack([pool[i].target for i in picks]).to(path.device)
        loss = recon_losses(path, state, targets, bypass=bypass).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % log_every == 0 or step == 0:
            print(f"[train:{label}] step {step + 1}/{steps} L_rec={float(loss):.5f}", flush=True)


def eval_losses(path: ReconPath, pool: List[Decision], condition: str,
                *, batch_size: int) -> np.ndarray:
    """Held-out per-decision L_rec [N] under 'live', 'foreign', or 'bypass'."""

    losses = []
    with torch.no_grad():
        for start in range(0, len(pool), batch_size):
            batch = pool[start : start + batch_size]
            states = [item.foreign if condition == "foreign" else item.live for item in batch]
            state = stack_states(states).to(device=path.device)
            targets = torch.stack([item.target for item in batch]).to(path.device)
            losses.extend(
                recon_losses(path, state, targets, bypass=condition == "bypass").tolist()
            )
    return np.asarray(losses, dtype=np.float64)


def paired_bootstrap_p(deltas: np.ndarray, *, draws: int, seed: int) -> float:
    """One-sided p for mean(delta) > 0 by paired resampling over decisions."""

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(draws, len(deltas)))
    return float((deltas[indices].mean(axis=1) <= 0.0).mean())


def unconditional_floor(train_pool: List[Decision], eval_pool: List[Decision],
                        *, max_train: int = 256) -> float:
    """L1 of the best constant prediction: eval loss of the train-set median."""

    median = (
        torch.stack([item.target.to(torch.float32) for item in train_pool[:max_train]])
        .median(dim=0)
        .values
    )
    return float(
        torch.stack(
            [(item.target.to(torch.float32) - median).abs().mean() for item in eval_pool]
        ).mean()
    )


def snapshot_parameters(path: ReconPath) -> List[torch.Tensor]:
    return [param.detach().clone() for param in path.trainable_parameters()]


def restore_parameters(path: ReconPath, snapshot: List[torch.Tensor]) -> None:
    with torch.no_grad():
        for param, saved in zip(path.trainable_parameters(), snapshot):
            param.copy_(saved)


def build_pools(caches: List[dict], trajectories, targets, *, eval_stride: int = 4):
    """Interleaved episode split -> (train, eval) Decision pools.

    Foreign states follow the replay-engine donor convention: nearest
    different-task episode, maturity-matched at ``min(d, len - 1)``.
    """

    eval_indices = [i for i in range(len(caches)) if i % eval_stride == 0]
    train_indices = [i for i in range(len(caches)) if i % eval_stride != 0]
    if not train_indices or not eval_indices:
        raise SystemExit("need at least two episodes to split train/eval pools")

    def decisions(cache_indices, local_caches):
        pool = []
        for local, index in enumerate(cache_indices):
            donor = pick_donor_index(local_caches, local)
            donor_states = trajectories[cache_indices[donor]] if donor is not None else None
            for d in range(len(targets[index])):
                foreign = (
                    donor_states[min(d, len(donor_states) - 1)]
                    if donor_states is not None
                    else trajectories[index][d]
                )
                pool.append(Decision(trajectories[index][d], foreign, targets[index][d]))
        return pool

    train_pool = decisions(train_indices, [caches[i] for i in train_indices])
    eval_pool = decisions(eval_indices, [caches[i] for i in eval_indices])
    return train_pool, eval_pool


def run_gate(path: ReconPath, train_pool: List[Decision], eval_pool: List[Decision],
             args) -> dict:
    """Both arms + verdicts; returns the result record (also printed)."""

    initial = snapshot_parameters(path)
    train_arm(path, train_pool, steps=args.steps, batch_size=args.batch_size,
              lr=args.lr, seed=args.seed, bypass=False)
    live = eval_losses(path, eval_pool, "live", batch_size=args.batch_size)
    foreign = eval_losses(path, eval_pool, "foreign", batch_size=args.batch_size)
    bypass_meter = eval_losses(path, eval_pool, "bypass", batch_size=args.batch_size)

    restore_parameters(path, initial)
    train_arm(path, train_pool, steps=args.steps, batch_size=args.batch_size,
              lr=args.lr, seed=args.seed, bypass=True)
    bypass_trained = eval_losses(path, eval_pool, "bypass", batch_size=args.batch_size)

    floor = unconditional_floor(train_pool, eval_pool)
    gap = float((foreign - live).mean())
    p_value = paired_bootstrap_p(
        foreign - live, draws=args.bootstrap, seed=stable_seed("g1-boot-v1", args.seed)
    )
    ratio = float(bypass_trained.mean() / max(floor, 1e-12))
    pass_a = gap > 0.0 and p_value < 0.05
    pass_b = ratio >= 1.0 - args.floor_tolerance

    result = {
        "eval_decisions": len(eval_pool),
        "train_decisions": len(train_pool),
        "l_rec_live": float(live.mean()),
        "l_rec_foreign": float(foreign.mean()),
        "foreign_gap": gap,
        "foreign_gap_p": p_value,
        "delta_bypass": float(bypass_meter.mean() - live.mean()),
        "l_rec_bypass_trained": float(bypass_trained.mean()),
        "unconditional_floor": floor,
        "bypass_floor_ratio": ratio,
        "pass_a": pass_a,
        "pass_b": pass_b,
    }
    print(
        f"held-out decisions={len(eval_pool)}  L_rec(live)={result['l_rec_live']:.5f}  "
        f"L_rec(foreign)={result['l_rec_foreign']:.5f}  "
        f"delta_bypass={result['delta_bypass']:+.5f}"
    )
    print("G1(a) PASS criterion: mean[L_rec(foreign) - L_rec(live)] > 0, "
          "one-sided paired bootstrap p < 0.05")
    print(f"G1(a) result: gap={gap:+.5f} p={p_value:.4f} -> {'PASS' if pass_a else 'FAIL'}")
    print(f"G1(b) PASS criterion: bypass-trained held-out L_rec >= "
          f"{1.0 - args.floor_tolerance:.2f} x unconditional floor (no mask leak)")
    print(
        f"G1(b) result: bypass={result['l_rec_bypass_trained']:.5f} floor={floor:.5f} "
        f"ratio={ratio:.3f} -> {'PASS' if pass_b else 'FAIL'}"
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"results -> {args.out}")
    return result


# --- real path ------------------------------------------------------------------


def _allowed_missing_prefixes(cfg) -> tuple:
    """The trainer's checkpoint-migration allowlist (train_vlajepa_cotrain.py)."""

    migration = cfg.trainer.get("checkpoint_migration", None)
    if not migration or not bool(migration.get("enabled", False)):
        return ()
    return tuple(migration.get("allow_missing_prefixes", []))


def teacher_latents(model, video: np.ndarray, device: torch.device) -> torch.Tensor:
    """Frozen vj_encoder targets [T, D] for one decision clip, per _compute_world_loss."""

    # video: [num_views, num_frames, H, W, C] -> per-view [num_frames, C, H, W]
    views = video.transpose(0, 1, 4, 2, 3)
    processed = [
        model.vj_processor(videos=views[i], return_tensors="pt")["pixel_values_videos"].to(device)
        for i in range(views.shape[0])
    ]
    autocast = (
        torch.autocast("cuda", torch.bfloat16) if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    with torch.no_grad(), autocast:
        embeddings = model.vj_encoder.get_vision_features(
            pixel_values_videos=torch.cat(processed, dim=0)
        )
        embeddings = torch.cat(torch.chunk(embeddings, chunks=views.shape[0], dim=0), dim=2)
    latent_frames = views.shape[1] // model.vj_encoder.config.tubelet_size
    tokens_per_frame = embeddings.shape[1] // latent_frames
    return embeddings[0, tokens_per_frame:].to(device="cpu", dtype=torch.float16)


def run(args) -> None:
    # Heavy imports stay out of --help/--smoke.
    from omegaconf import OmegaConf

    from starVLA.dataloader.gr00t_lerobot.datasets import (
        LeRobotMixtureDataset,
        SAMPLE_MODE_CONTIGUOUS_SEGMENT,
    )
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
    from starVLA.model.framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

    if not args.config or not args.cache_dir:
        raise SystemExit("--config and --cache-dir are required unless --smoke is set")
    torch.manual_seed(args.seed)
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    model = build_framework(cfg)
    if getattr(model, "memory_schema_version", 0) != 2:
        raise SystemExit("G1 requires a schema-2 config (framework.memory.schema_version: 2)")
    checkpoint = args.ckpt or cfg.trainer.get("pretrained_checkpoint", None)
    if checkpoint:
        TrainerUtils.load_pretrained_backbones(
            model, checkpoint, allowed_missing_prefixes=_allowed_missing_prefixes(cfg)
        )
    model = model.to(device).eval()
    model.memory_module.float()
    model.policy_memory_fusion.float()
    for param in model.parameters():
        param.requires_grad = False
    path = ReconPath(
        fusion=model.policy_memory_fusion,
        adapter=model.mem_cond_adapter,
        mask_token=model.wm_mask_token,
        predictor=model.vj_predictor,
        num_consumer_tokens=int(model.expected_embodied_token_count),
        consumer_dim=int(model.policy_memory_fusion.consumer_dim),
        device=device,
    )
    for param in path.trainable_parameters():
        param.requires_grad = True

    cache_paths = sorted(Path(args.cache_dir).glob("*.pt"))[: args.max_episodes]
    if not cache_paths:
        raise SystemExit(f"no episode caches under {args.cache_dir}")
    caches = [torch.load(cache_path, map_location="cpu") for cache_path in cache_paths]
    dataset_names = {cache["dataset"] for cache in caches}
    if len(dataset_names) != 1:
        raise SystemExit(f"cache dir must hold one dataset, got {sorted(dataset_names)}")

    dataset = make_LeRobotSingleDataset(
        Path(args.data_root),
        dataset_names.pop(),
        args.robot_type,
        delete_pause_frame=False,
        sample_mode=SAMPLE_MODE_CONTIGUOUS_SEGMENT,
        action_horizon=int(cfg.framework.action_model.action_horizon),
        video_horizon=int(cfg.framework.vj2_model.num_frames),
    )
    vla_data = cfg.datasets.vla_data
    mixture = LeRobotMixtureDataset(
        [(dataset, 1.0)],
        mode="val",
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        with_state=bool(vla_data.get("with_state", False)),
        resolution_size=int(vla_data.get("resolution_size", 224)),
        video_resolution_size=int(vla_data.get("video_resolution_size", 256)),
        seed=42,
        sample_mode=SAMPLE_MODE_CONTIGUOUS_SEGMENT,
        segment_length=int(vla_data.get("segment_length", 4)),
        burn_in_max_decisions=int(vla_data.get("burn_in_max_decisions", 8)),
        segment_stride=int(vla_data.get("segment_stride", 7)),
    )

    trajectories, targets = [], []
    with torch.no_grad():  # not inference_mode: states re-enter the autograd graph
        for cache in caches:
            states, _ = teacher_forced_states(
                model.memory_module, cache["action_tokens"].to(device)
            )
            trajectories.append([state.detach() for state in states])
            targets.append(
                [
                    teacher_latents(
                        model,
                        mixture._format_step(dataset, int(cache["episode"]), int(base))["video"],
                        device,
                    )
                    for base in cache["decision_bases"].tolist()
                ]
            )
            print(f"episode {int(cache['episode'])}: {len(targets[-1])} teacher targets", flush=True)

    train_pool, eval_pool = build_pools(caches, trajectories, targets)
    run_gate(path, train_pool, eval_pool, args)


# --- smoke path -----------------------------------------------------------------


class _TokenMixer(nn.Module):
    """Tiny stand-in for mem_cond_adapter: learned linear over the token dim."""

    def __init__(self, in_tokens: int, out_tokens: int):
        super().__init__()
        self.mix = nn.Linear(in_tokens, out_tokens)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mix(tokens.transpose(1, 2)).transpose(1, 2)


class _StubPredictor(nn.Module):
    """Frozen decoder: broadcasts a fixed projection of the conditioning mean."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, input_states: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        return input_states + self.proj(conditioning.mean(dim=1, keepdim=True))


def run_smoke(args) -> None:
    from starVLA.model.modules.memory.fusion import SparseKeyMemoryFusion
    from starVLA.model.modules.memory.recurrent_memory import RecurrentMemory

    try:
        from scripts.offline_eval.replay_engine import make_smoke_cache
    except ImportError:
        from replay_engine import make_smoke_cache

    torch.manual_seed(args.seed)
    dim, memory_dim, latents = 16, 16, 10
    memory = RecurrentMemory(
        source_dim=dim, memory_dim=memory_dim, num_slots=4, num_heads=2,
        use_keys=True, key_dim=8,
    ).float()
    for param in memory.parameters():
        param.requires_grad = False
    path = ReconPath(
        fusion=SparseKeyMemoryFusion(consumer_dim=dim, memory_dim=memory_dim, key_dim=8, num_slots=4),
        adapter=_TokenMixer(9, 6),  # 8 consumer tokens + 1 tap
        mask_token=nn.Parameter(torch.zeros(dim)),
        predictor=_StubPredictor(dim).requires_grad_(False),
        num_consumer_tokens=8,
        consumer_dim=dim,
        device=torch.device("cpu"),
    )

    caches = [make_smoke_cache(episode, 5 + episode % 4, seed=args.seed) for episode in range(8)]
    trajectories, targets = [], []
    with torch.no_grad():
        for cache in caches:
            states, _ = teacher_forced_states(memory, cache["action_tokens"])
            trajectories.append(states)
            # Episode-specific targets recoverable only from the written tokens.
            running = torch.cumsum(cache["action_tokens"].mean(dim=1), dim=0)
            running = running / torch.arange(1, len(running) + 1)[:, None]
            targets.append([
                running[d].repeat(latents, 1).to(torch.float16) for d in range(len(running))
            ])
    train_pool, eval_pool = build_pools(caches, trajectories, targets)
    result = run_gate(path, train_pool, eval_pool, args)
    print(f"smoke OK: {result['train_decisions']} train / {result['eval_decisions']} eval decisions")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=str, default=None, help="schema-2 training yaml")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="warm-start .pt (default: cfg.trainer.pretrained_checkpoint)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="one suite dir of cache_tokens.py episode .pt files")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--robot-type", type=str, default="libero_franka")
    parser.add_argument("--steps", type=int, default=500, help="train steps per arm")
    parser.add_argument("--batch-size", type=int, default=4, help="decisions per step")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--max-episodes", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--floor-tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default=None, help="optional JSON results path")
    parser.add_argument("--smoke", action="store_true",
                        help="run the full plumbing on tiny synthetic modules, CPU only")
    return parser


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    if parsed.smoke:
        run_smoke(parsed)
    else:
        run(parsed)
