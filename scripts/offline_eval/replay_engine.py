"""Teacher-forced offline replay engine (plan T0.2/T0.6/T0.7, hook H7).

Loads a checkpoint plus the per-episode Qwen token caches written by
``scripts/offline_eval/cache_tokens.py``, teacher-forces ``memory_module.write``
over each episode, and scores every decision under CLI-selected conditions with
N pre-sampled ``initial_noise`` draws that are shared bit-identically across all
conditions.  Emits one record per (episode, decision, condition, J, lambda) with
the pinned column schema

    {dataset, suite, episode, decision, condition, J, lam, mse, tf_loss, working_norm}

as parquet when pandas/pyarrow import, else gzipped JSONL.  ``J = -1`` and
``lam = 1.0`` mean "not applicable".  Conditions:

    live            state after all same-episode writes before the decision
    bypass          exact fusion skip (no read, no injection)
    prior           learned initial slots, never written (zero-as-served)
    shuffled        foreign-episode state at matched decision index
    frozen_after=J  writes stop after the first J decisions
    burnin_<order>  fresh J-decision write window per decision; order in
                    {forward, shuffled, reversed} (T0.2b)

The lambda sweep (T0.2c) drives ``fusion.residual_scale`` (hook H2) over the
{live, shuffled} conditions.  ``--state-dump`` additionally writes per-episode
npz shards (states [D,8,512] fp32, write/read diagnostics from hook H3, labels)
for ``scripts/probe/fit_probes.py``.  ``--smoke`` runs the full plumbing on
synthetic tensors with the stub modules below; the unit tests reuse the same
stubs (tests/memory/test_replay_schedule.py).
"""

import argparse
import gzip
import hashlib
import inspect
import json
from contextlib import nullcontext
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np
import torch

from starVLA.model.modules.memory.state import MemoryRead, MemoryState

RECORD_COLUMNS = (
    "dataset", "suite", "episode", "decision",
    "condition", "J", "lam", "mse", "tf_loss", "working_norm",
)
BASE_CONDITIONS = ("live", "bypass", "prior", "shuffled")
BURN_IN_ORDERS = ("forward", "shuffled", "reversed")


class PlanRow(NamedTuple):
    condition: str
    J: int = -1
    lam: float = 1.0


def stable_seed(*parts) -> int:
    """Deterministic 63-bit seed from a tuple, in the datasets.safe_hash style.

    63 bits keeps the value inside torch.Generator.manual_seed's int64 range.
    """

    digest = hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()
    return int(digest, 16) & 0x7FFFFFFFFFFFFFFF


def burn_in_indices(decision: int, J: int, order: str, rng=None) -> List[int]:
    """Decision indices written before scoring ``decision`` under a J-window burn-in."""

    if J < 0:
        raise ValueError("J must be non-negative")
    window = list(range(max(0, decision - J), decision))
    if order == "forward":
        return window
    if order == "reversed":
        return window[::-1]
    if order == "shuffled":
        if rng is None:
            raise ValueError("shuffled order requires an rng")
        return [window[i] for i in rng.permutation(len(window))]
    raise ValueError(f"unknown burn-in order {order!r}")


def parse_conditions(spec: str) -> List[PlanRow]:
    rows = []
    for token in filter(None, spec.split(",")):
        if token.startswith("frozen_after="):
            rows.append(PlanRow("frozen_after", J=int(token.split("=", 1)[1])))
        elif token in BASE_CONDITIONS:
            rows.append(PlanRow(token))
        else:
            raise ValueError(f"unknown condition {token!r}")
    return rows


def build_plan(conditions: str, burnin: str, burnin_orders: str, lam: str, lam_conditions: str) -> List[PlanRow]:
    plan = parse_conditions(conditions)
    orders = list(filter(None, burnin_orders.split(",")))
    for order in orders:
        if order not in BURN_IN_ORDERS:
            raise ValueError(f"unknown burn-in order {order!r}")
    for J in (int(v) for v in filter(None, burnin.split(","))):
        plan.extend(PlanRow(f"burnin_{order}", J=J) for order in orders)
    for value in (float(v) for v in filter(None, lam.split(","))):
        for condition in filter(None, lam_conditions.split(",")):
            if condition not in ("live", "shuffled"):
                raise ValueError(f"lambda sweep supports live/shuffled, got {condition!r}")
            plan.append(PlanRow(condition, lam=value))
    seen, unique = set(), []
    for row in plan:
        if row not in seen:
            seen.add(row)
            unique.append(row)
    return unique


def teacher_forced_states(memory, action_tokens: torch.Tensor, *, capture: bool = False):
    """Full write trajectory: ``states[d]`` is the state read at decision ``d``."""

    if capture:
        memory.capture_diagnostics = True
    state = memory.init_state(1, device=action_tokens.device)
    states, write_diags = [state], []
    for d in range(action_tokens.shape[0]):
        state = memory.write(action_tokens[d : d + 1], state)
        states.append(state)
        if capture:
            diag = getattr(memory, "last_write_diagnostics", None)
            if diag is None:
                raise RuntimeError(
                    "capture_diagnostics set but write() left no last_write_diagnostics; "
                    "land hook H3 before --state-dump runs"
                )
            write_diags.append(diag)
    return states, write_diags


def condition_state(
    row: PlanRow,
    decision: int,
    states: List[MemoryState],
    donor_states: Optional[List[MemoryState]],
    memory,
    action_tokens: torch.Tensor,
    *,
    episode_seed: int,
) -> Optional[MemoryState]:
    """State fed to the read for one (condition, decision); None means exact bypass."""

    if row.condition == "bypass":
        return None
    if row.condition == "live":
        return states[decision]
    if row.condition == "prior":
        return states[0]
    if row.condition == "frozen_after":
        return states[min(decision, row.J)]
    if row.condition == "shuffled":
        return donor_states[min(decision, len(donor_states) - 1)]
    if row.condition.startswith("burnin_"):
        order = row.condition[len("burnin_"):]
        rng = (
            np.random.default_rng(stable_seed("burnin-shuffle-v1", episode_seed, decision, row.J))
            if order == "shuffled"
            else None
        )
        state = memory.init_state(1, device=action_tokens.device)
        for index in burn_in_indices(decision, row.J, order, rng):
            state = memory.write(action_tokens[index : index + 1], state)
        return state
    raise ValueError(f"unknown condition {row.condition!r}")


def teacher_forced_flow_loss(head, vl_embs, actions, t, noise) -> float:
    """Deterministic flow-matching loss with pinned (t, noise) draws.

    Prefers the head's seeded pass-through kwargs (hook H2); otherwise replicates
    ``FlowmatchingActionHead.forward`` exactly with the pinned values injected,
    so the metric is identical either way.
    """

    params = inspect.signature(head.forward).parameters
    if "t" in params and "noise" in params:
        return float(head(vl_embs, actions, t=t, noise=noise))
    t = t[:, None, None]
    noisy_trajectory = (1 - t) * noise + t * actions
    velocity = actions - noise
    t_discretized = (t[:, 0, 0] * head.num_timestep_buckets).long()
    action_features = head.action_encoder(noisy_trajectory, t_discretized)
    if head.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=vl_embs.device)
        action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
    future_tokens = head.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
    sa_embs = torch.cat((future_tokens, action_features), dim=1)
    model_output = head.model(
        hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=t_discretized
    )
    pred_actions = head.action_decoder(model_output)[:, -actions.shape[1]:]
    return float(((pred_actions - velocity) ** 2).mean())


def replay_episode(
    memory,
    fusion,
    head,
    cache: dict,
    plan: List[PlanRow],
    states: List[MemoryState],
    *,
    episode_seed: int,
    noise_draws: int = 8,
    donor_states: Optional[List[MemoryState]] = None,
) -> List[dict]:
    """Score every (decision, plan row) of one episode with shared noise draws."""

    action_tokens = cache["action_tokens"]
    embodied = cache["embodied_action_tokens"]
    gt_actions = cache["gt_actions"].to(torch.float32)
    num_decisions = action_tokens.shape[0]
    device = embodied.device

    # Every stochastic input is pre-sampled once per episode and reused
    # bit-identically by all conditions; t is a fixed stratified grid.
    generator = torch.Generator(device="cpu").manual_seed(episode_seed)
    chunk_shape = tuple(gt_actions.shape[1:])
    noise = torch.randn((num_decisions, noise_draws) + chunk_shape, generator=generator).to(device)
    tf_noise = torch.randn((num_decisions, noise_draws) + chunk_shape, generator=generator).to(device)
    t_grid = ((torch.arange(noise_draws, dtype=torch.float32) + 0.5) / noise_draws).to(device)

    autocast = (
        torch.autocast("cuda", torch.bfloat16) if device.type == "cuda" else nullcontext()
    )
    records = []
    for row in plan:
        if row.condition == "shuffled" and donor_states is None:
            continue
        if row.lam != 1.0 and not hasattr(fusion, "residual_scale"):
            raise RuntimeError("fusion.residual_scale missing; land hook H2 before lambda sweeps")
        previous_scale = getattr(fusion, "residual_scale", 1.0)
        if hasattr(fusion, "residual_scale"):
            fusion.residual_scale = row.lam
        try:
            for decision in range(num_decisions):
                state = condition_state(
                    row, decision, states, donor_states, memory, action_tokens,
                    episode_seed=episode_seed,
                )
                if state is None:
                    fused = embodied[decision : decision + 1]
                    working_norm = float("nan")
                else:
                    read = memory.read(action_tokens[decision : decision + 1], state)
                    working_norm = float(read.diagnostics["working_norm"].to(torch.float32).mean())
                    fused = fusion(embodied[decision : decision + 1], read.tokens)
                fused = fused.expand(noise_draws, -1, -1)
                gt = gt_actions[decision]
                with autocast:
                    pred = head.predict_action(fused, None, initial_noise=noise[decision])
                    tf_loss = teacher_forced_flow_loss(
                        head,
                        fused,
                        gt.unsqueeze(0).expand(noise_draws, -1, -1).to(fused.dtype),
                        t_grid.to(fused.dtype),
                        tf_noise[decision].to(fused.dtype),
                    )
                mse = float(((pred.to(torch.float32) - gt.unsqueeze(0)) ** 2).mean())
                records.append(
                    {
                        "dataset": cache["dataset"],
                        "suite": cache["suite"],
                        "episode": int(cache["episode"]),
                        "decision": decision,
                        "condition": row.condition,
                        "J": row.J,
                        "lam": row.lam,
                        "mse": mse,
                        "tf_loss": tf_loss,
                        "working_norm": working_norm,
                    }
                )
        finally:
            if hasattr(fusion, "residual_scale"):
                fusion.residual_scale = previous_scale
    return records


def build_state_dump(memory, cache: dict, states: List[MemoryState], write_diags: List[dict]) -> dict:
    """Per-episode npz payload for the probe frontend (plan T0.6)."""

    num_decisions = cache["action_tokens"].shape[0]
    dump = {
        "states": np.stack(
            [states[d].working[0].to(torch.float32).cpu().numpy() for d in range(num_decisions)]
        ),
        "token_mean": cache["action_tokens"].to(torch.float32).mean(dim=1).cpu().numpy(),
        # gripper channel = last action dim (binarized at 0.5, base_framework rule)
        "gt_gripper": cache["gt_actions"][:, :, -1].to(torch.float32).cpu().numpy(),
        "decision": np.arange(num_decisions, dtype=np.int64),
        "progress": np.arange(num_decisions, dtype=np.float64) / max(num_decisions - 1, 1),
        "initial_slots": states[0].working[0].to(torch.float32).cpu().numpy(),
        "episode": np.int64(cache["episode"]),
        "suite": np.str_(cache["suite"]),
        "dataset": np.str_(cache["dataset"]),
        "task": np.str_(cache["lang"]),
    }
    for key in write_diags[0] if write_diags else ():
        dump[f"write_{key}"] = np.stack(
            [np.asarray(torch.as_tensor(diag[key]).to(torch.float32).cpu()) for diag in write_diags]
        )
    read_attentions = []
    for d in range(num_decisions):
        memory.read(cache["action_tokens"][d : d + 1], states[d])
        read_diag = getattr(memory, "last_read_diagnostics", None)
        # RecurrentMemory keeps the key with a None value (the read itself is
        # attention-free); dump the map only when a module actually produces one.
        if read_diag is None or read_diag.get("read_attention") is None:
            break
        read_attentions.append(
            np.asarray(torch.as_tensor(read_diag["read_attention"]).to(torch.float32).cpu())
        )
    if len(read_attentions) == num_decisions:
        dump["read_attention"] = np.stack(read_attentions)
    return dump


def write_records(records: List[dict], out_stem: Path) -> Path:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        path = out_stem.with_suffix(".parquet")
        pd.DataFrame(records, columns=list(RECORD_COLUMNS)).to_parquet(path, index=False)
    except ImportError:
        path = out_stem.with_suffix(".jsonl.gz")
        with gzip.open(path, "wt") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    return path


# --- stub modules: used by --smoke and by tests/memory/test_replay_schedule.py ----


class SmokeMemory:
    """Order-sensitive affine stand-in exposing the RecurrentMemory interface."""

    def __init__(self, num_slots: int = 2, memory_dim: int = 4):
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.capture_diagnostics = False
        self.last_write_diagnostics = None
        self.last_read_diagnostics = None

    def init_state(self, batch_size: int, device) -> MemoryState:
        return MemoryState(
            working=torch.ones(batch_size, self.num_slots, self.memory_dim, dtype=torch.float32),
            episodic=None,
            steps=torch.zeros(batch_size, dtype=torch.int64),
            valid=torch.ones(batch_size, dtype=torch.bool),
        )

    def read(self, source_tokens, state, read_mask=None) -> MemoryRead:
        if self.capture_diagnostics:
            self.last_read_diagnostics = {
                "read_attention": torch.zeros(state.batch_size, 1, self.num_slots)
            }
        return MemoryRead(
            tokens=state.working.clone(),
            diagnostics={
                "working_norm": state.working.norm(dim=-1).mean(dim=-1),
                "steps": state.steps.to(torch.float32),
                "active": state.valid.to(torch.float32),
            },
        )

    def write(self, source_tokens, state, update_mask=None) -> MemoryState:
        summary = source_tokens.to(torch.float32).mean(dim=1)[:, : self.memory_dim]
        working = 0.5 * state.working + summary[:, None, :]
        if self.capture_diagnostics:
            self.last_write_diagnostics = {
                "update_gate_mean": 0.5,
                "update_gate_p05": 0.05,
                "update_gate_p95": 0.95,
                "per_slot_delta_norm": (working - state.working).norm(dim=-1)[0],
                "slot_cosine_mean": 1.0,
                "write_attention": torch.zeros(
                    state.batch_size, self.num_slots, source_tokens.shape[1]
                ),
            }
        return MemoryState(
            working=working, episodic=None, steps=state.steps + 1, valid=state.valid
        )


class SmokeFusion:
    """Additive stand-in recording read tokens; honors residual_scale (hook H2)."""

    def __init__(self):
        self.residual_scale = 1.0
        self.calls = []  # (memory_tokens, residual_scale) per invocation
        self.last_fusion_diagnostics = None

    def __call__(self, consumer_tokens, memory_tokens):
        self.calls.append((memory_tokens.detach().clone(), float(self.residual_scale)))
        consumer = consumer_tokens.to(torch.float32)
        residual = memory_tokens.to(torch.float32).mean() * torch.ones_like(consumer)
        self.last_fusion_diagnostics = {
            "injection_ratio": float(
                (self.residual_scale * residual).norm() / consumer.norm().clamp_min(1e-12)
            )
        }
        return (consumer + self.residual_scale * residual).to(consumer_tokens.dtype)


class SmokeHead:
    """Deterministic stand-in recording every noise draw it consumes."""

    def __init__(self):
        self.predict_calls = []  # (vl_embs, initial_noise) per invocation
        self.forward_calls = []  # (t, noise) per invocation

    def predict_action(self, vl_embs, state=None, generator=None, initial_noise=None):
        self.predict_calls.append((vl_embs.detach().clone(), initial_noise.detach().clone()))
        return initial_noise + vl_embs.to(torch.float32).mean()

    def forward(self, vl_embs, actions, state=None, t=None, noise=None):
        self.forward_calls.append((t.detach().clone(), noise.detach().clone()))
        velocity = actions - noise
        return ((velocity - vl_embs.to(torch.float32).mean()) ** 2).mean()

    __call__ = forward


def make_smoke_cache(episode: int, num_decisions: int, *, token_dim: int = 16, seed: int = 0) -> dict:
    rng = np.random.default_rng(stable_seed("smoke-cache-v1", seed, episode))
    return {
        "suite": "smoke",
        "dataset": "smoke_dataset",
        "episode": episode,
        "lang": f"smoke task {episode % 2}",
        "action_tokens": torch.as_tensor(
            rng.standard_normal((num_decisions, 24, token_dim)), dtype=torch.float32
        ),
        "embodied_action_tokens": torch.as_tensor(
            rng.standard_normal((num_decisions, 32, token_dim)), dtype=torch.float32
        ),
        "gt_actions": torch.as_tensor(
            rng.standard_normal((num_decisions, 7, 7)), dtype=torch.float32
        ),
    }


# --- CLI ----------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=str, default=None, help="exported .pt checkpoint")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="directory of cache_tokens.py episode .pt files")
    parser.add_argument("--out", type=str, required=True,
                        help="output stem; .parquet or .jsonl.gz suffix added")
    parser.add_argument("--conditions", type=str,
                        default="live,bypass,prior,shuffled,frozen_after=8")
    parser.add_argument("--burnin", type=str, default="0,1,2,4,8,16,32",
                        help="comma-separated J values; empty disables the sweep")
    parser.add_argument("--burnin-orders", type=str, default="forward,shuffled,reversed")
    parser.add_argument("--lam", type=str, default="0,0.25,0.5,1,2,4,8",
                        help="fusion.residual_scale values; empty disables the sweep")
    parser.add_argument("--lam-conditions", type=str, default="live,shuffled")
    parser.add_argument("--noise-draws", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--state-dump", type=str, default=None,
                        help="directory for per-episode npz probe shards (hook H3)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true",
                        help="run the full plumbing on synthetic tensors, no checkpoint")
    return parser


def _replay_all(memory, fusion, head, caches, plan, args, device) -> List[dict]:
    capture = args.state_dump is not None
    dump_dir = Path(args.state_dump) if capture else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    trajectories, diagnostics = [], []
    for cache in caches:
        cache["action_tokens"] = cache["action_tokens"].to(device)
        cache["embodied_action_tokens"] = cache["embodied_action_tokens"].to(device)
        cache["gt_actions"] = cache["gt_actions"].to(device)
        states, write_diags = teacher_forced_states(
            memory, cache["action_tokens"], capture=capture
        )
        trajectories.append(states)
        diagnostics.append(write_diags)

    records = []
    for index, cache in enumerate(caches):
        donor_states = trajectories[(index - 1) % len(caches)] if len(caches) > 1 else None
        episode_seed = stable_seed(
            "replay-noise-v1", args.seed, cache["suite"], cache["dataset"], cache["episode"]
        )
        records.extend(
            replay_episode(
                memory, fusion, head, cache, plan, trajectories[index],
                episode_seed=episode_seed,
                noise_draws=args.noise_draws,
                donor_states=donor_states,
            )
        )
        if dump_dir is not None:
            dump = build_state_dump(memory, cache, trajectories[index], diagnostics[index])
            np.savez_compressed(
                dump_dir / f"{cache['suite']}__{cache['dataset']}__ep{int(cache['episode']):06d}.npz",
                **dump,
            )
    return records


def run(args) -> None:
    from starVLA.model.framework.base_framework import baseframework

    if not args.ckpt or not args.cache_dir:
        raise SystemExit("--ckpt and --cache-dir are required unless --smoke is set")
    device = torch.device(args.device)
    model = baseframework.from_pretrained(args.ckpt).to(device).eval()
    if not getattr(model, "memory_enabled", False):
        raise SystemExit("replay requires a memory-enabled checkpoint")
    model.memory_module.float()
    model.policy_memory_fusion.float()

    cache_paths = sorted(Path(args.cache_dir).glob("*.pt"))
    if args.max_episodes is not None:
        cache_paths = cache_paths[: args.max_episodes]
    if not cache_paths:
        raise SystemExit(f"no episode caches under {args.cache_dir}")
    caches = [torch.load(path, map_location="cpu") for path in cache_paths]

    plan = build_plan(args.conditions, args.burnin, args.burnin_orders, args.lam, args.lam_conditions)
    with torch.inference_mode():
        records = _replay_all(
            model.memory_module, model.policy_memory_fusion, model.action_model,
            caches, plan, args, device,
        )
    path = write_records(records, Path(args.out))
    print(f"{len(records)} records ({len(caches)} episodes, {len(plan)} plan rows) -> {path}")


def run_smoke(args) -> None:
    device = torch.device("cpu")
    memory, fusion, head = SmokeMemory(), SmokeFusion(), SmokeHead()
    caches = [make_smoke_cache(episode, num_decisions, seed=args.seed)
              for episode, num_decisions in enumerate((5, 9, 3))]
    plan = build_plan(args.conditions, args.burnin, args.burnin_orders, args.lam, args.lam_conditions)
    records = _replay_all(memory, fusion, head, caches, plan, args, device)
    path = write_records(records, Path(args.out))
    print(f"smoke OK: {len(records)} records ({len(plan)} plan rows) -> {path}")
    if args.state_dump:
        print(f"smoke state dumps under {args.state_dump}")


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    if parsed.smoke:
        run_smoke(parsed)
    else:
        run(parsed)
