"""memv2 forward-sequence content-read discriminator (harness-mismatch-free).

For each sampled training segment (built exactly like the trainer's
contiguous-segment VLA dataset), run the model's own ``forward_sequence``
twice under no-grad bf16 autocast:

  * live    -- the segment exactly as sampled;
  * foreign -- the burn-in prefix (steps AND the per-timestep control arrays
               ``base_indices``/``sequence_valid``/``update_mask``/``is_first``/
               ``is_last``) spliced in from a segment of a DIFFERENT episode of
               the same dataset with the SAME number of valid burn-in
               decisions (maturity-matched).  Everything downstream -- the
               supervised decisions, loss_mask, segment_start, mask_plan, and
               the masked step's clean reconstruction target -- is identical.

Every segment carries exactly one masked (blind) supervised decision at
supervised offset 1: the dataset's own mask plan with
``memory_mask_rate=1.0, memory_mask_max_per_segment=1,
memory_mask_ramp_samples=0`` (rate 1.0 fires the first eligible position and
the per-segment cap blocks the rest; position 0 is never masked by
construction).  ``rec_loss``/``action_loss`` from the returned dicts are
compared pairwise; the torch RNG is reseeded identically before both passes
so the flow-matching noise draws pair up.

Foreign - live > 0 on rec/action means the policy read consumes stored
burn-in *content* through the very code path the trainer optimizes.

Validation without a GPU/checkpoint: ``--limit N`` loads the dataset only,
prints one segment's structure, and verifies the splice construction.
"""

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_DATASETS = (
    "mikasa_shell_game_shuffle_touch_vla_v0,"
    "mikasa_take_it_back_vla_v0,"
    "mikasa_remember_color_9_long_vla_v0,"
    "mikasa_chain_of_colors_7_vla_v0"
)

# Per-timestep control arrays that must travel WITH the burn-in steps.
BURN_IN_ARRAY_KEYS = ("base_indices", "sequence_valid", "update_mask", "is_first", "is_last")


def stable_seed(*items) -> int:
    digest = hashlib.sha256(repr(items).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# --------------------------------------------------------------------- dataset


def build_mixture(cfg, data_root: str, dataset_name: str, robot_type: str):
    """One single-dataset mixture built with the trainer's exact parameters.

    Mirrors ``build_dataloader('lerobot_datasets')`` -> ``get_vla_dataset`` ->
    ``make_LeRobotSingleDataset``/``LeRobotMixtureDataset`` for one dataset
    (weight 1.0), then sets the blind-decision mask attributes the trainer
    pushes in ``VLAMTrainer._configure_mask_schedule`` -- pinned so EVERY
    segment has exactly one masked supervised decision, deterministically.
    """
    from starVLA.dataloader.gr00t_lerobot.datasets import (
        LeRobotMixtureDataset,
        SAMPLE_MODE_CONTIGUOUS_SEGMENT,
    )
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

    vla = cfg.datasets.vla_data
    sample_mode = str(vla.get("sample_mode", "single_step"))
    if sample_mode != SAMPLE_MODE_CONTIGUOUS_SEGMENT:
        raise SystemExit(f"config sample_mode={sample_mode!r}; this test needs contiguous_segment")

    single = make_LeRobotSingleDataset(
        Path(data_root),
        dataset_name,
        robot_type,
        delete_pause_frame=bool(vla.get("delete_pause_frame", True)),
        sample_mode=sample_mode,
        action_horizon=int(cfg.framework.action_model.action_horizon),
        video_horizon=int(cfg.framework.vj2_model.num_frames),
    )
    mixture = LeRobotMixtureDataset(
        [(single, 1.0)],
        mode="val",  # epoch-independent deterministic sampling per index
        balance_dataset_weights=False,
        balance_trajectory_weights=False,
        with_state=bool(vla.get("with_state", False)),
        resolution_size=int(vla.get("resolution_size", 224)),
        video_resolution_size=int(vla.get("video_resolution_size", 256)),
        seed=42,  # the trainer's get_vla_dataset default
        sample_mode=sample_mode,
        segment_length=int(vla.get("segment_length", 4)),
        burn_in_max_decisions=int(vla.get("burn_in_max_decisions", 8)),
        segment_stride=int(vla.get("segment_stride", cfg.framework.action_model.action_horizon)),
    )
    # sample_segment consumes these instance attributes (class defaults are the
    # no-mask memv1 behaviour).  rate=1.0 + cap=1 + ramp=0  =>  mask_plan is
    # all-False except supervised position 1, for every index.
    mixture.memory_mask_rate = 1.0
    mixture.memory_mask_max_per_segment = 1
    mixture.memory_mask_ramp_samples = 0
    return mixture


def sample_pool(mixture, burn_in: int, min_burn_in: int, target: int, rng) -> list:
    """Draw deterministic segments until ``target`` usable candidates exist."""
    total = len(mixture)
    order = rng.choice(total, size=min(total, max(target * 24, 64)), replace=False)
    pool, skipped_short, failures = [], 0, 0
    for index in order:
        if len(pool) >= target:
            break
        try:
            segment = mixture[int(index)]
        except Exception as exc:  # decoder retries already exhausted inside
            failures += 1
            print(f"  [skip] index {int(index)}: {type(exc).__name__}: {exc}", flush=True)
            continue
        valid = np.asarray(segment["sequence_valid"], dtype=bool)
        burn_count = int(valid[:burn_in].sum())
        if burn_count < min_burn_in:
            skipped_short += 1
            continue
        plan = np.asarray(segment["mask_plan"], dtype=bool)
        if int(plan.sum()) != 1 or not bool(plan[1]):
            raise AssertionError(f"unexpected mask plan {plan.tolist()} at index {int(index)}")
        pool.append(
            {
                "index": int(index),
                "segment": segment,
                "burn_count": burn_count,
                "episode_id": int(segment["episode_id"]),
            }
        )
    print(
        f"  pool: {len(pool)} usable segments "
        f"({skipped_short} below min burn-in {min_burn_in}, {failures} decode failures)",
        flush=True,
    )
    return pool


def form_pairs(pool: list, quota: int) -> list:
    """Maturity-matched (equal burn_count) live/donor pairs, donor episode differs."""
    groups = defaultdict(list)
    for entry in pool:
        groups[entry["burn_count"]].append(entry)
    pairs = []
    for burn_count in sorted(groups, reverse=True):  # prefer the longest burn-ins
        group = groups[burn_count]
        size = len(group)
        for i, live in enumerate(group):
            if len(pairs) >= quota:
                return pairs
            donor = next(
                (
                    group[(i + off) % size]
                    for off in range(1, size)
                    if group[(i + off) % size]["episode_id"] != live["episode_id"]
                ),
                None,
            )
            if donor is not None:
                pairs.append((live, donor))
    return pairs


def splice_foreign(live_segment: dict, donor_segment: dict, burn_in: int) -> dict:
    """Foreign twin: donor's burn-in prefix, live's supervised suffix.

    The whole per-timestep step dicts are swapped, so image/video/lang/action
    (and proprio ``state`` when with_state is on) always move together; the
    burn-in slices of the control arrays are swapped with them so validity,
    write gating, and episode-start resets stay consistent with the donor
    prefix.  loss_mask/segment_start/mask_plan are identical across segments
    by construction and stay live's.
    """
    twin = dict(live_segment)
    twin["steps"] = list(donor_segment["steps"][:burn_in]) + list(live_segment["steps"][burn_in:])
    for key in BURN_IN_ARRAY_KEYS:
        twin[key] = np.concatenate(
            (
                np.asarray(donor_segment[key][:burn_in]),
                np.asarray(live_segment[key][burn_in:]),
            )
        )
    return twin


# --------------------------------------------------------------------- model


def load_model(cfg, ckpt: str, device):
    import torch
    from starVLA.model.framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

    model = build_framework(cfg)
    if getattr(model, "memory_schema_version", 0) < 2:
        raise SystemExit("this discriminator requires framework.memory.schema_version >= 2")
    migration = cfg.trainer.get("checkpoint_migration", None)
    allowed = (
        tuple(migration.get("allow_missing_prefixes", []))
        if migration and bool(migration.get("enabled", False))
        else ()
    )
    TrainerUtils.load_pretrained_backbones(model, ckpt, allowed_missing_prefixes=allowed)
    return model.to(device).eval()


def run_pair(model, live_segment: dict, foreign_segment: dict, seg_seed: int, device) -> dict:
    """Both forward_sequence passes, identically seeded; returns the 4 losses."""
    import torch
    from contextlib import nullcontext

    record = {}
    for tag, segment in (("live", live_segment), ("foreign", foreign_segment)):
        torch.manual_seed(seg_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seg_seed)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            output = model.forward_sequence([segment])
        if "rec_loss" not in output or "action_loss" not in output:
            raise AssertionError(f"missing losses in forward_sequence output: {sorted(output)}")
        record[f"rec_{tag}"] = float(output["rec_loss"])
        record[f"act_{tag}"] = float(output["action_loss"])
    record["delta_rec"] = record["rec_foreign"] - record["rec_live"]
    record["delta_act"] = record["act_foreign"] - record["act_live"]
    return record


# --------------------------------------------------------------------- stats


def bootstrap_stats(deltas: np.ndarray, draws: int, seed: int) -> dict:
    """Mean, percentile 95% CI, and one-sided p for mean(delta) > 0."""
    deltas = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(draws, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return {
        "n": int(len(deltas)),
        "mean": float(deltas.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "p_one_sided": float((means <= 0.0).mean()),
    }


# --------------------------------------------------------------------- modes


def run(args) -> None:
    import torch
    from omegaconf import OmegaConf

    if not args.ckpt:
        raise SystemExit("--ckpt is required (use --limit for the model-free dataset check)")
    cfg = OmegaConf.load(args.config)
    data_root = args.data_root or str(cfg.datasets.vla_data.data_root_dir)
    burn_in = int(cfg.datasets.vla_data.get("burn_in_max_decisions", 8))
    device = torch.device(args.device)
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    quota = int(math.ceil(args.num_segments / len(dataset_names)))

    print(f"loading model: config={args.config} ckpt={args.ckpt}", flush=True)
    model = load_model(cfg, args.ckpt, device)

    records = []
    for dataset_name in dataset_names:
        if len(records) >= args.num_segments:
            break
        print(f"[{dataset_name}] building dataset ...", flush=True)
        mixture = build_mixture(cfg, data_root, dataset_name, args.robot_type)
        pool_rng = np.random.default_rng(stable_seed("fwdseq-pool-v1", args.seed, dataset_name))
        pool = sample_pool(mixture, burn_in, args.min_burn_in, target=quota * 2, rng=pool_rng)
        pairs = form_pairs(pool, min(quota, args.num_segments - len(records)))
        print(f"  paired {len(pairs)} live/donor segments", flush=True)
        for live, donor in pairs:
            foreign = splice_foreign(live["segment"], donor["segment"], burn_in)
            seg_seed = stable_seed("fwdseq-fwd-v1", args.seed, dataset_name, live["index"])
            started = time.perf_counter()
            record = run_pair(model, live["segment"], foreign, seg_seed, device)
            record.update(
                dataset=dataset_name,
                index=live["index"],
                episode_id=live["episode_id"],
                donor_episode_id=donor["episode_id"],
                burn_count=live["burn_count"],
            )
            records.append(record)
            print(
                f"  [{len(records):3d}] ep={record['episode_id']:<5d} "
                f"donor_ep={record['donor_episode_id']:<5d} burn={record['burn_count']} "
                f"d_rec={record['delta_rec']:+.5f} d_act={record['delta_act']:+.5f} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )
        del pool, pairs, mixture
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not records:
        raise SystemExit("no pairable segments were found; lower --min-burn-in or change --datasets")

    delta_rec = np.asarray([record["delta_rec"] for record in records])
    delta_act = np.asarray([record["delta_act"] for record in records])
    rec_stats = bootstrap_stats(delta_rec, args.bootstrap, stable_seed("fwdseq-boot-rec", args.seed))
    act_stats = bootstrap_stats(delta_act, args.bootstrap, stable_seed("fwdseq-boot-act", args.seed))

    per_dataset = {}
    for dataset_name in sorted({record["dataset"] for record in records}):
        rows = [record for record in records if record["dataset"] == dataset_name]
        per_dataset[dataset_name] = {
            "n": len(rows),
            "gap_rec": float(np.mean([row["delta_rec"] for row in rows])),
            "gap_act": float(np.mean([row["delta_act"] for row in rows])),
            "mean_burn_count": float(np.mean([row["burn_count"] for row in rows])),
        }

    result = {
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": args.seed,
        "min_burn_in": args.min_burn_in,
        "n": len(records),
        "rec": rec_stats,
        "act": act_stats,
        "per_dataset": per_dataset,
        "pairs": records,
    }
    print("\nper-dataset breakdown:")
    for dataset_name, row in per_dataset.items():
        print(
            f"  {dataset_name}: n={row['n']} gap_rec={row['gap_rec']:+.5f} "
            f"gap_act={row['gap_act']:+.5f} mean_burn={row['mean_burn_count']:.1f}"
        )
    print(
        f"rec: mean={rec_stats['mean']:+.5f} ci95=[{rec_stats['ci95'][0]:+.5f}, "
        f"{rec_stats['ci95'][1]:+.5f}] p={rec_stats['p_one_sided']:.4f}"
    )
    print(
        f"act: mean={act_stats['mean']:+.5f} ci95=[{act_stats['ci95'][0]:+.5f}, "
        f"{act_stats['ci95'][1]:+.5f}] p={act_stats['p_one_sided']:.4f}"
    )
    print(
        f"FWDSEQ_DISC gap_rec={rec_stats['mean']:+.5f} p={rec_stats['p_one_sided']:.4f} "
        f"gap_act={act_stats['mean']:+.5f} p={act_stats['p_one_sided']:.4f} n={len(records)}"
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"results -> {args.out}")


def describe_array(value) -> str:
    array = np.asarray(value)
    return f"ndarray shape={array.shape} dtype={array.dtype}"


def run_limit(args) -> None:
    """Model-free validation: dataset plumbing, sample structure, splice check."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    data_root = args.data_root or str(cfg.datasets.vla_data.data_root_dir)
    burn_in = int(cfg.datasets.vla_data.get("burn_in_max_decisions", 8))
    dataset_name = [name.strip() for name in args.datasets.split(",") if name.strip()][0]
    print(f"[limit] dataset={dataset_name} data_root={data_root} burn_in={burn_in}")
    mixture = build_mixture(cfg, data_root, dataset_name, args.robot_type)
    print(f"[limit] mixture length (segment starts): {len(mixture)}")

    rng = np.random.default_rng(stable_seed("fwdseq-pool-v1", args.seed, dataset_name))
    pool = sample_pool(mixture, burn_in, args.min_burn_in, target=args.limit, rng=rng)
    if not pool:
        raise SystemExit("no usable segments; lower --min-burn-in or pick another dataset")

    segment = pool[0]["segment"]
    print(f"\n[limit] segment keys: {sorted(segment.keys())}")
    for key in ("base_indices", "sequence_valid", "loss_mask", "update_mask",
                "is_first", "is_last", "segment_start", "mask_plan"):
        print(f"  {key:<15} = {np.asarray(segment[key]).astype(int).tolist()}")
    print(f"  dataset_id      = {segment['dataset_id']}")
    print(f"  episode_id      = {segment['episode_id']}")
    steps = segment["steps"]
    pad = sum(1 for step in steps if step is None)
    print(f"  steps           = {len(steps)} entries ({pad} None padding, burn-in slots 0..{burn_in - 1})")
    first_real = next(step for step in steps if step is not None)
    print("  step fields:")
    for key, value in first_real.items():
        if isinstance(value, np.ndarray):
            print(f"    {key:<12} {describe_array(value)}")
        elif isinstance(value, list):
            print(f"    {key:<12} list len={len(value)} of {type(value[0]).__name__}")
        else:
            print(f"    {key:<12} {type(value).__name__}: {str(value)[:70]}")
    masked_positions = [
        burn_in + offset for offset in np.flatnonzero(np.asarray(segment["mask_plan"], dtype=bool))
    ]
    print(f"  masked step positions: {masked_positions}")
    for position in masked_positions:
        has_clean = "video_clean" in steps[position]
        print(f"    steps[{position}] has video_clean: {has_clean} "
              f"({describe_array(steps[position]['video_clean']) if has_clean else 'MISSING'})")
        if not has_clean:
            raise AssertionError("masked step lacks video_clean")

    pairs = form_pairs(pool, quota=1)
    if not pairs:
        print("\n[limit] fewer than two same-maturity episodes sampled; splice check skipped")
    else:
        live, donor = pairs[0]
        twin = splice_foreign(live["segment"], donor["segment"], burn_in)
        same_burn = all(
            twin["steps"][i] is donor["segment"]["steps"][i] for i in range(burn_in)
        )
        same_tail = all(
            twin["steps"][i] is live["segment"]["steps"][i]
            for i in range(burn_in, len(twin["steps"]))
        )
        arrays_ok = all(
            np.array_equal(np.asarray(twin[key][:burn_in]), np.asarray(donor["segment"][key][:burn_in]))
            and np.array_equal(np.asarray(twin[key][burn_in:]), np.asarray(live["segment"][key][burn_in:]))
            for key in BURN_IN_ARRAY_KEYS
        )
        print(
            f"\n[limit] splice check: live ep={live['episode_id']} donor ep={donor['episode_id']} "
            f"burn_count={live['burn_count']}=={donor['burn_count']}"
        )
        print(f"  burn-in steps are donor's:      {same_burn}")
        print(f"  supervised steps are live's:    {same_tail}")
        print(f"  control arrays spliced at {burn_in}: {arrays_ok}")
        if not (same_burn and same_tail and arrays_ok):
            raise AssertionError("splice construction failed")
    print("\nLIMIT_OK")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=str, required=True, help="training yaml (run config.yaml)")
    parser.add_argument("--ckpt", type=str, default=None, help="exported weights .pt")
    parser.add_argument("--num-segments", type=int, default=48)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str, default=None, help="optional JSON results path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS,
                        help="comma list of dataset names from the memv2_stage1_mix")
    parser.add_argument("--robot-type", type=str, default="mikasa_robo")
    parser.add_argument("--data-root", type=str, default=None,
                        help="override cfg.datasets.vla_data.data_root_dir")
    parser.add_argument("--min-burn-in", type=int, default=1,
                        help="minimum valid burn-in decisions; pairs are matched on the exact count")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=None,
                        help="model-free validation: sample this many segments, print structure, exit")
    return parser


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    if parsed.limit is not None:
        run_limit(parsed)
    else:
        run(parsed)
