"""Cache per-decision Qwen tokens for offline teacher-forced replay (plan T0.2, hook H7).

Walks every episode of one LeRobot dataset on the training stride-7 decision lattice
(the exact ``_build_segment_catalog``/``sample_segment`` math, replicated in
``scripts/offline_eval/lattice.py``), builds each decision sample via the mixture's
own ``_format_step``, runs ``VLA_JEPA._encode_qwen_tokens`` once per decision, and
writes one ``.pt`` cache per episode:

    action_tokens          [D, 24, 2048]  (bf16, memory write source)
    embodied_action_tokens [D, 32, 2048]  (bf16, fusion consumer)
    gt_actions             [D, 7, 7]      (fp32, chunk rule VLA_JEPA.py:321)
    + episode/decision metadata

Images are the training-path 224px `_format_step` views (no serve-time resize), so
cached tokens match the teacher-forcing distribution.  Caches are checkpoint-portable
across the memv1 lineage because Qwen has been frozen since stage1.

Usage:
    python scripts/offline_eval/cache_tokens.py \
        --ckpt <export.pt> --dataset libero_10_no_noops_1.0.0_lerobot \
        --suite libero_10 --out-dir results/offline/token_cache [--max-episodes N]

``--episodes`` restricts caching to an explicit index list (one per line), for
enumerator-selected unseen episodes (plan T0.2d); every requested episode must
be cached or the run fails loudly.  ``--smoke`` exercises the full
lattice/save/index plumbing on synthetic tensors without a checkpoint or dataset.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.offline_eval.lattice import decision_lattice
except ImportError:  # executed as a plain file instead of a module
    from lattice import decision_lattice

CACHE_SCHEMA = "qwen_token_cache_v1"
DEFAULT_DATA_ROOT = "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=str, default=None, help="exported .pt checkpoint")
    parser.add_argument("--dataset", type=str, default=None, help="LeRobot dataset dir name")
    parser.add_argument("--suite", type=str, required=True, help="label recorded in every cache")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episodes", type=str, default=None,
                        help="file of episode indices (whitespace-separated); cache exactly these")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--robot-type", type=str, default="libero_franka",
                        help="ROBOT_TYPE_CONFIG_MAP key (droid_libero for DROID)")
    parser.add_argument("--batch-size", type=int, default=8, help="decisions per Qwen forward")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true",
                        help="run the full plumbing on synthetic tensors, no checkpoint")
    return parser


def load_episode_selection(path) -> set:
    """Episode indices from a whitespace/newline-separated file (plan T0.2d)."""
    tokens = Path(path).read_text().split()
    if not tokens:
        raise ValueError(f"empty episode selection file: {path}")
    return {int(token) for token in tokens}


def assert_selection_cached(selected: set, cached_ids: set) -> None:
    """Fail loudly when a requested episode was not cached (no silent truncation)."""
    missing = sorted(selected - cached_ids)
    if missing:
        preview = ", ".join(str(episode) for episode in missing[:10])
        raise SystemExit(
            f"{len(missing)} selected episodes not cached (absent from the catalog "
            f"or without a valid decision): {preview}"
            + (" ..." if len(missing) > 10 else "")
        )


def write_episode_cache(suite_dir: Path, record: dict) -> Path:
    path = suite_dir / f"{record['dataset']}__ep{int(record['episode']):06d}.pt"
    torch.save(record, path)
    return path


def append_index(suite_dir: Path, record: dict, path: Path) -> None:
    entry = {
        "suite": record["suite"],
        "dataset": record["dataset"],
        "episode": int(record["episode"]),
        "num_decisions": int(record["decision_bases"].shape[0]),
        "trajectory_length": int(record["trajectory_length"]),
        "lang": record["lang"],
        "file": path.name,
    }
    with open(suite_dir / "index.jsonl", "a") as handle:
        handle.write(json.dumps(entry) + "\n")


def cache_one_episode(model, mixture, dataset, episode_id, bases, args, cfg) -> dict:
    """Encode every lattice decision of one episode with the frozen Qwen backbone."""

    prompt = cfg.datasets.vla_data.get("CoT_prompt", "")
    chunk = int(cfg.framework.action_model.future_action_window_size) + 1
    action_tokens, embodied_tokens, gt_actions = [], [], []
    lang = None
    for start in range(0, len(bases), args.batch_size):
        steps = [
            mixture._format_step(dataset, int(episode_id), int(base))
            for base in bases[start : start + args.batch_size]
        ]
        if lang is None:
            lang = steps[0]["lang"]
        if any(step["lang"] != lang for step in steps):
            raise ValueError(f"episode {episode_id}: instruction changed mid-episode")
        qwen = model._encode_qwen_tokens(
            [step["image"] for step in steps],
            [step["lang"] for step in steps],
            prompt,
            require_embodied=True,
        )
        action_tokens.append(qwen.action_tokens.detach().cpu())
        embodied_tokens.append(qwen.embodied_action_tokens.detach().cpu())
        gt_actions.append(
            torch.stack(
                [
                    torch.as_tensor(np.asarray(step["action"], dtype=np.float32))[-chunk:]
                    for step in steps
                ]
            )
        )
    return {
        "schema": CACHE_SCHEMA,
        "suite": args.suite,
        "dataset": str(dataset.dataset_name),
        "episode": int(episode_id),
        "trajectory_length": None,  # filled by caller
        # torch tensor keeps the payload loadable under weights_only=True
        "decision_bases": torch.as_tensor(np.asarray(bases, dtype=np.int64)),
        "stride": int(mixture.segment_stride),
        "lang": lang,
        "ckpt": str(args.ckpt),
        "action_tokens": torch.cat(action_tokens),
        "embodied_action_tokens": torch.cat(embodied_tokens),
        "gt_actions": torch.cat(gt_actions),
    }


def run(args) -> None:
    # Heavy imports stay out of --help/--smoke.
    from starVLA.dataloader.gr00t_lerobot.datasets import (
        LeRobotMixtureDataset,
        SAMPLE_MODE_CONTIGUOUS_SEGMENT,
    )
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
    from starVLA.model.framework.base_framework import baseframework

    if not args.ckpt or not args.dataset:
        raise SystemExit("--ckpt and --dataset are required unless --smoke is set")

    model = baseframework.from_pretrained(args.ckpt)
    model = model.to(torch.device(args.device)).eval()
    cfg = model.config
    vla_data = cfg.datasets.vla_data

    dataset = make_LeRobotSingleDataset(
        Path(args.data_root),
        args.dataset,
        args.robot_type,
        delete_pause_frame=False,
        sample_mode=SAMPLE_MODE_CONTIGUOUS_SEGMENT,
        action_horizon=int(cfg.framework.action_model.action_horizon),
        video_horizon=int(cfg.framework.vj2_model.num_frames),
    )
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
    catalog = mixture._segment_catalogs[0]

    selected = load_episode_selection(args.episodes) if args.episodes else None
    suite_dir = Path(args.out_dir) / args.suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    cached, cached_ids = 0, set()
    with torch.inference_mode():
        for episode_index, episode_id in enumerate(catalog["trajectory_ids"]):
            if selected is not None and int(episode_id) not in selected:
                continue
            if args.max_episodes is not None and cached >= args.max_episodes:
                break
            length = int(catalog["trajectory_lengths"][episode_index])
            bases = decision_lattice(
                length,
                stride=mixture.segment_stride,
                min_delta=catalog["min_delta"],
                max_delta=catalog["max_delta"],
            )
            if bases.size == 0:
                continue
            record = cache_one_episode(model, mixture, dataset, episode_id, bases, args, cfg)
            record["trajectory_length"] = length
            path = write_episode_cache(suite_dir, record)
            append_index(suite_dir, record, path)
            cached += 1
            cached_ids.add(int(episode_id))
            print(f"[{cached}] episode {int(episode_id)}: {bases.size} decisions -> {path.name}")
    if selected is not None:
        assert_selection_cached(selected, cached_ids)
    print(f"cached {cached} episodes under {suite_dir}")


def run_smoke(args) -> None:
    """Synthetic end-to-end pass: lattice walk, cache write, index append."""

    rng = np.random.default_rng(0)
    suite_dir = Path(args.out_dir) / args.suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    stride, max_delta, hidden = 7, 7, 16
    for episode, length in enumerate((30, 90)):
        bases = decision_lattice(length, stride=stride, min_delta=0, max_delta=max_delta)
        num = int(bases.size)
        record = {
            "schema": CACHE_SCHEMA,
            "suite": args.suite,
            "dataset": "smoke_dataset",
            "episode": episode,
            "trajectory_length": length,
            "decision_bases": torch.as_tensor(bases),
            "stride": stride,
            "lang": "smoke task",
            "ckpt": "none",
            "action_tokens": torch.as_tensor(
                rng.standard_normal((num, 24, hidden)), dtype=torch.bfloat16
            ),
            "embodied_action_tokens": torch.as_tensor(
                rng.standard_normal((num, 32, hidden)), dtype=torch.bfloat16
            ),
            "gt_actions": torch.as_tensor(
                rng.standard_normal((num, 7, 7)), dtype=torch.float32
            ),
        }
        path = write_episode_cache(suite_dir, record)
        append_index(suite_dir, record, path)
        print(f"smoke episode {episode}: {num} decisions -> {path.name}")
    print(f"smoke OK: caches under {suite_dir}")


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    if parsed.smoke:
        run_smoke(parsed)
    else:
        run(parsed)
