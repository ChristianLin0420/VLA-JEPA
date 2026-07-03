"""Archaeology summary (plan T0.7): pathway/content emergence across checkpoints.

Joins ``<root>/<tag>/replay_*.parquet`` (mode-battery replays from
``cluster/archaeology_offline.sbatch``) into one row per checkpoint anchor:
pooled paired dMSE vs live for {bypass, prior, shuffled} (via
``offline_paired_dmse`` from log_eval_to_wandb) plus tanh(policy gate) read
from the archived ``model.safetensors`` (CPU partial read, no model build).
Writes ``<root>/summary.csv``, prints the table, and logs one resumable wandb
run (default id ``archaeology-step34729-caches``, group ``archaeology``) with
the summary table and per-metric line plots vs training step.

Step tags are ``N`` (cotrain) or ``stage1_N``; ``--step-dir TAG=DIR`` points a
tag at a non-standard replay dir, e.g. the single-checkpoint smoke against the
existing live-run battery:

    archaeology_summary.py --dry-run \\
        --step-dir 34729=results/offline/VLA-JEPA-memv1-live-step_34729

``--dry-run`` prints the table only (no csv, no wandb); ``--no-wandb`` writes
the csv without logging.  Failures are loud, matching log_eval_to_wandb.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.log_eval_to_wandb import (  # noqa: E402
    DMSE_CONDITIONS,
    offline_paired_dmse,
    sanitize_run_id,
)

GATE_KEY = "policy_memory_fusion.gate"
CSV_COLUMNS = ("tag", "phase", "step", "tanh_policy_gate",
               *(f"dmse_{c}" for c in DMSE_CONDITIONS),
               *(f"pairs_{c}" for c in DMSE_CONDITIONS))


def parse_tag(tag: str):
    """('stage1' | 'cotrain', numeric step) from 'stage1_N' / 'N'."""
    try:
        if tag.startswith("stage1_"):
            return "stage1", int(tag[len("stage1_"):])
        return "cotrain", int(tag)
    except ValueError:
        raise SystemExit(f"bad step tag {tag!r} (want N or stage1_N)") from None


def read_tanh_policy_gate(path: Path) -> float:
    """tanh of the scalar fusion gate, read from safetensors without the model."""
    from safetensors import safe_open

    if not path.is_file():
        raise SystemExit(f"missing archived checkpoint {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = [k for k in handle.keys()
                if k == GATE_KEY or (k.startswith("policy_memory_fusion.") and k.endswith(".gate"))]
        if len(keys) != 1:
            raise SystemExit(f"{path}: expected one policy gate key, found {keys}")
        return math.tanh(float(handle.get_tensor(keys[0])))


def checkpoint_row(tag: str, replay_dir: Path, archive_root: Path) -> dict:
    """One summary row: pooled paired dMSE per condition + gate for a checkpoint."""
    import pandas as pd

    paths = sorted(replay_dir.glob("replay_*.parquet"))
    if not paths:
        raise SystemExit(f"no replay_*.parquet under {replay_dir}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    pooled = {r["condition"]: r for r in offline_paired_dmse(df) if r["suite"] == "pooled"}
    phase, step = parse_tag(tag)
    gate_path = archive_root / phase / f"step_{step}" / "model.safetensors"
    row = {"tag": tag, "phase": phase, "step": step,
           "tanh_policy_gate": read_tanh_policy_gate(gate_path)}
    for condition in DMSE_CONDITIONS:
        row[f"dmse_{condition}"] = pooled[condition]["dmse"] if condition in pooled else None
        row[f"pairs_{condition}"] = pooled[condition]["pairs"] if condition in pooled else None
    return row


def resolve_step_dirs(args) -> list:
    """Ordered [(tag, replay_dir)]: --steps / --step-dir, else discovered under --root."""
    root = Path(args.root)
    dirs = {}
    for tag in filter(None, args.steps.split(",")):
        dirs[tag] = root / tag
    if not dirs and not args.step_dir and root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name != "exports" and any(child.glob("replay_*.parquet")):
                dirs[child.name] = child
    for spec in args.step_dir:
        tag, _, path = spec.partition("=")
        if not tag or not path:
            raise SystemExit(f"--step-dir wants TAG=DIR, got {spec!r}")
        dirs[tag] = Path(path)
    if not dirs:
        raise SystemExit(f"no checkpoint replay dirs under {root} (and no --steps/--step-dir)")

    def training_order(item):
        phase, step = parse_tag(item[0])
        return {"stage1": 0, "cotrain": 1}[phase], step

    return sorted(dirs.items(), key=training_order)


def print_table(rows: list) -> None:
    columns = ["tag", "phase", "step", "tanh_policy_gate",
               *(f"dmse_{c}" for c in DMSE_CONDITIONS), "pairs"]

    def cell(row, column):
        if column == "pairs":
            values = sorted({row[f"pairs_{c}"] for c in DMSE_CONDITIONS
                             if row[f"pairs_{c}"] is not None})
            return "/".join(str(v) for v in values) or "--"
        value = row[column]
        if value is None:
            return "--"
        if isinstance(value, float):
            return f"{value:+.6f}" if column.startswith("dmse") else f"{value:.6f}"
        return str(value)

    table = [columns] + [[cell(row, column) for column in columns] for row in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(columns))]
    for line in table:
        print("  ".join(value.rjust(width) for value, width in zip(line, widths)))


def log_to_wandb(args, rows: list) -> None:
    import wandb

    columns = list(CSV_COLUMNS)
    run = wandb.init(
        project=args.project, entity=args.entity,
        id=sanitize_run_id(args.run_id), name=args.run_id, group=args.group,
        job_type="eval", resume="allow",
        config={"steps": [r["tag"] for r in rows], "conditions": list(DMSE_CONDITIONS)},
    )
    try:
        table = wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in rows])
        logged = {"archaeology/summary": table}
        logged.update(
            (f"archaeology/{metric}_vs_step",
             wandb.plot.line(table, "step", metric, title=f"{metric} vs training step"))
            for metric in ("tanh_policy_gate", *(f"dmse_{c}" for c in DMSE_CONDITIONS))
        )
        run.log(logged)
        run.summary.update({f"latest/{k}": v for k, v in rows[-1].items() if v is not None})
    except Exception:
        run.finish(exit_code=1)
        raise
    run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=str, default="results/offline/archaeology")
    parser.add_argument("--steps", type=str, default="",
                        help="comma-separated tags (N or stage1_N); default: discover "
                             "<root> subdirs holding replay parquets")
    parser.add_argument("--step-dir", action="append", default=[], metavar="TAG=DIR",
                        help="explicit replay dir for a tag (overrides/extends --steps)")
    parser.add_argument("--archive-root", type=str,
                        default="/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/memv1_ckpt_archive",
                        help="checkpoint archive with {cotrain,stage1}/step_*/model.safetensors")
    parser.add_argument("--out", type=str, default=None,
                        help="csv path (default <root>/summary.csv)")
    parser.add_argument("--project", type=str, default="vla-jepa")
    parser.add_argument("--entity", type=str, default="crlc112358")
    parser.add_argument("--run-id", type=str, default="archaeology-step34729-caches")
    parser.add_argument("--group", type=str, default="archaeology")
    parser.add_argument("--no-wandb", action="store_true", help="write csv, skip wandb")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the table only (no csv, no wandb)")
    return parser.parse_args()


def main():
    args = parse_args()
    archive_root = Path(args.archive_root)
    step_dirs = resolve_step_dirs(args)
    rows = [checkpoint_row(tag, replay_dir, archive_root) for tag, replay_dir in step_dirs]
    print_table(rows)
    if args.dry_run:
        return
    out = Path(args.out) if args.out else Path(args.root) / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[archaeology_summary] {len(rows)} checkpoints -> {out}")
    if not args.no_wandb:
        log_to_wandb(args, rows)


if __name__ == "__main__":
    main()
