"""Temporal-reach summary for the DROID replay battery (plan T0.2d).

Reads the ``replay_droid.parquet`` written by ``scripts/offline_eval/replay_engine.py``
over enumerator-verified unseen DROID episodes and reports, per decision-depth bin
{[0,12), [12,25), [25,50), [50,100), [100,150+)}:

    paired dMSE            mean (bypass - live) and (shuffled - live) per decision,
                           with episode-level (cluster) bootstrap 95% CIs
    state stability        live working_norm mean and p95

The d=12 bin edge is the training unroll horizon; the last bin is the 8-12x
extrapolation regime the experiment exists to measure.  Output is one CSV row per
(condition, depth bin) — the working_norm columns are per-bin live-state statistics,
repeated on both condition rows — plus an optional wandb run (group ``droid_reach``,
resumable id, same entity/project as ``log_eval_to_wandb.py``).  ``--dry-run``
prints the rows without writing the CSV or touching wandb; ``--no-wandb`` writes
the CSV only.  Failures are loud; the aggregation helpers are unit-tested on
synthetic rows (tests/memory/test_droid_reach_summary.py).
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

DEPTH_EDGES = (12, 25, 50, 100)  # first edge = the d=12 training horizon
DEPTH_LABELS = ("0-11", "12-24", "25-49", "50-99", "100-150+")
DELTA_CONDITIONS = ("bypass", "shuffled")
PAIR_KEYS = ["dataset", "episode", "decision"]
CSV_COLUMNS = (
    "condition", "depth_bin", "pairs", "episodes",
    "dmse_mean", "dmse_ci_lo", "dmse_ci_hi",
    "working_norm_mean", "working_norm_p95",
)


def depth_label(decisions) -> np.ndarray:
    """Vectorized decision index -> depth-bin label."""
    return np.asarray(DEPTH_LABELS)[np.searchsorted(DEPTH_EDGES, np.asarray(decisions), side="right")]


def episode_bootstrap_ci(deltas, clusters, n_boot: int, rng) -> tuple:
    """Percentile 95% CI of the pooled mean under episode-cluster resampling."""
    import pandas as pd

    per = pd.Series(np.asarray(deltas, dtype=np.float64)).groupby(list(clusters)).agg(["sum", "size"])
    sums, counts = per["sum"].to_numpy(), per["size"].to_numpy()
    index = rng.integers(0, len(sums), size=(int(n_boot), len(sums)))
    boot = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    lo, hi = np.percentile(boot, (2.5, 97.5))
    return float(lo), float(hi)


def summarize(df, conditions=DELTA_CONDITIONS, n_boot: int = 1000, seed: int = 7) -> list:
    """One row per (condition, depth bin): paired dMSE + CI + live-state norms."""
    base = df[(df["J"] == -1) & (df["lam"] == 1.0)].copy()
    base["cluster"] = base["dataset"].astype(str) + ":" + base["episode"].astype(str)
    base["depth_bin"] = depth_label(base["decision"].to_numpy())
    live = base[base["condition"] == "live"].groupby(PAIR_KEYS)["mse"].mean()

    norms = {
        label: group["working_norm"].to_numpy(dtype=np.float64)
        for label, group in base[base["condition"] == "live"].groupby("depth_bin")
    }
    rows = []
    for condition in conditions:
        other = base[base["condition"] == condition].set_index(PAIR_KEYS)
        delta = (other["mse"] - live).dropna()
        if delta.empty:
            raise ValueError(f"no paired ({condition} - live) rows in the replay records")
        paired = other.loc[delta.index]
        rng = np.random.default_rng(seed)
        for label in DEPTH_LABELS:
            mask = (paired["depth_bin"] == label).to_numpy()
            if not mask.any():
                continue
            deltas = delta.to_numpy()[mask]
            clusters = paired["cluster"].to_numpy()[mask]
            lo, hi = episode_bootstrap_ci(deltas, clusters, n_boot, rng)
            bin_norms = norms.get(label, np.array([np.nan]))
            rows.append({
                "condition": condition,
                "depth_bin": label,
                "pairs": int(mask.sum()),
                "episodes": int(len(set(clusters))),
                "dmse_mean": float(deltas.mean()),
                "dmse_ci_lo": lo,
                "dmse_ci_hi": hi,
                "working_norm_mean": float(np.mean(bin_norms)),
                "working_norm_p95": float(np.percentile(bin_norms, 95)),
            })
    return rows


def load_records(path: Path):
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    with gzip.open(path, "rt") as handle:  # replay_engine's no-pandas fallback
        return pd.DataFrame([json.loads(line) for line in handle])


def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def log_wandb(args, rows: list) -> None:
    import wandb

    from scripts.analysis.log_eval_to_wandb import sanitize_run_id

    run = wandb.init(
        project=args.project, entity=args.entity,
        id=sanitize_run_id(f"droid-reach-{args.ckpt_name}"),
        name=f"droid-reach-{args.ckpt_name}", group=args.group,
        job_type="analysis", resume="allow",
        config={"ckpt_name": args.ckpt_name, "replay": str(args.replay),
                "n_boot": args.n_boot, "seed": args.seed,
                "depth_edges": list(DEPTH_EDGES)},
    )
    try:
        run.log({"droid_reach/summary": wandb.Table(
            columns=list(CSV_COLUMNS), data=[[row[c] for c in CSV_COLUMNS] for row in rows])})
        summary = {f"dmse/{r['condition']}/{r['depth_bin']}": r["dmse_mean"] for r in rows}
        summary.update((f"working_norm_p95/{r['depth_bin']}", r["working_norm_p95"]) for r in rows)
        run.summary.update(summary)
    except Exception:
        run.finish(exit_code=1)
        raise
    run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", type=Path, required=True,
                        help="replay_droid.parquet (or .jsonl.gz) from replay_engine.py")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--ckpt-name", type=str, default=None,
                        help="default: the replay file's parent directory name")
    parser.add_argument("--conditions", type=str, default=",".join(DELTA_CONDITIONS))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--project", type=str, default="vla-jepa")
    parser.add_argument("--entity", type=str, default="crlc112358")
    parser.add_argument("--group", type=str, default="droid_reach")
    parser.add_argument("--no-wandb", action="store_true", help="write the CSV only")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the summary rows; no CSV, no wandb")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ckpt_name is None:
        args.ckpt_name = args.replay.resolve().parent.name
    rows = summarize(load_records(args.replay),
                     conditions=tuple(filter(None, args.conditions.split(","))),
                     n_boot=args.n_boot, seed=args.seed)
    for row in rows:
        print(" ".join(f"{key}={row[key]}" for key in CSV_COLUMNS))
    if args.dry_run:
        return
    write_csv(rows, args.out_csv)
    print(f"wrote {len(rows)} rows -> {args.out_csv}")
    if not args.no_wandb:
        log_wandb(args, rows)
        print(f"wandb: group={args.group} run=droid-reach-{args.ckpt_name}")


if __name__ == "__main__":
    main()
