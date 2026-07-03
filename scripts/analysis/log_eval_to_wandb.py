"""Log memexp eval results to wandb (one resumable run per checkpoint arm).

Aggregates ``results/<suite>/<ckpt-name>/{episodes,decisions}.jsonl`` from the
closed-loop LIBERO arms and, when present,
``results/offline/<ckpt-name>/replay_<suite>.parquet`` from the offline pilot
(T0.2) into a single wandb run.  The run id is derived from the ckpt name and
``resume="allow"`` is used, so re-running overwrites the run instead of
duplicating it.  Logged per arm:

    config              memory_mode / memory_params / ckpt / shas / blackout / trials
    summary             success_rate/<suite>, success_rate/pooled, episode counts
    eval/episodes       wandb.Table of every episode record
    decisions/*         histograms (injection_ratio, working_norm,
                        cf_delta_action_l2 when present) + per-decision-index
                        mean line plots for injection_ratio and working_norm
    offline/*           paired dMSE vs live per condition, burn-in curve
                        (J x order -> mean mse), lambda curve tables

Failures are loud (nonzero exit, no silent partial run); ``--dry-run`` prints
what would be logged without importing wandb.  wandb is only imported inside
``log_to_wandb`` so the aggregation helpers stay unit-testable offline
(tests/memory/test_wandb_eval_logger.py).
"""

import argparse
import json
import re
from pathlib import Path

CONFIG_KEYS = ("memory_mode", "memory_params", "ckpt", "ckpt_sha", "git_sha", "blackout")
HIST_FIELDS = ("injection_ratio", "working_norm", "cf_delta_action_l2")
SERIES_FIELDS = ("injection_ratio", "working_norm")
DMSE_CONDITIONS = ("bypass", "prior", "shuffled")


def sanitize_run_id(name: str) -> str:
    """wandb run id from a ckpt name: [A-Za-z0-9._-] only, capped at 128."""
    run_id = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    if not run_id:
        raise ValueError(f"cannot derive a wandb run id from {name!r}")
    return run_id[:128]


def step_group(ckpt_name: str) -> str:
    """Group tag, e.g. 'step_34729' from 'VLA-JEPA-memv1-live-step_34729'."""
    match = re.search(r"step[_-]?(\d+)", ckpt_name)
    return f"step_{match.group(1)}" if match else ckpt_name


def load_jsonl(path: Path) -> list:
    records = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{lineno}: bad JSON ({err})") from err
    return records


def collect_arm(results_root: Path, ckpt_name: str) -> dict:
    """{suite: {"episodes": [...], "decisions": [...]}} for every closed-loop suite."""
    arms = {}
    for episodes_path in sorted(results_root.glob(f"*/{ckpt_name}/episodes.jsonl")):
        suite = episodes_path.parent.parent.name
        decisions_path = episodes_path.with_name("decisions.jsonl")
        arms[suite] = {
            "episodes": load_jsonl(episodes_path),
            "decisions": load_jsonl(decisions_path) if decisions_path.exists() else [],
        }
    return arms


def collect_offline(results_root: Path, ckpt_name: str):
    """Concatenated replay_*.parquet DataFrame for the arm, or None."""
    paths = sorted((results_root / "offline" / ckpt_name).glob("replay_*.parquet"))
    if not paths:
        return None
    import pandas as pd

    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def suite_summary(episodes_by_suite: dict) -> dict:
    """success_rate/<suite> + episodes/<suite> + pooled over all suites."""
    summary, pooled_n, pooled_s = {}, 0, 0
    for suite, episodes in sorted(episodes_by_suite.items()):
        if not episodes:
            continue
        successes = sum(1 for e in episodes if e["success"])
        summary[f"success_rate/{suite}"] = successes / len(episodes)
        summary[f"episodes/{suite}"] = len(episodes)
        pooled_n += len(episodes)
        pooled_s += successes
    if pooled_n:
        summary["success_rate/pooled"] = pooled_s / pooled_n
        summary["episodes/pooled"] = pooled_n
    return summary


def episode_table(episodes: list):
    """(columns, rows) over the union of record keys; dict/list cells as JSON."""
    columns = []
    for record in episodes:
        columns.extend(k for k in record if k not in columns)
    as_cell = lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
    return columns, [[as_cell(record.get(c)) for c in columns] for record in episodes]


def decision_values(decisions: list, field: str) -> list:
    return [d[field] for d in decisions if d.get(field) is not None]


def decision_index_means(decisions: list, fields=SERIES_FIELDS):
    """(columns, rows) of per-decision-index means for the diagnostic fields."""
    buckets = {}
    for record in decisions:
        buckets.setdefault(record["decision_index"], []).append(record)
    rows = []
    for index in sorted(buckets):
        row = [index]
        for field in fields:
            values = decision_values(buckets[index], field)
            row.append(sum(values) / len(values) if values else None)
        rows.append(row + [len(buckets[index])])
    return ["decision_index", *fields, "count"], rows


def offline_paired_dmse(df, conditions=DMSE_CONDITIONS) -> list:
    """Mean (mse_cond - mse_live) paired per (suite, episode, decision).

    Restricted to base-battery rows (J == -1, lam == 1.0); duplicate live rows
    from the lambda sweep at lam == 1 collapse via the groupby mean.
    """
    keys = ["suite", "episode", "decision"]
    base = df[(df["J"] == -1) & (df["lam"] == 1.0)]
    live = base[base["condition"] == "live"].groupby(keys)["mse"].mean()
    rows = []
    for condition in conditions:
        other = base[base["condition"] == condition].groupby(keys)["mse"].mean()
        delta = (other - live).dropna()
        if delta.empty:
            continue
        per_suite = delta.groupby(level="suite").agg(["mean", "size"])
        rows.extend(
            {"suite": suite, "condition": condition,
             "dmse": float(r["mean"]), "pairs": int(r["size"])}
            for suite, r in per_suite.iterrows()
        )
        rows.append({"suite": "pooled", "condition": condition,
                     "dmse": float(delta.mean()), "pairs": int(delta.size)})
    return rows


def offline_burnin_curve(df) -> list:
    """(J, order) -> mean mse over the burnin_<order> conditions."""
    burn = df[df["condition"].str.startswith("burnin_")].copy()
    if burn.empty:
        return []
    burn["order"] = burn["condition"].str.replace("burnin_", "", regex=False)
    grouped = burn.groupby(["J", "order"])["mse"].agg(["mean", "size"]).reset_index()
    return [{"J": int(r["J"]), "order": r["order"],
             "mse": float(r["mean"]), "rows": int(r["size"])}
            for r in grouped.to_dict("records")]


def offline_lambda_curve(df) -> list:
    """(condition, lam) -> mean mse over the lambda-sweep conditions."""
    sweep = df[(df["J"] == -1) & df["condition"].isin(["live", "shuffled"])]
    grouped = sweep.groupby(["condition", "lam"])["mse"].agg(["mean", "size"]).reset_index()
    return [{"condition": r["condition"], "lam": float(r["lam"]),
             "mse": float(r["mean"]), "rows": int(r["size"])}
            for r in grouped.to_dict("records")]


def build_payload(ckpt_name: str, arms: dict, offline_df) -> dict:
    episodes = [e for arm in arms.values() for e in arm["episodes"]]
    decisions = [d for arm in arms.values() for d in arm["decisions"]]
    config = {"ckpt_name": ckpt_name}
    if episodes:
        first = episodes[0]
        config.update({k: first[k] for k in CONFIG_KEYS if k in first})
        config["trials"] = max(e["episode_idx"] for e in episodes) + 1
    summary = suite_summary({s: a["episodes"] for s, a in arms.items()})
    payload = {
        "config": config,
        "summary": summary,
        "episode_table": episode_table(episodes),
        "histograms": {f: v for f in HIST_FIELDS if (v := decision_values(decisions, f))},
        "decision_series": decision_index_means(decisions),
        "offline": None,
    }
    if offline_df is not None:
        payload["offline"] = {
            "paired_dmse": offline_paired_dmse(offline_df),
            "burnin_curve": offline_burnin_curve(offline_df),
            "lambda_curve": offline_lambda_curve(offline_df),
        }
        summary.update(
            (f"offline/dmse/{row['condition']}", row["dmse"])
            for row in payload["offline"]["paired_dmse"] if row["suite"] == "pooled"
        )
    return payload


def print_payload(payload: dict) -> None:
    print(f"[dry-run] config: {json.dumps(payload['config'], sort_keys=True)}")
    for key, value in sorted(payload["summary"].items()):
        rendered = f"{value:.4f}" if isinstance(value, float) else value
        print(f"[dry-run] summary {key} = {rendered}")
    columns, rows = payload["episode_table"]
    print(f"[dry-run] eval/episodes table: {len(rows)} rows x {len(columns)} cols")
    for field, values in payload["histograms"].items():
        print(f"[dry-run] histogram decisions/{field}: "
              f"n={len(values)} mean={sum(values) / len(values):.4f}")
    columns, rows = payload["decision_series"]
    print(f"[dry-run] decisions/index_means: {len(rows)} indices ({', '.join(columns)})")
    if payload["offline"]:
        for name, rows in payload["offline"].items():
            print(f"[dry-run] offline/{name}: {len(rows)} rows")
            for row in rows if name == "paired_dmse" else rows[:4]:
                print(f"[dry-run]   {json.dumps(row)}")


def log_to_wandb(args, run_id: str, group: str, payload: dict, wandb=None) -> None:
    if wandb is None:
        import wandb
    run = wandb.init(
        project=args.project, entity=args.entity,
        id=run_id, name=args.ckpt_name, group=group,
        job_type="eval", resume="allow", config=payload["config"],
    )
    try:
        logged = {}
        columns, rows = payload["episode_table"]
        if rows:
            logged["eval/episodes"] = wandb.Table(columns=columns, data=rows)
        for field, values in payload["histograms"].items():
            logged[f"decisions/{field}"] = wandb.Histogram(values)
        columns, rows = payload["decision_series"]
        if rows:
            table = wandb.Table(columns=columns, data=rows)
            logged["decisions/index_means"] = table
            logged.update(
                (f"decisions/{field}_by_index",
                 wandb.plot.line(table, "decision_index", field,
                                 title=f"mean {field} vs decision index"))
                for field in SERIES_FIELDS
            )
        for name, table_rows in (payload["offline"] or {}).items():
            if table_rows:
                logged[f"offline/{name}"] = wandb.Table(
                    columns=list(table_rows[0]),
                    data=[list(r.values()) for r in table_rows],
                )
        run.log(logged)
        run.summary.update(payload["summary"])
    except Exception:
        run.finish(exit_code=1)
        raise
    run.finish()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt-name", type=str, required=True,
                        help="checkpoint basename without .pt, e.g. VLA-JEPA-memv1-live-step_34729")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--project", type=str, default="vla-jepa")
    parser.add_argument("--entity", type=str, default="crlc112358")
    parser.add_argument("--group", type=str, default="auto",
                        help="wandb group; 'auto' parses the step tag from the ckpt name")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be logged without wandb.init")
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    arms = collect_arm(results_root, args.ckpt_name)
    offline_df = collect_offline(results_root, args.ckpt_name)
    if not arms and offline_df is None:
        raise SystemExit(
            f"no episodes.jsonl or replay_*.parquet for {args.ckpt_name!r} under {results_root}/")
    run_id = sanitize_run_id(args.ckpt_name)
    group = step_group(args.ckpt_name) if args.group == "auto" else args.group
    payload = build_payload(args.ckpt_name, arms, offline_df)
    print(f"[log_eval_to_wandb] run id={run_id} group={group} "
          f"suites={sorted(arms)} offline={offline_df is not None}")
    if args.dry_run:
        print_payload(payload)
        return
    log_to_wandb(args, run_id, group, payload)


if __name__ == "__main__":
    main()
