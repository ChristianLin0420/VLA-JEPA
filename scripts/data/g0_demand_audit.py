"""G0 demand audit: the manufacturable demand ceiling of a corpus (design gate G0).

Regresses the expert action at decision t from (task, phase-bucket) cells alone
versus the same cells plus the episode's initial layout proxy — first-frame
proprio concatenated with the first expert action chunk.  The out-of-fold R2
gap per suite is the ceiling on episode-specific demand that observation
masking can manufacture: blind-step BC can only demand what the expert
conditioned on beyond (task, phase).  Gap ~ 0 on LIBERO means masking cannot
work on this corpus and the mixture must shift (gate table, G0).

CPU-only.  Ridge with per-episode grouped cross-validation, so layout features
must generalize across episodes instead of memorizing them; targets start at
t >= chunk so the action chunk inside the layout proxy can never contain the
regression target itself.

Usage:
    python scripts/data/g0_demand_audit.py [--suites libero_10,...] [--out g0.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot"
DEFAULT_SUITES = "libero_10,libero_goal,libero_object,libero_spatial"
DATASET_TEMPLATE = "{suite}_no_noops_1.0.0_lerobot"


def phase_bucket(index: int, length: int, num_buckets: int) -> int:
    """Bucket of the episode-progress fraction index/(length-1), clamped to the last."""

    if length <= 1:
        return 0
    return min(int(index / (length - 1) * num_buckets), num_buckets - 1)


def build_design(episodes, *, num_buckets: int, stride: int, start: int):
    """(task x phase)-cell one-hots, layout features, action targets, episode groups.

    Episode records are ``{"task": str, "layout": [L], "actions": [T, A]}``;
    decisions are the stride lattice ``start, start+stride, ...`` (start
    defaults to the chunk length so targets never overlap the layout proxy).
    Layout features are interacted with the task one-hot so the ceiling
    admits task-specific layout->action mappings, not one global map.
    """

    tasks = sorted({episode["task"] for episode in episodes})
    task_index = {task: i for i, task in enumerate(tasks)}
    num_cells = len(tasks) * num_buckets
    cell_rows, layout_rows, targets, groups = [], [], [], []
    for group, episode in enumerate(episodes):
        actions = np.asarray(episode["actions"], dtype=np.float64)
        layout = np.asarray(episode["layout"], dtype=np.float64)
        length = actions.shape[0]
        task_one_hot = np.zeros(len(tasks))
        task_one_hot[task_index[episode["task"]]] = 1.0
        interacted_layout = np.kron(task_one_hot, layout)
        for t in range(start, length, stride):
            one_hot = np.zeros(num_cells)
            one_hot[task_index[episode["task"]] * num_buckets + phase_bucket(t, length, num_buckets)] = 1.0
            cell_rows.append(one_hot)
            layout_rows.append(interacted_layout)
            targets.append(actions[t])
            groups.append(group)
    if not targets:
        raise ValueError("no regression targets; episodes shorter than the start offset")
    return np.array(cell_rows), np.array(layout_rows), np.array(targets), np.array(groups)


def _oof_r2(features, targets, groups, *, alphas, folds: int) -> float:
    """Best pooled out-of-fold R2 over the Ridge alpha grid (GroupKFold).

    Each feature set gets the regularization that suits it, so the gap
    compares model classes rather than one arbitrary alpha; selecting on the
    out-of-fold metric is mildly optimistic for both sets, which is the
    conservative direction for a demand *ceiling*.
    """

    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold

    splits = min(folds, len(np.unique(groups)))
    scores = []
    for alpha in alphas:
        predictions = np.zeros_like(targets)
        for train, test in GroupKFold(n_splits=splits).split(features, targets, groups):
            mean = features[train].mean(axis=0)
            std = features[train].std(axis=0) + 1e-8
            model = Ridge(alpha=alpha)
            model.fit((features[train] - mean) / std, targets[train])
            predictions[test] = model.predict((features[test] - mean) / std)
        scores.append(float(r2_score(targets, predictions, multioutput="uniform_average")))
    return max(scores)


def audit(episodes, *, num_buckets: int = 10, stride: int = 7, start: int = 7,
          alphas=(0.1, 1.0, 10.0, 100.0, 1000.0), folds: int = 5) -> dict:
    """R2 of (task, phase) vs (task, phase, init layout); gap = demand ceiling."""

    cells, layout, targets, groups = build_design(
        episodes, num_buckets=num_buckets, stride=stride, start=start
    )
    r2_base = _oof_r2(cells, targets, groups, alphas=alphas, folds=folds)
    r2_layout = _oof_r2(
        np.concatenate((cells, layout), axis=1), targets, groups, alphas=alphas, folds=folds
    )
    return {
        "r2_task_phase": r2_base,
        "r2_plus_layout": r2_layout,
        "gap": r2_layout - r2_base,
        "episodes": len(episodes),
        "samples": int(targets.shape[0]),
    }


def load_suite_episodes(data_root, dataset_name: str, *, chunk: int, max_episodes=None):
    """Episode records from one LeRobot dataset (expert parquet, no video)."""

    import pandas as pd

    root = Path(data_root) / dataset_name
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = {}
    with open(root / "meta" / "tasks.jsonl") as handle:
        for line in handle:
            entry = json.loads(line)
            tasks[int(entry["task_index"])] = entry["task"]
    total = int(info["total_episodes"])
    if max_episodes is not None:
        total = min(total, max_episodes)
    episodes = []
    for episode in range(total):
        path = root / info["data_path"].format(
            episode_chunk=episode // int(info["chunks_size"]), episode_index=episode
        )
        frame = pd.read_parquet(path, columns=["observation.state", "action", "task_index"])
        actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
        proprio0 = np.asarray(frame["observation.state"].iloc[0], dtype=np.float64)
        episodes.append(
            {
                "task": tasks[int(frame["task_index"].iloc[0])],
                "layout": np.concatenate((proprio0, actions[:chunk].ravel())),
                "actions": actions,
            }
        )
    return episodes


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--suites", type=str, default=DEFAULT_SUITES,
                        help=f"comma-separated suites, mapped through {DATASET_TEMPLATE!r}")
    parser.add_argument("--stride", type=int, default=7, help="training decision lattice stride")
    parser.add_argument("--chunk", type=int, default=7,
                        help="action-chunk length in the layout proxy (targets start here)")
    parser.add_argument("--phase-buckets", type=int, default=10)
    parser.add_argument("--alphas", type=str, default="0.1,1,10,100,1000",
                        help="Ridge alpha grid; each feature set takes its best")
    parser.add_argument("--folds", type=int, default=5, help="GroupKFold folds (per episode)")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="optional JSON results path")
    return parser


def main(args) -> None:
    results = {}
    for suite in filter(None, args.suites.split(",")):
        episodes = load_suite_episodes(
            args.data_root, DATASET_TEMPLATE.format(suite=suite),
            chunk=args.chunk, max_episodes=args.max_episodes,
        )
        results[suite] = audit(
            episodes, num_buckets=args.phase_buckets, stride=args.stride,
            start=args.chunk, folds=args.folds,
            alphas=tuple(float(alpha) for alpha in args.alphas.split(",") if alpha),
        )
        row = results[suite]
        print(
            f"{suite:16s} R2(task,phase)={row['r2_task_phase']:+.4f}  "
            f"R2(+init layout)={row['r2_plus_layout']:+.4f}  gap={row['gap']:+.4f}  "
            f"({row['episodes']} episodes, {row['samples']} decisions)"
        )
    print(
        "G0 verdict: the per-suite gap is the manufacturable demand ceiling; "
        "gap ~ 0 everywhere means masking cannot work on this corpus (shift the mixture)."
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"results -> {args.out}")


if __name__ == "__main__":
    main(build_argparser().parse_args())
