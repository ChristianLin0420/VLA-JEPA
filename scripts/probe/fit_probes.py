"""CPU probe battery over replay-engine state dumps (plan T0.6, hook H7).

Reads the per-episode npz shards written by ``scripts/offline_eval/replay_engine.py
--state-dump`` and decodes episode variables from the flattened [8x512] working
memory state against control feature sets:

    targets:   task_id, phase quintile (t/T), progress (t/T), gripper-event count
    features:  memory (flattened state), tokens (mean-pooled present tokens),
               const (initial slots, identical every row), shuffled (memory rows
               permuted across the corpus)

Five episode-disjoint splits (task-stratified, fixed across feature sets),
balanced accuracy / R^2 with episode-level bootstrap CIs.  Uses sklearn when
importable, else a closed-form ridge plus a torch softmax regression.  Writes
``probe_results.csv`` and one summary figure per target under ``--out-dir``.

``--smoke`` synthesizes shards and runs the full battery without real dumps.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

try:
    from sklearn.linear_model import LogisticRegression, Ridge

    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

CLASSIFICATION_TARGETS = ("task_id", "phase")
REGRESSION_TARGETS = ("progress", "gripper_events")
FEATURE_SETS = ("memory", "tokens", "const", "shuffled")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump-dir", type=str, default=None,
                        help="directory of replay_engine.py --state-dump npz shards")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--smoke", action="store_true",
                        help="run the full battery on synthetic shards")
    return parser


def gripper_event_counts(gt_gripper: np.ndarray) -> np.ndarray:
    """Cumulative gripper transitions observed before each decision.

    ``gt_gripper`` is [D, chunk] of normalized gripper commands; binarization at
    0.5 follows ``baseframework.unnormalize_actions``.
    """

    num_decisions, chunk = gt_gripper.shape
    stream = (gt_gripper.reshape(-1) >= 0.5).astype(np.int64)
    transitions = np.abs(np.diff(stream))
    return np.array(
        [transitions[: max(d * chunk - 1, 0)].sum() for d in range(num_decisions)],
        dtype=np.float64,
    )


def load_dumps(dump_dir: Path) -> dict:
    """Stack every shard into per-decision rows with paired feature sets."""

    memory, tokens, const = [], [], []
    episode_key, task, progress, gripper = [], [], [], []
    shards = sorted(dump_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"no npz shards under {dump_dir}")
    for shard in shards:
        data = np.load(shard, allow_pickle=False)
        states = data["states"].astype(np.float64)
        num_decisions = states.shape[0]
        memory.append(states.reshape(num_decisions, -1))
        tokens.append(data["token_mean"].astype(np.float64))
        const.append(
            np.tile(data["initial_slots"].astype(np.float64).reshape(1, -1), (num_decisions, 1))
        )
        key = f"{data['suite']}|{data['dataset']}|{int(data['episode'])}"
        episode_key.extend([key] * num_decisions)
        task.extend([str(data["task"])] * num_decisions)
        progress.append(data["progress"].astype(np.float64))
        gripper.append(gripper_event_counts(data["gt_gripper"]))
    progress = np.concatenate(progress)
    return {
        "memory": np.concatenate(memory),
        "tokens": np.concatenate(tokens),
        "const": np.concatenate(const),
        "episode_key": np.asarray(episode_key),
        "task": np.asarray(task),
        "progress": progress,
        "phase": np.minimum((progress * 5).astype(np.int64), 4),
        "gripper_events": np.concatenate(gripper),
    }


def build_folds(episode_key: np.ndarray, task: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Task-stratified, episode-disjoint fold id per row; fixed across feature sets."""

    episodes = sorted(set(zip(episode_key.tolist(), task.tolist())))
    rng = np.random.default_rng(seed)
    fold_of_episode = {}
    by_task = {}
    for key, task_name in episodes:
        by_task.setdefault(task_name, []).append(key)
    for task_name in sorted(by_task):
        keys = by_task[task_name]
        order = rng.permutation(len(keys))
        for position, index in enumerate(order):
            fold_of_episode[keys[index]] = position % n_splits
    return np.asarray([fold_of_episode[key] for key in episode_key.tolist()])


def _standardize(train_x: np.ndarray, test_x: np.ndarray):
    mean = train_x.mean(axis=0)
    std = np.maximum(train_x.std(axis=0), 1e-8)
    return (train_x - mean) / std, (test_x - mean) / std


def _fit_predict_regression(train_x, train_y, test_x, alpha: float) -> np.ndarray:
    if HAVE_SKLEARN:
        model = Ridge(alpha=alpha)
        model.fit(train_x, train_y)
        return model.predict(test_x)
    x = torch.as_tensor(np.hstack([train_x, np.ones((len(train_x), 1))]), dtype=torch.float64)
    y = torch.as_tensor(train_y, dtype=torch.float64)
    eye = torch.eye(x.shape[1], dtype=torch.float64)
    eye[-1, -1] = 0.0  # do not regularize the intercept
    weights = torch.linalg.solve(x.T @ x + alpha * eye, x.T @ y)
    test = torch.as_tensor(np.hstack([test_x, np.ones((len(test_x), 1))]), dtype=torch.float64)
    return (test @ weights).numpy()


def _fit_predict_classification(train_x, train_y, test_x, seed: int) -> np.ndarray:
    classes = np.unique(train_y)
    if len(classes) == 1:
        return np.full(len(test_x), classes[0])
    if HAVE_SKLEARN:
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(train_x, train_y)
        return model.predict(test_x)
    torch.manual_seed(seed)
    index_of = {label: i for i, label in enumerate(classes)}
    x = torch.as_tensor(train_x, dtype=torch.float32)
    y = torch.as_tensor([index_of[label] for label in train_y], dtype=torch.int64)
    linear = torch.nn.Linear(x.shape[1], len(classes))
    optimizer = torch.optim.Adam(linear.parameters(), lr=0.05)
    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(linear(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = linear(torch.as_tensor(test_x, dtype=torch.float32)).argmax(dim=1).numpy()
    return classes[predicted]


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = [
        float((y_pred[y_true == label] == label).mean()) for label in np.unique(y_true)
    ]
    return float(np.mean(recalls))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(((y_true - y_pred) ** 2).sum())
    total = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - residual / max(total, 1e-12)


def bootstrap_ci(episode_key, y_true, y_pred, metric_fn, n_boot: int, seed: int):
    """Episode-level bootstrap over pooled out-of-fold predictions."""

    episodes = np.asarray(sorted(set(episode_key.tolist())))
    rows_of = {key: np.flatnonzero(episode_key == key) for key in episodes}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(len(episodes), size=len(episodes), replace=True)
        rows = np.concatenate([rows_of[episodes[i]] for i in chosen])
        try:
            values.append(metric_fn(y_true[rows], y_pred[rows]))
        except ZeroDivisionError:
            continue
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def run_probes(data: dict, args) -> list:
    rng = np.random.default_rng(args.seed)
    folds = build_folds(data["episode_key"], data["task"], args.splits, args.seed)
    features = {
        "memory": data["memory"],
        "tokens": data["tokens"],
        "const": data["const"],
        "shuffled": data["memory"][rng.permutation(len(data["memory"]))],
    }
    task_codes = np.unique(data["task"], return_inverse=True)[1]
    targets = {
        "task_id": (task_codes, "classification"),
        "phase": (data["phase"], "classification"),
        "progress": (data["progress"], "regression"),
        "gripper_events": (data["gripper_events"], "regression"),
    }

    results = []
    for target_name, (y, kind) in targets.items():
        if kind == "classification" and len(np.unique(y)) < 2:
            print(f"skipping {target_name}: fewer than two classes in the dump")
            continue
        for feature_name in FEATURE_SETS:
            x = features[feature_name]
            y_pred = np.empty(len(y), dtype=y.dtype if kind == "classification" else np.float64)
            for fold in range(args.splits):
                train, test = folds != fold, folds == fold
                if not test.any():
                    continue
                train_x, test_x = _standardize(x[train], x[test])
                if kind == "classification":
                    y_pred[test] = _fit_predict_classification(train_x, y[train], test_x, args.seed)
                else:
                    y_pred[test] = _fit_predict_regression(train_x, y[train], test_x, args.ridge_alpha)
            metric_fn = balanced_accuracy if kind == "classification" else r_squared
            value = metric_fn(y, y_pred)
            ci_low, ci_high = bootstrap_ci(
                data["episode_key"], y, y_pred, metric_fn, args.bootstrap, args.seed
            )
            results.append(
                {
                    "target": target_name,
                    "features": feature_name,
                    "metric": "balanced_accuracy" if kind == "classification" else "r2",
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_rows": int(len(y)),
                    "n_episodes": int(len(set(data["episode_key"].tolist()))),
                    "splits": args.splits,
                }
            )
            print(f"{target_name:>14} | {feature_name:>8}: {value:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    return results


def write_outputs(results: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "probe_results.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for target in sorted({row["target"] for row in results}):
        rows = [row for row in results if row["target"] == target]
        labels = [row["features"] for row in rows]
        values = [row["value"] for row in rows]
        errors = [
            [row["value"] - row["ci_low"] for row in rows],
            [row["ci_high"] - row["value"] for row in rows],
        ]
        figure, axis = plt.subplots(figsize=(5, 3.2))
        axis.bar(labels, values, yerr=errors, capsize=4, color="#4878cf")
        axis.set_ylabel(rows[0]["metric"])
        axis.set_title(f"probe: {target}")
        figure.tight_layout()
        figure.savefig(out_dir / f"probe_{target}.png", dpi=150)
        plt.close(figure)
    print(f"results -> {csv_path}")


def make_smoke_dumps(dump_dir: Path, seed: int) -> None:
    """Synthetic shards with recoverable structure: task offset + progress drift."""

    rng = np.random.default_rng(seed)
    dump_dir.mkdir(parents=True, exist_ok=True)
    for episode in range(10):
        task_index = episode % 2
        num_decisions = int(rng.integers(6, 12))
        progress = np.arange(num_decisions) / max(num_decisions - 1, 1)
        states = (
            rng.standard_normal((num_decisions, 2, 4)) * 0.1
            + task_index
            + progress[:, None, None]
        )
        np.savez_compressed(
            dump_dir / f"smoke__smoke_dataset__ep{episode:06d}.npz",
            states=states.astype(np.float32),
            token_mean=rng.standard_normal((num_decisions, 8)).astype(np.float32),
            gt_gripper=rng.standard_normal((num_decisions, 7)).astype(np.float32),
            decision=np.arange(num_decisions, dtype=np.int64),
            progress=progress,
            initial_slots=np.zeros((2, 4), dtype=np.float32),
            episode=np.int64(episode),
            suite=np.str_("smoke"),
            dataset=np.str_("smoke_dataset"),
            task=np.str_(f"smoke task {task_index}"),
        )


if __name__ == "__main__":
    parsed = build_argparser().parse_args()
    out_dir = Path(parsed.out_dir)
    if parsed.smoke:
        parsed.dump_dir = str(out_dir / "smoke_dumps")
        parsed.bootstrap = min(parsed.bootstrap, 100)
        make_smoke_dumps(Path(parsed.dump_dir), parsed.seed)
    elif not parsed.dump_dir:
        raise SystemExit("--dump-dir is required unless --smoke is set")
    loaded = load_dumps(Path(parsed.dump_dir))
    probe_results = run_probes(loaded, parsed)
    if not probe_results:
        raise SystemExit("no probe targets were runnable on this dump")
    write_outputs(probe_results, out_dir)
    with open(out_dir / "probe_config.json", "w") as handle:
        json.dump(
            {k: v for k, v in vars(parsed).items()}, handle, indent=2, sort_keys=True
        )
