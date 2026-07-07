"""D3 masked-step demand audit on the memv2 demand-bearing corpora.

Adapts scripts/data/g0_demand_audit.py (worktree wt-memexp) to answer:
at the exact decision depths where memv2.2 masks landed (supervised
positions 1-3 of stride-7, K=4, burn-in-8 segments), is the expert action
episode-dependent beyond (task, phase)?

Differences vs G0:
- Corpora: libero_mem_1.0.0_lerobot + 6 MIKASA anchor datasets (the memv2
  mixture's demand-bearing subset).
- Targets: actions at t = s0 + 7p, p in {1,2,3}, for every valid supervised
  segment start s0 on the training lattice (s0 = 0,7,...  with
  s0 + 21 + max_delta <= L-1, max_delta = 7 from the video horizon 8 --
  matches LeRobotMixtureDataset._segment_catalogs / _sample_mask_plan:
  first supervised decision never masked).
- Features beyond (task x phase) cells:
    B: init-layout (proprio[0] ++ actions[0:7]) ++ burn-in summary
       ([proprio[s0-7j] ++ actions[s0-7j]] for j=1..8, zero-padded at
       episode start), task-interacted -- the D3-specified ceiling.
    C: B ++ previous-decision context (proprio[t-7] ++ actions[t-7]; the
       immediately preceding decision is always sighted under
       memory_mask_max_per_segment=1) -- the 1-step-buffer ceiling.
    D: B ++ visual burn-in summary (12x12x3 RGB thumbnails of the main
       camera at bases {0} + {s0-7j, j=1..8}) -- ceiling for visually
       carried cues that proprio/actions cannot encode (MIKASA cue colors,
       LIBERO-Mem occluded objects).  Not task-interacted (dimensionality).
- Same Ridge / GroupKFold-by-episode / pooled OOF R2 / per-set best-alpha
  machinery as G0.

CPU only.  Writes cache + results under /lustre/.../tmp/diag/.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATA_ROOT = "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot"
DIAG = Path("/lustre/fsw/portfolios/edgeai/users/chrislin/tmp/diag")
DEFAULT_CORPORA = ",".join(
    [
        "libero_mem_1.0.0_lerobot",
        "mikasa_shell_game_shuffle_touch_vla_v0",
        "mikasa_shell_game_shuffle_touch_long_vla_v0",
        "mikasa_remember_color_3_long_vla_v0",
        "mikasa_remember_color_5_long_vla_v0",
        "mikasa_take_it_back_vla_v0",
        "mikasa_chain_of_colors_3_vla_v0",
    ]
)
STRIDE = 7
CHUNK = 7
BURN_IN = 8
SUPERVISED = 4
MAX_DELTA = 7  # video horizon 8 (starVLA/dataloader/__init__.py) dominates action chunk 6
THUMB = 12

MAIN_CAMERA_PREFERENCE = ("observation.images.image", "observation.images.top")


def phase_bucket(index: int, length: int, num_buckets: int) -> int:
    if length <= 1:
        return 0
    return min(int(index / (length - 1) * num_buckets), num_buckets - 1)


# ---------------------------------------------------------------- loading

def _decode_thumbs(args):
    """Decode one episode's main-camera video at stride-7 bases -> [n,432]."""
    video_path, length = args
    import av

    bases = list(range(0, length, STRIDE))
    wanted = set(bases)
    thumbs = {}
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted:
                    img = frame.to_ndarray(format="rgb24")
                    h, w = img.shape[:2]
                    ys = (np.arange(THUMB) * h // THUMB)
                    xs = (np.arange(THUMB) * w // THUMB)
                    thumbs[i] = img[np.ix_(ys, xs)].astype(np.float32).ravel() / 255.0
                if i >= max(wanted):
                    break
    except Exception as exc:  # noqa: BLE001 -- corrupt video: zeros, keep sample
        print(f"  decode failure {video_path}: {exc}")
    dim = THUMB * THUMB * 3
    return np.stack([thumbs.get(b, np.zeros(dim, dtype=np.float32)) for b in bases])


def load_corpus(data_root: str, name: str, *, with_visual: bool, workers: int) -> list:
    """Episode records: task, actions [T,7], proprio [T,sd], thumbs [ceil(T/7),432]."""
    import pandas as pd

    root = Path(data_root) / name
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = {}
    with open(root / "meta" / "tasks.jsonl") as handle:
        for line in handle:
            entry = json.loads(line)
            tasks[int(entry["task_index"])] = entry["task"]
    camera = next(
        (k for k in MAIN_CAMERA_PREFERENCE if k in info["features"]),
        next(k for k, v in info["features"].items() if v.get("dtype") == "video"),
    )
    total = int(info["total_episodes"])
    chunks_size = int(info["chunks_size"])
    episodes = []
    video_jobs = []
    for episode in range(total):
        path = root / info["data_path"].format(
            episode_chunk=episode // chunks_size, episode_index=episode
        )
        frame = pd.read_parquet(path, columns=["observation.state", "action", "task_index"])
        actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
        proprio = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
        episodes.append(
            {
                "task": tasks[int(frame["task_index"].iloc[0])],
                "actions": actions,
                "proprio": proprio,
            }
        )
        if with_visual:
            video_jobs.append(
                (
                    str(
                        root
                        / info["video_path"].format(
                            episode_chunk=episode // chunks_size,
                            video_key=camera,
                            episode_index=episode,
                        )
                    ),
                    len(frame),
                )
            )
    if with_visual:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for episode, thumbs in zip(episodes, pool.map(_decode_thumbs, video_jobs, chunksize=4)):
                episode["thumbs"] = thumbs
    return episodes


# ---------------------------------------------------------------- design

def build_design(episodes, *, num_buckets: int, row_cap: int):
    """Rows for every (segment s0, masked position p in 1..3) on the lattice."""
    tasks = sorted({e["task"] for e in episodes})
    task_index = {task: i for i, task in enumerate(tasks)}
    num_tasks = len(tasks)
    with_visual = "thumbs" in episodes[0]
    vdim = THUMB * THUMB * 3

    # thinning factor to respect the row cap (deterministic, on s0)
    total_rows = 0
    for e in episodes:
        length = e["actions"].shape[0]
        last = length - 1 - MAX_DELTA - (SUPERVISED - 1) * STRIDE
        if last >= 0:
            total_rows += (last // STRIDE + 1) * (SUPERVISED - 1)
    thin = max(1, int(np.ceil(total_rows / row_cap)))

    cell_rows, layout_rows, prev_rows, vis_rows, targets, groups, depths = (
        [], [], [], [], [], [], []
    )
    for group, e in enumerate(episodes):
        actions, proprio = e["actions"], e["proprio"]
        length = actions.shape[0]
        last = length - 1 - MAX_DELTA - (SUPERVISED - 1) * STRIDE
        if last < 0:
            continue
        sd = proprio.shape[1]
        onehot = np.zeros(num_tasks)
        onehot[task_index[e["task"]]] = 1.0
        init_layout = np.concatenate((proprio[0], actions[:CHUNK].ravel()))
        for s0 in range(0, last + 1, STRIDE):
            if (s0 // STRIDE) % thin:
                continue
            burn = []
            for j in range(1, BURN_IN + 1):
                b = s0 - STRIDE * j
                if b >= 0:
                    burn.append(np.concatenate((proprio[b], actions[b])))
                else:
                    burn.append(np.zeros(sd + actions.shape[1]))
            layout = np.concatenate([init_layout] + burn)
            interacted = np.kron(onehot, layout)
            if with_visual:
                vis_bases = [0] + [max(s0 - STRIDE * j, -1) for j in range(1, BURN_IN + 1)]
                vis = np.concatenate(
                    [
                        e["thumbs"][b // STRIDE] if b >= 0 else np.zeros(vdim)
                        for b in vis_bases
                    ]
                )
            for p in range(1, SUPERVISED):
                t = s0 + STRIDE * p
                cell = np.zeros(num_tasks * num_buckets)
                cell[task_index[e["task"]] * num_buckets + phase_bucket(t, length, num_buckets)] = 1.0
                cell_rows.append(cell)
                layout_rows.append(interacted)
                prev_rows.append(np.kron(onehot, np.concatenate((proprio[t - STRIDE], actions[t - STRIDE]))))
                if with_visual:
                    vis_rows.append(vis)
                targets.append(actions[t])
                groups.append(group)
                depths.append(t)
    if not targets:
        raise ValueError("no rows")
    out = dict(
        cells=np.array(cell_rows),
        layout=np.array(layout_rows),
        prev=np.array(prev_rows),
        targets=np.array(targets),
        groups=np.array(groups),
        depths=np.array(depths),
        thin=thin,
    )
    if with_visual:
        out["visual"] = np.array(vis_rows)
    return out


# ---------------------------------------------------------------- regression

def _oof_r2(features, targets, groups, *, alphas, folds: int) -> float:
    """Best pooled out-of-fold R2 over the Ridge alpha grid (GroupKFold).

    Identical semantics to g0_demand_audit._oof_r2; restructured so each
    fold standardizes once and reuses the matrices across alphas.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold

    splits = min(folds, len(np.unique(groups)))
    predictions = {alpha: np.zeros_like(targets) for alpha in alphas}
    for train, test in GroupKFold(n_splits=splits).split(features, targets, groups):
        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0) + 1e-8
        ftrain = (features[train] - mean) / std
        ftest = (features[test] - mean) / std
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(ftrain, targets[train])
            predictions[alpha][test] = model.predict(ftest)
    return max(
        float(r2_score(targets, predictions[alpha], multioutput="uniform_average"))
        for alpha in alphas
    )


def audit(design, *, alphas, folds: int) -> dict:
    cells, targets, groups = design["cells"], design["targets"], design["groups"]
    r2_base = _oof_r2(cells, targets, groups, alphas=alphas, folds=folds)
    feats_b = np.concatenate((cells, design["layout"]), axis=1)
    r2_b = _oof_r2(feats_b, targets, groups, alphas=alphas, folds=folds)
    feats_c = np.concatenate((feats_b, design["prev"]), axis=1)
    r2_c = _oof_r2(feats_c, targets, groups, alphas=alphas, folds=folds)
    result = {
        "r2_task_phase": r2_base,
        "r2_plus_layout_burnin": r2_b,
        "gap": r2_b - r2_base,
        "r2_plus_prev_decision": r2_c,
        "gap_prev": r2_c - r2_base,
        "episodes": int(len(np.unique(groups))),
        "samples": int(targets.shape[0]),
        "thin": int(design["thin"]),
        "unexplained_by_task_phase": 1.0 - r2_base,
    }
    if "visual" in design:
        feats_d = np.concatenate((feats_b, design["visual"]), axis=1)
        r2_d = _oof_r2(feats_d, targets, groups, alphas=alphas, folds=folds)
        result["r2_plus_visual_burnin"] = r2_d
        result["gap_visual"] = r2_d - r2_base
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--corpora", default=DEFAULT_CORPORA)
    parser.add_argument("--phase-buckets", type=int, default=10)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--row-cap", type=int, default=60000)
    parser.add_argument("--visual", action="store_true", default=True)
    parser.add_argument("--no-visual", dest="visual", action="store_false")
    parser.add_argument("--decode-workers", type=int, default=24)
    parser.add_argument("--out", default=str(DIAG / "d3_results.json"))
    args = parser.parse_args()
    alphas = tuple(float(a) for a in args.alphas.split(",") if a)

    results = {}
    for corpus in filter(None, args.corpora.split(",")):
        print(f"== {corpus}: loading", flush=True)
        episodes = load_corpus(
            args.data_root, corpus, with_visual=args.visual, workers=args.decode_workers
        )
        design = build_design(episodes, num_buckets=args.phase_buckets, row_cap=args.row_cap)
        print(
            f"   rows={design['targets'].shape[0]} thin={design['thin']} "
            f"dims: cells={design['cells'].shape[1]} layout={design['layout'].shape[1]}",
            flush=True,
        )
        results[corpus] = audit(design, alphas=alphas, folds=args.folds)
        row = results[corpus]
        print(
            f"{corpus:45s} R2(task,phase)={row['r2_task_phase']:+.4f} "
            f"R2(+layout+burnin)={row['r2_plus_layout_burnin']:+.4f} gap={row['gap']:+.4f} "
            f"gap_prev={row['gap_prev']:+.4f} "
            + (f"gap_visual={row.get('gap_visual', float('nan')):+.4f} " if args.visual else "")
            + f"({row['episodes']} eps, {row['samples']} rows)",
            flush=True,
        )
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")

    # memv2_stage1_mix weights over the audited demand-bearing subset
    weights = {c: (2.0 if c.startswith("libero_mem") else 0.0952381) for c in results}
    wsum = sum(weights.values())
    headline = sum(weights[c] * results[c]["gap"] for c in results) / wsum
    results["_headline_mixture_weighted_gap"] = headline
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"HEADLINE mixture-weighted masked-depth R2 gap = {headline:+.4f}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "24")
    main()
