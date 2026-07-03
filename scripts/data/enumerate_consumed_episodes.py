#!/usr/bin/env python
"""Sampler-replay enumerator for training-consumed episodes (plan H7, T0.2 data).

Replays the deterministic contiguous-segment sampler
(``starVLA/dataloader/gr00t_lerobot/datasets.py:1668-1715``) for every dataset
index a cotrain run has consumed, without constructing datasets or loading any
step data.  The vla DataLoader is sequential (``per_device_batch_size: 1``, no
sampler, ``dataloader/__init__.py:59-65``); after ``accelerator.prepare`` the
Accelerate ``BatchSamplerShard`` hands rank ``p`` batch ``t * world_size + p``,
so outer step ``t`` of an epoch consumes exactly those dataset indices (the
modulo wrap applies only to the padded final group of an epoch).  Each index
draws ``(dataset, flat_start)`` from
``np.random.default_rng(safe_hash(("segment-v1", epoch, index, seed)))`` --
one uniform dataset choice, then one uniform valid-start choice.

Not modeled: corrupt-video decode retries during training replace a segment
with a deterministic alternate ("segment-decode-retry-v1"); the primary draw
is counted as consumed here while the alternate is not.  Decode failures are
rare and loudly logged in trainer stdout ("Video decode failed for
deterministic segment index ..."); grep those logs and drop the retried
episodes from any unseen list before publishing counts.

Note on ``--seed``: ``build_dataloader`` never forwards a seed, so the sampler
seed is ``get_vla_dataset``'s hardcoded default 42 (``lerobot_datasets.py:59``)
regardless of the training config's ``seed`` (which is also 42 for this run).

Usage:
  python scripts/data/enumerate_consumed_episodes.py \
      --config /lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/vlajepa_memv1_cotrain/config.yaml \
      --completed-steps 34729 --world-size 8 \
      --output-dir outputs/consumed_episodes [--limit-indices 2000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf  # noqa: E402
from tqdm import tqdm  # noqa: E402

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP  # noqa: E402
from starVLA.dataloader.gr00t_lerobot.datasets import safe_hash  # noqa: E402
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES  # noqa: E402

DECISION_THRESHOLDS = (100, 150, 300)


def read_episode_metadata(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mirror ``LeRobotSingleDataset._get_trajectories`` (datasets.py:412-423)."""
    trajectory_ids, trajectory_lengths = [], []
    with open(Path(dataset_dir) / "meta" / "episodes.jsonl") as f:
        for line in f:
            episode = json.loads(line)
            trajectory_ids.append(episode["episode_index"])
            trajectory_lengths.append(episode["length"])
    return np.array(trajectory_ids), np.array(trajectory_lengths)


def delta_bounds(robot_type: str, action_horizon: int, video_horizon: int) -> tuple[int, int]:
    """Mirror ``_get_delta_indices`` (datasets.py:668-674) folded to the
    min/max used by ``_build_segment_catalog`` (datasets.py:1592-1603)."""
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type](
        observation_indices=list(range(video_horizon)),
        action_indices=list(range(action_horizon)),
    )
    delta_arrays = []
    for modality_config in data_config.modality_config().values():
        values = np.asarray(modality_config.delta_indices, dtype=np.int64).reshape(-1)
        if values.size:
            delta_arrays.append(values)
    if not delta_arrays:
        return 0, 0
    all_deltas = np.concatenate(delta_arrays)
    return int(all_deltas.min()), int(all_deltas.max())


def build_catalog(
    trajectory_ids: np.ndarray,
    trajectory_lengths: np.ndarray,
    min_delta: int,
    max_delta: int,
    segment_length: int,
    segment_stride: int,
) -> dict:
    """Mirror ``LeRobotMixtureDataset._build_segment_catalog`` (datasets.py:1568-1627)."""
    trajectory_ids = np.asarray(trajectory_ids)
    trajectory_lengths = np.asarray(trajectory_lengths, dtype=np.int64)
    stride = int(segment_stride)
    minimum_valid_base = max(0, -min_delta)
    first_base = ((minimum_valid_base + stride - 1) // stride) * stride
    supervised_span = (int(segment_length) - 1) * stride
    last_start = trajectory_lengths - 1 - max_delta - supervised_span
    valid_start_counts = np.where(
        last_start >= first_base,
        ((last_start - first_base) // stride) + 1,
        0,
    ).astype(np.int64)
    cumulative_starts = np.cumsum(valid_start_counts, dtype=np.int64)
    total_starts = int(cumulative_starts[-1]) if cumulative_starts.size else 0
    return {
        "trajectory_ids": trajectory_ids,
        "trajectory_lengths": trajectory_lengths,
        "valid_start_counts": valid_start_counts,
        "cumulative_starts": cumulative_starts,
        "total_starts": total_starts,
        "first_base": first_base,
        "min_delta": min_delta,
        "max_delta": max_delta,
    }


class SamplerReplay:
    """Bit-faithful stand-in for ``LeRobotMixtureDataset`` segment selection.

    Holds only the per-dataset catalogs and the normalized sampling weights;
    ``sample_location`` reproduces ``_segment_rng`` + ``_sample_segment_location``
    (datasets.py:1668-1715) draw for draw.
    """

    def __init__(
        self,
        dataset_names: list[str],
        catalogs: list[dict],
        raw_weights: list[float],
        segment_stride: int,
        seed: int,
        mode: str = "train",
    ):
        # Mirror the __init__ filter (datasets.py:1463-1483): datasets without a
        # single fully valid segment are dropped before indexing.
        kept = [i for i, catalog in enumerate(catalogs) if catalog["total_starts"] > 0]
        if not kept:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")
        self.dataset_names = [dataset_names[i] for i in kept]
        self.catalogs = [catalogs[i] for i in kept]
        self.raw_weights = [raw_weights[i] for i in kept]
        self.segment_stride = int(segment_stride)
        self.seed = int(seed)
        self.mode = mode

        # Mirror weight normalization (datasets.py:1503-1523) with
        # balance_dataset_weights=False (get_vla_dataset default).
        weights = np.array(self.raw_weights)
        if np.any(weights <= 0):
            weights = np.maximum(weights, 1e-8)
        weights_sum = weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            weights = np.ones(len(self.catalogs)) / len(self.catalogs)
        else:
            weights /= weights_sum
        self.sampling_weights = weights

    def sample_location(self, epoch: int, index: int) -> tuple[int, int, int]:
        """Return (dataset_index, episode_id, supervised_start) for one index."""
        effective_epoch = epoch if self.mode == "train" else 0
        seed_items = ("segment-v1", effective_epoch, int(index), self.seed)
        rng = np.random.default_rng(safe_hash(seed_items))
        dataset_index = int(rng.choice(len(self.catalogs), p=self.sampling_weights))
        catalog = self.catalogs[dataset_index]
        flat_start = int(rng.integers(catalog["total_starts"]))
        trajectory_index = int(
            np.searchsorted(catalog["cumulative_starts"], flat_start, side="right")
        )
        previous_total = (
            int(catalog["cumulative_starts"][trajectory_index - 1])
            if trajectory_index > 0
            else 0
        )
        supervised_start = (
            int(catalog["first_base"])
            + (flat_start - previous_total) * self.segment_stride
        )
        episode_id = int(catalog["trajectory_ids"][trajectory_index])
        return dataset_index, episode_id, supervised_start

    def epoch_length(self) -> int:
        """Mirror ``LeRobotMixtureDataset.__len__`` (datasets.py:1946-2009), sane path."""
        dataset_lengths = np.asarray(
            [catalog["total_starts"] for catalog in self.catalogs], dtype=np.int64
        )
        primary = np.array(self.raw_weights) == 1.0
        if not np.any(primary):
            primary = np.array(self.raw_weights) == max(self.raw_weights)
        ratios = (dataset_lengths / self.sampling_weights)[primary]
        return int(ratios.max())


def batches_per_epoch(epoch_length: int, world_size: int) -> int:
    """Per-rank dataloader length: ``BatchSamplerShard.__len__`` with batch
    size 1, drop_last=False, even_batches=True (accelerate defaults)."""
    length, remainder = divmod(epoch_length, world_size)
    return length + 1 if remainder else length


def iter_consumed_indices(completed_steps: int, world_size: int, epoch_length: int):
    """Yield (epoch, dataset_index) in trainer consumption order.

    Outer step ``g`` sits at batch ``t = g % batches_per_epoch`` of epoch
    ``g // batches_per_epoch`` (train_vlajepa_cotrain.py:307-331) and consumes
    indices ``t*world_size + rank``; on the padded final group of an epoch,
    ``BatchSamplerShard`` cycles indices from the epoch start, which the
    modulo reproduces exactly for batch size 1.
    """
    per_rank = batches_per_epoch(epoch_length, world_size)
    for step in range(completed_steps):
        epoch, t = divmod(step, per_rank)
        for rank in range(world_size):
            yield epoch, (t * world_size + rank) % epoch_length


def validate_run_config(cfg) -> None:
    """Guards for the assumptions baked into ``iter_consumed_indices``."""
    data_cfg = cfg.datasets.vla_data
    if data_cfg.sample_mode != "contiguous_segment":
        raise ValueError(f"expected contiguous_segment sampling, got {data_cfg.sample_mode!r}")
    if int(data_cfg.per_device_batch_size) != 1:
        raise NotImplementedError(
            "index schedule is only exact for per_device_batch_size=1 "
            f"(got {data_cfg.per_device_batch_size})"
        )
    accumulation = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    if accumulation != 1:
        raise NotImplementedError(
            "index schedule is only exact for gradient_accumulation_steps=1 "
            f"(got {accumulation}): completed_steps advances once per optimizer "
            "sync, so a step maps to exactly one batch per rank only without "
            "accumulation"
        )


def filtered_mixture_spec(data_mix: str) -> list[tuple[str, float, str]]:
    """Mirror get_vla_dataset's duplicate filter (lerobot_datasets.py:100-108)."""
    included, spec = set(), []
    for d_name, d_weight, robot_type in DATASET_NAMED_MIXTURES[data_mix]:
        dataset_key = (d_name, robot_type)
        if dataset_key in included:
            continue
        included.add(dataset_key)
        spec.append((d_name, d_weight, robot_type))
    return spec


def summarize(
    replay: SamplerReplay,
    consumed: dict[str, set],
    segment_stride: int,
) -> tuple[dict, list[dict]]:
    """Build the per-dataset summary and the unseen-episode records.

    A decision is one stride-``segment_stride`` lattice step; an episode of
    ``length`` frames holds ``length // segment_stride`` decisions.
    """
    summary, unseen_records = {}, []
    for name, catalog in zip(replay.dataset_names, replay.catalogs):
        ids = catalog["trajectory_ids"]
        lengths = catalog["trajectory_lengths"]
        consumed_ids = consumed.get(name, set())
        unseen_mask = np.array([int(episode_id) not in consumed_ids for episode_id in ids])
        decisions = lengths // segment_stride
        entry = {
            "episodes": int(len(ids)),
            "consumed": int(len(consumed_ids)),
            "unseen": int(unseen_mask.sum()),
        }
        for threshold in DECISION_THRESHOLDS:
            entry[f"unseen_ge{threshold}"] = int((unseen_mask & (decisions >= threshold)).sum())
        summary[name] = entry
        for episode_id, length, num_decisions in zip(
            ids[unseen_mask], lengths[unseen_mask], decisions[unseen_mask]
        ):
            unseen_records.append(
                {
                    "dataset": name,
                    "episode_index": int(episode_id),
                    "length": int(length),
                    "num_decisions": int(num_decisions),
                }
            )
    return summary, unseen_records


def print_summary_table(summary: dict) -> None:
    columns = ["episodes", "consumed", "unseen"] + [
        f"unseen_ge{t}" for t in DECISION_THRESHOLDS
    ]
    name_width = max(len(name) for name in list(summary) + ["TOTAL"])
    header = f"{'dataset':<{name_width}} " + " ".join(f"{c:>12}" for c in columns)
    print(header)
    print("-" * len(header))
    totals = {c: 0 for c in columns}
    for name, entry in summary.items():
        print(f"{name:<{name_width}} " + " ".join(f"{entry[c]:>12}" for c in columns))
        for c in columns:
            totals[c] += entry[c]
    print(f"{'TOTAL':<{name_width}} " + " ".join(f"{totals[c]:>12}" for c in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_runs/"
            "vlajepa_memv1_cotrain/config.yaml"
        ),
        help="Training run config (datasets.vla_data + framework horizons).",
    )
    parser.add_argument("--completed-steps", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampler seed; the trainer uses get_vla_dataset's default 42 "
        "(never forwarded from the config).",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Override data_root_dir.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit-indices",
        type=int,
        default=None,
        help="Smoke mode: replay only the first N global indices.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    validate_run_config(cfg)
    data_cfg = cfg.datasets.vla_data
    segment_length = int(data_cfg.segment_length)
    segment_stride = int(data_cfg.segment_stride)
    action_horizon = int(cfg.framework.action_model.action_horizon)
    video_horizon = int(cfg.framework.vj2_model.num_frames)
    data_root = Path(args.data_root or data_cfg.data_root_dir)

    spec = filtered_mixture_spec(data_cfg.data_mix)
    names, catalogs, raw_weights = [], [], []
    for d_name, d_weight, robot_type in spec:
        min_delta, max_delta = delta_bounds(robot_type, action_horizon, video_horizon)
        trajectory_ids, trajectory_lengths = read_episode_metadata(data_root / d_name)
        catalogs.append(
            build_catalog(
                trajectory_ids, trajectory_lengths, min_delta, max_delta,
                segment_length, segment_stride,
            )
        )
        names.append(d_name)
        raw_weights.append(d_weight)

    replay = SamplerReplay(names, catalogs, raw_weights, segment_stride, args.seed)
    epoch_length = replay.epoch_length()
    total_indices = args.completed_steps * args.world_size
    if args.limit_indices is not None:
        total_indices = min(total_indices, args.limit_indices)

    consumed: dict[str, set] = {name: set() for name in replay.dataset_names}
    draws = {name: 0 for name in replay.dataset_names}
    indices = iter_consumed_indices(args.completed_steps, args.world_size, epoch_length)
    for _ in tqdm(range(total_indices), desc="Replaying sampler", unit="idx"):
        epoch, index = next(indices)
        dataset_index, episode_id, _ = replay.sample_location(epoch, index)
        name = replay.dataset_names[dataset_index]
        consumed[name].add(episode_id)
        draws[name] += 1

    summary, unseen_records = summarize(replay, consumed, segment_stride)
    metadata = {
        "config": str(args.config),
        "data_root": str(data_root),
        "data_mix": str(data_cfg.data_mix),
        "completed_steps": args.completed_steps,
        "world_size": args.world_size,
        "seed": args.seed,
        "segment_length": segment_length,
        "segment_stride": segment_stride,
        "min_max_delta": {
            name: [catalog["min_delta"], catalog["max_delta"]]
            for name, catalog in zip(replay.dataset_names, replay.catalogs)
        },
        "mixture_epoch_length": epoch_length,
        "batches_per_epoch": batches_per_epoch(epoch_length, args.world_size),
        "replayed_indices": total_indices,
        "limit_indices": args.limit_indices,
        "draws_per_dataset": draws,
        "decision_thresholds": list(DECISION_THRESHOLDS),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "consumed_episodes.json", "w") as f:
        json.dump(
            {
                "metadata": metadata,
                "datasets": {name: sorted(ids) for name, ids in consumed.items()},
            },
            f,
        )
    with open(output_dir / "unseen_episodes.jsonl", "w") as f:
        for record in unseen_records:
            f.write(json.dumps(record) + "\n")
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"metadata": metadata, "summary": summary}, f, indent=2)

    if args.limit_indices is not None:
        print(f"[smoke] replayed {total_indices} of {args.completed_steps * args.world_size} indices")
    print_summary_table(summary)
    print(f"Wrote consumed_episodes.json, unseen_episodes.jsonl, summary.json to {output_dir}")


if __name__ == "__main__":
    main()
