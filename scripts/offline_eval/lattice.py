"""Episode-anchored decision-lattice math shared by the offline evaluation scripts.

Replicates the training sampler exactly (``LeRobotMixtureDataset._build_segment_catalog``
and ``sample_segment``, ``starVLA/dataloader/gr00t_lerobot/datasets.py:1568-1794``):
decisions live on the lattice ``first_base, first_base + stride, ...`` and a
supervised start is valid only when every modality delta is in-bounds at all K
supervised decisions.  Pure numpy so tests never touch the dataloader stack.
"""

import numpy as np


def first_lattice_base(min_delta: int, stride: int) -> int:
    """First lattice point at which every (possibly negative) delta is in-bounds."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    minimum_valid_base = max(0, -int(min_delta))
    return ((minimum_valid_base + stride - 1) // stride) * stride


def decision_lattice(
    trajectory_length: int,
    *,
    stride: int,
    min_delta: int = 0,
    max_delta: int = 7,
) -> np.ndarray:
    """All fully valid decision bases of one episode, in temporal order.

    A base is valid when ``base + min_delta >= 0`` and
    ``base + max_delta <= trajectory_length - 1`` — the same bound the sampler
    enforces on every burn-in and supervised decision it emits.
    """

    if trajectory_length < 0:
        raise ValueError("trajectory_length must be non-negative")
    first_base = first_lattice_base(min_delta, stride)
    return np.arange(first_base, trajectory_length - max_delta, stride, dtype=np.int64)


def valid_segment_start_count(
    trajectory_length: int,
    *,
    stride: int,
    segment_length: int,
    min_delta: int = 0,
    max_delta: int = 7,
) -> int:
    """Number of fully valid supervised starts (``valid_start_counts`` per episode)."""

    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    first_base = first_lattice_base(min_delta, stride)
    supervised_span = (segment_length - 1) * stride
    last_start = trajectory_length - 1 - max_delta - supervised_span
    if last_start < first_base:
        return 0
    return int((last_start - first_base) // stride) + 1


def segment_base_indices(
    supervised_start: int,
    *,
    stride: int,
    segment_length: int,
    burn_in: int,
    min_delta: int = 0,
) -> np.ndarray:
    """Padded burn-in + supervised base vector, exactly as ``sample_segment`` builds it.

    The first ``burn_in`` positions are left-padded with ``-1`` when the episode
    prefix is shorter than the configured burn-in window.
    """

    first_base = first_lattice_base(min_delta, stride)
    if supervised_start < first_base or (supervised_start - first_base) % stride != 0:
        raise ValueError(
            f"supervised_start {supervised_start} is not on the lattice starting at {first_base}"
        )
    supervised_bases = supervised_start + np.arange(segment_length, dtype=np.int64) * stride
    first_burn_base = max(first_base, supervised_start - burn_in * stride)
    burn_bases = np.arange(first_burn_base, supervised_start, stride, dtype=np.int64)
    pad_count = burn_in - len(burn_bases)
    return np.concatenate(
        (np.full(pad_count, -1, dtype=np.int64), burn_bases, supervised_bases)
    )


def dataset_delta_bounds(delta_indices: dict) -> tuple[int, int]:
    """(min_delta, max_delta) over all modalities, as the segment catalog computes them."""

    delta_arrays = []
    for values in delta_indices.values():
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        if values.size:
            delta_arrays.append(values)
    if not delta_arrays:
        return 0, 0
    all_deltas = np.concatenate(delta_arrays)
    return int(all_deltas.min()), int(all_deltas.max())
