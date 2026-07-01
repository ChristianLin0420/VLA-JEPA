#!/usr/bin/env python3
"""Validate every DROID Parquet and rebuild full-data statistics/step caches.

The upstream LeRobot loader tolerates unreadable Parquets while deriving its
caches.  That is useful interactively but unsafe for a completion gate: a cache
can look valid while silently omitting episodes.  This script therefore checks
the exact episode manifest and reads every low-dimensional payload before it
atomically publishes caches and a content-bound completion marker.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq


DEFAULT_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot"
)
DEFAULT_STATE_DIR = Path(
    "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/all_robot_pipeline"
)
DATASET_NAME = "droid_lerobot"
DELETE_PAUSE_FRAME = True
EPSILON = 5e-4
CACHE_SCHEMA_VERSION = 2
PREFERRED_STEPS_NAME = "steps_332420bad1ab.pkl"
LEGACY_STEPS_NAME = "steps_2d5a34b904d2.pkl"


@dataclass(frozen=True)
class Episode:
    index: int
    length: int
    relative_path: str
    absolute_path: Path
    frame_offset: int


@dataclass
class PayloadResult:
    episode_index: int
    relative_path: str
    size: int
    rows: int
    content_sha256: str
    values: np.ndarray
    selected_steps: np.ndarray
    sums: np.ndarray
    sums_squared: np.ndarray
    minima: np.ndarray
    maxima: np.ndarray


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_pickle_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def backup(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}.{suffix}.bak")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.name}.{suffix}.{counter}.bak")
        counter += 1
    path.replace(target)
    print(f"[cache] preserved stale {path} as {target}", flush=True)
    return target


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"expected object in {path}:{line_number}")
            records.append(value)
    return records


def load_episode_manifest(dataset_dir: Path) -> tuple[dict[str, Any], list[Episode]]:
    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    expected_episodes = int(info["total_episodes"])
    expected_frames = int(info["total_frames"])
    chunk_size = int(info.get("chunks_size", 1000))
    data_template = str(info["data_path"])

    episode_records = read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    if len(episode_records) != expected_episodes:
        raise RuntimeError(
            f"episodes.jsonl has {len(episode_records)}/{expected_episodes} records"
        )

    tasks = read_jsonl(dataset_dir / "meta" / "tasks.jsonl")
    expected_tasks = int(info["total_tasks"])
    if len(tasks) != expected_tasks:
        raise RuntimeError(f"tasks.jsonl has {len(tasks)}/{expected_tasks} records")
    task_indices = [record.get("task_index") for record in tasks]
    if task_indices != list(range(expected_tasks)):
        raise RuntimeError("tasks.jsonl task_index values are not unique and contiguous")

    episodes: list[Episode] = []
    frame_offset = 0
    for expected_index, record in enumerate(episode_records):
        if record.get("episode_index") != expected_index:
            raise RuntimeError(
                "episodes.jsonl episode_index values are not unique and contiguous: "
                f"record {expected_index} contains {record.get('episode_index')!r}"
            )
        length = record.get("length")
        if not isinstance(length, int) or length <= 0:
            raise RuntimeError(f"episode {expected_index} has invalid length {length!r}")
        values = {
            "episode_chunk": expected_index // chunk_size,
            "episode_index": expected_index,
        }
        relative = data_template.format(**values)
        episodes.append(
            Episode(
                index=expected_index,
                length=length,
                relative_path=relative,
                absolute_path=dataset_dir / relative,
                frame_offset=frame_offset,
            )
        )
        frame_offset += length
    if frame_offset != expected_frames:
        raise RuntimeError(
            f"episode lengths sum to {frame_offset}, info.json expects {expected_frames}"
        )
    return info, episodes


def expected_columns(info: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        name
        for name, feature in info["features"].items()
        if not (isinstance(feature, dict) and feature.get("dtype") == "video")
    )


def manifest_digest(records: Iterable[tuple[int, str, int, int, str]]) -> str:
    digest = hashlib.sha256()
    for episode_index, relative_path, size, rows, content_sha256 in records:
        digest.update(
            f"{episode_index}\0{relative_path}\0{size}\0{rows}\0{content_sha256}\n".encode()
        )
    return digest.hexdigest()


def validate_footer(
    episode: Episode, columns: tuple[str, ...]
) -> tuple[int, str, int, int, str]:
    path = episode.absolute_path
    try:
        stat = path.stat()
    except OSError as exc:
        raise RuntimeError(f"missing Parquet {episode.relative_path}: {exc}") from exc
    if not path.is_file() or stat.st_size <= 0:
        raise RuntimeError(f"Parquet is not a nonempty regular file: {episode.relative_path}")
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise RuntimeError(f"unreadable Parquet footer {episode.relative_path}: {exc}") from exc
    if parquet.metadata.num_rows != episode.length:
        raise RuntimeError(
            f"{episode.relative_path} has {parquet.metadata.num_rows} rows; "
            f"episodes.jsonl expects {episode.length}"
        )
    if tuple(parquet.schema_arrow.names) != columns:
        raise RuntimeError(
            f"{episode.relative_path} schema columns {parquet.schema_arrow.names!r}; "
            f"expected {list(columns)!r}"
        )
    # Bind cache reuse to payload bytes, not merely path/size/row metadata.  The
    # complete DROID low-dimensional payload is small enough to hash directly.
    content_sha256 = sha256_file(path)
    return (
        episode.index,
        episode.relative_path,
        stat.st_size,
        episode.length,
        content_sha256,
    )


def run_bounded(
    function: Any,
    episodes: list[Episode],
    workers: int,
    *args: Any,
    batch_size: int = 1024,
) -> Iterable[Any]:
    """Run bounded batches so 92K Future objects are never retained at once."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(episodes), batch_size):
            batch = episodes[start : start + batch_size]
            futures = {executor.submit(function, episode, *args): episode for episode in batch}
            results: list[Any] = []
            for future in as_completed(futures):
                episode = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"DROID validation failed at episode {episode.index}: {exc}") from exc
            results.sort(
                key=lambda item: (
                    item.episode_index if hasattr(item, "episode_index") else item[0]
                )
            )
            yield from results
            completed = min(start + len(batch), len(episodes))
            if completed % 10_240 == 0 or completed == len(episodes):
                print(f"[cache] validated {completed}/{len(episodes)} Parquets", flush=True)


def footer_manifest(
    episodes: list[Episode], columns: tuple[str, ...], workers: int
) -> str:
    records = list(run_bounded(validate_footer, episodes, workers, columns))
    records.sort(key=lambda item: item[0])
    return manifest_digest(records)


def fixed_list_numpy(table: Any, column_name: str, width: int) -> np.ndarray:
    array = table[column_name].combine_chunks()
    list_size = getattr(array.type, "list_size", None)
    if list_size != width:
        raise RuntimeError(f"{column_name} has width {list_size}, expected {width}")
    if array.null_count or array.values.null_count:
        raise RuntimeError(
            f"{column_name} contains null list rows or null numeric values"
        )
    values = array.values.to_numpy(zero_copy_only=False).reshape(len(array), width)
    return np.asarray(values, dtype=np.float32)


def read_payload(
    episode: Episode,
    columns: tuple[str, ...],
    expected_tasks: int,
) -> PayloadResult:
    episode_index, relative_path, size, rows, content_sha256 = validate_footer(
        episode, columns
    )
    parquet = pq.ParquetFile(episode.absolute_path)
    try:
        table = parquet.read()
        state = fixed_list_numpy(table, "observation.state", 8)
        action = fixed_list_numpy(table, "action", 7)
        for scalar_name in (
            "episode_index",
            "frame_index",
            "index",
            "task_index",
            "timestamp",
        ):
            if table[scalar_name].null_count:
                raise RuntimeError(f"{scalar_name} contains null values")
        episode_values = table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False)
        frame_values = table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False)
        global_values = table["index"].combine_chunks().to_numpy(zero_copy_only=False)
        task_values = table["task_index"].combine_chunks().to_numpy(zero_copy_only=False)
        timestamps = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False)
    except Exception as exc:
        raise RuntimeError(f"cannot read full payload {relative_path}: {exc}") from exc

    if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(timestamps).all():
        raise RuntimeError(f"{relative_path} contains non-finite state/action/timestamp values")
    if not np.all(episode_values == episode_index):
        raise RuntimeError(f"{relative_path} contains a different episode_index")
    if not np.array_equal(frame_values, np.arange(rows, dtype=frame_values.dtype)):
        raise RuntimeError(f"{relative_path} frame_index is not contiguous from zero")
    expected_global = np.arange(
        episode.frame_offset, episode.frame_offset + rows, dtype=global_values.dtype
    )
    if not np.array_equal(global_values, expected_global):
        raise RuntimeError(f"{relative_path} global index is not contiguous")
    if np.any(task_values < 0) or np.any(task_values >= expected_tasks):
        raise RuntimeError(f"{relative_path} contains an out-of-range task_index")

    values = np.concatenate((state, action), axis=1)
    translation_changed = np.any(np.abs(action[:, :3]) > EPSILON, axis=1)
    gripper_changed = np.zeros(rows, dtype=bool)
    if rows > 1:
        gripper_changed[1:] = action[1:, 6] != action[:-1, 6]
    selected = np.flatnonzero(translation_changed | gripper_changed).astype(np.int32)
    values64 = values.astype(np.float64, copy=False)
    return PayloadResult(
        episode_index=episode_index,
        relative_path=relative_path,
        size=size,
        rows=rows,
        content_sha256=content_sha256,
        values=values,
        selected_steps=selected,
        sums=values64.sum(axis=0),
        sums_squared=np.square(values64).sum(axis=0),
        minima=values.min(axis=0),
        maxima=values.max(axis=0),
    )


def build_full_caches(
    dataset_dir: Path,
    state_dir: Path,
    info: dict[str, Any],
    episodes: list[Episode],
    columns: tuple[str, ...],
    workers: int,
) -> tuple[str, dict[str, Any], np.ndarray]:
    total_frames = int(info["total_frames"])
    expected_tasks = int(info["total_tasks"])
    mmap_path = state_dir / f"droid_state_action.{os.getpid()}.mmap"
    state_action = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(total_frames, 15))
    selected_by_episode: list[np.ndarray | None] = [None] * len(episodes)
    records: list[tuple[int, str, int, int, str]] = []
    sums = np.zeros(15, dtype=np.float64)
    sums_squared = np.zeros(15, dtype=np.float64)
    minima = np.full(15, np.inf, dtype=np.float64)
    maxima = np.full(15, -np.inf, dtype=np.float64)

    try:
        for result in run_bounded(
            read_payload, episodes, workers, columns, expected_tasks, batch_size=512
        ):
            episode = episodes[result.episode_index]
            frame_slice = slice(episode.frame_offset, episode.frame_offset + result.rows)
            state_action[frame_slice] = result.values
            selected_by_episode[result.episode_index] = result.selected_steps
            records.append(
                (
                    result.episode_index,
                    result.relative_path,
                    result.size,
                    result.rows,
                    result.content_sha256,
                )
            )
            sums += result.sums
            sums_squared += result.sums_squared
            minima = np.minimum(minima, result.minima)
            maxima = np.maximum(maxima, result.maxima)
        state_action.flush()

        if any(value is None for value in selected_by_episode):
            raise RuntimeError("internal error: not every episode produced a step selection")
        records.sort(key=lambda item: item[0])
        source_manifest = manifest_digest(records)

        means = sums / total_frames
        variances = np.maximum(sums_squared / total_frames - np.square(means), 0.0)
        standard_deviations = np.sqrt(variances)
        q01 = np.empty(15, dtype=np.float64)
        q99 = np.empty(15, dtype=np.float64)
        for dimension in range(15):
            q01[dimension], q99[dimension] = np.quantile(
                state_action[:, dimension], (0.01, 0.99), method="linear"
            )
            print(f"[cache] computed quantiles {dimension + 1}/15", flush=True)

        stats: dict[str, Any] = {}
        for name, start, end in (("observation.state", 0, 8), ("action", 8, 15)):
            stats[name] = {
                "mean": means[start:end].tolist(),
                "std": standard_deviations[start:end].tolist(),
                "min": minima[start:end].tolist(),
                "max": maxima[start:end].tolist(),
                "q01": q01[start:end].tolist(),
                "q99": q99[start:end].tolist(),
            }

        total_steps = sum(len(value) for value in selected_by_episode if value is not None)
        steps = np.empty((total_steps, 2), dtype=np.int32)
        cursor = 0
        for episode_index, selected in enumerate(selected_by_episode):
            assert selected is not None
            count = len(selected)
            steps[cursor : cursor + count, 0] = episode_index
            steps[cursor : cursor + count, 1] = selected
            cursor += count
        if cursor != total_steps or total_steps <= 0:
            raise RuntimeError(f"invalid rebuilt step count {cursor}/{total_steps}")
        return source_manifest, stats, steps
    finally:
        del state_action
        mmap_path.unlink(missing_ok=True)


def steps_config_key() -> str:
    value = str(
        sorted(
            {
                "delete_pause_frame": DELETE_PAUSE_FRAME,
                "dataset_name": DATASET_NAME,
            }.items()
        )
    )
    return hashlib.md5(value.encode()).hexdigest()[:12]


def source_revision(state_dir: Path) -> str:
    # The preparation pipeline persists immutable Hugging Face snapshots in a
    # versioned envelope.  Retain support for the early flat filename so an
    # interrupted pre-hardening run can still be inspected safely.
    for path in (state_dir / "hf_revisions.json", state_dir / "revisions.json"):
        if not path.exists():
            continue
        try:
            revisions = json.loads(path.read_text())
            if isinstance(revisions, dict) and isinstance(revisions.get("datasets"), dict):
                revisions = revisions["datasets"]
            value = revisions.get(DATASET_NAME)
            if isinstance(value, dict):
                value = value.get("revision") or value.get("sha")
            if value:
                return str(value)
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return "unknown"


def validate_stats(stats: Any) -> None:
    if not isinstance(stats, dict) or set(stats) != {"observation.state", "action"}:
        raise RuntimeError("stats cache does not contain exactly state and action")
    for name, width in (("observation.state", 8), ("action", 7)):
        values = stats[name]
        for statistic in ("mean", "std", "min", "max", "q01", "q99"):
            array = np.asarray(values.get(statistic), dtype=np.float64)
            if array.shape != (width,) or not np.isfinite(array).all():
                raise RuntimeError(f"invalid {name}.{statistic} in stats cache")


def marker_is_valid(
    marker_path: Path,
    stats_path: Path,
    steps_path: Path,
    expected_episodes: int,
    manifest: str,
    revision: str,
    config_key: str,
) -> bool:
    if not marker_path.is_file() or not stats_path.is_file() or not steps_path.is_file():
        return False
    try:
        state = json.loads(marker_path.read_text())
        expected = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "source_parquets": expected_episodes,
            "source_manifest_sha256": manifest,
            "source_revision": revision,
            "config_key": config_key,
            "delete_pause_frame": DELETE_PAUSE_FRAME,
            "stats_path": stats_path.name,
            "steps_path": steps_path.name,
            "stats_bytes": stats_path.stat().st_size,
            "steps_bytes": steps_path.stat().st_size,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            return False
        if state.get("stats_sha256") != sha256_file(stats_path):
            return False
        if state.get("steps_sha256") != sha256_file(steps_path):
            return False
        stats = json.loads(stats_path.read_text())
        validate_stats(stats)
        with steps_path.open("rb") as handle:
            cache = pickle.load(handle)
        steps = np.asarray(cache["steps"])
        if (
            cache.get("config_key") != config_key
            or cache.get("cache_schema_version") != CACHE_SCHEMA_VERSION
            or cache.get("delete_pause_frame") is not DELETE_PAUSE_FRAME
            or steps.ndim != 2
            or steps.shape[1] != 2
            or len(steps) != state.get("total_steps")
            or cache.get("total_steps") != state.get("total_steps")
            or cache.get("num_trajectories") != expected_episodes
        ):
            return False
        unique = int(np.unique(steps[:, 0]).size)
        return unique == state.get("unique_trajectory_ids")
    except Exception as exc:
        print(f"[cache] existing marker/cache validation failed: {exc}", flush=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_dir / "droid_cache.lock"
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        dataset_dir = args.data_root / DATASET_NAME
        info, episodes = load_episode_manifest(dataset_dir)
        columns = expected_columns(info)
        if columns != (
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ):
            raise RuntimeError(f"unexpected DROID non-video schema: {columns!r}")

        marker = args.state_dir / "droid_cache_complete.json"
        stats_path = dataset_dir / "meta" / "stats_gr00t.json"
        preferred_steps = dataset_dir / "meta" / PREFERRED_STEPS_NAME
        legacy_steps = dataset_dir / "meta" / LEGACY_STEPS_NAME
        revision = source_revision(args.state_dir)
        config_key = steps_config_key()

        # A valid marker gets a fresh footer/row-count manifest plus cache hashes.
        # This makes restart reuse safe without rereading 27M payload rows.
        if marker.exists():
            print("[cache] validating source manifest for existing cache marker", flush=True)
            source_manifest = footer_manifest(episodes, columns, args.workers)
            try:
                recorded = json.loads(marker.read_text())
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                print(f"[cache] ignoring malformed cache marker {marker}: {exc}", flush=True)
                recorded = {}
            recorded_steps_name = str(recorded.get("steps_path", ""))
            recorded_steps = dataset_dir / "meta" / recorded_steps_name
            if marker_is_valid(
                marker,
                stats_path,
                recorded_steps,
                len(episodes),
                source_manifest,
                revision,
                config_key,
            ):
                print(f"[cache] verified existing full-data cache marker {marker}", flush=True)
                return 0

        print(
            f"[cache] reading and validating all {len(episodes)} DROID Parquets ",
            f"({info['total_frames']} frames)",
            flush=True,
        )
        source_manifest, stats, steps = build_full_caches(
            dataset_dir, args.state_dir, info, episodes, columns, args.workers
        )
        validate_stats(stats)

        timestamp = datetime.now().astimezone().strftime("partial-%Y%m%dT%H%M%S%z")
        # A marker certifies a coherent stats/steps pair. Invalidate it before
        # replacing either member so no concurrent marker-aware consumer can
        # mistake a partially published rebuild for a complete cache.
        marker.unlink(missing_ok=True)
        backup(stats_path, timestamp)
        backup(preferred_steps, timestamp)
        backup(legacy_steps, timestamp)

        write_json_atomic(stats_path, stats)
        cache = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "config_key": config_key,
            "steps": steps,
            "num_trajectories": len(episodes),
            "total_steps": len(steps),
            "computed_timestamp": datetime.now().astimezone().isoformat(),
            "delete_pause_frame": DELETE_PAUSE_FRAME,
            "source_manifest_sha256": source_manifest,
        }
        write_pickle_atomic(preferred_steps, cache)

        unique_trajectory_ids = int(np.unique(steps[:, 0]).size)
        state = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "completed_at": datetime.now().astimezone().isoformat(),
            "source_parquets": len(episodes),
            "expected_episodes": int(info["total_episodes"]),
            "source_frames": int(info["total_frames"]),
            "source_manifest_sha256": source_manifest,
            "source_revision": revision,
            "config_key": config_key,
            "delete_pause_frame": DELETE_PAUSE_FRAME,
            "total_steps": len(steps),
            "unique_trajectory_ids": unique_trajectory_ids,
            "stats_path": stats_path.name,
            "steps_path": preferred_steps.name,
            "stats_bytes": stats_path.stat().st_size,
            "steps_bytes": preferred_steps.stat().st_size,
            "stats_sha256": sha256_file(stats_path),
            "steps_sha256": sha256_file(preferred_steps),
        }
        write_json_atomic(marker, state)
        print(f"[cache] DROID full-data cache complete: {state}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
