#!/usr/bin/env python3
"""Complete and verify the all_robot LeRobot mixture, then launch training once.

Completion is based on the exact paths described by each dataset's meta/info.json,
not on downloader progress output or approximate file counts.  The download mode
only requests categories that are incomplete and is safe to resume.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import getpass
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/lerobot"
)
DEFAULT_REPO_ROOT = Path(
    "/lustre/fsw/portfolios/edgeai/projects/"
    "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/VLA-JEPA"
)
DEFAULT_STATE_DIR = Path(
    "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/all_robot_pipeline"
)
SSV2_VIDEO_DIR = Path(
    "/lustre/fsw/portfolios/edgeai/users/chrislin/vlajepa_stage/datasets/ssv2/"
    "20bn-something-something-v2"
)
SSV2_LABELS = SSV2_VIDEO_DIR.parent / "labels_all.csv"
SSV2_EXPECTED_VIDEOS = 220_847
SSV2_EXPECTED_LABELS = 193_690

DATASETS = (
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
    "droid_lerobot",
    "bridge_orig_1.0.0_lerobot",
    "fractal20220817_data_0.1.0_lerobot",
)

REMOTE_REPOS = {
    "droid_lerobot": "IPEC-COMMUNITY/droid_lerobot",
    "bridge_orig_1.0.0_lerobot": "IPEC-COMMUNITY/bridge_orig_lerobot",
    "fractal20220817_data_0.1.0_lerobot": (
        "IPEC-COMMUNITY/fractal20220817_data_lerobot"
    ),
}

REMOTE_METADATA_PATHS = (
    "meta/info.json",
    "meta/episodes.jsonl",
    "meta/tasks.jsonl",
    "meta/stats.json",
)
HF_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
PIPELINE_SCHEMA_VERSION = 2
TRAIN_CONFIG = Path("scripts/config/vlajepa_cotrain_all.yaml")
TRAIN_RUN_ID = "vlajepa_cotrain_allv2"
TRAIN_EXPERIMENT_ID = "E20260626-allv2"


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def file_ok(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def present_payloads(directory: Path) -> dict[str, int]:
    """Return regular, non-empty entries and their sizes with one directory scan."""
    result: dict[str, int] = {}
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=True):
                        size = entry.stat(follow_symlinks=True).st_size
                        if size > 0:
                            result[entry.name] = size
                except OSError:
                    continue
    except OSError:
        pass
    return result


def listed_payloads(directory: Path, strict: bool) -> dict[str, int]:
    """List exact names cheaply during repair; stat/open only for final audits.

    Hugging Face publishes completed local-dir downloads with an atomic rename;
    partial files stay below ``.cache``.  Repeated repair passes therefore only
    need pathname membership.  The mandatory strict pass still verifies regular
    non-empty files, sizes, and container signatures before cache generation or
    submission.
    """
    if strict:
        return present_payloads(directory)
    try:
        return {name: 0 for name in os.listdir(directory)}
    except OSError:
        return {}


def payload_container_ok(path: Path, kind: str) -> bool:
    """Check cheap container signatures without decoding the full payload."""
    try:
        size = path.stat().st_size
        if kind == "parquet":
            if size < 12:
                return False
            with path.open("rb") as handle:
                if handle.read(4) != b"PAR1":
                    return False
                handle.seek(-4, os.SEEK_END)
                return handle.read(4) == b"PAR1"
        if kind == "mp4":
            if size < 12:
                return False
            with path.open("rb") as handle:
                return handle.read(12)[4:8] == b"ftyp"
        if kind == "webm":
            if size < 4:
                return False
            with path.open("rb") as handle:
                return handle.read(4) == b"\x1aE\xdf\xa3"
    except OSError:
        return False
    raise ValueError(f"unsupported payload kind: {kind}")


def validate_json_object(path: Path) -> tuple[bool, str | None]:
    if not file_ok(path):
        return False, f"missing or empty {path}"
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"invalid JSON in {path}: {exc}"
    if not isinstance(value, dict) or not value:
        return False, f"expected a non-empty JSON object in {path}"
    return True, None


def validate_indexed_jsonl(
    path: Path,
    expected_lines: int,
    index_key: str,
    record_kind: str,
) -> tuple[int, bool, str | None]:
    count = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for count, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    return count, False, f"invalid {record_kind} JSONL at line {count}: {exc}"
                if not isinstance(record, dict):
                    return count, False, f"non-object {record_kind} record at line {count}"
                expected_index = count - 1
                if record.get(index_key) != expected_index:
                    return (
                        count,
                        False,
                        f"{record_kind} {index_key} mismatch at line {count}: "
                        f"expected {expected_index}, got {record.get(index_key)!r}",
                    )
                if record_kind == "episode":
                    length = record.get("length")
                    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
                        return count, False, f"invalid episode length at line {count}: {length!r}"
                    if not isinstance(record.get("tasks"), list):
                        return count, False, f"invalid episode tasks at line {count}"
                elif record_kind == "task" and not isinstance(record.get("task"), str):
                    return count, False, f"invalid task text at line {count}"
    except (OSError, UnicodeError) as exc:
        return 0, False, f"cannot read {path}: {exc}"
    if count != expected_lines:
        return count, False, f"{path} has {count} records, expected {expected_lines}"
    return count, True, None


def hf_local_metadata_path(local_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    return local_dir / ".cache" / "huggingface" / "download" / relative.parent / (
        relative.name + ".metadata"
    )


def read_local_hf_revision(local_dir: Path, relative_path: str = "meta/info.json") -> str | None:
    metadata_path = hf_local_metadata_path(local_dir, relative_path)
    try:
        revision = metadata_path.open(encoding="utf-8").readline().strip()
    except (OSError, UnicodeError):
        return None
    return revision if HF_REVISION_RE.fullmatch(revision) else None


def exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_rate_limited(exc: BaseException) -> bool:
    for item in exception_chain(exc):
        response = getattr(item, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True
        message = str(item).lower()
        if "429 client error" in message or "too many requests" in message:
            return True
    return False


@dataclass
class Audit:
    name: str
    expected_episodes: int = 0
    data_present: int = 0
    data_missing: int = 0
    data_bytes: int = 0
    expected_videos: int = 0
    videos_present: int = 0
    videos_missing: int = 0
    video_bytes: int = 0
    episodes_metadata_lines: int = 0
    episodes_metadata_valid: bool = False
    expected_tasks: int = 0
    tasks_metadata_lines: int = 0
    tasks_metadata_valid: bool = False
    modality_present: bool = False
    stats_present: bool = False
    stats_gr00t_present: bool = False
    info_present: bool = False
    strict_payloads: bool = False
    pinned_revision: str | None = None
    local_revision: str | None = None
    revision_valid: bool = True
    metadata_errors: tuple[str, ...] = ()
    missing_samples: tuple[str, ...] = ()
    error: str | None = None

    @property
    def metadata_complete(self) -> bool:
        return (
            self.info_present
            and self.episodes_metadata_valid
            and self.tasks_metadata_valid
            and self.modality_present
            and self.stats_present
            and self.stats_gr00t_present
            and self.revision_valid
        )

    @property
    def complete(self) -> bool:
        return (
            self.error is None
            and self.data_missing == 0
            and self.videos_missing == 0
            and self.metadata_complete
        )


def audit_dataset(
    root: Path,
    name: str,
    sample_limit: int = 12,
    *,
    strict_payloads: bool = False,
    expected_revision: str | None = None,
) -> Audit:
    dataset_dir = root / name
    info_path = dataset_dir / "meta" / "info.json"
    result = Audit(
        name=name,
        info_present=file_ok(info_path),
        strict_payloads=strict_payloads,
        pinned_revision=expected_revision,
    )
    if not result.info_present:
        result.error = f"missing or empty {info_path}"
        result.metadata_errors = (result.error,)
        return result

    try:
        info: dict[str, Any] = json.loads(info_path.read_text())
        if not isinstance(info, dict):
            raise TypeError("top-level value is not an object")
        result.expected_episodes = int(info["total_episodes"])
        result.expected_videos = int(info["total_videos"])
        result.expected_tasks = int(info["total_tasks"])
        chunk_size = int(info.get("chunks_size", 1000))
        data_template = str(info["data_path"])
        video_template = str(info["video_path"])
        if result.expected_episodes <= 0:
            raise ValueError(f"non-positive total_episodes={result.expected_episodes}")
        if result.expected_videos < 0 or result.expected_tasks < 0:
            raise ValueError("negative total_videos or total_tasks")
        if chunk_size <= 0:
            raise ValueError(f"non-positive chunks_size={chunk_size}")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        result.error = f"invalid info.json: {exc}"
        result.info_present = False
        result.metadata_errors = (result.error,)
        return result

    metadata_errors: list[str] = []
    result.local_revision = read_local_hf_revision(dataset_dir)
    if expected_revision is not None:
        result.revision_valid = result.local_revision == expected_revision
        if not result.revision_valid:
            metadata_errors.append(
                f"local HF revision {result.local_revision!r} does not match pinned "
                f"revision {expected_revision}"
            )

    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    computed_videos = result.expected_episodes * len(video_keys)
    if computed_videos != result.expected_videos:
        result.error = (
            f"video schema mismatch: {len(video_keys)} keys x "
            f"{result.expected_episodes} episodes = {computed_videos}, "
            f"info expects {result.expected_videos}"
        )
        result.metadata_errors = tuple(metadata_errors + [result.error])
        return result

    samples: list[str] = []
    strict_executor = (
        ThreadPoolExecutor(max_workers=32) if result.strict_payloads else None
    )
    for chunk_start in range(0, result.expected_episodes, chunk_size):
        chunk_end = min(chunk_start + chunk_size, result.expected_episodes)
        first_values = {
            "episode_chunk": chunk_start // chunk_size,
            "episode_index": chunk_start,
        }
        first_data = Path(data_template.format(**first_values))
        present_data = listed_payloads(
            dataset_dir / first_data.parent, result.strict_payloads
        )
        data_relatives = [
            Path(
                data_template.format(
                    episode_chunk=episode_index // chunk_size,
                    episode_index=episode_index,
                )
            )
            for episode_index in range(chunk_start, chunk_end)
        ]
        strict_data_names: set[str] = set()
        if strict_executor is not None:
            checks = strict_executor.map(
                lambda relative: payload_container_ok(
                    dataset_dir / relative, "parquet"
                ),
                data_relatives,
            )
            strict_data_names = {
                relative.name
                for relative, valid in zip(data_relatives, checks)
                if valid
            }
        for episode_index in range(chunk_start, chunk_end):
            values = {
                "episode_chunk": episode_index // chunk_size,
                "episode_index": episode_index,
            }
            relative = Path(data_template.format(**values))
            payload_size = present_data.get(relative.name)
            valid = payload_size is not None
            if valid and strict_payloads:
                valid = relative.name in strict_data_names
            if valid:
                result.data_present += 1
                result.data_bytes += int(payload_size)
            else:
                result.data_missing += 1
                if len(samples) < sample_limit:
                    samples.append(str(relative))

        for video_key in video_keys:
            first_video = Path(video_template.format(**first_values, video_key=video_key))
            present_videos = listed_payloads(
                dataset_dir / first_video.parent, result.strict_payloads
            )
            video_relatives = [
                Path(
                    video_template.format(
                        episode_chunk=episode_index // chunk_size,
                        episode_index=episode_index,
                        video_key=video_key,
                    )
                )
                for episode_index in range(chunk_start, chunk_end)
            ]
            strict_video_names: set[str] = set()
            if strict_executor is not None:
                checks = strict_executor.map(
                    lambda relative: payload_container_ok(dataset_dir / relative, "mp4"),
                    video_relatives,
                )
                strict_video_names = {
                    relative.name
                    for relative, valid in zip(video_relatives, checks)
                    if valid
                }
            for episode_index in range(chunk_start, chunk_end):
                values = {
                    "episode_chunk": episode_index // chunk_size,
                    "episode_index": episode_index,
                }
                relative = Path(video_template.format(**values, video_key=video_key))
                payload_size = present_videos.get(relative.name)
                valid = payload_size is not None
                if valid and strict_payloads:
                    valid = relative.name in strict_video_names
                if valid:
                    result.videos_present += 1
                    result.video_bytes += int(payload_size)
                else:
                    result.videos_missing += 1
                    if len(samples) < sample_limit:
                        samples.append(str(relative))

    if strict_executor is not None:
        strict_executor.shutdown(wait=True, cancel_futures=True)

    (
        result.episodes_metadata_lines,
        result.episodes_metadata_valid,
        metadata_error,
    ) = validate_indexed_jsonl(
        dataset_dir / "meta" / "episodes.jsonl",
        result.expected_episodes,
        "episode_index",
        "episode",
    )
    if metadata_error:
        metadata_errors.append(metadata_error)
    (
        result.tasks_metadata_lines,
        result.tasks_metadata_valid,
        metadata_error,
    ) = validate_indexed_jsonl(
        dataset_dir / "meta" / "tasks.jsonl",
        result.expected_tasks,
        "task_index",
        "task",
    )
    if metadata_error:
        metadata_errors.append(metadata_error)

    result.modality_present, metadata_error = validate_json_object(
        dataset_dir / "meta" / "modality.json"
    )
    if metadata_error:
        metadata_errors.append(metadata_error)
    stats_path = dataset_dir / "meta" / "stats.json"
    if expected_revision is None and not stats_path.exists():
        # The local LIBERO conversions predate this optional upstream file; the
        # training loader consumes stats_gr00t.json below.
        result.stats_present = True
    else:
        result.stats_present, metadata_error = validate_json_object(stats_path)
        if metadata_error:
            metadata_errors.append(metadata_error)
    result.stats_gr00t_present, metadata_error = validate_json_object(
        dataset_dir / "meta" / "stats_gr00t.json"
    )
    if metadata_error:
        metadata_errors.append(metadata_error)
    result.metadata_errors = tuple(metadata_errors)
    result.missing_samples = tuple(samples)
    return result


def print_audit(result: Audit) -> None:
    status = "COMPLETE" if result.complete else "INCOMPLETE"
    log(
        f"{status} {result.name}: "
        f"data={result.data_present}/{result.expected_episodes}, "
        f"videos={result.videos_present}/{result.expected_videos}, "
        f"episodes_meta={result.episodes_metadata_lines}/{result.expected_episodes}, "
        f"tasks_meta={result.tasks_metadata_lines}/{result.expected_tasks}, "
        f"modality={result.modality_present}, stats={result.stats_present}, "
        f"stats_gr00t={result.stats_gr00t_present}, strict={result.strict_payloads}, "
        f"revision={result.local_revision or '-'}"
    )
    if result.error:
        log(f"  error: {result.error}")
    for error in result.metadata_errors[:8]:
        if error != result.error:
            log(f"  metadata error: {error}")
    if result.missing_samples:
        log(f"  missing samples: {', '.join(result.missing_samples)}")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def audit_ssv2(*, strict_payloads: bool = False) -> dict[str, Any]:
    video_names: set[str] = set()
    candidates: dict[str, tuple[Path, int]] = {}
    invalid_videos = 0
    video_bytes = 0
    try:
        with os.scandir(SSV2_VIDEO_DIR) as entries:
            for entry in entries:
                if not entry.name.lower().endswith(".webm"):
                    continue
                path = Path(entry.path)
                try:
                    size = entry.stat(follow_symlinks=True).st_size
                    if entry.is_file(follow_symlinks=True) and size > 0:
                        candidates[entry.name] = (path, size)
                    else:
                        invalid_videos += 1
                except OSError:
                    invalid_videos += 1
    except OSError:
        pass
    if strict_payloads:
        with ThreadPoolExecutor(max_workers=32) as executor:
            items = list(candidates.items())
            checks = executor.map(
                lambda item: payload_container_ok(item[1][0], "webm"), items
            )
            for (name, (_path, size)), valid in zip(items, checks):
                if valid:
                    video_names.add(name)
                    video_bytes += size
                else:
                    invalid_videos += 1
    else:
        video_names.update(candidates)
        video_bytes = sum(size for _path, size in candidates.values())
    label_ids: set[int] = set()
    labels_valid = True
    try:
        with SSV2_LABELS.open() as handle:
            for line in handle:
                value = line.partition(";")[0].strip()
                if value:
                    label_ids.add(int(value))
    except (OSError, ValueError):
        labels_valid = False
    labeled_missing = sum(1 for item in label_ids if f"{item}.webm" not in video_names)
    result = {
        "video_count": len(video_names),
        "video_bytes": video_bytes,
        "invalid_videos": invalid_videos,
        "expected_videos": SSV2_EXPECTED_VIDEOS,
        "label_count": len(label_ids),
        "expected_labels": SSV2_EXPECTED_LABELS,
        "labeled_missing": labeled_missing,
        "labels_valid": labels_valid,
        "strict_payloads": strict_payloads,
    }
    result["complete"] = (
        result["video_count"] == SSV2_EXPECTED_VIDEOS
        and result["label_count"] == SSV2_EXPECTED_LABELS
        and labeled_missing == 0
        and invalid_videos == 0
        and labels_valid
    )
    log(
        f"{'COMPLETE' if result['complete'] else 'INCOMPLETE'} SSV2: "
        f"videos={result['video_count']}/{SSV2_EXPECTED_VIDEOS}, "
        f"labels={result['label_count']}/{SSV2_EXPECTED_LABELS}, "
        f"labeled_missing={labeled_missing}, invalid={invalid_videos}, "
        f"strict={strict_payloads}"
    )
    return result


def hub_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".hf_token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def load_pinned_revisions(state_dir: Path) -> dict[str, dict[str, str]]:
    path = state_dir / "hf_revisions.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError(f"invalid Hugging Face revision state: {path}")
    datasets = value.get("datasets")
    if not isinstance(datasets, dict):
        raise RuntimeError(f"invalid dataset revision map: {path}")
    result: dict[str, dict[str, str]] = {}
    for name, record in datasets.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise RuntimeError(f"invalid revision record in {path}: {name!r}")
        repo_id = record.get("repo_id")
        revision = record.get("revision")
        if not isinstance(repo_id, str) or not isinstance(revision, str):
            raise RuntimeError(f"invalid revision record for {name} in {path}")
        if not HF_REVISION_RE.fullmatch(revision):
            raise RuntimeError(f"invalid pinned revision for {name}: {revision!r}")
        result[name] = {str(key): str(item) for key, item in record.items()}
    return result


def pin_dataset_revision(
    state_dir: Path,
    data_root: Path,
    name: str,
    repo_id: str,
) -> str:
    records = load_pinned_revisions(state_dir)
    existing = records.get(name)
    if existing is not None:
        if existing["repo_id"] != repo_id:
            raise RuntimeError(
                f"pinned repository mismatch for {name}: "
                f"{existing['repo_id']} != {repo_id}"
            )
        return existing["revision"]

    # Resume the exact snapshot that produced the existing metadata whenever
    # possible. This prevents a moving Hub `main` branch from mixing revisions.
    local_revision = read_local_hf_revision(data_root / name)
    source = "local-info-metadata"
    if local_revision is None:
        from huggingface_hub import HfApi

        local_revision = HfApi(token=hub_token()).repo_info(
            repo_id, repo_type="dataset"
        ).sha
        source = "hub-main"
    if not isinstance(local_revision, str) or not HF_REVISION_RE.fullmatch(local_revision):
        raise RuntimeError(f"could not resolve an immutable Hub revision for {repo_id}")

    records[name] = {
        "repo_id": repo_id,
        "revision": local_revision,
        "source": source,
        "pinned_at": datetime.now().astimezone().isoformat(),
    }
    write_json_atomic(
        state_dir / "hf_revisions.json",
        {"schema_version": 1, "datasets": records},
    )
    log(f"pinned {name} ({repo_id}) at {local_revision} from {source}")
    return local_revision


def pinned_revision_map(state_dir: Path) -> dict[str, str]:
    return {
        name: record["revision"]
        for name, record in load_pinned_revisions(state_dir).items()
    }


def missing_remote_paths(root: Path, name: str, result: Audit) -> list[str]:
    dataset_dir = root / name
    if not result.info_present:
        return ["meta/info.json"]
    if result.pinned_revision and not result.revision_valid:
        if result.local_revision is None:
            return ["meta/info.json"]
        raise RuntimeError(
            f"refusing to mix Hub revisions for {name}: local info metadata is "
            f"{result.local_revision}, pinned revision is {result.pinned_revision}"
        )
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    chunk_size = int(info.get("chunks_size", 1000))
    data_template = str(info["data_path"])
    video_template = str(info["video_path"])
    video_keys = [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]

    missing: list[str] = []
    for chunk_start in range(0, result.expected_episodes, chunk_size):
        chunk_end = min(chunk_start + chunk_size, result.expected_episodes)
        first_values = {
            "episode_chunk": chunk_start // chunk_size,
            "episode_index": chunk_start,
        }
        if result.data_missing:
            first_data = Path(data_template.format(**first_values))
            present_data = listed_payloads(
                dataset_dir / first_data.parent, result.strict_payloads
            )
            for episode_index in range(chunk_start, chunk_end):
                values = {
                    "episode_chunk": episode_index // chunk_size,
                    "episode_index": episode_index,
                }
                relative = Path(data_template.format(**values))
                valid = relative.name in present_data
                if valid and result.strict_payloads:
                    valid = payload_container_ok(dataset_dir / relative, "parquet")
                if not valid:
                    missing.append(str(relative))
        if result.videos_missing:
            for video_key in video_keys:
                first_video = Path(video_template.format(**first_values, video_key=video_key))
                present_videos = listed_payloads(
                    dataset_dir / first_video.parent, result.strict_payloads
                )
                for episode_index in range(chunk_start, chunk_end):
                    values = {
                        "episode_chunk": episode_index // chunk_size,
                        "episode_index": episode_index,
                    }
                    relative = Path(video_template.format(**values, video_key=video_key))
                    valid = relative.name in present_videos
                    if valid and result.strict_payloads:
                        valid = payload_container_ok(dataset_dir / relative, "mp4")
                    if not valid:
                        missing.append(str(relative))

    if not result.episodes_metadata_valid:
        missing.append("meta/episodes.jsonl")
    if not result.tasks_metadata_valid:
        missing.append("meta/tasks.jsonl")
    if not result.stats_present:
        missing.append("meta/stats.json")
    return list(dict.fromkeys(missing))


def download_pass(
    repo_id: str,
    revision: str,
    local_dir: Path,
    filenames: list[str],
    workers: int,
    max_files: int,
) -> tuple[int, int, bool]:
    from huggingface_hub import hf_hub_download

    token = hub_token()
    batch = filenames[:max_files]
    log(
        f"download pass {repo_id}: exact_missing={len(filenames)}, "
        f"batch={len(batch)}, workers={workers}, revision={revision}"
    )

    def fetch(filename: str) -> str:
        destination = local_dir / filename
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=token,
            # Every requested path failed exact validation. Preserve resume for
            # absent files, but replace a malformed existing final payload.
            force_download=destination.exists(),
        )
        return filename

    failures: list[tuple[str, str]] = []
    completed = 0
    failed = 0
    rate_limited = False
    next_index = 0
    pending: dict[Future[str], str] = {}
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        def fill_window() -> None:
            nonlocal next_index
            while not rate_limited and len(pending) < workers and next_index < len(batch):
                name = batch[next_index]
                next_index += 1
                pending[executor.submit(fetch, name)] = name

        fill_window()
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                name = pending.pop(future)
                try:
                    future.result()
                    completed += 1
                except Exception as exc:
                    failed += 1
                    if is_rate_limited(exc):
                        rate_limited = True
                    if len(failures) < 8:
                        failures.append((name, f"{type(exc).__name__}: {exc}"))
            if rate_limited:
                for future in pending:
                    future.cancel()
            else:
                fill_window()
            if completed and completed % 250 == 0:
                log(
                    f"download batch progress: completed={completed}/{len(batch)}, "
                    f"failed={failed}, submitted={next_index}"
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    log(f"download batch finished: completed={completed}, failed={failed}, rate_limited={rate_limited}")
    for name, error in failures:
        log(f"  failed {name}: {error[:500].replace(chr(10), ' ')}")
    return completed, failed, rate_limited


def complete_remote_dataset(
    root: Path,
    name: str,
    repo_id: str,
    revision: str,
    workers: int,
    retry_seconds: int,
    max_attempts: int,
    max_files_per_attempt: int,
) -> Audit:
    result = audit_dataset(root, name, expected_revision=revision)
    print_audit(result)
    attempt = 0
    while True:
        if result.complete and not result.strict_payloads:
            log(f"running strict payload/container audit for {name}")
            result = audit_dataset(
                root,
                name,
                strict_payloads=True,
                expected_revision=revision,
            )
            print_audit(result)
        if result.complete:
            return result
        filenames = missing_remote_paths(root, name, result)
        if not filenames:
            log(f"{name} is incomplete but has no remotely recoverable missing paths")
            return result
        attempt += 1
        if attempt > max_attempts:
            log(f"giving up {name} after {max_attempts} download attempts")
            return result
        failed = 0
        rate_limited = False
        try:
            _, failed, rate_limited = download_pass(
                repo_id,
                revision,
                root / name,
                filenames,
                workers,
                max_files_per_attempt,
            )
        except Exception as exc:  # downloader errors are retried after exact re-audit
            message = str(exc).replace("\n", " ")
            log(f"download attempt {attempt} raised {type(exc).__name__}: {message[:500]}")
            failed = 1
            rate_limited = is_rate_limited(exc)

        result = audit_dataset(
            root,
            name,
            strict_payloads=result.strict_payloads,
            expected_revision=revision,
        )
        print_audit(result)
        if not result.complete:
            delay = retry_seconds if rate_limited else (60 if failed else 20)
            log(f"{name} remains incomplete; cooling down {delay}s")
            time.sleep(delay)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def pipeline_identity(
    repo_root: Path,
    data_root: Path,
    audits: list[Audit],
    ssv2_audit: dict[str, Any],
    revisions: dict[str, str],
) -> dict[str, Any]:
    code_paths = sorted((repo_root / "starVLA").rglob("*.py"))
    code_paths.extend(
        repo_root / relative
        for relative in (
            "cluster/prepare_all_robot_data.py",
            "cluster/rebuild_droid_cache.py",
            "cluster/smoke_all_robot_data.py",
            "cluster/launch_vlajepa_cotrain_8gpu.sbatch",
            "cluster/eval_after_all.sbatch",
            "cluster/export_vlajepa_ckpt.py",
        )
    )
    code_manifest: dict[str, str] = {}
    for path in sorted(set(code_paths)):
        if path.is_file():
            code_manifest[str(path.relative_to(repo_root))] = sha256_file(path)

    config_path = repo_root / TRAIN_CONFIG
    if not config_path.is_file():
        raise RuntimeError(f"training config is missing: {config_path}")
    config_manifest = {str(TRAIN_CONFIG): sha256_file(config_path)}

    metadata_names = (
        *REMOTE_METADATA_PATHS,
        "meta/modality.json",
        "meta/stats_gr00t.json",
    )
    dataset_manifest: dict[str, Any] = {}
    for audit in sorted(audits, key=lambda item: item.name):
        dataset_dir = data_root / audit.name
        metadata: dict[str, dict[str, Any]] = {}
        for relative in metadata_names:
            path = dataset_dir / relative
            if not path.is_file():
                if relative == "meta/stats.json" and audit.pinned_revision is None:
                    metadata[relative] = {"missing_optional": True}
                    continue
                raise RuntimeError(f"cannot fingerprint missing metadata: {path}")
            metadata[relative] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        dataset_manifest[audit.name] = {
            "expected_episodes": audit.expected_episodes,
            "data_present": audit.data_present,
            "data_bytes": audit.data_bytes,
            "expected_videos": audit.expected_videos,
            "videos_present": audit.videos_present,
            "video_bytes": audit.video_bytes,
            "pinned_revision": revisions.get(audit.name),
            "metadata": metadata,
        }

    data_manifest = {
        "datasets": dataset_manifest,
        "hf_revisions": revisions,
        "ssv2": {
            "video_count": ssv2_audit["video_count"],
            "video_bytes": ssv2_audit["video_bytes"],
            "label_count": ssv2_audit["label_count"],
            "labels_sha256": sha256_file(SSV2_LABELS),
        },
    }
    fingerprints = {
        "code": object_fingerprint(code_manifest),
        "config": object_fingerprint(config_manifest),
        "data": object_fingerprint(data_manifest),
    }
    pipeline = object_fingerprint(
        {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "fingerprints": fingerprints,
            "run_id": TRAIN_RUN_ID,
            "experiment_id": TRAIN_EXPERIMENT_ID,
        }
    )
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "pipeline_fingerprint": pipeline,
        "code_fingerprint": fingerprints["code"],
        "config_fingerprint": fingerprints["config"],
        "data_fingerprint": fingerprints["data"],
        "run_id": TRAIN_RUN_ID,
        "experiment_id": TRAIN_EXPERIMENT_ID,
    }


def query_slurm_jobs(comment: str, start_date: str) -> dict[str, dict[str, str]]:
    user = os.environ.get("USER") or getpass.getuser()
    commands = (
        (
            "squeue",
            [
                "squeue",
                "--noheader",
                f"--user={user}",
                "--format=%A|%k|%T",
            ],
        ),
        (
            "sacct",
            [
                "sacct",
                "--noheader",
                f"--user={user}",
                f"--starttime={start_date}",
                "-X",
                "--parsable2",
                "--format=JobIDRaw,Comment%128,State",
            ],
        ),
    )
    matches: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for source, command in commands:
        try:
            output = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"{source}: {exc}")
            continue
        for line in output.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            job_id, found_comment, state = parts[:3]
            job_id = job_id.split(".", 1)[0].strip()
            if found_comment.strip() != comment or not job_id.isdigit():
                continue
            matches[job_id] = {
                "job_id": job_id,
                "state": state.strip(),
                "source": source,
            }
    if errors:
        raise RuntimeError(
            "refusing Slurm submission because reconciliation was incomplete: "
            + "; ".join(errors)
        )
    return matches


def parse_sbatch_job_id(output: str) -> str:
    for line in reversed(output.splitlines()):
        candidate = line.strip().split(";", 1)[0]
        if candidate.isdigit():
            return candidate
    raise RuntimeError(f"unexpected sbatch response: {output!r}")


def reconcile_or_submit(
    *,
    kind: str,
    receipt: dict[str, Any],
    receipt_path: Path,
    repo_root: Path,
    command: list[str],
) -> tuple[str, str]:
    comment = str(receipt[f"{kind}_comment"])
    recorded_job = str(receipt.get(f"{kind}_job", ""))
    start_date = str(receipt["created_at"])[:10]

    def reconcile() -> dict[str, dict[str, str]]:
        matches = query_slurm_jobs(comment, start_date)
        if len(matches) > 1:
            raise RuntimeError(
                f"multiple Slurm jobs share idempotency comment {comment!r}: "
                f"{sorted(matches)}"
            )
        return matches

    matches = reconcile()
    if recorded_job:
        if matches and recorded_job not in matches:
            raise RuntimeError(
                f"receipt records {kind} job {recorded_job}, but comment {comment!r} "
                f"belongs to {sorted(matches)}"
            )
        state = matches.get(recorded_job, {}).get(
            "state", str(receipt.get(f"{kind}_state", "RECORDED"))
        )
        receipt[f"{kind}_state"] = state
        receipt[f"{kind}_reconciled_at"] = datetime.now().astimezone().isoformat()
        write_json_atomic(receipt_path, receipt)
        log(f"{kind} job already recorded as {recorded_job} ({state}); not resubmitting")
        return recorded_job, state

    if matches:
        job_id, match = next(iter(matches.items()))
        receipt[f"{kind}_job"] = job_id
        receipt[f"{kind}_state"] = match["state"]
        receipt[f"{kind}_recovered_at"] = datetime.now().astimezone().isoformat()
        write_json_atomic(receipt_path, receipt)
        log(f"recovered {kind} job {job_id} from Slurm comment {comment!r}")
        return job_id, match["state"]

    # A prior process may have died after sbatch accepted the job but before the
    # receipt was updated. Give squeue/sacct time to expose that allocation.
    if receipt.get(f"{kind}_submit_started_at"):
        for _ in range(3):
            time.sleep(5)
            matches = reconcile()
            if matches:
                job_id, match = next(iter(matches.items()))
                receipt[f"{kind}_job"] = job_id
                receipt[f"{kind}_state"] = match["state"]
                receipt[f"{kind}_recovered_at"] = datetime.now().astimezone().isoformat()
                write_json_atomic(receipt_path, receipt)
                log(f"recovered delayed {kind} job {job_id} from Slurm")
                return job_id, match["state"]

    receipt[f"{kind}_submit_started_at"] = datetime.now().astimezone().isoformat()
    write_json_atomic(receipt_path, receipt)
    output = subprocess.check_output(
        command,
        cwd=repo_root,
        text=True,
        stderr=subprocess.STDOUT,
    )
    job_id = parse_sbatch_job_id(output)
    receipt[f"{kind}_job"] = job_id
    receipt[f"{kind}_state"] = "SUBMITTED"
    receipt[f"{kind}_submitted_at"] = datetime.now().astimezone().isoformat()
    write_json_atomic(receipt_path, receipt)
    log(f"submitted {kind} job {job_id} with comment {comment!r}")
    return job_id, "SUBMITTED"


def submit_pipeline(
    repo_root: Path,
    state_dir: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = state_dir / "launch_receipt.json"
    fingerprint = str(identity["pipeline_fingerprint"])
    token = fingerprint[:24]
    receipt: dict[str, Any]
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise RuntimeError(
                f"legacy or invalid launch receipt requires manual reconciliation: {receipt_path}"
            )
        if receipt.get("pipeline_fingerprint") != fingerprint:
            raise RuntimeError(
                "launch receipt fingerprint does not match current code/config/data; "
                f"refusing to reuse {receipt_path}"
            )
    else:
        receipt = {
            **identity,
            "created_at": datetime.now().astimezone().isoformat(),
            "train_comment": f"vla-jepa-allv2:{token}:train",
            "eval_comment": f"vla-jepa-allv2:{token}:eval",
            "train_job_name": f"vj-all-{token[:12]}-train",
            "eval_job_name": f"vj-all-{token[:12]}-eval",
        }
        # Persist the deterministic intent before any external side effect.
        write_json_atomic(receipt_path, receipt)

    train_command = [
        "sbatch",
        "--parsable",
        f"--comment={receipt['train_comment']}",
        f"--job-name={receipt['train_job_name']}",
        "--export=ALL,CONFIG=./scripts/config/vlajepa_cotrain_all.yaml,"
        f"RUN_ID={TRAIN_RUN_ID},EXP_ID={TRAIN_EXPERIMENT_ID}",
        "cluster/launch_vlajepa_cotrain_8gpu.sbatch",
    ]
    train_job, train_state = reconcile_or_submit(
        kind="train",
        receipt=receipt,
        receipt_path=receipt_path,
        repo_root=repo_root,
        command=train_command,
    )

    failed_states = {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
    normalized_train_state = train_state.split()[0].rstrip("+")
    if normalized_train_state in failed_states and not receipt.get("eval_job"):
        raise RuntimeError(
            f"training job {train_job} is terminal with state {train_state}; "
            "refusing to submit evaluation"
        )

    eval_command = [
        "sbatch",
        "--parsable",
        f"--comment={receipt['eval_comment']}",
        f"--job-name={receipt['eval_job_name']}",
        f"--dependency=afterok:{train_job}",
        "cluster/eval_after_all.sbatch",
    ]
    eval_job, _ = reconcile_or_submit(
        kind="eval",
        receipt=receipt,
        receipt_path=receipt_path,
        repo_root=repo_root,
        command=eval_command,
    )
    log(f"evaluation job {eval_job} is bound afterok:{train_job}")
    return receipt


def ensure_droid_cache(repo_root: Path, data_root: Path, state_dir: Path) -> None:
    command = [
        sys.executable,
        str(repo_root / "cluster" / "rebuild_droid_cache.py"),
        "--data-root",
        str(data_root),
        "--state-dir",
        str(state_dir),
    ]
    log("verifying/rebuilding DROID full-data statistics and step cache")
    subprocess.check_call(command, cwd=repo_root)


def run_data_smoke(repo_root: Path, state_dir: Path) -> None:
    command = [
        sys.executable,
        str(repo_root / "cluster" / "smoke_all_robot_data.py"),
        "--config",
        str(repo_root / "scripts" / "config" / "vlajepa_cotrain_all.yaml"),
        "--state-dir",
        str(state_dir),
    ]
    log("running representative decode smoke across all_robot and SSV2")
    subprocess.check_call(command, cwd=repo_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-seconds", type=int, default=330)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--max-files-per-attempt", type=int, default=2500)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.retry_seconds < 0:
        parser.error("--retry-seconds must be non-negative")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.max_files_per_attempt < 1:
        parser.error("--max-files-per-attempt must be at least 1")
    return args


@contextmanager
def whole_pipeline_lock(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "launch.lock"
    with lock_path.open("a+") as lock_handle:
        log(f"waiting for whole-pipeline lock {lock_path}")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        log(f"acquired whole-pipeline lock {lock_path}")
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            log(f"released whole-pipeline lock {lock_path}")


def run_pipeline(args: argparse.Namespace) -> int:
    revisions = pinned_revision_map(args.state_dir)
    if args.download or args.launch:
        for name, repo_id in REMOTE_REPOS.items():
            revisions[name] = pin_dataset_revision(
                args.state_dir,
                args.data_root,
                name,
                repo_id,
            )

    remote_results: dict[str, Audit] = {}
    if args.download:
        for name, repo_id in REMOTE_REPOS.items():
            result = complete_remote_dataset(
                root=args.data_root,
                name=name,
                repo_id=repo_id,
                revision=revisions[name],
                workers=args.workers,
                retry_seconds=args.retry_seconds,
                max_attempts=args.max_attempts,
                max_files_per_attempt=args.max_files_per_attempt,
            )
            if not result.complete:
                log(f"cannot continue: {name} is still incomplete")
                return 1
            remote_results[name] = result

    audits = [
        remote_results.get(name)
        or audit_dataset(
            args.data_root,
            name,
            strict_payloads=True,
            expected_revision=revisions.get(name),
        )
        for name in DATASETS
    ]
    for result in audits:
        print_audit(result)
    write_json_atomic(args.state_dir / "latest_audit.json", [asdict(item) for item in audits])

    ssv2_audit = audit_ssv2(strict_payloads=True)
    incomplete = [item.name for item in audits if not item.complete]
    if not ssv2_audit["complete"]:
        incomplete.append("ssv2")
    if incomplete:
        log(f"all_robot verification FAILED: {', '.join(incomplete)}")
        return 1

    completion = {
        "completed_at": datetime.now().astimezone().isoformat(),
        "datasets": [asdict(item) for item in audits],
        "ssv2": ssv2_audit,
        "hf_revisions": revisions,
    }
    write_json_atomic(args.state_dir / "datasets_complete.json", completion)
    log("all_robot verification PASSED for all seven datasets")

    if args.launch:
        ensure_droid_cache(args.repo_root, args.data_root, args.state_dir)
        run_data_smoke(args.repo_root, args.state_dir)
        identity = pipeline_identity(
            args.repo_root,
            args.data_root,
            audits,
            ssv2_audit,
            revisions,
        )
        log(
            "launch identity: "
            f"pipeline={identity['pipeline_fingerprint']}, "
            f"code={identity['code_fingerprint']}, "
            f"config={identity['config_fingerprint']}, "
            f"data={identity['data_fingerprint']}"
        )
        receipt = submit_pipeline(args.repo_root, args.state_dir, identity)
        log(f"launch receipt: {receipt}")
    return 0


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    with whole_pipeline_lock(args.state_dir):
        return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
