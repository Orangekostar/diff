"""Scope, path, and immutable-input validation for order analysis."""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cmc_bbdm.cai_active_image.environment import NativeCellGrid

TASK_ID = "CAI_ORDER_MECHANISM_R1_37B3C404"
SOURCE_COMMIT = "37b3c40414c00c6633b64656c4bbb2b178ef9848"
BRANCH = "research/cai-vlm-agent-v3-controlled-reuse"
ARCHIVED_METHODS = (
    "CENTER_FIRST",
    "GEOMETRY_SPREAD",
    "SERPENTINE",
    "RANDOM",
    "LEARNED_STATIC_TRUE",
    "NO_VLM_SPATIAL_FEEDBACK",
)
MAIN_METHOD = "NO_VLM_SPATIAL_FEEDBACK"
EXPECTED_DOMAIN_COUNTS = {
    "74t7kcdgkr": 9,
    "cgtnjyggtm": 9,
    "w68dtmpfyf": 8,
    "xcmzfsbd9t": 9,
    "yfxyg8jm46": 8,
    "ykhs7s2dck": 7,
}
_ZERO_LIMITS = (
    "optimizer_updates",
    "actor_forwards",
    "qwen_forwards",
    "cnn_forwards",
    "oof_predictor_forwards",
    "autograd",
    "test_forwards",
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return destination


def atomic_text(path: str | Path, value: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return destination


def write_csv_atomic(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError("empty CSV requires explicit fieldnames")
    fields = list(fieldnames or dict.fromkeys(key for row in rows for key in row))
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return destination


def write_gzip_csv_atomic(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError("empty CSV requires explicit fieldnames")
    fields = list(fieldnames or dict.fromkeys(key for row in rows for key in row))
    temporary = destination.with_name(destination.name + ".tmp")
    raw = temporary.open("wb")
    try:
        with (
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle,
        ):
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        raw.flush()
        os.fsync(raw.fileno())
    finally:
        raw.close()
    temporary.replace(destination)
    return destination


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _validate_relative(raw: object, name: str) -> Path:
    path = Path(str(raw))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative")
    return path


def _validate_scope(scope: Mapping[str, Any], branch: str) -> None:
    if scope.get("schema_version") != 1 or scope.get("task_id") != TASK_ID:
        raise ValueError("unexpected task identity")
    if scope.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("unexpected source commit")
    if scope.get("branch") != BRANCH or branch != BRANCH:
        raise ValueError(f"scope must run on branch {BRANCH}")
    if scope.get("task_type") != (
        "FROZEN_TRAJECTORY_ANALYSIS_AND_FIXED_FINAL_SET_ORDER_INTERVENTION"
    ):
        raise ValueError("unexpected task type")
    cohort = scope.get("cohort", {})
    if cohort.get("split") != "VALID" or cohort.get("test_inference_or_label_join") is not False:
        raise ValueError("only the VALID cohort without TEST access is authorized")
    if (cohort.get("physical_n"), cohort.get("capture_groups"), cohort.get("domains")) != (
        50,
        48,
        6,
    ):
        raise ValueError("VALID cohort identity changed")
    models = scope.get("frozen_models", {})
    if models.get("actor", {}).get("forward_allowed") is not False:
        raise ValueError("actor forwards must remain disabled")
    predictor = models.get("predictor", {})
    if (
        predictor.get("model") != "MEAN_SC"
        or predictor.get("selected_update") != 1750
        or predictor.get("forward_allowed") is not True
        or models.get("selection") != "LOCK_EXISTING_SELECTION_NO_RESELECTION"
    ):
        raise ValueError("frozen predictor selection changed")
    limits = scope.get("resource_limits", {})
    for name in _ZERO_LIMITS:
        if limits.get(name) != 0:
            raise ValueError(f"forbidden resource must remain zero: {name}")
    if (
        limits.get("predictor_evaluated_state_rows_including_padding_retries_qa")
        != 7000
        or limits.get("max_visible_gpus") != 1
        or limits.get("max_cpu_threads") != 4
    ):
        raise ValueError("resource limits changed")
    roots = scope.get("roots", {})
    for name in ("code", "tests", "scope", "results", "artifacts"):
        if name not in roots:
            raise ValueError(f"missing configured root: {name}")
        _validate_relative(roots[name], f"root {name}")


@dataclass(frozen=True, slots=True)
class TaskContext:
    root: Path
    scope: dict[str, Any]
    scope_path: Path | None
    branch: str
    head: str

    @classmethod
    def from_mapping(
        cls,
        root: str | Path,
        scope: Mapping[str, Any],
        *,
        branch: str,
        head: str,
        scope_path: str | Path | None = None,
    ) -> TaskContext:
        copied = json.loads(json.dumps(scope))
        _validate_scope(copied, branch)
        return cls(
            root=Path(root).resolve(),
            scope=copied,
            scope_path=Path(scope_path).resolve() if scope_path is not None else None,
            branch=branch,
            head=head,
        )

    @classmethod
    def load(cls, root: str | Path, scope_path: str | Path) -> TaskContext:
        repository = Path(root).resolve(strict=True)
        scope_file = Path(scope_path)
        if not scope_file.is_absolute():
            scope_file = repository / scope_file
        scope_file = scope_file.resolve(strict=True)
        actual_root = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
        if actual_root != repository:
            raise ValueError("--root must be the repository root")
        branch = _git(repository, "branch", "--show-current")
        head = _git(repository, "rev-parse", "HEAD")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
            cwd=repository,
            check=True,
        )
        return cls.from_mapping(
            repository,
            json.loads(scope_file.read_text(encoding="utf-8")),
            branch=branch,
            head=head,
            scope_path=scope_file,
        )

    def root_path(self, name: str) -> Path:
        relative = _validate_relative(self.scope["roots"][name], f"root {name}")
        return self.root / relative

    def source_path(self, relative: str | Path) -> Path:
        path = _validate_relative(relative, "source path")
        local = self.root / path
        if local.exists():
            return local
        common = Path(_git(self.root, "rev-parse", "--git-common-dir")).resolve()
        return (common.parent / path).resolve(strict=True)

    def source(self, dotted: str) -> Path:
        value: object = self.scope
        for part in dotted.split("."):
            if not isinstance(value, Mapping):
                raise KeyError(dotted)
            value = value[part]
        if isinstance(value, Mapping):
            value = value["path"]
        return self.source_path(_validate_relative(value, dotted))

    @property
    def results_root(self) -> Path:
        return self.root_path("results")

    @property
    def artifacts_root(self) -> Path:
        return self.root_path("artifacts")

    @property
    def scope_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.scope))


@dataclass(frozen=True, slots=True)
class ValidFeature:
    specimen_key: str
    dataset_id: str
    specimen_id: str
    capture_group_id: str
    target_mpa: float
    surface_tokens: np.ndarray
    cscan_tokens: np.ndarray
    native_shape: tuple[int, int]
    shard_path: str
    shard_index: int
    shard_sha256: str

    @property
    def grid(self) -> NativeCellGrid:
        return NativeCellGrid.from_shape(self.native_shape)


def _validate_episode_panel(
    rows: Sequence[Mapping[str, str]], selected_rows: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, str]], list[str]]:
    from scripts.cai_order_mechanism.analysis import parse_episode

    if len(rows) != 650:
        raise ValueError(f"assembled episode source must contain 650 rows, found {len(rows)}")
    filtered = [dict(row) for row in rows if row["method"] in ARCHIVED_METHODS]
    if len(filtered) != 500:
        raise ValueError(f"archived analysis must contain 500 rows, found {len(filtered)}")
    counts = Counter(row["method"] for row in filtered)
    expected = {method: (250 if method == "RANDOM" else 50) for method in ARCHIVED_METHODS}
    if counts != expected:
        raise ValueError(f"archived method counts changed: {dict(counts)}")
    episodes = [parse_episode(row) for row in filtered]
    main = [episode for episode in episodes if episode.method == MAIN_METHOD]
    if len(main) != 50 or any(len(episode.cells) not in {15, 16} for episode in main):
        raise ValueError("main trajectories must be 50 rows of 15 or 16 actions")
    selected = {
        (row["specimen_key"], row["method"], int(row["run"])): row
        for row in selected_rows
    }
    if len(selected) != 50:
        raise ValueError("selected @1000 episode file must contain 50 unique rows")
    for row in filtered:
        if row["method"] != MAIN_METHOD:
            continue
        key = (row["specimen_key"], row["method"], int(row["run"]))
        if key not in selected:
            raise ValueError(f"main selected row is missing: {key}")
        for field in ("cells", "costs", "predictions_mpa", "target_mpa"):
            if row[field] != selected[key][field]:
                raise ValueError(f"main selected field differs for {key}: {field}")
    metadata: dict[str, tuple[str, str, float]] = {}
    initials: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        identity = (episode.dataset_id, episode.capture_group_id, episode.target_mpa)
        prior = metadata.setdefault(episode.specimen_key, identity)
        if prior != identity:
            raise ValueError(f"method metadata differs for {episode.specimen_key}")
        initials[episode.specimen_key].append(float(episode.predictions_mpa[0]))
    if any(max(values) - min(values) > 1e-4 for values in initials.values()):
        raise ValueError("initial predictions differ by more than 1e-4 MPa")
    keys = sorted(metadata)
    domains = Counter(metadata[key][0] for key in keys)
    groups = {metadata[key][1] for key in keys}
    if len(keys) != 50 or len(groups) != 48 or domains != EXPECTED_DOMAIN_COUNTS:
        raise ValueError("50/48/6 VALID cohort identity changed")
    return filtered, keys


def load_valid_features(
    context: TaskContext,
    keys: Sequence[str],
    *,
    shard_hashes: Mapping[str, str] | None = None,
) -> dict[str, ValidFeature]:
    rows = [
        row
        for row in read_csv_rows(context.source("sources.feature_index"))
        if row["split"] == "VALID"
    ]
    wanted = set(keys)
    if len(rows) != 50 or {row["specimen_key"] for row in rows} != wanted:
        raise ValueError("feature-index VALID keys differ from the locked cohort")
    by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_path"]].append(row)
    output: dict[str, ValidFeature] = {}
    for shard_relative, shard_rows in sorted(by_shard.items()):
        shard_path = context.source_path(shard_relative)
        expected_hash = shard_rows[0]["shard_sha256"]
        if any(row["shard_sha256"] != expected_hash for row in shard_rows):
            raise ValueError("feature-index shard hashes disagree")
        if shard_hashes is not None and shard_hashes.get(shard_relative) != expected_hash:
            raise ValueError("prepared feature-shard binding differs")
        with np.load(shard_path, allow_pickle=False) as archive:
            for row in shard_rows:
                offset = int(row["shard_index"])
                key = str(archive["specimen_keys"][offset])
                dataset = str(archive["dataset_ids"][offset])
                group = str(archive["capture_group_ids"][offset])
                split = str(archive["splits"][offset])
                if (
                    key != row["specimen_key"]
                    or dataset != row["dataset_id"]
                    or group != row["capture_group_id"]
                    or split != "VALID"
                ):
                    raise ValueError(f"feature shard index mismatch: {row['specimen_key']}")
                surface = np.asarray(archive["surface_tokens"][offset], dtype=np.float32)
                cscan = np.asarray(archive["cscan_tokens"][offset], dtype=np.float32)
                shape = tuple(int(value) for value in archive["native_shapes"][offset])
                target = float(archive["targets_mpa"][offset])
                if surface.shape != (64, 512) or cscan.shape != (64, 512):
                    raise ValueError(f"feature tensor shape changed: {key}")
                output[key] = ValidFeature(
                    specimen_key=key,
                    dataset_id=dataset,
                    specimen_id=str(archive["specimen_ids"][offset]),
                    capture_group_id=group,
                    target_mpa=target,
                    surface_tokens=surface.copy(),
                    cscan_tokens=cscan.copy(),
                    native_shape=shape,
                    shard_path=shard_relative,
                    shard_index=offset,
                    shard_sha256=expected_hash,
                )
    return output


def load_bootstrap_weights(
    context: TaskContext,
    keys: Sequence[str],
    domains: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    with np.load(context.source("sources.bootstrap_weights"), allow_pickle=False) as archive:
        old_keys = [str(value) for value in archive["specimen_keys"]]
        if len(old_keys) != len(set(old_keys)) or set(old_keys) != set(keys):
            raise ValueError("fixed bootstrap specimen keys changed")
        lookup = {key: index for index, key in enumerate(old_keys)}
        order = np.asarray([lookup[key] for key in keys], dtype=np.int64)
        if not np.array_equal(archive["domains"][order].astype(str), domains.astype(str)):
            raise ValueError("fixed bootstrap domains changed")
        if not np.array_equal(
            archive["capture_groups"][order].astype(str), groups.astype(str)
        ):
            raise ValueError("fixed bootstrap capture groups changed")
        weights = np.asarray(archive["weights"][:, order])
    if weights.shape != (5000, 50):
        raise ValueError("fixed bootstrap dimensions changed")
    for group in sorted(set(groups.astype(str))):
        local = weights[:, groups.astype(str) == group]
        if local.shape[1] > 1 and not np.all(local == local[:, :1]):
            raise ValueError("capture-group members have different bootstrap multiplicity")
    for domain in sorted(set(domains.astype(str))):
        if np.any(weights[:, domains.astype(str) == domain].sum(axis=1) <= 0):
            raise ValueError("fixed bootstrap contains an empty domain draw")
    return weights


def prepare_inputs(context: TaskContext) -> list[Path]:
    results = context.results_root
    results.mkdir(parents=True, exist_ok=True)
    assembled_path = context.source("sources.assembled_episodes")
    selected_path = context.source("sources.main_selected_episodes")
    job_path = context.source("sources.main_job")
    binding_path = context.source("sources.feature_bindings")
    actor_path = context.source("frozen_models.actor")
    predictor_path = context.source("frozen_models.predictor")
    index_path = context.source("sources.feature_index")
    scope_sources = context.scope["sources"]
    scope_models = context.scope["frozen_models"]

    assembled_hash = sha256_file(assembled_path)
    index_hash = sha256_file(index_path)
    actor_hash = sha256_file(actor_path)
    predictor_hash = sha256_file(predictor_path)
    expected_hashes = {
        "assembled episode": (assembled_hash, scope_sources["assembled_episodes"]["sha256"]),
        "feature index": (index_hash, scope_sources["feature_index"]["sha256"]),
        "actor checkpoint": (actor_hash, scope_models["actor"]["sha256"]),
        "predictor checkpoint": (predictor_hash, scope_models["predictor"]["sha256"]),
    }
    for label, (actual, expected) in expected_hashes.items():
        if actual != expected:
            raise ValueError(f"{label} SHA256 changed")

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    shard_hashes: dict[str, str] = {}
    shard_records = []
    for row in binding["feature_shards"]:
        relative = row["shard_path"]
        actual = sha256_file(context.source_path(relative))
        if actual != row["sha256"]:
            raise ValueError(f"feature-shard SHA256 changed: {relative}")
        shard_hashes[relative] = actual
        shard_records.append({**row, "actual_sha256": actual})

    filtered, keys = _validate_episode_panel(
        read_gzip_csv_rows(assembled_path), read_gzip_csv_rows(selected_path)
    )
    job = json.loads(job_path.read_text(encoding="utf-8"))
    actor_scope = scope_models["actor"]
    if (
        job["manifest"]["method"] != MAIN_METHOD
        or job["manifest"]["selected_update"] != actor_scope["selected_update"]
        or job["manifest"]["training_seed"] != actor_scope["training_seed"]
        or job["manifest"]["checkpoint_sha256"] != actor_hash
        or job["manifest"]["selected_episodes"] != scope_sources["main_selected_episodes"]
    ):
        raise ValueError("main actor selection binding changed")

    features = load_valid_features(context, keys, shard_hashes=shard_hashes)
    main_target = {
        row["specimen_key"]: float(row["target_mpa"])
        for row in filtered
        if row["method"] == MAIN_METHOD
    }
    cohort_rows = []
    for key in keys:
        feature = features[key]
        if abs(feature.target_mpa - main_target[key]) > 1e-5:
            raise ValueError(f"feature target differs from episode target: {key}")
        grid = feature.grid
        cohort_rows.append(
            {
                "identity_family": "VALID50_CAPTURE48_DOMAIN6",
                "cohort": "VALID",
                "specimen_key": key,
                "dataset_id": feature.dataset_id,
                "specimen_id": feature.specimen_id,
                "capture_group_id": feature.capture_group_id,
                "target_mpa": feature.target_mpa,
                "native_height": feature.native_shape[0],
                "native_width": feature.native_shape[1],
                "total_pixels": int(np.prod(feature.native_shape)),
                "cell_pixel_counts": ";".join(
                    str(cell.pixel_count) for cell in grid.cells
                ),
                "shard_path": feature.shard_path,
                "shard_index": feature.shard_index,
                "shard_sha256": feature.shard_sha256,
            }
        )

    input_bindings = {
        "task_id": TASK_ID,
        "source_commit": SOURCE_COMMIT,
        "executed_head": context.head,
        "scope_sha256": context.scope_sha256,
        "large_files_hashed_once_at_prepare": True,
        "assembled_episodes": {
            "path": scope_sources["assembled_episodes"]["path"],
            "sha256": assembled_hash,
        },
        "feature_index": {
            "path": scope_sources["feature_index"]["path"],
            "sha256": index_hash,
        },
        "actor_checkpoint": {
            "path": scope_models["actor"]["path"],
            "sha256": actor_hash,
            "loaded": False,
        },
        "predictor_checkpoint": {
            "path": scope_models["predictor"]["path"],
            "sha256": predictor_hash,
        },
        "feature_shards": shard_records,
        "archive_rows": 500,
        "valid_physical_n": 50,
        "capture_groups": 48,
        "domain_counts": EXPECTED_DOMAIN_COUNTS,
        "test_access": 0,
    }
    model_lock = {
        "selection": "LOCK_EXISTING_SELECTION_NO_RESELECTION",
        "actor": {**scope_models["actor"], "loaded": False, "forward_calls": 0},
        "predictor": scope_models["predictor"],
        "new_model_selection": 0,
    }
    package_versions = {}
    for package in ("numpy", "pandas", "torch", "matplotlib"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "NOT_INSTALLED"
    runtime_lock = {
        "task_id": TASK_ID,
        "source_commit": SOURCE_COMMIT,
        "executed_head": context.head,
        "branch": context.branch,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "packages": package_versions,
        "max_cpu_threads": 4,
        "max_visible_gpus": 1,
        "model_dtype": "float32",
        "metric_dtype": "float64",
        "amp": False,
        "tf32": False,
        "engine_status": "NOT_LOCKED_BEFORE_INFER",
    }
    outputs = [
        atomic_text(
            results / "ORDER_MECHANISM_SCOPE.json",
            context.source("roots.scope").read_text(encoding="utf-8"),
        ),
        atomic_json(results / "input_bindings.json", input_bindings),
        atomic_json(results / "MODEL_SELECTION_LOCK.json", model_lock),
        atomic_json(results / "runtime_lock.json", runtime_lock),
        write_csv_atomic(results / "cohort_manifest.csv", cohort_rows),
    ]
    quality_rows = [
        {"target_mpa": float(value), "target_type": "INTEGER_FIXED"}
        for value in range(41, 62)
    ] + [{"target_mpa": 41.69001007080078, "target_type": "FULL_REFERENCE_FIXED"}]
    outputs.append(write_csv_atomic(results / "quality_targets.csv", quality_rows))
    return outputs


__all__ = [
    "ARCHIVED_METHODS",
    "BRANCH",
    "EXPECTED_DOMAIN_COUNTS",
    "MAIN_METHOD",
    "SOURCE_COMMIT",
    "TASK_ID",
    "TaskContext",
    "ValidFeature",
    "atomic_json",
    "atomic_text",
    "canonical_json",
    "load_bootstrap_weights",
    "load_valid_features",
    "prepare_inputs",
    "read_csv_rows",
    "read_gzip_csv_rows",
    "sha256_bytes",
    "sha256_file",
    "write_csv_atomic",
    "write_gzip_csv_atomic",
]
