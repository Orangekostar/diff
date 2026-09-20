"""Assemble the frozen controls and selected C trajectories."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.cai_c_retrain.context import (
    TaskContext,
    atomic_json,
    canonical_json,
    sha256_bytes,
    sha256_file,
)

CONTROL_METHODS = (
    "CENTER_FIRST",
    "GEOMETRY_SPREAD",
    "SERPENTINE",
    "RANDOM",
    "LEARNED_STATIC_TRUE",
    "NO_VLM_SPATIAL_FEEDBACK",
)
C_METHODS = (
    "VLM_SPATIAL_FEEDBACK",
    "VLM_SPATIAL_OPEN_LOOP",
    "VLM_MEAN_FEEDBACK",
)
HISTORICAL_A_METHODS = C_METHODS
ALL_METHODS = frozenset((*CONTROL_METHODS, *C_METHODS))
EXPECTED_CONTROL_COUNTS = {
    "CENTER_FIRST": 50,
    "GEOMETRY_SPREAD": 50,
    "SERPENTINE": 50,
    "RANDOM": 250,
    "LEARNED_STATIC_TRUE": 50,
    "NO_VLM_SPATIAL_FEEDBACK": 50,
}
EXPECTED_C_COUNTS = {method: 50 for method in C_METHODS}
EXPECTED_HISTORICAL_COUNTS = {method: 50 for method in HISTORICAL_A_METHODS}
PRIMARY_COUNT = 650
REUSED_CONTROL_COUNT = 500
NEW_C_COUNT = 150
HISTORICAL_A_COUNT = 150
TARGET_TOLERANCE = 1e-12
OLD_SOURCE_LABEL = "old_w3/policy_validation_episodes.csv.gz"
NEW_SOURCE_LABEL = "w3/new_selected_policy_episodes.csv.gz"


def read_episodes(path: str | Path) -> list[dict[str, str]]:
    """Read a gzip CSV trajectory file without changing its row values."""
    with gzip.open(Path(path), "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = list(fieldnames or _field_union(rows))
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _write_plain_csv_atomic(
    path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _field_union(rows: Iterable[Mapping[str, object]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def _method_counts(rows: Iterable[Mapping[str, object]]) -> Counter[str]:
    return Counter(str(row.get("method")) for row in rows)


def _require_counts(
    rows: Sequence[Mapping[str, object]], expected: Mapping[str, int], label: str
) -> None:
    observed = _method_counts(rows)
    if observed != Counter(expected):
        raise ValueError(
            f"{label} method counts invalid: expected {dict(expected)}, observed {dict(observed)}"
        )


def _check_allowed_methods(
    rows: Sequence[Mapping[str, object]], allowed: set[str], label: str
) -> None:
    observed = {row.get("method") for row in rows}
    unexpected = observed - allowed
    if unexpected:
        raise ValueError(
            f"{label} contains unauthorized methods: {sorted(map(str, unexpected))}"
        )


def _check_primary_identity(rows: Sequence[Mapping[str, object]]) -> None:
    if len(rows) != PRIMARY_COUNT:
        raise ValueError(f"primary matrix must contain {PRIMARY_COUNT} rows")

    specimens = {row.get("specimen_key") for row in rows}
    groups = {row.get("capture_group_id") for row in rows}
    datasets = {row.get("dataset_id") for row in rows}
    if len(specimens) != 50:
        raise ValueError("primary matrix must contain exactly 50 specimen keys")
    if len(groups) != 48:
        raise ValueError("primary matrix must contain exactly 48 capture groups")
    if len(datasets) != 6:
        raise ValueError("primary matrix must contain exactly 6 dataset ids")
    if None in specimens or None in groups or None in datasets:
        raise ValueError("primary matrix contains an empty identity field")

    by_method: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("method"))].append(row)
    expected_methods = set(EXPECTED_CONTROL_COUNTS) | set(EXPECTED_C_COUNTS)
    if set(by_method) != expected_methods:
        raise ValueError("primary matrix method set is invalid")

    reference_keys: set[object] | None = None
    for method in (*CONTROL_METHODS, *C_METHODS):
        method_rows = by_method[method]
        by_specimen: dict[object, list[Mapping[str, object]]] = defaultdict(list)
        for row in method_rows:
            by_specimen[row.get("specimen_key")].append(row)
        if set(by_specimen) != specimens:
            raise ValueError(f"{method} does not cover the same 50 specimens")
        if method == "RANDOM":
            if any(len(case_rows) != 5 for case_rows in by_specimen.values()):
                raise ValueError("RANDOM must contain exactly five runs per specimen")
            for specimen, case_rows in by_specimen.items():
                runs = [row.get("run") for row in case_rows]
                if len(set(runs)) != 5:
                    raise ValueError(
                        f"RANDOM has duplicate runs for specimen {specimen}"
                    )
        elif any(len(case_rows) != 1 for case_rows in by_specimen.values()):
            raise ValueError(f"{method} must contain exactly one row per specimen")
        if reference_keys is None:
            reference_keys = set(by_specimen)

    target_by_specimen: dict[object, float] = {}
    group_by_specimen: dict[object, object] = {}
    dataset_by_specimen: dict[object, object] = {}
    for row in rows:
        specimen = row.get("specimen_key")
        try:
            target = float(row["target_mpa"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid target_mpa for specimen {specimen}") from error
        if not math.isfinite(target):
            raise ValueError(f"nonfinite target_mpa for specimen {specimen}")
        if (
            specimen in target_by_specimen
            and abs(target - target_by_specimen[specimen]) > TARGET_TOLERANCE
        ):
            raise ValueError(f"target_mpa drift for specimen {specimen}")
        target_by_specimen.setdefault(specimen, target)
        group = row.get("capture_group_id")
        dataset = row.get("dataset_id")
        if specimen in group_by_specimen and group != group_by_specimen[specimen]:
            raise ValueError(f"capture group drift for specimen {specimen}")
        if specimen in dataset_by_specimen and dataset != dataset_by_specimen[specimen]:
            raise ValueError(f"dataset id drift for specimen {specimen}")
        group_by_specimen.setdefault(specimen, group)
        dataset_by_specimen.setdefault(specimen, dataset)


def _provenance_rows(
    rows: Sequence[Mapping[str, object]], old_count: int
) -> list[dict[str, object]]:
    provenance: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        is_old = index < old_count
        provenance.append(
            {
                "row_index": index,
                "specimen_key": row["specimen_key"],
                "method": row["method"],
                "provenance": "REUSED_FROZEN_CONTROL" if is_old else "NEW_C_RETRAIN",
                "source_path": OLD_SOURCE_LABEL if is_old else NEW_SOURCE_LABEL,
                "prior_version": "UNCHANGED_CONTROL" if is_old else "C_P0_R1_GLOBAL_V1",
            }
        )
    return provenance


def assemble_rows(
    old_rows: Sequence[Mapping[str, object]], new_rows: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Return primary rows, provenance rows and the historical A sidecar."""
    old_rows = list(old_rows)
    new_rows = list(new_rows)
    _check_allowed_methods(old_rows, ALL_METHODS, "old W3 input")
    _check_allowed_methods(new_rows, set(C_METHODS), "new C input")

    old_controls = [
        deepcopy(row) for row in old_rows if row.get("method") in CONTROL_METHODS
    ]
    historical = [
        deepcopy(row) for row in old_rows if row.get("method") in HISTORICAL_A_METHODS
    ]
    selected_c = [deepcopy(row) for row in new_rows if row.get("method") in C_METHODS]
    _require_counts(old_controls, EXPECTED_CONTROL_COUNTS, "frozen controls")
    _require_counts(historical, EXPECTED_HISTORICAL_COUNTS, "historical A")
    _require_counts(selected_c, EXPECTED_C_COUNTS, "new C")

    primary = old_controls + selected_c
    _check_primary_identity(primary)
    provenance = _provenance_rows(primary, len(old_controls))
    if len(historical) != HISTORICAL_A_COUNT:
        raise ValueError("historical A sidecar must contain 150 rows")
    return primary, provenance, historical


def _relative_path(context: TaskContext, path: Path) -> str:
    try:
        return path.resolve().relative_to(context.root).as_posix()
    except ValueError:
        return str(path.resolve())


def _with_source_paths(
    rows: Sequence[Mapping[str, object]],
    context: TaskContext,
    old_path: Path,
    new_path: Path,
) -> list[dict[str, object]]:
    old_source = _relative_path(context, old_path)
    new_source = _relative_path(context, new_path)
    updated = []
    for row in rows:
        copied = dict(row)
        copied["source_path"] = (
            old_source if row["provenance"] == "REUSED_FROZEN_CONTROL" else new_source
        )
        updated.append(copied)
    return updated


def _audit_frozen_bindings(
    context: TaskContext, actor_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    old_path = context.path("old_w3_artifacts") / "INPUT_BINDINGS.json"
    new_path = context.path("w3") / "input_bindings.json"
    if not old_path.is_file() or not new_path.is_file():
        raise FileNotFoundError("old and current W3 input bindings are required")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    new_files = new.get("files", {})
    common = {
        "feature_bank_manifest.json": "feature_bank_manifest",
        "feature_bank_index.csv": "feature_bank_index",
        "oof_fold_manifest.csv": "oof_fold_manifest",
        "split_manifest.csv": "split_manifest",
    }
    for old_name, new_name in common.items():
        if old["data_metadata"].get(old_name) != new_files.get(new_name):
            raise ValueError(
                f"old control and current C data binding differ: {old_name}"
            )
    for name in ("candidate_queue", "cost_precision_audit", "feature_bank_shards"):
        path = (
            context.path("data")
            / {
                "candidate_queue": "candidate_queue.csv",
                "cost_precision_audit": "cost_precision_audit.json",
                "feature_bank_shards": "feature_bank_shards.csv",
            }[name]
        )
        if new_files.get(name) != sha256_file(path):
            raise ValueError(f"current C data binding changed: {name}")

    old_shards = {row["dataset_id"]: row for row in old["feature_shards"]}
    if len(old_shards) != 6:
        raise ValueError("old control binding does not contain six feature shards")
    for dataset_id, row in old_shards.items():
        path = context.root / row["shard_path"]
        digest = sha256_file(path)
        if (
            digest != row["sha256"]
            or new_files.get(f"feature_shard_{dataset_id}") != digest
        ):
            raise ValueError(
                f"old control and current C feature shard differ: {dataset_id}"
            )

    expected_p_all_path = context.scope["predictors"]["common_checkpoint"]
    expected_p_all_hash = context.scope["predictors"]["common_checkpoint_sha256"]
    p_all = next(
        row
        for row in old["predictors"]
        if row["checkpoint_path"] == expected_p_all_path
    )
    if p_all["checkpoint_sha256"] != expected_p_all_hash:
        raise ValueError("old controls did not use the frozen W2 P_all checkpoint")
    for predictor in old["predictors"]:
        path = context.root / predictor["checkpoint_path"]
        if sha256_file(path) != predictor["checkpoint_sha256"]:
            raise ValueError(f"frozen W2 checkpoint changed: {path}")

    vlm_manifest = context.path("vlm") / "vlm_manifest_fit.json"
    vlm_features = context.path("vlm") / "vlm_actor_features_fit.csv"
    if new_files.get("vlm_manifest") != sha256_file(vlm_manifest) or new_files.get(
        "vlm_features"
    ) != sha256_file(vlm_features):
        raise ValueError("current C prior differs from the training binding")
    expected_signature = sha256_bytes(canonical_json(new))
    method_manifests = actor_manifest.get("actor_manifests", [])
    if len(method_manifests) != 3 or any(
        row.get("input_signature") != expected_signature for row in method_manifests
    ):
        raise ValueError(
            "selected C actors do not share the frozen training input signature"
        )
    return {
        "status": "FROZEN_W2_FEATURE_COST_AND_C_PRIOR_BINDINGS_MATCH",
        "old_input_bindings": {
            "path": _relative_path(context, old_path),
            "sha256": sha256_file(old_path),
        },
        "current_input_bindings": {
            "path": _relative_path(context, new_path),
            "sha256": sha256_file(new_path),
            "signature": expected_signature,
        },
        "p_all": {"path": expected_p_all_path, "sha256": expected_p_all_hash},
        "feature_shard_count": 6,
    }


def assemble_stage(context: TaskContext) -> dict[str, Any]:
    """Assemble the release matrix after the three C actors are complete."""
    w3 = context.path("w3")
    actor_manifest_path = w3 / "actor_manifests.json"
    if not actor_manifest_path.is_file():
        raise FileNotFoundError(actor_manifest_path)
    actor_manifest = json.loads(actor_manifest_path.read_text(encoding="utf-8"))
    if actor_manifest.get("status") != "THREE_C_ACTORS_COMPLETE":
        raise ValueError("three C actors are not complete")
    if actor_manifest.get("selected_episode_count") != NEW_C_COUNT:
        raise ValueError("selected C episodes must contain exactly 150 rows")
    binding_audit = _audit_frozen_bindings(context, actor_manifest)

    old_path = context.path("old_w3") / "policy_validation_episodes.csv.gz"
    new_path = w3 / "new_selected_policy_episodes.csv.gz"
    if not old_path.is_file() or not new_path.is_file():
        raise FileNotFoundError("W3 source trajectory file is missing")
    old_hash = sha256_file(old_path)
    new_hash = sha256_file(new_path)
    primary, provenance, historical = assemble_rows(
        read_episodes(old_path), read_episodes(new_path)
    )
    provenance = _with_source_paths(provenance, context, old_path, new_path)

    primary_path = w3 / "policy_validation_episodes.csv.gz"
    provenance_path = w3 / "row_provenance.csv"
    historical_path = (
        w3 / "historical_A_comparison" / "policy_validation_episodes.csv.gz"
    )
    _write_csv_atomic(primary_path, primary)
    _write_plain_csv_atomic(
        provenance_path,
        provenance,
        [
            "row_index",
            "specimen_key",
            "method",
            "provenance",
            "source_path",
            "prior_version",
        ],
    )
    _write_csv_atomic(historical_path, historical)

    method_counts = dict(_method_counts(primary))
    manifest = {
        "schema_version": 1,
        "task_id": context.task_id,
        "status": "PRIMARY_MATRIX_COMPLETE",
        "prior_version": context.scope.get("prior_version", "C_P0_R1_GLOBAL_V1"),
        "primary_count": len(primary),
        "provenance_count": len(provenance),
        "reused_control_count": REUSED_CONTROL_COUNT,
        "new_C_count": NEW_C_COUNT,
        "historical_A_count": len(historical),
        "method_counts": method_counts,
        "sources": {
            "old_w3": {"path": _relative_path(context, old_path), "sha256": old_hash},
            "new_C": {"path": _relative_path(context, new_path), "sha256": new_hash},
            "actor_manifests": {
                "path": _relative_path(context, actor_manifest_path),
                "sha256": sha256_file(actor_manifest_path),
            },
        },
        "binding_audit": binding_audit,
        "outputs": {
            "primary": {
                "path": _relative_path(context, primary_path),
                "sha256": sha256_file(primary_path),
            },
            "provenance": {
                "path": _relative_path(context, provenance_path),
                "sha256": sha256_file(provenance_path),
            },
            "historical_A": {
                "path": _relative_path(context, historical_path),
                "sha256": sha256_file(historical_path),
            },
        },
    }
    manifest_path = w3 / "assembly_manifest.json"
    atomic_json(manifest_path, manifest)
    context.transition(
        "assemble",
        "COMPLETE",
        primary_rows=PRIMARY_COUNT,
        reused_control_rows=REUSED_CONTROL_COUNT,
        new_c_rows=NEW_C_COUNT,
        historical_a_rows=HISTORICAL_A_COUNT,
    )
    return manifest


__all__ = [
    "assemble_rows",
    "assemble_stage",
    "read_episodes",
]
