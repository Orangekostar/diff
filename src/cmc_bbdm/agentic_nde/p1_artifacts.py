"""Deterministic, checksum-bound P1 visual-observability packages."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import polars as pl


class P1ArtifactError(ValueError):
    """Raised when a P1 package is incomplete, unsafe, or changed."""


REQUIRED_P1_FILES = frozenset(
    {
        "config.yaml",
        "authorized_roster.csv",
        "visual_feature_manifest.csv",
        "outer_model_selection.csv",
        "per_state_scores.parquet",
        "per_specimen_metrics.csv",
        "domain_metrics.csv",
        "bootstrap.csv",
        "acquisition_curves.csv",
        "control_results.csv",
        "summary.json",
        "REPORT.md",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
)
_PAYLOAD_FILES = REQUIRED_P1_FILES - {
    "artifact_manifest.json",
    "CHECKSUMS.sha256",
}
_TABLE_FILES = {
    "authorized_roster": "authorized_roster.csv",
    "visual_feature_manifest": "visual_feature_manifest.csv",
    "outer_model_selection": "outer_model_selection.csv",
    "per_state_scores": "per_state_scores.parquet",
    "per_specimen_metrics": "per_specimen_metrics.csv",
    "domain_metrics": "domain_metrics.csv",
    "bootstrap": "bootstrap.csv",
    "acquisition_curves": "acquisition_curves.csv",
    "control_results": "control_results.csv",
}
_SORT_KEYS = {
    "authorized_roster": ("dataset_id", "specimen_id"),
    "visual_feature_manifest": ("array_name",),
    "outer_model_selection": (
        "outer_domain",
        "stage",
        "representation",
        "candidate_id",
        "validation_domain",
        "method",
    ),
    "per_state_scores": ("outer_domain", "specimen_id", "method", "cell_index"),
    "per_specimen_metrics": ("outer_domain", "specimen_id", "method"),
    "domain_metrics": ("outer_domain", "method"),
    "bootstrap": ("effect_key",),
    "acquisition_curves": (
        "outer_domain",
        "specimen_id",
        "method",
        "nominal_checkpoint",
    ),
    "control_results": ("method",),
}
_STATUSES = {
    "P1_SPATIAL_VISUAL_OBSERVABILITY_GO",
    "P1_GLOBAL_VISUAL_CONTEXT_GO",
    "P1_DESCRIPTIVE_SPATIAL_SIGNAL_ONLY",
    "P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO",
}
_HEX = frozenset("0123456789abcdef")
_ABSOLUTE_TEXT = re.compile(
    r"(?:^|[\s:=\"'(])(?:/|\\\\|[A-Za-z]:[\\/])", flags=re.MULTILINE
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and not set(value) - _HEX


def _json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as error:
        raise P1ArtifactError("P1 JSON payload is not canonicalizable") from error
    return (text + "\n").encode("ascii")


def _contains_absolute_path(value: object) -> bool:
    if type(value) is str:
        return value.startswith(("/", "\\\\")) or (
            len(value) > 2 and value[1:3] in {":/", ":\\"}
        )
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _canonical_table(name: str, table: pl.DataFrame) -> pl.DataFrame:
    if type(table) is not pl.DataFrame or table.is_empty():
        raise P1ArtifactError(f"P1 {name} table is empty or invalid")
    keys = tuple(key for key in _SORT_KEYS[name] if key in table.columns)
    if not keys:
        raise P1ArtifactError(f"P1 {name} sort identity is unavailable")
    try:
        return table.sort(list(keys), nulls_last=True)
    except pl.exceptions.PolarsError as error:
        raise P1ArtifactError(f"P1 {name} cannot be sorted") from error


def _table_bytes(name: str, table: pl.DataFrame) -> bytes:
    canonical = _canonical_table(name, table)
    if name == "per_state_scores":
        stream = io.BytesIO()
        canonical.write_parquet(stream, compression="zstd", statistics=True)
        return stream.getvalue()
    try:
        return canonical.write_csv().encode("utf-8")
    except (UnicodeEncodeError, pl.exceptions.PolarsError) as error:
        raise P1ArtifactError(f"P1 {name} cannot be encoded") from error


def _validate_summary(value: object) -> dict[str, object]:
    if (
        type(value) is not dict
        or value.get("schema_version") != 1
        or value.get("stage") != "P1_VISUAL_OBSERVABILITY"
        or value.get("status") not in _STATUSES
        or _contains_absolute_path(value)
    ):
        raise P1ArtifactError("P1 summary identity changed")
    return dict(value)


def write_p1_package(
    destination: str | Path,
    *,
    config_bytes: bytes,
    tables: Mapping[str, pl.DataFrame],
    summary: Mapping[str, object],
    report: str,
) -> Path:
    """Write the exact P1 package by verified atomic rename without overwrite."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise P1ArtifactError("P1 output already exists")
    if (
        type(config_bytes) is not bytes
        or not config_bytes
        or type(report) is not str
        or not report
        or _ABSOLUTE_TEXT.search(report)
        or set(tables) != set(_TABLE_FILES)
        or any(type(value) is not pl.DataFrame for value in tables.values())
    ):
        raise P1ArtifactError("P1 package input changed")
    try:
        config_bytes.decode("ascii")
        report_bytes = report.encode("ascii")
    except UnicodeError as error:
        raise P1ArtifactError("P1 text payload must be ASCII") from error
    summary_payload = _validate_summary(dict(summary))
    payload: dict[str, bytes] = {
        "config.yaml": config_bytes,
        "summary.json": _json(summary_payload),
        "REPORT.md": report_bytes,
    }
    payload.update(
        {
            filename: _table_bytes(name, tables[name])
            for name, filename in _TABLE_FILES.items()
        }
    )
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise P1ArtifactError("P1 output parent is unavailable")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        manifest = _json(
            {
                "files": {
                    name: {"sha256": _sha256(value), "size": len(value)}
                    for name, value in sorted(payload.items())
                },
                "schema_version": 1,
                "stage": "P1_VISUAL_OBSERVABILITY",
                "status": summary_payload["status"],
            }
        )
        complete = dict(payload)
        complete["artifact_manifest.json"] = manifest
        complete["CHECKSUMS.sha256"] = "".join(
            f"{_sha256(value)}  {name}\n"
            for name, value in sorted(complete.items())
        ).encode("ascii")
        if set(complete) != REQUIRED_P1_FILES:
            raise P1ArtifactError("P1 package membership is internally inconsistent")
        for name, value in complete.items():
            (temporary / name).write_bytes(value)
        replay_p1_package(temporary)
        if target.exists() or target.is_symlink():
            raise P1ArtifactError("P1 output already exists")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _load_json(root: Path, name: str) -> object:
    try:
        raw = (root / name).read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P1ArtifactError(f"P1 {name} is invalid") from error
    if raw != _json(value):
        raise P1ArtifactError(f"P1 {name} is not canonical")
    return value


def replay_p1_package(path: str | Path) -> dict[str, object]:
    """Verify exact package membership, canonical metadata, and all hashes."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise P1ArtifactError("P1 package is unavailable")
    entries = tuple(root.iterdir())
    if (
        any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or {entry.name for entry in entries} != REQUIRED_P1_FILES
    ):
        raise P1ArtifactError("P1 package membership changed")
    manifest = _load_json(root, "artifact_manifest.json")
    if (
        type(manifest) is not dict
        or set(manifest) != {"files", "schema_version", "stage", "status"}
        or manifest["schema_version"] != 1
        or manifest["stage"] != "P1_VISUAL_OBSERVABILITY"
        or manifest["status"] not in _STATUSES
        or type(manifest["files"]) is not dict
        or set(manifest["files"]) != _PAYLOAD_FILES
    ):
        raise P1ArtifactError("P1 manifest schema changed")
    for name, record in manifest["files"].items():
        if (
            type(record) is not dict
            or set(record) != {"sha256", "size"}
            or type(record["size"]) is not int
            or record["size"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise P1ArtifactError("P1 manifest file record changed")
        value = (root / name).read_bytes()
        if len(value) != record["size"] or _sha256(value) != record["sha256"]:
            raise P1ArtifactError(f"P1 file hash or size changed: {name}")
    expected_checksums = "".join(
        f"{_sha256((root / name).read_bytes())}  {name}\n"
        for name in sorted(_PAYLOAD_FILES | {"artifact_manifest.json"})
    ).encode("ascii")
    if (root / "CHECKSUMS.sha256").read_bytes() != expected_checksums:
        raise P1ArtifactError("P1 checksum ledger changed")
    summary = _load_json(root, "summary.json")
    validated = _validate_summary(summary)
    if validated["status"] != manifest["status"]:
        raise P1ArtifactError("P1 summary and manifest statuses differ")
    try:
        for name, filename in _TABLE_FILES.items():
            if filename.endswith(".parquet"):
                table = pl.read_parquet(root / filename)
            else:
                table = pl.read_csv(root / filename)
            _canonical_table(name, table)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise P1ArtifactError("P1 table payload cannot be read") from error
    return validated


__all__ = [
    "REQUIRED_P1_FILES",
    "P1ArtifactError",
    "replay_p1_package",
    "write_p1_package",
]
