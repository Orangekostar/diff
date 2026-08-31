"""Deterministic, checksum-bound P0R authority packages."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .author_authority import build_author_registration_authority
from .contracts import P0RGateFacts, P0RStatus, decide_p0r


class P0RArtifactError(ValueError):
    """Raised when a P0R package is unsafe, incomplete, or changed."""


REQUIRED_P0R_FILES = frozenset(
    {
        "config.yaml",
        "author_authority.json",
        "surface_manifest.csv",
        "scan_processing_provenance.csv",
        "registration.csv",
        "registration_qc.csv",
        "grid_mapping_qc.csv",
        "summary.json",
        "REPORT.md",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
)
_PAYLOAD_FILES = REQUIRED_P0R_FILES - {
    "artifact_manifest.json",
    "CHECKSUMS.sha256",
}
_CSV_KEYS = {
    "surface_manifest.csv": ("dataset_id", "specimen_id"),
    "scan_processing_provenance.csv": ("dataset_id", "specimen_id"),
    "registration.csv": ("dataset_id", "specimen_id"),
    "registration_qc.csv": ("dataset_id", "specimen_id"),
    "grid_mapping_qc.csv": ("dataset_id", "specimen_id", "cell_id"),
}
_STATUSES = frozenset(status.value for status in P0RStatus)
_HEX = frozenset("0123456789abcdef")
_ABSOLUTE_TEXT = re.compile(
    r"(?:^|[\s:=\"'(])(?:/|\\\\|[A-Za-z]:[\\/])",
    flags=re.MULTILINE,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and not set(value) - _HEX


def _json(value: object) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            ensure_ascii=True,
        )
    except (TypeError, ValueError) as error:
        raise P0RArtifactError("P0R JSON payload is not canonicalizable") from error
    return (payload + "\n").encode("ascii")


def _has_absolute_path(value: object) -> bool:
    if type(value) is str:
        return value.startswith(("/", "\\\\")) or (
            len(value) >= 3 and value[1:3] in {":\\", ":/"}
        )
    if isinstance(value, Mapping):
        return any(_has_absolute_path(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_absolute_path(item) for item in value)
    return False


def _validate_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or _ABSOLUTE_TEXT.search(value):
        raise P0RArtifactError(f"P0R {label} text is invalid")
    return value


def _validate_cell(value: object, *, name: str) -> None:
    if type(value) in {str, int, bool}:
        return
    if type(value) is float and math.isfinite(value):
        return
    raise P0RArtifactError(f"{name} contains a non-scalar value")


def _canonical_rows(
    name: str, rows: Sequence[Mapping[str, object]]
) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
    if not rows or any(type(row) is not dict for row in rows):
        raise P0RArtifactError(f"{name} rows are invalid")
    fields = tuple(sorted({field for row in rows for field in row}))
    if not fields or any(type(field) is not str or not field for field in fields):
        raise P0RArtifactError(f"{name} schema is invalid")
    required_keys = _CSV_KEYS[name]
    if any(key not in fields for key in required_keys):
        raise P0RArtifactError(f"{name} key schema is invalid")

    canonical: list[dict[str, object]] = []
    for row in rows:
        if _has_absolute_path(row):
            raise P0RArtifactError(f"{name} contains an absolute path")
        for value in row.values():
            _validate_cell(value, name=name)
        canonical.append({field: row.get(field, "") for field in fields})

    canonical.sort(
        key=lambda row: (
            tuple(str(row[key]) for key in required_keys),
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
        )
    )
    identities = [
        tuple(str(row[key]) for key in required_keys) for row in canonical
    ]
    if len(set(identities)) != len(identities):
        raise P0RArtifactError(f"{name} contains duplicate keys")
    return fields, tuple(canonical)


def _csv_bytes(name: str, rows: Sequence[Mapping[str, object]]) -> bytes:
    fields, canonical = _canonical_rows(name, rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(canonical)
    try:
        return stream.getvalue().encode("utf-8")
    except UnicodeEncodeError as error:
        raise P0RArtifactError(f"{name} cannot be encoded") from error


def _validate_author_authority(payload: object) -> dict[str, object]:
    if type(payload) is not dict or _has_absolute_path(payload):
        raise P0RArtifactError("P0R author authority is invalid")
    try:
        expected = build_author_registration_authority(
            original_artifact_sha256=payload.get("original_artifact_sha256")
        ).as_dict()
    except (TypeError, ValueError) as error:
        raise P0RArtifactError("P0R author authority is invalid") from error
    if payload != expected:
        raise P0RArtifactError("P0R author authority is invalid")
    return dict(payload)


def _verify_gate(summary: dict[str, Any]) -> None:
    facts_payload = summary.get("gate_facts")
    if facts_payload is None:
        raise P0RArtifactError("P0R gate facts are required")
    if type(facts_payload) is not dict:
        raise P0RArtifactError("P0R gate facts schema changed")
    try:
        facts = P0RGateFacts(**facts_payload)
    except (TypeError, ValueError) as error:
        raise P0RArtifactError("P0R gate facts are invalid") from error
    decision = decide_p0r(facts)
    expected = decision.as_dict()
    for field, value in expected.items():
        if summary.get(field) != value:
            raise P0RArtifactError(
                f"P0R {field} is inconsistent with recomputed gate"
            )


def _validate_summary(summary: object) -> dict[str, Any]:
    if (
        type(summary) is not dict
        or summary.get("status") not in _STATUSES
        or _has_absolute_path(summary)
    ):
        raise P0RArtifactError("P0R summary status is invalid")
    payload = dict(summary)
    _verify_gate(payload)
    return payload


def write_p0r_package(
    destination: str | Path,
    *,
    config_text: str,
    author_authority: Mapping[str, object],
    surface_manifest: Sequence[Mapping[str, object]],
    scan_processing_provenance: Sequence[Mapping[str, object]],
    registration: Sequence[Mapping[str, object]],
    registration_qc: Sequence[Mapping[str, object]],
    grid_mapping_qc: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    report: str,
) -> Path:
    """Write one complete P0R package by atomic rename without overwrite."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise P0RArtifactError("P0R output already exists")
    config_payload = _validate_text(config_text, label="config").encode("utf-8")
    report_payload = _validate_text(report, label="report").encode("utf-8")
    authority_payload = _validate_author_authority(author_authority)
    summary_payload = _validate_summary(summary)
    payload = {
        "config.yaml": config_payload,
        "author_authority.json": _json(authority_payload),
        "surface_manifest.csv": _csv_bytes(
            "surface_manifest.csv", surface_manifest
        ),
        "scan_processing_provenance.csv": _csv_bytes(
            "scan_processing_provenance.csv", scan_processing_provenance
        ),
        "registration.csv": _csv_bytes("registration.csv", registration),
        "registration_qc.csv": _csv_bytes(
            "registration_qc.csv", registration_qc
        ),
        "grid_mapping_qc.csv": _csv_bytes(
            "grid_mapping_qc.csv", grid_mapping_qc
        ),
        "summary.json": _json(summary_payload),
        "REPORT.md": report_payload,
    }

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise P0RArtifactError("P0R output parent is unavailable")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        file_records = {
            name: {"sha256": _sha256(value), "size": len(value)}
            for name, value in sorted(payload.items())
        }
        manifest = _json(
            {
                "schema_version": 1,
                "stage": "P0R",
                "status": summary_payload["status"],
                "files": file_records,
            }
        )
        complete = dict(payload)
        complete["artifact_manifest.json"] = manifest
        complete["CHECKSUMS.sha256"] = "".join(
            f"{_sha256(value)}  {name}\n"
            for name, value in sorted(complete.items())
        ).encode("ascii")
        if set(complete) != REQUIRED_P0R_FILES:
            raise P0RArtifactError(
                "P0R package membership is internally inconsistent"
            )
        for name, value in complete.items():
            (temporary / name).write_bytes(value)
        replay_p0r_package(temporary)
        if target.exists() or target.is_symlink():
            raise P0RArtifactError("P0R output already exists")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _load_json(root: Path, name: str) -> object:
    try:
        value = (root / name).read_bytes()
        payload = json.loads(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P0RArtifactError(f"P0R {name} is invalid") from error
    if value != _json(payload):
        raise P0RArtifactError(f"P0R {name} is not canonical")
    return payload


def _load_csv(root: Path, name: str) -> list[dict[str, str]]:
    try:
        value = (root / name).read_bytes()
        with io.StringIO(value.decode("utf-8"), newline="") as stream:
            reader = csv.DictReader(stream)
            if (
                reader.fieldnames is None
                or len(set(reader.fieldnames)) != len(reader.fieldnames)
            ):
                raise P0RArtifactError(f"{name} schema changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise P0RArtifactError(f"{name} is invalid") from error
    if value != _csv_bytes(name, rows):
        raise P0RArtifactError(f"{name} is not canonical")
    return rows


def replay_p0r_package(path: str | Path) -> dict[str, Any]:
    """Verify exact membership, hashes, schemas, authority, and P0R gate."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise P0RArtifactError("P0R package is unavailable")
    entries = tuple(root.iterdir())
    if (
        any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or {entry.name for entry in entries} != REQUIRED_P0R_FILES
    ):
        raise P0RArtifactError("P0R package membership changed")

    manifest_payload = _load_json(root, "artifact_manifest.json")
    if (
        type(manifest_payload) is not dict
        or set(manifest_payload)
        != {"schema_version", "stage", "status", "files"}
        or manifest_payload["schema_version"] != 1
        or manifest_payload["stage"] != "P0R"
        or manifest_payload["status"] not in _STATUSES
        or type(manifest_payload["files"]) is not dict
        or set(manifest_payload["files"]) != _PAYLOAD_FILES
    ):
        raise P0RArtifactError("P0R manifest schema changed")
    for name, record in manifest_payload["files"].items():
        if (
            type(record) is not dict
            or set(record) != {"sha256", "size"}
            or type(record["size"]) is not int
            or record["size"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise P0RArtifactError("P0R manifest file record changed")
        value = (root / name).read_bytes()
        if len(value) != record["size"] or _sha256(value) != record["sha256"]:
            raise P0RArtifactError(f"P0R file hash or size changed: {name}")

    expected_checksums = "".join(
        f"{_sha256((root / name).read_bytes())}  {name}\n"
        for name in sorted(_PAYLOAD_FILES | {"artifact_manifest.json"})
    ).encode("ascii")
    if (root / "CHECKSUMS.sha256").read_bytes() != expected_checksums:
        raise P0RArtifactError("P0R checksum ledger hash changed")

    for name in _CSV_KEYS:
        _load_csv(root, name)
    authority = _load_json(root, "author_authority.json")
    _validate_author_authority(authority)
    summary = _load_json(root, "summary.json")
    if (
        type(summary) is not dict
        or summary.get("status") != manifest_payload["status"]
    ):
        raise P0RArtifactError("P0R summary status is inconsistent")
    return _validate_summary(summary)


__all__ = [
    "REQUIRED_P0R_FILES",
    "P0RArtifactError",
    "replay_p0r_package",
    "write_p0r_package",
]
