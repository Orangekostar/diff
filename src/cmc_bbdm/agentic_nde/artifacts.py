"""Deterministic, checksum-bound P0 authority packages."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import P0GateFacts, StageStatus, decide_p0


class ArtifactError(ValueError):
    """Raised when a P0 package is unsafe, incomplete, or changed."""


REQUIRED_P0_FILES = frozenset(
    {
        "config.yaml",
        "surface_manifest.csv",
        "surface_qc.csv",
        "registration.csv",
        "registration_qc.csv",
        "source_hashes.csv",
        "summary.json",
        "REPORT.md",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
)
_PAYLOAD_FILES = REQUIRED_P0_FILES - {"artifact_manifest.json", "CHECKSUMS.sha256"}
_CSV_KEYS = {
    "surface_manifest.csv": ("dataset_id", "specimen_id"),
    "surface_qc.csv": ("dataset_id", "specimen_id"),
    "registration.csv": ("dataset_id", "specimen_id"),
    "registration_qc.csv": ("dataset_id", "specimen_id"),
    "source_hashes.csv": ("logical_path",),
}
_STATUSES = frozenset(
    {
        StageStatus.P0_GO.value,
        StageStatus.P0_SPATIAL_REGISTRATION_NO_GO.value,
        StageStatus.P0_IDENTITY_AUTHORITY_NO_GO.value,
    }
)
_HEX = frozenset("0123456789abcdef")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ArtifactError("P0 JSON payload is not canonicalizable") from error
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


def _canonical_rows(
    name: str, rows: Sequence[Mapping[str, object]]
) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
    if not rows or any(type(row) is not dict for row in rows):
        raise ArtifactError(f"{name} rows are invalid")
    fields = tuple(sorted({field for row in rows for field in row}))
    if not fields or any(type(field) is not str or not field for field in fields):
        raise ArtifactError(f"{name} schema is invalid")
    canonical: list[dict[str, object]] = []
    for row in rows:
        if _has_absolute_path(row):
            raise ArtifactError(f"{name} contains an absolute path")
        canonical.append({field: row.get(field, "") for field in fields})
    canonical.sort(
        key=lambda row: json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    keys = _CSV_KEYS[name]
    if all(key in fields for key in keys):
        identities = [tuple(str(row[key]) for key in keys) for row in canonical]
        if len(set(identities)) != len(identities):
            raise ArtifactError(f"{name} contains duplicate keys")
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
        raise ArtifactError(f"{name} cannot be encoded") from error


def _validate_summary(summary: Mapping[str, object]) -> dict[str, object]:
    if type(summary) is not dict or summary.get("status") not in _STATUSES:
        raise ArtifactError("P0 summary status is invalid")
    if _has_absolute_path(summary):
        raise ArtifactError("P0 summary contains an absolute path")
    return dict(summary)


def write_p0_package(
    destination: str | Path,
    *,
    config_text: str,
    surface_manifest: Sequence[Mapping[str, object]],
    surface_qc: Sequence[Mapping[str, object]],
    registration: Sequence[Mapping[str, object]],
    registration_qc: Sequence[Mapping[str, object]],
    source_hashes: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    report: str,
) -> Path:
    """Write one complete P0 package by atomic rename without overwrite."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise ArtifactError("P0 output already exists")
    if type(config_text) is not str or not config_text or _has_absolute_path(config_text):
        raise ArtifactError("P0 config text is invalid")
    if type(report) is not str or not report or _has_absolute_path(report):
        raise ArtifactError("P0 report text is invalid")
    summary_payload = _validate_summary(summary)
    _verify_gate(summary_payload)
    payload = {
        "config.yaml": config_text.encode("utf-8"),
        "surface_manifest.csv": _csv_bytes("surface_manifest.csv", surface_manifest),
        "surface_qc.csv": _csv_bytes("surface_qc.csv", surface_qc),
        "registration.csv": _csv_bytes("registration.csv", registration),
        "registration_qc.csv": _csv_bytes("registration_qc.csv", registration_qc),
        "source_hashes.csv": _csv_bytes("source_hashes.csv", source_hashes),
        "summary.json": _json(summary_payload),
        "REPORT.md": report.encode("utf-8"),
    }
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ArtifactError("P0 output parent is unavailable")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        file_records = {
            name: {"sha256": _sha256(value), "size": len(value)}
            for name, value in sorted(payload.items())
        }
        manifest = _json(
            {
                "schema_version": 1,
                "stage": "P0",
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
        if set(complete) != REQUIRED_P0_FILES:
            raise ArtifactError("P0 package membership is internally inconsistent")
        for name, value in complete.items():
            (temporary / name).write_bytes(value)
        replay_p0(temporary)
        if target.exists() or target.is_symlink():
            raise ArtifactError("P0 output already exists")
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _load_csv(root: Path, name: str) -> list[dict[str, str]]:
    try:
        with (root / name).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ArtifactError(f"{name} schema changed")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ArtifactError(f"{name} is invalid") from error
    if not rows:
        raise ArtifactError(f"{name} contains no rows")
    keys = _CSV_KEYS[name]
    if all(key in reader.fieldnames for key in keys):
        identities = [tuple(row[key] for key in keys) for row in rows]
        if len(set(identities)) != len(identities):
            raise ArtifactError(f"{name} contains duplicate keys")
    if _has_absolute_path(rows):
        raise ArtifactError(f"{name} contains an absolute path")
    return rows


def _verify_gate(summary: dict[str, Any]) -> None:
    facts_payload = summary.get("gate_facts")
    if facts_payload is None:
        raise ArtifactError("P0 gate facts are required")
    if type(facts_payload) is not dict:
        raise ArtifactError("P0 gate facts schema changed")
    try:
        facts = P0GateFacts(**facts_payload)
    except (TypeError, ValueError) as error:
        raise ArtifactError("P0 gate facts are invalid") from error
    decision = decide_p0(facts)
    if summary.get("status") != decision.status.value:
        raise ArtifactError("P0 status is inconsistent with recomputed gate")
    if summary.get("reasons") != list(decision.reasons):
        raise ArtifactError("P0 reasons are inconsistent with recomputed gate")
    expected_downstream = {stage: status for stage, status in decision.downstream}
    if summary.get("downstream") != expected_downstream:
        raise ArtifactError("P0 downstream authorization is inconsistent")


def _verify_source_rows(
    rows: list[dict[str, str]],
    *,
    surface_root: str | Path | None,
    project_root: str | Path | None,
) -> None:
    roots: dict[str, Path] = {}
    for logical_root, raw_root in (
        ("external_hasebe", surface_root),
        ("compact_repository", project_root),
    ):
        if raw_root is None:
            continue
        authority_root = Path(raw_root)
        if authority_root.is_symlink() or not authority_root.is_dir():
            raise ArtifactError(f"{logical_root} authority root is unavailable")
        roots[logical_root] = authority_root.resolve(strict=True)
    if not roots:
        return
    required = {"logical_path", "logical_root", "relative_path", "bytes", "sha256"}
    for row in rows:
        if not required <= set(row):
            raise ArtifactError("P0 source hash schema changed")
        logical_root = row["logical_root"]
        if logical_root not in {"external_hasebe", "compact_repository"}:
            raise ArtifactError("P0 source logical root is invalid")
        if logical_root not in roots:
            continue
        relative = Path(row["relative_path"])
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or row["logical_path"] != f"{logical_root}/{relative.as_posix()}"
            or not _is_sha256(row["sha256"])
        ):
            raise ArtifactError("P0 source authority path or hash is invalid")
        try:
            expected_size = int(row["bytes"])
        except ValueError as error:
            raise ArtifactError("P0 source authority size is invalid") from error
        root = roots[logical_root]
        unresolved = root
        for part in relative.parts:
            unresolved /= part
            if unresolved.is_symlink():
                raise ArtifactError("P0 source authority contains a symlink")
        try:
            source = unresolved.resolve(strict=True)
            source.relative_to(root)
        except (OSError, ValueError) as error:
            raise ArtifactError("P0 source authority path escapes its root") from error
        if (
            not source.is_file()
            or source.stat().st_size != expected_size
            or _file_sha256(source) != row["sha256"]
        ):
            raise ArtifactError("P0 source authority hash or size changed")


def replay_p0(
    path: str | Path,
    *,
    surface_root: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify exact membership, hashes, schemas, and the machine-readable gate."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactError("P0 package is unavailable")
    entries = tuple(root.iterdir())
    if (
        any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or {entry.name for entry in entries} != REQUIRED_P0_FILES
    ):
        raise ArtifactError("P0 package membership changed")
    try:
        manifest_bytes = (root / "artifact_manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError("P0 manifest is invalid") from error
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema_version", "stage", "status", "files"}
        or manifest["schema_version"] != 1
        or manifest["stage"] != "P0"
        or manifest["status"] not in _STATUSES
        or type(manifest["files"]) is not dict
        or set(manifest["files"]) != _PAYLOAD_FILES
    ):
        raise ArtifactError("P0 manifest schema changed")
    for name, record in manifest["files"].items():
        if (
            type(record) is not dict
            or set(record) != {"sha256", "size"}
            or type(record["size"]) is not int
            or record["size"] < 0
            or not _is_sha256(record["sha256"])
        ):
            raise ArtifactError("P0 manifest file record changed")
        value = (root / name).read_bytes()
        if len(value) != record["size"] or _sha256(value) != record["sha256"]:
            raise ArtifactError(f"P0 file hash or size changed: {name}")
    expected_checksums = "".join(
        f"{_sha256((root / name).read_bytes())}  {name}\n"
        for name in sorted(_PAYLOAD_FILES | {"artifact_manifest.json"})
    ).encode("ascii")
    if (root / "CHECKSUMS.sha256").read_bytes() != expected_checksums:
        raise ArtifactError("P0 checksum ledger hash changed")
    csv_rows = {name: _load_csv(root, name) for name in _CSV_KEYS}
    try:
        summary = json.loads((root / "summary.json").read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError("P0 summary is invalid") from error
    if type(summary) is not dict or summary.get("status") != manifest["status"]:
        raise ArtifactError("P0 summary status is inconsistent")
    _validate_summary(summary)
    _verify_gate(summary)
    _verify_source_rows(
        csv_rows["source_hashes.csv"],
        surface_root=surface_root,
        project_root=project_root,
    )
    return summary


__all__ = ["REQUIRED_P0_FILES", "ArtifactError", "replay_p0", "write_p0_package"]
