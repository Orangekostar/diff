"""Deterministic publication and replay validation for the formal G0 package."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

REQUIRED_OUTPUTS = (
    "authorized_roster.csv",
    "zero_state_audit.csv",
    "cai_assessor_metrics.csv",
    "state_bank_manifest.csv",
    "initialization_curves.csv",
    "hierarchical_trajectories.parquet",
    "task_swap.csv",
    "stopping_results.csv",
    "domain_metrics.csv",
    "bootstrap.csv",
    "decision_summary.json",
    "REPORT.md",
)
_METADATA_FILES = frozenset(("artifact_manifest.json", "CHECKSUMS.sha256"))
_PACKAGE_FILES = frozenset((*REQUIRED_OUTPUTS, "config.yaml"))
_VALID_STATUSES = frozenset(
    (
        "G0_TASK_CONDITIONED_AGENTIC_OPPORTUNITY_GO",
        "G0_ACTIVE_INSPECTION_OPPORTUNITY_GO",
        "G0_FIELD_ONLY_OPPORTUNITY_GO",
        "G0_NO_AGENTIC_HEADROOM_NO_GO",
        "G0_CAI_ASSESSOR_NO_GO",
    )
)


class InspectionArtifactError(ValueError):
    """Raised when a G0 package is incomplete, inconsistent, or tampered."""


@dataclass(frozen=True, slots=True)
class G0PackageValidation:
    status: str
    output_tree_sha256: str
    manifest_sha256: str
    file_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class G0ReplayComparison:
    byte_identical: bool
    package_sha256: str
    replay_sha256: str


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _config(config_file: Path) -> dict[str, object]:
    try:
        loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise InspectionArtifactError("G0 config cannot be read") from error
    if not isinstance(loaded, dict) or not isinstance(loaded.get("sources"), dict):
        raise InspectionArtifactError("G0 source config is invalid")
    return loaded


def _source_records(
    root: Path,
    config: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    sources = config.get("sources")
    if not isinstance(sources, dict):
        raise InspectionArtifactError("G0 source config is invalid")
    output: dict[str, dict[str, object]] = {}
    for name, raw in sorted(sources.items()):
        if (
            not isinstance(name, str)
            or not isinstance(raw, dict)
            or set(raw) != {"path", "sha256"}
            or not isinstance(raw["path"], str)
            or not isinstance(raw["sha256"], str)
        ):
            raise InspectionArtifactError("G0 source config is invalid")
        relative = Path(raw["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise InspectionArtifactError("G0 source path is invalid")
        path = root / relative
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise InspectionArtifactError(f"G0 source is missing: {name}") from error
        actual = _sha(payload)
        if actual != raw["sha256"]:
            raise InspectionArtifactError(f"G0 source SHA-256 mismatch: {name}")
        output[name] = {
            "bytes": len(payload),
            "path": relative.as_posix(),
            "sha256": actual,
        }
    return output


def _records(output: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise InspectionArtifactError("G0 package must not contain symlinks")
        if not path.is_file() or path.name in _METADATA_FILES:
            continue
        relative = path.relative_to(output).as_posix()
        payload = path.read_bytes()
        records[relative] = {"bytes": len(payload), "sha256": _sha(payload)}
    if set(records) != _PACKAGE_FILES:
        raise InspectionArtifactError("G0 output file roster changed")
    return records


def _tree_sha(records: Mapping[str, Mapping[str, object]]) -> str:
    return _sha(_json_bytes({name: dict(records[name]) for name in sorted(records)}))


def _status(output: Path) -> str:
    try:
        payload = json.loads(
            (output / "decision_summary.json").read_text(encoding="utf-8")
        )
        value = payload["status"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise InspectionArtifactError("G0 decision status is invalid") from error
    if value not in _VALID_STATUSES:
        raise InspectionArtifactError("G0 decision status is invalid")
    return str(value)


def _check_required(output: Path, config_file: Path) -> None:
    if not output.is_dir():
        raise InspectionArtifactError("G0 output directory is missing")
    missing = [name for name in _PACKAGE_FILES if not (output / name).is_file()]
    if missing:
        raise InspectionArtifactError(f"required G0 outputs are missing: {missing}")
    if (output / "config.yaml").read_bytes() != config_file.read_bytes():
        raise InspectionArtifactError("published G0 config differs from frozen config")


def _check_path_privacy(output: Path, root: Path) -> None:
    forbidden = {root.as_posix().encode(), Path.home().as_posix().encode()}
    text_suffixes = {".csv", ".json", ".md", ".sha256", ".yaml"}
    for path in output.rglob("*"):
        if path.is_file() and (
            path.suffix in text_suffixes or path.name == "CHECKSUMS.sha256"
        ):
            payload = path.read_bytes()
            if any(value and value in payload for value in forbidden):
                raise InspectionArtifactError("G0 artifact contains a private absolute path")


def publish_g0_manifest(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> G0PackageValidation:
    root = Path(project_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = _config(config_file)
    _check_required(output, config_file)
    _check_path_privacy(output, root)
    records = _records(output)
    manifest = {
        "schema_version": 1,
        "scope": "inspection_agent_g0_opportunity_audit",
        "status": _status(output),
        "config_sha256": _sha(config_file.read_bytes()),
        "repository_base_sha": config.get("repository_base_sha"),
        "sources": _source_records(root, config),
        "outputs": records,
        "output_tree_sha256": _tree_sha(records),
    }
    manifest_bytes = _json_bytes(manifest)
    (output / "artifact_manifest.json").write_bytes(manifest_bytes)
    checksums = {
        **{name: str(record["sha256"]) for name, record in records.items()},
        "artifact_manifest.json": _sha(manifest_bytes),
    }
    (output / "CHECKSUMS.sha256").write_text(
        "".join(f"{checksums[name]}  {name}\n" for name in sorted(checksums)),
        encoding="ascii",
    )
    return validate_g0_package(
        output,
        project_root=root,
        config_path=config_file,
    )


def validate_g0_package(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> G0PackageValidation:
    root = Path(project_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = _config(config_file)
    _check_required(output, config_file)
    _check_path_privacy(output, root)
    try:
        manifest_payload = (output / "artifact_manifest.json").read_bytes()
        manifest = json.loads(manifest_payload)
        checksum_lines = (
            (output / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InspectionArtifactError("G0 artifact metadata cannot be read") from error
    expected_keys = {
        "schema_version",
        "scope",
        "status",
        "config_sha256",
        "repository_base_sha",
        "sources",
        "outputs",
        "output_tree_sha256",
    }
    if set(manifest) != expected_keys:
        raise InspectionArtifactError("G0 artifact manifest schema changed")
    records = _records(output)
    if manifest["outputs"] != records or manifest["output_tree_sha256"] != _tree_sha(
        records
    ):
        raise InspectionArtifactError("G0 output checksum mismatch")
    if (
        manifest["status"] != _status(output)
        or manifest["config_sha256"] != _sha(config_file.read_bytes())
        or manifest["repository_base_sha"] != config.get("repository_base_sha")
        or manifest["sources"] != _source_records(root, config)
    ):
        raise InspectionArtifactError("G0 authority binding changed")
    checksums: dict[str, str] = {}
    for line in checksum_lines:
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or parts[1] in checksums:
            raise InspectionArtifactError("G0 CHECKSUMS.sha256 is invalid")
        checksums[parts[1]] = parts[0]
    expected_checksums = {
        **{name: str(record["sha256"]) for name, record in records.items()},
        "artifact_manifest.json": _sha(manifest_payload),
    }
    if checksums != expected_checksums:
        raise InspectionArtifactError("G0 CHECKSUMS.sha256 mismatch")
    return G0PackageValidation(
        status=str(manifest["status"]),
        output_tree_sha256=str(manifest["output_tree_sha256"]),
        manifest_sha256=_sha(manifest_payload),
        file_sha256=MappingProxyType(
            {name: str(record["sha256"]) for name, record in records.items()}
        ),
    )


def _package_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("ascii")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def compare_g0_packages(
    formal_dir: str | Path,
    replay_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> G0ReplayComparison:
    formal = Path(formal_dir).resolve(strict=True)
    replay = Path(replay_dir).resolve(strict=True)
    validate_g0_package(formal, project_root=project_root, config_path=config_path)
    validate_g0_package(replay, project_root=project_root, config_path=config_path)
    formal_files = {
        path.relative_to(formal).as_posix(): path
        for path in formal.rglob("*")
        if path.is_file()
    }
    replay_files = {
        path.relative_to(replay).as_posix(): path
        for path in replay.rglob("*")
        if path.is_file()
    }
    if set(formal_files) != set(replay_files) or any(
        formal_files[name].read_bytes() != replay_files[name].read_bytes()
        for name in formal_files
    ):
        raise InspectionArtifactError("G0 replay package is not byte-identical")
    formal_sha = _package_sha(formal)
    replay_sha = _package_sha(replay)
    return G0ReplayComparison(
        byte_identical=True,
        package_sha256=formal_sha,
        replay_sha256=replay_sha,
    )


__all__ = [
    "REQUIRED_OUTPUTS",
    "G0PackageValidation",
    "G0ReplayComparison",
    "InspectionArtifactError",
    "compare_g0_packages",
    "publish_g0_manifest",
    "validate_g0_package",
]
