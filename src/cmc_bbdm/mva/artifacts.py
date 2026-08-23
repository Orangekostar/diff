"""Checksum-bound publication and validation for the formal MVA A2 package."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .config import load_mva_config

REQUIRED_OUTPUTS = (
    "oracle_values.parquet",
    "oracle_trajectories.parquet",
    "state_metrics.parquet",
    "uniform_curve.csv",
    "random_curve.csv",
    "appearance_curve.csv",
    "reconstruction_oracle_curve.csv",
    "mechanical_oracle_curve.csv",
    "budget_metrics.csv",
    "domain_metrics.csv",
    "map_similarity.csv",
    "stability_diagnostics.csv",
    "bootstrap.csv",
    "summary.json",
    "REPORT.md",
    "figures/O1_current_sparse_scan_mask.png",
    "figures/O2_reconstruction_value_map.png",
    "figures/O3_appearance_value_map.png",
    "figures/O4_mechanical_value_map.png",
    "figures/O5_acquisition_trajectories.png",
    "figures/error_budget_curve.png",
    "figures/source_data.csv",
)
_METADATA_FILES = frozenset(("artifact_manifest.json", "CHECKSUMS.sha256"))
_VALID_STATUSES = frozenset(("MVA_ORACLE_GO", "MVA_ORACLE_NO_GO"))


class MVAArtifactError(ValueError):
    """Raised when an MVA package is incomplete, inconsistent, or tampered."""


@dataclass(frozen=True, slots=True)
class MVAPackageValidation:
    status: str
    output_tree_sha256: str
    manifest_sha256: str
    file_sha256: Mapping[str, str]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _records(output: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise MVAArtifactError("artifact package must not contain symlinks")
        if not path.is_file() or path.name in _METADATA_FILES:
            continue
        relative = path.relative_to(output).as_posix()
        payload = path.read_bytes()
        records[relative] = {"bytes": len(payload), "sha256": _sha(payload)}
    return records


def _tree_sha(records: Mapping[str, Mapping[str, object]]) -> str:
    return _sha(_json_bytes({name: dict(records[name]) for name in sorted(records)}))


def _upstream_records(root: Path, output_root: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for leaf in ("a0_acquisition_audit", "a1_simulator"):
        directory = root / output_root / leaf
        if not directory.is_dir():
            raise MVAArtifactError(f"required upstream MVA output is missing: {leaf}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            records[relative] = {"bytes": len(payload), "sha256": _sha(payload)}
    if not records:
        raise MVAArtifactError("upstream MVA outputs are empty")
    return records


def _status(output: Path) -> str:
    try:
        value = json.loads((output / "summary.json").read_text(encoding="utf-8"))[
            "status"
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise MVAArtifactError("summary status is invalid") from error
    if value not in _VALID_STATUSES:
        raise MVAArtifactError("summary status is invalid")
    return str(value)


def _check_required(output: Path) -> None:
    missing = [
        relative
        for relative in (*REQUIRED_OUTPUTS, "config.yaml")
        if not (output / relative).is_file()
    ]
    if missing:
        raise MVAArtifactError(f"required MVA outputs are missing: {missing}")


def _check_path_privacy(output: Path, root: Path) -> None:
    forbidden = {root.as_posix().encode(), Path.home().as_posix().encode()}
    text_suffixes = {".csv", ".json", ".md", ".sha256", ".svg", ".yaml"}
    for path in output.rglob("*"):
        if path.is_file() and (
            path.suffix in text_suffixes or path.name == "CHECKSUMS.sha256"
        ):
            payload = path.read_bytes()
            if any(value and value in payload for value in forbidden):
                raise MVAArtifactError("artifact contains a private absolute path")


def publish_mva_manifest(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> MVAPackageValidation:
    """Bind a completed MVA output tree to its frozen config and source hashes."""

    root = Path(project_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_mva_config(config_file, project_root=root)
    _check_required(output)
    _check_path_privacy(output, root)
    if (output / "config.yaml").read_bytes() != config_file.read_bytes():
        raise MVAArtifactError("published config differs from the frozen config")
    status = _status(output)
    records = _records(output)
    upstream = _upstream_records(root, config.output_dir)
    manifest = {
        "schema_version": 1,
        "scope": "mva_a0_a3_oracle_headroom",
        "status": status,
        "config_sha256": _sha(config_file.read_bytes()),
        "sources": {
            name: {"path": source.path.as_posix(), "sha256": source.sha256}
            for name, source in sorted(config.sources.items())
        },
        "upstream_outputs": upstream,
        "outputs": records,
        "output_tree_sha256": _tree_sha(records),
    }
    manifest_bytes = _json_bytes(manifest)
    (output / "artifact_manifest.json").write_bytes(manifest_bytes)
    checksum_records = {
        **{name: str(record["sha256"]) for name, record in records.items()},
        "artifact_manifest.json": _sha(manifest_bytes),
    }
    checksum_payload = "".join(
        f"{checksum_records[name]}  {name}\n" for name in sorted(checksum_records)
    ).encode("ascii")
    (output / "CHECKSUMS.sha256").write_bytes(checksum_payload)
    return validate_mva_package(output, project_root=root, config_path=config_file)


def validate_mva_package(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> MVAPackageValidation:
    """Independently validate package completeness, hashes, and authority binding."""

    root = Path(project_root).resolve(strict=True)
    output = Path(output_dir).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_mva_config(config_file, project_root=root)
    _check_required(output)
    _check_path_privacy(output, root)
    if (output / "config.yaml").read_bytes() != config_file.read_bytes():
        raise MVAArtifactError("published config differs from the frozen config")
    try:
        manifest_payload = (output / "artifact_manifest.json").read_bytes()
        manifest = json.loads(manifest_payload)
        checksum_lines = (
            (output / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MVAArtifactError("artifact metadata cannot be read") from error
    records = _records(output)
    expected_manifest_keys = {
        "schema_version",
        "scope",
        "status",
        "config_sha256",
        "sources",
        "upstream_outputs",
        "outputs",
        "output_tree_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise MVAArtifactError("artifact manifest schema changed")
    if manifest["outputs"] != records or manifest["output_tree_sha256"] != _tree_sha(
        records
    ):
        raise MVAArtifactError("artifact checksum mismatch")
    if manifest["config_sha256"] != _sha(config_file.read_bytes()):
        raise MVAArtifactError("config SHA-256 mismatch")
    expected_sources = {
        name: {"path": source.path.as_posix(), "sha256": source.sha256}
        for name, source in sorted(config.sources.items())
    }
    if manifest["sources"] != expected_sources or manifest["status"] != _status(output):
        raise MVAArtifactError("artifact authority binding changed")
    if manifest["upstream_outputs"] != _upstream_records(root, config.output_dir):
        raise MVAArtifactError("upstream output checksum mismatch")
    checksums: dict[str, str] = {}
    for line in checksum_lines:
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2 or parts[1] in checksums:
            raise MVAArtifactError("CHECKSUMS.sha256 is invalid")
        checksums[parts[1]] = parts[0]
    expected_checksums = {
        **{name: str(record["sha256"]) for name, record in records.items()},
        "artifact_manifest.json": _sha(manifest_payload),
    }
    if checksums != expected_checksums:
        raise MVAArtifactError("CHECKSUMS.sha256 mismatch")
    return MVAPackageValidation(
        status=str(manifest["status"]),
        output_tree_sha256=str(manifest["output_tree_sha256"]),
        manifest_sha256=_sha(manifest_payload),
        file_sha256=MappingProxyType(
            {name: str(record["sha256"]) for name, record in records.items()}
        ),
    )


__all__ = [
    "REQUIRED_OUTPUTS",
    "MVAArtifactError",
    "MVAPackageValidation",
    "publish_mva_manifest",
    "validate_mva_package",
]
