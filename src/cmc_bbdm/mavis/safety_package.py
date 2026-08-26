"""Finalize and verify the MAVIS P6 source-only safety calibration package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from .config import load_mavis_config
from .dynamic_package import verify_dynamic_package
from .mris_package import verify_mris_package
from .state_bank_package import (
    load_state_manifest_package,
    verify_state_bank_package,
)


class MAVISSafetyPackageError(RuntimeError):
    """Raised when P6 safety workers or their package are inconsistent."""


_WORKER_FILES = {
    "calibration_predictions.parquet",
    "selection.json",
    "source_metrics.csv",
    "threshold_audit.csv",
}
_CODE_PATHS = (
    "src/cmc_bbdm/mavis/closed_loop_execution.py",
    "src/cmc_bbdm/mavis/closed_loop_metrics.py",
    "src/cmc_bbdm/mavis/dynamic_training.py",
    "src/cmc_bbdm/mavis/fallback.py",
    "src/cmc_bbdm/mavis/mris_training.py",
    "src/cmc_bbdm/mavis/policy.py",
    "src/cmc_bbdm/mavis/rollout.py",
    "src/cmc_bbdm/mavis/safety_execution.py",
    "src/cmc_bbdm/mavis/safety_package.py",
)
_GIT_BLOB_LIMIT = 100 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_checksums(output: Path) -> None:
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    (output / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in files
        ),
        encoding="ascii",
    )


def _repository_state(root: Path) -> tuple[str, str, bool]:
    digest = hashlib.sha256()
    for relative in _CODE_PATHS:
        path = root / relative
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise MAVISSafetyPackageError(
                f"P6 runtime source is unavailable: {relative}"
            ) from error
        digest.update(relative.encode("utf-8"))
        digest.update(hashlib.sha256(payload).digest())
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise MAVISSafetyPackageError("P6 Git provenance is unavailable") from error
    if len(git_sha) != 40:
        raise MAVISSafetyPackageError("P6 Git SHA is invalid")
    return git_sha, digest.hexdigest(), dirty


def _worker(
    path: Path,
    *,
    outer_domain: str,
    config_sha256: str,
    source_domains: tuple[str, ...],
    source_specimen_count: int,
) -> dict[str, object]:
    try:
        payload = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISSafetyPackageError("P6 worker summary is invalid") from error
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("outer_domain") != outer_domain
        or payload.get("config_sha256") != config_sha256
        or tuple(payload.get("source_domains", ())) != source_domains
        or payload.get("source_specimen_count") != source_specimen_count
        or set(payload.get("p2_inner_model_state_sha256", {})) != set(source_domains)
        or set(payload.get("p3_inner_model_state_sha256", {})) != set(source_domains)
        or payload.get("baseline") not in {"uniform", "reconstruction"}
        or payload.get("calibration_policy") != "nested_p3_pre_aggregation"
        or payload.get("target_outcomes_used_for_selection") is not False
        or type(payload.get("files")) is not dict
        or set(payload["files"]) != _WORKER_FILES
    ):
        raise MAVISSafetyPackageError("P6 worker contract changed")
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "complete.json"
    }
    if actual != _WORKER_FILES:
        raise MAVISSafetyPackageError("P6 worker file roster changed")
    for relative, expected in payload["files"].items():
        if type(expected) is not str or _sha256(path / relative) != expected:
            raise MAVISSafetyPackageError(
                f"P6 worker checksum mismatch: {relative}"
            )
    return payload


def finalize_safety_package(
    config_path: str | Path,
    *,
    project_root: str | Path,
    worker_root: str | Path,
    p1_package: str | Path,
    p2_package: str | Path,
    p3_package: str | Path,
) -> Path:
    root = Path(project_root).resolve(strict=True)
    config = load_mavis_config(Path(config_path).resolve(strict=True), project_root=root)
    p1_root = Path(p1_package).resolve(strict=True)
    p2_root = Path(p2_package).resolve(strict=True)
    p3_root = Path(p3_package).resolve(strict=True)
    p1_manifest = verify_state_bank_package(p1_root)
    p2_manifest = verify_mris_package(p2_root)
    p3_manifest = verify_dynamic_package(p3_root)
    state_manifest = load_state_manifest_package(p1_root)
    workers_root = Path(worker_root).resolve(strict=True)
    predictions_parts: list[pl.DataFrame] = []
    metrics_parts: list[pl.DataFrame] = []
    audit_parts: list[pl.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    workers: list[dict[str, object]] = []
    for outer_domain in config.domain_order:
        source_domains = tuple(
            domain for domain in config.domain_order if domain != outer_domain
        )
        source_count = state_manifest.filter(
            pl.col("domain_id") != outer_domain
        ).get_column("specimen_id").n_unique()
        worker_path = workers_root / outer_domain
        worker = _worker(
            worker_path,
            outer_domain=outer_domain,
            config_sha256=config.config_sha256,
            source_domains=source_domains,
            source_specimen_count=source_count,
        )
        workers.append(worker)
        try:
            predictions_parts.append(
                pl.read_parquet(worker_path / "calibration_predictions.parquet")
                .with_columns(pl.lit(outer_domain).alias("selection_outer_domain"))
            )
            metrics_parts.append(
                pl.read_csv(worker_path / "source_metrics.csv").with_columns(
                    pl.lit(outer_domain).alias("selection_outer_domain")
                )
            )
            audit_parts.append(
                pl.read_csv(worker_path / "threshold_audit.csv").with_columns(
                    pl.lit(outer_domain).alias("selection_outer_domain")
                )
            )
            selection = json.loads(
                (worker_path / "selection.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError, pl.exceptions.PolarsError) as error:
            raise MAVISSafetyPackageError("P6 worker artifact is invalid") from error
        selection_rows.append(
            {
                "outer_domain": outer_domain,
                "baseline": selection["baseline"],
                "threshold": selection["threshold"],
                "source_domain_count": len(source_domains),
                "source_specimen_count": source_count,
                "selection_state_sha256": selection["state_sha256"],
                "target_outcomes_used": selection["target_outcomes_used"],
            }
        )
    predictions = pl.concat(predictions_parts, how="vertical_relaxed").sort(
        [
            "selection_outer_domain",
            "outer_domain",
            "specimen_id",
            "method",
            "nominal_checkpoint",
        ]
    )
    metrics = pl.concat(metrics_parts, how="vertical_relaxed").sort(
        ["selection_outer_domain", "domain_id", "specimen_id"]
    )
    audits = pl.concat(audit_parts, how="vertical_relaxed").sort(
        ["selection_outer_domain", "threshold"]
    )
    selections = pl.DataFrame(selection_rows, infer_schema_length=None).sort(
        "outer_domain"
    )
    expected_pairs = sum(
        state_manifest.filter(pl.col("domain_id") != outer).get_column("specimen_id").n_unique()
        for outer in config.domain_order
    )
    if (
        metrics.height != expected_pairs
        or predictions.height
        != expected_pairs * 3 * len(config.checkpoints)
        or audits.height != len(config.domain_order) * len(config.confidence_thresholds)
        or selections.height != len(config.domain_order)
        or selections.filter(pl.col("target_outcomes_used") != False).height
        or metrics.filter(
            pl.col("selection_outer_domain") == pl.col("domain_id")
        ).height
    ):
        raise MAVISSafetyPackageError("P6 scientific table roster is incomplete")
    git_sha, code_state_sha256, dirty = _repository_state(root)
    p6_state = hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "config_sha256": config.config_sha256,
                "p1_state_sha256": p1_manifest.get("state_bank_state_sha256"),
                "p2_state_sha256": p2_manifest.get("p2_state_sha256"),
                "p3_state_sha256": p3_manifest.get("p3_state_sha256"),
                "workers": workers,
                "runtime_code_state_sha256": code_state_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    output = root / config.output_root / "p6_safety"
    if output.exists():
        raise MAVISSafetyPackageError("P6 formal package already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p6_safety.", dir=output.parent))
    try:
        predictions.write_parquet(
            temporary / "calibration_predictions.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        metrics.write_csv(temporary / "source_metrics.csv")
        audits.write_csv(temporary / "threshold_audit.csv")
        selections.write_csv(temporary / "selections.csv")
        _write_json(
            temporary / "summary.json",
            {
                "schema_version": 1,
                "stage": "P6_SOURCE_ONLY_CONFIDENCE_FALLBACK",
                "status": "COMPLETE",
                "git_sha": git_sha,
                "git_worktree_dirty_at_run": dirty,
                "runtime_code_state_sha256": code_state_sha256,
                "config_sha256": config.config_sha256,
                "p1_state_sha256": p1_manifest.get("state_bank_state_sha256"),
                "p2_state_sha256": p2_manifest.get("p2_state_sha256"),
                "p3_state_sha256": p3_manifest.get("p3_state_sha256"),
                "p6_state_sha256": p6_state,
                "outer_domain_count": len(config.domain_order),
                "nested_source_specimen_pairs": metrics.height,
                "target_outcomes_used_for_selection": False,
                "calibration_policy": "nested_p3_pre_aggregation",
            },
        )
        (temporary / "REPORT.md").write_text(
            "# MAVIS P6 Source-only Confidence Fallback\n\n"
            "Status: `COMPLETE`. Baseline and confidence threshold are selected "
            "separately for each outer target using only double-held-out source "
            "validation curves. The first post-scout decision confidence routes a "
            "physical specimen to MAVIS or the selected robust baseline. Outer-target "
            "outcomes are not used. Calibration uses nested pre-aggregation P3 models "
            "so it remains independent of source on-policy training outcomes.\n",
            encoding="utf-8",
        )
        scientific_files = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        )
        manifest = {
            "schema_version": 1,
            "artifact": "mavis_p6_safety",
            "p6_state_sha256": p6_state,
            "config_sha256": config.config_sha256,
            "runtime_code_state_sha256": code_state_sha256,
            "files": {
                name: {
                    "bytes": (temporary / name).stat().st_size,
                    "sha256": _sha256(temporary / name),
                }
                for name in scientific_files
            },
        }
        _write_json(temporary / "artifact_manifest.json", manifest)
        _write_checksums(temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    verify_safety_package(output)
    return output


def verify_safety_package(path: str | Path) -> dict[str, object]:
    root = Path(path)
    try:
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        lines = (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISSafetyPackageError("P6 package metadata is invalid") from error
    if type(manifest) is not dict or type(manifest.get("files")) is not dict:
        raise MAVISSafetyPackageError("P6 package manifest is invalid")
    expected = set(manifest["files"]) | {"artifact_manifest.json"}
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "CHECKSUMS.sha256"
    }
    if actual != expected or len(lines) != len(expected):
        raise MAVISSafetyPackageError("P6 package file roster changed")
    ledger: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise MAVISSafetyPackageError("P6 checksum ledger is invalid") from error
        if name in ledger or len(digest) != 64:
            raise MAVISSafetyPackageError("P6 checksum ledger is invalid")
        ledger[name] = digest
    if set(ledger) != expected:
        raise MAVISSafetyPackageError("P6 checksum roster changed")
    for name, expected_digest in ledger.items():
        if _sha256(root / name) != expected_digest:
            raise MAVISSafetyPackageError(f"P6 checksum mismatch: {name}")
    for name, metadata in manifest["files"].items():
        file_path = root / name
        if (
            type(metadata) is not dict
            or metadata.get("bytes") != file_path.stat().st_size
            or metadata.get("sha256") != _sha256(file_path)
        ):
            raise MAVISSafetyPackageError(f"P6 manifest mismatch: {name}")
    if any((root / name).stat().st_size >= _GIT_BLOB_LIMIT for name in expected):
        raise MAVISSafetyPackageError("P6 package contains an oversized Git blob")
    return manifest


__all__ = [
    "MAVISSafetyPackageError",
    "finalize_safety_package",
    "verify_safety_package",
]
