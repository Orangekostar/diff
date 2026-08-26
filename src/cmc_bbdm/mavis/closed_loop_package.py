"""Finalize and verify the MAVIS P4 closed-loop evidence package."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from .closed_loop_execution import PRIMARY_CLOSED_LOOP_METHODS
from .closed_loop_figure import render_task_specificity_curve
from .closed_loop_metrics import (
    bootstrap_closed_loop_contrasts,
    evaluate_closed_loop_predictions,
)
from .config import load_mavis_config
from .dynamic_package import verify_dynamic_package
from .mris_package import verify_mris_package
from .state_bank_package import (
    load_state_manifest_package,
    verify_state_bank_package,
)


class MAVISClosedLoopPackageError(RuntimeError):
    """Raised when P4 workers or their formal package are inconsistent."""


_WORKER_FILES = {
    "full_scan_anchors.parquet",
    "predictions.parquet",
    "trajectories.parquet",
}
_CODE_PATHS = (
    "src/cmc_bbdm/mavis/closed_loop_execution.py",
    "src/cmc_bbdm/mavis/closed_loop_figure.py",
    "src/cmc_bbdm/mavis/closed_loop_metrics.py",
    "src/cmc_bbdm/mavis/closed_loop_package.py",
    "src/cmc_bbdm/mavis/historical_sources.py",
    "src/cmc_bbdm/mavis/policy.py",
    "src/cmc_bbdm/mavis/rollout.py",
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
            raise MAVISClosedLoopPackageError(
                f"P4 runtime source is unavailable: {relative}"
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
        raise MAVISClosedLoopPackageError("P4 Git provenance is unavailable") from error
    if len(git_sha) != 40:
        raise MAVISClosedLoopPackageError("P4 Git SHA is invalid")
    return git_sha, digest.hexdigest(), dirty


def _worker(
    path: Path,
    *,
    outer_domain: str,
    target_specimen_count: int,
    checkpoint_count: int,
) -> dict[str, object]:
    try:
        payload = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISClosedLoopPackageError("P4 worker summary is invalid") from error
    expected_predictions = (
        target_specimen_count * len(PRIMARY_CLOSED_LOOP_METHODS) * checkpoint_count
    )
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("outer_domain") != outer_domain
        or tuple(payload.get("methods", ())) != PRIMARY_CLOSED_LOOP_METHODS
        or payload.get("target_specimen_count") != target_specimen_count
        or payload.get("prediction_count") != expected_predictions
        or payload.get("full_scan_anchor_count") != target_specimen_count
        or payload.get("target_true_cai_used_by_policy") is not False
        or payload.get("future_target_content_used_by_policy") is not False
        or set(payload.get("p2_model_state_sha256", {}))
        != {"positions_only", "real", "shuffled"}
        or set(payload.get("p3_model_state_sha256", {}))
        != {"positions_only", "real", "shuffled"}
        or type(payload.get("historical_source_state_sha256")) is not str
        or len(payload["historical_source_state_sha256"]) != 64
        or type(payload.get("files")) is not dict
        or set(payload["files"]) != _WORKER_FILES
    ):
        raise MAVISClosedLoopPackageError("P4 worker contract changed")
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "complete.json"
    }
    if actual != _WORKER_FILES:
        raise MAVISClosedLoopPackageError("P4 worker file roster changed")
    for relative, expected in payload["files"].items():
        if type(expected) is not str or _sha256(path / relative) != expected:
            raise MAVISClosedLoopPackageError(
                f"P4 worker checksum mismatch: {relative}"
            )
    return payload


def _validate_tables(
    predictions: pl.DataFrame,
    trajectories: pl.DataFrame,
    full_scan: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    expected_specimens: pl.DataFrame,
) -> None:
    expected_roster = expected_specimens.select(
        pl.col("domain_id").alias("outer_domain"),
        "specimen_id",
    ).unique().sort(["outer_domain", "specimen_id"])
    specimen_count = expected_roster.height
    rosters_match = all(
        table.select("outer_domain", "specimen_id")
        .unique()
        .sort(["outer_domain", "specimen_id"])
        .equals(expected_roster)
        for table in (predictions, trajectories, full_scan)
    )
    prediction_key = [
        "outer_domain",
        "specimen_id",
        "method",
        "nominal_checkpoint",
    ]
    expected_predictions = (
        specimen_count * len(PRIMARY_CLOSED_LOOP_METHODS) * len(checkpoints)
    )
    if (
        predictions.height != expected_predictions
        or not rosters_match
        or predictions.unique(subset=prediction_key).height != predictions.height
        or set(predictions.get_column("outer_domain").unique()) != set(domain_order)
        or set(predictions.get_column("method").unique())
        != set(PRIMARY_CLOSED_LOOP_METHODS)
        or set(predictions.get_column("nominal_checkpoint").unique())
        != set(checkpoints)
        or trajectories.height == 0
        or set(trajectories.get_column("outer_domain").unique()) != set(domain_order)
        or not set(trajectories.get_column("method").unique())
        <= set(PRIMARY_CLOSED_LOOP_METHODS)
        or full_scan.height != specimen_count
        or full_scan.unique(subset=["outer_domain", "specimen_id"]).height
        != specimen_count
        or set(full_scan.get_column("outer_domain").unique()) != set(domain_order)
        or full_scan.filter(
            (pl.col("method") != "full_scan")
            | (pl.col("nominal_checkpoint") != 1.0)
            | (pl.col("exact_acquired_cost") != pl.col("native_count"))
        ).height
    ):
        raise MAVISClosedLoopPackageError("P4 scientific table roster is incomplete")
    for table in (predictions, full_scan):
        if table.select(pl.any_horizontal(pl.selectors.numeric().is_nan())).to_series().any():
            raise MAVISClosedLoopPackageError("P4 scientific table contains NaN")


def _bootstrap_summary(bootstrap: pl.DataFrame) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for control in sorted(bootstrap.get_column("control_method").unique()):
        table = bootstrap.filter(pl.col("control_method") == control)
        values = table.get_column("control_minus_reference_cai_auebc").to_numpy()
        improved = table.get_column("improved_domain_count").to_numpy()
        result[str(control)] = {
            "mean_control_minus_mavis_full_cai_auebc": float(np.mean(values)),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
            "median_improved_domain_count": float(np.median(improved)),
        }
    return result


def finalize_closed_loop_package(
    config_path: str | Path,
    *,
    project_root: str | Path,
    worker_root: str | Path,
    p1_package: str | Path,
    p2_package: str | Path,
    p3_package: str | Path,
    bootstrap_replicates: int,
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
    prediction_parts: list[pl.DataFrame] = []
    trajectory_parts: list[pl.DataFrame] = []
    full_scan_parts: list[pl.DataFrame] = []
    workers: list[dict[str, object]] = []
    for outer_domain in config.domain_order:
        target_specimens = state_manifest.filter(
            pl.col("domain_id") == outer_domain
        ).get_column("specimen_id").n_unique()
        worker_path = workers_root / outer_domain
        workers.append(
            _worker(
                worker_path,
                outer_domain=outer_domain,
                target_specimen_count=target_specimens,
                checkpoint_count=len(config.checkpoints),
            )
        )
        try:
            prediction_parts.append(pl.read_parquet(worker_path / "predictions.parquet"))
            trajectory_parts.append(pl.read_parquet(worker_path / "trajectories.parquet"))
            full_scan_parts.append(
                pl.read_parquet(worker_path / "full_scan_anchors.parquet")
            )
        except (OSError, pl.exceptions.PolarsError) as error:
            raise MAVISClosedLoopPackageError("P4 worker table is invalid") from error
    predictions = pl.concat(prediction_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "method", "nominal_checkpoint"]
    )
    trajectories = pl.concat(trajectory_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "method", "step"]
    )
    full_scan = pl.concat(full_scan_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id"]
    )
    _validate_tables(
        predictions,
        trajectories,
        full_scan,
        domain_order=config.domain_order,
        checkpoints=config.checkpoints,
        expected_specimens=state_manifest,
    )
    metrics = evaluate_closed_loop_predictions(
        predictions,
        domain_order=config.domain_order,
        method_order=PRIMARY_CLOSED_LOOP_METHODS,
        checkpoints=config.checkpoints,
    )
    controls = tuple(
        method for method in PRIMARY_CLOSED_LOOP_METHODS if method != "mavis_full"
    )
    bootstrap = bootstrap_closed_loop_contrasts(
        metrics.specimen_auebc,
        reference_method="mavis_full",
        control_methods=controls,
        domain_order=config.domain_order,
        replicates=bootstrap_replicates,
        seed=config.seed + 400,
    )
    git_sha, code_state_sha256, dirty = _repository_state(root)
    p4_state = hashlib.sha256(
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
    output = root / config.output_root / "p4_closed_loop"
    if output.exists():
        raise MAVISClosedLoopPackageError("P4 formal package already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p4_closed_loop.", dir=output.parent))
    try:
        predictions.write_parquet(
            temporary / "closed_loop_predictions.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        trajectories.write_parquet(
            temporary / "action_trajectories.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        full_scan.write_csv(temporary / "full_scan_anchors.csv")
        metrics.per_specimen_curve.write_csv(temporary / "per_specimen_curves.csv")
        metrics.domain_curve.write_csv(temporary / "domain_curves.csv")
        metrics.aggregate_curve.write_csv(temporary / "aggregate_curves.csv")
        metrics.specimen_auebc.write_csv(temporary / "per_specimen_auebc.csv")
        metrics.domain_auebc.write_csv(temporary / "domain_auebc.csv")
        metrics.aggregate_auebc.write_csv(temporary / "aggregate_auebc.csv")
        bootstrap.write_parquet(
            temporary / "paired_bootstrap.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        task_specificity = metrics.aggregate_curve.filter(
            pl.col("method").is_in(
                [
                    "uniform",
                    "reconstruction_driven",
                    "mavis_no_feedback",
                    "mavis_positions_only",
                    "mavis_shuffled_content",
                    "mavis_full",
                ]
            )
        )
        task_specificity.write_csv(temporary / "task_specificity.csv")
        figure_root = temporary / "figures"
        figure_root.mkdir()
        task_specificity.write_csv(figure_root / "source_data.csv")
        render_task_specificity_curve(task_specificity, output_root=figure_root)
        (temporary / "FIGURE_CAPTION.md").write_text(
            "Domain-balanced CAI mean absolute error and normalized C-scan "
            "reconstruction mean squared error versus mean exact acquired native-"
            "raster fraction. The comparison separates mechanics-driven sensing "
            "from reconstruction-driven sensing and includes feedback and "
            "measurement-content controls. Source: figures/source_data.csv; "
            "retrospective normalized-raster evaluation.\n",
            encoding="utf-8",
        )
        aggregate = metrics.aggregate_auebc
        mavis = aggregate.filter(pl.col("method") == "mavis_full").row(0, named=True)
        deployable = (
            "uniform",
            "random",
            "reconstruction_driven",
            "global_mechanical",
            "mva_a5",
            "mvd_m1_o2",
        )
        baseline = (
            aggregate.filter(pl.col("method").is_in(deployable))
            .sort("domain_balanced_cai_auebc")
            .row(0, named=True)
        )
        summary = {
            "schema_version": 1,
            "stage": "P4_CLOSED_LOOP_SCOUT_AND_FOCUS",
            "status": "COMPLETE",
            "git_sha": git_sha,
            "git_worktree_dirty_at_run": dirty,
            "runtime_code_state_sha256": code_state_sha256,
            "config_sha256": config.config_sha256,
            "p1_state_sha256": p1_manifest.get("state_bank_state_sha256"),
            "p2_state_sha256": p2_manifest.get("p2_state_sha256"),
            "p3_state_sha256": p3_manifest.get("p3_state_sha256"),
            "p4_state_sha256": p4_state,
            "specimen_count": config.specimen_count,
            "method_count": len(PRIMARY_CLOSED_LOOP_METHODS),
            "prediction_count": predictions.height,
            "trajectory_row_count": trajectories.height,
            "mavis_full_domain_balanced_cai_auebc": float(
                mavis["domain_balanced_cai_auebc"]
            ),
            "strongest_observed_deployable_baseline": str(baseline["method"]),
            "strongest_observed_deployable_baseline_cai_auebc": float(
                baseline["domain_balanced_cai_auebc"]
            ),
            "bootstrap": _bootstrap_summary(bootstrap),
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "target_data_used_for_model_selection": False,
            "claim_tier_assigned": False,
        }
        _write_json(temporary / "summary.json", summary)
        report = (
            "# MAVIS P4 Closed-loop Scout-and-Focus\n\n"
            "Status: `COMPLETE`. This development stage does not assign the final "
            "MAVIS claim tier.\n\n"
            f"The equal-domain MAVIS full CAI AUEBC is "
            f"`{float(mavis['domain_balanced_cai_auebc']):.10f}`. The strongest "
            f"observed deployable baseline is `{baseline['method']}` at "
            f"`{float(baseline['domain_balanced_cai_auebc']):.10f}`.\n\n"
            "Every method uses the same frozen cohort, exact native-raster cost, "
            "checkpoints, and real-state MRIS CAI endpoint. Full scan is reported "
            "separately as a 100% anchor. Target CAI and future target measurements "
            "are unavailable to all deployed scorers. Statistics pair methods "
            "within physical specimens, resample within held-out domains, and weight "
            "the six domains equally. This is retrospective normalized-raster "
            "closed-loop feasibility evidence.\n"
        )
        (temporary / "REPORT.md").write_text(report, encoding="utf-8")
        scientific_files = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        )
        manifest = {
            "schema_version": 1,
            "artifact": "mavis_p4_closed_loop",
            "p4_state_sha256": p4_state,
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
    verify_closed_loop_package(output)
    return output


def verify_closed_loop_package(path: str | Path) -> dict[str, object]:
    root = Path(path)
    try:
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        lines = (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISClosedLoopPackageError("P4 package metadata is invalid") from error
    if type(manifest) is not dict or type(manifest.get("files")) is not dict:
        raise MAVISClosedLoopPackageError("P4 package manifest is invalid")
    expected = set(manifest["files"]) | {"artifact_manifest.json"}
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "CHECKSUMS.sha256"
    }
    if actual != expected or len(lines) != len(expected):
        raise MAVISClosedLoopPackageError("P4 package file roster changed")
    ledger: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise MAVISClosedLoopPackageError("P4 checksum ledger is invalid") from error
        if name in ledger or len(digest) != 64:
            raise MAVISClosedLoopPackageError("P4 checksum ledger is invalid")
        ledger[name] = digest
    if set(ledger) != expected:
        raise MAVISClosedLoopPackageError("P4 checksum roster changed")
    for name, expected_digest in ledger.items():
        if _sha256(root / name) != expected_digest:
            raise MAVISClosedLoopPackageError(f"P4 checksum mismatch: {name}")
    for name, metadata in manifest["files"].items():
        file_path = root / name
        if (
            type(metadata) is not dict
            or metadata.get("bytes") != file_path.stat().st_size
            or metadata.get("sha256") != _sha256(file_path)
        ):
            raise MAVISClosedLoopPackageError(f"P4 manifest mismatch: {name}")
    if any((root / name).stat().st_size >= _GIT_BLOB_LIMIT for name in expected):
        raise MAVISClosedLoopPackageError("P4 package contains an oversized Git blob")
    return manifest


__all__ = [
    "MAVISClosedLoopPackageError",
    "finalize_closed_loop_package",
    "verify_closed_loop_package",
]
