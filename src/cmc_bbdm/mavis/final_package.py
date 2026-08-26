"""Finalize and verify the frozen MAVIS outer-evaluation package."""

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

from .aggregation_package import verify_aggregation_package
from .closed_loop_execution import PRIMARY_CLOSED_LOOP_METHODS
from .closed_loop_metrics import (
    bootstrap_closed_loop_contrasts,
    evaluate_closed_loop_predictions,
)
from .closed_loop_package import verify_closed_loop_package
from .config import load_mavis_config
from .final_execution import (
    assign_claim_tier,
    build_risk_coverage,
    compose_final_predictions,
)
from .final_figure import render_final_claim_figure
from .safety_package import verify_safety_package
from .state_bank_package import (
    load_state_manifest_package,
    verify_state_bank_package,
)


class MAVISFinalPackageError(RuntimeError):
    """Raised when frozen final evidence is incomplete or inconsistent."""


_GIT_BLOB_LIMIT = 100 * 1024 * 1024
_WORKER_FILES = {
    "aggregated_predictions.parquet",
    "aggregated_trajectories.parquet",
    "routing.csv",
}
_CODE_PATHS = (
    "src/cmc_bbdm/mavis/closed_loop_execution.py",
    "src/cmc_bbdm/mavis/closed_loop_metrics.py",
    "src/cmc_bbdm/mavis/dynamic_training.py",
    "src/cmc_bbdm/mavis/final_execution.py",
    "src/cmc_bbdm/mavis/final_figure.py",
    "src/cmc_bbdm/mavis/final_package.py",
    "src/cmc_bbdm/mavis/mris_training.py",
    "src/cmc_bbdm/mavis/policy.py",
    "src/cmc_bbdm/mavis/rollout.py",
)
_DEPLOYABLE_BASELINES = (
    "uniform",
    "random",
    "reconstruction_driven",
    "global_mechanical",
    "mva_a5",
    "mvd_m1_o2",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
            raise MAVISFinalPackageError(
                f"final runtime source is unavailable: {relative}"
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
        raise MAVISFinalPackageError("final Git provenance is unavailable") from error
    if len(git_sha) != 40:
        raise MAVISFinalPackageError("final Git SHA is invalid")
    return git_sha, digest.hexdigest(), dirty


def development_package_sha256(
    p4_manifest: dict[str, object],
    p5_manifest: dict[str, object],
    p6_manifest: dict[str, object],
) -> str:
    manifests = {
        "p4": (p4_manifest, "p4_state_sha256"),
        "p5": (p5_manifest, "p5_state_sha256"),
        "p6": (p6_manifest, "p6_state_sha256"),
    }
    if any(type(manifest) is not dict for manifest, _ in manifests.values()):
        raise MAVISFinalPackageError("development package manifest is invalid")
    configs = {manifest.get("config_sha256") for manifest, _ in manifests.values()}
    if len(configs) != 1 or not _is_sha(next(iter(configs), None)):
        raise MAVISFinalPackageError("development package config changed")
    states = {
        name: manifest.get(state_key)
        for name, (manifest, state_key) in manifests.items()
    }
    if any(not _is_sha(value) for value in states.values()):
        raise MAVISFinalPackageError("development package state is invalid")
    payload = {
        "schema": 1,
        "development_config_sha256": next(iter(configs)),
        "stage_states": states,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _worker(
    path: Path,
    *,
    outer_domain: str,
    target_specimen_count: int,
    checkpoint_count: int,
    config_sha256: str,
    development_sha256: str,
) -> dict[str, object]:
    try:
        payload = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISFinalPackageError("final worker summary is invalid") from error
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("outer_domain") != outer_domain
        or payload.get("config_sha256") != config_sha256
        or payload.get("development_package_sha256") != development_sha256
        or payload.get("target_specimen_count") != target_specimen_count
        or payload.get("prediction_count") != target_specimen_count * checkpoint_count
        or payload.get("routing_count") != target_specimen_count
        or payload.get("baseline") not in {"uniform", "reconstruction"}
        or payload.get("target_true_cai_used_by_policy") is not False
        or payload.get("future_target_content_used_by_policy") is not False
        or not _is_sha(payload.get("p2_model_state_sha256"))
        or not _is_sha(payload.get("p5_model_state_sha256"))
        or isinstance(payload.get("runtime_seconds"), bool)
        or not isinstance(payload.get("runtime_seconds"), (int, float))
        or not np.isfinite(float(payload["runtime_seconds"]))
        or float(payload["runtime_seconds"]) <= 0.0
        or type(payload.get("files")) is not dict
        or set(payload["files"]) != _WORKER_FILES
    ):
        raise MAVISFinalPackageError("final worker contract changed")
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "complete.json"
    }
    if actual != _WORKER_FILES:
        raise MAVISFinalPackageError("final worker file roster changed")
    for relative, expected in payload["files"].items():
        if type(expected) is not str or _sha256(path / relative) != expected:
            raise MAVISFinalPackageError(f"final worker checksum mismatch: {relative}")
    return payload


def _bootstrap_summary(
    bootstrap: pl.DataFrame,
    *,
    control_method: str,
    reference_method: str,
) -> dict[str, object]:
    table = bootstrap.filter(
        (pl.col("control_method") == control_method)
        & (pl.col("reference_method") == reference_method)
    )
    if table.height == 0:
        raise MAVISFinalPackageError("final bootstrap contrast is unavailable")
    values = table.get_column("control_minus_reference_cai_auebc").to_numpy()
    return {
        "control_method": control_method,
        "reference_method": reference_method,
        "point_estimate": float(np.mean(values, dtype=np.float64)),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "median_improved_domain_count": float(
            np.median(table.get_column("improved_domain_count").to_numpy())
        ),
    }


def _high_confidence_bootstrap(
    specimen_auebc: pl.DataFrame,
    routing: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    replicates: int,
    seed: int,
) -> tuple[pl.DataFrame, dict[str, object]]:
    mavis = specimen_auebc.filter(pl.col("method") == "mavis_full").select(
        "outer_domain",
        "specimen_id",
        pl.col("cai_auebc").alias("mavis_cai_auebc"),
    )
    control = specimen_auebc.filter(
        pl.col("method") == "source_selected_fallback"
    ).select(
        "outer_domain",
        "specimen_id",
        pl.col("cai_auebc").alias("control_cai_auebc"),
    )
    paired = (
        routing.select(
            "outer_domain", "specimen_id", "confidence", "threshold"
        )
        .join(mavis, on=["outer_domain", "specimen_id"], how="inner", validate="1:1")
        .join(
            control,
            on=["outer_domain", "specimen_id"],
            how="inner",
            validate="1:1",
        )
        .filter(pl.col("confidence") >= pl.col("threshold"))
        .with_columns(
            (pl.col("control_cai_auebc") - pl.col("mavis_cai_auebc")).alias(
                "control_minus_mavis_cai_auebc"
            )
        )
    )
    empty_schema = {
        "replicate": pl.Int64,
        "control_minus_mavis_cai_auebc": pl.Float64,
        "covered_domain_count": pl.Int64,
        "high_confidence_specimen_count": pl.Int64,
        "statistical_unit": pl.String,
    }
    if paired.height == 0:
        return pl.DataFrame(schema=empty_schema), {
            "point_estimate": 0.0,
            "ci95_lower": 0.0,
            "ci95_upper": 0.0,
            "covered_domain_count": 0,
            "high_confidence_specimen_count": 0,
        }
    arrays = {
        domain: paired.filter(pl.col("outer_domain") == domain)
        .get_column("control_minus_mavis_cai_auebc")
        .to_numpy()
        for domain in domain_order
        if paired.filter(pl.col("outer_domain") == domain).height
    }
    point = float(
        np.mean([np.mean(values, dtype=np.float64) for values in arrays.values()])
    )
    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        domain_values = []
        for values in arrays.values():
            indices = generator.integers(0, values.size, values.size)
            domain_values.append(np.mean(values[indices], dtype=np.float64))
        rows.append(
            {
                "replicate": replicate,
                "control_minus_mavis_cai_auebc": float(
                    np.mean(domain_values, dtype=np.float64)
                ),
                "covered_domain_count": len(arrays),
                "high_confidence_specimen_count": paired.height,
                "statistical_unit": "paired_high_confidence_specimen_within_domain",
            }
        )
    table = pl.DataFrame(rows, schema=empty_schema).sort("replicate")
    values = table.get_column("control_minus_mavis_cai_auebc").to_numpy()
    return table, {
        "point_estimate": point,
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "covered_domain_count": len(arrays),
        "high_confidence_specimen_count": paired.height,
    }


def _domain_effects(
    domain_auebc: pl.DataFrame,
    *,
    strongest_baseline: str,
    domain_order: tuple[str, ...],
) -> pl.DataFrame:
    contrasts = (
        ("baseline_minus_mavis", strongest_baseline, "mavis_full"),
        ("fallback_minus_safe", "source_selected_fallback", "mavis_safe"),
    )
    rows: list[dict[str, object]] = []
    for contrast, control, reference in contrasts:
        control_table = domain_auebc.filter(pl.col("method") == control).select(
            "outer_domain", pl.col("cai_auebc").alias("control")
        )
        reference_table = domain_auebc.filter(pl.col("method") == reference).select(
            "outer_domain", pl.col("cai_auebc").alias("reference")
        )
        paired = control_table.join(
            reference_table, on="outer_domain", how="inner", validate="1:1"
        )
        if paired.height != len(domain_order):
            raise MAVISFinalPackageError("final domain contrast roster is incomplete")
        lookup = {
            str(row["outer_domain"]): float(row["control"] - row["reference"])
            for row in paired.iter_rows(named=True)
        }
        rows.extend(
            {
                "outer_domain": domain,
                "contrast": contrast,
                "control_method": control,
                "reference_method": reference,
                "control_minus_reference_cai_auebc": lookup[domain],
                "improved": lookup[domain] > 0.0,
            }
            for domain in domain_order
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["contrast", "outer_domain"]
    )


def _external_audit(root: Path) -> dict[str, object]:
    source = root / "artifacts/external_data/EXTERNAL_DATA_MANIFEST.json"
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISFinalPackageError("external data audit is unavailable") from error
    if payload.get("method_performance_present") is not False:
        raise MAVISFinalPackageError("external data role changed")
    return {
        "schema_version": 1,
        "source_manifest_sha256": _sha256(source),
        "datasets": payload.get("datasets"),
        "external_validation_performed": False,
        "method_performance_present": False,
        "physical_coordinate_status": {
            "cranfield_wp2": "indexed_grid_recoverable_physical_spacing_unavailable",
            "imperial_interlock": "paired_images_no_registered_scanner_path_or_time",
            "imperial_rss": "exact_pairing_unresolved",
            "tudelft": "three_microcases_only",
        },
        "scanner_path_or_time_available": False,
        "interpretation": "retrospective_normalized_raster_closed_loop_feasibility",
        "prospective_validation_required": True,
    }


def finalize_final_package(
    config_path: str | Path,
    *,
    project_root: str | Path,
    worker_root: str | Path,
    p1_package: str | Path,
    p4_package: str | Path,
    p5_package: str | Path,
    p6_package: str | Path,
    bootstrap_replicates: int,
    output_path: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve(strict=True)
    config_path_value = Path(config_path).resolve(strict=True)
    config = load_mavis_config(config_path_value, project_root=root)
    config.require_finalized()
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise MAVISFinalPackageError("final bootstrap count is invalid")
    p1_root = Path(p1_package).resolve(strict=True)
    p4_root = Path(p4_package).resolve(strict=True)
    p5_root = Path(p5_package).resolve(strict=True)
    p6_root = Path(p6_package).resolve(strict=True)
    p1_manifest = verify_state_bank_package(p1_root)
    p4_manifest = verify_closed_loop_package(p4_root)
    p5_manifest = verify_aggregation_package(p5_root)
    p6_manifest = verify_safety_package(p6_root)
    development_sha = development_package_sha256(
        p4_manifest, p5_manifest, p6_manifest
    )
    if config.development_package_sha256 != development_sha:
        raise MAVISFinalPackageError("frozen development package hash changed")
    state_manifest = load_state_manifest_package(p1_root)
    try:
        p4_predictions = pl.read_parquet(
            p4_root / "closed_loop_predictions.parquet"
        )
        full_scan = pl.read_csv(p4_root / "full_scan_anchors.csv")
        selections = pl.read_csv(p6_root / "selections.csv")
    except (OSError, pl.exceptions.PolarsError) as error:
        raise MAVISFinalPackageError("final dependency table is invalid") from error
    workers_root = Path(worker_root).resolve(strict=True)
    prediction_parts: list[pl.DataFrame] = []
    trajectory_parts: list[pl.DataFrame] = []
    routing_parts: list[pl.DataFrame] = []
    workers: list[dict[str, object]] = []
    for outer_domain in config.domain_order:
        target_count = state_manifest.filter(
            pl.col("domain_id") == outer_domain
        ).get_column("specimen_id").n_unique()
        worker_path = workers_root / outer_domain
        worker = _worker(
            worker_path,
            outer_domain=outer_domain,
            target_specimen_count=target_count,
            checkpoint_count=len(config.checkpoints),
            config_sha256=config.config_sha256,
            development_sha256=development_sha,
        )
        workers.append(worker)
        try:
            prediction_parts.append(
                pl.read_parquet(worker_path / "aggregated_predictions.parquet")
            )
            trajectory_parts.append(
                pl.read_parquet(worker_path / "aggregated_trajectories.parquet")
            )
            routing_parts.append(pl.read_csv(worker_path / "routing.csv"))
        except (OSError, pl.exceptions.PolarsError) as error:
            raise MAVISFinalPackageError("final worker table is invalid") from error
    aggregated = pl.concat(prediction_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "nominal_checkpoint"]
    )
    trajectories = pl.concat(trajectory_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "step"]
    )
    routing = pl.concat(routing_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id"]
    )
    selection_lookup = {
        str(row["outer_domain"]): (
            str(row["baseline"]),
            float(row["threshold"]),
            str(row["selection_state_sha256"]),
        )
        for row in selections.iter_rows(named=True)
    }
    if set(selection_lookup) != set(config.domain_order):
        raise MAVISFinalPackageError("final safety selection roster changed")
    for outer_domain in config.domain_order:
        expected_baseline, expected_threshold, expected_state = selection_lookup[
            outer_domain
        ]
        table = routing.filter(pl.col("outer_domain") == outer_domain)
        if (
            set(table.get_column("baseline")) != {expected_baseline}
            or set(table.get_column("selection_state_sha256")) != {expected_state}
            or any(
                abs(float(value) - expected_threshold) > 1.0e-15
                for value in table.get_column("threshold")
            )
        ):
            raise MAVISFinalPackageError("final worker routing changed")
    predictions, routing_audit = compose_final_predictions(
        p4_predictions,
        aggregated,
        routing,
    )
    method_order = tuple(
        "mavis_no_aggregation" if method == "mavis_full" else method
        for method in PRIMARY_CLOSED_LOOP_METHODS
    ) + ("mavis_full", "mavis_safe", "source_selected_fallback")
    if (
        predictions.height
        != config.specimen_count * len(method_order) * len(config.checkpoints)
        or set(predictions.get_column("method").unique()) != set(method_order)
        or routing_audit.height != config.specimen_count
        or routing_audit.unique(subset=["outer_domain", "specimen_id"]).height
        != routing_audit.height
        or set(full_scan.get_column("outer_domain").unique()) != set(config.domain_order)
    ):
        raise MAVISFinalPackageError("final scientific roster is incomplete")
    metrics = evaluate_closed_loop_predictions(
        predictions,
        domain_order=config.domain_order,
        method_order=method_order,
        checkpoints=config.checkpoints,
    )
    aggregate = metrics.aggregate_auebc
    baseline = (
        aggregate.filter(pl.col("method").is_in(_DEPLOYABLE_BASELINES))
        .sort(["domain_balanced_cai_auebc", "method"])
        .row(0, named=True)
    )
    strongest_baseline = str(baseline["method"])
    mavis_controls = tuple(
        dict.fromkeys(
            (
                strongest_baseline,
                "source_selected_fallback",
                "mavis_no_aggregation",
                "mavis_no_feedback",
                "mavis_positions_only",
                "mavis_shuffled_content",
                "mavis_raw_value",
                "mavis_value_per_cost",
            )
        )
    )
    mavis_bootstrap = bootstrap_closed_loop_contrasts(
        metrics.specimen_auebc,
        reference_method="mavis_full",
        control_methods=mavis_controls,
        domain_order=config.domain_order,
        replicates=bootstrap_replicates,
        seed=config.seed + 700,
    )
    safe_bootstrap = bootstrap_closed_loop_contrasts(
        metrics.specimen_auebc,
        reference_method="mavis_safe",
        control_methods=("source_selected_fallback",),
        domain_order=config.domain_order,
        replicates=bootstrap_replicates,
        seed=config.seed + 701,
    )
    bootstrap = pl.concat([mavis_bootstrap, safe_bootstrap], how="vertical_relaxed").sort(
        ["reference_method", "control_method", "replicate"]
    )
    high_bootstrap, high_summary = _high_confidence_bootstrap(
        metrics.specimen_auebc,
        routing,
        domain_order=config.domain_order,
        replicates=bootstrap_replicates,
        seed=config.seed + 702,
    )
    risk_coverage = build_risk_coverage(
        metrics.specimen_auebc,
        routing,
        thresholds=config.confidence_thresholds,
        domain_order=config.domain_order,
    )
    effects = _domain_effects(
        metrics.domain_auebc,
        strongest_baseline=strongest_baseline,
        domain_order=config.domain_order,
    )
    mavis_effect = _bootstrap_summary(
        bootstrap,
        control_method=strongest_baseline,
        reference_method="mavis_full",
    )
    safe_effect = _bootstrap_summary(
        bootstrap,
        control_method="source_selected_fallback",
        reference_method="mavis_safe",
    )
    value_by_method = {
        str(row["method"]): float(row["domain_balanced_cai_auebc"])
        for row in aggregate.iter_rows(named=True)
    }
    improved_domains = effects.filter(
        (pl.col("contrast") == "baseline_minus_mavis") & pl.col("improved")
    ).height
    claim_tier = assign_claim_tier(
        baseline_cai_auebc=value_by_method[strongest_baseline],
        safe_control_cai_auebc=value_by_method["source_selected_fallback"],
        mavis_cai_auebc=value_by_method["mavis_full"],
        safe_cai_auebc=value_by_method["mavis_safe"],
        sequential_oracle_cai_auebc=value_by_method[
            "sequential_mechanical_oracle"
        ],
        mavis_improved_domain_count=improved_domains,
        domain_count=len(config.domain_order),
        mavis_bootstrap_ci_lower=float(mavis_effect["ci95_lower"]),
        safe_bootstrap_ci_lower=float(safe_effect["ci95_lower"]),
        high_confidence_control_minus_mavis_auebc=float(
            high_summary["point_estimate"]
        ),
        high_confidence_bootstrap_ci_lower=float(high_summary["ci95_lower"]),
        high_confidence_specimen_count=int(
            high_summary["high_confidence_specimen_count"]
        ),
    )
    git_sha, code_state_sha256, dirty = _repository_state(root)
    dependency_states = {
        "p1_state_sha256": p1_manifest.get("state_bank_state_sha256"),
        "p4_state_sha256": p4_manifest.get("p4_state_sha256"),
        "p5_state_sha256": p5_manifest.get("p5_state_sha256"),
        "p6_state_sha256": p6_manifest.get("p6_state_sha256"),
    }
    p7_state = hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "config_sha256": config.config_sha256,
                "development_package_sha256": development_sha,
                "dependencies": dependency_states,
                "workers": workers,
                "runtime_code_state_sha256": code_state_sha256,
                "claim_tier": claim_tier,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if output_path is None:
        output = root / config.output_root / "p7_final_frozen_eval"
    else:
        output = Path(output_path)
        if not output.is_absolute():
            output = root / output
        output = output.resolve()
        if root not in output.parents:
            raise MAVISFinalPackageError("final output path leaves the repository")
    if output.exists():
        raise MAVISFinalPackageError("final formal package already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p7_final.", dir=output.parent))
    try:
        predictions.write_parquet(
            temporary / "closed_loop_predictions.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        trajectories.write_parquet(
            temporary / "aggregated_action_trajectories.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        routing_audit.write_csv(temporary / "routing_audit.csv")
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
        high_bootstrap.write_parquet(
            temporary / "high_confidence_bootstrap.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        risk_coverage.write_csv(temporary / "risk_coverage.csv")
        effects.write_csv(temporary / "domain_effects.csv")
        selections.write_csv(temporary / "source_selection.csv")
        pl.DataFrame(
            [
                {
                    "outer_domain": worker["outer_domain"],
                    "runtime_seconds": worker["runtime_seconds"],
                    "target_specimen_count": worker["target_specimen_count"],
                    "trajectory_row_count": worker["trajectory_row_count"],
                }
                for worker in workers
            ],
            infer_schema_length=None,
        ).sort("outer_domain").write_csv(temporary / "runtime.csv")
        claim_evidence = {
            "claim_tier": claim_tier,
            "strongest_deployable_baseline": strongest_baseline,
            "baseline_cai_auebc": value_by_method[strongest_baseline],
            "mavis_cai_auebc": value_by_method["mavis_full"],
            "mavis_no_aggregation_cai_auebc": value_by_method[
                "mavis_no_aggregation"
            ],
            "mavis_safe_cai_auebc": value_by_method["mavis_safe"],
            "source_selected_fallback_cai_auebc": value_by_method[
                "source_selected_fallback"
            ],
            "sequential_oracle_cai_auebc": value_by_method[
                "sequential_mechanical_oracle"
            ],
            "mavis_improved_domain_count": improved_domains,
            "domain_count": len(config.domain_order),
            "mavis_control_minus_reference_ci95_lower": mavis_effect["ci95_lower"],
            "mavis_control_minus_reference_ci95_upper": mavis_effect["ci95_upper"],
            "safe_control_minus_reference_ci95_lower": safe_effect["ci95_lower"],
            "safe_control_minus_reference_ci95_upper": safe_effect["ci95_upper"],
            "high_confidence_control_minus_mavis_auebc": high_summary[
                "point_estimate"
            ],
            "high_confidence_ci95_lower": high_summary["ci95_lower"],
            "high_confidence_ci95_upper": high_summary["ci95_upper"],
            "high_confidence_specimen_count": high_summary[
                "high_confidence_specimen_count"
            ],
            "high_confidence_domain_count": high_summary["covered_domain_count"],
        }
        pl.DataFrame([claim_evidence], infer_schema_length=None).write_csv(
            temporary / "claim_evidence.csv"
        )
        dependencies = {
            "schema_version": 1,
            "development_package_sha256": development_sha,
            "stage_states": dependency_states,
            "development_config_sha256": p4_manifest.get("config_sha256"),
            "final_config_sha256": config.config_sha256,
        }
        _write_json(temporary / "development_dependencies.json", dependencies)
        _write_json(temporary / "external_data_audit.json", _external_audit(root))
        evidence = pl.DataFrame(
            {
                "experiment": [f"E{index}" for index in range(1, 11)],
                "status": ["FROZEN"] + ["COMPLETE"] * 8 + ["AUDIT_ONLY"],
                "artifact": [
                    "artifacts/mavis/P0_FROZEN_EVIDENCE_LEDGER.md",
                    "results/mavis/p2_mris",
                    "results/mavis/p3_dynamic_voi",
                    "results/mavis/p7_final_frozen_eval",
                    "results/mavis/p4_closed_loop/figures",
                    "results/mavis/p7_final_frozen_eval",
                    "results/mavis/p7_final_frozen_eval",
                    "results/mavis/p7_final_frozen_eval",
                    "results/mavis/p7_final_frozen_eval",
                    "results/mavis/p7_final_frozen_eval/external_data_audit.json",
                ],
            }
        )
        evidence.write_csv(temporary / "experiment_evidence_map.csv")
        (temporary / "config.yaml").write_bytes(config_path_value.read_bytes())
        figure_root = temporary / "figures"
        figure_root.mkdir()
        figure_methods = {
            strongest_baseline,
            "mavis_no_aggregation",
            "mavis_full",
            "mavis_safe",
            "sequential_mechanical_oracle",
        }
        figure_curves = metrics.aggregate_curve.filter(
            pl.col("method").is_in(figure_methods)
        )
        figure_curves.write_csv(figure_root / "aggregate_curves.csv")
        effects.write_csv(figure_root / "domain_effects.csv")
        render_final_claim_figure(
            figure_curves,
            effects,
            strongest_baseline=strongest_baseline,
            domain_order=config.domain_order,
            output_root=figure_root,
        )
        (temporary / "FIGURE_CAPTION.md").write_text(
            "Frozen equal-domain CAI error curves at exact native-raster cost and "
            "paired held-out-domain CAI AUEBC effects. Positive bars favor aggregated "
            "MAVIS or the source-selected safe system. D1-D6 follow the domain order "
            "in config.yaml. Source: figures/aggregate_curves.csv and "
            "figures/domain_effects.csv; retrospective normalized-raster evaluation.\n",
            encoding="utf-8",
        )
        summary = {
            "schema_version": 1,
            "stage": "P7_FROZEN_OUTER_EVALUATION",
            "status": "COMPLETE",
            "claim_tier": claim_tier,
            "configuration_frozen": True,
            "git_sha": git_sha,
            "git_worktree_dirty_at_run": dirty,
            "runtime_code_state_sha256": code_state_sha256,
            "config_sha256": config.config_sha256,
            "development_package_sha256": development_sha,
            "p7_state_sha256": p7_state,
            **dependency_states,
            "specimen_count": config.specimen_count,
            "domain_count": len(config.domain_order),
            "method_count": len(method_order),
            "prediction_count": predictions.height,
            "strongest_deployable_baseline": strongest_baseline,
            "claim_evidence": claim_evidence,
            "target_data_used_for_training_or_selection": False,
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "external_validation_performed": False,
        }
        _write_json(temporary / "summary.json", summary)
        report = (
            "# MAVIS P7 Frozen Outer Evaluation\n\n"
            f"Status: `COMPLETE`. Frozen claim tier: `Tier {claim_tier}`.\n\n"
            f"Aggregated MAVIS CAI AUEBC is `{value_by_method['mavis_full']:.10f}`; "
            f"the strongest observed deployable baseline is `{strongest_baseline}` at "
            f"`{value_by_method[strongest_baseline]:.10f}`. The source-selected "
            f"fallback control is `{value_by_method['source_selected_fallback']:.10f}` "
            f"and the routed safe system is `{value_by_method['mavis_safe']:.10f}`. "
            f"MAVIS improves {improved_domains}/{len(config.domain_order)} held-out "
            "domains relative to the strongest deployable baseline.\n\n"
            "All policy training, aggregation, baseline choice, confidence thresholds, "
            "and claim rules were fixed from source domains before this aggregated "
            "outer evaluation. Statistics pair physical specimens within held-out "
            "domains and weight the six domains equally.\n\n"
            "This package supports only retrospective normalized-raster closed-loop "
            "feasibility. The external-data audit contains no method-performance "
            "evaluation and does not establish scanner-time reduction, industrial "
            "deployment, or external generalization. Missing positive controls are "
            "not claimed even when the conservative tier remains B.\n"
        )
        (temporary / "REPORT.md").write_text(report, encoding="utf-8")
        scientific_files = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        )
        manifest = {
            "schema_version": 1,
            "artifact": "mavis_p7_final_frozen_eval",
            "p7_state_sha256": p7_state,
            "config_sha256": config.config_sha256,
            "development_package_sha256": development_sha,
            "runtime_code_state_sha256": code_state_sha256,
            "claim_tier": claim_tier,
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
    verify_final_package(output)
    return output


def verify_final_package(path: str | Path) -> dict[str, object]:
    root = Path(path)
    try:
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        lines = (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISFinalPackageError("final package metadata is invalid") from error
    if type(manifest) is not dict or type(manifest.get("files")) is not dict:
        raise MAVISFinalPackageError("final package manifest is invalid")
    expected = set(manifest["files"]) | {"artifact_manifest.json"}
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "CHECKSUMS.sha256"
    }
    if actual != expected or len(lines) != len(expected):
        raise MAVISFinalPackageError("final package file roster changed")
    ledger: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise MAVISFinalPackageError("final checksum ledger is invalid") from error
        if name in ledger or len(digest) != 64:
            raise MAVISFinalPackageError("final checksum ledger is invalid")
        ledger[name] = digest
    if set(ledger) != expected:
        raise MAVISFinalPackageError("final checksum roster changed")
    for name, expected_digest in ledger.items():
        if _sha256(root / name) != expected_digest:
            raise MAVISFinalPackageError(f"final checksum mismatch: {name}")
    for name, metadata in manifest["files"].items():
        file_path = root / name
        if (
            type(metadata) is not dict
            or metadata.get("bytes") != file_path.stat().st_size
            or metadata.get("sha256") != _sha256(file_path)
        ):
            raise MAVISFinalPackageError(f"final manifest mismatch: {name}")
    if any((root / name).stat().st_size >= _GIT_BLOB_LIMIT for name in expected):
        raise MAVISFinalPackageError("final package contains an oversized Git blob")
    try:
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISFinalPackageError("final summary is invalid") from error
    if (
        type(summary) is not dict
        or summary.get("status") != "COMPLETE"
        or summary.get("claim_tier") not in {"S", "A", "B"}
        or summary.get("configuration_frozen") is not True
        or not _is_sha(summary.get("development_package_sha256"))
        or not _is_sha(manifest.get("p7_state_sha256"))
        or summary.get("p7_state_sha256") != manifest.get("p7_state_sha256")
        or summary.get("config_sha256") != manifest.get("config_sha256")
        or summary.get("development_package_sha256")
        != manifest.get("development_package_sha256")
        or summary.get("claim_tier") != manifest.get("claim_tier")
    ):
        raise MAVISFinalPackageError("final package state is inconsistent")
    return manifest


__all__ = [
    "MAVISFinalPackageError",
    "development_package_sha256",
    "finalize_final_package",
    "verify_final_package",
]
