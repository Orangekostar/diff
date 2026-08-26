"""Frozen final MAVIS evaluation with aggregation and safe routing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import polars as pl

from .authority import MAVISAuthority
from .closed_loop_execution import evaluate_inspection_curve
from .config import MAVISConfig
from .dynamic_training import load_fitted_dynamic_checkpoint
from .mris_training import load_fitted_mris_checkpoint
from .policy import DeployedDynamicScorer
from .rollout import rollout_scout_and_focus_curve


class MAVISFinalExecutionError(RuntimeError):
    """Raised when final evaluation changes a frozen development decision."""


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


def _is_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fold_checkpoint(root: str | Path, outer_domain: str) -> Path:
    base = Path(root)
    formal = base / f"{outer_domain}__real.npz"
    local = base / "real.npz"
    return formal if formal.is_file() else local


def run_final_outer_domain(
    authority: MAVISAuthority,
    config: MAVISConfig,
    *,
    outer_domain: str,
    p2_checkpoint_root: str | Path,
    p5_checkpoint_root: str | Path,
    selections: pl.DataFrame,
    output_root: str | Path,
    device: str,
) -> Path:
    started = time.perf_counter()
    if (
        type(authority) is not MAVISAuthority
        or type(config) is not MAVISConfig
        or outer_domain not in config.domain_order
        or not isinstance(selections, pl.DataFrame)
        or type(device) is not str
        or not device
    ):
        raise MAVISFinalExecutionError("final worker request is invalid")
    config.require_finalized()
    required_selection = {
        "outer_domain",
        "baseline",
        "threshold",
        "selection_state_sha256",
        "target_outcomes_used",
    }
    selected = selections.filter(pl.col("outer_domain") == outer_domain)
    if (
        not required_selection <= set(selections.columns)
        or selected.height != 1
        or selected.row(0, named=True)["baseline"]
        not in {"uniform", "reconstruction"}
        or selected.row(0, named=True)["target_outcomes_used"] is not False
        or not _is_sha(selected.row(0, named=True)["selection_state_sha256"])
    ):
        raise MAVISFinalExecutionError("final source-selected routing changed")
    selection = selected.row(0, named=True)
    threshold = float(selection["threshold"])
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise MAVISFinalExecutionError("final source-selected threshold changed")
    p2 = load_fitted_mris_checkpoint(
        _fold_checkpoint(p2_checkpoint_root, outer_domain)
    )
    p5 = load_fitted_dynamic_checkpoint(
        _fold_checkpoint(p5_checkpoint_root, outer_domain)
    )
    if (
        p2.mode != "real"
        or p2.outer_domain != outer_domain
        or p5.outer_domain != outer_domain
        or p2.mris_dimension != p5.mris_dimension
        or outer_domain in p2.audit.fit_domains
        or outer_domain in p5.audit.fit_domains
        or set(p5.audit.fit_domains) != set(config.domain_order) - {outer_domain}
    ):
        raise MAVISFinalExecutionError("final outer-fold model changed")
    scorer = DeployedDynamicScorer(
        mris_model=p2,
        dynamic_model=p5,
        device=device,
    )
    specimen_ids = tuple(
        sorted(
            specimen_id
            for specimen_id, domain_id in zip(
                authority.specimen_ids,
                authority.dataset_ids,
                strict=True,
            )
            if domain_id == outer_domain
        )
    )
    if not specimen_ids:
        raise MAVISFinalExecutionError("final target roster is empty")
    prediction_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    for specimen_id in specimen_ids:
        curve = rollout_scout_and_focus_curve(
            authority,
            specimen_id=specimen_id,
            initial_budget=config.initial_budget_by_domain[outer_domain],
            checkpoints=config.checkpoints,
            scorer=scorer,
            objective="direct_cost_aware",
            feedback=True,
        )
        if not curve.steps:
            raise MAVISFinalExecutionError("final target rollout has no decision")
        confidence = float(curve.steps[0].decision_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise MAVISFinalExecutionError("final target confidence is invalid")
        prediction_rows.extend(
            evaluate_inspection_curve(
                authority,
                outer_domain=outer_domain,
                method="mavis_full",
                checkpoints=config.checkpoints,
                states=curve.checkpoint_states,
                cai_evaluator=p2,
                device=device,
            )
        )
        trajectory_rows.extend(
            {
                "outer_domain": outer_domain,
                "specimen_id": specimen_id,
                "method": "mavis_full",
                "step": step.step,
                "nominal_checkpoint": step.nominal_checkpoint,
                "cell_index": step.action.cell_index,
                "from_level": step.action.from_level,
                "to_level": step.action.to_level,
                "exact_cost_before": step.exact_cost_before,
                "exact_cost_after": step.exact_cost_after,
                "state_sha256_before": step.state_sha256_before,
                "state_sha256_after": step.state_sha256_after,
                "decision_confidence": step.decision_confidence,
            }
            for step in curve.steps
        )
        routing_rows.append(
            {
                "outer_domain": outer_domain,
                "specimen_id": specimen_id,
                "confidence": confidence,
                "threshold": threshold,
                "baseline": str(selection["baseline"]),
                "used_fallback": confidence < threshold,
                "selection_state_sha256": str(
                    selection["selection_state_sha256"]
                ),
            }
        )
    predictions = pl.DataFrame(prediction_rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id", "nominal_checkpoint"]
    )
    trajectories = pl.DataFrame(trajectory_rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id", "step"]
    )
    routing = pl.DataFrame(routing_rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id"]
    )
    expected_predictions = len(specimen_ids) * len(config.checkpoints)
    if (
        predictions.height != expected_predictions
        or predictions.unique(
            subset=["outer_domain", "specimen_id", "nominal_checkpoint"]
        ).height
        != predictions.height
        or routing.height != len(specimen_ids)
        or routing.unique(subset=["outer_domain", "specimen_id"]).height
        != routing.height
    ):
        raise MAVISFinalExecutionError("final worker table roster is incomplete")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / outer_domain
    if destination.exists():
        raise MAVISFinalExecutionError("final worker output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{outer_domain}.", dir=root))
    try:
        predictions.write_parquet(
            temporary / "aggregated_predictions.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        trajectories.write_parquet(
            temporary / "aggregated_trajectories.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        routing.write_csv(temporary / "routing.csv")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        _write_json(
            temporary / "complete.json",
            {
                "schema_version": 1,
                "outer_domain": outer_domain,
                "config_sha256": config.config_sha256,
                "development_package_sha256": config.development_package_sha256,
                "target_specimen_count": len(specimen_ids),
                "prediction_count": predictions.height,
                "trajectory_row_count": trajectories.height,
                "routing_count": routing.height,
                "baseline": str(selection["baseline"]),
                "threshold": threshold,
                "fallback_count": routing.get_column("used_fallback").sum(),
                "p2_model_state_sha256": p2.model_state_sha256,
                "p5_model_state_sha256": p5.model_state_sha256,
                "target_true_cai_used_by_policy": False,
                "future_target_content_used_by_policy": False,
                "runtime_seconds": time.perf_counter() - started,
                "files": {
                    path.relative_to(temporary).as_posix(): _sha256(path)
                    for path in files
                },
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "complete.json"


def compose_final_predictions(
    p4_predictions: pl.DataFrame,
    aggregated_predictions: pl.DataFrame,
    routing: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    key = ["outer_domain", "specimen_id", "nominal_checkpoint"]
    required = {*key, "method"}
    routing_required = {
        "outer_domain",
        "specimen_id",
        "confidence",
        "threshold",
        "baseline",
    }
    if (
        not isinstance(p4_predictions, pl.DataFrame)
        or not isinstance(aggregated_predictions, pl.DataFrame)
        or not isinstance(routing, pl.DataFrame)
        or not required <= set(p4_predictions.columns)
        or set(p4_predictions.columns) != set(aggregated_predictions.columns)
        or not routing_required <= set(routing.columns)
        or p4_predictions.height == 0
        or aggregated_predictions.height == 0
        or routing.height == 0
        or {"mavis_no_aggregation", "mavis_safe"}
        & set(p4_predictions.get_column("method"))
        or set(aggregated_predictions.get_column("method")) != {"mavis_full"}
        or p4_predictions.unique(subset=[*key, "method"]).height
        != p4_predictions.height
        or aggregated_predictions.unique(subset=key).height
        != aggregated_predictions.height
        or routing.unique(subset=["outer_domain", "specimen_id"]).height
        != routing.height
    ):
        raise MAVISFinalExecutionError("final prediction inputs are invalid")
    prior = p4_predictions.filter(pl.col("method") == "mavis_full").sort(key)
    aggregated = aggregated_predictions.sort(key)
    if prior.height != aggregated.height or prior.select(key).rows() != aggregated.select(
        key
    ).rows():
        raise MAVISFinalExecutionError("final aggregated curve roster changed")
    expected_specimens = aggregated.select("outer_domain", "specimen_id").unique().sort(
        ["outer_domain", "specimen_id"]
    )
    if routing.select("outer_domain", "specimen_id").sort(
        ["outer_domain", "specimen_id"]
    ).rows() != expected_specimens.rows():
        raise MAVISFinalExecutionError("final routing specimen roster changed")
    safe_parts: list[pl.DataFrame] = []
    fallback_parts: list[pl.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for route in routing.sort(["outer_domain", "specimen_id"]).iter_rows(named=True):
        outer_domain = str(route["outer_domain"])
        specimen_id = str(route["specimen_id"])
        baseline = str(route["baseline"])
        baseline_method = {
            "uniform": "uniform",
            "reconstruction": "reconstruction_driven",
        }.get(baseline)
        if baseline_method is None:
            raise MAVISFinalExecutionError("final routing baseline changed")
        specimen_filter = (pl.col("outer_domain") == outer_domain) & (
            pl.col("specimen_id") == specimen_id
        )
        mavis_rows = aggregated.filter(specimen_filter).sort(key)
        baseline_rows = p4_predictions.filter(
            specimen_filter & (pl.col("method") == baseline_method)
        ).sort(key)
        selected, audit = select_safe_curve_rows(
            mavis_rows,
            baseline_rows,
            confidence=float(route["confidence"]),
            threshold=float(route["threshold"]),
            baseline=baseline,
        )
        safe_parts.append(selected)
        fallback_parts.append(
            baseline_rows.with_columns(
                pl.lit("source_selected_fallback").alias("method")
            )
        )
        audit_rows.append(
            {
                "outer_domain": outer_domain,
                "specimen_id": specimen_id,
                **audit,
            }
        )
    renamed = p4_predictions.with_columns(
        pl.when(pl.col("method") == "mavis_full")
        .then(pl.lit("mavis_no_aggregation"))
        .otherwise(pl.col("method"))
        .alias("method")
    )
    result = pl.concat(
        [renamed, aggregated, *safe_parts, *fallback_parts],
        how="vertical_relaxed",
    ).sort(["outer_domain", "specimen_id", "method", "nominal_checkpoint"])
    return result, pl.DataFrame(audit_rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id"]
    )


def assign_claim_tier(
    *,
    baseline_cai_auebc: float,
    safe_control_cai_auebc: float,
    mavis_cai_auebc: float,
    safe_cai_auebc: float,
    sequential_oracle_cai_auebc: float,
    mavis_improved_domain_count: int,
    domain_count: int,
    mavis_bootstrap_ci_lower: float,
    safe_bootstrap_ci_lower: float,
    high_confidence_control_minus_mavis_auebc: float,
    high_confidence_bootstrap_ci_lower: float,
    high_confidence_specimen_count: int,
) -> str:
    raw_numeric = (
        baseline_cai_auebc,
        safe_control_cai_auebc,
        mavis_cai_auebc,
        safe_cai_auebc,
        sequential_oracle_cai_auebc,
        mavis_bootstrap_ci_lower,
        safe_bootstrap_ci_lower,
        high_confidence_control_minus_mavis_auebc,
        high_confidence_bootstrap_ci_lower,
    )
    try:
        numeric = tuple(float(value) for value in raw_numeric)
    except (TypeError, ValueError, OverflowError) as error:
        raise MAVISFinalExecutionError("final claim evidence is invalid") from error
    if (
        any(isinstance(value, bool) for value in raw_numeric)
        or not all(math.isfinite(value) for value in numeric)
        or type(mavis_improved_domain_count) is not int
        or type(domain_count) is not int
        or type(high_confidence_specimen_count) is not int
        or not 0 <= mavis_improved_domain_count <= domain_count
        or domain_count <= 0
        or high_confidence_specimen_count < 0
    ):
        raise MAVISFinalExecutionError("final claim evidence is invalid")
    (
        baseline,
        safe_control,
        mavis,
        safe,
        oracle,
        mavis_lower,
        safe_lower,
        high_gain,
        high_lower,
    ) = numeric
    oracle_gap = baseline - oracle
    recovered_gap = (
        (baseline - mavis) / oracle_gap if oracle_gap > 0.0 else float("-inf")
    )
    if (
        mavis < baseline
        and mavis_improved_domain_count > domain_count / 2
        and mavis_lower > 0.0
        and recovered_gap >= 0.1
    ):
        return "S"
    if (
        safe <= safe_control + 1.0e-12
        and safe_lower >= 0.0
        and high_confidence_specimen_count > 0
        and high_gain > 0.0
        and high_lower > 0.0
    ):
        return "A"
    return "B"


def build_risk_coverage(
    specimen_auebc: pl.DataFrame,
    routing: pl.DataFrame,
    *,
    thresholds: tuple[float, ...],
    domain_order: tuple[str, ...],
) -> pl.DataFrame:
    required_metrics = {"outer_domain", "specimen_id", "method", "cai_auebc"}
    required_routing = {"outer_domain", "specimen_id", "confidence"}
    if (
        not isinstance(specimen_auebc, pl.DataFrame)
        or not isinstance(routing, pl.DataFrame)
        or not required_metrics <= set(specimen_auebc.columns)
        or not required_routing <= set(routing.columns)
        or specimen_auebc.height == 0
        or routing.height == 0
        or type(thresholds) is not tuple
        or not thresholds
        or type(domain_order) is not tuple
        or not domain_order
        or len(set(thresholds)) != len(thresholds)
        or tuple(sorted(thresholds)) != thresholds
        or any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in thresholds
        )
        or len(set(domain_order)) != len(domain_order)
        or set(routing.get_column("outer_domain").unique()) != set(domain_order)
        or routing.unique(subset=["outer_domain", "specimen_id"]).height
        != routing.height
        or routing.filter(
            (pl.col("confidence") < 0.0) | (pl.col("confidence") > 1.0)
        ).height
    ):
        raise MAVISFinalExecutionError("final risk-coverage request is invalid")
    selected = specimen_auebc.filter(
        pl.col("method").is_in(["mavis_full", "source_selected_fallback"])
    )
    if (
        set(selected.get_column("method").unique())
        != {"mavis_full", "source_selected_fallback"}
        or selected.unique(subset=["outer_domain", "specimen_id", "method"]).height
        != selected.height
    ):
        raise MAVISFinalExecutionError("final risk-coverage methods are invalid")
    mavis = selected.filter(pl.col("method") == "mavis_full").select(
        "outer_domain",
        "specimen_id",
        pl.col("cai_auebc").alias("mavis_cai_auebc"),
    )
    fallback = selected.filter(
        pl.col("method") == "source_selected_fallback"
    ).select(
        "outer_domain",
        "specimen_id",
        pl.col("cai_auebc").alias("fallback_cai_auebc"),
    )
    paired = (
        routing.select("outer_domain", "specimen_id", "confidence")
        .join(mavis, on=["outer_domain", "specimen_id"], how="inner", validate="1:1")
        .join(
            fallback,
            on=["outer_domain", "specimen_id"],
            how="inner",
            validate="1:1",
        )
    )
    if paired.height != routing.height:
        raise MAVISFinalExecutionError("final risk-coverage pairing is incomplete")
    fallback_domain = paired.group_by("outer_domain").agg(
        pl.col("fallback_cai_auebc").mean().alias("fallback")
    )
    fallback_aggregate = float(fallback_domain.get_column("fallback").mean())
    rows: list[dict[str, object]] = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        routed = paired.with_columns(
            pl.when(pl.col("confidence") >= threshold)
            .then(pl.col("mavis_cai_auebc"))
            .otherwise(pl.col("fallback_cai_auebc"))
            .alias("safe_cai_auebc")
        )
        domains = routed.group_by("outer_domain").agg(
            pl.col("safe_cai_auebc").mean().alias("safe"),
            pl.col("fallback_cai_auebc").mean().alias("fallback"),
        )
        if domains.height != len(domain_order):
            raise MAVISFinalExecutionError("final risk-coverage domains are incomplete")
        rows.append(
            {
                "threshold": threshold,
                "coverage": float((routed.get_column("confidence") >= threshold).mean()),
                "fallback_frequency": float(
                    (routed.get_column("confidence") < threshold).mean()
                ),
                "domain_balanced_cai_auebc": float(domains.get_column("safe").mean()),
                "source_selected_fallback_cai_auebc": fallback_aggregate,
                "worst_domain_cai_auebc": float(domains.get_column("safe").max()),
                "improved_domain_count": domains.filter(
                    pl.col("safe") < pl.col("fallback")
                ).height,
                "domain_count": len(domain_order),
                "statistical_unit": "equal_domain_with_paired_physical_specimens",
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort("threshold")


def select_safe_curve_rows(
    mavis: pl.DataFrame,
    baseline_rows: pl.DataFrame,
    *,
    confidence: float,
    threshold: float,
    baseline: str,
) -> tuple[pl.DataFrame, dict[str, object]]:
    value = float(confidence)
    cutoff = float(threshold)
    baseline_method = {
        "uniform": "uniform",
        "reconstruction": "reconstruction_driven",
    }.get(baseline)
    required = {
        "outer_domain",
        "specimen_id",
        "method",
        "nominal_checkpoint",
    }
    if (
        not isinstance(mavis, pl.DataFrame)
        or not isinstance(baseline_rows, pl.DataFrame)
        or not required <= set(mavis.columns)
        or set(mavis.columns) != set(baseline_rows.columns)
        or mavis.height == 0
        or mavis.height != baseline_rows.height
        or baseline_method is None
        or set(mavis.get_column("method")) != {"mavis_full"}
        or set(baseline_rows.get_column("method")) != {baseline_method}
        or mavis.select("outer_domain", "specimen_id", "nominal_checkpoint").rows()
        != baseline_rows.select(
            "outer_domain", "specimen_id", "nominal_checkpoint"
        ).rows()
        or isinstance(confidence, bool)
        or isinstance(threshold, bool)
        or not math.isfinite(value)
        or not math.isfinite(cutoff)
        or not 0.0 <= value <= 1.0
        or not 0.0 <= cutoff <= 1.0
    ):
        raise MAVISFinalExecutionError("final safe curve request is invalid")
    used_fallback = value < cutoff
    selected_method = baseline_method if used_fallback else "mavis_full"
    selected = (baseline_rows if used_fallback else mavis).with_columns(
        pl.lit("mavis_safe").alias("method")
    )
    return selected, {
        "confidence": value,
        "threshold": cutoff,
        "baseline": baseline,
        "used_fallback": used_fallback,
        "selected_method": selected_method,
    }


__all__ = [
    "MAVISFinalExecutionError",
    "assign_claim_tier",
    "build_risk_coverage",
    "compose_final_predictions",
    "run_final_outer_domain",
    "select_safe_curve_rows",
]
