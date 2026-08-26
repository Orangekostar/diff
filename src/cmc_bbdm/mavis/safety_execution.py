"""Nested source-validation execution for MAVIS confidence fallback."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import polars as pl

from .authority import MAVISAuthority
from .closed_loop_execution import (
    evaluate_inspection_curve,
    replay_state_manifest_row,
)
from .closed_loop_metrics import evaluate_closed_loop_predictions
from .config import MAVISConfig
from .dynamic_training import load_fitted_dynamic_checkpoint
from .fallback import SafePolicySelection, select_source_safe_policy
from .mris_training import load_fitted_mris_checkpoint
from .policy import DeployedDynamicScorer
from .rollout import rollout_scout_and_focus_curve


class MAVISSafetyExecutionError(RuntimeError):
    """Raised when confidence calibration crosses its nested source fold."""


_METHODS = ("mavis", "uniform", "reconstruction_driven")


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


def _inner_checkpoint(
    root: str | Path,
    *,
    outer_domain: str,
    validation_domain: str,
    mode: str,
) -> Path:
    base = Path(root)
    candidates = (
        base / "inner" / f"{validation_domain}__{mode}.npz",
        base / "inner" / f"{outer_domain}__{validation_domain}__{mode}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise MAVISSafetyExecutionError("nested safety checkpoint is unavailable")


def build_source_safe_metrics(
    specimen_auebc: pl.DataFrame,
    confidence: pl.DataFrame,
    *,
    outer_domain: str,
) -> pl.DataFrame:
    required_auebc = {"outer_domain", "specimen_id", "method", "cai_auebc"}
    required_confidence = {"domain_id", "specimen_id", "confidence"}
    if (
        not isinstance(specimen_auebc, pl.DataFrame)
        or not isinstance(confidence, pl.DataFrame)
        or not required_auebc <= set(specimen_auebc.columns)
        or not required_confidence <= set(confidence.columns)
        or type(outer_domain) is not str
        or not outer_domain
        or specimen_auebc.height == 0
        or confidence.height == 0
        or outer_domain in specimen_auebc.get_column("outer_domain").unique()
        or outer_domain in confidence.get_column("domain_id").unique()
        or set(specimen_auebc.get_column("method").unique()) != set(_METHODS)
    ):
        raise MAVISSafetyExecutionError("safe source metric request is invalid")
    tables = {}
    for method, output in (
        ("mavis", "mavis_auebc"),
        ("uniform", "uniform_auebc"),
        ("reconstruction_driven", "reconstruction_auebc"),
    ):
        tables[method] = specimen_auebc.filter(pl.col("method") == method).select(
            pl.col("outer_domain").alias("domain_id"),
            "specimen_id",
            pl.col("cai_auebc").alias(output),
        )
    result = (
        confidence.join(
            tables["mavis"],
            on=["domain_id", "specimen_id"],
            how="inner",
            validate="1:1",
        )
        .join(
            tables["uniform"],
            on=["domain_id", "specimen_id"],
            how="inner",
            validate="1:1",
        )
        .join(
            tables["reconstruction_driven"],
            on=["domain_id", "specimen_id"],
            how="inner",
            validate="1:1",
        )
        .sort(["domain_id", "specimen_id"])
    )
    expected = confidence.height
    if (
        result.height != expected
        or result.unique(subset=["domain_id", "specimen_id"]).height != expected
        or any(table.height != expected for table in tables.values())
    ):
        raise MAVISSafetyExecutionError("safe source metric pairing is incomplete")
    return result


def _selection_payload(selection: SafePolicySelection) -> dict[str, object]:
    return {
        "outer_domain": selection.outer_domain,
        "baseline": selection.baseline,
        "threshold": selection.threshold,
        "source_domains": list(selection.source_domains),
        "source_specimen_ids": list(selection.source_specimen_ids),
        "target_outcomes_used": selection.target_outcomes_used,
        "state_sha256": selection.state_sha256,
    }


def run_safety_outer_domain(
    authority: MAVISAuthority,
    config: MAVISConfig,
    *,
    outer_domain: str,
    p1_states: pl.DataFrame,
    p2_checkpoint_root: str | Path,
    p3_checkpoint_root: str | Path,
    output_root: str | Path,
    device: str,
) -> Path:
    if (
        type(authority) is not MAVISAuthority
        or type(config) is not MAVISConfig
        or outer_domain not in config.domain_order
        or not isinstance(p1_states, pl.DataFrame)
        or type(device) is not str
        or not device
    ):
        raise MAVISSafetyExecutionError("safety worker request is invalid")
    source_domains = tuple(
        domain for domain in config.domain_order if domain != outer_domain
    )
    prediction_rows: list[dict[str, object]] = []
    confidence_rows: list[dict[str, object]] = []
    p2_states: dict[str, str] = {}
    p3_states: dict[str, str] = {}
    for validation_domain in source_domains:
        p2 = load_fitted_mris_checkpoint(
            _inner_checkpoint(
                p2_checkpoint_root,
                outer_domain=outer_domain,
                validation_domain=validation_domain,
                mode="real",
            )
        )
        p3 = load_fitted_dynamic_checkpoint(
            _inner_checkpoint(
                p3_checkpoint_root,
                outer_domain=outer_domain,
                validation_domain=validation_domain,
                mode="real",
            )
        )
        if (
            p2.outer_domain != outer_domain
            or p2.audit.validation_domains != (validation_domain,)
            or validation_domain in p2.audit.fit_domains
            or p3.outer_domain != outer_domain
            or p3.audit.validation_domain != validation_domain
            or validation_domain in p3.audit.fit_domains
            or outer_domain in p3.audit.fit_domains
            or p2.mris_dimension != p3.mris_dimension
        ):
            raise MAVISSafetyExecutionError("nested safety model fold changed")
        p2_states[validation_domain] = p2.model_state_sha256
        p3_states[validation_domain] = p3.model_state_sha256
        scorer = DeployedDynamicScorer(
            mris_model=p2,
            dynamic_model=p3,
            device=device,
        )
        validation_states = p1_states.filter(
            pl.col("domain_id") == validation_domain
        )
        specimen_ids = tuple(
            sorted(validation_states.get_column("specimen_id").unique())
        )
        for specimen_id in specimen_ids:
            initial_budget = float(config.initial_budget_by_domain[validation_domain])
            curve = rollout_scout_and_focus_curve(
                authority,
                specimen_id=specimen_id,
                initial_budget=initial_budget,
                checkpoints=config.checkpoints,
                scorer=scorer,
                objective="direct_cost_aware",
                feedback=True,
            )
            if not curve.steps:
                raise MAVISSafetyExecutionError("safe calibration has no decision")
            confidence_rows.append(
                {
                    "domain_id": validation_domain,
                    "specimen_id": specimen_id,
                    "confidence": curve.steps[0].decision_confidence,
                }
            )
            prediction_rows.extend(
                evaluate_inspection_curve(
                    authority,
                    outer_domain=validation_domain,
                    method="mavis",
                    checkpoints=config.checkpoints,
                    states=curve.checkpoint_states,
                    cai_evaluator=p2,
                    device=device,
                )
            )
            specimen_states = validation_states.filter(
                pl.col("specimen_id") == specimen_id
            )
            for method in ("uniform", "reconstruction_driven"):
                table = specimen_states.filter(pl.col("method") == method).sort(
                    "nominal_checkpoint"
                )
                if table.height != len(config.checkpoints):
                    raise MAVISSafetyExecutionError(
                        "safe calibration baseline curve is incomplete"
                    )
                prediction_rows.extend(
                    evaluate_inspection_curve(
                        authority,
                        outer_domain=validation_domain,
                        method=method,
                        checkpoints=config.checkpoints,
                        states=tuple(
                            replay_state_manifest_row(authority, row)
                            for row in table.iter_rows(named=True)
                        ),
                        cai_evaluator=p2,
                        device=device,
                    )
                )
    predictions = pl.DataFrame(prediction_rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id", "method", "nominal_checkpoint"]
    )
    confidence = pl.DataFrame(confidence_rows, infer_schema_length=None).sort(
        ["domain_id", "specimen_id"]
    )
    metrics = evaluate_closed_loop_predictions(
        predictions,
        domain_order=source_domains,
        method_order=_METHODS,
        checkpoints=config.checkpoints,
    )
    source_metrics = build_source_safe_metrics(
        metrics.specimen_auebc,
        confidence,
        outer_domain=outer_domain,
    )
    selection = select_source_safe_policy(
        source_metrics,
        outer_domain=outer_domain,
        thresholds=config.confidence_thresholds,
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / outer_domain
    if destination.exists():
        raise MAVISSafetyExecutionError("safety worker output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{outer_domain}.", dir=root))
    try:
        predictions.write_parquet(
            temporary / "calibration_predictions.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        source_metrics.write_csv(temporary / "source_metrics.csv")
        selection.audit.write_csv(temporary / "threshold_audit.csv")
        _write_json(temporary / "selection.json", _selection_payload(selection))
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        _write_json(
            temporary / "complete.json",
            {
                "schema_version": 1,
                "outer_domain": outer_domain,
                "config_sha256": config.config_sha256,
                "source_domains": list(source_domains),
                "source_specimen_count": source_metrics.height,
                "p2_inner_model_state_sha256": p2_states,
                "p3_inner_model_state_sha256": p3_states,
                "baseline": selection.baseline,
                "threshold": selection.threshold,
                "selection_state_sha256": selection.state_sha256,
                "calibration_policy": "nested_p3_pre_aggregation",
                "target_outcomes_used_for_selection": False,
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


__all__ = [
    "MAVISSafetyExecutionError",
    "build_source_safe_metrics",
    "run_safety_outer_domain",
]
