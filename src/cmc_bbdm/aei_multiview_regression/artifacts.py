"""Atomic, checksum-bound result packages for formal multi-view stages."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .formal_outer import FormalChainResult, PerformanceMetrics, performance_metrics
from .protocol import MultiViewProtocol


class ArtifactError(ValueError):
    """Raised when a formal package is incomplete, changed, or unsafe."""


_REQUIRED = frozenset(
    {
        "config.yaml",
        "aggregate_metrics.csv",
        "domain_metrics.csv",
        "summary.json",
        "REPORT.md",
    }
)


@dataclass(frozen=True, slots=True)
class ReplayedStage:
    stage: str
    files: tuple[str, ...]
    manifest_sha256: str


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _csv(fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    return stream.getvalue().encode("ascii")


def _safe_files(files: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(files, Mapping) or not _REQUIRED <= set(files):
        raise ArtifactError("stage package is missing required files")
    result: dict[str, bytes] = {}
    for name, value in files.items():
        path = Path(name)
        if (
            type(name) is not str
            or not name
            or path.is_absolute()
            or len(path.parts) != 1
            or name in {"artifact_manifest.json", "CHECKSUMS.sha256"}
            or not isinstance(value, bytes)
        ):
            raise ArtifactError("stage package contains an unsafe file")
        result[name] = value
    return result


def write_stage_package(
    destination: str | Path, *, stage: str, files: Mapping[str, bytes]
) -> Path:
    """Write one complete stage by temporary-directory rename, without overwrite."""

    target = Path(destination)
    if type(stage) is not str or stage not in {"E1", "E2", "E3", "E4", "E5"}:
        raise ArtifactError("stage name is invalid")
    payload = _safe_files(files)
    if target.exists() or target.is_symlink():
        raise ArtifactError("stage destination already exists")
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ArtifactError("stage parent is unavailable")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=parent))
    try:
        records = {
            name: {"sha256": _sha256(value), "size": len(value)}
            for name, value in sorted(payload.items())
        }
        manifest = _json({"schema_version": 1, "stage": stage, "files": records})
        complete = dict(payload)
        complete["artifact_manifest.json"] = manifest
        checksums = "".join(
            f"{_sha256(value)}  {name}\n" for name, value in sorted(complete.items())
        ).encode("ascii")
        complete["CHECKSUMS.sha256"] = checksums
        for name, value in complete.items():
            (temporary / name).write_bytes(value)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def replay_stage(path: str | Path) -> ReplayedStage:
    """Verify exact manifest membership and every stored checksum."""

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ArtifactError("stage package is unavailable")
    try:
        manifest_bytes = (root / "artifact_manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError("stage manifest is invalid") from error
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema_version", "stage", "files"}
        or manifest["schema_version"] != 1
        or manifest["stage"] not in {"E1", "E2", "E3", "E4", "E5"}
        or type(manifest["files"]) is not dict
    ):
        raise ArtifactError("stage manifest schema changed")
    expected_names = set(manifest["files"]) | {
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
    actual_names = {item.name for item in root.iterdir() if item.is_file()}
    if actual_names != expected_names or any(
        item.is_symlink() for item in root.iterdir()
    ):
        raise ArtifactError("stage file membership changed")
    for name, record in manifest["files"].items():
        if type(record) is not dict or set(record) != {"sha256", "size"}:
            raise ArtifactError("stage manifest file record changed")
        value = (root / name).read_bytes()
        if len(value) != record["size"] or _sha256(value) != record["sha256"]:
            raise ArtifactError(f"stage checksum changed: {name}")
    expected_checksums = "".join(
        f"{_sha256((root / name).read_bytes())}  {name}\n"
        for name in sorted(set(manifest["files"]) | {"artifact_manifest.json"})
    ).encode("ascii")
    if (root / "CHECKSUMS.sha256").read_bytes() != expected_checksums:
        raise ArtifactError("stage checksum ledger changed")
    return ReplayedStage(
        stage=manifest["stage"],
        files=tuple(sorted(expected_names)),
        manifest_sha256=_sha256(manifest_bytes),
    )


def e1_oof_csv(
    *,
    specimen_ids: Sequence[str],
    domain_ids: Sequence[str],
    targets: object,
    predictions: object,
    cai_strength_mpa: object,
    intact_strength_mpa: object,
) -> bytes:
    """Serialize the registered one-row-per-specimen E1 prediction table."""

    ids = tuple(specimen_ids)
    domains = tuple(domain_ids)
    y = np.asarray(targets, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    strength = np.asarray(cai_strength_mpa, dtype=np.float64)
    intact = np.asarray(intact_strength_mpa, dtype=np.float64)
    if (
        pred.shape != (len(ids), 3)
        or y.shape != (len(ids),)
        or len(domains) != len(ids)
        or strength.shape != y.shape
        or intact.shape != y.shape
    ):
        raise ArtifactError("E1 OOF rows do not align")
    fields = (
        "specimen_id",
        "domain_id",
        "y_true",
        "pred_full",
        "pred_50",
        "pred_25",
        "err_full",
        "err_50",
        "err_25",
        "cai_strength_mpa",
        "intact_strength_mpa",
        "pred_full_mpa",
        "pred_50_mpa",
        "pred_25_mpa",
        "abs_err_full_mpa",
        "abs_err_50_mpa",
        "abs_err_25_mpa",
    )
    rows = []
    for index, specimen in enumerate(ids):
        rows.append(
            {
                "specimen_id": specimen,
                "domain_id": domains[index],
                "y_true": repr(float(y[index])),
                "pred_full": repr(float(pred[index, 0])),
                "pred_50": repr(float(pred[index, 1])),
                "pred_25": repr(float(pred[index, 2])),
                "err_full": repr(float(y[index] - pred[index, 0])),
                "err_50": repr(float(y[index] - pred[index, 1])),
                "err_25": repr(float(y[index] - pred[index, 2])),
                "cai_strength_mpa": repr(float(strength[index])),
                "intact_strength_mpa": repr(float(intact[index])),
                "pred_full_mpa": repr(float(pred[index, 0] * intact[index])),
                "pred_50_mpa": repr(float(pred[index, 1] * intact[index])),
                "pred_25_mpa": repr(float(pred[index, 2] * intact[index])),
                "abs_err_full_mpa": repr(
                    float(abs(y[index] - pred[index, 0]) * intact[index])
                ),
                "abs_err_50_mpa": repr(
                    float(abs(y[index] - pred[index, 1]) * intact[index])
                ),
                "abs_err_25_mpa": repr(
                    float(abs(y[index] - pred[index, 2]) * intact[index])
                ),
            }
        )
    return _csv(fields, rows)


def _metric_files(metrics: Sequence[PerformanceMetrics]) -> tuple[bytes, bytes]:
    aggregate_fields = (
        "method",
        "equal_domain_mae",
        "worst_domain_mae",
        "domain_mae_sd",
        "rmse",
        "r2",
        "equal_domain_mae_mpa",
        "rmse_mpa",
    )
    aggregate = [
        {
            "method": item.method,
            "equal_domain_mae": repr(item.equal_domain_mae),
            "worst_domain_mae": repr(item.worst_domain_mae),
            "domain_mae_sd": repr(item.domain_mae_sd),
            "rmse": repr(item.rmse),
            "r2": repr(item.r2),
            "equal_domain_mae_mpa": repr(item.equal_domain_mae_mpa),
            "rmse_mpa": repr(item.rmse_mpa),
        }
        for item in metrics
    ]
    domain = [
        {
            "method": item.method,
            "domain": name,
            "mae": repr(value),
            "mae_mpa": repr(dict(item.domain_mae_mpa)[name]),
        }
        for item in metrics
        for name, value in item.domain_mae
    ]
    return _csv(aggregate_fields, aggregate), _csv(
        ("method", "domain", "mae", "mae_mpa"), domain
    )


def _p1_reproduction(
    result: FormalChainResult, protocol: MultiViewProtocol
) -> tuple[float, float]:
    matches = tuple(item for item in protocol.sources if item.name == "p1_predictions")
    if len(matches) != 1:
        raise ArtifactError("frozen P1 prediction authority is unavailable")
    try:
        with matches[0].path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            rows = {
                row["specimen_id"]: (float(row["target"]), float(row["prediction"]))
                for row in reader
                if row["method"] == "I_frozen"
            }
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ArtifactError("frozen P1 predictions cannot be decoded") from error
    independent = result.e1.independent
    if set(rows) != set(independent.specimen_ids):
        raise ArtifactError("frozen P1 prediction roster changed")
    frozen_targets = np.asarray([rows[item][0] for item in independent.specimen_ids])
    frozen_predictions = np.asarray(
        [rows[item][1] for item in independent.specimen_ids]
    )
    return (
        float(np.max(np.abs(independent.targets - frozen_targets))),
        float(np.max(np.abs(independent.predictions[:, 0] - frozen_predictions))),
    )


def _prediction_csv(
    specimen_ids: tuple[str, ...],
    domain_ids: tuple[str, ...],
    targets: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    intact_strength_mpa: np.ndarray,
) -> bytes:
    methods = tuple(predictions)
    intact = np.asarray(intact_strength_mpa, dtype=np.float64)
    if intact.shape != targets.shape or not np.all(np.isfinite(intact)):
        raise ArtifactError("MPa prediction scale does not align")
    fields = (
        "specimen_id",
        "domain_id",
        "y_true",
        "y_true_mpa",
        *methods,
        *(f"{method}_mpa" for method in methods),
    )
    rows = []
    for index, specimen in enumerate(specimen_ids):
        row: dict[str, object] = {
            "specimen_id": specimen,
            "domain_id": domain_ids[index],
            "y_true": repr(float(targets[index])),
            "y_true_mpa": repr(float(targets[index] * intact[index])),
        }
        row.update(
            {method: repr(float(predictions[method][index])) for method in methods}
        )
        row.update(
            {
                f"{method}_mpa": repr(
                    float(predictions[method][index] * intact[index])
                )
                for method in methods
            }
        )
        rows.append(row)
    return _csv(fields, rows)


def _e1_files(
    result: FormalChainResult, protocol: MultiViewProtocol
) -> dict[str, bytes]:
    e1 = result.e1
    maximum_target_error, maximum_prediction_error = _p1_reproduction(
        result, protocol
    )
    if maximum_target_error > 1e-12 or maximum_prediction_error > 1e-12:
        raise ArtifactError("FULL does not reproduce the frozen P1 authority")
    metrics = tuple(
        performance_metrics(
            item.view.lower(),
            e1.independent.targets,
            e1.independent.predictions[:, index],
            e1.independent.dataset_ids,
            intact_strength_mpa=e1.independent.intact_strength_mpa,
        )
        for index, item in enumerate(e1.audit.view_metrics)
    )
    aggregate, domain = _metric_files(metrics)
    pair_rows = []
    for left in range(3):
        for right in range(left + 1, 3):
            pair_rows.append(
                {
                    "view_left": e1.audit.view_names[left],
                    "view_right": e1.audit.view_names[right],
                    "prediction_pearson_r": repr(
                        float(e1.audit.prediction_correlations[left, right])
                    ),
                    "residual_pearson_r": repr(
                        float(e1.audit.residual_correlations[left, right])
                    ),
                    "mean_absolute_disagreement": repr(
                        float(e1.audit.mean_absolute_disagreement[left, right])
                    ),
                }
            )
    reliability_rows = [
        {
            "method": method.method,
            "pearson_r": repr(method.pearson_r),
            "pearson_p_value": repr(method.pearson_p_value),
            "spearman_r": repr(method.spearman_r),
            "spearman_p_value": repr(method.spearman_p_value),
            "stratum": stratum.name,
            "count": stratum.count,
            "mean_dispersion": repr(stratum.mean_dispersion),
            "mean_absolute_error": repr(stratum.mean_absolute_error),
        }
        for method in e1.reliability.methods
        for stratum in method.strata
    ]
    reliability_oof_rows = [
        {
            "specimen_id": e1.independent.specimen_ids[index],
            "domain_id": e1.independent.dataset_ids[index],
            "method": method.method,
            "cross_view_dispersion": repr(float(e1.reliability.dispersion[index])),
            "absolute_error": repr(float(method.absolute_errors[index])),
            "stratum": e1.reliability.stratum_labels[index],
        }
        for method in e1.reliability.methods
        for index in range(len(e1.independent.specimen_ids))
    ]
    selection_rows = [
        {
            "outer_domain": item.outer_domain,
            "view": item.view,
            "pca_dimension": item.pca_dimension,
            "dimension_scores": json.dumps(dict(item.dimension_scores), sort_keys=True),
        }
        for item in e1.independent.selections
    ]
    best_rows = [
        {
            "group_name": item.group_name,
            "group_value": item.group_value,
            "specimen_count": item.specimen_count,
            "full_count": item.counts[0],
            "bilinear_50_count": item.counts[1],
            "bilinear_25_count": item.counts[2],
            "full_frequency": repr(item.frequencies[0]),
            "bilinear_50_frequency": repr(item.frequencies[1]),
            "bilinear_25_frequency": repr(item.frequencies[2]),
        }
        for item in e1.audit.grouped_best_view
    ]
    summary = {
        "schema_version": 1,
        "stage": "E1",
        "gate_status": e1.gate_status,
        "predictive_equivalence": e1.audit.predictive_equivalence,
        "complementarity_signal": e1.audit.complementarity_signal,
        "oracle_is_deployable": False,
        "oracle_mae": e1.audit.oracle_mae,
        "oracle_improvement_vs_full": e1.audit.oracle_improvement_vs_full,
        "baseline_equal_domain_mae": protocol.baseline_mae,
        "maximum_full_prediction_error": maximum_prediction_error,
        "maximum_target_error": maximum_target_error,
        "reproduction_tolerance": 1e-12,
        "best_view_counts": dict(
            zip(e1.audit.view_names, e1.audit.best_view_counts, strict=True)
        ),
        "independent_state_sha256": e1.independent.state_sha256,
    }
    return {
        "config.yaml": protocol.config_path.read_bytes(),
        "aggregate_metrics.csv": aggregate,
        "domain_metrics.csv": domain,
        "summary.json": _json(summary),
        "REPORT.md": (
            f"# E1 Cross-View Audit\n\nGate: `{e1.gate_status}`. "
            f"Oracle MAE `{e1.audit.oracle_mae:.12f}` is diagnostic only.\n"
        ).encode("ascii"),
        "oof_predictions.csv": e1_oof_csv(
            specimen_ids=e1.independent.specimen_ids,
            domain_ids=e1.independent.dataset_ids,
            targets=e1.independent.targets,
            predictions=e1.independent.predictions,
            cai_strength_mpa=e1.independent.cai_strength_mpa,
            intact_strength_mpa=e1.independent.intact_strength_mpa,
        ),
        "agreement.csv": _csv(tuple(pair_rows[0]), pair_rows),
        "reliability.csv": _csv(tuple(reliability_rows[0]), reliability_rows),
        "reliability_oof.csv": _csv(
            tuple(reliability_oof_rows[0]), reliability_oof_rows
        ),
        "selections.csv": _csv(tuple(selection_rows[0]), selection_rows),
        "best_view_frequencies.csv": _csv(tuple(best_rows[0]), best_rows),
    }


def _e2_files(
    result: FormalChainResult, protocol: MultiViewProtocol
) -> dict[str, bytes]:
    if result.e2 is None:
        raise ArtifactError("E2 result is unavailable")
    e2 = result.e2
    aggregate, domain = _metric_files(e2.metrics)
    selections = [
        {
            "outer_domain": item.outer_domain,
            "loss": item.selected.loss,
            "lambda_consistency": repr(item.selected.lambda_consistency),
            "validation_weights": json.dumps(item.validation_weights.tolist()),
        }
        for item in e2.outer_states
    ]
    searches = [
        {
            "outer_domain": outer.outer_domain,
            "loss": score.candidate.loss,
            "lambda_consistency": repr(score.candidate.lambda_consistency),
            "equal_domain_mae": repr(score.equal_domain_mae),
            "worst_domain_mae": repr(score.worst_domain_mae),
            "domain_mae_sd": repr(score.domain_mae_sd),
            "mean_absolute_disagreement": repr(score.mean_absolute_disagreement),
            "prediction_variances": json.dumps(score.prediction_variances),
            "residual_correlations": json.dumps(score.residual_correlations),
            "collapsed": score.collapsed,
        }
        for outer in e2.outer_states
        for score in outer.search.scores
    ]
    metric_by_name = {item.method: item for item in e2.metrics}
    full_mae = metric_by_name["full"].equal_domain_mae
    cooperative_mae = metric_by_name["cooperative_selected"].equal_domain_mae
    summary = {
        "schema_version": 1,
        "stage": "E2",
        "gate_status": e2.gate_status,
        "cooperative_improved_domains": e2.cooperative_improved_domains,
        "cooperative_equal_domain_mae": cooperative_mae,
        "fusion_gain_vs_full": full_mae - cooperative_mae,
        "fusion_gain_fraction_vs_full": (full_mae - cooperative_mae) / full_mae,
        "selected_candidates": [
            {
                "outer_domain": item.outer_domain,
                "loss": item.selected.loss,
                "lambda_consistency": item.selected.lambda_consistency,
            }
            for item in e2.outer_states
        ],
    }
    return {
        "config.yaml": protocol.config_path.read_bytes(),
        "aggregate_metrics.csv": aggregate,
        "domain_metrics.csv": domain,
        "summary.json": _json(summary),
        "REPORT.md": (
            f"# E2 Cooperative Regression\n\nGate: `{e2.gate_status}`; "
            f"improved domains: `{e2.cooperative_improved_domains}/6`.\n"
        ).encode("ascii"),
        "oof_predictions.csv": _prediction_csv(
            result.e1.independent.specimen_ids,
            result.e1.independent.dataset_ids,
            result.e1.independent.targets,
            e2.predictions,
            result.e1.independent.intact_strength_mpa,
        ),
        "selections.csv": _csv(tuple(selections[0]), selections),
        "search_diagnostics.csv": _csv(tuple(searches[0]), searches),
    }


def _e3_files(
    result: FormalChainResult, protocol: MultiViewProtocol
) -> dict[str, bytes]:
    if result.e3 is None or result.bootstrap is None or result.stress is None:
        raise ArtifactError("E3 result is unavailable")
    e3 = result.e3
    aggregate, domain = _metric_files(e3.metrics)
    states = [
        {
            "outer_domain": item.outer_domain,
            "stacking_method": item.stacking_method,
            "gmvr_lambda_consistency": repr(item.gmvr.lambda_consistency),
            "gmvr_lambda_complementarity": repr(item.gmvr.lambda_complementarity),
            "gmvr_weights": json.dumps(item.gmvr.weights.tolist()),
            "gmvr_contributions": json.dumps(item.gmvr.mean_absolute_contributions),
            "gmvr_candidates": json.dumps(item.gmvr_candidates),
        }
        for item in e3.outer_states
    ]
    bootstrap_rows = [
        {
            "effect": item.name,
            "point_estimate": repr(item.point_estimate),
            "bootstrap_mean": repr(item.bootstrap_mean),
            "ordinary_low": repr(item.ordinary_interval[0]),
            "ordinary_high": repr(item.ordinary_interval[1]),
            "familywise_low": repr(item.familywise_interval[0]),
            "familywise_high": repr(item.familywise_interval[1]),
            "probability_positive": repr(item.probability_positive),
        }
        for item in result.bootstrap.effects
    ]
    stress_aggregate = [
        {
            "scheme": metric.scheme,
            "method": metric.method,
            "equal_group_mae": repr(metric.equal_group_mae),
            "worst_group_mae": repr(metric.worst_group_mae),
        }
        for scheme in result.stress.schemes
        for metric in scheme.metrics
    ]
    stress_groups = [
        {
            "scheme": metric.scheme,
            "method": metric.method,
            "heldout_group": group,
            "mae": repr(mae),
        }
        for scheme in result.stress.schemes
        for metric in scheme.metrics
        for group, mae in metric.group_mae
    ]
    stress_selections = [
        {
            "scheme": state.scheme,
            "heldout_group": state.heldout_group,
            "pca_dimensions": json.dumps(state.pca_dimensions),
            "cooperative_loss": state.cooperative_candidate.loss,
            "cooperative_lambda": repr(state.cooperative_candidate.lambda_consistency),
            "stacking_method": state.stacking_method,
            "gmvr_weights": json.dumps(state.gmvr_weights.tolist()),
        }
        for scheme in result.stress.schemes
        for state in scheme.fold_states
    ]
    stress_prediction_rows = [
        {
            "scheme": scheme.scheme,
            "specimen_id": result.e1.independent.specimen_ids[index],
            "heldout_group": scheme.group_values[index],
            "y_true": repr(float(result.e1.independent.targets[index])),
            **{
                method: repr(float(values[index]))
                for method, values in scheme.predictions.items()
            },
        }
        for scheme in result.stress.schemes
        for index in range(len(result.e1.independent.specimen_ids))
    ]
    metric_by_name = {item.method: item for item in e3.metrics}
    best_mae = metric_by_name[e3.best_method].equal_domain_mae
    full_mae = result.e1.audit.view_metrics[0].equal_domain_mae
    best_single_mae = min(
        item.equal_domain_mae for item in result.e1.audit.view_metrics
    )
    summary = {
        "schema_version": 1,
        "stage": "E3",
        "gate_status": e3.gate_status,
        "best_method": e3.best_method,
        "best_equal_domain_mae": best_mae,
        "best_single_view_mae": best_single_mae,
        "fusion_gain_vs_full": full_mae - best_mae,
        "fusion_gain_vs_best_single": best_single_mae - best_mae,
        "best_improved_domains": e3.best_improved_domains,
        "e4_status": result.e4_status,
        "e5_status": result.e5_status,
        "bootstrap_seed": result.bootstrap.seed,
        "bootstrap_resamples": result.bootstrap.resamples,
    }
    return {
        "config.yaml": protocol.config_path.read_bytes(),
        "aggregate_metrics.csv": aggregate,
        "domain_metrics.csv": domain,
        "summary.json": _json(summary),
        "REPORT.md": (
            f"# E3 Complementarity\n\nGate: `{e3.gate_status}`; best method: "
            f"`{e3.best_method}`; improved domains: `{e3.best_improved_domains}/6`.\n"
        ).encode("ascii"),
        "oof_predictions.csv": _prediction_csv(
            result.e1.independent.specimen_ids,
            result.e1.independent.dataset_ids,
            result.e1.independent.targets,
            e3.predictions,
            result.e1.independent.intact_strength_mpa,
        ),
        "fusion_selections.csv": _csv(tuple(states[0]), states),
        "bootstrap_effects.csv": _csv(tuple(bootstrap_rows[0]), bootstrap_rows),
        "stress_aggregate_metrics.csv": _csv(
            tuple(stress_aggregate[0]), stress_aggregate
        ),
        "stress_group_metrics.csv": _csv(tuple(stress_groups[0]), stress_groups),
        "stress_selections.csv": _csv(tuple(stress_selections[0]), stress_selections),
        "stress_oof_predictions.csv": _csv(
            tuple(stress_prediction_rows[0]), stress_prediction_rows
        ),
    }


def publish_formal_chain(
    result: FormalChainResult,
    *,
    protocol: MultiViewProtocol,
) -> tuple[Path, ...]:
    """Publish every authorized completed stage under one atomic output root."""

    root = protocol.output_root
    if root.exists() or root.is_symlink():
        raise ArtifactError("formal output root already exists")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise ArtifactError("formal output parent is unavailable")
    temporary = Path(tempfile.mkdtemp(prefix=".multiview.", dir=root.parent))
    names: list[str] = []
    try:
        write_stage_package(
            temporary / "e1_audit", stage="E1", files=_e1_files(result, protocol)
        )
        names.append("e1_audit")
        if result.e2 is not None:
            write_stage_package(
                temporary / "e2_cooperative",
                stage="E2",
                files=_e2_files(result, protocol),
            )
            names.append("e2_cooperative")
        if result.e3 is not None:
            write_stage_package(
                temporary / "e3_complementarity",
                stage="E3",
                files=_e3_files(result, protocol),
            )
            names.append("e3_complementarity")
        os.replace(temporary, root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return tuple(root / name for name in names)
