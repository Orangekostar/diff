"""Assemble prediction-derived formal M0 metrics and the frozen gate."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .evaluation import PredictionRecord
from .m0_components import ComponentEvaluation
from .m0_residual_audit import ResidualAudit, SourceResidualRecord
from .protocol import MGMRProtocol
from .statistics import (
    DomainBootstrap,
    M0Decision,
    MetricSummary,
    decide_m0,
    paired_domain_bootstrap,
    prediction_metrics,
)


class MGMRFormalError(ValueError):
    """Raised when formal M0 evidence is incomplete or misaligned."""


@dataclass(frozen=True, slots=True)
class M0FormalResult:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    prediction_records: Mapping[str, tuple[PredictionRecord, ...]]
    metrics: Mapping[str, MetricSummary]
    bootstrap: DomainBootstrap
    decision: M0Decision
    source_residuals: tuple[SourceResidualRecord, ...]
    component_state_sha256: str
    residual_state_sha256: str
    state_sha256: str


def _records(
    method: str,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    targets: np.ndarray,
    predictions: np.ndarray,
    dimensions: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[PredictionRecord, ...]:
    if targets.shape != predictions.shape or targets.shape != (len(specimen_ids),):
        raise MGMRFormalError("formal prediction arrays are not aligned")
    return tuple(
        PredictionRecord(
            method=method,
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            target=float(targets[index]),
            prediction=float(predictions[index]),
            dimensions=dimensions[dataset_id] if dimensions is not None else (),
        )
        for index, (specimen_id, dataset_id) in enumerate(
            zip(specimen_ids, dataset_ids, strict=True)
        )
    )


def _domain_effect(reference: MetricSummary, candidate: MetricSummary) -> tuple[float, ...]:
    reference_rows = reference.domain_metrics
    candidate_rows = candidate.domain_metrics
    if tuple(row.domain for row in reference_rows) != tuple(
        row.domain for row in candidate_rows
    ):
        raise MGMRFormalError("formal metric domains are not aligned")
    return tuple(
        left.mae - right.mae
        for left, right in zip(reference_rows, candidate_rows, strict=True)
    )


def evaluate_formal_outer(
    protocol: MGMRProtocol,
    components: ComponentEvaluation,
    residual: ResidualAudit,
) -> M0FormalResult:
    """Derive every reported M0 statistic from ordered outer predictions."""

    if type(protocol) is not MGMRProtocol:
        raise MGMRFormalError("issued M0 protocol is required")
    if (
        components.specimen_ids != residual.specimen_ids
        or components.dataset_ids != residual.dataset_ids
        or not np.array_equal(components.targets, residual.targets)
        or components.specimen_ids != tuple(residual.specimen_ids)
    ):
        raise MGMRFormalError("component and residual evidence are not aligned")
    samples = components.specimen_ids
    datasets = components.dataset_ids
    targets = components.targets
    dimensions: dict[str, Mapping[str, tuple[int, ...]]] = {
        "B0": {
            domain: (components.baseline_dimensions[index],)
            for index, domain in enumerate(protocol.domain_order)
        }
    }
    for method, run in components.runs.items():
        dimensions[method] = {
            domain: selection.dimensions
            for domain, selection in run.selection_by_domain.items()
        }
    records: dict[str, tuple[PredictionRecord, ...]] = {}
    for method in components.methods:
        records[method] = _records(
            method,
            samples,
            datasets,
            targets,
            components.predictions[method],
            dimensions[method],
        )
    residual_roster = {
        "R_coarse": residual.coarse,
        "R_full": residual.full,
        **{
            f"P3_{seed}": residual.shuffles[seed]
            for seed in protocol.specificity_seeds
        },
    }
    for method, branch in residual_roster.items():
        selected = {
            row.outer_domain: (row.selected_dimension,) for row in branch.outer_records
        }
        records[method] = _records(
            method,
            samples,
            datasets,
            targets,
            branch.corrected_predictions,
            selected,
        )
        signal_method = f"S_{method}"
        records[signal_method] = _records(
            signal_method,
            samples,
            datasets,
            targets - branch.baseline_predictions,
            branch.corrections,
            selected,
        )
    metrics = {
        method: prediction_metrics(rows, domain_order=protocol.domain_order)
        for method, rows in records.items()
    }
    direct = {method: metrics[method] for method in ("B1", "B2", "B3")}
    shuffled_metrics = {
        seed: metrics[f"P3_{seed}"] for seed in protocol.specificity_seeds
    }
    decision = decide_m0(
        direct=direct,
        coarse_baseline=metrics["B1"],
        coarse_corrected=metrics["R_coarse"],
        full_baseline=metrics["B0"],
        full_corrected=metrics["R_full"],
        shuffled=shuffled_metrics,
        required_gates=protocol.gate_required,
        minimum_positive_domains=protocol.minimum_positive_domains,
    )
    effects: dict[str, tuple[float, ...]] = {
        "B1_minus_B3": _domain_effect(metrics["B1"], metrics["B3"]),
        "B2_minus_B3": _domain_effect(metrics["B2"], metrics["B3"]),
        "B0_minus_B4": _domain_effect(metrics["B0"], metrics["B4"]),
        "B1_minus_R_coarse": _domain_effect(metrics["B1"], metrics["R_coarse"]),
        "B0_minus_R_full": _domain_effect(metrics["B0"], metrics["R_full"]),
    }
    for seed in protocol.specificity_seeds:
        shuffle_name = f"P3_{seed}"
        effects[f"B1_minus_{shuffle_name}"] = _domain_effect(
            metrics["B1"], metrics[shuffle_name]
        )
        effects[f"real_minus_{shuffle_name}"] = _domain_effect(
            metrics[shuffle_name], metrics["R_coarse"]
        )
    bootstrap = paired_domain_bootstrap(
        effects,
        domain_order=protocol.domain_order,
        seed=protocol.bootstrap_seed,
        resamples=protocol.bootstrap_resamples,
        quantiles=protocol.bootstrap_quantiles,
    )
    immutable_records = MappingProxyType(records)
    immutable_metrics = MappingProxyType(metrics)
    digest = hashlib.sha256()
    digest.update(protocol.config_sha256.encode("ascii"))
    digest.update(components.state_sha256.encode("ascii"))
    digest.update(residual.state_sha256.encode("ascii"))
    digest.update(decision.status.encode("ascii"))
    digest.update(bootstrap.draw_sha256.encode("ascii"))
    for method, rows in records.items():
        digest.update(method.encode("ascii"))
        for row in rows:
            digest.update(repr((row.specimen_id, row.target, row.prediction)).encode("ascii"))
    return M0FormalResult(
        specimen_ids=samples,
        dataset_ids=datasets,
        prediction_records=immutable_records,
        metrics=immutable_metrics,
        bootstrap=bootstrap,
        decision=decision,
        source_residuals=residual.source_residuals,
        component_state_sha256=components.state_sha256,
        residual_state_sha256=residual.state_sha256,
        state_sha256=digest.hexdigest(),
    )


__all__ = ["M0FormalResult", "MGMRFormalError", "evaluate_formal_outer"]
