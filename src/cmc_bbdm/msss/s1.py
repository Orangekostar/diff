"""S1 scale-curve aggregation and frozen three-axis decision logic."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .msss_selector import (
    AxisGate,
    S1Gate,
    StabilityResult,
    axis_gate,
    s1_gate,
    selection_stability,
)
from .noninferiority import NoninferiorityResult, select_noninferior
from .protocol import MSSSProtocol
from .scale_evaluator import AxisEvaluation
from .scale_features import ScaleFeatureBank
from .spatial_specificity import SpatialSpecificityEvaluation, SpecificityResult
from .statistics import CommonBootstrap, EffectInterval, common_stratified_bootstrap


class S1EvaluationError(ValueError):
    """Raised when completed S1 evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class DomainScaleMetric:
    axis: str
    condition_id: str
    dataset_id: str
    specimen_count: int
    mae: float


@dataclass(frozen=True, slots=True)
class ScaleCurveMetric:
    axis: str
    condition_id: str
    value: float
    coarse_rank: int
    primary_eligible: bool
    wavelet: str | None
    level: int | None
    mode: str | None
    normalized_retention_index: float
    equal_domain_mae: float
    ci_low: float
    ci_high: float
    full_equal_domain_mae: float
    relative_gap: float
    noninferior_025: bool
    noninferior_05: bool
    noninferior_075: bool


@dataclass(frozen=True, slots=True)
class AxisSummary:
    axis: str
    full_condition_id: str
    global_selected_condition_id: str
    global_over_coarse_condition_id: str | None
    global_noninferiority: NoninferiorityResult
    selected_equal_domain_mae: float
    selected_ci_low: float
    selected_ci_high: float
    full_equal_domain_mae: float
    mechanically_sufficient: bool
    stability: StabilityResult
    specificity: SpecificityResult
    specificity_interval: EffectInterval
    specificity_simultaneous_interval: EffectInterval
    gate: AxisGate


@dataclass(frozen=True, slots=True)
class S1Run:
    evaluations: tuple[AxisEvaluation, ...]
    specificity_evaluations: tuple[SpatialSpecificityEvaluation, ...]
    curves: tuple[ScaleCurveMetric, ...]
    domain_metrics: tuple[DomainScaleMetric, ...]
    axis_summaries: tuple[AxisSummary, ...]
    curve_bootstrap: CommonBootstrap
    specificity_bootstrap: CommonBootstrap
    specificity_simultaneous_bootstrap: CommonBootstrap
    gate: S1Gate
    feature_bank_state_sha256: str
    state_sha256: str


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise S1EvaluationError("mean requires nonempty values")
    result = float(math.fsum(float(value) for value in values) / len(values))
    if not math.isfinite(result):
        raise S1EvaluationError("mean is non-finite")
    return result


def _prediction_errors(
    evaluation: AxisEvaluation,
    *,
    specimen_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    positions = {specimen: index for index, specimen in enumerate(specimen_ids)}
    conditions = tuple(
        dict.fromkeys(row.condition_id for row in evaluation.candidate_predictions)
    )
    output = {condition: np.full(len(specimen_ids), np.nan) for condition in conditions}
    seen: set[tuple[str, str]] = set()
    for row in evaluation.candidate_predictions:
        key = (row.condition_id, row.specimen_id)
        if key in seen or row.specimen_id not in positions:
            raise S1EvaluationError("candidate prediction roster is invalid")
        seen.add(key)
        output[row.condition_id][positions[row.specimen_id]] = row.absolute_error
    if any(not np.all(np.isfinite(value)) for value in output.values()):
        raise S1EvaluationError("candidate prediction roster is incomplete")
    return output


def _selected_errors(
    evaluation: AxisEvaluation, *, specimen_ids: tuple[str, ...]
) -> np.ndarray:
    positions = {specimen: index for index, specimen in enumerate(specimen_ids)}
    output = np.full(len(specimen_ids), np.nan)
    for row in evaluation.selected_predictions:
        if row.specimen_id not in positions or np.isfinite(output[positions[row.specimen_id]]):
            raise S1EvaluationError("selected prediction roster is invalid")
        output[positions[row.specimen_id]] = row.absolute_error
    if not np.all(np.isfinite(output)):
        raise S1EvaluationError("selected prediction roster is incomplete")
    return output


def _specificity_effects(
    evaluation: SpatialSpecificityEvaluation,
    *,
    specimen_ids: tuple[str, ...],
) -> np.ndarray:
    by_specimen: dict[str, list[float]] = {specimen: [] for specimen in specimen_ids}
    regular: dict[str, float] = {}
    for row in evaluation.predictions:
        if row.specimen_id not in by_specimen:
            raise S1EvaluationError("specificity specimen roster changed")
        by_specimen[row.specimen_id].append(row.shuffled_absolute_error)
        previous = regular.setdefault(row.specimen_id, row.regular_absolute_error)
        if previous != row.regular_absolute_error:
            raise S1EvaluationError("regular specificity error changed across seeds")
    if any(not values for values in by_specimen.values()):
        raise S1EvaluationError("specificity prediction roster is incomplete")
    return np.asarray(
        [_mean(by_specimen[item]) - regular[item] for item in specimen_ids],
        dtype=np.float64,
    )


def summarize_s1(
    protocol: MSSSProtocol,
    *,
    bank: ScaleFeatureBank,
    evaluations: Sequence[AxisEvaluation],
    specificity: Sequence[SpatialSpecificityEvaluation],
    bootstrap_resamples: int | None = None,
) -> S1Run:
    """Aggregate completed S1 predictions and apply the frozen gate."""

    if type(protocol) is not MSSSProtocol or type(bank) is not ScaleFeatureBank:
        raise S1EvaluationError("issued MSSS protocol and feature bank are required")
    axis_order = ("sampling", "gaussian", "wavelet")
    evaluation_tuple = tuple(evaluations)
    specificity_tuple = tuple(specificity)
    if (
        tuple(item.axis for item in evaluation_tuple) != axis_order
        or tuple(item.axis for item in specificity_tuple) != axis_order
        or tuple(dict.fromkeys(bank.dataset_ids)) != protocol.domain_order
    ):
        raise S1EvaluationError("S1 axis or domain order changed")
    specimen_ids = bank.specimen_ids
    groups = np.asarray(bank.dataset_ids, dtype=str)
    errors_by_axis: dict[str, dict[str, np.ndarray]] = {}
    selected_errors: dict[str, np.ndarray] = {}
    curve_effects: dict[str, np.ndarray] = {}
    for evaluation in evaluation_tuple:
        errors = _prediction_errors(evaluation, specimen_ids=specimen_ids)
        errors_by_axis[evaluation.axis] = errors
        selected_errors[evaluation.axis] = _selected_errors(
            evaluation, specimen_ids=specimen_ids
        )
        for condition_id, values in errors.items():
            curve_effects[f"curve|{condition_id}"] = values
        curve_effects[f"selected|{evaluation.axis}"] = selected_errors[evaluation.axis]
    resamples = protocol.bootstrap_resamples if bootstrap_resamples is None else bootstrap_resamples
    curve_bootstrap = common_stratified_bootstrap(
        curve_effects,
        groups=bank.dataset_ids,
        group_order=protocol.domain_order,
        seed=protocol.bootstrap_seed,
        resamples=resamples,
        quantiles=(0.025, 0.975),
    )

    domain_metrics: list[DomainScaleMetric] = []
    curves: list[ScaleCurveMetric] = []
    curve_by_condition: dict[str, ScaleCurveMetric] = {}
    for evaluation in evaluation_tuple:
        axis_conditions = tuple(
            item for item in bank.conditions if item.axis == evaluation.axis
        )
        primary = tuple(item for item in axis_conditions if item.primary_eligible)
        full = tuple(item for item in primary if item.is_full_identity)
        if len(full) != 1:
            raise S1EvaluationError("axis FULL condition is invalid")
        full_errors = errors_by_axis[evaluation.axis][full[0].condition_id]
        full_domain_values = tuple(
            float(np.mean(full_errors[groups == group])) for group in protocol.domain_order
        )
        full_mae = _mean(full_domain_values)
        maximum_rank = max(item.coarse_rank for item in primary)
        for condition in axis_conditions:
            values = errors_by_axis[evaluation.axis][condition.condition_id]
            domain_values = []
            for group in protocol.domain_order:
                selected = values[groups == group]
                mae = float(np.mean(selected))
                domain_values.append(mae)
                domain_metrics.append(
                    DomainScaleMetric(
                        axis=evaluation.axis,
                        condition_id=condition.condition_id,
                        dataset_id=group,
                        specimen_count=len(selected),
                        mae=mae,
                    )
                )
            equal_mae = _mean(domain_values)
            interval = curve_bootstrap.effects[f"curve|{condition.condition_id}"]
            metric = ScaleCurveMetric(
                axis=evaluation.axis,
                condition_id=condition.condition_id,
                value=condition.value,
                coarse_rank=condition.coarse_rank,
                primary_eligible=condition.primary_eligible,
                wavelet=condition.wavelet,
                level=condition.level,
                mode=condition.mode,
                normalized_retention_index=(
                    1.0
                    if maximum_rank == 0
                    else 1.0 - condition.coarse_rank / maximum_rank
                ),
                equal_domain_mae=equal_mae,
                ci_low=interval.low,
                ci_high=interval.high,
                full_equal_domain_mae=full_mae,
                relative_gap=(equal_mae - full_mae) / full_mae,
                noninferior_025=equal_mae <= full_mae * 1.025,
                noninferior_05=equal_mae <= full_mae * 1.05,
                noninferior_075=equal_mae <= full_mae * 1.075,
            )
            curves.append(metric)
            curve_by_condition[condition.condition_id] = metric

    specificity_effects = {
        item.axis: _specificity_effects(item, specimen_ids=specimen_ids)
        for item in specificity_tuple
    }
    specificity_bootstrap = common_stratified_bootstrap(
        specificity_effects,
        groups=bank.dataset_ids,
        group_order=protocol.domain_order,
        seed=protocol.bootstrap_seed,
        resamples=resamples,
        quantiles=(0.025, 0.975),
    )
    specificity_simultaneous = common_stratified_bootstrap(
        specificity_effects,
        groups=bank.dataset_ids,
        group_order=protocol.domain_order,
        seed=protocol.bootstrap_seed,
        resamples=resamples,
        quantiles=(0.008333333333333333, 0.9916666666666667),
    )

    specificity_by_axis = {item.axis: item for item in specificity_tuple}
    axis_summaries: list[AxisSummary] = []
    for evaluation in evaluation_tuple:
        primary = tuple(
            item
            for item in bank.conditions
            if item.axis == evaluation.axis and item.primary_eligible
        )
        primary_curves = tuple(curve_by_condition[item.condition_id] for item in primary)
        noninferiority = select_noninferior(
            tuple(item.value for item in primary),
            tuple(item.equal_domain_mae for item in primary_curves),
            margin=protocol.primary_margin,
        )
        selected_primary_index = tuple(item.value for item in primary).index(
            noninferiority.selected
        )
        global_selected = primary[selected_primary_index]
        global_over = (
            None
            if noninferiority.over_coarse is None
            else primary[
                tuple(item.value for item in primary).index(noninferiority.over_coarse)
            ].condition_id
        )
        selected_interval = curve_bootstrap.effects[f"selected|{evaluation.axis}"]
        selected_mae = selected_interval.estimate
        full_condition = next(item for item in primary if item.is_full_identity)
        full_mae = curve_by_condition[full_condition.condition_id].equal_domain_mae
        stable = selection_stability(
            tuple(item.selected_condition_id for item in evaluation.scale_selections),
            candidate_order=tuple(item.condition_id for item in primary),
            minimum_outer_folds=4,
            window_steps=1,
        )
        spatial = specificity_by_axis[evaluation.axis]
        gate = axis_gate(
            plateau=noninferiority.plateau,
            boundary_confirmed=noninferiority.boundary_confirmed,
            stable=stable.passed,
            mechanically_sufficient=selected_mae <= full_mae * (1.0 + protocol.primary_margin),
            spatially_specific=spatial.result.status == "PASS",
        )
        axis_summaries.append(
            AxisSummary(
                axis=evaluation.axis,
                full_condition_id=full_condition.condition_id,
                global_selected_condition_id=global_selected.condition_id,
                global_over_coarse_condition_id=global_over,
                global_noninferiority=noninferiority,
                selected_equal_domain_mae=selected_mae,
                selected_ci_low=selected_interval.low,
                selected_ci_high=selected_interval.high,
                full_equal_domain_mae=full_mae,
                mechanically_sufficient=gate.mechanically_sufficient,
                stability=stable,
                specificity=spatial.result,
                specificity_interval=specificity_bootstrap.effects[evaluation.axis],
                specificity_simultaneous_interval=specificity_simultaneous.effects[evaluation.axis],
                gate=gate,
            )
        )
    final_gate = s1_gate(tuple(item.gate for item in axis_summaries))
    state_payload = {
        "evaluations": [item.state_sha256 for item in evaluation_tuple],
        "specificity": [item.state_sha256 for item in specificity_tuple],
        "feature_bank": bank.state_sha256,
        "axis_gates": [item.gate.status for item in axis_summaries],
        "s1_gate": final_gate.status,
        "curve_draws": curve_bootstrap.draws_sha256,
        "specificity_draws": specificity_bootstrap.draws_sha256,
    }
    state = hashlib.sha256(
        json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return S1Run(
        evaluations=evaluation_tuple,
        specificity_evaluations=specificity_tuple,
        curves=tuple(curves),
        domain_metrics=tuple(domain_metrics),
        axis_summaries=tuple(axis_summaries),
        curve_bootstrap=curve_bootstrap,
        specificity_bootstrap=specificity_bootstrap,
        specificity_simultaneous_bootstrap=specificity_simultaneous,
        gate=final_gate,
        feature_bank_state_sha256=bank.state_sha256,
        state_sha256=state,
    )


__all__ = [
    "AxisSummary",
    "DomainScaleMetric",
    "S1EvaluationError",
    "S1Run",
    "ScaleCurveMetric",
    "summarize_s1",
]
