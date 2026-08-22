"""Nested source-only fixed-candidate and selected-scale evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.aei_multiview_regression.view_experts import (
    fit_pca_basis,
    fit_view_expert,
)

from .scale_features import ScaleCondition, ScaleFeatureBank
from .source_only_selection import SourceScaleDecision, select_source_scale


class ScaleEvaluationError(ValueError):
    """Raised when nested scale evaluation loses identity or isolation."""


@dataclass(frozen=True, slots=True)
class FitEvent:
    stage: str
    condition_id: str
    outer_group: str
    inner_group: str | None
    pca_dimension: int
    fit_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    fit_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InnerScore:
    outer_group: str
    inner_group: str
    condition_id: str
    pca_dimension: int
    mae: float
    fit_count: int
    query_count: int
    pca_state_sha256: str
    model_state_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    outer_group: str
    condition_id: str
    selected_pca_dimension: int
    source_equal_group_mae: float
    dimension_scores: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    condition_id: str
    specimen_id: str
    dataset_id: str
    outer_group: str
    target: float
    prediction: float
    absolute_error: float
    selected_pca_dimension: int
    fit_state_sha256: str


@dataclass(frozen=True, slots=True)
class ScaleSelection:
    outer_group: str
    selected_condition_id: str
    full_condition_id: str
    over_coarse_condition_id: str | None
    boundary_confirmed: bool
    sufficient_sets: tuple[tuple[float, tuple[str, ...]], ...]
    candidate_scores: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class SelectedPrediction:
    axis: str
    selected_condition_id: str
    specimen_id: str
    dataset_id: str
    outer_group: str
    target: float
    prediction: float
    absolute_error: float
    selected_pca_dimension: int
    fit_state_sha256: str


@dataclass(frozen=True, slots=True)
class AxisEvaluation:
    axis: str
    group_order: tuple[str, ...]
    inner_scores: tuple[InnerScore, ...]
    candidate_selections: tuple[CandidateSelection, ...]
    candidate_predictions: tuple[CandidatePrediction, ...]
    scale_selections: tuple[ScaleSelection, ...]
    selected_predictions: tuple[SelectedPrediction, ...]
    state_sha256: str


FitHook = Callable[[FitEvent], None]


def _vector(value: object, rows: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ScaleEvaluationError(f"{label} must be numeric") from error
    if array.shape != (rows,) or not np.all(np.isfinite(array)):
        raise ScaleEvaluationError(f"{label} must be a finite vector")
    return np.ascontiguousarray(array)


def _metadata(value: object, rows: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ScaleEvaluationError("metadata must be numeric") from error
    if array.shape != (rows, 13) or np.any(np.isinf(array)):
        raise ScaleEvaluationError("metadata must have shape (n, 13)")
    return np.ascontiguousarray(array)


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.shape != prediction.shape or target.ndim != 1 or not target.size:
        raise ScaleEvaluationError("MAE inputs are incomplete")
    errors = np.abs(target - prediction)
    if not np.all(np.isfinite(errors)):
        raise ScaleEvaluationError("MAE is non-finite")
    return float(math.fsum(float(value) for value in errors) / len(errors))


def _equal_group_mae(
    target: np.ndarray, prediction: np.ndarray, groups: np.ndarray
) -> float:
    order = tuple(dict.fromkeys(str(value) for value in groups.tolist()))
    if len(order) < 2:
        raise ScaleEvaluationError("source evaluation needs at least two groups")
    return float(
        math.fsum(
            _mae(target[groups == group], prediction[groups == group])
            for group in order
        )
        / len(order)
    )


def _event(
    hook: FitHook | None,
    *,
    stage: str,
    condition_id: str,
    outer_group: str,
    inner_group: str | None,
    dimension: int,
    fit: np.ndarray,
    query: np.ndarray,
    specimen_ids: tuple[str, ...],
    groups: np.ndarray,
) -> None:
    if hook is None:
        return
    hook(
        FitEvent(
            stage=stage,
            condition_id=condition_id,
            outer_group=outer_group,
            inner_group=inner_group,
            pca_dimension=dimension,
            fit_ids=tuple(specimen_ids[int(index)] for index in fit),
            query_ids=tuple(specimen_ids[int(index)] for index in query),
            fit_groups=tuple(dict.fromkeys(str(groups[int(index)]) for index in fit)),
        )
    )


def _conditions(bank: ScaleFeatureBank, axis: str) -> tuple[ScaleCondition, ...]:
    if type(axis) is not str or axis not in {"sampling", "gaussian", "wavelet"}:
        raise ScaleEvaluationError("axis is not registered")
    result = tuple(item for item in bank.conditions if item.axis == axis)
    primary = tuple(item for item in result if item.primary_eligible)
    if not result or len(tuple(item for item in primary if item.is_full_identity)) != 1:
        raise ScaleEvaluationError("axis primary registry is incomplete")
    return result


def _scale_selection(
    outer_group: str, decision: SourceScaleDecision
) -> ScaleSelection:
    return ScaleSelection(
        outer_group=outer_group,
        selected_condition_id=decision.selected_condition_id,
        full_condition_id=decision.full_condition_id,
        over_coarse_condition_id=decision.over_coarse_condition_id,
        boundary_confirmed=decision.boundary_confirmed,
        sufficient_sets=decision.sufficient_sets,
        candidate_scores=decision.candidate_scores,
    )


def evaluate_axis(
    bank: ScaleFeatureBank,
    *,
    targets: object,
    metadata13: object,
    axis: str,
    pca_dimensions: Sequence[int] = (8, 16, 32),
    primary_margin: float = 0.05,
    margins: Sequence[float] = (0.025, 0.05, 0.075),
    fit_hook: FitHook | None = None,
) -> AxisEvaluation:
    """Evaluate fixed candidates and source-selected scales for one axis."""

    if type(bank) is not ScaleFeatureBank:
        raise ScaleEvaluationError("issued ScaleFeatureBank is required")
    rows = len(bank.specimen_ids)
    response = _vector(targets, rows, "targets")
    metadata = _metadata(metadata13, rows)
    groups = np.asarray(bank.dataset_ids, dtype=str)
    group_order = tuple(dict.fromkeys(bank.dataset_ids))
    if len(group_order) < 3:
        raise ScaleEvaluationError("nested scale evaluation requires three groups")
    dimensions = tuple(pca_dimensions)
    if (
        not dimensions
        or any(type(value) is not int or value < 1 for value in dimensions)
        or len(set(dimensions)) != len(dimensions)
    ):
        raise ScaleEvaluationError("PCA dimensions are invalid")
    candidates = _conditions(bank, axis)
    selection_candidates = tuple(item for item in candidates if item.primary_eligible)
    indices = np.arange(rows, dtype=np.int64)
    inner_rows: list[InnerScore] = []
    candidate_selections: list[CandidateSelection] = []
    candidate_predictions: list[CandidatePrediction] = []
    scale_selections: list[ScaleSelection] = []
    selected_predictions: list[SelectedPrediction] = []

    for outer_group in group_order:
        source = indices[groups != outer_group]
        query = indices[groups == outer_group]
        source_groups = groups[source]
        inner_order = tuple(dict.fromkeys(str(value) for value in source_groups.tolist()))
        if len(inner_order) < 2 or len(query) == 0 or len(source) <= max(dimensions):
            raise ScaleEvaluationError("outer scale fold is underdetermined")
        outer_candidate_rows: dict[str, tuple[CandidatePrediction, ...]] = {}
        outer_candidate_selection: dict[str, CandidateSelection] = {}
        for condition in candidates:
            embeddings = np.asarray(bank.features[condition.condition_id], dtype=np.float64)
            source_positions = {
                int(index): position for position, index in enumerate(source)
            }
            source_oof_by_dimension = {
                dimension: np.full(len(source), np.nan, dtype=np.float64)
                for dimension in dimensions
            }
            dimension_inner_rows: dict[int, list[InnerScore]] = {
                dimension: [] for dimension in dimensions
            }
            for inner_group in inner_order:
                inner_fit = source[source_groups != inner_group]
                inner_query = source[source_groups == inner_group]
                if len(inner_fit) <= max(dimensions) or not len(inner_query):
                    raise ScaleEvaluationError("inner scale fold is underdetermined")
                basis = fit_pca_basis(
                    embeddings, inner_fit, maximum_dimension=max(dimensions)
                )
                positions = [source_positions[int(index)] for index in inner_query]
                for dimension in dimensions:
                    expert = fit_view_expert(
                        embeddings,
                        metadata,
                        response,
                        inner_fit,
                        pca_dimension=dimension,
                        alpha=10.0,
                        pca_basis=basis,
                    )
                    prediction = expert.predict(
                        embeddings[inner_query], metadata[inner_query]
                    )
                    source_oof_by_dimension[dimension][positions] = prediction
                    score = _mae(response[inner_query], prediction)
                    dimension_inner_rows[dimension].append(
                        InnerScore(
                            outer_group=outer_group,
                            inner_group=inner_group,
                            condition_id=condition.condition_id,
                            pca_dimension=dimension,
                            mae=score,
                            fit_count=len(inner_fit),
                            query_count=len(inner_query),
                            pca_state_sha256=basis.state_sha256,
                            model_state_sha256=expert.state_sha256,
                        )
                    )
                    _event(
                        fit_hook,
                        stage="inner",
                        condition_id=condition.condition_id,
                        outer_group=outer_group,
                        inner_group=inner_group,
                        dimension=dimension,
                        fit=inner_fit,
                        query=inner_query,
                        specimen_ids=bank.specimen_ids,
                        groups=groups,
                    )
            dimension_scores: list[tuple[int, float]] = []
            for dimension in dimensions:
                source_oof = source_oof_by_dimension[dimension]
                if not np.all(np.isfinite(source_oof)):
                    raise ScaleEvaluationError("source OOF predictions are incomplete")
                dimension_scores.append(
                    (
                        dimension,
                        _equal_group_mae(response[source], source_oof, source_groups),
                    )
                )
            selected_dimension, selected_score = dimension_scores[0]
            for dimension, score in dimension_scores[1:]:
                if score < selected_score - 1.0e-12:
                    selected_dimension, selected_score = dimension, score
            inner_rows.extend(dimension_inner_rows[selected_dimension])
            selection = CandidateSelection(
                outer_group=outer_group,
                condition_id=condition.condition_id,
                selected_pca_dimension=selected_dimension,
                source_equal_group_mae=selected_score,
                dimension_scores=tuple(dimension_scores),
            )
            candidate_selections.append(selection)
            outer_candidate_selection[condition.condition_id] = selection
            basis = fit_pca_basis(
                embeddings, source, maximum_dimension=selected_dimension
            )
            expert = fit_view_expert(
                embeddings,
                metadata,
                response,
                source,
                pca_dimension=selected_dimension,
                alpha=10.0,
                pca_basis=basis,
            )
            predictions = expert.predict(embeddings[query], metadata[query])
            _event(
                fit_hook,
                stage="outer",
                condition_id=condition.condition_id,
                outer_group=outer_group,
                inner_group=None,
                dimension=selected_dimension,
                fit=source,
                query=query,
                specimen_ids=bank.specimen_ids,
                groups=groups,
            )
            rows_for_condition = tuple(
                CandidatePrediction(
                    condition_id=condition.condition_id,
                    specimen_id=bank.specimen_ids[int(index)],
                    dataset_id=bank.dataset_ids[int(index)],
                    outer_group=outer_group,
                    target=float(response[int(index)]),
                    prediction=float(predictions[position]),
                    absolute_error=float(
                        abs(response[int(index)] - predictions[position])
                    ),
                    selected_pca_dimension=selected_dimension,
                    fit_state_sha256=expert.state_sha256,
                )
                for position, index in enumerate(query)
            )
            candidate_predictions.extend(rows_for_condition)
            outer_candidate_rows[condition.condition_id] = rows_for_condition

        decision = select_source_scale(
            selection_candidates,
            {
                condition_id: selection.source_equal_group_mae
                for condition_id, selection in outer_candidate_selection.items()
                if condition_id
                in {item.condition_id for item in selection_candidates}
            },
            margins=margins,
            primary_margin=primary_margin,
        )
        scale_selections.append(_scale_selection(outer_group, decision))
        selected_rows = outer_candidate_rows[decision.selected_condition_id]
        selected_predictions.extend(
            SelectedPrediction(
                axis=axis,
                selected_condition_id=row.condition_id,
                specimen_id=row.specimen_id,
                dataset_id=row.dataset_id,
                outer_group=row.outer_group,
                target=row.target,
                prediction=row.prediction,
                absolute_error=row.absolute_error,
                selected_pca_dimension=row.selected_pca_dimension,
                fit_state_sha256=row.fit_state_sha256,
            )
            for row in selected_rows
        )

    state_payload = {
        "axis": axis,
        "groups": group_order,
        "candidate_selections": [
            (
                row.outer_group,
                row.condition_id,
                row.selected_pca_dimension,
                row.source_equal_group_mae,
            )
            for row in candidate_selections
        ],
        "scale_selections": [
            (
                row.outer_group,
                row.selected_condition_id,
                row.over_coarse_condition_id,
                row.boundary_confirmed,
            )
            for row in scale_selections
        ],
        "predictions": [
            (row.condition_id, row.specimen_id, row.prediction)
            for row in candidate_predictions
        ],
    }
    state = hashlib.sha256(
        json.dumps(
            state_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return AxisEvaluation(
        axis=axis,
        group_order=group_order,
        inner_scores=tuple(inner_rows),
        candidate_selections=tuple(candidate_selections),
        candidate_predictions=tuple(candidate_predictions),
        scale_selections=tuple(scale_selections),
        selected_predictions=tuple(selected_predictions),
        state_sha256=state,
    )


__all__ = [
    "AxisEvaluation",
    "CandidatePrediction",
    "CandidateSelection",
    "FitEvent",
    "InnerScore",
    "ScaleEvaluationError",
    "ScaleSelection",
    "SelectedPrediction",
    "evaluate_axis",
]
