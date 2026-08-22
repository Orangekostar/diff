"""Source-only sampling-scale selection for S2 transfer tasks."""

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

from .scale_features import ScaleFeatureBank
from .source_only_selection import select_source_scale


class TransferTaskError(ValueError):
    """Raised when an S2 task, source selection, or prediction is invalid."""


@dataclass(frozen=True, slots=True)
class TransferTask:
    family: str
    task_id: str
    target_label: str
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    source_domains: tuple[str, ...]
    target_domains: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransferFitEvent:
    stage: str
    task_id: str
    condition_id: str
    inner_group: str | None
    pca_dimension: int
    fit_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    target_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransferSelection:
    task_id: str
    full_condition_id: str
    fixed25_condition_id: str
    selected_condition_id: str
    over_coarse_condition_id: str
    boundary_confirmed: bool
    sufficient_sets: tuple[tuple[float, tuple[str, ...]], ...]
    candidate_scores: tuple[tuple[str, float], ...]
    candidate_pca_dimensions: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class TransferPrediction:
    task_family: str
    task_id: str
    target_label: str
    comparator: str
    condition_id: str
    specimen_id: str
    dataset_id: str
    target: float
    prediction: float
    absolute_error: float
    selected_pca_dimension: int
    fit_state_sha256: str


@dataclass(frozen=True, slots=True)
class TransferTaskEvaluation:
    task: TransferTask
    selection: TransferSelection
    predictions: tuple[TransferPrediction, ...]
    state_sha256: str


TransferFitHook = Callable[[TransferFitEvent], None]


def _identities(value: Sequence[object], label: str) -> tuple[str, ...]:
    result = tuple(value)
    if not result or any(type(item) is not str or not item for item in result):
        raise TransferTaskError(f"{label} are invalid")
    return result  # type: ignore[return-value]


def _issue_task(
    *,
    family: str,
    task_id: str,
    target_label: str,
    target_mask: np.ndarray,
    datasets: tuple[str, ...],
    group_order: tuple[str, ...],
) -> TransferTask:
    target = np.flatnonzero(target_mask)
    source = np.flatnonzero(~target_mask)
    source_domains = tuple(group for group in group_order if any(datasets[int(index)] == group for index in source))
    target_domains = tuple(group for group in group_order if any(datasets[int(index)] == group for index in target))
    if not len(target) or not len(source) or len(source_domains) < 3:
        raise TransferTaskError(f"transfer task is underdetermined: {task_id}")
    return TransferTask(
        family=family,
        task_id=task_id,
        target_label=target_label,
        source_indices=tuple(int(index) for index in source),
        target_indices=tuple(int(index) for index in target),
        source_domains=source_domains,
        target_domains=target_domains,
    )


def build_task_registry(
    *,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    ply_count: Sequence[int],
    layup_family: Sequence[str],
    domain_order: Sequence[str],
) -> tuple[TransferTask, ...]:
    """Build the frozen six-domain, three-ply, and two-layup task order."""

    specimens = _identities(specimen_ids, "specimen IDs")
    datasets = _identities(dataset_ids, "dataset IDs")
    groups = _identities(domain_order, "domain order")
    if (
        len(set(specimens)) != len(specimens)
        or len(datasets) != len(specimens)
        or tuple(dict.fromkeys(datasets)) != groups
        or len(set(groups)) != len(groups)
    ):
        raise TransferTaskError("transfer cohort identity changed")
    try:
        ply = np.asarray(tuple(ply_count), dtype=np.int64)
        layup = np.asarray(tuple(layup_family), dtype=str)
    except (TypeError, ValueError, OverflowError) as error:
        raise TransferTaskError("structural task labels are invalid") from error
    if (
        ply.shape != (len(specimens),)
        or layup.shape != (len(specimens),)
        or set(ply.tolist()) != {8, 16, 24}
        or set(layup.tolist()) != {"cross_ply", "quasi_isotropic"}
    ):
        raise TransferTaskError("structural task registry changed")
    dataset_array = np.asarray(datasets, dtype=str)
    tasks: list[TransferTask] = []
    for group in groups:
        tasks.append(
            _issue_task(
                family="domain",
                task_id=f"domain:{group}",
                target_label=group,
                target_mask=dataset_array == group,
                datasets=datasets,
                group_order=groups,
            )
        )
    for value in (8, 16, 24):
        tasks.append(
            _issue_task(
                family="ply",
                task_id=f"ply:{value}",
                target_label=str(value),
                target_mask=ply == value,
                datasets=datasets,
                group_order=groups,
            )
        )
    for value in ("cross_ply", "quasi_isotropic"):
        tasks.append(
            _issue_task(
                family="layup",
                task_id=f"layup:{value}",
                target_label=value,
                target_mask=layup == value,
                datasets=datasets,
                group_order=groups,
            )
        )
    return tuple(tasks)


def _numeric(value: object, shape: tuple[int, ...], label: str, *, allow_nan: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TransferTaskError(f"{label} must be numeric") from error
    valid = ~np.isinf(array) if allow_nan else np.isfinite(array)
    if array.shape != shape or not np.all(valid):
        raise TransferTaskError(f"{label} shape or values are invalid")
    return np.ascontiguousarray(array)


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    values = np.abs(target - prediction)
    if not len(values) or not np.all(np.isfinite(values)):
        raise TransferTaskError("transfer MAE is invalid")
    return float(math.fsum(float(value) for value in values) / len(values))


def _equal_domain_mae(target: np.ndarray, prediction: np.ndarray, groups: np.ndarray, order: tuple[str, ...]) -> float:
    return float(
        math.fsum(_mae(target[groups == group], prediction[groups == group]) for group in order)
        / len(order)
    )


def _event(
    hook: TransferFitHook | None,
    *,
    stage: str,
    task: TransferTask,
    condition_id: str,
    inner_group: str | None,
    dimension: int,
    fit: np.ndarray,
    query: np.ndarray,
    specimen_ids: tuple[str, ...],
) -> None:
    if hook is None:
        return
    hook(
        TransferFitEvent(
            stage=stage,
            task_id=task.task_id,
            condition_id=condition_id,
            inner_group=inner_group,
            pca_dimension=dimension,
            fit_ids=tuple(specimen_ids[int(index)] for index in fit),
            query_ids=tuple(specimen_ids[int(index)] for index in query),
            target_ids=tuple(specimen_ids[index] for index in task.target_indices),
        )
    )


def evaluate_transfer_task(
    bank: ScaleFeatureBank,
    *,
    targets: object,
    metadata13: object,
    task: TransferTask,
    pca_dimensions: Sequence[int] = (8, 16, 32),
    primary_margin: float = 0.05,
    margins: Sequence[float] = (0.025, 0.05, 0.075),
    fit_hook: TransferFitHook | None = None,
) -> TransferTaskEvaluation:
    """Source-select sampling scale/PCA and predict one untouched target group."""

    if type(bank) is not ScaleFeatureBank or type(task) is not TransferTask:
        raise TransferTaskError("issued sampling bank and transfer task are required")
    rows = len(bank.specimen_ids)
    response = _numeric(targets, (rows,), "targets")
    metadata = _numeric(metadata13, (rows, 13), "metadata13", allow_nan=True)
    groups = np.asarray(bank.dataset_ids, dtype=str)
    dimensions = tuple(pca_dimensions)
    if not dimensions or any(type(value) is not int or value < 1 for value in dimensions):
        raise TransferTaskError("transfer PCA dimensions are invalid")
    candidates = tuple(item for item in bank.conditions if item.axis == "sampling" and item.primary_eligible)
    full_matches = tuple(item for item in candidates if item.is_full_identity)
    fixed_matches = tuple(item for item in candidates if math.isclose(item.value, 0.25, rel_tol=0.0, abs_tol=1.0e-15))
    if len(full_matches) != 1 or len(fixed_matches) != 1 or len(candidates) < 3:
        raise TransferTaskError("sampling comparator registry is incomplete")
    source = np.asarray(task.source_indices, dtype=np.int64)
    query = np.asarray(task.target_indices, dtype=np.int64)
    if (
        len(source) <= max(dimensions)
        or not len(query)
        or set(source.tolist()) & set(query.tolist())
        or set(source.tolist()) | set(query.tolist()) != set(range(rows))
    ):
        raise TransferTaskError("transfer task roster is invalid")
    score_by_condition: dict[str, float] = {}
    dimension_by_condition: dict[str, int] = {}
    for condition in candidates:
        embeddings = np.asarray(bank.features[condition.condition_id], dtype=np.float64)
        source_positions = {int(index): position for position, index in enumerate(source)}
        oof = {dimension: np.full(len(source), np.nan) for dimension in dimensions}
        for inner_group in task.source_domains:
            inner_fit = source[groups[source] != inner_group]
            inner_query = source[groups[source] == inner_group]
            if len(inner_fit) <= max(dimensions) or not len(inner_query):
                raise TransferTaskError("source-domain inner fold is underdetermined")
            basis = fit_pca_basis(embeddings, inner_fit, maximum_dimension=max(dimensions))
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
                oof[dimension][positions] = expert.predict(embeddings[inner_query], metadata[inner_query])
                _event(
                    fit_hook,
                    stage="inner",
                    task=task,
                    condition_id=condition.condition_id,
                    inner_group=inner_group,
                    dimension=dimension,
                    fit=inner_fit,
                    query=inner_query,
                    specimen_ids=bank.specimen_ids,
                )
        dimension_scores = tuple(
            (
                dimension,
                _equal_domain_mae(response[source], values, groups[source], task.source_domains),
            )
            for dimension, values in oof.items()
        )
        if any(not np.all(np.isfinite(values)) for values in oof.values()):
            raise TransferTaskError("source OOF prediction roster is incomplete")
        selected_dimension, selected_score = dimension_scores[0]
        for dimension, score in dimension_scores[1:]:
            if score < selected_score - 1.0e-12:
                selected_dimension, selected_score = dimension, score
        dimension_by_condition[condition.condition_id] = selected_dimension
        score_by_condition[condition.condition_id] = selected_score
    decision = select_source_scale(
        candidates,
        score_by_condition,
        margins=margins,
        primary_margin=primary_margin,
    )
    over = decision.over_coarse_condition_id or candidates[-1].condition_id
    selection = TransferSelection(
        task_id=task.task_id,
        full_condition_id=decision.full_condition_id,
        fixed25_condition_id=fixed_matches[0].condition_id,
        selected_condition_id=decision.selected_condition_id,
        over_coarse_condition_id=over,
        boundary_confirmed=decision.boundary_confirmed,
        sufficient_sets=decision.sufficient_sets,
        candidate_scores=decision.candidate_scores,
        candidate_pca_dimensions=tuple(
            (condition.condition_id, dimension_by_condition[condition.condition_id])
            for condition in candidates
        ),
    )
    comparator_conditions = (
        ("FULL", selection.full_condition_id),
        ("FIXED_25", selection.fixed25_condition_id),
        ("SOURCE_MSSS", selection.selected_condition_id),
        ("OVER_COARSE", selection.over_coarse_condition_id),
    )
    fitted: dict[str, tuple[np.ndarray, int, str]] = {}
    for condition_id in dict(comparator_conditions).values():
        if condition_id in fitted:
            continue
        embeddings = np.asarray(bank.features[condition_id], dtype=np.float64)
        dimension = dimension_by_condition[condition_id]
        basis = fit_pca_basis(embeddings, source, maximum_dimension=dimension)
        expert = fit_view_expert(
            embeddings,
            metadata,
            response,
            source,
            pca_dimension=dimension,
            alpha=10.0,
            pca_basis=basis,
        )
        fitted[condition_id] = (
            expert.predict(embeddings[query], metadata[query]),
            dimension,
            expert.state_sha256,
        )
        _event(
            fit_hook,
            stage="outer",
            task=task,
            condition_id=condition_id,
            inner_group=None,
            dimension=dimension,
            fit=source,
            query=query,
            specimen_ids=bank.specimen_ids,
        )
    predictions: list[TransferPrediction] = []
    for comparator, condition_id in comparator_conditions:
        values, dimension, state = fitted[condition_id]
        predictions.extend(
            TransferPrediction(
                task_family=task.family,
                task_id=task.task_id,
                target_label=task.target_label,
                comparator=comparator,
                condition_id=condition_id,
                specimen_id=bank.specimen_ids[int(index)],
                dataset_id=bank.dataset_ids[int(index)],
                target=float(response[int(index)]),
                prediction=float(values[position]),
                absolute_error=float(abs(response[int(index)] - values[position])),
                selected_pca_dimension=dimension,
                fit_state_sha256=state,
            )
            for position, index in enumerate(query)
        )
    state = hashlib.sha256(
        json.dumps(
            {
                "task": task.task_id,
                "selection": (
                    selection.selected_condition_id,
                    selection.over_coarse_condition_id,
                    selection.candidate_pca_dimensions,
                ),
                "predictions": [
                    (item.comparator, item.specimen_id, item.prediction) for item in predictions
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return TransferTaskEvaluation(
        task=task,
        selection=selection,
        predictions=tuple(predictions),
        state_sha256=state,
    )


__all__ = [
    "TransferFitEvent",
    "TransferPrediction",
    "TransferSelection",
    "TransferTask",
    "TransferTaskError",
    "TransferTaskEvaluation",
    "build_task_registry",
    "evaluate_transfer_task",
]
