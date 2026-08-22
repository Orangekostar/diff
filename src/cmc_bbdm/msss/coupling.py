"""Post-hoc configuration-dependent scale diagnostics after an S1 NO_GO."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

from .scale_features import ScaleCondition


class CouplingError(ValueError):
    """Raised when a coupling diagnostic loses its frozen roster or semantics."""


@dataclass(frozen=True, slots=True)
class GroupCurveRow:
    axis: str
    grouping: str
    group_value: str
    condition_id: str
    coarse_rank: int
    specimen_count: int
    domain_count: int
    equal_domain_mae: float
    full_equal_domain_mae: float
    relative_gap: float
    noninferior_05: bool


@dataclass(frozen=True, slots=True)
class GroupScaleSelection:
    axis: str
    grouping: str
    group_value: str
    selected_condition_id: str
    selected_coarse_rank: int
    full_condition_id: str
    over_coarse_condition_id: str | None
    boundary_confirmed: bool
    sufficient_condition_ids: tuple[str, ...]
    specimen_count: int
    domain_count: int


@dataclass(frozen=True, slots=True)
class FactorAlignment:
    status: str
    direction: str | None
    axis_count: int


@dataclass(frozen=True, slots=True)
class DamageSizeBin:
    specimen_id: str
    dataset_id: str
    metric: str
    value: float
    tertile: str


@dataclass(frozen=True, slots=True)
class FactorTrend:
    axis: str
    factor: str
    group_order: tuple[str, ...]
    selected_condition_ids: tuple[str, ...]
    coarse_ranks: tuple[int, ...]
    direction: str


@dataclass(frozen=True, slots=True)
class FactorAlignmentRow:
    factor: str
    status: str
    direction: str | None
    axis_count: int


@dataclass(frozen=True, slots=True)
class CouplingDiagnostic:
    curves: tuple[GroupCurveRow, ...]
    selections: tuple[GroupScaleSelection, ...]
    damage_bins: tuple[DamageSizeBin, ...]
    trends: tuple[FactorTrend, ...]
    alignments: tuple[FactorAlignmentRow, ...]
    coupling_status: str
    validation_status: str
    s2_status: str
    state_sha256: str


def _identities(values: Sequence[str], label: str, *, unique: bool) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or any(not isinstance(value, str) or not value for value in result)
        or (unique and len(set(result)) != len(result))
    ):
        raise CouplingError(f"{label} are invalid")
    return result


def stable_rank_tertiles(
    values: object, *, specimen_ids: Sequence[str]
) -> tuple[str, ...]:
    """Assign stable rank-balanced low/middle/high labels."""

    specimens = _identities(specimen_ids, "specimen IDs", unique=True)
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CouplingError("damage values must be numeric") from error
    if (
        len(specimens) < 3
        or array.shape != (len(specimens),)
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
    ):
        raise CouplingError("damage values are incomplete")
    order = sorted(
        range(len(specimens)), key=lambda index: (float(array[index]), specimens[index])
    )
    names = ("low", "middle", "high")
    labels = [""] * len(specimens)
    for position, index in enumerate(order):
        labels[index] = names[min(2, position * 3 // len(specimens))]
    if any(not labels.count(name) for name in names):
        raise CouplingError("damage tertiles are empty")
    return tuple(labels)


def _condition_registry(
    conditions: Sequence[ScaleCondition],
) -> tuple[ScaleCondition, ...]:
    registry = tuple(conditions)
    if (
        len(registry) < 2
        or any(type(item) is not ScaleCondition for item in registry)
        or len({item.condition_id for item in registry}) != len(registry)
        or len({item.axis for item in registry}) != 1
        or any(not item.primary_eligible for item in registry)
        or tuple(item.coarse_rank for item in registry) != tuple(range(len(registry)))
        or sum(item.is_full_identity for item in registry) != 1
        or not registry[0].is_full_identity
    ):
        raise CouplingError("candidate registry is invalid")
    return registry


def evaluate_group_curve(
    conditions: Sequence[ScaleCondition],
    *,
    absolute_errors: Mapping[str, object],
    dataset_ids: Sequence[str],
    selected_indices: Sequence[int],
    grouping: str,
    group_value: str,
    margin: float,
) -> tuple[tuple[GroupCurveRow, ...], GroupScaleSelection]:
    """Compute one target-diagnostic curve using equal-domain aggregation."""

    registry = _condition_registry(conditions)
    datasets = _identities(dataset_ids, "dataset IDs", unique=False)
    if not isinstance(absolute_errors, Mapping) or set(absolute_errors) != {
        item.condition_id for item in registry
    }:
        raise CouplingError("candidate error roster is incomplete")
    indices = tuple(selected_indices)
    if (
        not indices
        or any(type(index) is not int for index in indices)
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= len(datasets)
        or type(grouping) is not str
        or not grouping
        or type(group_value) is not str
        or not group_value
        or isinstance(margin, bool)
        or not math.isfinite(float(margin))
        or float(margin) <= 0.0
    ):
        raise CouplingError("group curve registry is invalid")
    arrays: dict[str, np.ndarray] = {}
    for condition in registry:
        try:
            array = np.asarray(
                absolute_errors[condition.condition_id], dtype=np.float64
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise CouplingError("candidate errors must be numeric") from error
        if (
            array.shape != (len(datasets),)
            or not np.all(np.isfinite(array))
            or np.any(array < 0.0)
        ):
            raise CouplingError("candidate error roster is incomplete")
        arrays[condition.condition_id] = np.ascontiguousarray(array)
    selected = np.asarray(indices, dtype=np.int64)
    domain_order = tuple(dict.fromkeys(datasets[index] for index in indices))
    scores: list[float] = []
    for condition in registry:
        array = arrays[condition.condition_id]
        domain_scores = []
        for domain in domain_order:
            domain_indices = selected[
                np.asarray([datasets[int(index)] == domain for index in selected])
            ]
            if not len(domain_indices):
                raise CouplingError("diagnostic group has an empty domain")
            domain_scores.append(float(np.mean(array[domain_indices])))
        scores.append(math.fsum(domain_scores) / len(domain_scores))
    full_score = scores[0]
    if not math.isfinite(full_score) or full_score <= 0.0:
        raise CouplingError("FULL diagnostic score must be positive")
    threshold = full_score * (1.0 + float(margin))
    eligible = tuple(index for index, score in enumerate(scores) if score <= threshold)
    if 0 not in eligible:
        raise CouplingError("FULL is absent from the diagnostic sufficient set")
    selected_index = eligible[-1]
    over_index = next(
        (index for index in range(selected_index + 1, len(registry)) if index not in eligible),
        None,
    )
    rows = tuple(
        GroupCurveRow(
            axis=condition.axis,
            grouping=grouping,
            group_value=group_value,
            condition_id=condition.condition_id,
            coarse_rank=condition.coarse_rank,
            specimen_count=len(indices),
            domain_count=len(domain_order),
            equal_domain_mae=score,
            full_equal_domain_mae=full_score,
            relative_gap=score / full_score - 1.0,
            noninferior_05=index in eligible,
        )
        for index, (condition, score) in enumerate(zip(registry, scores, strict=True))
    )
    decision = GroupScaleSelection(
        axis=registry[0].axis,
        grouping=grouping,
        group_value=group_value,
        selected_condition_id=registry[selected_index].condition_id,
        selected_coarse_rank=registry[selected_index].coarse_rank,
        full_condition_id=registry[0].condition_id,
        over_coarse_condition_id=(
            None if over_index is None else registry[over_index].condition_id
        ),
        boundary_confirmed=over_index is not None,
        sufficient_condition_ids=tuple(registry[index].condition_id for index in eligible),
        specimen_count=len(indices),
        domain_count=len(domain_order),
    )
    return rows, decision


def classify_ordered_direction(coarse_ranks: Sequence[int]) -> str:
    """Classify an ordered coarse-rank sequence without statistical promotion."""

    values = tuple(coarse_ranks)
    if len(values) < 2 or any(type(value) is not int or value < 0 for value in values):
        raise CouplingError("coarse-rank sequence is invalid")
    differences = tuple(right - left for left, right in pairwise(values))
    if all(value == 0 for value in differences):
        return "SAME"
    if all(value >= 0 for value in differences):
        return "COARSER"
    if all(value <= 0 for value in differences):
        return "FINER"
    return "NON_MONOTONIC"


def align_factor_directions(directions: Sequence[str]) -> FactorAlignment:
    """Require at least two axes with one matching non-neutral direction."""

    values = tuple(directions)
    allowed = {"COARSER", "FINER", "SAME", "NON_MONOTONIC"}
    if len(values) != 3 or any(value not in allowed for value in values):
        raise CouplingError("factor directions are invalid")
    counts = Counter(value for value in values if value in {"COARSER", "FINER"})
    if not counts:
        return FactorAlignment("NO_CROSS_AXIS_ALIGNMENT", None, 0)
    direction, count = counts.most_common(1)[0]
    if count >= 2:
        return FactorAlignment("CROSS_AXIS_ALIGNED", direction, count)
    return FactorAlignment("NO_CROSS_AXIS_ALIGNMENT", None, count)


def _error_mapping(
    absolute_errors: Mapping[str, object],
    *,
    conditions: Sequence[ScaleCondition],
    rows: int,
) -> dict[str, np.ndarray]:
    expected = {item.condition_id for item in conditions}
    if not isinstance(absolute_errors, Mapping) or set(absolute_errors) != expected:
        raise CouplingError("candidate error roster is incomplete")
    result: dict[str, np.ndarray] = {}
    for condition_id in expected:
        try:
            array = np.asarray(absolute_errors[condition_id], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise CouplingError("candidate errors must be numeric") from error
        if array.shape != (rows,) or not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise CouplingError("candidate error roster is incomplete")
        result[condition_id] = np.ascontiguousarray(array)
    return result


def diagnose_coupling(
    *,
    conditions: Sequence[ScaleCondition],
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    ply_count: Sequence[int],
    layup_family: Sequence[str],
    damage_sizes: Mapping[str, object],
    absolute_errors: Mapping[str, object],
    margin: float,
) -> CouplingDiagnostic:
    """Run the frozen post-hoc grouping diagnostic on cross-fitted errors."""

    registry = tuple(conditions)
    specimens = _identities(specimen_ids, "specimen IDs", unique=True)
    datasets = _identities(dataset_ids, "dataset IDs", unique=False)
    if len(datasets) != len(specimens):
        raise CouplingError("specimen and dataset rosters differ")
    axes = ("sampling", "gaussian", "wavelet")
    by_axis: dict[str, tuple[ScaleCondition, ...]] = {}
    for axis in axes:
        axis_conditions = tuple(
            item for item in registry if item.axis == axis and item.primary_eligible
        )
        by_axis[axis] = _condition_registry(axis_conditions)
    if (
        len(registry) != sum(len(value) for value in by_axis.values())
        or len({item.condition_id for item in registry}) != len(registry)
    ):
        raise CouplingError("diagnostic requires only the three primary axes")
    try:
        ply = tuple(int(value) for value in ply_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise CouplingError("ply-count authority is invalid") from error
    layup = _identities(layup_family, "layup authority", unique=False)
    if (
        len(ply) != len(specimens)
        or len(layup) != len(specimens)
        or set(ply) != {8, 16, 24}
        or set(layup) != {"cross_ply", "quasi_isotropic"}
        or any(isinstance(value, bool) for value in ply_count)
    ):
        raise CouplingError("structural authority is invalid")
    damage_names = ("damage_area", "damage_height", "damage_width")
    if not isinstance(damage_sizes, Mapping) or tuple(damage_sizes) != damage_names:
        raise CouplingError("damage-size registry is invalid")
    damage_arrays: dict[str, np.ndarray] = {}
    damage_labels: dict[str, tuple[str, ...]] = {}
    damage_bins: list[DamageSizeBin] = []
    for name in damage_names:
        try:
            values = np.asarray(damage_sizes[name], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise CouplingError("damage-size values must be numeric") from error
        if (
            values.shape != (len(specimens),)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise CouplingError("damage-size values are incomplete")
        labels = stable_rank_tertiles(values, specimen_ids=specimens)
        damage_arrays[name] = np.ascontiguousarray(values)
        damage_labels[name] = labels
        damage_bins.extend(
            DamageSizeBin(
                specimen_id=specimen,
                dataset_id=dataset,
                metric=name,
                value=float(value),
                tertile=label,
            )
            for specimen, dataset, value, label in zip(
                specimens, datasets, values, labels, strict=True
            )
        )
    errors = _error_mapping(
        absolute_errors, conditions=registry, rows=len(specimens)
    )
    domain_order = tuple(dict.fromkeys(datasets))
    grouping_registry = (
        ("domain", datasets, domain_order),
        ("ply_count", tuple(str(value) for value in ply), ("8", "16", "24")),
        (
            "layup_family",
            layup,
            ("cross_ply", "quasi_isotropic"),
        ),
        *(
            (name, damage_labels[name], ("low", "middle", "high"))
            for name in damage_names
        ),
    )
    curves: list[GroupCurveRow] = []
    selections: list[GroupScaleSelection] = []
    for axis in axes:
        axis_errors = {
            condition.condition_id: errors[condition.condition_id]
            for condition in by_axis[axis]
        }
        for grouping, labels, group_order in grouping_registry:
            for group_value in group_order:
                selected_indices = tuple(
                    index for index, label in enumerate(labels) if label == group_value
                )
                rows, decision = evaluate_group_curve(
                    by_axis[axis],
                    absolute_errors=axis_errors,
                    dataset_ids=datasets,
                    selected_indices=selected_indices,
                    grouping=grouping,
                    group_value=group_value,
                    margin=margin,
                )
                curves.extend(rows)
                selections.append(decision)
    selection_index = {
        (item.axis, item.grouping, item.group_value): item for item in selections
    }
    factor_order = ("ply_count", "layup_family", *damage_names)
    group_orders = {
        grouping: group_order
        for grouping, _labels, group_order in grouping_registry
        if grouping != "domain"
    }
    trends: list[FactorTrend] = []
    alignments: list[FactorAlignmentRow] = []
    for factor in factor_order:
        directions: list[str] = []
        for axis in axes:
            selected = tuple(
                selection_index[(axis, factor, group)]
                for group in group_orders[factor]
            )
            ranks = tuple(item.selected_coarse_rank for item in selected)
            direction = classify_ordered_direction(ranks)
            directions.append(direction)
            trends.append(
                FactorTrend(
                    axis=axis,
                    factor=factor,
                    group_order=group_orders[factor],
                    selected_condition_ids=tuple(
                        item.selected_condition_id for item in selected
                    ),
                    coarse_ranks=ranks,
                    direction=direction,
                )
            )
        alignment = align_factor_directions(directions)
        alignments.append(
            FactorAlignmentRow(
                factor=factor,
                status=alignment.status,
                direction=alignment.direction,
                axis_count=alignment.axis_count,
            )
        )
    coupling_status = (
        "EXPLORATORY_SIGNAL"
        if any(item.status == "CROSS_AXIS_ALIGNED" for item in alignments)
        else "NO_CONSISTENT_SIGNAL"
    )
    payload = {
        "curves": [asdict(item) for item in curves],
        "selections": [asdict(item) for item in selections],
        "damage_bins": [asdict(item) for item in damage_bins],
        "trends": [asdict(item) for item in trends],
        "alignments": [asdict(item) for item in alignments],
        "coupling_status": coupling_status,
        "validation_status": "NOT_VALIDATED_POST_HOC",
        "s2_status": "NOT_RUN_NOT_AUTHORIZED",
    }
    state = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()
    return CouplingDiagnostic(
        curves=tuple(curves),
        selections=tuple(selections),
        damage_bins=tuple(damage_bins),
        trends=tuple(trends),
        alignments=tuple(alignments),
        coupling_status=coupling_status,
        validation_status="NOT_VALIDATED_POST_HOC",
        s2_status="NOT_RUN_NOT_AUTHORIZED",
        state_sha256=state,
    )


__all__ = [
    "CouplingDiagnostic",
    "CouplingError",
    "DamageSizeBin",
    "FactorAlignment",
    "FactorAlignmentRow",
    "FactorTrend",
    "GroupCurveRow",
    "GroupScaleSelection",
    "align_factor_directions",
    "classify_ordered_direction",
    "diagnose_coupling",
    "evaluate_group_curve",
    "stable_rank_tertiles",
]
