"""E1 individual accuracy, agreement, and complementarity diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ViewMetrics:
    view: str
    domain_mae: tuple[tuple[str, float], ...]
    equal_domain_mae: float
    worst_domain_mae: float
    domain_mae_sd: float
    rmse: float
    r2: float


@dataclass(frozen=True, slots=True)
class GroupedBestView:
    group_name: str
    group_value: str
    specimen_count: int
    counts: tuple[int, ...]
    frequencies: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AgreementAudit:
    view_names: tuple[str, ...]
    view_metrics: tuple[ViewMetrics, ...]
    prediction_correlations: np.ndarray
    residual_correlations: np.ndarray
    mean_absolute_disagreement: np.ndarray
    oracle_mae: float
    oracle_improvement_vs_full: float
    best_view_indices: np.ndarray
    best_view_counts: tuple[int, ...]
    grouped_best_view: tuple[GroupedBestView, ...]
    deployable_methods: tuple[str, ...]
    predictive_equivalence: bool
    complementarity_signal: bool
    gate_status: str


def _readonly(value: np.ndarray, *, dtype: str = "<f8") -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(order="C"), dtype=dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _validated(
    targets: object,
    predictions: object,
    domains: Sequence[str],
    view_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if np.iscomplexobj(np.asarray(targets)) or np.iscomplexobj(np.asarray(predictions)):
        raise ValueError("audit values must be real")
    y = np.asarray(targets, dtype=np.float64)
    values = np.asarray(predictions, dtype=np.float64)
    domain_ids = tuple(domains)
    names = tuple(view_names)
    if (
        y.ndim != 1
        or values.ndim != 2
        or values.shape != (len(y), len(names))
        or len(domain_ids) != len(y)
        or len(y) < 2
        or len(names) < 2
        or len(set(names)) != len(names)
        or any(type(item) is not str or not item for item in domain_ids + names)
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("audit inputs are invalid or misaligned")
    return y, values, domain_ids, names


def _metrics(
    view: str, targets: np.ndarray, predictions: np.ndarray, domains: tuple[str, ...]
) -> ViewMetrics:
    order = tuple(dict.fromkeys(domains))
    domain_array = np.asarray(domains)
    domain_mae = tuple(
        (
            domain,
            float(
                np.mean(
                    np.abs(
                        targets[domain_array == domain]
                        - predictions[domain_array == domain]
                    )
                )
            ),
        )
        for domain in order
    )
    values = np.asarray([item[1] for item in domain_mae], dtype=np.float64)
    residual = targets - predictions
    denominator = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0.0 else 0.0
    return ViewMetrics(
        view=view,
        domain_mae=domain_mae,
        equal_domain_mae=float(np.mean(values)),
        worst_domain_mae=float(np.max(values)),
        domain_mae_sd=float(np.std(values, ddof=0)),
        rmse=float(np.sqrt(np.mean(residual**2))),
        r2=r2,
    )


def _correlations(values: np.ndarray) -> np.ndarray:
    output = np.corrcoef(values, rowvar=False)
    if output.shape != (values.shape[1], values.shape[1]):
        raise ValueError("correlation matrix shape changed")
    return _readonly(np.asarray(output, dtype=np.float64))


def _group_rows(
    best: np.ndarray,
    group_name: str,
    raw_values: Sequence[object],
    view_count: int,
) -> tuple[GroupedBestView, ...]:
    values = tuple(str(item) for item in raw_values)
    if len(values) != len(best) or any(not item for item in values):
        raise ValueError(f"group {group_name} is invalid")
    result: list[GroupedBestView] = []
    array = np.asarray(values)
    for value in dict.fromkeys(values):
        selected = best[array == value]
        counts = tuple(int(np.sum(selected == index)) for index in range(view_count))
        result.append(
            GroupedBestView(
                group_name=group_name,
                group_value=value,
                specimen_count=len(selected),
                counts=counts,
                frequencies=tuple(count / len(selected) for count in counts),
            )
        )
    return tuple(result)


def audit_predictions(
    targets: object,
    predictions: object,
    domains: Sequence[str],
    *,
    view_names: Sequence[str],
    groups: Mapping[str, Sequence[object]] | None = None,
    baseline_mae: float | None = None,
) -> AgreementAudit:
    """Compute the registered E1 diagnostics from strict outer OOF predictions."""

    y, values, domain_ids, names = _validated(targets, predictions, domains, view_names)
    metrics = tuple(
        _metrics(name, y, values[:, index], domain_ids)
        for index, name in enumerate(names)
    )
    residuals = y[:, None] - values
    prediction_correlations = _correlations(values)
    residual_correlations = _correlations(residuals)
    disagreement = np.zeros((len(names), len(names)), dtype=np.float64)
    for left in range(len(names)):
        for right in range(len(names)):
            disagreement[left, right] = float(
                np.mean(np.abs(values[:, left] - values[:, right]))
            )
    errors = np.abs(residuals)
    best = np.argmin(errors, axis=1).astype(np.int64)
    counts = tuple(int(np.sum(best == index)) for index in range(len(names)))
    oracle_mae = float(np.mean(np.min(errors, axis=1)))
    comparator = (
        metrics[0].equal_domain_mae if baseline_mae is None else float(baseline_mae)
    )
    if not np.isfinite(comparator) or comparator <= 0.0:
        raise ValueError("baseline MAE must be positive and finite")
    grouped: list[GroupedBestView] = list(
        _group_rows(best, "domain", domain_ids, len(names))
    )
    if groups is not None:
        for group_name, group_values in groups.items():
            if type(group_name) is not str or not group_name or group_name == "domain":
                raise ValueError("group name is invalid")
            grouped.extend(_group_rows(best, group_name, group_values, len(names)))
    upper = prediction_correlations[np.triu_indices(len(names), k=1)]
    residual_upper = residual_correlations[np.triu_indices(len(names), k=1)]
    predictive_equivalence = bool(
        np.min(upper) >= 0.95
        and max(item.equal_domain_mae for item in metrics) <= 1.10 * comparator
    )
    useful = sum(item.equal_domain_mae <= 1.10 * comparator for item in metrics) >= 2
    complementarity_signal = bool(
        useful
        and np.min(residual_upper) <= 0.90
        and (comparator - oracle_mae) / comparator >= 0.10
    )
    immutable_best = _readonly(best, dtype="<i8")
    return AgreementAudit(
        view_names=names,
        view_metrics=metrics,
        prediction_correlations=prediction_correlations,
        residual_correlations=residual_correlations,
        mean_absolute_disagreement=_readonly(disagreement),
        oracle_mae=oracle_mae,
        oracle_improvement_vs_full=float((comparator - oracle_mae) / comparator),
        best_view_indices=immutable_best,
        best_view_counts=counts,
        grouped_best_view=tuple(grouped),
        deployable_methods=names,
        predictive_equivalence=predictive_equivalence,
        complementarity_signal=complementarity_signal,
        gate_status="GO"
        if predictive_equivalence or complementarity_signal
        else "NO_GO",
    )
