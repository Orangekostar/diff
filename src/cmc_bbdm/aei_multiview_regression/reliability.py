"""Prediction-dispersion reliability diagnostics for deployable methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class ReliabilityStratum:
    name: str
    count: int
    mean_dispersion: float
    mean_absolute_error: float


@dataclass(frozen=True, slots=True)
class MethodReliability:
    method: str
    pearson_r: float
    pearson_p_value: float
    spearman_r: float
    spearman_p_value: float
    absolute_errors: np.ndarray
    strata: tuple[ReliabilityStratum, ...]


@dataclass(frozen=True, slots=True)
class ReliabilityAudit:
    dispersion: np.ndarray
    stratum_labels: tuple[str, ...]
    methods: tuple[MethodReliability, ...]


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    result = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    result.setflags(write=False)
    return result


def _correlation(function, left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return 0.0, 1.0
    result = function(left, right)
    value = float(result.statistic)
    p_value = float(result.pvalue)
    return (
        value if np.isfinite(value) else 0.0,
        p_value if np.isfinite(p_value) else 1.0,
    )


def audit_reliability(
    targets: object,
    view_predictions: object,
    *,
    deployable_predictions: Mapping[str, object],
) -> ReliabilityAudit:
    """Relate cross-view OOF dispersion to deployable absolute error."""

    if np.iscomplexobj(np.asarray(targets)) or np.iscomplexobj(
        np.asarray(view_predictions)
    ):
        raise ValueError("reliability inputs must be real")
    y = np.asarray(targets, dtype=np.float64)
    views = np.asarray(view_predictions, dtype=np.float64)
    if (
        y.ndim != 1
        or views.ndim != 2
        or views.shape[0] != len(y)
        or views.shape[1] < 2
        or len(y) < 4
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(views))
        or not isinstance(deployable_predictions, Mapping)
        or not deployable_predictions
    ):
        raise ValueError("reliability inputs are invalid or misaligned")
    dispersion = np.std(views, axis=1, ddof=0)
    order = np.argsort(dispersion, kind="stable")
    quarter = len(y) // 4
    if quarter < 1:
        raise ValueError("reliability strata require at least four specimens")
    strata_indices = (
        ("lowest_25_percent", order[:quarter]),
        ("middle_50_percent", order[quarter : len(y) - quarter]),
        ("highest_25_percent", order[len(y) - quarter :]),
    )
    stratum_labels = [""] * len(y)
    for name, indices in strata_indices:
        for index in indices:
            stratum_labels[int(index)] = name
    if any(not item for item in stratum_labels):
        raise ValueError("reliability strata do not cover every specimen")
    methods: list[MethodReliability] = []
    for method, raw in deployable_predictions.items():
        if type(method) is not str or not method or "oracle" in method.lower():
            raise ValueError("oracle cannot be a deployable reliability method")
        prediction = np.asarray(raw, dtype=np.float64)
        if prediction.shape != y.shape or not np.all(np.isfinite(prediction)):
            raise ValueError(f"deployable prediction is invalid: {method}")
        error = np.abs(y - prediction)
        pearson_r, pearson_p_value = _correlation(pearsonr, dispersion, error)
        spearman_r, spearman_p_value = _correlation(spearmanr, dispersion, error)
        strata = tuple(
            ReliabilityStratum(
                name=name,
                count=len(indices),
                mean_dispersion=float(np.mean(dispersion[indices])),
                mean_absolute_error=float(np.mean(error[indices])),
            )
            for name, indices in strata_indices
        )
        methods.append(
            MethodReliability(
                method=method,
                pearson_r=pearson_r,
                pearson_p_value=pearson_p_value,
                spearman_r=spearman_r,
                spearman_p_value=spearman_p_value,
                absolute_errors=_readonly(error),
                strata=strata,
            )
        )
    return ReliabilityAudit(
        dispersion=_readonly(dispersion),
        stratum_labels=tuple(stratum_labels),
        methods=tuple(methods),
    )
