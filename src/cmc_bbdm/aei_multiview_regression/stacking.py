"""Domain-held-out level-1 stacking over strict source OOF predictions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear  # type: ignore[import-untyped]
from sklearn.linear_model import HuberRegressor  # type: ignore[import-untyped]


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    result = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class StackerFit:
    method: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, predictions: object) -> np.ndarray:
        matrix = np.asarray(predictions, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(self.coef)
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError("stacking query predictions are invalid")
        return np.asarray(
            ((matrix - self.feature_mean) / self.feature_scale) @ self.coef
            + self.intercept,
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class MetaFitEvent:
    method: str
    query_domain: str
    fit_domains: tuple[str, ...]
    base_prediction_role: str = "source_oof"


@dataclass(frozen=True, slots=True)
class StackerScore:
    method: str
    domain_mae: tuple[tuple[str, float], ...]
    equal_domain_mae: float
    worst_domain_mae: float
    domain_mae_sd: float


@dataclass(frozen=True, slots=True)
class StackerSelection:
    selected_method: str
    scores: tuple[StackerScore, ...]
    fitted: StackerFit


def _inputs(predictions: object, targets: object) -> tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(np.asarray(predictions)) or np.iscomplexobj(np.asarray(targets)):
        raise ValueError("stacking inputs must be real")
    matrix = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if (
        matrix.ndim != 2
        or min(matrix.shape) < 1
        or y.shape != (len(matrix),)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("stacking inputs are invalid or misaligned")
    return np.array(matrix, copy=True), np.array(y, copy=True)


def fit_stacker(
    predictions: object,
    targets: object,
    *,
    method: str,
    alpha: float = 1.0,
) -> StackerFit:
    """Fit one registered linear meta-regressor."""

    matrix, y = _inputs(predictions, targets)
    if method not in {"ridge", "nonnegative_ridge", "huber"}:
        raise ValueError("stacker method is not registered")
    if type(alpha) not in (int, float) or not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("stacker alpha is invalid")
    mean = np.mean(matrix, axis=0, dtype=np.float64)
    scale = np.std(matrix, axis=0, dtype=np.float64, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    x = (matrix - mean) / scale
    if method == "ridge":
        x_mean = np.mean(x, axis=0)
        y_mean = float(np.mean(y))
        centered = x - x_mean
        system = centered.T @ centered + float(alpha) * np.eye(x.shape[1])
        rhs = centered.T @ (y - y_mean)
        try:
            coef = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(system, rhs, rcond=None)[0]
        intercept = y_mean - float(x_mean @ coef)
    elif method == "nonnegative_ridge":
        x_mean = np.mean(x, axis=0, dtype=np.float64)
        y_mean = float(np.mean(y, dtype=np.float64))
        centered_x = x - x_mean
        centered_y = y - y_mean
        system = np.vstack(
            (centered_x, np.sqrt(float(alpha)) * np.eye(x.shape[1]))
        )
        rhs = np.concatenate((centered_y, np.zeros(x.shape[1])))
        result = lsq_linear(
            system,
            rhs,
            bounds=(np.zeros(x.shape[1]), np.full(x.shape[1], np.inf)),
            method="bvls",
            tol=1e-12,
            max_iter=2_000,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise ValueError(
                f"nonnegative Ridge stacker failed: {result.message}"
            )
        coef = np.asarray(result.x, dtype=np.float64)
        intercept = y_mean - float(x_mean @ coef)
    else:
        model = HuberRegressor(
            alpha=max(float(alpha), 1e-12), epsilon=1.35, max_iter=2_000, tol=1e-10
        )
        model.fit(x, y)
        coef = np.asarray(model.coef_, dtype=np.float64)
        intercept = float(model.intercept_)
    if not np.all(np.isfinite(coef)) or not np.isfinite(intercept):
        raise ValueError("stacker fit returned non-finite parameters")
    return StackerFit(
        method=method,
        feature_mean=_readonly(mean),
        feature_scale=_readonly(scale),
        coef=_readonly(np.asarray(coef, dtype=np.float64)),
        intercept=float(intercept),
    )


def select_stacker_oof(
    predictions: object,
    targets: object,
    domains: Sequence[str],
    *,
    methods: tuple[str, ...] = ("ridge", "nonnegative_ridge", "huber"),
    alpha: float = 1.0,
    fit_hook: Callable[[MetaFitEvent], None] | None = None,
) -> StackerSelection:
    """Select a stacker by domain-held-out fits over source OOF base predictions."""

    matrix, y = _inputs(predictions, targets)
    domain_ids = tuple(domains)
    if len(domain_ids) != len(y) or len(set(domain_ids)) < 3:
        raise ValueError("stacker selection requires at least three aligned domains")
    domain_order = tuple(dict.fromkeys(domain_ids))
    domain_array = np.asarray(domain_ids)
    scores: list[StackerScore] = []
    ranking: list[tuple[tuple[float, float, float, int], str]] = []
    for method_order, method in enumerate(methods):
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for query_domain in domain_order:
            fit_indices = np.flatnonzero(domain_array != query_domain)
            query_indices = np.flatnonzero(domain_array == query_domain)
            if fit_hook is not None:
                fit_hook(
                    MetaFitEvent(
                        method=method,
                        query_domain=query_domain,
                        fit_domains=tuple(domain_array[fit_indices].tolist()),
                    )
                )
            model = fit_stacker(
                matrix[fit_indices], y[fit_indices], method=method, alpha=alpha
            )
            oof[query_indices] = model.predict(matrix[query_indices])
        domain_mae = tuple(
            (
                domain,
                float(
                    np.mean(
                        np.abs(y[domain_array == domain] - oof[domain_array == domain])
                    )
                ),
            )
            for domain in domain_order
        )
        values = np.asarray([item[1] for item in domain_mae])
        score = StackerScore(
            method=method,
            domain_mae=domain_mae,
            equal_domain_mae=float(np.mean(values)),
            worst_domain_mae=float(np.max(values)),
            domain_mae_sd=float(np.std(values, ddof=0)),
        )
        scores.append(score)
        ranking.append(
            (
                (
                    score.equal_domain_mae,
                    score.worst_domain_mae,
                    score.domain_mae_sd,
                    method_order,
                ),
                method,
            )
        )
    selected = min(ranking, key=lambda item: item[0])[1]
    return StackerSelection(
        selected_method=selected,
        scores=tuple(scores),
        fitted=fit_stacker(matrix, y, method=selected, alpha=alpha),
    )
