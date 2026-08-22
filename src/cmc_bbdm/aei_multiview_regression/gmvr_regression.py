"""Lightweight prediction-space consistency and complementarity weighting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    result = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class GMvRWeights:
    weights: np.ndarray
    predictions: np.ndarray
    lambda_consistency: float
    lambda_complementarity: float
    prediction_loss: float
    consistency_loss: float
    concentration: float
    mean_absolute_contributions: tuple[float, ...]

    def predict(self, view_predictions: object) -> np.ndarray:
        matrix = np.asarray(view_predictions, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(self.weights)
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError("GMvR query predictions are invalid")
        return np.asarray(matrix @ self.weights, dtype=np.float64)


def fit_gmvr_weights(
    view_predictions: object,
    targets: object,
    *,
    domains: Sequence[str],
    lambda_consistency: float,
    lambda_complementarity: float,
) -> GMvRWeights:
    """Learn nonnegative view contributions without feature-space forcing."""

    matrix = np.asarray(view_predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    domain_ids = tuple(domains)
    if (
        matrix.ndim != 2
        or matrix.shape[1] < 2
        or y.shape != (len(matrix),)
        or len(domain_ids) != len(y)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(y))
        or type(lambda_consistency) not in (int, float)
        or type(lambda_complementarity) not in (int, float)
        or not np.isfinite(lambda_consistency)
        or not np.isfinite(lambda_complementarity)
        or lambda_consistency < 0.0
        or lambda_complementarity < 0.0
    ):
        raise ValueError("GMvR inputs or penalties are invalid")
    domain_array = np.asarray(domain_ids)
    order = tuple(dict.fromkeys(domain_ids))
    sample_weights = np.asarray(
        [1.0 / (len(order) * np.sum(domain_array == domain)) for domain in domain_ids]
    )
    squared_views = matrix**2

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        prediction = matrix @ weights
        residual = prediction - y
        prediction_loss = float(np.sum(sample_weights * residual**2))
        consistency = float(
            np.sum(sample_weights * (squared_views @ weights - prediction**2))
        )
        concentration = float(weights @ weights)
        value = (
            prediction_loss
            + float(lambda_consistency) * consistency
            + float(lambda_complementarity) * concentration
        )
        gradient = 2.0 * matrix.T @ (sample_weights * residual)
        gradient += float(lambda_consistency) * (
            squared_views.T @ sample_weights
            - 2.0 * matrix.T @ (sample_weights * prediction)
        )
        gradient += 2.0 * float(lambda_complementarity) * weights
        return value, gradient

    views = matrix.shape[1]
    result = minimize(
        objective,
        np.full(views, 1.0 / views),
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * views,
        constraints={
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-13, "maxiter": 2_000, "disp": False},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"GMvR weight fit failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    weights /= np.sum(weights)
    prediction = matrix @ weights
    consistency = float(
        np.sum(sample_weights * (squared_views @ weights - prediction**2))
    )
    return GMvRWeights(
        weights=_readonly(weights),
        predictions=_readonly(prediction),
        lambda_consistency=float(lambda_consistency),
        lambda_complementarity=float(lambda_complementarity),
        prediction_loss=float(np.sum(sample_weights * (prediction - y) ** 2)),
        consistency_loss=max(consistency, 0.0),
        concentration=float(weights @ weights),
        mean_absolute_contributions=tuple(
            float(np.mean(np.abs(weights[index] * matrix[:, index])))
            for index in range(views)
        ),
    )
