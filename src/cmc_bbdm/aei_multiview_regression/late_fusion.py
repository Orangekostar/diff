"""Leakage-safe equal and validation-weighted late fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog  # type: ignore[import-untyped]


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    result = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    result.setflags(write=False)
    return result


def _matrix(value: object) -> np.ndarray:
    if np.iscomplexobj(np.asarray(value)):
        raise ValueError("predictions must be real")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1 or not np.all(np.isfinite(result)):
        raise ValueError("predictions must be a finite matrix")
    return np.array(result, copy=True)


@dataclass(frozen=True, slots=True)
class ValidationWeights:
    weights: np.ndarray
    objective_equal_domain_mae: float

    def predict(self, predictions: object) -> np.ndarray:
        matrix = _matrix(predictions)
        if matrix.shape[1] != len(self.weights):
            raise ValueError("fusion view roster changed")
        return np.asarray(matrix @ self.weights, dtype=np.float64)


def equal_fusion(predictions: object) -> np.ndarray:
    """Return the registered arithmetic-mean fusion."""

    return np.asarray(np.mean(_matrix(predictions), axis=1), dtype=np.float64)


def fit_validation_weights(
    predictions: object,
    targets: object,
    *,
    domains: Sequence[str],
) -> ValidationWeights:
    """Minimize source-OOF equal-domain MAE on the nonnegative simplex."""

    matrix = _matrix(predictions)
    y = np.asarray(targets, dtype=np.float64)
    domain_ids = tuple(domains)
    if (
        y.shape != (len(matrix),)
        or not np.all(np.isfinite(y))
        or len(domain_ids) != len(y)
        or any(type(item) is not str or not item for item in domain_ids)
    ):
        raise ValueError("fusion targets or domains do not align")
    domain_array = np.asarray(domain_ids)
    domain_order = tuple(dict.fromkeys(domain_ids))
    sample_weights = np.asarray(
        [
            1.0 / (len(domain_order) * np.sum(domain_array == domain))
            for domain in domain_ids
        ],
        dtype=np.float64,
    )
    rows, views = matrix.shape
    objective = np.concatenate((np.zeros(views), sample_weights))
    upper = np.zeros((2 * rows, views + rows), dtype=np.float64)
    bound = np.zeros(2 * rows, dtype=np.float64)
    for index in range(rows):
        upper[2 * index, :views] = matrix[index]
        upper[2 * index, views + index] = -1.0
        bound[2 * index] = y[index]
        upper[2 * index + 1, :views] = -matrix[index]
        upper[2 * index + 1, views + index] = -1.0
        bound[2 * index + 1] = -y[index]
    equality = np.zeros((1, views + rows), dtype=np.float64)
    equality[0, :views] = 1.0
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=bound,
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * views + [(0.0, None)] * rows,
        method="highs",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"validation-weight optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x[:views]), 0.0, 1.0)
    weights /= np.sum(weights)
    prediction = matrix @ weights
    domain_mae = [
        np.mean(np.abs(y[domain_array == domain] - prediction[domain_array == domain]))
        for domain in domain_order
    ]
    return ValidationWeights(
        weights=_readonly(weights),
        objective_equal_domain_mae=float(np.mean(domain_mae)),
    )
