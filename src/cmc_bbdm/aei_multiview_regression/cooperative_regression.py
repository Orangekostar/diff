"""Continuous cooperative regression with soft pairwise agreement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]


class CooperativeRegressionError(ValueError):
    """Raised when a cooperative fit is invalid or fails closed."""


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    result = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ViewLinearState:
    imputer_statistics: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coef: np.ndarray
    intercept: float


@dataclass(frozen=True, slots=True)
class CooperativeFit:
    views: tuple[ViewLinearState, ...]
    lambda_consistency: float
    loss: str
    alpha: float
    huber_delta: float
    train_predictions: np.ndarray
    mean_absolute_disagreement: float
    prediction_variances: tuple[float, ...]
    optimizer_success: bool
    optimizer_iterations: int

    def predict(self, designs: tuple[object, ...]) -> np.ndarray:
        if not isinstance(designs, tuple) or len(designs) != len(self.views):
            raise CooperativeRegressionError("query view roster changed")
        predictions: list[np.ndarray] = []
        rows: int | None = None
        for raw, state in zip(designs, self.views, strict=True):
            matrix = _matrix(raw, "query design", allow_nan=True)
            if rows is None:
                rows = len(matrix)
            if len(matrix) != rows or matrix.shape[1] != state.coef.size:
                raise CooperativeRegressionError("query designs do not align")
            imputed = np.where(np.isnan(matrix), state.imputer_statistics, matrix)
            scaled = (imputed - state.feature_mean) / state.feature_scale
            predictions.append(scaled @ state.coef + state.intercept)
        output = np.column_stack(predictions)
        if not np.all(np.isfinite(output)):
            raise CooperativeRegressionError("cooperative prediction is non-finite")
        return np.asarray(output, dtype=np.float64)


def _matrix(value: object, label: str, *, allow_nan: bool) -> np.ndarray:
    if np.iscomplexobj(np.asarray(value)):
        raise CooperativeRegressionError(f"{label} must be real")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CooperativeRegressionError(f"{label} must be numeric") from error
    valid = ~np.isinf(result) if allow_nan else np.isfinite(result)
    if result.ndim != 2 or min(result.shape) < 1 or not np.all(valid):
        raise CooperativeRegressionError(f"{label} must be a valid matrix")
    return np.array(result, dtype=np.float64, copy=True, order="C")


def _prepare(
    designs: tuple[object, ...], targets: object
) -> tuple[tuple[np.ndarray, ...], np.ndarray, tuple[tuple[np.ndarray, ...], ...]]:
    if not isinstance(designs, tuple) or not designs:
        raise CooperativeRegressionError("designs must be a nonempty view tuple")
    if np.iscomplexobj(np.asarray(targets)):
        raise CooperativeRegressionError("targets must be real")
    try:
        y = np.asarray(targets, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CooperativeRegressionError("targets must be numeric") from error
    matrices = tuple(_matrix(item, "design", allow_nan=True) for item in designs)
    if (
        y.ndim != 1
        or len(y) < 2
        or not np.all(np.isfinite(y))
        or any(len(item) != len(y) for item in matrices)
    ):
        raise CooperativeRegressionError("cooperative inputs do not align")
    scaled: list[np.ndarray] = []
    states: list[tuple[np.ndarray, ...]] = []
    for matrix in matrices:
        imputer = np.nanmean(matrix, axis=0)
        if not np.all(np.isfinite(imputer)):
            raise CooperativeRegressionError(
                "every design feature needs a finite fit value"
            )
        filled = np.where(np.isnan(matrix), imputer, matrix)
        mean = np.mean(filled, axis=0, dtype=np.float64)
        scale = np.std(filled, axis=0, dtype=np.float64, ddof=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        transformed = (filled - mean) / scale
        scaled.append(transformed)
        states.append((imputer, mean, scale))
    return tuple(scaled), np.array(y, copy=True), tuple(states)


def _layout(designs: tuple[np.ndarray, ...]) -> tuple[tuple[slice, ...], int]:
    offsets: list[slice] = []
    start = 0
    for matrix in designs:
        stop = start + matrix.shape[1] + 1
        offsets.append(slice(start, stop))
        start = stop
    return tuple(offsets), start


def _augmented(designs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.column_stack((matrix, np.ones(len(matrix), dtype=np.float64)))
        for matrix in designs
    )


def _mse_solution(
    designs: tuple[np.ndarray, ...],
    targets: np.ndarray,
    *,
    lambda_consistency: float,
    alpha: float,
) -> np.ndarray:
    augmented = _augmented(designs)
    slices, size = _layout(designs)
    gram = np.zeros((size, size), dtype=np.float64)
    rhs = np.zeros(size, dtype=np.float64)
    for matrix, block in zip(augmented, slices, strict=True):
        gram[block, block] += matrix.T @ matrix
        rhs[block] += matrix.T @ targets
        coefficient_indices = np.arange(block.start, block.stop - 1)
        gram[coefficient_indices, coefficient_indices] += alpha
    for left in range(len(designs)):
        for right in range(left + 1, len(designs)):
            left_matrix = augmented[left]
            right_matrix = augmented[right]
            left_block = slices[left]
            right_block = slices[right]
            gram[left_block, left_block] += lambda_consistency * (
                left_matrix.T @ left_matrix
            )
            gram[right_block, right_block] += lambda_consistency * (
                right_matrix.T @ right_matrix
            )
            cross = lambda_consistency * (left_matrix.T @ right_matrix)
            gram[left_block, right_block] -= cross
            gram[right_block, left_block] -= cross.T
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _predictions(
    theta: np.ndarray,
    augmented: tuple[np.ndarray, ...],
    slices: tuple[slice, ...],
) -> np.ndarray:
    return np.column_stack(
        [matrix @ theta[block] for matrix, block in zip(augmented, slices, strict=True)]
    )


def _huber_solution(
    initial: np.ndarray,
    designs: tuple[np.ndarray, ...],
    targets: np.ndarray,
    *,
    lambda_consistency: float,
    alpha: float,
    delta: float,
) -> tuple[np.ndarray, bool, int]:
    augmented = _augmented(designs)
    slices, _size = _layout(designs)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        prediction = _predictions(theta, augmented, slices)
        residual = prediction - targets[:, None]
        absolute = np.abs(residual)
        target_loss = np.where(
            absolute <= delta,
            0.5 * residual**2,
            delta * (absolute - 0.5 * delta),
        )
        value = float(np.sum(target_loss))
        derivative = np.clip(residual, -delta, delta)
        gradient = np.zeros_like(theta)
        for view, (matrix, block) in enumerate(zip(augmented, slices, strict=True)):
            gradient[block] += matrix.T @ derivative[:, view]
            coefficients = theta[block][:-1]
            value += 0.5 * alpha * float(coefficients @ coefficients)
            gradient[block.start : block.stop - 1] += alpha * coefficients
        for left in range(len(designs)):
            for right in range(left + 1, len(designs)):
                difference = prediction[:, left] - prediction[:, right]
                value += 0.5 * lambda_consistency * float(difference @ difference)
                gradient[slices[left]] += (
                    lambda_consistency * augmented[left].T @ difference
                )
                gradient[slices[right]] -= (
                    lambda_consistency * augmented[right].T @ difference
                )
        return value, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2_000, "ftol": 1e-13, "gtol": 1e-8, "maxls": 50},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise CooperativeRegressionError(
            f"Huber cooperative fit did not converge: {result.message}"
        )
    return np.asarray(result.x, dtype=np.float64), True, int(result.nit)


def fit_cooperative(
    designs: tuple[object, ...],
    targets: object,
    *,
    lambda_consistency: float,
    loss: str,
    alpha: float = 10.0,
    huber_delta: float = 0.05,
) -> CooperativeFit:
    """Fit view-specific regressors under a soft prediction-agreement penalty."""

    if (
        type(lambda_consistency) not in (int, float)
        or not np.isfinite(lambda_consistency)
        or lambda_consistency < 0.0
    ):
        raise CooperativeRegressionError(
            "consistency strength must be finite and nonnegative"
        )
    if type(alpha) not in (int, float) or float(alpha) != 10.0:
        raise CooperativeRegressionError("alpha must equal 10.0")
    if loss not in {"mse", "huber"}:
        raise CooperativeRegressionError("target loss must be mse or huber")
    if type(huber_delta) not in (int, float) or float(huber_delta) <= 0.0:
        raise CooperativeRegressionError("Huber delta must be positive")
    scaled, y, preprocessing = _prepare(designs, targets)
    initial = _mse_solution(
        scaled,
        y,
        lambda_consistency=float(lambda_consistency),
        alpha=10.0,
    )
    if loss == "huber":
        theta, success, iterations = _huber_solution(
            initial,
            scaled,
            y,
            lambda_consistency=float(lambda_consistency),
            alpha=10.0,
            delta=float(huber_delta),
        )
    else:
        theta, success, iterations = initial, True, 0
    augmented = _augmented(scaled)
    slices, _size = _layout(scaled)
    train_predictions = _predictions(theta, augmented, slices)
    states: list[ViewLinearState] = []
    for raw_state, block in zip(preprocessing, slices, strict=True):
        imputer, mean, scale = raw_state
        parameters = theta[block]
        states.append(
            ViewLinearState(
                imputer_statistics=_readonly(imputer),
                feature_mean=_readonly(mean),
                feature_scale=_readonly(scale),
                coef=_readonly(parameters[:-1]),
                intercept=float(parameters[-1]),
            )
        )
    disagreements = [
        np.mean(np.abs(train_predictions[:, left] - train_predictions[:, right]))
        for left in range(len(states))
        for right in range(left + 1, len(states))
    ]
    mean_disagreement = float(np.mean(disagreements)) if disagreements else 0.0
    return CooperativeFit(
        views=tuple(states),
        lambda_consistency=float(lambda_consistency),
        loss=loss,
        alpha=10.0,
        huber_delta=float(huber_delta),
        train_predictions=_readonly(train_predictions),
        mean_absolute_disagreement=mean_disagreement,
        prediction_variances=tuple(
            float(np.var(train_predictions[:, index], ddof=0))
            for index in range(train_predictions.shape[1])
        ),
        optimizer_success=success,
        optimizer_iterations=iterations,
    )
