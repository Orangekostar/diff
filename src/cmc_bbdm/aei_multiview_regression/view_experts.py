"""Fold-local mechanics-consistent view experts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


class ViewExpertError(ValueError):
    """Raised when a view expert would violate the registered fit contract."""


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    result = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    result.setflags(write=False)
    return result


def _matrix(value: object, label: str, *, allow_nan: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ViewExpertError(f"{label} must be real")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ViewExpertError(f"{label} must be numeric") from error
    if result.ndim != 2 or min(result.shape) < 1:
        raise ViewExpertError(f"{label} must be a nonempty matrix")
    valid = ~np.isinf(result) if allow_nan else np.isfinite(result)
    if not np.all(valid):
        raise ViewExpertError(f"{label} contains invalid values")
    return np.array(result, dtype=np.float64, copy=True, order="C")


def _vector(value: object, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ViewExpertError(f"{label} must be real")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ViewExpertError(f"{label} must be numeric") from error
    if result.ndim != 1 or result.size < 1 or not np.all(np.isfinite(result)):
        raise ViewExpertError(f"{label} must be a finite vector")
    return np.array(result, dtype=np.float64, copy=True)


def _indices(value: object, *, rows: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ViewExpertError("fit indices must be an integer vector")
    result = np.asarray(raw, dtype=np.int64)
    if (
        result.size < 2
        or np.any(result < 0)
        or np.any(result >= rows)
        or len(np.unique(result)) != len(result)
    ):
        raise ViewExpertError("fit indices are invalid")
    return result


@dataclass(frozen=True, slots=True)
class PCABasis:
    mean: np.ndarray
    components: np.ndarray
    fit_indices: tuple[int, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class FittedViewExpert:
    pca_mean: np.ndarray
    components: np.ndarray
    imputer_statistics: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coef: np.ndarray
    intercept: float
    fit_indices: tuple[int, ...]
    alpha: float
    state_sha256: str

    def predict(self, embeddings: object, metadata: object) -> np.ndarray:
        embedding_matrix = _matrix(embeddings, "query embeddings")
        metadata_matrix = _matrix(metadata, "query metadata", allow_nan=True)
        if (
            len(embedding_matrix) != len(metadata_matrix)
            or embedding_matrix.shape[1] != self.pca_mean.size
            or metadata_matrix.shape[1] + self.components.shape[0] != self.coef.size
        ):
            raise ViewExpertError("query feature shapes do not match the fitted expert")
        projected = (embedding_matrix - self.pca_mean) @ self.components.T
        design = np.column_stack((metadata_matrix, projected))
        imputed = np.where(np.isnan(design), self.imputer_statistics, design)
        scaled = (imputed - self.feature_mean) / self.feature_scale
        predictions = scaled @ self.coef + self.intercept
        if not np.all(np.isfinite(predictions)):
            raise ViewExpertError("view expert returned non-finite predictions")
        return np.asarray(predictions, dtype=np.float64)


def fit_pca_basis(
    embeddings: object,
    fit_indices: object,
    *,
    maximum_dimension: int,
) -> PCABasis:
    """Fit one canonical full-SVD basis for all registered prefix dimensions."""

    matrix = _matrix(embeddings, "embeddings")
    fit = _indices(fit_indices, rows=len(matrix))
    if (
        type(maximum_dimension) is not int
        or maximum_dimension < 1
        or maximum_dimension > min(len(fit) - 1, matrix.shape[1])
    ):
        raise ViewExpertError("PCA dimension exceeds the fit rank bound")
    mean = np.mean(matrix[fit], axis=0, dtype=np.float64)
    centered = matrix[fit] - mean
    try:
        _left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
        raise ViewExpertError("fold-local PCA failed") from error
    if not np.all(np.isfinite(singular_values)) or not np.all(np.isfinite(right)):
        raise ViewExpertError("fold-local PCA returned non-finite values")
    components = np.asarray(right[:maximum_dimension], dtype=np.float64).copy()
    for row in range(len(components)):
        pivot = int(np.argmax(np.abs(components[row])))
        if components[row, pivot] < 0.0:
            components[row] *= -1.0
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mean, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(components, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(fit, dtype="<i8").tobytes())
    return PCABasis(
        mean=_readonly(mean),
        components=_readonly(components),
        fit_indices=tuple(int(item) for item in fit),
        state_sha256=digest.hexdigest(),
    )


def fit_view_expert(
    embeddings: object,
    metadata: object,
    targets: object,
    fit_indices: object,
    *,
    pca_dimension: int,
    alpha: float = 10.0,
    pca_basis: PCABasis | None = None,
) -> FittedViewExpert:
    """Fit the exact P1 metadata + PCA + standardized Ridge estimator."""

    embedding_matrix = _matrix(embeddings, "embeddings")
    metadata_matrix = _matrix(metadata, "metadata", allow_nan=True)
    response = _vector(targets, "targets")
    if len(embedding_matrix) != len(metadata_matrix) or len(response) != len(
        embedding_matrix
    ):
        raise ViewExpertError("expert inputs do not align")
    fit = _indices(fit_indices, rows=len(response))
    if type(pca_dimension) is not int or pca_dimension < 1:
        raise ViewExpertError("PCA dimension must be positive")
    if type(alpha) not in (int, float) or float(alpha) != 10.0:
        raise ViewExpertError("Ridge alpha must equal the registered value 10.0")
    basis = pca_basis or fit_pca_basis(
        embedding_matrix, fit, maximum_dimension=pca_dimension
    )
    if basis.fit_indices != tuple(int(item) for item in fit):
        raise ViewExpertError("PCA basis was fitted on different rows")
    if basis.mean.size != embedding_matrix.shape[1] or pca_dimension > len(
        basis.components
    ):
        raise ViewExpertError("PCA basis does not support the requested dimension")
    components = np.asarray(basis.components[:pca_dimension], dtype=np.float64)
    projected = (embedding_matrix[fit] - basis.mean) @ components.T
    design = np.column_stack((metadata_matrix[fit], projected))
    imputer = np.nanmean(design, axis=0)
    if not np.all(np.isfinite(imputer)):
        raise ViewExpertError("every fitted feature must contain a finite value")
    filled = np.where(np.isnan(design), imputer, design)
    mean = np.mean(filled, axis=0, dtype=np.float64)
    scale = np.std(filled, axis=0, dtype=np.float64, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    scaled = (filled - mean) / scale
    x_mean = np.mean(scaled, axis=0, dtype=np.float64)
    y_mean = float(np.mean(response[fit], dtype=np.float64))
    centered_x = scaled - x_mean
    centered_y = response[fit] - y_mean
    system = centered_x.T @ centered_x + 10.0 * np.eye(scaled.shape[1])
    rhs = centered_x.T @ centered_y
    try:
        coef = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(system, rhs, rcond=None)[0]
    intercept = y_mean - float(x_mean @ coef)
    if not np.all(np.isfinite(coef)) or not np.isfinite(intercept):
        raise ViewExpertError("Ridge fit returned non-finite parameters")
    digest = hashlib.sha256()
    for item in (basis.mean, components, imputer, mean, scale, coef):
        digest.update(np.ascontiguousarray(item, dtype="<f8").tobytes())
        digest.update(b"\0")
    digest.update(repr(intercept).encode("ascii"))
    digest.update(np.ascontiguousarray(fit, dtype="<i8").tobytes())
    return FittedViewExpert(
        pca_mean=_readonly(np.asarray(basis.mean)),
        components=_readonly(components),
        imputer_statistics=_readonly(imputer),
        feature_mean=_readonly(mean),
        feature_scale=_readonly(scale),
        coef=_readonly(np.asarray(coef, dtype=np.float64)),
        intercept=float(intercept),
        fit_indices=tuple(int(item) for item in fit),
        alpha=10.0,
        state_sha256=digest.hexdigest(),
    )
