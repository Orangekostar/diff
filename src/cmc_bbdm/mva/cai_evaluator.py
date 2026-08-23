"""Fold-local PCA/Ridge CAI predictor used by MVA P-A and P-B."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.cpb_v3.models import (
    FoldPreprocessor,
    FoldRidgeModel,
    fit_fold_preprocessor,
    fit_fold_ridge,
)


class CAIEvaluatorError(ValueError):
    """Raised when an MVA CAI predictor request is invalid."""


def _readonly(value: object, *, dtype: object = np.float64) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    output.setflags(write=False)
    return output


def _state(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(
                json.dumps(value.shape, separators=(",", ":")).encode("ascii")
            )
            digest.update(value.tobytes(order="C"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PCAProjection:
    mean: np.ndarray
    components: np.ndarray
    fit_specimen_ids: tuple[str, ...]
    fit_domains: tuple[str, ...]
    state_sha256: str

    @property
    def dimension(self) -> int:
        return int(self.components.shape[0])

    def transform(self, embeddings: object) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != self.mean.size
            or not np.all(np.isfinite(values))
        ):
            raise CAIEvaluatorError("query embeddings are invalid")
        output = (values - self.mean) @ self.components.T
        if not np.all(np.isfinite(output)):
            raise CAIEvaluatorError("PCA transform is nonfinite")
        return np.asarray(output, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SensitivityRidgeModel:
    preprocessor: FoldPreprocessor
    alpha: float
    coef_: np.ndarray
    intercept_: float
    fit_sample_ids: tuple[str, ...]
    fit_domain_ids: tuple[str, ...]
    state_sha256: str

    def predict(self, matrix: object) -> np.ndarray:
        transformed = self.preprocessor.transform(matrix)
        output = transformed @ self.coef_ + self.intercept_
        if not np.all(np.isfinite(output)):
            raise CAIEvaluatorError("sensitivity Ridge prediction is nonfinite")
        return np.asarray(output, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CAIPredictor:
    method: str
    outer_domain: str
    pca: PCAProjection
    ridge: FoldRidgeModel | SensitivityRidgeModel
    metadata_features: int
    fit_specimen_ids: tuple[str, ...]
    fit_domains: tuple[str, ...]
    state_sha256: str

    def predict(self, metadata: object, embeddings: object) -> np.ndarray:
        meta = np.asarray(metadata, dtype=np.float64)
        values = np.asarray(embeddings, dtype=np.float64)
        if (
            meta.ndim != 2
            or meta.shape[1] != self.metadata_features
            or values.ndim != 2
            or values.shape[0] != meta.shape[0]
        ):
            raise CAIEvaluatorError("predictor query arrays are misaligned")
        projected = self.pca.transform(values)
        return self.ridge.predict(np.column_stack((meta, projected)))


def fit_pca_projection(
    embeddings: object,
    *,
    dimension: int,
    fit_specimen_ids: tuple[str, ...],
    fit_domains: tuple[str, ...],
) -> PCAProjection:
    values = np.asarray(embeddings, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != len(fit_specimen_ids) != 0
        or values.shape[0] != len(fit_domains)
        or not np.all(np.isfinite(values))
    ):
        raise CAIEvaluatorError("PCA fit arrays are invalid")
    if type(dimension) is not int or not 0 < dimension <= min(
        values.shape[0] - 1, values.shape[1]
    ):
        raise CAIEvaluatorError("PCA dimension is invalid")
    mean = np.mean(values, axis=0, dtype=np.float64)
    try:
        _left, singular, right = np.linalg.svd(values - mean, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise CAIEvaluatorError("PCA fit failed") from error
    tolerance = max(values.shape) * np.finfo(np.float64).eps * float(singular[0])
    if np.count_nonzero(singular > tolerance) < dimension:
        raise CAIEvaluatorError("PCA training rank is insufficient")
    components = np.asarray(right[:dimension], dtype=np.float64).copy()
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    frozen_mean = _readonly(mean)
    frozen_components = _readonly(components)
    state = _state(
        "mva-pca",
        frozen_mean,
        frozen_components,
        fit_specimen_ids,
        fit_domains,
    )
    return PCAProjection(
        mean=frozen_mean,
        components=frozen_components,
        fit_specimen_ids=fit_specimen_ids,
        fit_domains=fit_domains,
        state_sha256=state,
    )


def fit_cai_predictor(
    *,
    method: str,
    outer_domain: str,
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    targets: np.ndarray,
    metadata: np.ndarray,
    embeddings: np.ndarray,
    dimension: int,
    fit_indices: np.ndarray,
    ridge_alpha: float,
) -> CAIPredictor:
    indices = np.asarray(fit_indices, dtype=np.int64)
    fit_ids = tuple(specimen_ids[index] for index in indices)
    fit_domains = tuple(dataset_ids[index] for index in indices)
    pca = fit_pca_projection(
        embeddings[indices],
        dimension=dimension,
        fit_specimen_ids=fit_ids,
        fit_domains=fit_domains,
    )
    train = np.column_stack((metadata[indices], pca.transform(embeddings[indices])))
    ridge = fit_fold_ridge(
        train,
        targets[indices],
        alpha=ridge_alpha,
        fit_sample_ids=fit_ids,
        fit_domain_ids=fit_domains,
    )
    ordered_domains = tuple(dict.fromkeys(fit_domains))
    state = _state(
        "mva-cai-predictor",
        method,
        outer_domain,
        pca.state_sha256,
        ridge.state_sha256,
        fit_ids,
        fit_domains,
    )
    return CAIPredictor(
        method=method,
        outer_domain=outer_domain,
        pca=pca,
        ridge=ridge,
        metadata_features=int(metadata.shape[1]),
        fit_specimen_ids=fit_ids,
        fit_domains=ordered_domains,
        state_sha256=state,
    )


def fit_sensitivity_cai_predictor(
    *,
    method: str,
    outer_domain: str,
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    targets: np.ndarray,
    metadata: np.ndarray,
    embeddings: np.ndarray,
    dimension: int,
    fit_indices: np.ndarray,
    ridge_alpha: float,
) -> CAIPredictor:
    """Fit one preregistered nonselecting Ridge-alpha sensitivity model."""

    alpha = float(ridge_alpha)
    if alpha not in {1.0, 100.0}:
        raise CAIEvaluatorError("registered sensitivity Ridge alpha must be 1 or 100")
    indices = np.asarray(fit_indices, dtype=np.int64)
    y = np.asarray(targets, dtype=np.float64)
    meta = np.asarray(metadata, dtype=np.float64)
    values = np.asarray(embeddings, dtype=np.float64)
    count = len(specimen_ids)
    if (
        indices.ndim != 1
        or indices.size < 2
        or len(np.unique(indices)) != indices.size
        or int(np.min(indices)) < 0
        or int(np.max(indices)) >= count
        or len(dataset_ids) != count
        or y.shape != (count,)
        or meta.ndim != 2
        or meta.shape[0] != count
        or values.ndim != 2
        or values.shape[0] != count
        or not np.all(np.isfinite(y))
        or np.any(np.isinf(meta))
        or not np.all(np.isfinite(values))
    ):
        raise CAIEvaluatorError("sensitivity fit arrays are invalid")
    fit_ids = tuple(specimen_ids[index] for index in indices)
    fit_domains = tuple(dataset_ids[index] for index in indices)
    pca = fit_pca_projection(
        values[indices],
        dimension=dimension,
        fit_specimen_ids=fit_ids,
        fit_domains=fit_domains,
    )
    train = np.column_stack((meta[indices], pca.transform(values[indices])))
    preprocessor = fit_fold_preprocessor(
        train,
        fit_sample_ids=fit_ids,
        fit_domain_ids=fit_domains,
    )
    transformed = preprocessor.transform(train)
    x_mean = np.mean(transformed, axis=0, dtype=np.float64)
    response = y[indices]
    y_mean = float(np.mean(response, dtype=np.float64))
    centered_x = transformed - x_mean
    centered_y = response - y_mean
    regularized = centered_x.T @ centered_x + alpha * np.eye(
        train.shape[1], dtype=np.float64
    )
    rhs = centered_x.T @ centered_y
    try:
        coefficients = np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(regularized, rhs, rcond=None)[0]
    intercept = y_mean - float(x_mean @ coefficients)
    if not np.all(np.isfinite(coefficients)) or not np.isfinite(intercept):
        raise CAIEvaluatorError("sensitivity Ridge fit is nonfinite")
    frozen_coefficients = _readonly(coefficients)
    ridge_state = _state(
        "mva-sensitivity-ridge",
        preprocessor.state_sha256,
        alpha,
        frozen_coefficients,
        intercept,
        fit_ids,
        fit_domains,
    )
    ridge = SensitivityRidgeModel(
        preprocessor=preprocessor,
        alpha=alpha,
        coef_=frozen_coefficients,
        intercept_=intercept,
        fit_sample_ids=fit_ids,
        fit_domain_ids=fit_domains,
        state_sha256=ridge_state,
    )
    state = _state(
        "mva-cai-sensitivity-predictor",
        method,
        outer_domain,
        pca.state_sha256,
        ridge.state_sha256,
        fit_ids,
        fit_domains,
    )
    return CAIPredictor(
        method=method,
        outer_domain=outer_domain,
        pca=pca,
        ridge=ridge,
        metadata_features=int(meta.shape[1]),
        fit_specimen_ids=fit_ids,
        fit_domains=tuple(dict.fromkeys(fit_domains)),
        state_sha256=state,
    )


__all__ = [
    "CAIEvaluatorError",
    "CAIPredictor",
    "PCAProjection",
    "SensitivityRidgeModel",
    "fit_cai_predictor",
    "fit_pca_projection",
    "fit_sensitivity_cai_predictor",
]
