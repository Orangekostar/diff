from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.damage_response.p2_features import (
    PROFILE_STAT_NAMES,
    SCALAR_DAMAGE_NAMES,
    P2FeatureAuthority,
)
from cmc_bbdm.damage_response.sources import (
    IMPACTOR_CATEGORIES,
    LAMINATE_CATEGORIES,
)

P2_VIEW_FIELDS = {
    "F0": ("laminate_type", "ply_count", "width_mm", "thickness_mm"),
    "F1": ("F0", "surface_profile_stats21"),
    "F2": ("F0", *SCALAR_DAMAGE_NAMES),
    "F3": ("F0", "full_cscan_embedding512"),
    "F4": ("F0", "surface_profile_stats21", "full_cscan_embedding512"),
    "F5": (
        "F4",
        "privileged_total_impact_energy_j",
        "privileged_impactor",
    ),
}
DEPLOYABLE_P2_VIEWS = ("F0", "F1", "F2", "F3", "F4")
PRIVILEGED_P2_VIEWS = ("F5",)
EMBEDDING_P2_VIEWS = ("F3", "F4", "F5")
FORBIDDEN_P2_INPUT_FIELDS = (
    "published_cai_strength_mpa",
    "true_cai_strength",
    "true_peak_strength",
    "raw_cai_trace",
    "true_cai_trace",
    "derived_response",
    "response_curve",
    "extension_peak_mm",
    "slope_u20_u60_mpa_per_mm",
    "normalized_prepeak_auc",
    "post_cai_image",
)

_BASE_NUMERIC_NAMES = ("ply_count", "width_mm", "thickness_mm")
_LAMINATE_NAMES = tuple(
    f"laminate_type={category}" for category in LAMINATE_CATEGORIES
)
_IMPACTOR_NAMES = tuple(
    f"impactor={category}" for category in IMPACTOR_CATEGORIES
)


class P2ViewError(ValueError):
    """Raised when a P2 view or fold-local transform violates its contract."""


def validate_p2_view(view_name: str, fields: Iterable[str]) -> tuple[str, ...]:
    """Require exact symbolic membership for one registered P2 view."""

    expected = P2_VIEW_FIELDS.get(view_name)
    if expected is None:
        raise P2ViewError(f"unknown P2 view: {view_name!r}")
    observed = tuple(fields)
    forbidden = set(observed) & set(FORBIDDEN_P2_INPUT_FIELDS)
    if forbidden:
        raise P2ViewError(f"P2 view contains outcome fields: {sorted(forbidden)!r}")
    if view_name in DEPLOYABLE_P2_VIEWS and set(observed) & {
        "privileged_total_impact_energy_j",
        "privileged_impactor",
    }:
        raise P2ViewError("deployable P2 view contains privileged impact context")
    if observed != expected:
        raise P2ViewError(f"P2 view {view_name} must equal {expected!r}")
    return observed


def _pca_dimension(view_name: str, value: int | None) -> int | None:
    if view_name in EMBEDDING_P2_VIEWS:
        if type(value) is not int or value < 1:
            raise P2ViewError("PCA dimension is required for embedding views")
        return value
    if value is not None:
        raise P2ViewError("PCA dimension is forbidden for nonembedding views")
    return None


def _pca_names(dimension: int | None) -> tuple[str, ...]:
    if dimension is None:
        return ()
    return tuple(f"full_cscan_pca_{index:02d}" for index in range(dimension))


def _numeric_names(view_name: str, pca_dimension: int | None) -> tuple[str, ...]:
    names: tuple[str, ...] = _BASE_NUMERIC_NAMES
    if view_name in {"F1", "F4", "F5"}:
        names = (*names, *PROFILE_STAT_NAMES)
    elif view_name == "F2":
        names = (*names, *SCALAR_DAMAGE_NAMES)
    if view_name in EMBEDDING_P2_VIEWS:
        names = (*names, *_pca_names(pca_dimension))
    if view_name == "F5":
        names = (*names, "privileged_total_impact_energy_j")
    return names


def _categorical_names(view_name: str) -> tuple[str, ...]:
    if view_name == "F5":
        return (*_LAMINATE_NAMES, *_IMPACTOR_NAMES)
    return _LAMINATE_NAMES


def view_feature_names(
    view_name: str, pca_dimension: int | None
) -> tuple[str, ...]:
    validate_p2_view(view_name, P2_VIEW_FIELDS.get(view_name, ()))
    dimension = _pca_dimension(view_name, pca_dimension)
    return (*_numeric_names(view_name, dimension), *_categorical_names(view_name))


def _indices(value: object, *, rows: int, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
        raise P2ViewError(f"{label} must be one-dimensional integer indices")
    result = np.asarray(raw, dtype=np.int64)
    if (
        len(result) == 0
        or np.any(result < 0)
        or np.any(result >= rows)
        or len(np.unique(result)) != len(result)
    ):
        raise P2ViewError(f"{label} are empty, duplicate, or out of range")
    return result


def _readonly(value: np.ndarray, *, dtype: str = "<f8") -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _roster_sha256(authority: P2FeatureAuthority) -> str:
    payload = {
        "domains": authority.domain_ids,
        "embedding_state_sha256": authority.embedding_state_sha256,
        "encoder_sha256": authority.encoder_sha256,
        "sources": dict(authority.source_sha256),
        "specimens": authority.specimen_ids,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _fit_pca(
    embeddings: np.ndarray, fit: np.ndarray, dimension: int
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(embeddings, dtype=np.float64)
    maximum = min(len(fit) - 1, matrix.shape[1])
    if dimension > maximum:
        raise P2ViewError(
            f"PCA dimension {dimension} exceeds source-fold rank bound {maximum}"
        )
    mean = np.mean(matrix[fit], axis=0, dtype=np.float64)
    centered = matrix[fit] - mean
    try:
        _left, singular_values, right = np.linalg.svd(
            centered, full_matrices=False
        )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
        raise P2ViewError("fold-local embedding PCA failed") from error
    if not np.all(np.isfinite(singular_values)) or not np.all(np.isfinite(right)):
        raise P2ViewError("fold-local embedding PCA returned nonfinite values")
    components = np.asarray(right[:dimension], dtype=np.float64).copy()
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return mean, components


def _base_numeric(authority: P2FeatureAuthority) -> np.ndarray:
    return np.column_stack(
        (authority.ply_counts, authority.widths_mm, authority.thicknesses_mm)
    ).astype(np.float64, copy=False)


def _numeric_matrix(
    authority: P2FeatureAuthority,
    view_name: str,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
) -> np.ndarray:
    blocks = [_base_numeric(authority)]
    if view_name in {"F1", "F4", "F5"}:
        blocks.append(np.asarray(authority.surface_profile_stats, dtype=np.float64))
    elif view_name == "F2":
        blocks.append(np.asarray(authority.scalar_damage, dtype=np.float64))
    if view_name in EMBEDDING_P2_VIEWS:
        embeddings = np.asarray(authority.full_cscan_embedding, dtype=np.float64)
        blocks.append((embeddings - pca_mean) @ pca_components.T)
    if view_name == "F5":
        blocks.append(
            np.asarray(authority.privileged_total_energy_j, dtype=np.float64)[:, None]
        )
    result = np.column_stack(blocks)
    if np.any(np.isinf(result)):
        raise P2ViewError("P2 numeric features contain infinite values")
    return result


def _categorical_matrix(
    authority: P2FeatureAuthority, view_name: str
) -> np.ndarray:
    laminate = np.asarray(
        [
            [float(value == category) for category in LAMINATE_CATEGORIES]
            for value in authority.laminate_types
        ],
        dtype=np.float64,
    )
    if view_name != "F5":
        return laminate
    impactor = np.asarray(
        [
            [float(value == category) for category in IMPACTOR_CATEGORIES]
            for value in authority.privileged_impactors
        ],
        dtype=np.float64,
    )
    return np.column_stack((laminate, impactor))


def _state_sha256(
    *,
    view_name: str,
    pca_dimension: int | None,
    roster_sha256: str,
    fit_indices: tuple[int, ...],
    fit_specimen_ids: tuple[str, ...],
    numeric_feature_names: tuple[str, ...],
    categorical_feature_names: tuple[str, ...],
    imputer_statistics: np.ndarray,
    numeric_means: np.ndarray,
    numeric_scales: np.ndarray,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
) -> str:
    metadata = {
        "categorical_feature_names": categorical_feature_names,
        "fit_indices": fit_indices,
        "fit_specimen_ids": fit_specimen_ids,
        "numeric_feature_names": numeric_feature_names,
        "pca_dimension": pca_dimension,
        "roster_sha256": roster_sha256,
        "view_name": view_name,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    for value in (
        imputer_statistics,
        numeric_means,
        numeric_scales,
        pca_mean,
        pca_components,
    ):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class P2FoldPreprocessor:
    view_name: str
    pca_dimension: int | None
    roster_sha256: str
    fit_indices: tuple[int, ...]
    fit_specimen_ids: tuple[str, ...]
    numeric_feature_names: tuple[str, ...]
    categorical_feature_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    imputer_statistics: np.ndarray
    numeric_means: np.ndarray
    numeric_scales: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    state_sha256: str


def fit_p2_preprocessor(
    authority: P2FeatureAuthority,
    view_name: str,
    fit_indices: object,
    *,
    pca_dimension: int | None,
) -> P2FoldPreprocessor:
    """Fit one source-only feature transform without accepting response data."""

    if not isinstance(authority, P2FeatureAuthority):
        raise P2ViewError("P2 feature authority type changed")
    validate_p2_view(view_name, P2_VIEW_FIELDS.get(view_name, ()))
    dimension = _pca_dimension(view_name, pca_dimension)
    fit = _indices(fit_indices, rows=len(authority.specimen_ids), label="fit indices")
    if dimension is None:
        pca_mean = np.empty(0, dtype=np.float64)
        pca_components = np.empty((0, 512), dtype=np.float64)
    else:
        pca_mean, pca_components = _fit_pca(
            authority.full_cscan_embedding, fit, dimension
        )
    numeric = _numeric_matrix(authority, view_name, pca_mean, pca_components)
    source = numeric[fit]
    with np.errstate(all="ignore"):
        imputer = np.nanmedian(source, axis=0)
    if not np.all(np.isfinite(imputer)):
        raise P2ViewError("every fitted numeric feature requires a finite value")
    filled = np.where(np.isnan(source), imputer, source)
    means = np.mean(filled, axis=0, dtype=np.float64)
    scales = np.std(filled, axis=0, ddof=0, dtype=np.float64)
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)):
        raise P2ViewError("fold-local numeric scaling is nonfinite")
    scales[scales <= np.finfo(np.float64).eps] = 1.0

    immutable_imputer = _readonly(imputer)
    immutable_means = _readonly(means)
    immutable_scales = _readonly(scales)
    immutable_pca_mean = _readonly(pca_mean)
    immutable_components = _readonly(pca_components)
    numeric_names = _numeric_names(view_name, dimension)
    categorical_names = _categorical_names(view_name)
    if len(numeric_names) != len(imputer):
        raise P2ViewError("P2 numeric feature-name registry differs")
    fit_ids = tuple(authority.specimen_ids[int(index)] for index in fit)
    roster_sha256 = _roster_sha256(authority)
    state_sha256 = _state_sha256(
        view_name=view_name,
        pca_dimension=dimension,
        roster_sha256=roster_sha256,
        fit_indices=tuple(int(index) for index in fit),
        fit_specimen_ids=fit_ids,
        numeric_feature_names=numeric_names,
        categorical_feature_names=categorical_names,
        imputer_statistics=immutable_imputer,
        numeric_means=immutable_means,
        numeric_scales=immutable_scales,
        pca_mean=immutable_pca_mean,
        pca_components=immutable_components,
    )
    return P2FoldPreprocessor(
        view_name=view_name,
        pca_dimension=dimension,
        roster_sha256=roster_sha256,
        fit_indices=tuple(int(index) for index in fit),
        fit_specimen_ids=fit_ids,
        numeric_feature_names=numeric_names,
        categorical_feature_names=categorical_names,
        feature_names=(*numeric_names, *categorical_names),
        imputer_statistics=immutable_imputer,
        numeric_means=immutable_means,
        numeric_scales=immutable_scales,
        pca_mean=immutable_pca_mean,
        pca_components=immutable_components,
        state_sha256=state_sha256,
    )


def transform_p2_view(
    authority: P2FeatureAuthority,
    preprocessor: P2FoldPreprocessor,
    indices: object,
) -> np.ndarray:
    """Apply a registered source-fitted transform to explicit specimen rows."""

    if not isinstance(authority, P2FeatureAuthority) or not isinstance(
        preprocessor, P2FoldPreprocessor
    ):
        raise P2ViewError("P2 transform authority or state type changed")
    if _roster_sha256(authority) != preprocessor.roster_sha256:
        raise P2ViewError("P2 transform roster differs from fitted state")
    query = _indices(indices, rows=len(authority.specimen_ids), label="query indices")
    numeric = _numeric_matrix(
        authority,
        preprocessor.view_name,
        preprocessor.pca_mean,
        preprocessor.pca_components,
    )[query]
    filled = np.where(np.isnan(numeric), preprocessor.imputer_statistics, numeric)
    scaled = (filled - preprocessor.numeric_means) / preprocessor.numeric_scales
    categorical = _categorical_matrix(authority, preprocessor.view_name)[query]
    result = np.column_stack((scaled, categorical))
    if (
        result.shape != (len(query), len(preprocessor.feature_names))
        or not np.all(np.isfinite(result))
    ):
        raise P2ViewError("P2 transformed feature matrix is invalid")
    return _readonly(result)
