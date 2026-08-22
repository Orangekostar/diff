"""Deterministic nested leave-one-domain-out evaluation for MGMR."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from cmc_bbdm.cpb_v3.models import fit_fold_ridge


class MGMREvaluationError(ValueError):
    """Raised when an MGMR evaluation request violates the frozen protocol."""


def _readonly(value: object) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )
    output.setflags(write=False)
    return output


def _hash_state(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return digest.hexdigest()


def _ids(value: Sequence[str], length: int, label: str, *, unique: bool) -> tuple[str, ...]:
    output = tuple(value)
    if (
        len(output) != length
        or any(type(item) is not str or not item for item in output)
        or (unique and len(set(output)) != length)
    ):
        raise MGMREvaluationError(f"{label} are invalid or misaligned")
    return output


def _matrix(value: object, rows: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MGMREvaluationError(f"{label} must be numeric") from error
    if array.ndim != 2 or array.shape[0] != rows or not np.all(np.isfinite(array)):
        raise MGMREvaluationError(f"{label} must be a finite aligned matrix")
    return array


def _vector(value: object, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MGMREvaluationError(f"{label} must be numeric") from error
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise MGMREvaluationError(f"{label} must be a finite vector")
    return array


def _ordered_unique(values: Sequence[str], order: Sequence[str]) -> tuple[str, ...]:
    present = set(values)
    return tuple(item for item in order if item in present)


@dataclass(frozen=True, slots=True)
class FoldPCA:
    mean: np.ndarray
    components: np.ndarray
    fit_specimen_ids: tuple[str, ...]
    fit_domains: tuple[str, ...]
    state_sha256: str

    @property
    def dimension(self) -> int:
        return int(self.components.shape[0])

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean.size:
            raise MGMREvaluationError("PCA query feature count changed")
        output = (array - self.mean) @ self.components.T
        if not np.all(np.isfinite(output)):
            raise MGMREvaluationError("PCA transform produced non-finite values")
        return output


@dataclass(frozen=True, slots=True)
class FitRecord:
    stage: str
    method: str
    outer_domain: str
    query_domains: tuple[str, ...]
    fit_domains: tuple[str, ...]
    query_specimen_ids: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    dimensions: tuple[int, ...]
    ridge_feature_count: int
    pca_state_sha256: tuple[str, ...]
    ridge_state_sha256: str


@dataclass(frozen=True, slots=True)
class InnerScore:
    method: str
    outer_domain: str
    query_domain: str
    dimensions: tuple[int, ...]
    mae: float


@dataclass(frozen=True, slots=True)
class DimensionSelection:
    method: str
    outer_domain: str
    dimensions: tuple[int, ...]
    mean_inner_mae: float


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    method: str
    specimen_id: str
    dataset_id: str
    target: float
    prediction: float
    dimensions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NestedLODORun:
    method: str
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    predictions: np.ndarray
    records: tuple[PredictionRecord, ...]
    selections: tuple[DimensionSelection, ...]
    selection_by_domain: Mapping[str, DimensionSelection]
    inner_scores: tuple[InnerScore, ...]
    fit_records: tuple[FitRecord, ...]
    state_sha256: str


def _fit_pca(
    values: np.ndarray,
    dimension: int,
    fit_specimen_ids: tuple[str, ...],
    fit_domains: tuple[str, ...],
) -> FoldPCA:
    if dimension <= 0 or dimension > min(values.shape[0] - 1, values.shape[1]):
        raise MGMREvaluationError("PCA dimension is invalid for a training fold")
    mean = np.mean(values, axis=0, dtype=np.float64)
    centered = values - mean
    try:
        _left, _singular, right = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise MGMREvaluationError("PCA SVD did not converge") from error
    components = np.asarray(right[:dimension], dtype=np.float64).copy()
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    mean = _readonly(mean)
    components = _readonly(components)
    state = _hash_state(
        "fold-pca", mean, components, fit_specimen_ids, fit_domains
    )
    return FoldPCA(
        mean=mean,
        components=components,
        fit_specimen_ids=fit_specimen_ids,
        fit_domains=fit_domains,
        state_sha256=state,
    )


def _slice_pca(model: FoldPCA, dimension: int) -> FoldPCA:
    if dimension <= 0 or dimension > model.dimension:
        raise MGMREvaluationError("PCA slice dimension is invalid")
    if dimension == model.dimension:
        return model
    components = _readonly(model.components[:dimension])
    state = _hash_state(
        "fold-pca",
        model.mean,
        components,
        model.fit_specimen_ids,
        model.fit_domains,
    )
    return FoldPCA(
        mean=model.mean,
        components=components,
        fit_specimen_ids=model.fit_specimen_ids,
        fit_domains=model.fit_domains,
        state_sha256=state,
    )


def _validate_request(
    *,
    method: str,
    metadata: object,
    blocks: Mapping[str, object],
    targets: object,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    domain_order: Sequence[str],
    pca_dimensions: Sequence[int],
    ridge_alpha: float,
    tie_tolerance: float,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    if type(method) is not str or not method:
        raise MGMREvaluationError("method is required")
    y = _vector(targets, "targets")
    rows = y.size
    samples = _ids(specimen_ids, rows, "specimen IDs", unique=True)
    datasets = _ids(dataset_ids, rows, "dataset IDs", unique=False)
    domains = tuple(domain_order)
    if (
        len(domains) < 3
        or len(set(domains)) != len(domains)
        or any(type(item) is not str or not item for item in domains)
        or set(datasets) != set(domains)
    ):
        raise MGMREvaluationError("domain order is incomplete or invalid")
    meta = _matrix(metadata, rows, "metadata")
    if not isinstance(blocks, Mapping) or not blocks:
        raise MGMREvaluationError("at least one PCA feature block is required")
    matrices: dict[str, np.ndarray] = {}
    for name, value in blocks.items():
        if type(name) is not str or not name:
            raise MGMREvaluationError("feature block names are invalid")
        matrices[name] = _matrix(value, rows, f"feature block {name}")
    dimensions = tuple(pca_dimensions)
    if (
        not dimensions
        or len(set(dimensions)) != len(dimensions)
        or tuple(sorted(dimensions)) != dimensions
        or any(type(item) is not int or item <= 0 for item in dimensions)
    ):
        raise MGMREvaluationError("PCA dimensions must be unique increasing integers")
    if isinstance(ridge_alpha, bool) or float(ridge_alpha) != 10.0:
        raise MGMREvaluationError("ridge alpha must equal 10.0")
    if (
        isinstance(tie_tolerance, bool)
        or not math.isfinite(float(tie_tolerance))
        or float(tie_tolerance) < 0.0
    ):
        raise MGMREvaluationError("tie tolerance must be finite and non-negative")
    return meta, matrices, y, samples, datasets, domains, dimensions


def nested_lodo_predictions(
    *,
    method: str,
    metadata: object,
    blocks: Mapping[str, object],
    targets: object,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    domain_order: Sequence[str],
    pca_dimensions: Sequence[int],
    ridge_alpha: float,
    tie_tolerance: float,
) -> NestedLODORun:
    """Run deterministic nested LODO with independently fitted PCA blocks."""

    meta, matrices, y, samples, datasets, domains, dimensions = _validate_request(
        method=method,
        metadata=metadata,
        blocks=blocks,
        targets=targets,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domain_order,
        pca_dimensions=pca_dimensions,
        ridge_alpha=ridge_alpha,
        tie_tolerance=tie_tolerance,
    )
    dataset_array = np.asarray(datasets, dtype=object)
    predictions = np.full(y.size, np.nan, dtype=np.float64)
    inner_rows: list[InnerScore] = []
    fit_rows: list[FitRecord] = []
    selection_rows: list[DimensionSelection] = []
    combinations = tuple(itertools.product(dimensions, repeat=len(matrices)))
    pca_cache: dict[tuple[str, tuple[str, ...]], FoldPCA] = {}

    def fit_predict(
        fit_indices: np.ndarray,
        query_indices: np.ndarray,
        selected: tuple[int, ...],
        *,
        stage: str,
        outer_domain: str,
    ) -> np.ndarray:
        fit_ids = tuple(samples[index] for index in fit_indices)
        query_ids = tuple(samples[index] for index in query_indices)
        fit_domains = _ordered_unique(dataset_array[fit_indices].tolist(), domains)
        query_domains = _ordered_unique(dataset_array[query_indices].tolist(), domains)
        fit_parts = [meta[fit_indices]]
        query_parts = [meta[query_indices]]
        pca_states: list[str] = []
        for (name, values), dimension in zip(matrices.items(), selected, strict=True):
            cache_key = (name, fit_ids)
            maximum = pca_cache.get(cache_key)
            if maximum is None:
                maximum = _fit_pca(
                    values[fit_indices], max(dimensions), fit_ids, fit_domains
                )
                pca_cache[cache_key] = maximum
            pca = _slice_pca(maximum, dimension)
            fit_parts.append(pca.transform(values[fit_indices]))
            query_parts.append(pca.transform(values[query_indices]))
            pca_states.append(pca.state_sha256)
        train_x = np.concatenate(fit_parts, axis=1)
        query_x = np.concatenate(query_parts, axis=1)
        ridge = fit_fold_ridge(
            train_x,
            y[fit_indices],
            alpha=float(ridge_alpha),
            fit_sample_ids=fit_ids,
            fit_domain_ids=tuple(datasets[index] for index in fit_indices),
        )
        output = ridge.predict(query_x)
        fit_rows.append(
            FitRecord(
                stage=stage,
                method=method,
                outer_domain=outer_domain,
                query_domains=query_domains,
                fit_domains=fit_domains,
                query_specimen_ids=query_ids,
                fit_specimen_ids=fit_ids,
                dimensions=selected,
                ridge_feature_count=int(train_x.shape[1]),
                pca_state_sha256=tuple(pca_states),
                ridge_state_sha256=ridge.state_sha256,
            )
        )
        return output

    for outer_domain in domains:
        outer_query = np.flatnonzero(dataset_array == outer_domain)
        outer_train = np.flatnonzero(dataset_array != outer_domain)
        scores: dict[tuple[int, ...], float] = {}
        for selected in combinations:
            fold_scores: list[float] = []
            for inner_domain in domains:
                if inner_domain == outer_domain:
                    continue
                inner_query = np.flatnonzero(dataset_array == inner_domain)
                inner_fit = np.flatnonzero(
                    (dataset_array != outer_domain) & (dataset_array != inner_domain)
                )
                prediction = fit_predict(
                    inner_fit,
                    inner_query,
                    selected,
                    stage="inner",
                    outer_domain=outer_domain,
                )
                mae = float(np.mean(np.abs(y[inner_query] - prediction), dtype=np.float64))
                inner_rows.append(
                    InnerScore(
                        method=method,
                        outer_domain=outer_domain,
                        query_domain=inner_domain,
                        dimensions=selected,
                        mae=mae,
                    )
                )
                fold_scores.append(mae)
            scores[selected] = float(math.fsum(fold_scores) / len(fold_scores))
        selected = combinations[0]
        selected_score = scores[selected]
        for candidate in combinations[1:]:
            candidate_score = scores[candidate]
            if candidate_score < selected_score - float(tie_tolerance) or (
                abs(candidate_score - selected_score) <= float(tie_tolerance)
                and (sum(candidate), candidate) < (sum(selected), selected)
            ):
                selected, selected_score = candidate, candidate_score
        selection_rows.append(
            DimensionSelection(
                method=method,
                outer_domain=outer_domain,
                dimensions=selected,
                mean_inner_mae=selected_score,
            )
        )
        predictions[outer_query] = fit_predict(
            outer_train,
            outer_query,
            selected,
            stage="outer",
            outer_domain=outer_domain,
        )

    if not np.all(np.isfinite(predictions)):
        raise MGMREvaluationError("each specimen must receive one finite prediction")
    selections = tuple(selection_rows)
    by_domain = MappingProxyType({row.outer_domain: row for row in selections})
    records = tuple(
        PredictionRecord(
            method=method,
            specimen_id=samples[index],
            dataset_id=datasets[index],
            target=float(y[index]),
            prediction=float(predictions[index]),
            dimensions=by_domain[datasets[index]].dimensions,
        )
        for index in range(y.size)
    )
    immutable_predictions = _readonly(predictions)
    state = _hash_state(
        "nested-lodo",
        method,
        samples,
        datasets,
        immutable_predictions,
        [(row.outer_domain, row.dimensions, row.mean_inner_mae) for row in selections],
    )
    return NestedLODORun(
        method=method,
        specimen_ids=samples,
        dataset_ids=datasets,
        predictions=immutable_predictions,
        records=records,
        selections=selections,
        selection_by_domain=by_domain,
        inner_scores=tuple(inner_rows),
        fit_records=tuple(fit_rows),
        state_sha256=state,
    )


__all__ = [
    "DimensionSelection",
    "FitRecord",
    "FoldPCA",
    "InnerScore",
    "MGMREvaluationError",
    "NestedLODORun",
    "PredictionRecord",
    "nested_lodo_predictions",
]
