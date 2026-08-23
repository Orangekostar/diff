"""Strict source-only cross-fitting and initial-survey selection for MVA."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .cai_evaluator import CAIPredictor, fit_cai_predictor


class CrossfitError(ValueError):
    """Raised when a cross-fitting or source-only selection contract fails."""


@dataclass(frozen=True, slots=True)
class FitAudit:
    stage: str
    method: str
    outer_domain: str
    query_domains: tuple[str, ...]
    fit_domains: tuple[str, ...]
    query_specimen_ids: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    pca_dimension: int
    predictor_state_sha256: str


@dataclass(frozen=True, slots=True)
class OuterPredictorFit:
    model: CAIPredictor
    selected_pca_dimension: int
    inner_dimension_mae: tuple[tuple[int, float], ...]
    selected_inner_domain_mae: tuple[tuple[str, float], ...]
    fit_audits: tuple[FitAudit, ...]


@dataclass(frozen=True, slots=True)
class CrossFittedEvaluator:
    method: str
    models: Mapping[str, CAIPredictor]
    selected_pca_dimensions: tuple[int, ...]
    predictions: np.ndarray
    fit_audits: tuple[FitAudit, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class InitialSurveySelection:
    outer_domain: str
    selected_budget: float
    status: str
    source_full_mae: float
    source_candidate_mae: tuple[tuple[float, float], ...]


def _state(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(value.dtype.str.encode("ascii"))
            digest.update(
                json.dumps(value.shape, separators=(",", ":")).encode("ascii")
            )
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
    return digest.hexdigest()


def _validate_arrays(
    *,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    domain_order: Sequence[str],
    targets: object,
    metadata: object,
    embeddings: object,
    pca_dimensions: Sequence[int],
    ridge_alpha: float,
    tie_tolerance: float,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
]:
    samples = tuple(specimen_ids)
    datasets = tuple(dataset_ids)
    domains = tuple(domain_order)
    try:
        y = np.asarray(targets, dtype=np.float64)
        meta = np.asarray(metadata, dtype=np.float64)
        values = np.asarray(embeddings, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CrossfitError("crossfit arrays must be numeric") from error
    count = len(samples)
    dimensions = tuple(pca_dimensions)
    if (
        count == 0
        or len(set(samples)) != count
        or len(datasets) != count
        or len(domains) < 3
        or len(set(domains)) != len(domains)
        or set(datasets) != set(domains)
        or y.shape != (count,)
        or meta.ndim != 2
        or meta.shape[0] != count
        or values.ndim != 2
        or values.shape[0] != count
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(values))
        or np.any(np.isinf(meta))
        or not dimensions
        or tuple(sorted(set(dimensions))) != dimensions
        or any(type(value) is not int or value <= 0 for value in dimensions)
        or float(ridge_alpha) != 10.0
        or not math.isfinite(float(tie_tolerance))
        or float(tie_tolerance) < 0.0
    ):
        raise CrossfitError("crossfit arrays or protocol are invalid")
    return samples, datasets, domains, y, meta, values, dimensions


def fit_outer_source_predictor(
    *,
    method: str,
    outer_domain: str,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    domain_order: Sequence[str],
    targets: object,
    metadata: object,
    embeddings: object,
    pca_dimensions: Sequence[int],
    ridge_alpha: float,
    tie_tolerance: float,
) -> OuterPredictorFit:
    """Select and fit one predictor without using any outer-domain row."""

    samples, datasets, domains, y, meta, values, dimensions = _validate_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domain_order,
        targets=targets,
        metadata=metadata,
        embeddings=embeddings,
        pca_dimensions=pca_dimensions,
        ridge_alpha=ridge_alpha,
        tie_tolerance=tie_tolerance,
    )
    if outer_domain not in domains or type(method) is not str or not method:
        raise CrossfitError("outer domain or method is invalid")
    dataset_array = np.asarray(datasets, dtype=object)
    source_domains = tuple(domain for domain in domains if domain != outer_domain)
    audits: list[FitAudit] = []
    scores: dict[int, float] = {}
    per_dimension_domain: dict[int, tuple[tuple[str, float], ...]] = {}
    for dimension in dimensions:
        domain_scores: list[float] = []
        for inner_domain in source_domains:
            fit_indices = np.flatnonzero(
                (dataset_array != outer_domain) & (dataset_array != inner_domain)
            )
            query_indices = np.flatnonzero(dataset_array == inner_domain)
            model = fit_cai_predictor(
                method=method,
                outer_domain=outer_domain,
                specimen_ids=samples,
                dataset_ids=datasets,
                targets=y,
                metadata=meta,
                embeddings=values,
                dimension=dimension,
                fit_indices=fit_indices,
                ridge_alpha=ridge_alpha,
            )
            prediction = model.predict(meta[query_indices], values[query_indices])
            domain_scores.append(
                float(np.mean(np.abs(y[query_indices] - prediction), dtype=np.float64))
            )
            audits.append(
                FitAudit(
                    stage="inner",
                    method=method,
                    outer_domain=outer_domain,
                    query_domains=(inner_domain,),
                    fit_domains=model.fit_domains,
                    query_specimen_ids=tuple(samples[index] for index in query_indices),
                    fit_specimen_ids=model.fit_specimen_ids,
                    pca_dimension=dimension,
                    predictor_state_sha256=model.state_sha256,
                )
            )
        scores[dimension] = float(sum(domain_scores) / len(domain_scores))
        per_dimension_domain[dimension] = tuple(
            zip(source_domains, domain_scores, strict=True)
        )
    selected = dimensions[0]
    selected_score = scores[selected]
    for dimension in dimensions[1:]:
        score = scores[dimension]
        if score < selected_score - float(tie_tolerance):
            selected, selected_score = dimension, score
    fit_indices = np.flatnonzero(dataset_array != outer_domain)
    query_indices = np.flatnonzero(dataset_array == outer_domain)
    model = fit_cai_predictor(
        method=method,
        outer_domain=outer_domain,
        specimen_ids=samples,
        dataset_ids=datasets,
        targets=y,
        metadata=meta,
        embeddings=values,
        dimension=selected,
        fit_indices=fit_indices,
        ridge_alpha=ridge_alpha,
    )
    audits.append(
        FitAudit(
            stage="outer",
            method=method,
            outer_domain=outer_domain,
            query_domains=(outer_domain,),
            fit_domains=model.fit_domains,
            query_specimen_ids=tuple(samples[index] for index in query_indices),
            fit_specimen_ids=model.fit_specimen_ids,
            pca_dimension=selected,
            predictor_state_sha256=model.state_sha256,
        )
    )
    return OuterPredictorFit(
        model=model,
        selected_pca_dimension=selected,
        inner_dimension_mae=tuple(
            (dimension, scores[dimension]) for dimension in dimensions
        ),
        selected_inner_domain_mae=per_dimension_domain[selected],
        fit_audits=tuple(audits),
    )


def fit_cross_fitted_evaluator(
    *,
    method: str,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    domain_order: Sequence[str],
    targets: object,
    metadata: object,
    embeddings: object,
    pca_dimensions: Sequence[int],
    ridge_alpha: float,
    tie_tolerance: float,
) -> CrossFittedEvaluator:
    """Fit one strict OOF predictor per domain and predict every specimen."""

    samples, datasets, domains, y, meta, values, _dimensions = _validate_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=domain_order,
        targets=targets,
        metadata=metadata,
        embeddings=embeddings,
        pca_dimensions=pca_dimensions,
        ridge_alpha=ridge_alpha,
        tie_tolerance=tie_tolerance,
    )
    dataset_array = np.asarray(datasets, dtype=object)
    predictions = np.full(len(samples), np.nan, dtype=np.float64)
    models: dict[str, CAIPredictor] = {}
    selected: list[int] = []
    audits: list[FitAudit] = []
    for outer_domain in domains:
        fitted = fit_outer_source_predictor(
            method=method,
            outer_domain=outer_domain,
            specimen_ids=samples,
            dataset_ids=datasets,
            domain_order=domains,
            targets=y,
            metadata=meta,
            embeddings=values,
            pca_dimensions=pca_dimensions,
            ridge_alpha=ridge_alpha,
            tie_tolerance=tie_tolerance,
        )
        query = np.flatnonzero(dataset_array == outer_domain)
        predictions[query] = fitted.model.predict(meta[query], values[query])
        models[outer_domain] = fitted.model
        selected.append(fitted.selected_pca_dimension)
        audits.extend(fitted.fit_audits)
    if not np.all(np.isfinite(predictions)):
        raise CrossfitError("cross-fitted prediction roster is incomplete")
    frozen = np.frombuffer(predictions.tobytes(order="C"), dtype=np.float64)
    frozen.setflags(write=False)
    state = _state(
        "mva-crossfit",
        method,
        tuple(model.state_sha256 for model in models.values()),
        tuple(selected),
        frozen,
    )
    return CrossFittedEvaluator(
        method=method,
        models=MappingProxyType(models),
        selected_pca_dimensions=tuple(selected),
        predictions=frozen,
        fit_audits=tuple(audits),
        state_sha256=state,
    )


def select_initial_survey(
    *,
    outer_domain: str,
    domain_order: Sequence[str],
    full_domain_mae: Mapping[str, float],
    candidate_domain_mae: Mapping[float, Mapping[str, float]],
) -> InitialSurveySelection:
    """Apply the frozen source-only initial-budget rule without target access."""

    domains = tuple(domain_order)
    if outer_domain not in domains:
        raise CrossfitError("outer domain is not registered")
    source_domains = tuple(domain for domain in domains if domain != outer_domain)
    budgets = tuple(sorted(float(value) for value in candidate_domain_mae))
    if budgets != (0.015625, 0.03125, 0.0625):
        raise CrossfitError("initial survey candidate set changed")
    try:
        full = float(
            sum(float(full_domain_mae[domain]) for domain in source_domains)
            / len(source_domains)
        )
        scores = {
            budget: float(
                sum(
                    float(candidate_domain_mae[budget][domain])
                    for domain in source_domains
                )
                / len(source_domains)
            )
            for budget in budgets
        }
    except (KeyError, TypeError, ValueError) as error:
        raise CrossfitError("initial survey scores are incomplete") from error
    if (
        not math.isfinite(full)
        or full <= 0.0
        or not all(math.isfinite(value) and value >= 0.0 for value in scores.values())
    ):
        raise CrossfitError("initial survey scores are invalid")
    upper = 1.5 * full
    lower = 1.025 * full
    selected = next(
        (budget for budget in budgets if lower <= scores[budget] <= upper), None
    )
    status = "selected"
    if selected is None:
        selected = next((budget for budget in budgets if scores[budget] <= upper), None)
        status = "weak_headroom"
    if selected is None:
        raise CrossfitError("no initial survey candidate retains the FULL upper bound")
    return InitialSurveySelection(
        outer_domain=outer_domain,
        selected_budget=selected,
        status=status,
        source_full_mae=full,
        source_candidate_mae=tuple((budget, scores[budget]) for budget in budgets),
    )


__all__ = [
    "CrossFittedEvaluator",
    "CrossfitError",
    "FitAudit",
    "InitialSurveySelection",
    "OuterPredictorFit",
    "fit_cross_fitted_evaluator",
    "fit_outer_source_predictor",
    "select_initial_survey",
]
