"""Strict source-only LODO redundancy evaluation for P1 response endpoints."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS
from cmc_bbdm.damage_response.feature_views import (
    DESIGN_FEATURE_NAMES,
    PRIMARY_TARGET_FIELDS,
    DesignMetadata,
    fit_fold_local_design_encoder,
)

RIDGE_ALPHA = 1e-6
REDUNDANCY_MODELS = ("strength_only", "strength_plus_design")
STRENGTH_FEATURE_NAMES = (
    "published_cai_strength_mpa^1",
    "published_cai_strength_mpa^2",
    "published_cai_strength_mpa^3",
)
DOMAIN_ORDER = tuple(PRIMARY_COUNTS)


class EvaluationError(RuntimeError):
    """Raised when strict P1 LODO evaluation cannot satisfy its contract."""


def _readonly(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype="<f8").reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class P1RedundancyRecord:
    specimen_id: str
    domain_id: str
    published_cai_strength_mpa: float
    extension_peak_mm: float
    slope_u20_u60_mpa_per_mm: float
    normalized_prepeak_auc: float
    design: DesignMetadata


@dataclass(frozen=True, slots=True)
class FoldState:
    held_out_domain: str
    endpoint: str
    model: str
    fit_domains: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    strength_means: np.ndarray
    strength_scales: np.ndarray
    design_encoder_sha256: str | None
    coefficients: np.ndarray
    intercept: float
    state_sha256: str


@dataclass(frozen=True, slots=True)
class OOFPrediction:
    specimen_id: str
    domain_id: str
    held_out_domain: str
    endpoint: str
    model: str
    truth: float
    prediction: float
    fold_state_sha256: str


@dataclass(frozen=True, slots=True)
class DomainSpearman:
    domain_id: str
    specimen_count: int
    spearman: float


@dataclass(frozen=True, slots=True)
class RedundancyMetric:
    endpoint: str
    model: str
    pooled_r2: float
    domain_spearman: tuple[DomainSpearman, ...]


@dataclass(frozen=True, slots=True)
class RedundancyEvaluation:
    predictions: tuple[OOFPrediction, ...]
    fold_states: tuple[FoldState, ...]
    metrics: tuple[RedundancyMetric, ...]


def _validate_records(
    records: tuple[P1RedundancyRecord, ...],
) -> tuple[P1RedundancyRecord, ...]:
    if not records:
        raise EvaluationError("P1 redundancy records are empty")
    specimen_ids: set[str] = set()
    observed_domains: set[str] = set()
    domain_rank = {domain: index for index, domain in enumerate(DOMAIN_ORDER)}
    for record in records:
        if type(record) is not P1RedundancyRecord:
            raise EvaluationError("P1 redundancy record type changed")
        specimen_id = record.specimen_id
        domain_id = record.domain_id
        if (
            not specimen_id
            or specimen_id != specimen_id.strip().casefold()
            or specimen_id in specimen_ids
        ):
            raise EvaluationError("P1 specimen identity is empty, noncanonical, or duplicate")
        if domain_id not in domain_rank:
            raise EvaluationError("P1 evaluation requires the six canonical domains")
        if (
            record.design.specimen_id != specimen_id
            or record.design.domain_id != domain_id
        ):
            raise EvaluationError(f"P1 design identity differs: {specimen_id}")
        numeric = (
            record.published_cai_strength_mpa,
            record.extension_peak_mm,
            record.slope_u20_u60_mpa_per_mm,
            record.normalized_prepeak_auc,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise EvaluationError(f"P1 redundancy value is invalid: {specimen_id}")
        if record.published_cai_strength_mpa <= 0.0 or record.extension_peak_mm <= 0.0:
            raise EvaluationError(f"P1 strength or extension is nonpositive: {specimen_id}")
        specimen_ids.add(specimen_id)
        observed_domains.add(domain_id)
    if observed_domains != set(DOMAIN_ORDER):
        raise EvaluationError("P1 evaluation requires the six canonical domains")
    counts = {
        domain: sum(record.domain_id == domain for record in records)
        for domain in DOMAIN_ORDER
    }
    if any(count < 2 for count in counts.values()):
        raise EvaluationError("each canonical domain needs at least two valid rows")
    return tuple(
        sorted(
            records,
            key=lambda record: (domain_rank[record.domain_id], record.specimen_id),
        )
    )


def _target(record: P1RedundancyRecord, endpoint: str) -> float:
    if endpoint not in PRIMARY_TARGET_FIELDS:
        raise EvaluationError(f"unknown P1 endpoint: {endpoint!r}")
    return float(getattr(record, endpoint))


def _strength_polynomial(records: tuple[P1RedundancyRecord, ...]) -> np.ndarray:
    strength = np.asarray(
        [record.published_cai_strength_mpa for record in records], dtype=np.float64
    )
    matrix = np.column_stack((strength, strength**2, strength**3))
    if not np.all(np.isfinite(matrix)):
        raise EvaluationError("strength polynomial is nonfinite")
    return matrix


def _state_sha256(
    *,
    held_out_domain: str,
    endpoint: str,
    model_name: str,
    fit_domains: tuple[str, ...],
    fit_specimen_ids: tuple[str, ...],
    feature_names: tuple[str, ...],
    means: np.ndarray,
    scales: np.ndarray,
    design_encoder_sha256: str | None,
    coefficients: np.ndarray,
    intercept: float,
) -> str:
    payload = {
        "alpha": RIDGE_ALPHA,
        "design_encoder_sha256": design_encoder_sha256,
        "endpoint": endpoint,
        "feature_names": feature_names,
        "fit_domains": fit_domains,
        "fit_specimen_ids": fit_specimen_ids,
        "held_out_domain": held_out_domain,
        "model": model_name,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    for values in (
        means,
        scales,
        coefficients,
        np.asarray([intercept], dtype="<f8"),
    ):
        digest.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _fit_fold_model(
    records: tuple[P1RedundancyRecord, ...],
    *,
    held_out_domain: str,
    endpoint: str,
    model_name: str,
) -> tuple[FoldState, tuple[OOFPrediction, ...]]:
    source = tuple(record for record in records if record.domain_id != held_out_domain)
    query = tuple(record for record in records if record.domain_id == held_out_domain)
    fit_domains = tuple(domain for domain in DOMAIN_ORDER if domain != held_out_domain)
    fit_specimen_ids = tuple(record.specimen_id for record in source)

    source_polynomial = _strength_polynomial(source)
    query_polynomial = _strength_polynomial(query)
    means = np.mean(source_polynomial, axis=0, dtype=np.float64)
    scales = np.std(source_polynomial, axis=0, ddof=0, dtype=np.float64)
    scales[scales <= np.finfo(np.float64).eps] = 1.0
    source_features = (source_polynomial - means) / scales
    query_features = (query_polynomial - means) / scales
    feature_names = STRENGTH_FEATURE_NAMES
    design_encoder_sha256: str | None = None

    if model_name == "strength_plus_design":
        encoder = fit_fold_local_design_encoder(
            (record.design for record in records), held_out_domain
        )
        source_design = encoder.transform(record.design for record in source)
        query_design = encoder.transform(record.design for record in query)
        source_features = np.column_stack((source_features, source_design))
        query_features = np.column_stack((query_features, query_design))
        feature_names = (*STRENGTH_FEATURE_NAMES, *DESIGN_FEATURE_NAMES)
        design_encoder_sha256 = encoder.state_sha256
        if encoder.fit_specimen_ids != fit_specimen_ids:
            raise EvaluationError("design encoder source identities changed")
    elif model_name != "strength_only":
        raise EvaluationError(f"unknown P1 redundancy model: {model_name!r}")

    target = np.asarray([_target(record, endpoint) for record in source], dtype=np.float64)
    estimator = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="svd")
    estimator.fit(source_features, target)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    intercept = float(estimator.intercept_)
    predictions = np.asarray(estimator.predict(query_features), dtype=np.float64)
    if (
        coefficients.shape != (len(feature_names),)
        or not np.all(np.isfinite(coefficients))
        or not math.isfinite(intercept)
        or predictions.shape != (len(query),)
        or not np.all(np.isfinite(predictions))
    ):
        raise EvaluationError("P1 Ridge fit or prediction is invalid")

    immutable_means = _readonly(means)
    immutable_scales = _readonly(scales)
    immutable_coefficients = _readonly(coefficients)
    state_sha256 = _state_sha256(
        held_out_domain=held_out_domain,
        endpoint=endpoint,
        model_name=model_name,
        fit_domains=fit_domains,
        fit_specimen_ids=fit_specimen_ids,
        feature_names=feature_names,
        means=immutable_means,
        scales=immutable_scales,
        design_encoder_sha256=design_encoder_sha256,
        coefficients=immutable_coefficients,
        intercept=intercept,
    )
    state = FoldState(
        held_out_domain=held_out_domain,
        endpoint=endpoint,
        model=model_name,
        fit_domains=fit_domains,
        fit_specimen_ids=fit_specimen_ids,
        feature_names=feature_names,
        strength_means=immutable_means,
        strength_scales=immutable_scales,
        design_encoder_sha256=design_encoder_sha256,
        coefficients=immutable_coefficients,
        intercept=intercept,
        state_sha256=state_sha256,
    )
    rows = tuple(
        OOFPrediction(
            specimen_id=record.specimen_id,
            domain_id=record.domain_id,
            held_out_domain=held_out_domain,
            endpoint=endpoint,
            model=model_name,
            truth=_target(record, endpoint),
            prediction=float(prediction),
            fold_state_sha256=state_sha256,
        )
        for record, prediction in zip(query, predictions, strict=True)
    )
    return state, rows


def _metrics(
    predictions: tuple[OOFPrediction, ...], endpoint: str, model_name: str
) -> RedundancyMetric:
    rows = tuple(
        row
        for row in predictions
        if row.endpoint == endpoint and row.model == model_name
    )
    truth = np.asarray([row.truth for row in rows], dtype=np.float64)
    predicted = np.asarray([row.prediction for row in rows], dtype=np.float64)
    pooled_r2 = float(r2_score(truth, predicted))
    if not math.isfinite(pooled_r2):
        raise EvaluationError(f"pooled R-squared is undefined: {endpoint}/{model_name}")

    correlations: list[DomainSpearman] = []
    for domain_id in DOMAIN_ORDER:
        domain_rows = tuple(row for row in rows if row.domain_id == domain_id)
        statistic = float(
            spearmanr(
                [row.truth for row in domain_rows],
                [row.prediction for row in domain_rows],
            ).statistic
        )
        if not math.isfinite(statistic):
            raise EvaluationError(
                f"domain Spearman is undefined: {endpoint}/{model_name}/{domain_id}"
            )
        correlations.append(
            DomainSpearman(
                domain_id=domain_id,
                specimen_count=len(domain_rows),
                spearman=statistic,
            )
        )
    return RedundancyMetric(
        endpoint=endpoint,
        model=model_name,
        pooled_r2=pooled_r2,
        domain_spearman=tuple(correlations),
    )


def evaluate_p1_redundancy(
    records: tuple[P1RedundancyRecord, ...],
) -> RedundancyEvaluation:
    """Evaluate both fixed Ridge views in six source-only outer folds."""

    values = _validate_records(tuple(records))
    predictions: list[OOFPrediction] = []
    states: list[FoldState] = []
    for endpoint in PRIMARY_TARGET_FIELDS:
        for model_name in REDUNDANCY_MODELS:
            for held_out_domain in DOMAIN_ORDER:
                state, rows = _fit_fold_model(
                    values,
                    held_out_domain=held_out_domain,
                    endpoint=endpoint,
                    model_name=model_name,
                )
                states.append(state)
                predictions.extend(rows)

    prediction_values = tuple(predictions)
    metrics = tuple(
        _metrics(prediction_values, endpoint, model_name)
        for endpoint in PRIMARY_TARGET_FIELDS
        for model_name in REDUNDANCY_MODELS
    )
    return RedundancyEvaluation(
        predictions=prediction_values,
        fold_states=tuple(states),
        metrics=metrics,
    )
