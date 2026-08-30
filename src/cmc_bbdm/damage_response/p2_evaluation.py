from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from sklearn.linear_model import Ridge

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS
from cmc_bbdm.damage_response.feature_views import PRIMARY_TARGET_FIELDS
from cmc_bbdm.damage_response.p2_features import P2FeatureAuthority
from cmc_bbdm.damage_response.p2_views import (
    EMBEDDING_P2_VIEWS,
    P2_VIEW_FIELDS,
    P2FoldPreprocessor,
    fit_p2_pca_basis,
    fit_p2_preprocessor,
    transform_p2_view,
)

P2_ENDPOINTS = PRIMARY_TARGET_FIELDS
P2_VIEWS = tuple(P2_VIEW_FIELDS)
DOMAIN_ORDER = tuple(PRIMARY_COUNTS)


class P2EvaluationError(RuntimeError):
    """Raised when strict nested P2 LODO evaluation cannot be completed."""


@dataclass(frozen=True, slots=True)
class P2EvaluationProtocol:
    ridge_alphas: tuple[float, ...]
    pca_dimensions: tuple[int, ...]
    tie_tolerance: float

    def __post_init__(self) -> None:
        alphas = tuple(float(value) for value in self.ridge_alphas)
        dimensions = tuple(self.pca_dimensions)
        tolerance = float(self.tie_tolerance)
        if (
            not alphas
            or any(not math.isfinite(value) or value <= 0.0 for value in alphas)
            or len(set(alphas)) != len(alphas)
            or not dimensions
            or any(type(value) is not int or value < 1 for value in dimensions)
            or tuple(sorted(set(dimensions))) != dimensions
            or not math.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise P2EvaluationError("P2 evaluation protocol is invalid")
        object.__setattr__(self, "ridge_alphas", alphas)
        object.__setattr__(self, "pca_dimensions", dimensions)
        object.__setattr__(self, "tie_tolerance", tolerance)


REGISTERED_P2_PROTOCOL = P2EvaluationProtocol(
    ridge_alphas=(0.1, 1.0, 10.0, 100.0),
    pca_dimensions=(8, 16, 32),
    tie_tolerance=1e-12,
)


@dataclass(frozen=True, slots=True)
class InnerCandidateScore:
    held_out_domain: str
    endpoint: str
    view_name: str
    ridge_alpha: float
    pca_dimension: int | None
    inner_domain_mae: tuple[tuple[str, float], ...]
    inner_equal_domain_mae: float
    selected: bool


@dataclass(frozen=True, slots=True)
class P2FoldState:
    held_out_domain: str
    endpoint: str
    view_name: str
    fit_domains: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    ridge_alpha: float
    pca_dimension: int | None
    preprocessor_state_sha256: str
    source_target_mean: float
    source_target_std: float
    coefficients: np.ndarray
    intercept: float
    state_sha256: str


@dataclass(frozen=True, slots=True)
class P2OOFPrediction:
    specimen_id: str
    domain_id: str
    held_out_domain: str
    endpoint: str
    view_name: str
    truth: float
    prediction: float
    absolute_error: float
    standardized_absolute_error: float
    source_target_std: float
    selected_ridge_alpha: float
    selected_pca_dimension: int | None
    preprocessor_state_sha256: str
    fold_state_sha256: str


@dataclass(frozen=True, slots=True)
class P2DomainMetric:
    endpoint: str
    view_name: str
    domain_id: str
    specimen_count: int
    mae: float
    rmse: float
    standardized_mae: float


@dataclass(frozen=True, slots=True)
class P2AggregateMetric:
    endpoint: str
    view_name: str
    specimen_count: int
    equal_domain_mae: float
    pooled_rmse: float
    pooled_r2: float
    equal_domain_standardized_mae: float


@dataclass(frozen=True, slots=True)
class P2OuterFoldEvaluation:
    held_out_domain: str
    inner_scores: tuple[InnerCandidateScore, ...]
    fold_states: tuple[P2FoldState, ...]
    predictions: tuple[P2OOFPrediction, ...]


@dataclass(frozen=True, slots=True)
class P2Evaluation:
    protocol: P2EvaluationProtocol
    inner_scores: tuple[InnerCandidateScore, ...]
    fold_states: tuple[P2FoldState, ...]
    predictions: tuple[P2OOFPrediction, ...]
    domain_metrics: tuple[P2DomainMetric, ...]
    metrics: tuple[P2AggregateMetric, ...]


@dataclass(frozen=True, slots=True)
class _PreparedSplit:
    state: P2FoldPreprocessor
    fit_indices: np.ndarray
    query_indices: np.ndarray
    fit_features: np.ndarray
    query_features: np.ndarray


def _readonly(value: np.ndarray, *, dtype: str = "<f8") -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _validate_authority(authority: P2FeatureAuthority) -> None:
    if not isinstance(authority, P2FeatureAuthority):
        raise P2EvaluationError("P2 feature authority type changed")
    observed = set(authority.domain_ids)
    if observed != set(DOMAIN_ORDER):
        raise P2EvaluationError("P2 evaluation requires the six canonical domains")
    counts = {
        domain: authority.domain_ids.count(domain) for domain in DOMAIN_ORDER
    }
    if any(value < 2 for value in counts.values()):
        raise P2EvaluationError("each P2 domain requires at least two specimens")


def _validate_targets(
    authority: P2FeatureAuthority, targets: Mapping[str, object]
) -> Mapping[str, np.ndarray]:
    if not isinstance(targets, Mapping) or set(targets) != set(P2_ENDPOINTS):
        raise P2EvaluationError("P2 target registry changed")
    result: dict[str, np.ndarray] = {}
    n = len(authority.specimen_ids)
    for endpoint in P2_ENDPOINTS:
        try:
            values = np.asarray(targets[endpoint], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise P2EvaluationError(f"P2 target is not numeric: {endpoint}") from error
        if (
            values.shape != (n,)
            or not np.all(np.isfinite(values))
            or float(np.std(values, ddof=0)) <= np.finfo(np.float64).eps
        ):
            raise P2EvaluationError(f"P2 target is invalid: {endpoint}")
        result[endpoint] = _readonly(values)
    return result


def _candidate_dimensions(
    view_name: str, protocol: P2EvaluationProtocol
) -> tuple[int | None, ...]:
    if view_name in EMBEDDING_P2_VIEWS:
        return tuple(protocol.pca_dimensions)
    return (None,)


def _ridge_fit_predict(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    query_features: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    estimator = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
    try:
        estimator.fit(fit_features, fit_targets)
        prediction = np.asarray(estimator.predict(query_features), dtype=np.float64)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        raise P2EvaluationError("P2 Ridge fit failed") from error
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    intercept = float(estimator.intercept_)
    if (
        coefficients.shape != (fit_features.shape[1],)
        or prediction.shape != (len(query_features),)
        or not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(prediction))
        or not math.isfinite(intercept)
    ):
        raise P2EvaluationError("P2 Ridge returned invalid parameters")
    return prediction, _readonly(coefficients), intercept


def select_inner_candidate(
    candidates: Sequence[InnerCandidateScore], *, tie_tolerance: float
) -> InnerCandidateScore:
    """Apply the registered score tolerance and deterministic simplicity tie rule."""

    values = tuple(candidates)
    tolerance = float(tie_tolerance)
    if (
        not values
        or not math.isfinite(tolerance)
        or tolerance < 0.0
        or any(
            not math.isfinite(row.inner_equal_domain_mae)
            or row.inner_equal_domain_mae < 0.0
            or not math.isfinite(row.ridge_alpha)
            or row.ridge_alpha <= 0.0
            for row in values
        )
    ):
        raise P2EvaluationError("inner candidate registry is invalid")
    identities = {
        (row.held_out_domain, row.endpoint, row.view_name) for row in values
    }
    if len(identities) != 1:
        raise P2EvaluationError("inner candidates cross outer/model identities")
    best_score = min(row.inner_equal_domain_mae for row in values)
    eligible = tuple(
        row
        for row in values
        if row.inner_equal_domain_mae <= best_score + tolerance
    )
    return min(
        eligible,
        key=lambda row: (
            -1 if row.pca_dimension is None else row.pca_dimension,
            -row.ridge_alpha,
        ),
    )


def _fold_state_sha256(
    *,
    held_out_domain: str,
    endpoint: str,
    view_name: str,
    fit_domains: tuple[str, ...],
    fit_specimen_ids: tuple[str, ...],
    alpha: float,
    pca_dimension: int | None,
    preprocessor_state_sha256: str,
    source_target_mean: float,
    source_target_std: float,
    coefficients: np.ndarray,
    intercept: float,
) -> str:
    metadata = {
        "endpoint": endpoint,
        "fit_domains": fit_domains,
        "fit_specimen_ids": fit_specimen_ids,
        "held_out_domain": held_out_domain,
        "pca_dimension": pca_dimension,
        "preprocessor_state_sha256": preprocessor_state_sha256,
        "ridge_alpha": alpha,
        "source_target_mean": source_target_mean,
        "source_target_std": source_target_std,
        "view_name": view_name,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    digest.update(np.ascontiguousarray(coefficients, dtype="<f8").tobytes())
    digest.update(np.asarray([intercept], dtype="<f8").tobytes())
    return digest.hexdigest()


def evaluate_p2_outer_fold(
    authority: P2FeatureAuthority,
    targets: Mapping[str, object],
    held_out_domain: str,
    *,
    protocol: P2EvaluationProtocol = REGISTERED_P2_PROTOCOL,
) -> P2OuterFoldEvaluation:
    """Evaluate one strict outer domain with source-domain inner selection."""

    _validate_authority(authority)
    if not isinstance(protocol, P2EvaluationProtocol):
        raise P2EvaluationError("P2 evaluation protocol type changed")
    if held_out_domain not in DOMAIN_ORDER:
        raise P2EvaluationError(f"unknown P2 held-out domain: {held_out_domain!r}")
    target_values = _validate_targets(authority, targets)
    domains = np.asarray(authority.domain_ids)
    outer_fit = np.flatnonzero(domains != held_out_domain)
    outer_query = np.flatnonzero(domains == held_out_domain)
    fit_domains = tuple(domain for domain in DOMAIN_ORDER if domain != held_out_domain)
    max_pca = max(protocol.pca_dimensions)
    prepared: dict[tuple[str, str, int | None], _PreparedSplit] = {}
    bases: dict[str, object] = {}

    def prepared_split(
        split_name: str,
        view_name: str,
        dimension: int | None,
    ) -> _PreparedSplit:
        key = (split_name, view_name, dimension)
        if key in prepared:
            return prepared[key]
        if split_name == "outer":
            fit = outer_fit
            query = outer_query
        else:
            if split_name not in fit_domains:
                raise P2EvaluationError("inner split is outside outer source domains")
            fit = np.flatnonzero(
                (domains != held_out_domain) & (domains != split_name)
            )
            query = np.flatnonzero(domains == split_name)
        basis = None
        if dimension is not None:
            if split_name not in bases:
                bases[split_name] = fit_p2_pca_basis(
                    authority, fit, maximum_dimension=max_pca
                )
            basis = bases[split_name]
        try:
            state = fit_p2_preprocessor(
                authority,
                view_name,
                fit,
                pca_dimension=dimension,
                pca_basis=basis,
            )
            fit_features = transform_p2_view(authority, state, fit)
            query_features = transform_p2_view(authority, state, query)
        except ValueError as error:
            raise P2EvaluationError(str(error)) from error
        value = _PreparedSplit(
            state=state,
            fit_indices=_readonly(fit, dtype="<i8"),
            query_indices=_readonly(query, dtype="<i8"),
            fit_features=fit_features,
            query_features=query_features,
        )
        prepared[key] = value
        return value

    inner_scores: list[InnerCandidateScore] = []
    fold_states: list[P2FoldState] = []
    predictions: list[P2OOFPrediction] = []
    for view_name in P2_VIEWS:
        dimensions = _candidate_dimensions(view_name, protocol)
        for endpoint in P2_ENDPOINTS:
            endpoint_candidates: list[InnerCandidateScore] = []
            target = target_values[endpoint]
            for dimension in dimensions:
                for alpha in protocol.ridge_alphas:
                    domain_scores: list[tuple[str, float]] = []
                    for inner_domain in fit_domains:
                        split = prepared_split(inner_domain, view_name, dimension)
                        prediction, _coefficients, _intercept = _ridge_fit_predict(
                            split.fit_features,
                            target[split.fit_indices],
                            split.query_features,
                            alpha=alpha,
                        )
                        mae = float(
                            np.mean(
                                np.abs(target[split.query_indices] - prediction),
                                dtype=np.float64,
                            )
                        )
                        domain_scores.append((inner_domain, mae))
                    equal_domain_mae = float(
                        np.mean([value for _domain, value in domain_scores])
                    )
                    endpoint_candidates.append(
                        InnerCandidateScore(
                            held_out_domain=held_out_domain,
                            endpoint=endpoint,
                            view_name=view_name,
                            ridge_alpha=alpha,
                            pca_dimension=dimension,
                            inner_domain_mae=tuple(domain_scores),
                            inner_equal_domain_mae=equal_domain_mae,
                            selected=False,
                        )
                    )
            selected = select_inner_candidate(
                endpoint_candidates, tie_tolerance=protocol.tie_tolerance
            )
            selected_key = (selected.ridge_alpha, selected.pca_dimension)
            marked = tuple(
                replace(
                    row,
                    selected=(row.ridge_alpha, row.pca_dimension) == selected_key,
                )
                for row in endpoint_candidates
            )
            if sum(row.selected for row in marked) != 1:
                raise P2EvaluationError("inner selection did not produce one candidate")
            inner_scores.extend(marked)

            outer = prepared_split("outer", view_name, selected.pca_dimension)
            outer_prediction, coefficients, intercept = _ridge_fit_predict(
                outer.fit_features,
                target[outer.fit_indices],
                outer.query_features,
                alpha=selected.ridge_alpha,
            )
            source_mean = float(np.mean(target[outer.fit_indices], dtype=np.float64))
            source_std = float(np.std(target[outer.fit_indices], ddof=0))
            if not math.isfinite(source_std) or source_std <= np.finfo(np.float64).eps:
                raise P2EvaluationError("outer-source target standard deviation is zero")
            fit_ids = tuple(
                authority.specimen_ids[int(index)] for index in outer.fit_indices
            )
            state_sha256 = _fold_state_sha256(
                held_out_domain=held_out_domain,
                endpoint=endpoint,
                view_name=view_name,
                fit_domains=fit_domains,
                fit_specimen_ids=fit_ids,
                alpha=selected.ridge_alpha,
                pca_dimension=selected.pca_dimension,
                preprocessor_state_sha256=outer.state.state_sha256,
                source_target_mean=source_mean,
                source_target_std=source_std,
                coefficients=coefficients,
                intercept=intercept,
            )
            fold_states.append(
                P2FoldState(
                    held_out_domain=held_out_domain,
                    endpoint=endpoint,
                    view_name=view_name,
                    fit_domains=fit_domains,
                    fit_specimen_ids=fit_ids,
                    ridge_alpha=selected.ridge_alpha,
                    pca_dimension=selected.pca_dimension,
                    preprocessor_state_sha256=outer.state.state_sha256,
                    source_target_mean=source_mean,
                    source_target_std=source_std,
                    coefficients=coefficients,
                    intercept=intercept,
                    state_sha256=state_sha256,
                )
            )
            for index, prediction in zip(
                outer.query_indices, outer_prediction, strict=True
            ):
                row = int(index)
                truth = float(target[row])
                predicted = float(prediction)
                absolute_error = abs(truth - predicted)
                predictions.append(
                    P2OOFPrediction(
                        specimen_id=authority.specimen_ids[row],
                        domain_id=authority.domain_ids[row],
                        held_out_domain=held_out_domain,
                        endpoint=endpoint,
                        view_name=view_name,
                        truth=truth,
                        prediction=predicted,
                        absolute_error=absolute_error,
                        standardized_absolute_error=absolute_error / source_std,
                        source_target_std=source_std,
                        selected_ridge_alpha=selected.ridge_alpha,
                        selected_pca_dimension=selected.pca_dimension,
                        preprocessor_state_sha256=outer.state.state_sha256,
                        fold_state_sha256=state_sha256,
                    )
                )
    return P2OuterFoldEvaluation(
        held_out_domain=held_out_domain,
        inner_scores=tuple(inner_scores),
        fold_states=tuple(fold_states),
        predictions=tuple(predictions),
    )


def _metrics(
    predictions: tuple[P2OOFPrediction, ...], *, specimen_count: int
) -> tuple[tuple[P2DomainMetric, ...], tuple[P2AggregateMetric, ...]]:
    domain_rows: list[P2DomainMetric] = []
    aggregate_rows: list[P2AggregateMetric] = []
    for endpoint in P2_ENDPOINTS:
        for view_name in P2_VIEWS:
            selected = tuple(
                row
                for row in predictions
                if row.endpoint == endpoint and row.view_name == view_name
            )
            if len(selected) != specimen_count:
                raise P2EvaluationError("P2 OOF prediction coverage is incomplete")
            domain_values: list[P2DomainMetric] = []
            for domain in DOMAIN_ORDER:
                rows = tuple(row for row in selected if row.domain_id == domain)
                if not rows:
                    raise P2EvaluationError("P2 OOF domain coverage is incomplete")
                errors = np.asarray(
                    [row.truth - row.prediction for row in rows], dtype=np.float64
                )
                standardized = np.asarray(
                    [row.standardized_absolute_error for row in rows],
                    dtype=np.float64,
                )
                metric = P2DomainMetric(
                    endpoint=endpoint,
                    view_name=view_name,
                    domain_id=domain,
                    specimen_count=len(rows),
                    mae=float(np.mean(np.abs(errors), dtype=np.float64)),
                    rmse=float(np.sqrt(np.mean(errors**2, dtype=np.float64))),
                    standardized_mae=float(np.mean(standardized, dtype=np.float64)),
                )
                domain_rows.append(metric)
                domain_values.append(metric)
            truth = np.asarray([row.truth for row in selected], dtype=np.float64)
            predicted = np.asarray(
                [row.prediction for row in selected], dtype=np.float64
            )
            residual = truth - predicted
            denominator = float(np.sum((truth - np.mean(truth)) ** 2))
            if denominator <= np.finfo(np.float64).eps:
                raise P2EvaluationError("P2 pooled target variance is zero")
            aggregate_rows.append(
                P2AggregateMetric(
                    endpoint=endpoint,
                    view_name=view_name,
                    specimen_count=len(selected),
                    equal_domain_mae=float(
                        np.mean([row.mae for row in domain_values])
                    ),
                    pooled_rmse=float(
                        np.sqrt(np.mean(residual**2, dtype=np.float64))
                    ),
                    pooled_r2=float(1.0 - np.sum(residual**2) / denominator),
                    equal_domain_standardized_mae=float(
                        np.mean([row.standardized_mae for row in domain_values])
                    ),
                )
            )
    return tuple(domain_rows), tuple(aggregate_rows)


def evaluate_p2_nested_lodo(
    authority: P2FeatureAuthority,
    targets: Mapping[str, object],
    *,
    protocol: P2EvaluationProtocol = REGISTERED_P2_PROTOCOL,
) -> P2Evaluation:
    """Execute all six strict outer folds and derive deterministic OOF metrics."""

    _validate_authority(authority)
    _validate_targets(authority, targets)
    outer = tuple(
        evaluate_p2_outer_fold(
            authority, targets, domain, protocol=protocol
        )
        for domain in DOMAIN_ORDER
    )
    inner_scores = tuple(row for fold in outer for row in fold.inner_scores)
    fold_states = tuple(row for fold in outer for row in fold.fold_states)
    predictions = tuple(row for fold in outer for row in fold.predictions)
    expected_predictions = len(authority.specimen_ids) * len(P2_ENDPOINTS) * len(
        P2_VIEWS
    )
    prediction_keys = {
        (row.specimen_id, row.endpoint, row.view_name) for row in predictions
    }
    if (
        len(predictions) != expected_predictions
        or len(prediction_keys) != expected_predictions
        or len(fold_states) != len(DOMAIN_ORDER) * len(P2_ENDPOINTS) * len(P2_VIEWS)
    ):
        raise P2EvaluationError("P2 nested LODO output membership is incomplete")
    domain_metrics, metrics = _metrics(
        predictions, specimen_count=len(authority.specimen_ids)
    )
    return P2Evaluation(
        protocol=protocol,
        inner_scores=inner_scores,
        fold_states=fold_states,
        predictions=predictions,
        domain_metrics=domain_metrics,
        metrics=metrics,
    )
