"""Fold-local bounded regressor registry for D8 candidate evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .authority import D8InnerFold, validate_inner_fold
from .features import aggregate_features, aggregate_predictions

_PCA_DIMENSIONS = frozenset({4, 8, 16, 32, 64})
_REGRESSORS = frozenset(
    {
        "ridge",
        "elastic_net",
        "pls",
        "huber",
        "kernel_ridge",
        "svr",
        "hist_gradient_boosting",
        "shallow_mlp",
    }
)
_PARAMETER_KEYS = {
    "ridge": frozenset({"alpha"}),
    "elastic_net": frozenset({"alpha", "l1_ratio"}),
    "pls": frozenset({"n_components"}),
    "huber": frozenset({"alpha", "epsilon"}),
    "kernel_ridge": frozenset({"alpha", "gamma"}),
    "svr": frozenset({"C", "epsilon", "gamma"}),
    "hist_gradient_boosting": frozenset(
        {"l2_regularization", "learning_rate", "max_leaf_nodes"}
    ),
    "shallow_mlp": frozenset({"alpha", "hidden_layer_size"}),
}


def _number(value: object, *, label: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{label} is outside the registered range")
    return result


def _integer(value: object, *, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} is outside the registered range")
    return value


def _validated_parameters(name: str, value: object) -> dict[str, int | float]:
    if type(value) is not dict or frozenset(value) != _PARAMETER_KEYS[name]:
        raise ValueError("regressor parameters do not match the registered schema")
    if name == "ridge":
        return {
            "alpha": _number(value["alpha"], label="alpha", minimum=1e-6, maximum=1e3)
        }
    if name == "elastic_net":
        return {
            "alpha": _number(value["alpha"], label="alpha", minimum=1e-6, maximum=10.0),
            "l1_ratio": _number(
                value["l1_ratio"], label="l1_ratio", minimum=0.0, maximum=1.0
            ),
        }
    if name == "pls":
        return {
            "n_components": _integer(
                value["n_components"], label="n_components", minimum=1, maximum=16
            )
        }
    if name == "huber":
        return {
            "alpha": _number(value["alpha"], label="alpha", minimum=0.0, maximum=1.0),
            "epsilon": _number(
                value["epsilon"], label="epsilon", minimum=1.0, maximum=3.0
            ),
        }
    if name in {"kernel_ridge", "svr"}:
        result = {
            "alpha" if name == "kernel_ridge" else "C": _number(
                value["alpha" if name == "kernel_ridge" else "C"],
                label="regularization",
                minimum=1e-6,
                maximum=1e3,
            ),
            "gamma": _number(value["gamma"], label="gamma", minimum=1e-6, maximum=1e3),
        }
        if name == "svr":
            result["epsilon"] = _number(
                value["epsilon"], label="epsilon", minimum=0.0, maximum=1.0
            )
        return result
    if name == "hist_gradient_boosting":
        return {
            "l2_regularization": _number(
                value["l2_regularization"],
                label="l2_regularization",
                minimum=0.0,
                maximum=100.0,
            ),
            "learning_rate": _number(
                value["learning_rate"], label="learning_rate", minimum=1e-4, maximum=1.0
            ),
            "max_leaf_nodes": _integer(
                value["max_leaf_nodes"], label="max_leaf_nodes", minimum=2, maximum=31
            ),
        }
    return {
        "alpha": _number(value["alpha"], label="alpha", minimum=1e-8, maximum=100.0),
        "hidden_layer_size": _integer(
            value["hidden_layer_size"],
            label="hidden_layer_size",
            minimum=4,
            maximum=64,
        ),
    }


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One bounded fold-local PCA/regressor configuration."""

    pca_dimension: int
    regressor: str
    parameters: Mapping[str, int | float]
    seed: int
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.pca_dimension) is not int
            or self.pca_dimension not in _PCA_DIMENSIONS
        ):
            raise ValueError("PCA dimension is not registered")
        if type(self.regressor) is not str or self.regressor not in _REGRESSORS:
            raise ValueError("regressor is not registered")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("candidate seed must be an unsigned 32-bit integer")
        parameters = _validated_parameters(self.regressor, self.parameters)
        payload = {
            "pca_dimension": self.pca_dimension,
            "regressor": self.regressor,
            "parameters": parameters,
            "seed": self.seed,
        }
        state = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "state_sha256", state)


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    """One inner-query prediction vector with explicit fit/query identities."""

    fit_specimen_ids: tuple[str, ...]
    query_specimen_ids: tuple[str, ...]
    targets: np.ndarray
    predictions: np.ndarray
    fit_state_sha256: str
    state_sha256: str


def _matrix(value: object, *, rows: int) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("features must be real")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("features must be numeric") from error
    if (
        array.ndim != 2
        or array.shape[0] != rows
        or array.shape[1] < 1
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("features must be a finite aligned matrix")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _variant_matrix(value: object, *, rows: int, label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if (
        array.ndim != 3
        or array.shape[0] != rows
        or array.shape[1] not in (1, 2, 4, 8, 16)
        or array.shape[2] < 1
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            f"{label} must be a finite aligned (specimen, K, feature) matrix"
        )
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _vector(value: object, *, rows: int, label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if array.shape != (rows,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite aligned vector")
    return np.array(array, dtype=np.float64, copy=True)


def _readonly(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64).reshape(
        contiguous.shape
    )
    result.setflags(write=False)
    return result


def _regressor(spec: CandidateSpec):
    values = dict(spec.parameters)
    if spec.regressor == "ridge":
        return Ridge(alpha=values["alpha"])
    if spec.regressor == "elastic_net":
        return ElasticNet(
            alpha=values["alpha"],
            l1_ratio=values["l1_ratio"],
            max_iter=10_000,
            selection="cyclic",
            random_state=spec.seed,
        )
    if spec.regressor == "pls":
        return PLSRegression(n_components=int(values["n_components"]), scale=False)
    if spec.regressor == "huber":
        return HuberRegressor(
            alpha=values["alpha"], epsilon=values["epsilon"], max_iter=1_000
        )
    if spec.regressor == "kernel_ridge":
        return KernelRidge(alpha=values["alpha"], kernel="rbf", gamma=values["gamma"])
    if spec.regressor == "svr":
        return SVR(
            C=values["C"],
            epsilon=values["epsilon"],
            gamma=values["gamma"],
            kernel="rbf",
        )
    if spec.regressor == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=values["learning_rate"],
            max_leaf_nodes=int(values["max_leaf_nodes"]),
            l2_regularization=values["l2_regularization"],
            max_iter=200,
            random_state=spec.seed,
        )
    return MLPRegressor(
        hidden_layer_sizes=(int(values["hidden_layer_size"]),),
        activation="relu",
        solver="lbfgs",
        alpha=values["alpha"],
        max_iter=1_000,
        random_state=spec.seed,
    )


def _fit_pipeline(
    spec: CandidateSpec,
    *,
    matrix: np.ndarray,
    response: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[StandardScaler, PCA, object]:
    if spec.pca_dimension > min(len(fit_indices) - 1, matrix.shape[1]):
        raise ValueError("PCA dimension exceeds the inner-fit rank bound")
    scaler = StandardScaler(copy=True)
    fit_scaled = scaler.fit_transform(matrix[fit_indices])
    pca = PCA(n_components=spec.pca_dimension, svd_solver="full")
    fit_transformed = pca.fit_transform(fit_scaled)
    model = _regressor(spec)
    try:
        model.fit(fit_transformed, response[fit_indices])
    except (FloatingPointError, TypeError, ValueError) as error:
        raise ValueError("candidate regressor failed") from error
    return scaler, pca, model


def _predict_pipeline(
    pipeline: tuple[StandardScaler, PCA, object], matrix: np.ndarray
) -> np.ndarray:
    scaler, pca, model = pipeline
    try:
        transformed = pca.transform(scaler.transform(matrix))
        predictions = np.asarray(model.predict(transformed), dtype=np.float64).reshape(-1)
    except (FloatingPointError, TypeError, ValueError) as error:
        raise ValueError("candidate regressor failed") from error
    if predictions.shape != (len(matrix),) or not np.all(np.isfinite(predictions)):
        raise ValueError("candidate regressor returned invalid predictions")
    return predictions


def _candidate_result(
    *,
    spec: CandidateSpec,
    inner_fold: D8InnerFold,
    fold_state: str,
    response: np.ndarray,
    predictions: np.ndarray,
    pipeline: tuple[StandardScaler, PCA, object],
    semantics: Mapping[str, object] | None = None,
) -> CandidatePrediction:
    model_state = pickle.dumps(pipeline, protocol=5)
    semantic_bytes = (
        b""
        if semantics is None
        else json.dumps(
            dict(semantics), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    )
    fit_state = hashlib.sha256(
        spec.state_sha256.encode("ascii")
        + fold_state.encode("ascii")
        + semantic_bytes
        + model_state
    ).hexdigest()
    query_indices = np.asarray(inner_fold.query_indices, dtype=np.int64)
    query_targets = _readonly(response[query_indices])
    query_predictions = _readonly(predictions)
    state = hashlib.sha256(
        fit_state.encode("ascii")
        + query_targets.tobytes(order="C")
        + query_predictions.tobytes(order="C")
    ).hexdigest()
    return CandidatePrediction(
        fit_specimen_ids=inner_fold.fit_specimen_ids,
        query_specimen_ids=inner_fold.query_specimen_ids,
        targets=query_targets,
        predictions=query_predictions,
        fit_state_sha256=fit_state,
        state_sha256=state,
    )


def fit_candidate(
    spec: CandidateSpec,
    *,
    inner_fold: D8InnerFold,
    specimen_ids: tuple[str, ...],
    features: np.ndarray,
    targets: np.ndarray,
) -> CandidatePrediction:
    """Fit scaling, PCA, and a regressor on inner-fit domains only."""

    if type(spec) is not CandidateSpec:
        raise TypeError("exact CandidateSpec is required")
    fold_state = validate_inner_fold(inner_fold)
    if specimen_ids != inner_fold.search_view.specimen_ids:
        raise ValueError("candidate specimen roster differs from search authority")
    rows = inner_fold.search_view.specimen_count
    matrix = _matrix(features, rows=rows)
    response = _vector(targets, rows=rows, label="targets")
    authority_response = np.asarray(
        inner_fold.search_view.data_view.cai_ratio, dtype=np.float64
    )
    if not np.array_equal(response, authority_response):
        raise ValueError("candidate targets differ from search authority")
    fit_indices = np.asarray(inner_fold.fit_indices, dtype=np.int64)
    query_indices = np.asarray(inner_fold.query_indices, dtype=np.int64)
    pipeline = _fit_pipeline(
        spec, matrix=matrix, response=response, fit_indices=fit_indices
    )
    predictions = _predict_pipeline(pipeline, matrix[query_indices])
    return _candidate_result(
        spec=spec,
        inner_fold=inner_fold,
        fold_state=fold_state,
        response=response,
        predictions=predictions,
        pipeline=pipeline,
    )


def _contract_variants(values: np.ndarray, *, weight: float) -> np.ndarray:
    center = np.mean(values, axis=1, keepdims=True, dtype=np.float64)
    return center + (1.0 - weight) * (values - center)


def _ranking_consistent_predictions(
    values: np.ndarray, *, weight: float
) -> np.ndarray:
    if values.shape[0] < 2 or values.shape[1] < 2 or weight == 0.0:
        return values
    order = np.argsort(values, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    positions = np.arange(values.shape[0], dtype=np.float64)[:, None]
    np.put_along_axis(ranks, order, positions, axis=0)
    consensus = np.mean(ranks, axis=1, dtype=np.float64)
    centered_consensus = consensus - np.mean(consensus, dtype=np.float64)
    consensus_norm = float(np.linalg.norm(centered_consensus))
    correlations = np.zeros(values.shape[1], dtype=np.float64)
    if consensus_norm > 0.0:
        for column in range(values.shape[1]):
            centered = ranks[:, column] - np.mean(ranks[:, column], dtype=np.float64)
            denominator = float(np.linalg.norm(centered)) * consensus_norm
            if denominator > 0.0:
                correlations[column] = float(
                    np.dot(centered, centered_consensus) / denominator
                )
    logits = weight * correlations
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights, dtype=np.float64)
    consensus_prediction = np.sum(values * weights[None, :], axis=1)
    return (1.0 - weight) * values + weight * consensus_prediction[:, None]


def fit_marginalized_candidate(
    spec: CandidateSpec,
    *,
    inner_fold: D8InnerFold,
    specimen_ids: tuple[str, ...],
    train_variant_features: np.ndarray,
    query_variant_features: np.ndarray,
    targets: np.ndarray,
    marginalization_stage: str,
    feature_aggregation: str,
    prediction_aggregation: str,
    morphology_distances: np.ndarray | None,
    morphology_beta: float | None,
    consistency: str,
    consistency_weight: float,
) -> CandidatePrediction:
    """Fit one fold-local D8 candidate with registered K-axis marginalization."""

    if type(spec) is not CandidateSpec:
        raise TypeError("exact CandidateSpec is required")
    fold_state = validate_inner_fold(inner_fold)
    if specimen_ids != inner_fold.search_view.specimen_ids:
        raise ValueError("candidate specimen roster differs from search authority")
    rows = inner_fold.search_view.specimen_count
    train = _variant_matrix(
        train_variant_features, rows=rows, label="train variant features"
    )
    query = _variant_matrix(
        query_variant_features, rows=rows, label="query variant features"
    )
    if train.shape[2] != query.shape[2]:
        raise ValueError("train and query variant features are not aligned")
    response = _vector(targets, rows=rows, label="targets")
    authority_response = np.asarray(
        inner_fold.search_view.data_view.cai_ratio, dtype=np.float64
    )
    if not np.array_equal(response, authority_response):
        raise ValueError("candidate targets differ from search authority")
    if marginalization_stage not in {"feature", "prediction"}:
        raise ValueError("marginalization stage is not registered")
    if consistency not in {
        "none",
        "prediction_variance",
        "feature_variance",
        "pairwise_ranking",
    }:
        raise ValueError("consistency strategy is not registered")
    weight = _number(
        consistency_weight,
        label="consistency weight",
        minimum=0.0,
        maximum=1.0,
    )
    if consistency == "none" and weight != 0.0:
        raise ValueError("none consistency requires zero weight")
    fit_indices = np.asarray(inner_fold.fit_indices, dtype=np.int64)
    query_indices = np.asarray(inner_fold.query_indices, dtype=np.int64)
    semantics = {
        "marginalization_stage": marginalization_stage,
        "feature_aggregation": feature_aggregation,
        "prediction_aggregation": prediction_aggregation,
        "morphology_beta": morphology_beta,
        "consistency": consistency,
        "consistency_weight": weight,
    }

    if marginalization_stage == "feature":
        if prediction_aggregation != "mean":
            raise ValueError("feature marginalization requires mean prediction aggregation")
        if morphology_distances is not None or morphology_beta is not None:
            raise ValueError("feature marginalization does not use morphology weights")
        if consistency not in {"none", "feature_variance"}:
            raise ValueError("feature marginalization requires feature consistency")
        if consistency == "feature_variance":
            train = _contract_variants(train, weight=weight)
            query = _contract_variants(query, weight=weight)
        train_aggregated = aggregate_features(train, method=feature_aggregation)
        query_aggregated = aggregate_features(query, method=feature_aggregation)
        if train_aggregated.shape[1] != query_aggregated.shape[1]:
            raise ValueError("feature aggregation produced misaligned matrices")
        combined = np.zeros((rows, train_aggregated.shape[1]), dtype=np.float64)
        combined[fit_indices] = train_aggregated[fit_indices]
        combined[query_indices] = query_aggregated[query_indices]
        pipeline = _fit_pipeline(
            spec, matrix=combined, response=response, fit_indices=fit_indices
        )
        predictions = _predict_pipeline(pipeline, combined[query_indices])
    else:
        if feature_aggregation != "mean":
            raise ValueError("prediction marginalization requires mean feature aggregation")
        if consistency == "feature_variance":
            raise ValueError("prediction marginalization cannot use feature consistency")
        train_aggregated = aggregate_features(train, method="mean")
        pipeline = _fit_pipeline(
            spec, matrix=train_aggregated, response=response, fit_indices=fit_indices
        )
        query_values = query[query_indices]
        flat = query_values.reshape(-1, query_values.shape[2])
        variant_predictions = _predict_pipeline(pipeline, flat).reshape(
            len(query_indices), query_values.shape[1]
        )
        if consistency == "prediction_variance":
            variant_predictions = _contract_variants(
                variant_predictions[:, :, None], weight=weight
            )[:, :, 0]
        elif consistency == "pairwise_ranking":
            variant_predictions = _ranking_consistent_predictions(
                variant_predictions, weight=weight
            )
        distances: np.ndarray | None = None
        if morphology_distances is not None:
            distances_full = _matrix(morphology_distances, rows=rows)
            if distances_full.shape != (rows, query.shape[1]) or np.any(
                distances_full < 0.0
            ):
                raise ValueError("morphology distances are not aligned")
            distances = distances_full[query_indices]
        predictions = aggregate_predictions(
            variant_predictions,
            method=prediction_aggregation,
            morphology_distances=distances,
            beta=morphology_beta,
        )

    return _candidate_result(
        spec=spec,
        inner_fold=inner_fold,
        fold_state=fold_state,
        response=response,
        predictions=predictions,
        pipeline=pipeline,
        semantics=semantics,
    )


__all__ = [
    "CandidatePrediction",
    "CandidateSpec",
    "fit_candidate",
    "fit_marginalized_candidate",
]
