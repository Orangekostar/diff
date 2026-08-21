"""Repeated-seed reranking and prospective D8 selection freezing."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .authority import D8InnerFold, D8SearchView, issue_inner_fold, validate_search_view
from .config import DOMAIN_ORDER
from .regression import CandidatePrediction, CandidateSpec
from .search import D8Candidate, robust_inner_objective

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_REGRESSOR_COMPLEXITY = {
    "ridge": 0,
    "huber": 1,
    "elastic_net": 2,
    "pls": 3,
    "kernel_ridge": 4,
    "svr": 5,
    "hist_gradient_boosting": 6,
    "shallow_mlp": 7,
}
_LAYER_COMPLEXITY = {"global": 0, "layer3": 1, "multi_layer": 2}
_FEATURE_AGGREGATION_COMPLEXITY = {
    "mean": 0,
    "median": 1,
    "trimmed": 2,
    "mean_variance": 3,
}
_PREDICTION_AGGREGATION_COMPLEXITY = {
    "mean": 0,
    "median": 1,
    "trimmed": 2,
    "morphology_weighted": 3,
}
_CONSISTENCY_COMPLEXITY = {
    "none": 0,
    "prediction_variance": 1,
    "feature_variance": 2,
    "pairwise_ranking": 3,
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    )


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - _SHA256_CHARACTERS)
    )


def _readonly(value: object, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite and aligned")
    payload = np.ascontiguousarray(array).tobytes(order="C")
    result = np.frombuffer(payload, dtype=np.float64).reshape(shape)
    result.setflags(write=False)
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty sequence")
    return math.fsum(float(value) for value in values) / len(values)


def _seeded_candidate(
    candidate: D8Candidate, *, seed: int, K_test: int | None = None
) -> D8Candidate:
    if type(candidate) is not D8Candidate:
        raise TypeError("reranking requires exact D8Candidate values")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("reranking seed must be an unsigned 32-bit integer")
    regressor = candidate.regressor_spec
    return D8Candidate(
        control_id=candidate.control_id,
        decomposition_family=candidate.decomposition_family,
        band=candidate.band,
        decomposition_parameters=dict(candidate.decomposition_parameters),
        alpha=candidate.alpha,
        K_train=candidate.K_train,
        K_test=candidate.K_test if K_test is None else K_test,
        thresholds=candidate.thresholds,
        feature_layer=candidate.feature_layer,
        feature_aggregation=candidate.feature_aggregation,
        prediction_aggregation=candidate.prediction_aggregation,
        morphology_beta=candidate.morphology_beta,
        consistency=candidate.consistency,
        consistency_weight=candidate.consistency_weight,
        regressor_spec=CandidateSpec(
            pca_dimension=regressor.pca_dimension,
            regressor=regressor.regressor,
            parameters=dict(regressor.parameters),
            seed=seed,
        ),
        seed=seed,
        config_sha256=candidate.config_sha256,
        marginalization_stage=candidate.marginalization_stage,
    )


def _complexity_key(candidate: D8Candidate) -> tuple[int, ...]:
    return (
        candidate.K_train,
        candidate.K_test,
        0 if candidate.marginalization_stage == "feature" else 1,
        _LAYER_COMPLEXITY[candidate.feature_layer],
        _FEATURE_AGGREGATION_COMPLEXITY[candidate.feature_aggregation],
        _PREDICTION_AGGREGATION_COMPLEXITY[candidate.prediction_aggregation],
        _CONSISTENCY_COMPLEXITY[candidate.consistency],
        candidate.regressor_spec.pca_dimension,
        _REGRESSOR_COMPLEXITY[candidate.regressor_spec.regressor],
    )


@dataclass(frozen=True, slots=True)
class SeedRerankEvaluation:
    seed: int
    seeded_candidate_sha256: str
    domain_mae: tuple[tuple[str, float], ...]
    mean_mae: float
    worst_mae: float
    domain_sd: float
    objective: float
    oof_targets: np.ndarray
    oof_predictions: np.ndarray
    fold_evidence_sha256: tuple[str, ...]
    evidence_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class RerankRow:
    candidate: D8Candidate
    seeds: tuple[SeedRerankEvaluation, ...]
    domain_mae: tuple[tuple[str, float], ...]
    mean_mae: float
    worst_mae: float
    domain_sd: float
    objective: float
    complexity_key: tuple[int, ...]
    oof_targets: np.ndarray
    oof_predictions: np.ndarray
    state_sha256: str

    @property
    def rank_key(self) -> tuple[object, ...]:
        return (
            self.objective,
            self.mean_mae,
            self.worst_mae,
            self.complexity_key,
            self.candidate.state_sha256,
        )


@dataclass(frozen=True, slots=True)
class RerankResult:
    outer_domain: str
    config_sha256: str
    search_view_sha256: str
    seed_count: int
    rows: tuple[RerankRow, ...]
    finalists: tuple[RerankRow, ...]
    state_sha256: str

    @property
    def selected(self) -> RerankRow:
        return self.rows[0]


@dataclass(frozen=True, slots=True)
class FinalistResult:
    outer_domain: str
    config_sha256: str
    search_view_sha256: str
    cells: tuple[RerankRow, ...]
    selected: tuple[RerankRow, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    candidate_sha256: tuple[str, ...]
    accepted: bool
    weights: np.ndarray
    crossfit_weights: np.ndarray
    best_member_index: int
    best_member_objective: float
    objective: float
    objective_gain: float
    predictions: np.ndarray
    state_sha256: str


@dataclass(frozen=True, slots=True)
class FrozenOuterSelection:
    outer_domain: str
    config_sha256: str
    search_view_sha256: str
    selected_candidate_sha256: tuple[str, ...]
    ensemble_sha256: str
    outer_evaluation_started: bool
    state_sha256: str


def _validated_prediction(
    value: object,
    *,
    candidate: D8Candidate,
    fold: D8InnerFold,
) -> tuple[np.ndarray, np.ndarray, str]:
    if type(value) is not CandidatePrediction:
        raise TypeError("rerank evaluator must return exact CandidatePrediction")
    if value.fit_specimen_ids != fold.fit_specimen_ids:
        raise ValueError("rerank prediction fit identities changed")
    if value.query_specimen_ids != fold.query_specimen_ids:
        raise ValueError("rerank prediction query identities changed")
    rows = len(fold.query_indices)
    targets = _readonly(value.targets, shape=(rows,), label="rerank targets")
    predictions = _readonly(
        value.predictions, shape=(rows,), label="rerank predictions"
    )
    authority = np.asarray(
        fold.search_view.data_view.cai_ratio[fold.query_indices], dtype=np.float64
    )
    if not np.array_equal(targets, authority):
        raise ValueError("rerank targets differ from search authority")
    if not _valid_sha256(value.fit_state_sha256) or not _valid_sha256(
        value.state_sha256
    ):
        raise ValueError("rerank prediction state hash is invalid")
    expected_state = _sha256(
        value.fit_state_sha256.encode("ascii")
        + targets.tobytes(order="C")
        + predictions.tobytes(order="C")
    )
    if value.state_sha256 != expected_state:
        raise ValueError("rerank prediction state changed")
    evidence = _canonical_sha256(
        {
            "candidate_sha256": candidate.state_sha256,
            "fold_sha256": fold.state_sha256,
            "prediction_sha256": value.state_sha256,
        }
    )
    return targets, predictions, evidence


def _evaluate_seed(
    candidate: D8Candidate,
    *,
    view: D8SearchView,
    seed: int,
    evaluator: Callable[[D8Candidate, D8InnerFold], CandidatePrediction],
) -> SeedRerankEvaluation:
    seeded = _seeded_candidate(candidate, seed=seed)
    targets = np.full(view.specimen_count, np.nan, dtype=np.float64)
    predictions = np.full(view.specimen_count, np.nan, dtype=np.float64)
    evidence: list[str] = []
    domain_values: list[tuple[str, float]] = []
    domains = tuple(dict.fromkeys(view.dataset_ids))
    for domain in domains:
        fold = issue_inner_fold(view, query_domain=domain)
        result = evaluator(seeded, fold)
        target, prediction, state = _validated_prediction(
            result, candidate=seeded, fold=fold
        )
        indices = np.asarray(fold.query_indices, dtype=np.int64)
        if np.any(np.isfinite(targets[indices])):
            raise ValueError("rerank OOF rows overlap")
        targets[indices] = target
        predictions[indices] = prediction
        evidence.append(state)
        domain_values.append(
            (domain, float(np.mean(np.abs(prediction - target), dtype=np.float64)))
        )
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(predictions)):
        raise ValueError("rerank OOF rows are incomplete")
    authority = np.asarray(view.data_view.cai_ratio, dtype=np.float64)
    if not np.array_equal(targets, authority):
        raise ValueError("rerank OOF targets changed")
    values = np.asarray([value for _, value in domain_values], dtype=np.float64)
    objective = robust_inner_objective(values)
    mean_mae = _mean(values.tolist())
    worst = float(np.max(values))
    deviation = float(np.std(values))
    immutable_targets = _readonly(
        targets, shape=(view.specimen_count,), label="OOF targets"
    )
    immutable_predictions = _readonly(
        predictions, shape=(view.specimen_count,), label="OOF predictions"
    )
    evidence_sha = _sha256("\0".join(evidence).encode("ascii"))
    payload = {
        "seed": seed,
        "seeded_candidate_sha256": seeded.state_sha256,
        "domain_mae": domain_values,
        "mean_mae": mean_mae,
        "worst_mae": worst,
        "domain_sd": deviation,
        "objective": objective,
        "target_sha256": _sha256(immutable_targets.tobytes(order="C")),
        "prediction_sha256": _sha256(immutable_predictions.tobytes(order="C")),
        "evidence_sha256": evidence_sha,
    }
    return SeedRerankEvaluation(
        seed=seed,
        seeded_candidate_sha256=seeded.state_sha256,
        domain_mae=tuple(domain_values),
        mean_mae=mean_mae,
        worst_mae=worst,
        domain_sd=deviation,
        objective=objective,
        oof_targets=immutable_targets,
        oof_predictions=immutable_predictions,
        fold_evidence_sha256=tuple(evidence),
        evidence_sha256=evidence_sha,
        state_sha256=_canonical_sha256(payload),
    )


def _evaluate_candidate(
    candidate: D8Candidate,
    *,
    view: D8SearchView,
    seeds: tuple[int, ...],
    evaluator: Callable[[D8Candidate, D8InnerFold], CandidatePrediction],
) -> RerankRow:
    evaluations = tuple(
        _evaluate_seed(candidate, view=view, seed=seed, evaluator=evaluator)
        for seed in seeds
    )
    targets = evaluations[0].oof_targets
    if any(not np.array_equal(item.oof_targets, targets) for item in evaluations[1:]):
        raise ValueError("rerank seed targets changed")
    domain_names = tuple(domain for domain, _ in evaluations[0].domain_mae)
    if any(
        tuple(domain for domain, _ in item.domain_mae) != domain_names
        for item in evaluations[1:]
    ):
        raise ValueError("rerank seed domain order changed")
    domain_mae = tuple(
        (
            domain,
            _mean([dict(item.domain_mae)[domain] for item in evaluations]),
        )
        for domain in domain_names
    )
    predictions = _readonly(
        np.mean(
            np.vstack([item.oof_predictions for item in evaluations]),
            axis=0,
            dtype=np.float64,
        ),
        shape=(view.specimen_count,),
        label="mean seed OOF predictions",
    )
    mean_mae = _mean([item.mean_mae for item in evaluations])
    worst_mae = _mean([item.worst_mae for item in evaluations])
    domain_sd = _mean([item.domain_sd for item in evaluations])
    objective = _mean([item.objective for item in evaluations])
    complexity = _complexity_key(candidate)
    payload = {
        "candidate_sha256": candidate.state_sha256,
        "seed_states": [item.state_sha256 for item in evaluations],
        "domain_mae": domain_mae,
        "mean_mae": mean_mae,
        "worst_mae": worst_mae,
        "domain_sd": domain_sd,
        "objective": objective,
        "complexity_key": complexity,
        "target_sha256": _sha256(targets.tobytes(order="C")),
        "prediction_sha256": _sha256(predictions.tobytes(order="C")),
    }
    return RerankRow(
        candidate=candidate,
        seeds=evaluations,
        domain_mae=domain_mae,
        mean_mae=mean_mae,
        worst_mae=worst_mae,
        domain_sd=domain_sd,
        objective=objective,
        complexity_key=complexity,
        oof_targets=targets,
        oof_predictions=predictions,
        state_sha256=_canonical_sha256(payload),
    )


def _validated_candidates(
    values: object, *, count: int, view: D8SearchView
) -> tuple[D8Candidate, ...]:
    if type(values) is not tuple or len(values) != count:
        raise ValueError(f"selection requires exactly {count} candidates")
    if any(type(item) is not D8Candidate for item in values):
        raise TypeError("selection requires exact D8Candidate values")
    candidates = values
    if len({item.state_sha256 for item in candidates}) != len(candidates):
        raise ValueError("selection candidates are not unique")
    if any(item.config_sha256 != view.config_sha256 for item in candidates):
        raise ValueError("selection candidate config differs from search authority")
    return candidates


def _validated_seeds(values: object) -> tuple[int, ...]:
    if (
        type(values) is not tuple
        or len(values) != 3
        or len(set(values)) != 3
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values)
    ):
        raise ValueError("reranking requires three unique registered seeds")
    return values


def rerank_candidates(
    candidates: tuple[D8Candidate, ...],
    *,
    view: D8SearchView,
    seeds: tuple[int, ...],
    evaluator: Callable[[D8Candidate, D8InnerFold], CandidatePrediction],
) -> RerankResult:
    """Rerun the top twelve candidates across three fixed inner-LODO seeds."""

    view_state = validate_search_view(view)
    values = _validated_candidates(candidates, count=12, view=view)
    registered_seeds = _validated_seeds(seeds)
    if not callable(evaluator):
        raise TypeError("rerank evaluator must be callable")
    rows = tuple(
        sorted(
            (
                _evaluate_candidate(
                    candidate,
                    view=view,
                    seeds=registered_seeds,
                    evaluator=evaluator,
                )
                for candidate in values
            ),
            key=lambda item: item.rank_key,
        )
    )
    payload = {
        "outer_domain": view.outer_domain,
        "config_sha256": view.config_sha256,
        "search_view_sha256": view_state,
        "seeds": registered_seeds,
        "row_states": [item.state_sha256 for item in rows],
        "finalist_states": [item.state_sha256 for item in rows[:4]],
    }
    return RerankResult(
        outer_domain=view.outer_domain,
        config_sha256=view.config_sha256,
        search_view_sha256=view_state,
        seed_count=len(registered_seeds),
        rows=rows,
        finalists=rows[:4],
        state_sha256=_canonical_sha256(payload),
    )


def evaluate_finalists(
    candidates: tuple[D8Candidate, ...],
    *,
    view: D8SearchView,
    seeds: tuple[int, ...],
    K_test_values: tuple[int, ...],
    evaluator: Callable[[D8Candidate, D8InnerFold], CandidatePrediction],
) -> FinalistResult:
    """Evaluate each of four finalists at the registered large test ensembles."""

    view_state = validate_search_view(view)
    values = _validated_candidates(candidates, count=4, view=view)
    registered_seeds = _validated_seeds(seeds)
    if K_test_values != (8, 16):
        raise ValueError("finalist K_test values changed")
    cells: list[RerankRow] = []
    selected: list[RerankRow] = []
    for candidate in values:
        rows = [
            _evaluate_candidate(
                _seeded_candidate(candidate, seed=candidate.seed, K_test=K_test),
                view=view,
                seeds=registered_seeds,
                evaluator=evaluator,
            )
            for K_test in K_test_values
        ]
        cells.extend(rows)
        selected.append(min(rows, key=lambda item: item.rank_key))
    ordered_cells = tuple(sorted(cells, key=lambda item: item.rank_key))
    ordered_selected = tuple(sorted(selected, key=lambda item: item.rank_key))
    payload = {
        "outer_domain": view.outer_domain,
        "config_sha256": view.config_sha256,
        "search_view_sha256": view_state,
        "cell_states": [item.state_sha256 for item in ordered_cells],
        "selected_states": [item.state_sha256 for item in ordered_selected],
    }
    return FinalistResult(
        outer_domain=view.outer_domain,
        config_sha256=view.config_sha256,
        search_view_sha256=view_state,
        cells=ordered_cells,
        selected=ordered_selected,
        state_sha256=_canonical_sha256(payload),
    )


def _prediction_objective(
    predictions: np.ndarray, targets: np.ndarray, domain_ids: tuple[str, ...]
) -> float:
    order = tuple(dict.fromkeys(domain_ids))
    if len(order) != 5:
        raise ValueError("ensemble OOF rows must cover five domains")
    domains = np.asarray(domain_ids)
    values = np.asarray(
        [
            np.mean(np.abs(predictions[domains == domain] - targets[domains == domain]))
            for domain in order
        ],
        dtype=np.float64,
    )
    return robust_inner_objective(values)


def _fit_simplex_weights(matrix: np.ndarray, truth: np.ndarray) -> np.ndarray:
    member_count = matrix.shape[0]
    initial = np.full(member_count, 1.0 / member_count, dtype=np.float64)
    solution = minimize(
        lambda weights: float(np.sum((weights @ matrix - truth) ** 2)),
        initial,
        jac=lambda weights: 2.0 * (weights @ matrix - truth) @ matrix.T,
        bounds=tuple((0.0, 1.0) for _ in range(member_count)),
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        method="SLSQP",
        options={"ftol": 1.0e-14, "maxiter": 10_000, "disp": False},
    )
    if not solution.success:
        raise ValueError("nonnegative ensemble optimization failed")
    weights = np.asarray(solution.x, dtype=np.float64)
    weights[np.abs(weights) <= 1.0e-12] = 0.0
    total = float(np.sum(weights))
    if np.any(weights < 0.0) or not math.isfinite(total) or total <= 0.0:
        raise ValueError("nonnegative ensemble returned invalid weights")
    weights /= total
    return weights


def fit_nonnegative_ensemble(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    specimen_ids: tuple[str, ...],
    domain_ids: tuple[str, ...],
    candidate_sha256: tuple[str, ...],
    minimum_j_gain: float = 1.0e-4,
) -> EnsembleResult:
    """Fit deterministic simplex weights from aligned inner-OOF predictions."""

    if np.iscomplexobj(predictions):
        raise ValueError("ensemble predictions must be real")
    try:
        matrix = np.asarray(predictions, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("ensemble predictions must be numeric") from error
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("ensemble predictions must be a finite member matrix")
    member_count, row_count = matrix.shape
    truth = _readonly(targets, shape=(row_count,), label="ensemble targets")
    if (
        type(specimen_ids) is not tuple
        or len(specimen_ids) != row_count
        or len(set(specimen_ids)) != row_count
        or any(type(item) is not str or not item for item in specimen_ids)
    ):
        raise ValueError("ensemble specimen identities are invalid")
    if (
        type(domain_ids) is not tuple
        or len(domain_ids) != row_count
        or any(type(item) is not str or not item for item in domain_ids)
    ):
        raise ValueError("ensemble domain identities are invalid")
    if (
        type(candidate_sha256) is not tuple
        or len(candidate_sha256) != member_count
        or len(set(candidate_sha256)) != member_count
        or any(not _valid_sha256(item) for item in candidate_sha256)
    ):
        raise ValueError("ensemble candidate identities are invalid")
    if (
        type(minimum_j_gain) is not float
        or not math.isfinite(minimum_j_gain)
        or minimum_j_gain < 0.0
    ):
        raise ValueError("ensemble minimum objective gain is invalid")
    member_objectives = np.asarray(
        [
            _prediction_objective(matrix[index], truth, domain_ids)
            for index in range(member_count)
        ],
        dtype=np.float64,
    )
    best = min(
        range(member_count),
        key=lambda index: (member_objectives[index], candidate_sha256[index]),
    )
    domain_order = tuple(dict.fromkeys(domain_ids))
    domain_array = np.asarray(domain_ids)
    crossfit_weights = np.empty((len(domain_order), member_count), dtype=np.float64)
    crossfit_predictions = np.empty(row_count, dtype=np.float64)
    for domain_index, domain in enumerate(domain_order):
        fit_mask = domain_array != domain
        query_mask = ~fit_mask
        fitted = _fit_simplex_weights(matrix[:, fit_mask], truth[fit_mask])
        crossfit_weights[domain_index] = fitted
        crossfit_predictions[query_mask] = fitted @ matrix[:, query_mask]
    objective = _prediction_objective(crossfit_predictions, truth, domain_ids)
    best_objective = float(member_objectives[best])
    gain = best_objective - objective
    accepted = gain >= minimum_j_gain
    if accepted:
        weights = _fit_simplex_weights(matrix, truth)
        ensemble_predictions = crossfit_predictions
    else:
        weights = np.zeros(member_count, dtype=np.float64)
        weights[best] = 1.0
        ensemble_predictions = np.array(matrix[best], dtype=np.float64, copy=True)
        objective = best_objective
        gain = 0.0
    immutable_weights = _readonly(
        weights, shape=(member_count,), label="ensemble weights"
    )
    immutable_predictions = _readonly(
        ensemble_predictions,
        shape=(row_count,),
        label="ensemble predictions",
    )
    immutable_crossfit_weights = _readonly(
        crossfit_weights,
        shape=(len(domain_order), member_count),
        label="cross-fitted ensemble weights",
    )
    payload = {
        "candidate_sha256": candidate_sha256,
        "accepted": accepted,
        "weights": immutable_weights.tolist(),
        "crossfit_weights": immutable_crossfit_weights.tolist(),
        "best_member_index": best,
        "best_member_objective": best_objective,
        "objective": objective,
        "objective_gain": gain,
        "specimen_ids": specimen_ids,
        "domain_ids": domain_ids,
        "target_sha256": _sha256(truth.tobytes(order="C")),
        "prediction_sha256": _sha256(immutable_predictions.tobytes(order="C")),
    }
    return EnsembleResult(
        candidate_sha256=candidate_sha256,
        accepted=accepted,
        weights=immutable_weights,
        crossfit_weights=immutable_crossfit_weights,
        best_member_index=best,
        best_member_objective=best_objective,
        objective=objective,
        objective_gain=gain,
        predictions=immutable_predictions,
        state_sha256=_canonical_sha256(payload),
    )


def _exact_mapping(
    value: object, *, keys: set[str], label: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} schema changed")
    return dict(value)


def _finite_float(value: object, *, label: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _same_float(left: object, right: float, *, label: str) -> None:
    observed = _finite_float(left, label=label)
    if not math.isclose(observed, right, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(f"{label} changed")


def _seed_evidence_payload(value: SeedRerankEvaluation) -> dict[str, object]:
    if type(value) is not SeedRerankEvaluation:
        raise TypeError("selection seed evidence type changed")
    return {
        "seed": value.seed,
        "seeded_candidate_sha256": value.seeded_candidate_sha256,
        "domain_mae": [[domain, mae] for domain, mae in value.domain_mae],
        "mean_mae": value.mean_mae,
        "worst_mae": value.worst_mae,
        "domain_sd": value.domain_sd,
        "objective": value.objective,
        "oof_targets": value.oof_targets.tolist(),
        "oof_predictions": value.oof_predictions.tolist(),
        "fold_evidence_sha256": list(value.fold_evidence_sha256),
        "evidence_sha256": value.evidence_sha256,
        "state_sha256": value.state_sha256,
    }


def _row_evidence_payload(value: RerankRow) -> dict[str, object]:
    if type(value) is not RerankRow:
        raise TypeError("selection row evidence type changed")
    return {
        "candidate": value.candidate.to_payload(),
        "seeds": [_seed_evidence_payload(item) for item in value.seeds],
        "domain_mae": [[domain, mae] for domain, mae in value.domain_mae],
        "mean_mae": value.mean_mae,
        "worst_mae": value.worst_mae,
        "domain_sd": value.domain_sd,
        "objective": value.objective,
        "complexity_key": list(value.complexity_key),
        "oof_targets": value.oof_targets.tolist(),
        "oof_predictions": value.oof_predictions.tolist(),
        "state_sha256": value.state_sha256,
    }


def _authority_payload(view: D8SearchView) -> dict[str, object]:
    targets = _readonly(
        view.data_view.cai_ratio,
        shape=(view.specimen_count,),
        label="selection authority targets",
    )
    return {
        "specimen_ids": list(view.specimen_ids),
        "domain_ids": list(view.dataset_ids),
        "targets": targets.tolist(),
        "target_sha256": _sha256(targets.tobytes(order="C")),
    }


def _parse_domain_mae(
    value: object,
    *,
    domain_ids: tuple[str, ...],
    targets: np.ndarray,
    predictions: np.ndarray,
    label: str,
) -> tuple[tuple[str, float], ...]:
    order = tuple(dict.fromkeys(domain_ids))
    if not isinstance(value, list) or len(value) != len(order):
        raise ValueError(f"{label} domain roster changed")
    domain_array = np.asarray(domain_ids)
    expected = tuple(
        (
            domain,
            float(
                np.mean(
                    np.abs(
                        predictions[domain_array == domain]
                        - targets[domain_array == domain]
                    ),
                    dtype=np.float64,
                )
            ),
        )
        for domain in order
    )
    for observed, (domain, mae) in zip(value, expected, strict=True):
        if (
            not isinstance(observed, list)
            or len(observed) != 2
            or observed[0] != domain
        ):
            raise ValueError(f"{label} domain roster changed")
        _same_float(observed[1], mae, label=f"{label} domain MAE")
    return expected


def _parse_seed_evidence(
    value: object,
    *,
    candidate: D8Candidate,
    specimen_count: int,
    domain_ids: tuple[str, ...],
    targets: np.ndarray,
) -> SeedRerankEvaluation:
    item = _exact_mapping(
        value,
        keys={
            "seed",
            "seeded_candidate_sha256",
            "domain_mae",
            "mean_mae",
            "worst_mae",
            "domain_sd",
            "objective",
            "oof_targets",
            "oof_predictions",
            "fold_evidence_sha256",
            "evidence_sha256",
            "state_sha256",
        },
        label="selection seed evidence",
    )
    seed = item["seed"]
    if type(seed) is not int:
        raise ValueError("selection seed changed")
    seeded = _seeded_candidate(candidate, seed=seed)
    if item["seeded_candidate_sha256"] != seeded.state_sha256:
        raise ValueError("selection seeded candidate changed")
    observed_targets = _readonly(
        item["oof_targets"], shape=(specimen_count,), label="selection seed targets"
    )
    predictions = _readonly(
        item["oof_predictions"],
        shape=(specimen_count,),
        label="selection seed predictions",
    )
    if not np.array_equal(observed_targets, targets):
        raise ValueError("selection seed targets changed")
    domain_mae = _parse_domain_mae(
        item["domain_mae"],
        domain_ids=domain_ids,
        targets=targets,
        predictions=predictions,
        label="selection seed",
    )
    values = np.asarray([mae for _, mae in domain_mae], dtype=np.float64)
    mean_mae = _mean(values.tolist())
    worst_mae = float(np.max(values))
    domain_sd = float(np.std(values))
    objective = robust_inner_objective(values)
    for key, expected in (
        ("mean_mae", mean_mae),
        ("worst_mae", worst_mae),
        ("domain_sd", domain_sd),
        ("objective", objective),
    ):
        _same_float(item[key], expected, label=f"selection seed {key}")
    evidence_values = item["fold_evidence_sha256"]
    if (
        not isinstance(evidence_values, list)
        or len(evidence_values) != 5
        or any(not _valid_sha256(entry) for entry in evidence_values)
    ):
        raise ValueError("selection fold evidence changed")
    evidence = tuple(evidence_values)
    evidence_sha = _sha256("\0".join(evidence).encode("ascii"))
    if item["evidence_sha256"] != evidence_sha:
        raise ValueError("selection fold evidence state changed")
    state_payload = {
        "seed": seed,
        "seeded_candidate_sha256": seeded.state_sha256,
        "domain_mae": domain_mae,
        "mean_mae": mean_mae,
        "worst_mae": worst_mae,
        "domain_sd": domain_sd,
        "objective": objective,
        "target_sha256": _sha256(targets.tobytes(order="C")),
        "prediction_sha256": _sha256(predictions.tobytes(order="C")),
        "evidence_sha256": evidence_sha,
    }
    state = _canonical_sha256(state_payload)
    if item["state_sha256"] != state:
        raise ValueError("selection seed state changed")
    return SeedRerankEvaluation(
        seed=seed,
        seeded_candidate_sha256=seeded.state_sha256,
        domain_mae=domain_mae,
        mean_mae=mean_mae,
        worst_mae=worst_mae,
        domain_sd=domain_sd,
        objective=objective,
        oof_targets=targets,
        oof_predictions=predictions,
        fold_evidence_sha256=evidence,
        evidence_sha256=evidence_sha,
        state_sha256=state,
    )


def _parse_row_evidence(
    value: object,
    *,
    specimen_count: int,
    domain_ids: tuple[str, ...],
    targets: np.ndarray,
) -> RerankRow:
    item = _exact_mapping(
        value,
        keys={
            "candidate",
            "seeds",
            "domain_mae",
            "mean_mae",
            "worst_mae",
            "domain_sd",
            "objective",
            "complexity_key",
            "oof_targets",
            "oof_predictions",
            "state_sha256",
        },
        label="selection rerank row",
    )
    candidate = D8Candidate.from_payload(item["candidate"])
    seed_values = item["seeds"]
    if not isinstance(seed_values, list) or len(seed_values) != 3:
        raise ValueError("selection rerank seed roster changed")
    seeds = tuple(
        _parse_seed_evidence(
            seed_value,
            candidate=candidate,
            specimen_count=specimen_count,
            domain_ids=domain_ids,
            targets=targets,
        )
        for seed_value in seed_values
    )
    if len({entry.seed for entry in seeds}) != 3:
        raise ValueError("selection rerank seeds are not unique")
    domain_order = tuple(domain for domain, _ in seeds[0].domain_mae)
    domain_mae = tuple(
        (
            domain,
            _mean([dict(seed.domain_mae)[domain] for seed in seeds]),
        )
        for domain in domain_order
    )
    provided_domain_mae = item["domain_mae"]
    if not isinstance(provided_domain_mae, list) or len(provided_domain_mae) != 5:
        raise ValueError("selection rerank domain roster changed")
    for observed, (domain, mae) in zip(
        provided_domain_mae, domain_mae, strict=True
    ):
        if not isinstance(observed, list) or observed[0] != domain:
            raise ValueError("selection rerank domain roster changed")
        _same_float(observed[1], mae, label="selection rerank domain MAE")
    predictions = _readonly(
        np.mean(
            np.vstack([seed.oof_predictions for seed in seeds]),
            axis=0,
            dtype=np.float64,
        ),
        shape=(specimen_count,),
        label="selection mean OOF predictions",
    )
    observed_targets = _readonly(
        item["oof_targets"], shape=(specimen_count,), label="selection OOF targets"
    )
    observed_predictions = _readonly(
        item["oof_predictions"],
        shape=(specimen_count,),
        label="selection OOF predictions",
    )
    if not np.array_equal(observed_targets, targets) or not np.array_equal(
        observed_predictions, predictions
    ):
        raise ValueError("selection rerank OOF vectors changed")
    mean_mae = _mean([seed.mean_mae for seed in seeds])
    worst_mae = _mean([seed.worst_mae for seed in seeds])
    domain_sd = _mean([seed.domain_sd for seed in seeds])
    objective = _mean([seed.objective for seed in seeds])
    for key, expected in (
        ("mean_mae", mean_mae),
        ("worst_mae", worst_mae),
        ("domain_sd", domain_sd),
        ("objective", objective),
    ):
        _same_float(item[key], expected, label=f"selection rerank {key}")
    complexity = _complexity_key(candidate)
    if item["complexity_key"] != list(complexity):
        raise ValueError("selection rerank complexity changed")
    state_payload = {
        "candidate_sha256": candidate.state_sha256,
        "seed_states": [seed.state_sha256 for seed in seeds],
        "domain_mae": domain_mae,
        "mean_mae": mean_mae,
        "worst_mae": worst_mae,
        "domain_sd": domain_sd,
        "objective": objective,
        "complexity_key": complexity,
        "target_sha256": _sha256(targets.tobytes(order="C")),
        "prediction_sha256": _sha256(predictions.tobytes(order="C")),
    }
    state = _canonical_sha256(state_payload)
    if item["state_sha256"] != state:
        raise ValueError("selection rerank row state changed")
    return RerankRow(
        candidate=candidate,
        seeds=seeds,
        domain_mae=domain_mae,
        mean_mae=mean_mae,
        worst_mae=worst_mae,
        domain_sd=domain_sd,
        objective=objective,
        complexity_key=complexity,
        oof_targets=targets,
        oof_predictions=predictions,
        state_sha256=state,
    )


def _candidate_family_payload(candidate: D8Candidate) -> dict[str, object]:
    payload = candidate.to_payload(include_state=False)
    payload.pop("K_test")
    return payload


def validate_selection_evidence(
    value: object,
    *,
    outer_domain: str,
    config_sha256: str,
    search_view_sha256: str,
    specimen_ids: tuple[str, ...],
    domain_ids: tuple[str, ...],
    targets: np.ndarray,
    search_candidates: tuple[D8Candidate, ...],
) -> FrozenOuterSelection:
    """Recompute a serialized selection against registered pre-outer authority."""

    if (
        outer_domain not in DOMAIN_ORDER
        or not _valid_sha256(config_sha256)
        or not _valid_sha256(search_view_sha256)
        or type(specimen_ids) is not tuple
        or type(domain_ids) is not tuple
        or len(specimen_ids) != len(domain_ids)
        or len(specimen_ids) == 0
        or len(set(specimen_ids)) != len(specimen_ids)
        or any(type(item) is not str or not item for item in specimen_ids + domain_ids)
        or outer_domain in domain_ids
        or tuple(dict.fromkeys(domain_ids))
        != tuple(domain for domain in DOMAIN_ORDER if domain != outer_domain)
    ):
        raise ValueError("selection validation authority changed")
    specimen_count = len(specimen_ids)
    expected_targets = _readonly(
        targets,
        shape=(specimen_count,),
        label="selection validation targets",
    )
    item = _exact_mapping(
        value,
        keys={
            "schema_version",
            "scope",
            "outer_domain",
            "config_sha256",
            "search_view_sha256",
            "search_authority",
            "rerank",
            "finalists",
            "ensemble",
            "selected_candidates",
            "outer_evaluation_started",
            "state_sha256",
        },
        label="frozen selection",
    )
    if (
        item["schema_version"] != 2
        or item["scope"] != "d8_prospective_outer_selection"
        or item["outer_domain"] != outer_domain
        or item["config_sha256"] != config_sha256
        or item["search_view_sha256"] != search_view_sha256
        or item["outer_evaluation_started"] is not False
    ):
        raise ValueError("frozen selection authority changed")
    authority = _exact_mapping(
        item["search_authority"],
        keys={"specimen_ids", "domain_ids", "targets", "target_sha256"},
        label="selection search authority",
    )
    observed_specimen_ids = tuple(authority["specimen_ids"])
    observed_domain_ids = tuple(authority["domain_ids"])
    if observed_specimen_ids != specimen_ids or observed_domain_ids != domain_ids:
        raise ValueError("selection search roster changed")
    observed_targets = _readonly(
        authority["targets"],
        shape=(specimen_count,),
        label="selection search targets",
    )
    if (
        not np.array_equal(observed_targets, expected_targets)
        or authority["target_sha256"]
        != _sha256(observed_targets.tobytes(order="C"))
    ):
        raise ValueError("selection search targets changed")
    if (
        type(search_candidates) is not tuple
        or len(search_candidates) != 12
        or len({candidate.state_sha256 for candidate in search_candidates}) != 12
        or any(
            type(candidate) is not D8Candidate
            or candidate.config_sha256 != config_sha256
            for candidate in search_candidates
        )
    ):
        raise ValueError("selection search candidates changed")
    registered = search_candidates

    rerank_payload = _exact_mapping(
        item["rerank"],
        keys={"seed_count", "seeds", "rows", "finalist_states", "state_sha256"},
        label="selection rerank",
    )
    rows_payload = rerank_payload["rows"]
    if not isinstance(rows_payload, list) or len(rows_payload) != 12:
        raise ValueError("selection rerank row roster changed")
    rows = tuple(
        _parse_row_evidence(
            row,
            specimen_count=specimen_count,
            domain_ids=domain_ids,
            targets=expected_targets,
        )
        for row in rows_payload
    )
    if rows != tuple(sorted(rows, key=lambda row: row.rank_key)):
        raise ValueError("selection rerank ordering changed")
    if {row.candidate.state_sha256 for row in rows} != {
        candidate.state_sha256 for candidate in registered
    }:
        raise ValueError("selection rerank candidates differ from search")
    seeds = tuple(seed.seed for seed in rows[0].seeds)
    if (
        rerank_payload["seed_count"] != 3
        or rerank_payload["seeds"] != list(seeds)
        or any(tuple(seed.seed for seed in row.seeds) != seeds for row in rows)
    ):
        raise ValueError("selection rerank seed roster changed")
    finalist_states = [row.state_sha256 for row in rows[:4]]
    if rerank_payload["finalist_states"] != finalist_states:
        raise ValueError("selection rerank finalists changed")
    rerank_state = _canonical_sha256(
        {
            "outer_domain": outer_domain,
            "config_sha256": config_sha256,
            "search_view_sha256": search_view_sha256,
            "seeds": seeds,
            "row_states": [row.state_sha256 for row in rows],
            "finalist_states": finalist_states,
        }
    )
    if rerank_payload["state_sha256"] != rerank_state:
        raise ValueError("selection rerank state changed")

    finalist_payload = _exact_mapping(
        item["finalists"],
        keys={"cells", "selected_states", "state_sha256"},
        label="selection finalists",
    )
    cells_payload = finalist_payload["cells"]
    if not isinstance(cells_payload, list) or len(cells_payload) != 8:
        raise ValueError("selection finalist cell roster changed")
    cells = tuple(
        _parse_row_evidence(
            row,
            specimen_count=specimen_count,
            domain_ids=domain_ids,
            targets=expected_targets,
        )
        for row in cells_payload
    )
    if cells != tuple(sorted(cells, key=lambda row: row.rank_key)):
        raise ValueError("selection finalist ordering changed")
    selected: list[RerankRow] = []
    for source in rows[:4]:
        family = _candidate_family_payload(source.candidate)
        matches = tuple(
            cell
            for cell in cells
            if _candidate_family_payload(cell.candidate) == family
        )
        if len(matches) != 2 or {cell.candidate.K_test for cell in matches} != {8, 16}:
            raise ValueError("selection finalist K roster changed")
        selected.append(min(matches, key=lambda row: row.rank_key))
    selected_rows = tuple(sorted(selected, key=lambda row: row.rank_key))
    selected_states = [row.state_sha256 for row in selected_rows]
    if finalist_payload["selected_states"] != selected_states:
        raise ValueError("selection finalist choices changed")
    finalist_state = _canonical_sha256(
        {
            "outer_domain": outer_domain,
            "config_sha256": config_sha256,
            "search_view_sha256": search_view_sha256,
            "cell_states": [cell.state_sha256 for cell in cells],
            "selected_states": selected_states,
        }
    )
    if finalist_payload["state_sha256"] != finalist_state:
        raise ValueError("selection finalist state changed")

    ensemble_payload = _exact_mapping(
        item["ensemble"],
        keys={
            "accepted",
            "candidate_sha256",
            "weights",
            "crossfit_weights",
            "best_member_index",
            "best_member_objective",
            "objective",
            "objective_gain",
            "predictions",
            "state_sha256",
        },
        label="selection ensemble",
    )
    candidate_sha256 = tuple(row.candidate.state_sha256 for row in selected_rows)
    recomputed_ensemble = fit_nonnegative_ensemble(
        np.vstack([row.oof_predictions for row in selected_rows]),
        expected_targets,
        specimen_ids=specimen_ids,
        domain_ids=domain_ids,
        candidate_sha256=candidate_sha256,
        minimum_j_gain=1.0e-4,
    )
    if (
        ensemble_payload["accepted"] is not recomputed_ensemble.accepted
        or ensemble_payload["candidate_sha256"] != list(candidate_sha256)
        or ensemble_payload["best_member_index"]
        != recomputed_ensemble.best_member_index
        or ensemble_payload["state_sha256"] != recomputed_ensemble.state_sha256
    ):
        raise ValueError("selection ensemble state changed")
    for key, expected in (
        ("best_member_objective", recomputed_ensemble.best_member_objective),
        ("objective", recomputed_ensemble.objective),
        ("objective_gain", recomputed_ensemble.objective_gain),
    ):
        _same_float(ensemble_payload[key], expected, label=f"selection ensemble {key}")
    for key, expected in (
        ("weights", recomputed_ensemble.weights),
        ("crossfit_weights", recomputed_ensemble.crossfit_weights),
        ("predictions", recomputed_ensemble.predictions),
    ):
        observed = np.asarray(ensemble_payload[key], dtype=np.float64)
        if not np.array_equal(observed, expected):
            raise ValueError(f"selection ensemble {key} changed")
    selected_candidates = item["selected_candidates"]
    if (
        not isinstance(selected_candidates, list)
        or [D8Candidate.from_payload(value).state_sha256 for value in selected_candidates]
        != list(candidate_sha256)
    ):
        raise ValueError("selection selected candidates changed")
    document_without_state = dict(item)
    state = document_without_state.pop("state_sha256")
    expected_state = _canonical_sha256(document_without_state)
    if state != expected_state:
        raise ValueError("frozen selection state changed")
    return FrozenOuterSelection(
        outer_domain=outer_domain,
        config_sha256=config_sha256,
        search_view_sha256=search_view_sha256,
        selected_candidate_sha256=candidate_sha256,
        ensemble_sha256=recomputed_ensemble.state_sha256,
        outer_evaluation_started=False,
        state_sha256=expected_state,
    )


def validate_frozen_outer_selection(
    path: str | Path,
    *,
    view: D8SearchView,
    search_candidates: tuple[D8Candidate, ...],
) -> FrozenOuterSelection:
    """Reload and independently recompute a prospective selection document."""

    selection_path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(selection_path, flags)
    except OSError as error:
        raise ValueError("frozen selection is unavailable") from error
    try:
        info = os.fstat(descriptor)
        if info.st_size <= 0 or info.st_size > 64 * 1024 * 1024:
            raise ValueError("frozen selection size is invalid")
        payload = b""
        while len(payload) <= info.st_size:
            chunk = os.read(descriptor, min(1024 * 1024, info.st_size - len(payload)))
            if not chunk:
                break
            payload += chunk
    finally:
        os.close(descriptor)
    if len(payload) != info.st_size:
        raise ValueError("frozen selection changed while reading")
    try:
        document = json.loads(payload.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen selection is invalid JSON") from error
    view_state = validate_search_view(view)
    return validate_selection_evidence(
        document,
        outer_domain=view.outer_domain,
        config_sha256=view.config_sha256,
        search_view_sha256=view_state,
        specimen_ids=view.specimen_ids,
        domain_ids=view.dataset_ids,
        targets=np.asarray(view.data_view.cai_ratio, dtype=np.float64),
        search_candidates=search_candidates,
    )


def _selection_payload(
    reranked: RerankResult,
    finalists: FinalistResult,
    ensemble: EnsembleResult,
    view: D8SearchView,
) -> dict[str, object]:
    view_state = validate_search_view(view)
    if (
        type(reranked) is not RerankResult
        or type(finalists) is not FinalistResult
        or type(ensemble) is not EnsembleResult
    ):
        raise TypeError("selection freeze requires exact result values")
    authorities = {
        (reranked.outer_domain, reranked.config_sha256, reranked.search_view_sha256),
        (finalists.outer_domain, finalists.config_sha256, finalists.search_view_sha256),
        (view.outer_domain, view.config_sha256, view_state),
    }
    if len(authorities) != 1:
        raise ValueError("selection authorities differ")
    selected_hashes = tuple(item.candidate.state_sha256 for item in finalists.selected)
    if selected_hashes != ensemble.candidate_sha256:
        raise ValueError("ensemble candidates differ from selected finalists")
    return {
        "schema_version": 2,
        "scope": "d8_prospective_outer_selection",
        "outer_domain": view.outer_domain,
        "config_sha256": view.config_sha256,
        "search_view_sha256": view_state,
        "search_authority": _authority_payload(view),
        "rerank": {
            "seed_count": reranked.seed_count,
            "seeds": [item.seed for item in reranked.rows[0].seeds],
            "rows": [_row_evidence_payload(item) for item in reranked.rows],
            "finalist_states": [item.state_sha256 for item in reranked.finalists],
            "state_sha256": reranked.state_sha256,
        },
        "finalists": {
            "cells": [_row_evidence_payload(item) for item in finalists.cells],
            "selected_states": [item.state_sha256 for item in finalists.selected],
            "state_sha256": finalists.state_sha256,
        },
        "selected_candidates": [
            item.candidate.to_payload() for item in finalists.selected
        ],
        "ensemble": {
            "accepted": ensemble.accepted,
            "candidate_sha256": list(ensemble.candidate_sha256),
            "weights": ensemble.weights.tolist(),
            "crossfit_weights": ensemble.crossfit_weights.tolist(),
            "best_member_index": ensemble.best_member_index,
            "best_member_objective": ensemble.best_member_objective,
            "objective": ensemble.objective,
            "objective_gain": ensemble.objective_gain,
            "predictions": ensemble.predictions.tolist(),
            "state_sha256": ensemble.state_sha256,
        },
        "outer_evaluation_started": False,
    }


def freeze_outer_selection(
    reranked: RerankResult,
    *,
    finalists: FinalistResult,
    ensemble: EnsembleResult,
    view: D8SearchView,
    output: Path,
) -> FrozenOuterSelection:
    """Write one immutable prospective selection before outer data issuance."""

    payload = _selection_payload(reranked, finalists, ensemble, view)
    state = _canonical_sha256(payload)
    document = dict(payload)
    document["state_sha256"] = state
    encoded = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    path = Path(output)
    if path.suffix != ".json":
        raise ValueError("selection output must be JSON")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("existing frozen selection differs")
    else:
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with staged.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            staged.replace(path)
        finally:
            if staged.exists():
                staged.unlink()
    selected = tuple(item.candidate.state_sha256 for item in finalists.selected)
    return FrozenOuterSelection(
        outer_domain=view.outer_domain,
        config_sha256=view.config_sha256,
        search_view_sha256=view.state_sha256,
        selected_candidate_sha256=selected,
        ensemble_sha256=ensemble.state_sha256,
        outer_evaluation_started=False,
        state_sha256=state,
    )


__all__ = [
    "EnsembleResult",
    "FinalistResult",
    "FrozenOuterSelection",
    "RerankResult",
    "RerankRow",
    "SeedRerankEvaluation",
    "evaluate_finalists",
    "fit_nonnegative_ensemble",
    "freeze_outer_selection",
    "rerank_candidates",
    "validate_frozen_outer_selection",
    "validate_selection_evidence",
]
