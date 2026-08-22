"""Ordered formal E1--E5 evaluation for mechanics-consistent multi-view CAI."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .agreement_audit import AgreementAudit, audit_predictions
from .cooperative_regression import (
    CooperativeRegressionError,
    fit_cooperative,
)
from .gmvr_regression import GMvRWeights, fit_gmvr_weights
from .late_fusion import equal_fusion, fit_validation_weights
from .oof_predictions import (
    AuthoritativeInputs,
    IndependentOOFResult,
    evaluate_independent_views,
)
from .protocol import MultiViewProtocol
from .reliability import ReliabilityAudit, audit_reliability
from .search import (
    CooperativeCandidate,
    CooperativeSearchResult,
    select_cooperative_oof,
)
from .stacking import fit_stacker, select_stacker_oof
from .statistics import CommonBootstrap, common_domain_bootstrap
from .view_experts import fit_pca_basis, fit_view_expert


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    method: str
    domain_mae: tuple[tuple[str, float], ...]
    equal_domain_mae: float
    worst_domain_mae: float
    domain_mae_sd: float
    rmse: float
    r2: float
    domain_mae_mpa: tuple[tuple[str, float], ...]
    equal_domain_mae_mpa: float
    rmse_mpa: float


@dataclass(frozen=True, slots=True)
class E1FormalResult:
    independent: IndependentOOFResult
    audit: AgreementAudit
    reliability: ReliabilityAudit
    gate_status: str


@dataclass(frozen=True, slots=True)
class E2OuterState:
    outer_domain: str
    selected: CooperativeCandidate
    search: CooperativeSearchResult
    source_indices: tuple[int, ...]
    source_oof_predictions: np.ndarray
    outer_view_predictions: np.ndarray
    validation_weights: np.ndarray


@dataclass(frozen=True, slots=True)
class E2FormalResult:
    predictions: Mapping[str, np.ndarray]
    metrics: tuple[PerformanceMetrics, ...]
    outer_states: tuple[E2OuterState, ...]
    cooperative_improved_domains: int
    gate_status: str


@dataclass(frozen=True, slots=True)
class E3OuterState:
    outer_domain: str
    stacking_method: str
    gmvr: GMvRWeights
    gmvr_candidates: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class E3FormalResult:
    predictions: Mapping[str, np.ndarray]
    metrics: tuple[PerformanceMetrics, ...]
    outer_states: tuple[E3OuterState, ...]
    best_method: str
    best_improved_domains: int
    gate_status: str


@dataclass(frozen=True, slots=True)
class StressFoldState:
    scheme: str
    heldout_group: str
    pca_dimensions: tuple[int, ...]
    cooperative_candidate: CooperativeCandidate
    stacking_method: str
    gmvr_weights: np.ndarray


@dataclass(frozen=True, slots=True)
class StressMethodMetrics:
    scheme: str
    method: str
    group_mae: tuple[tuple[str, float], ...]
    equal_group_mae: float
    worst_group_mae: float


@dataclass(frozen=True, slots=True)
class StressSchemeResult:
    scheme: str
    group_values: tuple[str, ...]
    predictions: Mapping[str, np.ndarray]
    metrics: tuple[StressMethodMetrics, ...]
    fold_states: tuple[StressFoldState, ...]


@dataclass(frozen=True, slots=True)
class EngineeringStressResult:
    schemes: tuple[StressSchemeResult, ...]


@dataclass(frozen=True, slots=True)
class FormalChainResult:
    e1: E1FormalResult
    e2: E2FormalResult | None
    e3: E3FormalResult | None
    bootstrap: CommonBootstrap | None
    stress: EngineeringStressResult | None
    e4_status: str
    e5_status: str


def authorize_e4(
    *,
    fusion_mae: float,
    baseline_mae: float,
    best_single_mae: float,
    improved_domain_count: int,
) -> bool:
    """Apply the registered E3 complementarity success gate."""

    values = (fusion_mae, baseline_mae, best_single_mae)
    if any(
        not isinstance(item, (int, float)) or not math.isfinite(item) for item in values
    ):
        raise ValueError("E4 gate metrics must be finite")
    if type(improved_domain_count) is not int or not 0 <= improved_domain_count <= 6:
        raise ValueError("E4 improved-domain count is invalid")
    return bool(
        fusion_mae < baseline_mae
        and fusion_mae < best_single_mae
        and improved_domain_count >= 4
    )


def authorize_e5(
    *,
    e1_nontrivial: bool,
    complementarity_confirmed: bool,
    oracle_gap_fraction: float,
) -> bool:
    """Authorize transport only after a material deterministic oracle gap remains."""

    if type(e1_nontrivial) is not bool or type(complementarity_confirmed) is not bool:
        raise ValueError("E5 gate flags must be boolean")
    if not isinstance(oracle_gap_fraction, (int, float)) or not math.isfinite(
        oracle_gap_fraction
    ):
        raise ValueError("E5 oracle gap must be finite")
    return bool(
        e1_nontrivial and complementarity_confirmed and oracle_gap_fraction >= 0.05
    )


def stress_group_splits(
    groups: tuple[object, ...],
) -> tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]:
    """Return deterministic exhaustive leave-one-group-out index contracts."""

    if not isinstance(groups, tuple) or len(groups) < 2:
        raise ValueError("stress groups must be a nonempty tuple")
    values = tuple(str(item) for item in groups)
    if any(not item for item in values) or len(set(values)) < 2:
        raise ValueError("stress groups require at least two nonempty values")
    result = []
    for group in dict.fromkeys(values):
        fit = tuple(index for index, value in enumerate(values) if value != group)
        query = tuple(index for index, value in enumerate(values) if value == group)
        if not fit or not query:
            raise ValueError("stress group split is empty")
        result.append((group, fit, query))
    return tuple(result)


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    result = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    result.setflags(write=False)
    return result


def performance_metrics(
    method: str,
    targets: np.ndarray,
    predictions: np.ndarray,
    domains: tuple[str, ...],
    *,
    intact_strength_mpa: object,
) -> PerformanceMetrics:
    """Evaluate one deployable OOF method under equal-domain aggregation."""

    y = np.asarray(targets, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    domain_ids = tuple(domains)
    intact = np.asarray(intact_strength_mpa, dtype=np.float64)
    if (
        type(method) is not str
        or not method
        or y.shape != pred.shape
        or y.ndim != 1
        or intact.shape != y.shape
        or len(domain_ids) != len(y)
        or not np.all(np.isfinite(y))
        or not np.all(np.isfinite(pred))
        or not np.all(np.isfinite(intact))
        or np.any(intact <= 0.0)
    ):
        raise ValueError("method metric inputs are invalid")
    domain_array = np.asarray(domain_ids)
    order = tuple(dict.fromkeys(domain_ids))
    domain_mae = tuple(
        (
            domain,
            float(
                np.mean(
                    np.abs(y[domain_array == domain] - pred[domain_array == domain])
                )
            ),
        )
        for domain in order
    )
    domain_values = np.asarray([item[1] for item in domain_mae])
    residual = y - pred
    residual_mpa = residual * intact
    domain_mae_mpa = tuple(
        (
            domain,
            float(np.mean(np.abs(residual_mpa[domain_array == domain]))),
        )
        for domain in order
    )
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return PerformanceMetrics(
        method=method,
        domain_mae=domain_mae,
        equal_domain_mae=float(np.mean(domain_values)),
        worst_domain_mae=float(np.max(domain_values)),
        domain_mae_sd=float(np.std(domain_values, ddof=0)),
        rmse=float(np.sqrt(np.mean(residual**2))),
        r2=float(1.0 - np.sum(residual**2) / denominator) if denominator > 0.0 else 0.0,
        domain_mae_mpa=domain_mae_mpa,
        equal_domain_mae_mpa=float(
            np.mean([item[1] for item in domain_mae_mpa])
        ),
        rmse_mpa=float(np.sqrt(np.mean(residual_mpa**2))),
    )


def _descriptor_bins(values: np.ndarray) -> tuple[str, ...]:
    quartiles = np.quantile(values, (0.25, 0.5, 0.75), method="linear")
    labels = ("Q1_low", "Q2", "Q3", "Q4_high")
    return tuple(
        labels[int(np.searchsorted(quartiles, value, side="left"))] for value in values
    )


def _e1_groups(inputs: AuthoritativeInputs) -> dict[str, tuple[object, ...]]:
    ply = tuple(round(value * 24.0) for value in inputs.metadata[:, 1])
    layup = tuple(
        "cross_ply" if value >= 0.5 else "quasi_isotropic"
        for value in inputs.metadata[:, 2]
    )
    groups: dict[str, tuple[object, ...]] = {"ply_count": ply, "layup": layup}
    for index, name in enumerate(inputs.damage_descriptor_names):
        groups[f"{name}_quartile"] = _descriptor_bins(
            np.asarray(inputs.damage_descriptors[:, index], dtype=np.float64)
        )
    return groups


def run_e1(
    inputs: AuthoritativeInputs, *, protocol: MultiViewProtocol
) -> E1FormalResult:
    independent = evaluate_independent_views(inputs, protocol=protocol)
    audit = audit_predictions(
        independent.targets,
        independent.predictions,
        independent.dataset_ids,
        view_names=independent.view_names,
        groups=_e1_groups(inputs),
        baseline_mae=protocol.baseline_mae,
    )
    reliability = audit_reliability(
        independent.targets,
        independent.predictions,
        deployable_predictions={
            "independent_equal_fusion": equal_fusion(independent.predictions)
        },
    )
    return E1FormalResult(
        independent=independent,
        audit=audit,
        reliability=reliability,
        gate_status=audit.gate_status,
    )


def _selection_dimensions(
    independent: IndependentOOFResult, outer_domain: str
) -> tuple[int, ...]:
    values = tuple(
        item.pca_dimension
        for view in independent.view_names
        for item in independent.selections
        if item.outer_domain == outer_domain and item.view == view
    )
    if len(values) != len(independent.view_names):
        raise ValueError("independent view dimensions are incomplete")
    return values


def _fold_designs(
    inputs: AuthoritativeInputs,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    dimensions: tuple[int, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    fit_designs: list[np.ndarray] = []
    query_designs: list[np.ndarray] = []
    for view_index, dimension in enumerate(dimensions):
        embeddings = np.asarray(inputs.features[:, view_index], dtype=np.float64)
        basis = fit_pca_basis(embeddings, fit_indices, maximum_dimension=dimension)
        components = np.asarray(basis.components[:dimension])
        fit_projected = (embeddings[fit_indices] - basis.mean) @ components.T
        query_projected = (embeddings[query_indices] - basis.mean) @ components.T
        fit_designs.append(
            np.column_stack((inputs.metadata[fit_indices], fit_projected))
        )
        query_designs.append(
            np.column_stack((inputs.metadata[query_indices], query_projected))
        )
    return tuple(fit_designs), tuple(query_designs)


def _method_mapping(
    values: dict[str, np.ndarray], *, rows: int
) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, raw in values.items():
        array = np.asarray(raw, dtype=np.float64)
        if array.shape != (rows,) or not np.all(np.isfinite(array)):
            raise ValueError(f"formal OOF method is incomplete: {name}")
        result[name] = _readonly(array)
    return MappingProxyType(result)


def _improved_domains(candidate: PerformanceMetrics, full: PerformanceMetrics) -> int:
    full_values = dict(full.domain_mae)
    return sum(value < full_values[domain] for domain, value in candidate.domain_mae)


def run_e2(
    inputs: AuthoritativeInputs,
    *,
    protocol: MultiViewProtocol,
    e1: E1FormalResult,
) -> E2FormalResult:
    independent = e1.independent
    y = np.asarray(independent.targets)
    domains = np.asarray(independent.dataset_ids)
    rows = len(y)
    outputs = {
        independent.view_names[index].lower(): np.asarray(
            independent.predictions[:, index]
        ).copy()
        for index in range(len(independent.view_names))
    }
    outputs["independent_equal_fusion"] = equal_fusion(independent.predictions)
    outputs["validation_weighted"] = np.full(rows, np.nan)
    outputs["cooperative_selected"] = np.full(rows, np.nan)
    source_by_outer = {item.outer_domain: item for item in independent.source_oof}
    outer_states: list[E2OuterState] = []
    candidates = tuple(
        CooperativeCandidate(loss, strength)
        for loss in protocol.target_losses
        for strength in protocol.consistency_grid
    )
    for outer_domain in protocol.domain_order:
        outer_query = np.flatnonzero(domains == outer_domain).astype(np.int64)
        source = np.flatnonzero(domains != outer_domain).astype(np.int64)
        source_domains = domains[source]
        dimensions = _selection_dimensions(independent, outer_domain)
        positions = {int(index): position for position, index in enumerate(source)}
        candidate_predictions = {
            item: np.full((len(source), len(protocol.views)), np.nan)
            for item in candidates
        }
        unavailable: set[CooperativeCandidate] = set()
        for inner_domain in tuple(sorted(set(source_domains.tolist()))):
            inner_fit = source[source_domains != inner_domain]
            inner_query = source[source_domains == inner_domain]
            fit_designs, query_designs = _fold_designs(
                inputs, inner_fit, inner_query, dimensions
            )
            query_positions = [positions[int(index)] for index in inner_query]
            for candidate in candidates:
                if candidate in unavailable:
                    continue
                try:
                    fitted = fit_cooperative(
                        fit_designs,
                        y[inner_fit],
                        lambda_consistency=candidate.lambda_consistency,
                        loss=candidate.loss,
                        alpha=protocol.ridge_alpha,
                        huber_delta=protocol.huber_delta,
                    )
                    candidate_predictions[candidate][query_positions] = fitted.predict(
                        query_designs
                    )
                except CooperativeRegressionError:
                    unavailable.add(candidate)
        candidate_predictions = {
            candidate: values
            for candidate, values in candidate_predictions.items()
            if candidate not in unavailable and np.all(np.isfinite(values))
        }
        if not candidate_predictions:
            raise ValueError("all cooperative candidates failed")
        search = select_cooperative_oof(
            candidate_predictions,
            targets=y[source],
            domains=tuple(domains[source].tolist()),
        )
        fit_designs, query_designs = _fold_designs(
            inputs, source, outer_query, dimensions
        )
        selected_fit = fit_cooperative(
            fit_designs,
            y[source],
            lambda_consistency=search.selected.lambda_consistency,
            loss=search.selected.loss,
            alpha=protocol.ridge_alpha,
            huber_delta=protocol.huber_delta,
        )
        outer_view_predictions = selected_fit.predict(query_designs)
        outputs["cooperative_selected"][outer_query] = np.mean(
            outer_view_predictions, axis=1
        )
        source_record = source_by_outer[outer_domain]
        weight_fit = fit_validation_weights(
            source_record.predictions,
            y[source],
            domains=source_record.dataset_ids,
        )
        outputs["validation_weighted"][outer_query] = weight_fit.predict(
            independent.predictions[outer_query]
        )
        outer_states.append(
            E2OuterState(
                outer_domain=outer_domain,
                selected=search.selected,
                search=search,
                source_indices=tuple(int(item) for item in source),
                source_oof_predictions=_readonly(
                    candidate_predictions[search.selected]
                ),
                outer_view_predictions=_readonly(outer_view_predictions),
                validation_weights=_readonly(np.asarray(weight_fit.weights)),
            )
        )
    prediction_mapping = _method_mapping(outputs, rows=rows)
    metrics = tuple(
        performance_metrics(
            name,
            y,
            values,
            independent.dataset_ids,
            intact_strength_mpa=independent.intact_strength_mpa,
        )
        for name, values in prediction_mapping.items()
    )
    metric_by_name = {item.method: item for item in metrics}
    full = metric_by_name["full"]
    cooperative = metric_by_name["cooperative_selected"]
    improved = _improved_domains(cooperative, full)
    status = (
        "GO"
        if cooperative.equal_domain_mae < protocol.baseline_mae and improved >= 4
        else "NO_GO"
    )
    return E2FormalResult(
        predictions=prediction_mapping,
        metrics=metrics,
        outer_states=tuple(outer_states),
        cooperative_improved_domains=improved,
        gate_status=status,
    )


def run_e3(
    inputs: AuthoritativeInputs,
    *,
    protocol: MultiViewProtocol,
    e1: E1FormalResult,
    e2: E2FormalResult,
) -> E3FormalResult:
    independent = e1.independent
    y = np.asarray(independent.targets)
    domains = np.asarray(independent.dataset_ids)
    rows = len(y)
    outputs = {
        "equal_fusion": np.full(rows, np.nan),
        "validation_weighted": np.full(rows, np.nan),
        "stacking_ridge": np.full(rows, np.nan),
        "stacking_nonnegative_ridge": np.full(rows, np.nan),
        "stacking_huber": np.full(rows, np.nan),
        "stacking_selected": np.full(rows, np.nan),
        "gmvr_selected": np.full(rows, np.nan),
    }
    source_by_outer = {item.outer_domain: item for item in independent.source_oof}
    cooperative_by_outer = {item.outer_domain: item for item in e2.outer_states}
    outer_states: list[E3OuterState] = []
    for outer_domain in protocol.domain_order:
        query = np.flatnonzero(domains == outer_domain).astype(np.int64)
        source = np.flatnonzero(domains != outer_domain).astype(np.int64)
        outer_base = np.asarray(independent.predictions[query])
        source_record = source_by_outer[outer_domain]
        source_base = np.asarray(source_record.predictions)
        source_domain_ids = source_record.dataset_ids
        outputs["equal_fusion"][query] = equal_fusion(outer_base)
        validation = fit_validation_weights(
            source_base, y[source], domains=source_domain_ids
        )
        outputs["validation_weighted"][query] = validation.predict(outer_base)
        selection = select_stacker_oof(
            source_base,
            y[source],
            source_domain_ids,
            methods=("ridge", "nonnegative_ridge", "huber"),
            alpha=1.0,
        )
        for method in ("ridge", "nonnegative_ridge", "huber"):
            fitted = fit_stacker(source_base, y[source], method=method, alpha=1.0)
            outputs[f"stacking_{method}"][query] = fitted.predict(outer_base)
        outputs["stacking_selected"][query] = selection.fitted.predict(outer_base)
        cooperative = cooperative_by_outer[outer_domain]
        gmvr_ranked: list[tuple[tuple[float, float], GMvRWeights]] = []
        gmvr_scores: list[tuple[float, float]] = []
        for complementarity_strength in protocol.complementarity_grid:
            gmvr = fit_gmvr_weights(
                cooperative.source_oof_predictions,
                y[source],
                domains=source_domain_ids,
                lambda_consistency=cooperative.selected.lambda_consistency,
                lambda_complementarity=complementarity_strength,
            )
            score = performance_metrics(
                "gmvr_source",
                y[source],
                gmvr.predictions,
                source_domain_ids,
                intact_strength_mpa=independent.intact_strength_mpa[source],
            ).equal_domain_mae
            gmvr_scores.append((complementarity_strength, score))
            gmvr_ranked.append(((score, complementarity_strength), gmvr))
        selected_gmvr = min(gmvr_ranked, key=lambda item: item[0])[1]
        outputs["gmvr_selected"][query] = selected_gmvr.predict(
            cooperative.outer_view_predictions
        )
        outer_states.append(
            E3OuterState(
                outer_domain=outer_domain,
                stacking_method=selection.selected_method,
                gmvr=selected_gmvr,
                gmvr_candidates=tuple(gmvr_scores),
            )
        )
    prediction_mapping = _method_mapping(outputs, rows=rows)
    metrics = tuple(
        performance_metrics(
            name,
            y,
            values,
            independent.dataset_ids,
            intact_strength_mpa=independent.intact_strength_mpa,
        )
        for name, values in prediction_mapping.items()
    )
    best = min(
        enumerate(metrics),
        key=lambda item: (item[1].equal_domain_mae, item[0]),
    )[1]
    full = performance_metrics(
        "full",
        y,
        independent.predictions[:, 0],
        independent.dataset_ids,
        intact_strength_mpa=independent.intact_strength_mpa,
    )
    best_single_mae = min(item.equal_domain_mae for item in e1.audit.view_metrics)
    improved = _improved_domains(best, full)
    status = (
        "GO"
        if authorize_e4(
            fusion_mae=best.equal_domain_mae,
            baseline_mae=protocol.baseline_mae,
            best_single_mae=best_single_mae,
            improved_domain_count=improved,
        )
        else "NO_GO"
    )
    return E3FormalResult(
        predictions=prediction_mapping,
        metrics=metrics,
        outer_states=tuple(outer_states),
        best_method=best.method,
        best_improved_domains=improved,
        gate_status=status,
    )


def _domain_effect(
    full: PerformanceMetrics, candidate: PerformanceMetrics
) -> np.ndarray:
    full_values = dict(full.domain_mae)
    return np.asarray(
        [full_values[domain] - value for domain, value in candidate.domain_mae],
        dtype=np.float64,
    )


def _confirmatory_bootstrap(
    e1: E1FormalResult, e2: E2FormalResult, e3: E3FormalResult
) -> CommonBootstrap:
    y = np.asarray(e1.independent.targets)
    full = performance_metrics(
        "full",
        y,
        e1.independent.predictions[:, 0],
        e1.independent.dataset_ids,
        intact_strength_mpa=e1.independent.intact_strength_mpa,
    )
    e2_metrics = {item.method: item for item in e2.metrics}
    e3_metrics = {item.method: item for item in e3.metrics}
    return common_domain_bootstrap(
        {
            "e2_cooperative": _domain_effect(full, e2_metrics["cooperative_selected"]),
            "e3_validation_weighted": _domain_effect(
                full, e3_metrics["validation_weighted"]
            ),
            "e3_stacking_selected": _domain_effect(
                full, e3_metrics["stacking_selected"]
            ),
            "e3_gmvr_selected": _domain_effect(full, e3_metrics["gmvr_selected"]),
        }
    )


def _stress_groups(inputs: AuthoritativeInputs) -> dict[str, tuple[str, ...]]:
    return {
        "leave_ply_count_out": tuple(
            str(round(value * 24.0)) for value in inputs.metadata[:, 1]
        ),
        "leave_layup_family_out": tuple(
            "cross_ply" if value >= 0.5 else "quasi_isotropic"
            for value in inputs.metadata[:, 2]
        ),
    }


def _stress_independent_fold(
    inputs: AuthoritativeInputs,
    *,
    protocol: MultiViewProtocol,
    source: np.ndarray,
    query: np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    y = np.asarray(inputs.targets)
    domains = np.asarray(inputs.dataset_ids)
    source_domains = domains[source]
    positions = {int(index): position for position, index in enumerate(source)}
    source_predictions = np.full((len(source), len(protocol.views)), np.nan)
    query_predictions = np.full((len(query), len(protocol.views)), np.nan)
    selected_dimensions: list[int] = []
    for view_index in range(len(protocol.views)):
        embeddings = np.asarray(inputs.features[:, view_index], dtype=np.float64)
        by_dimension = {
            dimension: np.full(len(source), np.nan)
            for dimension in protocol.pca_dimensions
        }
        for inner_domain in tuple(sorted(set(source_domains.tolist()))):
            inner_fit = source[source_domains != inner_domain]
            inner_query = source[source_domains == inner_domain]
            if len(inner_fit) <= max(protocol.pca_dimensions) or not len(inner_query):
                raise ValueError("stress inner domain fold is underdetermined")
            basis = fit_pca_basis(
                embeddings,
                inner_fit,
                maximum_dimension=max(protocol.pca_dimensions),
            )
            local_query = [positions[int(index)] for index in inner_query]
            for dimension in protocol.pca_dimensions:
                expert = fit_view_expert(
                    embeddings,
                    inputs.metadata,
                    y,
                    inner_fit,
                    pca_dimension=dimension,
                    alpha=protocol.ridge_alpha,
                    pca_basis=basis,
                )
                by_dimension[dimension][local_query] = expert.predict(
                    embeddings[inner_query], inputs.metadata[inner_query]
                )
        ranked: list[tuple[float, int]] = []
        for dimension in protocol.pca_dimensions:
            fold_mae = []
            for inner_domain in tuple(sorted(set(source_domains.tolist()))):
                local = np.flatnonzero(source_domains == inner_domain)
                fold_mae.append(
                    float(
                        np.mean(
                            np.abs(y[source[local]] - by_dimension[dimension][local])
                        )
                    )
                )
            ranked.append((float(np.mean(fold_mae)), dimension))
        selected_score, selected = ranked[0]
        for score, dimension in ranked[1:]:
            if score < selected_score - 1e-12:
                selected_score, selected = score, dimension
        selected_dimensions.append(selected)
        source_predictions[:, view_index] = by_dimension[selected]
        basis = fit_pca_basis(embeddings, source, maximum_dimension=selected)
        expert = fit_view_expert(
            embeddings,
            inputs.metadata,
            y,
            source,
            pca_dimension=selected,
            alpha=protocol.ridge_alpha,
            pca_basis=basis,
        )
        query_predictions[:, view_index] = expert.predict(
            embeddings[query], inputs.metadata[query]
        )
    if not np.all(np.isfinite(source_predictions)) or not np.all(
        np.isfinite(query_predictions)
    ):
        raise ValueError("stress independent predictions are incomplete")
    return (
        tuple(selected_dimensions),
        _readonly(source_predictions),
        _readonly(query_predictions),
    )


def _stress_cooperative_fold(
    inputs: AuthoritativeInputs,
    *,
    protocol: MultiViewProtocol,
    source: np.ndarray,
    query: np.ndarray,
    dimensions: tuple[int, ...],
) -> tuple[CooperativeCandidate, np.ndarray, np.ndarray]:
    y = np.asarray(inputs.targets)
    domains = np.asarray(inputs.dataset_ids)
    source_domains = domains[source]
    positions = {int(index): position for position, index in enumerate(source)}
    candidates = tuple(
        CooperativeCandidate(loss, strength)
        for loss in protocol.target_losses
        for strength in protocol.consistency_grid
    )
    predictions = {
        candidate: np.full((len(source), len(protocol.views)), np.nan)
        for candidate in candidates
    }
    unavailable: set[CooperativeCandidate] = set()
    for inner_domain in tuple(sorted(set(source_domains.tolist()))):
        inner_fit = source[source_domains != inner_domain]
        inner_query = source[source_domains == inner_domain]
        fit_designs, query_designs = _fold_designs(
            inputs, inner_fit, inner_query, dimensions
        )
        local_query = [positions[int(index)] for index in inner_query]
        for candidate in candidates:
            if candidate in unavailable:
                continue
            try:
                fitted = fit_cooperative(
                    fit_designs,
                    y[inner_fit],
                    lambda_consistency=candidate.lambda_consistency,
                    loss=candidate.loss,
                    alpha=protocol.ridge_alpha,
                    huber_delta=protocol.huber_delta,
                )
                predictions[candidate][local_query] = fitted.predict(query_designs)
            except CooperativeRegressionError:
                unavailable.add(candidate)
    predictions = {
        candidate: values
        for candidate, values in predictions.items()
        if candidate not in unavailable and np.all(np.isfinite(values))
    }
    search = select_cooperative_oof(
        predictions,
        targets=y[source],
        domains=tuple(source_domains.tolist()),
    )
    fit_designs, query_designs = _fold_designs(inputs, source, query, dimensions)
    fitted = fit_cooperative(
        fit_designs,
        y[source],
        lambda_consistency=search.selected.lambda_consistency,
        loss=search.selected.loss,
        alpha=protocol.ridge_alpha,
        huber_delta=protocol.huber_delta,
    )
    return (
        search.selected,
        _readonly(predictions[search.selected]),
        _readonly(fitted.predict(query_designs)),
    )


def _stress_scheme(
    inputs: AuthoritativeInputs,
    *,
    protocol: MultiViewProtocol,
    scheme: str,
    group_values: tuple[str, ...],
) -> StressSchemeResult:
    y = np.asarray(inputs.targets)
    domains = np.asarray(inputs.dataset_ids)
    methods = (
        "full",
        "bilinear_50",
        "bilinear_25",
        "equal_fusion",
        "validation_weighted",
        "stacking_selected",
        "cooperative_selected",
        "gmvr_selected",
    )
    outputs = {method: np.full(len(y), np.nan) for method in methods}
    states: list[StressFoldState] = []
    for heldout_group, fit_tuple, query_tuple in stress_group_splits(group_values):
        source = np.asarray(fit_tuple, dtype=np.int64)
        query = np.asarray(query_tuple, dtype=np.int64)
        dimensions, source_base, query_base = _stress_independent_fold(
            inputs, protocol=protocol, source=source, query=query
        )
        outputs["full"][query] = query_base[:, 0]
        outputs["bilinear_50"][query] = query_base[:, 1]
        outputs["bilinear_25"][query] = query_base[:, 2]
        outputs["equal_fusion"][query] = equal_fusion(query_base)
        validation = fit_validation_weights(
            source_base,
            y[source],
            domains=tuple(domains[source].tolist()),
        )
        outputs["validation_weighted"][query] = validation.predict(query_base)
        stacking = select_stacker_oof(
            source_base,
            y[source],
            tuple(domains[source].tolist()),
            methods=("ridge", "nonnegative_ridge", "huber"),
            alpha=1.0,
        )
        outputs["stacking_selected"][query] = stacking.fitted.predict(query_base)
        cooperative_candidate, cooperative_source, cooperative_query = (
            _stress_cooperative_fold(
                inputs,
                protocol=protocol,
                source=source,
                query=query,
                dimensions=dimensions,
            )
        )
        outputs["cooperative_selected"][query] = np.mean(cooperative_query, axis=1)
        gmvr_candidates: list[tuple[tuple[float, float], GMvRWeights]] = []
        for strength in protocol.complementarity_grid:
            gmvr = fit_gmvr_weights(
                cooperative_source,
                y[source],
                domains=tuple(domains[source].tolist()),
                lambda_consistency=cooperative_candidate.lambda_consistency,
                lambda_complementarity=strength,
            )
            score = performance_metrics(
                "stress_gmvr_source",
                y[source],
                gmvr.predictions,
                tuple(domains[source].tolist()),
                intact_strength_mpa=inputs.intact_strength_mpa[source],
            ).equal_domain_mae
            gmvr_candidates.append(((score, strength), gmvr))
        gmvr = min(gmvr_candidates, key=lambda item: item[0])[1]
        outputs["gmvr_selected"][query] = gmvr.predict(cooperative_query)
        states.append(
            StressFoldState(
                scheme=scheme,
                heldout_group=heldout_group,
                pca_dimensions=dimensions,
                cooperative_candidate=cooperative_candidate,
                stacking_method=stacking.selected_method,
                gmvr_weights=_readonly(np.asarray(gmvr.weights)),
            )
        )
    predictions = _method_mapping(outputs, rows=len(y))
    group_array = np.asarray(group_values)
    group_order = tuple(dict.fromkeys(group_values))
    metrics: list[StressMethodMetrics] = []
    for method, prediction in predictions.items():
        group_mae = tuple(
            (
                group,
                float(
                    np.mean(
                        np.abs(
                            y[group_array == group] - prediction[group_array == group]
                        )
                    )
                ),
            )
            for group in group_order
        )
        values = np.asarray([item[1] for item in group_mae])
        metrics.append(
            StressMethodMetrics(
                scheme=scheme,
                method=method,
                group_mae=group_mae,
                equal_group_mae=float(np.mean(values)),
                worst_group_mae=float(np.max(values)),
            )
        )
    return StressSchemeResult(
        scheme=scheme,
        group_values=group_values,
        predictions=predictions,
        metrics=tuple(metrics),
        fold_states=tuple(states),
    )


def run_engineering_stress_tests(
    inputs: AuthoritativeInputs, *, protocol: MultiViewProtocol
) -> EngineeringStressResult:
    """Run registered leave-ply-count-out and leave-layup-family-out tests."""

    return EngineeringStressResult(
        schemes=tuple(
            _stress_scheme(
                inputs,
                protocol=protocol,
                scheme=scheme,
                group_values=values,
            )
            for scheme, values in _stress_groups(inputs).items()
        )
    )


def run_formal_chain(
    inputs: AuthoritativeInputs, *, protocol: MultiViewProtocol
) -> FormalChainResult:
    """Execute the registered stages in order and stop unauthorized branches."""

    e1 = run_e1(inputs, protocol=protocol)
    if e1.gate_status != "GO":
        return FormalChainResult(
            e1=e1,
            e2=None,
            e3=None,
            bootstrap=None,
            stress=None,
            e4_status="NOT_AUTHORIZED",
            e5_status="NOT_AUTHORIZED",
        )
    e2 = run_e2(inputs, protocol=protocol, e1=e1)
    e3 = run_e3(inputs, protocol=protocol, e1=e1, e2=e2)
    bootstrap = _confirmatory_bootstrap(e1, e2, e3)
    stress = run_engineering_stress_tests(inputs, protocol=protocol)
    e4_status = "AUTHORIZED_NOT_RUN" if e3.gate_status == "GO" else "NO_GO"
    best_metrics = next(item for item in e3.metrics if item.method == e3.best_method)
    oracle_gap = max(best_metrics.equal_domain_mae - e1.audit.oracle_mae, 0.0)
    oracle_gap_fraction = oracle_gap / best_metrics.equal_domain_mae
    e5_status = (
        "AUTHORIZED_NOT_RUN"
        if authorize_e5(
            e1_nontrivial=e1.gate_status == "GO",
            complementarity_confirmed=e3.gate_status == "GO",
            oracle_gap_fraction=oracle_gap_fraction,
        )
        else "NO_GO"
    )
    return FormalChainResult(
        e1=e1,
        e2=e2,
        e3=e3,
        bootstrap=bootstrap,
        stress=stress,
        e4_status=e4_status,
        e5_status=e5_status,
    )
