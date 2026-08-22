"""Strict nested domain-held-out predictions for the three registered views."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmc_bbdm.aei_selective_invariance.feature_bank import PairedFeatureBank
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data
from cmc_bbdm.hasebe_cai import read_cai_outcomes

from .protocol import MultiViewProtocol, SourceAuthority
from .view_experts import PCABasis, fit_pca_basis, fit_view_expert


class OOFPredictionError(ValueError):
    """Raised when nested multi-view evaluation loses identity or isolation."""


def _readonly(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    result = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(array.shape)
    result.setflags(write=False)
    return result


def _source(protocol: MultiViewProtocol, name: str) -> SourceAuthority:
    matches = tuple(item for item in protocol.sources if item.name == name)
    if len(matches) != 1:
        raise OOFPredictionError(f"protocol source is missing: {name}")
    return matches[0]


@dataclass(frozen=True, slots=True)
class AuthoritativeInputs:
    features: np.ndarray
    metadata: np.ndarray
    damage_descriptors: np.ndarray
    damage_descriptor_names: tuple[str, ...]
    targets: np.ndarray
    cai_strength_mpa: np.ndarray
    intact_strength_mpa: np.ndarray
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    view_names: tuple[str, ...]
    feature_state_sha256: str


@dataclass(frozen=True, slots=True)
class FitEvent:
    fit_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    fit_domains: tuple[str, ...]
    outer_domain: str
    inner_domain: str | None
    view: str
    pca_dimension: int


@dataclass(frozen=True, slots=True)
class ViewSelection:
    outer_domain: str
    view: str
    pca_dimension: int
    dimension_scores: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class SourceOOFPredictions:
    outer_domain: str
    source_indices: tuple[int, ...]
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    predictions: np.ndarray


@dataclass(frozen=True, slots=True)
class IndependentOOFResult:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    view_names: tuple[str, ...]
    targets: np.ndarray
    cai_strength_mpa: np.ndarray
    intact_strength_mpa: np.ndarray
    predictions: np.ndarray
    selections: tuple[ViewSelection, ...]
    source_oof: tuple[SourceOOFPredictions, ...]
    state_sha256: str


FitHook = Callable[[FitEvent], None]


def load_authoritative_inputs(
    protocol: MultiViewProtocol, *, project_root: str | Path
) -> AuthoritativeInputs:
    """Load and cross-bind the immutable A2 bank and frozen V3 cohort."""

    root = Path(project_root).resolve()
    bank_source = _source(protocol, "paired_feature_bank")
    bank = PairedFeatureBank.load(bank_source.path, expected_sha256=bank_source.sha256)
    p1_config_source = _source(protocol, "p1_config")
    config = load_v3_config(p1_config_source.path, project_root=root)
    data = load_data(config, root)
    specimen_ids = tuple(str(item) for item in data.sample_ids.tolist())
    dataset_ids = tuple(str(item) for item in data.dataset_ids.tolist())
    if (
        bank.specimen_ids != specimen_ids
        or bank.dataset_ids != dataset_ids
        or bank.view_names != protocol.views
        or len(specimen_ids) != protocol.specimen_count
    ):
        raise OOFPredictionError("A2 feature bank and V3 cohort identities changed")
    outcomes = read_cai_outcomes(config.sources["cai_workbook"].path)
    try:
        cai_strength_mpa = np.asarray(
            [outcomes.records[item].cai_strength_mpa for item in specimen_ids],
            dtype=np.float64,
        )
    except KeyError as error:
        raise OOFPredictionError("registered MPa response roster changed") from error
    targets = np.asarray(data.cai_ratio, dtype=np.float64)
    if (
        cai_strength_mpa.shape != targets.shape
        or not np.all(np.isfinite(cai_strength_mpa))
        or np.any(cai_strength_mpa <= 0.0)
        or np.any(targets <= 0.0)
    ):
        raise OOFPredictionError("registered MPa response scale is invalid")
    intact_strength_mpa = cai_strength_mpa / targets
    return AuthoritativeInputs(
        features=_readonly(np.asarray(bank.features, dtype=np.float64)),
        metadata=_readonly(np.asarray(data.metadata13, dtype=np.float64)),
        damage_descriptors=_readonly(
            np.asarray(data.scalar_internal3, dtype=np.float64)
        ),
        damage_descriptor_names=(
            "projected_damage_area",
            "damage_height",
            "damage_width",
        ),
        targets=_readonly(targets),
        cai_strength_mpa=_readonly(cai_strength_mpa),
        intact_strength_mpa=_readonly(intact_strength_mpa),
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        view_names=bank.view_names,
        feature_state_sha256=bank.state_sha256,
    )


def _emit(
    hook: FitHook | None,
    *,
    inputs: AuthoritativeInputs,
    fit: np.ndarray,
    query: np.ndarray,
    outer_domain: str,
    inner_domain: str | None,
    view: str,
    dimension: int,
) -> None:
    if hook is None:
        return
    hook(
        FitEvent(
            fit_ids=tuple(inputs.specimen_ids[int(index)] for index in fit),
            query_ids=tuple(inputs.specimen_ids[int(index)] for index in query),
            fit_domains=tuple(inputs.dataset_ids[int(index)] for index in fit),
            outer_domain=outer_domain,
            inner_domain=inner_domain,
            view=view,
            pca_dimension=dimension,
        )
    )


def evaluate_independent_views(
    inputs: AuthoritativeInputs,
    *,
    protocol: MultiViewProtocol,
    fit_hook: FitHook | None = None,
) -> IndependentOOFResult:
    """Select each view on source domains and predict every outer domain once."""

    features = np.asarray(inputs.features, dtype=np.float64)
    metadata = np.asarray(inputs.metadata, dtype=np.float64)
    targets = np.asarray(inputs.targets, dtype=np.float64)
    domains = np.asarray(inputs.dataset_ids, dtype=str)
    if (
        features.shape != (protocol.specimen_count, len(protocol.views), 512)
        or metadata.shape != (protocol.specimen_count, 13)
        or targets.shape != (protocol.specimen_count,)
        or inputs.cai_strength_mpa.shape != targets.shape
        or inputs.intact_strength_mpa.shape != targets.shape
        or not np.all(np.isfinite(inputs.cai_strength_mpa))
        or not np.all(np.isfinite(inputs.intact_strength_mpa))
        or np.any(inputs.cai_strength_mpa <= 0.0)
        or np.any(inputs.intact_strength_mpa <= 0.0)
        or inputs.view_names != protocol.views
        or tuple(dict.fromkeys(inputs.dataset_ids)) != protocol.domain_order
    ):
        raise OOFPredictionError("authoritative multi-view inputs changed")

    predictions = np.full((len(targets), len(protocol.views)), np.nan, dtype=np.float64)
    selections: list[ViewSelection] = []
    source_records: list[SourceOOFPredictions] = []
    basis_cache: dict[tuple[int, tuple[int, ...]], PCABasis] = {}

    for outer_domain in protocol.domain_order:
        source = np.flatnonzero(domains != outer_domain).astype(np.int64)
        query = np.flatnonzero(domains == outer_domain).astype(np.int64)
        inner_domains = tuple(sorted(set(domains[source].tolist())))
        source_predictions = np.full((len(source), len(protocol.views)), np.nan)
        source_positions = {
            int(index): position for position, index in enumerate(source)
        }
        for view_index, view in enumerate(protocol.views):
            by_dimension = {
                dimension: np.full(len(targets), np.nan, dtype=np.float64)
                for dimension in protocol.pca_dimensions
            }
            for inner_domain in inner_domains:
                inner_fit = source[domains[source] != inner_domain]
                inner_query = source[domains[source] == inner_domain]
                cache_key = (view_index, tuple(int(item) for item in inner_fit))
                basis = basis_cache.get(cache_key)
                if basis is None:
                    basis = fit_pca_basis(
                        features[:, view_index],
                        inner_fit,
                        maximum_dimension=max(protocol.pca_dimensions),
                    )
                    basis_cache[cache_key] = basis
                for dimension in protocol.pca_dimensions:
                    _emit(
                        fit_hook,
                        inputs=inputs,
                        fit=inner_fit,
                        query=inner_query,
                        outer_domain=outer_domain,
                        inner_domain=inner_domain,
                        view=view,
                        dimension=dimension,
                    )
                    expert = fit_view_expert(
                        features[:, view_index],
                        metadata,
                        targets,
                        inner_fit,
                        pca_dimension=dimension,
                        alpha=protocol.ridge_alpha,
                        pca_basis=basis,
                    )
                    by_dimension[dimension][inner_query] = expert.predict(
                        features[inner_query, view_index], metadata[inner_query]
                    )
            scores: list[tuple[int, float]] = []
            selected_dimension: int | None = None
            selected_score: float | None = None
            for dimension in protocol.pca_dimensions:
                fold_scores = tuple(
                    float(
                        np.mean(
                            np.abs(
                                targets[source[domains[source] == inner_domain]]
                                - by_dimension[dimension][
                                    source[domains[source] == inner_domain]
                                ]
                            )
                        )
                    )
                    for inner_domain in inner_domains
                )
                score = float(np.mean(fold_scores))
                scores.append((dimension, score))
                if selected_score is None or score < selected_score - 1e-12:
                    selected_dimension = dimension
                    selected_score = score
            if selected_dimension is None:
                raise OOFPredictionError("PCA selection did not produce a dimension")
            for index in source:
                source_predictions[source_positions[int(index)], view_index] = (
                    by_dimension[selected_dimension][int(index)]
                )
            outer_cache_key = (view_index, tuple(int(item) for item in source))
            outer_basis = basis_cache.get(outer_cache_key)
            if outer_basis is None:
                outer_basis = fit_pca_basis(
                    features[:, view_index],
                    source,
                    maximum_dimension=max(protocol.pca_dimensions),
                )
                basis_cache[outer_cache_key] = outer_basis
            _emit(
                fit_hook,
                inputs=inputs,
                fit=source,
                query=query,
                outer_domain=outer_domain,
                inner_domain=None,
                view=view,
                dimension=selected_dimension,
            )
            expert = fit_view_expert(
                features[:, view_index],
                metadata,
                targets,
                source,
                pca_dimension=selected_dimension,
                alpha=protocol.ridge_alpha,
                pca_basis=outer_basis,
            )
            predictions[query, view_index] = expert.predict(
                features[query, view_index], metadata[query]
            )
            selections.append(
                ViewSelection(
                    outer_domain=outer_domain,
                    view=view,
                    pca_dimension=selected_dimension,
                    dimension_scores=tuple(scores),
                )
            )
        if not np.all(np.isfinite(source_predictions)):
            raise OOFPredictionError("source-domain OOF predictions are incomplete")
        source_records.append(
            SourceOOFPredictions(
                outer_domain=outer_domain,
                source_indices=tuple(int(item) for item in source),
                specimen_ids=tuple(inputs.specimen_ids[int(item)] for item in source),
                dataset_ids=tuple(inputs.dataset_ids[int(item)] for item in source),
                predictions=_readonly(source_predictions),
            )
        )
    if not np.all(np.isfinite(predictions)):
        raise OOFPredictionError("outer OOF predictions are incomplete")
    immutable_targets = _readonly(targets)
    immutable_predictions = _readonly(predictions)
    digest = hashlib.sha256()
    digest.update(inputs.feature_state_sha256.encode("ascii"))
    digest.update(immutable_targets.tobytes(order="C"))
    digest.update(np.asarray(inputs.cai_strength_mpa).tobytes(order="C"))
    digest.update(np.asarray(inputs.intact_strength_mpa).tobytes(order="C"))
    digest.update(immutable_predictions.tobytes(order="C"))
    for selection in selections:
        digest.update(repr(selection).encode("utf-8"))
    return IndependentOOFResult(
        specimen_ids=inputs.specimen_ids,
        dataset_ids=inputs.dataset_ids,
        view_names=inputs.view_names,
        targets=immutable_targets,
        cai_strength_mpa=_readonly(inputs.cai_strength_mpa),
        intact_strength_mpa=_readonly(inputs.intact_strength_mpa),
        predictions=immutable_predictions,
        selections=tuple(selections),
        source_oof=tuple(source_records),
        state_sha256=digest.hexdigest(),
    )
