"""Stage-A/B roster, aggregation, and promotion contracts for D8 residual models."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import optuna

from cmc_bbdm.cpb_v3.artifacts import (
    _validate_serialized_replay_against_data,
    _validate_tabular_replay_metadata,
)
from cmc_bbdm.cpb_v3.data import V3Data, validate_issued_data_authority
from cmc_bbdm.cpb_v3.morphology import REGISTERED_EXTRACTION_RULE

from .artifacts import validate_d8_search_package
from .authority import (
    D8InnerFold,
    D8SearchView,
    validate_inner_fold,
    validate_search_view,
)
from .config import DOMAIN_ORDER, load_d8_config
from .pilot import (
    D8FeatureBundle,
    RegisteredPilotAssets,
    _morphology_distance,
    evaluate_feature_bundle,
)
from .residual_config import ResidualDiffusionConfig
from .residual_model import (
    ResidualCheckpoint,
    _is_registered_runtime_device,
    _validate_checkpoint,
    sample_residual_targets,
)
from .residual_targets import (
    PilotDiffusionScaffold,
    ResidualFieldBank,
    ResidualTargetBatch,
    build_fit_residual_target_batch,
    build_outer_fit_residual_target_batch,
    load_pilot_diffusion_scaffolds,
    residual_replacement_perturbations,
    validate_residual_target_batch,
    validate_search_residual_field_bank,
)
from .residual_training import (
    ResidualFinalTrainingResult,
    ResidualTrainingResult,
    train_inner_residual_model,
    train_outer_fit_residual_model,
)
from .search import D8Candidate
from .selection import EnsembleResult, fit_nonnegative_ensemble
from .variants import build_variant_batch

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_PARAMETER_COUNTS = {False: 2_471_747, True: 10_128_515}


class ResidualSearchError(ValueError):
    """Raised when staged residual search evidence is incomplete or altered."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(repr(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value) <= _SHA256_CHARACTERS
    )


def _readonly_vector(value: object, *, label: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ResidualSearchError(f"{label} must be real")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ResidualSearchError(f"{label} must be numeric") from error
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ResidualSearchError(f"{label} must be a nonempty finite vector")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    output = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.float64
    ).reshape(contiguous.shape)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class ResidualSearchCell:
    """One exact outer/query/candidate/training-seed search cell."""

    stage: str
    outer_domain: str
    query_domain: str
    candidate_id: str
    training_seed: int
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.stage not in {"A", "B"}
            or self.outer_domain not in DOMAIN_ORDER
            or self.query_domain not in DOMAIN_ORDER
            or self.query_domain == self.outer_domain
            or self.candidate_id not in {f"RD{index}" for index in range(8)}
            or type(self.training_seed) is not int
            or self.training_seed not in {20260823, 20260824, 20260825}
        ):
            raise ResidualSearchError("residual search cell is not registered")
        object.__setattr__(
            self,
            "state_sha256",
            _canonical_sha256(
                {
                    "stage": self.stage,
                    "outer_domain": self.outer_domain,
                    "query_domain": self.query_domain,
                    "candidate_id": self.candidate_id,
                    "training_seed": self.training_seed,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ResidualCellEvaluation:
    """One query-domain prediction vector and morphology-gate record."""

    cell: ResidualSearchCell
    specimen_ids: tuple[str, ...]
    targets: np.ndarray
    predictions: np.ndarray
    accepted_proposals: int
    proposed_variants: int
    checkpoint_sha256: str
    prediction_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.cell) is not ResidualSearchCell:
            raise TypeError("exact ResidualSearchCell is required")
        targets = _readonly_vector(self.targets, label="cell targets")
        predictions = _readonly_vector(self.predictions, label="cell predictions")
        if (
            type(self.specimen_ids) is not tuple
            or len(self.specimen_ids) != len(targets)
            or len(set(self.specimen_ids)) != len(self.specimen_ids)
            or any(type(value) is not str or not value for value in self.specimen_ids)
            or predictions.shape != targets.shape
            or type(self.accepted_proposals) is not int
            or type(self.proposed_variants) is not int
            or self.proposed_variants < 1
            or not 0 <= self.accepted_proposals <= self.proposed_variants
            or not _valid_sha256(self.checkpoint_sha256)
            or not _valid_sha256(self.prediction_sha256)
        ):
            raise ResidualSearchError("residual cell evaluation is invalid")
        state = _canonical_sha256(
            {
                "cell_sha256": self.cell.state_sha256,
                "specimen_ids": self.specimen_ids,
                "target_sha256": _array_sha256(targets),
                "prediction_array_sha256": _array_sha256(predictions),
                "accepted_proposals": self.accepted_proposals,
                "proposed_variants": self.proposed_variants,
                "checkpoint_sha256": self.checkpoint_sha256,
                "prediction_sha256": self.prediction_sha256,
            }
        )
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "state_sha256", state)


@dataclass(frozen=True, slots=True)
class ResidualFeatureBundle:
    """Residual-generator evidence wrapped around one Pilot feature bundle."""

    cell: ResidualSearchCell
    pilot_candidate_sha256: str
    checkpoint_sha256: str
    sampled_target_sha256: str
    scaffold_sha256: str
    field_bank_sha256: str
    target_state_sha256: str
    asset_state_sha256: str
    feature_bundle: D8FeatureBundle
    variant_state_sha256: tuple[str, ...]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.cell) is not ResidualSearchCell
            or type(self.feature_bundle) is not D8FeatureBundle
            or any(
                not _valid_sha256(value)
                for value in (
                    self.pilot_candidate_sha256,
                    self.checkpoint_sha256,
                    self.sampled_target_sha256,
                    self.scaffold_sha256,
                    self.field_bank_sha256,
                    self.target_state_sha256,
                    self.asset_state_sha256,
                )
            )
            or self.feature_bundle.candidate_sha256
            != self.pilot_candidate_sha256
            or type(self.variant_state_sha256) is not tuple
            or len(self.variant_state_sha256) != len(self.feature_bundle.specimen_ids)
            or any(not _valid_sha256(value) for value in self.variant_state_sha256)
        ):
            raise ResidualSearchError("residual feature bundle is invalid")
        payload = {
            "cell_sha256": self.cell.state_sha256,
            "pilot_candidate_sha256": self.pilot_candidate_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "sampled_target_sha256": self.sampled_target_sha256,
            "scaffold_sha256": self.scaffold_sha256,
            "field_bank_sha256": self.field_bank_sha256,
            "target_state_sha256": self.target_state_sha256,
            "asset_state_sha256": self.asset_state_sha256,
            "feature_bundle_sha256": self.feature_bundle.state_sha256,
            "variant_state_sha256": self.variant_state_sha256,
        }
        object.__setattr__(self, "state_sha256", _canonical_sha256(payload))


@dataclass(frozen=True, slots=True)
class ResidualCellRun:
    """One generator fit followed by frozen sampling and query scoring."""

    cell: ResidualSearchCell
    training: ResidualTrainingResult
    feature_bundle_sha256: str
    evaluation: ResidualCellEvaluation
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.cell) is not ResidualSearchCell
            or type(self.training) is not ResidualTrainingResult
            or type(self.evaluation) is not ResidualCellEvaluation
            or not _valid_sha256(self.feature_bundle_sha256)
            or self.training.outer_domain != self.cell.outer_domain
            or self.training.query_domain != self.cell.query_domain
            or self.training.candidate_id != self.cell.candidate_id
            or self.training.seed != self.cell.training_seed
            or self.evaluation.cell.state_sha256 != self.cell.state_sha256
            or self.training.checkpoint.scientific_digest
            != self.evaluation.checkpoint_sha256
        ):
            raise ResidualSearchError("residual cell run state is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _canonical_sha256(
                {
                    "cell_sha256": self.cell.state_sha256,
                    "training_sha256": self.training.state_sha256,
                    "feature_bundle_sha256": self.feature_bundle_sha256,
                    "evaluation_sha256": self.evaluation.state_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ResidualCandidateSummary:
    """One candidate's specimen-first, domain-level staged objective."""

    stage: str
    outer_domain: str
    candidate_id: str
    training_seeds: tuple[int, ...]
    domain_mae: tuple[tuple[str, float], ...]
    mean_mae: float
    worst_mae: float
    domain_sd: float
    objective: float
    overall_acceptance: float
    domain_acceptance: tuple[tuple[str, float], ...]
    eligible: bool
    failed_domains: tuple[str, ...]
    oof_specimen_ids: tuple[str, ...]
    oof_domain_ids: tuple[str, ...]
    oof_targets: np.ndarray
    oof_predictions: np.ndarray
    parameter_count: int
    cell_state_sha256: tuple[str, ...]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        targets = _readonly_vector(self.oof_targets, label="summary targets")
        predictions = _readonly_vector(
            self.oof_predictions, label="summary predictions"
        )
        inner_domains = tuple(
            domain for domain in DOMAIN_ORDER if domain != self.outer_domain
        )
        if (
            self.stage not in {"A", "B"}
            or self.outer_domain not in DOMAIN_ORDER
            or self.candidate_id not in {f"RD{index}" for index in range(8)}
            or tuple(domain for domain, _value in self.domain_mae) != inner_domains
            or tuple(domain for domain, _value in self.domain_acceptance)
            != inner_domains
            or len(self.oof_specimen_ids) != len(targets)
            or len(self.oof_domain_ids) != len(targets)
            or predictions.shape != targets.shape
            or len(set(self.oof_specimen_ids)) != len(self.oof_specimen_ids)
            or any(not _valid_sha256(value) for value in self.cell_state_sha256)
            or type(self.eligible) is not bool
            or self.parameter_count not in _PARAMETER_COUNTS.values()
        ):
            raise ResidualSearchError("candidate summary structure is invalid")
        payload = {
            "stage": self.stage,
            "outer_domain": self.outer_domain,
            "candidate_id": self.candidate_id,
            "training_seeds": self.training_seeds,
            "domain_mae": self.domain_mae,
            "mean_mae": self.mean_mae,
            "worst_mae": self.worst_mae,
            "domain_sd": self.domain_sd,
            "objective": self.objective,
            "overall_acceptance": self.overall_acceptance,
            "domain_acceptance": self.domain_acceptance,
            "eligible": self.eligible,
            "failed_domains": self.failed_domains,
            "oof_specimen_ids": self.oof_specimen_ids,
            "oof_domain_ids": self.oof_domain_ids,
            "target_sha256": _array_sha256(targets),
            "prediction_sha256": _array_sha256(predictions),
            "parameter_count": self.parameter_count,
            "cell_state_sha256": self.cell_state_sha256,
        }
        object.__setattr__(self, "oof_targets", targets)
        object.__setattr__(self, "oof_predictions", predictions)
        object.__setattr__(self, "state_sha256", _canonical_sha256(payload))

    @property
    def rank_key(self) -> tuple[float, float, float, int, str]:
        return (
            self.objective,
            self.mean_mae,
            self.worst_mae,
            self.parameter_count,
            self.candidate_id,
        )


@dataclass(frozen=True, slots=True)
class ResidualIncumbentEvidence:
    """One frozen Pilot or raw-B0 inner-OOF comparison vector."""

    pipeline_id: str
    outer_domain: str
    specimen_ids: tuple[str, ...]
    domain_ids: tuple[str, ...]
    targets: np.ndarray
    predictions: np.ndarray
    evidence_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        targets = _readonly_vector(self.targets, label="incumbent targets")
        predictions = _readonly_vector(
            self.predictions, label="incumbent predictions"
        )
        inner_domains = tuple(
            domain for domain in DOMAIN_ORDER if domain != self.outer_domain
        )
        if (
            self.pipeline_id not in {"PILOT", "B0"}
            or self.outer_domain not in DOMAIN_ORDER
            or type(self.specimen_ids) is not tuple
            or type(self.domain_ids) is not tuple
            or len(self.specimen_ids) != len(targets)
            or len(self.domain_ids) != len(targets)
            or len(set(self.specimen_ids)) != len(self.specimen_ids)
            or predictions.shape != targets.shape
            or any(type(value) is not str or not value for value in self.specimen_ids)
            or any(type(value) is not str or not value for value in self.domain_ids)
            or tuple(dict.fromkeys(self.domain_ids)) != inner_domains
            or not _valid_sha256(self.evidence_sha256)
        ):
            raise ResidualSearchError("incumbent evidence is invalid")
        payload = {
            "pipeline_id": self.pipeline_id,
            "outer_domain": self.outer_domain,
            "specimen_ids": self.specimen_ids,
            "domain_ids": self.domain_ids,
            "target_sha256": _array_sha256(targets),
            "prediction_sha256": _array_sha256(predictions),
            "evidence_sha256": self.evidence_sha256,
        }
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "state_sha256", _canonical_sha256(payload))


@dataclass(frozen=True, slots=True)
class ResidualOuterSelection:
    """Frozen pre-outer comparison among residual, Pilot, B0, and ensemble."""

    outer_domain: str
    candidate_summaries: tuple[ResidualCandidateSummary, ...]
    incumbents: tuple[ResidualIncumbentEvidence, ...]
    best_residual: ResidualCandidateSummary | None
    best_incumbent: ResidualIncumbentEvidence
    best_incumbent_objective: float
    residual_improvement: float | None
    residual_promoted: bool
    ensemble: EnsembleResult | None
    ensemble_promoted: bool
    selected_pipeline: str
    selected_components: tuple[str, ...]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.outer_domain not in DOMAIN_ORDER
            or len(self.candidate_summaries) != 2
            or {value.outer_domain for value in self.candidate_summaries}
            != {self.outer_domain}
            or {value.stage for value in self.candidate_summaries} != {"B"}
            or tuple(value.pipeline_id for value in self.incumbents)
            not in {("PILOT", "B0"), ("B0", "PILOT")}
            or self.best_incumbent not in self.incumbents
            or self.selected_pipeline not in {"INCUMBENT", "RESIDUAL", "ENSEMBLE"}
            or type(self.residual_promoted) is not bool
            or type(self.ensemble_promoted) is not bool
            or not math.isfinite(self.best_incumbent_objective)
        ):
            raise ResidualSearchError("outer selection structure is invalid")
        if self.residual_promoted and self.best_residual is None:
            raise ResidualSearchError("residual promotion state is inconsistent")
        if (self.best_residual is None) != (self.residual_improvement is None):
            raise ResidualSearchError("residual comparison evidence is inconsistent")
        if self.residual_improvement is not None and not math.isfinite(
            self.residual_improvement
        ):
            raise ResidualSearchError("residual improvement is invalid")
        if self.ensemble_promoted != (self.ensemble is not None):
            raise ResidualSearchError("ensemble promotion state is inconsistent")
        expected_components: tuple[str, ...]
        if self.selected_pipeline == "INCUMBENT":
            expected_components = (self.best_incumbent.pipeline_id,)
        elif self.selected_pipeline == "RESIDUAL":
            assert self.best_residual is not None
            expected_components = (self.best_residual.candidate_id,)
        else:
            assert self.best_residual is not None and self.ensemble is not None
            expected_components = (
                self.best_residual.candidate_id,
                self.best_incumbent.pipeline_id,
            )
        if self.selected_components != expected_components:
            raise ResidualSearchError("selected component roster is inconsistent")
        payload = {
            "outer_domain": self.outer_domain,
            "candidate_states": [
                value.state_sha256 for value in self.candidate_summaries
            ],
            "incumbent_states": [value.state_sha256 for value in self.incumbents],
            "best_residual_state": (
                None if self.best_residual is None else self.best_residual.state_sha256
            ),
            "best_incumbent_state": self.best_incumbent.state_sha256,
            "best_incumbent_objective": self.best_incumbent_objective,
            "residual_improvement": self.residual_improvement,
            "residual_promoted": self.residual_promoted,
            "ensemble_state": (
                None if self.ensemble is None else self.ensemble.state_sha256
            ),
            "ensemble_promoted": self.ensemble_promoted,
            "selected_pipeline": self.selected_pipeline,
            "selected_components": self.selected_components,
        }
        object.__setattr__(self, "state_sha256", _canonical_sha256(payload))

    @property
    def requires_final_residual_checkpoints(self) -> bool:
        return self.selected_pipeline in {"RESIDUAL", "ENSEMBLE"}


@dataclass(frozen=True, slots=True)
class StageAOuterPromotion:
    """Exactly two eligible screening candidates for one prospective outer."""

    outer_domain: str
    finalists: tuple[str, str]
    summaries: tuple[ResidualCandidateSummary, ...]
    test_scale_override: bool = False
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.outer_domain not in DOMAIN_ORDER
            or type(self.finalists) is not tuple
            or len(self.finalists) != 2
            or len(set(self.finalists)) != 2
            or len(self.summaries) != 8
            or type(self.test_scale_override) is not bool
            or tuple(value.candidate_id for value in self.summaries)
            != tuple(f"RD{index}" for index in range(8))
            or any(
                value.stage != "A" or value.outer_domain != self.outer_domain
                for value in self.summaries
            )
        ):
            raise ResidualSearchError("Stage-A outer promotion roster is invalid")
        ranking_pool = (
            self.summaries
            if self.test_scale_override
            else tuple(summary for summary in self.summaries if summary.eligible)
        )
        finalists = tuple(
            value.candidate_id
            for value in sorted(
                ranking_pool,
                key=lambda value: value.rank_key,
            )[:2]
        )
        if len(finalists) != 2 or self.finalists != finalists:
            raise ResidualSearchError("Stage-A outer finalists changed")
        object.__setattr__(
            self,
            "state_sha256",
            _canonical_sha256(
                {
                    "outer_domain": self.outer_domain,
                    "finalists": self.finalists,
                    "test_scale_override": self.test_scale_override,
                    "summary_states": [
                        summary.state_sha256 for summary in self.summaries
                    ],
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ResidualOuterSearchRun:
    """One complete pre-outer Stage-A/B study with streamed checkpoints."""

    outer_domain: str
    stage_a: StageAOuterPromotion
    stage_a_run_sha256: tuple[str, ...]
    stage_b_run_sha256: tuple[str, ...]
    selection: ResidualOuterSelection
    final_training_sha256: tuple[str, ...]
    outer_evaluation_count: int
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        final_expected = 3 if self.selection.requires_final_residual_checkpoints else 0
        if (
            self.outer_domain not in DOMAIN_ORDER
            or type(self.stage_a) is not StageAOuterPromotion
            or self.stage_a.outer_domain != self.outer_domain
            or type(self.selection) is not ResidualOuterSelection
            or self.selection.outer_domain != self.outer_domain
            or tuple(
                value.candidate_id for value in self.selection.candidate_summaries
            )
            != self.stage_a.finalists
            or type(self.stage_a_run_sha256) is not tuple
            or len(self.stage_a_run_sha256) != 5 * 8
            or len(set(self.stage_a_run_sha256)) != len(self.stage_a_run_sha256)
            or type(self.stage_b_run_sha256) is not tuple
            or len(self.stage_b_run_sha256) != 5 * 2 * 3
            or len(set(self.stage_b_run_sha256)) != len(self.stage_b_run_sha256)
            or type(self.final_training_sha256) is not tuple
            or len(self.final_training_sha256) != final_expected
            or len(set(self.final_training_sha256)) != len(
                self.final_training_sha256
            )
            or any(
                not _valid_sha256(value)
                for value in (
                    *self.stage_a_run_sha256,
                    *self.stage_b_run_sha256,
                    *self.final_training_sha256,
                )
            )
            or self.outer_evaluation_count != 0
        ):
            raise ResidualSearchError("outer residual search run is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _canonical_sha256(
                {
                    "outer_domain": self.outer_domain,
                    "stage_a_sha256": self.stage_a.state_sha256,
                    "stage_a_run_sha256": self.stage_a_run_sha256,
                    "stage_b_run_sha256": self.stage_b_run_sha256,
                    "selection_sha256": self.selection.state_sha256,
                    "final_training_sha256": self.final_training_sha256,
                    "outer_evaluation_count": self.outer_evaluation_count,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class StageAPromotion:
    """Exactly two eligible Stage-A candidates for every prospective outer."""

    finalists: Mapping[str, tuple[str, str]]
    summaries: tuple[ResidualCandidateSummary, ...]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        values = dict(self.finalists)
        if (
            tuple(values) != DOMAIN_ORDER
            or any(
                type(candidate_ids) is not tuple
                or len(candidate_ids) != 2
                or len(set(candidate_ids)) != 2
                for candidate_ids in values.values()
            )
            or len(self.summaries) != len(DOMAIN_ORDER) * 8
        ):
            raise ResidualSearchError("Stage-A promotion roster is invalid")
        payload = {
            "finalists": values,
            "summary_states": [summary.state_sha256 for summary in self.summaries],
        }
        object.__setattr__(self, "finalists", MappingProxyType(values))
        object.__setattr__(self, "state_sha256", _canonical_sha256(payload))


def _config(value: object) -> ResidualDiffusionConfig:
    if type(value) is not ResidualDiffusionConfig:
        raise TypeError("exact ResidualDiffusionConfig is required")
    if value.outer_evaluation_count != 0:
        raise ResidualSearchError("outer evaluation count changed")
    return value


def stage_a_cell_keys(
    config: ResidualDiffusionConfig,
) -> tuple[ResidualSearchCell, ...]:
    """Return the exact 6 x 5 x 8 screening roster."""

    clean = _config(config)
    return tuple(
        ResidualSearchCell("A", outer, query, candidate_id, clean.screening_seed)
        for outer in DOMAIN_ORDER
        for query in DOMAIN_ORDER
        if query != outer
        for candidate_id in clean.candidate_ids
    )


def load_pilot_scaffold_candidates(
    config: ResidualDiffusionConfig,
    *,
    project_root: str | Path,
) -> Mapping[str, D8Candidate]:
    """Recover all frozen Pilot downstream candidates from validated trials."""

    clean = _config(config)
    root = Path(project_root).resolve(strict=True)
    scaffolds = load_pilot_diffusion_scaffolds(clean, project_root=root)
    storage = f"sqlite:///{(root / 'results/d8_search/study.db').resolve(strict=True)}"
    candidates: dict[str, D8Candidate] = {}
    for outer_domain in DOMAIN_ORDER:
        scaffold = scaffolds[outer_domain]
        study = optuna.load_study(
            study_name=f"d8::{outer_domain}",
            storage=storage,
        )
        matches = tuple(
            trial
            for trial in study.trials
            if trial.number == scaffold.trial_number
            and type(trial.user_attrs.get("candidate")) is dict
        )
        if len(matches) != 1:
            raise ResidualSearchError("Pilot scaffold trial is unavailable")
        candidate = D8Candidate.from_payload(matches[0].user_attrs["candidate"])
        parameters = dict(candidate.decomposition_parameters)
        parameters["band"] = candidate.band
        if (
            candidate.state_sha256 != scaffold.candidate_sha256
            or candidate.config_sha256 != scaffold.config_sha256
            or candidate.control_id != scaffold.control_id
            or candidate.decomposition_family != scaffold.decomposition_family
            or parameters != dict(scaffold.decomposition_parameters)
            or matches[0].value != scaffold.objective
        ):
            raise ResidualSearchError("Pilot scaffold candidate binding changed")
        candidates[outer_domain] = candidate
    return MappingProxyType(candidates)


def load_pilot_incumbent_evidence(
    config: ResidualDiffusionConfig,
    *,
    project_root: str | Path,
) -> tuple[ResidualIncumbentEvidence, ...]:
    """Load the six frozen Pilot OOF vectors after full package validation."""

    clean = _config(config)
    root = Path(project_root).resolve(strict=True)
    pilot_output = (root / clean.sources["pilot_manifest"].path).parent
    validated = validate_d8_search_package(
        pilot_output,
        project_root=root,
        config_path=root / clean.sources["exploration_config"].path,
    )
    if (
        validated.outer_evaluation_count != 0
        or validated.escalation_status != clean.pilot_decision
        or validated.scientific_digest != clean.pilot_scientific_digest
    ):
        raise ResidualSearchError("validated Pilot package differs from residual gate")
    selection_path = pilot_output / "selected_configs.json"
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResidualSearchError("Pilot selections are unavailable") from error
    expected_root = {
        "schema_version",
        "scope",
        "config_sha256",
        "outer_evaluation_count",
        "selections",
        "state_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_root
        or payload["schema_version"] != 2
        or payload["scope"] != "d8_prospective_outer_selections"
        or payload["config_sha256"] != clean.sources["exploration_config"].sha256
        or payload["outer_evaluation_count"] != 0
        or not isinstance(payload["selections"], list)
        or len(payload["selections"]) != len(DOMAIN_ORDER)
    ):
        raise ResidualSearchError("Pilot selection document changed")
    results: list[ResidualIncumbentEvidence] = []
    for outer_domain, selection in zip(
        DOMAIN_ORDER, payload["selections"], strict=True
    ):
        if (
            not isinstance(selection, dict)
            or selection.get("outer_domain") != outer_domain
            or selection.get("outer_evaluation_started") is not False
            or selection.get("config_sha256")
            != clean.sources["exploration_config"].sha256
            or not _valid_sha256(selection.get("state_sha256"))
        ):
            raise ResidualSearchError("Pilot outer selection changed")
        authority = selection.get("search_authority")
        ensemble = selection.get("ensemble")
        if (
            not isinstance(authority, dict)
            or set(authority)
            != {"domain_ids", "specimen_ids", "target_sha256", "targets"}
            or not isinstance(ensemble, dict)
            or not _valid_sha256(ensemble.get("state_sha256"))
        ):
            raise ResidualSearchError("Pilot OOF authority changed")
        specimen_ids = tuple(authority["specimen_ids"])
        domain_ids = tuple(authority["domain_ids"])
        targets = _readonly_vector(authority["targets"], label="Pilot targets")
        predictions = _readonly_vector(
            ensemble.get("predictions"), label="Pilot predictions"
        )
        evidence_sha256 = _canonical_sha256(
            {
                "artifact_manifest_sha256": validated.artifact_manifest_sha256,
                "selection_sha256": selection["state_sha256"],
                "ensemble_sha256": ensemble["state_sha256"],
                "search_view_sha256": selection.get("search_view_sha256"),
            }
        )
        results.append(
            ResidualIncumbentEvidence(
                pipeline_id="PILOT",
                outer_domain=outer_domain,
                specimen_ids=specimen_ids,
                domain_ids=domain_ids,
                targets=targets,
                predictions=predictions,
                evidence_sha256=evidence_sha256,
            )
        )
    return tuple(results)


def load_b0_incumbent_evidence(
    data: V3Data,
    *,
    config: ResidualDiffusionConfig,
    project_root: str | Path,
) -> tuple[ResidualIncumbentEvidence, ...]:
    """Rebuild the raw I_frozen incumbent from validated P1 inner replay."""

    if type(data) is not V3Data:
        raise TypeError("exact V3Data is required")
    validate_issued_data_authority(data)
    clean = _config(config)
    root = Path(project_root).resolve(strict=True)
    exploration = load_d8_config(
        root / clean.sources["exploration_config"].path,
        project_root=root,
    )
    manifest_source = exploration.sources["p1_manifest"]
    manifest_path = root / manifest_source.path
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResidualSearchError("P1 artifact manifest is unavailable") from error
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != manifest_source.sha256
        or not isinstance(manifest, dict)
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ResidualSearchError("P1 artifact manifest authority changed")
    package = manifest_path.parent

    def registered_bytes(name: str) -> bytes:
        record = manifest["files"].get(name)
        path = package / name
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ResidualSearchError(f"P1 {name} is unavailable") from error
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or record["bytes"] != len(payload)
            or record["sha256"] != hashlib.sha256(payload).hexdigest()
        ):
            raise ResidualSearchError(f"P1 {name} differs from its manifest")
        return payload

    try:
        summary = json.loads(registered_bytes("run_summary.json").decode("utf-8"))

        def csv_rows(name: str) -> tuple[dict[str, str], ...]:
            text = registered_bytes(name).decode("utf-8")
            reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
            if reader.fieldnames is None:
                raise ResidualSearchError(f"P1 {name} has no header")
            return tuple(dict(row) for row in reader)

        learned_rows = csv_rows("inner_predictions.csv")
        selection_rows = csv_rows("inner_selection.csv")
    except (UnicodeError, json.JSONDecodeError, csv.Error) as error:
        raise ResidualSearchError("P1 replay evidence cannot be decoded") from error
    try:
        typed_replay = summary["replay_authority"]["typed_replay"]
        tabular_replay = typed_replay["tabular_replay"]
        _validate_tabular_replay_metadata(tabular_replay)
        _validate_serialized_replay_against_data(
            typed_replay,
            data,
            learned_rows,
            selection_rows,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ResidualSearchError("P1 typed replay authority changed") from error
    records = {
        (
            record["identity"]["outer_dataset_id"],
            record["identity"]["query_dataset_id"],
        ): record
        for record in tabular_replay["records"]
        if record["identity"]["candidate"] == "I_frozen"
    }
    if len(records) != len(DOMAIN_ORDER) * (len(DOMAIN_ORDER) - 1):
        raise ResidualSearchError("P1 I_frozen replay roster is incomplete")
    results: list[ResidualIncumbentEvidence] = []
    sample_ids = tuple(str(value) for value in data.sample_ids.tolist())
    dataset_ids = tuple(str(value) for value in data.dataset_ids.tolist())
    for outer_domain in DOMAIN_ORDER:
        inner_domains = tuple(
            value for value in DOMAIN_ORDER if value != outer_domain
        )
        ordered = tuple(records[(outer_domain, query)] for query in inner_domains)
        specimen_ids = tuple(
            value
            for record in ordered
            for value in record["state"]["query_ids"]
        )
        domains = tuple(
            value
            for record in ordered
            for value in record["state"]["query_domains"]
        )
        expected_ids = tuple(
            specimen
            for specimen, domain in zip(sample_ids, dataset_ids, strict=True)
            if domain != outer_domain
        )
        expected_domains = tuple(
            domain for domain in dataset_ids if domain != outer_domain
        )
        if specimen_ids != expected_ids or domains != expected_domains:
            raise ResidualSearchError("P1 I_frozen replay order changed")
        targets = tuple(
            value for record in ordered for value in record["state"]["targets"]
        )
        predictions = tuple(
            value
            for record in ordered
            for value in record["state"]["predictions"]
        )
        evidence_sha256 = _canonical_sha256(
            {
                "p1_manifest_sha256": manifest_source.sha256,
                "tabular_replay_sha256": tabular_replay["state_sha256"],
                "record_sha256": [
                    record["state"]["state_sha256"] for record in ordered
                ],
            }
        )
        results.append(
            ResidualIncumbentEvidence(
                pipeline_id="B0",
                outer_domain=outer_domain,
                specimen_ids=specimen_ids,
                domain_ids=domains,
                targets=np.asarray(targets, dtype=np.float64),
                predictions=np.asarray(predictions, dtype=np.float64),
                evidence_sha256=evidence_sha256,
            )
        )
    return tuple(results)


def _validated_finalists(
    finalists: object,
    *,
    config: ResidualDiffusionConfig,
) -> dict[str, tuple[str, str]]:
    if not isinstance(finalists, Mapping):
        raise TypeError("Stage-B finalists must be a mapping")
    values = dict(finalists)
    if tuple(values) != DOMAIN_ORDER:
        raise ResidualSearchError("Stage-B outer roster changed")
    for candidate_ids in values.values():
        if (
            type(candidate_ids) is not tuple
            or len(candidate_ids) != config.finalists_per_outer
            or len(set(candidate_ids)) != len(candidate_ids)
            or any(value not in config.candidate_ids for value in candidate_ids)
        ):
            raise ResidualSearchError("Stage-B finalist roster changed")
    return values


def stage_b_cell_keys(
    config: ResidualDiffusionConfig,
    *,
    finalists: Mapping[str, tuple[str, str]],
) -> tuple[ResidualSearchCell, ...]:
    """Return the exact 6 x 5 x 2 x 3 reranking roster."""

    clean = _config(config)
    values = _validated_finalists(finalists, config=clean)
    return tuple(
        ResidualSearchCell("B", outer, query, candidate_id, seed)
        for outer in DOMAIN_ORDER
        for query in DOMAIN_ORDER
        if query != outer
        for candidate_id in values[outer]
        for seed in clean.training_seeds
    )


def _seed_mean(values: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.asarray(
        [
            math.fsum(float(value[index]) for value in values) / len(values)
            for index in range(len(values[0]))
        ],
        dtype=np.float64,
    )


def summarize_candidate_cells(
    evaluations: tuple[ResidualCellEvaluation, ...],
    *,
    config: ResidualDiffusionConfig,
    stage: str,
) -> ResidualCandidateSummary:
    """Average seed predictions per specimen before computing domain metrics."""

    clean = _config(config)
    if (
        stage not in {"A", "B"}
        or type(evaluations) is not tuple
        or not evaluations
        or any(type(value) is not ResidualCellEvaluation for value in evaluations)
    ):
        raise ResidualSearchError("candidate cell evidence is invalid")
    outer = evaluations[0].cell.outer_domain
    candidate_id = evaluations[0].cell.candidate_id
    seeds = (
        (clean.screening_seed,) if stage == "A" else clean.training_seeds
    )
    inner_domains = tuple(domain for domain in DOMAIN_ORDER if domain != outer)
    expected = {
        (domain, seed)
        for domain in inner_domains
        for seed in seeds
    }
    observed = {
        (value.cell.query_domain, value.cell.training_seed) for value in evaluations
    }
    if (
        len(evaluations) != len(expected)
        or observed != expected
        or len(observed) != len(evaluations)
        or any(
            value.cell.stage != stage
            or value.cell.outer_domain != outer
            or value.cell.candidate_id != candidate_id
            for value in evaluations
        )
    ):
        raise ResidualSearchError("candidate cell roster is incomplete")
    by_key = {
        (value.cell.query_domain, value.cell.training_seed): value
        for value in evaluations
    }
    domain_mae: list[tuple[str, float]] = []
    domain_acceptance: list[tuple[str, float]] = []
    failed_domains: list[str] = []
    oof_ids: list[str] = []
    oof_domains: list[str] = []
    oof_targets: list[np.ndarray] = []
    oof_predictions: list[np.ndarray] = []
    accepted_total = 0
    proposed_total = 0
    for domain in inner_domains:
        cells = tuple(by_key[(domain, seed)] for seed in seeds)
        reference = cells[0]
        if any(
            value.specimen_ids != reference.specimen_ids
            or not np.array_equal(value.targets, reference.targets)
            for value in cells[1:]
        ):
            raise ResidualSearchError("seed cells do not share query identities")
        predictions = _seed_mean(tuple(value.predictions for value in cells))
        mae = math.fsum(
            abs(float(prediction) - float(target))
            for prediction, target in zip(
                predictions, reference.targets, strict=True
            )
        ) / len(predictions)
        accepted = sum(value.accepted_proposals for value in cells)
        proposed = sum(value.proposed_variants for value in cells)
        rate = accepted / proposed
        domain_mae.append((domain, mae))
        domain_acceptance.append((domain, rate))
        if rate < clean.minimum_domain_acceptance:
            failed_domains.append(domain)
        accepted_total += accepted
        proposed_total += proposed
        oof_ids.extend(reference.specimen_ids)
        oof_domains.extend((domain,) * len(reference.specimen_ids))
        oof_targets.append(reference.targets)
        oof_predictions.append(predictions)
    values = np.asarray([value for _domain, value in domain_mae], dtype=np.float64)
    mean_mae = math.fsum(float(value) for value in values) / len(values)
    worst_mae = float(np.max(values))
    domain_sd = float(np.std(values, dtype=np.float64))
    weights = clean.objective_weights
    objective = (
        weights[0] * mean_mae
        + weights[1] * worst_mae
        + weights[2] * domain_sd
    )
    overall = accepted_total / proposed_total
    candidate = clean.candidate(candidate_id)
    return ResidualCandidateSummary(
        stage=stage,
        outer_domain=outer,
        candidate_id=candidate_id,
        training_seeds=seeds,
        domain_mae=tuple(domain_mae),
        mean_mae=mean_mae,
        worst_mae=worst_mae,
        domain_sd=domain_sd,
        objective=objective,
        overall_acceptance=overall,
        domain_acceptance=tuple(domain_acceptance),
        eligible=(
            overall >= clean.minimum_overall_acceptance and not failed_domains
        ),
        failed_domains=tuple(failed_domains),
        oof_specimen_ids=tuple(oof_ids),
        oof_domain_ids=tuple(oof_domains),
        oof_targets=np.concatenate(oof_targets),
        oof_predictions=np.concatenate(oof_predictions),
        parameter_count=_PARAMETER_COUNTS[candidate.bottleneck_attention],
        cell_state_sha256=tuple(value.state_sha256 for value in evaluations),
    )


def build_residual_feature_bundle(
    cell: ResidualSearchCell,
    *,
    fold: D8InnerFold,
    config: ResidualDiffusionConfig,
    checkpoint: ResidualCheckpoint,
    scaffold: PilotDiffusionScaffold,
    pilot_candidate: D8Candidate,
    field_bank: ResidualFieldBank,
    target_batch: ResidualTargetBatch,
    assets: RegisteredPilotAssets,
    encoder: object,
    device: str,
) -> ResidualFeatureBundle:
    """Sample a frozen generator, rerun native gates, and encode one cell."""

    clean = _config(config)
    if (
        type(cell) is not ResidualSearchCell
        or type(fold) is not D8InnerFold
        or type(scaffold) is not PilotDiffusionScaffold
        or type(pilot_candidate) is not D8Candidate
        or type(field_bank) is not ResidualFieldBank
        or type(target_batch) is not ResidualTargetBatch
        or type(assets) is not RegisteredPilotAssets
        or not hasattr(encoder, "encode")
        or not callable(encoder.encode)
        or not _is_registered_runtime_device(device, allow_cpu=False)
    ):
        raise TypeError("exact residual feature authorities are required")
    split_state = validate_inner_fold(fold)
    field_state = validate_search_residual_field_bank(
        field_bank,
        search_view=fold.search_view,
    )
    target_state = validate_residual_target_batch(target_batch)
    frozen = _validate_checkpoint(checkpoint)
    parameters = dict(pilot_candidate.decomposition_parameters)
    parameters["band"] = pilot_candidate.band
    if (
        cell.outer_domain != fold.outer_domain
        or cell.query_domain != fold.query_domain
        or cell.candidate_id != frozen.candidate_id
        or cell.training_seed != frozen.training_seed
        or frozen.config_sha256 != clean.config_sha256
        or frozen.split_sha256 != split_state
        or scaffold.outer_domain != fold.outer_domain
        or scaffold.candidate_sha256 != pilot_candidate.state_sha256
        or scaffold.config_sha256 != pilot_candidate.config_sha256
        or scaffold.control_id != pilot_candidate.control_id
        or scaffold.decomposition_family != pilot_candidate.decomposition_family
        or dict(scaffold.decomposition_parameters) != parameters
        or target_batch.role != "outer_fit"
        or target_batch.outer_domain != fold.outer_domain
        or target_batch.authority_sha256 != fold.search_view.state_sha256
        or target_batch.scaffold_sha256 != scaffold.state_sha256
        or target_batch.specimen_ids != field_bank.specimen_ids
        or target_batch.dataset_ids != field_bank.dataset_ids
        or target_batch.source_sha256 != field_bank.source_sha256
        or not np.array_equal(target_batch.measured, field_bank.measured)
    ):
        raise ResidualSearchError("residual feature authority changed")
    targets = target_batch
    sampled = sample_residual_targets(
        frozen,
        targets.stable_condition,
        specimen_ids=targets.specimen_ids,
        draws=32,
        steps=clean.sample_steps,
        eta=clean.sample_eta,
        device=device,
    )
    sampled_array = np.asarray(sampled, dtype=np.float32)
    if (
        sampled_array.shape != (len(targets.specimen_ids), 32, 3, 64, 64)
        or not np.all(np.isfinite(sampled_array))
        or float(np.min(sampled_array)) < -1.0
        or float(np.max(sampled_array)) > 1.0
    ):
        raise ResidualSearchError("sampled residual target roster is invalid")
    positions = {value: index for index, value in enumerate(assets.specimen_ids)}
    if set(targets.specimen_ids) - set(positions):
        raise ResidualSearchError("registered assets do not cover the search view")
    maximum_k = max(pilot_candidate.K_train, pilot_candidate.K_test)

    def build_row(
        index: int,
    ) -> tuple[tuple[np.ndarray, ...], tuple[float, ...], int, int, str]:
        specimen_id = targets.specimen_ids[index]
        dataset_id = targets.dataset_ids[index]
        position = positions[specimen_id]
        if (
            assets.dataset_ids[position] != dataset_id
            or assets.source_sha256[position]
            != field_bank.native_source_sha256[index]
            or not np.array_equal(
                assets.measured_fields[position], targets.measured[index]
            )
        ):
            raise ResidualSearchError("registered asset row changed")
        perturbations = residual_replacement_perturbations(
            sampled_array[index],
            observed_residual=targets.residual[index],
        )
        batch = build_variant_batch(
            targets.measured[index],
            tuple(perturbations[draw] for draw in range(32)),
            native_source=assets.native_images[position],
            alpha=pilot_candidate.alpha,
            requested_count=maximum_k,
            rule=REGISTERED_EXTRACTION_RULE,
            calibration=assets.calibrations[dataset_id],
            thresholds=pilot_candidate.thresholds,
        )
        accepted_records = tuple(record for record in batch.records if record.accepted)
        distances = tuple(
            _morphology_distance(record, pilot_candidate)
            for record in accepted_records[:maximum_k]
        ) + tuple(0.0 for _ in range(batch.fallback_count))
        if len(distances) != maximum_k or len(batch.encoder_images) != maximum_k:
            raise ResidualSearchError("residual variant K roster changed")
        return (
            batch.encoder_images,
            distances,
            batch.accepted_count,
            batch.proposal_count,
            batch.state_sha256,
        )

    with ThreadPoolExecutor(
        max_workers=8,
        thread_name_prefix="d8-residual-variant",
    ) as workers:
        rows = tuple(workers.map(build_row, range(len(targets.specimen_ids))))
    image_grid = tuple(value[0] for value in rows)
    encoded = np.asarray(
        encoder.encode(image_grid, layer=pilot_candidate.feature_layer),
        dtype=np.float64,
    )
    if (
        encoded.ndim != 3
        or encoded.shape[:2] != (len(targets.specimen_ids), maximum_k)
    ):
        raise ResidualSearchError("residual encoder output is misaligned")
    feature_bundle = D8FeatureBundle(
        candidate_sha256=pilot_candidate.state_sha256,
        search_view_sha256=fold.search_view.state_sha256,
        specimen_ids=fold.search_view.specimen_ids,
        train_variant_features=encoded[:, : pilot_candidate.K_train],
        query_variant_features=encoded[:, : pilot_candidate.K_test],
        morphology_distances=np.asarray(
            tuple(value[1] for value in rows), dtype=np.float64
        )[:, : pilot_candidate.K_test],
        accepted_proposals=np.asarray(
            tuple(value[2] for value in rows), dtype=np.int64
        ),
        proposed_variants=np.asarray(
            tuple(value[3] for value in rows), dtype=np.int64
        ),
    )
    return ResidualFeatureBundle(
        cell=cell,
        pilot_candidate_sha256=pilot_candidate.state_sha256,
        checkpoint_sha256=frozen.scientific_digest,
        sampled_target_sha256=_array_sha256(sampled_array),
        scaffold_sha256=scaffold.state_sha256,
        field_bank_sha256=field_state,
        target_state_sha256=target_state,
        asset_state_sha256=assets.state_sha256,
        feature_bundle=feature_bundle,
        variant_state_sha256=tuple(value[4] for value in rows),
    )


def evaluate_residual_feature_bundle(
    cell: ResidualSearchCell,
    *,
    fold: D8InnerFold,
    pilot_candidate: D8Candidate,
    bundle: ResidualFeatureBundle,
) -> ResidualCellEvaluation:
    """Fit the frozen Pilot downstream model and score one residual cell."""

    if (
        type(cell) is not ResidualSearchCell
        or type(fold) is not D8InnerFold
        or type(pilot_candidate) is not D8Candidate
        or type(bundle) is not ResidualFeatureBundle
    ):
        raise TypeError("exact residual evaluation authorities are required")
    validate_inner_fold(fold)
    if (
        bundle.cell.state_sha256 != cell.state_sha256
        or cell.outer_domain != fold.outer_domain
        or cell.query_domain != fold.query_domain
        or bundle.pilot_candidate_sha256 != pilot_candidate.state_sha256
        or bundle.feature_bundle.search_view_sha256
        != fold.search_view.state_sha256
        or bundle.feature_bundle.specimen_ids != fold.search_view.specimen_ids
    ):
        raise ResidualSearchError("residual feature authority changed")
    evaluated = evaluate_feature_bundle(
        pilot_candidate,
        fold=fold,
        bundle=bundle.feature_bundle,
    )
    prediction_sha256 = _canonical_sha256(
        {
            "cell_sha256": cell.state_sha256,
            "bundle_sha256": bundle.state_sha256,
            "prediction_sha256": evaluated.prediction.state_sha256,
        }
    )
    return ResidualCellEvaluation(
        cell=cell,
        specimen_ids=evaluated.prediction.query_specimen_ids,
        targets=evaluated.prediction.targets,
        predictions=evaluated.prediction.predictions,
        accepted_proposals=evaluated.inner.accepted_proposals,
        proposed_variants=evaluated.inner.proposed_variants,
        checkpoint_sha256=bundle.checkpoint_sha256,
        prediction_sha256=prediction_sha256,
    )


def run_residual_search_cell(
    cell: ResidualSearchCell,
    *,
    fold: D8InnerFold,
    config: ResidualDiffusionConfig,
    fit_target_batch: ResidualTargetBatch,
    outer_target_batch: ResidualTargetBatch,
    scaffold: PilotDiffusionScaffold,
    pilot_candidate: D8Candidate,
    field_bank: ResidualFieldBank,
    assets: RegisteredPilotAssets,
    encoder: object,
    device: str,
    test_scale_override: bool = False,
) -> ResidualCellRun:
    """Execute one fit -> frozen sample -> inner-query score cell."""

    clean = _config(config)
    if (
        type(cell) is not ResidualSearchCell
        or type(fold) is not D8InnerFold
        or cell.outer_domain != fold.outer_domain
        or cell.query_domain != fold.query_domain
        or type(test_scale_override) is not bool
    ):
        raise ResidualSearchError("residual cell authority changed")
    validate_inner_fold(fold)
    epochs = (
        1
        if test_scale_override
        else (
            clean.screening_epochs
            if cell.stage == "A"
            else clean.rerank_epochs
        )
    )
    training = train_inner_residual_model(
        fold,
        fit_target_batch,
        config=clean,
        candidate=clean.candidate(cell.candidate_id),
        epochs=epochs,
        seed=cell.training_seed,
        device=device,
        test_scale_override=test_scale_override,
    )
    bundle = build_residual_feature_bundle(
        cell,
        fold=fold,
        config=clean,
        checkpoint=training.checkpoint,
        scaffold=scaffold,
        pilot_candidate=pilot_candidate,
        field_bank=field_bank,
        target_batch=outer_target_batch,
        assets=assets,
        encoder=encoder,
        device=device,
    )
    evaluation = evaluate_residual_feature_bundle(
        cell,
        fold=fold,
        pilot_candidate=pilot_candidate,
        bundle=bundle,
    )
    return ResidualCellRun(
        cell=cell,
        training=training,
        feature_bundle_sha256=bundle.state_sha256,
        evaluation=evaluation,
    )


def promote_stage_a_outer(
    evaluations: tuple[ResidualCellEvaluation, ...],
    *,
    config: ResidualDiffusionConfig,
    test_scale_override: bool = False,
) -> StageAOuterPromotion:
    """Rank the exact 5 x 8 screening cells for one prospective outer."""

    clean = _config(config)
    if (
        type(evaluations) is not tuple
        or not evaluations
        or any(type(value) is not ResidualCellEvaluation for value in evaluations)
        or type(test_scale_override) is not bool
    ):
        raise ResidualSearchError("Stage-A outer evidence must be an exact tuple")
    outer = evaluations[0].cell.outer_domain
    expected_states = {
        cell.state_sha256
        for cell in stage_a_cell_keys(clean)
        if cell.outer_domain == outer
    }
    observed_states = {value.cell.state_sha256 for value in evaluations}
    if (
        len(evaluations) != len(expected_states)
        or len(observed_states) != len(evaluations)
        or observed_states != expected_states
    ):
        raise ResidualSearchError("Stage-A outer Cartesian evidence is incomplete")
    summaries = tuple(
        summarize_candidate_cells(
            tuple(
                value
                for value in evaluations
                if value.cell.candidate_id == candidate_id
            ),
            config=clean,
            stage="A",
        )
        for candidate_id in clean.candidate_ids
    )
    ranking_pool = sorted(
        (
            summaries
            if test_scale_override
            else tuple(summary for summary in summaries if summary.eligible)
        ),
        key=lambda value: value.rank_key,
    )
    if len(ranking_pool) < clean.finalists_per_outer:
        raise ResidualSearchError("fewer than two Stage-A candidates are eligible")
    finalists = tuple(
        summary.candidate_id
        for summary in ranking_pool[: clean.finalists_per_outer]
    )
    return StageAOuterPromotion(
        outer_domain=outer,
        finalists=finalists,
        summaries=summaries,
        test_scale_override=test_scale_override,
    )


def promote_stage_a(
    evaluations: tuple[ResidualCellEvaluation, ...],
    *,
    config: ResidualDiffusionConfig,
) -> StageAPromotion:
    """Rank all screening cells and promote exactly two eligible candidates."""

    clean = _config(config)
    if type(evaluations) is not tuple or any(
        type(value) is not ResidualCellEvaluation for value in evaluations
    ):
        raise ResidualSearchError("Stage-A evidence must be an exact tuple")
    expected_states = {cell.state_sha256 for cell in stage_a_cell_keys(clean)}
    observed_states = {value.cell.state_sha256 for value in evaluations}
    if (
        len(evaluations) != len(expected_states)
        or len(observed_states) != len(evaluations)
        or observed_states != expected_states
    ):
        raise ResidualSearchError("Stage-A Cartesian evidence is incomplete")
    summaries: list[ResidualCandidateSummary] = []
    finalists: dict[str, tuple[str, str]] = {}
    for outer in DOMAIN_ORDER:
        promoted = promote_stage_a_outer(
            tuple(
                value
                for value in evaluations
                if value.cell.outer_domain == outer
            ),
            config=clean,
        )
        summaries.extend(promoted.summaries)
        finalists[outer] = promoted.finalists
    return StageAPromotion(
        finalists=finalists,
        summaries=tuple(summaries),
    )


def _prediction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    domain_ids: tuple[str, ...],
    *,
    outer_domain: str,
    config: ResidualDiffusionConfig,
) -> tuple[tuple[tuple[str, float], ...], float, float, float, float]:
    inner_domains = tuple(domain for domain in DOMAIN_ORDER if domain != outer_domain)
    domains = np.asarray(domain_ids)
    if tuple(dict.fromkeys(domain_ids)) != inner_domains:
        raise ResidualSearchError("comparison domain order changed")
    domain_mae = tuple(
        (
            domain,
            math.fsum(
                abs(float(prediction) - float(target))
                for prediction, target in zip(
                    predictions[domains == domain],
                    targets[domains == domain],
                    strict=True,
                )
            )
            / int(np.sum(domains == domain)),
        )
        for domain in inner_domains
    )
    values = np.asarray([value for _domain, value in domain_mae], dtype=np.float64)
    mean_mae = math.fsum(float(value) for value in values) / len(values)
    worst_mae = float(np.max(values))
    domain_sd = float(np.std(values, dtype=np.float64))
    weights = config.objective_weights
    objective = (
        weights[0] * mean_mae
        + weights[1] * worst_mae
        + weights[2] * domain_sd
    )
    return domain_mae, mean_mae, worst_mae, domain_sd, objective


def select_stage_b_pipeline(
    evaluations: tuple[ResidualCellEvaluation, ...],
    *,
    incumbents: tuple[ResidualIncumbentEvidence, ...],
    finalists: tuple[str, str],
    config: ResidualDiffusionConfig,
) -> ResidualOuterSelection:
    """Freeze one outer's pre-outer residual/incumbent/ensemble decision."""

    clean = _config(config)
    if (
        type(evaluations) is not tuple
        or not evaluations
        or any(type(value) is not ResidualCellEvaluation for value in evaluations)
        or type(finalists) is not tuple
        or len(finalists) != clean.finalists_per_outer
        or len(set(finalists)) != len(finalists)
        or any(value not in clean.candidate_ids for value in finalists)
    ):
        raise ResidualSearchError("Stage-B selection evidence is invalid")
    outer = evaluations[0].cell.outer_domain
    expected = {
        (query, candidate_id, seed)
        for query in DOMAIN_ORDER
        if query != outer
        for candidate_id in finalists
        for seed in clean.training_seeds
    }
    observed = {
        (
            value.cell.query_domain,
            value.cell.candidate_id,
            value.cell.training_seed,
        )
        for value in evaluations
    }
    if (
        len(evaluations) != len(expected)
        or len(observed) != len(evaluations)
        or observed != expected
        or any(
            value.cell.stage != "B" or value.cell.outer_domain != outer
            for value in evaluations
        )
    ):
        raise ResidualSearchError("Stage-B Cartesian evidence is incomplete")
    summaries = tuple(
        summarize_candidate_cells(
            tuple(
                value
                for value in evaluations
                if value.cell.candidate_id == candidate_id
            ),
            config=clean,
            stage="B",
        )
        for candidate_id in finalists
    )
    if (
        type(incumbents) is not tuple
        or len(incumbents) != 2
        or any(type(value) is not ResidualIncumbentEvidence for value in incumbents)
        or {value.pipeline_id for value in incumbents} != {"PILOT", "B0"}
        or {value.outer_domain for value in incumbents} != {outer}
    ):
        raise ResidualSearchError("incumbent roster is invalid")
    reference = summaries[0]
    for incumbent in incumbents:
        if (
            incumbent.specimen_ids != reference.oof_specimen_ids
            or incumbent.domain_ids != reference.oof_domain_ids
            or not np.array_equal(incumbent.targets, reference.oof_targets)
        ):
            raise ResidualSearchError("incumbent identities or targets changed")
    incumbent_metrics = {
        value.pipeline_id: _prediction_metrics(
            value.predictions,
            value.targets,
            value.domain_ids,
            outer_domain=outer,
            config=clean,
        )
        for value in incumbents
    }
    best_incumbent = min(
        incumbents,
        key=lambda value: (
            incumbent_metrics[value.pipeline_id][4],
            incumbent_metrics[value.pipeline_id][1],
            incumbent_metrics[value.pipeline_id][2],
            value.pipeline_id,
        ),
    )
    incumbent_objective = incumbent_metrics[best_incumbent.pipeline_id][4]
    eligible = tuple(summary for summary in summaries if summary.eligible)
    best_residual = min(eligible, key=lambda value: value.rank_key) if eligible else None
    residual_improvement = (
        None
        if best_residual is None
        else incumbent_objective - best_residual.objective
    )
    residual_promoted = (
        best_residual is not None
        and residual_improvement is not None
        and residual_improvement >= clean.promotion_margin
    )
    ensemble: EnsembleResult | None = None
    if residual_promoted:
        assert best_residual is not None
        candidate_states = (
            best_residual.state_sha256,
            best_incumbent.state_sha256,
        )
        candidate = fit_nonnegative_ensemble(
            np.vstack(
                (best_residual.oof_predictions, best_incumbent.predictions)
            ),
            best_residual.oof_targets,
            specimen_ids=best_residual.oof_specimen_ids,
            domain_ids=best_residual.oof_domain_ids,
            candidate_sha256=candidate_states,
            minimum_j_gain=clean.ensemble_margin,
        )
        if candidate.accepted:
            ensemble = candidate
    if ensemble is not None:
        assert best_residual is not None
        selected_pipeline = "ENSEMBLE"
        selected_components = (
            best_residual.candidate_id,
            best_incumbent.pipeline_id,
        )
    elif residual_promoted:
        assert best_residual is not None
        selected_pipeline = "RESIDUAL"
        selected_components = (best_residual.candidate_id,)
    else:
        selected_pipeline = "INCUMBENT"
        selected_components = (best_incumbent.pipeline_id,)
    return ResidualOuterSelection(
        outer_domain=outer,
        candidate_summaries=summaries,
        incumbents=incumbents,
        best_residual=best_residual,
        best_incumbent=best_incumbent,
        best_incumbent_objective=incumbent_objective,
        residual_improvement=residual_improvement,
        residual_promoted=residual_promoted,
        ensemble=ensemble,
        ensemble_promoted=ensemble is not None,
        selected_pipeline=selected_pipeline,
        selected_components=selected_components,
    )


def run_residual_outer_search(
    search_view: D8SearchView,
    *,
    folds: Mapping[str, D8InnerFold],
    config: ResidualDiffusionConfig,
    scaffold: PilotDiffusionScaffold,
    pilot_candidate: D8Candidate,
    field_bank: ResidualFieldBank,
    incumbents: tuple[ResidualIncumbentEvidence, ...],
    assets: RegisteredPilotAssets,
    encoder: object,
    device: str,
    cell_recorder: Callable[..., None],
    final_recorder: Callable[[ResidualFinalTrainingResult], None],
    test_scale_override: bool = False,
) -> ResidualOuterSearchRun:
    """Run one prospective outer study while streaming model-bearing results."""

    clean = _config(config)
    if type(search_view) is not D8SearchView:
        raise TypeError("exact D8SearchView is required")
    validate_search_view(search_view)
    outer = search_view.outer_domain
    inner_domains = tuple(domain for domain in DOMAIN_ORDER if domain != outer)
    if not isinstance(folds, Mapping):
        raise TypeError("inner folds must be a mapping")
    fold_values = dict(folds)
    if (
        tuple(fold_values) != inner_domains
        or any(
            type(fold) is not D8InnerFold
            or fold.search_view is not search_view
            or fold.query_domain != domain
            for domain, fold in fold_values.items()
        )
        or not callable(cell_recorder)
        or not callable(final_recorder)
        or type(test_scale_override) is not bool
    ):
        raise ResidualSearchError("outer search authority changed")
    for fold in fold_values.values():
        validate_inner_fold(fold)
    outer_target = build_outer_fit_residual_target_batch(
        search_view,
        scaffold,
        field_bank=field_bank,
    )
    fit_targets = {
        domain: build_fit_residual_target_batch(
            fold_values[domain],
            scaffold,
            field_bank=field_bank,
        )
        for domain in inner_domains
    }

    def execute(cell: ResidualSearchCell, *, retain_checkpoint: bool) -> ResidualCellRun:
        run = run_residual_search_cell(
            cell,
            fold=fold_values[cell.query_domain],
            config=clean,
            fit_target_batch=fit_targets[cell.query_domain],
            outer_target_batch=outer_target,
            scaffold=scaffold,
            pilot_candidate=pilot_candidate,
            field_bank=field_bank,
            assets=assets,
            encoder=encoder,
            device=device,
            test_scale_override=test_scale_override,
        )
        if type(run) is not ResidualCellRun:
            raise ResidualSearchError("cell runner returned an invalid result")
        recorded = cell_recorder(run, retain_checkpoint=retain_checkpoint)
        if recorded is not None:
            raise ResidualSearchError("cell recorder must not alter search state")
        return run

    stage_a_evaluations: list[ResidualCellEvaluation] = []
    stage_a_states: list[str] = []
    for cell in stage_a_cell_keys(clean):
        if cell.outer_domain != outer:
            continue
        run = execute(cell, retain_checkpoint=False)
        stage_a_evaluations.append(run.evaluation)
        stage_a_states.append(run.state_sha256)
    stage_a = promote_stage_a_outer(
        tuple(stage_a_evaluations),
        config=clean,
        test_scale_override=test_scale_override,
    )
    stage_b_cells = tuple(
        ResidualSearchCell("B", outer, query, candidate_id, seed)
        for query in inner_domains
        for candidate_id in stage_a.finalists
        for seed in clean.training_seeds
    )
    stage_b_evaluations: list[ResidualCellEvaluation] = []
    stage_b_states: list[str] = []
    for cell in stage_b_cells:
        run = execute(cell, retain_checkpoint=True)
        stage_b_evaluations.append(run.evaluation)
        stage_b_states.append(run.state_sha256)
    selection = select_stage_b_pipeline(
        tuple(stage_b_evaluations),
        incumbents=incumbents,
        finalists=stage_a.finalists,
        config=clean,
    )
    final_digests: list[str] = []
    if selection.requires_final_residual_checkpoints:
        if selection.best_residual is None:
            raise ResidualSearchError("selected residual candidate is unavailable")
        candidate = clean.candidate(selection.best_residual.candidate_id)
        for seed in clean.training_seeds:
            final = train_outer_fit_residual_model(
                search_view,
                outer_target,
                config=clean,
                candidate=candidate,
                epochs=1 if test_scale_override else clean.rerank_epochs,
                seed=seed,
                device=device,
                test_scale_override=test_scale_override,
            )
            if (
                type(final) is not ResidualFinalTrainingResult
                or final.outer_domain != outer
                or final.candidate_id != candidate.candidate_id
                or final.seed != seed
                or final.split_sha256 != search_view.state_sha256
            ):
                raise ResidualSearchError("final residual training state changed")
            recorded = final_recorder(final)
            if recorded is not None:
                raise ResidualSearchError("final recorder must not alter search state")
            final_digests.append(final.checkpoint.scientific_digest)
    return ResidualOuterSearchRun(
        outer_domain=outer,
        stage_a=stage_a,
        stage_a_run_sha256=tuple(stage_a_states),
        stage_b_run_sha256=tuple(stage_b_states),
        selection=selection,
        final_training_sha256=tuple(final_digests),
        outer_evaluation_count=0,
    )


__all__ = [
    "ResidualCandidateSummary",
    "ResidualCellEvaluation",
    "ResidualCellRun",
    "ResidualFeatureBundle",
    "ResidualIncumbentEvidence",
    "ResidualOuterSearchRun",
    "ResidualOuterSelection",
    "ResidualSearchCell",
    "ResidualSearchError",
    "StageAOuterPromotion",
    "StageAPromotion",
    "build_residual_feature_bundle",
    "evaluate_residual_feature_bundle",
    "load_b0_incumbent_evidence",
    "load_pilot_incumbent_evidence",
    "load_pilot_scaffold_candidates",
    "promote_stage_a",
    "promote_stage_a_outer",
    "run_residual_outer_search",
    "run_residual_search_cell",
    "select_stage_b_pipeline",
    "stage_a_cell_keys",
    "stage_b_cell_keys",
    "summarize_candidate_cells",
]
