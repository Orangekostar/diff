"""Registered G0 component gates and final decision vocabulary."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl
import yaml
from PIL import Image

from cmc_bbdm.agentic_nde.surface_cells import (
    SurfaceCellAuthority,
    SurfaceCellRecord,
    load_surface_cell_authority,
)
from cmc_bbdm.mavis.authority import MAVISAuthority, load_mavis_authority
from cmc_bbdm.mavis.config import load_mavis_config
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.mva.encoder_session import MVAEncoderSession
from cmc_bbdm.mva.measurement_state import MeasurementState
from cmc_bbdm.mva.measurement_state import measurement_mask as mva_measurement_mask
from cmc_bbdm.mva.oracle import uniform_cell_order
from cmc_bbdm.mva.pipeline import _encoder

from .artifacts import G0PackageValidation, publish_g0_manifest
from .cai_assessor import (
    StateCAIAssessor,
    StateFeatureRow,
    fit_state_cai_assessor,
    state_scalars,
)
from .contracts import InspectionObservation, InspectionTask
from .evaluation import (
    project_oracle_checkpoints,
    task_swap_advantages,
    trajectory_overlap,
    zero_inclusive_auebc,
)
from .field_task import (
    InternalSignalSaliency,
    field_loss,
    internal_signal_saliency,
    normalized_capture_auc,
    signal_capture,
)
from .generalized_reconstruction import (
    SourceBackgroundPrior,
    fit_source_background_prior,
    reconstruct_observation,
)
from .oracle import (
    OracleTrajectory,
    run_cai_oracle,
    run_discovery_oracle,
    run_field_oracle,
)
from .state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    action_added_positions_from_mask,
    apply_action,
    budget_record,
    measurement_mask,
    zero_state,
)
from .state_bank import (
    StateBankPolicy,
    materialize_state_bank,
    plan_policy_actions,
)
from .statistics import PairedBootstrapSummary, synchronized_paired_bootstrap
from .stopping import (
    ReferenceEndpoint,
    earliest_sufficient_state,
    select_strongest_fixed_reference,
)
from .surface_hypothesis import SurfaceHypothesis, compute_surface_hypothesis
from .world import CausalInspectionWorld

_BASE_SHA = "892d92ea4979d9ca8ceeafef3348cd43266ed1b8"
_PROMPT_SHA = "5e5ad7bdf871f445b6cb60476540e01b267b42d4b3194c4463ca4a273e83f8bf"
_DOMAIN_ORDER = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
_DOMAIN_COUNTS = MappingProxyType(
    {
        "74t7kcdgkr": 45,
        "cgtnjyggtm": 49,
        "w68dtmpfyf": 43,
        "xcmzfsbd9t": 59,
        "yfxyg8jm46": 42,
        "ykhs7s2dck": 38,
    }
)
_INITIAL_BUDGETS = MappingProxyType(
    {
        "74t7kcdgkr": 0.03125,
        "cgtnjyggtm": 0.015625,
        "w68dtmpfyf": 0.015625,
        "xcmzfsbd9t": 0.015625,
        "yfxyg8jm46": 0.015625,
        "ykhs7s2dck": 0.015625,
    }
)
_CHECKPOINTS = (0.0, 0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25)


class G0ExecutionError(RuntimeError):
    """Raised when the preregistered G0 execution contract cannot be honored."""


@dataclass(frozen=True, slots=True)
class G0Protocol:
    config_path: Path
    config_sha256: str
    specimen_count: int
    domain_order: tuple[str, ...]
    domain_counts: MappingProxyType
    initial_budget_by_domain: MappingProxyType
    endpoint_budget: float
    checkpoints: tuple[float, ...]
    snapshot_fractions: tuple[float, ...]
    state_bank_seed: int
    bootstrap_replicates: int
    bootstrap_seed: int
    default_device: str
    encoder_batch_size: int
    source_bindings: MappingProxyType


@dataclass(frozen=True, slots=True)
class G0RunResult:
    output_dir: Path
    status: str
    package: G0PackageValidation
    cai_assessor_authorized: bool
    specimen_count: int


@dataclass(frozen=True, slots=True)
class _SurfaceDatum:
    record: SurfaceCellRecord
    image: np.ndarray
    hypothesis: SurfaceHypothesis


@dataclass(frozen=True, slots=True)
class _RuntimeAuthority:
    mavis: MAVISAuthority
    surfaces: Mapping[tuple[str, str], _SurfaceDatum]
    surface_authority_sha256: str


@dataclass(frozen=True, slots=True)
class _AssessorAudit:
    assessors: Mapping[str, StateCAIAssessor]
    priors: Mapping[str, SourceBackgroundPrior]
    state_bank_rows: tuple[dict[str, object], ...]
    metric_rows: tuple[dict[str, object], ...]
    bootstrap: PairedBootstrapSummary
    zero_mae: float
    endpoint_mae: float
    replay_valid: bool
    outer_exclusion_valid: bool
    authorized: bool


@dataclass(frozen=True, slots=True)
class _InitializationMetric:
    dataset_id: str
    specimen_id: str
    method: str
    auc: float
    budget_to_25: float | None
    budget_to_50: float | None
    budget_to_75: float | None
    first_high_saliency_budget: float | None
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _InitializationAudit:
    metrics: tuple[_InitializationMetric, ...]
    rows: tuple[dict[str, object], ...]
    bootstrap: PairedBootstrapSummary
    relative_auc_improvement: float
    capture_budget_reduction: float
    headroom: bool


@dataclass(frozen=True, slots=True)
class _FixedCurve:
    observations: tuple[InspectionObservation, ...]
    exact_budgets: np.ndarray
    field_losses: np.ndarray
    cai_losses: np.ndarray | None
    field_auebc: float
    cai_auebc: float | None


@dataclass(frozen=True, slots=True)
class _HistoricalMAVISTrajectory:
    actions: tuple[InspectionCellAction, ...]
    exact_cost_before: tuple[int, ...]
    exact_cost_after: tuple[int, ...]
    source_state_sha256_before: tuple[str, ...]
    source_state_sha256_after: tuple[str, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class _HierarchyAudit:
    endpoint_rows: tuple[dict[str, object], ...]
    trajectory_rows: tuple[dict[str, object], ...]
    stopping_rows: tuple[dict[str, object], ...]
    field_fixed_curves: Mapping[tuple[str, str], _FixedCurve]
    historical_mavis_curves: Mapping[tuple[str, str], _FixedCurve]
    historical_mavis_sha256: Mapping[tuple[str, str], str]
    field_oracles: Mapping[tuple[str, str], OracleTrajectory]
    field_bootstrap: PairedBootstrapSummary
    field_relative_auebc_improvement: float
    field_mean_stopping_saving: float
    field_stopping_loss_ratio: float
    field_headroom: bool
    field_stopping_bootstrap: PairedBootstrapSummary
    field_stopping_headroom: bool
    reference_methods: Mapping[tuple[str, str], str]


@dataclass(frozen=True, slots=True)
class _CAIHierarchyAudit:
    status: str
    trajectory_rows: tuple[dict[str, object], ...]
    stopping_rows: tuple[dict[str, object], ...]
    task_swap_rows: tuple[dict[str, object], ...]
    cai_bootstrap: PairedBootstrapSummary | None
    cai_relative_auebc_improvement: float | None
    cai_mean_stopping_saving: float | None
    cai_stopping_loss_ratio: float | None
    cai_headroom: bool
    cai_stopping_bootstrap: PairedBootstrapSummary | None
    cai_stopping_headroom: bool
    field_swap_bootstrap: PairedBootstrapSummary | None
    cai_swap_bootstrap: PairedBootstrapSummary | None
    task_conditioning_headroom: bool


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise G0ExecutionError(f"{label} must be a mapping")
    return value


def _tuple_float(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise G0ExecutionError(f"{label} must be a nonempty list")
    try:
        output = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise G0ExecutionError(f"{label} is invalid") from error
    if not all(math.isfinite(item) for item in output):
        raise G0ExecutionError(f"{label} is invalid")
    return output


def load_g0_protocol(
    path: str | Path,
    *,
    project_root: str | Path,
) -> G0Protocol:
    root = Path(project_root).resolve(strict=True)
    config_path = Path(path).resolve(strict=True)
    try:
        payload = config_path.read_bytes()
        raw = yaml.safe_load(payload)
    except (OSError, yaml.YAMLError) as error:
        raise G0ExecutionError("G0 protocol config is unavailable") from error
    config = _mapping(raw, "G0 config")
    if (
        config.get("schema_version") != 1
        or config.get("stage") != "INSPECTION_AGENT_G0"
        or config.get("mode") != "formal"
        or config.get("repository_base_sha") != _BASE_SHA
        or config.get("controlling_prompt_sha256") != _PROMPT_SHA
        or config.get("configuration_frozen") is not True
    ):
        raise G0ExecutionError("G0 protocol identity changed")

    cohort = _mapping(config.get("cohort"), "G0 cohort")
    domain_order = tuple(cohort.get("domain_order", ()))
    domain_counts = _mapping(cohort.get("domain_counts"), "G0 domain counts")
    if (
        cohort.get("specimen_count") != 276
        or domain_order != _DOMAIN_ORDER
        or domain_counts != dict(_DOMAIN_COUNTS)
        or sum(int(value) for value in domain_counts.values()) != 276
        or cohort.get("outer_split") != "leave_one_domain_out"
        or cohort.get("p0r_status") != "P0R_AUTHOR_REGISTRATION_GO"
        or cohort.get("p0r_orientation") != "ROT90"
    ):
        raise G0ExecutionError("G0 cohort contract changed")

    acquisition = _mapping(config.get("acquisition"), "G0 acquisition")
    initial_budgets = _mapping(
        acquisition.get("initial_budget_by_domain"),
        "G0 initial budgets",
    )
    checkpoints = _tuple_float(
        acquisition.get("evaluation_checkpoints"),
        "G0 checkpoints",
    )
    if (
        acquisition.get("cell_shape") != [8, 8]
        or tuple(acquisition.get("levels", ())) != (-1, 0, 1, 2)
        or tuple(acquisition.get("allowed_transitions", ()))
        != ("-1->0", "0->1", "1->2")
        or acquisition.get("budget_unit") != "unique_native_raster_locations"
        or float(acquisition.get("endpoint_budget", -1.0)) != 0.25
        or initial_budgets != dict(_INITIAL_BUDGETS)
        or checkpoints != _CHECKPOINTS
        or acquisition.get("nonfitting_action") != "skip_without_reordering"
    ):
        raise G0ExecutionError("G0 acquisition contract changed")

    state_bank = _mapping(config.get("state_bank"), "G0 state bank")
    fractions = _tuple_float(
        state_bank.get("snapshot_fractions"),
        "G0 state-bank fractions",
    )
    if (
        state_bank.get("include_zero_anchor") is not True
        or state_bank.get("snapshots_per_policy") != 3
        or fractions != (1 / 3, 2 / 3, 1.0)
        or state_bank.get("snapshot_axis") != "action_count"
        or state_bank.get("label_independent") is not True
        or state_bank.get("equal_states_per_specimen") is not True
        or state_bank.get("expected_states_per_specimen") != 19
        or state_bank.get("random_seed") != 2026083101
    ):
        raise G0ExecutionError("G0 state-bank contract changed")

    assessor = _mapping(config.get("cai_assessor"), "G0 CAI assessor")
    statistics = _mapping(config.get("statistics"), "G0 statistics")
    execution = _mapping(config.get("execution"), "G0 execution")
    gates = _mapping(config.get("gates"), "G0 gates")
    if (
        assessor.get("embedding_dimension") != 512
        or assessor.get("pca_dimension") != 32
        or float(assessor.get("ridge_alpha", -1.0)) != 10.0
        or assessor.get("metadata13_used") is not False
        or assessor.get("profile_stats21_used") is not False
        or assessor.get("target_fold_used_for_fit_normalization_selection") is not False
        or statistics.get("bootstrap") != "synchronized_specimen_within_domain"
        or statistics.get("bootstrap_replicates") != 100_000
        or statistics.get("bootstrap_seed") != 2026083102
        or float(statistics.get("confidence_level", -1.0)) != 0.95
        or float(gates.get("relative_improvement", -1.0)) != 0.10
        or gates.get("improved_domains_minimum") != 4
        or float(gates.get("stopping_budget_saving", -1.0)) != 0.10
        or float(gates.get("stopping_noninferiority_relative", -1.0)) != 0.05
        or execution.get("default_device") != "cuda:0"
        or execution.get("encoder_batch_size") != 32
        or execution.get("deterministic_algorithms") is not True
        or execution.get("no_new_planner_training") is not True
        or execution.get("allowed_learned_component") != "StateCAIAssessor"
    ):
        raise G0ExecutionError("G0 model, statistics, or gate contract changed")

    sources = _mapping(config.get("sources"), "G0 sources")
    source_bindings: dict[str, tuple[str, str]] = {}
    for name, value in sorted(sources.items()):
        binding = _mapping(value, f"G0 source {name}")
        if set(binding) != {"path", "sha256"}:
            raise G0ExecutionError("G0 source binding schema changed")
        relative = Path(str(binding["path"]))
        expected = str(binding["sha256"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(expected) != 64
            or set(expected) - set("0123456789abcdef")
        ):
            raise G0ExecutionError("G0 source binding is invalid")
        try:
            source_payload = (root / relative).read_bytes()
        except OSError as error:
            raise G0ExecutionError(f"G0 source is unavailable: {name}") from error
        if hashlib.sha256(source_payload).hexdigest() != expected:
            raise G0ExecutionError(f"G0 source hash changed: {name}")
        source_bindings[str(name)] = (relative.as_posix(), expected)

    return G0Protocol(
        config_path=config_path,
        config_sha256=hashlib.sha256(payload).hexdigest(),
        specimen_count=276,
        domain_order=_DOMAIN_ORDER,
        domain_counts=_DOMAIN_COUNTS,
        initial_budget_by_domain=_INITIAL_BUDGETS,
        endpoint_budget=0.25,
        checkpoints=_CHECKPOINTS,
        snapshot_fractions=fractions,
        state_bank_seed=2026083101,
        bootstrap_replicates=100_000,
        bootstrap_seed=2026083102,
        default_device="cuda:0",
        encoder_batch_size=32,
        source_bindings=MappingProxyType(source_bindings),
    )


def _load_historical_mavis_trajectories(
    protocol: G0Protocol,
    project_root: str | Path,
) -> Mapping[tuple[str, str], _HistoricalMAVISTrajectory]:
    root = Path(project_root).resolve(strict=True)
    relative, _expected_sha = protocol.source_bindings[
        "historical_mavis_trajectories"
    ]
    required = {
        "outer_domain",
        "specimen_id",
        "method",
        "step",
        "nominal_checkpoint",
        "cell_index",
        "from_level",
        "to_level",
        "exact_cost_before",
        "exact_cost_after",
        "state_sha256_before",
        "state_sha256_after",
        "feedback_used",
        "source",
    }
    try:
        frame = pl.read_parquet(root / relative)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise G0ExecutionError("frozen MAVIS trajectories cannot be read") from error
    if not required <= set(frame.columns):
        raise G0ExecutionError("frozen MAVIS trajectory schema changed")
    selected = frame.filter(pl.col("method") == "mavis_full").sort(
        ["outer_domain", "specimen_id", "step"]
    )
    groups = selected.partition_by(
        ["outer_domain", "specimen_id"],
        maintain_order=True,
        as_dict=True,
    )
    expected_keys = {
        (domain, specimen)
        for domain in protocol.domain_order
        for specimen in selected.filter(pl.col("outer_domain") == domain)
        .get_column("specimen_id")
        .unique()
        .to_list()
    }
    if (
        len(groups) != protocol.specimen_count
        or set(groups) != expected_keys
        or {
            domain: sum(key[0] == domain for key in groups)
            for domain in protocol.domain_order
        }
        != dict(protocol.domain_counts)
    ):
        raise G0ExecutionError("frozen MAVIS trajectory roster changed")
    output: dict[tuple[str, str], _HistoricalMAVISTrajectory] = {}
    for raw_key, group in groups.items():
        key = (str(raw_key[0]), str(raw_key[1]))
        rows = group.to_dicts()
        steps = tuple(int(row["step"]) for row in rows)
        actions = tuple(
            InspectionCellAction(
                int(row["cell_index"]),
                int(row["from_level"]),
                int(row["to_level"]),
            )
            for row in rows
        )
        before = tuple(int(row["exact_cost_before"]) for row in rows)
        after = tuple(int(row["exact_cost_after"]) for row in rows)
        state_before = tuple(str(row["state_sha256_before"]) for row in rows)
        state_after = tuple(str(row["state_sha256_after"]) for row in rows)
        if (
            not rows
            or steps != tuple(range(len(rows)))
            or any(row["feedback_used"] is not True for row in rows)
            or any(row["source"] != "mavis_causal_rollout" for row in rows)
            or any(
                float(row["nominal_checkpoint"]) not in protocol.checkpoints
                for row in rows
            )
            or any(
                action.from_level not in (0, 1)
                or action.to_level != action.from_level + 1
                for action in actions
            )
            or any(value <= 0 for value in before)
            or any(end <= start for start, end in zip(before, after, strict=True))
            or before[1:] != after[:-1]
            or any(
                len(value) != 64 or set(value) - set("0123456789abcdef")
                for value in (*state_before, *state_after)
            )
        ):
            raise G0ExecutionError("frozen MAVIS trajectory content changed")
        identity = [
            [
                steps[index],
                actions[index].cell_index,
                actions[index].from_level,
                actions[index].to_level,
                before[index],
                after[index],
                state_before[index],
                state_after[index],
            ]
            for index in range(len(rows))
        ]
        payload = json.dumps(identity, separators=(",", ":"), ensure_ascii=True)
        output[key] = _HistoricalMAVISTrajectory(
            actions=actions,
            exact_cost_before=before,
            exact_cost_after=after,
            source_state_sha256_before=state_before,
            source_state_sha256_after=state_after,
            state_sha256=hashlib.sha256(payload.encode("ascii")).hexdigest(),
        )
    return MappingProxyType(output)


def plan_staged_actions(
    grid: AcquisitionGrid,
    cell_order: tuple[int, ...],
    *,
    endpoint_budget: float,
) -> tuple[InspectionCellAction, ...]:
    endpoint = float(endpoint_budget)
    if (
        type(grid) is not AcquisitionGrid
        or type(cell_order) is not tuple
        or set(cell_order) != set(range(64))
        or len(cell_order) != 64
        or any(type(cell) is not int for cell in cell_order)
        or isinstance(endpoint_budget, bool)
        or not math.isfinite(endpoint)
        or not 0.0 < endpoint <= 1.0
    ):
        raise G0ExecutionError("fixed staged plan request is invalid")
    state = zero_state(grid)
    mask = np.zeros(grid.native_shape, dtype=np.bool_)
    measured_count = 0
    output: list[InspectionCellAction] = []
    for source, target in ((-1, 0), (0, 1), (1, 2)):
        for cell in cell_order:
            if state.levels[cell] != source:
                continue
            action = InspectionCellAction(cell, source, target)
            added = action_added_positions_from_mask(grid, state, action, mask)
            candidate_count = measured_count + len(added)
            if candidate_count / mask.size > endpoint + 1.0e-15:
                continue
            output.append(action)
            state = apply_action(grid, state, action)
            mask[added[:, 0], added[:, 1]] = True
            measured_count = candidate_count
    return tuple(output)


class G0Status(str, Enum):
    TASK_CONDITIONED = "G0_TASK_CONDITIONED_AGENTIC_OPPORTUNITY_GO"
    ACTIVE_INSPECTION = "G0_ACTIVE_INSPECTION_OPPORTUNITY_GO"
    FIELD_ONLY = "G0_FIELD_ONLY_OPPORTUNITY_GO"
    NO_AGENTIC_HEADROOM = "G0_NO_AGENTIC_HEADROOM_NO_GO"
    CAI_ASSESSOR_NO_GO = "G0_CAI_ASSESSOR_NO_GO"


@dataclass(frozen=True, slots=True)
class GateEvidence:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    improved_domains: int

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(float(value))
                for value in (self.point_estimate, self.ci_lower, self.ci_upper)
            )
            or self.ci_lower > self.ci_upper
            or type(self.improved_domains) is not int
            or not 0 <= self.improved_domains <= 6
        ):
            raise ValueError("G0 gate evidence is invalid")


@dataclass(frozen=True, slots=True)
class FinalG0Evidence:
    initialization_headroom: bool
    field_hierarchical_headroom: bool
    cai_assessor_authorized: bool
    cai_hierarchical_headroom: bool
    task_conditioning_headroom: bool
    field_stopping_headroom: bool
    cai_stopping_headroom: bool

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in (
            self.initialization_headroom,
            self.field_hierarchical_headroom,
            self.cai_assessor_authorized,
            self.cai_hierarchical_headroom,
            self.task_conditioning_headroom,
            self.field_stopping_headroom,
            self.cai_stopping_headroom,
        )):
            raise ValueError("final G0 evidence must be boolean")


def _positive(evidence: GateEvidence) -> bool:
    return (
        evidence.point_estimate > 0.0
        and evidence.ci_lower > 0.0
        and evidence.improved_domains >= 4
    )


def initialization_headroom_gate(
    evidence: GateEvidence,
    *,
    relative_auc_improvement: float,
    capture_budget_reduction: float,
) -> bool:
    return _positive(evidence) and (
        float(relative_auc_improvement) >= 0.10
        or float(capture_budget_reduction) >= 0.10
    )


def hierarchical_headroom_gate(
    evidence: GateEvidence,
    *,
    relative_auebc_improvement: float,
    sufficiency_budget_reduction: float,
) -> bool:
    return _positive(evidence) and (
        float(relative_auebc_improvement) >= 0.10
        or float(sufficiency_budget_reduction) >= 0.10
    )


def task_conditioning_gate(
    field_evidence: GateEvidence,
    cai_evidence: GateEvidence,
) -> bool:
    return _positive(field_evidence) and _positive(cai_evidence)


def stopping_headroom_gate(
    evidence: GateEvidence,
    *,
    mean_budget_saving: float,
    task_loss_ratio: float,
) -> bool:
    return (
        _positive(evidence)
        and float(mean_budget_saving) >= 0.10
        and float(task_loss_ratio) <= 1.05
    )


def assessor_authorization_gate(
    *,
    zero_mae: float,
    endpoint_mae: float,
    improvement: GateEvidence,
    replay_valid: bool,
    outer_exclusion_valid: bool,
) -> bool:
    return (
        math.isfinite(float(zero_mae))
        and math.isfinite(float(endpoint_mae))
        and float(endpoint_mae) < float(zero_mae)
        and _positive(improvement)
        and replay_valid is True
        and outer_exclusion_valid is True
    )


def decide_g0_status(evidence: FinalG0Evidence) -> G0Status:
    if type(evidence) is not FinalG0Evidence:
        raise ValueError("issued final G0 evidence is required")
    adaptive = evidence.field_hierarchical_headroom or (
        evidence.cai_assessor_authorized and evidence.cai_hierarchical_headroom
    )
    if not adaptive:
        return G0Status.NO_AGENTIC_HEADROOM
    if not evidence.cai_assessor_authorized:
        return (
            G0Status.FIELD_ONLY
            if evidence.field_hierarchical_headroom
            else G0Status.NO_AGENTIC_HEADROOM
        )
    stopping = evidence.field_stopping_headroom or evidence.cai_stopping_headroom
    if (
        evidence.initialization_headroom
        and evidence.task_conditioning_headroom
        and stopping
    ):
        return G0Status.TASK_CONDITIONED
    return G0Status.ACTIVE_INSPECTION


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _immutable_image(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.uint8)
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.uint8).reshape(
        array.shape
    )
    output.setflags(write=False)
    return output


def _load_surface_datum(
    record: SurfaceCellRecord,
    *,
    source_project_root: Path,
) -> _SurfaceDatum:
    path = source_project_root / record.surface_path
    try:
        payload = path.read_bytes()
        with Image.open(io.BytesIO(payload)) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise G0ExecutionError("authorized impacted-surface image cannot be decoded") from error
    if (
        hashlib.sha256(payload).hexdigest() != record.surface_sha256
        or image.shape != (record.source.height_px, record.source.width_px, 3)
    ):
        raise G0ExecutionError("authorized impacted-surface identity changed")
    frozen = _immutable_image(image)
    return _SurfaceDatum(
        record=record,
        image=frozen,
        hypothesis=compute_surface_hypothesis(
            frozen,
            record.cell_boxes,
            top_k=8,
        ),
    )


def _load_runtime_authority(
    protocol: G0Protocol,
    *,
    project_root: Path,
    source_project_root: Path,
    specimen_keys: frozenset[tuple[str, str]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> _RuntimeAuthority:
    mavis_path = project_root / protocol.source_bindings["mavis_config"][0]
    mavis_config = load_mavis_config(mavis_path, project_root=project_root)
    mavis = load_mavis_authority(
        mavis_config,
        source_project_root=source_project_root,
    )
    p0r = project_root / "results/agentic_task_driven_nde/p0r_author_registration"
    surface_authority: SurfaceCellAuthority = load_surface_cell_authority(
        p0r / "surface_manifest.csv",
        p0r / "registration.csv",
        p0r / "grid_mapping_qc.csv",
    )
    mavis_keys = set(zip(mavis.dataset_ids, mavis.specimen_ids, strict=True))
    surface_keys = set(
        zip(
            surface_authority.dataset_ids,
            surface_authority.specimen_ids,
            strict=True,
        )
    )
    if (
        mavis.specimen_count != protocol.specimen_count
        or surface_authority.specimen_count != protocol.specimen_count
        or mavis_keys != surface_keys
        or tuple(dict.fromkeys(mavis.dataset_ids)) != protocol.domain_order
        or {
            domain: mavis.dataset_ids.count(domain) for domain in protocol.domain_order
        }
        != dict(protocol.domain_counts)
    ):
        raise G0ExecutionError("MAVIS and P0R authorized rosters differ")
    selected = surface_keys if specimen_keys is None else set(specimen_keys)
    if not selected <= surface_keys:
        raise G0ExecutionError("requested G0 smoke roster is unauthorized")
    surfaces: dict[tuple[str, str], _SurfaceDatum] = {}
    records = [
        record
        for record in surface_authority.records
        if (record.dataset_id, record.specimen_id) in selected
    ]
    for index, record in enumerate(records, start=1):
        surfaces[(record.dataset_id, record.specimen_id)] = _load_surface_datum(
            record,
            source_project_root=source_project_root,
        )
        if index % 25 == 0 or index == len(records):
            _progress(progress, f"surface authority decoded: {index}/{len(records)}")
    return _RuntimeAuthority(
        mavis=mavis,
        surfaces=MappingProxyType(surfaces),
        surface_authority_sha256=surface_authority.state_sha256,
    )


def _grid(
    protocol: G0Protocol,
    domain: str,
    shape: tuple[int, int],
) -> AcquisitionGrid:
    from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid

    return build_acquisition_grid(
        *shape,
        initial_budget=float(protocol.initial_budget_by_domain[domain]),
    )


def _world(
    runtime: _RuntimeAuthority,
    datum: _SurfaceDatum,
    *,
    grid: AcquisitionGrid,
    task: InspectionTask,
    endpoint_budget: float,
) -> CausalInspectionWorld:
    return CausalInspectionWorld(
        runtime.mavis,
        specimen_id=datum.record.specimen_id,
        task=task,
        surface_rgb=datum.image,
        surface_sha256=datum.record.surface_sha256,
        grid=grid,
        endpoint_budget=endpoint_budget,
    )


def _source_priors(
    authority: MAVISAuthority,
    protocol: G0Protocol,
) -> Mapping[str, SourceBackgroundPrior]:
    return MappingProxyType(
        {
            outer: fit_source_background_prior(authority, outer_domain=outer)
            for outer in protocol.domain_order
        }
    )


def _zero_state_rows(
    runtime: _RuntimeAuthority,
    protocol: G0Protocol,
) -> tuple[dict[str, object], ...]:
    cache: dict[tuple[str, tuple[int, int]], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for specimen, domain in zip(
        runtime.mavis.specimen_ids,
        runtime.mavis.dataset_ids,
        strict=True,
    ):
        key = (domain, runtime.mavis.policy_context(specimen).native_shape)
        template = cache.get(key)
        if template is None:
            grid = _grid(protocol, domain, key[1])
            zero = zero_state(grid)
            level0 = GeneralizedMeasurementState(grid.state_sha256, (0,) * 64)
            level1 = GeneralizedMeasurementState(grid.state_sha256, (1,) * 64)
            level2 = GeneralizedMeasurementState(grid.state_sha256, (2,) * 64)
            zero_mask = measurement_mask(grid, zero)
            level0_mask = measurement_mask(grid, level0)
            level1_mask = measurement_mask(grid, level1)
            level2_mask = measurement_mask(grid, level2)
            template = {
                "native_height": key[1][0],
                "native_width": key[1][1],
                "grid_sha256": grid.state_sha256,
                "zero_state_sha256": zero.state_sha256,
                "zero_count": int(np.count_nonzero(zero_mask)),
                "zero_budget": 0.0,
                "level0_count": int(np.count_nonzero(level0_mask)),
                "level0_budget": budget_record(grid, level0).effective_budget,
                "level1_count": int(np.count_nonzero(level1_mask)),
                "level1_budget": budget_record(grid, level1).effective_budget,
                "level2_count": int(np.count_nonzero(level2_mask)),
                "level2_budget": budget_record(grid, level2).effective_budget,
                "level0_matches_mva": bool(
                    np.array_equal(
                        level0_mask,
                        mva_measurement_mask(
                            grid,
                            MeasurementState(grid.state_sha256, (0,) * 64),
                        ),
                    )
                ),
                "level1_matches_mva": bool(
                    np.array_equal(
                        level1_mask,
                        mva_measurement_mask(
                            grid,
                            MeasurementState(grid.state_sha256, (1,) * 64),
                        ),
                    )
                ),
                "level2_is_full_raster": bool(np.all(level2_mask)),
            }
            cache[key] = template
        rows.append({"dataset_id": domain, "specimen_id": specimen, **template})
    if any(
        row["zero_count"] != 0
        or row["zero_budget"] != 0.0
        or row["level0_matches_mva"] is not True
        or row["level1_matches_mva"] is not True
        or row["level2_is_full_raster"] is not True
        for row in rows
    ):
        raise G0ExecutionError("zero-state equivalence audit failed")
    return tuple(rows)


def _registered_encoder(project_root: Path, device: str) -> MVAEncoderSession:
    return MVAEncoderSession(_encoder(project_root, device))


def _equal_domain_mean(
    dataset_ids: Iterable[str],
    values: Iterable[float],
) -> float:
    pairs = tuple(zip(dataset_ids, values, strict=True))
    domains = tuple(sorted({domain for domain, _value in pairs}))
    if not pairs or not domains:
        raise G0ExecutionError("equal-domain aggregation is empty")
    return float(
        np.mean(
            [
                np.mean(
                    [float(value) for row_domain, value in pairs if row_domain == domain],
                    dtype=np.float64,
                )
                for domain in domains
            ],
            dtype=np.float64,
        )
    )


def _build_assessor_audit(
    runtime: _RuntimeAuthority,
    protocol: G0Protocol,
    *,
    encoder_project_root: Path,
    device: str,
    progress: Callable[[str], None] | None,
) -> _AssessorAudit:
    priors = _source_priors(runtime.mavis, protocol)
    encoder = _registered_encoder(encoder_project_root, device)
    rows_by_outer: dict[str, list[StateFeatureRow]] = {
        outer: [] for outer in protocol.domain_order
    }
    manifest_rows: list[dict[str, object]] = []
    roster = [
        (domain, specimen)
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if (domain, specimen) in runtime.surfaces
    ]
    for specimen_index, (domain, specimen) in enumerate(roster, start=1):
        datum = runtime.surfaces[(domain, specimen)]
        view = runtime.mavis.evaluation_view(specimen)
        grid = _grid(protocol, domain, view.full_scan.shape[:2])
        world = _world(
            runtime,
            datum,
            grid=grid,
            task=InspectionTask.CAI,
            endpoint_budget=protocol.endpoint_budget,
        )
        bank = materialize_state_bank(
            world,
            grid,
            datum.hypothesis,
            random_seed=protocol.state_bank_seed,
            snapshot_fractions=protocol.snapshot_fractions,
        )
        images: list[np.ndarray] = []
        identities: list[tuple[str, object]] = []
        for outer in protocol.domain_order:
            if outer == domain:
                continue
            for snapshot in bank:
                images.append(
                    reconstruct_observation(
                        snapshot.observation,
                        grid,
                        priors[outer],
                    ).image
                )
                identities.append((outer, snapshot))
        embeddings = encoder.encode(tuple(images)).astype(np.float64, copy=False)
        if embeddings.shape != (len(identities), 512):
            raise G0ExecutionError("state-bank embedding roster changed")
        for embedding, (outer, raw_snapshot) in zip(
            embeddings,
            identities,
            strict=True,
        ):
            snapshot = raw_snapshot
            observation = snapshot.observation
            sample_id = (
                f"{outer}::{domain}::{specimen}::{snapshot.policy}::"
                f"{snapshot.snapshot_index}"
            )
            feature = StateFeatureRow(
                sample_id=sample_id,
                specimen_id=specimen,
                dataset_id=domain,
                policy=snapshot.policy,
                observation_sha256=observation.state_sha256,
                embedding=embedding,
                effective_budget=observation.effective_budget,
                observed_cell_fraction=float(np.count_nonzero(
                    np.asarray(observation.measurement_state.levels) >= 0
                ) / 64),
                mean_observed_level=float(state_scalars(observation)[2]),
                true_cai=view.true_cai,
            )
            rows_by_outer[outer].append(feature)
            manifest_rows.append(
                {
                    "outer_domain": outer,
                    "dataset_id": domain,
                    "specimen_id": specimen,
                    "sample_id": sample_id,
                    "policy": snapshot.policy,
                    "snapshot_index": snapshot.snapshot_index,
                    "progress_fraction": snapshot.progress_fraction,
                    "action_count": len(observation.action_history),
                    "exact_acquired_count": observation.exact_acquired_count,
                    "effective_budget": observation.effective_budget,
                    "observation_sha256": observation.state_sha256,
                    "snapshot_sha256": snapshot.state_sha256,
                    "source_prior_sha256": priors[outer].state_sha256,
                    "embedding_sha256": hashlib.sha256(
                        feature.embedding.tobytes(order="C")
                    ).hexdigest(),
                    "outer_target_excluded": True,
                }
            )
        if specimen_index % 10 == 0 or specimen_index == len(roster):
            _progress(
                progress,
                f"CAI state bank encoded: {specimen_index}/{len(roster)}",
            )
    assessors = MappingProxyType(
        {
            outer: fit_state_cai_assessor(
                tuple(rows_by_outer[outer]),
                outer_domain=outer,
                pca_dimension=32,
                ridge_alpha=10.0,
            )
            for outer in protocol.domain_order
        }
    )
    metric_rows: list[dict[str, object]] = []
    replay_valid = True
    outer_exclusion_valid = True
    for outer in protocol.domain_order:
        target_roster = [
            (domain, specimen)
            for domain, specimen in roster
            if domain == outer
        ]
        observations: list[tuple[InspectionObservation, InspectionObservation]] = []
        images: list[np.ndarray] = []
        scalars: list[np.ndarray] = []
        for domain, specimen in target_roster:
            datum = runtime.surfaces[(domain, specimen)]
            view = runtime.mavis.evaluation_view(specimen)
            grid = _grid(protocol, domain, view.full_scan.shape[:2])
            world = _world(
                runtime,
                datum,
                grid=grid,
                task=InspectionTask.CAI,
                endpoint_budget=protocol.endpoint_budget,
            )
            zero = world.reset()
            endpoint = world.replay(
                plan_staged_actions(
                    grid,
                    uniform_cell_order(),
                    endpoint_budget=protocol.endpoint_budget,
                )
            )
            observations.append((zero, endpoint))
            for observation in (zero, endpoint):
                images.append(
                    reconstruct_observation(
                        observation,
                        grid,
                        priors[outer],
                    ).image
                )
                scalars.append(state_scalars(observation))
        embeddings = encoder.encode(tuple(images)).astype(np.float64, copy=False)
        scalar_matrix = np.asarray(scalars, dtype=np.float64)
        predictions = assessors[outer].predict(embeddings, scalar_matrix)
        repeated = assessors[outer].predict(embeddings, scalar_matrix)
        replay_valid = replay_valid and np.array_equal(predictions, repeated)
        assessor = assessors[outer]
        fit_ids = set(assessor.fit_physical_specimen_ids)
        target_ids = {specimen for _domain, specimen in target_roster}
        excluded = (
            outer not in assessor.fit_domains
            and not fit_ids.intersection(target_ids)
            and all(row.dataset_id != outer for row in rows_by_outer[outer])
        )
        outer_exclusion_valid = outer_exclusion_valid and excluded
        for target_index, ((domain, specimen), pair) in enumerate(
            zip(target_roster, observations, strict=True)
        ):
            view = runtime.mavis.evaluation_view(specimen)
            zero_prediction = float(predictions[2 * target_index])
            endpoint_prediction = float(predictions[2 * target_index + 1])
            zero_error = abs(view.true_cai - zero_prediction)
            endpoint_error = abs(view.true_cai - endpoint_prediction)
            metric_rows.append(
                {
                    "outer_domain": outer,
                    "dataset_id": domain,
                    "specimen_id": specimen,
                    "true_cai": view.true_cai,
                    "zero_prediction": zero_prediction,
                    "endpoint_prediction": endpoint_prediction,
                    "zero_absolute_error": zero_error,
                    "endpoint_absolute_error": endpoint_error,
                    "improvement": zero_error - endpoint_error,
                    "zero_state_sha256": pair[0].state_sha256,
                    "endpoint_state_sha256": pair[1].state_sha256,
                    "model_state_sha256": assessor.model_state_sha256,
                    "state_bank_sha256": assessor.state_bank_sha256,
                    "source_prior_sha256": priors[outer].state_sha256,
                    "fit_physical_specimen_count": len(
                        assessor.fit_physical_specimen_ids
                    ),
                    "fit_state_count": len(assessor.fit_sample_ids),
                    "outer_target_excluded": excluded,
                    "prediction_replay_identical": bool(
                        predictions[2 * target_index]
                        == repeated[2 * target_index]
                        and predictions[2 * target_index + 1]
                        == repeated[2 * target_index + 1]
                    ),
                }
            )
        _progress(progress, f"CAI assessor target fold evaluated: {outer}")
    dataset_ids = tuple(str(row["dataset_id"]) for row in metric_rows)
    specimen_ids = tuple(str(row["specimen_id"]) for row in metric_rows)
    effects = np.asarray([float(row["improvement"]) for row in metric_rows])
    bootstrap = synchronized_paired_bootstrap(
        dataset_ids=dataset_ids,
        specimen_ids=specimen_ids,
        effects=effects,
        replicates=protocol.bootstrap_replicates,
        seed=protocol.bootstrap_seed,
    )
    zero_mae = _equal_domain_mean(
        dataset_ids,
        (float(row["zero_absolute_error"]) for row in metric_rows),
    )
    endpoint_mae = _equal_domain_mean(
        dataset_ids,
        (float(row["endpoint_absolute_error"]) for row in metric_rows),
    )
    improvement = GateEvidence(
        bootstrap.point_estimate,
        bootstrap.ci_lower,
        bootstrap.ci_upper,
        bootstrap.improved_domains,
    )
    authorized = assessor_authorization_gate(
        zero_mae=zero_mae,
        endpoint_mae=endpoint_mae,
        improvement=improvement,
        replay_valid=replay_valid,
        outer_exclusion_valid=outer_exclusion_valid,
    )
    return _AssessorAudit(
        assessors=assessors,
        priors=priors,
        state_bank_rows=tuple(manifest_rows),
        metric_rows=tuple(metric_rows),
        bootstrap=bootstrap,
        zero_mae=zero_mae,
        endpoint_mae=endpoint_mae,
        replay_valid=replay_valid,
        outer_exclusion_valid=outer_exclusion_valid,
        authorized=authorized,
    )


def _verify_controlling_prompt(source_project_root: Path) -> None:
    prompt = (
        source_project_root.parent
        / "CODEX_INSPECTION_AGENT_G0_AUTONOMOUS_REASONING_PROMPT.md"
    )
    try:
        actual = hashlib.sha256(prompt.read_bytes()).hexdigest()
    except OSError as error:
        raise G0ExecutionError("controlling G0 prompt is unavailable") from error
    if actual != _PROMPT_SHA:
        raise G0ExecutionError("controlling G0 prompt hash changed")


def smoke_g0_assessor(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_project_root: str | Path,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    source_root = Path(source_project_root).resolve(strict=True)
    protocol = load_g0_protocol(config_path, project_root=root)
    _verify_controlling_prompt(source_root)
    manifest_path = (
        root
        / "results/agentic_task_driven_nde/p0r_author_registration/surface_manifest.csv"
    )
    roster = pl.read_csv(manifest_path).sort(["dataset_id", "specimen_id"])
    selected = frozenset(
        (str(row["dataset_id"]), str(row["specimen_id"]))
        for row in roster.group_by("dataset_id", maintain_order=True).head(1).iter_rows(
            named=True
        )
    )
    if len(selected) != 6:
        raise G0ExecutionError("G0 smoke roster does not cover six domains")
    runtime = _load_runtime_authority(
        protocol,
        project_root=root,
        source_project_root=source_root,
        specimen_keys=selected,
        progress=progress,
    )
    zero_rows = tuple(
        row
        for row in _zero_state_rows(runtime, protocol)
        if (str(row["dataset_id"]), str(row["specimen_id"])) in selected
    )
    if len(zero_rows) != 6:
        raise G0ExecutionError("G0 smoke zero-state roster changed")
    audit = _build_assessor_audit(
        runtime,
        protocol,
        encoder_project_root=source_root,
        device=device or protocol.default_device,
        progress=progress,
    )
    return {
        "specimen_count": len(selected),
        "zero_state_rows": len(zero_rows),
        "state_bank_rows": len(audit.state_bank_rows),
        "metric_rows": len(audit.metric_rows),
        "zero_mae": audit.zero_mae,
        "endpoint_mae": audit.endpoint_mae,
        "ci_lower": audit.bootstrap.ci_lower,
        "ci_upper": audit.bootstrap.ci_upper,
        "improved_domains": audit.bootstrap.improved_domains,
        "authorized": audit.authorized,
        "replay_valid": audit.replay_valid,
        "outer_exclusion_valid": audit.outer_exclusion_valid,
    }


def _capture_budget(
    budgets: np.ndarray,
    captures: np.ndarray,
    target: float,
) -> float | None:
    passing = np.flatnonzero(captures >= target)
    return None if not passing.size else float(budgets[int(passing[0])])


def _initialization_metric_from_actions(
    *,
    dataset_id: str,
    specimen_id: str,
    method: str,
    grid: AcquisitionGrid,
    actions: tuple[InspectionCellAction, ...],
    saliency: InternalSignalSaliency,
) -> _InitializationMetric:
    state = zero_state(grid)
    mask = np.zeros(grid.native_shape, dtype=np.bool_)
    budgets = [0.0]
    captures = [0.0]
    rows: list[dict[str, object]] = [
        {
            "step": -1,
            "cell_index": None,
            "from_level": None,
            "to_level": None,
            "exact_added_cost": 0,
            "exact_acquired_count": 0,
            "effective_budget": 0.0,
            "signal_capture": 0.0,
            "state_sha256": state.state_sha256,
        }
    ]
    high_threshold = float(np.quantile(saliency.pixel_mass, 0.95))
    first_high: float | None = None
    measured_count = 0
    for step, action in enumerate(actions):
        added = action_added_positions_from_mask(grid, state, action, mask)
        if not len(added):
            state = apply_action(grid, state, action)
            continue
        state = apply_action(grid, state, action)
        mask[added[:, 0], added[:, 1]] = True
        measured_count += len(added)
        budget = float(measured_count / mask.size)
        capture = signal_capture(mask, saliency)
        if (
            first_high is None
            and high_threshold > 0.0
            and np.any(
                saliency.pixel_mass[added[:, 0], added[:, 1]] >= high_threshold
            )
        ):
            first_high = budget
        budgets.append(budget)
        captures.append(capture)
        rows.append(
            {
                "step": step,
                "cell_index": action.cell_index,
                "from_level": action.from_level,
                "to_level": action.to_level,
                "exact_added_cost": len(added),
                "exact_acquired_count": measured_count,
                "effective_budget": budget,
                "signal_capture": capture,
                "state_sha256": state.state_sha256,
            }
        )
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(captures, dtype=np.float64)
    if x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise G0ExecutionError("initialization curve budget axis is invalid")
    auc = normalized_capture_auc(x, y, scout_endpoint=float(x[-1]))
    return _InitializationMetric(
        dataset_id=dataset_id,
        specimen_id=specimen_id,
        method=method,
        auc=auc,
        budget_to_25=_capture_budget(x, y, 0.25),
        budget_to_50=_capture_budget(x, y, 0.50),
        budget_to_75=_capture_budget(x, y, 0.75),
        first_high_saliency_budget=first_high,
        rows=tuple(rows),
    )


def _initialization_metric_from_oracle(
    *,
    dataset_id: str,
    specimen_id: str,
    trajectory: OracleTrajectory,
) -> _InitializationMetric:
    if not trajectory.steps:
        raise G0ExecutionError("discovery oracle trajectory is empty")
    budgets = np.asarray(
        [0.0, *(step.budget_after for step in trajectory.steps)],
        dtype=np.float64,
    )
    captures = np.asarray(
        [
            1.0 - trajectory.steps[0].task_loss_before,
            *(1.0 - step.task_loss_after for step in trajectory.steps),
        ],
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = [
        {
            "step": -1,
            "cell_index": None,
            "from_level": None,
            "to_level": None,
            "exact_added_cost": 0,
            "exact_acquired_count": 0,
            "effective_budget": 0.0,
            "signal_capture": float(captures[0]),
            "state_sha256": trajectory.steps[0].state_sha256_before,
        }
    ]
    for step in trajectory.steps:
        rows.append(
            {
                "step": step.step,
                "cell_index": step.action.cell_index,
                "from_level": step.action.from_level,
                "to_level": step.action.to_level,
                "exact_added_cost": step.exact_cost_after - step.exact_cost_before,
                "exact_acquired_count": step.exact_cost_after,
                "effective_budget": step.budget_after,
                "signal_capture": 1.0 - step.task_loss_after,
                "state_sha256": step.state_sha256_after,
            }
        )
    return _InitializationMetric(
        dataset_id=dataset_id,
        specimen_id=specimen_id,
        method="ORACLE_DISCOVERY",
        auc=normalized_capture_auc(
            budgets,
            captures,
            scout_endpoint=float(budgets[-1]),
        ),
        budget_to_25=_capture_budget(budgets, captures, 0.25),
        budget_to_50=_capture_budget(budgets, captures, 0.50),
        budget_to_75=_capture_budget(budgets, captures, 0.75),
        first_high_saliency_budget=None,
        rows=tuple(rows),
    )


def _surface_internal_stratum(
    grid: AcquisitionGrid,
    hypothesis: SurfaceHypothesis,
    saliency: InternalSignalSaliency,
) -> tuple[str, float, tuple[int, ...]]:
    masses = np.empty(64, dtype=np.float64)
    for cell in grid.cells:
        row_start, row_stop = cell.rows[2][0], cell.rows[2][-1]
        column_start, column_stop = cell.columns[2][0], cell.columns[2][-1]
        owned_row_stop = row_stop + 1 if cell.row == 7 else row_stop
        owned_column_stop = column_stop + 1 if cell.column == 7 else column_stop
        masses[cell.index] = float(
            np.sum(
                saliency.pixel_mass[
                    row_start:owned_row_stop,
                    column_start:owned_column_stop,
                ],
                dtype=np.float64,
            )
        )
    internal = tuple(
        sorted(range(64), key=lambda cell: (-float(masses[cell]), cell))[:8]
    )
    surface_set = set(hypothesis.top_cells)
    internal_set = set(internal)
    jaccard = float(len(surface_set & internal_set) / len(surface_set | internal_set))
    if jaccard >= 0.5:
        stratum = "SURFACE_INTERNAL_AGREE"
    elif jaccard >= 0.125:
        stratum = "SURFACE_INTERNAL_PARTIAL"
    else:
        stratum = "SURFACE_INTERNAL_MISLEADING"
    return stratum, jaccard, internal


def _initialization_audit(
    runtime: _RuntimeAuthority,
    protocol: G0Protocol,
    *,
    progress: Callable[[str], None] | None,
) -> _InitializationAudit:
    metrics: list[_InitializationMetric] = []
    all_rows: list[dict[str, object]] = []
    roster = [
        (domain, specimen)
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if (domain, specimen) in runtime.surfaces
    ]
    policy_methods = (
        ("RANDOM", StateBankPolicy.RANDOM_BROADEN),
        ("ZERO_UNIFORM", StateBankPolicy.UNIFORM_BROADEN),
        ("CENTER_FIRST", StateBankPolicy.CENTER_BROADEN),
        ("SURFACE_FOCUS", StateBankPolicy.SURFACE_FOCUS),
    )
    for index, (domain, specimen) in enumerate(roster, start=1):
        datum = runtime.surfaces[(domain, specimen)]
        view = runtime.mavis.evaluation_view(specimen)
        grid = _grid(protocol, domain, view.full_scan.shape[:2])
        saliency = internal_signal_saliency(view.full_scan)
        stratum, jaccard, internal_cells = _surface_internal_stratum(
            grid,
            datum.hypothesis,
            saliency,
        )
        scout_state = GeneralizedMeasurementState(grid.state_sha256, (0,) * 64)
        scout_endpoint = budget_record(grid, scout_state).effective_budget
        specimen_metrics: list[_InitializationMetric] = []
        for method, policy in policy_methods:
            actions = plan_policy_actions(
                grid,
                policy,
                surface_hypothesis=datum.hypothesis,
                surface_sha256=datum.record.surface_sha256,
                random_seed=protocol.state_bank_seed,
                endpoint_budget=scout_endpoint,
            )
            specimen_metrics.append(
                _initialization_metric_from_actions(
                    dataset_id=domain,
                    specimen_id=specimen,
                    method=method,
                    grid=grid,
                    actions=actions,
                    saliency=saliency,
                )
            )
        oracle_world = _world(
            runtime,
            datum,
            grid=grid,
            task=InspectionTask.DISCOVERY,
            endpoint_budget=scout_endpoint,
        )
        specimen_metrics.append(
            _initialization_metric_from_oracle(
                dataset_id=domain,
                specimen_id=specimen,
                trajectory=run_discovery_oracle(
                    oracle_world,
                    grid,
                    full_scan=view.full_scan,
                    surface_hypothesis_cells=datum.hypothesis.top_cells,
                ),
            )
        )
        for metric in specimen_metrics:
            metrics.append(metric)
            for row in metric.rows:
                all_rows.append(
                    {
                        "dataset_id": domain,
                        "specimen_id": specimen,
                        "method": metric.method,
                        "surface_internal_stratum": stratum,
                        "surface_internal_jaccard": jaccard,
                        "surface_top_cells": ";".join(
                            str(cell) for cell in datum.hypothesis.top_cells
                        ),
                        "internal_top_cells": ";".join(
                            str(cell) for cell in internal_cells
                        ),
                        "capture_auc": metric.auc,
                        "budget_to_25": metric.budget_to_25,
                        "budget_to_50": metric.budget_to_50,
                        "budget_to_75": metric.budget_to_75,
                        "first_high_saliency_budget": (
                            metric.first_high_saliency_budget
                        ),
                        **row,
                    }
                )
        if index % 25 == 0 or index == len(roster):
            _progress(progress, f"G0-A initialization: {index}/{len(roster)}")
    by_key = {
        (metric.dataset_id, metric.specimen_id, metric.method): metric
        for metric in metrics
    }
    dataset_ids = tuple(domain for domain, _specimen in roster)
    specimen_ids = tuple(specimen for _domain, specimen in roster)
    effects = np.asarray(
        [
            by_key[(domain, specimen, "ORACLE_DISCOVERY")].auc
            - by_key[(domain, specimen, "ZERO_UNIFORM")].auc
            for domain, specimen in roster
        ],
        dtype=np.float64,
    )
    bootstrap = synchronized_paired_bootstrap(
        dataset_ids=dataset_ids,
        specimen_ids=specimen_ids,
        effects=effects,
        replicates=protocol.bootstrap_replicates,
        seed=protocol.bootstrap_seed,
    )
    baseline_auc = _equal_domain_mean(
        dataset_ids,
        (
            by_key[(domain, specimen, "ZERO_UNIFORM")].auc
            for domain, specimen in roster
        ),
    )
    relative = 0.0 if baseline_auc == 0.0 else bootstrap.point_estimate / baseline_auc
    reductions = []
    reduction_domains = []
    for domain, specimen in roster:
        baseline_budget = by_key[(domain, specimen, "ZERO_UNIFORM")].budget_to_25
        oracle_budget = by_key[(domain, specimen, "ORACLE_DISCOVERY")].budget_to_25
        if baseline_budget is not None and oracle_budget is not None and baseline_budget > 0:
            reductions.append((baseline_budget - oracle_budget) / baseline_budget)
            reduction_domains.append(domain)
    capture_reduction = (
        0.0
        if not reductions
        else _equal_domain_mean(reduction_domains, reductions)
    )
    evidence = GateEvidence(
        bootstrap.point_estimate,
        bootstrap.ci_lower,
        bootstrap.ci_upper,
        bootstrap.improved_domains,
    )
    return _InitializationAudit(
        metrics=tuple(metrics),
        rows=tuple(all_rows),
        bootstrap=bootstrap,
        relative_auc_improvement=relative,
        capture_budget_reduction=capture_reduction,
        headroom=initialization_headroom_gate(
            evidence,
            relative_auc_improvement=relative,
            capture_budget_reduction=capture_reduction,
        ),
    )


_FIXED_REFERENCE_METHODS = (
    "RANDOM",
    "ZERO_UNIFORM",
    "CENTER_FIRST",
    "SURFACE_FOCUS",
    "SURVEY_THEN_REFINE_FIXED",
)


def _cell_order_for_method(
    method: str,
    datum: _SurfaceDatum,
    grid: AcquisitionGrid,
    protocol: G0Protocol,
) -> tuple[int, ...]:
    if method in {"ZERO_UNIFORM", "SURVEY_THEN_REFINE_FIXED"}:
        return uniform_cell_order()
    if method == "CENTER_FIRST":
        return tuple(
            sorted(
                range(64),
                key=lambda cell: (
                    (cell // 8 - 3.5) ** 2 + (cell % 8 - 3.5) ** 2,
                    cell,
                ),
            )
        )
    if method == "SURFACE_FOCUS":
        return tuple(
            sorted(
                range(64),
                key=lambda cell: (-float(datum.hypothesis.scores[cell]), cell),
            )
        )
    if method == "RANDOM":
        actions = plan_policy_actions(
            grid,
            StateBankPolicy.RANDOM_BROADEN,
            surface_hypothesis=datum.hypothesis,
            surface_sha256=datum.record.surface_sha256,
            random_seed=protocol.state_bank_seed,
            endpoint_budget=protocol.endpoint_budget,
        )
        order = tuple(action.cell_index for action in actions)
        if set(order) != set(range(64)) or len(order) != 64:
            raise G0ExecutionError("random fixed cell order is incomplete")
        return order
    raise G0ExecutionError(f"unknown fixed reference method: {method}")


def _checkpoint_histories(
    grid: AcquisitionGrid,
    actions: tuple[InspectionCellAction, ...],
    checkpoints: tuple[float, ...],
) -> tuple[tuple[InspectionCellAction, ...], ...]:
    state = zero_state(grid)
    mask = np.zeros(grid.native_shape, dtype=np.bool_)
    measured = 0
    action_budgets: list[float] = []
    for action in actions:
        added = action_added_positions_from_mask(grid, state, action, mask)
        state = apply_action(grid, state, action)
        mask[added[:, 0], added[:, 1]] = True
        measured += len(added)
        action_budgets.append(float(measured / mask.size))
    histories = []
    for checkpoint in checkpoints:
        prefix = sum(budget <= checkpoint + 1.0e-15 for budget in action_budgets)
        histories.append(actions[:prefix])
    return tuple(histories)


def _curve_from_actions(
    runtime: _RuntimeAuthority,
    protocol: G0Protocol,
    datum: _SurfaceDatum,
    prior: SourceBackgroundPrior,
    *,
    actions: tuple[InspectionCellAction, ...],
    assessor: StateCAIAssessor | None,
    encoder: MVAEncoderSession | None,
) -> _FixedCurve:
    specimen = datum.record.specimen_id
    domain = datum.record.dataset_id
    view = runtime.mavis.evaluation_view(specimen)
    grid = _grid(protocol, domain, view.full_scan.shape[:2])
    world = _world(
        runtime,
        datum,
        grid=grid,
        task=InspectionTask.FIELD,
        endpoint_budget=protocol.endpoint_budget,
    )
    histories = _checkpoint_histories(grid, actions, protocol.checkpoints)
    observations = tuple(world.replay(history) for history in histories)
    reconstructions = tuple(
        reconstruct_observation(observation, grid, prior).image
        for observation in observations
    )
    exact_budgets = np.asarray(
        [observation.effective_budget for observation in observations],
        dtype=np.float64,
    )
    field_losses = np.asarray(
        [field_loss(view.full_scan, image) for image in reconstructions],
        dtype=np.float64,
    )
    cai_losses: np.ndarray | None = None
    cai_auebc: float | None = None
    if assessor is not None:
        if encoder is None:
            raise G0ExecutionError("authorized CAI fixed curve has no encoder")
        embeddings = encoder.encode(reconstructions).astype(np.float64, copy=False)
        scalar_matrix = np.asarray(
            [state_scalars(observation) for observation in observations],
            dtype=np.float64,
        )
        predictions = assessor.predict(embeddings, scalar_matrix)
        cai_losses = np.abs(predictions - view.true_cai)
        cai_auebc = zero_inclusive_auebc(protocol.checkpoints, cai_losses)
    return _FixedCurve(
        observations=observations,
        exact_budgets=exact_budgets,
        field_losses=field_losses,
        cai_losses=cai_losses,
        field_auebc=zero_inclusive_auebc(protocol.checkpoints, field_losses),
        cai_auebc=cai_auebc,
    )


def _fixed_curve(
    runtime: _RuntimeAuthority,
    protocol: G0Protocol,
    datum: _SurfaceDatum,
    prior: SourceBackgroundPrior,
    *,
    method: str,
    assessor: StateCAIAssessor | None,
    encoder: MVAEncoderSession | None,
) -> _FixedCurve:
    view = runtime.mavis.evaluation_view(datum.record.specimen_id)
    grid = _grid(
        protocol,
        datum.record.dataset_id,
        view.full_scan.shape[:2],
    )
    actions = plan_staged_actions(
        grid,
        _cell_order_for_method(method, datum, grid, protocol),
        endpoint_budget=protocol.endpoint_budget,
    )
    return _curve_from_actions(
        runtime,
        protocol,
        datum,
        prior,
        actions=actions,
        assessor=assessor,
        encoder=encoder,
    )


def _historical_mavis_curve(
    runtime: _RuntimeAuthority,
    protocol: G0Protocol,
    datum: _SurfaceDatum,
    prior: SourceBackgroundPrior,
    trajectory: _HistoricalMAVISTrajectory,
    *,
    assessor: StateCAIAssessor | None,
    encoder: MVAEncoderSession | None,
) -> _FixedCurve:
    view = runtime.mavis.evaluation_view(datum.record.specimen_id)
    grid = _grid(
        protocol,
        datum.record.dataset_id,
        view.full_scan.shape[:2],
    )
    scout = tuple(
        InspectionCellAction(cell, -1, 0) for cell in uniform_cell_order()
    )
    state = zero_state(grid)
    mask = np.zeros(grid.native_shape, dtype=np.bool_)
    measured = 0
    for action in scout:
        added = action_added_positions_from_mask(grid, state, action, mask)
        state = apply_action(grid, state, action)
        mask[added[:, 0], added[:, 1]] = True
        measured += len(added)
    if measured != trajectory.exact_cost_before[0]:
        raise G0ExecutionError("frozen MAVIS scout cost changed")
    for index, action in enumerate(trajectory.actions):
        if measured != trajectory.exact_cost_before[index]:
            raise G0ExecutionError("frozen MAVIS cost chain changed")
        added = action_added_positions_from_mask(grid, state, action, mask)
        state = apply_action(grid, state, action)
        mask[added[:, 0], added[:, 1]] = True
        measured += len(added)
        if measured != trajectory.exact_cost_after[index]:
            raise G0ExecutionError("frozen MAVIS action cost changed")
    if measured / mask.size > protocol.endpoint_budget + 1.0e-15:
        raise G0ExecutionError("frozen MAVIS endpoint exceeds G0 budget")
    return _curve_from_actions(
        runtime,
        protocol,
        datum,
        prior,
        actions=(*scout, *trajectory.actions),
        assessor=assessor,
        encoder=encoder,
    )


def _fixed_endpoint_rows(
    runtime: _RuntimeAuthority,
    protocol: G0Protocol,
    audit: _AssessorAudit,
    *,
    encoder: MVAEncoderSession | None,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    roster = [
        (domain, specimen)
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if (domain, specimen) in runtime.surfaces
    ]
    for specimen_index, (domain, specimen) in enumerate(roster, start=1):
        datum = runtime.surfaces[(domain, specimen)]
        view = runtime.mavis.evaluation_view(specimen)
        grid = _grid(protocol, domain, view.full_scan.shape[:2])
        images: list[np.ndarray] = []
        scalar_rows: list[np.ndarray] = []
        identities: list[tuple[str, str, InspectionObservation, float]] = []
        world = _world(
            runtime,
            datum,
            grid=grid,
            task=InspectionTask.FIELD,
            endpoint_budget=protocol.endpoint_budget,
        )
        observations_by_method: dict[str, InspectionObservation] = {}
        for method in _FIXED_REFERENCE_METHODS:
            order = _cell_order_for_method(method, datum, grid, protocol)
            matching = next(
                (
                    observation
                    for known_method, observation in observations_by_method.items()
                    if _cell_order_for_method(
                        known_method,
                        datum,
                        grid,
                        protocol,
                    )
                    == order
                ),
                None,
            )
            observations_by_method[method] = (
                matching
                if matching is not None
                else world.replay(
                    plan_staged_actions(
                        grid,
                        order,
                        endpoint_budget=protocol.endpoint_budget,
                    )
                )
            )
        for outer in protocol.domain_order:
            for method in _FIXED_REFERENCE_METHODS:
                observation = observations_by_method[method]
                image = reconstruct_observation(
                    observation,
                    grid,
                    audit.priors[outer],
                ).image
                images.append(image)
                scalar_rows.append(state_scalars(observation))
                identities.append(
                    (
                        outer,
                        method,
                        observation,
                        field_loss(view.full_scan, image),
                    )
                )
        predictions: dict[int, float] = {}
        if audit.authorized:
            if encoder is None:
                raise G0ExecutionError("authorized endpoint audit has no encoder")
            embeddings = encoder.encode(tuple(images)).astype(np.float64, copy=False)
            for outer in protocol.domain_order:
                indices = [
                    index
                    for index, identity in enumerate(identities)
                    if identity[0] == outer
                ]
                values = audit.assessors[outer].predict(
                    embeddings[indices],
                    np.asarray([scalar_rows[index] for index in indices]),
                )
                predictions.update(
                    {index: float(value) for index, value in zip(indices, values, strict=True)}
                )
        for identity_index, (outer, method, observation, field_value) in enumerate(
            identities
        ):
            prediction = predictions.get(identity_index)
            rows.append(
                {
                    "outer_domain": outer,
                    "dataset_id": domain,
                    "specimen_id": specimen,
                    "method": method,
                    "effective_budget": observation.effective_budget,
                    "exact_acquired_count": observation.exact_acquired_count,
                    "state_sha256": observation.state_sha256,
                    "field_loss": field_value,
                    "cai_prediction": prediction,
                    "cai_loss": (
                        None if prediction is None else abs(view.true_cai - prediction)
                    ),
                    "source_prior_sha256": audit.priors[outer].state_sha256,
                    "assessor_state_sha256": (
                        None
                        if not audit.authorized
                        else audit.assessors[outer].model_state_sha256
                    ),
                }
            )
        if specimen_index % 10 == 0 or specimen_index == len(roster):
            _progress(
                progress,
                f"fixed 25% endpoints: {specimen_index}/{len(roster)}",
            )
    return tuple(rows)


def _reference_methods(
    endpoint_rows: tuple[dict[str, object], ...],
    protocol: G0Protocol,
    *,
    cai_authorized: bool,
) -> Mapping[tuple[str, str], str]:
    selections: dict[tuple[str, str], str] = {}
    for outer in protocol.domain_order:
        for task, column in (
            ("FIELD", "field_loss"),
            ("CAI", "cai_loss"),
        ):
            if task == "CAI" and not cai_authorized:
                continue
            source_rows = tuple(
                ReferenceEndpoint(
                    method=str(row["method"]),
                    dataset_id=str(row["dataset_id"]),
                    specimen_id=str(row["specimen_id"]),
                    task_loss=float(row[column]),
                )
                for row in endpoint_rows
                if row["outer_domain"] == outer and row["dataset_id"] != outer
            )
            selection = select_strongest_fixed_reference(
                source_rows,
                allowed_methods=_FIXED_REFERENCE_METHODS,
            )
            selections[(outer, task)] = selection.method
    return MappingProxyType(selections)


def _candidate_roster(step: object) -> tuple[str, str]:
    rows = [
        (
            candidate.action.cell_index,
            candidate.action.from_level,
            candidate.action.to_level,
            candidate.exact_added_cost,
            candidate.raw_value,
            candidate.objective_value,
            candidate.task_loss_after,
            candidate.candidate_state_sha256,
        )
        for candidate in step.candidates
    ]
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True)
    return payload, hashlib.sha256(payload.encode("ascii")).hexdigest()


def _oracle_trajectory_rows(
    domain: str,
    specimen: str,
    trajectory: OracleTrajectory,
) -> list[dict[str, object]]:
    rows = []
    for step in trajectory.steps:
        candidate_roster, candidate_hash = _candidate_roster(step)
        rows.append(
            {
                "outer_domain": domain,
                "dataset_id": domain,
                "specimen_id": specimen,
                "task": trajectory.task.value,
                "method": trajectory.method,
                "step": step.step,
                "decision": step.decision.value,
                "cell_index": step.action.cell_index,
                "from_level": step.action.from_level,
                "to_level": step.action.to_level,
                "exact_cost_before": step.exact_cost_before,
                "exact_cost_after": step.exact_cost_after,
                "budget_before": step.budget_before,
                "budget_after": step.budget_after,
                "task_loss_before": step.task_loss_before,
                "task_loss_after": step.task_loss_after,
                "teacher_value": step.teacher_value,
                "objective_value": step.objective_value,
                "candidate_count": len(step.candidates),
                "candidate_roster_json": candidate_roster,
                "candidate_scores_sha256": candidate_hash,
                "state_sha256_before": step.state_sha256_before,
                "state_sha256_after": step.state_sha256_after,
                "trajectory_sha256": trajectory.state_sha256,
                "stop_status": step.stop_status,
            }
        )
    return rows


def _baseline_curve_rows(
    domain: str,
    specimen: str,
    protocol: G0Protocol,
    curve: _FixedCurve,
    *,
    method: str,
    decision: str,
    trajectory_sha256: str | None,
) -> list[dict[str, object]]:
    tasks: list[tuple[str, np.ndarray]] = [("FIELD", curve.field_losses)]
    if curve.cai_losses is not None:
        tasks.append(("CAI", curve.cai_losses))
    rows: list[dict[str, object]] = []
    empty_hash = hashlib.sha256(b"[]").hexdigest()
    for task, losses in tasks:
        for checkpoint_index, checkpoint in enumerate(protocol.checkpoints):
            rows.append(
                {
                    "outer_domain": domain,
                    "dataset_id": domain,
                    "specimen_id": specimen,
                    "task": task,
                    "method": method,
                    "step": checkpoint_index - 1,
                    "decision": decision,
                    "cell_index": None,
                    "from_level": None,
                    "to_level": None,
                    "exact_cost_before": None,
                    "exact_cost_after": curve.observations[
                        checkpoint_index
                    ].exact_acquired_count,
                    "budget_before": None,
                    "budget_after": curve.exact_budgets[checkpoint_index],
                    "task_loss_before": None,
                    "task_loss_after": losses[checkpoint_index],
                    "teacher_value": None,
                    "objective_value": None,
                    "candidate_count": 0,
                    "candidate_roster_json": "[]",
                    "candidate_scores_sha256": empty_hash,
                    "state_sha256_before": None,
                    "state_sha256_after": curve.observations[
                        checkpoint_index
                    ].state_sha256,
                    "trajectory_sha256": trajectory_sha256,
                    "stop_status": checkpoint == protocol.endpoint_budget,
                }
            )
    return rows


def _fixed_endpoint_trajectory_rows(
    endpoint_rows: tuple[dict[str, object], ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    empty_hash = hashlib.sha256(b"[]").hexdigest()
    for endpoint in endpoint_rows:
        if (
            endpoint["outer_domain"] != endpoint["dataset_id"]
            or endpoint["method"] == "SURVEY_THEN_REFINE_FIXED"
        ):
            continue
        for task, loss_key in (("FIELD", "field_loss"), ("CAI", "cai_loss")):
            loss = endpoint[loss_key]
            if loss is None:
                continue
            rows.append(
                {
                    "outer_domain": endpoint["outer_domain"],
                    "dataset_id": endpoint["dataset_id"],
                    "specimen_id": endpoint["specimen_id"],
                    "task": task,
                    "method": endpoint["method"],
                    "step": 6,
                    "decision": "FIXED_ENDPOINT",
                    "cell_index": None,
                    "from_level": None,
                    "to_level": None,
                    "exact_cost_before": None,
                    "exact_cost_after": endpoint["exact_acquired_count"],
                    "budget_before": None,
                    "budget_after": endpoint["effective_budget"],
                    "task_loss_before": None,
                    "task_loss_after": loss,
                    "teacher_value": None,
                    "objective_value": None,
                    "candidate_count": 0,
                    "candidate_roster_json": "[]",
                    "candidate_scores_sha256": empty_hash,
                    "state_sha256_before": None,
                    "state_sha256_after": endpoint["state_sha256"],
                    "trajectory_sha256": None,
                    "stop_status": True,
                }
            )
    return rows


def _hierarchy_audit(
    runtime: _RuntimeAuthority,
    protocol: G0Protocol,
    assessor_audit: _AssessorAudit,
    *,
    encoder_project_root: Path,
    device: str,
    progress: Callable[[str], None] | None,
) -> _HierarchyAudit:
    encoder = (
        _registered_encoder(encoder_project_root, device)
        if assessor_audit.authorized
        else None
    )
    endpoint_rows = _fixed_endpoint_rows(
        runtime,
        protocol,
        assessor_audit,
        encoder=encoder,
        progress=progress,
    )
    references = _reference_methods(
        endpoint_rows,
        protocol,
        cai_authorized=assessor_audit.authorized,
    )
    historical_trajectories = _load_historical_mavis_trajectories(
        protocol,
        protocol.config_path.parents[2],
    )
    roster = [
        (domain, specimen)
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if (domain, specimen) in runtime.surfaces
    ]
    fixed_curves: dict[tuple[str, str], _FixedCurve] = {}
    historical_curves: dict[tuple[str, str], _FixedCurve] = {}
    historical_hashes: dict[tuple[str, str], str] = {}
    field_oracles: dict[tuple[str, str], OracleTrajectory] = {}
    trajectory_rows = _fixed_endpoint_trajectory_rows(endpoint_rows)
    stopping_rows: list[dict[str, object]] = []
    field_effects: list[float] = []
    stopping_effects: list[float] = []
    fixed_auebcs: list[float] = []
    final_stop_losses: list[float] = []
    reference_losses: list[float] = []
    for index, (domain, specimen) in enumerate(roster, start=1):
        datum = runtime.surfaces[(domain, specimen)]
        view = runtime.mavis.evaluation_view(specimen)
        grid = _grid(protocol, domain, view.full_scan.shape[:2])
        fixed = _fixed_curve(
            runtime,
            protocol,
            datum,
            assessor_audit.priors[domain],
            method="SURVEY_THEN_REFINE_FIXED",
            assessor=(
                assessor_audit.assessors[domain]
                if assessor_audit.authorized
                else None
            ),
            encoder=encoder,
        )
        fixed_curves[(domain, specimen)] = fixed
        trajectory_rows.extend(
            _baseline_curve_rows(
                domain,
                specimen,
                protocol,
                fixed,
                method="SURVEY_THEN_REFINE_FIXED",
                decision="FIXED",
                trajectory_sha256=None,
            )
        )
        historical_source = historical_trajectories[(domain, specimen)]
        historical = _historical_mavis_curve(
            runtime,
            protocol,
            datum,
            assessor_audit.priors[domain],
            historical_source,
            assessor=(
                assessor_audit.assessors[domain]
                if assessor_audit.authorized
                else None
            ),
            encoder=encoder,
        )
        historical_curves[(domain, specimen)] = historical
        historical_hashes[(domain, specimen)] = historical_source.state_sha256
        trajectory_rows.extend(
            _baseline_curve_rows(
                domain,
                specimen,
                protocol,
                historical,
                method="FIXED_UNIFORM_THEN_MAVIS",
                decision="HISTORICAL_METADATA_AUGMENTED",
                trajectory_sha256=historical_source.state_sha256,
            )
        )
        oracle = run_field_oracle(
            _world(
                runtime,
                datum,
                grid=grid,
                task=InspectionTask.FIELD,
                endpoint_budget=protocol.endpoint_budget,
            ),
            grid,
            assessor_audit.priors[domain],
            full_scan=view.full_scan,
            surface_hypothesis_cells=datum.hypothesis.top_cells,
        )
        field_oracles[(domain, specimen)] = oracle
        trajectory_rows.extend(_oracle_trajectory_rows(domain, specimen, oracle))
        projected = project_oracle_checkpoints(oracle, protocol.checkpoints)
        oracle_auebc = zero_inclusive_auebc(
            protocol.checkpoints,
            projected.task_losses,
        )
        effect = fixed.field_auebc - oracle_auebc
        field_effects.append(effect)
        fixed_auebcs.append(fixed.field_auebc)
        reference_method = references[(domain, "FIELD")]
        reference_row = next(
            row
            for row in endpoint_rows
            if row["outer_domain"] == domain
            and row["dataset_id"] == domain
            and row["specimen_id"] == specimen
            and row["method"] == reference_method
        )
        budgets = np.asarray(
            [0.0, *(step.budget_after for step in oracle.steps)],
            dtype=np.float64,
        )
        losses = np.asarray(
            [oracle.steps[0].task_loss_before, *(step.task_loss_after for step in oracle.steps)],
            dtype=np.float64,
        )
        stopping = earliest_sufficient_state(
            budgets=budgets,
            losses=losses,
            reference_budget=protocol.endpoint_budget,
            reference_loss=float(reference_row["field_loss"]),
            tolerance=0.05,
        )
        stopping_effects.append(stopping.normalized_measurement_saving)
        final_stop_losses.append(stopping.final_task_loss)
        reference_losses.append(stopping.reference_loss)
        stopping_rows.append(
            {
                "outer_domain": domain,
                "dataset_id": domain,
                "specimen_id": specimen,
                "task": "FIELD",
                "method": "ORACLE_FIELD",
                "reference_method": reference_method,
                "reached": stopping.reached,
                "stop_index": stopping.stop_index,
                "budget_to_sufficiency": stopping.budget_to_sufficiency,
                "normalized_measurement_saving": stopping.normalized_measurement_saving,
                "final_task_loss": stopping.final_task_loss,
                "reference_budget": stopping.reference_budget,
                "reference_loss": stopping.reference_loss,
                "threshold_loss": stopping.threshold_loss,
            }
        )
        if index % 10 == 0 or index == len(roster):
            _progress(progress, f"G0-B FIELD oracle: {index}/{len(roster)}")
    dataset_ids = tuple(domain for domain, _specimen in roster)
    specimen_ids = tuple(specimen for _domain, specimen in roster)
    field_bootstrap = synchronized_paired_bootstrap(
        dataset_ids=dataset_ids,
        specimen_ids=specimen_ids,
        effects=np.asarray(field_effects),
        replicates=protocol.bootstrap_replicates,
        seed=protocol.bootstrap_seed,
    )
    stopping_bootstrap = synchronized_paired_bootstrap(
        dataset_ids=dataset_ids,
        specimen_ids=specimen_ids,
        effects=np.asarray(stopping_effects),
        replicates=protocol.bootstrap_replicates,
        seed=protocol.bootstrap_seed,
    )
    fixed_mean = _equal_domain_mean(dataset_ids, fixed_auebcs)
    relative = (
        0.0 if fixed_mean == 0.0 else field_bootstrap.point_estimate / fixed_mean
    )
    mean_saving = _equal_domain_mean(dataset_ids, stopping_effects)
    mean_stop_loss = _equal_domain_mean(dataset_ids, final_stop_losses)
    mean_reference_loss = _equal_domain_mean(dataset_ids, reference_losses)
    loss_ratio = (
        math.inf if mean_reference_loss == 0.0 and mean_stop_loss > 0.0
        else 1.0 if mean_reference_loss == 0.0
        else mean_stop_loss / mean_reference_loss
    )
    hierarchical_evidence = GateEvidence(
        field_bootstrap.point_estimate,
        field_bootstrap.ci_lower,
        field_bootstrap.ci_upper,
        field_bootstrap.improved_domains,
    )
    stopping_evidence = GateEvidence(
        stopping_bootstrap.point_estimate,
        stopping_bootstrap.ci_lower,
        stopping_bootstrap.ci_upper,
        stopping_bootstrap.improved_domains,
    )
    return _HierarchyAudit(
        endpoint_rows=endpoint_rows,
        trajectory_rows=tuple(trajectory_rows),
        stopping_rows=tuple(stopping_rows),
        field_fixed_curves=MappingProxyType(fixed_curves),
        historical_mavis_curves=MappingProxyType(historical_curves),
        historical_mavis_sha256=MappingProxyType(historical_hashes),
        field_oracles=MappingProxyType(field_oracles),
        field_bootstrap=field_bootstrap,
        field_relative_auebc_improvement=relative,
        field_mean_stopping_saving=mean_saving,
        field_stopping_loss_ratio=loss_ratio,
        field_headroom=hierarchical_headroom_gate(
            hierarchical_evidence,
            relative_auebc_improvement=relative,
            sufficiency_budget_reduction=mean_saving,
        ),
        field_stopping_bootstrap=stopping_bootstrap,
        field_stopping_headroom=stopping_headroom_gate(
            stopping_evidence,
            mean_budget_saving=mean_saving,
            task_loss_ratio=loss_ratio,
        ),
        reference_methods=references,
    )


def _oracle_checkpoint_observations(
    runtime: _RuntimeAuthority,
    datum: _SurfaceDatum,
    grid: AcquisitionGrid,
    trajectory: OracleTrajectory,
    checkpoints: tuple[float, ...],
) -> tuple[InspectionObservation, ...]:
    actions = tuple(step.action for step in trajectory.steps)
    world = _world(
        runtime,
        datum,
        grid=grid,
        task=trajectory.task,
        endpoint_budget=float(checkpoints[-1]),
    )
    output = []
    for checkpoint in checkpoints:
        prefix = sum(
            step.budget_after <= checkpoint + 1.0e-15
            for step in trajectory.steps
        )
        output.append(world.replay(actions[:prefix]))
    return tuple(output)


def _conditional_cai_audit(
    runtime: _RuntimeAuthority,
    protocol: G0Protocol,
    assessor_audit: _AssessorAudit,
    hierarchy: _HierarchyAudit,
    *,
    encoder_project_root: Path,
    device: str,
    progress: Callable[[str], None] | None,
) -> _CAIHierarchyAudit:
    if not assessor_audit.authorized:
        return _CAIHierarchyAudit(
            status="NOT_RUN_NOT_AUTHORIZED",
            trajectory_rows=(),
            stopping_rows=(),
            task_swap_rows=(
                {
                    "status": "NOT_RUN_NOT_AUTHORIZED",
                    "reason": "CAI_ASSESSOR_GATE_FAILED",
                },
            ),
            cai_bootstrap=None,
            cai_relative_auebc_improvement=None,
            cai_mean_stopping_saving=None,
            cai_stopping_loss_ratio=None,
            cai_headroom=False,
            cai_stopping_bootstrap=None,
            cai_stopping_headroom=False,
            field_swap_bootstrap=None,
            cai_swap_bootstrap=None,
            task_conditioning_headroom=False,
        )
    encoder = _registered_encoder(encoder_project_root, device)
    roster = [
        (domain, specimen)
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if (domain, specimen) in runtime.surfaces
    ]
    trajectory_rows: list[dict[str, object]] = []
    stopping_rows: list[dict[str, object]] = []
    swap_rows: list[dict[str, object]] = []
    cai_effects: list[float] = []
    fixed_cai_auebcs: list[float] = []
    stopping_effects: list[float] = []
    stopping_final_losses: list[float] = []
    stopping_reference_losses: list[float] = []
    field_swap_effects: list[float] = []
    cai_swap_effects: list[float] = []
    for index, (domain, specimen) in enumerate(roster, start=1):
        datum = runtime.surfaces[(domain, specimen)]
        view = runtime.mavis.evaluation_view(specimen)
        grid = _grid(protocol, domain, view.full_scan.shape[:2])
        assessor = assessor_audit.assessors[domain]
        cai_oracle = run_cai_oracle(
            _world(
                runtime,
                datum,
                grid=grid,
                task=InspectionTask.CAI,
                endpoint_budget=protocol.endpoint_budget,
            ),
            grid,
            assessor_audit.priors[domain],
            full_scan=view.full_scan,
            true_cai=view.true_cai,
            assessor=assessor,
            encoder=encoder,
            surface_hypothesis_cells=datum.hypothesis.top_cells,
        )
        trajectory_rows.extend(_oracle_trajectory_rows(domain, specimen, cai_oracle))
        projected_cai = project_oracle_checkpoints(cai_oracle, protocol.checkpoints)
        cai_auebc = zero_inclusive_auebc(
            protocol.checkpoints,
            projected_cai.task_losses,
        )
        fixed = hierarchy.field_fixed_curves[(domain, specimen)]
        if fixed.cai_auebc is None or fixed.cai_losses is None:
            raise G0ExecutionError("authorized CAI fixed curve is absent")
        cai_effects.append(fixed.cai_auebc - cai_auebc)
        fixed_cai_auebcs.append(fixed.cai_auebc)
        reference_method = hierarchy.reference_methods[(domain, "CAI")]
        reference_row = next(
            row
            for row in hierarchy.endpoint_rows
            if row["outer_domain"] == domain
            and row["dataset_id"] == domain
            and row["specimen_id"] == specimen
            and row["method"] == reference_method
        )
        budgets = np.asarray(
            [0.0, *(step.budget_after for step in cai_oracle.steps)],
            dtype=np.float64,
        )
        losses = np.asarray(
            [
                cai_oracle.steps[0].task_loss_before,
                *(step.task_loss_after for step in cai_oracle.steps),
            ],
            dtype=np.float64,
        )
        stopping = earliest_sufficient_state(
            budgets=budgets,
            losses=losses,
            reference_budget=protocol.endpoint_budget,
            reference_loss=float(reference_row["cai_loss"]),
            tolerance=0.05,
        )
        stopping_effects.append(stopping.normalized_measurement_saving)
        stopping_final_losses.append(stopping.final_task_loss)
        stopping_reference_losses.append(stopping.reference_loss)
        stopping_rows.append(
            {
                "outer_domain": domain,
                "dataset_id": domain,
                "specimen_id": specimen,
                "task": "CAI",
                "method": "ORACLE_CAI",
                "reference_method": reference_method,
                "reached": stopping.reached,
                "stop_index": stopping.stop_index,
                "budget_to_sufficiency": stopping.budget_to_sufficiency,
                "normalized_measurement_saving": stopping.normalized_measurement_saving,
                "final_task_loss": stopping.final_task_loss,
                "reference_budget": stopping.reference_budget,
                "reference_loss": stopping.reference_loss,
                "threshold_loss": stopping.threshold_loss,
            }
        )

        field_oracle = hierarchy.field_oracles[(domain, specimen)]
        projected_field = project_oracle_checkpoints(
            field_oracle,
            protocol.checkpoints,
        )
        cai_observations = _oracle_checkpoint_observations(
            runtime,
            datum,
            grid,
            cai_oracle,
            protocol.checkpoints,
        )
        field_observations = _oracle_checkpoint_observations(
            runtime,
            datum,
            grid,
            field_oracle,
            protocol.checkpoints,
        )
        cai_images = tuple(
            reconstruct_observation(
                observation,
                grid,
                assessor_audit.priors[domain],
            ).image
            for observation in cai_observations
        )
        field_images = tuple(
            reconstruct_observation(
                observation,
                grid,
                assessor_audit.priors[domain],
            ).image
            for observation in field_observations
        )
        cai_on_field = np.asarray(
            [field_loss(view.full_scan, image) for image in cai_images],
            dtype=np.float64,
        )
        field_embeddings = encoder.encode(field_images).astype(np.float64, copy=False)
        field_on_cai = np.abs(
            assessor.predict(
                field_embeddings,
                np.asarray([state_scalars(value) for value in field_observations]),
            )
            - view.true_cai
        )
        advantages = task_swap_advantages(
            field_on_field=projected_field.task_losses,
            cai_on_field=cai_on_field,
            cai_on_cai=projected_cai.task_losses,
            field_on_cai=field_on_cai,
        )
        field_swap_effect = zero_inclusive_auebc(
            protocol.checkpoints,
            advantages.field_advantage,
        )
        cai_swap_effect = zero_inclusive_auebc(
            protocol.checkpoints,
            advantages.cai_advantage,
        )
        field_swap_effects.append(field_swap_effect)
        cai_swap_effects.append(cai_swap_effect)
        overlap = trajectory_overlap(
            tuple(step.action for step in field_oracle.steps),
            tuple(step.action for step in cai_oracle.steps),
            field_decisions=tuple(step.decision for step in field_oracle.steps),
            cai_decisions=tuple(step.decision for step in cai_oracle.steps),
        )
        for checkpoint_index, checkpoint in enumerate(protocol.checkpoints):
            swap_rows.append(
                {
                    "status": "RUN_AUTHORIZED",
                    "outer_domain": domain,
                    "dataset_id": domain,
                    "specimen_id": specimen,
                    "nominal_budget": checkpoint,
                    "field_on_field_loss": projected_field.task_losses[
                        checkpoint_index
                    ],
                    "cai_on_field_loss": cai_on_field[checkpoint_index],
                    "field_advantage": advantages.field_advantage[checkpoint_index],
                    "cai_on_cai_loss": projected_cai.task_losses[checkpoint_index],
                    "field_on_cai_loss": field_on_cai[checkpoint_index],
                    "cai_advantage": advantages.cai_advantage[checkpoint_index],
                    "field_advantage_auebc": field_swap_effect,
                    "cai_advantage_auebc": cai_swap_effect,
                    "action_jaccard": overlap.action_jaccard,
                    "cell_jaccard": overlap.cell_jaccard,
                    "high_level_action_overlap": overlap.high_level_action_overlap,
                    "normalized_edit_distance": overlap.normalized_edit_distance,
                }
            )
        if index % 5 == 0 or index == len(roster):
            _progress(progress, f"G0-B/C/D CAI oracle: {index}/{len(roster)}")
    dataset_ids = tuple(domain for domain, _specimen in roster)
    specimen_ids = tuple(specimen for _domain, specimen in roster)

    def bootstrap(values: list[float]) -> PairedBootstrapSummary:
        return synchronized_paired_bootstrap(
            dataset_ids=dataset_ids,
            specimen_ids=specimen_ids,
            effects=np.asarray(values, dtype=np.float64),
            replicates=protocol.bootstrap_replicates,
            seed=protocol.bootstrap_seed,
        )

    cai_bootstrap = bootstrap(cai_effects)
    stopping_bootstrap = bootstrap(stopping_effects)
    field_swap_bootstrap = bootstrap(field_swap_effects)
    cai_swap_bootstrap = bootstrap(cai_swap_effects)
    fixed_mean = _equal_domain_mean(dataset_ids, fixed_cai_auebcs)
    relative = 0.0 if fixed_mean == 0.0 else cai_bootstrap.point_estimate / fixed_mean
    mean_saving = _equal_domain_mean(dataset_ids, stopping_effects)
    mean_final = _equal_domain_mean(dataset_ids, stopping_final_losses)
    mean_reference = _equal_domain_mean(dataset_ids, stopping_reference_losses)
    loss_ratio = (
        math.inf if mean_reference == 0.0 and mean_final > 0.0
        else 1.0 if mean_reference == 0.0
        else mean_final / mean_reference
    )
    cai_evidence = GateEvidence(
        cai_bootstrap.point_estimate,
        cai_bootstrap.ci_lower,
        cai_bootstrap.ci_upper,
        cai_bootstrap.improved_domains,
    )
    stopping_evidence = GateEvidence(
        stopping_bootstrap.point_estimate,
        stopping_bootstrap.ci_lower,
        stopping_bootstrap.ci_upper,
        stopping_bootstrap.improved_domains,
    )
    field_swap_evidence = GateEvidence(
        field_swap_bootstrap.point_estimate,
        field_swap_bootstrap.ci_lower,
        field_swap_bootstrap.ci_upper,
        field_swap_bootstrap.improved_domains,
    )
    cai_swap_evidence = GateEvidence(
        cai_swap_bootstrap.point_estimate,
        cai_swap_bootstrap.ci_lower,
        cai_swap_bootstrap.ci_upper,
        cai_swap_bootstrap.improved_domains,
    )
    return _CAIHierarchyAudit(
        status="RUN_AUTHORIZED",
        trajectory_rows=tuple(trajectory_rows),
        stopping_rows=tuple(stopping_rows),
        task_swap_rows=tuple(swap_rows),
        cai_bootstrap=cai_bootstrap,
        cai_relative_auebc_improvement=relative,
        cai_mean_stopping_saving=mean_saving,
        cai_stopping_loss_ratio=loss_ratio,
        cai_headroom=hierarchical_headroom_gate(
            cai_evidence,
            relative_auebc_improvement=relative,
            sufficiency_budget_reduction=mean_saving,
        ),
        cai_stopping_bootstrap=stopping_bootstrap,
        cai_stopping_headroom=stopping_headroom_gate(
            stopping_evidence,
            mean_budget_saving=mean_saving,
            task_loss_ratio=loss_ratio,
        ),
        field_swap_bootstrap=field_swap_bootstrap,
        cai_swap_bootstrap=cai_swap_bootstrap,
        task_conditioning_headroom=task_conditioning_gate(
            field_swap_evidence,
            cai_swap_evidence,
        ),
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    values = tuple(dict(row) for row in rows)
    if not values:
        raise G0ExecutionError(f"G0 CSV has no rows: {path.name}")
    fieldnames = tuple(sorted({key for row in values for key in row}))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(values)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )


def _bootstrap_row(
    analysis: str,
    task: str,
    summary: PairedBootstrapSummary,
    *,
    effect_definition: str,
) -> dict[str, object]:
    return {
        "analysis": analysis,
        "task": task,
        "effect_definition": effect_definition,
        "point_estimate": summary.point_estimate,
        "ci_lower": summary.ci_lower,
        "ci_upper": summary.ci_upper,
        "improved_domains": summary.improved_domains,
        "replicates": summary.replicates,
        "seed": summary.seed,
        "distribution_sha256": summary.distribution_sha256,
    }


def _domain_rows(
    analysis: str,
    task: str,
    summary: PairedBootstrapSummary,
) -> list[dict[str, object]]:
    return [
        {
            "analysis": analysis,
            "task": task,
            "dataset_id": domain,
            "effect": effect,
        }
        for domain, effect in summary.domain_effects
    ]


def _authorized_roster_rows(
    runtime: _RuntimeAuthority,
    protocol: G0Protocol,
) -> tuple[dict[str, object], ...]:
    rows = []
    for index, (specimen, domain, source_sha, decoded_sha) in enumerate(
        zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            runtime.mavis.source_image_sha256,
            runtime.mavis.decoded_image_sha256,
            strict=True,
        )
    ):
        datum = runtime.surfaces[(domain, specimen)]
        context = runtime.mavis.policy_context(specimen)
        rows.append(
            {
                "roster_index": index,
                "dataset_id": domain,
                "specimen_id": specimen,
                "native_height": context.native_shape[0],
                "native_width": context.native_shape[1],
                "initial_nominal_budget": protocol.initial_budget_by_domain[domain],
                "cscan_source_sha256": source_sha,
                "cscan_decoded_sha256": decoded_sha,
                "surface_path": datum.record.surface_path.as_posix(),
                "surface_sha256": datum.record.surface_sha256,
                "surface_transform_sha256": datum.record.transform_sha256,
                "surface_hypothesis_sha256": datum.hypothesis.state_sha256,
                "surface_top_cells": ";".join(
                    str(cell) for cell in datum.hypothesis.top_cells
                ),
                "mavis_authority_sha256": runtime.mavis.state_sha256,
                "surface_authority_sha256": runtime.surface_authority_sha256,
                "authorization_status": "AUTHORIZED",
            }
        )
    return tuple(rows)


def _decision_payload(
    protocol: G0Protocol,
    assessor: _AssessorAudit,
    initialization: _InitializationAudit,
    hierarchy: _HierarchyAudit,
    cai: _CAIHierarchyAudit,
) -> dict[str, object]:
    historical_keys = tuple(sorted(hierarchy.historical_mavis_curves))
    historical_domains = tuple(domain for domain, _specimen in historical_keys)
    historical_field_auebc = _equal_domain_mean(
        historical_domains,
        (
            hierarchy.historical_mavis_curves[key].field_auebc
            for key in historical_keys
        ),
    )
    historical_cai_values = tuple(
        hierarchy.historical_mavis_curves[key].cai_auebc
        for key in historical_keys
    )
    historical_cai_auebc = (
        None
        if any(value is None for value in historical_cai_values)
        else _equal_domain_mean(
            historical_domains,
            (float(value) for value in historical_cai_values),
        )
    )
    evidence = FinalG0Evidence(
        initialization_headroom=initialization.headroom,
        field_hierarchical_headroom=hierarchy.field_headroom,
        cai_assessor_authorized=assessor.authorized,
        cai_hierarchical_headroom=cai.cai_headroom,
        task_conditioning_headroom=cai.task_conditioning_headroom,
        field_stopping_headroom=hierarchy.field_stopping_headroom,
        cai_stopping_headroom=cai.cai_stopping_headroom,
    )
    status = decide_g0_status(evidence)
    return {
        "schema_version": 1,
        "status": status.value,
        "repository_base_sha": _BASE_SHA,
        "config_sha256": protocol.config_sha256,
        "specimen_count": protocol.specimen_count,
        "domain_count": len(protocol.domain_order),
        "cai_conditional_status": cai.status,
        "component_gates": {
            "initialization_headroom": initialization.headroom,
            "field_hierarchical_headroom": hierarchy.field_headroom,
            "cai_assessor_authorized": assessor.authorized,
            "cai_hierarchical_headroom": cai.cai_headroom,
            "task_conditioning_headroom": cai.task_conditioning_headroom,
            "field_stopping_headroom": hierarchy.field_stopping_headroom,
            "cai_stopping_headroom": cai.cai_stopping_headroom,
        },
        "assessor": {
            "zero_equal_domain_mae": assessor.zero_mae,
            "endpoint_equal_domain_mae": assessor.endpoint_mae,
            "improvement": assessor.bootstrap.point_estimate,
            "ci_lower": assessor.bootstrap.ci_lower,
            "ci_upper": assessor.bootstrap.ci_upper,
            "improved_domains": assessor.bootstrap.improved_domains,
            "replay_valid": assessor.replay_valid,
            "outer_exclusion_valid": assessor.outer_exclusion_valid,
        },
        "initialization": {
            "oracle_minus_uniform_auc": initialization.bootstrap.point_estimate,
            "ci_lower": initialization.bootstrap.ci_lower,
            "ci_upper": initialization.bootstrap.ci_upper,
            "improved_domains": initialization.bootstrap.improved_domains,
            "relative_auc_improvement": initialization.relative_auc_improvement,
            "capture_budget_reduction": initialization.capture_budget_reduction,
        },
        "field_hierarchy": {
            "fixed_minus_oracle_auebc": hierarchy.field_bootstrap.point_estimate,
            "ci_lower": hierarchy.field_bootstrap.ci_lower,
            "ci_upper": hierarchy.field_bootstrap.ci_upper,
            "improved_domains": hierarchy.field_bootstrap.improved_domains,
            "relative_auebc_improvement": (
                hierarchy.field_relative_auebc_improvement
            ),
        },
        "field_stopping": {
            "mean_normalized_measurement_saving": (
                hierarchy.field_mean_stopping_saving
            ),
            "ci_lower": hierarchy.field_stopping_bootstrap.ci_lower,
            "ci_upper": hierarchy.field_stopping_bootstrap.ci_upper,
            "improved_domains": hierarchy.field_stopping_bootstrap.improved_domains,
            "task_loss_ratio": hierarchy.field_stopping_loss_ratio,
        },
        "historical_upper_bound": {
            "method": "FIXED_UNIFORM_THEN_MAVIS",
            "privilege": "METADATA_AUGMENTED_UPPER_BOUND",
            "gate_eligible": False,
            "field_equal_domain_auebc": historical_field_auebc,
            "cai_equal_domain_auebc": historical_cai_auebc,
        },
        "cai_hierarchy": (
            "NOT_RUN_NOT_AUTHORIZED"
            if cai.cai_bootstrap is None
            else {
                "fixed_minus_oracle_auebc": cai.cai_bootstrap.point_estimate,
                "ci_lower": cai.cai_bootstrap.ci_lower,
                "ci_upper": cai.cai_bootstrap.ci_upper,
                "improved_domains": cai.cai_bootstrap.improved_domains,
                "relative_auebc_improvement": cai.cai_relative_auebc_improvement,
            }
        ),
        "task_conditioning": (
            "NOT_RUN_NOT_AUTHORIZED"
            if cai.field_swap_bootstrap is None or cai.cai_swap_bootstrap is None
            else {
                "field_advantage": {
                    "point_estimate": cai.field_swap_bootstrap.point_estimate,
                    "ci_lower": cai.field_swap_bootstrap.ci_lower,
                    "ci_upper": cai.field_swap_bootstrap.ci_upper,
                    "improved_domains": cai.field_swap_bootstrap.improved_domains,
                },
                "cai_advantage": {
                    "point_estimate": cai.cai_swap_bootstrap.point_estimate,
                    "ci_lower": cai.cai_swap_bootstrap.ci_lower,
                    "ci_upper": cai.cai_swap_bootstrap.ci_upper,
                    "improved_domains": cai.cai_swap_bootstrap.improved_domains,
                },
            }
        ),
        "no_result_rescue": True,
        "new_planner_training": False,
        "scanner_time_claim": False,
    }


def _report(payload: Mapping[str, object]) -> str:
    components = payload["component_gates"]
    assessor = payload["assessor"]
    initialization = payload["initialization"]
    field = payload["field_hierarchy"]
    stopping = payload["field_stopping"]
    historical = payload["historical_upper_bound"]
    return "\n".join(
        (
            "# Inspection Agent G0 Opportunity Audit",
            "",
            f"Status: `{payload['status']}`",
            "",
            "This package evaluates privileged opportunity from a strict zero-ultrasound state. ",
            "It does not contain a learned planner and does not claim deployment readiness.",
            "",
            "## CAI assessor gate",
            "",
            f"- Zero-state equal-domain MAE: {assessor['zero_equal_domain_mae']:.12g}",
            f"- 25% equal-domain MAE: {assessor['endpoint_equal_domain_mae']:.12g}",
            f"- Paired improvement 95% CI: [{assessor['ci_lower']:.12g}, {assessor['ci_upper']:.12g}]",
            f"- Authorized: `{components['cai_assessor_authorized']}`",
            "",
            "## Initialization",
            "",
            f"- Oracle-minus-uniform capture AUC: {initialization['oracle_minus_uniform_auc']:.12g}",
            f"- Relative improvement: {initialization['relative_auc_improvement']:.6%}",
            f"- Gate: `{components['initialization_headroom']}`",
            "",
            "## Hierarchical FIELD allocation",
            "",
            f"- Fixed-minus-oracle AUEBC: {field['fixed_minus_oracle_auebc']:.12g}",
            f"- Relative improvement: {field['relative_auebc_improvement']:.6%}",
            f"- Gate: `{components['field_hierarchical_headroom']}`",
            "",
            "## Historical MAVIS upper bound",
            "",
            f"- FIELD AUEBC: {historical['field_equal_domain_auebc']:.12g}",
            "- Privilege: `METADATA_AUGMENTED_UPPER_BOUND`",
            "- Gate eligible: `False`",
            "",
            "## FIELD stopping",
            "",
            f"- Mean normalized measurement saving: {stopping['mean_normalized_measurement_saving']:.6%}",
            f"- Task-loss ratio: {stopping['task_loss_ratio']:.12g}",
            f"- Gate: `{components['field_stopping_headroom']}`",
            "",
            "## Conditional CAI route",
            "",
            f"Status: `{payload['cai_conditional_status']}`",
            "",
            "All contrasts use synchronized specimen-within-domain bootstrap and equal-domain aggregation.",
            "Full C-scans and true CAI are evaluation/teacher privilege only.",
            "",
        )
    )


def _materialize_package(
    output: Path,
    *,
    project_root: Path,
    protocol: G0Protocol,
    runtime: _RuntimeAuthority,
    zero_rows: tuple[dict[str, object], ...],
    assessor: _AssessorAudit,
    initialization: _InitializationAudit,
    hierarchy: _HierarchyAudit,
    cai: _CAIHierarchyAudit,
) -> tuple[str, G0PackageValidation]:
    if output.exists():
        raise G0ExecutionError("G0 output already exists; refusing to overwrite")
    expected_parent = (project_root / "results/inspection_agent").resolve()
    output_parent = output.parent.resolve()
    if output_parent != expected_parent and output_parent != (
        expected_parent / "replay"
    ):
        raise G0ExecutionError("G0 output is outside the authorized result roots")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    try:
        shutil.copyfile(protocol.config_path, temporary / "config.yaml")
        _write_csv(
            temporary / "authorized_roster.csv",
            _authorized_roster_rows(runtime, protocol),
        )
        _write_csv(temporary / "zero_state_audit.csv", zero_rows)
        _write_csv(temporary / "cai_assessor_metrics.csv", assessor.metric_rows)
        _write_csv(temporary / "state_bank_manifest.csv", assessor.state_bank_rows)
        _write_csv(temporary / "initialization_curves.csv", initialization.rows)
        trajectory_rows = (*hierarchy.trajectory_rows, *cai.trajectory_rows)
        frame = pl.DataFrame(trajectory_rows).sort(
            ["task", "dataset_id", "specimen_id", "method", "step"]
        )
        frame.write_parquet(
            temporary / "hierarchical_trajectories.parquet",
            compression="zstd",
            statistics=False,
            row_group_size=8192,
        )
        _write_csv(temporary / "task_swap.csv", cai.task_swap_rows)
        _write_csv(
            temporary / "stopping_results.csv",
            (*hierarchy.stopping_rows, *cai.stopping_rows),
        )
        bootstrap_rows = [
            _bootstrap_row(
                "CAI_ASSESSOR",
                "CAI",
                assessor.bootstrap,
                effect_definition="zero_absolute_error_minus_endpoint_absolute_error",
            ),
            _bootstrap_row(
                "INITIALIZATION",
                "DISCOVERY",
                initialization.bootstrap,
                effect_definition="oracle_capture_auc_minus_zero_uniform_capture_auc",
            ),
            _bootstrap_row(
                "HIERARCHICAL",
                "FIELD",
                hierarchy.field_bootstrap,
                effect_definition="fixed_auebc_minus_oracle_auebc",
            ),
            _bootstrap_row(
                "STOPPING",
                "FIELD",
                hierarchy.field_stopping_bootstrap,
                effect_definition="normalized_measurement_saving",
            ),
        ]
        domain_rows = [
            *_domain_rows("CAI_ASSESSOR", "CAI", assessor.bootstrap),
            *_domain_rows("INITIALIZATION", "DISCOVERY", initialization.bootstrap),
            *_domain_rows("HIERARCHICAL", "FIELD", hierarchy.field_bootstrap),
            *_domain_rows("STOPPING", "FIELD", hierarchy.field_stopping_bootstrap),
        ]
        for domain in protocol.domain_order:
            for method in _FIXED_REFERENCE_METHODS:
                matching = tuple(
                    row
                    for row in hierarchy.endpoint_rows
                    if row["outer_domain"] == domain
                    and row["dataset_id"] == domain
                    and row["method"] == method
                )
                for task, column in (("FIELD", "field_loss"), ("CAI", "cai_loss")):
                    values = tuple(row[column] for row in matching)
                    if not values or any(value is None for value in values):
                        continue
                    domain_rows.append(
                        {
                            "analysis": "FIXED_ENDPOINT_BASELINE",
                            "task": task,
                            "dataset_id": domain,
                            "effect": float(
                                np.mean(
                                    [float(value) for value in values],
                                    dtype=np.float64,
                                )
                            ),
                            "selected_method": method,
                            "gate_eligible": True,
                        }
                    )
        for domain in protocol.domain_order:
            domain_curves = tuple(
                curve
                for (row_domain, _specimen), curve in sorted(
                    hierarchy.historical_mavis_curves.items()
                )
                if row_domain == domain
            )
            domain_rows.append(
                {
                    "analysis": "HISTORICAL_UPPER_BOUND",
                    "task": "FIELD",
                    "dataset_id": domain,
                    "effect": float(
                        np.mean(
                            [curve.field_auebc for curve in domain_curves],
                            dtype=np.float64,
                        )
                    ),
                    "selected_method": "FIXED_UNIFORM_THEN_MAVIS",
                    "gate_eligible": False,
                }
            )
            if all(curve.cai_auebc is not None for curve in domain_curves):
                domain_rows.append(
                    {
                        "analysis": "HISTORICAL_UPPER_BOUND",
                        "task": "CAI",
                        "dataset_id": domain,
                        "effect": float(
                            np.mean(
                                [
                                    float(curve.cai_auebc)
                                    for curve in domain_curves
                                ],
                                dtype=np.float64,
                            )
                        ),
                        "selected_method": "FIXED_UNIFORM_THEN_MAVIS",
                        "gate_eligible": False,
                    }
                )
        for analysis, task, summary, definition in (
            ("HIERARCHICAL", "CAI", cai.cai_bootstrap, "fixed_auebc_minus_oracle_auebc"),
            ("STOPPING", "CAI", cai.cai_stopping_bootstrap, "normalized_measurement_saving"),
            ("TASK_SWAP", "FIELD", cai.field_swap_bootstrap, "wrong_minus_correct_auebc"),
            ("TASK_SWAP", "CAI", cai.cai_swap_bootstrap, "wrong_minus_correct_auebc"),
        ):
            if summary is not None:
                bootstrap_rows.append(
                    _bootstrap_row(
                        analysis,
                        task,
                        summary,
                        effect_definition=definition,
                    )
                )
                domain_rows.extend(_domain_rows(analysis, task, summary))
        for (outer, task), method in sorted(hierarchy.reference_methods.items()):
            domain_rows.append(
                {
                    "analysis": "SOURCE_REFERENCE_SELECTION",
                    "task": task,
                    "dataset_id": outer,
                    "effect": None,
                    "selected_method": method,
                }
            )
        _write_csv(temporary / "domain_metrics.csv", domain_rows)
        _write_csv(temporary / "bootstrap.csv", bootstrap_rows)
        decision = _decision_payload(
            protocol,
            assessor,
            initialization,
            hierarchy,
            cai,
        )
        _write_json(temporary / "decision_summary.json", decision)
        (temporary / "REPORT.md").write_text(_report(decision), encoding="ascii")
        package = publish_g0_manifest(
            temporary,
            project_root=project_root,
            config_path=protocol.config_path,
        )
        os.replace(temporary, output)
        return str(decision["status"]), package
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def run_g0(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_project_root: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> G0RunResult:
    root = Path(project_root).resolve(strict=True)
    source_root = Path(source_project_root).resolve(strict=True)
    protocol = load_g0_protocol(config_path, project_root=root)
    _verify_controlling_prompt(source_root)
    runtime = _load_runtime_authority(
        protocol,
        project_root=root,
        source_project_root=source_root,
        progress=progress,
    )
    if len(runtime.surfaces) != protocol.specimen_count:
        raise G0ExecutionError("formal G0 runtime does not contain 276 surfaces")
    zero_rows = _zero_state_rows(runtime, protocol)
    _progress(progress, "zero-state audit complete")
    assessor = _build_assessor_audit(
        runtime,
        protocol,
        encoder_project_root=source_root,
        device=device or protocol.default_device,
        progress=progress,
    )
    _progress(
        progress,
        f"CAI assessor authorization: {assessor.authorized}",
    )
    initialization = _initialization_audit(runtime, protocol, progress=progress)
    hierarchy = _hierarchy_audit(
        runtime,
        protocol,
        assessor,
        encoder_project_root=source_root,
        device=device or protocol.default_device,
        progress=progress,
    )
    cai = _conditional_cai_audit(
        runtime,
        protocol,
        assessor,
        hierarchy,
        encoder_project_root=source_root,
        device=device or protocol.default_device,
        progress=progress,
    )
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    status, package = _materialize_package(
        output,
        project_root=root,
        protocol=protocol,
        runtime=runtime,
        zero_rows=zero_rows,
        assessor=assessor,
        initialization=initialization,
        hierarchy=hierarchy,
        cai=cai,
    )
    return G0RunResult(
        output_dir=output,
        status=status,
        package=package,
        cai_assessor_authorized=assessor.authorized,
        specimen_count=protocol.specimen_count,
    )


__all__ = [
    "FinalG0Evidence",
    "G0ExecutionError",
    "G0Protocol",
    "G0RunResult",
    "G0Status",
    "GateEvidence",
    "assessor_authorization_gate",
    "decide_g0_status",
    "hierarchical_headroom_gate",
    "initialization_headroom_gate",
    "load_g0_protocol",
    "plan_staged_actions",
    "run_g0",
    "smoke_g0_assessor",
    "stopping_headroom_gate",
    "task_conditioning_gate",
]
