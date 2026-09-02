"""Source-only learned-policy curves and inner engineering evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .batch_rollout import G1BatchRolloutRequest, run_g1_closed_loop_batch
from .formal import G1ObservableStateBuilder, evaluate_g1_action_history
from .g1 import (
    G1Protocol,
    G1Runtime,
    G1SourceDependencies,
    build_g1_world,
)
from .metrics import (
    EngineeringCurve,
    G1MetricError,
    replay_engineering_curve,
    validate_engineering_curve,
)
from .policy_selection import InnerPolicyEngineeringMetric
from .policy_training import (
    REGISTERED_MAX_EPOCHS,
    PolicyTrainingHyperparameters,
)
from .rollout import ClosedLoopTrajectory, run_closed_loop
from .source_bridge import (
    SOURCE_ORACLE_METHODS,
    G1SourceBridgeRecord,
    SourceFixedBridgeSelection,
    select_source_fixed_bridge,
)
from .teacher import (
    SourceTeacherAuthorization,
    validate_source_teacher_dependencies,
)

LEARNED_POLICY_METHOD = "LEARNED_POLICY"


class G1SourcePolicyEvaluationError(ValueError):
    """Raised when learned source evidence is incomplete or not fold-safe."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class G1LearnedSourceRecord:
    outer_target: str
    source_domain: str
    specimen_id: str
    fit_domains: tuple[str, ...]
    dependency_sha256: str
    hyperparameters_sha256: str
    model_state_sha256: str
    fit_audit_sha256: str
    selected_epoch: int
    trajectory_sha256: str
    action_history_sha256: str
    curve: EngineeringCurve
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            validate_engineering_curve(self.curve)
        except G1MetricError as error:
            raise G1SourcePolicyEvaluationError(
                "learned source curve identity changed"
            ) from error
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or self.source_domain == self.outer_target
            or type(self.specimen_id) is not str
            or not self.specimen_id
            or type(self.fit_domains) is not tuple
            or len(self.fit_domains) != 4
            or len(set(self.fit_domains)) != 4
            or self.outer_target in self.fit_domains
            or self.source_domain in self.fit_domains
            or not all(
                _valid_sha256(value)
                for value in (
                    self.dependency_sha256,
                    self.hyperparameters_sha256,
                    self.model_state_sha256,
                    self.fit_audit_sha256,
                    self.trajectory_sha256,
                    self.action_history_sha256,
                )
            )
            or type(self.selected_epoch) is not int
            or not 1 <= self.selected_epoch <= REGISTERED_MAX_EPOCHS
            or self.curve.method != LEARNED_POLICY_METHOD
            or self.curve.target_domain != self.source_domain
        ):
            raise G1SourcePolicyEvaluationError("learned source record is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-learned-source-record",
                    "outer_target": self.outer_target,
                    "source_domain": self.source_domain,
                    "specimen_id": self.specimen_id,
                    "fit_domains": self.fit_domains,
                    "dependency": self.dependency_sha256,
                    "hyperparameters": self.hyperparameters_sha256,
                    "model": self.model_state_sha256,
                    "fit_audit": self.fit_audit_sha256,
                    "selected_epoch": self.selected_epoch,
                    "trajectory": self.trajectory_sha256,
                    "action_history": self.action_history_sha256,
                    "curve": self.curve.state_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1LearnedSourceBankFile:
    row_count: int
    parquet_sha256: str
    records_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class G1LearnedSourceBuild:
    path: Path
    outer_target: str
    source_domain: str
    specimen_count: int
    dependency_sha256: str
    hyperparameters_sha256: str
    model_state_sha256: str
    bank: G1LearnedSourceBankFile


@dataclass(frozen=True, slots=True)
class G1InnerPolicyBridgeEvaluation:
    outer_target: str
    validation_domain: str
    hyperparameters_sha256: str
    model_state_sha256: str
    metrics: tuple[InnerPolicyEngineeringMetric, ...]
    fixed_selections: tuple[SourceFixedBridgeSelection, ...]
    learned_evidence_sha256: str
    bridge_evidence_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        expected_tasks = (InspectionTask.FIELD, InspectionTask.CAI)
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.validation_domain) is not str
            or not self.validation_domain
            or self.validation_domain == self.outer_target
            or not _valid_sha256(self.hyperparameters_sha256)
            or not _valid_sha256(self.model_state_sha256)
            or type(self.metrics) is not tuple
            or len(self.metrics) != 2
            or tuple(row.task for row in self.metrics) != expected_tasks
            or any(
                type(row) is not InnerPolicyEngineeringMetric
                or row.outer_target != self.outer_target
                or row.validation_domain != self.validation_domain
                or row.hyperparameters_sha256 != self.hyperparameters_sha256
                or row.model_state_sha256 != self.model_state_sha256
                for row in self.metrics
            )
            or type(self.fixed_selections) is not tuple
            or len(self.fixed_selections) != 2
            or tuple(row.task for row in self.fixed_selections) != expected_tasks
            or any(
                type(row) is not SourceFixedBridgeSelection
                or row.outer_target != self.outer_target
                or row.validation_domain != self.validation_domain
                for row in self.fixed_selections
            )
            or not _valid_sha256(self.learned_evidence_sha256)
            or not _valid_sha256(self.bridge_evidence_sha256)
        ):
            raise G1SourcePolicyEvaluationError(
                "inner policy bridge evaluation is invalid"
            )
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-inner-policy-bridge-evaluation",
                    "outer_target": self.outer_target,
                    "validation_domain": self.validation_domain,
                    "hyperparameters": self.hyperparameters_sha256,
                    "model": self.model_state_sha256,
                    "metrics": tuple(row.state_sha256 for row in self.metrics),
                    "fixed_selections": tuple(
                        row.state_sha256 for row in self.fixed_selections
                    ),
                    "learned_evidence": self.learned_evidence_sha256,
                    "bridge_evidence": self.bridge_evidence_sha256,
                }
            ),
        )


def _record_key(record: G1LearnedSourceRecord) -> tuple[object, ...]:
    return (
        record.outer_target,
        record.source_domain,
        record.specimen_id,
        record.curve.task.value,
    )


def _action_history_sha(actions: tuple[object, ...]) -> str:
    try:
        tokens = tuple(
            (action.cell_index, action.from_level, action.to_level)
            for action in actions
        )
    except AttributeError as error:
        raise G1SourcePolicyEvaluationError(
            "learned source action history is invalid"
        ) from error
    if type(actions) is not tuple or not actions:
        raise G1SourcePolicyEvaluationError(
            "learned source action history is invalid"
        )
    return _json_sha(
        {
            "schema": 1,
            "kind": "g1-learned-source-action-history",
            "actions": tokens,
        }
    )


def _learned_source_record_from_trajectory(
    trajectory: ClosedLoopTrajectory,
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    *,
    assessor: object,
    encoder: object,
    actor: object,
    outer_target: str,
    source_domain: str,
    specimen_id: str,
    dependency_sha256: str,
    full_scan: np.ndarray,
    true_cai: float,
) -> G1LearnedSourceRecord:
    audit = getattr(actor, "audit", None)
    hyperparameters = getattr(actor, "hyperparameters", None)
    if (
        type(trajectory) is not ClosedLoopTrajectory
        or trajectory.stopped
        or trajectory.model_sha256 != getattr(actor, "model_state_sha256", None)
        or type(hyperparameters) is not PolicyTrainingHyperparameters
        or not _valid_sha256(getattr(audit, "state_sha256", None))
    ):
        raise G1SourcePolicyEvaluationError(
            "learned source trajectory identity changed"
        )
    curve = evaluate_g1_action_history(
        world,
        grid,
        prior,
        assessor=assessor,
        encoder=encoder,
        method=LEARNED_POLICY_METHOD,
        target_domain=source_domain,
        specimen_sha256=trajectory.specimen_sha256,
        actions=trajectory.action_history,
        full_scan=full_scan,
        true_cai=true_cai,
    )
    return G1LearnedSourceRecord(
        outer_target=outer_target,
        source_domain=source_domain,
        specimen_id=specimen_id,
        fit_domains=tuple(audit.fit_domains),
        dependency_sha256=dependency_sha256,
        hyperparameters_sha256=hyperparameters.state_sha256,
        model_state_sha256=actor.model_state_sha256,
        fit_audit_sha256=audit.state_sha256,
        selected_epoch=audit.selected_epoch,
        trajectory_sha256=trajectory.state_sha256,
        action_history_sha256=_action_history_sha(trajectory.action_history),
        curve=curve,
    )


def materialize_learned_source_for_world(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    surface_hypothesis: SurfaceHypothesis,
    prior: SourceBackgroundPrior,
    authorization: SourceTeacherAuthorization,
    *,
    assessor: object,
    encoder: object,
    actor: object,
    outer_target: str,
    source_domain: str,
    specimen_id: str,
    specimen_sha256: str,
    dependency_sha256: str,
    full_scan: np.ndarray,
    true_cai: float,
) -> G1LearnedSourceRecord:
    audit = getattr(actor, "audit", None)
    hyperparameters = getattr(actor, "hyperparameters", None)
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(prior) is not SourceBackgroundPrior
        or type(authorization) is not SourceTeacherAuthorization
        or not callable(getattr(assessor, "predict", None))
        or not _valid_sha256(getattr(assessor, "model_state_sha256", None))
        or not callable(getattr(encoder, "encode", None))
        or not callable(actor)
        or not _valid_sha256(getattr(actor, "model_state_sha256", None))
        or type(hyperparameters) is not PolicyTrainingHyperparameters
        or outer_target != authorization.outer_target
        or source_domain != authorization.labeled_domain
        or source_domain == outer_target
        or type(specimen_id) is not str
        or not specimen_id
        or not _valid_sha256(specimen_sha256)
        or not _valid_sha256(dependency_sha256)
        or getattr(audit, "outer_target", None) != outer_target
        or getattr(audit, "validation_domain", None) != source_domain
        or tuple(getattr(audit, "fit_domains", ())) != authorization.fit_domains
        or not _valid_sha256(getattr(audit, "state_sha256", None))
        or type(getattr(audit, "selected_epoch", None)) is not int
    ):
        raise G1SourcePolicyEvaluationError(
            "learned source world request is invalid"
        )
    validate_source_teacher_dependencies(
        authorization,
        prior,
        assessor=assessor,
    )
    state_builder = G1ObservableStateBuilder(
        grid=grid,
        surface_hypothesis=surface_hypothesis,
        prior=prior,
        assessor=assessor,
        encoder=encoder,
        cai_context_mode=hyperparameters.cai_context_mode,
        task_token_mode=hyperparameters.task_token_mode,
    )
    trajectory = run_closed_loop(
        world,
        grid,
        target_domain=source_domain,
        specimen_sha256=specimen_sha256,
        state_builder=state_builder,
        actor=actor,
        stop_threshold=None,
    )
    return _learned_source_record_from_trajectory(
        trajectory,
        world,
        grid,
        prior,
        assessor=assessor,
        encoder=encoder,
        actor=actor,
        outer_target=outer_target,
        source_domain=source_domain,
        specimen_id=specimen_id,
        dependency_sha256=dependency_sha256,
        full_scan=full_scan,
        true_cai=true_cai,
    )


def learned_source_bank_path(
    work_root: str | Path,
    outer_target: str,
    hyperparameters_sha256: str,
    source_domain: str,
) -> Path:
    if (
        type(outer_target) is not str
        or not outer_target
        or type(source_domain) is not str
        or not source_domain
        or outer_target == source_domain
        or not _valid_sha256(hyperparameters_sha256)
        or any(
            "/" in value or "\\" in value or value in {".", ".."}
            for value in (outer_target, source_domain)
        )
    ):
        raise G1SourcePolicyEvaluationError(
            "learned source bank identity is invalid"
        )
    return (
        Path(work_root)
        / outer_target
        / hyperparameters_sha256
        / f"{source_domain}.parquet"
    )


def _materialize_g1_learned_source_records_batched(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    actor: object,
    specimen_ids: tuple[str, ...],
    progress: Callable[[str], None] | None,
) -> tuple[G1LearnedSourceRecord, ...]:
    roster = dependencies.roster
    prior = dependencies.prior_fit.prior
    assessor = dependencies.assessor_fit.assessor
    hyperparameters = actor.hyperparameters
    batch_specimens = max(1, int(protocol.encoder_batch_size) // 2)
    output = []
    for start in range(0, len(specimen_ids), batch_specimens):
        selected = specimen_ids[start : start + batch_specimens]
        requests = []
        payloads = []
        for specimen in selected:
            teacher_view = runtime.mavis.source_teacher_view(specimen)
            specimen_sha = runtime.specimen_sha256(roster.labeled_domain, specimen)
            for task in (InspectionTask.FIELD, InspectionTask.CAI):
                world, grid, surface = build_g1_world(
                    runtime,
                    dataset_id=roster.labeled_domain,
                    specimen_id=specimen,
                    task=task,
                    endpoint_budget=protocol.endpoint_budget,
                )
                builder = G1ObservableStateBuilder(
                    grid=grid,
                    surface_hypothesis=surface.hypothesis,
                    prior=prior,
                    assessor=assessor,
                    encoder=encoder,
                    cai_context_mode=hyperparameters.cai_context_mode,
                    task_token_mode=hyperparameters.task_token_mode,
                )
                requests.append(
                    G1BatchRolloutRequest(
                        world=world,
                        grid=grid,
                        target_domain=roster.labeled_domain,
                        specimen_sha256=specimen_sha,
                        state_builder=builder,
                    )
                )
                payloads.append(
                    (
                        world,
                        grid,
                        specimen,
                        teacher_view.full_scan,
                        teacher_view.true_cai,
                    )
                )
        trajectories = run_g1_closed_loop_batch(
            tuple(requests),
            actor=actor,
            stop_threshold=None,
        )
        for trajectory, payload in zip(trajectories, payloads, strict=True):
            world, grid, specimen, full_scan, true_cai = payload
            output.append(
                _learned_source_record_from_trajectory(
                    trajectory,
                    world,
                    grid,
                    prior,
                    assessor=assessor,
                    encoder=encoder,
                    actor=actor,
                    outer_target=roster.outer_target,
                    source_domain=roster.labeled_domain,
                    specimen_id=specimen,
                    dependency_sha256=dependencies.state_sha256,
                    full_scan=full_scan,
                    true_cai=true_cai,
                )
            )
        completed = start + len(selected)
        if progress is not None:
            progress(
                f"G1 learned source {roster.outer_target}/"
                f"{roster.labeled_domain}: {completed}/"
                f"{len(specimen_ids)} specimens"
            )
    return tuple(output)


def materialize_g1_learned_source_records(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    actor: object,
    specimen_ids: tuple[str, ...],
    progress: Callable[[str], None] | None = None,
) -> tuple[G1LearnedSourceRecord, ...]:
    audit = getattr(actor, "audit", None)
    hyperparameters = getattr(actor, "hyperparameters", None)
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or type(dependencies) is not G1SourceDependencies
        or not callable(getattr(encoder, "encode", None))
        or not callable(actor)
        or not _valid_sha256(getattr(actor, "model_state_sha256", None))
        or type(hyperparameters) is not PolicyTrainingHyperparameters
        or type(specimen_ids) is not tuple
        or not specimen_ids
        or len(set(specimen_ids)) != len(specimen_ids)
        or (progress is not None and not callable(progress))
    ):
        raise G1SourcePolicyEvaluationError(
            "learned source materialization request is invalid"
        )
    roster = dependencies.roster
    if (
        getattr(audit, "outer_target", None) != roster.outer_target
        or getattr(audit, "validation_domain", None) != roster.labeled_domain
        or tuple(getattr(audit, "fit_domains", ())) != roster.fit_domains
        or not _valid_sha256(getattr(audit, "state_sha256", None))
    ):
        raise G1SourcePolicyEvaluationError("learned source actor fold changed")
    available = {
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == roster.labeled_domain
    }
    if not set(specimen_ids) <= available:
        raise G1SourcePolicyEvaluationError(
            "learned source specimen is outside its validation fold"
        )
    if callable(getattr(actor, "score_batch", None)):
        records = _materialize_g1_learned_source_records_batched(
            runtime,
            protocol,
            dependencies,
            encoder=encoder,
            actor=actor,
            specimen_ids=specimen_ids,
            progress=progress,
        )
        _ordered_records(records)
        if (
            len(records) != 2 * len(specimen_ids)
            or {row.specimen_id for row in records} != set(specimen_ids)
            or any(
                row.dependency_sha256 != dependencies.state_sha256
                or row.hyperparameters_sha256 != hyperparameters.state_sha256
                or row.model_state_sha256 != actor.model_state_sha256
                or row.fit_audit_sha256 != audit.state_sha256
                for row in records
            )
        ):
            raise G1SourcePolicyEvaluationError(
                "learned source materialization evidence changed"
            )
        return records
    prior = dependencies.prior_fit.prior
    assessor = dependencies.assessor_fit.assessor
    output = []
    for specimen_index, specimen in enumerate(specimen_ids, start=1):
        teacher_view = runtime.mavis.source_teacher_view(specimen)
        specimen_sha = runtime.specimen_sha256(roster.labeled_domain, specimen)
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            world, grid, surface = build_g1_world(
                runtime,
                dataset_id=roster.labeled_domain,
                specimen_id=specimen,
                task=task,
                endpoint_budget=protocol.endpoint_budget,
            )
            output.append(
                materialize_learned_source_for_world(
                    world,
                    grid,
                    surface.hypothesis,
                    prior,
                    dependencies.authorization,
                    assessor=assessor,
                    encoder=encoder,
                    actor=actor,
                    outer_target=roster.outer_target,
                    source_domain=roster.labeled_domain,
                    specimen_id=specimen,
                    specimen_sha256=specimen_sha,
                    dependency_sha256=dependencies.state_sha256,
                    full_scan=teacher_view.full_scan,
                    true_cai=teacher_view.true_cai,
                )
            )
        if progress is not None and (
            specimen_index % 10 == 0 or specimen_index == len(specimen_ids)
        ):
            progress(
                f"G1 learned source {roster.outer_target}/"
                f"{roster.labeled_domain}: {specimen_index}/"
                f"{len(specimen_ids)} specimens"
            )
    records = tuple(output)
    _ordered_records(records)
    if (
        len(records) != 2 * len(specimen_ids)
        or {row.specimen_id for row in records} != set(specimen_ids)
        or any(
            row.dependency_sha256 != dependencies.state_sha256
            or row.hyperparameters_sha256 != hyperparameters.state_sha256
            or row.model_state_sha256 != actor.model_state_sha256
            or row.fit_audit_sha256 != audit.state_sha256
            for row in records
        )
    ):
        raise G1SourcePolicyEvaluationError(
            "learned source materialization evidence changed"
        )
    return records


def _ordered_records(
    records: tuple[G1LearnedSourceRecord, ...],
) -> tuple[G1LearnedSourceRecord, ...]:
    scalar_fields = (
        "outer_target",
        "source_domain",
        "fit_domains",
        "dependency_sha256",
        "hyperparameters_sha256",
        "model_state_sha256",
        "fit_audit_sha256",
        "selected_epoch",
    )
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1LearnedSourceRecord for row in records)
        or len({row.state_sha256 for row in records}) != len(records)
        or len({_record_key(row) for row in records}) != len(records)
        or any(len({getattr(row, name) for row in records}) != 1 for name in scalar_fields)
    ):
        raise G1SourcePolicyEvaluationError("learned source bank roster is invalid")
    for specimen in {row.specimen_id for row in records}:
        rows = tuple(row for row in records if row.specimen_id == specimen)
        if (
            {row.curve.task for row in rows}
            != {InspectionTask.FIELD, InspectionTask.CAI}
            or len(rows) != 2
            or len({row.curve.specimen_sha256 for row in rows}) != 1
            or len({row.curve.grid_sha256 for row in rows}) != 1
        ):
            raise G1SourcePolicyEvaluationError(
                "learned source specimen roster is incomplete"
            )
    return tuple(sorted(records, key=_record_key))


def _records_sha(records: tuple[G1LearnedSourceRecord, ...]) -> str:
    return _json_sha(tuple(row.state_sha256 for row in records))


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.manifest.json")


def _row(record: G1LearnedSourceRecord) -> dict[str, object]:
    curve = record.curve
    return {
        "outer_target": record.outer_target,
        "source_domain": record.source_domain,
        "specimen_id": record.specimen_id,
        "specimen_sha256": curve.specimen_sha256,
        "fit_domains_json": json.dumps(record.fit_domains, separators=(",", ":")),
        "dependency_sha256": record.dependency_sha256,
        "hyperparameters_sha256": record.hyperparameters_sha256,
        "model_state_sha256": record.model_state_sha256,
        "fit_audit_sha256": record.fit_audit_sha256,
        "selected_epoch": record.selected_epoch,
        "trajectory_sha256": record.trajectory_sha256,
        "action_history_sha256": record.action_history_sha256,
        "task": curve.task.value,
        "method": curve.method,
        "grid_sha256": curve.grid_sha256,
        "evaluator_sha256": curve.evaluator_sha256,
        "warm_start_sha256": curve.warm_start_sha256,
        "nominal_budgets": curve.nominal_budgets.tolist(),
        "exact_budgets": curve.exact_budgets.tolist(),
        "task_losses": curve.task_losses.tolist(),
        "projected_state_sha256_json": json.dumps(
            curve.projected_state_sha256, separators=(",", ":")
        ),
        "auebc": curve.auebc,
        "curve_sha256": curve.state_sha256,
        "record_sha256": record.state_sha256,
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_learned_source_bank(
    path: str | Path,
    records: tuple[G1LearnedSourceRecord, ...],
) -> G1LearnedSourceBankFile:
    destination = Path(path)
    ordered = _ordered_records(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        pl.DataFrame([_row(row) for row in ordered]).write_parquet(
            temporary,
            compression="zstd",
            statistics=False,
            row_group_size=512,
        )
        parquet_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        records_sha = _records_sha(ordered)
        manifest = {
            "schema_version": 1,
            "scope": "inspection_agent_g1_learned_source_engineering",
            "row_count": len(ordered),
            "outer_target": ordered[0].outer_target,
            "source_domain": ordered[0].source_domain,
            "fit_domains": list(ordered[0].fit_domains),
            "dependency_sha256": ordered[0].dependency_sha256,
            "hyperparameters_sha256": ordered[0].hyperparameters_sha256,
            "model_state_sha256": ordered[0].model_state_sha256,
            "fit_audit_sha256": ordered[0].fit_audit_sha256,
            "selected_epoch": ordered[0].selected_epoch,
            "parquet_sha256": parquet_sha,
            "records_sha256": records_sha,
        }
        manifest_payload = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        os.replace(temporary, destination)
        _atomic_bytes(_manifest_path(destination), manifest_payload)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    identity, replay = read_learned_source_bank(destination)
    if tuple(row.state_sha256 for row in replay) != tuple(
        row.state_sha256 for row in ordered
    ):
        raise G1SourcePolicyEvaluationError(
            "written learned source bank did not replay exactly"
        )
    return identity


def _record_from_row(row: dict[str, object]) -> G1LearnedSourceRecord:
    try:
        fit_raw = json.loads(str(row["fit_domains_json"]))
        projected_raw = json.loads(str(row["projected_state_sha256_json"]))
        if not isinstance(fit_raw, list) or not isinstance(projected_raw, list):
            raise TypeError
        curve = replay_engineering_curve(
            method=str(row["method"]),
            target_domain=str(row["source_domain"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            grid_sha256=str(row["grid_sha256"]),
            evaluator_sha256=str(row["evaluator_sha256"]),
            warm_start_sha256=str(row["warm_start_sha256"]),
            exact_budgets=row["exact_budgets"],
            task_losses=row["task_losses"],
            projected_state_sha256=tuple(str(value) for value in projected_raw),
            state_sha256=str(row["curve_sha256"]),
        )
        if (
            tuple(float(value) for value in row["nominal_budgets"])
            != tuple(float(value) for value in curve.nominal_budgets)
            or float(row["auebc"]) != curve.auebc
        ):
            raise G1SourcePolicyEvaluationError(
                "learned source curve summary changed"
            )
        record = G1LearnedSourceRecord(
            outer_target=str(row["outer_target"]),
            source_domain=str(row["source_domain"]),
            specimen_id=str(row["specimen_id"]),
            fit_domains=tuple(str(value) for value in fit_raw),
            dependency_sha256=str(row["dependency_sha256"]),
            hyperparameters_sha256=str(row["hyperparameters_sha256"]),
            model_state_sha256=str(row["model_state_sha256"]),
            fit_audit_sha256=str(row["fit_audit_sha256"]),
            selected_epoch=int(row["selected_epoch"]),
            trajectory_sha256=str(row["trajectory_sha256"]),
            action_history_sha256=str(row["action_history_sha256"]),
            curve=curve,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, G1SourcePolicyEvaluationError):
            raise
        raise G1SourcePolicyEvaluationError(
            "learned source record cannot be decoded"
        ) from error
    if record.state_sha256 != str(row.get("record_sha256", "")):
        raise G1SourcePolicyEvaluationError("learned source record hash changed")
    return record


def read_learned_source_bank(
    path: str | Path,
) -> tuple[G1LearnedSourceBankFile, tuple[G1LearnedSourceRecord, ...]]:
    source = Path(path)
    manifest_path = _manifest_path(source)
    try:
        parquet_payload = source.read_bytes()
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        rows = pl.read_parquet(source).to_dicts()
    except (OSError, json.JSONDecodeError, UnicodeError, pl.exceptions.PolarsError) as error:
        raise G1SourcePolicyEvaluationError(
            "learned source bank cannot be read"
        ) from error
    expected_keys = {
        "schema_version",
        "scope",
        "row_count",
        "outer_target",
        "source_domain",
        "fit_domains",
        "dependency_sha256",
        "hyperparameters_sha256",
        "model_state_sha256",
        "fit_audit_sha256",
        "selected_epoch",
        "parquet_sha256",
        "records_sha256",
    }
    parquet_sha = hashlib.sha256(parquet_payload).hexdigest()
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("scope")
        != "inspection_agent_g1_learned_source_engineering"
        or manifest.get("parquet_sha256") != parquet_sha
    ):
        raise G1SourcePolicyEvaluationError("learned source bank manifest changed")
    records = tuple(_record_from_row(row) for row in rows)
    ordered = _ordered_records(records)
    records_sha = _records_sha(ordered)
    first = records[0]
    if (
        records != ordered
        or manifest.get("row_count") != len(records)
        or manifest.get("outer_target") != first.outer_target
        or manifest.get("source_domain") != first.source_domain
        or manifest.get("fit_domains") != list(first.fit_domains)
        or manifest.get("dependency_sha256") != first.dependency_sha256
        or manifest.get("hyperparameters_sha256") != first.hyperparameters_sha256
        or manifest.get("model_state_sha256") != first.model_state_sha256
        or manifest.get("fit_audit_sha256") != first.fit_audit_sha256
        or manifest.get("selected_epoch") != first.selected_epoch
        or manifest.get("records_sha256") != records_sha
    ):
        raise G1SourcePolicyEvaluationError("learned source bank evidence changed")
    return (
        G1LearnedSourceBankFile(
            row_count=len(records),
            parquet_sha256=parquet_sha,
            records_sha256=records_sha,
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        ),
        records,
    )


def build_g1_learned_source_bank(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    actor: object,
    work_root: str | Path,
    progress: Callable[[str], None] | None = None,
) -> G1LearnedSourceBuild:
    hyperparameters = getattr(actor, "hyperparameters", None)
    audit = getattr(actor, "audit", None)
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or type(dependencies) is not G1SourceDependencies
        or not callable(getattr(encoder, "encode", None))
        or not callable(actor)
        or not _valid_sha256(getattr(actor, "model_state_sha256", None))
        or type(hyperparameters) is not PolicyTrainingHyperparameters
        or not _valid_sha256(getattr(audit, "state_sha256", None))
        or (progress is not None and not callable(progress))
    ):
        raise G1SourcePolicyEvaluationError(
            "learned source bank build request is invalid"
        )
    roster = dependencies.roster
    if (
        getattr(audit, "outer_target", None) != roster.outer_target
        or getattr(audit, "validation_domain", None) != roster.labeled_domain
        or tuple(getattr(audit, "fit_domains", ())) != roster.fit_domains
    ):
        raise G1SourcePolicyEvaluationError("learned source actor fold changed")
    specimen_ids = tuple(
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == roster.labeled_domain
    )
    if len(specimen_ids) != int(protocol.domain_counts[roster.labeled_domain]):
        raise G1SourcePolicyEvaluationError(
            "formal learned source specimen roster changed"
        )
    path = learned_source_bank_path(
        work_root,
        roster.outer_target,
        hyperparameters.state_sha256,
        roster.labeled_domain,
    )
    manifest_path = _manifest_path(path)
    if path.exists() or manifest_path.exists():
        bank, records = read_learned_source_bank(path)
        expected_specimens = {
            runtime.specimen_sha256(roster.labeled_domain, specimen)
            for specimen in specimen_ids
        }
        if (
            {row.curve.specimen_sha256 for row in records} != expected_specimens
            or len(records) != 2 * len(specimen_ids)
            or any(
                row.fit_domains != roster.fit_domains
                or row.dependency_sha256 != dependencies.state_sha256
                or row.hyperparameters_sha256 != hyperparameters.state_sha256
                or row.model_state_sha256 != actor.model_state_sha256
                or row.fit_audit_sha256 != audit.state_sha256
                for row in records
            )
        ):
            raise G1SourcePolicyEvaluationError(
                "existing learned source bank has stale evidence"
            )
        if progress is not None:
            progress(f"G1 learned source bank reused: {path}")
    else:
        records = materialize_g1_learned_source_records(
            runtime,
            protocol,
            dependencies,
            encoder=encoder,
            actor=actor,
            specimen_ids=specimen_ids,
            progress=progress,
        )
        bank = write_learned_source_bank(path, records)
    return G1LearnedSourceBuild(
        path=path,
        outer_target=roster.outer_target,
        source_domain=roster.labeled_domain,
        specimen_count=len(specimen_ids),
        dependency_sha256=dependencies.state_sha256,
        hyperparameters_sha256=hyperparameters.state_sha256,
        model_state_sha256=actor.model_state_sha256,
        bank=bank,
    )


def _curve_geometry(curve: EngineeringCurve) -> tuple[object, ...]:
    return (
        curve.target_domain,
        curve.specimen_sha256,
        curve.task,
        curve.grid_sha256,
        curve.evaluator_sha256,
        curve.warm_start_sha256,
        tuple(float(value) for value in curve.nominal_budgets),
    )


def evaluate_inner_policy_bridge(
    learned_records: tuple[G1LearnedSourceRecord, ...],
    bridge_records: tuple[G1SourceBridgeRecord, ...],
    *,
    hyperparameters: PolicyTrainingHyperparameters,
) -> G1InnerPolicyBridgeEvaluation:
    learned = _ordered_records(learned_records)
    if (
        type(hyperparameters) is not PolicyTrainingHyperparameters
        or hyperparameters.state_sha256 != learned[0].hyperparameters_sha256
        or type(bridge_records) is not tuple
        or not bridge_records
        or any(type(row) is not G1SourceBridgeRecord for row in bridge_records)
    ):
        raise G1SourcePolicyEvaluationError("inner policy bridge request is invalid")
    outer = learned[0].outer_target
    source = learned[0].source_domain
    if any(row.outer_target != outer for row in bridge_records):
        raise G1SourcePolicyEvaluationError("inner policy bridge outer fold changed")
    validation_bridge = tuple(
        row for row in bridge_records if row.source_domain == source
    )
    if (
        not validation_bridge
        or any(
            row.dependency_sha256 != learned[0].dependency_sha256
            or row.fit_domains != learned[0].fit_domains
            for row in validation_bridge
        )
    ):
        raise G1SourcePolicyEvaluationError(
            "inner policy bridge dependency bundle changed"
        )
    metrics = []
    selections = []
    for task in (InspectionTask.FIELD, InspectionTask.CAI):
        learned_task = tuple(row for row in learned if row.curve.task is task)
        selection = select_source_fixed_bridge(
            bridge_records,
            validation_domain=source,
            task=task,
        )
        if selection.fit_domains != learned[0].fit_domains:
            raise G1SourcePolicyEvaluationError(
                "inner policy bridge fit-domain roster changed"
            )
        validation = tuple(
            row
            for row in bridge_records
            if row.source_domain == source and row.curve.task is task
        )
        fixed = {
            row.curve.specimen_sha256: row.curve
            for row in validation
            if row.curve.method == selection.method
        }
        oracle = {
            row.curve.specimen_sha256: row.curve
            for row in validation
            if row.curve.method == SOURCE_ORACLE_METHODS[task]
        }
        learned_by_specimen = {
            row.curve.specimen_sha256: row.curve for row in learned_task
        }
        if (
            not learned_by_specimen
            or set(learned_by_specimen) != set(fixed)
            or set(learned_by_specimen) != set(oracle)
        ):
            raise G1SourcePolicyEvaluationError(
                "inner policy bridge specimen pairing changed"
            )
        for specimen, learned_curve in learned_by_specimen.items():
            if len(
                {
                    _curve_geometry(learned_curve),
                    _curve_geometry(fixed[specimen]),
                    _curve_geometry(oracle[specimen]),
                }
            ) != 1:
                raise G1SourcePolicyEvaluationError(
                    "inner policy curves do not share evaluation geometry"
                )
        metric = InnerPolicyEngineeringMetric(
            outer_target=outer,
            validation_domain=source,
            task=task,
            hyperparameters_sha256=hyperparameters.state_sha256,
            model_state_sha256=learned[0].model_state_sha256,
            learned_auebc=float(
                np.mean([curve.auebc for curve in learned_by_specimen.values()])
            ),
            fixed_auebc=float(np.mean([curve.auebc for curve in fixed.values()])),
            oracle_auebc=float(
                np.mean([curve.auebc for curve in oracle.values()])
            ),
            selected_epoch=learned[0].selected_epoch,
        )
        metrics.append(metric)
        selections.append(selection)
    bridge_evidence = _json_sha(
        tuple(
            row.state_sha256
            for row in sorted(
                bridge_records,
                key=lambda row: (
                    row.outer_target,
                    row.source_domain,
                    row.specimen_id,
                    row.curve.task.value,
                    row.curve.method,
                ),
            )
        )
    )
    return G1InnerPolicyBridgeEvaluation(
        outer_target=outer,
        validation_domain=source,
        hyperparameters_sha256=hyperparameters.state_sha256,
        model_state_sha256=learned[0].model_state_sha256,
        metrics=tuple(metrics),
        fixed_selections=tuple(selections),
        learned_evidence_sha256=_records_sha(learned),
        bridge_evidence_sha256=bridge_evidence,
    )


__all__ = [
    "LEARNED_POLICY_METHOD",
    "G1InnerPolicyBridgeEvaluation",
    "G1LearnedSourceBankFile",
    "G1LearnedSourceBuild",
    "G1LearnedSourceRecord",
    "G1SourcePolicyEvaluationError",
    "build_g1_learned_source_bank",
    "evaluate_inner_policy_bridge",
    "learned_source_bank_path",
    "materialize_g1_learned_source_records",
    "materialize_learned_source_for_world",
    "read_learned_source_bank",
    "write_learned_source_bank",
]
