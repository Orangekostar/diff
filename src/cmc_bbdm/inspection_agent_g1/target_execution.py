"""Hash-bound target policy trajectories frozen before target-truth access."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.state import InspectionCellAction

from .batch_rollout import G1BatchRolloutRequest, run_g1_closed_loop_batch
from .contracts import ACTION_SLOT_COUNT, CAIContextMode, TaskTokenMode
from .features import canonical_action_from_slot
from .formal import G1ObservableStateBuilder
from .g1 import G1FinalDependencies, G1Protocol, G1Runtime, build_g1_world
from .rollout import (
    ClosedLoopTrajectory,
    ObservablePolicyScores,
    RolloutStep,
    SurfaceVariant,
    controlled_surface_hypothesis,
    shuffled_surface_donors,
)
from .stop_selection_execution import G1OuterStopSelectionRun
from .stopping_policy import REGISTERED_STOP_THRESHOLDS, StopThresholdSelection

SHUFFLED_SURFACE_SEED = 2026090103


class G1TargetExecutionError(ValueError):
    """Raised when a frozen target trajectory cannot be verified exactly."""


class TargetPolicyVariant(str, Enum):
    PROPOSED = "PROPOSED"
    PROPOSED_STOP = "PROPOSED_STOP"
    NO_TASK = "NO_TASK"
    WRONG_TASK = "WRONG_TASK"
    NO_SURFACE = "NO_SURFACE"
    SHUFFLED_SURFACE = "SHUFFLED_SURFACE"


TARGET_ACTION_VARIANTS = (
    TargetPolicyVariant.PROPOSED,
    TargetPolicyVariant.NO_TASK,
    TargetPolicyVariant.WRONG_TASK,
    TargetPolicyVariant.NO_SURFACE,
    TargetPolicyVariant.SHUFFLED_SURFACE,
)


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


_VARIANT_MODES = {
    TargetPolicyVariant.PROPOSED: (
        TaskTokenMode.CORRECT,
        SurfaceVariant.CORRECT_SURFACE,
        False,
    ),
    TargetPolicyVariant.PROPOSED_STOP: (
        TaskTokenMode.CORRECT,
        SurfaceVariant.CORRECT_SURFACE,
        True,
    ),
    TargetPolicyVariant.NO_TASK: (
        TaskTokenMode.NO_TASK,
        SurfaceVariant.CORRECT_SURFACE,
        False,
    ),
    TargetPolicyVariant.WRONG_TASK: (
        TaskTokenMode.WRONG_TASK,
        SurfaceVariant.CORRECT_SURFACE,
        False,
    ),
    TargetPolicyVariant.NO_SURFACE: (
        TaskTokenMode.CORRECT,
        SurfaceVariant.NO_SURFACE,
        False,
    ),
    TargetPolicyVariant.SHUFFLED_SURFACE: (
        TaskTokenMode.CORRECT,
        SurfaceVariant.SHUFFLED_SURFACE,
        False,
    ),
}


@dataclass(frozen=True, slots=True)
class G1TargetTrajectoryRecord:
    outer_target: str
    specimen_id: str
    specimen_sha256: str
    task: InspectionTask
    variant: TargetPolicyVariant
    task_token_mode: TaskTokenMode
    surface_variant: SurfaceVariant
    donor_surface_sha256: str | None
    final_dependency_sha256: str
    action_selection_sha256: str
    action_model_sha256: str
    stop_model_sha256: str
    trajectory: ClosedLoopTrajectory
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        expected = _VARIANT_MODES.get(self.variant)
        has_donor = self.donor_surface_sha256 is not None
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.specimen_id) is not str
            or not self.specimen_id
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or type(self.variant) is not TargetPolicyVariant
            or type(self.task_token_mode) is not TaskTokenMode
            or type(self.surface_variant) is not SurfaceVariant
            or expected is None
            or (self.task_token_mode, self.surface_variant) != expected[:2]
            or has_donor
            != (self.surface_variant is SurfaceVariant.SHUFFLED_SURFACE)
            or (has_donor and not _valid_sha256(self.donor_surface_sha256))
            or not all(
                _valid_sha256(value)
                for value in (
                    self.final_dependency_sha256,
                    self.action_selection_sha256,
                    self.action_model_sha256,
                    self.stop_model_sha256,
                )
            )
            or type(self.trajectory) is not ClosedLoopTrajectory
            or self.trajectory.target_domain != self.outer_target
            or self.trajectory.specimen_sha256 != self.specimen_sha256
            or self.trajectory.task is not self.task
            or self.trajectory.model_sha256 != self.stop_model_sha256
            or (self.trajectory.stop_threshold is not None) != expected[2]
        ):
            raise G1TargetExecutionError("target trajectory record is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-trajectory-record",
                    "outer_target": self.outer_target,
                    "specimen_id": self.specimen_id,
                    "specimen_sha256": self.specimen_sha256,
                    "task": self.task.value,
                    "variant": self.variant.value,
                    "task_token_mode": self.task_token_mode.value,
                    "surface_variant": self.surface_variant.value,
                    "donor_surface": self.donor_surface_sha256,
                    "final_dependency": self.final_dependency_sha256,
                    "action_selection": self.action_selection_sha256,
                    "action_model": self.action_model_sha256,
                    "stop_model": self.stop_model_sha256,
                    "trajectory": self.trajectory.state_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1TargetTrajectoryBankFile:
    row_count: int
    outer_target: str
    parquet_sha256: str
    records_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.row_count) is not int
            or self.row_count <= 0
            or type(self.outer_target) is not str
            or not self.outer_target
            or not all(
                _valid_sha256(value)
                for value in (
                    self.parquet_sha256,
                    self.records_sha256,
                    self.manifest_sha256,
                )
            )
        ):
            raise G1TargetExecutionError("target trajectory bank identity is invalid")


@dataclass(frozen=True, slots=True)
class G1OuterTargetTrajectoryBuild:
    path: Path
    outer_target: str
    specimen_count: int
    record_count: int
    final_dependency_sha256: str
    outer_selection_sha256: str
    fragment_manifest_sha256s: tuple[str, ...]
    bank: G1TargetTrajectoryBankFile
    target_outcomes_opened: bool = False
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, Path)
            or type(self.outer_target) is not str
            or not self.outer_target
            or type(self.specimen_count) is not int
            or self.specimen_count <= 0
            or type(self.record_count) is not int
            or self.record_count < 10 * self.specimen_count
            or not _valid_sha256(self.final_dependency_sha256)
            or not _valid_sha256(self.outer_selection_sha256)
            or type(self.fragment_manifest_sha256s) is not tuple
            or not self.fragment_manifest_sha256s
            or not all(
                _valid_sha256(value) for value in self.fragment_manifest_sha256s
            )
            or type(self.bank) is not G1TargetTrajectoryBankFile
            or self.bank.outer_target != self.outer_target
            or self.bank.row_count != self.record_count
            or type(self.target_outcomes_opened) is not bool
            or self.target_outcomes_opened
        ):
            raise G1TargetExecutionError("outer target trajectory build is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-outer-target-trajectory-build",
                    "path": self.path.as_posix(),
                    "outer_target": self.outer_target,
                    "specimen_count": self.specimen_count,
                    "record_count": self.record_count,
                    "final_dependency": self.final_dependency_sha256,
                    "outer_selection": self.outer_selection_sha256,
                    "fragments": self.fragment_manifest_sha256s,
                    "bank": self.bank.manifest_sha256,
                    "target_outcomes_opened": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1TargetTrajectoryBankSeal:
    outer_target: str
    record_count: int
    bank_manifest_sha256: str
    records_sha256: str
    trajectory_sha256s: tuple[str, ...]
    acquired_positions_sha256s: tuple[str, ...]
    acquired_values_sha256s: tuple[str, ...]
    state_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.record_count) is not int
            or self.record_count <= 0
            or not all(
                _valid_sha256(value)
                for value in (self.bank_manifest_sha256, self.records_sha256)
            )
            or any(
                type(values) is not tuple
                or len(values) != self.record_count
                or not all(_valid_sha256(value) for value in values)
                for values in (
                    self.trajectory_sha256s,
                    self.acquired_positions_sha256s,
                    self.acquired_values_sha256s,
                )
            )
            or not _valid_sha256(self.state_sha256)
        ):
            raise G1TargetExecutionError("target trajectory bank seal is invalid")


def target_trajectory_bank_path(work_root: str | Path, outer_target: str) -> Path:
    if (
        type(outer_target) is not str
        or not outer_target
        or "/" in outer_target
        or "\\" in outer_target
        or outer_target in {".", ".."}
    ):
        raise G1TargetExecutionError("target trajectory path identity is invalid")
    return Path(work_root) / outer_target / "target_trajectories.parquet"


def target_trajectory_fragment_path(
    work_root: str | Path,
    outer_target: str,
    task: InspectionTask,
    variant: TargetPolicyVariant,
) -> Path:
    if task not in (InspectionTask.FIELD, InspectionTask.CAI) or type(
        variant
    ) is not TargetPolicyVariant:
        raise G1TargetExecutionError("target trajectory fragment identity is invalid")
    return (
        target_trajectory_bank_path(work_root, outer_target).parent
        / "fragments"
        / f"{task.value.lower()}_{variant.value.lower()}.parquet"
    )


def plan_g1_target_variants(
    outer_target: str,
    thresholds: tuple[StopThresholdSelection, ...],
) -> tuple[tuple[InspectionTask, TargetPolicyVariant, float | None], ...]:
    if (
        type(outer_target) is not str
        or not outer_target
        or type(thresholds) is not tuple
        or len(thresholds) != 2
        or any(type(value) is not StopThresholdSelection for value in thresholds)
        or tuple(value.task for value in thresholds)
        != (InspectionTask.FIELD, InspectionTask.CAI)
        or any(value.outer_target != outer_target for value in thresholds)
        or any(not _valid_sha256(value.state_sha256) for value in thresholds)
    ):
        raise G1TargetExecutionError("target STOP threshold plan is invalid")
    plan = []
    for selection in thresholds:
        if selection.status == "STOP_AUTHORIZED_SOURCE_ONLY":
            if selection.threshold not in REGISTERED_STOP_THRESHOLDS:
                raise G1TargetExecutionError("authorized target STOP threshold is invalid")
            stop_threshold = float(selection.threshold)
        elif selection.status == "STOP_NOT_AUTHORIZED":
            if selection.threshold is not None:
                raise G1TargetExecutionError("unauthorized target STOP threshold is present")
            stop_threshold = None
        else:
            raise G1TargetExecutionError("target STOP authorization status is invalid")
        plan.extend((selection.task, variant, None) for variant in TARGET_ACTION_VARIANTS)
        if stop_threshold is not None:
            plan.append(
                (
                    selection.task,
                    TargetPolicyVariant.PROPOSED_STOP,
                    stop_threshold,
                )
            )
    return tuple(plan)


def materialize_g1_target_variant_records(
    runtime: G1Runtime,
    *,
    outer_target: str,
    specimen_ids: tuple[str, ...],
    task: InspectionTask,
    variant: TargetPolicyVariant,
    prior: SourceBackgroundPrior,
    assessor: object,
    encoder: object,
    actor: object,
    cai_context_mode: CAIContextMode,
    endpoint_budget: float,
    final_dependency_sha256: str,
    action_selection_sha256: str,
    action_model_sha256: str,
    stop_model_sha256: str,
    stop_threshold: float | None,
    progress: object | None = None,
) -> tuple[G1TargetTrajectoryRecord, ...]:
    if type(runtime) is not G1Runtime:
        raise G1TargetExecutionError("target variant materialization is invalid")
    try:
        endpoint = float(endpoint_budget)
    except (TypeError, ValueError, OverflowError) as error:
        raise G1TargetExecutionError(
            "target variant materialization is invalid"
        ) from error
    expected = _VARIANT_MODES.get(variant)
    available = tuple(
        specimen_id
        for specimen_id, domain in zip(
            getattr(getattr(runtime, "mavis", None), "specimen_ids", ()),
            getattr(getattr(runtime, "mavis", None), "dataset_ids", ()),
            strict=True,
        )
        if domain == outer_target
    )
    if (
        type(outer_target) is not str
        or not outer_target
        or type(specimen_ids) is not tuple
        or not specimen_ids
        or len(set(specimen_ids)) != len(specimen_ids)
        or any(type(value) is not str or not value for value in specimen_ids)
        or not set(specimen_ids) <= set(available)
        or task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or type(variant) is not TargetPolicyVariant
        or expected is None
        or type(prior) is not SourceBackgroundPrior
        or prior.outer_domain != outer_target
        or outer_target in prior.source_domains
        or not callable(getattr(assessor, "predict", None))
        or not _valid_sha256(getattr(assessor, "model_state_sha256", None))
        or not callable(getattr(encoder, "encode", None))
        or not callable(actor)
        or not callable(getattr(actor, "score_batch", None))
        or getattr(actor, "model_state_sha256", None) != stop_model_sha256
        or type(cai_context_mode) is not CAIContextMode
        or endpoint != 0.25
        or not all(
            _valid_sha256(value)
            for value in (
                final_dependency_sha256,
                action_selection_sha256,
                action_model_sha256,
                stop_model_sha256,
            )
        )
        or (stop_threshold is not None) != expected[2]
        or (progress is not None and not callable(progress))
    ):
        raise G1TargetExecutionError("target variant materialization is invalid")
    task_token_mode, surface_variant, _uses_stop = expected
    donor_by_recipient: dict[str, str] = {}
    if surface_variant is SurfaceVariant.SHUFFLED_SURFACE:
        donors = shuffled_surface_donors(
            available,
            (outer_target,) * len(available),
            seed=SHUFFLED_SURFACE_SEED,
        )
        donor_by_recipient = dict(zip(available, donors, strict=True))
    requests = []
    donor_shas = []
    specimen_shas = []
    for specimen_id in specimen_ids:
        world, grid, surface = build_g1_world(
            runtime,
            dataset_id=outer_target,
            specimen_id=specimen_id,
            task=task,
            endpoint_budget=endpoint_budget,
        )
        donor_id = donor_by_recipient.get(specimen_id)
        donor = (
            None
            if donor_id is None
            else runtime.surface(outer_target, donor_id).hypothesis
        )
        controlled = controlled_surface_hypothesis(
            surface.hypothesis,
            surface_variant,
            donor=donor,
        )
        specimen_sha = runtime.specimen_sha256(outer_target, specimen_id)
        requests.append(
            G1BatchRolloutRequest(
                world=world,
                grid=grid,
                target_domain=outer_target,
                specimen_sha256=specimen_sha,
                state_builder=G1ObservableStateBuilder(
                    grid=grid,
                    surface_hypothesis=controlled,
                    prior=prior,
                    assessor=assessor,
                    encoder=encoder,
                    cai_context_mode=cai_context_mode,
                    task_token_mode=task_token_mode,
                ),
            )
        )
        donor_shas.append(None if donor is None else donor.state_sha256)
        specimen_shas.append(specimen_sha)
    trajectories = run_g1_closed_loop_batch(
        tuple(requests),
        actor=actor,
        stop_threshold=stop_threshold,
    )
    output = tuple(
        G1TargetTrajectoryRecord(
            outer_target=outer_target,
            specimen_id=specimen_id,
            specimen_sha256=specimen_sha,
            task=task,
            variant=variant,
            task_token_mode=task_token_mode,
            surface_variant=surface_variant,
            donor_surface_sha256=donor_sha,
            final_dependency_sha256=final_dependency_sha256,
            action_selection_sha256=action_selection_sha256,
            action_model_sha256=action_model_sha256,
            stop_model_sha256=stop_model_sha256,
            trajectory=trajectory,
        )
        for specimen_id, specimen_sha, donor_sha, trajectory in zip(
            specimen_ids,
            specimen_shas,
            donor_shas,
            trajectories,
            strict=True,
        )
    )
    if progress is not None:
        progress(
            f"G1 target trajectories {outer_target}/{task.value}/{variant.value}: "
            f"{len(output)} specimens"
        )
    return output


def _validate_target_fragment(
    records: tuple[G1TargetTrajectoryRecord, ...],
    *,
    outer_target: str,
    specimen_ids: tuple[str, ...],
    task: InspectionTask,
    variant: TargetPolicyVariant,
    stop_threshold: float | None,
    final_dependency_sha256: str,
    outer_selection_sha256: str,
    action_model_sha256: str,
    stop_model_sha256: str,
) -> None:
    if (
        len(records) != len(specimen_ids)
        or {row.specimen_id for row in records} != set(specimen_ids)
        or any(
            row.outer_target != outer_target
            or row.task is not task
            or row.variant is not variant
            or row.trajectory.stop_threshold != stop_threshold
            or row.final_dependency_sha256 != final_dependency_sha256
            or row.action_selection_sha256 != outer_selection_sha256
            or row.action_model_sha256 != action_model_sha256
            or row.stop_model_sha256 != stop_model_sha256
            for row in records
        )
    ):
        raise G1TargetExecutionError("target trajectory fragment changed")


def build_g1_outer_target_trajectory_bank(
    runtime: G1Runtime,
    protocol: G1Protocol,
    final_dependencies: G1FinalDependencies,
    stop_selection: G1OuterStopSelectionRun,
    *,
    encoder: object,
    work_root: str | Path,
    progress: object | None = None,
) -> G1OuterTargetTrajectoryBuild:
    outer_target = getattr(stop_selection, "outer_target", None)
    source_domains = tuple(
        domain for domain in getattr(protocol, "domain_order", ()) if domain != outer_target
    )
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or runtime.domain_order != protocol.domain_order
        or type(final_dependencies) is not G1FinalDependencies
        or type(stop_selection) is not G1OuterStopSelectionRun
        or outer_target not in protocol.domain_order
        or stop_selection.target_outcomes_opened
        or final_dependencies.outer_target != outer_target
        or final_dependencies.fit_domains != source_domains
        or stop_selection.action_policy.audit.fit_domains != source_domains
        or stop_selection.stop_policy.audit.fit_domains != source_domains
        or stop_selection.action_policy.audit.validation_domain is not None
        or stop_selection.stop_policy.audit.validation_domain is not None
        or stop_selection.stop_policy.base_action_model_sha256
        != stop_selection.action_policy.model_state_sha256
        or stop_selection.action_policy.hyperparameters.task_token_mode
        is not TaskTokenMode.CORRECT
        or not callable(getattr(encoder, "encode", None))
        or (progress is not None and not callable(progress))
    ):
        raise G1TargetExecutionError("outer target trajectory request is invalid")
    specimen_ids = tuple(
        specimen_id
        for specimen_id, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == outer_target
    )
    if len(specimen_ids) != int(protocol.domain_counts[outer_target]):
        raise G1TargetExecutionError("outer target specimen roster changed")
    plan = plan_g1_target_variants(outer_target, stop_selection.thresholds)
    action_model_sha = stop_selection.action_policy.model_state_sha256
    stop_model_sha = stop_selection.stop_policy.model_state_sha256
    outer_selection_sha = stop_selection.state_sha256
    fragments: list[G1TargetTrajectoryRecord] = []
    fragment_manifests = []
    for task, variant, threshold in plan:
        path = target_trajectory_fragment_path(
            work_root,
            outer_target,
            task,
            variant,
        )
        present = path.exists(), _manifest_path(path).exists()
        if present == (True, True):
            fragment_bank, records = read_g1_target_trajectory_bank(path)
            if progress is not None:
                progress(
                    f"G1 target fragment replay {outer_target}/{task.value}/"
                    f"{variant.value}: {len(records)} specimens"
                )
        elif present == (False, False):
            records = materialize_g1_target_variant_records(
                runtime,
                outer_target=outer_target,
                specimen_ids=specimen_ids,
                task=task,
                variant=variant,
                prior=final_dependencies.prior,
                assessor=final_dependencies.assessor,
                encoder=encoder,
                actor=stop_selection.stop_policy,
                cai_context_mode=(
                    stop_selection.action_policy.hyperparameters.cai_context_mode
                ),
                endpoint_budget=protocol.endpoint_budget,
                final_dependency_sha256=final_dependencies.state_sha256,
                action_selection_sha256=outer_selection_sha,
                action_model_sha256=action_model_sha,
                stop_model_sha256=stop_model_sha,
                stop_threshold=threshold,
                progress=progress,
            )
            fragment_bank = write_g1_target_trajectory_bank(path, records)
        else:
            raise G1TargetExecutionError("target trajectory fragment is incomplete")
        _validate_target_fragment(
            records,
            outer_target=outer_target,
            specimen_ids=specimen_ids,
            task=task,
            variant=variant,
            stop_threshold=threshold,
            final_dependency_sha256=final_dependencies.state_sha256,
            outer_selection_sha256=outer_selection_sha,
            action_model_sha256=action_model_sha,
            stop_model_sha256=stop_model_sha,
        )
        fragments.extend(records)
        fragment_manifests.append(fragment_bank.manifest_sha256)
    ordered = _ordered_records(tuple(fragments))
    destination = target_trajectory_bank_path(work_root, outer_target)
    present = destination.exists(), _manifest_path(destination).exists()
    if present == (True, True):
        bank, replay = read_g1_target_trajectory_bank(destination)
        if tuple(row.state_sha256 for row in replay) != tuple(
            row.state_sha256 for row in ordered
        ):
            raise G1TargetExecutionError("outer target trajectory bank changed")
    elif present == (False, False):
        bank = write_g1_target_trajectory_bank(destination, ordered)
    else:
        raise G1TargetExecutionError("outer target trajectory bank is incomplete")
    result = G1OuterTargetTrajectoryBuild(
        path=destination,
        outer_target=outer_target,
        specimen_count=len(specimen_ids),
        record_count=len(ordered),
        final_dependency_sha256=final_dependencies.state_sha256,
        outer_selection_sha256=outer_selection_sha,
        fragment_manifest_sha256s=tuple(fragment_manifests),
        bank=bank,
    )
    if progress is not None:
        progress(
            f"G1 target bank complete {outer_target}: {len(ordered)} trajectories"
        )
    return result


def _replay_target_record_causally(
    runtime: G1Runtime,
    record: G1TargetTrajectoryRecord,
) -> None:
    trajectory = record.trajectory
    world, _grid, _surface = build_g1_world(
        runtime,
        dataset_id=record.outer_target,
        specimen_id=record.specimen_id,
        task=record.task,
        endpoint_budget=0.25,
    )
    current = world.replay(trajectory.action_history[:8])
    action_index = 8
    for step_index, step in enumerate(trajectory.steps):
        if step.observation_sha256 != current.state_sha256:
            raise G1TargetExecutionError(
                "target trajectory observation changed during causal replay"
            )
        if step.selected_slot is None:
            if step_index != len(trajectory.steps) - 1:
                raise G1TargetExecutionError("target STOP step is not terminal")
            continue
        action = canonical_action_from_slot(step.selected_slot)
        if (
            action_index >= len(trajectory.action_history)
            or trajectory.action_history[action_index] != action
        ):
            raise G1TargetExecutionError("target trajectory action history changed")
        current = world.step(current, action)
        action_index += 1
    if (
        action_index != len(trajectory.action_history)
        or current.action_history != trajectory.action_history
        or current.state_sha256 != trajectory.final_observation_sha256
        or current.native_count != trajectory.native_count
        or current.effective_budget != trajectory.effective_budget
        or not np.array_equal(
            current.acquired_positions,
            trajectory.acquired_positions,
        )
        or not np.array_equal(
            current.measurement_values,
            trajectory.acquired_values,
        )
    ):
        raise G1TargetExecutionError("target trajectory final causal state changed")


def seal_g1_target_trajectory_bank(
    runtime: G1Runtime,
    bank: G1TargetTrajectoryBankFile,
    records: tuple[G1TargetTrajectoryRecord, ...],
) -> G1TargetTrajectoryBankSeal:
    if type(runtime) is not G1Runtime or type(bank) is not G1TargetTrajectoryBankFile:
        raise G1TargetExecutionError("target trajectory seal request is invalid")
    ordered = _ordered_records(records)
    records_sha = _records_sha(ordered)
    if (
        bank.row_count != len(ordered)
        or bank.outer_target != ordered[0].outer_target
        or bank.records_sha256 != records_sha
    ):
        raise G1TargetExecutionError("target trajectory bank identity changed before seal")
    for record in ordered:
        _replay_target_record_causally(runtime, record)
    trajectory_shas = tuple(row.trajectory.state_sha256 for row in ordered)
    position_shas = tuple(
        row.trajectory.acquired_positions_sha256 for row in ordered
    )
    value_shas = tuple(row.trajectory.acquired_values_sha256 for row in ordered)
    state_sha = _json_sha(
        {
            "schema": 1,
            "kind": "g1-target-trajectory-bank-seal",
            "outer_target": bank.outer_target,
            "record_count": len(ordered),
            "bank_manifest": bank.manifest_sha256,
            "records": records_sha,
            "trajectories": trajectory_shas,
            "acquired_positions": position_shas,
            "acquired_values": value_shas,
        }
    )
    seal = G1TargetTrajectoryBankSeal(
        outer_target=bank.outer_target,
        record_count=len(ordered),
        bank_manifest_sha256=bank.manifest_sha256,
        records_sha256=records_sha,
        trajectory_sha256s=trajectory_shas,
        acquired_positions_sha256s=position_shas,
        acquired_values_sha256s=value_shas,
        state_sha256=state_sha,
    )
    validate_g1_target_trajectory_bank_seal(bank, ordered, seal)
    return seal


def validate_g1_target_trajectory_bank_seal(
    bank: G1TargetTrajectoryBankFile,
    records: tuple[G1TargetTrajectoryRecord, ...],
    seal: G1TargetTrajectoryBankSeal,
) -> None:
    if (
        type(bank) is not G1TargetTrajectoryBankFile
        or type(seal) is not G1TargetTrajectoryBankSeal
    ):
        raise G1TargetExecutionError("target trajectory bank seal is invalid")
    ordered = _ordered_records(records)
    records_sha = _records_sha(ordered)
    trajectory_shas = tuple(row.trajectory.state_sha256 for row in ordered)
    position_shas = tuple(
        row.trajectory.acquired_positions_sha256 for row in ordered
    )
    value_shas = tuple(row.trajectory.acquired_values_sha256 for row in ordered)
    expected_state = _json_sha(
        {
            "schema": 1,
            "kind": "g1-target-trajectory-bank-seal",
            "outer_target": bank.outer_target,
            "record_count": len(ordered),
            "bank_manifest": bank.manifest_sha256,
            "records": records_sha,
            "trajectories": trajectory_shas,
            "acquired_positions": position_shas,
            "acquired_values": value_shas,
        }
    )
    if (
        bank.row_count != len(ordered)
        or bank.records_sha256 != records_sha
        or seal.outer_target != bank.outer_target
        or seal.record_count != len(ordered)
        or seal.bank_manifest_sha256 != bank.manifest_sha256
        or seal.records_sha256 != records_sha
        or seal.trajectory_sha256s != trajectory_shas
        or seal.acquired_positions_sha256s != position_shas
        or seal.acquired_values_sha256s != value_shas
        or seal.state_sha256 != expected_state
    ):
        raise G1TargetExecutionError("target trajectory bank seal changed")


def _record_key(record: G1TargetTrajectoryRecord) -> tuple[str, str, str]:
    return record.specimen_id, record.task.value, record.variant.value


def _ordered_records(
    records: tuple[G1TargetTrajectoryRecord, ...],
) -> tuple[G1TargetTrajectoryRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1TargetTrajectoryRecord for row in records)
        or len({_record_key(row) for row in records}) != len(records)
        or len({row.state_sha256 for row in records}) != len(records)
        or len({row.trajectory.state_sha256 for row in records}) != len(records)
        or len({row.outer_target for row in records}) != 1
        or len({row.final_dependency_sha256 for row in records}) != 1
        or len({row.action_selection_sha256 for row in records}) != 1
        or len({row.action_model_sha256 for row in records}) != 1
        or len({row.stop_model_sha256 for row in records}) != 1
    ):
        raise G1TargetExecutionError("target trajectory bank roster is invalid")
    return tuple(sorted(records, key=_record_key))


def _records_sha(records: tuple[G1TargetTrajectoryRecord, ...]) -> str:
    return _json_sha(tuple(row.state_sha256 for row in records))


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.manifest.json")


def _action_payload(action: InspectionCellAction) -> list[int]:
    return [action.cell_index, action.from_level, action.to_level]


def _step_payload(step: RolloutStep) -> dict[str, object]:
    return {
        "step_index": step.step_index,
        "observation_sha256": step.observation_sha256,
        "policy_state_sha256": step.policy_state_sha256,
        "model_sha256": step.scores.model_sha256,
        "stop_probability": step.scores.stop_probability,
        "selected_slot": step.selected_slot,
        "scores_sha256": step.scores.state_sha256,
        "step_sha256": step.state_sha256,
    }


def _record_row(record: G1TargetTrajectoryRecord) -> dict[str, object]:
    trajectory = record.trajectory
    logits = np.ascontiguousarray(
        np.stack([step.scores.action_logits for step in trajectory.steps]),
        dtype="<f8",
    )
    positions = np.ascontiguousarray(trajectory.acquired_positions, dtype="<i8")
    values = np.ascontiguousarray(trajectory.acquired_values, dtype=np.uint8)
    return {
        "outer_target": record.outer_target,
        "specimen_id": record.specimen_id,
        "specimen_sha256": record.specimen_sha256,
        "task": record.task.value,
        "variant": record.variant.value,
        "task_token_mode": record.task_token_mode.value,
        "surface_variant": record.surface_variant.value,
        "donor_surface_sha256": record.donor_surface_sha256,
        "final_dependency_sha256": record.final_dependency_sha256,
        "action_selection_sha256": record.action_selection_sha256,
        "action_model_sha256": record.action_model_sha256,
        "stop_model_sha256": record.stop_model_sha256,
        "stop_threshold": trajectory.stop_threshold,
        "stopped": trajectory.stopped,
        "termination_reason": trajectory.termination_reason,
        "action_history_json": json.dumps(
            [_action_payload(action) for action in trajectory.action_history],
            separators=(",", ":"),
        ),
        "steps_json": json.dumps(
            [_step_payload(step) for step in trajectory.steps],
            separators=(",", ":"),
        ),
        "action_logits_le_f64": logits.tobytes(order="C"),
        "final_observation_sha256": trajectory.final_observation_sha256,
        "acquired_count": len(positions),
        "acquired_positions_le_i64": positions.tobytes(order="C"),
        "acquired_values_u8": values.tobytes(order="C"),
        "native_count": trajectory.native_count,
        "effective_budget": trajectory.effective_budget,
        "acquired_positions_sha256": trajectory.acquired_positions_sha256,
        "acquired_values_sha256": trajectory.acquired_values_sha256,
        "trajectory_sha256": trajectory.state_sha256,
        "record_sha256": record.state_sha256,
    }


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
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


def write_g1_target_trajectory_bank(
    path: str | Path,
    records: tuple[G1TargetTrajectoryRecord, ...],
) -> G1TargetTrajectoryBankFile:
    destination = Path(path)
    ordered = _ordered_records(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        pl.DataFrame(
            [_record_row(record) for record in ordered], infer_schema_length=None
        ).write_parquet(
            temporary,
            compression="zstd",
            statistics=False,
            row_group_size=128,
        )
        parquet_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        records_sha = _records_sha(ordered)
        manifest = {
            "schema_version": 1,
            "scope": "inspection_agent_g1_target_trajectory_bank",
            "row_count": len(ordered),
            "outer_target": ordered[0].outer_target,
            "final_dependency_sha256": ordered[0].final_dependency_sha256,
            "action_selection_sha256": ordered[0].action_selection_sha256,
            "action_model_sha256": ordered[0].action_model_sha256,
            "stop_model_sha256": ordered[0].stop_model_sha256,
            "parquet_sha256": parquet_sha,
            "records_sha256": records_sha,
        }
        manifest_payload = (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("ascii")
        os.replace(temporary, destination)
        _atomic_replace_bytes(_manifest_path(destination), manifest_payload)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    identity, replay = read_g1_target_trajectory_bank(destination)
    if tuple(row.state_sha256 for row in replay) != tuple(
        row.state_sha256 for row in ordered
    ):
        raise G1TargetExecutionError(
            "written target trajectory bank did not replay exactly"
        )
    return identity


def _decode_actions(raw: object) -> tuple[InspectionCellAction, ...]:
    parsed = json.loads(str(raw))
    if not isinstance(parsed, list):
        raise TypeError
    return tuple(
        InspectionCellAction(int(value[0]), int(value[1]), int(value[2]))
        for value in parsed
        if isinstance(value, list) and len(value) == 3
    )


def _decode_steps(row: dict[str, object]) -> tuple[RolloutStep, ...]:
    parsed = json.loads(str(row["steps_json"]))
    if not isinstance(parsed, list) or not parsed:
        raise TypeError
    payload = bytes(row["action_logits_le_f64"])
    logits = np.frombuffer(payload, dtype="<f8")
    if logits.size != len(parsed) * ACTION_SLOT_COUNT:
        raise TypeError
    logits = logits.reshape(len(parsed), ACTION_SLOT_COUNT)
    steps = []
    for value, action_logits in zip(parsed, logits, strict=True):
        if not isinstance(value, dict):
            raise TypeError
        scores = ObservablePolicyScores(
            policy_state_sha256=str(value["policy_state_sha256"]),
            model_sha256=str(value["model_sha256"]),
            action_logits=action_logits,
            stop_probability=float(value["stop_probability"]),
            state_sha256=str(value["scores_sha256"]),
        )
        selected_raw = value["selected_slot"]
        step = RolloutStep(
            step_index=int(value["step_index"]),
            observation_sha256=str(value["observation_sha256"]),
            policy_state_sha256=str(value["policy_state_sha256"]),
            scores=scores,
            selected_slot=None if selected_raw is None else int(selected_raw),
            state_sha256=str(value["step_sha256"]),
        )
        steps.append(step)
    return tuple(steps)


def _record_from_row(row: dict[str, object]) -> G1TargetTrajectoryRecord:
    try:
        count = int(row["acquired_count"])
        positions = np.frombuffer(
            bytes(row["acquired_positions_le_i64"]), dtype="<i8"
        )
        values = np.frombuffer(bytes(row["acquired_values_u8"]), dtype=np.uint8)
        if count <= 0 or positions.size != count * 2 or values.size != count * 3:
            raise TypeError
        threshold_raw = row["stop_threshold"]
        trajectory = ClosedLoopTrajectory(
            target_domain=str(row["outer_target"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            model_sha256=str(row["stop_model_sha256"]),
            stop_threshold=(
                None if threshold_raw is None else float(threshold_raw)
            ),
            stopped=bool(row["stopped"]),
            termination_reason=str(row["termination_reason"]),
            action_history=_decode_actions(row["action_history_json"]),
            steps=_decode_steps(row),
            final_observation_sha256=str(row["final_observation_sha256"]),
            acquired_positions=positions.reshape(count, 2),
            acquired_values=values.reshape(count, 3),
            native_count=int(row["native_count"]),
            effective_budget=float(row["effective_budget"]),
        )
        if (
            trajectory.acquired_positions_sha256
            != str(row["acquired_positions_sha256"])
            or trajectory.acquired_values_sha256
            != str(row["acquired_values_sha256"])
            or trajectory.state_sha256 != str(row["trajectory_sha256"])
        ):
            raise G1TargetExecutionError("target trajectory hash changed")
        record = G1TargetTrajectoryRecord(
            outer_target=str(row["outer_target"]),
            specimen_id=str(row["specimen_id"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            variant=TargetPolicyVariant(str(row["variant"])),
            task_token_mode=TaskTokenMode(str(row["task_token_mode"])),
            surface_variant=SurfaceVariant(str(row["surface_variant"])),
            donor_surface_sha256=(
                None
                if row["donor_surface_sha256"] is None
                else str(row["donor_surface_sha256"])
            ),
            final_dependency_sha256=str(row["final_dependency_sha256"]),
            action_selection_sha256=str(row["action_selection_sha256"]),
            action_model_sha256=str(row["action_model_sha256"]),
            stop_model_sha256=str(row["stop_model_sha256"]),
            trajectory=trajectory,
        )
        if record.state_sha256 != str(row["record_sha256"]):
            raise G1TargetExecutionError("target trajectory record hash changed")
        return record
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ) as error:
        if isinstance(error, G1TargetExecutionError):
            raise
        raise G1TargetExecutionError(
            "target trajectory row cannot be reconstructed"
        ) from error


def read_g1_target_trajectory_bank(
    path: str | Path,
) -> tuple[G1TargetTrajectoryBankFile, tuple[G1TargetTrajectoryRecord, ...]]:
    source = Path(path)
    try:
        parquet_payload = source.read_bytes()
        manifest_payload = _manifest_path(source).read_bytes()
        manifest = json.loads(manifest_payload)
        rows = pl.read_parquet(source).to_dicts()
    except (OSError, UnicodeError, json.JSONDecodeError, pl.exceptions.PolarsError) as error:
        raise G1TargetExecutionError(
            "target trajectory bank cannot be read"
        ) from error
    expected_keys = {
        "schema_version",
        "scope",
        "row_count",
        "outer_target",
        "final_dependency_sha256",
        "action_selection_sha256",
        "action_model_sha256",
        "stop_model_sha256",
        "parquet_sha256",
        "records_sha256",
    }
    parquet_sha = hashlib.sha256(parquet_payload).hexdigest()
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("scope") != "inspection_agent_g1_target_trajectory_bank"
        or manifest.get("parquet_sha256") != parquet_sha
    ):
        raise G1TargetExecutionError("target trajectory bank manifest changed")
    records = tuple(_record_from_row(row) for row in rows)
    ordered = _ordered_records(records)
    records_sha = _records_sha(ordered)
    first = ordered[0]
    if (
        records != ordered
        or manifest.get("row_count") != len(records)
        or manifest.get("outer_target") != first.outer_target
        or manifest.get("final_dependency_sha256")
        != first.final_dependency_sha256
        or manifest.get("action_selection_sha256")
        != first.action_selection_sha256
        or manifest.get("action_model_sha256") != first.action_model_sha256
        or manifest.get("stop_model_sha256") != first.stop_model_sha256
        or manifest.get("records_sha256") != records_sha
    ):
        raise G1TargetExecutionError("target trajectory bank evidence changed")
    identity = G1TargetTrajectoryBankFile(
        row_count=len(records),
        outer_target=first.outer_target,
        parquet_sha256=parquet_sha,
        records_sha256=records_sha,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )
    return identity, records


__all__ = [
    "G1OuterTargetTrajectoryBuild",
    "G1TargetExecutionError",
    "G1TargetTrajectoryBankFile",
    "G1TargetTrajectoryBankSeal",
    "G1TargetTrajectoryRecord",
    "TargetPolicyVariant",
    "build_g1_outer_target_trajectory_bank",
    "materialize_g1_target_variant_records",
    "plan_g1_target_variants",
    "read_g1_target_trajectory_bank",
    "seal_g1_target_trajectory_bank",
    "target_trajectory_bank_path",
    "target_trajectory_fragment_path",
    "validate_g1_target_trajectory_bank_seal",
    "write_g1_target_trajectory_bank",
]
