"""Source-only STOP threshold validation from frozen observable rollouts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

from cmc_bbdm.inspection_agent.cai_assessor import state_scalars
from cmc_bbdm.inspection_agent.contracts import InspectionObservation, InspectionTask
from cmc_bbdm.inspection_agent.field_task import field_loss
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    SourceBackgroundPrior,
    reconstruct_observation,
)
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .batch_rollout import G1BatchRolloutRequest, run_g1_closed_loop_batch
from .contracts import TaskTokenMode
from .dagger_selection_execution import G1OuterDaggerSelectionRun
from .formal import G1ObservableStateBuilder
from .g1 import (
    G1Protocol,
    G1Runtime,
    G1SourceDependencies,
    build_g1_source_dependencies,
    build_g1_world,
)
from .policy_training import (
    TrainedObservablePolicy,
    fit_final_observable_policy,
    fit_inner_observable_policy,
    rebind_training_example_modes,
)
from .rollout import ClosedLoopTrajectory
from .stop_bank import read_stop_bank
from .stop_execution import (
    G1FixedEndpointRecord,
    read_g1_outer_fixed_endpoint_records,
    select_g1_source_fixed_reference,
)
from .stop_training import (
    TrainedObservableStopPolicy,
    fit_final_observable_stop_head,
    fit_inner_observable_stop_head,
    join_g1_stop_training_examples,
)
from .stopping_policy import (
    SourceStopValidationTrajectory,
    StopThresholdSelection,
    select_conservative_stop_threshold,
)
from .teacher_bank import read_teacher_bank


class G1StopSelectionExecutionError(ValueError):
    """Raised when STOP selection opens truth before a fixed source rollout."""


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


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _threshold_payload(value: StopThresholdSelection) -> dict[str, object]:
    return {
        "task": value.task.value,
        "status": value.status,
        "threshold": value.threshold,
        "state_sha256": value.state_sha256,
        "trajectory_sha256": list(value.trajectory_sha256),
        "candidates": [
            {
                "threshold": row.threshold,
                "equal_domain_saving": row.equal_domain_saving,
                "equal_domain_premature_rate": row.equal_domain_premature_rate,
                "equal_domain_task_loss_ratio": row.equal_domain_task_loss_ratio,
                "equal_domain_stopped_fraction": row.equal_domain_stopped_fraction,
                "feasible": row.feasible,
            }
            for row in value.candidates
        ],
    }


@dataclass(frozen=True, slots=True)
class G1OuterStopSelectionRun:
    outer_target: str
    action_selection: G1OuterDaggerSelectionRun
    action_policy: TrainedObservablePolicy
    stop_policy: TrainedObservableStopPolicy
    selected_stop_epochs: int
    thresholds: tuple[StopThresholdSelection, ...]
    source_validation_trajectories: tuple[SourceStopValidationTrajectory, ...]
    teacher_bank_manifest_sha256s: tuple[str, ...]
    stop_bank_manifest_sha256s: tuple[str, ...]
    fixed_endpoint_record_count: int
    path: Path
    target_outcomes_opened: bool = False
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        sources = tuple(
            domain
            for domain in self.action_selection.selection.source_validation_domains
        )
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.action_selection) is not G1OuterDaggerSelectionRun
            or self.action_selection.outer_target != self.outer_target
            or type(self.action_policy) is not TrainedObservablePolicy
            or self.action_policy.audit.outer_target != self.outer_target
            or self.action_policy.audit.validation_domain is not None
            or self.action_policy.audit.fit_domains != sources
            or type(self.stop_policy) is not TrainedObservableStopPolicy
            or self.stop_policy.audit.outer_target != self.outer_target
            or self.stop_policy.audit.validation_domain is not None
            or self.stop_policy.audit.fit_domains != sources
            or self.stop_policy.base_action_model_sha256
            != self.action_policy.model_state_sha256
            or type(self.selected_stop_epochs) is not int
            or not 1 <= self.selected_stop_epochs <= 80
            or type(self.thresholds) is not tuple
            or tuple(row.task for row in self.thresholds)
            != (InspectionTask.FIELD, InspectionTask.CAI)
            or any(type(row) is not StopThresholdSelection for row in self.thresholds)
            or type(self.source_validation_trajectories) is not tuple
            or not self.source_validation_trajectories
            or {row.source_domain for row in self.source_validation_trajectories}
            != set(sources)
            or type(self.teacher_bank_manifest_sha256s) is not tuple
            or len(self.teacher_bank_manifest_sha256s) != 5
            or not all(
                _valid_sha256(value)
                for value in self.teacher_bank_manifest_sha256s
            )
            or type(self.stop_bank_manifest_sha256s) is not tuple
            or len(self.stop_bank_manifest_sha256s) != 5
            or not all(
                _valid_sha256(value) for value in self.stop_bank_manifest_sha256s
            )
            or type(self.fixed_endpoint_record_count) is not int
            or self.fixed_endpoint_record_count <= 0
            or not isinstance(self.path, Path)
            or self.target_outcomes_opened
        ):
            raise G1StopSelectionExecutionError(
                "outer STOP selection run is invalid"
            )
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-outer-stop-selection-run",
                    "outer_target": self.outer_target,
                    "action_selection": self.action_selection.selection.state_sha256,
                    "action_model": self.action_policy.model_state_sha256,
                    "stop_model": self.stop_policy.model_state_sha256,
                    "selected_stop_epochs": self.selected_stop_epochs,
                    "thresholds": tuple(row.state_sha256 for row in self.thresholds),
                    "source_trajectories": tuple(
                        row.state_sha256
                        for row in self.source_validation_trajectories
                    ),
                    "teacher_banks": self.teacher_bank_manifest_sha256s,
                    "stop_banks": self.stop_bank_manifest_sha256s,
                    "fixed_endpoint_record_count": self.fixed_endpoint_record_count,
                    "target_outcomes_opened": False,
                }
            ),
        )


def _frozen_rollout_observations(
    world: CausalInspectionWorld,
    trajectory: ClosedLoopTrajectory,
) -> tuple[InspectionObservation, ...]:
    current = world.replay(trajectory.action_history[:8])
    observations = [current]
    for step, action in zip(
        trajectory.steps,
        trajectory.action_history[8:],
        strict=True,
    ):
        if step.observation_sha256 != current.state_sha256:
            raise G1StopSelectionExecutionError(
                "source STOP rollout observation hash changed"
            )
        current = world.step(current, action)
        observations.append(current)
    if current.state_sha256 != trajectory.final_observation_sha256:
        raise G1StopSelectionExecutionError(
            "source STOP rollout final observation changed"
        )
    return tuple(observations)


def evaluate_source_stop_validation_trajectory(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    trajectory: ClosedLoopTrajectory,
    prior: SourceBackgroundPrior,
    *,
    assessor: object,
    encoder: object,
    outer_target: str,
    source_domain: str,
    full_scan: np.ndarray,
    true_cai: float,
    reference_true_loss: float,
) -> SourceStopValidationTrajectory:
    image = np.asarray(full_scan)
    target = float(true_cai)
    reference = float(reference_true_loss)
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(trajectory) is not ClosedLoopTrajectory
        or type(prior) is not SourceBackgroundPrior
        or not callable(getattr(assessor, "predict", None))
        or not callable(getattr(encoder, "encode", None))
        or type(outer_target) is not str
        or not outer_target
        or type(source_domain) is not str
        or not source_domain
        or source_domain == outer_target
        or trajectory.target_domain != source_domain
        or trajectory.task is not world.reset().task
        or trajectory.stop_threshold is not None
        or trajectory.stopped
        or len(trajectory.steps) != len(trajectory.action_history) - 8
        or prior.outer_domain != outer_target
        or source_domain in prior.source_domains
        or image.dtype != np.uint8
        or image.shape != (*grid.native_shape, 3)
        or not math.isfinite(target)
        or not math.isfinite(reference)
        or reference < 0.0
    ):
        raise G1StopSelectionExecutionError(
            "source STOP validation trajectory request is invalid"
        )
    observations = _frozen_rollout_observations(world, trajectory)
    reconstructions = tuple(
        reconstruct_observation(observation, grid, prior)
        for observation in observations
    )
    if trajectory.task is InspectionTask.FIELD:
        losses = tuple(field_loss(image, row.image) for row in reconstructions)
    elif trajectory.task is InspectionTask.CAI:
        embeddings = np.asarray(
            encoder.encode(tuple(row.image for row in reconstructions)),
            dtype=np.float64,
        )
        scalars = np.asarray(
            [state_scalars(observation) for observation in observations],
            dtype=np.float64,
        )
        predictions = np.asarray(
            assessor.predict(embeddings, scalars),
            dtype=np.float64,
        )
        if (
            embeddings.shape != (len(observations), 512)
            or predictions.shape != (len(observations),)
            or not np.all(np.isfinite(embeddings))
            or not np.all(np.isfinite(predictions))
        ):
            raise G1StopSelectionExecutionError(
                "source STOP CAI evaluation is invalid"
            )
        losses = tuple(abs(target - float(value)) for value in predictions)
    else:
        raise G1StopSelectionExecutionError("source STOP task changed")
    probabilities = tuple(
        step.scores.stop_probability for step in trajectory.steps
    ) + (0.0,)
    return SourceStopValidationTrajectory(
        outer_target=outer_target,
        source_domain=source_domain,
        specimen_sha256=trajectory.specimen_sha256,
        task=trajectory.task,
        budgets=tuple(value.effective_budget for value in observations),
        stop_probabilities=probabilities,
        true_task_losses=losses,
        reference_true_loss=reference,
        endpoint_budget=0.25,
    )


def materialize_g1_source_stop_validation_trajectories(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    action_policy: TrainedObservablePolicy,
    stop_policy: TrainedObservableStopPolicy,
    fixed_endpoint_records: tuple[G1FixedEndpointRecord, ...],
    progress: Callable[[str], None] | None = None,
) -> tuple[SourceStopValidationTrajectory, ...]:
    roster = getattr(dependencies, "roster", None)
    action_audit = getattr(action_policy, "audit", None)
    stop_audit = getattr(stop_policy, "audit", None)
    hyperparameters = getattr(action_policy, "hyperparameters", None)
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or runtime.domain_order != protocol.domain_order
        or type(dependencies) is not G1SourceDependencies
        or not callable(getattr(encoder, "encode", None))
        or type(action_policy) is not TrainedObservablePolicy
        or type(stop_policy) is not TrainedObservableStopPolicy
        or not callable(getattr(stop_policy, "score_batch", None))
        or type(fixed_endpoint_records) is not tuple
        or not fixed_endpoint_records
        or any(
            type(row) is not G1FixedEndpointRecord
            or row.outer_target != getattr(roster, "outer_target", None)
            for row in fixed_endpoint_records
        )
        or getattr(action_audit, "outer_target", None)
        != getattr(roster, "outer_target", None)
        or getattr(action_audit, "validation_domain", None)
        != getattr(roster, "labeled_domain", None)
        or tuple(getattr(action_audit, "fit_domains", ()))
        != tuple(getattr(roster, "fit_domains", ()))
        or getattr(stop_audit, "outer_target", None)
        != getattr(roster, "outer_target", None)
        or getattr(stop_audit, "validation_domain", None)
        != getattr(roster, "labeled_domain", None)
        or tuple(getattr(stop_audit, "fit_domains", ()))
        != tuple(getattr(roster, "fit_domains", ()))
        or getattr(stop_policy, "base_action_model_sha256", None)
        != action_policy.model_state_sha256
        or getattr(hyperparameters, "task_token_mode", None)
        is not TaskTokenMode.CORRECT
        or (progress is not None and not callable(progress))
    ):
        raise G1StopSelectionExecutionError(
            "source STOP validation materialization request is invalid"
        )
    outer_target = roster.outer_target
    source_domain = roster.labeled_domain
    specimen_ids = tuple(
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == source_domain
    )
    if len(specimen_ids) != int(protocol.domain_counts[source_domain]):
        raise G1StopSelectionExecutionError(
            "source STOP validation specimen roster changed"
        )
    references = {
        task: select_g1_source_fixed_reference(
            dependencies.authorization,
            fixed_endpoint_records,
            task=task,
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    }
    expected_sha = {
        runtime.specimen_sha256(source_domain, specimen) for specimen in specimen_ids
    }
    endpoint_map = {
        (row.specimen_sha256, row.task): row
        for row in fixed_endpoint_records
        if row.source_domain == source_domain
        and row.method == references[row.task].method
    }
    if (
        len(endpoint_map) != 2 * len(specimen_ids)
        or {key[0] for key in endpoint_map} != expected_sha
        or any(
            row.dependency_sha256 != dependencies.state_sha256
            or row.fit_domains != roster.fit_domains
            for row in endpoint_map.values()
        )
    ):
        raise G1StopSelectionExecutionError(
            "source STOP fixed endpoint bridge changed"
        )
    requests = []
    contexts = []
    prior = dependencies.prior_fit.prior
    assessor = dependencies.assessor_fit.assessor
    for specimen in specimen_ids:
        specimen_sha = runtime.specimen_sha256(source_domain, specimen)
        truth = runtime.mavis.source_teacher_view(specimen)
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            world, grid, surface = build_g1_world(
                runtime,
                dataset_id=source_domain,
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
                    target_domain=source_domain,
                    specimen_sha256=specimen_sha,
                    state_builder=builder,
                )
            )
            contexts.append(
                (
                    world,
                    grid,
                    truth,
                    endpoint_map[(specimen_sha, task)].task_loss,
                )
            )
    if progress is not None:
        progress(f"G1 source STOP rollout {outer_target}/{source_domain}")
    trajectories = run_g1_closed_loop_batch(
        tuple(requests),
        actor=stop_policy,
        stop_threshold=None,
    )
    if (
        type(trajectories) is not tuple
        or len(trajectories) != len(contexts)
        or any(type(row) is not ClosedLoopTrajectory for row in trajectories)
    ):
        raise G1StopSelectionExecutionError(
            "source STOP rollout result roster changed"
        )
    output = tuple(
        evaluate_source_stop_validation_trajectory(
            world,
            grid,
            trajectory,
            prior,
            assessor=assessor,
            encoder=encoder,
            outer_target=outer_target,
            source_domain=source_domain,
            full_scan=truth.full_scan,
            true_cai=truth.true_cai,
            reference_true_loss=reference_loss,
        )
        for trajectory, (world, grid, truth, reference_loss) in zip(
            trajectories,
            contexts,
            strict=True,
        )
    )
    if progress is not None:
        progress(
            f"G1 source STOP validation complete {outer_target}/{source_domain}: "
            f"{len(output)} trajectories"
        )
    return output


def run_outer_stop_selection(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    outer_target: str,
    action_selection: G1OuterDaggerSelectionRun,
    encoder: object,
    teacher_bank_root: str | Path,
    stop_bank_root: str | Path,
    fixed_endpoint_root: str | Path,
    work_root: str | Path,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> G1OuterStopSelectionRun:
    domain_order = tuple(getattr(protocol, "domain_order", ()))
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or runtime.domain_order != domain_order
        or len(domain_order) != 6
        or outer_target not in domain_order
        or type(action_selection) is not G1OuterDaggerSelectionRun
        or action_selection.outer_target != outer_target
        or getattr(action_selection, "target_outcomes_opened", False)
        or action_selection.aawr_authorization.status
        != "NOT_RUN_NOT_AUTHORIZED"
        or not callable(getattr(encoder, "encode", None))
        or type(device) is not str
        or not device
        or (progress is not None and not callable(progress))
    ):
        raise G1StopSelectionExecutionError(
            "outer STOP selection request is invalid or AAWR is unresolved"
        )
    selected_candidates = tuple(
        row
        for row in action_selection.candidates
        if row.candidate.hyperparameters.state_sha256
        == action_selection.selection.selected_hyperparameters_sha256
    )
    if len(selected_candidates) != 1:
        raise G1StopSelectionExecutionError(
            "outer STOP selection has no unique action policy"
        )
    hyperparameters = selected_candidates[0].candidate.hyperparameters
    source_domains = tuple(domain for domain in domain_order if domain != outer_target)
    selected_records = tuple(
        row
        for row in action_selection.dagger_build.records
        if row.example.dagger_iteration <= hyperparameters.dagger_iterations
    )
    if (
        not selected_records
        or {row.example.source_domain for row in selected_records}
        != set(source_domains)
        or any(row.example.outer_target != outer_target for row in selected_records)
    ):
        raise G1StopSelectionExecutionError(
            "selected action-training roster changed"
        )
    selected_examples = tuple(
        rebind_training_example_modes(
            row.example,
            cai_context_mode=hyperparameters.cai_context_mode,
            task_token_mode=hyperparameters.task_token_mode,
        )
        for row in selected_records
    )
    base_records = []
    stop_records = []
    teacher_manifests = []
    stop_manifests = []
    for source in source_domains:
        teacher_bank, teacher_rows = read_teacher_bank(
            Path(teacher_bank_root) / outer_target / f"{source}.parquet"
        )
        stop_bank, stop_rows = read_stop_bank(
            Path(stop_bank_root) / outer_target / f"{source}.parquet"
        )
        if (
            not teacher_rows
            or not stop_rows
            or any(
                row.example.outer_target != outer_target
                or row.example.source_domain != source
                for row in teacher_rows
            )
            or any(
                row.outer_target != outer_target or row.source_domain != source
                for row in stop_rows
            )
            or not _valid_sha256(teacher_bank.manifest_sha256)
            or not _valid_sha256(stop_bank.manifest_sha256)
        ):
            raise G1StopSelectionExecutionError(
                "outer STOP source bank identity changed"
            )
        base_records.extend(teacher_rows)
        stop_records.extend(stop_rows)
        teacher_manifests.append(teacher_bank.manifest_sha256)
        stop_manifests.append(stop_bank.manifest_sha256)
    fixed_records = read_g1_outer_fixed_endpoint_records(
        protocol,
        outer_target=outer_target,
        work_root=fixed_endpoint_root,
    )
    source_trajectories = []
    inner_stop_epochs = []
    for source in source_domains:
        if progress is not None:
            progress(f"G1 STOP inner validation {outer_target}/{source}")
        action_policy = fit_inner_observable_policy(
            selected_examples,
            validation_domain=source,
            hyperparameters=hyperparameters,
            max_epochs=int(protocol.epochs),
            patience=int(protocol.patience),
            device=device,
        )
        stop_examples = join_g1_stop_training_examples(
            tuple(base_records),
            tuple(stop_records),
            action_policy=action_policy,
        )
        stop_policy = fit_inner_observable_stop_head(
            action_policy,
            stop_examples,
            validation_domain=source,
            max_epochs=int(protocol.epochs),
            patience=int(protocol.patience),
            device=device,
        )
        dependencies = build_g1_source_dependencies(
            runtime,
            protocol,
            outer_target=outer_target,
            labeled_domain=source,
            encoder=encoder,
            progress=progress,
        )
        source_trajectories.extend(
            materialize_g1_source_stop_validation_trajectories(
                runtime,
                protocol,
                dependencies,
                encoder=encoder,
                action_policy=action_policy,
                stop_policy=stop_policy,
                fixed_endpoint_records=fixed_records,
                progress=progress,
            )
        )
        inner_stop_epochs.append(stop_policy.audit.selected_epoch)
    trajectories = tuple(source_trajectories)
    thresholds = tuple(
        select_conservative_stop_threshold(
            tuple(row for row in trajectories if row.task is task)
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    )
    action_policy = fit_final_observable_policy(
        selected_examples,
        hyperparameters=hyperparameters,
        selected_epochs=action_selection.selection.final_refit_epochs,
        device=device,
    )
    final_stop_examples = join_g1_stop_training_examples(
        tuple(base_records),
        tuple(stop_records),
        action_policy=action_policy,
    )
    selected_stop_epochs = int(median(inner_stop_epochs))
    stop_policy = fit_final_observable_stop_head(
        action_policy,
        final_stop_examples,
        selected_epochs=selected_stop_epochs,
        device=device,
    )
    destination = Path(work_root) / outer_target / "selection.json"
    result = G1OuterStopSelectionRun(
        outer_target=outer_target,
        action_selection=action_selection,
        action_policy=action_policy,
        stop_policy=stop_policy,
        selected_stop_epochs=selected_stop_epochs,
        thresholds=thresholds,
        source_validation_trajectories=trajectories,
        teacher_bank_manifest_sha256s=tuple(teacher_manifests),
        stop_bank_manifest_sha256s=tuple(stop_manifests),
        fixed_endpoint_record_count=len(fixed_records),
        path=destination,
        target_outcomes_opened=False,
    )
    payload = {
        "schema_version": 1,
        "scope": "inspection_agent_g1_outer_stop_selection",
        "outer_target": outer_target,
        "source_domains": list(source_domains),
        "action_selection_sha256": action_selection.selection.state_sha256,
        "aawr_status": action_selection.aawr_authorization.status,
        "aawr_authorization_sha256": (
            action_selection.aawr_authorization.state_sha256
        ),
        "selected_hyperparameters_sha256": hyperparameters.state_sha256,
        "selected_action_epochs": action_selection.selection.final_refit_epochs,
        "inner_stop_epochs": inner_stop_epochs,
        "selected_stop_epochs": selected_stop_epochs,
        "teacher_bank_manifest_sha256s": teacher_manifests,
        "stop_bank_manifest_sha256s": stop_manifests,
        "fixed_endpoint_record_count": len(fixed_records),
        "source_validation_trajectory_sha256s": [
            row.state_sha256 for row in trajectories
        ],
        "thresholds": [_threshold_payload(row) for row in thresholds],
        "final_action_model_sha256": action_policy.model_state_sha256,
        "final_action_fit_audit_sha256": action_policy.audit.state_sha256,
        "final_stop_model_sha256": stop_policy.model_state_sha256,
        "final_stop_fit_audit_sha256": stop_policy.audit.state_sha256,
        "target_outcomes_opened": False,
        "state_sha256": result.state_sha256,
    }
    _atomic_json(destination, payload)
    return result


__all__ = [
    "G1OuterStopSelectionRun",
    "G1StopSelectionExecutionError",
    "evaluate_source_stop_validation_trajectory",
    "materialize_g1_source_stop_validation_trajectories",
    "run_outer_stop_selection",
]
