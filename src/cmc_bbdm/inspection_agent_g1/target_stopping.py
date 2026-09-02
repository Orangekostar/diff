"""Held-out target STOP outcomes and task-specific formal gates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .g1 import StopGateEvidence, StopGateResult, evaluate_stop_gate
from .source_bridge import OuterFixedBridgeSelection
from .statistics import (
    FORMAL_BOOTSTRAP_SEED,
    G1PairedBootstrap,
    formal_synchronized_bootstrap,
)
from .stopping_policy import REGISTERED_STOP_THRESHOLDS
from .target_evaluation import G1TargetCurveRecord
from .target_execution import (
    G1TargetTrajectoryRecord,
    TargetPolicyVariant,
)
from .target_reference import G1TargetReferenceCurveRecord


class G1TargetStoppingError(ValueError):
    """Raised when held-out target STOP evidence is incomplete or inconsistent."""


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
class G1TargetStopOutcome:
    outer_target: str
    specimen_id: str
    specimen_sha256: str
    task: InspectionTask
    stop_authorized: bool
    threshold: float | None
    stopped: bool
    budget: float
    normalized_measurement_saving: float
    task_loss: float
    reference_task_loss: float
    task_loss_ratio: float | None
    premature_stop: bool
    false_continue: bool | None
    proposed_trajectory_sha256: str
    deployed_trajectory_sha256: str
    deployed_curve_sha256: str
    reference_curve_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        budget = float(self.budget)
        saving = float(self.normalized_measurement_saving)
        loss = float(self.task_loss)
        reference = float(self.reference_task_loss)
        ratio = None if self.task_loss_ratio is None else float(self.task_loss_ratio)
        expected_saving = max(0.0, 1.0 - budget / 0.25)
        expected_ratio = (
            loss / reference
            if reference > 0.0
            else (1.0 if loss == 0.0 else None)
        )
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.specimen_id) is not str
            or not self.specimen_id
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or type(self.stop_authorized) is not bool
            or (
                self.stop_authorized
                and self.threshold not in REGISTERED_STOP_THRESHOLDS
            )
            or (not self.stop_authorized and self.threshold is not None)
            or type(self.stopped) is not bool
            or (self.stopped and not self.stop_authorized)
            or not math.isfinite(budget)
            or not 0.0 <= budget <= 0.25
            or not math.isfinite(saving)
            or not 0.0 <= saving <= 1.0
            or not math.isclose(
                saving,
                expected_saving if self.stop_authorized else 0.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isfinite(loss)
            or loss < 0.0
            or not math.isfinite(reference)
            or reference < 0.0
            or ratio != expected_ratio
            or type(self.premature_stop) is not bool
            or (self.premature_stop and not self.stopped)
            or (
                self.stop_authorized
                and type(self.false_continue) is not bool
            )
            or (not self.stop_authorized and self.false_continue is not None)
            or not all(
                _valid_sha256(value)
                for value in (
                    self.proposed_trajectory_sha256,
                    self.deployed_trajectory_sha256,
                    self.deployed_curve_sha256,
                    self.reference_curve_sha256,
                )
            )
        ):
            raise G1TargetStoppingError("target STOP outcome is invalid")
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "normalized_measurement_saving", saving)
        object.__setattr__(self, "task_loss", loss)
        object.__setattr__(self, "reference_task_loss", reference)
        object.__setattr__(self, "task_loss_ratio", ratio)
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-stop-outcome",
                    "outer_target": self.outer_target,
                    "specimen_id": self.specimen_id,
                    "specimen_sha256": self.specimen_sha256,
                    "task": self.task.value,
                    "stop_authorized": self.stop_authorized,
                    "threshold": self.threshold,
                    "stopped": self.stopped,
                    "budget": budget,
                    "normalized_measurement_saving": saving,
                    "task_loss": loss,
                    "reference_task_loss": reference,
                    "task_loss_ratio": ratio,
                    "premature_stop": self.premature_stop,
                    "false_continue": self.false_continue,
                    "proposed_trajectory": self.proposed_trajectory_sha256,
                    "deployed_trajectory": self.deployed_trajectory_sha256,
                    "deployed_curve": self.deployed_curve_sha256,
                    "reference_curve": self.reference_curve_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1TaskStopAnalysis:
    task: InspectionTask
    normalized_measurement_saving: float
    task_loss_ratio: float | None
    premature_stop_rate: float
    false_continue_rate: float | None
    fraction_stopped: float
    fraction_never_stopped: float
    authorized_domains: int
    saving_bootstrap: G1PairedBootstrap
    gate: StopGateResult | None
    status: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        rates = (
            self.normalized_measurement_saving,
            self.premature_stop_rate,
            self.fraction_stopped,
            self.fraction_never_stopped,
        )
        expected_status = (
            self.gate.status
            if self.gate is not None
            else f"G1_{self.task.value}_STOPPING_NO_GO"
        )
        if (
            self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in rates)
            or (
                self.task_loss_ratio is not None
                and (
                    not math.isfinite(float(self.task_loss_ratio))
                    or float(self.task_loss_ratio) < 0.0
                )
            )
            or (
                self.false_continue_rate is not None
                and (
                    not math.isfinite(float(self.false_continue_rate))
                    or not 0.0 <= float(self.false_continue_rate) <= 1.0
                )
            )
            or not math.isclose(
                self.fraction_stopped + self.fraction_never_stopped,
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or type(self.authorized_domains) is not int
            or not 0 <= self.authorized_domains <= 6
            or type(self.saving_bootstrap) is not G1PairedBootstrap
            or (self.gate is not None and self.gate.task is not self.task)
            or self.status != expected_status
        ):
            raise G1TargetStoppingError("target task STOP analysis is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-task-stop-analysis",
                    "task": self.task.value,
                    "normalized_measurement_saving": self.normalized_measurement_saving,
                    "task_loss_ratio": self.task_loss_ratio,
                    "premature_stop_rate": self.premature_stop_rate,
                    "false_continue_rate": self.false_continue_rate,
                    "fraction_stopped": self.fraction_stopped,
                    "fraction_never_stopped": self.fraction_never_stopped,
                    "authorized_domains": self.authorized_domains,
                    "saving_bootstrap": self.saving_bootstrap.distribution_sha256,
                    "gate": None if self.gate is None else self.gate.state_sha256,
                    "status": self.status,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1TargetStoppingAnalysis:
    field: G1TaskStopAnalysis
    cai: G1TaskStopAnalysis
    outcome_sha256s: tuple[str, ...]
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.field) is not G1TaskStopAnalysis
            or self.field.task is not InspectionTask.FIELD
            or type(self.cai) is not G1TaskStopAnalysis
            or self.cai.task is not InspectionTask.CAI
            or type(self.outcome_sha256s) is not tuple
            or not self.outcome_sha256s
            or not all(_valid_sha256(value) for value in self.outcome_sha256s)
        ):
            raise G1TargetStoppingError("target STOP analysis is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-stopping-analysis",
                    "field": self.field.state_sha256,
                    "cai": self.cai.state_sha256,
                    "outcomes": self.outcome_sha256s,
                }
            ),
        )


def materialize_g1_target_stop_outcomes(
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
    learned_curves: tuple[G1TargetCurveRecord, ...],
    reference_curves: tuple[G1TargetReferenceCurveRecord, ...],
    fixed_selections: tuple[OuterFixedBridgeSelection, ...],
) -> tuple[G1TargetStopOutcome, ...]:
    if (
        type(trajectories) is not tuple
        or not trajectories
        or any(type(row) is not G1TargetTrajectoryRecord for row in trajectories)
        or type(learned_curves) is not tuple
        or any(type(row) is not G1TargetCurveRecord for row in learned_curves)
        or type(reference_curves) is not tuple
        or any(
            type(row) is not G1TargetReferenceCurveRecord for row in reference_curves
        )
        or type(fixed_selections) is not tuple
        or any(type(row) is not OuterFixedBridgeSelection for row in fixed_selections)
    ):
        raise G1TargetStoppingError("target STOP materialization roster is invalid")
    proposed_trajectories = {
        (row.outer_target, row.specimen_id, row.task): row
        for row in trajectories
        if row.variant is TargetPolicyVariant.PROPOSED
    }
    stop_trajectories = {
        (row.outer_target, row.specimen_id, row.task): row
        for row in trajectories
        if row.variant is TargetPolicyVariant.PROPOSED_STOP
    }
    learned = {
        (row.outer_target, row.specimen_id, row.task, row.variant): row
        for row in learned_curves
        if row.variant in (
            TargetPolicyVariant.PROPOSED,
            TargetPolicyVariant.PROPOSED_STOP,
        )
    }
    selections = {
        (row.outer_target, row.task): row for row in fixed_selections
    }
    references = {
        (row.outer_target, row.specimen_id, row.task, row.method): row
        for row in reference_curves
    }
    keys = tuple(sorted(proposed_trajectories, key=lambda row: (row[0], row[1], row[2].value)))
    if (
        not keys
        or len(proposed_trajectories)
        != len(
            [row for row in trajectories if row.variant is TargetPolicyVariant.PROPOSED]
        )
        or len(stop_trajectories)
        != len(
            [
                row
                for row in trajectories
                if row.variant is TargetPolicyVariant.PROPOSED_STOP
            ]
        )
        or len(selections) != len({(domain, task) for domain, _specimen, task in keys})
    ):
        raise G1TargetStoppingError("target STOP trajectory roster changed")
    output = []
    for key in keys:
        domain, specimen_id, task = key
        proposed = proposed_trajectories[key]
        stop = stop_trajectories.get(key)
        stop_authorized = stop is not None
        deployed = proposed if stop is None else stop
        proposed_curve = learned.get((*key, TargetPolicyVariant.PROPOSED))
        deployed_variant = (
            TargetPolicyVariant.PROPOSED
            if stop is None
            else TargetPolicyVariant.PROPOSED_STOP
        )
        deployed_curve = learned.get((*key, deployed_variant))
        selection = selections.get((domain, task))
        reference = (
            None
            if selection is None
            else references.get((domain, specimen_id, task, selection.method))
        )
        if (
            proposed_curve is None
            or deployed_curve is None
            or reference is None
            or proposed_curve.target_record_sha256 != proposed.state_sha256
            or proposed_curve.trajectory_sha256
            != proposed.trajectory.state_sha256
            or deployed_curve.target_record_sha256 != deployed.state_sha256
            or deployed_curve.trajectory_sha256 != deployed.trajectory.state_sha256
            or proposed.trajectory.stop_threshold is not None
            or proposed.trajectory.stopped
            or (stop is not None and stop.trajectory.stop_threshold is None)
            or (stop is not None and stop.trajectory.action_history != proposed.trajectory.action_history[: len(stop.trajectory.action_history)])
            or deployed_curve.curve.grid_sha256 != reference.curve.grid_sha256
            or deployed_curve.curve.evaluator_sha256
            != reference.curve.evaluator_sha256
            or deployed_curve.curve.warm_start_sha256
            != reference.curve.warm_start_sha256
            or deployed_curve.bank_seal_sha256 != reference.bank_seal_sha256
        ):
            raise G1TargetStoppingError("target STOP curve pairing changed")
        loss = float(deployed_curve.curve.task_losses[-1])
        reference_loss = float(reference.curve.task_losses[-1])
        ratio = (
            loss / reference_loss
            if reference_loss > 0.0
            else (1.0 if loss == 0.0 else None)
        )
        budget = float(deployed.trajectory.effective_budget)
        saving = max(0.0, 1.0 - budget / 0.25) if stop_authorized else 0.0
        stopped = deployed.trajectory.stopped
        premature = bool(
            stop_authorized
            and stopped
            and loss > 1.05 * reference_loss
        )
        false_continue = (
            None
            if not stop_authorized
            else bool(not stopped and loss <= 1.05 * reference_loss)
        )
        output.append(
            G1TargetStopOutcome(
                outer_target=domain,
                specimen_id=specimen_id,
                specimen_sha256=proposed.specimen_sha256,
                task=task,
                stop_authorized=stop_authorized,
                threshold=(
                    None if stop is None else stop.trajectory.stop_threshold
                ),
                stopped=stopped,
                budget=budget,
                normalized_measurement_saving=saving,
                task_loss=loss,
                reference_task_loss=reference_loss,
                task_loss_ratio=ratio,
                premature_stop=premature,
                false_continue=false_continue,
                proposed_trajectory_sha256=proposed.trajectory.state_sha256,
                deployed_trajectory_sha256=deployed.trajectory.state_sha256,
                deployed_curve_sha256=deployed_curve.curve.state_sha256,
                reference_curve_sha256=reference.curve.state_sha256,
            )
        )
    return tuple(output)


def _equal_domain_mean(
    records: tuple[G1TargetStopOutcome, ...],
    value: object,
) -> float:
    domains = tuple(sorted({row.outer_target for row in records}))
    return float(
        np.mean(
            [
                np.mean([float(value(row)) for row in records if row.outer_target == domain])
                for domain in domains
            ],
            dtype=np.float64,
        )
    )


def _task_analysis(
    outcomes: tuple[G1TargetStopOutcome, ...],
    task: InspectionTask,
) -> G1TaskStopAnalysis:
    records = tuple(row for row in outcomes if row.task is task)
    keys = tuple((row.outer_target, row.specimen_id) for row in records)
    savings = np.asarray(
        [row.normalized_measurement_saving for row in records], dtype=np.float64
    )
    bootstrap = formal_synchronized_bootstrap(
        dataset_ids=tuple(domain for domain, _specimen in keys),
        specimen_ids=tuple(specimen for _domain, specimen in keys),
        baseline_values=np.ones(len(records), dtype=np.float64),
        learned_values=1.0 - savings,
        seed=FORMAL_BOOTSTRAP_SEED,
    )
    saving = _equal_domain_mean(records, lambda row: row.normalized_measurement_saving)
    premature = _equal_domain_mean(records, lambda row: float(row.premature_stop))
    stopped = _equal_domain_mean(records, lambda row: float(row.stopped))
    ratios = tuple(row.task_loss_ratio for row in records)
    ratio = (
        None
        if any(value is None for value in ratios)
        else _equal_domain_mean(records, lambda row: float(row.task_loss_ratio))
    )
    authorized = tuple(row for row in records if row.stop_authorized)
    authorized_domains = len({row.outer_target for row in authorized})
    false_continue = (
        None
        if not authorized
        else _equal_domain_mean(authorized, lambda row: float(row.false_continue))
    )
    gate = (
        None
        if ratio is None
        else evaluate_stop_gate(
            StopGateEvidence(
                task=task,
                normalized_measurement_saving=saving,
                task_loss_ratio=ratio,
                premature_stop_rate=premature,
                saving_bootstrap=bootstrap,
            )
        )
    )
    return G1TaskStopAnalysis(
        task=task,
        normalized_measurement_saving=saving,
        task_loss_ratio=ratio,
        premature_stop_rate=premature,
        false_continue_rate=false_continue,
        fraction_stopped=stopped,
        fraction_never_stopped=1.0 - stopped,
        authorized_domains=authorized_domains,
        saving_bootstrap=bootstrap,
        gate=gate,
        status=(gate.status if gate is not None else f"G1_{task.value}_STOPPING_NO_GO"),
    )


def analyze_g1_target_stopping(
    outcomes: tuple[G1TargetStopOutcome, ...],
) -> G1TargetStoppingAnalysis:
    if (
        type(outcomes) is not tuple
        or not outcomes
        or any(type(row) is not G1TargetStopOutcome for row in outcomes)
    ):
        raise G1TargetStoppingError("target STOP outcome roster is invalid")
    ordered = tuple(
        sorted(outcomes, key=lambda row: (row.outer_target, row.specimen_id, row.task.value))
    )
    domains = tuple(sorted({row.outer_target for row in ordered}))
    keys = {(row.outer_target, row.specimen_id, row.task) for row in ordered}
    specimens = {(row.outer_target, row.specimen_id) for row in ordered}
    if (
        len(domains) != 6
        or len(keys) != len(ordered)
        or keys
        != {
            (domain, specimen, task)
            for domain, specimen in specimens
            for task in (InspectionTask.FIELD, InspectionTask.CAI)
        }
        or any(
            len(
                {
                    row.stop_authorized
                    for row in ordered
                    if row.outer_target == domain and row.task is task
                }
            )
            != 1
            for domain in domains
            for task in (InspectionTask.FIELD, InspectionTask.CAI)
        )
    ):
        raise G1TargetStoppingError("target STOP outcome roster changed")
    return G1TargetStoppingAnalysis(
        field=_task_analysis(ordered, InspectionTask.FIELD),
        cai=_task_analysis(ordered, InspectionTask.CAI),
        outcome_sha256s=tuple(row.state_sha256 for row in ordered),
    )


__all__ = [
    "G1TargetStopOutcome",
    "G1TargetStoppingAnalysis",
    "G1TargetStoppingError",
    "G1TaskStopAnalysis",
    "analyze_g1_target_stopping",
    "materialize_g1_target_stop_outcomes",
]
