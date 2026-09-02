"""Source-only STOP labels and conservative observable threshold selection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import torch
from torch.nn import functional

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.stopping import (
    ReferenceEndpoint,
    select_strongest_fixed_reference,
)

from .teacher import SourceTeacherAuthorization

GATE_ELIGIBLE_FIXED_METHODS = (
    "RANDOM",
    "ZERO_UNIFORM",
    "CENTER_FIRST",
    "SURFACE_FOCUS",
    "SURVEY_THEN_REFINE_FIXED",
)
REGISTERED_STOP_THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
STOP_TEACHER_TOLERANCE = 0.05
MAXIMUM_PREMATURE_RATE = 0.05
MAXIMUM_TASK_LOSS_RATIO = 1.05


class G1StoppingError(ValueError):
    """Raised when STOP teacher data or source threshold evidence is invalid."""


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
class SourceFixedReference:
    authorization_sha256: str
    fit_domains: tuple[str, ...]
    method: str
    equal_domain_loss: float
    domain_losses: tuple[tuple[str, float], ...]
    state_sha256: str


def select_source_fixed_reference(
    authorization: SourceTeacherAuthorization,
    rows: tuple[ReferenceEndpoint, ...],
) -> SourceFixedReference:
    if (
        type(authorization) is not SourceTeacherAuthorization
        or type(rows) is not tuple
        or not rows
        or any(type(row) is not ReferenceEndpoint for row in rows)
        or tuple(sorted({row.dataset_id for row in rows}))
        != tuple(sorted(authorization.fit_domains))
        or authorization.outer_target in {row.dataset_id for row in rows}
        or authorization.labeled_domain in {row.dataset_id for row in rows}
    ):
        raise G1StoppingError("STOP reference must use exactly the other four source domains")
    selection = select_strongest_fixed_reference(
        rows,
        allowed_methods=GATE_ELIGIBLE_FIXED_METHODS,
    )
    payload = {
        "schema": 1,
        "kind": "g1-source-fixed-reference",
        "authorization": authorization.state_sha256,
        "fit_domains": authorization.fit_domains,
        "method": selection.method,
        "equal_domain_loss": selection.equal_domain_loss,
        "domain_losses": selection.domain_losses,
    }
    return SourceFixedReference(
        authorization_sha256=authorization.state_sha256,
        fit_domains=authorization.fit_domains,
        method=selection.method,
        equal_domain_loss=selection.equal_domain_loss,
        domain_losses=selection.domain_losses,
        state_sha256=_json_sha(payload),
    )


@dataclass(frozen=True, slots=True)
class SourceStopLabel:
    authorization_sha256: str
    fixed_reference_sha256: str
    source_domain: str
    specimen_sha256: str
    task: InspectionTask
    policy_state_sha256: str
    reference_method: str
    current_true_loss: float
    reference_true_loss: float
    tolerance: float
    is_sufficient: bool
    state_sha256: str = ""

    def __post_init__(self) -> None:
        current = float(self.current_true_loss)
        reference = float(self.reference_true_loss)
        tolerance = float(self.tolerance)
        sufficient = current <= (1.0 + tolerance) * reference
        if (
            not _valid_sha256(self.authorization_sha256)
            or not _valid_sha256(self.fixed_reference_sha256)
            or type(self.source_domain) is not str
            or not self.source_domain
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or not _valid_sha256(self.policy_state_sha256)
            or self.reference_method not in GATE_ELIGIBLE_FIXED_METHODS
            or not math.isfinite(current)
            or current < 0.0
            or not math.isfinite(reference)
            or reference < 0.0
            or tolerance != STOP_TEACHER_TOLERANCE
            or type(self.is_sufficient) is not bool
            or self.is_sufficient != sufficient
        ):
            raise G1StoppingError("source STOP label is invalid")
        payload = {
            "schema": 1,
            "kind": "g1-source-stop-label",
            "authorization": self.authorization_sha256,
            "fixed_reference": self.fixed_reference_sha256,
            "source_domain": self.source_domain,
            "specimen_sha256": self.specimen_sha256,
            "task": self.task.value,
            "policy_state_sha256": self.policy_state_sha256,
            "reference_method": self.reference_method,
            "current_true_loss": current,
            "reference_true_loss": reference,
            "tolerance": tolerance,
            "is_sufficient": sufficient,
        }
        state = _json_sha(payload)
        if self.state_sha256 not in ("", state):
            raise G1StoppingError("source STOP label hash changed")
        object.__setattr__(self, "current_true_loss", current)
        object.__setattr__(self, "reference_true_loss", reference)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "state_sha256", state)


def build_source_stop_label(
    authorization: SourceTeacherAuthorization,
    fixed_reference: SourceFixedReference,
    *,
    source_domain: str,
    specimen_sha256: str,
    task: InspectionTask,
    policy_state_sha256: str,
    current_true_loss: float,
    reference_true_loss: float,
) -> SourceStopLabel:
    current = float(current_true_loss)
    reference = float(reference_true_loss)
    if (
        type(authorization) is not SourceTeacherAuthorization
        or type(fixed_reference) is not SourceFixedReference
        or fixed_reference.authorization_sha256 != authorization.state_sha256
        or source_domain != authorization.labeled_domain
        or source_domain == authorization.outer_target
        or not _valid_sha256(specimen_sha256)
        or task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or not _valid_sha256(policy_state_sha256)
        or not math.isfinite(current)
        or current < 0.0
        or not math.isfinite(reference)
        or reference < 0.0
    ):
        raise G1StoppingError("source STOP label request is invalid")
    sufficient = current <= (1.0 + STOP_TEACHER_TOLERANCE) * reference
    payload = {
        "schema": 1,
        "kind": "g1-source-stop-label",
        "authorization": authorization.state_sha256,
        "fixed_reference": fixed_reference.state_sha256,
        "source_domain": source_domain,
        "specimen_sha256": specimen_sha256,
        "task": task.value,
        "policy_state_sha256": policy_state_sha256,
        "reference_method": fixed_reference.method,
        "current_true_loss": current,
        "reference_true_loss": reference,
        "tolerance": STOP_TEACHER_TOLERANCE,
        "is_sufficient": sufficient,
    }
    return SourceStopLabel(
        authorization_sha256=authorization.state_sha256,
        fixed_reference_sha256=fixed_reference.state_sha256,
        source_domain=source_domain,
        specimen_sha256=specimen_sha256,
        task=task,
        policy_state_sha256=policy_state_sha256,
        reference_method=fixed_reference.method,
        current_true_loss=current,
        reference_true_loss=reference,
        tolerance=STOP_TEACHER_TOLERANCE,
        is_sufficient=sufficient,
        state_sha256=_json_sha(payload),
    )


def observable_stop_loss(
    stop_logits: torch.Tensor,
    sufficient_labels: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        not isinstance(stop_logits, torch.Tensor)
        or not isinstance(sufficient_labels, torch.Tensor)
        or stop_logits.ndim != 1
        or sufficient_labels.shape != stop_logits.shape
        or sufficient_labels.dtype is not torch.bool
        or not stop_logits.is_floating_point()
        or not torch.isfinite(stop_logits).all()
    ):
        raise G1StoppingError("observable STOP tensors are invalid")
    losses = functional.binary_cross_entropy_with_logits(
        stop_logits,
        sufficient_labels.to(dtype=stop_logits.dtype),
        reduction="none",
    )
    if sample_weights is None:
        return losses.mean()
    if (
        not isinstance(sample_weights, torch.Tensor)
        or sample_weights.shape != losses.shape
        or sample_weights.device != losses.device
        or not sample_weights.is_floating_point()
        or not torch.isfinite(sample_weights).all()
        or torch.any(sample_weights <= 0.0)
    ):
        raise G1StoppingError("observable STOP sample weights are invalid")
    return torch.sum(losses * sample_weights) / torch.sum(sample_weights)


@dataclass(frozen=True, slots=True)
class SourceStopValidationTrajectory:
    outer_target: str
    source_domain: str
    specimen_sha256: str
    task: InspectionTask
    budgets: tuple[float, ...]
    stop_probabilities: tuple[float, ...]
    true_task_losses: tuple[float, ...]
    reference_true_loss: float
    endpoint_budget: float
    state_sha256: str = ""

    def __post_init__(self) -> None:
        budgets = tuple(float(value) for value in self.budgets)
        probabilities = tuple(float(value) for value in self.stop_probabilities)
        losses = tuple(float(value) for value in self.true_task_losses)
        reference = float(self.reference_true_loss)
        endpoint = float(self.endpoint_budget)
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or not budgets
            or len(probabilities) != len(budgets)
            or len(losses) != len(budgets)
            or any(not math.isfinite(value) or value < 0.0 for value in budgets)
            or any(right <= left for left, right in pairwise(budgets))
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
            or any(not math.isfinite(value) or value < 0.0 for value in losses)
            or not math.isfinite(reference)
            or reference < 0.0
            or not math.isfinite(endpoint)
            or endpoint != 0.25
            or budgets[-1] > endpoint + 1.0e-15
        ):
            raise G1StoppingError("source STOP validation trajectory is invalid")
        payload = {
            "schema": 1,
            "kind": "g1-source-stop-validation-trajectory",
            "outer_target": self.outer_target,
            "source_domain": self.source_domain,
            "specimen_sha256": self.specimen_sha256,
            "task": self.task.value,
            "budgets": budgets,
            "stop_probabilities": probabilities,
            "true_task_losses": losses,
            "reference_true_loss": reference,
            "endpoint_budget": endpoint,
        }
        state = _json_sha(payload)
        if self.state_sha256 not in ("", state):
            raise G1StoppingError("source STOP validation trajectory hash changed")
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "stop_probabilities", probabilities)
        object.__setattr__(self, "true_task_losses", losses)
        object.__setattr__(self, "reference_true_loss", reference)
        object.__setattr__(self, "endpoint_budget", endpoint)
        object.__setattr__(self, "state_sha256", state)


@dataclass(frozen=True, slots=True)
class StopThresholdCandidate:
    threshold: float
    equal_domain_saving: float
    equal_domain_premature_rate: float
    equal_domain_task_loss_ratio: float | None
    equal_domain_stopped_fraction: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class StopThresholdSelection:
    outer_target: str
    task: InspectionTask
    status: str
    threshold: float | None
    candidates: tuple[StopThresholdCandidate, ...]
    trajectory_sha256: tuple[str, ...]
    state_sha256: str


def _trajectory_at_threshold(
    row: SourceStopValidationTrajectory,
    threshold: float,
) -> tuple[float, float, float | None, float]:
    indices = tuple(
        index
        for index, probability in enumerate(row.stop_probabilities)
        if probability >= threshold
    )
    signaled = bool(indices)
    index = indices[0] if signaled else len(row.budgets) - 1
    budget = row.budgets[index]
    loss = row.true_task_losses[index]
    early = signaled and budget < row.endpoint_budget - 1.0e-15
    saving = max(0.0, 1.0 - budget / row.endpoint_budget) if signaled else 0.0
    premature = float(
        early and loss > (1.0 + STOP_TEACHER_TOLERANCE) * row.reference_true_loss
    )
    if row.reference_true_loss > 0.0:
        ratio: float | None = loss / row.reference_true_loss
    else:
        ratio = 1.0 if loss == 0.0 else None
    return saving, premature, ratio, float(signaled)


def select_conservative_stop_threshold(
    trajectories: tuple[SourceStopValidationTrajectory, ...],
) -> StopThresholdSelection:
    if (
        type(trajectories) is not tuple
        or not trajectories
        or any(type(row) is not SourceStopValidationTrajectory for row in trajectories)
        or len({row.state_sha256 for row in trajectories}) != len(trajectories)
        or len({row.outer_target for row in trajectories}) != 1
        or len({row.task for row in trajectories}) != 1
    ):
        raise G1StoppingError("source STOP threshold evidence is invalid")
    outer_target = trajectories[0].outer_target
    if any(row.source_domain == outer_target for row in trajectories):
        raise G1StoppingError("outer target is forbidden from STOP threshold selection")
    domains = tuple(sorted({row.source_domain for row in trajectories}))
    if len(domains) != 5:
        raise G1StoppingError("STOP threshold selection requires five source domains")
    candidates = []
    for threshold in REGISTERED_STOP_THRESHOLDS:
        domain_metrics = []
        for domain in domains:
            outputs = tuple(
                _trajectory_at_threshold(row, threshold)
                for row in trajectories
                if row.source_domain == domain
            )
            savings, premature, ratios, stopped = zip(*outputs, strict=True)
            domain_ratio = (
                None
                if any(value is None for value in ratios)
                else float(np.mean(ratios, dtype=np.float64))
            )
            domain_metrics.append(
                (
                    float(np.mean(savings, dtype=np.float64)),
                    float(np.mean(premature, dtype=np.float64)),
                    domain_ratio,
                    float(np.mean(stopped, dtype=np.float64)),
                )
            )
        saving = float(np.mean([row[0] for row in domain_metrics], dtype=np.float64))
        premature_rate = float(
            np.mean([row[1] for row in domain_metrics], dtype=np.float64)
        )
        ratio = (
            None
            if any(row[2] is None for row in domain_metrics)
            else float(np.mean([row[2] for row in domain_metrics], dtype=np.float64))
        )
        stopped_fraction = float(
            np.mean([row[3] for row in domain_metrics], dtype=np.float64)
        )
        feasible = (
            premature_rate <= MAXIMUM_PREMATURE_RATE
            and ratio is not None
            and ratio <= MAXIMUM_TASK_LOSS_RATIO
        )
        candidates.append(
            StopThresholdCandidate(
                threshold=threshold,
                equal_domain_saving=saving,
                equal_domain_premature_rate=premature_rate,
                equal_domain_task_loss_ratio=ratio,
                equal_domain_stopped_fraction=stopped_fraction,
                feasible=feasible,
            )
        )
    feasible = tuple(candidate for candidate in candidates if candidate.feasible)
    selected = (
        max(feasible, key=lambda row: (row.equal_domain_saving, row.threshold))
        if feasible
        else None
    )
    status = "STOP_AUTHORIZED_SOURCE_ONLY" if selected is not None else "STOP_NOT_AUTHORIZED"
    payload = {
        "schema": 1,
        "kind": "g1-stop-threshold-selection",
        "outer_target": outer_target,
        "task": trajectories[0].task.value,
        "status": status,
        "threshold": None if selected is None else selected.threshold,
        "candidates": [
            (
                row.threshold,
                row.equal_domain_saving,
                row.equal_domain_premature_rate,
                row.equal_domain_task_loss_ratio,
                row.equal_domain_stopped_fraction,
                row.feasible,
            )
            for row in candidates
        ],
        "trajectories": tuple(row.state_sha256 for row in trajectories),
    }
    return StopThresholdSelection(
        outer_target=outer_target,
        task=trajectories[0].task,
        status=status,
        threshold=None if selected is None else selected.threshold,
        candidates=tuple(candidates),
        trajectory_sha256=tuple(row.state_sha256 for row in trajectories),
        state_sha256=_json_sha(payload),
    )


__all__ = [
    "GATE_ELIGIBLE_FIXED_METHODS",
    "MAXIMUM_PREMATURE_RATE",
    "MAXIMUM_TASK_LOSS_RATIO",
    "REGISTERED_STOP_THRESHOLDS",
    "STOP_TEACHER_TOLERANCE",
    "G1StoppingError",
    "SourceFixedReference",
    "SourceStopLabel",
    "SourceStopValidationTrajectory",
    "StopThresholdCandidate",
    "StopThresholdSelection",
    "build_source_stop_label",
    "observable_stop_loss",
    "select_conservative_stop_threshold",
    "select_source_fixed_reference",
]
