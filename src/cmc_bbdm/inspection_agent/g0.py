"""Registered G0 component gates and final decision vocabulary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


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


__all__ = [
    "FinalG0Evidence",
    "G0Status",
    "GateEvidence",
    "assessor_authorization_gate",
    "decide_g0_status",
    "hierarchical_headroom_gate",
    "initialization_headroom_gate",
    "stopping_headroom_gate",
    "task_conditioning_gate",
]
