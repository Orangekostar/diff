"""Source-only privileged one-step labels for observable G1 actor states."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from cmc_bbdm.inspection_agent.contracts import (
    InspectionDecision,
    InspectionObservation,
    InspectionTask,
)
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.oracle import (
    CAIStatePredictor,
    OracleSelection,
    ReconstructionEncoder,
    choose_cai_action,
    choose_field_action,
)
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .crossfit import CrossfitRoster
from .features import canonical_slot, decision_type


class G1TeacherError(ValueError):
    """Raised when a privileged source teacher crosses its authorization boundary."""


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
class SourceTeacherAuthorization:
    outer_target: str
    labeled_domain: str
    fit_domains: tuple[str, ...]
    roster_sha256: str
    state_sha256: str

    def __post_init__(self) -> None:
        payload = {
            "schema": 1,
            "kind": "g1-source-teacher-authorization",
            "outer_target": self.outer_target,
            "labeled_domain": self.labeled_domain,
            "fit_domains": self.fit_domains,
            "roster": self.roster_sha256,
        }
        state = _json_sha(payload)
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.labeled_domain) is not str
            or not self.labeled_domain
            or self.outer_target == self.labeled_domain
            or type(self.fit_domains) is not tuple
            or len(self.fit_domains) != 4
            or len(set(self.fit_domains)) != 4
            or self.outer_target in self.fit_domains
            or self.labeled_domain in self.fit_domains
            or not _valid_sha256(self.roster_sha256)
            or self.state_sha256 != state
        ):
            raise G1TeacherError("source teacher authorization is invalid")


@dataclass(frozen=True, slots=True)
class TeacherCandidateRecord:
    slot: int
    action: InspectionCellAction
    decision: InspectionDecision
    exact_added_cost: int
    raw_value: float
    objective_value: float
    task_loss_after: float
    candidate_state_sha256: str
    selected: bool

    def __post_init__(self) -> None:
        numbers = (
            float(self.raw_value),
            float(self.objective_value),
            float(self.task_loss_after),
        )
        if (
            type(self.slot) is not int
            or type(self.action) is not InspectionCellAction
            or self.slot != canonical_slot(self.action)
            or type(self.decision) is not InspectionDecision
            or self.decision is InspectionDecision.STOP
            or (self.action.from_level >= 0 and self.decision is not InspectionDecision.REFINE)
            or (
                self.action.from_level == -1
                and self.decision not in (InspectionDecision.FOCUS, InspectionDecision.BROADEN)
            )
            or type(self.exact_added_cost) is not int
            or self.exact_added_cost <= 0
            or not all(math.isfinite(value) for value in numbers)
            or not _valid_sha256(self.candidate_state_sha256)
            or type(self.selected) is not bool
        ):
            raise G1TeacherError("teacher candidate record is invalid")
        object.__setattr__(self, "raw_value", numbers[0])
        object.__setattr__(self, "objective_value", numbers[1])
        object.__setattr__(self, "task_loss_after", numbers[2])


@dataclass(frozen=True, slots=True)
class PrivilegedTeacherLabel:
    task: InspectionTask
    authorization_sha256: str
    observation_sha256: str
    policy_state_sha256: str
    selected_slot: int
    candidates: tuple[TeacherCandidateRecord, ...]
    state_sha256: str = ""

    def __post_init__(self) -> None:
        slots = tuple(candidate.slot for candidate in self.candidates)
        selected = tuple(candidate.slot for candidate in self.candidates if candidate.selected)
        if (
            self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or not all(
                _valid_sha256(value)
                for value in (
                    self.authorization_sha256,
                    self.observation_sha256,
                    self.policy_state_sha256,
                )
            )
            or type(self.selected_slot) is not int
            or type(self.candidates) is not tuple
            or not self.candidates
            or any(type(candidate) is not TeacherCandidateRecord for candidate in self.candidates)
            or slots != tuple(sorted(slots))
            or len(set(slots)) != len(slots)
            or selected != (self.selected_slot,)
        ):
            raise G1TeacherError("privileged teacher label is invalid")
        payload = {
            "schema": 1,
            "kind": "g1-privileged-teacher-label",
            "task": self.task.value,
            "authorization": self.authorization_sha256,
            "observation": self.observation_sha256,
            "policy_state": self.policy_state_sha256,
            "selected_slot": self.selected_slot,
            "candidates": [
                (
                    candidate.slot,
                    candidate.action.cell_index,
                    candidate.action.from_level,
                    candidate.action.to_level,
                    candidate.decision.value,
                    candidate.exact_added_cost,
                    candidate.raw_value,
                    candidate.objective_value,
                    candidate.task_loss_after,
                    candidate.candidate_state_sha256,
                    candidate.selected,
                )
                for candidate in self.candidates
            ],
        }
        state = _json_sha(payload)
        if self.state_sha256 not in ("", state):
            raise G1TeacherError("privileged teacher-label hash changed")
        object.__setattr__(self, "state_sha256", state)


def authorize_source_teacher(
    roster: CrossfitRoster,
    *,
    query_domain: str,
) -> SourceTeacherAuthorization:
    if (
        type(roster) is not CrossfitRoster
        or type(query_domain) is not str
        or query_domain != roster.labeled_domain
        or query_domain == roster.outer_target
        or query_domain in roster.fit_domains
    ):
        raise G1TeacherError("teacher query must be the labeled source domain")
    payload = {
        "schema": 1,
        "kind": "g1-source-teacher-authorization",
        "outer_target": roster.outer_target,
        "labeled_domain": roster.labeled_domain,
        "fit_domains": roster.fit_domains,
        "roster": roster.state_sha256,
    }
    return SourceTeacherAuthorization(
        outer_target=roster.outer_target,
        labeled_domain=roster.labeled_domain,
        fit_domains=roster.fit_domains,
        roster_sha256=roster.state_sha256,
        state_sha256=_json_sha(payload),
    )


def _teacher_label(
    selection: OracleSelection,
    observation: InspectionObservation,
    surface_hypothesis: SurfaceHypothesis,
    authorization: SourceTeacherAuthorization,
    *,
    policy_state_sha256: str,
) -> PrivilegedTeacherLabel:
    if (
        type(selection) is not OracleSelection
        or type(observation) is not InspectionObservation
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(authorization) is not SourceTeacherAuthorization
        or not _valid_sha256(policy_state_sha256)
    ):
        raise G1TeacherError("teacher label request is invalid")
    selected_slot = canonical_slot(selection.action)
    candidates = tuple(
        sorted(
            (
                TeacherCandidateRecord(
                    slot=canonical_slot(candidate.action),
                    action=candidate.action,
                    decision=decision_type(candidate.action, surface_hypothesis),
                    exact_added_cost=candidate.exact_added_cost,
                    raw_value=candidate.raw_value,
                    objective_value=candidate.objective_value,
                    task_loss_after=candidate.task_loss_after,
                    candidate_state_sha256=candidate.candidate_state_sha256,
                    selected=canonical_slot(candidate.action) == selected_slot,
                )
                for candidate in selection.candidates
            ),
            key=lambda candidate: candidate.slot,
        )
    )
    return PrivilegedTeacherLabel(
        task=observation.task,
        authorization_sha256=authorization.state_sha256,
        observation_sha256=observation.state_sha256,
        policy_state_sha256=policy_state_sha256,
        selected_slot=selected_slot,
        candidates=candidates,
    )


def validate_source_teacher_dependencies(
    authorization: SourceTeacherAuthorization,
    prior: SourceBackgroundPrior,
    *,
    assessor: CAIStatePredictor | None = None,
) -> None:
    if (
        type(authorization) is not SourceTeacherAuthorization
        or type(prior) is not SourceBackgroundPrior
        or prior.outer_domain != authorization.outer_target
        or prior.source_domains != authorization.fit_domains
        or authorization.labeled_domain in prior.source_domains
    ):
        raise G1TeacherError("teacher dependencies do not match authorization")
    if assessor is not None and (
        tuple(getattr(assessor, "fit_domains", ())) != authorization.fit_domains
        or getattr(assessor, "outer_domain", None) != authorization.outer_target
        or not _valid_sha256(getattr(assessor, "model_state_sha256", None))
    ):
        raise G1TeacherError("CAI assessor does not match teacher authorization")


def field_teacher_label(
    observation: InspectionObservation,
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    surface_hypothesis: SurfaceHypothesis,
    authorization: SourceTeacherAuthorization,
    *,
    full_scan: object,
    policy_state_sha256: str,
) -> PrivilegedTeacherLabel:
    if type(observation) is not InspectionObservation or observation.task is not InspectionTask.FIELD:
        raise G1TeacherError("FIELD source observation is required")
    validate_source_teacher_dependencies(authorization, prior)
    selection = choose_field_action(
        observation,
        grid,
        prior,
        full_scan=full_scan,
        checkpoint=observation.endpoint_budget,
    )
    return _teacher_label(
        selection,
        observation,
        surface_hypothesis,
        authorization,
        policy_state_sha256=policy_state_sha256,
    )


def cai_teacher_label(
    observation: InspectionObservation,
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    surface_hypothesis: SurfaceHypothesis,
    authorization: SourceTeacherAuthorization,
    *,
    full_scan: object,
    true_cai: float,
    assessor: CAIStatePredictor,
    encoder: ReconstructionEncoder,
    policy_state_sha256: str,
) -> PrivilegedTeacherLabel:
    if type(observation) is not InspectionObservation or observation.task is not InspectionTask.CAI:
        raise G1TeacherError("CAI source observation is required")
    validate_source_teacher_dependencies(authorization, prior, assessor=assessor)
    selection = choose_cai_action(
        observation,
        grid,
        prior,
        full_scan=full_scan,
        true_cai=true_cai,
        assessor=assessor,
        encoder=encoder,
        checkpoint=observation.endpoint_budget,
    )
    return _teacher_label(
        selection,
        observation,
        surface_hypothesis,
        authorization,
        policy_state_sha256=policy_state_sha256,
    )


__all__ = [
    "G1TeacherError",
    "PrivilegedTeacherLabel",
    "SourceTeacherAuthorization",
    "TeacherCandidateRecord",
    "authorize_source_teacher",
    "cai_teacher_label",
    "field_teacher_label",
    "validate_source_teacher_dependencies",
]
