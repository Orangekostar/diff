"""Causal uniform-scout then feedback-driven exact-cost rollout."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

import numpy as np

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    RefinementAction,
    fitting_actions,
    measurement_mask,
)

from .authority import MAVISAuthority
from .contracts import InspectionState
from .dynamic_voi import CandidateDescriptor
from .policy import PolicySelection, select_cost_aware_action
from .reveal import reveal_action, reveal_action_history, reveal_uniform_scout


class MAVISRolloutError(ValueError):
    """Raised when a causal rollout request or transition is invalid."""


class DeployableRolloutScorer(Protocol):
    def score_actions(
        self,
        state: InspectionState,
        candidates: tuple[CandidateDescriptor, ...],
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class RolloutStep:
    step: int
    nominal_checkpoint: float
    action: RefinementAction
    candidate: CandidateDescriptor
    raw_score: float
    objective_score: float
    decision_confidence: float
    exact_cost_before: int
    exact_cost_after: int
    state_sha256_before: str
    state_sha256_after: str
    feedback_used: bool


@dataclass(frozen=True, slots=True)
class ScoutAndFocusRollout:
    specimen_id: str
    initial_budget: float
    endpoint_budget: float
    objective: str
    feedback: bool
    states: tuple[InspectionState, ...]
    steps: tuple[RolloutStep, ...]
    state_sha256: str

    @property
    def final_state(self) -> InspectionState:
        return self.states[-1]


@dataclass(frozen=True, slots=True)
class ScoutAndFocusCurve:
    specimen_id: str
    initial_budget: float
    checkpoints: tuple[float, ...]
    objective: str
    feedback: bool
    checkpoint_states: tuple[InspectionState, ...]
    scoring_states: tuple[InspectionState, ...]
    steps: tuple[RolloutStep, ...]
    state_sha256: str


def _candidate_descriptors(
    state: InspectionState,
    *,
    endpoint_budget: float,
    action_budget: float | None = None,
) -> tuple[tuple[RefinementAction, ...], tuple[CandidateDescriptor, ...]]:
    grid = build_acquisition_grid(
        *state.native_shape,
        initial_budget=state.initial_budget,
    )
    cap = endpoint_budget if action_budget is None else action_budget
    actions = fitting_actions(grid, state.measurement_state, cap)
    endpoint_count = math.floor(endpoint_budget * state.native_count)
    remaining = endpoint_count - state.exact_acquired_count
    acquired = measurement_mask(grid, state.measurement_state)
    descriptors: list[CandidateDescriptor] = []
    for action in actions:
        cell = grid.cells[action.cell_index]
        rows = np.asarray(cell.rows[action.to_level], dtype=np.int64)
        columns = np.asarray(cell.columns[action.to_level], dtype=np.int64)
        added = int(np.count_nonzero(~acquired[np.ix_(rows, columns)]))
        descriptors.append(
            CandidateDescriptor(
                cell_index=action.cell_index,
                from_level=action.from_level,
                to_level=action.to_level,
                exact_added_cost=added,
                native_count=state.native_count,
                remaining_cost=remaining,
            )
        )
    return actions, tuple(descriptors)


def _scores(value: object, count: int, *, objective: str) -> np.ndarray:
    if objective == "direct_cost_aware":
        raw = getattr(value, "scores", value)
    else:
        raw = getattr(value, "value_predictions", value)
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    scores = np.asarray(raw, dtype=np.float64)
    if scores.shape != (count,) or not np.all(np.isfinite(scores)):
        raise MAVISRolloutError("rollout scorer output is invalid")
    return scores


def _decision_confidence(
    candidates: tuple[CandidateDescriptor, ...],
    scores: np.ndarray,
    *,
    objective: str,
) -> float:
    objective_scores = (
        scores
        if objective != "value_per_exact_cost"
        else scores
        / np.asarray(
            [candidate.exact_added_cost for candidate in candidates],
            dtype=np.float64,
        )
    )
    if objective_scores.size == 1:
        return 1.0
    ordered = np.sort(objective_scores)
    best = float(ordered[-1])
    second = float(ordered[-2])
    scale = abs(best) + abs(second)
    if scale <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip((best - second) / scale, 0.0, 1.0))


def rollout_scout_and_focus(
    authority: MAVISAuthority,
    *,
    specimen_id: str,
    initial_budget: float,
    endpoint_budget: float,
    scorer: DeployableRolloutScorer,
    objective: str,
    feedback: bool,
) -> ScoutAndFocusRollout:
    curve, states = _rollout_curve(
        authority,
        specimen_id=specimen_id,
        initial_budget=initial_budget,
        checkpoints=(endpoint_budget,),
        scorer=scorer,
        objective=objective,
        feedback=feedback,
    )
    return ScoutAndFocusRollout(
        specimen_id=curve.specimen_id,
        initial_budget=curve.initial_budget,
        endpoint_budget=curve.checkpoints[-1],
        objective=curve.objective,
        feedback=curve.feedback,
        states=states,
        steps=curve.steps,
        state_sha256=curve.state_sha256,
    )


def rollout_scout_and_focus_curve(
    authority: MAVISAuthority,
    *,
    specimen_id: str,
    initial_budget: float,
    checkpoints: tuple[float, ...],
    scorer: DeployableRolloutScorer,
    objective: str,
    feedback: bool,
) -> ScoutAndFocusCurve:
    curve, _states = _rollout_curve(
        authority,
        specimen_id=specimen_id,
        initial_budget=initial_budget,
        checkpoints=checkpoints,
        scorer=scorer,
        objective=objective,
        feedback=feedback,
    )
    return curve


def _rollout_curve(
    authority: MAVISAuthority,
    *,
    specimen_id: str,
    initial_budget: float,
    checkpoints: tuple[float, ...],
    scorer: DeployableRolloutScorer,
    objective: str,
    feedback: bool,
) -> tuple[ScoutAndFocusCurve, tuple[InspectionState, ...]]:
    initial = float(initial_budget)
    if type(checkpoints) is not tuple or not checkpoints:
        raise MAVISRolloutError("rollout checkpoint roster is invalid")
    caps = tuple(float(value) for value in checkpoints)
    endpoint = caps[-1]
    if (
        type(authority) is not MAVISAuthority
        or type(specimen_id) is not str
        or not specimen_id
        or isinstance(initial_budget, bool)
        or any(isinstance(value, bool) for value in checkpoints)
        or not math.isfinite(initial)
        or any(not math.isfinite(value) for value in caps)
        or not 0.0 < initial <= caps[0]
        or endpoint > 1.0
        or any(second <= first for first, second in pairwise(caps))
        or not hasattr(scorer, "score_actions")
        or type(feedback) is not bool
    ):
        raise MAVISRolloutError("rollout request is invalid")
    current = reveal_uniform_scout(
        authority,
        authority.policy_context(specimen_id),
        initial_budget=initial,
        checkpoint=endpoint,
    )
    states = [current]
    checkpoint_states: list[InspectionState] = []
    scoring_states: list[InspectionState] = []
    steps: list[RolloutStep] = []
    frozen_cell_scores: dict[int, float] | None = None
    for checkpoint_index, checkpoint in enumerate(caps):
        if checkpoint_index or current.effective_budget <= checkpoint + 1.0e-15:
            current = reveal_action_history(
                authority,
                authority.policy_context(specimen_id),
                initial_budget=initial,
                checkpoint=checkpoint,
                actions=current.action_history,
            )
        while True:
            actions, candidates = _candidate_descriptors(
                current,
                endpoint_budget=endpoint,
                action_budget=checkpoint,
            )
            if not candidates:
                break
            if len(steps) >= 128:
                raise MAVISRolloutError("rollout exceeded the finite action roster")
            if feedback:
                scoring_states.append(current)
                score_values = _scores(
                    scorer.score_actions(current, candidates),
                    len(candidates),
                    objective=objective,
                )
            elif frozen_cell_scores is None:
                scoring_states.append(current)
                _all_actions, all_candidates = _candidate_descriptors(
                    current,
                    endpoint_budget=endpoint,
                    action_budget=endpoint,
                )
                initial_scores = _scores(
                    scorer.score_actions(current, all_candidates),
                    len(all_candidates),
                    objective=objective,
                )
                frozen_cell_scores = {
                    candidate.cell_index: float(score)
                    for candidate, score in zip(
                        all_candidates,
                        initial_scores,
                        strict=True,
                    )
                }
                score_values = np.asarray(
                    [frozen_cell_scores[candidate.cell_index] for candidate in candidates],
                    dtype=np.float64,
                )
            else:
                if any(
                    candidate.cell_index not in frozen_cell_scores
                    for candidate in candidates
                ):
                    raise MAVISRolloutError("no-feedback cell roster changed")
                score_values = np.asarray(
                    [frozen_cell_scores[candidate.cell_index] for candidate in candidates],
                    dtype=np.float64,
                )
            selection: PolicySelection = select_cost_aware_action(
                candidates,
                score_values,
                objective=objective,
            )
            confidence = _decision_confidence(
                candidates,
                score_values,
                objective=objective,
            )
            action = actions[selection.candidate_index]
            next_state = reveal_action(authority, current, action)
            if (
                next_state.exact_acquired_count - current.exact_acquired_count
                != selection.candidate.exact_added_cost
                or next_state.effective_budget > checkpoint + 1.0e-15
            ):
                raise MAVISRolloutError("rollout exact-cost transition changed")
            steps.append(
                RolloutStep(
                    step=len(steps),
                    nominal_checkpoint=checkpoint,
                    action=action,
                    candidate=selection.candidate,
                    raw_score=selection.raw_score,
                    objective_score=selection.objective_score,
                    decision_confidence=confidence,
                    exact_cost_before=current.exact_acquired_count,
                    exact_cost_after=next_state.exact_acquired_count,
                    state_sha256_before=current.state_sha256,
                    state_sha256_after=next_state.state_sha256,
                    feedback_used=feedback,
                )
            )
            current = next_state
            states.append(current)
        checkpoint_states.append(current)
    payload = {
        "schema": 1,
        "specimen_id": specimen_id,
        "initial_budget": initial,
        "checkpoints": caps,
        "objective": objective,
        "feedback": feedback,
        "states": [state.state_sha256 for state in checkpoint_states],
        "scoring_states": [state.state_sha256 for state in scoring_states],
        "actions": [
            (step.action.cell_index, step.action.from_level, step.action.to_level)
            for step in steps
        ],
        "decision_confidences": [step.decision_confidence for step in steps],
    }
    state_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    curve = ScoutAndFocusCurve(
        specimen_id=specimen_id,
        initial_budget=initial,
        checkpoints=caps,
        objective=objective,
        feedback=feedback,
        checkpoint_states=tuple(checkpoint_states),
        scoring_states=tuple(scoring_states),
        steps=tuple(steps),
        state_sha256=state_sha256,
    )
    return curve, tuple(states)


__all__ = [
    "DeployableRolloutScorer",
    "MAVISRolloutError",
    "RolloutStep",
    "ScoutAndFocusCurve",
    "ScoutAndFocusRollout",
    "rollout_scout_and_focus",
    "rollout_scout_and_focus_curve",
]
