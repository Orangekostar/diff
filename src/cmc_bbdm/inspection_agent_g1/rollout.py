"""Observable closed-loop rollout and post-trajectory target-truth sealing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from cmc_bbdm.agentic_nde.surface_cells import (
    shuffled_surface_donors as _frozen_shuffled_surface_donors,
)
from cmc_bbdm.inspection_agent.contracts import InspectionObservation, InspectionTask
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .contracts import ACTION_SLOT_COUNT, G1PolicyState
from .features import canonical_action_from_slot
from .stopping_policy import REGISTERED_STOP_THRESHOLDS
from .warm_start import PRIMARY_WARM_START_CELLS


class G1RolloutError(ValueError):
    """Raised when observable policy rollout violates the causal contract."""


class TargetTruthVaultError(PermissionError):
    """Raised when hidden target truth is requested before a matching seal."""


class SurfaceVariant(str, Enum):
    CORRECT_SURFACE = "CORRECT_SURFACE"
    NO_SURFACE = "NO_SURFACE"
    SHUFFLED_SURFACE = "SHUFFLED_SURFACE"


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


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape:
        raise G1RolloutError("rollout array shape is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def shuffled_surface_donors(
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    *,
    seed: int,
) -> tuple[str, ...]:
    if type(seed) is not int:
        raise G1RolloutError("shuffled-surface seed is invalid")
    try:
        return _frozen_shuffled_surface_donors(
            specimen_ids,
            dataset_ids,
            seed=f"inspection-agent-g1|{seed}",
        )
    except (TypeError, ValueError) as error:
        raise G1RolloutError("shuffled-surface roster is invalid") from error


def controlled_surface_hypothesis(
    recipient: SurfaceHypothesis,
    variant: SurfaceVariant,
    *,
    donor: SurfaceHypothesis | None = None,
) -> SurfaceHypothesis:
    if type(recipient) is not SurfaceHypothesis or type(variant) is not SurfaceVariant:
        raise G1RolloutError("surface-control request is invalid")
    if variant is SurfaceVariant.CORRECT_SURFACE:
        if donor is not None:
            raise G1RolloutError("correct surface accepts no donor")
        return recipient
    if variant is SurfaceVariant.NO_SURFACE:
        if donor is not None:
            raise G1RolloutError("no-surface control accepts no donor")
        scores = np.zeros(64, dtype=np.float64)
        top_cells: tuple[int, ...] = ()
        donor_sha256 = None
    else:
        if (
            type(donor) is not SurfaceHypothesis
            or donor.state_sha256 == recipient.state_sha256
        ):
            raise G1RolloutError("shuffled surface requires a nonself donor")
        scores = np.asarray(donor.scores, dtype=np.float64)
        top_cells = donor.top_cells
        donor_sha256 = donor.state_sha256
    frozen_scores = _readonly(scores, dtype="<f8", shape=(64,))
    frozen_median = _readonly(
        recipient.border_median_rgb,
        dtype="<f8",
        shape=(3,),
    )
    digest = hashlib.sha256()
    digest.update(b"inspection-agent-g1-surface-control-v1")
    digest.update(
        json.dumps(
            {
                "recipient": recipient.state_sha256,
                "variant": variant.value,
                "donor": donor_sha256,
                "top_cells": top_cells,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(frozen_scores.tobytes(order="C"))
    digest.update(frozen_median.tobytes(order="C"))
    return SurfaceHypothesis(
        scores=frozen_scores,
        top_cells=top_cells,
        border_median_rgb=frozen_median,
        state_sha256=digest.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class ObservablePolicyScores:
    policy_state_sha256: str
    model_sha256: str
    action_logits: np.ndarray
    stop_probability: float
    state_sha256: str = ""

    def __post_init__(self) -> None:
        probability = float(self.stop_probability)
        logits = _readonly(
            self.action_logits,
            dtype="<f8",
            shape=(ACTION_SLOT_COUNT,),
        )
        if (
            not _valid_sha256(self.policy_state_sha256)
            or not _valid_sha256(self.model_sha256)
            or np.any(np.isnan(logits))
            or np.any(np.isposinf(logits))
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise G1RolloutError("observable policy scores are invalid")
        digest = hashlib.sha256()
        digest.update(b"inspection-agent-g1-observable-policy-scores-v1")
        digest.update(self.policy_state_sha256.encode("ascii"))
        digest.update(self.model_sha256.encode("ascii"))
        digest.update(logits.tobytes(order="C"))
        digest.update(f"{probability:.17g}".encode("ascii"))
        state = digest.hexdigest()
        if self.state_sha256 not in ("", state):
            raise G1RolloutError("observable policy-score hash changed")
        object.__setattr__(self, "action_logits", logits)
        object.__setattr__(self, "stop_probability", probability)
        object.__setattr__(self, "state_sha256", state)


@dataclass(frozen=True, slots=True)
class RolloutStep:
    step_index: int
    observation_sha256: str
    policy_state_sha256: str
    scores: ObservablePolicyScores
    selected_slot: int | None
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.step_index) is not int
            or self.step_index < 0
            or not _valid_sha256(self.observation_sha256)
            or not _valid_sha256(self.policy_state_sha256)
            or type(self.scores) is not ObservablePolicyScores
            or self.scores.policy_state_sha256 != self.policy_state_sha256
            or (
                self.selected_slot is not None
                and (
                    type(self.selected_slot) is not int
                    or not 0 <= self.selected_slot < ACTION_SLOT_COUNT
                )
            )
        ):
            raise G1RolloutError("rollout step is invalid")
        state = _json_sha(
            {
                "schema": 1,
                "kind": "g1-rollout-step",
                "step_index": self.step_index,
                "observation": self.observation_sha256,
                "policy_state": self.policy_state_sha256,
                "scores": self.scores.state_sha256,
                "selected_slot": self.selected_slot,
            }
        )
        if self.state_sha256 not in ("", state):
            raise G1RolloutError("rollout-step hash changed")
        object.__setattr__(self, "state_sha256", state)


@dataclass(frozen=True, slots=True)
class ClosedLoopTrajectory:
    target_domain: str
    specimen_sha256: str
    task: InspectionTask
    model_sha256: str
    stop_threshold: float | None
    stopped: bool
    termination_reason: str
    action_history: tuple[InspectionCellAction, ...]
    steps: tuple[RolloutStep, ...]
    final_observation_sha256: str
    acquired_positions: np.ndarray
    acquired_values: np.ndarray
    native_count: int
    effective_budget: float
    acquired_positions_sha256: str = field(init=False)
    acquired_values_sha256: str = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        count = len(self.acquired_positions)
        positions = _readonly(self.acquired_positions, dtype="<i8", shape=(count, 2))
        values = _readonly(self.acquired_values, dtype=np.uint8, shape=(count, 3))
        budget = float(self.effective_budget)
        selected = tuple(step.selected_slot for step in self.steps if step.selected_slot is not None)
        if (
            type(self.target_domain) is not str
            or not self.target_domain
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or not _valid_sha256(self.model_sha256)
            or (
                self.stop_threshold is not None
                and float(self.stop_threshold) not in REGISTERED_STOP_THRESHOLDS
            )
            or type(self.stopped) is not bool
            or self.termination_reason not in {"STOP", "NO_LEGAL_ACTION"}
            or self.stopped != (self.termination_reason == "STOP")
            or type(self.action_history) is not tuple
            or any(type(action) is not InspectionCellAction for action in self.action_history)
            or tuple(action.cell_index for action in self.action_history[:8])
            != PRIMARY_WARM_START_CELLS
            or len(self.action_history) != 8 + len(selected)
            or tuple(canonical_action_from_slot(slot) for slot in selected)
            != self.action_history[8:]
            or type(self.steps) is not tuple
            or any(type(step) is not RolloutStep for step in self.steps)
            or tuple(step.step_index for step in self.steps) != tuple(range(len(self.steps)))
            or any(step.scores.model_sha256 != self.model_sha256 for step in self.steps)
            or not _valid_sha256(self.final_observation_sha256)
            or type(self.native_count) is not int
            or self.native_count < count
            or not math.isfinite(budget)
            or budget != count / self.native_count
            or budget > 0.25 + 1.0e-15
            or (self.stopped and (not self.steps or self.steps[-1].selected_slot is not None))
        ):
            raise G1RolloutError("closed-loop trajectory is invalid")
        positions_sha = hashlib.sha256(positions.tobytes(order="C")).hexdigest()
        values_sha = hashlib.sha256(values.tobytes(order="C")).hexdigest()
        payload = {
            "schema": 1,
            "kind": "g1-closed-loop-trajectory",
            "target_domain": self.target_domain,
            "specimen_sha256": self.specimen_sha256,
            "task": self.task.value,
            "model_sha256": self.model_sha256,
            "stop_threshold": self.stop_threshold,
            "stopped": self.stopped,
            "termination_reason": self.termination_reason,
            "action_history": tuple(
                (action.cell_index, action.from_level, action.to_level)
                for action in self.action_history
            ),
            "steps": tuple(step.state_sha256 for step in self.steps),
            "final_observation": self.final_observation_sha256,
            "positions": positions_sha,
            "values": values_sha,
            "native_count": self.native_count,
            "effective_budget": budget,
        }
        object.__setattr__(self, "acquired_positions", positions)
        object.__setattr__(self, "acquired_values", values)
        object.__setattr__(self, "effective_budget", budget)
        object.__setattr__(self, "acquired_positions_sha256", positions_sha)
        object.__setattr__(self, "acquired_values_sha256", values_sha)
        object.__setattr__(self, "state_sha256", _json_sha(payload))


class ObservableStateBuilder(Protocol):
    def __call__(self, observation: InspectionObservation) -> G1PolicyState: ...


class ObservableActor(Protocol):
    def __call__(self, state: G1PolicyState) -> ObservablePolicyScores: ...


def _validate_scores(
    state: G1PolicyState,
    scores: ObservablePolicyScores,
) -> None:
    mask = state.legal_action_mask
    if (
        scores.policy_state_sha256 != state.state_sha256
        or not np.all(np.isfinite(scores.action_logits[mask]))
        or not np.all(np.isneginf(scores.action_logits[~mask]))
    ):
        raise G1RolloutError("actor scores violate the legal-action mask")


def run_closed_loop(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    *,
    target_domain: str,
    specimen_sha256: str,
    state_builder: ObservableStateBuilder,
    actor: ObservableActor,
    stop_threshold: float | None,
) -> ClosedLoopTrajectory:
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(target_domain) is not str
        or not target_domain
        or not _valid_sha256(specimen_sha256)
        or not callable(state_builder)
        or not callable(actor)
        or (
            stop_threshold is not None
            and float(stop_threshold) not in REGISTERED_STOP_THRESHOLDS
        )
    ):
        raise G1RolloutError("closed-loop rollout request is invalid")
    warm_actions = tuple(
        canonical_action_from_slot(cell) for cell in PRIMARY_WARM_START_CELLS
    )
    observation = world.replay(warm_actions)
    steps: list[RolloutStep] = []
    model_sha256: str | None = None
    stopped = False
    termination_reason = "NO_LEGAL_ACTION"
    for step_index in range(ACTION_SLOT_COUNT):
        state = state_builder(observation)
        if (
            type(state) is not G1PolicyState
            or state.observation_sha256 != observation.state_sha256
            or state.grid_sha256 != grid.state_sha256
            or state.task is not observation.task
        ):
            raise G1RolloutError("state builder did not issue the current observable state")
        if not np.any(state.legal_action_mask):
            break
        scores = actor(state)
        if type(scores) is not ObservablePolicyScores:
            raise G1RolloutError("actor did not issue observable policy scores")
        _validate_scores(state, scores)
        if model_sha256 is None:
            model_sha256 = scores.model_sha256
        elif scores.model_sha256 != model_sha256:
            raise G1RolloutError("actor model identity changed during rollout")
        if stop_threshold is not None and scores.stop_probability >= float(stop_threshold):
            steps.append(
                RolloutStep(
                    step_index=step_index,
                    observation_sha256=observation.state_sha256,
                    policy_state_sha256=state.state_sha256,
                    scores=scores,
                    selected_slot=None,
                )
            )
            stopped = True
            termination_reason = "STOP"
            break
        selected_slot = int(np.argmax(scores.action_logits))
        if not state.legal_action_mask[selected_slot]:
            raise G1RolloutError("actor selected an illegal action")
        steps.append(
            RolloutStep(
                step_index=step_index,
                observation_sha256=observation.state_sha256,
                policy_state_sha256=state.state_sha256,
                scores=scores,
                selected_slot=selected_slot,
            )
        )
        observation = world.step(
            observation,
            canonical_action_from_slot(selected_slot),
        )
    else:
        raise G1RolloutError("closed-loop rollout exceeded the finite action roster")
    if model_sha256 is None:
        raise G1RolloutError("closed-loop rollout produced no actor score")
    return ClosedLoopTrajectory(
        target_domain=target_domain,
        specimen_sha256=specimen_sha256,
        task=observation.task,
        model_sha256=model_sha256,
        stop_threshold=stop_threshold,
        stopped=stopped,
        termination_reason=termination_reason,
        action_history=observation.action_history,
        steps=tuple(steps),
        final_observation_sha256=observation.state_sha256,
        acquired_positions=observation.acquired_positions,
        acquired_values=observation.measurement_values,
        native_count=observation.native_count,
        effective_budget=observation.effective_budget,
    )


@dataclass(frozen=True, slots=True)
class TargetTrajectorySeal:
    target_domain: str
    specimen_sha256: str
    trajectory_sha256: str
    truth_binding_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class TargetTruthView:
    target_domain: str
    specimen_sha256: str
    trajectory_sha256: str
    full_scan: np.ndarray
    full_scan_sha256: str
    true_cai: float
    state_sha256: str


class TargetTruthVault:
    """Hold target truth until a complete matching policy trajectory is sealed."""

    __slots__ = (
        "_full_scan",
        "_full_scan_sha256",
        "_seal",
        "_specimen_sha256",
        "_target_domain",
        "_true_cai",
    )

    def __init__(
        self,
        *,
        target_domain: str,
        specimen_sha256: str,
        full_scan: object,
        true_cai: float,
    ) -> None:
        image = np.asarray(full_scan)
        cai = float(true_cai)
        if (
            type(target_domain) is not str
            or not target_domain
            or not _valid_sha256(specimen_sha256)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or not math.isfinite(cai)
        ):
            raise TargetTruthVaultError("target truth identity is invalid")
        self._target_domain = target_domain
        self._specimen_sha256 = specimen_sha256
        self._full_scan = _readonly(image, dtype=np.uint8, shape=image.shape)
        self._full_scan_sha256 = hashlib.sha256(
            self._full_scan.tobytes(order="C")
        ).hexdigest()
        self._true_cai = cai
        self._seal: TargetTrajectorySeal | None = None

    def seal_trajectory(self, trajectory: ClosedLoopTrajectory) -> TargetTrajectorySeal:
        if (
            type(trajectory) is not ClosedLoopTrajectory
            or trajectory.target_domain != self._target_domain
            or trajectory.specimen_sha256 != self._specimen_sha256
        ):
            raise TargetTruthVaultError("target trajectory identity does not match the vault")
        truth_binding = _json_sha(
            {
                "schema": 1,
                "kind": "g1-target-truth-binding",
                "full_scan": self._full_scan_sha256,
                "true_cai": self._true_cai,
            }
        )
        state = _json_sha(
            {
                "schema": 1,
                "kind": "g1-target-trajectory-seal",
                "target_domain": self._target_domain,
                "specimen_sha256": self._specimen_sha256,
                "trajectory": trajectory.state_sha256,
                "truth_binding": truth_binding,
            }
        )
        seal = TargetTrajectorySeal(
            target_domain=self._target_domain,
            specimen_sha256=self._specimen_sha256,
            trajectory_sha256=trajectory.state_sha256,
            truth_binding_sha256=truth_binding,
            state_sha256=state,
        )
        if self._seal is not None and self._seal != seal:
            raise TargetTruthVaultError("target truth vault was sealed to a different trajectory")
        self._seal = seal
        return seal

    def open_truth(self, seal: TargetTrajectorySeal | None) -> TargetTruthView:
        if self._seal is None:
            raise TargetTruthVaultError("target trajectory is not sealed")
        if type(seal) is not TargetTrajectorySeal or seal != self._seal:
            raise TargetTruthVaultError("target truth seal is invalid")
        image = _readonly(
            self._full_scan,
            dtype=np.uint8,
            shape=self._full_scan.shape,
        )
        state = _json_sha(
            {
                "schema": 1,
                "kind": "g1-target-truth-view",
                "target_domain": self._target_domain,
                "specimen_sha256": self._specimen_sha256,
                "trajectory": seal.trajectory_sha256,
                "full_scan": self._full_scan_sha256,
                "true_cai": self._true_cai,
            }
        )
        return TargetTruthView(
            target_domain=self._target_domain,
            specimen_sha256=self._specimen_sha256,
            trajectory_sha256=seal.trajectory_sha256,
            full_scan=image,
            full_scan_sha256=self._full_scan_sha256,
            true_cai=self._true_cai,
            state_sha256=state,
        )


__all__ = [
    "ClosedLoopTrajectory",
    "G1RolloutError",
    "ObservableActor",
    "ObservablePolicyScores",
    "ObservableStateBuilder",
    "RolloutStep",
    "SurfaceVariant",
    "TargetTrajectorySeal",
    "TargetTruthVault",
    "TargetTruthVaultError",
    "TargetTruthView",
    "controlled_surface_hypothesis",
    "run_closed_loop",
    "shuffled_surface_donors",
]
