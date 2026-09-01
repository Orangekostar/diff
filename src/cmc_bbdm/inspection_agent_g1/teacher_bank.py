"""Deterministic post-warm source states for the fold-safe G1 teacher bank."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionObservation, InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.oracle import (
    CAIStatePredictor,
    ReconstructionEncoder,
    choose_cai_action,
    choose_field_action,
)
from cmc_bbdm.inspection_agent.state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    action_added_positions,
    apply_action,
    fitting_actions,
)
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.mva.oracle import uniform_cell_order

from .features import canonical_action_from_slot
from .teacher import (
    SourceTeacherAuthorization,
    validate_source_teacher_dependencies,
)
from .warm_start import PRIMARY_WARM_START_CELLS, apply_warm_start


class G1TeacherBankError(ValueError):
    """Raised when source-state generation violates the frozen G1 roster."""


class ContinuationPolicy(str, Enum):
    UNIFORM_CONTINUE = "UNIFORM_CONTINUE"
    SURFACE_FOCUS_CONTINUE = "SURFACE_FOCUS_CONTINUE"
    RANDOM_CONTINUE = "RANDOM_CONTINUE"
    ALTERNATE_BROADEN_REFINE = "ALTERNATE_BROADEN_REFINE"


@dataclass(frozen=True, slots=True)
class TeacherBankState:
    outer_target: str
    source: str
    snapshot_index: int
    progress_fraction: float
    label_independent: bool
    observation: InspectionObservation
    state_sha256: str


@dataclass(frozen=True, slots=True)
class OracleCheckpointState:
    checkpoints: tuple[float, ...]
    label_independent: bool
    observation: InspectionObservation
    state_sha256: str


def _state_hash(kind: str, values: tuple[object, ...]) -> str:
    return hashlib.sha256(
        ("|".join(("inspection-agent-g1-teacher-bank-v1", kind, *(str(v) for v in values)))).encode(
            "ascii"
        )
    ).hexdigest()


def _ordered_action(
    actions: tuple[InspectionCellAction, ...],
    order: tuple[int, ...],
    *,
    minimum_level: bool,
) -> InspectionCellAction:
    eligible = actions
    if minimum_level:
        level = min(action.from_level for action in actions)
        eligible = tuple(action for action in actions if action.from_level == level)
    by_cell = {action.cell_index: action for action in eligible}
    for cell in order:
        if cell in by_cell:
            return by_cell[cell]
    raise G1TeacherBankError("continuation order has no eligible action")


def _positive_cost_actions(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    endpoint_budget: float,
) -> tuple[InspectionCellAction, ...]:
    return tuple(
        action
        for action in fitting_actions(grid, state, endpoint_budget)
        if len(action_added_positions(grid, state, action)) > 0
    )


def plan_continuation_actions(
    grid: AcquisitionGrid,
    warm_state: GeneralizedMeasurementState,
    policy: ContinuationPolicy,
    surface_hypothesis: SurfaceHypothesis,
    *,
    outer_target: str,
    task: InspectionTask,
    surface_sha256: str,
    random_seed: int,
    endpoint_budget: float,
) -> tuple[InspectionCellAction, ...]:
    endpoint = float(endpoint_budget)
    if (
        type(grid) is not AcquisitionGrid
        or type(warm_state) is not GeneralizedMeasurementState
        or warm_state != apply_warm_start(grid, k=8)
        or type(policy) is not ContinuationPolicy
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(outer_target) is not str
        or not outer_target
        or task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or type(surface_sha256) is not str
        or len(surface_sha256) != 64
        or type(random_seed) is not int
        or isinstance(endpoint_budget, bool)
        or not math.isfinite(endpoint)
        or not 0.0 < endpoint <= 1.0
    ):
        raise G1TeacherBankError("continuation policy request is invalid")
    uniform_order = uniform_cell_order()
    surface_order = tuple(
        sorted(
            range(64),
            key=lambda cell: (-float(surface_hypothesis.scores[cell]), cell),
        )
    )
    token = hashlib.sha256(
        (
            f"g1-continuation|{random_seed}|{outer_target}|"
            f"{surface_sha256}|{task.value}"
        ).encode("ascii")
    ).hexdigest()
    generator = np.random.Generator(np.random.PCG64(int(token[:16], 16)))
    state = warm_state
    output: list[InspectionCellAction] = []
    pending_refine: int | None = None
    for _ in range(192):
        actions = _positive_cost_actions(grid, state, endpoint)
        if not actions:
            break
        if policy is ContinuationPolicy.UNIFORM_CONTINUE:
            selected = _ordered_action(actions, uniform_order, minimum_level=True)
        elif policy is ContinuationPolicy.SURFACE_FOCUS_CONTINUE:
            selected = _ordered_action(actions, surface_order, minimum_level=True)
        elif policy is ContinuationPolicy.RANDOM_CONTINUE:
            selected = actions[int(generator.integers(0, len(actions)))]
        else:
            immediate = tuple(
                action
                for action in actions
                if pending_refine is not None
                and action.cell_index == pending_refine
                and action.from_level == 0
            )
            if immediate:
                selected = immediate[0]
                pending_refine = None
            else:
                broaden = tuple(action for action in actions if action.from_level == -1)
                selected = _ordered_action(
                    broaden if broaden else actions,
                    uniform_order,
                    minimum_level=not broaden,
                )
                pending_refine = selected.cell_index if selected.from_level == -1 else None
        output.append(selected)
        state = apply_action(grid, state, selected)
    else:
        raise G1TeacherBankError("continuation policy exceeded the finite action roster")
    if not output:
        raise G1TeacherBankError("continuation policy produced no action")
    return tuple(output)


def _bank_state(
    outer_target: str,
    source: str,
    snapshot_index: int,
    fraction: float,
    label_independent: bool,
    observation: InspectionObservation,
) -> TeacherBankState:
    return TeacherBankState(
        outer_target=outer_target,
        source=source,
        snapshot_index=snapshot_index,
        progress_fraction=fraction,
        label_independent=label_independent,
        observation=observation,
        state_sha256=_state_hash(
            "state",
            (
                source,
                outer_target,
                snapshot_index,
                f"{fraction:.17g}",
                label_independent,
                observation.state_sha256,
            ),
        ),
    )


def materialize_label_independent_states(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    surface_hypothesis: SurfaceHypothesis,
    *,
    outer_target: str,
    random_seed: int,
    snapshot_fractions: tuple[float, ...],
) -> tuple[TeacherBankState, ...]:
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(outer_target) is not str
        or not outer_target
        or type(snapshot_fractions) is not tuple
        or len(snapshot_fractions) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 1.0
            for value in snapshot_fractions
        )
        or any(float(right) <= float(left) for left, right in pairwise(snapshot_fractions))
        or float(snapshot_fractions[-1]) != 1.0
    ):
        raise G1TeacherBankError("label-independent state request is invalid")
    warm_actions = tuple(
        canonical_action_from_slot(cell) for cell in PRIMARY_WARM_START_CELLS
    )
    warm = world.replay(warm_actions)
    rows = [_bank_state(outer_target, "WARM_START", 0, 0.0, True, warm)]
    for policy in ContinuationPolicy:
        actions = plan_continuation_actions(
            grid,
            warm.measurement_state,
            policy,
            surface_hypothesis,
            outer_target=outer_target,
            task=warm.task,
            surface_sha256=warm.surface_sha256,
            random_seed=random_seed,
            endpoint_budget=warm.endpoint_budget,
        )
        indices = tuple(
            min(len(actions) - 1, math.ceil(float(fraction) * len(actions)) - 1)
            for fraction in snapshot_fractions
        )
        if len(set(indices)) != 3:
            raise G1TeacherBankError("continuation snapshots are not unique")
        for snapshot_index, action_index in enumerate(indices):
            observation = world.replay(
                (*warm_actions, *actions[: action_index + 1])
            )
            rows.append(
                _bank_state(
                    outer_target,
                    policy.value,
                    snapshot_index,
                    float(snapshot_fractions[snapshot_index]),
                    True,
                    observation,
                )
            )
    if len(rows) != 13:
        raise G1TeacherBankError("label-independent state count changed")
    return tuple(rows)


def materialize_oracle_checkpoint_states(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    surface_hypothesis: SurfaceHypothesis,
    prior: SourceBackgroundPrior,
    authorization: SourceTeacherAuthorization,
    *,
    full_scan: object,
    checkpoints: tuple[float, ...],
    true_cai: float | None = None,
    assessor: CAIStatePredictor | None = None,
    encoder: ReconstructionEncoder | None = None,
) -> tuple[OracleCheckpointState, ...]:
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(checkpoints) is not tuple
        or len(checkpoints) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 0.25
            for value in checkpoints
        )
        or any(float(right) <= float(left) for left, right in pairwise(checkpoints))
    ):
        raise G1TeacherBankError("oracle checkpoint request is invalid")
    validate_source_teacher_dependencies(
        authorization,
        prior,
        assessor=assessor,
    )
    warm_actions = tuple(
        canonical_action_from_slot(cell) for cell in PRIMARY_WARM_START_CELLS
    )
    current = world.replay(warm_actions)
    trajectory = [current]
    for _ in range(192):
        if not _positive_cost_actions(
            grid,
            current.measurement_state,
            current.endpoint_budget,
        ):
            break
        if current.task is InspectionTask.FIELD:
            selection = choose_field_action(
                current,
                grid,
                prior,
                full_scan=full_scan,
                checkpoint=current.endpoint_budget,
            )
        elif current.task is InspectionTask.CAI:
            if true_cai is None or assessor is None or encoder is None:
                raise G1TeacherBankError("CAI oracle dependencies are incomplete")
            selection = choose_cai_action(
                current,
                grid,
                prior,
                full_scan=full_scan,
                true_cai=true_cai,
                assessor=assessor,
                encoder=encoder,
                checkpoint=current.endpoint_budget,
            )
        else:
            raise G1TeacherBankError("G1 teacher bank supports FIELD and CAI only")
        current = world.step(current, selection.action)
        trajectory.append(current)
    else:
        raise G1TeacherBankError("oracle trajectory exceeded the finite action roster")
    grouped: dict[str, tuple[InspectionObservation, list[float]]] = {}
    for checkpoint in checkpoints:
        eligible = [
            observation
            for observation in trajectory
            if observation.effective_budget <= float(checkpoint) + 1.0e-15
        ]
        if not eligible:
            raise G1TeacherBankError("no oracle state fits a checkpoint")
        selected = eligible[-1]
        if selected.state_sha256 not in grouped:
            grouped[selected.state_sha256] = (selected, [])
        grouped[selected.state_sha256][1].append(float(checkpoint))
    output = []
    for observation, grouped_checkpoints in grouped.values():
        checkpoints_tuple = tuple(grouped_checkpoints)
        output.append(
            OracleCheckpointState(
                checkpoints=checkpoints_tuple,
                label_independent=False,
                observation=observation,
                state_sha256=_state_hash(
                    "oracle-checkpoint",
                    (
                        authorization.state_sha256,
                        checkpoints_tuple,
                        observation.state_sha256,
                    ),
                ),
            )
        )
    return tuple(output)


__all__ = [
    "ContinuationPolicy",
    "G1TeacherBankError",
    "OracleCheckpointState",
    "TeacherBankState",
    "materialize_label_independent_states",
    "materialize_oracle_checkpoint_states",
    "plan_continuation_actions",
]
