"""Causal scout and action-bound ultrasonic reveal."""

from __future__ import annotations

import math

import numpy as np

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import (
    MeasurementStateError,
    RefinementAction,
    apply_action,
    budget_record,
    initial_state,
    measurement_mask,
)

from .authority import MAVISAuthority, MAVISAuthorityError
from .contracts import InspectionState, MAVISContractError, PolicyContext


class MAVISRevealError(ValueError):
    """Raised when causal acquisition or exact-cost constraints are violated."""


def _checkpoint(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MAVISRevealError("checkpoint is invalid")
    checkpoint = float(value)
    if not math.isfinite(checkpoint) or not 0.0 < checkpoint <= 1.0:
        raise MAVISRevealError("checkpoint is invalid")
    return checkpoint


def _materialize(
    authority: MAVISAuthority,
    context: PolicyContext,
    *,
    initial_budget: float,
    checkpoint: float,
    levels: tuple[int, ...],
    action_history: tuple[RefinementAction, ...],
) -> InspectionState:
    grid = build_acquisition_grid(*context.native_shape, initial_budget=initial_budget)
    from cmc_bbdm.mva.measurement_state import MeasurementState

    state = MeasurementState(grid_sha256=grid.state_sha256, levels=levels)
    mask = measurement_mask(grid, state)
    positions = np.argwhere(mask).astype("<i8", copy=False)
    values = authority._reveal_values(context.specimen_id, positions)
    record = budget_record(grid, state)
    try:
        return InspectionState(
            specimen_id=context.specimen_id,
            context_features=context.context_features,
            native_shape=context.native_shape,
            native_count=context.native_count,
            initial_budget=initial_budget,
            checkpoint=checkpoint,
            grid_state_sha256=grid.state_sha256,
            measurement_levels=levels,
            acquired_positions=positions,
            measurement_values=values,
            exact_acquired_count=record.measured_count,
            action_history=action_history,
        )
    except MAVISContractError as error:
        raise MAVISRevealError("revealed state violates the causal contract") from error


def reveal_uniform_scout(
    authority: MAVISAuthority,
    context: PolicyContext,
    *,
    initial_budget: float,
    checkpoint: float,
) -> InspectionState:
    if type(authority) is not MAVISAuthority or type(context) is not PolicyContext:
        raise MAVISRevealError("issued authority and policy context are required")
    if authority.policy_context(context.specimen_id) != context:
        raise MAVISRevealError("policy context does not belong to the authority")
    cap = _checkpoint(checkpoint)
    try:
        grid = build_acquisition_grid(*context.native_shape, initial_budget=initial_budget)
        scout = initial_state(grid)
        if budget_record(grid, scout).effective_budget > cap + 1.0e-15:
            raise MAVISRevealError("uniform scout exceeds the budget")
        return _materialize(
            authority,
            context,
            initial_budget=float(initial_budget),
            checkpoint=cap,
            levels=scout.levels,
            action_history=(),
        )
    except (MAVISAuthorityError, MeasurementStateError, ValueError) as error:
        if isinstance(error, MAVISRevealError):
            raise
        raise MAVISRevealError("uniform scout cannot be revealed") from error


def reveal_action_history(
    authority: MAVISAuthority,
    context: PolicyContext,
    *,
    initial_budget: float,
    checkpoint: float,
    actions: tuple[RefinementAction, ...],
) -> InspectionState:
    if (
        type(authority) is not MAVISAuthority
        or type(context) is not PolicyContext
        or authority.policy_context(context.specimen_id) != context
        or type(actions) is not tuple
        or any(type(action) is not RefinementAction for action in actions)
    ):
        raise MAVISRevealError("issued action history is invalid")
    cap = _checkpoint(checkpoint)
    try:
        grid = build_acquisition_grid(
            *context.native_shape,
            initial_budget=initial_budget,
        )
        state = initial_state(grid)
        acquired = measurement_mask(grid, state)
        for action in actions:
            refined = apply_action(grid, state, action)
            cell = grid.cells[action.cell_index]
            rows = np.asarray(cell.rows[action.to_level], dtype=np.int64)
            columns = np.asarray(cell.columns[action.to_level], dtype=np.int64)
            added = int(np.count_nonzero(~acquired[np.ix_(rows, columns)]))
            if added <= 0:
                raise MAVISRevealError("action history adds no new measurement")
            acquired[np.ix_(rows, columns)] = True
            state = refined
        if float(np.count_nonzero(acquired) / acquired.size) > cap + 1.0e-15:
            raise MAVISRevealError("action history exceeds the exact acquisition budget")
        return _materialize(
            authority,
            context,
            initial_budget=float(initial_budget),
            checkpoint=cap,
            levels=state.levels,
            action_history=actions,
        )
    except (MAVISAuthorityError, MeasurementStateError, ValueError) as error:
        if isinstance(error, MAVISRevealError):
            raise
        raise MAVISRevealError("action history cannot be revealed") from error


def reveal_action(
    authority: MAVISAuthority,
    state: InspectionState,
    action: RefinementAction,
) -> InspectionState:
    if type(authority) is not MAVISAuthority or type(state) is not InspectionState:
        raise MAVISRevealError("issued authority and inspection state are required")
    context = authority.policy_context(state.specimen_id)
    if not np.array_equal(context.context_features, state.context_features):
        raise MAVISRevealError("inspection state does not belong to the authority")
    grid = build_acquisition_grid(*state.native_shape, initial_budget=state.initial_budget)
    if grid.state_sha256 != state.grid_state_sha256:
        raise MAVISRevealError("inspection grid changed")
    try:
        refined = apply_action(grid, state.measurement_state, action)
    except MeasurementStateError as error:
        raise MAVISRevealError("action is not legal for the current state") from error
    record = budget_record(grid, refined)
    if record.effective_budget > state.checkpoint + 1.0e-15:
        raise MAVISRevealError("action exceeds the exact acquisition budget")
    revealed = _materialize(
        authority,
        context,
        initial_budget=state.initial_budget,
        checkpoint=state.checkpoint,
        levels=refined.levels,
        action_history=(*state.action_history, action),
    )
    current_linear = (
        state.acquired_positions[:, 0] * state.native_shape[1]
        + state.acquired_positions[:, 1]
    )
    revealed_linear = (
        revealed.acquired_positions[:, 0] * state.native_shape[1]
        + revealed.acquired_positions[:, 1]
    )
    retained = np.searchsorted(revealed_linear, current_linear)
    if (
        revealed.exact_acquired_count <= state.exact_acquired_count
        or np.any(retained >= revealed.exact_acquired_count)
        or not np.array_equal(revealed_linear[retained], current_linear)
        or not np.array_equal(
            revealed.measurement_values[retained],
            state.measurement_values,
        )
    ):
        raise MAVISRevealError("action did not add a causal measurement")
    return revealed


__all__ = [
    "MAVISRevealError",
    "reveal_action",
    "reveal_action_history",
    "reveal_uniform_scout",
]
