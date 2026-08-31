"""Causal zero-start world that reveals only newly acquired positions."""

from __future__ import annotations

import math

import numpy as np

from cmc_bbdm.mavis.authority import MAVISAuthority, MAVISAuthorityError
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .contracts import InspectionObservation, InspectionTask
from .state import (
    GeneralizedMeasurementState,
    GeneralizedMeasurementStateError,
    InspectionCellAction,
    action_added_positions,
    action_added_positions_from_mask,
    apply_action,
    budget_record,
    fitting_actions,
    measurement_mask,
    zero_state,
)


class CausalInspectionWorldError(ValueError):
    """Raised when a causal reveal or observation transition is invalid."""


class CausalInspectionWorld:
    """Keep privileged identity private and return policy-visible observations."""

    def __init__(
        self,
        authority: MAVISAuthority,
        *,
        specimen_id: str,
        task: InspectionTask,
        surface_rgb: np.ndarray,
        surface_sha256: str,
        grid: AcquisitionGrid,
        endpoint_budget: float,
    ) -> None:
        endpoint = float(endpoint_budget)
        if (
            type(authority) is not MAVISAuthority
            or type(specimen_id) is not str
            or not specimen_id
            or type(task) is not InspectionTask
            or type(grid) is not AcquisitionGrid
            or isinstance(endpoint_budget, bool)
            or not math.isfinite(endpoint)
            or not 0.0 < endpoint <= 1.0
        ):
            raise CausalInspectionWorldError("causal world request is invalid")
        try:
            context = authority.policy_context(specimen_id)
        except MAVISAuthorityError as error:
            raise CausalInspectionWorldError("specimen is unavailable") from error
        if context.native_shape != grid.native_shape:
            raise CausalInspectionWorldError("world grid does not match the specimen")
        self._authority = authority
        self._specimen_key = specimen_id
        self._task = task
        self._surface_rgb = surface_rgb
        self._surface_sha256 = surface_sha256
        self._grid = grid
        self._endpoint_budget = endpoint
        self._current_observation_sha256: str | None = None

    def _materialize(
        self,
        state: GeneralizedMeasurementState,
        positions: np.ndarray,
        values: np.ndarray,
        history: tuple[InspectionCellAction, ...],
    ) -> InspectionObservation:
        record = budget_record(self._grid, state)
        if positions.shape[0] != record.measured_count:
            raise CausalInspectionWorldError("revealed positions do not match the state")
        observation = InspectionObservation(
            surface_rgb=self._surface_rgb,
            surface_sha256=self._surface_sha256,
            task=self._task,
            native_shape=self._grid.native_shape,
            native_count=record.native_count,
            grid_sha256=self._grid.state_sha256,
            measurement_state=state,
            acquired_positions=positions,
            measurement_values=values,
            exact_acquired_count=record.measured_count,
            endpoint_budget=self._endpoint_budget,
            action_history=history,
        )
        self._surface_rgb = observation.surface_rgb
        self._current_observation_sha256 = observation.state_sha256
        return observation

    def reset(self) -> InspectionObservation:
        state = zero_state(self._grid)
        return self._materialize(
            state,
            np.empty((0, 2), dtype="<i8"),
            np.empty((0, 3), dtype=np.uint8),
            (),
        )

    def legal_actions(
        self,
        observation: InspectionObservation,
    ) -> tuple[InspectionCellAction, ...]:
        self._validate_current(observation)
        return fitting_actions(
            self._grid,
            observation.measurement_state,
            self._endpoint_budget,
        )

    def _validate_current(self, observation: InspectionObservation) -> None:
        if (
            type(observation) is not InspectionObservation
            or observation.state_sha256 != self._current_observation_sha256
            or observation.grid_sha256 != self._grid.state_sha256
            or observation.task is not self._task
            or observation.surface_sha256 != self._surface_sha256
        ):
            raise CausalInspectionWorldError("observation is stale or belongs elsewhere")

    def step(
        self,
        observation: InspectionObservation,
        action: InspectionCellAction,
    ) -> InspectionObservation:
        self._validate_current(observation)
        try:
            candidate = apply_action(self._grid, observation.measurement_state, action)
            record = budget_record(self._grid, candidate)
            if record.effective_budget > self._endpoint_budget + 1.0e-15:
                raise CausalInspectionWorldError("action exceeds the endpoint budget")
            added_positions = action_added_positions(
                self._grid,
                observation.measurement_state,
                action,
            )
            added_values = self._authority._reveal_values(
                self._specimen_key,
                added_positions,
            )
        except (GeneralizedMeasurementStateError, MAVISAuthorityError) as error:
            raise CausalInspectionWorldError("action cannot be revealed") from error
        candidate_positions = np.argwhere(
            measurement_mask(self._grid, candidate)
        ).astype("<i8", copy=False)
        candidate_linear = (
            candidate_positions[:, 0] * self._grid.native_shape[1]
            + candidate_positions[:, 1]
        )
        values = np.empty((len(candidate_positions), 3), dtype=np.uint8)
        if observation.exact_acquired_count:
            old_linear = (
                observation.acquired_positions[:, 0] * self._grid.native_shape[1]
                + observation.acquired_positions[:, 1]
            )
            old_indices = np.searchsorted(candidate_linear, old_linear)
            values[old_indices] = observation.measurement_values
        added_linear = (
            added_positions[:, 0] * self._grid.native_shape[1]
            + added_positions[:, 1]
        )
        added_indices = np.searchsorted(candidate_linear, added_linear)
        values[added_indices] = added_values
        return self._materialize(
            candidate,
            candidate_positions,
            values,
            (*observation.action_history, action),
        )

    def replay(
        self,
        action_history: tuple[InspectionCellAction, ...],
    ) -> InspectionObservation:
        if type(action_history) is not tuple or any(
            type(action) is not InspectionCellAction for action in action_history
        ):
            raise CausalInspectionWorldError("action history is invalid")
        state = zero_state(self._grid)
        mask = np.zeros(self._grid.native_shape, dtype=np.bool_)
        measured_count = 0
        try:
            for action in action_history:
                added_positions = action_added_positions_from_mask(
                    self._grid,
                    state,
                    action,
                    mask,
                )
                candidate_count = measured_count + len(added_positions)
                if (
                    candidate_count / mask.size
                    > self._endpoint_budget + 1.0e-15
                ):
                    raise CausalInspectionWorldError(
                        "action history exceeds the endpoint budget"
                    )
                state = apply_action(self._grid, state, action)
                mask[added_positions[:, 0], added_positions[:, 1]] = True
                measured_count = candidate_count
            positions = np.argwhere(mask).astype("<i8", copy=False)
            values = self._authority._reveal_values(self._specimen_key, positions)
        except (GeneralizedMeasurementStateError, MAVISAuthorityError) as error:
            raise CausalInspectionWorldError("action history cannot be replayed") from error
        return self._materialize(state, positions, values, action_history)


__all__ = ["CausalInspectionWorld", "CausalInspectionWorldError"]
