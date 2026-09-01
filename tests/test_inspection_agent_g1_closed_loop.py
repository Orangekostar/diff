from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.state import action_added_positions, fitting_actions
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.contracts import (
    CAIContextMode,
    G1PolicyState,
    TaskTokenMode,
)
from cmc_bbdm.inspection_agent_g1.features import canonical_slot
from cmc_bbdm.inspection_agent_g1.rollout import (
    ObservablePolicyScores,
    run_closed_loop,
)
from cmc_bbdm.inspection_agent_g1.warm_start import (
    PRIMARY_WARM_START_CELLS,
    build_deployment_grid,
)
from cmc_bbdm.mavis.authority import MAVISAuthority


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _runtime():
    rows, columns = np.indices((41, 43))
    full_scan = np.stack((rows, columns, rows + columns), axis=2).astype(np.uint8)
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("sample",),
        dataset_ids=("target",),
        images=(full_scan,),
        targets=np.asarray([0.4]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    surface = np.zeros((80, 80, 3), dtype=np.uint8)
    grid = build_deployment_grid(full_scan.shape[:2])
    world = CausalInspectionWorld(
        authority,
        specimen_id="sample",
        task=InspectionTask.FIELD,
        surface_rgb=surface,
        surface_sha256=hashlib.sha256(surface.tobytes()).hexdigest(),
        grid=grid,
        endpoint_budget=0.25,
    )
    return world, grid, full_scan


def _state_builder(grid):
    def build(observation):
        mask = np.zeros(192, dtype=np.bool_)
        for action in fitting_actions(
            grid,
            observation.measurement_state,
            observation.endpoint_budget,
        ):
            if len(action_added_positions(grid, observation.measurement_state, action)):
                mask[canonical_slot(action)] = True
        global_scalars = np.zeros(17)
        global_scalars[0] = observation.effective_budget
        global_scalars[1] = observation.remaining_budget
        return G1PolicyState(
            task=InspectionTask.FIELD,
            task_token_mode=TaskTokenMode.CORRECT,
            cai_context_mode=CAIContextMode.TASK_SPECIFIC_MASKED,
            observation_sha256=observation.state_sha256,
            reconstruction_sha256=_sha(f"reconstruction-{observation.state_sha256}"),
            surface_hypothesis_sha256=_sha("surface-hypothesis"),
            grid_sha256=grid.state_sha256,
            reconstruction_embedding=np.zeros(512),
            global_scalars=global_scalars,
            task_token=np.asarray([1.0, 0.0]),
            cell_features=np.zeros((64, 18)),
            candidate_features=np.zeros((192, 12)),
            legal_action_mask=mask,
        )

    return build


class _StopAfterOneAction:
    def __init__(self) -> None:
        self.states: list[G1PolicyState] = []

    def __call__(self, state: G1PolicyState) -> ObservablePolicyScores:
        self.states.append(state)
        logits = np.full(192, -np.inf)
        legal = np.flatnonzero(state.legal_action_mask)
        logits[legal] = -legal
        return ObservablePolicyScores(
            policy_state_sha256=state.state_sha256,
            model_sha256=_sha("model"),
            action_logits=logits,
            stop_probability=0.0 if len(self.states) == 1 else 1.0,
        )


def test_closed_loop_uses_only_policy_state_and_causal_legal_actions() -> None:
    world, grid, _ = _runtime()
    actor = _StopAfterOneAction()
    trajectory = run_closed_loop(
        world,
        grid,
        target_domain="target",
        specimen_sha256=_sha("sample"),
        state_builder=_state_builder(grid),
        actor=actor,
        stop_threshold=0.90,
    )
    assert all(type(state) is G1PolicyState for state in actor.states)
    assert len(actor.states) == 2
    assert trajectory.stopped is True
    assert trajectory.termination_reason == "STOP"
    assert len(trajectory.action_history) == 9
    assert tuple(action.cell_index for action in trajectory.action_history[:8]) == (
        PRIMARY_WARM_START_CELLS
    )
    assert trajectory.steps[0].selected_slot is not None
    assert actor.states[0].legal_action_mask[trajectory.steps[0].selected_slot]
    assert trajectory.steps[-1].selected_slot is None
    assert trajectory.acquired_positions.shape[0] == trajectory.acquired_values.shape[0]
    assert trajectory.effective_budget <= 0.25


def test_unauthorized_stop_score_cannot_terminate_the_policy() -> None:
    world, grid, _ = _runtime()

    def actor(state: G1PolicyState) -> ObservablePolicyScores:
        logits = np.full(192, -np.inf)
        legal = np.flatnonzero(state.legal_action_mask)
        logits[legal] = -legal
        return ObservablePolicyScores(
            policy_state_sha256=state.state_sha256,
            model_sha256=_sha("model"),
            action_logits=logits,
            stop_probability=1.0,
        )

    trajectory = run_closed_loop(
        world,
        grid,
        target_domain="target",
        specimen_sha256=_sha("sample"),
        state_builder=_state_builder(grid),
        actor=actor,
        stop_threshold=None,
    )
    assert trajectory.stopped is False
    assert trajectory.termination_reason == "NO_LEGAL_ACTION"
    assert trajectory.steps
