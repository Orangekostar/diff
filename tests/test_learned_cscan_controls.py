from __future__ import annotations

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionObservation, InspectionTask
from cmc_bbdm.inspection_agent.state import (
    InspectionCellAction,
    action_added_positions,
    apply_action,
    candidate_budget_record,
    legal_actions,
    zero_state,
)
from cmc_bbdm.learned_cscan import policies as policy_module
from cmc_bbdm.learned_cscan import rollouts
from cmc_bbdm.learned_cscan.contracts import Task
from cmc_bbdm.learned_cscan.observation import build_observation_packet
from cmc_bbdm.learned_cscan.perception import SurfacePercept, SurfaceRegion
from cmc_bbdm.learned_cscan.policies import RuleMethod, select_rule_action
from cmc_bbdm.learned_cscan.readout import BackgroundPrior, TaskReportV2
from cmc_bbdm.learned_cscan.stopping import RuleStopController
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.vlm_cscan.route import compile_route


def _observation(
    actions: tuple[InspectionCellAction, ...],
) -> tuple[InspectionObservation, object]:
    grid = build_acquisition_grid(33, 33, initial_budget=0.015625)
    state = zero_state(grid)
    mask = np.zeros(grid.native_shape, dtype=np.bool_)
    hidden = np.full((*grid.native_shape, 3), 120, dtype=np.uint8)
    hidden[0:5, 0:5] = (250, 20, 20)
    for action in actions:
        added = action_added_positions(grid, state, action)
        mask[added[:, 0], added[:, 1]] = True
        state = apply_action(grid, state, action)
    positions = np.argwhere(mask).astype(np.int64)
    observation = InspectionObservation(
        surface_rgb=np.full((32, 32, 3), 100, dtype=np.uint8),
        surface_sha256="a" * 64,
        task=InspectionTask.FIELD,
        native_shape=grid.native_shape,
        native_count=int(mask.size),
        grid_sha256=grid.state_sha256,
        measurement_state=state,
        acquired_positions=positions,
        measurement_values=hidden[positions[:, 0], positions[:, 1]],
        exact_acquired_count=len(positions),
        endpoint_budget=1.0,
        action_history=actions,
    )
    return observation, grid


def _packet(actions: tuple[InspectionCellAction, ...] = ()):
    observation, grid = _observation(actions)
    cue_cell = actions[0].cell_index if actions else 27
    percept = SurfacePercept(
        regions=(
            SurfaceRegion((cue_cell,), "shape_change", "lighting", "medium"),
        ),
        no_reliable_cue=False,
    )
    packet = build_observation_packet(
        observation,
        grid=grid,
        percept=percept,
        task=Task.LOCATE,
        prior=BackgroundPrior(
            rgb=np.asarray([120, 120, 120], dtype=np.uint8),
            fit_specimen_keys=("train:one",),
            fit_split="TRAIN",
        ),
        distance_threshold=0.18,
        probe_position=(0.0, 0.0),
        route_cost=0.0,
    )
    return packet, grid, observation


def test_rules_emit_progressive_legal_actions_and_balanced_forces_coverage() -> None:
    packet, grid, observation = _packet()
    legal = set(legal_actions(grid, observation.measurement_state))

    for method in RuleMethod:
        action = select_rule_action(method, packet, grid=grid, coverage_period=4)
        assert action in legal
        assert action.to_level == action.from_level + 1
        candidate = candidate_budget_record(grid, observation.measurement_state, action)
        assert candidate.measured_count > observation.exact_acquired_count
        assert candidate.effective_budget > observation.effective_budget
        positions = action_added_positions(
            grid, observation.measurement_state, action
        )
        full_route = compile_route(
            positions,
            native_shape=grid.native_shape,
            start_position=(0.0, 0.0),
        )
        route_cost, route_end = rollouts.compile_route_cost(
            positions,
            native_shape=grid.native_shape,
            start_position=(0.0, 0.0),
        )
        assert route_cost == full_route.total_length
        assert route_end == full_route.end_position

    visible_balanced = policy_module.select_balanced_visible_action(
        cell_levels=packet.cell_levels,
        candidate_cells=packet.report.candidate_cells,
        percept=SurfacePercept(
            regions=(
                SurfaceRegion((27,), "shape_change", "lighting", "medium"),
            ),
            no_reliable_cue=False,
        ),
        measured_mask=packet.measured_mask,
        probe_position=(
            float(packet.global_features[4]),
            float(packet.global_features[5]),
        ),
        grid=grid,
        coverage_period=4,
    )
    assert visible_balanced == select_rule_action(
        RuleMethod.R_BALANCED, packet, grid=grid, coverage_period=4
    )

    cue_history = (
        InspectionCellAction(0, -1, 0),
        InspectionCellAction(1, -1, 0),
        InspectionCellAction(8, -1, 0),
        InspectionCellAction(9, -1, 0),
        InspectionCellAction(2, -1, 0),
        InspectionCellAction(3, -1, 0),
        InspectionCellAction(0, 0, 1),
    )
    fourth_packet, grid, _ = _packet(cue_history)
    fourth = select_rule_action(
        RuleMethod.R_BALANCED,
        fourth_packet,
        grid=grid,
        coverage_period=4,
    )
    assert fourth.from_level == -1
    assert fourth.to_level == 0


def test_rule_stop_stability_ignores_unrelated_measurements() -> None:
    mask = np.zeros((64, 64), dtype=np.bool_)
    mask[24:32, 24:32] = True
    report = TaskReportV2(
        task=Task.LOCATE,
        predicted_mask=mask,
        support_positions=np.asarray([[25, 25], [25, 30], [30, 25]], dtype=np.int64),
        candidate_cells=(27,),
        unverified_boundary_cells=(),
        signal_strength=0.2,
        reason_code="VISIBLE_CANDIDATE",
    )
    levels = [0] * 64
    levels[27] = 1
    controller = RuleStopController(Task.LOCATE)

    first = controller.update(
        report,
        cell_levels=tuple(levels),
        last_action=InspectionCellAction(27, 0, 1),
    )
    unrelated_one = controller.update(
        report,
        cell_levels=tuple(levels),
        last_action=InspectionCellAction(0, 0, 1),
    )
    unrelated_two = controller.update(
        report,
        cell_levels=tuple(levels),
        last_action=InspectionCellAction(1, 0, 1),
    )
    relevant_one = controller.update(
        report,
        cell_levels=tuple(levels),
        last_action=InspectionCellAction(26, 0, 1),
    )
    relevant_two = controller.update(
        report,
        cell_levels=tuple(levels),
        last_action=InspectionCellAction(28, 0, 1),
    )

    assert first.stable_relevant_updates == 0
    assert unrelated_one.stable_relevant_updates == 0
    assert unrelated_two.stable_relevant_updates == 0
    assert relevant_one.stable_relevant_updates == 1
    assert relevant_two.stable_relevant_updates == 2
    assert relevant_two.should_stop is True
    assert not hasattr(relevant_two, "completion_probability")
