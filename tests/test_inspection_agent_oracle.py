from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask
from cmc_bbdm.inspection_agent.field_task import (
    internal_signal_saliency,
    normalized_capture_auc,
)
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    SourceBackgroundPrior,
    reconstruct_observation,
)
from cmc_bbdm.inspection_agent.oracle import (
    choose_cai_action,
    choose_discovery_action,
    choose_field_action,
    run_discovery_oracle,
    run_field_oracle,
)
from cmc_bbdm.inspection_agent.state import (
    InspectionCellAction,
    apply_action,
    budget_record,
    fitting_actions,
    zero_state,
)
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.reconstruction_value import normalized_rgb_mse


def _prior() -> SourceBackgroundPrior:
    return SourceBackgroundPrior(
        outer_domain="target",
        source_domains=("a", "b"),
        fit_specimen_ids=("a1", "b1"),
        source_authority_sha256="a" * 64,
        domain_border_medians=np.zeros((2, 3)),
        background_rgb=np.zeros(3, dtype=np.uint8),
    )


def _world(image: np.ndarray, task: InspectionTask, endpoint: float) -> CausalInspectionWorld:
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("sample",),
        dataset_ids=("target",),
        images=(image,),
        targets=np.asarray([0.5]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    grid = build_acquisition_grid(*image.shape[:2], initial_budget=0.015625)
    surface = np.zeros((9, 9, 3), dtype=np.uint8)
    return CausalInspectionWorld(
        authority,
        specimen_id="sample",
        task=task,
        surface_rgb=surface,
        surface_sha256=hashlib.sha256(surface.tobytes()).hexdigest(),
        grid=grid,
        endpoint_budget=endpoint,
    )


def test_internal_signal_saliency_and_capture_auc_use_exact_budget_axis() -> None:
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    image[4, 4] = (10, 20, 30)
    signal = internal_signal_saliency(image)
    assert signal.total_mass == 60.0
    assert signal.pixel_mass[4, 4] == 60.0
    assert signal.border_median_rgb.tolist() == [0.0, 0.0, 0.0]
    assert normalized_capture_auc(
        np.asarray((0.0, 0.01, 0.02)),
        np.asarray((0.0, 0.5, 1.0)),
        scout_endpoint=0.02,
    ) == pytest.approx(0.5)


def test_discovery_oracle_selects_highest_incremental_signal_per_cost() -> None:
    image = np.zeros((41, 43, 3), dtype=np.uint8)
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    target_cell = 63
    cell = grid.cells[target_cell]
    rows = np.asarray(cell.rows[0])
    columns = np.asarray(cell.columns[0])
    image[np.ix_(rows, columns)] = 255
    selection = choose_discovery_action(
        grid,
        zero_state(grid),
        full_scan=image,
        checkpoint=0.25,
    )
    assert selection.action.cell_index == target_cell
    assert selection.raw_value > 0.0
    assert selection.objective_value == pytest.approx(
        selection.raw_value / selection.exact_added_cost
    )
    assert len(selection.candidates) == 64


def test_field_oracle_matches_exhaustive_observed_only_reconstruction() -> None:
    rows, columns = np.indices((41, 43))
    image = np.stack((rows * 5, columns * 3, rows + columns), axis=2).astype(np.uint8)
    world = _world(image, InspectionTask.FIELD, 0.25)
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    observation = world.reset()
    selection = choose_field_action(
        observation,
        grid,
        _prior(),
        full_scan=image,
        checkpoint=0.25,
    )
    current_loss = normalized_rgb_mse(
        image, reconstruct_observation(observation, grid, _prior()).image
    )
    exhaustive = []
    exhaustive_losses = {}
    for action in fitting_actions(grid, observation.measurement_state, 0.25):
        candidate_world = _world(image, InspectionTask.FIELD, 0.25)
        candidate = candidate_world.step(candidate_world.reset(), action)
        candidate_loss = normalized_rgb_mse(
            image, reconstruct_observation(candidate, grid, _prior()).image
        )
        added = candidate.exact_acquired_count
        exhaustive.append(((current_loss - candidate_loss) / added, action))
        exhaustive_losses[action] = candidate_loss
    expected = max(
        exhaustive,
        key=lambda item: (item[0], -item[1].cell_index, -item[1].to_level),
    )[1]
    assert selection.action == expected
    assert len(selection.candidates) == len(exhaustive)
    for score in selection.candidates:
        assert score.task_loss_after == pytest.approx(
            exhaustive_losses[score.action], abs=1.0e-15
        )
        assert score.candidate_state_sha256 == apply_action(
            grid,
            observation.measurement_state,
            score.action,
        ).state_sha256


def test_field_oracle_candidate_losses_match_exhaustive_mixed_state() -> None:
    rows, columns = np.indices((41, 43))
    image = np.stack((rows * 5, columns * 3, rows + columns), axis=2).astype(np.uint8)
    world = _world(image, InspectionTask.FIELD, 0.25)
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    history = (
        InspectionCellAction(0, -1, 0),
        InspectionCellAction(1, -1, 0),
        InspectionCellAction(0, 0, 1),
        InspectionCellAction(9, -1, 0),
    )
    observation = world.replay(history)
    selection = choose_field_action(
        observation,
        grid,
        _prior(),
        full_scan=image,
        checkpoint=0.25,
    )
    for score in selection.candidates:
        candidate_world = _world(image, InspectionTask.FIELD, 0.25)
        candidate = candidate_world.replay((*history, score.action))
        expected_loss = normalized_rgb_mse(
            image,
            reconstruct_observation(candidate, grid, _prior()).image,
        )
        assert score.task_loss_after == pytest.approx(expected_loss, abs=1.0e-15)
        assert score.candidate_state_sha256 == candidate.measurement_state.state_sha256


def test_discovery_trajectory_records_candidates_and_structured_decisions() -> None:
    image = np.zeros((49, 51, 3), dtype=np.uint8)
    image[20:30, 20:30] = 200
    grid = build_acquisition_grid(49, 51, initial_budget=0.015625)
    all_level0 = tuple(
        InspectionCellAction(cell, -1, 0) for cell in range(64)
    )
    state = zero_state(grid)
    for action in all_level0:
        state = apply_action(grid, state, action)
    endpoint = budget_record(grid, state).effective_budget
    world = _world(image, InspectionTask.DISCOVERY, endpoint)
    trajectory = run_discovery_oracle(
        world,
        grid,
        full_scan=image,
        surface_hypothesis_cells=(27,),
    )
    assert 1 <= len(trajectory.steps) <= 64
    assert trajectory.steps[0].decision in {
        InspectionDecision.FOCUS,
        InspectionDecision.BROADEN,
    }
    assert trajectory.steps[0].candidates
    assert trajectory.steps[-1].budget_after == pytest.approx(endpoint)
    assert trajectory.steps[-1].task_loss_after <= trajectory.steps[0].task_loss_before
    assert not hasattr(trajectory.final_observation, "full_scan")
    assert not hasattr(trajectory.final_observation, "true_cai")


def test_field_oracle_trajectory_records_loss_and_respects_endpoint() -> None:
    rows, columns = np.indices((49, 51))
    image = np.stack((rows * 4, columns * 3, rows + columns), axis=2).astype(np.uint8)
    endpoint = 0.02
    world = _world(image, InspectionTask.FIELD, endpoint)
    grid = build_acquisition_grid(49, 51, initial_budget=0.015625)
    trajectory = run_field_oracle(
        world,
        grid,
        _prior(),
        full_scan=image,
        surface_hypothesis_cells=(27,),
    )
    assert trajectory.task is InspectionTask.FIELD
    assert trajectory.method == "ORACLE_FIELD"
    assert trajectory.steps
    assert all(step.budget_after <= endpoint + 1.0e-15 for step in trajectory.steps)
    assert all(
        step.task_loss_after <= step.task_loss_before + 1.0e-12
        for step in trajectory.steps
    )
    assert (
        trajectory.final_observation.state_sha256
        == trajectory.steps[-1].state_sha256_after
    )


class _FakeEncoder:
    def encode(self, images: object) -> np.ndarray:
        values = tuple(images)
        output = np.zeros((len(values), 512), dtype=np.float64)
        output[:, 0] = [np.mean(image, dtype=np.float64) / 255.0 for image in values]
        return output


class _FakeAssessor:
    model_state_sha256 = "e" * 64

    def predict(self, embeddings: object, scalars: object) -> np.ndarray:
        values = np.asarray(embeddings)
        state = np.asarray(scalars)
        return values[:, 0] + state[:, 0]


def test_cai_oracle_scores_candidate_error_reduction_with_frozen_interfaces() -> None:
    image = np.zeros((41, 43, 3), dtype=np.uint8)
    image[20:30, 20:30] = 255
    world = _world(image, InspectionTask.CAI, 0.25)
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    selection = choose_cai_action(
        world.reset(),
        grid,
        _prior(),
        full_scan=image,
        true_cai=1.0,
        assessor=_FakeAssessor(),
        encoder=_FakeEncoder(),
        checkpoint=0.25,
    )
    assert selection.candidates
    assert selection.exact_added_cost > 0
    assert np.isfinite(selection.objective_value)
