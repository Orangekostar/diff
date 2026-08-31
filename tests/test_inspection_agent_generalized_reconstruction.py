from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    SourceBackgroundPrior,
    reconstruct_observation,
)
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.interpolation import reconstruct_measurement_state
from cmc_bbdm.mva.measurement_state import MeasurementState
from cmc_bbdm.mva.oracle import uniform_cell_order
from cmc_bbdm.mva.reconstruction_value import normalized_rgb_mse


def _image(shape: tuple[int, int] = (41, 43)) -> np.ndarray:
    rows, columns = np.indices(shape)
    return np.stack(
        (
            (rows * 4 + columns * 2) % 256,
            (rows * 3 + columns * 5) % 256,
            (rows * 7 + columns) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def _authority(image: np.ndarray) -> MAVISAuthority:
    return MAVISAuthority.from_arrays(
        specimen_ids=("sample",),
        dataset_ids=("target",),
        images=(image,),
        targets=np.asarray([0.4]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )


def _prior() -> SourceBackgroundPrior:
    return SourceBackgroundPrior(
        outer_domain="target",
        source_domains=("source-a", "source-b"),
        fit_specimen_ids=("a", "b"),
        source_authority_sha256="a" * 64,
        domain_border_medians=np.asarray(((10, 20, 30), (30, 40, 50))),
        background_rgb=np.asarray((20, 30, 40), dtype=np.uint8),
    )


def _world(image: np.ndarray, *, endpoint: float = 1.0) -> CausalInspectionWorld:
    grid = build_acquisition_grid(*image.shape[:2], initial_budget=0.015625)
    surface = np.zeros((17, 19, 3), dtype=np.uint8)
    return CausalInspectionWorld(
        _authority(image),
        specimen_id="sample",
        task=InspectionTask.FIELD,
        surface_rgb=surface,
        surface_sha256=hashlib.sha256(surface.tobytes()).hexdigest(),
        grid=grid,
        endpoint_budget=endpoint,
    )


def test_zero_reconstruction_uses_only_source_background_prior() -> None:
    image = _image()
    world = _world(image)
    grid = build_acquisition_grid(*image.shape[:2], initial_budget=0.015625)
    reconstruction = reconstruct_observation(world.reset(), grid, _prior())
    expected = np.broadcast_to(np.asarray((20, 30, 40), dtype=np.uint8), image.shape)
    np.testing.assert_array_equal(reconstruction.image, expected)
    assert reconstruction.measured_count == 0


def test_unmeasured_target_pixels_cannot_change_reconstruction() -> None:
    image = _image()
    changed = image.copy()
    first = _world(image)
    grid = build_acquisition_grid(*image.shape[:2], initial_budget=0.015625)
    current = first.step(first.reset(), InspectionCellAction(0, -1, 0))
    mask = np.zeros(image.shape[:2], dtype=np.bool_)
    mask[current.acquired_positions[:, 0], current.acquired_positions[:, 1]] = True
    changed[~mask] = 255 - changed[~mask]
    second = _world(changed)
    other = second.step(second.reset(), InspectionCellAction(0, -1, 0))

    np.testing.assert_array_equal(current.measurement_values, other.measurement_values)
    np.testing.assert_array_equal(
        reconstruct_observation(current, grid, _prior()).image,
        reconstruct_observation(other, grid, _prior()).image,
    )


def test_observed_pixels_are_exact_and_smooth_refinement_is_monotone() -> None:
    image = _image()
    world = _world(image)
    grid = build_acquisition_grid(*image.shape[:2], initial_budget=0.015625)
    observation = world.step(world.reset(), InspectionCellAction(27, -1, 0))
    losses = []
    for action in (
        None,
        InspectionCellAction(27, 0, 1),
        InspectionCellAction(27, 1, 2),
    ):
        if action is not None:
            observation = world.step(observation, action)
        result = reconstruct_observation(observation, grid, _prior())
        np.testing.assert_array_equal(
            result.image[
                observation.acquired_positions[:, 0],
                observation.acquired_positions[:, 1],
            ],
            observation.measurement_values,
        )
        losses.append(normalized_rgb_mse(image, result.image))
    assert losses[0] >= losses[1] >= losses[2]


def test_complete_nonnegative_state_matches_frozen_mva_reconstruction() -> None:
    image = _image()
    world = _world(image)
    grid = build_acquisition_grid(*image.shape[:2], initial_budget=0.015625)
    observation = world.reset()
    for cell in uniform_cell_order():
        observation = world.step(observation, InspectionCellAction(cell, -1, 0))
    for cell in (0, 9, 27, 63):
        observation = world.step(observation, InspectionCellAction(cell, 0, 1))

    generalized = reconstruct_observation(observation, grid, _prior())
    legacy_state = MeasurementState(
        grid_sha256=grid.state_sha256,
        levels=observation.measurement_state.levels,
    )
    legacy = reconstruct_measurement_state(
        image,
        grid,
        legacy_state,
        interpolation="bilinear",
        specimen_id="sample",
        dataset_id="target",
    )
    np.testing.assert_array_equal(generalized.image, legacy.image)


def test_all_level2_reconstructs_hidden_full_scan_exactly() -> None:
    image = _image()
    world = _world(image)
    grid = build_acquisition_grid(*image.shape[:2], initial_budget=0.015625)
    observation = world.reset()
    for source, target in ((-1, 0), (0, 1), (1, 2)):
        for cell in uniform_cell_order():
            observation = world.step(
                observation,
                InspectionCellAction(cell, source, target),
            )
    result = reconstruct_observation(observation, grid, _prior())
    np.testing.assert_array_equal(result.image, image)
    assert result.measured_count == image.shape[0] * image.shape[1]
    assert not result.image.flags.writeable
