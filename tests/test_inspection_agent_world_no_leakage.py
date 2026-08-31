from __future__ import annotations

import hashlib

import numpy as np
from mavis_test_support import synthetic_inputs

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.state import (
    InspectionCellAction,
    action_added_positions,
    zero_state,
)
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid


def _world_pair() -> tuple[CausalInspectionWorld, CausalInspectionWorld]:
    first_inputs = synthetic_inputs(true_cai=0.1)
    second_inputs = synthetic_inputs(true_cai=99.0)
    first_image = first_inputs["images"][0]
    second_image = second_inputs["images"][0]
    grid = build_acquisition_grid(*first_image.shape[:2], initial_budget=0.015625)
    action = InspectionCellAction(0, -1, 0)
    visible = action_added_positions(grid, zero_state(grid), action)
    keep = np.zeros(first_image.shape[:2], dtype=np.bool_)
    keep[visible[:, 0], visible[:, 1]] = True
    second_image[~keep] = 255 - first_image[~keep]
    first = MAVISAuthority.from_arrays(**first_inputs)
    second = MAVISAuthority.from_arrays(**second_inputs)
    surface = np.full((17, 19, 3), 127, dtype=np.uint8)
    surface_hash = hashlib.sha256(surface.tobytes(order="C")).hexdigest()
    return (
        CausalInspectionWorld(
            first,
            specimen_id="sample-001",
            task=InspectionTask.FIELD,
            surface_rgb=surface,
            surface_sha256=surface_hash,
            grid=grid,
            endpoint_budget=0.25,
        ),
        CausalInspectionWorld(
            second,
            specimen_id="sample-001",
            task=InspectionTask.FIELD,
            surface_rgb=surface,
            surface_sha256=surface_hash,
            grid=grid,
            endpoint_budget=0.25,
        ),
    )


def test_world_observation_excludes_privileged_identity_and_future_values() -> None:
    first, second = _world_pair()
    first_zero = first.reset()
    second_zero = second.reset()
    assert first_zero == second_zero
    for forbidden in (
        "full_scan",
        "true_cai",
        "specimen_id",
        "dataset_id",
        "context_features",
        "oracle_value",
    ):
        assert not hasattr(first_zero, forbidden)

    action = InspectionCellAction(0, -1, 0)
    first_state = first.step(first_zero, action)
    second_state = second.step(second_zero, action)
    assert first_state == second_state
    np.testing.assert_array_equal(
        first_state.measurement_values,
        second_state.measurement_values,
    )


def test_world_transition_preserves_all_previously_acquired_values() -> None:
    first, _second = _world_pair()
    current = first.step(first.reset(), InspectionCellAction(0, -1, 0))
    old_positions = current.acquired_positions.copy()
    old_values = current.measurement_values.copy()

    refined = first.step(current, InspectionCellAction(0, 0, 1))
    lookup = {
        tuple(position): value
        for position, value in zip(
            refined.acquired_positions, refined.measurement_values, strict=True
        )
    }
    np.testing.assert_array_equal(
        np.asarray([lookup[tuple(position)] for position in old_positions]),
        old_values,
    )
    assert not refined.acquired_positions.flags.writeable
    assert not refined.measurement_values.flags.writeable
