from __future__ import annotations

import inspect

import numpy as np

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import apply_action, initial_state
from cmc_bbdm.mva.policy_state import build_policy_observation


def _reconstruction(rows: int, columns: int) -> np.ndarray:
    row = np.arange(rows, dtype=np.uint16)[:, None]
    column = np.arange(columns, dtype=np.uint16)[None, :]
    return np.stack(
        (
            np.broadcast_to((3 * row) % 256, (rows, columns)),
            np.broadcast_to((5 * column) % 256, (rows, columns)),
            (row + column) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def test_policy_observation_has_registered_shapes_and_action_order() -> None:
    grid = build_acquisition_grid(338, 352, initial_budget=0.015625)
    state = initial_state(grid)
    reconstruction = _reconstruction(*grid.native_shape)
    embedding = np.linspace(-1.0, 1.0, 512, dtype=np.float64)

    observation = build_policy_observation(
        grid,
        state,
        current_reconstruction=reconstruction,
        current_embedding=embedding,
        current_prediction=0.73,
        checkpoint=0.0625,
        maximum_budget=0.25,
    )

    assert observation.global_features.shape == (579,)
    assert observation.candidate_features.shape == (len(observation.actions), 8)
    assert tuple(action.cell_index for action in observation.actions) == tuple(
        range(len(observation.actions))
    )
    assert np.all(np.isfinite(observation.global_features))
    assert np.all(np.isfinite(observation.candidate_features))
    assert np.all(observation.candidate_features[:, :3] >= 0.0)
    assert np.all(observation.candidate_features[:, :3] <= 1.0)
    assert not observation.global_features.flags.writeable
    assert not observation.candidate_features.flags.writeable
    assert observation.used_budget > 0.0
    assert observation.remaining_budget == 0.25 - observation.used_budget


def test_policy_features_change_only_with_observed_state() -> None:
    grid = build_acquisition_grid(338, 340, initial_budget=0.03125)
    state = initial_state(grid)
    reconstruction = _reconstruction(*grid.native_shape)
    embedding = np.ones(512, dtype=np.float64)
    first = build_policy_observation(
        grid,
        state,
        current_reconstruction=reconstruction,
        current_embedding=embedding,
        current_prediction=0.5,
        checkpoint=0.09375,
        maximum_budget=0.25,
    )

    unseen_true_image = reconstruction.copy()
    unseen_true_image[~np.zeros(grid.native_shape, dtype=np.bool_)] = 255
    second = build_policy_observation(
        grid,
        state,
        current_reconstruction=reconstruction,
        current_embedding=embedding,
        current_prediction=0.5,
        checkpoint=0.09375,
        maximum_budget=0.25,
    )

    assert np.array_equal(first.global_features, second.global_features)
    assert np.array_equal(first.candidate_features, second.candidate_features)
    assert unseen_true_image is not reconstruction

    refined = apply_action(grid, state, first.actions[0])
    changed = build_policy_observation(
        grid,
        refined,
        current_reconstruction=reconstruction,
        current_embedding=embedding,
        current_prediction=0.5,
        checkpoint=0.09375,
        maximum_budget=0.25,
    )
    assert not np.array_equal(first.global_features[512:576], changed.global_features[512:576])


def test_policy_observation_api_cannot_receive_forbidden_evidence() -> None:
    parameters = set(inspect.signature(build_policy_observation).parameters)

    assert not parameters & {
        "target",
        "true_cai",
        "full_image",
        "source_image",
        "oracle_values",
        "unmeasured_pixels",
    }
