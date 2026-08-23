from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.interpolation import (
    RefinementPatchCache,
    reconstruct_measurement_state,
    refine_reconstruction,
)
from cmc_bbdm.mva.measurement_state import (
    apply_action,
    initial_state,
    legal_actions,
)


@pytest.mark.parametrize("method", ("nearest", "bilinear", "bicubic"))
def test_incremental_candidate_is_byte_exact_full_reconstruction(method: str) -> None:
    rng = np.random.default_rng(20260823)
    image = rng.integers(0, 256, size=(91, 93, 3), dtype=np.uint8)
    grid = build_acquisition_grid(91, 93, initial_budget=0.03125)
    state = initial_state(grid)
    for cell_index in (0, 9, 27, 63, 0, 27, 36):
        action = next(
            value
            for value in legal_actions(grid, state)
            if value.cell_index == cell_index
        )
        state = apply_action(grid, state, action)
    current = reconstruct_measurement_state(
        image, grid, state, interpolation=method, specimen_id="s", dataset_id="d"
    )

    for action in legal_actions(grid, state):
        candidate_state = apply_action(grid, state, action)
        incremental = refine_reconstruction(
            image,
            grid,
            state,
            current.image,
            action,
            interpolation=method,
        )
        full = reconstruct_measurement_state(
            image,
            grid,
            candidate_state,
            interpolation=method,
            specimen_id="s",
            dataset_id="d",
        )

        assert np.array_equal(incremental, full.image), action
        assert not incremental.flags.writeable


def test_incremental_refinement_does_not_mutate_current_image() -> None:
    image = np.arange(41 * 43 * 3, dtype=np.uint32).reshape(41, 43, 3).astype(np.uint8)
    grid = build_acquisition_grid(41, 43, initial_budget=0.0625)
    state = initial_state(grid)
    current = reconstruct_measurement_state(
        image, grid, state, interpolation="bilinear", specimen_id="s", dataset_id="d"
    )
    before = current.image.tobytes()

    _candidate = refine_reconstruction(
        image,
        grid,
        state,
        current.image,
        legal_actions(grid, state)[10],
        interpolation="bilinear",
    )

    assert current.image.tobytes() == before


def test_refinement_patch_cache_is_source_bound_and_byte_exact() -> None:
    image = np.arange(61 * 67 * 3, dtype=np.uint32).reshape(61, 67, 3).astype(np.uint8)
    grid = build_acquisition_grid(61, 67, initial_budget=0.03125)
    state = initial_state(grid)
    current = reconstruct_measurement_state(
        image, grid, state, interpolation="bilinear", specimen_id="s", dataset_id="d"
    )
    action = legal_actions(grid, state)[12]
    cache = RefinementPatchCache(image=image, grid=grid)

    first = refine_reconstruction(
        image,
        grid,
        state,
        current.image,
        action,
        interpolation="bilinear",
        patch_cache=cache,
    )
    second = refine_reconstruction(
        image,
        grid,
        state,
        current.image,
        action,
        interpolation="bilinear",
        patch_cache=cache,
    )

    assert np.array_equal(first, second)
    assert len(cache.patches) == 1
    with pytest.raises(ValueError, match="source"):
        refine_reconstruction(
            image.copy(),
            grid,
            state,
            current.image,
            action,
            interpolation="bilinear",
            patch_cache=cache,
        )
