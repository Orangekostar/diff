from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.interpolation import reconstruct_measurement_state
from cmc_bbdm.mva.measurement_state import (
    apply_action,
    initial_state,
    legal_actions,
    measurement_mask,
)


def _image(shape: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(20260823)
    return rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)


@pytest.mark.parametrize("method", ("nearest", "bilinear", "bicubic"))
def test_mixed_reconstruction_restores_every_measured_rgb(method: str) -> None:
    image = _image((338, 340))
    original_hash = hashlib.sha256(image.tobytes()).hexdigest()
    grid = build_acquisition_grid(338, 340, initial_budget=0.03125)
    state = initial_state(grid)
    for cell_index in (0, 9, 27, 63, 0, 27):
        action = next(
            item for item in legal_actions(grid, state) if item.cell_index == cell_index
        )
        state = apply_action(grid, state, action)

    result = reconstruct_measurement_state(
        image,
        grid,
        state,
        interpolation=method,
        specimen_id="synthetic",
        dataset_id="test",
    )
    mask = measurement_mask(grid, state)

    assert result.image.shape == image.shape
    assert result.image.dtype == image.dtype == np.uint8
    assert np.array_equal(result.image[mask], image[mask])
    assert result.measured_values_exact
    assert result.measured_count == int(np.count_nonzero(mask))
    assert hashlib.sha256(image.tobytes()).hexdigest() == original_hash
    assert not result.image.flags.writeable


def test_reconstruction_is_deterministic() -> None:
    image = _image((91, 93))
    grid = build_acquisition_grid(91, 93, initial_budget=0.0625)
    state = apply_action(
        grid, initial_state(grid), legal_actions(grid, initial_state(grid))[4]
    )

    first = reconstruct_measurement_state(
        image, grid, state, interpolation="bilinear", specimen_id="s", dataset_id="d"
    )
    second = reconstruct_measurement_state(
        image, grid, state, interpolation="bilinear", specimen_id="s", dataset_id="d"
    )

    assert first == second
    assert np.array_equal(first.image, second.image)
    assert first.output_sha256 == second.output_sha256
