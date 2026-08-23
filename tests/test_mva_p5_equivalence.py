from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.cpb_sparse_scan.sampling import reconstruct_sparse_rgb
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.interpolation import reconstruct_measurement_state
from cmc_bbdm.mva.measurement_state import MeasurementState, measurement_mask


@pytest.mark.parametrize("shape", ((674, 675), (338, 352), (338, 340)))
@pytest.mark.parametrize("method", ("nearest", "bilinear", "bicubic"))
def test_all_level1_is_byte_exact_p5_reconstruction(
    shape: tuple[int, int], method: str
) -> None:
    rng = np.random.default_rng(31 + shape[0] + shape[1])
    image = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    grid = build_acquisition_grid(*shape, initial_budget=0.015625)
    state = MeasurementState(grid_sha256=grid.state_sha256, levels=(1,) * 64)

    mva = reconstruct_measurement_state(
        image,
        grid,
        state,
        interpolation=method,
        specimen_id="same",
        dataset_id="same-domain",
    )
    p5, p5_record = reconstruct_sparse_rgb(
        image,
        specimen_id="same",
        dataset_id="same-domain",
        density=0.25,
        interpolation=method,
    )

    expected_mask = np.zeros(shape, dtype=np.bool_)
    expected_mask[np.ix_(grid.level1_rows, grid.level1_columns)] = True
    assert np.array_equal(measurement_mask(grid, state), expected_mask)
    assert np.array_equal(mva.image, p5)
    assert mva.output_sha256 == p5_record.output_sha256
    assert mva.p5_equivalent
