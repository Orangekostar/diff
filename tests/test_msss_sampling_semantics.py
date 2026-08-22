from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.cpb_sparse_scan.sampling import reconstruct_sparse_rgb
from cmc_bbdm.msss.sampling_scale import (
    MSSSSamplingError,
    reconstruct_sampling_scale,
)


def _rgb(height: int = 17, width: int = 19) -> np.ndarray:
    values = np.arange(height * width * 3, dtype=np.int64).reshape(height, width, 3)
    return np.asarray(values % 256, dtype=np.uint8)


@pytest.mark.parametrize("density", [0.5, 0.25, 0.125])
def test_msss_sampling_is_bit_exact_with_p5(density: float) -> None:
    image = _rgb()
    expected, p5 = reconstruct_sparse_rgb(
        image,
        specimen_id="s1",
        dataset_id="d1",
        density=density,
        interpolation="bilinear",
    )
    actual, record = reconstruct_sampling_scale(
        image,
        specimen_id="s1",
        dataset_id="d1",
        requested_density=density,
    )

    np.testing.assert_array_equal(actual, expected)
    assert record.row_indices_sha256 == p5.row_indices_sha256
    assert record.column_indices_sha256 == p5.column_indices_sha256
    assert record.measured_points_exact
    assert record.effective_density == p5.actual_density


def test_msss_sampling_records_effective_integer_grid() -> None:
    image = _rgb(23, 29)
    output, record = reconstruct_sampling_scale(
        image,
        specimen_id="s1",
        dataset_id="d1",
        requested_density=0.1875,
    )

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert record.requested_density == 0.1875
    assert record.measured_points == record.row_count * record.column_count
    assert record.effective_density == pytest.approx(
        record.measured_points / (23 * 29)
    )
    assert record.rows[0] == 0 and record.rows[-1] == 22
    assert record.columns[0] == 0 and record.columns[-1] == 28
    assert record.vertical_stride_px == pytest.approx(22 / (record.row_count - 1))
    assert record.horizontal_stride_px == pytest.approx(28 / (record.column_count - 1))


def test_sampling_full_density_is_bit_exact_identity() -> None:
    image = _rgb()
    output, record = reconstruct_sampling_scale(
        image,
        specimen_id="s1",
        dataset_id="d1",
        requested_density=1.0,
    )

    np.testing.assert_array_equal(output, image)
    assert record.effective_density == 1.0
    assert record.measured_points == image.shape[0] * image.shape[1]


@pytest.mark.parametrize("density", [0.0, 0.2, 1.1, True, float("nan")])
def test_sampling_rejects_unregistered_density(density: object) -> None:
    with pytest.raises(MSSSSamplingError, match="registered"):
        reconstruct_sampling_scale(
            _rgb(),
            specimen_id="s1",
            dataset_id="d1",
            requested_density=density,
        )
