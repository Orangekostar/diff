from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.msss.gaussian_scale import MSSSGaussianError, gaussian_scale


def _rgb() -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(17))
    return generator.integers(0, 256, size=(31, 27, 3), dtype=np.uint8)


def test_gaussian_zero_is_bit_exact_identity() -> None:
    image = _rgb()
    output, record = gaussian_scale(image, sigma_px=0.0)

    np.testing.assert_array_equal(output, image)
    assert output.dtype == np.uint8
    assert not output.flags.writeable
    assert record.input_sha256 == record.output_sha256
    assert record.sigma_mm is None


def test_gaussian_preserves_shape_range_and_channel_isolation() -> None:
    image = np.zeros((19, 21, 3), dtype=np.uint8)
    image[9, 10, 0] = 255
    output, record = gaussian_scale(image, sigma_px=2.0)

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert int(output.min()) >= 0 and int(output.max()) <= 255
    assert np.count_nonzero(output[:, :, 0]) > 1
    assert np.count_nonzero(output[:, :, 1:]) == 0
    assert record.shape_preserved and record.dtype_preserved
    repeated, repeated_record = gaussian_scale(image, sigma_px=2.0)
    np.testing.assert_array_equal(repeated, output)
    assert repeated_record == record


def test_gaussian_constant_field_retains_intensity_semantics() -> None:
    image = np.full((17, 23, 3), 117, dtype=np.uint8)
    output, _ = gaussian_scale(image, sigma_px=8.0)
    np.testing.assert_array_equal(output, image)


@pytest.mark.parametrize("sigma", [-1.0, 0.25, 5.0, True, float("inf")])
def test_gaussian_rejects_unregistered_sigma(sigma: object) -> None:
    with pytest.raises(MSSSGaussianError, match="registered"):
        gaussian_scale(_rgb(), sigma_px=sigma)
