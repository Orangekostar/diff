from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.msss.wavelet_scale import MSSSWaveletError, wavelet_scale


def _rgb(height: int = 31, width: int = 27) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(19))
    return generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize("wavelet", ["db2", "haar", "db4"])
def test_wavelet_level_zero_is_bit_exact_identity(wavelet: str) -> None:
    image = _rgb()
    output, record = wavelet_scale(
        image, wavelet=wavelet, level=0, mode="low_only"
    )

    np.testing.assert_array_equal(output, image)
    assert not output.flags.writeable
    assert record.input_sha256 == record.output_sha256
    assert record.reconstruction_shape == image.shape


def test_wavelet_low_pass_preserves_original_odd_shape_and_dtype() -> None:
    image = _rgb(43, 37)
    output, record = wavelet_scale(
        image, wavelet="db2", level=3, mode="low_only"
    )

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert record.shape_preserved and record.dtype_preserved
    assert record.retained_approximation
    assert record.retained_detail_levels == ()
    assert record.coefficient_sha256 != record.output_sha256


def test_wavelet_boundary_details_are_audited_and_deterministic() -> None:
    image = _rgb(43, 37)
    output, record = wavelet_scale(
        image,
        wavelet="db2",
        level=2,
        mode="low_plus_boundary_details",
    )
    repeated, repeated_record = wavelet_scale(
        image,
        wavelet="db2",
        level=2,
        mode="low_plus_boundary_details",
    )

    np.testing.assert_array_equal(repeated, output)
    assert repeated_record == record
    assert record.retained_detail_levels == (2,)


def test_wavelet_level_one_with_boundary_details_reconstructs_uint8_input() -> None:
    image = _rgb(45, 39)
    output, _ = wavelet_scale(
        image,
        wavelet="db2",
        level=1,
        mode="low_plus_boundary_details",
    )
    np.testing.assert_array_equal(output, image)


@pytest.mark.parametrize(
    ("wavelet", "level", "mode"),
    [
        ("sym4", 1, "low_only"),
        ("db2", 4, "low_only"),
        ("db2", -1, "low_only"),
        ("db2", 1, "high_only"),
        ("db2", True, "low_only"),
    ],
)
def test_wavelet_rejects_unregistered_condition(
    wavelet: object, level: object, mode: object
) -> None:
    with pytest.raises(MSSSWaveletError, match="registered"):
        wavelet_scale(_rgb(), wavelet=wavelet, level=level, mode=mode)
