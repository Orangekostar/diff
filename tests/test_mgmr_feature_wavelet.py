from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mgmr.feature_wavelet import (
    FeatureWaveletError,
    directional_gap,
    dwt2_feature_maps,
    idwt2_feature_maps,
)


@pytest.mark.parametrize("wavelet", ("db2", "haar"))
@pytest.mark.parametrize("side", (14, 28))
def test_feature_dwt_reconstructs_registered_map_sizes(
    wavelet: str, side: int
) -> None:
    rng = np.random.Generator(np.random.PCG64(20260822 + side))
    maps = rng.normal(size=(2, 3, side, side)).astype(np.float32)

    bands = dwt2_feature_maps(maps, wavelet=wavelet, mode="periodization")
    rebuilt = idwt2_feature_maps(bands)

    assert bands.ll.shape == (2, 3, side // 2, side // 2)
    assert bands.horizontal.shape == bands.ll.shape
    assert bands.vertical.shape == bands.ll.shape
    assert bands.diagonal.shape == bands.ll.shape
    assert rebuilt.shape == maps.shape
    assert rebuilt.dtype == np.float32
    np.testing.assert_allclose(rebuilt, maps, rtol=2.0e-6, atol=2.0e-6)


@pytest.mark.parametrize("wavelet", ("db2", "haar"))
def test_feature_dwt_band_orientation_is_explicit(wavelet: str) -> None:
    row_change = np.tile((np.arange(14) % 2)[:, None], (1, 14))
    column_change = row_change.T
    diagonal_change = (np.indices((14, 14)).sum(axis=0) % 2).astype(np.float32)
    maps = np.stack((row_change, column_change, diagonal_change), axis=0)[None].astype(
        np.float32
    )

    bands = dwt2_feature_maps(maps, wavelet=wavelet, mode="periodization")
    energies = np.stack(
        [
            np.sum(bands.horizontal**2, axis=(-2, -1)),
            np.sum(bands.vertical**2, axis=(-2, -1)),
            np.sum(bands.diagonal**2, axis=(-2, -1)),
        ],
        axis=-1,
    )[0]

    assert np.argmax(energies[0]) == 0
    assert np.argmax(energies[1]) == 1
    assert np.argmax(energies[2]) == 2


def test_directional_gap_keeps_three_bands_separate() -> None:
    maps = np.arange(2 * 4 * 14 * 14, dtype=np.float32).reshape(2, 4, 14, 14)
    bands = dwt2_feature_maps(maps, wavelet="db2", mode="periodization")

    feature = directional_gap(bands)

    expected = np.concatenate(
        [
            bands.horizontal.mean(axis=(-2, -1)),
            bands.vertical.mean(axis=(-2, -1)),
            bands.diagonal.mean(axis=(-2, -1)),
        ],
        axis=1,
    )
    assert feature.shape == (2, 12)
    assert feature.dtype == np.float32
    assert feature.flags.writeable is False
    np.testing.assert_array_equal(feature, expected)


def test_feature_dwt_is_bitwise_deterministic_and_immutable() -> None:
    maps = np.linspace(-2.0, 3.0, 2 * 5 * 14 * 14, dtype=np.float32).reshape(
        2, 5, 14, 14
    )

    first = dwt2_feature_maps(maps, wavelet="db2", mode="periodization")
    second = dwt2_feature_maps(maps, wavelet="db2", mode="periodization")

    assert first.state_sha256 == second.state_sha256
    for left, right in zip(first.coefficients, second.coefficients, strict=True):
        assert left.dtype == np.float32
        assert left.flags.writeable is False
        np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError):
        first.horizontal[0, 0, 0, 0] = 1.0


@pytest.mark.parametrize(
    ("maps", "wavelet", "mode", "message"),
    (
        (np.zeros((1, 2, 14, 14), dtype=np.float64), "db2", "periodization", "float32"),
        (np.zeros((1, 2, 14, 14), dtype=np.float32), "db4", "periodization", "wavelet"),
        (np.zeros((1, 2, 14, 14), dtype=np.float32), "db2", "reflect", "mode"),
        (np.zeros((2, 14, 14), dtype=np.float32), "db2", "periodization", "NCHW"),
    ),
)
def test_feature_dwt_rejects_unregistered_inputs(
    maps: np.ndarray, wavelet: str, mode: str, message: str
) -> None:
    with pytest.raises(FeatureWaveletError, match=message):
        dwt2_feature_maps(maps, wavelet=wavelet, mode=mode)
