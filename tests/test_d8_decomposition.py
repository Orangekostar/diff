from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.cpb_diffusion_marginalization.decomposition import (
    decompose_residual,
    empirical_control,
    gaussian_control,
    phase_randomized_control,
)
from cmc_bbdm.cpb_diffusion_marginalization.residuals import (
    P6ResidualBank,
    ResidualRecord,
)


@pytest.fixture
def residual() -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(20260820))
    return generator.normal(0.0, 0.2, size=(3, 64, 64)).astype(np.float32)


@pytest.mark.parametrize(
    ("family", "parameters"),
    (
        ("gaussian", {"band": "mid", "sigma": 2.0}),
        (
            "fourier",
            {"band": "high", "cutoff": 0.20, "transition": 0.05},
        ),
        (
            "wavelet",
            {"band": "mid", "wavelet": "db2", "level": 2},
        ),
    ),
)
def test_decomposition_reconstructs_residual_and_preserves_shape(
    residual: np.ndarray, family: str, parameters: dict[str, object]
) -> None:
    bands = decompose_residual(
        residual,
        family=family,
        parameters=parameters,
    )
    assert bands.selected.shape == residual.shape
    assert bands.selected.dtype == np.float32
    assert bands.selected.flags.writeable is False
    assert np.isfinite(bands.selected).all()
    assert bands.reconstruction_error <= 1.0e-6
    np.testing.assert_allclose(
        bands.low + bands.mid + bands.high,
        residual,
        atol=1.0e-6,
        rtol=0.0,
    )


def test_fourier_high_band_rejects_constant_field() -> None:
    residual = np.ones((3, 64, 64), dtype=np.float32)
    bands = decompose_residual(
        residual,
        family="fourier",
        parameters={"band": "high", "cutoff": 0.2, "transition": 0.05},
    )
    assert np.max(np.abs(bands.selected)) <= 1.0e-6


def test_fourier_low_band_preserves_dc_when_transition_exceeds_cutoff() -> None:
    residual = np.ones((3, 64, 64), dtype=np.float32)
    bands = decompose_residual(
        residual,
        family="fourier",
        parameters={"band": "low", "cutoff": 0.08, "transition": 0.10},
    )
    np.testing.assert_allclose(bands.selected, residual, atol=1.0e-6, rtol=0.0)


def test_noise_controls_are_seeded_and_match_registered_second_order_state(
    residual: np.ndarray,
) -> None:
    gaussian_first = gaussian_control(residual, seed=11)
    gaussian_second = gaussian_control(residual, seed=11)
    np.testing.assert_array_equal(gaussian_first, gaussian_second)
    np.testing.assert_allclose(
        gaussian_first.mean(axis=(1, 2)),
        residual.mean(axis=(1, 2)),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        gaussian_first.std(axis=(1, 2)),
        residual.std(axis=(1, 2)),
        atol=1.0e-6,
    )

    phase_first = phase_randomized_control(residual, seed=17)
    phase_second = phase_randomized_control(residual, seed=17)
    np.testing.assert_array_equal(phase_first, phase_second)
    np.testing.assert_allclose(
        np.abs(np.fft.fft2(phase_first, axes=(-2, -1))),
        np.abs(np.fft.fft2(residual, axes=(-2, -1))),
        atol=2.0e-4,
        rtol=1.0e-5,
    )
    assert not np.array_equal(phase_first, residual)


def test_empirical_control_uses_only_fit_domains_and_excludes_self_donation() -> None:
    def record(specimen_id: str, dataset_id: str, value: float) -> ResidualRecord:
        return ResidualRecord(
            specimen_id=specimen_id,
            dataset_id=dataset_id,
            heldout_domain=dataset_id,
            draw_index=0,
            residual_64=np.full((3, 64, 64), value, dtype=np.float32),
            source_sha256="a" * 64,
            checkpoint_scientific_digest="b" * 64,
            checkpoint_train_ids=("fit-specimen",),
            checkpoint_train_domains=("other-domain",),
        )

    records = (
        record("fit-a", "fit-domain", 1.0),
        record("fit-b", "fit-domain", 2.0),
        record("excluded", "query-domain", 99.0),
    )
    bank = P6ResidualBank(
        records=records,
        specimen_count=3,
        draw_count=1,
        maximum_mean_error=0.0,
        maximum_variance_error=0.0,
        state_sha256="c" * 64,
    )

    first = empirical_control(
        bank,
        fit_domains=("fit-domain",),
        query_ids=("fit-a", "fit-b", "heldout"),
        seed=23,
    )
    second = empirical_control(
        bank,
        fit_domains=("fit-domain",),
        query_ids=("fit-a", "fit-b", "heldout"),
        seed=23,
    )

    assert all(not value.flags.writeable for value in first)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert np.all(first[0] == 2.0)
    assert np.all(first[1] == 1.0)
    assert all(not np.all(value == 99.0) for value in first)


@pytest.mark.parametrize(
    ("family", "parameters"),
    (
        ("unknown", {"band": "low"}),
        ("gaussian", {"band": "low", "sigma": -1.0}),
        (
            "fourier",
            {"band": "low", "cutoff": 0.9, "transition": 0.2},
        ),
        (
            "wavelet",
            {"band": "low", "wavelet": "not-a-wavelet", "level": 2},
        ),
    ),
)
def test_decomposition_rejects_unregistered_or_invalid_parameters(
    residual: np.ndarray, family: str, parameters: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        decompose_residual(residual, family=family, parameters=parameters)
