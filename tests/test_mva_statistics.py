from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mva.statistics import (
    paired_domain_bootstrap,
    synchronized_bootstrap_indices,
)


def test_bootstrap_indices_are_shared_deterministic_pcg64() -> None:
    first = synchronized_bootstrap_indices(seed=20260823, resamples=100_000, domains=6)
    second = synchronized_bootstrap_indices(seed=20260823, resamples=100_000, domains=6)

    assert first.shape == (100_000, 6)
    assert first.dtype == np.int16
    assert np.array_equal(first, second)
    assert not first.flags.writeable


def test_paired_effect_reuses_supplied_bootstrap_matrix() -> None:
    indices = synchronized_bootstrap_indices(seed=7, resamples=500, domains=6)
    baseline = np.asarray([0.20, 0.18, 0.22, 0.19, 0.21, 0.17])
    adaptive = baseline - 0.02

    result = paired_domain_bootstrap(
        baseline,
        adaptive,
        indices=indices,
        effect_id="baseline_minus_adaptive",
    )

    assert result.point_estimate == pytest.approx(0.02)
    assert result.lower == pytest.approx(0.02)
    assert result.upper == pytest.approx(0.02)
    assert result.improved_domains == 6
    assert (
        result.indices_sha256
        == paired_domain_bootstrap(
            baseline + 1.0,
            adaptive + 1.0,
            indices=indices,
            effect_id="shifted",
        ).indices_sha256
    )


def test_bootstrap_preserves_adverse_domains() -> None:
    indices = synchronized_bootstrap_indices(seed=9, resamples=1000, domains=6)
    baseline = np.asarray([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    adaptive = np.asarray([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])

    result = paired_domain_bootstrap(
        baseline, adaptive, indices=indices, effect_id="mixed"
    )

    assert result.improved_domains == 3
    assert result.domain_effects == pytest.approx((0.1, 0.1, 0.1, -0.1, -0.1, -0.1))
