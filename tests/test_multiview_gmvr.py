from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.aei_multiview_regression.gmvr_regression import fit_gmvr_weights


def test_gmvr_weights_are_nonnegative_simplex_with_named_contributions() -> None:
    rng = np.random.default_rng(8)
    predictions = rng.normal(size=(60, 3))
    targets = 0.8 * predictions[:, 0] + 0.1 * predictions[:, 1]
    fit = fit_gmvr_weights(
        predictions,
        targets,
        domains=tuple(f"d{index}" for index in range(3) for _ in range(20)),
        lambda_consistency=0.01,
        lambda_complementarity=0.1,
    )

    assert np.all(fit.weights >= 0.0)
    assert fit.weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert len(fit.mean_absolute_contributions) == 3
    assert fit.predictions.shape == targets.shape


def test_complementarity_penalty_reduces_weight_concentration() -> None:
    rng = np.random.default_rng(9)
    predictions = rng.normal(size=(80, 3))
    targets = predictions[:, 0] + rng.normal(scale=0.05, size=80)
    domains = tuple(f"d{index}" for index in range(4) for _ in range(20))
    low = fit_gmvr_weights(
        predictions,
        targets,
        domains=domains,
        lambda_consistency=0.0,
        lambda_complementarity=0.0,
    )
    high = fit_gmvr_weights(
        predictions,
        targets,
        domains=domains,
        lambda_consistency=0.0,
        lambda_complementarity=1.0,
    )

    assert np.max(high.weights) < np.max(low.weights)
