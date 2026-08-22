from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.aei_multiview_regression.late_fusion import (
    equal_fusion,
    fit_validation_weights,
)


def test_validation_weights_are_nonnegative_simplex_and_minimize_mae() -> None:
    predictions = np.asarray(
        [
            [0.0, 0.3, 0.5],
            [1.0, 0.7, 0.5],
            [2.0, 1.7, 1.5],
            [3.0, 2.7, 2.5],
        ]
    )
    targets = predictions[:, 0]
    fit = fit_validation_weights(predictions, targets, domains=("a", "a", "b", "b"))

    assert np.all(fit.weights >= 0.0)
    assert fit.weights.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(fit.predict(predictions), targets, atol=1e-12)
    assert fit.weights.flags.writeable is False


def test_equal_fusion_is_row_mean() -> None:
    predictions = np.arange(12, dtype=np.float64).reshape(4, 3)
    np.testing.assert_array_equal(
        equal_fusion(predictions), np.mean(predictions, axis=1)
    )
