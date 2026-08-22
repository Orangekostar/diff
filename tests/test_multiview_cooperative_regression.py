from __future__ import annotations

import numpy as np

from cmc_bbdm.aei_multiview_regression.cooperative_regression import (
    fit_cooperative,
)


def _designs() -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    rng = np.random.default_rng(20260821)
    latent = rng.normal(size=(80, 4))
    targets = 0.7 * latent[:, 0] - 0.4 * latent[:, 1] + rng.normal(scale=0.2, size=80)
    designs = tuple(
        np.column_stack(
            (
                latent + rng.normal(scale=scale, size=latent.shape),
                rng.normal(size=(80, 2)),
            )
        )
        for scale in (0.05, 0.2, 0.4)
    )
    return designs, targets


def test_lambda_zero_matches_independent_ridge() -> None:
    designs, targets = _designs()
    joint = fit_cooperative(
        designs, targets, lambda_consistency=0.0, loss="mse", alpha=10.0
    )
    independent = np.column_stack(
        [
            fit_cooperative(
                (design,), targets, lambda_consistency=0.0, loss="mse", alpha=10.0
            ).train_predictions[:, 0]
            for design in designs
        ]
    )

    np.testing.assert_allclose(
        joint.train_predictions, independent, atol=1e-12, rtol=0.0
    )


def test_large_lambda_reduces_training_disagreement() -> None:
    designs, targets = _designs()
    low = fit_cooperative(
        designs, targets, lambda_consistency=0.0, loss="mse", alpha=10.0
    )
    high = fit_cooperative(
        designs, targets, lambda_consistency=1.0, loss="mse", alpha=10.0
    )

    assert high.mean_absolute_disagreement < low.mean_absolute_disagreement
    assert high.train_predictions.shape == (80, 3)
    assert high.train_predictions.flags.writeable is False


def test_huber_fit_is_finite_and_predicts_aligned_queries() -> None:
    designs, targets = _designs()
    fit = fit_cooperative(
        designs,
        targets,
        lambda_consistency=0.03,
        loss="huber",
        huber_delta=0.05,
        alpha=10.0,
    )

    prediction = fit.predict(tuple(item[:7] for item in designs))
    assert prediction.shape == (7, 3)
    assert np.all(np.isfinite(prediction))
    assert fit.optimizer_success is True
