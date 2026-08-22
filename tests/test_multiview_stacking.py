from __future__ import annotations

import numpy as np

from cmc_bbdm.aei_multiview_regression.stacking import (
    fit_stacker,
    select_stacker_oof,
)


def _data() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(20260821)
    domains = tuple(f"d{index}" for index in range(4) for _ in range(12))
    predictions = rng.normal(size=(48, 3))
    targets = (
        0.7 * predictions[:, 0]
        + 0.2 * predictions[:, 1]
        + rng.normal(scale=0.05, size=48)
    )
    return predictions, targets, domains


def test_nonnegative_ridge_stacker_has_nonnegative_coefficients() -> None:
    predictions, targets, _domains = _data()
    fit = fit_stacker(predictions, targets, method="nonnegative_ridge", alpha=1.0)

    assert np.all(fit.coef >= 0.0)
    assert np.all(np.isfinite(fit.predict(predictions[:5])))


def test_nonnegative_ridge_is_stable_for_nearly_collinear_views() -> None:
    rng = np.random.default_rng(99)
    latent = rng.normal(size=120)
    predictions = np.column_stack(
        (
            latent + rng.normal(scale=0.005, size=120),
            latent + rng.normal(scale=0.007, size=120),
            latent + rng.normal(scale=0.009, size=120),
        )
    )
    targets = 0.6 * predictions[:, 0] + 0.3 * predictions[:, 1]

    fit = fit_stacker(
        predictions, targets, method="nonnegative_ridge", alpha=1.0
    )

    assert np.all(fit.coef >= 0.0)
    assert np.all(np.isfinite(fit.predict(predictions)))
    assert np.mean((fit.predict(predictions) - targets) ** 2) < np.var(targets)


def test_meta_selection_uses_only_domain_heldout_base_oof_predictions() -> None:
    predictions, targets, domains = _data()
    events = []
    result = select_stacker_oof(
        predictions,
        targets,
        domains,
        methods=("ridge", "nonnegative_ridge", "huber"),
        fit_hook=events.append,
    )

    assert result.selected_method in {"ridge", "nonnegative_ridge", "huber"}
    assert len(events) == 12
    assert all(event.base_prediction_role == "source_oof" for event in events)
    assert all(event.query_domain not in event.fit_domains for event in events)
