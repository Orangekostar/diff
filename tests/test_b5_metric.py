from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mva.budget_metrics import simulated_saving, sufficiency_budget

FULL = 0.08963580465761432


def test_b5_is_first_registered_checkpoint_at_threshold() -> None:
    budgets = np.asarray([0.0625, 0.09375, 0.125, 0.1875, 0.25])
    mae = np.asarray([0.12, 0.10, 1.05 * FULL, 0.09, 0.089])

    assert sufficiency_budget(budgets, mae, full_mae=FULL, tolerance=0.05) == 0.125


def test_sufficiency_budget_remains_none_when_unreached() -> None:
    budgets = np.asarray([0.0625, 0.125, 0.25])
    mae = np.asarray([0.2, 0.15, np.nextafter(1.05 * FULL, np.inf)])

    assert sufficiency_budget(budgets, mae, full_mae=FULL, tolerance=0.05) is None


def test_simulated_saving_requires_two_observed_budgets() -> None:
    assert simulated_saving(0.125, 0.25) == pytest.approx(0.5)
    assert simulated_saving(None, 0.25) is None
    assert simulated_saving(0.125, None) is None


def test_sufficiency_rejects_nan_instead_of_hiding_it() -> None:
    with pytest.raises(ValueError, match="finite"):
        sufficiency_budget(
            np.asarray([0.0625, 0.125]),
            np.asarray([0.2, np.nan]),
            full_mae=FULL,
            tolerance=0.05,
        )
