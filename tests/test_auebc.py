from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mva.budget_metrics import auebc


def test_auebc_uses_registered_nominal_interval_and_trapezoids() -> None:
    budgets = np.asarray([0.03125, 0.0625, 0.125, 0.1875, 0.25, 0.5])
    mae = np.asarray([0.20, 0.18, 0.14, 0.12, 0.10, 0.09])

    result = auebc(budgets, mae, lower=0.0625, upper=0.25)

    expected = np.trapezoid(
        np.asarray([0.18, 0.14, 0.12, 0.10]),
        np.asarray([0.0625, 0.125, 0.1875, 0.25]),
    )
    assert result == pytest.approx(expected)


def test_auebc_requires_both_registered_endpoints() -> None:
    with pytest.raises(ValueError, match="endpoints"):
        auebc(
            np.asarray([0.09375, 0.125, 0.1875, 0.25]),
            np.asarray([0.2, 0.18, 0.16, 0.14]),
            lower=0.0625,
            upper=0.25,
        )


def test_auebc_rejects_unsorted_or_duplicate_budgets() -> None:
    with pytest.raises(ValueError, match="increasing"):
        auebc(
            np.asarray([0.0625, 0.125, 0.125, 0.25]),
            np.asarray([0.2, 0.18, 0.16, 0.14]),
            lower=0.0625,
            upper=0.25,
        )
