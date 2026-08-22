from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.aei_multiview_regression.reliability import audit_reliability


def test_reliability_uses_population_dispersion_and_rank_strata() -> None:
    centers = np.linspace(0.1, 0.8, 8)
    spreads = np.asarray([0.01, 0.08, 0.02, 0.07, 0.03, 0.06, 0.04, 0.05])
    predictions = np.column_stack((centers - spreads, centers, centers + spreads))
    targets = centers + np.asarray([0.01, 0.08, 0.02, 0.07, 0.03, 0.06, 0.04, 0.05])

    result = audit_reliability(
        targets,
        predictions,
        deployable_predictions={"equal_fusion": centers},
    )

    np.testing.assert_allclose(
        result.dispersion,
        np.std(predictions, axis=1, ddof=0),
    )
    record = result.methods[0]
    assert record.method == "equal_fusion"
    assert record.pearson_r == pytest.approx(1.0)
    assert record.spearman_r == pytest.approx(1.0)
    assert record.pearson_p_value < 1e-10
    assert record.spearman_p_value < 1e-10
    np.testing.assert_allclose(record.absolute_errors, spreads)
    assert result.stratum_labels == (
        "lowest_25_percent",
        "highest_25_percent",
        "lowest_25_percent",
        "highest_25_percent",
        "middle_50_percent",
        "middle_50_percent",
        "middle_50_percent",
        "middle_50_percent",
    )
    assert tuple(item.count for item in record.strata) == (2, 4, 2)
    assert tuple(item.name for item in record.strata) == (
        "lowest_25_percent",
        "middle_50_percent",
        "highest_25_percent",
    )


def test_reliability_rejects_oracle_as_deployable_method() -> None:
    predictions = np.ones((8, 3))
    with pytest.raises(ValueError, match="oracle"):
        audit_reliability(
            np.ones(8),
            predictions,
            deployable_predictions={"oracle": np.ones(8)},
        )


def test_constant_reliability_signal_is_nonsignificant() -> None:
    result = audit_reliability(
        np.arange(8, dtype=np.float64),
        np.ones((8, 3)),
        deployable_predictions={"constant": np.ones(8)},
    )

    record = result.methods[0]
    assert record.pearson_r == 0.0
    assert record.spearman_r == 0.0
    assert record.pearson_p_value == 1.0
    assert record.spearman_p_value == 1.0
