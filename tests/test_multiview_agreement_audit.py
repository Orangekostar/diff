from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.aei_multiview_regression.agreement_audit import audit_predictions

Y = np.asarray([0.0, 1.0, 2.0, 3.0])
PREDICTIONS = np.asarray(
    [
        [0.1, 0.2, 0.3],
        [1.4, 1.1, 1.2],
        [1.8, 1.7, 2.1],
        [2.9, 3.3, 3.2],
    ]
)
DOMAINS = ("d1", "d1", "d2", "d2")
VIEWS = ("FULL", "BILINEAR_50", "BILINEAR_25")


def test_oracle_and_best_view_are_diagnostic() -> None:
    audit = audit_predictions(Y, PREDICTIONS, DOMAINS, view_names=VIEWS)

    assert audit.oracle_mae == pytest.approx(0.1)
    assert audit.best_view_counts == (2, 1, 1)
    assert audit.best_view_indices.tolist() == [0, 1, 2, 0]
    assert "oracle" not in audit.deployable_methods
    assert audit.oracle_improvement_vs_full == pytest.approx(0.5)
    assert audit.prediction_correlations.shape == (3, 3)
    assert audit.residual_correlations.shape == (3, 3)
    assert audit.mean_absolute_disagreement[0, 1] == pytest.approx(0.225)


def test_individual_metrics_use_equal_domain_aggregation() -> None:
    audit = audit_predictions(Y, PREDICTIONS, DOMAINS, view_names=VIEWS)
    full = audit.view_metrics[0]

    assert tuple(item[0] for item in full.domain_mae) == ("d1", "d2")
    assert tuple(item[1] for item in full.domain_mae) == pytest.approx((0.25, 0.15))
    assert full.equal_domain_mae == pytest.approx(0.2)
    assert full.worst_domain_mae == pytest.approx(0.25)
    assert full.domain_mae_sd == pytest.approx(0.05)
    assert full.rmse == pytest.approx(np.sqrt(0.055))


def test_grouped_best_view_frequency_preserves_registered_order() -> None:
    audit = audit_predictions(
        Y,
        PREDICTIONS,
        DOMAINS,
        view_names=VIEWS,
        groups={"ply_count": (8, 8, 24, 24)},
    )

    rows = [item for item in audit.grouped_best_view if item.group_name == "ply_count"]
    assert [(row.group_value, row.counts) for row in rows] == [
        ("8", (1, 1, 0)),
        ("24", (1, 0, 1)),
    ]
