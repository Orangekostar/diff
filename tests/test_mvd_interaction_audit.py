from __future__ import annotations

import polars as pl
import pytest

from cmc_bbdm.mvd.interaction_audit import summarize_interactions


def test_interaction_summary_reports_signed_bias_and_mae() -> None:
    table = pl.DataFrame(
        {
            "method": ["m"] * 4,
            "nominal_checkpoint": [0.0625] * 4,
            "additive_gain": [1.0, 2.0, 3.0, 4.0],
            "joint_gain": [0.5, 1.5, 2.5, 3.5],
        }
    )

    summary = summarize_interactions(table)

    assert summary.height == 1
    assert summary["pearson"][0] == pytest.approx(1.0)
    assert summary["spearman"][0] == pytest.approx(1.0)
    assert summary["signed_bias_additive_minus_joint"][0] == pytest.approx(0.5)
    assert summary["mean_absolute_error"][0] == pytest.approx(0.5)
