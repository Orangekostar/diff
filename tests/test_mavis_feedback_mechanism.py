from __future__ import annotations

import polars as pl

from cmc_bbdm.mavis.feedback_mechanism import (
    align_feedback_effect,
    assign_outcome_blind_tertiles,
)


def test_feedback_benefit_sign_is_no_feedback_minus_feedback() -> None:
    curves = pl.DataFrame(
        {
            "outer_domain": ["d0", "d0"],
            "specimen_id": ["s0", "s0"],
            "method": ["mavis_full", "mavis_no_feedback"],
            "nominal_checkpoint": [0.125, 0.125],
            "mean_exact_acquired_cost": [100.0, 100.0],
            "mean_effective_budget": [0.125, 0.125],
            "cai_mae": [0.2, 0.3],
            "reconstruction_mse": [0.01, 0.01],
            "trajectory_count": [5, 5],
        }
    )

    effect = align_feedback_effect(curves)

    assert effect.get_column("feedback_benefit").item() == 0.1


def test_feedback_strata_do_not_use_outcomes() -> None:
    frame = pl.DataFrame(
        {
            "outer_domain": ["d0"] * 6,
            "specimen_id": [f"s{index}" for index in range(6)],
            "mechanism_metric": [0.4, 0.1, 0.6, 0.2, 0.5, 0.3],
            "feedback_benefit": [10.0, -2.0, 4.0, 1.0, -8.0, 3.0],
        }
    )
    changed_outcome = frame.with_columns(
        (pl.col("feedback_benefit") * -100.0).alias("feedback_benefit")
    )

    first, _ = assign_outcome_blind_tertiles(frame, metric="mechanism_metric")
    second, _ = assign_outcome_blind_tertiles(
        changed_outcome, metric="mechanism_metric"
    )

    assert first.select("specimen_id", "stratum").equals(
        second.select("specimen_id", "stratum")
    )
