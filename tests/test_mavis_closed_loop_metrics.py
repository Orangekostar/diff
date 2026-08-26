from __future__ import annotations

import polars as pl
import pytest

from cmc_bbdm.mavis.closed_loop_metrics import (
    MAVISClosedLoopMetricError,
    bootstrap_closed_loop_contrasts,
    evaluate_closed_loop_predictions,
)


def _predictions() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, specimens in (("d0", ("a", "b")), ("d1", ("c",))):
        for specimen in specimens:
            for method, offset in (("uniform", 0.2), ("mavis_full", 0.1)):
                for checkpoint in (0.1, 0.2):
                    target = 0.5
                    error = offset + checkpoint
                    rows.append(
                        {
                            "outer_domain": domain,
                            "specimen_id": specimen,
                            "method": method,
                            "nominal_checkpoint": checkpoint,
                            "initial_budget": 0.1,
                            "action_count": 0 if checkpoint == 0.1 else 1,
                            "exact_acquired_cost": int(checkpoint * 100),
                            "native_count": 100,
                            "effective_budget": checkpoint,
                            "target": target,
                            "prediction": target + error,
                            "absolute_error": error,
                            "reconstruction_mse": error * 2.0,
                        }
                    )
    return pl.DataFrame(rows)


def test_closed_loop_metrics_are_specimen_first_and_equal_domain() -> None:
    metrics = evaluate_closed_loop_predictions(
        _predictions(),
        domain_order=("d0", "d1"),
        method_order=("uniform", "mavis_full"),
        checkpoints=(0.1, 0.2),
    )

    assert metrics.per_specimen_curve.height == 12

    assert metrics.specimen_auebc.height == 6
    assert metrics.domain_auebc.height == 4
    assert metrics.aggregate_auebc.height == 2
    mavis = metrics.aggregate_auebc.filter(pl.col("method") == "mavis_full").row(
        0, named=True
    )
    uniform = metrics.aggregate_auebc.filter(pl.col("method") == "uniform").row(
        0, named=True
    )
    assert mavis["domain_balanced_cai_auebc"] < uniform["domain_balanced_cai_auebc"]
    assert mavis["statistical_unit"] == "equal_domain"


def test_closed_loop_metrics_accept_quantized_initial_scout_only() -> None:
    predictions = _predictions().with_columns(
        pl.lit(0.1).alias("initial_budget"),
        pl.when(pl.col("nominal_checkpoint") == 0.1)
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("action_count"),
        pl.lit(1000).alias("native_count"),
        pl.when(pl.col("nominal_checkpoint") == 0.1)
        .then(pl.lit(101))
        .otherwise(pl.lit(200))
        .alias("exact_acquired_cost"),
        pl.when(pl.col("nominal_checkpoint") == 0.1)
        .then(pl.lit(0.101))
        .otherwise(pl.lit(0.2))
        .alias("effective_budget"),
    )

    metrics = evaluate_closed_loop_predictions(
        predictions,
        domain_order=("d0", "d1"),
        method_order=("uniform", "mavis_full"),
        checkpoints=(0.1, 0.2),
    )

    assert metrics.per_specimen_curve.height == 12

    invalid = predictions.with_columns(
        pl.when(pl.col("nominal_checkpoint") == 0.1)
        .then(pl.lit(1))
        .otherwise(pl.col("action_count"))
        .alias("action_count")
    )
    with pytest.raises(MAVISClosedLoopMetricError):
        evaluate_closed_loop_predictions(
            invalid,
            domain_order=("d0", "d1"),
            method_order=("uniform", "mavis_full"),
            checkpoints=(0.1, 0.2),
        )


def test_closed_loop_bootstrap_is_paired_within_domain_and_deterministic() -> None:
    metrics = evaluate_closed_loop_predictions(
        _predictions(),
        domain_order=("d0", "d1"),
        method_order=("uniform", "mavis_full"),
        checkpoints=(0.1, 0.2),
    )

    first = bootstrap_closed_loop_contrasts(
        metrics.specimen_auebc,
        reference_method="mavis_full",
        control_methods=("uniform",),
        domain_order=("d0", "d1"),
        replicates=20,
        seed=17,
    )
    second = bootstrap_closed_loop_contrasts(
        metrics.specimen_auebc,
        reference_method="mavis_full",
        control_methods=("uniform",),
        domain_order=("d0", "d1"),
        replicates=20,
        seed=17,
    )

    assert first.equals(second)
    assert first.height == 20
    assert first.get_column("improved_domain_count").unique().to_list() == [2]
    assert (
        first.get_column("control_minus_reference_cai_auebc") > 0.0
    ).all()
    assert (
        first.get_column("control_minus_reference_reconstruction_auebc") > 0.0
    ).all()
