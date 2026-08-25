from __future__ import annotations

import polars as pl

from cmc_bbdm.mavis.mris_metrics import (
    bootstrap_mris_contrasts,
    evaluate_mris_predictions,
)


def _predictions() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    roster = (("d0", "a", 0.0), ("d1", "b", 1.0), ("d1", "c", 1.0), ("d1", "d", 1.0))
    for domain, specimen, static_error in roster:
        for checkpoint in (0.1, 0.2):
            for mode, error in (("real", 0.0), ("static", static_error)):
                rows.append(
                    {
                        "outer_domain": domain,
                        "state_id": f"{domain}-{specimen}-{checkpoint}",
                        "specimen_id": specimen,
                        "trajectory_id": f"t-{specimen}",
                        "method": "uniform",
                        "nominal_checkpoint": checkpoint,
                        "exact_acquired_cost": int(checkpoint * 100),
                        "native_count": 100,
                        "effective_budget": checkpoint,
                        "mode": mode,
                        "target": 0.5,
                        "prediction": 0.5 + error,
                        "absolute_error": error,
                        "model_state_sha256": "a" * 64,
                    }
                )
    return pl.DataFrame(rows)


def test_mavis_metrics_use_specimen_then_equal_domain_aggregation() -> None:
    metrics = evaluate_mris_predictions(_predictions(), domain_order=("d0", "d1"))

    static = metrics.aggregate_metrics.filter(pl.col("mode") == "static")

    assert metrics.per_specimen_metrics.height == 16
    assert metrics.domain_metrics.filter(pl.col("mode") == "static").height == 4
    assert static.get_column("domain_balanced_mae").to_list() == [0.5, 0.5]


def test_mavis_bootstrap_is_deterministic_and_specimen_resampled() -> None:
    metrics = evaluate_mris_predictions(_predictions(), domain_order=("d0", "d1"))

    first = bootstrap_mris_contrasts(
        metrics.per_specimen_metrics,
        reference_mode="real",
        control_modes=("static",),
        domain_order=("d0", "d1"),
        replicates=50,
        seed=20260825,
    )
    second = bootstrap_mris_contrasts(
        metrics.per_specimen_metrics,
        reference_mode="real",
        control_modes=("static",),
        domain_order=("d0", "d1"),
        replicates=50,
        seed=20260825,
    )

    assert first.equals(second)
    assert first.height == 50
    assert first.get_column("control_minus_reference_auebc").min() == 0.5
