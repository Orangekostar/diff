from __future__ import annotations

import polars as pl

from cmc_bbdm.mavis.final_execution import (
    assign_claim_tier,
    build_risk_coverage,
    compose_final_predictions,
    select_safe_curve_rows,
)


def _curve(method: str, value: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "outer_domain": ["d0", "d0"],
            "specimen_id": ["s1", "s1"],
            "method": [method, method],
            "nominal_checkpoint": [0.1, 0.2],
            "absolute_error": [value, value],
        }
    )


def test_final_safe_curve_routes_whole_specimen_from_first_decision_confidence() -> None:
    mavis = _curve("mavis_full", 0.1)
    baseline = _curve("uniform", 0.4)

    selected, audit = select_safe_curve_rows(
        mavis,
        baseline,
        confidence=0.2,
        threshold=0.5,
        baseline="uniform",
    )

    assert set(selected.get_column("method")) == {"mavis_safe"}
    assert selected.get_column("absolute_error").to_list() == [0.4, 0.4]
    assert audit == {
        "confidence": 0.2,
        "threshold": 0.5,
        "baseline": "uniform",
        "used_fallback": True,
        "selected_method": "uniform",
    }


def test_final_safe_curve_uses_mavis_at_registered_threshold() -> None:
    selected, audit = select_safe_curve_rows(
        _curve("mavis_full", 0.1),
        _curve("reconstruction_driven", 0.4),
        confidence=0.5,
        threshold=0.5,
        baseline="reconstruction",
    )

    assert selected.get_column("absolute_error").to_list() == [0.1, 0.1]
    assert audit["used_fallback"] is False
    assert audit["selected_method"] == "mavis_full"


def test_final_claim_tier_rules_are_frozen_before_outer_evaluation() -> None:
    common = {
        "baseline_cai_auebc": 0.30,
        "safe_control_cai_auebc": 0.35,
        "sequential_oracle_cai_auebc": 0.20,
        "domain_count": 6,
        "high_confidence_specimen_count": 12,
    }
    assert assign_claim_tier(
        **common,
        mavis_cai_auebc=0.28,
        safe_cai_auebc=0.27,
        mavis_improved_domain_count=4,
        mavis_bootstrap_ci_lower=0.001,
        safe_bootstrap_ci_lower=0.001,
        high_confidence_control_minus_mavis_auebc=0.01,
        high_confidence_bootstrap_ci_lower=0.001,
    ) == "S"
    assert assign_claim_tier(
        **common,
        mavis_cai_auebc=0.31,
        safe_cai_auebc=0.29,
        mavis_improved_domain_count=3,
        mavis_bootstrap_ci_lower=-0.01,
        safe_bootstrap_ci_lower=0.0,
        high_confidence_control_minus_mavis_auebc=0.01,
        high_confidence_bootstrap_ci_lower=0.001,
    ) == "A"
    assert assign_claim_tier(
        **common,
        mavis_cai_auebc=0.31,
        safe_cai_auebc=0.31,
        mavis_improved_domain_count=2,
        mavis_bootstrap_ci_lower=-0.01,
        safe_bootstrap_ci_lower=-0.01,
        high_confidence_control_minus_mavis_auebc=-0.01,
        high_confidence_bootstrap_ci_lower=-0.01,
    ) == "B"


def test_final_predictions_distinguish_aggregation_and_safe_routing() -> None:
    p4 = pl.concat(
        [
            _curve("uniform", 0.4),
            _curve("reconstruction_driven", 0.3),
            _curve("mavis_full", 0.2),
        ]
    )
    aggregated = _curve("mavis_full", 0.1)
    routing = pl.DataFrame(
        {
            "outer_domain": ["d0"],
            "specimen_id": ["s1"],
            "confidence": [0.2],
            "threshold": [0.5],
            "baseline": ["uniform"],
        }
    )

    predictions, audit = compose_final_predictions(p4, aggregated, routing)

    assert set(predictions.get_column("method")) == {
        "uniform",
        "reconstruction_driven",
        "mavis_no_aggregation",
        "mavis_full",
        "mavis_safe",
        "source_selected_fallback",
    }
    assert predictions.filter(pl.col("method") == "mavis_safe").get_column(
        "absolute_error"
    ).to_list() == [0.4, 0.4]
    assert audit.select("used_fallback", "selected_method").row(0) == (
        True,
        "uniform",
    )
    assert predictions.filter(
        pl.col("method") == "source_selected_fallback"
    ).get_column("absolute_error").to_list() == [0.4, 0.4]


def test_final_risk_coverage_is_specimen_paired_and_domain_balanced() -> None:
    specimen = pl.DataFrame(
        {
            "outer_domain": ["d0", "d0", "d0", "d0", "d1", "d1", "d1", "d1"],
            "specimen_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "method": ["mavis_full", "source_selected_fallback"] * 4,
            "cai_auebc": [0.1, 0.4, 0.6, 0.4, 0.2, 0.5, 0.7, 0.5],
            "reconstruction_auebc": [0.0] * 8,
        }
    )
    routing = pl.DataFrame(
        {
            "outer_domain": ["d0", "d0", "d1", "d1"],
            "specimen_id": ["s1", "s2", "s3", "s4"],
            "confidence": [0.9, 0.1, 0.8, 0.2],
            "threshold": [0.5] * 4,
            "baseline": ["uniform"] * 4,
        }
    )

    result = build_risk_coverage(
        specimen,
        routing,
        thresholds=(0.0, 0.5, 1.0),
        domain_order=("d0", "d1"),
    )

    middle = result.filter(pl.col("threshold") == 0.5).row(0, named=True)
    assert middle["coverage"] == 0.5
    assert middle["fallback_frequency"] == 0.5
    assert middle["domain_balanced_cai_auebc"] == 0.3
    assert middle["source_selected_fallback_cai_auebc"] == 0.45
    assert middle["improved_domain_count"] == 2
