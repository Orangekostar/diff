from __future__ import annotations

import polars as pl

from cmc_bbdm.mavis.dynamic_voi import CandidateDescriptor
from cmc_bbdm.mavis.fallback import (
    apply_safe_action,
    select_source_safe_policy,
)
from cmc_bbdm.mavis.policy import PolicySelection
from cmc_bbdm.mavis.safety_execution import build_source_safe_metrics


def _source_table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "domain_id": ["d1", "d1", "d2", "d2"],
            "specimen_id": ["s1", "s2", "s3", "s4"],
            "confidence": [0.9, 0.1, 0.8, 0.2],
            "mavis_auebc": [0.1, 0.5, 0.2, 0.6],
            "uniform_auebc": [0.4, 0.4, 0.5, 0.5],
            "reconstruction_auebc": [0.45, 0.45, 0.55, 0.55],
        }
    )


def test_safe_policy_selection_is_source_only() -> None:
    selected = select_source_safe_policy(
        _source_table(),
        outer_domain="d0",
        thresholds=(0.0, 0.5, 1.0),
    )

    assert selected.baseline == "uniform"
    assert selected.threshold == 0.5
    assert selected.source_domains == ("d1", "d2")
    assert selected.target_outcomes_used is False
    assert selected.audit.height == 3


def test_low_confidence_safe_action_exactly_uses_fallback() -> None:
    selected = select_source_safe_policy(
        _source_table(),
        outer_domain="d0",
        thresholds=(0.0, 0.5, 1.0),
    )
    candidates = (
        CandidateDescriptor(1, 0, 1, 5, 100, 20),
        CandidateDescriptor(2, 0, 1, 5, 100, 20),
    )
    mavis = PolicySelection(0, candidates[0], 0.8, 0.8, "raw_score")
    fallback = PolicySelection(1, candidates[1], 0.0, 0.0, "raw_score")

    action = apply_safe_action(
        mavis,
        fallback,
        confidence=0.2,
        safe_policy=selected,
    )

    assert action.selection == fallback
    assert action.used_fallback is True


def test_safe_metrics_pair_nested_source_curves_by_physical_specimen() -> None:
    specimen_auebc = pl.DataFrame(
        {
            "outer_domain": ["d1"] * 3 + ["d2"] * 3,
            "specimen_id": ["s1"] * 3 + ["s2"] * 3,
            "method": [
                "mavis",
                "uniform",
                "reconstruction_driven",
            ]
            * 2,
            "cai_auebc": [0.1, 0.4, 0.5, 0.2, 0.5, 0.6],
        }
    )
    confidence = pl.DataFrame(
        {
            "domain_id": ["d1", "d2"],
            "specimen_id": ["s1", "s2"],
            "confidence": [0.9, 0.8],
        }
    )

    metrics = build_source_safe_metrics(
        specimen_auebc,
        confidence,
        outer_domain="d0",
    )

    assert metrics.to_dicts() == _source_table().head(0).vstack(
        pl.DataFrame(
            {
                "domain_id": ["d1", "d2"],
                "specimen_id": ["s1", "s2"],
                "confidence": [0.9, 0.8],
                "mavis_auebc": [0.1, 0.2],
                "uniform_auebc": [0.4, 0.5],
                "reconstruction_auebc": [0.5, 0.6],
            }
        )
    ).to_dicts()
