from __future__ import annotations

import polars as pl
import pytest

from cmc_bbdm.agentic_nde.p1_cai import aggregate_p1_evaluation

DOMAINS = ("a", "b", "c", "d", "e", "f")
CHECKPOINTS = (0.0625, 0.09375, 0.125, 0.1875, 0.25)
METHOD_VALUES = {
    "c0_mvd_m1_o2": 1.0,
    "proposed": 0.6,
    "c2_global_context": 0.82,
    "c3_shuffled_surface": 0.9,
    "c4_wrong_orientation": 0.88,
    "c5_spatial_derangement": 0.86,
    "c3_shuffled_global": 0.94,
    "mechanical_oracle_diagnostic": 0.2,
}


def test_p1_cai_aggregation_is_normalized_specimen_first_and_gated() -> None:
    curves = pl.DataFrame(
        [
            {
                "outer_domain": domain,
                "dataset_id": domain,
                "specimen_id": f"{domain}-{index}",
                "method": method,
                "nominal_checkpoint": checkpoint,
                "effective_budget": checkpoint - 0.001,
                "p_a_absolute_error": value + 0.1,
                "p_b_absolute_error": value + 0.001 * index,
            }
            for domain in DOMAINS
            for index in range(2)
            for method, value in METHOD_VALUES.items()
            for checkpoint in CHECKPOINTS
        ]
    )
    ranking = pl.DataFrame(
        [
            {
                "outer_domain": domain,
                "dataset_id": domain,
                "specimen_id": f"{domain}-{index}",
                "method": method,
                "ndcg_10": 0.7 if method == "proposed" else 0.5,
                "next_action_regret": 0.3 if method == "proposed" else 0.5,
                "one_step_cai_utility": 0.4,
                "spearman": 0.2,
                "recall_5": 0.4,
                "top_10_percent_overlap": 0.5,
                "top_1_oracle_match": 0.1,
                "model_state_sha256": "a" * 64,
            }
            for domain in DOMAINS
            for index in range(2)
            for method in METHOD_VALUES
        ]
    )
    result = aggregate_p1_evaluation(
        curves,
        ranking,
        domain_order=DOMAINS,
        checkpoints=CHECKPOINTS,
        bootstrap_seed=20260831,
        bootstrap_resamples=1000,
    )
    proposed = result.per_specimen_metrics.filter(
        pl.col("method") == "proposed"
    )
    assert set(proposed["cai_auebc"]) == {0.6, 0.601}
    assert result.decision.status == "P1_SPATIAL_VISUAL_OBSERVABILITY_GO"
    assert result.decision.oracle_gap_closure == pytest.approx(0.5)
    assert result.domain_metrics.height == len(DOMAINS) * len(METHOD_VALUES)
    assert result.bootstrap.height == 7
    assert result.control_results.height == len(METHOD_VALUES)
    assert len(result.state_sha256) == 64
