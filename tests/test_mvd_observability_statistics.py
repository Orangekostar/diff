from __future__ import annotations

import polars as pl

from cmc_bbdm.mvd.observability_statistics import aggregate_observability_metrics


def test_observability_gate_requires_all_frozen_domain_effects() -> None:
    rows = []
    for domain_index in range(6):
        for method, spearman, ndcg, regret in (
            ("o2_global_candidate", 0.4, 0.8, 0.1),
            ("global_mechanical", 0.1, 0.5, 0.4),
            ("random_median", 0.0, 0.3, 0.7),
        ):
            rows.append(
                {
                    "outer_domain": f"d{domain_index}",
                    "specimen_id": f"s{domain_index}",
                    "method": method,
                    "spearman": spearman,
                    "ndcg_5": ndcg,
                    "ndcg_10": ndcg,
                    "recall_5": ndcg,
                    "recall_10": ndcg,
                    "regret_1": regret,
                    "mean_budgeted_regret": regret,
                }
            )

    result = aggregate_observability_metrics(
        pl.DataFrame(rows),
        domain_order=tuple(f"d{index}" for index in range(6)),
        bootstrap_seed=20260824,
        bootstrap_resamples=1000,
        minimum_improved_domains=4,
    )

    assert result.gate.go
    assert result.gate.status == "MVD_OBSERVABILITY_GO"
    assert all(effect.lower > 0.0 for effect in result.bootstrap_effects)
