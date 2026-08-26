from __future__ import annotations

from pathlib import Path

import numpy as np

from cmc_bbdm.mavis.historical_sources import HistoricalPolicySource

ROOT = Path(__file__).resolve().parents[1]


def _source() -> HistoricalPolicySource:
    return HistoricalPolicySource(
        a4_path=ROOT / "results/mva/a4_global_task_mask/fixed_trajectories.parquet",
        a4_sha256="12293c89c8c066d7db1febc4f1e7e573744b3522a832968f91e209090028e257",
        a5_path=ROOT / "results/mva/a5_imitation_policy/target_trajectories.parquet",
        a5_sha256="58f9b72723a5973383e1c5b16dae0f662b4b560392348345e2a07c827e80e3b8",
        mvd_m1_path=ROOT / "results/mvd/m1_observability/observability_predictions.parquet",
        mvd_m1_sha256="3164e5e6e6a293cdd1376fd4d58438c74d23e791a1c1378deff14c254d9903b3",
        checkpoints=(0.03125, 0.0625, 0.09375, 0.125, 0.1875, 0.25),
    )


def test_historical_source_issues_actions_and_predicted_o2_scores_only() -> None:
    source = _source()
    plans = source.action_plans(
        specimen_id="c8-10t",
        dataset_id="74t7kcdgkr",
        outer_domain="74t7kcdgkr",
    )
    scores = source.o2_scores(
        specimen_id="c8-10t",
        dataset_id="74t7kcdgkr",
        outer_domain="74t7kcdgkr",
    )

    assert set(plans) == {"global_mechanical", "mva_a5"}
    assert len(plans["global_mechanical"]) == 63
    assert len(plans["mva_a5"]) == 29
    assert scores.shape == (64,)
    assert np.all(np.isfinite(scores))
    assert not hasattr(source, "teacher_values")
