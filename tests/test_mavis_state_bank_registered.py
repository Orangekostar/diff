from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cmc_bbdm.mavis.authority import load_mavis_authority
from cmc_bbdm.mavis.config import load_mavis_config
from cmc_bbdm.mavis.state_bank import materialize_action_plan
from cmc_bbdm.mavis.trajectory_sources import load_frozen_action_plans

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/home/ww/paper3/cmc_damage_inference")
CONFIG = ROOT / "paper_v3/configs/mavis_development.yaml"


@pytest.mark.slow
def test_mavis_registered_plans_reproduce_all_five_checkpoint_cost_curves() -> None:
    config = load_mavis_config(CONFIG, project_root=ROOT)
    authority = load_mavis_authority(config, source_project_root=SOURCE_ROOT)
    specimen_id = "c8-10t"
    dataset_id = "74t7kcdgkr"
    plans = load_frozen_action_plans(
        config,
        project_root=ROOT,
        specimen_id=specimen_id,
        dataset_id=dataset_id,
    )
    trajectories = {
        method: materialize_action_plan(
            authority,
            specimen_id=specimen_id,
            method=method,
            seed=config.trajectory_random_seed if method == "random" else None,
            initial_budget=config.initial_budget_by_domain[dataset_id],
            checkpoints=config.checkpoints,
            actions=actions,
        )
        for method, actions in plans.items()
    }

    a2 = pl.read_parquet(
        ROOT / "results/mva/a2_oracle_value/state_metrics.parquet"
    )
    mvd = pl.read_parquet(
        ROOT / "results/mvd/m0_one_shot_oracle/state_metrics.parquet"
    )
    source_methods = {
        "random": "random",
        "uniform": "uniform",
        "reconstruction_driven": "reconstruction_oracle",
        "sequential_mechanical_oracle": "mechanical_oracle",
    }
    for method, source_method in source_methods.items():
        expected = a2.filter(
            (pl.col("specimen_id") == specimen_id)
            & (pl.col("dataset_id") == dataset_id)
            & (pl.col("method") == source_method)
            & (
                (pl.col("seed") == config.trajectory_random_seed)
                if method == "random"
                else pl.col("seed").is_null()
            )
        ).sort("nominal_checkpoint")
        assert expected.height == len(config.checkpoints)
        assert tuple(
            snapshot.inspection_state.exact_acquired_count
            for snapshot in trajectories[method].snapshots
        ) == tuple(expected.get_column("measured_count"))

    expected_one_shot = mvd.filter(
        (pl.col("specimen_id") == specimen_id)
        & (pl.col("dataset_id") == dataset_id)
        & (pl.col("method") == "one_shot_mechanical_oracle")
    ).sort("nominal_checkpoint")
    assert expected_one_shot.height == len(config.checkpoints)
    assert tuple(
        snapshot.inspection_state.exact_acquired_count
        for snapshot in trajectories["one_shot_mechanical_oracle"].snapshots
    ) == tuple(expected_one_shot.get_column("measured_count"))
