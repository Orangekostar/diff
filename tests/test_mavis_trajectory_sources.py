from __future__ import annotations

from pathlib import Path

from cmc_bbdm.mavis.config import load_mavis_config
from cmc_bbdm.mavis.trajectory_sources import load_frozen_action_plans

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mavis_development.yaml"


def test_mavis_frozen_sources_recover_five_registered_trajectory_plans() -> None:
    config = load_mavis_config(CONFIG, project_root=ROOT)

    plans = load_frozen_action_plans(
        config,
        project_root=ROOT,
        specimen_id="c8-10t",
        dataset_id="74t7kcdgkr",
    )

    assert set(plans) == {
        "random",
        "uniform",
        "reconstruction_driven",
        "one_shot_mechanical_oracle",
        "sequential_mechanical_oracle",
    }
    assert {method: len(actions) for method, actions in plans.items()} == {
        "random": 43,
        "uniform": 63,
        "reconstruction_driven": 28,
        "one_shot_mechanical_oracle": 63,
        "sequential_mechanical_oracle": 41,
    }
    for actions in plans.values():
        assert tuple(item.nominal_checkpoint for item in actions) == tuple(
            sorted(item.nominal_checkpoint for item in actions)
        )
        assert all(
            item.nominal_checkpoint in config.checkpoints for item in actions
        )


def test_mavis_frozen_sources_are_byte_bound_and_specimen_exact() -> None:
    config = load_mavis_config(CONFIG, project_root=ROOT)

    plans = load_frozen_action_plans(
        config,
        project_root=ROOT,
        specimen_id="q24-1",
        dataset_id="cgtnjyggtm",
    )

    assert config.sources["a2_oracle_trajectories"].sha256 == (
        "4906d188b8a369d0e233d7cf79d98b694342eaf9923d7ec702d611a525123d16"
    )
    assert config.sources["mvd_m0_actions"].sha256 == (
        "a7c40acff28fb307de385423379451d4aaf4f0f936a54c74fee7d62c67fb631d"
    )
    assert plans["random"][0].action.cell_index >= 0
    assert plans["random"][0].nominal_checkpoint == 0.0625
