from __future__ import annotations

from pathlib import Path

from cmc_bbdm.inspection_agent.g0 import (
    _load_historical_mavis_trajectories,
    _trajectory_frame,
    load_g0_protocol,
    plan_staged_actions,
)
from cmc_bbdm.inspection_agent.state import apply_action, budget_record, zero_state
from cmc_bbdm.inspection_agent.state_bank import (
    StateBankPolicy,
    plan_policy_actions,
)
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.oracle import uniform_cell_order

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/inspection_agent_g0.yaml"


def _hypothesis() -> SurfaceHypothesis:
    import numpy as np

    scores = np.linspace(0.0, 1.0, 64)
    scores.setflags(write=False)
    median = np.zeros(3)
    median.setflags(write=False)
    return SurfaceHypothesis(scores, tuple(range(63, 55, -1)), median, "a" * 64)


def test_frozen_g0_protocol_loads_exact_registered_contract() -> None:
    protocol = load_g0_protocol(CONFIG, project_root=ROOT)
    assert protocol.specimen_count == 276
    assert len(protocol.domain_order) == 6
    assert protocol.endpoint_budget == 0.25
    assert protocol.checkpoints[0] == 0.0
    assert protocol.checkpoints[-1] == 0.25
    assert protocol.bootstrap_replicates == 100_000
    assert protocol.default_device == "cuda:0"


def test_fixed_staged_plan_matches_registered_uniform_state_bank_policy() -> None:
    grid = build_acquisition_grid(73, 79, initial_budget=0.015625)
    actions = plan_staged_actions(
        grid,
        uniform_cell_order(),
        endpoint_budget=0.25,
    )
    registered = plan_policy_actions(
        grid,
        StateBankPolicy.UNIFORM_THEN_REFINE,
        surface_hypothesis=_hypothesis(),
        surface_sha256="b" * 64,
        random_seed=2026083101,
        endpoint_budget=0.25,
    )
    assert actions == registered
    state = zero_state(grid)
    for action in actions:
        state = apply_action(grid, state, action)
        assert budget_record(grid, state).effective_budget <= 0.25 + 1.0e-15
    assert actions[:64] == tuple(
        type(actions[0])(cell, -1, 0) for cell in uniform_cell_order()
    )


def test_frozen_mavis_upper_bound_loads_all_authorized_action_histories() -> None:
    protocol = load_g0_protocol(CONFIG, project_root=ROOT)

    trajectories = _load_historical_mavis_trajectories(protocol, ROOT)

    assert len(trajectories) == 276
    assert {
        domain: sum(key[0] == domain for key in trajectories)
        for domain in protocol.domain_order
    } == dict(protocol.domain_counts)
    assert all(value.actions for value in trajectories.values())
    assert all(
        action.from_level in (0, 1) and action.to_level == action.from_level + 1
        for value in trajectories.values()
        for action in value.actions
    )
    assert all(len(value.source_state_sha256_before) == len(value.actions) for value in trajectories.values())
    assert all(len(value.state_sha256) == 64 for value in trajectories.values())


def test_trajectory_frame_infers_hash_columns_beyond_first_100_rows() -> None:
    rows = [
        {
            "task": "FIELD",
            "dataset_id": "source",
            "specimen_id": f"specimen-{index:03d}",
            "method": "FIXED",
            "step": 0,
            "trajectory_sha256": None,
        }
        for index in range(101)
    ]
    rows.append(
        {
            "task": "CAI",
            "dataset_id": "target",
            "specimen_id": "specimen-101",
            "method": "ORACLE",
            "step": 1,
            "trajectory_sha256": "a" * 64,
        }
    )

    frame = _trajectory_frame(tuple(rows))

    assert frame.filter(frame["task"] == "CAI")["trajectory_sha256"].item() == "a" * 64
