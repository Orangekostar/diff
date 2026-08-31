from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.state_bank import (
    StateBankPolicy,
    materialize_state_bank,
    plan_policy_actions,
)
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid


def _authority(*, invert: bool, true_cai: float) -> MAVISAuthority:
    rows, columns = np.indices((41, 43))
    image = np.stack((rows * 3, columns * 4, rows + columns), axis=2).astype(np.uint8)
    if invert:
        image = 255 - image
    return MAVISAuthority.from_arrays(
        specimen_ids=("sample",),
        dataset_ids=("domain",),
        images=(image,),
        targets=np.asarray([true_cai]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )


def _hypothesis() -> SurfaceHypothesis:
    scores = np.linspace(0.0, 1.0, 64)
    scores.setflags(write=False)
    median = np.zeros(3)
    median.setflags(write=False)
    return SurfaceHypothesis(
        scores=scores,
        top_cells=tuple(range(63, 55, -1)),
        border_median_rgb=median,
        state_sha256="b" * 64,
    )


def _world(authority: MAVISAuthority) -> tuple[CausalInspectionWorld, object]:
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    surface = np.full((17, 19, 3), 80, dtype=np.uint8)
    world = CausalInspectionWorld(
        authority,
        specimen_id="sample",
        task=InspectionTask.CAI,
        surface_rgb=surface,
        surface_sha256=hashlib.sha256(surface.tobytes()).hexdigest(),
        grid=grid,
        endpoint_budget=0.25,
    )
    return world, grid


def test_state_bank_has_one_zero_anchor_and_three_states_per_policy() -> None:
    world, grid = _world(_authority(invert=False, true_cai=0.1))
    bank = materialize_state_bank(
        world,
        grid,
        _hypothesis(),
        random_seed=2026083101,
        snapshot_fractions=(1 / 3, 2 / 3, 1.0),
    )
    assert len(bank) == 19
    assert bank[0].policy == "ZERO_ANCHOR"
    assert bank[0].observation.exact_acquired_count == 0
    for policy in StateBankPolicy:
        selected = [row for row in bank if row.policy == policy.value]
        assert len(selected) == 3
        assert [row.snapshot_index for row in selected] == [0, 1, 2]
        assert [row.observation.exact_acquired_count for row in selected] == sorted(
            row.observation.exact_acquired_count for row in selected
        )
        assert selected[-1].observation.effective_budget <= 0.25


def test_state_bank_actions_do_not_depend_on_cscan_values_or_true_cai() -> None:
    first_world, grid = _world(_authority(invert=False, true_cai=0.1))
    second_world, _ = _world(_authority(invert=True, true_cai=99.0))
    kwargs = {
        "random_seed": 2026083101,
        "snapshot_fractions": (1 / 3, 2 / 3, 1.0),
    }
    first = materialize_state_bank(first_world, grid, _hypothesis(), **kwargs)
    second = materialize_state_bank(second_world, grid, _hypothesis(), **kwargs)
    assert [(row.policy, row.observation.action_history) for row in first] == [
        (row.policy, row.observation.action_history) for row in second
    ]


def test_staged_and_alternating_policy_semantics_are_distinct_and_fixed() -> None:
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    common = {
        "surface_hypothesis": _hypothesis(),
        "surface_sha256": "c" * 64,
        "random_seed": 2026083101,
        "endpoint_budget": 0.25,
    }
    staged = plan_policy_actions(
        grid, StateBankPolicy.UNIFORM_THEN_REFINE, **common
    )
    alternating = plan_policy_actions(
        grid, StateBankPolicy.ALTERNATE_BROADEN_REFINE, **common
    )
    first_refine = next(index for index, action in enumerate(staged) if action.from_level == 0)
    assert first_refine == 64
    assert alternating[0].from_level == -1
    assert alternating[1].from_level == 0
    assert staged != alternating
