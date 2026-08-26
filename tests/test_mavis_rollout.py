from __future__ import annotations

import numpy as np
from mavis_test_support import synthetic_authority

from cmc_bbdm.mavis.dynamic_voi import ActionScoreBatch
from cmc_bbdm.mavis.reveal import reveal_uniform_scout
from cmc_bbdm.mavis.rollout import (
    _candidate_descriptors,
    rollout_scout_and_focus,
    rollout_scout_and_focus_curve,
)
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import candidate_budget_record


class _CountingScorer:
    def __init__(self) -> None:
        self.state_hashes: list[str] = []

    def score_actions(self, state, candidates):
        self.state_hashes.append(state.state_sha256)
        return np.asarray(
            [1.0 - candidate.cell_index / 100.0 for candidate in candidates],
            dtype=np.float64,
        )


class _DualScorer:
    def score_actions(self, _state, candidates):
        cells = np.asarray([candidate.cell_index for candidate in candidates])
        return ActionScoreBatch(
            scores=np.asarray(cells, dtype=np.float64),
            value_predictions=-np.asarray(cells, dtype=np.float64),
        )


def test_mavis_rollout_respects_budget_and_has_no_duplicate_refinement() -> None:
    authority = synthetic_authority()
    scorer = _CountingScorer()

    rollout = rollout_scout_and_focus(
        authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        endpoint_budget=0.0625,
        scorer=scorer,
        objective="raw_score",
        feedback=True,
    )

    actions = tuple(
        (step.action.cell_index, step.action.from_level, step.action.to_level)
        for step in rollout.steps
    )
    assert len(actions) == len(set(actions))
    assert rollout.final_state.effective_budget <= 0.0625 + 1.0e-15
    assert all(
        later.exact_acquired_count > earlier.exact_acquired_count
        for earlier, later in zip(rollout.states, rollout.states[1:])
    )
    assert len(scorer.state_hashes) == len(rollout.steps)
    assert scorer.state_hashes == [state.state_sha256 for state in rollout.states[:-1]]
    assert all(0.0 <= step.decision_confidence <= 1.0 for step in rollout.steps)
    assert all(np.isfinite(step.decision_confidence) for step in rollout.steps)


def test_mavis_no_feedback_freezes_post_scout_ranking() -> None:
    authority = synthetic_authority()
    scorer = _CountingScorer()

    rollout = rollout_scout_and_focus(
        authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        endpoint_budget=0.0625,
        scorer=scorer,
        objective="raw_score",
        feedback=False,
    )

    assert rollout.steps
    assert len(scorer.state_hashes) == 1
    assert all(step.feedback_used is False for step in rollout.steps)


def test_mavis_true_cai_cannot_change_rollout_actions() -> None:
    first_authority = synthetic_authority(true_cai=0.1)
    changed_authority = synthetic_authority(true_cai=1000.0)

    first = rollout_scout_and_focus(
        first_authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        endpoint_budget=0.0625,
        scorer=_CountingScorer(),
        objective="raw_score",
        feedback=True,
    )
    changed = rollout_scout_and_focus(
        changed_authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        endpoint_budget=0.0625,
        scorer=_CountingScorer(),
        objective="raw_score",
        feedback=True,
    )

    assert tuple(step.action for step in first.steps) == tuple(
        step.action for step in changed.steps
    )


def test_mavis_replay_is_deterministic() -> None:
    authority = synthetic_authority()

    first = rollout_scout_and_focus_curve(
        authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        checkpoints=(0.0625, 0.125),
        scorer=_CountingScorer(),
        objective="direct_cost_aware",
        feedback=True,
    )
    replay = rollout_scout_and_focus_curve(
        authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        checkpoints=(0.0625, 0.125),
        scorer=_CountingScorer(),
        objective="direct_cost_aware",
        feedback=True,
    )

    assert tuple(step.action for step in first.steps) == tuple(
        step.action for step in replay.steps
    )
    assert tuple(state.state_sha256 for state in first.checkpoint_states) == tuple(
        state.state_sha256 for state in replay.checkpoint_states
    )


def test_mavis_rollout_curve_respects_every_registered_checkpoint() -> None:
    authority = synthetic_authority()
    scorer = _CountingScorer()
    checkpoints = (0.0625, 0.125, 0.25)

    curve = rollout_scout_and_focus_curve(
        authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        checkpoints=checkpoints,
        scorer=scorer,
        objective="value_per_exact_cost",
        feedback=True,
    )

    assert curve.checkpoints == checkpoints
    assert len(curve.checkpoint_states) == len(checkpoints)
    assert all(
        state.effective_budget <= checkpoint + 1.0e-15
        for state, checkpoint in zip(
            curve.checkpoint_states,
            checkpoints,
            strict=True,
        )
    )
    assert all(
        later.exact_acquired_count >= earlier.exact_acquired_count
        for earlier, later in zip(
            curve.checkpoint_states,
            curve.checkpoint_states[1:],
        )
    )
    assert len(scorer.state_hashes) == len(curve.steps)
    assert len(curve.scoring_states) == len(curve.steps)
    assert scorer.state_hashes == [
        state.state_sha256 for state in curve.scoring_states
    ]


def test_mavis_cost_ablation_separates_raw_value_from_direct_score() -> None:
    authority = synthetic_authority()
    raw = rollout_scout_and_focus(
        authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        endpoint_budget=0.0625,
        scorer=_DualScorer(),
        objective="raw_score",
        feedback=True,
    )
    direct = rollout_scout_and_focus(
        authority,
        specimen_id="sample-001",
        initial_budget=0.015625,
        endpoint_budget=0.0625,
        scorer=_DualScorer(),
        objective="direct_cost_aware",
        feedback=True,
    )

    assert raw.steps[0].action.cell_index == 0
    assert direct.steps[0].action.cell_index == 63


def test_mavis_rollout_accepts_quantized_initial_scout() -> None:
    authority = synthetic_authority(native_shape=(338, 340))

    curve = rollout_scout_and_focus_curve(
        authority,
        specimen_id="sample-001",
        initial_budget=0.03125,
        checkpoints=(0.03125, 0.0625),
        scorer=_CountingScorer(),
        objective="direct_cost_aware",
        feedback=True,
    )

    assert curve.checkpoint_states[0].effective_budget > 0.03125
    assert curve.checkpoint_states[1].effective_budget <= 0.0625


def test_rollout_candidate_costs_match_authoritative_legacy_records() -> None:
    authority = synthetic_authority(native_shape=(338, 340))
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("sample-001"),
        initial_budget=0.03125,
        checkpoint=0.25,
    )

    actions, descriptors = _candidate_descriptors(
        state,
        endpoint_budget=0.25,
        action_budget=0.0625,
    )

    grid = build_acquisition_grid(338, 340, initial_budget=0.03125)
    expected = tuple(
        candidate_budget_record(grid, state.measurement_state, action).measured_count
        - state.exact_acquired_count
        for action in actions
    )
    assert tuple(item.exact_added_cost for item in descriptors) == expected
