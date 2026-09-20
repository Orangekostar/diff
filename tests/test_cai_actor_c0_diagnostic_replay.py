import numpy as np

from scripts.cai_actor_c0_diagnostic.replay import (
    PhysicalState,
    choose_action,
    physical_state_sha256,
    snapshot_indices,
)


def state(*, history=None, prior=None):
    measured = np.zeros(64, dtype=bool)
    measured[[1, 4]] = True
    surface = np.arange(64 * 4, dtype=np.float32).reshape(64, 4)
    observed = np.zeros_like(surface)
    observed[measured] = surface[measured] * 2
    return PhysicalState(
        surface=surface,
        observed_cscan=observed,
        measured=measured,
        history=np.asarray(history or [0.0, 1 / 64, 0.0, 0.0, 2 / 64] + [0.0] * 59, dtype=np.float32),
        exact_cost=0.125,
        actor_cost=np.float32(0.125),
        remaining_cost=np.float32(0.125),
        current_prediction_mpa=np.float32(250.0),
        environment_legal=~measured,
        policy_prior=np.asarray(prior if prior is not None else np.zeros(64), dtype=np.float32),
    )


def test_physical_hash_excludes_prior_but_includes_ordered_history():
    first = state(prior=np.ones(64))
    second = state(prior=np.zeros(64))
    assert physical_state_sha256(first) == physical_state_sha256(second)
    changed = state(history=[0.0, 2 / 64, 0.0, 0.0, 1 / 64] + [0.0] * 59)
    assert physical_state_sha256(first) != physical_state_sha256(changed)


def test_snapshot_indices_are_pre_action_and_terminal_is_separate():
    assert snapshot_indices(16) == ((0, 1, 2, 8, 15), 16)
    assert snapshot_indices(8) == ((0, 1, 2, 7), 8)
    assert snapshot_indices(2) == ((0, 1), 2)


def test_choose_action_uses_original_first_index_tie_break_and_mask_only_c0():
    logits = np.zeros(64, dtype=np.float32)
    logits[3] = logits[5] = 2.0
    legal = np.ones(64, dtype=bool)
    c0 = np.zeros(64, dtype=bool)
    c0[5] = True
    assert choose_action(logits, c0) == 5
    assert choose_action(logits, legal) == 3
    assert np.array_equal(logits, logits.copy())
