import numpy as np

from scripts.cai_actor_c0_diagnostic.diagnose import (
    attribution_cells,
    masked_softmax,
    select_probe_cells,
)


def test_conditional_softmax_uses_identical_logits_and_reports_blocked_mass():
    logits = np.linspace(-2.0, 2.0, 64)
    legal = np.ones(64, dtype=bool)
    legal[0] = False
    proposal = np.zeros(64, dtype=bool)
    proposal[[5, 7, 9]] = True
    p_env = masked_softmax(logits, legal)
    p_policy = masked_softmax(logits, proposal)
    conditional = p_env[proposal] / p_env[proposal].sum()
    assert np.allclose(p_policy[proposal], conditional, atol=1e-12, rtol=0)
    assert np.isclose(p_env.sum(), 1.0)
    assert np.isclose(p_policy.sum(), 1.0)
    assert np.isclose(p_env[~proposal].sum(), 1.0 - p_env[proposal].sum())


def test_attribution_cells_preserve_signed_sum_and_l1():
    attribution = np.zeros((2, 64, 2))
    attribution[:, :2] = np.asarray(
        [[[1.0, -2.0], [3.0, 4.0]], [[-1.0, 1.0], [2.0, -2.0]]]
    )
    signed, l1 = attribution_cells(attribution)
    assert np.array_equal(signed[:, :2], np.asarray([[-1.0, 7.0], [0.0, 0.0]]))
    assert np.array_equal(l1[:, :2], np.asarray([[3.0, 7.0], [2.0, 4.0]]))
    assert np.allclose(signed.sum(axis=1), attribution.sum(axis=(1, 2)))


def test_probe_selection_is_deterministic_and_disjoint():
    magnitude = np.asarray([1.0] * 64)
    top, bottom = select_probe_cells(magnitude, count=3)
    assert top == (0, 1, 2)
    assert bottom == (3, 4, 5)
