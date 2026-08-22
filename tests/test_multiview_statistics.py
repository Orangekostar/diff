from __future__ import annotations

import numpy as np

from cmc_bbdm.aei_multiview_regression.statistics import common_domain_bootstrap


def test_common_bootstrap_uses_one_registered_index_matrix_for_all_effects() -> None:
    effects = {
        "e2_cooperative": np.arange(6, dtype=np.float64) / 100.0,
        "e3_weighted": np.arange(6, dtype=np.float64) / 200.0,
        "e3_stacking": np.arange(6, dtype=np.float64) / 300.0,
        "e3_gmvr": np.arange(6, dtype=np.float64) / 400.0,
    }

    result = common_domain_bootstrap(effects)

    assert result.indices.shape == (100_000, 6)
    assert result.indices.flags.writeable is False
    assert result.seed == 20260811
    assert len(result.effects) == 4
    first = result.effects[0]
    expected = np.mean(effects[first.name][result.indices], axis=1)
    assert first.bootstrap_mean == np.mean(expected)


def test_common_bootstrap_is_deterministic() -> None:
    effects = {f"effect_{index}": np.full(6, index / 100.0) for index in range(4)}
    first = common_domain_bootstrap(effects)
    second = common_domain_bootstrap(effects)

    np.testing.assert_array_equal(first.indices, second.indices)
    assert first.effects == second.effects
