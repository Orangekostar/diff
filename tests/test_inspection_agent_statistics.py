from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.statistics import (
    InspectionStatisticsError,
    synchronized_paired_bootstrap,
)


def test_synchronized_bootstrap_equal_weights_domains_after_specimen_resampling() -> None:
    summary = synchronized_paired_bootstrap(
        dataset_ids=("a", "a", "b", "b"),
        specimen_ids=("a1", "a2", "b1", "b2"),
        effects=np.asarray((1.0, 3.0, -1.0, 1.0)),
        replicates=5000,
        seed=17,
    )
    assert summary.point_estimate == pytest.approx(1.0)
    assert summary.domain_effects == (("a", 2.0), ("b", 0.0))
    assert summary.improved_domains == 1
    assert summary.replicates == 5000
    assert summary.ci_lower <= summary.point_estimate <= summary.ci_upper
    assert len(summary.distribution_sha256) == 64


def test_bootstrap_is_invariant_to_input_row_order() -> None:
    first = synchronized_paired_bootstrap(
        dataset_ids=("a", "a", "b", "b"),
        specimen_ids=("a1", "a2", "b1", "b2"),
        effects=np.asarray((1.0, 3.0, -1.0, 1.0)),
        replicates=2000,
        seed=21,
    )
    second = synchronized_paired_bootstrap(
        dataset_ids=("b", "a", "b", "a"),
        specimen_ids=("b2", "a2", "b1", "a1"),
        effects=np.asarray((1.0, 3.0, -1.0, 1.0)),
        replicates=2000,
        seed=21,
    )
    assert first == second


def test_bootstrap_rejects_duplicate_specimen_rows_as_pseudoreplication() -> None:
    with pytest.raises(InspectionStatisticsError, match="one paired effect"):
        synchronized_paired_bootstrap(
            dataset_ids=("a", "a"),
            specimen_ids=("s1", "s1"),
            effects=np.asarray((1.0, 2.0)),
            replicates=100,
            seed=1,
        )
