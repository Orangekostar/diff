from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.msss.pipeline import MSSSPipelineError, execution_indices

GROUPS = ("d1", "d2", "d3", "d4", "d5", "d6")


def test_smoke_indices_take_nine_specimens_per_domain_in_registered_order() -> None:
    dataset_ids = tuple(group for group in GROUPS for _ in range(12))
    indices = execution_indices(dataset_ids, group_order=GROUPS, mode="smoke")

    assert indices.dtype == np.int64
    assert indices.shape == (54,)
    assert tuple(dataset_ids[index] for index in indices[::9]) == GROUPS
    assert all(sum(dataset_ids[index] == group for index in indices) == 9 for group in GROUPS)


def test_full_indices_preserve_the_complete_roster() -> None:
    dataset_ids = tuple(group for group in GROUPS for _ in range(10))
    indices = execution_indices(dataset_ids, group_order=GROUPS, mode="full")

    np.testing.assert_array_equal(indices, np.arange(60, dtype=np.int64))


def test_smoke_indices_fail_when_a_domain_is_too_small() -> None:
    with pytest.raises(MSSSPipelineError, match="nine"):
        execution_indices(("d1",) * 8 + tuple(group for group in GROUPS[1:] for _ in range(9)), group_order=GROUPS, mode="smoke")
