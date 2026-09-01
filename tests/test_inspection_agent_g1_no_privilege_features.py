from __future__ import annotations

import inspect
from dataclasses import fields

import pytest

from cmc_bbdm.inspection_agent_g1.contracts import G1PolicyState
from cmc_bbdm.inspection_agent_g1.features import (
    PRIVILEGED_INPUT_KEYS,
    G1FeatureError,
    build_policy_state,
    reject_privileged_columns,
)

EXPECTED_FORBIDDEN = {
    "true_cai",
    "full_scan",
    "dataset_id",
    "specimen_id",
    "oracle_value",
    "oracle_utility",
    "future_measurement",
    "true_task_loss",
}


def test_policy_state_and_builder_have_no_privileged_surface() -> None:
    assert PRIVILEGED_INPUT_KEYS == frozenset(EXPECTED_FORBIDDEN)
    assert EXPECTED_FORBIDDEN.isdisjoint(field.name for field in fields(G1PolicyState))
    assert EXPECTED_FORBIDDEN.isdisjoint(inspect.signature(build_policy_state).parameters)


@pytest.mark.parametrize("name", sorted(EXPECTED_FORBIDDEN))
def test_actor_feature_builder_rejects_each_privileged_column(name: str) -> None:
    with pytest.raises(G1FeatureError, match="privileged actor input"):
        reject_privileged_columns({name: object()})


def test_actor_feature_builder_rejects_unknown_supplemental_columns() -> None:
    with pytest.raises(G1FeatureError, match="unexpected actor input"):
        reject_privileged_columns({"unregistered_hint": 1.0})
