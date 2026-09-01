from __future__ import annotations

import hashlib

import numpy as np
import pytest
from test_inspection_agent_g1_closed_loop import (
    _runtime,
    _sha,
    _state_builder,
    _StopAfterOneAction,
)

from cmc_bbdm.inspection_agent_g1.rollout import (
    TargetTruthVault,
    TargetTruthVaultError,
    run_closed_loop,
)


def test_target_truth_access_fails_until_policy_trajectory_is_sealed() -> None:
    world, grid, full_scan = _runtime()
    vault = TargetTruthVault(
        target_domain="target",
        specimen_sha256=_sha("sample"),
        full_scan=full_scan,
        true_cai=0.4,
    )
    with pytest.raises(TargetTruthVaultError, match="not sealed"):
        vault.open_truth(None)

    trajectory = run_closed_loop(
        world,
        grid,
        target_domain="target",
        specimen_sha256=_sha("sample"),
        state_builder=_state_builder(grid),
        actor=_StopAfterOneAction(),
        stop_threshold=0.90,
    )
    seal = vault.seal_trajectory(trajectory)
    truth = vault.open_truth(seal)
    np.testing.assert_array_equal(truth.full_scan, full_scan)
    assert truth.true_cai == pytest.approx(0.4)
    assert truth.trajectory_sha256 == trajectory.state_sha256
    assert hashlib.sha256(truth.full_scan.tobytes()).hexdigest() == truth.full_scan_sha256


def test_target_truth_vault_rejects_a_different_trajectory_identity() -> None:
    world, grid, full_scan = _runtime()
    trajectory = run_closed_loop(
        world,
        grid,
        target_domain="target",
        specimen_sha256=_sha("sample"),
        state_builder=_state_builder(grid),
        actor=_StopAfterOneAction(),
        stop_threshold=0.90,
    )
    vault = TargetTruthVault(
        target_domain="target",
        specimen_sha256=_sha("different"),
        full_scan=full_scan,
        true_cai=0.4,
    )
    with pytest.raises(TargetTruthVaultError, match="identity"):
        vault.seal_trajectory(trajectory)
