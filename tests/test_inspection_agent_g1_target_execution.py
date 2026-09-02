from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.contracts import TaskTokenMode
from cmc_bbdm.inspection_agent_g1.features import canonical_slot
from cmc_bbdm.inspection_agent_g1.formal import plan_g1_fixed_actions
from cmc_bbdm.inspection_agent_g1.rollout import (
    ClosedLoopTrajectory,
    ObservablePolicyScores,
    RolloutStep,
    SurfaceVariant,
)
from cmc_bbdm.inspection_agent_g1.target_execution import (
    G1TargetExecutionError,
    G1TargetTrajectoryRecord,
    TargetPolicyVariant,
    read_g1_target_trajectory_bank,
    write_g1_target_trajectory_bank,
)
from cmc_bbdm.inspection_agent_g1.warm_start import build_deployment_grid
from cmc_bbdm.mavis.authority import MAVISAuthority


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _record() -> G1TargetTrajectoryRecord:
    rows, columns = np.indices((41, 43))
    full_scan = np.stack((rows, columns, rows + columns), axis=2).astype(np.uint8)
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("sample",),
        dataset_ids=("d6",),
        images=(full_scan,),
        targets=np.asarray([0.4]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    grid = build_deployment_grid(full_scan.shape[:2])
    hypothesis = SurfaceHypothesis(
        scores=np.linspace(0.0, 1.0, 64),
        top_cells=tuple(range(63, 55, -1)),
        border_median_rgb=np.zeros(3),
        state_sha256=_sha("hypothesis"),
    )
    world = CausalInspectionWorld(
        authority,
        specimen_id="sample",
        task=InspectionTask.CAI,
        surface_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        surface_sha256=_sha("surface"),
        grid=grid,
        endpoint_budget=0.25,
    )
    actions = plan_g1_fixed_actions(
        grid,
        hypothesis,
        surface_sha256=_sha("surface"),
        specimen_sha256=_sha("sample"),
        method="ZERO_UNIFORM",
        random_seed=2026090101,
        endpoint_budget=0.25,
    )
    current = world.replay(actions[:8])
    model_sha = _sha("stop-model")
    policy_sha = _sha("policy-0")
    logits = np.full(192, -np.inf, dtype=np.float64)
    logits[canonical_slot(actions[8])] = 1.0
    selected = RolloutStep(
        step_index=0,
        observation_sha256=current.state_sha256,
        policy_state_sha256=policy_sha,
        scores=ObservablePolicyScores(
            policy_state_sha256=policy_sha,
            model_sha256=model_sha,
            action_logits=logits,
            stop_probability=0.4,
        ),
        selected_slot=canonical_slot(actions[8]),
    )
    current = world.step(current, actions[8])
    stop_policy_sha = _sha("policy-1")
    stop_logits = np.zeros(192, dtype=np.float64)
    stopped = RolloutStep(
        step_index=1,
        observation_sha256=current.state_sha256,
        policy_state_sha256=stop_policy_sha,
        scores=ObservablePolicyScores(
            policy_state_sha256=stop_policy_sha,
            model_sha256=model_sha,
            action_logits=stop_logits,
            stop_probability=0.95,
        ),
        selected_slot=None,
    )
    trajectory = ClosedLoopTrajectory(
        target_domain="d6",
        specimen_sha256=_sha("sample"),
        task=InspectionTask.CAI,
        model_sha256=model_sha,
        stop_threshold=0.9,
        stopped=True,
        termination_reason="STOP",
        action_history=current.action_history,
        steps=(selected, stopped),
        final_observation_sha256=current.state_sha256,
        acquired_positions=current.acquired_positions,
        acquired_values=current.measurement_values,
        native_count=current.native_count,
        effective_budget=current.effective_budget,
    )
    return G1TargetTrajectoryRecord(
        outer_target="d6",
        specimen_id="sample",
        specimen_sha256=_sha("sample"),
        task=InspectionTask.CAI,
        variant=TargetPolicyVariant.PROPOSED_STOP,
        task_token_mode=TaskTokenMode.CORRECT,
        surface_variant=SurfaceVariant.CORRECT_SURFACE,
        donor_surface_sha256=None,
        final_dependency_sha256=_sha("dependencies"),
        action_selection_sha256=_sha("selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=model_sha,
        trajectory=trajectory,
    )


def test_target_trajectory_bank_round_trips_every_frozen_policy_value(
    tmp_path: Path,
) -> None:
    record = _record()
    path = tmp_path / "target.parquet"

    identity = write_g1_target_trajectory_bank(path, (record,))
    replay_identity, replay = read_g1_target_trajectory_bank(path)

    assert replay_identity == identity
    assert len(replay) == 1
    actual = replay[0]
    assert actual.state_sha256 == record.state_sha256
    assert actual.trajectory.state_sha256 == record.trajectory.state_sha256
    np.testing.assert_array_equal(
        actual.trajectory.acquired_positions,
        record.trajectory.acquired_positions,
    )
    np.testing.assert_array_equal(
        actual.trajectory.acquired_values,
        record.trajectory.acquired_values,
    )
    for expected, observed in zip(
        record.trajectory.steps,
        actual.trajectory.steps,
        strict=True,
    ):
        np.testing.assert_array_equal(
            observed.scores.action_logits,
            expected.scores.action_logits,
        )
        assert observed.scores.stop_probability == expected.scores.stop_probability


def test_target_trajectory_record_rejects_a_variant_mode_mismatch() -> None:
    with pytest.raises(G1TargetExecutionError, match="record is invalid"):
        replace(_record(), task_token_mode=TaskTokenMode.NO_TASK)


def test_target_trajectory_bank_rejects_manifest_tampering(tmp_path: Path) -> None:
    path = tmp_path / "target.parquet"
    write_g1_target_trajectory_bank(path, (_record(),))
    manifest_path = path.with_suffix(".parquet.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["records_sha256"] = _sha("tampered")
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    with pytest.raises(G1TargetExecutionError, match="evidence changed"):
        read_g1_target_trajectory_bank(path)
