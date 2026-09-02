from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.features import canonical_slot
from cmc_bbdm.inspection_agent_g1.formal import plan_g1_fixed_actions
from cmc_bbdm.inspection_agent_g1.g1 import (
    G1FinalDependencies,
    G1Protocol,
    G1Runtime,
    G1RuntimeSurface,
)
from cmc_bbdm.inspection_agent_g1.policy_training import TrainedObservablePolicy
from cmc_bbdm.inspection_agent_g1.rollout import (
    ClosedLoopTrajectory,
    ObservablePolicyScores,
    RolloutStep,
    SurfaceVariant,
)
from cmc_bbdm.inspection_agent_g1.stop_selection_execution import (
    G1OuterStopSelectionRun,
)
from cmc_bbdm.inspection_agent_g1.stop_training import TrainedObservableStopPolicy
from cmc_bbdm.inspection_agent_g1.stopping_policy import StopThresholdSelection
from cmc_bbdm.inspection_agent_g1.target_execution import (
    G1TargetExecutionError,
    G1TargetTrajectoryRecord,
    TargetPolicyVariant,
    build_g1_outer_target_trajectory_bank,
    materialize_g1_target_variant_records,
    plan_g1_target_variants,
    read_g1_target_trajectory_bank,
    seal_g1_target_trajectory_bank,
    target_trajectory_fragment_path,
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


class _Encoder:
    def encode(self, images: object) -> np.ndarray:
        return np.zeros((len(tuple(images)), 512), dtype=np.float64)


class _Assessor:
    model_state_sha256 = _sha("assessor")

    def predict(self, embeddings: object, scalars: object) -> np.ndarray:
        del scalars
        return np.full(len(np.asarray(embeddings)), 0.4, dtype=np.float64)


class _Actor:
    model_state_sha256 = _sha("stop-model")

    def __call__(self, state: object) -> ObservablePolicyScores:
        return self.score_batch((state,))[0]

    def score_batch(
        self, states: tuple[object, ...]
    ) -> tuple[ObservablePolicyScores, ...]:
        output = []
        for state in states:
            logits = np.full(192, -np.inf, dtype=np.float64)
            legal = np.flatnonzero(state.legal_action_mask)
            logits[legal] = -legal.astype(np.float64)
            output.append(
                ObservablePolicyScores(
                    policy_state_sha256=state.state_sha256,
                    model_sha256=self.model_state_sha256,
                    action_logits=logits,
                    stop_probability=0.0,
                )
            )
        return tuple(output)


def _target_runtime() -> G1Runtime:
    specimen_ids = ("target-a", "target-b")
    dataset_ids = ("d6", "d6")
    rows, columns = np.indices((41, 43))
    images = tuple(
        np.stack((rows + index, columns + index, rows + columns), axis=2).astype(
            np.uint8
        )
        for index in range(2)
    )
    authority = MAVISAuthority.from_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        images=images,
        targets=np.asarray([0.4, 0.5]),
        metadata13=np.zeros((2, 13)),
        profile_stats21=np.zeros((2, 21)),
    )
    surfaces = {}
    for index, specimen_id in enumerate(specimen_ids):
        hypothesis = SurfaceHypothesis(
            scores=np.linspace(float(index), float(index + 1), 64),
            top_cells=tuple(range(63 - index, 55 - index, -1)),
            border_median_rgb=np.zeros(3),
            state_sha256=_sha(f"hypothesis-{specimen_id}"),
        )
        surfaces[("d6", specimen_id)] = G1RuntimeSurface(
            dataset_id="d6",
            specimen_id=specimen_id,
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            surface_sha256=_sha(f"surface-{specimen_id}"),
            hypothesis=hypothesis,
        )
    return G1Runtime(
        mavis=authority,
        surfaces=MappingProxyType(surfaces),
        surface_authority_sha256=_sha("surface-authority"),
    )


def test_target_variant_materialization_never_opens_target_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = _target_runtime()

    def forbidden_truth_view(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("target truth view was opened before trajectory freeze")

    monkeypatch.setattr(MAVISAuthority, "source_teacher_view", forbidden_truth_view)
    monkeypatch.setattr(MAVISAuthority, "evaluation_view", forbidden_truth_view)
    prior = SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=("d1", "d2", "d3", "d4", "d5"),
        fit_specimen_ids=("a", "b", "c", "d", "e"),
        source_authority_sha256=_sha("source-authority"),
        domain_border_medians=np.zeros((5, 3)),
        background_rgb=np.zeros(3, dtype=np.uint8),
    )

    records = materialize_g1_target_variant_records(
        runtime,
        outer_target="d6",
        specimen_ids=("target-a", "target-b"),
        task=InspectionTask.CAI,
        variant=TargetPolicyVariant.NO_TASK,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        actor=_Actor(),
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        endpoint_budget=0.25,
        final_dependency_sha256=_sha("dependencies"),
        action_selection_sha256=_sha("selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        stop_threshold=None,
    )

    assert len(records) == 2
    assert all(row.variant is TargetPolicyVariant.NO_TASK for row in records)
    assert all(row.task_token_mode is TaskTokenMode.NO_TASK for row in records)
    assert all(row.trajectory.effective_budget <= 0.25 for row in records)
    assert all(row.trajectory.stop_threshold is None for row in records)
    path = tmp_path / "target.parquet"
    identity = write_g1_target_trajectory_bank(path, records)
    seal = seal_g1_target_trajectory_bank(runtime, identity, records)
    assert seal.outer_target == "d6"
    assert seal.record_count == 2
    assert seal.bank_manifest_sha256 == identity.manifest_sha256


def test_target_variant_plan_adds_stop_only_for_a_source_authorized_task() -> None:
    thresholds = (
        StopThresholdSelection(
            outer_target="d6",
            task=InspectionTask.FIELD,
            status="STOP_AUTHORIZED_SOURCE_ONLY",
            threshold=0.9,
            candidates=(),
            trajectory_sha256=(),
            state_sha256=_sha("field-threshold"),
        ),
        StopThresholdSelection(
            outer_target="d6",
            task=InspectionTask.CAI,
            status="STOP_NOT_AUTHORIZED",
            threshold=None,
            candidates=(),
            trajectory_sha256=(),
            state_sha256=_sha("cai-threshold"),
        ),
    )

    plan = plan_g1_target_variants("d6", thresholds)

    assert len(plan) == 11
    assert (
        InspectionTask.FIELD,
        TargetPolicyVariant.PROPOSED_STOP,
        0.9,
    ) in plan
    assert not any(
        task is InspectionTask.CAI and variant is TargetPolicyVariant.PROPOSED_STOP
        for task, variant, _threshold in plan
    )


def _thresholds() -> tuple[StopThresholdSelection, ...]:
    return (
        StopThresholdSelection(
            outer_target="d6",
            task=InspectionTask.FIELD,
            status="STOP_AUTHORIZED_SOURCE_ONLY",
            threshold=0.9,
            candidates=(),
            trajectory_sha256=(),
            state_sha256=_sha("field-threshold"),
        ),
        StopThresholdSelection(
            outer_target="d6",
            task=InspectionTask.CAI,
            status="STOP_NOT_AUTHORIZED",
            threshold=None,
            candidates=(),
            trajectory_sha256=(),
            state_sha256=_sha("cai-threshold"),
        ),
    )


def _outer_build_inputs():
    runtime = _target_runtime()
    protocol = object.__new__(G1Protocol)
    object.__setattr__(protocol, "domain_order", ("d6",))
    object.__setattr__(protocol, "domain_counts", MappingProxyType({"d6": 2}))
    object.__setattr__(protocol, "endpoint_budget", 0.25)
    prior = SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=("d1", "d2", "d3", "d4", "d5"),
        fit_specimen_ids=("a", "b", "c", "d", "e"),
        source_authority_sha256=_sha("source-authority"),
        domain_border_medians=np.zeros((5, 3)),
        background_rgb=np.zeros(3, dtype=np.uint8),
    )
    dependencies = object.__new__(G1FinalDependencies)
    object.__setattr__(dependencies, "outer_target", "d6")
    object.__setattr__(dependencies, "fit_domains", ())
    object.__setattr__(dependencies, "prior", prior)
    object.__setattr__(dependencies, "assessor", _Assessor())
    object.__setattr__(dependencies, "state_sha256", _sha("dependencies"))
    action = object.__new__(TrainedObservablePolicy)
    object.__setattr__(
        action,
        "audit",
        SimpleNamespace(fit_domains=(), validation_domain=None),
    )
    object.__setattr__(
        action,
        "hyperparameters",
        SimpleNamespace(
            task_token_mode=TaskTokenMode.CORRECT,
            cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        ),
    )
    object.__setattr__(action, "model_state_sha256", _sha("action-model"))
    stop = object.__new__(TrainedObservableStopPolicy)
    object.__setattr__(
        stop,
        "audit",
        SimpleNamespace(fit_domains=(), validation_domain=None),
    )
    object.__setattr__(stop, "base_action_model_sha256", _sha("action-model"))
    object.__setattr__(stop, "model_state_sha256", _sha("stop-model"))
    selection = object.__new__(G1OuterStopSelectionRun)
    object.__setattr__(selection, "outer_target", "d6")
    object.__setattr__(selection, "action_policy", action)
    object.__setattr__(selection, "stop_policy", stop)
    object.__setattr__(selection, "thresholds", _thresholds())
    object.__setattr__(selection, "target_outcomes_opened", False)
    object.__setattr__(selection, "state_sha256", _sha("outer-selection"))
    return runtime, protocol, dependencies, selection


def _sealed_mock_record(
    runtime: G1Runtime,
    *,
    specimen_id: str,
    task: InspectionTask,
    variant: TargetPolicyVariant,
    threshold: float | None,
) -> G1TargetTrajectoryRecord:
    base = _record().trajectory
    token_mode, surface_variant = {
        TargetPolicyVariant.PROPOSED: (
            TaskTokenMode.CORRECT,
            SurfaceVariant.CORRECT_SURFACE,
        ),
        TargetPolicyVariant.PROPOSED_STOP: (
            TaskTokenMode.CORRECT,
            SurfaceVariant.CORRECT_SURFACE,
        ),
        TargetPolicyVariant.NO_TASK: (
            TaskTokenMode.NO_TASK,
            SurfaceVariant.CORRECT_SURFACE,
        ),
        TargetPolicyVariant.WRONG_TASK: (
            TaskTokenMode.WRONG_TASK,
            SurfaceVariant.CORRECT_SURFACE,
        ),
        TargetPolicyVariant.NO_SURFACE: (
            TaskTokenMode.CORRECT,
            SurfaceVariant.NO_SURFACE,
        ),
        TargetPolicyVariant.SHUFFLED_SURFACE: (
            TaskTokenMode.CORRECT,
            SurfaceVariant.SHUFFLED_SURFACE,
        ),
    }[variant]
    policy_sha = _sha(f"{specimen_id}-{task.value}-{variant.value}-policy")
    selected_base = base.steps[0]
    selected_scores = ObservablePolicyScores(
        policy_state_sha256=policy_sha,
        model_sha256=_sha("stop-model"),
        action_logits=selected_base.scores.action_logits,
        stop_probability=0.4,
    )
    steps = [
        RolloutStep(
            step_index=0,
            observation_sha256=selected_base.observation_sha256,
            policy_state_sha256=policy_sha,
            scores=selected_scores,
            selected_slot=selected_base.selected_slot,
        )
    ]
    stopped = variant is TargetPolicyVariant.PROPOSED_STOP
    if stopped:
        stop_policy_sha = _sha(
            f"{specimen_id}-{task.value}-{variant.value}-stop-policy"
        )
        steps.append(
            RolloutStep(
                step_index=1,
                observation_sha256=base.steps[1].observation_sha256,
                policy_state_sha256=stop_policy_sha,
                scores=ObservablePolicyScores(
                    policy_state_sha256=stop_policy_sha,
                    model_sha256=_sha("stop-model"),
                    action_logits=base.steps[1].scores.action_logits,
                    stop_probability=0.95,
                ),
                selected_slot=None,
            )
        )
    specimen_sha = runtime.specimen_sha256("d6", specimen_id)
    trajectory = ClosedLoopTrajectory(
        target_domain="d6",
        specimen_sha256=specimen_sha,
        task=task,
        model_sha256=_sha("stop-model"),
        stop_threshold=threshold,
        stopped=stopped,
        termination_reason="STOP" if stopped else "NO_LEGAL_ACTION",
        action_history=base.action_history,
        steps=tuple(steps),
        final_observation_sha256=_sha(
            f"{specimen_id}-{task.value}-{variant.value}-final"
        ),
        acquired_positions=base.acquired_positions,
        acquired_values=base.acquired_values,
        native_count=base.native_count,
        effective_budget=base.effective_budget,
    )
    return G1TargetTrajectoryRecord(
        outer_target="d6",
        specimen_id=specimen_id,
        specimen_sha256=specimen_sha,
        task=task,
        variant=variant,
        task_token_mode=token_mode,
        surface_variant=surface_variant,
        donor_surface_sha256=(
            _sha(f"donor-{specimen_id}")
            if variant is TargetPolicyVariant.SHUFFLED_SURFACE
            else None
        ),
        final_dependency_sha256=_sha("dependencies"),
        action_selection_sha256=_sha("outer-selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        trajectory=trajectory,
    )


def test_outer_target_bank_replays_complete_fragments_without_rerollout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cmc_bbdm.inspection_agent_g1 import target_execution as module

    runtime, protocol, dependencies, selection = _outer_build_inputs()
    calls = []

    def fake_materialize(_runtime: G1Runtime, **kwargs: object):
        calls.append((kwargs["task"], kwargs["variant"]))
        return tuple(
            _sealed_mock_record(
                runtime,
                specimen_id=specimen_id,
                task=kwargs["task"],
                variant=kwargs["variant"],
                threshold=kwargs["stop_threshold"],
            )
            for specimen_id in kwargs["specimen_ids"]
        )

    monkeypatch.setattr(module, "materialize_g1_target_variant_records", fake_materialize)
    first = build_g1_outer_target_trajectory_bank(
        runtime,
        protocol,
        dependencies,
        selection,
        encoder=_Encoder(),
        work_root=tmp_path,
    )
    assert first.record_count == 22
    assert len(calls) == 11

    monkeypatch.setattr(
        module,
        "materialize_g1_target_variant_records",
        lambda *_args, **_kwargs: pytest.fail("complete fragment was rerun"),
    )
    replay = build_g1_outer_target_trajectory_bank(
        runtime,
        protocol,
        dependencies,
        selection,
        encoder=_Encoder(),
        work_root=tmp_path,
    )
    assert replay.state_sha256 == first.state_sha256


def test_outer_target_bank_rejects_a_half_written_fragment(
    tmp_path: Path,
) -> None:
    runtime, protocol, dependencies, selection = _outer_build_inputs()
    fragment = target_trajectory_fragment_path(
        tmp_path,
        "d6",
        InspectionTask.FIELD,
        TargetPolicyVariant.PROPOSED,
    )
    fragment.parent.mkdir(parents=True)
    fragment.write_bytes(b"incomplete")

    with pytest.raises(G1TargetExecutionError, match="fragment is incomplete"):
        build_g1_outer_target_trajectory_bank(
            runtime,
            protocol,
            dependencies,
            selection,
            encoder=_Encoder(),
            work_root=tmp_path,
        )
