from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.features import canonical_slot
from cmc_bbdm.inspection_agent_g1.formal import plan_g1_fixed_actions
from cmc_bbdm.inspection_agent_g1.rollout import (
    ClosedLoopTrajectory,
    ObservablePolicyScores,
    RolloutStep,
)
from cmc_bbdm.inspection_agent_g1.stop_selection_execution import (
    evaluate_source_stop_validation_trajectory,
)
from cmc_bbdm.inspection_agent_g1.warm_start import build_deployment_grid
from cmc_bbdm.mavis.authority import MAVISAuthority


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _Encoder:
    def encode(self, images: object) -> np.ndarray:
        return np.zeros((len(tuple(images)), 512), dtype=np.float64)


class _Assessor:
    model_state_sha256 = _sha("assessor")

    def predict(self, embeddings: object, scalars: object) -> np.ndarray:
        del embeddings
        values = np.asarray(scalars, dtype=np.float64)
        return 0.2 + values[:, 0]


def _fixture(task: InspectionTask):
    rows, columns = np.indices((41, 43))
    full_scan = np.stack((rows, columns, rows + columns), axis=2).astype(np.uint8)
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("sample",),
        dataset_ids=("d1",),
        images=(full_scan,),
        targets=np.asarray([0.4]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    grid = build_deployment_grid(full_scan.shape[:2])
    surface = SurfaceHypothesis(
        scores=np.linspace(0.0, 1.0, 64),
        top_cells=tuple(range(63, 55, -1)),
        border_median_rgb=np.zeros(3),
        state_sha256=_sha("surface-hypothesis"),
    )
    world = CausalInspectionWorld(
        authority,
        specimen_id="sample",
        task=task,
        surface_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        surface_sha256=_sha("surface"),
        grid=grid,
        endpoint_budget=0.25,
    )
    prior = SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=("d2", "d3", "d4", "d5"),
        fit_specimen_ids=("a", "b", "c", "d"),
        source_authority_sha256=_sha("authority"),
        domain_border_medians=np.zeros((4, 3)),
        background_rgb=np.zeros(3, dtype=np.uint8),
    )
    actions = plan_g1_fixed_actions(
        grid,
        surface,
        surface_sha256=_sha("surface"),
        specimen_sha256=_sha("sample"),
        method="ZERO_UNIFORM",
        random_seed=2026090101,
        endpoint_budget=0.25,
    )
    current = world.replay(actions[:8])
    model_sha = _sha("stop-model")
    steps = []
    for index, action in enumerate(actions[8:]):
        policy_state_sha = _sha(f"policy-{task.value}-{index}")
        logits = np.full(192, -np.inf, dtype=np.float64)
        logits[canonical_slot(action)] = 1.0
        steps.append(
            RolloutStep(
                step_index=index,
                observation_sha256=current.state_sha256,
                policy_state_sha256=policy_state_sha,
                scores=ObservablePolicyScores(
                    policy_state_sha256=policy_state_sha,
                    model_sha256=model_sha,
                    action_logits=logits,
                    stop_probability=min(0.99, 0.1 + current.effective_budget),
                ),
                selected_slot=canonical_slot(action),
            )
        )
        current = world.step(current, action)
    trajectory = ClosedLoopTrajectory(
        target_domain="d1",
        specimen_sha256=_sha("sample"),
        task=task,
        model_sha256=model_sha,
        stop_threshold=None,
        stopped=False,
        termination_reason="NO_LEGAL_ACTION",
        action_history=current.action_history,
        steps=tuple(steps),
        final_observation_sha256=current.state_sha256,
        acquired_positions=current.acquired_positions,
        acquired_values=current.measurement_values,
        native_count=current.native_count,
        effective_budget=current.effective_budget,
    )
    return world, grid, prior, full_scan, trajectory


def test_frozen_source_rollout_becomes_field_stop_validation_curve() -> None:
    world, grid, prior, full_scan, trajectory = _fixture(InspectionTask.FIELD)

    result = evaluate_source_stop_validation_trajectory(
        world,
        grid,
        trajectory,
        prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        outer_target="d6",
        source_domain="d1",
        full_scan=full_scan,
        true_cai=0.4,
        reference_true_loss=0.01,
    )

    assert result.task is InspectionTask.FIELD
    assert result.budgets[0] > 0.0
    assert result.budgets[-1] == trajectory.effective_budget
    assert result.stop_probabilities[-1] == 0.0
    assert len(result.budgets) == len(trajectory.steps) + 1
    assert all(value >= 0.0 for value in result.true_task_losses)


def test_frozen_source_rollout_becomes_cai_stop_validation_curve() -> None:
    world, grid, prior, full_scan, trajectory = _fixture(InspectionTask.CAI)

    result = evaluate_source_stop_validation_trajectory(
        world,
        grid,
        trajectory,
        prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        outer_target="d6",
        source_domain="d1",
        full_scan=full_scan,
        true_cai=0.4,
        reference_true_loss=0.1,
    )

    expected = tuple(abs(0.4 - (0.2 + budget)) for budget in result.budgets)
    np.testing.assert_allclose(result.true_task_losses, expected, rtol=0.0, atol=0.0)
