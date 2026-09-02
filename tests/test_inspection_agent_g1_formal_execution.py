from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.state import (
    action_added_positions,
    apply_action,
    fitting_actions,
    zero_state,
)
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.formal import (
    FIXED_BASELINE_METHODS,
    G1ObservableStateBuilder,
    evaluate_g1_action_history,
    plan_g1_fixed_actions,
    run_g1_warm_started_oracle_actions,
)
from cmc_bbdm.inspection_agent_g1.warm_start import (
    PRIMARY_WARM_START_CELLS,
    build_deployment_grid,
)
from cmc_bbdm.mavis.authority import MAVISAuthority


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _fixture(task: InspectionTask):
    rows, columns = np.indices((41, 43))
    full_scan = np.stack((rows, columns, rows + columns), axis=2).astype(np.uint8)
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("sample",),
        dataset_ids=("target",),
        images=(full_scan,),
        targets=np.asarray([0.4]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    surface = np.zeros((1, 1, 3), dtype=np.uint8)
    grid = build_deployment_grid(full_scan.shape[:2])
    hypothesis = SurfaceHypothesis(
        scores=np.linspace(0.0, 1.0, 64),
        top_cells=tuple(range(63, 55, -1)),
        border_median_rgb=np.zeros(3),
        state_sha256=_sha("surface-hypothesis"),
    )
    world = CausalInspectionWorld(
        authority,
        specimen_id="sample",
        task=task,
        surface_rgb=surface,
        surface_sha256=_sha("surface"),
        grid=grid,
        endpoint_budget=0.25,
    )
    prior = SourceBackgroundPrior(
        outer_domain="target",
        source_domains=("s1", "s2", "s3", "s4", "s5"),
        fit_specimen_ids=("a", "b", "c", "d", "e"),
        source_authority_sha256=_sha("authority"),
        domain_border_medians=np.zeros((5, 3)),
        background_rgb=np.zeros(3, dtype=np.uint8),
    )
    return world, grid, hypothesis, prior, full_scan


class _Encoder:
    def encode(self, images: object) -> np.ndarray:
        return np.zeros((len(tuple(images)), 512), dtype=np.float32)


class _Assessor:
    model_state_sha256 = _sha("assessor")

    def predict(self, embeddings: object, scalars: object) -> np.ndarray:
        del scalars
        return np.full(len(np.asarray(embeddings)), 0.25, dtype=np.float64)


def test_fixed_baselines_preserve_the_exact_k8_prefix_and_positive_cost() -> None:
    _world, grid, hypothesis, _prior, _scan = _fixture(InspectionTask.FIELD)
    for method in FIXED_BASELINE_METHODS:
        actions = plan_g1_fixed_actions(
            grid,
            hypothesis,
            surface_sha256=_sha("surface"),
            specimen_sha256=_sha("sample"),
            method=method,
            random_seed=2026090101,
            endpoint_budget=0.25,
        )
        assert tuple(action.cell_index for action in actions[:8]) == (
            PRIMARY_WARM_START_CELLS
        )
        state = zero_state(grid)
        for action in actions:
            assert len(action_added_positions(grid, state, action)) > 0
            state = apply_action(grid, state, action)
        assert not any(
            len(action_added_positions(grid, state, action)) > 0
            for action in fitting_actions(grid, state, 0.25)
        )


def test_observable_state_builder_uses_current_reconstruction_only() -> None:
    world, grid, hypothesis, prior, _scan = _fixture(InspectionTask.CAI)
    observation = world.replay(
        tuple(
            action
            for action in plan_g1_fixed_actions(
                grid,
                hypothesis,
                surface_sha256=_sha("surface"),
                specimen_sha256=_sha("sample"),
                method="ZERO_UNIFORM",
                random_seed=2026090101,
                endpoint_budget=0.25,
            )[:8]
        )
    )
    builder = G1ObservableStateBuilder(
        grid=grid,
        surface_hypothesis=hypothesis,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
    )
    state = builder(observation)
    assert state.observation_sha256 == observation.state_sha256
    assert state.task is InspectionTask.CAI
    assert state.global_scalars[12] == 0.25
    assert state.global_scalars[13] == 1.0
    assert np.count_nonzero(state.legal_action_mask) > 0


def test_action_history_evaluation_builds_same_geometry_field_and_cai_curves() -> None:
    for task in (InspectionTask.FIELD, InspectionTask.CAI):
        world, grid, hypothesis, prior, full_scan = _fixture(task)
        actions = plan_g1_fixed_actions(
            grid,
            hypothesis,
            surface_sha256=_sha("surface"),
            specimen_sha256=_sha("sample"),
            method="ZERO_UNIFORM",
            random_seed=2026090101,
            endpoint_budget=0.25,
        )
        curve = evaluate_g1_action_history(
            world,
            grid,
            prior,
            assessor=_Assessor(),
            encoder=_Encoder(),
            method="ZERO_UNIFORM",
            target_domain="target",
            specimen_sha256=_sha("sample"),
            actions=actions,
            full_scan=full_scan,
            true_cai=0.4,
        )
        assert curve.task is task
        assert tuple(curve.nominal_budgets) == (0.0, 0.0625, 0.125, 0.1875, 0.25)
        assert np.all(np.diff(curve.exact_budgets) >= 0.0)
        assert np.isfinite(curve.auebc)
        assert curve.warm_start_sha256 == world.replay(actions[:8]).state_sha256


def test_evaluation_oracles_start_from_k8_and_reach_the_same_endpoint() -> None:
    for task in (InspectionTask.FIELD, InspectionTask.CAI):
        world, grid, hypothesis, prior, full_scan = _fixture(task)
        actions = run_g1_warm_started_oracle_actions(
            world,
            grid,
            prior,
            surface_hypothesis=hypothesis,
            full_scan=full_scan,
            true_cai=0.4,
            assessor=_Assessor(),
            encoder=_Encoder(),
        )
        assert tuple(action.cell_index for action in actions[:8]) == (
            PRIMARY_WARM_START_CELLS
        )
        final = world.replay(actions)
        assert not any(
            len(action_added_positions(grid, final.measurement_state, action)) > 0
            for action in fitting_actions(grid, final.measurement_state, 0.25)
        )
        assert final.effective_budget <= 0.25
