from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    SourceBackgroundPrior,
    reconstruct_observation,
)
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.contracts import (
    CAIContextMode,
    G1PolicyState,
    TaskTokenMode,
)
from cmc_bbdm.inspection_agent_g1.features import (
    build_policy_state,
    canonical_action_from_slot,
    canonical_slot,
    decision_type,
)
from cmc_bbdm.inspection_agent_g1.warm_start import (
    PRIMARY_WARM_START_CELLS,
    build_deployment_grid,
)
from cmc_bbdm.mavis.authority import MAVISAuthority


def _image() -> np.ndarray:
    rows, columns = np.indices((41, 43))
    return np.stack(
        (
            (3 * rows + columns) % 256,
            (rows + 5 * columns) % 256,
            (7 * rows + 2 * columns) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def _surface() -> tuple[np.ndarray, SurfaceHypothesis]:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    scores = np.linspace(0.0, 1.0, 64, dtype=np.float64)
    scores.setflags(write=False)
    median = np.zeros(3, dtype=np.float64)
    median.setflags(write=False)
    return image, SurfaceHypothesis(scores, tuple(range(63, 55, -1)), median, "b" * 64)


def _prior() -> SourceBackgroundPrior:
    return SourceBackgroundPrior(
        outer_domain="target",
        source_domains=("s1", "s2", "s3", "s4", "s5"),
        fit_specimen_ids=("a", "b", "c", "d", "e"),
        source_authority_sha256="a" * 64,
        domain_border_medians=np.full((5, 3), 100.0),
        background_rgb=np.full(3, 100, dtype=np.uint8),
    )


def _observation(task: InspectionTask = InspectionTask.FIELD):
    full_scan = _image()
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("sample",),
        dataset_ids=("target",),
        images=(full_scan,),
        targets=np.asarray([0.4]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    surface, hypothesis = _surface()
    grid = build_deployment_grid(full_scan.shape[:2])
    world = CausalInspectionWorld(
        authority,
        specimen_id="sample",
        task=task,
        surface_rgb=surface,
        surface_sha256=hashlib.sha256(surface.tobytes()).hexdigest(),
        grid=grid,
        endpoint_budget=0.25,
    )
    observation = world.reset()
    for cell in PRIMARY_WARM_START_CELLS:
        action = canonical_action_from_slot(cell)
        observation = world.step(observation, action)
    return observation, grid, hypothesis


def test_policy_state_has_fixed_observable_tensor_contract() -> None:
    observation, grid, hypothesis = _observation()
    prior = _prior()
    reconstruction = reconstruct_observation(observation, grid, prior)
    state = build_policy_state(
        observation,
        hypothesis,
        grid,
        prior,
        reconstruction,
        reconstruction_embedding=np.linspace(-1.0, 1.0, 512),
        cai_estimate=0.35,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
    )
    assert isinstance(state, G1PolicyState)
    assert state.reconstruction_embedding.shape == (512,)
    assert state.global_scalars.shape == (17,)
    assert state.task_token.shape == (2,)
    assert state.cell_features.shape == (64, 18)
    assert state.candidate_features.shape == (192, 12)
    assert state.legal_action_mask.shape == (192,)
    assert int(np.count_nonzero(state.legal_action_mask)) == 64
    assert not any(
        value.flags.writeable
        for value in (
            state.reconstruction_embedding,
            state.global_scalars,
            state.task_token,
            state.cell_features,
            state.candidate_features,
            state.legal_action_mask,
        )
    )
    assert len(state.state_sha256) == 64


def test_cell_tokens_use_only_currently_observed_native_values() -> None:
    observation, grid, hypothesis = _observation()
    prior = _prior()
    state = build_policy_state(
        observation,
        hypothesis,
        grid,
        prior,
        reconstruct_observation(observation, grid, prior),
        reconstruction_embedding=np.zeros(512),
        cai_estimate=None,
        cai_context_mode=CAIContextMode.TASK_SPECIFIC_MASKED,
        task_token_mode=TaskTokenMode.CORRECT,
    )
    measured = PRIMARY_WARM_START_CELLS[0]
    unmeasured = 1
    assert state.cell_features[measured, 16] == 1.0
    assert state.cell_features[unmeasured, 16] == 0.0
    np.testing.assert_array_equal(state.cell_features[unmeasured, 9:16], 0.0)
    assert state.cell_features[PRIMARY_WARM_START_CELLS[-1], 17] == 1.0
    assert np.count_nonzero(state.cell_features[:, 17]) == 1


def test_canonical_slots_and_decisions_cannot_contradict_primitives() -> None:
    observation, _grid, hypothesis = _observation()
    for slot in range(192):
        assert canonical_slot(canonical_action_from_slot(slot)) == slot
    measured_refine = canonical_action_from_slot(64 + PRIMARY_WARM_START_CELLS[0])
    unmeasured_focus = canonical_action_from_slot(62)
    unmeasured_broaden = canonical_action_from_slot(1)
    assert observation.measurement_state.levels[measured_refine.cell_index] == 0
    assert decision_type(measured_refine, hypothesis) is InspectionDecision.REFINE
    assert decision_type(unmeasured_focus, hypothesis) is InspectionDecision.FOCUS
    assert decision_type(unmeasured_broaden, hypothesis) is InspectionDecision.BROADEN
