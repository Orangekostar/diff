from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.dagger import (
    DAGGER_MAX_NEW_STATES,
    trajectory_quantile_indices,
)
from cmc_bbdm.inspection_agent_g1.dagger_execution import (
    materialize_g1_dagger_relabels_for_world,
)
from cmc_bbdm.inspection_agent_g1.rollout import ObservablePolicyScores
from cmc_bbdm.inspection_agent_g1.teacher import authorize_source_teacher
from cmc_bbdm.inspection_agent_g1.warm_start import build_deployment_grid
from cmc_bbdm.mavis.authority import MAVISAuthority

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _Assessor:
    outer_domain = "d6"
    fit_domains = ("d2", "d3", "d4", "d5")
    model_state_sha256 = _sha("assessor")

    def predict(self, embeddings: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        assert len(embeddings) == len(scalars)
        return np.full(len(embeddings), 0.4, dtype=np.float64)


class _Encoder:
    def encode(self, images: object) -> np.ndarray:
        count = len(tuple(images))
        return np.zeros((count, 512), dtype=np.float64)


class _Actor:
    model_state_sha256 = _sha("actor")
    audit = SimpleNamespace(
        outer_target="d6",
        validation_domain="d1",
        fit_domains=("d2", "d3", "d4", "d5"),
    )

    def __init__(
        self,
        *,
        cai_context_mode: CAIContextMode = (
            CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT
        ),
    ) -> None:
        self.hyperparameters = SimpleNamespace(
            cai_context_mode=cai_context_mode,
            task_token_mode=TaskTokenMode.CORRECT,
            dagger_iterations=0,
        )
        self.seen_context_modes: list[CAIContextMode] = []

    def __call__(self, state: object) -> ObservablePolicyScores:
        self.seen_context_modes.append(state.cai_context_mode)
        logits = np.full(192, -np.inf, dtype=np.float64)
        logits[np.flatnonzero(state.legal_action_mask)] = 0.0
        return ObservablePolicyScores(
            policy_state_sha256=state.state_sha256,
            model_sha256=self.model_state_sha256,
            action_logits=logits,
            stop_probability=0.0,
        )


def _fixture():
    rows, columns = np.indices((41, 43))
    image = np.stack((3 * rows, 4 * columns, rows + columns), axis=2).astype(np.uint8)
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("d1-sample",),
        dataset_ids=("d1",),
        images=(image,),
        targets=np.asarray([0.4]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    surface_rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    grid = build_deployment_grid(image.shape[:2])
    world = CausalInspectionWorld(
        authority,
        specimen_id="d1-sample",
        task=InspectionTask.FIELD,
        surface_rgb=surface_rgb,
        surface_sha256=_sha("surface-rgb"),
        grid=grid,
        endpoint_budget=0.25,
    )
    scores = np.linspace(0.0, 1.0, 64)
    scores.setflags(write=False)
    median = np.zeros(3, dtype=np.float64)
    median.setflags(write=False)
    surface = SurfaceHypothesis(
        scores=scores,
        top_cells=tuple(range(63, 55, -1)),
        border_median_rgb=median,
        state_sha256=_sha("surface-hypothesis"),
    )
    prior = SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=("d2", "d3", "d4", "d5"),
        fit_specimen_ids=("s2", "s3", "s4", "s5"),
        source_authority_sha256=_sha("source-authority"),
        domain_border_medians=np.full((4, 3), 90.0),
        background_rgb=np.full(3, 90, dtype=np.uint8),
    )
    authorization = authorize_source_teacher(
        build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1"),
        query_domain="d1",
    )
    return world, grid, surface, prior, authorization, image


def test_dagger_world_relabels_only_quantile_capped_source_states() -> None:
    world, grid, surface, prior, authorization, image = _fixture()
    batch = materialize_g1_dagger_relabels_for_world(
        world,
        grid,
        surface,
        prior,
        authorization,
        assessor=_Assessor(),
        encoder=_Encoder(),
        actor=_Actor(),
        outer_target="d6",
        source_domain="d1",
        specimen_sha256=_sha("d1-sample"),
        iteration=1,
        full_scan=image,
        true_cai=0.4,
    )

    assert len(batch.records) <= DAGGER_MAX_NEW_STATES
    assert len(batch.records) == len(batch.visited_states)
    assert tuple(row.trajectory_index for row in batch.visited_states) == (
        trajectory_quantile_indices(batch.trajectory_length)
    )
    assert all(row.example.dagger_iteration == 1 for row in batch.records)
    assert all(row.example.source_domain == "d1" for row in batch.records)
    assert all(row.example.outer_target == "d6" for row in batch.records)
    assert all(
        row.source_state_sha256 == visited.state_sha256
        for row, visited in zip(batch.records, batch.visited_states, strict=True)
    )


def test_dagger_rolls_out_actor_mode_but_persists_canonical_states() -> None:
    world, grid, surface, prior, authorization, image = _fixture()
    actor = _Actor(cai_context_mode=CAIContextMode.TASK_SPECIFIC_MASKED)

    batch = materialize_g1_dagger_relabels_for_world(
        world,
        grid,
        surface,
        prior,
        authorization,
        assessor=_Assessor(),
        encoder=_Encoder(),
        actor=actor,
        outer_target="d6",
        source_domain="d1",
        specimen_sha256=_sha("d1-sample"),
        iteration=1,
        full_scan=image,
        true_cai=0.4,
    )

    assert set(actor.seen_context_modes) == {CAIContextMode.TASK_SPECIFIC_MASKED}
    assert all(
        row.example.policy_state.cai_context_mode
        is CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT
        for row in batch.records
    )
    assert all(
        row.example.policy_state.task_token_mode is TaskTokenMode.CORRECT
        for row in batch.records
    )
