from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.batch_rollout import (
    G1BatchRolloutRequest,
    run_g1_closed_loop_batch,
)
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.formal import G1ObservableStateBuilder
from cmc_bbdm.inspection_agent_g1.rollout import (
    ObservablePolicyScores,
    run_closed_loop,
)
from cmc_bbdm.inspection_agent_g1.warm_start import build_deployment_grid
from cmc_bbdm.mavis.authority import MAVISAuthority


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _Encoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode(self, images: object) -> np.ndarray:
        values = tuple(images)
        self.batch_sizes.append(len(values))
        return np.zeros((len(values), 512), dtype=np.float64)


class _Assessor:
    model_state_sha256 = _sha("assessor")

    def predict(self, embeddings: object, scalars: object) -> np.ndarray:
        del scalars
        return np.full(len(np.asarray(embeddings)), 0.4, dtype=np.float64)


class _Actor:
    model_state_sha256 = _sha("actor")

    def __call__(self, state: object) -> ObservablePolicyScores:
        return self.score_batch((state,))[0]

    def score_batch(self, states: tuple[object, ...]) -> tuple[ObservablePolicyScores, ...]:
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


def _fixture():
    rows, columns = np.indices((41, 43))
    full_scan = np.stack((rows, columns, rows + columns), axis=2).astype(np.uint8)
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("sample",),
        dataset_ids=("source",),
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
    prior = SourceBackgroundPrior(
        outer_domain="target",
        source_domains=("s1", "s2", "s3", "s4"),
        fit_specimen_ids=("a", "b", "c", "d"),
        source_authority_sha256=_sha("authority"),
        domain_border_medians=np.zeros((4, 3)),
        background_rgb=np.zeros(3, dtype=np.uint8),
    )
    encoder = _Encoder()
    assessor = _Assessor()
    worlds = tuple(
        CausalInspectionWorld(
            authority,
            specimen_id="sample",
            task=task,
            surface_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            surface_sha256=_sha("surface"),
            grid=grid,
            endpoint_budget=0.25,
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    )
    builders = tuple(
        G1ObservableStateBuilder(
            grid=grid,
            surface_hypothesis=hypothesis,
            prior=prior,
            assessor=assessor,
            encoder=encoder,
            cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
            task_token_mode=TaskTokenMode.CORRECT,
        )
        for _world in worlds
    )
    return worlds, grid, builders, encoder


def test_batched_rollout_replays_the_same_trajectories_as_scalar_execution() -> None:
    worlds, grid, builders, encoder = _fixture()
    actor = _Actor()
    requests = tuple(
        G1BatchRolloutRequest(
            world=world,
            grid=grid,
            target_domain="source",
            specimen_sha256=_sha("sample"),
            state_builder=builder,
        )
        for world, builder in zip(worlds, builders, strict=True)
    )

    batched = run_g1_closed_loop_batch(
        requests,
        actor=actor,
        stop_threshold=None,
    )
    scalar = tuple(
        run_closed_loop(
            world,
            grid,
            target_domain="source",
            specimen_sha256=_sha("sample"),
            state_builder=builder,
            actor=actor,
            stop_threshold=None,
        )
        for world, builder in zip(worlds, builders, strict=True)
    )

    assert tuple(row.state_sha256 for row in batched) == tuple(
        row.state_sha256 for row in scalar
    )
    assert 2 in encoder.batch_sizes
