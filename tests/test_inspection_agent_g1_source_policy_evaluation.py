from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.formal import FIXED_BASELINE_METHODS
from cmc_bbdm.inspection_agent_g1.metrics import build_engineering_curve
from cmc_bbdm.inspection_agent_g1.policy_training import (
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
)
from cmc_bbdm.inspection_agent_g1.rollout import ObservablePolicyScores
from cmc_bbdm.inspection_agent_g1.source_bridge import G1SourceBridgeRecord
from cmc_bbdm.inspection_agent_g1.source_policy_evaluation import (
    G1LearnedSourceRecord,
    G1SourcePolicyEvaluationError,
    evaluate_inner_policy_bridge,
    materialize_learned_source_for_world,
    read_learned_source_bank,
    write_learned_source_bank,
)
from cmc_bbdm.inspection_agent_g1.teacher import authorize_source_teacher
from cmc_bbdm.inspection_agent_g1.warm_start import build_deployment_grid
from cmc_bbdm.mavis.authority import MAVISAuthority

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _curve(
    source: str,
    task: InspectionTask,
    method: str,
    loss: float,
):
    specimen = f"{source}-specimen"
    return build_engineering_curve(
        method=method,
        target_domain=source,
        specimen_sha256=_sha(specimen),
        task=task,
        grid_sha256=_sha(f"grid-{source}"),
        evaluator_sha256=_sha(f"evaluator-{source}-{task.value}"),
        warm_start_sha256=_sha(f"warm-{source}-{task.value}"),
        state_budgets=(0.0, 0.05, 0.1, 0.18, 0.24),
        state_losses=(loss,) * 5,
        state_sha256=tuple(
            _sha(f"state-{source}-{task.value}-{method}-{index}")
            for index in range(5)
        ),
    )


def _fit_domains(source: str) -> tuple[str, ...]:
    return tuple(domain for domain in DOMAINS if domain not in {"d6", source})


def _hyperparameters() -> PolicyTrainingHyperparameters:
    return PolicyTrainingHyperparameters(
        model_name=PolicyModelName.STRUCTURED_INSPECTION_POLICY,
        route=TrainingRoute.HARD_BC,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=None,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=0,
    )


def _learned(task: InspectionTask) -> G1LearnedSourceRecord:
    hp = _hyperparameters()
    return G1LearnedSourceRecord(
        outer_target="d6",
        source_domain="d1",
        specimen_id="d1-specimen",
        fit_domains=("d2", "d3", "d4", "d5"),
        dependency_sha256=_sha("dependencies-d1"),
        hyperparameters_sha256=hp.state_sha256,
        model_state_sha256=_sha("model-d1"),
        fit_audit_sha256=_sha("fit-audit-d1"),
        selected_epoch=7,
        trajectory_sha256=_sha(f"trajectory-{task.value}"),
        action_history_sha256=_sha(f"actions-{task.value}"),
        curve=_curve("d1", task, "LEARNED_POLICY", 0.8),
    )


def _bridges() -> tuple[G1SourceBridgeRecord, ...]:
    output = []
    for source in DOMAINS[:-1]:
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            for index, method in enumerate(FIXED_BASELINE_METHODS):
                output.append(
                    G1SourceBridgeRecord(
                        outer_target="d6",
                        source_domain=source,
                        specimen_id=f"{source}-specimen",
                        fit_domains=_fit_domains(source),
                        dependency_sha256=_sha(f"dependencies-{source}"),
                        action_history_sha256=_sha(
                            f"actions-{source}-{task.value}-{method}"
                        ),
                        curve=_curve(source, task, method, 1.0 + index),
                    )
                )
            output.append(
                G1SourceBridgeRecord(
                    outer_target="d6",
                    source_domain=source,
                    specimen_id=f"{source}-specimen",
                    fit_domains=_fit_domains(source),
                    dependency_sha256=_sha(f"dependencies-{source}"),
                    action_history_sha256=_sha(
                        f"actions-{source}-{task.value}-oracle"
                    ),
                    curve=_curve(source, task, f"ORACLE_{task.value}", 0.5),
                )
            )
    return tuple(output)


def test_learned_source_bank_round_trips_per_specimen_curves(tmp_path: Path) -> None:
    records = (_learned(InspectionTask.FIELD), _learned(InspectionTask.CAI))
    path = tmp_path / "learned.parquet"

    identity = write_learned_source_bank(path, records)
    replay_identity, replay = read_learned_source_bank(path)

    assert replay_identity == identity
    assert tuple(row.state_sha256 for row in replay) == tuple(
        row.state_sha256
        for row in sorted(records, key=lambda row: row.curve.task.value)
    )
    assert all(np.isfinite(row.curve.auebc) for row in replay)


def test_inner_policy_bridge_uses_same_geometry_fixed_and_oracle_curves() -> None:
    evaluation = evaluate_inner_policy_bridge(
        (_learned(InspectionTask.FIELD), _learned(InspectionTask.CAI)),
        _bridges(),
        hyperparameters=_hyperparameters(),
    )

    assert evaluation.validation_domain == "d1"
    assert {row.task for row in evaluation.metrics} == {
        InspectionTask.FIELD,
        InspectionTask.CAI,
    }
    assert all(row.learned_auebc == 0.2 for row in evaluation.metrics)
    assert all(row.fixed_auebc == 0.25 for row in evaluation.metrics)
    assert all(row.oracle_auebc == 0.125 for row in evaluation.metrics)
    assert all(
        row.oracle_gap_closure == pytest.approx(0.4) for row in evaluation.metrics
    )
    assert all(selection.method == "RANDOM" for selection in evaluation.fixed_selections)


def test_inner_policy_bridge_rejects_other_dependency_bundle() -> None:
    learned = tuple(
        replace(
            _learned(task),
            dependency_sha256=_sha("other-dependencies"),
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    )

    with pytest.raises(G1SourcePolicyEvaluationError, match="dependency"):
        evaluate_inner_policy_bridge(
            learned,
            _bridges(),
            hyperparameters=_hyperparameters(),
        )


class _Assessor:
    outer_domain = "d6"
    fit_domains = ("d2", "d3", "d4", "d5")
    model_state_sha256 = _sha("assessor")

    def predict(self, embeddings: object, scalars: object) -> np.ndarray:
        del scalars
        return np.full(len(np.asarray(embeddings)), 0.4, dtype=np.float64)


class _Encoder:
    def encode(self, images: object) -> np.ndarray:
        return np.zeros((len(tuple(images)), 512), dtype=np.float64)


class _Actor:
    model_state_sha256 = _sha("actor")
    hyperparameters = _hyperparameters()
    audit = type(
        "Audit",
        (),
        {
            "outer_target": "d6",
            "validation_domain": "d1",
            "fit_domains": ("d2", "d3", "d4", "d5"),
            "selected_epoch": 7,
            "state_sha256": _sha("actor-fit-audit"),
        },
    )()

    def __call__(self, state: object) -> ObservablePolicyScores:
        logits = np.full(192, -np.inf, dtype=np.float64)
        logits[np.flatnonzero(state.legal_action_mask)] = 0.0
        return ObservablePolicyScores(
            policy_state_sha256=state.state_sha256,
            model_sha256=self.model_state_sha256,
            action_logits=logits,
            stop_probability=0.0,
        )


def _world(task: InspectionTask):
    rows, columns = np.indices((41, 43))
    full_scan = np.stack((3 * rows, 4 * columns, rows + columns), axis=2).astype(
        np.uint8
    )
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("d1-specimen",),
        dataset_ids=("d1",),
        images=(full_scan,),
        targets=np.asarray([0.4]),
        metadata13=np.zeros((1, 13)),
        profile_stats21=np.zeros((1, 21)),
    )
    grid = build_deployment_grid(full_scan.shape[:2])
    world = CausalInspectionWorld(
        authority,
        specimen_id="d1-specimen",
        task=task,
        surface_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        surface_sha256=_sha("surface"),
        grid=grid,
        endpoint_budget=0.25,
    )
    hypothesis = SurfaceHypothesis(
        scores=np.linspace(0.0, 1.0, 64),
        top_cells=tuple(range(63, 55, -1)),
        border_median_rgb=np.zeros(3),
        state_sha256=_sha("hypothesis"),
    )
    prior = SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=("d2", "d3", "d4", "d5"),
        fit_specimen_ids=("s2", "s3", "s4", "s5"),
        source_authority_sha256=_sha("source-authority"),
        domain_border_medians=np.zeros((4, 3)),
        background_rgb=np.zeros(3, dtype=np.uint8),
    )
    authorization = authorize_source_teacher(
        build_crossfit_roster(DOMAINS, outer_target="d6", labeled_domain="d1"),
        query_domain="d1",
    )
    return world, grid, hypothesis, prior, authorization, full_scan


@pytest.mark.parametrize("task", (InspectionTask.FIELD, InspectionTask.CAI))
def test_learned_source_world_rolls_out_observable_actor_before_evaluation(
    task: InspectionTask,
) -> None:
    world, grid, hypothesis, prior, authorization, full_scan = _world(task)

    record = materialize_learned_source_for_world(
        world,
        grid,
        hypothesis,
        prior,
        authorization,
        assessor=_Assessor(),
        encoder=_Encoder(),
        actor=_Actor(),
        outer_target="d6",
        source_domain="d1",
        specimen_id="d1-specimen",
        specimen_sha256=_sha("d1-specimen"),
        dependency_sha256=_sha("dependencies-d1"),
        full_scan=full_scan,
        true_cai=0.4,
    )

    assert record.curve.task is task
    assert record.curve.method == "LEARNED_POLICY"
    assert record.fit_domains == ("d2", "d3", "d4", "d5")
    assert record.hyperparameters_sha256 == _hyperparameters().state_sha256
    assert record.model_state_sha256 == _sha("actor")
    assert np.isfinite(record.curve.auebc)
