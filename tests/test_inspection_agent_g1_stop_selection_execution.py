from __future__ import annotations

import hashlib
from types import MappingProxyType, SimpleNamespace

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.crossfit import build_crossfit_roster
from cmc_bbdm.inspection_agent_g1.dagger_selection_execution import (
    G1OuterDaggerSelectionRun,
)
from cmc_bbdm.inspection_agent_g1.features import canonical_slot
from cmc_bbdm.inspection_agent_g1.formal import plan_g1_fixed_actions
from cmc_bbdm.inspection_agent_g1.g1 import (
    G1Protocol,
    G1Runtime,
    G1RuntimeSurface,
    G1SourceDependencies,
)
from cmc_bbdm.inspection_agent_g1.policy_training import (
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainedObservablePolicy,
    TrainingRoute,
)
from cmc_bbdm.inspection_agent_g1.rollout import (
    ClosedLoopTrajectory,
    ObservablePolicyScores,
    RolloutStep,
)
from cmc_bbdm.inspection_agent_g1.stop_execution import G1FixedEndpointRecord
from cmc_bbdm.inspection_agent_g1.stop_selection_execution import (
    evaluate_source_stop_validation_trajectory,
    materialize_g1_source_stop_validation_trajectories,
    run_outer_stop_selection,
)
from cmc_bbdm.inspection_agent_g1.stop_training import TrainedObservableStopPolicy
from cmc_bbdm.inspection_agent_g1.stopping_policy import (
    SourceStopValidationTrajectory,
)
from cmc_bbdm.inspection_agent_g1.teacher import authorize_source_teacher
from cmc_bbdm.inspection_agent_g1.warm_start import build_deployment_grid
from cmc_bbdm.mavis.authority import MAVISAuthority

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


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


def _source_runtime() -> G1Runtime:
    specimen_ids = ("d1-a", "d1-b", "d2-a", "d3-a", "d4-a", "d5-a", "d6-a")
    dataset_ids = ("d1", "d1", "d2", "d3", "d4", "d5", "d6")
    images = []
    surfaces = {}
    for index, (specimen, domain) in enumerate(
        zip(specimen_ids, dataset_ids, strict=True)
    ):
        rows, columns = np.indices((41, 43))
        image = np.stack(
            (rows + index, columns + index, rows + columns + index),
            axis=2,
        ).astype(np.uint8)
        images.append(image)
        hypothesis = SurfaceHypothesis(
            scores=np.linspace(0.0, 1.0, 64),
            top_cells=tuple(range(63, 55, -1)),
            border_median_rgb=np.zeros(3),
            state_sha256=_sha(f"hypothesis-{specimen}"),
        )
        surfaces[(domain, specimen)] = G1RuntimeSurface(
            dataset_id=domain,
            specimen_id=specimen,
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            surface_sha256=_sha(f"surface-{specimen}"),
            hypothesis=hypothesis,
        )
    authority = MAVISAuthority.from_arrays(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        images=tuple(images),
        targets=np.linspace(0.2, 0.5, len(images)),
        metadata13=np.zeros((len(images), 13)),
        profile_stats21=np.zeros((len(images), 21)),
    )
    return G1Runtime(
        mavis=authority,
        surfaces=MappingProxyType(surfaces),
        surface_authority_sha256=_sha("surface-authority"),
    )


def _fixed_records(runtime: G1Runtime) -> tuple[G1FixedEndpointRecord, ...]:
    output = []
    for specimen, source in zip(
        runtime.mavis.specimen_ids,
        runtime.mavis.dataset_ids,
        strict=True,
    ):
        if source == "d6":
            continue
        fit_domains = tuple(
            domain for domain in DOMAINS if domain not in {"d6", source}
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            for method in (
                "RANDOM",
                "ZERO_UNIFORM",
                "CENTER_FIRST",
                "SURFACE_FOCUS",
                "SURVEY_THEN_REFINE_FIXED",
            ):
                output.append(
                    G1FixedEndpointRecord(
                        outer_target="d6",
                        source_domain=source,
                        specimen_id=specimen,
                        specimen_sha256=runtime.specimen_sha256(source, specimen),
                        task=task,
                        method=method,
                        fit_domains=fit_domains,
                        dependency_sha256=_sha(f"dependency-{source}"),
                        action_history_sha256=_sha(
                            f"actions-{source}-{specimen}-{task.value}-{method}"
                        ),
                        endpoint_observation_sha256=_sha(
                            f"endpoint-{source}-{specimen}-{task.value}-{method}"
                        ),
                        effective_budget=0.25,
                        exact_acquired_count=100,
                        native_count=400,
                        task_loss=0.1 if method == "RANDOM" else 1.0,
                    )
                )
    return tuple(output)


def test_source_stop_validation_materialization_is_crossfit_and_batched(
    monkeypatch,
) -> None:
    runtime = _source_runtime()
    protocol = object.__new__(G1Protocol)
    object.__setattr__(protocol, "domain_order", DOMAINS)
    object.__setattr__(protocol, "domain_counts", MappingProxyType({"d1": 2}))
    object.__setattr__(protocol, "endpoint_budget", 0.25)
    roster = build_crossfit_roster(
        DOMAINS,
        outer_target="d6",
        labeled_domain="d1",
    )
    prior = SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=roster.fit_domains,
        fit_specimen_ids=("d2-a", "d3-a", "d4-a", "d5-a"),
        source_authority_sha256=_sha("authority"),
        domain_border_medians=np.zeros((4, 3)),
        background_rgb=np.zeros(3, dtype=np.uint8),
    )
    dependencies = object.__new__(G1SourceDependencies)
    object.__setattr__(dependencies, "roster", roster)
    object.__setattr__(dependencies, "prior_fit", SimpleNamespace(prior=prior))
    object.__setattr__(
        dependencies,
        "assessor_fit",
        SimpleNamespace(assessor=_Assessor()),
    )
    object.__setattr__(
        dependencies,
        "authorization",
        authorize_source_teacher(roster, query_domain="d1"),
    )
    object.__setattr__(dependencies, "state_sha256", _sha("dependency-d1"))
    hyperparameters = PolicyTrainingHyperparameters(
        model_name=PolicyModelName.SHARED_ACTION_MLP,
        route=TrainingRoute.HARD_BC,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=None,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=0,
    )
    action_policy = object.__new__(TrainedObservablePolicy)
    object.__setattr__(action_policy, "hyperparameters", hyperparameters)
    object.__setattr__(
        action_policy,
        "audit",
        SimpleNamespace(
            outer_target="d6",
            validation_domain="d1",
            fit_domains=roster.fit_domains,
        ),
    )
    object.__setattr__(action_policy, "model_state_sha256", _sha("action-model"))
    stop_policy = object.__new__(TrainedObservableStopPolicy)
    object.__setattr__(
        stop_policy,
        "audit",
        SimpleNamespace(
            outer_target="d6",
            validation_domain="d1",
            fit_domains=roster.fit_domains,
        ),
    )
    object.__setattr__(
        stop_policy,
        "base_action_model_sha256",
        action_policy.model_state_sha256,
    )
    object.__setattr__(stop_policy, "model_state_sha256", _sha("stop-model"))

    def fake_scores(_policy, states):
        output = []
        for state in states:
            logits = np.full(192, -np.inf, dtype=np.float64)
            logits[state.legal_action_mask] = 0.0
            logits[int(np.flatnonzero(state.legal_action_mask)[0])] = 1.0
            output.append(
                ObservablePolicyScores(
                    policy_state_sha256=state.state_sha256,
                    model_sha256=_sha("stop-model"),
                    action_logits=logits,
                    stop_probability=min(0.99, 0.1 + state.global_scalars[0]),
                )
            )
        return tuple(output)

    monkeypatch.setattr(TrainedObservableStopPolicy, "score_batch", fake_scores)
    results = materialize_g1_source_stop_validation_trajectories(
        runtime,
        protocol,
        dependencies,
        encoder=_Encoder(),
        action_policy=action_policy,
        stop_policy=stop_policy,
        fixed_endpoint_records=_fixed_records(runtime),
    )

    assert len(results) == 4
    assert {row.source_domain for row in results} == {"d1"}
    assert {row.task for row in results} == {
        InspectionTask.FIELD,
        InspectionTask.CAI,
    }
    assert len({row.specimen_sha256 for row in results}) == 2
    assert all(row.reference_true_loss == 0.1 for row in results)


def test_outer_stop_selection_freezes_all_five_sources_before_target(
    monkeypatch,
    tmp_path,
) -> None:
    from cmc_bbdm.inspection_agent_g1 import stop_selection_execution as module

    hyperparameters = PolicyTrainingHyperparameters(
        model_name=PolicyModelName.SHARED_ACTION_MLP,
        route=TrainingRoute.HARD_BC,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=None,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=0,
    )
    selected_records = tuple(
        SimpleNamespace(
            example=SimpleNamespace(
                outer_target="d6",
                source_domain=source,
                dagger_iteration=0,
            )
        )
        for source in DOMAINS[:-1]
    )
    action_selection = object.__new__(G1OuterDaggerSelectionRun)
    object.__setattr__(action_selection, "outer_target", "d6")
    object.__setattr__(
        action_selection,
        "selection",
        SimpleNamespace(
            selected_hyperparameters_sha256=hyperparameters.state_sha256,
            final_refit_epochs=2,
            state_sha256=_sha("action-selection"),
            source_validation_domains=DOMAINS[:-1],
        ),
    )
    object.__setattr__(
        action_selection,
        "candidates",
        (
            SimpleNamespace(
                candidate=SimpleNamespace(hyperparameters=hyperparameters)
            ),
        ),
    )
    object.__setattr__(
        action_selection,
        "dagger_build",
        SimpleNamespace(records=selected_records),
    )
    object.__setattr__(
        action_selection,
        "aawr_authorization",
        SimpleNamespace(
            status="NOT_RUN_NOT_AUTHORIZED",
            state_sha256=_sha("aawr"),
        ),
    )
    protocol = object.__new__(G1Protocol)
    object.__setattr__(protocol, "domain_order", DOMAINS)
    object.__setattr__(protocol, "epochs", 80)
    object.__setattr__(protocol, "patience", 12)
    runtime = object.__new__(G1Runtime)
    object.__setattr__(runtime, "mavis", SimpleNamespace(dataset_ids=DOMAINS))
    object.__setattr__(runtime, "surfaces", MappingProxyType({}))
    object.__setattr__(runtime, "surface_authority_sha256", _sha("surface"))
    object.__setattr__(runtime, "state_sha256", _sha("runtime"))

    monkeypatch.setattr(
        module,
        "read_teacher_bank",
        lambda path: (
            SimpleNamespace(manifest_sha256=_sha(f"teacher-{path.parent.name}-{path.stem}")),
            (
                SimpleNamespace(
                    example=SimpleNamespace(
                        outer_target=path.parent.name,
                        source_domain=path.stem,
                    )
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "read_stop_bank",
        lambda path: (
            SimpleNamespace(manifest_sha256=_sha(f"stop-{path.parent.name}-{path.stem}")),
            (
                SimpleNamespace(
                    outer_target=path.parent.name,
                    source_domain=path.stem,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "read_g1_outer_fixed_endpoint_records",
        lambda *_args, **_kwargs: (SimpleNamespace(),),
    )
    monkeypatch.setattr(
        module,
        "rebind_training_example_modes",
        lambda example, **_kwargs: example,
    )
    inner_sources = []

    def fake_action_fit(_examples, *, validation_domain, **_kwargs):
        inner_sources.append(validation_domain)
        actor = object.__new__(TrainedObservablePolicy)
        object.__setattr__(actor, "hyperparameters", hyperparameters)
        object.__setattr__(
            actor,
            "audit",
            SimpleNamespace(
                outer_target="d6",
                validation_domain=validation_domain,
                fit_domains=tuple(
                    domain for domain in DOMAINS[:-1] if domain != validation_domain
                ),
            ),
        )
        object.__setattr__(actor, "model_state_sha256", _sha(f"action-{validation_domain}"))
        return actor

    monkeypatch.setattr(module, "fit_inner_observable_policy", fake_action_fit)
    monkeypatch.setattr(
        module,
        "join_g1_stop_training_examples",
        lambda *_args, **_kwargs: (SimpleNamespace(),),
    )

    def fake_stop_fit(action_policy, _examples, *, validation_domain, **_kwargs):
        actor = object.__new__(TrainedObservableStopPolicy)
        object.__setattr__(
            actor,
            "audit",
            SimpleNamespace(
                outer_target="d6",
                validation_domain=validation_domain,
                fit_domains=action_policy.audit.fit_domains,
                selected_epoch=DOMAINS.index(validation_domain) + 1,
                state_sha256=_sha(f"stop-audit-{validation_domain}"),
            ),
        )
        object.__setattr__(
            actor,
            "base_action_model_sha256",
            action_policy.model_state_sha256,
        )
        object.__setattr__(actor, "model_state_sha256", _sha(f"stop-{validation_domain}"))
        return actor

    monkeypatch.setattr(module, "fit_inner_observable_stop_head", fake_stop_fit)

    def fake_dependencies(_runtime, _protocol, *, labeled_domain, **_kwargs):
        value = object.__new__(G1SourceDependencies)
        object.__setattr__(
            value,
            "roster",
            build_crossfit_roster(
                DOMAINS,
                outer_target="d6",
                labeled_domain=labeled_domain,
            ),
        )
        return value

    monkeypatch.setattr(module, "build_g1_source_dependencies", fake_dependencies)

    def fake_trajectories(
        _runtime,
        _protocol,
        dependencies,
        **_kwargs,
    ):
        source = dependencies.roster.labeled_domain
        return tuple(
            SourceStopValidationTrajectory(
                outer_target="d6",
                source_domain=source,
                specimen_sha256=_sha(f"specimen-{source}-{task.value}"),
                task=task,
                budgets=(0.05, 0.25),
                stop_probabilities=(0.9, 0.0),
                true_task_losses=(0.9, 1.0),
                reference_true_loss=1.0,
                endpoint_budget=0.25,
            )
            for task in (InspectionTask.FIELD, InspectionTask.CAI)
        )

    monkeypatch.setattr(
        module,
        "materialize_g1_source_stop_validation_trajectories",
        fake_trajectories,
    )
    final_action = object.__new__(TrainedObservablePolicy)
    object.__setattr__(final_action, "hyperparameters", hyperparameters)
    object.__setattr__(
        final_action,
        "audit",
        SimpleNamespace(
            outer_target="d6",
            validation_domain=None,
            fit_domains=DOMAINS[:-1],
            state_sha256=_sha("final-action-audit"),
        ),
    )
    object.__setattr__(final_action, "model_state_sha256", _sha("final-action"))
    monkeypatch.setattr(
        module,
        "fit_final_observable_policy",
        lambda *_args, **_kwargs: final_action,
    )
    final_stop = object.__new__(TrainedObservableStopPolicy)
    object.__setattr__(
        final_stop,
        "audit",
        SimpleNamespace(
            outer_target="d6",
            validation_domain=None,
            fit_domains=DOMAINS[:-1],
            selected_epoch=3,
            state_sha256=_sha("final-stop-audit"),
        ),
    )
    object.__setattr__(
        final_stop,
        "base_action_model_sha256",
        final_action.model_state_sha256,
    )
    object.__setattr__(final_stop, "model_state_sha256", _sha("final-stop"))
    final_stop_epochs = []

    def fake_final_stop(_action, _examples, *, selected_epochs, **_kwargs):
        final_stop_epochs.append(selected_epochs)
        return final_stop

    monkeypatch.setattr(module, "fit_final_observable_stop_head", fake_final_stop)

    result = run_outer_stop_selection(
        runtime,
        protocol,
        outer_target="d6",
        action_selection=action_selection,
        encoder=_Encoder(),
        teacher_bank_root=tmp_path / "teacher",
        stop_bank_root=tmp_path / "stop",
        fixed_endpoint_root=tmp_path / "fixed",
        work_root=tmp_path / "selection",
        device="cpu",
    )

    assert tuple(inner_sources) == DOMAINS[:-1]
    assert final_stop_epochs == [3]
    assert result.action_policy is final_action
    assert result.stop_policy is final_stop
    assert tuple(row.task for row in result.thresholds) == (
        InspectionTask.FIELD,
        InspectionTask.CAI,
    )
    assert tuple(row.threshold for row in result.thresholds) == (0.9, 0.9)
    assert result.target_outcomes_opened is False
    payload = result.path.read_text(encoding="ascii")
    assert f'"state_sha256":"{result.state_sha256}"' in payload
    assert payload.endswith("\n")
