from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.inspection_agent_g1 import dagger_orchestration as module
from cmc_bbdm.inspection_agent_g1.contracts import (
    ACTION_SLOT_COUNT,
    CANDIDATE_FEATURE_DIMENSION,
    CELL_COUNT,
    CELL_FEATURE_DIMENSION,
    GLOBAL_SCALAR_DIMENSION,
    RECONSTRUCTION_EMBEDDING_DIMENSION,
    CAIContextMode,
    G1PolicyState,
    TaskTokenMode,
)
from cmc_bbdm.inspection_agent_g1.dagger import DaggerVisitedState
from cmc_bbdm.inspection_agent_g1.dagger_execution import G1DaggerRelabelBatch
from cmc_bbdm.inspection_agent_g1.g1 import G1Protocol, G1Runtime, G1SourceDependencies
from cmc_bbdm.inspection_agent_g1.policy_training import (
    G1PolicyTrainingExample,
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
)
from cmc_bbdm.inspection_agent_g1.teacher import (
    PrivilegedTeacherLabel,
    TeacherCandidateRecord,
)
from cmc_bbdm.inspection_agent_g1.teacher_bank import G1TeacherBankRecord

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _hyperparameters(iteration: int = 0) -> PolicyTrainingHyperparameters:
    return PolicyTrainingHyperparameters(
        model_name=PolicyModelName.STRUCTURED_INSPECTION_POLICY,
        route=TrainingRoute.SOFT_UTILITY_DISTILL,
        cai_context_mode=CAIContextMode.TASK_SPECIFIC_MASKED,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=0.5,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=iteration,
    )


def _record(source: str, iteration: int, index: int = 0):
    return SimpleNamespace(
        example=SimpleNamespace(
            outer_target="d6",
            source_domain=source,
            dagger_iteration=iteration,
        ),
        state_sha256=_sha(f"record-{source}-{iteration}-{index}"),
    )


def _dependencies(source: str) -> G1SourceDependencies:
    result = object.__new__(G1SourceDependencies)
    object.__setattr__(
        result,
        "roster",
        SimpleNamespace(
            outer_target="d6",
            labeled_domain=source,
            fit_domains=tuple(
                domain for domain in DOMAINS if domain not in {"d6", source}
            ),
        ),
    )
    object.__setattr__(result, "state_sha256", _sha(f"dependency-{source}"))
    return result


def _batch(task: InspectionTask) -> G1DaggerRelabelBatch:
    observation_sha = _sha(f"observation-{task.value}")
    global_scalars = np.zeros(GLOBAL_SCALAR_DIMENSION, dtype=np.float64)
    global_scalars[12:14] = (0.4, 1.0)
    legal = np.zeros(ACTION_SLOT_COUNT, dtype=np.bool_)
    legal[0] = True
    state = G1PolicyState(
        task=task,
        task_token_mode=TaskTokenMode.CORRECT,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        observation_sha256=observation_sha,
        reconstruction_sha256=_sha(f"reconstruction-{task.value}"),
        surface_hypothesis_sha256=_sha("surface"),
        grid_sha256=_sha("grid"),
        reconstruction_embedding=np.zeros(
            RECONSTRUCTION_EMBEDDING_DIMENSION, dtype=np.float64
        ),
        global_scalars=global_scalars,
        task_token=np.asarray(
            (1.0, 0.0) if task is InspectionTask.FIELD else (0.0, 1.0),
            dtype=np.float64,
        ),
        cell_features=np.zeros(
            (CELL_COUNT, CELL_FEATURE_DIMENSION), dtype=np.float64
        ),
        candidate_features=np.zeros(
            (ACTION_SLOT_COUNT, CANDIDATE_FEATURE_DIMENSION), dtype=np.float64
        ),
        legal_action_mask=legal,
    )
    label = PrivilegedTeacherLabel(
        task=task,
        authorization_sha256=_sha("authorization"),
        observation_sha256=observation_sha,
        policy_state_sha256=state.state_sha256,
        selected_slot=0,
        candidates=(
            TeacherCandidateRecord(
                slot=0,
                action=InspectionCellAction(0, -1, 0),
                decision=InspectionDecision.BROADEN,
                exact_added_cost=1,
                raw_value=1.0,
                objective_value=1.0,
                task_loss_after=0.0,
                candidate_state_sha256=_sha(f"candidate-{task.value}"),
                selected=True,
            ),
        ),
    )
    visited = DaggerVisitedState(
        outer_target="d6",
        source_domain="d1",
        specimen_sha256=_sha("specimen"),
        task=task,
        iteration=1,
        trajectory_index=0,
        trajectory_length=1,
        observation_sha256=observation_sha,
        policy_state_sha256=state.state_sha256,
        teacher_label_sha256=label.state_sha256,
    )
    record = G1TeacherBankRecord(
        example=G1PolicyTrainingExample(
            outer_target="d6",
            source_domain="d1",
            specimen_sha256=_sha("specimen"),
            task=task,
            dagger_iteration=1,
            policy_state=state,
            teacher_label=label,
        ),
        fit_domains=("d2", "d3", "d4", "d5"),
        state_source="DAGGER_ACTOR_VISITED",
        source_state_sha256=visited.state_sha256,
        prior_sha256=_sha("prior"),
        assessor_sha256=_sha("assessor"),
    )
    return G1DaggerRelabelBatch(
        outer_target="d6",
        source_domain="d1",
        specimen_sha256=_sha("specimen"),
        task=task,
        iteration=1,
        actor_model_sha256=_sha("actor"),
        trajectory_sha256=_sha(f"trajectory-{task.value}"),
        trajectory_length=1,
        records=(record,),
        visited_states=(visited,),
    )


def test_dagger_hyperparameters_change_only_registered_iteration() -> None:
    base = _hyperparameters()

    iteration_two = module.with_dagger_iterations(base, 2)

    assert iteration_two.dagger_iterations == 2
    assert iteration_two.model_name is base.model_name
    assert iteration_two.route is base.route
    assert iteration_two.cai_context_mode is base.cai_context_mode
    assert iteration_two.task_token_mode is base.task_token_mode
    assert iteration_two.tau == base.tau
    assert iteration_two.learning_rate == base.learning_rate
    assert iteration_two.weight_decay == base.weight_decay
    assert iteration_two.state_sha256 != base.state_sha256


def test_dagger_source_bank_round_trip_binds_actor_and_trajectory_lineage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "d1.parquet"
    batches = (_batch(InspectionTask.FIELD), _batch(InspectionTask.CAI))

    identity = module.write_g1_dagger_source_bank(
        path,
        batches,
        dependency_sha256=_sha("dependencies"),
        actor_fit_audit_sha256=_sha("actor-audit"),
        generation_hyperparameters_sha256=_hyperparameters().state_sha256,
    )
    replayed_identity, records = module.read_g1_dagger_source_bank(path)

    assert replayed_identity == identity
    assert identity.actor_model_sha256 == _sha("actor")
    assert identity.dependency_sha256 == _sha("dependencies")
    assert identity.specimen_count == 1
    assert identity.row_count == 2
    assert len(identity.batch_state_sha256s) == 2
    assert {row.example.task for row in records} == {
        InspectionTask.FIELD,
        InspectionTask.CAI,
    }

    manifest_path = module.dagger_manifest_path(path)
    payload = json.loads(manifest_path.read_bytes())
    payload["batches"][0]["trajectory_sha256"] = _sha("tampered")
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(module.G1DaggerOrchestrationError, match="batch hash"):
        module.read_g1_dagger_source_bank(path)


def test_outer_dagger_banks_are_crossfit_and_progress_in_registered_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = object.__new__(G1Runtime)
    protocol = object.__new__(G1Protocol)
    object.__setattr__(protocol, "domain_order", DOMAINS)
    object.__setattr__(protocol, "epochs", 80)
    object.__setattr__(protocol, "patience", 12)
    base = _hyperparameters()
    fitted: list[tuple[int, str]] = []
    dependency_calls: list[str] = []
    built: list[tuple[int, str, int]] = []

    def fake_read_teacher_bank(path):
        source = Path(path).stem
        return SimpleNamespace(manifest_sha256=_sha(f"base-{source}")), (
            _record(source, 0),
        )

    monkeypatch.setattr(module, "read_teacher_bank", fake_read_teacher_bank)
    monkeypatch.setattr(
        module,
        "rebind_training_example_modes",
        lambda example, **_kwargs: example,
    )

    def fake_fit(examples, *, validation_domain, hyperparameters, **_kwargs):
        fitted.append((hyperparameters.dagger_iterations, validation_domain))
        assert {row.source_domain for row in examples} == set(DOMAINS[:-1])
        assert {
            row.dagger_iteration for row in examples
        } == set(range(hyperparameters.dagger_iterations + 1))
        fit_domains = tuple(
            domain for domain in DOMAINS if domain not in {"d6", validation_domain}
        )
        return SimpleNamespace(
            hyperparameters=hyperparameters,
            model_state_sha256=_sha(
                f"actor-{hyperparameters.dagger_iterations}-{validation_domain}"
            ),
            audit=SimpleNamespace(
                outer_target="d6",
                validation_domain=validation_domain,
                fit_domains=fit_domains,
                state_sha256=_sha(
                    f"audit-{hyperparameters.dagger_iterations}-{validation_domain}"
                ),
            ),
        )

    def fake_dependencies(_runtime, _protocol, *, labeled_domain, **_kwargs):
        dependency_calls.append(labeled_domain)
        return _dependencies(labeled_domain)

    def fake_build(
        _runtime,
        _protocol,
        dependencies,
        *,
        actor,
        iteration,
        work_root,
        **_kwargs,
    ):
        source = dependencies.roster.labeled_domain
        built.append((iteration, source, actor.hyperparameters.dagger_iterations))
        records = (_record(source, iteration),)
        return module.G1DaggerSourceBankBuild(
            path=module.dagger_source_bank_path(
                work_root, "d6", iteration, source
            ),
            outer_target="d6",
            source_domain=source,
            iteration=iteration,
            specimen_count=1,
            dependency_sha256=dependencies.state_sha256,
            actor_model_sha256=actor.model_state_sha256,
            generation_hyperparameters_sha256=actor.hyperparameters.state_sha256,
            bank=SimpleNamespace(manifest_sha256=_sha(f"bank-{iteration}-{source}")),
            records=records,
        )

    monkeypatch.setattr(module, "fit_inner_observable_policy", fake_fit)
    monkeypatch.setattr(module, "build_g1_source_dependencies", fake_dependencies)
    monkeypatch.setattr(module, "build_g1_dagger_source_bank", fake_build)

    result = module.build_g1_outer_dagger_banks(
        runtime,
        protocol,
        outer_target="d6",
        base_hyperparameters=base,
        encoder=SimpleNamespace(encode=lambda _images: None),
        teacher_bank_root=tmp_path / "teacher",
        work_root=tmp_path / "dagger",
        device="cpu",
    )

    assert fitted == [
        (iteration - 1, source)
        for iteration in (1, 2)
        for source in DOMAINS[:-1]
    ]
    assert built == [
        (iteration, source, iteration - 1)
        for iteration in (1, 2)
        for source in DOMAINS[:-1]
    ]
    assert dependency_calls == list(DOMAINS[:-1])
    assert len(result.banks) == 10
    assert {row.example.dagger_iteration for row in result.records} == {0, 1, 2}
    assert result.target_outcomes_opened is False


def test_outer_dagger_bank_reuses_hash_bound_cached_iteration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = object.__new__(G1Runtime)
    protocol = object.__new__(G1Protocol)
    object.__setattr__(protocol, "domain_order", DOMAINS)
    object.__setattr__(protocol, "epochs", 80)
    object.__setattr__(protocol, "patience", 12)
    base = _hyperparameters()
    read_calls: list[tuple[int, str]] = []

    monkeypatch.setattr(
        module,
        "read_teacher_bank",
        lambda path: (
            SimpleNamespace(manifest_sha256=_sha(f"base-{Path(path).stem}")),
            (_record(Path(path).stem, 0),),
        ),
    )
    monkeypatch.setattr(
        module,
        "rebind_training_example_modes",
        lambda example, **_kwargs: example,
    )

    for iteration in (1, 2):
        for source in DOMAINS[:-1]:
            path = module.dagger_source_bank_path(
                tmp_path / "dagger", "d6", iteration, source
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            module.dagger_manifest_path(path).touch()

    def fake_read(path):
        source = Path(path).stem
        iteration = int(Path(path).parent.name.removeprefix("iteration_"))
        read_calls.append((iteration, source))
        hp = module.with_dagger_iterations(base, iteration - 1)
        return (
            SimpleNamespace(
                outer_target="d6",
                source_domain=source,
                fit_domains=tuple(
                    domain for domain in DOMAINS[:-1] if domain != source
                ),
                iteration=iteration,
                specimen_count=1,
                dependency_sha256=_sha(f"dependency-{source}"),
                actor_model_sha256=_sha(f"actor-{iteration}-{source}"),
                generation_hyperparameters_sha256=hp.state_sha256,
                manifest_sha256=_sha(f"manifest-{iteration}-{source}"),
            ),
            (_record(source, iteration),),
        )

    monkeypatch.setattr(module, "read_g1_dagger_source_bank", fake_read)
    monkeypatch.setattr(
        module,
        "fit_inner_observable_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fit")),
    )
    monkeypatch.setattr(
        module,
        "build_g1_source_dependencies",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dependency")),
    )

    result = module.build_g1_outer_dagger_banks(
        runtime,
        protocol,
        outer_target="d6",
        base_hyperparameters=base,
        encoder=SimpleNamespace(encode=lambda _images: None),
        teacher_bank_root=tmp_path / "teacher",
        work_root=tmp_path / "dagger",
        device="cpu",
    )

    assert read_calls == [
        (iteration, source)
        for iteration in (1, 2)
        for source in DOMAINS[:-1]
    ]
    assert len(result.banks) == 10
