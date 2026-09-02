from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1 import engineering_selection_execution as module
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.g1 import (
    G1Protocol,
    G1Runtime,
    G1SourceDependencies,
)
from cmc_bbdm.inspection_agent_g1.policy_selection import (
    InnerPolicyEngineeringMetric,
    PolicyCandidateEvaluation,
)
from cmc_bbdm.inspection_agent_g1.policy_training import (
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
)
from cmc_bbdm.inspection_agent_g1.source_policy_evaluation import (
    G1LearnedSourceBankFile,
    learned_source_bank_path,
)

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


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


def _candidate(
    hyperparameters: PolicyTrainingHyperparameters,
    relative: float,
) -> PolicyCandidateEvaluation:
    return PolicyCandidateEvaluation(
        hyperparameters=hyperparameters,
        inner_metrics=tuple(
            InnerPolicyEngineeringMetric(
                outer_target="d6",
                validation_domain=source,
                task=task,
                hyperparameters_sha256=hyperparameters.state_sha256,
                model_state_sha256=_sha(
                    f"model-{hyperparameters.state_sha256}-{source}"
                ),
                learned_auebc=relative,
                fixed_auebc=1.0,
                oracle_auebc=0.5,
                selected_epoch=7,
            )
            for source in DOMAINS[:-1]
            for task in (InspectionTask.FIELD, InspectionTask.CAI)
        ),
    )


def _dependencies(source: str) -> G1SourceDependencies:
    result = object.__new__(G1SourceDependencies)
    roster = SimpleNamespace(
        outer_target="d6",
        labeled_domain=source,
        fit_domains=tuple(
            domain for domain in DOMAINS if domain not in {"d6", source}
        ),
    )
    object.__setattr__(result, "roster", roster)
    object.__setattr__(result, "state_sha256", _sha(f"dependencies-{source}"))
    return result


def test_engineering_candidate_fits_and_evaluates_each_source_fold_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = object.__new__(G1Runtime)
    protocol = object.__new__(G1Protocol)
    object.__setattr__(protocol, "epochs", 80)
    object.__setattr__(protocol, "patience", 12)
    examples = tuple(
        SimpleNamespace(outer_target="d6", source_domain=domain)
        for domain in DOMAINS[:-1]
    )
    hp = _hyperparameters()
    fitted: list[str] = []
    built: list[str] = []
    dependency_calls: list[str] = []

    monkeypatch.setattr(
        module,
        "rebind_training_example_modes",
        lambda row, **_kwargs: row,
    )

    def fake_fit(_examples, *, validation_domain, **_kwargs):
        fitted.append(validation_domain)
        return SimpleNamespace(
            model_state_sha256=_sha(f"model-{validation_domain}"),
            hyperparameters=hp,
            audit=SimpleNamespace(
                outer_target="d6",
                validation_domain=validation_domain,
                fit_domains=tuple(
                    domain
                    for domain in DOMAINS
                    if domain not in {"d6", validation_domain}
                ),
                selected_epoch=7,
            ),
        )

    def fake_build(_runtime, _protocol, dependencies, **_kwargs):
        built.append(dependencies.roster.labeled_domain)

    def fake_read(path):
        source = Path(path).stem
        return (
            G1LearnedSourceBankFile(
                row_count=2,
                parquet_sha256=_sha(f"parquet-{source}"),
                records_sha256=_sha(f"records-{source}"),
                manifest_sha256=_sha(f"manifest-{source}"),
            ),
            (SimpleNamespace(source_domain=source),),
        )

    def fake_evaluate(learned, _bridges, *, hyperparameters):
        source = learned[0].source_domain
        metrics = tuple(
            InnerPolicyEngineeringMetric(
                outer_target="d6",
                validation_domain=source,
                task=task,
                hyperparameters_sha256=hyperparameters.state_sha256,
                model_state_sha256=_sha(f"model-{source}"),
                learned_auebc=0.8,
                fixed_auebc=1.0,
                oracle_auebc=0.5,
                selected_epoch=7,
            )
            for task in (InspectionTask.FIELD, InspectionTask.CAI)
        )
        return SimpleNamespace(
            outer_target="d6",
            validation_domain=source,
            hyperparameters_sha256=hyperparameters.state_sha256,
            metrics=metrics,
            state_sha256=_sha(f"evaluation-{source}"),
        )

    def dependency_factory(source: str) -> G1SourceDependencies:
        dependency_calls.append(source)
        return _dependencies(source)

    monkeypatch.setattr(module, "fit_inner_observable_policy", fake_fit)
    monkeypatch.setattr(module, "build_g1_learned_source_bank", fake_build)
    monkeypatch.setattr(module, "read_learned_source_bank", fake_read)
    monkeypatch.setattr(module, "evaluate_inner_policy_bridge", fake_evaluate)

    result = module.run_engineering_candidate(
        runtime,
        protocol,
        outer_target="d6",
        hyperparameters=hp,
        teacher_examples=examples,
        bridge_records=(SimpleNamespace(),),
        dependency_factory=dependency_factory,
        encoder=SimpleNamespace(encode=lambda _images: None),
        learned_root=tmp_path / "learned",
        device="cpu",
    )

    assert dependency_calls == list(DOMAINS[:-1])
    assert fitted == list(DOMAINS[:-1])
    assert built == list(DOMAINS[:-1])
    assert result.candidate.source_validation_domains == DOMAINS[:-1]
    assert len(result.candidate.inner_metrics) == 10
    assert result.candidate.equal_domain_mean_relative_auebc == 0.8
    assert len(result.inner_evaluation_sha256s) == 5
    assert len(result.learned_bank_manifest_sha256s) == 5


def test_engineering_candidate_replays_complete_banks_without_refitting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = object.__new__(G1Runtime)
    protocol = object.__new__(G1Protocol)
    object.__setattr__(protocol, "epochs", 80)
    object.__setattr__(protocol, "patience", 12)
    hp = _hyperparameters()
    examples = tuple(
        SimpleNamespace(outer_target="d6", source_domain=domain)
        for domain in DOMAINS[:-1]
    )
    learned_root = tmp_path / "learned"
    for source in DOMAINS[:-1]:
        path = learned_source_bank_path(
            learned_root,
            "d6",
            hp.state_sha256,
            source,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cached")
        path.with_suffix(f"{path.suffix}.manifest.json").write_bytes(b"cached")

    monkeypatch.setattr(
        module,
        "rebind_training_example_modes",
        lambda row, **_kwargs: row,
    )

    def fake_read(path):
        source = Path(path).stem
        return (
            G1LearnedSourceBankFile(
                row_count=2,
                parquet_sha256=_sha(f"parquet-{source}"),
                records_sha256=_sha(f"records-{source}"),
                manifest_sha256=_sha(f"manifest-{source}"),
            ),
            (SimpleNamespace(source_domain=source),),
        )

    def fake_evaluate(learned, _bridges, *, hyperparameters):
        source = learned[0].source_domain
        return SimpleNamespace(
            outer_target="d6",
            validation_domain=source,
            hyperparameters_sha256=hyperparameters.state_sha256,
            metrics=tuple(
                InnerPolicyEngineeringMetric(
                    outer_target="d6",
                    validation_domain=source,
                    task=task,
                    hyperparameters_sha256=hyperparameters.state_sha256,
                    model_state_sha256=_sha(f"model-{source}"),
                    learned_auebc=0.9,
                    fixed_auebc=1.0,
                    oracle_auebc=0.5,
                    selected_epoch=8,
                )
                for task in (InspectionTask.FIELD, InspectionTask.CAI)
            ),
            state_sha256=_sha(f"evaluation-{source}"),
        )

    monkeypatch.setattr(module, "read_learned_source_bank", fake_read)
    monkeypatch.setattr(module, "evaluate_inner_policy_bridge", fake_evaluate)
    monkeypatch.setattr(
        module,
        "fit_inner_observable_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cached candidate must not refit")
        ),
    )

    result = module.run_engineering_candidate(
        runtime,
        protocol,
        outer_target="d6",
        hyperparameters=hp,
        teacher_examples=examples,
        bridge_records=(SimpleNamespace(),),
        dependency_factory=lambda _source: (_ for _ in ()).throw(
            AssertionError("cached candidate must not rebuild dependencies")
        ),
        encoder=SimpleNamespace(encode=lambda _images: None),
        learned_root=learned_root,
        device="cpu",
    )

    assert result.candidate.equal_domain_mean_relative_auebc == 0.9


def test_outer_engineering_selection_freezes_engineering_winner_without_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = object.__new__(G1Runtime)
    protocol = object.__new__(G1Protocol)
    object.__setattr__(protocol, "domain_order", DOMAINS)
    object.__setattr__(protocol, "epochs", 80)
    object.__setattr__(protocol, "patience", 12)
    first = _hyperparameters()
    second = PolicyTrainingHyperparameters(
        model_name=PolicyModelName.SHARED_ACTION_MLP,
        route=TrainingRoute.HARD_BC,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=None,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=0,
    )
    supervised = SimpleNamespace(
        core_results=(SimpleNamespace(hyperparameters=first),),
        tuning_results=(SimpleNamespace(hyperparameters=second),),
        selection=SimpleNamespace(state_sha256=_sha("supervised-selection")),
    )
    monkeypatch.setattr(
        module,
        "run_outer_supervised_selection",
        lambda *_args, **_kwargs: supervised,
    )

    def fake_teacher(path):
        source = Path(path).stem
        return SimpleNamespace(manifest_sha256=_sha(f"teacher-{source}")), (
            SimpleNamespace(
                example=SimpleNamespace(
                    outer_target="d6",
                    source_domain=source,
                )
            ),
        )

    def fake_bridge(path):
        source = Path(path).stem
        return SimpleNamespace(manifest_sha256=_sha(f"bridge-{source}")), (
            SimpleNamespace(outer_target="d6", source_domain=source),
        )

    monkeypatch.setattr(module, "read_teacher_bank", fake_teacher)
    monkeypatch.setattr(module, "read_source_bridge_bank", fake_bridge)

    def fake_candidate(_runtime, _protocol, *, hyperparameters, **_kwargs):
        relative = 0.9 if hyperparameters == first else 0.8
        return module.G1EngineeringCandidateRun(
            candidate=_candidate(hyperparameters, relative),
            inner_evaluation_sha256s=tuple(
                _sha(f"evaluation-{hyperparameters.state_sha256}-{source}")
                for source in DOMAINS[:-1]
            ),
            learned_bank_manifest_sha256s=tuple(
                _sha(f"learned-{hyperparameters.state_sha256}-{source}")
                for source in DOMAINS[:-1]
            ),
        )

    monkeypatch.setattr(module, "run_engineering_candidate", fake_candidate)

    result = module.run_outer_engineering_selection(
        runtime,
        protocol,
        outer_target="d6",
        encoder=SimpleNamespace(encode=lambda _images: None),
        teacher_bank_root=tmp_path / "teachers",
        bridge_root=tmp_path / "bridges",
        supervised_root=tmp_path / "supervised",
        learned_root=tmp_path / "learned",
        work_root=tmp_path / "engineering",
        device="cpu",
    )

    payload = json.loads(result.path.read_bytes())
    assert result.selection.selected_hyperparameters_sha256 == second.state_sha256
    assert result.selection.target_outcomes_opened is False
    assert payload["selection"]["target_outcomes_opened"] is False
    assert payload["selection"]["selected_hyperparameters_sha256"] == (
        second.state_sha256
    )
    assert len(payload["candidate_run_sha256s"]) == 2
