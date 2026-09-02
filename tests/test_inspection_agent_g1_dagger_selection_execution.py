from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1 import dagger_selection_execution as module
from cmc_bbdm.inspection_agent_g1.aawr_selection_execution import (
    G1OuterAAWRSelectionRun,
)
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.dagger_orchestration import G1OuterDaggerBuild
from cmc_bbdm.inspection_agent_g1.engineering_selection_execution import (
    G1EngineeringCandidateRun,
    G1OuterEngineeringSelectionRun,
)
from cmc_bbdm.inspection_agent_g1.g1 import G1Protocol, G1Runtime, G1SourceDependencies
from cmc_bbdm.inspection_agent_g1.policy_selection import (
    InnerPolicyEngineeringMetric,
    PolicyCandidateEvaluation,
    select_outer_policy,
)
from cmc_bbdm.inspection_agent_g1.policy_training import (
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
)

DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _hyperparameters(iteration: int) -> PolicyTrainingHyperparameters:
    return PolicyTrainingHyperparameters(
        model_name=PolicyModelName.STRUCTURED_INSPECTION_POLICY,
        route=TrainingRoute.SOFT_UTILITY_DISTILL,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=0.5,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=iteration,
    )


def _candidate(iteration: int) -> G1EngineeringCandidateRun:
    hyperparameters = _hyperparameters(iteration)
    learned = {
        0: {InspectionTask.FIELD: 0.96, InspectionTask.CAI: 0.80},
        1: {InspectionTask.FIELD: 0.95, InspectionTask.CAI: 0.70},
        2: {InspectionTask.FIELD: 0.97, InspectionTask.CAI: 0.75},
    }
    candidate = PolicyCandidateEvaluation(
        hyperparameters=hyperparameters,
        inner_metrics=tuple(
            InnerPolicyEngineeringMetric(
                outer_target="d6",
                validation_domain=source,
                task=task,
                hyperparameters_sha256=hyperparameters.state_sha256,
                model_state_sha256=_sha(f"model-{iteration}-{source}"),
                learned_auebc=learned[iteration][task],
                fixed_auebc=1.0,
                oracle_auebc=0.5,
                selected_epoch=7 + iteration,
            )
            for source in DOMAINS[:-1]
            for task in (InspectionTask.FIELD, InspectionTask.CAI)
        ),
    )
    return G1EngineeringCandidateRun(
        candidate=candidate,
        inner_evaluation_sha256s=tuple(
            _sha(f"evaluation-{iteration}-{source}") for source in DOMAINS[:-1]
        ),
        learned_bank_manifest_sha256s=tuple(
            _sha(f"learned-{iteration}-{source}") for source in DOMAINS[:-1]
        ),
    )


def _dependencies(source: str) -> G1SourceDependencies:
    result = object.__new__(G1SourceDependencies)
    object.__setattr__(
        result,
        "roster",
        SimpleNamespace(
            outer_target="d6",
            labeled_domain=source,
            fit_domains=tuple(domain for domain in DOMAINS[:-1] if domain != source),
        ),
    )
    object.__setattr__(result, "state_sha256", _sha(f"dependency-{source}"))
    return result


def test_dagger_selection_compares_zero_one_two_and_authorizes_aawr_source_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = object.__new__(G1Runtime)
    protocol = object.__new__(G1Protocol)
    object.__setattr__(protocol, "domain_order", DOMAINS)
    base_candidate = _candidate(0)
    base_selection = select_outer_policy((base_candidate.candidate,))
    base_run = G1OuterEngineeringSelectionRun(
        outer_target="d6",
        example_count=10,
        candidates=(base_candidate,),
        selection=base_selection,
        path=tmp_path / "base" / "selection.json",
    )
    dependency_calls: list[str] = []
    candidate_calls: list[int] = []

    monkeypatch.setattr(
        module,
        "run_outer_engineering_selection",
        lambda *_args, **_kwargs: base_run,
    )

    def fake_dependencies(_runtime, _protocol, *, labeled_domain, **_kwargs):
        dependency_calls.append(labeled_domain)
        return _dependencies(labeled_domain)

    monkeypatch.setattr(module, "build_g1_source_dependencies", fake_dependencies)

    def fake_dagger_build(*_args, dependency_factory, **_kwargs):
        for source in DOMAINS[:-1]:
            dependency_factory(source)
        return G1OuterDaggerBuild(
            outer_target="d6",
            base_hyperparameters_sha256=_hyperparameters(0).state_sha256,
            base_teacher_manifest_sha256s=tuple(
                _sha(f"teacher-{source}") for source in DOMAINS[:-1]
            ),
            banks=tuple(
                SimpleNamespace(
                    bank=SimpleNamespace(
                        manifest_sha256=_sha(f"dagger-{iteration}-{source}")
                    )
                )
                for iteration in (1, 2)
                for source in DOMAINS[:-1]
            ),
            records=(
                SimpleNamespace(
                    example=SimpleNamespace(
                        outer_target="d6",
                        source_domain="d1",
                        dagger_iteration=0,
                    )
                ),
            ),
        )

    monkeypatch.setattr(module, "build_g1_outer_dagger_banks", fake_dagger_build)
    monkeypatch.setattr(
        module,
        "read_source_bridge_bank",
        lambda path: (
            SimpleNamespace(manifest_sha256=_sha(f"bridge-{Path(path).stem}")),
            (
                SimpleNamespace(
                    outer_target="d6", source_domain=Path(path).stem
                ),
            ),
        ),
    )

    def fake_candidate(*_args, hyperparameters, dependency_factory, **_kwargs):
        candidate_calls.append(hyperparameters.dagger_iterations)
        for source in DOMAINS[:-1]:
            dependency_factory(source)
        return _candidate(hyperparameters.dagger_iterations)

    monkeypatch.setattr(module, "run_engineering_candidate", fake_candidate)
    aawr_calls = []

    def fake_aawr_selection(*_args, authorization, base_candidate, **_kwargs):
        aawr_calls.append((authorization.state_sha256, base_candidate.state_sha256))
        result = object.__new__(G1OuterAAWRSelectionRun)
        object.__setattr__(result, "outer_target", "d6")
        object.__setattr__(result, "authorization", authorization)
        object.__setattr__(result, "base_candidate", base_candidate)
        object.__setattr__(result, "candidates", ())
        object.__setattr__(
            result,
            "selection",
            select_outer_policy((base_candidate.candidate,)),
        )
        object.__setattr__(result, "path", tmp_path / "aawr" / "selection.json")
        return result

    monkeypatch.setattr(module, "run_outer_aawr_selection", fake_aawr_selection)

    result = module.run_outer_dagger_selection(
        runtime,
        protocol,
        outer_target="d6",
        encoder=SimpleNamespace(encode=lambda _images: None),
        teacher_bank_root=tmp_path / "teachers",
        bridge_root=tmp_path / "bridges",
        supervised_root=tmp_path / "supervised",
        learned_root=tmp_path / "learned",
        base_work_root=tmp_path / "base",
        dagger_bank_root=tmp_path / "dagger-banks",
        work_root=tmp_path / "dagger-selection",
        device="cpu",
    )

    assert candidate_calls == [1, 2]
    assert dependency_calls == list(DOMAINS[:-1])
    assert result.selection.selected_hyperparameters_sha256 == _hyperparameters(
        1
    ).state_sha256
    assert result.aawr_authorization.status == "AUTHORIZED_SOURCE_ONLY"
    assert result.aawr_authorization.authorized_tasks == (InspectionTask.FIELD,)
    assert len(aawr_calls) == 1
    assert result.aawr_selection is not None
    assert (
        result.final_selection.selected_hyperparameters_sha256
        == result.selection.selected_hyperparameters_sha256
    )
    assert result.target_outcomes_opened is False
    payload = json.loads(result.path.read_bytes())
    assert payload["target_outcomes_opened"] is False
    assert payload["selected_dagger_iterations"] == 1
    assert payload["aawr_authorization"]["status"] == "AUTHORIZED_SOURCE_ONLY"
