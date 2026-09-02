from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from cmc_bbdm.inspection_agent_g1 import selection_execution as module
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode, TaskTokenMode
from cmc_bbdm.inspection_agent_g1.policy_training import (
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
)
from cmc_bbdm.inspection_agent_g1.selection_execution import (
    TeacherRegretCandidateResult,
    core_policy_candidates,
    fit_teacher_regret_candidate,
    read_teacher_regret_candidate_result,
    select_teacher_regret_candidate,
    tuning_policy_candidates,
    write_teacher_regret_candidate_result,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _hp(
    model: PolicyModelName,
    route: TrainingRoute,
    *,
    tau: float | None,
) -> PolicyTrainingHyperparameters:
    return PolicyTrainingHyperparameters(
        model_name=model,
        route=route,
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=tau,
        learning_rate=0.0003,
        weight_decay=0.0001,
        dagger_iterations=0,
    )


def _result(
    hp: PolicyTrainingHyperparameters,
    regret: float,
) -> TeacherRegretCandidateResult:
    domains = ("d1", "d2", "d3", "d4", "d5")
    return TeacherRegretCandidateResult(
        outer_target="d6",
        hyperparameters=hp,
        validation_domains=domains,
        validation_regrets=tuple(regret for _ in domains),
        selected_epochs=(10, 11, 12, 13, 14),
        model_state_sha256s=tuple(_sha(f"{hp.state_sha256}-{d}") for d in domains),
    )


def test_core_then_winner_conditioned_tuning_covers_every_registered_axis() -> None:
    core = core_policy_candidates()
    assert len(core) == 4
    assert {row.model_name for row in core} == set(PolicyModelName)
    assert {row.route for row in core} == {
        TrainingRoute.HARD_BC,
        TrainingRoute.SOFT_UTILITY_DISTILL,
    }
    assert all(row.route is not TrainingRoute.PRIVILEGED_AAWR for row in core)
    soft = next(
        row
        for row in core
        if row.model_name is PolicyModelName.SHARED_ACTION_MLP
        and row.route is TrainingRoute.SOFT_UTILITY_DISTILL
    )
    tuning = tuning_policy_candidates(soft)
    assert {row.tau for row in tuning} == {0.25, 0.5, 1.0, 2.0}
    assert {row.cai_context_mode for row in tuning} == set(CAIContextMode)
    assert {row.learning_rate for row in tuning} == {0.0001, 0.0003}
    assert {row.weight_decay for row in tuning} == {0.0001, 0.001}
    assert all(row.model_name is soft.model_name for row in tuning)
    assert all(row.route is soft.route for row in tuning)
    assert all(row.task_token_mode is TaskTokenMode.CORRECT for row in tuning)
    assert all(row.dagger_iterations == 0 for row in tuning)


def test_teacher_regret_selection_is_source_only_and_prefers_lower_regret() -> None:
    simple = _result(
        _hp(PolicyModelName.SHARED_ACTION_MLP, TrainingRoute.HARD_BC, tau=None),
        0.2,
    )
    structured = _result(
        _hp(
            PolicyModelName.STRUCTURED_INSPECTION_POLICY,
            TrainingRoute.SOFT_UTILITY_DISTILL,
            tau=0.5,
        ),
        0.1,
    )
    selected = select_teacher_regret_candidate((simple, structured))
    assert selected.outer_target == "d6"
    assert selected.selected_hyperparameters_sha256 == (
        structured.hyperparameters.state_sha256
    )
    assert selected.target_outcomes_opened is False


def test_candidate_fit_visits_each_source_validation_domain_once(monkeypatch) -> None:
    examples = tuple(
        SimpleNamespace(
            outer_target="d6",
            source_domain=domain,
            policy_state=SimpleNamespace(
                cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
                task_token_mode=TaskTokenMode.CORRECT,
            ),
        )
        for domain in ("d1", "d2", "d3", "d4", "d5")
    )
    calls: list[str] = []

    monkeypatch.setattr(module, "rebind_training_example_modes", lambda row, **_k: row)

    def fake_fit(_examples: object, *, validation_domain: str, **_kwargs: object):
        calls.append(validation_domain)
        return SimpleNamespace(
            audit=SimpleNamespace(
                outer_target="d6",
                validation_domain=validation_domain,
                best_validation_regret=float(len(calls)),
                selected_epoch=10 + len(calls),
            ),
            model_state_sha256=_sha(validation_domain),
        )

    monkeypatch.setattr(module, "fit_inner_observable_policy", fake_fit)
    hp = core_policy_candidates()[0]
    result = fit_teacher_regret_candidate(
        examples,
        hp,
        max_epochs=80,
        patience=12,
        device="cpu",
    )
    assert tuple(calls) == ("d1", "d2", "d3", "d4", "d5")
    assert result.validation_regrets == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert result.equal_domain_mean_regret == 3.0


def test_candidate_result_cache_round_trips_exactly(tmp_path: Path) -> None:
    result = _result(
        _hp(
            PolicyModelName.STRUCTURED_INSPECTION_POLICY,
            TrainingRoute.SOFT_UTILITY_DISTILL,
            tau=0.5,
        ),
        0.125,
    )
    path = tmp_path / "candidate.json"
    write_teacher_regret_candidate_result(path, result)
    replay = read_teacher_regret_candidate_result(path)
    assert replay == result
    assert path.read_bytes().endswith(b"\n")
