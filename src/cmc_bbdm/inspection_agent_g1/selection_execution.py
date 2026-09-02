"""Source-only staged execution for observable-policy model selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

from .contracts import CAIContextMode, TaskTokenMode
from .policy_training import (
    REGISTERED_MAX_EPOCHS,
    G1PolicyTrainingExample,
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
    fit_inner_observable_policy,
    rebind_training_example_modes,
)
from .teacher_bank import read_teacher_bank


class G1SelectionExecutionError(ValueError):
    """Raised when a source-only staged selection contract is incomplete."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _hyperparameters(
    *,
    model_name: PolicyModelName,
    route: TrainingRoute,
    cai_context_mode: CAIContextMode = CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
    tau: float | None,
    learning_rate: float = 0.0003,
    weight_decay: float = 0.0001,
) -> PolicyTrainingHyperparameters:
    return PolicyTrainingHyperparameters(
        model_name=model_name,
        route=route,
        cai_context_mode=cai_context_mode,
        task_token_mode=TaskTokenMode.CORRECT,
        tau=tau,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dagger_iterations=0,
    )


def core_policy_candidates() -> tuple[PolicyTrainingHyperparameters, ...]:
    """Compare both registered models and supervised routes at frozen defaults."""

    output = []
    for model in (
        PolicyModelName.SHARED_ACTION_MLP,
        PolicyModelName.STRUCTURED_INSPECTION_POLICY,
    ):
        output.append(
            _hyperparameters(
                model_name=model,
                route=TrainingRoute.HARD_BC,
                tau=None,
            )
        )
        output.append(
            _hyperparameters(
                model_name=model,
                route=TrainingRoute.SOFT_UTILITY_DISTILL,
                tau=0.5,
            )
        )
    return tuple(output)


def tuning_policy_candidates(
    source_winner: PolicyTrainingHyperparameters,
) -> tuple[PolicyTrainingHyperparameters, ...]:
    """Expand every registered axis only around the source-selected core route."""

    if (
        type(source_winner) is not PolicyTrainingHyperparameters
        or source_winner.dagger_iterations != 0
        or source_winner.task_token_mode is not TaskTokenMode.CORRECT
    ):
        raise G1SelectionExecutionError("core source winner is invalid")
    values: list[PolicyTrainingHyperparameters] = []

    def add(
        *,
        tau: float | None = source_winner.tau,
        context: CAIContextMode = source_winner.cai_context_mode,
        learning_rate: float = source_winner.learning_rate,
        weight_decay: float = source_winner.weight_decay,
    ) -> None:
        row = _hyperparameters(
            model_name=source_winner.model_name,
            route=source_winner.route,
            cai_context_mode=context,
            tau=tau,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        if row.state_sha256 not in {value.state_sha256 for value in values}:
            values.append(row)

    add()
    if source_winner.route is TrainingRoute.SOFT_UTILITY_DISTILL:
        for temperature in (0.25, 0.5, 1.0, 2.0):
            add(tau=temperature)
    for context in (
        CAIContextMode.TASK_SPECIFIC_MASKED,
        CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
    ):
        add(context=context)
    for learning_rate in (0.0001, 0.0003):
        add(learning_rate=learning_rate)
    for weight_decay in (0.0001, 0.001):
        add(weight_decay=weight_decay)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class TeacherRegretCandidateResult:
    outer_target: str
    hyperparameters: PolicyTrainingHyperparameters
    validation_domains: tuple[str, ...]
    validation_regrets: tuple[float, ...]
    selected_epochs: tuple[int, ...]
    model_state_sha256s: tuple[str, ...]
    equal_domain_mean_regret: float = field(init=False)
    final_refit_epochs: int = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        regrets = tuple(float(value) for value in self.validation_regrets)
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.hyperparameters) is not PolicyTrainingHyperparameters
            or type(self.validation_domains) is not tuple
            or len(self.validation_domains) != 5
            or len(set(self.validation_domains)) != 5
            or self.validation_domains != tuple(sorted(self.validation_domains))
            or self.outer_target in self.validation_domains
            or type(self.validation_regrets) is not tuple
            or len(regrets) != 5
            or any(not math.isfinite(value) or value < 0.0 for value in regrets)
            or type(self.selected_epochs) is not tuple
            or len(self.selected_epochs) != 5
            or any(
                type(value) is not int or not 1 <= value <= REGISTERED_MAX_EPOCHS
                for value in self.selected_epochs
            )
            or type(self.model_state_sha256s) is not tuple
            or len(self.model_state_sha256s) != 5
            or not all(_valid_sha256(value) for value in self.model_state_sha256s)
        ):
            raise G1SelectionExecutionError("teacher-regret candidate is invalid")
        mean = float(np.mean(regrets, dtype=np.float64))
        epochs = int(median(self.selected_epochs))
        object.__setattr__(self, "validation_regrets", regrets)
        object.__setattr__(self, "equal_domain_mean_regret", mean)
        object.__setattr__(self, "final_refit_epochs", epochs)
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-teacher-regret-candidate",
                    "outer_target": self.outer_target,
                    "hyperparameters": self.hyperparameters.state_sha256,
                    "validation_domains": self.validation_domains,
                    "validation_regrets": regrets,
                    "selected_epochs": self.selected_epochs,
                    "models": self.model_state_sha256s,
                    "equal_domain_mean_regret": mean,
                    "final_refit_epochs": epochs,
                }
            ),
        )


def _complexity(row: TeacherRegretCandidateResult) -> tuple[int, int, int, str]:
    hp = row.hyperparameters
    return (
        int(hp.model_name is PolicyModelName.STRUCTURED_INSPECTION_POLICY),
        int(hp.route is TrainingRoute.SOFT_UTILITY_DISTILL),
        int(hp.cai_context_mode is CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT),
        hp.state_sha256,
    )


@dataclass(frozen=True, slots=True)
class TeacherRegretSelection:
    outer_target: str
    selected_hyperparameters: PolicyTrainingHyperparameters
    selected_hyperparameters_sha256: str
    equal_domain_mean_regret: float
    final_refit_epochs: int
    candidate_state_sha256s: tuple[str, ...]
    target_outcomes_opened: bool
    state_sha256: str


def select_teacher_regret_candidate(
    candidates: tuple[TeacherRegretCandidateResult, ...],
) -> TeacherRegretSelection:
    if (
        type(candidates) is not tuple
        or not candidates
        or any(type(row) is not TeacherRegretCandidateResult for row in candidates)
        or len({row.state_sha256 for row in candidates}) != len(candidates)
        or len({row.outer_target for row in candidates}) != 1
        or len({row.validation_domains for row in candidates}) != 1
    ):
        raise G1SelectionExecutionError("teacher-regret selection is invalid")
    selected = min(
        candidates,
        key=lambda row: (row.equal_domain_mean_regret, *_complexity(row)),
    )
    payload = {
        "schema": 1,
        "kind": "g1-teacher-regret-selection",
        "outer_target": selected.outer_target,
        "selected_hyperparameters": selected.hyperparameters.state_sha256,
        "equal_domain_mean_regret": selected.equal_domain_mean_regret,
        "final_refit_epochs": selected.final_refit_epochs,
        "candidates": tuple(row.state_sha256 for row in candidates),
        "target_outcomes_opened": False,
    }
    return TeacherRegretSelection(
        outer_target=selected.outer_target,
        selected_hyperparameters=selected.hyperparameters,
        selected_hyperparameters_sha256=selected.hyperparameters.state_sha256,
        equal_domain_mean_regret=selected.equal_domain_mean_regret,
        final_refit_epochs=selected.final_refit_epochs,
        candidate_state_sha256s=tuple(row.state_sha256 for row in candidates),
        target_outcomes_opened=False,
        state_sha256=_json_sha(payload),
    )


def fit_teacher_regret_candidate(
    examples: tuple[G1PolicyTrainingExample, ...],
    hyperparameters: PolicyTrainingHyperparameters,
    *,
    max_epochs: int,
    patience: int,
    device: str,
) -> TeacherRegretCandidateResult:
    if type(examples) is not tuple or not examples:
        raise G1SelectionExecutionError("source teacher examples are empty")
    outer_targets = {row.outer_target for row in examples}
    source_domains = tuple(sorted({row.source_domain for row in examples}))
    if (
        len(outer_targets) != 1
        or len(source_domains) != 5
        or next(iter(outer_targets)) in source_domains
        or type(hyperparameters) is not PolicyTrainingHyperparameters
    ):
        raise G1SelectionExecutionError("source teacher roster is invalid")
    rebound = tuple(
        rebind_training_example_modes(
            row,
            cai_context_mode=hyperparameters.cai_context_mode,
            task_token_mode=hyperparameters.task_token_mode,
        )
        for row in examples
    )
    regrets = []
    epochs = []
    models = []
    outer = next(iter(outer_targets))
    for domain in source_domains:
        fitted = fit_inner_observable_policy(
            rebound,
            validation_domain=domain,
            hyperparameters=hyperparameters,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
        )
        audit = fitted.audit
        if (
            audit.outer_target != outer
            or audit.validation_domain != domain
            or audit.best_validation_regret is None
        ):
            raise G1SelectionExecutionError("inner policy fit identity changed")
        regrets.append(float(audit.best_validation_regret))
        epochs.append(audit.selected_epoch)
        models.append(fitted.model_state_sha256)
    return TeacherRegretCandidateResult(
        outer_target=outer,
        hyperparameters=hyperparameters,
        validation_domains=source_domains,
        validation_regrets=tuple(regrets),
        selected_epochs=tuple(epochs),
        model_state_sha256s=tuple(models),
    )


def _hyperparameters_payload(
    value: PolicyTrainingHyperparameters,
) -> dict[str, object]:
    return {
        "model_name": value.model_name.value,
        "route": value.route.value,
        "cai_context_mode": value.cai_context_mode.value,
        "task_token_mode": value.task_token_mode.value,
        "tau": value.tau,
        "learning_rate": value.learning_rate,
        "weight_decay": value.weight_decay,
        "dagger_iterations": value.dagger_iterations,
        "state_sha256": value.state_sha256,
    }


def teacher_regret_candidate_payload(
    result: TeacherRegretCandidateResult,
) -> dict[str, object]:
    if type(result) is not TeacherRegretCandidateResult:
        raise G1SelectionExecutionError("issued candidate result is required")
    return {
        "schema_version": 1,
        "scope": "inspection_agent_g1_teacher_regret_candidate",
        "outer_target": result.outer_target,
        "hyperparameters": _hyperparameters_payload(result.hyperparameters),
        "validation_domains": list(result.validation_domains),
        "validation_regrets": list(result.validation_regrets),
        "selected_epochs": list(result.selected_epochs),
        "model_state_sha256s": list(result.model_state_sha256s),
        "equal_domain_mean_regret": result.equal_domain_mean_regret,
        "final_refit_epochs": result.final_refit_epochs,
        "state_sha256": result.state_sha256,
        "target_outcomes_opened": False,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_teacher_regret_candidate_result(
    path: str | Path,
    result: TeacherRegretCandidateResult,
) -> None:
    _atomic_json(Path(path), teacher_regret_candidate_payload(result))


def read_teacher_regret_candidate_result(
    path: str | Path,
) -> TeacherRegretCandidateResult:
    try:
        payload = json.loads(Path(path).read_bytes())
        hp = payload["hyperparameters"]
        if not isinstance(payload, dict) or not isinstance(hp, dict):
            raise TypeError
        hyperparameters = PolicyTrainingHyperparameters(
            model_name=PolicyModelName(str(hp["model_name"])),
            route=TrainingRoute(str(hp["route"])),
            cai_context_mode=CAIContextMode(str(hp["cai_context_mode"])),
            task_token_mode=TaskTokenMode(str(hp["task_token_mode"])),
            tau=None if hp["tau"] is None else float(hp["tau"]),
            learning_rate=float(hp["learning_rate"]),
            weight_decay=float(hp["weight_decay"]),
            dagger_iterations=int(hp["dagger_iterations"]),
        )
        result = TeacherRegretCandidateResult(
            outer_target=str(payload["outer_target"]),
            hyperparameters=hyperparameters,
            validation_domains=tuple(str(value) for value in payload["validation_domains"]),
            validation_regrets=tuple(
                float(value) for value in payload["validation_regrets"]
            ),
            selected_epochs=tuple(int(value) for value in payload["selected_epochs"]),
            model_state_sha256s=tuple(
                str(value) for value in payload["model_state_sha256s"]
            ),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise G1SelectionExecutionError("candidate result cache is invalid") from error
    if (
        payload != teacher_regret_candidate_payload(result)
        or hp.get("state_sha256") != hyperparameters.state_sha256
    ):
        raise G1SelectionExecutionError("candidate result cache identity changed")
    return result


def teacher_regret_selection_payload(
    selection: TeacherRegretSelection,
) -> dict[str, object]:
    if type(selection) is not TeacherRegretSelection:
        raise G1SelectionExecutionError("issued teacher-regret selection is required")
    return {
        "outer_target": selection.outer_target,
        "selected_hyperparameters": _hyperparameters_payload(
            selection.selected_hyperparameters
        ),
        "selected_hyperparameters_sha256": (
            selection.selected_hyperparameters_sha256
        ),
        "equal_domain_mean_regret": selection.equal_domain_mean_regret,
        "final_refit_epochs": selection.final_refit_epochs,
        "candidate_state_sha256s": list(selection.candidate_state_sha256s),
        "target_outcomes_opened": selection.target_outcomes_opened,
        "state_sha256": selection.state_sha256,
    }


@dataclass(frozen=True, slots=True)
class OuterSupervisedSelectionRun:
    outer_target: str
    example_count: int
    core_results: tuple[TeacherRegretCandidateResult, ...]
    tuning_results: tuple[TeacherRegretCandidateResult, ...]
    selection: TeacherRegretSelection
    path: Path


def run_outer_supervised_selection(
    protocol: object,
    *,
    outer_target: str,
    bank_root: str | Path,
    work_root: str | Path,
    device: str,
    progress: Callable[[str], None] | None = None,
) -> OuterSupervisedSelectionRun:
    domain_order = tuple(getattr(protocol, "domain_order", ()))
    if (
        len(domain_order) != 6
        or outer_target not in domain_order
        or type(device) is not str
        or not device
    ):
        raise G1SelectionExecutionError("outer supervised selection request is invalid")
    source_domains = tuple(domain for domain in domain_order if domain != outer_target)
    banks = Path(bank_root)
    examples: list[G1PolicyTrainingExample] = []
    for source in source_domains:
        _identity, records = read_teacher_bank(banks / outer_target / f"{source}.parquet")
        if (
            not records
            or any(
                row.example.outer_target != outer_target
                or row.example.source_domain != source
                for row in records
            )
        ):
            raise G1SelectionExecutionError("outer teacher-bank identity changed")
        examples.extend(row.example for row in records)
    example_tuple = tuple(examples)
    destination = Path(work_root) / outer_target

    def load_or_fit(hp: PolicyTrainingHyperparameters) -> TeacherRegretCandidateResult:
        path = destination / f"{hp.state_sha256}.json"
        if path.exists():
            result = read_teacher_regret_candidate_result(path)
            if (
                result.outer_target != outer_target
                or result.hyperparameters != hp
            ):
                raise G1SelectionExecutionError("cached candidate request changed")
            if progress is not None:
                progress(f"G1 source candidate reused {outer_target}: {hp.state_sha256}")
            return result
        if progress is not None:
            progress(f"G1 source candidate fitting {outer_target}: {hp.state_sha256}")
        result = fit_teacher_regret_candidate(
            example_tuple,
            hp,
            max_epochs=int(protocol.epochs),
            patience=int(protocol.patience),
            device=device,
        )
        write_teacher_regret_candidate_result(path, result)
        if progress is not None:
            progress(
                f"G1 source candidate complete {outer_target}: "
                f"{result.equal_domain_mean_regret:.17g}"
            )
        return result

    core_results = tuple(load_or_fit(hp) for hp in core_policy_candidates())
    core_selection = select_teacher_regret_candidate(core_results)
    tuning_hyperparameters = tuning_policy_candidates(
        core_selection.selected_hyperparameters
    )
    existing = {row.hyperparameters.state_sha256 for row in core_results}
    tuning_results = tuple(
        load_or_fit(hp)
        for hp in tuning_hyperparameters
        if hp.state_sha256 not in existing
    )
    selection = select_teacher_regret_candidate((*core_results, *tuning_results))
    selection_path = destination / "selection.json"
    _atomic_json(
        selection_path,
        {
            "schema_version": 1,
            "scope": "inspection_agent_g1_outer_supervised_selection",
            "example_count": len(example_tuple),
            "core_candidate_sha256s": [row.state_sha256 for row in core_results],
            "core_winner_sha256": core_selection.state_sha256,
            "tuning_candidate_sha256s": [
                row.state_sha256 for row in tuning_results
            ],
            "selection": teacher_regret_selection_payload(selection),
        },
    )
    return OuterSupervisedSelectionRun(
        outer_target=outer_target,
        example_count=len(example_tuple),
        core_results=core_results,
        tuning_results=tuning_results,
        selection=selection,
        path=selection_path,
    )


__all__ = [
    "G1SelectionExecutionError",
    "OuterSupervisedSelectionRun",
    "TeacherRegretCandidateResult",
    "TeacherRegretSelection",
    "core_policy_candidates",
    "fit_teacher_regret_candidate",
    "read_teacher_regret_candidate_result",
    "run_outer_supervised_selection",
    "select_teacher_regret_candidate",
    "teacher_regret_candidate_payload",
    "teacher_regret_selection_payload",
    "tuning_policy_candidates",
    "write_teacher_regret_candidate_result",
]
