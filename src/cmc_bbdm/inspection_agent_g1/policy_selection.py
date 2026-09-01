"""Source-only inner validation and outer-fold policy freezing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from statistics import median

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .contracts import CAIContextMode, TaskTokenMode
from .policy_training import (
    REGISTERED_MAX_EPOCHS,
    PolicyModelName,
    PolicyTrainingHyperparameters,
    TrainingRoute,
)

SELECTION_PRIORITY = (
    "equal_domain_equal_task_mean_relative_engineering_auebc",
    "mean_oracle_gap_closure",
    "improved_source_validation_domains",
    "simpler_model_and_fewer_stages",
)


class G1PolicySelectionError(ValueError):
    """Raised when policy selection sees a target or inconsistent source bridge."""


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


@dataclass(frozen=True, slots=True)
class InnerPolicyEngineeringMetric:
    outer_target: str
    validation_domain: str
    task: InspectionTask
    hyperparameters_sha256: str
    model_state_sha256: str
    learned_auebc: float
    fixed_auebc: float
    oracle_auebc: float
    selected_epoch: int
    relative_engineering_auebc: float = field(init=False)
    oracle_gap_closure: float | None = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        learned = float(self.learned_auebc)
        fixed = float(self.fixed_auebc)
        oracle = float(self.oracle_auebc)
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.validation_domain) is not str
            or not self.validation_domain
            or self.validation_domain == self.outer_target
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or not _valid_sha256(self.hyperparameters_sha256)
            or not _valid_sha256(self.model_state_sha256)
            or not all(math.isfinite(value) and value >= 0.0 for value in (learned, fixed, oracle))
            or fixed <= 0.0
            or type(self.selected_epoch) is not int
            or not 1 <= self.selected_epoch <= REGISTERED_MAX_EPOCHS
        ):
            message = (
                "outer target is forbidden from inner policy selection"
                if self.validation_domain == self.outer_target
                else "inner policy engineering metric is invalid"
            )
            raise G1PolicySelectionError(message)
        relative = learned / fixed
        denominator = fixed - oracle
        closure = (fixed - learned) / denominator if denominator > 0.0 else None
        payload = {
            "schema": 1,
            "kind": "g1-inner-policy-engineering-metric",
            "outer_target": self.outer_target,
            "validation_domain": self.validation_domain,
            "task": self.task.value,
            "hyperparameters": self.hyperparameters_sha256,
            "model": self.model_state_sha256,
            "learned_auebc": learned,
            "fixed_auebc": fixed,
            "oracle_auebc": oracle,
            "relative_engineering_auebc": relative,
            "oracle_gap_closure": closure,
            "selected_epoch": self.selected_epoch,
        }
        object.__setattr__(self, "learned_auebc", learned)
        object.__setattr__(self, "fixed_auebc", fixed)
        object.__setattr__(self, "oracle_auebc", oracle)
        object.__setattr__(self, "relative_engineering_auebc", relative)
        object.__setattr__(self, "oracle_gap_closure", closure)
        object.__setattr__(self, "state_sha256", _json_sha(payload))


@dataclass(frozen=True, slots=True)
class PolicyCandidateEvaluation:
    hyperparameters: PolicyTrainingHyperparameters
    inner_metrics: tuple[InnerPolicyEngineeringMetric, ...]
    outer_target: str = field(init=False)
    source_validation_domains: tuple[str, ...] = field(init=False)
    equal_domain_mean_relative_auebc: float = field(init=False)
    mean_oracle_gap_closure: float | None = field(init=False)
    improved_source_domains: int = field(init=False)
    final_refit_epochs: int = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.hyperparameters) is not PolicyTrainingHyperparameters
            or type(self.inner_metrics) is not tuple
            or len(self.inner_metrics) != 10
            or any(
                type(metric) is not InnerPolicyEngineeringMetric
                for metric in self.inner_metrics
            )
            or len({metric.state_sha256 for metric in self.inner_metrics}) != 10
            or any(
                metric.hyperparameters_sha256
                != self.hyperparameters.state_sha256
                for metric in self.inner_metrics
            )
            or len({metric.outer_target for metric in self.inner_metrics}) != 1
        ):
            raise G1PolicySelectionError("policy candidate evaluation is invalid")
        outer_target = self.inner_metrics[0].outer_target
        domains = tuple(sorted({metric.validation_domain for metric in self.inner_metrics}))
        if len(domains) != 5 or outer_target in domains:
            raise G1PolicySelectionError("policy candidate inner-domain roster is invalid")
        expected_tasks = {InspectionTask.FIELD, InspectionTask.CAI}
        epochs: list[int] = []
        for domain in domains:
            rows = tuple(
                metric
                for metric in self.inner_metrics
                if metric.validation_domain == domain
            )
            if (
                {metric.task for metric in rows} != expected_tasks
                or len(rows) != 2
                or len({metric.model_state_sha256 for metric in rows}) != 1
                or len({metric.selected_epoch for metric in rows}) != 1
            ):
                raise G1PolicySelectionError(
                    "each inner domain must evaluate both tasks from one joint actor"
                )
            epochs.append(rows[0].selected_epoch)
        relative = float(
            np.mean(
                [metric.relative_engineering_auebc for metric in self.inner_metrics],
                dtype=np.float64,
            )
        )
        closures = tuple(
            metric.oracle_gap_closure
            for metric in self.inner_metrics
            if metric.oracle_gap_closure is not None
        )
        mean_closure = (
            None
            if not closures
            else float(np.mean(np.asarray(closures, dtype=np.float64)))
        )
        improved = sum(
            float(
                np.mean(
                    [
                        metric.relative_engineering_auebc
                        for metric in self.inner_metrics
                        if metric.validation_domain == domain
                    ],
                    dtype=np.float64,
                )
            )
            < 1.0
            for domain in domains
        )
        refit_epochs = int(median(epochs))
        payload = {
            "schema": 1,
            "kind": "g1-policy-candidate-evaluation",
            "hyperparameters": self.hyperparameters.state_sha256,
            "outer_target": outer_target,
            "source_validation_domains": domains,
            "inner_metrics": tuple(
                metric.state_sha256
                for metric in sorted(
                    self.inner_metrics,
                    key=lambda value: (
                        value.validation_domain,
                        value.task.value,
                    ),
                )
            ),
            "equal_domain_mean_relative_auebc": relative,
            "mean_oracle_gap_closure": mean_closure,
            "improved_source_domains": improved,
            "final_refit_epochs": refit_epochs,
        }
        object.__setattr__(self, "outer_target", outer_target)
        object.__setattr__(self, "source_validation_domains", domains)
        object.__setattr__(self, "equal_domain_mean_relative_auebc", relative)
        object.__setattr__(self, "mean_oracle_gap_closure", mean_closure)
        object.__setattr__(self, "improved_source_domains", improved)
        object.__setattr__(self, "final_refit_epochs", refit_epochs)
        object.__setattr__(self, "state_sha256", _json_sha(payload))


@dataclass(frozen=True, slots=True)
class OuterPolicySelection:
    outer_target: str
    source_validation_domains: tuple[str, ...]
    candidates: tuple[PolicyCandidateEvaluation, ...]
    selected_hyperparameters_sha256: str
    final_refit_epochs: int
    selection_priority: tuple[str, ...]
    target_outcomes_opened: bool
    state_sha256: str


def _complexity(candidate: PolicyCandidateEvaluation) -> tuple[int, ...]:
    hyperparameters = candidate.hyperparameters
    return (
        int(
            hyperparameters.model_name
            is PolicyModelName.STRUCTURED_INSPECTION_POLICY
        ),
        hyperparameters.dagger_iterations,
        int(hyperparameters.route is TrainingRoute.SOFT_UTILITY_DISTILL),
        int(
            hyperparameters.cai_context_mode
            is CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT
        ),
        int(hyperparameters.task_token_mode is TaskTokenMode.CORRECT),
    )


def _selection_key(candidate: PolicyCandidateEvaluation) -> tuple[object, ...]:
    closure_key = (
        math.inf
        if candidate.mean_oracle_gap_closure is None
        else -candidate.mean_oracle_gap_closure
    )
    return (
        candidate.equal_domain_mean_relative_auebc,
        closure_key,
        -candidate.improved_source_domains,
        _complexity(candidate),
        candidate.hyperparameters.state_sha256,
    )


def select_outer_policy(
    candidates: tuple[PolicyCandidateEvaluation, ...],
) -> OuterPolicySelection:
    if (
        type(candidates) is not tuple
        or not candidates
        or any(type(candidate) is not PolicyCandidateEvaluation for candidate in candidates)
        or len({candidate.state_sha256 for candidate in candidates}) != len(candidates)
        or len({candidate.hyperparameters.state_sha256 for candidate in candidates})
        != len(candidates)
        or len({candidate.outer_target for candidate in candidates}) != 1
        or len({candidate.source_validation_domains for candidate in candidates}) != 1
        or len(
            {
                candidate.hyperparameters.task_token_mode
                for candidate in candidates
            }
        )
        != 1
    ):
        raise G1PolicySelectionError("outer policy candidate roster is invalid")
    bridge_by_candidate: list[dict[tuple[str, str], tuple[float, float]]] = []
    for candidate in candidates:
        bridge_by_candidate.append(
            {
                (metric.validation_domain, metric.task.value): (
                    metric.fixed_auebc,
                    metric.oracle_auebc,
                )
                for metric in candidate.inner_metrics
            }
        )
    if any(bridge != bridge_by_candidate[0] for bridge in bridge_by_candidate[1:]):
        raise G1PolicySelectionError("policy candidates do not share one fixed/oracle bridge")
    ordered = tuple(sorted(candidates, key=lambda value: value.hyperparameters.state_sha256))
    selected = min(ordered, key=_selection_key)
    outer_target = selected.outer_target
    domains = selected.source_validation_domains
    payload = {
        "schema": 1,
        "kind": "g1-outer-policy-selection",
        "outer_target": outer_target,
        "source_validation_domains": domains,
        "candidates": tuple(candidate.state_sha256 for candidate in ordered),
        "selected_hyperparameters": selected.hyperparameters.state_sha256,
        "final_refit_epochs": selected.final_refit_epochs,
        "selection_priority": SELECTION_PRIORITY,
        "target_outcomes_opened": False,
    }
    return OuterPolicySelection(
        outer_target=outer_target,
        source_validation_domains=domains,
        candidates=ordered,
        selected_hyperparameters_sha256=selected.hyperparameters.state_sha256,
        final_refit_epochs=selected.final_refit_epochs,
        selection_priority=SELECTION_PRIORITY,
        target_outcomes_opened=False,
        state_sha256=_json_sha(payload),
    )


def outer_selection_payload(selection: OuterPolicySelection) -> dict[str, object]:
    if type(selection) is not OuterPolicySelection or selection.target_outcomes_opened:
        raise G1PolicySelectionError("issued source-only outer selection is required")
    return {
        "schema_version": 1,
        "scope": "inspection_agent_g1_outer_source_selection",
        "outer_target": selection.outer_target,
        "source_validation_domains": list(selection.source_validation_domains),
        "selection_priority": list(selection.selection_priority),
        "target_outcomes_opened": False,
        "selected_hyperparameters_sha256": selection.selected_hyperparameters_sha256,
        "final_refit_epochs": selection.final_refit_epochs,
        "selection_state_sha256": selection.state_sha256,
        "candidates": [
            {
                "hyperparameters": {
                    "model_name": candidate.hyperparameters.model_name.value,
                    "route": candidate.hyperparameters.route.value,
                    "cai_context_mode": candidate.hyperparameters.cai_context_mode.value,
                    "task_token_mode": candidate.hyperparameters.task_token_mode.value,
                    "tau": candidate.hyperparameters.tau,
                    "learning_rate": candidate.hyperparameters.learning_rate,
                    "weight_decay": candidate.hyperparameters.weight_decay,
                    "dagger_iterations": candidate.hyperparameters.dagger_iterations,
                    "state_sha256": candidate.hyperparameters.state_sha256,
                },
                "equal_domain_mean_relative_auebc": candidate.equal_domain_mean_relative_auebc,
                "mean_oracle_gap_closure": candidate.mean_oracle_gap_closure,
                "improved_source_domains": candidate.improved_source_domains,
                "final_refit_epochs": candidate.final_refit_epochs,
                "state_sha256": candidate.state_sha256,
                "inner_metrics": [
                    {
                        "validation_domain": metric.validation_domain,
                        "task": metric.task.value,
                        "model_state_sha256": metric.model_state_sha256,
                        "learned_auebc": metric.learned_auebc,
                        "fixed_auebc": metric.fixed_auebc,
                        "oracle_auebc": metric.oracle_auebc,
                        "relative_engineering_auebc": metric.relative_engineering_auebc,
                        "oracle_gap_closure": metric.oracle_gap_closure,
                        "selected_epoch": metric.selected_epoch,
                        "state_sha256": metric.state_sha256,
                    }
                    for metric in sorted(
                        candidate.inner_metrics,
                        key=lambda value: (
                            value.validation_domain,
                            value.task.value,
                        ),
                    )
                ],
            }
            for candidate in selection.candidates
        ],
    }


__all__ = [
    "SELECTION_PRIORITY",
    "G1PolicySelectionError",
    "InnerPolicyEngineeringMetric",
    "OuterPolicySelection",
    "PolicyCandidateEvaluation",
    "outer_selection_payload",
    "select_outer_policy",
]
