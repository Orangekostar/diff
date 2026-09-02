"""Six-domain paired analysis of frozen G1 target engineering curves."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .formal import FIXED_BASELINE_METHODS
from .g1 import (
    G1FinalDecision,
    PolicyGateEvidence,
    PolicyGateResult,
    TaskConditioningGateResult,
    evaluate_final_g1_decision,
    evaluate_policy_gate,
    evaluate_task_conditioning_gate,
)
from .metrics import EngineeringCurve, oracle_gap_closure
from .source_bridge import OuterFixedBridgeSelection
from .statistics import (
    FORMAL_BOOTSTRAP_SEED,
    G1PairedBootstrap,
    formal_synchronized_bootstrap,
)
from .target_evaluation import G1TargetCurveRecord
from .target_execution import TARGET_ACTION_VARIANTS, TargetPolicyVariant
from .target_reference import (
    TARGET_ORACLE_METHODS,
    G1TargetReferenceCurveRecord,
)


class G1TargetAnalysisError(ValueError):
    """Raised when six-domain target curve evidence is incomplete or unpaired."""


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class G1TaskCurveAnalysis:
    task: InspectionTask
    fixed_methods: tuple[tuple[str, str], ...]
    fixed_auebc: float
    learned_auebc: float
    oracle_auebc: float
    oracle_gap_closure: float | None
    baseline_minus_learned: G1PairedBootstrap
    policy_gate: PolicyGateResult
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        values = (self.fixed_auebc, self.learned_auebc, self.oracle_auebc)
        if (
            self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or type(self.fixed_methods) is not tuple
            or len(self.fixed_methods) != 6
            or len({domain for domain, _method in self.fixed_methods}) != 6
            or any(method not in FIXED_BASELINE_METHODS for _domain, method in self.fixed_methods)
            or any(not math.isfinite(value) or value < 0.0 for value in values)
            or (
                self.oracle_gap_closure is not None
                and not math.isfinite(float(self.oracle_gap_closure))
            )
            or type(self.baseline_minus_learned) is not G1PairedBootstrap
            or type(self.policy_gate) is not PolicyGateResult
            or self.policy_gate.task is not self.task
            or self.policy_gate.oracle_gap_closure != self.oracle_gap_closure
        ):
            raise G1TargetAnalysisError("target task analysis is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-task-curve-analysis",
                    "task": self.task.value,
                    "fixed_methods": self.fixed_methods,
                    "fixed_auebc": self.fixed_auebc,
                    "learned_auebc": self.learned_auebc,
                    "oracle_auebc": self.oracle_auebc,
                    "oracle_gap_closure": self.oracle_gap_closure,
                    "bootstrap": self.baseline_minus_learned.distribution_sha256,
                    "policy_gate": self.policy_gate.state_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1TaskConditioningAnalysis:
    field_wrong_minus_correct: G1PairedBootstrap
    field_no_task_minus_correct: G1PairedBootstrap
    cai_wrong_minus_correct: G1PairedBootstrap
    cai_no_task_minus_correct: G1PairedBootstrap
    gate: TaskConditioningGateResult
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.field_wrong_minus_correct,
            self.field_no_task_minus_correct,
            self.cai_wrong_minus_correct,
            self.cai_no_task_minus_correct,
        )
        if (
            any(type(value) is not G1PairedBootstrap for value in values)
            or type(self.gate) is not TaskConditioningGateResult
        ):
            raise G1TargetAnalysisError("task-conditioning analysis is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-task-conditioning-analysis",
                    "bootstrap": tuple(value.distribution_sha256 for value in values),
                    "gate": self.gate.state_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1SurfaceRobustnessAnalysis:
    field_no_surface_minus_correct: G1PairedBootstrap
    field_shuffled_surface_minus_correct: G1PairedBootstrap
    cai_no_surface_minus_correct: G1PairedBootstrap
    cai_shuffled_surface_minus_correct: G1PairedBootstrap
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.field_no_surface_minus_correct,
            self.field_shuffled_surface_minus_correct,
            self.cai_no_surface_minus_correct,
            self.cai_shuffled_surface_minus_correct,
        )
        if any(type(value) is not G1PairedBootstrap for value in values):
            raise G1TargetAnalysisError("surface-robustness analysis is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-surface-robustness-analysis",
                    "bootstrap": tuple(value.distribution_sha256 for value in values),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1TargetCurveAnalysis:
    field: G1TaskCurveAnalysis
    cai: G1TaskCurveAnalysis
    task_conditioning: G1TaskConditioningAnalysis
    surface_robustness: G1SurfaceRobustnessAnalysis
    final_decision: G1FinalDecision
    no_target_leakage: bool
    deterministic_replay: bool
    deployment_bridge_valid: bool
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.field) is not G1TaskCurveAnalysis
            or self.field.task is not InspectionTask.FIELD
            or type(self.cai) is not G1TaskCurveAnalysis
            or self.cai.task is not InspectionTask.CAI
            or type(self.task_conditioning) is not G1TaskConditioningAnalysis
            or type(self.surface_robustness) is not G1SurfaceRobustnessAnalysis
            or type(self.final_decision) is not G1FinalDecision
            or any(
                type(value) is not bool
                for value in (
                    self.no_target_leakage,
                    self.deterministic_replay,
                    self.deployment_bridge_valid,
                )
            )
        ):
            raise G1TargetAnalysisError("target curve analysis is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-curve-analysis",
                    "field": self.field.state_sha256,
                    "cai": self.cai.state_sha256,
                    "task_conditioning": self.task_conditioning.state_sha256,
                    "surface_robustness": self.surface_robustness.state_sha256,
                    "final_decision": self.final_decision.state_sha256,
                    "no_target_leakage": self.no_target_leakage,
                    "deterministic_replay": self.deterministic_replay,
                    "deployment_bridge_valid": self.deployment_bridge_valid,
                }
            ),
        )


def _curve_geometry(curve: EngineeringCurve) -> tuple[object, ...]:
    return (
        curve.target_domain,
        curve.specimen_sha256,
        curve.task,
        curve.grid_sha256,
        curve.evaluator_sha256,
        curve.warm_start_sha256,
        tuple(float(value) for value in curve.nominal_budgets),
    )


def _validate_rosters(
    learned: tuple[G1TargetCurveRecord, ...],
    references: tuple[G1TargetReferenceCurveRecord, ...],
    selections: tuple[OuterFixedBridgeSelection, ...],
) -> tuple[str, ...]:
    if (
        type(learned) is not tuple
        or not learned
        or any(type(row) is not G1TargetCurveRecord for row in learned)
        or type(references) is not tuple
        or not references
        or any(type(row) is not G1TargetReferenceCurveRecord for row in references)
        or type(selections) is not tuple
        or any(type(row) is not OuterFixedBridgeSelection for row in selections)
    ):
        raise G1TargetAnalysisError("target curve analysis roster is invalid")
    domains = tuple(sorted({row.outer_target for row in learned}))
    if (
        len(domains) != 6
        or {row.outer_target for row in references} != set(domains)
        or len(selections) != 12
        or {(row.outer_target, row.task) for row in selections}
        != {
            (domain, task)
            for domain in domains
            for task in (InspectionTask.FIELD, InspectionTask.CAI)
        }
        or any(set(row.source_domains) != set(domains) - {row.outer_target} for row in selections)
    ):
        raise G1TargetAnalysisError("target curve domain roster changed")
    relevant = tuple(row for row in learned if row.variant in TARGET_ACTION_VARIANTS)
    learned_keys = {
        (row.outer_target, row.specimen_id, row.task, row.variant) for row in relevant
    }
    reference_keys = {
        (row.outer_target, row.specimen_id, row.task, row.method) for row in references
    }
    if len(learned_keys) != len(relevant) or len(reference_keys) != len(references):
        raise G1TargetAnalysisError("target curve roster contains duplicates")
    for domain in domains:
        specimens = {
            row.specimen_id
            for row in relevant
            if row.outer_target == domain
            and row.task is InspectionTask.FIELD
            and row.variant is TargetPolicyVariant.PROPOSED
        }
        if not specimens:
            raise G1TargetAnalysisError("target curve specimen roster is empty")
        expected_learned = {
            (domain, specimen, task, variant)
            for specimen in specimens
            for task in (InspectionTask.FIELD, InspectionTask.CAI)
            for variant in TARGET_ACTION_VARIANTS
        }
        expected_references = {
            (domain, specimen, task, method)
            for specimen in specimens
            for task in (InspectionTask.FIELD, InspectionTask.CAI)
            for method in (*FIXED_BASELINE_METHODS, TARGET_ORACLE_METHODS[task])
        }
        if (
            {key for key in learned_keys if key[0] == domain} != expected_learned
            or {key for key in reference_keys if key[0] == domain}
            != expected_references
        ):
            raise G1TargetAnalysisError("target curve control roster changed")
        learned_domain = tuple(row for row in relevant if row.outer_target == domain)
        reference_domain = tuple(
            row for row in references if row.outer_target == domain
        )
        if (
            len({row.bank_seal_sha256 for row in (*learned_domain, *reference_domain)})
            != 1
            or any(
                row.truth_view_sha256
                != next(
                    value.truth_view_sha256
                    for value in learned_domain
                    if value.specimen_id == row.specimen_id
                )
                for row in reference_domain
            )
        ):
            raise G1TargetAnalysisError("target curve authority roster changed")
    return domains


def _paired_rows(
    learned: tuple[G1TargetCurveRecord, ...],
    references: tuple[G1TargetReferenceCurveRecord, ...],
    selections: tuple[OuterFixedBridgeSelection, ...],
    task: InspectionTask,
) -> tuple[
    tuple[tuple[str, str], ...],
    dict[tuple[str, str], G1TargetCurveRecord],
    dict[tuple[str, str], G1TargetReferenceCurveRecord],
    dict[tuple[str, str], G1TargetReferenceCurveRecord],
]:
    proposed = {
        (row.outer_target, row.specimen_id): row
        for row in learned
        if row.task is task and row.variant is TargetPolicyVariant.PROPOSED
    }
    fixed_methods = {
        row.outer_target: row.method for row in selections if row.task is task
    }
    fixed = {
        (row.outer_target, row.specimen_id): row
        for row in references
        if row.task is task and row.method == fixed_methods[row.outer_target]
    }
    oracle = {
        (row.outer_target, row.specimen_id): row
        for row in references
        if row.task is task and row.method == TARGET_ORACLE_METHODS[task]
    }
    keys = tuple(sorted(proposed))
    if not keys or set(keys) != set(fixed) or set(keys) != set(oracle):
        raise G1TargetAnalysisError("target task curve pairing changed")
    for key in keys:
        if len(
            {
                _curve_geometry(proposed[key].curve),
                _curve_geometry(fixed[key].curve),
                _curve_geometry(oracle[key].curve),
            }
        ) != 1:
            raise G1TargetAnalysisError("target task curve geometry changed")
        oracle_gap_closure(
            fixed=fixed[key].curve,
            learned=proposed[key].curve,
            oracle=oracle[key].curve,
        )
    return keys, proposed, fixed, oracle


def _equal_domain_mean(
    keys: tuple[tuple[str, str], ...],
    values: dict[tuple[str, str], float],
) -> float:
    domains = tuple(sorted({domain for domain, _specimen in keys}))
    return float(
        np.mean(
            [
                np.mean(
                    [values[key] for key in keys if key[0] == domain],
                    dtype=np.float64,
                )
                for domain in domains
            ],
            dtype=np.float64,
        )
    )


def _bootstrap(
    keys: tuple[tuple[str, str], ...],
    baseline: dict[tuple[str, str], float],
    learned: dict[tuple[str, str], float],
) -> G1PairedBootstrap:
    return formal_synchronized_bootstrap(
        dataset_ids=tuple(domain for domain, _specimen in keys),
        specimen_ids=tuple(specimen for _domain, specimen in keys),
        baseline_values=tuple(baseline[key] for key in keys),
        learned_values=tuple(learned[key] for key in keys),
        seed=FORMAL_BOOTSTRAP_SEED,
    )


def _variant_values(
    records: tuple[G1TargetCurveRecord, ...],
    *,
    task: InspectionTask,
    variant: TargetPolicyVariant,
) -> dict[tuple[str, str], float]:
    return {
        (row.outer_target, row.specimen_id): row.curve.auebc
        for row in records
        if row.task is task and row.variant is variant
    }


def _task_analysis(
    learned: tuple[G1TargetCurveRecord, ...],
    references: tuple[G1TargetReferenceCurveRecord, ...],
    selections: tuple[OuterFixedBridgeSelection, ...],
    *,
    task: InspectionTask,
    no_target_leakage: bool,
    deterministic_replay: bool,
) -> G1TaskCurveAnalysis:
    keys, proposed, fixed, oracle = _paired_rows(
        learned, references, selections, task
    )
    proposed_values = {key: proposed[key].curve.auebc for key in keys}
    fixed_values = {key: fixed[key].curve.auebc for key in keys}
    oracle_values = {key: oracle[key].curve.auebc for key in keys}
    fixed_mean = _equal_domain_mean(keys, fixed_values)
    learned_mean = _equal_domain_mean(keys, proposed_values)
    oracle_mean = _equal_domain_mean(keys, oracle_values)
    bootstrap = _bootstrap(keys, fixed_values, proposed_values)
    gate = evaluate_policy_gate(
        PolicyGateEvidence(
            task=task,
            fixed_auebc=fixed_mean,
            learned_auebc=learned_mean,
            oracle_auebc=oracle_mean,
            baseline_minus_learned=bootstrap,
            no_target_leakage=no_target_leakage,
            deterministic_replay=deterministic_replay,
        )
    )
    return G1TaskCurveAnalysis(
        task=task,
        fixed_methods=tuple(
            sorted(
                (row.outer_target, row.method)
                for row in selections
                if row.task is task
            )
        ),
        fixed_auebc=fixed_mean,
        learned_auebc=learned_mean,
        oracle_auebc=oracle_mean,
        oracle_gap_closure=gate.oracle_gap_closure,
        baseline_minus_learned=bootstrap,
        policy_gate=gate,
    )


def analyze_g1_target_curves(
    learned: tuple[G1TargetCurveRecord, ...],
    references: tuple[G1TargetReferenceCurveRecord, ...],
    fixed_selections: tuple[OuterFixedBridgeSelection, ...],
    *,
    no_target_leakage: bool,
    deterministic_replay: bool,
    deployment_bridge_valid: bool,
) -> G1TargetCurveAnalysis:
    if any(
        type(value) is not bool
        for value in (
            no_target_leakage,
            deterministic_replay,
            deployment_bridge_valid,
        )
    ):
        raise G1TargetAnalysisError("target analysis integrity flags are invalid")
    _validate_rosters(learned, references, fixed_selections)
    field = _task_analysis(
        learned,
        references,
        fixed_selections,
        task=InspectionTask.FIELD,
        no_target_leakage=no_target_leakage,
        deterministic_replay=deterministic_replay,
    )
    cai = _task_analysis(
        learned,
        references,
        fixed_selections,
        task=InspectionTask.CAI,
        no_target_leakage=no_target_leakage,
        deterministic_replay=deterministic_replay,
    )
    keys = tuple(
        sorted(
            _variant_values(
                learned,
                task=InspectionTask.FIELD,
                variant=TargetPolicyVariant.PROPOSED,
            )
        )
    )
    proposed = {
        task: _variant_values(
            learned,
            task=task,
            variant=TargetPolicyVariant.PROPOSED,
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    }
    task_bootstrap = {
        (task, variant): _bootstrap(
            keys,
            _variant_values(learned, task=task, variant=variant),
            proposed[task],
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        for variant in (TargetPolicyVariant.WRONG_TASK, TargetPolicyVariant.NO_TASK)
    }
    task_gate = evaluate_task_conditioning_gate(
        field_wrong_minus_correct=task_bootstrap[
            (InspectionTask.FIELD, TargetPolicyVariant.WRONG_TASK)
        ],
        field_no_task_minus_correct=task_bootstrap[
            (InspectionTask.FIELD, TargetPolicyVariant.NO_TASK)
        ],
        cai_wrong_minus_correct=task_bootstrap[
            (InspectionTask.CAI, TargetPolicyVariant.WRONG_TASK)
        ],
        cai_no_task_minus_correct=task_bootstrap[
            (InspectionTask.CAI, TargetPolicyVariant.NO_TASK)
        ],
    )
    task_conditioning = G1TaskConditioningAnalysis(
        field_wrong_minus_correct=task_bootstrap[
            (InspectionTask.FIELD, TargetPolicyVariant.WRONG_TASK)
        ],
        field_no_task_minus_correct=task_bootstrap[
            (InspectionTask.FIELD, TargetPolicyVariant.NO_TASK)
        ],
        cai_wrong_minus_correct=task_bootstrap[
            (InspectionTask.CAI, TargetPolicyVariant.WRONG_TASK)
        ],
        cai_no_task_minus_correct=task_bootstrap[
            (InspectionTask.CAI, TargetPolicyVariant.NO_TASK)
        ],
        gate=task_gate,
    )
    surface_bootstrap = {
        (task, variant): _bootstrap(
            keys,
            _variant_values(learned, task=task, variant=variant),
            proposed[task],
        )
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        for variant in (
            TargetPolicyVariant.NO_SURFACE,
            TargetPolicyVariant.SHUFFLED_SURFACE,
        )
    }
    surface = G1SurfaceRobustnessAnalysis(
        field_no_surface_minus_correct=surface_bootstrap[
            (InspectionTask.FIELD, TargetPolicyVariant.NO_SURFACE)
        ],
        field_shuffled_surface_minus_correct=surface_bootstrap[
            (InspectionTask.FIELD, TargetPolicyVariant.SHUFFLED_SURFACE)
        ],
        cai_no_surface_minus_correct=surface_bootstrap[
            (InspectionTask.CAI, TargetPolicyVariant.NO_SURFACE)
        ],
        cai_shuffled_surface_minus_correct=surface_bootstrap[
            (InspectionTask.CAI, TargetPolicyVariant.SHUFFLED_SURFACE)
        ],
    )
    decision = evaluate_final_g1_decision(
        deployment_bridge_valid=deployment_bridge_valid,
        field_policy_go=field.policy_gate.passed,
        cai_policy_go=cai.policy_gate.passed,
        task_conditioning_go=task_conditioning.gate.passed,
    )
    return G1TargetCurveAnalysis(
        field=field,
        cai=cai,
        task_conditioning=task_conditioning,
        surface_robustness=surface,
        final_decision=decision,
        no_target_leakage=no_target_leakage,
        deterministic_replay=deterministic_replay,
        deployment_bridge_valid=deployment_bridge_valid,
    )


__all__ = [
    "G1SurfaceRobustnessAnalysis",
    "G1TargetAnalysisError",
    "G1TargetCurveAnalysis",
    "G1TaskConditioningAnalysis",
    "G1TaskCurveAnalysis",
    "analyze_g1_target_curves",
]
