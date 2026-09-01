"""Registered G1 component gates and final decision vocabulary."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .statistics import (
    FORMAL_BOOTSTRAP_REPLICATES,
    FORMAL_BOOTSTRAP_SEED,
    G1PairedBootstrap,
)

POLICY_GAP_CLOSURE_MINIMUM = 0.20
POLICY_IMPROVED_DOMAINS_MINIMUM = 4
STOPPING_SAVING_MINIMUM = 0.10
STOPPING_TASK_LOSS_RATIO_MAXIMUM = 1.05
STOPPING_PREMATURE_RATE_MAXIMUM = 0.05

FINAL_G1_STATUSES = (
    "G1_TASK_CONDITIONED_POLICY_GO",
    "G1_ACTIVE_POLICY_GO",
    "G1_CAI_ONLY_POLICY_GO",
    "G1_FIELD_ONLY_POLICY_GO",
    "G1_POLICY_OBSERVABILITY_NO_GO",
    "G1_DEPLOYMENT_BRIDGE_NO_GO",
)
G2_AUTHORIZING_STATUSES = FINAL_G1_STATUSES[:4]


class G1GateError(ValueError):
    """Raised when G1 gate evidence is contradictory or incomplete."""


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _validate_bootstrap(value: G1PairedBootstrap) -> None:
    if (
        type(value) is not G1PairedBootstrap
        or value.replicates != FORMAL_BOOTSTRAP_REPLICATES
        or value.seed != FORMAL_BOOTSTRAP_SEED
        or len(value.domain_effects) != 6
        or len({domain for domain, _effect in value.domain_effects}) != 6
        or not 0 <= value.improved_domains <= 6
        or not all(
            math.isfinite(number)
            for number in (value.point_estimate, value.ci_lower, value.ci_upper)
        )
        or value.ci_lower > value.point_estimate
        or value.point_estimate > value.ci_upper
    ):
        raise G1GateError("formal bootstrap evidence is invalid")


@dataclass(frozen=True, slots=True)
class PolicyGateEvidence:
    task: InspectionTask
    fixed_auebc: float
    learned_auebc: float
    oracle_auebc: float
    baseline_minus_learned: G1PairedBootstrap
    no_target_leakage: bool
    deterministic_replay: bool


@dataclass(frozen=True, slots=True)
class PolicyGateResult:
    task: InspectionTask
    status: str
    passed: bool
    oracle_gap_closure: float | None
    baseline_effect: float
    ci_lower: float
    improved_domains: int
    no_target_leakage: bool
    deterministic_replay: bool
    state_sha256: str


def evaluate_policy_gate(evidence: PolicyGateEvidence) -> PolicyGateResult:
    if type(evidence) is not PolicyGateEvidence:
        raise G1GateError("issued policy-gate evidence is required")
    fixed = float(evidence.fixed_auebc)
    learned = float(evidence.learned_auebc)
    oracle = float(evidence.oracle_auebc)
    _validate_bootstrap(evidence.baseline_minus_learned)
    effect = fixed - learned
    if (
        evidence.task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or not all(math.isfinite(value) and value >= 0.0 for value in (fixed, learned, oracle))
        or not math.isclose(
            effect,
            evidence.baseline_minus_learned.point_estimate,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
        or type(evidence.no_target_leakage) is not bool
        or type(evidence.deterministic_replay) is not bool
    ):
        raise G1GateError("policy-gate evidence is invalid")
    denominator = fixed - oracle
    closure = effect / denominator if denominator > 0.0 else None
    statistical = (
        learned < fixed
        and evidence.baseline_minus_learned.ci_lower > 0.0
        and evidence.baseline_minus_learned.improved_domains
        >= POLICY_IMPROVED_DOMAINS_MINIMUM
        and evidence.no_target_leakage
        and evidence.deterministic_replay
    )
    passed = (
        statistical
        and closure is not None
        and closure >= POLICY_GAP_CLOSURE_MINIMUM
    )
    prefix = f"G1_{evidence.task.value}"
    if passed:
        status = f"{prefix}_POLICY_GO"
    elif statistical and closure is not None and closure < POLICY_GAP_CLOSURE_MINIMUM:
        status = f"{prefix}_DESCRIPTIVE_POLICY_SIGNAL_ONLY"
    else:
        status = f"{prefix}_POLICY_NO_GO"
    payload = {
        "schema": 1,
        "kind": "g1-policy-gate",
        "task": evidence.task.value,
        "status": status,
        "passed": passed,
        "fixed_auebc": fixed,
        "learned_auebc": learned,
        "oracle_auebc": oracle,
        "oracle_gap_closure": closure,
        "bootstrap": evidence.baseline_minus_learned.distribution_sha256,
        "no_target_leakage": evidence.no_target_leakage,
        "deterministic_replay": evidence.deterministic_replay,
    }
    return PolicyGateResult(
        task=evidence.task,
        status=status,
        passed=passed,
        oracle_gap_closure=closure,
        baseline_effect=effect,
        ci_lower=evidence.baseline_minus_learned.ci_lower,
        improved_domains=evidence.baseline_minus_learned.improved_domains,
        no_target_leakage=evidence.no_target_leakage,
        deterministic_replay=evidence.deterministic_replay,
        state_sha256=_json_sha(payload),
    )


@dataclass(frozen=True, slots=True)
class TaskConditioningGateResult:
    status: str
    passed: bool
    state_sha256: str


def evaluate_task_conditioning_gate(
    *,
    field_wrong_minus_correct: G1PairedBootstrap,
    field_no_task_minus_correct: G1PairedBootstrap,
    cai_wrong_minus_correct: G1PairedBootstrap,
    cai_no_task_minus_correct: G1PairedBootstrap,
) -> TaskConditioningGateResult:
    values = (
        field_wrong_minus_correct,
        field_no_task_minus_correct,
        cai_wrong_minus_correct,
        cai_no_task_minus_correct,
    )
    for value in values:
        _validate_bootstrap(value)
    passed = all(
        value.ci_lower > 0.0
        and value.improved_domains >= POLICY_IMPROVED_DOMAINS_MINIMUM
        for value in values
    )
    status = "G1_TASK_CONDITIONING_GO" if passed else "G1_TASK_CONDITIONING_NO_GO"
    payload = {
        "schema": 1,
        "kind": "g1-task-conditioning-gate",
        "status": status,
        "bootstrap": tuple(value.distribution_sha256 for value in values),
    }
    return TaskConditioningGateResult(
        status=status,
        passed=passed,
        state_sha256=_json_sha(payload),
    )


@dataclass(frozen=True, slots=True)
class StopGateEvidence:
    task: InspectionTask
    normalized_measurement_saving: float
    task_loss_ratio: float
    premature_stop_rate: float
    saving_bootstrap: G1PairedBootstrap


@dataclass(frozen=True, slots=True)
class StopGateResult:
    task: InspectionTask
    status: str
    passed: bool
    state_sha256: str


def evaluate_stop_gate(evidence: StopGateEvidence) -> StopGateResult:
    if type(evidence) is not StopGateEvidence:
        raise G1GateError("issued STOP-gate evidence is required")
    _validate_bootstrap(evidence.saving_bootstrap)
    saving = float(evidence.normalized_measurement_saving)
    ratio = float(evidence.task_loss_ratio)
    premature = float(evidence.premature_stop_rate)
    if (
        evidence.task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or not math.isfinite(saving)
        or not 0.0 <= saving <= 1.0
        or not math.isfinite(ratio)
        or ratio < 0.0
        or not math.isfinite(premature)
        or not 0.0 <= premature <= 1.0
        or not math.isclose(
            saving,
            evidence.saving_bootstrap.point_estimate,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
    ):
        raise G1GateError("STOP-gate evidence is invalid")
    passed = (
        saving >= STOPPING_SAVING_MINIMUM
        and ratio <= STOPPING_TASK_LOSS_RATIO_MAXIMUM
        and premature <= STOPPING_PREMATURE_RATE_MAXIMUM
        and evidence.saving_bootstrap.improved_domains
        >= POLICY_IMPROVED_DOMAINS_MINIMUM
        and evidence.saving_bootstrap.ci_lower > 0.0
    )
    status = (
        f"G1_{evidence.task.value}_STOPPING_GO"
        if passed
        else f"G1_{evidence.task.value}_STOPPING_NO_GO"
    )
    payload = {
        "schema": 1,
        "kind": "g1-stop-gate",
        "task": evidence.task.value,
        "status": status,
        "saving": saving,
        "task_loss_ratio": ratio,
        "premature_stop_rate": premature,
        "bootstrap": evidence.saving_bootstrap.distribution_sha256,
    }
    return StopGateResult(
        task=evidence.task,
        status=status,
        passed=passed,
        state_sha256=_json_sha(payload),
    )


@dataclass(frozen=True, slots=True)
class G1FinalDecision:
    status: str
    g2_authorized: bool
    field_policy_go: bool
    cai_policy_go: bool
    task_conditioning_go: bool
    deployment_bridge_valid: bool
    state_sha256: str


def evaluate_final_g1_decision(
    *,
    deployment_bridge_valid: bool,
    field_policy_go: bool,
    cai_policy_go: bool,
    task_conditioning_go: bool,
) -> G1FinalDecision:
    values = (
        deployment_bridge_valid,
        field_policy_go,
        cai_policy_go,
        task_conditioning_go,
    )
    if any(type(value) is not bool for value in values):
        raise G1GateError("final G1 decision flags must be boolean")
    if not deployment_bridge_valid:
        status = "G1_DEPLOYMENT_BRIDGE_NO_GO"
    elif field_policy_go and cai_policy_go and task_conditioning_go:
        status = "G1_TASK_CONDITIONED_POLICY_GO"
    elif field_policy_go and cai_policy_go:
        status = "G1_ACTIVE_POLICY_GO"
    elif field_policy_go:
        status = "G1_FIELD_ONLY_POLICY_GO"
    elif cai_policy_go:
        status = "G1_CAI_ONLY_POLICY_GO"
    else:
        status = "G1_POLICY_OBSERVABILITY_NO_GO"
    payload = {
        "schema": 1,
        "kind": "g1-final-decision",
        "status": status,
        "deployment_bridge_valid": deployment_bridge_valid,
        "field_policy_go": field_policy_go,
        "cai_policy_go": cai_policy_go,
        "task_conditioning_go": task_conditioning_go,
    }
    return G1FinalDecision(
        status=status,
        g2_authorized=status in G2_AUTHORIZING_STATUSES,
        field_policy_go=field_policy_go,
        cai_policy_go=cai_policy_go,
        task_conditioning_go=task_conditioning_go,
        deployment_bridge_valid=deployment_bridge_valid,
        state_sha256=_json_sha(payload),
    )


__all__ = [
    "FINAL_G1_STATUSES",
    "G2_AUTHORIZING_STATUSES",
    "POLICY_GAP_CLOSURE_MINIMUM",
    "POLICY_IMPROVED_DOMAINS_MINIMUM",
    "STOPPING_PREMATURE_RATE_MAXIMUM",
    "STOPPING_SAVING_MINIMUM",
    "STOPPING_TASK_LOSS_RATIO_MAXIMUM",
    "G1FinalDecision",
    "G1GateError",
    "PolicyGateEvidence",
    "PolicyGateResult",
    "StopGateEvidence",
    "StopGateResult",
    "TaskConditioningGateResult",
    "evaluate_final_g1_decision",
    "evaluate_policy_gate",
    "evaluate_stop_gate",
    "evaluate_task_conditioning_gate",
]
