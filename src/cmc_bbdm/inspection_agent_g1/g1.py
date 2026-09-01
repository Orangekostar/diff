"""Frozen G1 protocol, component gates, and final decision vocabulary."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .statistics import (
    FORMAL_BOOTSTRAP_REPLICATES,
    FORMAL_BOOTSTRAP_SEED,
    G1PairedBootstrap,
)

_G1_CONFIG_SHA256 = "aaf216ab9033fffc390f123131bca29952b0d5ca34752434aa9686c1d7ff1f05"
_G1_BASE_SHA = "7a10cd425de582fa158bf6639285731ccd8ff7a7"
_G1_PROMPT_SHA256 = "37b7e4b9860dd338589cf2d9dd8f2cc8d202451a84eeafbc4961b2c5a28fcda5"


class G1ExecutionError(RuntimeError):
    """Raised before execution when the frozen G1 authority cannot be honored."""


@dataclass(frozen=True, slots=True)
class G1Protocol:
    config_path: Path
    config_sha256: str
    specimen_count: int
    domain_order: tuple[str, ...]
    domain_counts: MappingProxyType
    endpoint_budget: float
    deployment_initial_nominal_budget: float
    warm_start_k: int
    warm_start_cells: tuple[int, ...]
    evaluation_checkpoints: tuple[float, ...]
    snapshot_fractions: tuple[float, ...]
    oracle_checkpoints: tuple[float, ...]
    teacher_bank_seed: int
    teacher_temperatures: tuple[float, ...]
    model_candidates: tuple[str, ...]
    learning_rates: tuple[float, ...]
    weight_decays: tuple[float, ...]
    dagger_iterations: tuple[int, ...]
    epochs: int
    patience: int
    batch_specimens: int
    gradient_clip_norm: float
    initialization_seed: int
    stop_thresholds: tuple[float, ...]
    bootstrap_replicates: int
    bootstrap_seed: int
    default_device: str
    encoder_batch_size: int
    formal_output: str
    replay_output: str
    work_output: str
    source_bindings: MappingProxyType


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise G1ExecutionError(f"{label} must be a mapping")
    return value


def _numbers(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise G1ExecutionError(f"{label} must be a nonempty list")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise G1ExecutionError(f"{label} is invalid") from error
    if not all(math.isfinite(item) for item in result):
        raise G1ExecutionError(f"{label} is invalid")
    return result


def load_g1_protocol(
    path: str | Path,
    *,
    project_root: str | Path,
) -> G1Protocol:
    root = Path(project_root).resolve(strict=True)
    config_path = Path(path).resolve(strict=True)
    try:
        payload = config_path.read_bytes()
    except OSError as error:
        raise G1ExecutionError("G1 protocol config is unavailable") from error
    config_sha = hashlib.sha256(payload).hexdigest()
    if config_sha != _G1_CONFIG_SHA256:
        raise G1ExecutionError("G1 config SHA-256 changed")
    try:
        loaded = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise G1ExecutionError("G1 protocol config cannot be decoded") from error
    config = _mapping(loaded, "G1 config")
    if (
        config.get("schema_version") != 1
        or config.get("stage") != "INSPECTION_AGENT_G1"
        or config.get("mode") != "formal"
        or config.get("repository_base_sha") != _G1_BASE_SHA
        or config.get("controlling_prompt_sha256") != _G1_PROMPT_SHA256
        or config.get("configuration_frozen") is not True
    ):
        raise G1ExecutionError("G1 protocol identity changed")

    cohort = _mapping(config.get("cohort"), "G1 cohort")
    acquisition = _mapping(config.get("acquisition"), "G1 acquisition")
    teacher_bank = _mapping(config.get("teacher_bank"), "G1 teacher bank")
    distribution = _mapping(
        config.get("teacher_distribution"), "G1 teacher distribution"
    )
    models = _mapping(config.get("models"), "G1 models")
    training = _mapping(config.get("training"), "G1 training")
    stopping = _mapping(config.get("stopping"), "G1 stopping")
    statistics = _mapping(config.get("statistics"), "G1 statistics")
    execution = _mapping(config.get("execution"), "G1 execution")

    domain_order = tuple(str(value) for value in cohort.get("domain_order", ()))
    domain_counts_raw = _mapping(cohort.get("domain_counts"), "G1 domain counts")
    domain_counts = {str(key): int(value) for key, value in domain_counts_raw.items()}
    checkpoints = _numbers(
        acquisition.get("evaluation_checkpoints"), "G1 evaluation checkpoints"
    )
    snapshot_fractions = _numbers(
        teacher_bank.get("continuation_snapshot_fractions"),
        "G1 teacher snapshots",
    )
    oracle_checkpoints = _numbers(
        teacher_bank.get("oracle_checkpoints"), "G1 oracle checkpoints"
    )
    temperatures = _numbers(
        distribution.get("temperatures"), "G1 teacher temperatures"
    )
    learning_rates = _numbers(training.get("learning_rates"), "G1 learning rates")
    weight_decays = _numbers(training.get("weight_decays"), "G1 weight decays")
    stop_thresholds = _numbers(
        stopping.get("threshold_candidates"), "G1 STOP thresholds"
    )
    try:
        dagger_iterations = tuple(int(value) for value in training["dagger_iterations"])
        warm_cells = tuple(int(value) for value in acquisition["warm_start_primary_cells"])
        model_candidates = tuple(str(value) for value in models["candidates"])
    except (KeyError, TypeError, ValueError) as error:
        raise G1ExecutionError("G1 registered roster is invalid") from error
    if (
        cohort.get("specimen_count") != 276
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or set(domain_counts) != set(domain_order)
        or sum(domain_counts.values()) != 276
        or checkpoints != (0.0, 0.0625, 0.125, 0.1875, 0.25)
        or snapshot_fractions != (1 / 3, 2 / 3, 1.0)
        or oracle_checkpoints != (0.0625, 0.125, 0.1875, 0.25)
        or temperatures != (0.25, 0.5, 1.0, 2.0)
        or model_candidates != ("SharedActionMLP", "StructuredInspectionPolicy")
        or learning_rates != (0.0001, 0.0003)
        or weight_decays != (0.0001, 0.001)
        or dagger_iterations != (0, 1, 2)
        or stop_thresholds != (0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99)
    ):
        raise G1ExecutionError("G1 protocol roster changed")

    sources = _mapping(config.get("sources"), "G1 sources")
    source_bindings: dict[str, tuple[str, str]] = {}
    for name, raw in sorted(sources.items()):
        binding = _mapping(raw, f"G1 source {name}")
        if set(binding) != {"path", "sha256"}:
            raise G1ExecutionError("G1 source binding schema changed")
        relative = Path(str(binding["path"]))
        expected = str(binding["sha256"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(expected) != 64
            or set(expected) - set("0123456789abcdef")
        ):
            raise G1ExecutionError("G1 source binding is invalid")
        try:
            source_payload = (root / relative).read_bytes()
        except OSError as error:
            raise G1ExecutionError(f"G1 source is unavailable: {name}") from error
        if hashlib.sha256(source_payload).hexdigest() != expected:
            raise G1ExecutionError(f"G1 source SHA-256 mismatch: {name}")
        source_bindings[str(name)] = (relative.as_posix(), expected)

    return G1Protocol(
        config_path=config_path,
        config_sha256=config_sha,
        specimen_count=276,
        domain_order=domain_order,
        domain_counts=MappingProxyType(domain_counts),
        endpoint_budget=float(acquisition["endpoint_budget"]),
        deployment_initial_nominal_budget=float(
            acquisition["deployment_initial_nominal_budget"]
        ),
        warm_start_k=int(acquisition["warm_start_primary_k"]),
        warm_start_cells=warm_cells,
        evaluation_checkpoints=checkpoints,
        snapshot_fractions=snapshot_fractions,
        oracle_checkpoints=oracle_checkpoints,
        teacher_bank_seed=int(teacher_bank["random_seed"]),
        teacher_temperatures=temperatures,
        model_candidates=model_candidates,
        learning_rates=learning_rates,
        weight_decays=weight_decays,
        dagger_iterations=dagger_iterations,
        epochs=int(training["epochs"]),
        patience=int(training["early_stopping_patience"]),
        batch_specimens=int(training["batch_specimens"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        initialization_seed=int(training["initialization_seed"]),
        stop_thresholds=stop_thresholds,
        bootstrap_replicates=int(statistics["bootstrap_replicates"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        default_device=str(execution["default_device"]),
        encoder_batch_size=int(execution["encoder_batch_size"]),
        formal_output=str(execution["formal_output"]),
        replay_output=str(execution["replay_output"]),
        work_output=str(execution["work_output"]),
        source_bindings=MappingProxyType(source_bindings),
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
    "G1ExecutionError",
    "G1FinalDecision",
    "G1GateError",
    "G1Protocol",
    "PolicyGateEvidence",
    "PolicyGateResult",
    "StopGateEvidence",
    "StopGateResult",
    "TaskConditioningGateResult",
    "evaluate_final_g1_decision",
    "evaluate_policy_gate",
    "evaluate_stop_gate",
    "evaluate_task_conditioning_gate",
    "load_g1_protocol",
]
