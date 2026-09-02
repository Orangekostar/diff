"""Conditionally authorized source-only privileged advantage training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .contracts import ACTION_SLOT_COUNT
from .policy_training import (
    REGISTERED_AAWR_BETAS,
    REGISTERED_AAWR_EXPECTILES,
    REGISTERED_BATCH_SPECIMENS,
    REGISTERED_EARLY_STOPPING_PATIENCE,
    REGISTERED_GRADIENT_CLIP_NORM,
    REGISTERED_INITIALIZATION_SEED,
    REGISTERED_MAX_EPOCHS,
    G1ActorNormalizer,
    G1PolicyTrainingExample,
    PolicyFitAudit,
    PolicyTrainingHyperparameters,
    TrainedObservablePolicy,
    TrainingRoute,
    _deterministic_torch,
    _model_hash,
)

REGISTERED_EXPECTILES = REGISTERED_AAWR_EXPECTILES
REGISTERED_BETAS = REGISTERED_AAWR_BETAS
REGISTERED_MAXIMUM_ADVANTAGE_WEIGHT = 100.0
AAWR_GAP_CLOSURE_LIMIT = 0.20
PRIVILEGED_CRITIC_FEATURE_DIMENSION = 515
AAWR_TRAJECTORY_SOURCES = frozenset(
    (
        "WARM_START",
        "UNIFORM_CONTINUE",
        "SURFACE_FOCUS_CONTINUE",
        "RANDOM_CONTINUE",
        "ALTERNATE_BROADEN_REFINE",
        "ORACLE_CHECKPOINT",
        "DAGGER_ACTOR_VISITED",
    )
)


class PrivilegedAWRError(ValueError):
    """Raised when AAWR evidence or training tensors violate the source-only gate."""


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _finite_nonnegative(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) and value >= 0.0 for value in values)


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


@dataclass(frozen=True, slots=True)
class AAWRSourceEvidence:
    outer_target: str
    task: InspectionTask
    source_domains: tuple[str, ...]
    fixed_auebc: tuple[float, ...]
    policy_auebc: tuple[float, ...]
    oracle_auebc: tuple[float, ...]
    equal_domain_effect: float = field(init=False)
    oracle_gap_available: float = field(init=False)
    oracle_gap_closure: float | None = field(init=False)
    improved_domains: int = field(init=False)
    positive_action_observability: bool = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(
            tuple(float(value) for value in values)
            for values in (self.fixed_auebc, self.policy_auebc, self.oracle_auebc)
        )
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or type(self.source_domains) is not tuple
            or len(self.source_domains) != 5
            or len(set(self.source_domains)) != 5
            or any(type(domain) is not str or not domain for domain in self.source_domains)
            or self.outer_target in self.source_domains
            or any(len(values) != 5 or not _finite_nonnegative(values) for values in rows)
        ):
            message = (
                "outer target is forbidden from AAWR source evidence"
                if self.outer_target in self.source_domains
                else "AAWR source evidence is invalid"
            )
            raise PrivilegedAWRError(message)
        fixed, policy, oracle = rows
        effects = tuple(left - right for left, right in zip(fixed, policy, strict=True))
        gaps = tuple(left - right for left, right in zip(fixed, oracle, strict=True))
        effect = float(np.mean(effects, dtype=np.float64))
        gap = float(np.mean(gaps, dtype=np.float64))
        improved = sum(value > 0.0 for value in effects)
        closure = None if gap <= 0.0 else effect / gap
        positive = effect > 0.0 and improved >= 3
        state = _json_sha(
            {
                "schema": 1,
                "kind": "g1-aawr-source-evidence",
                "outer_target": self.outer_target,
                "task": self.task.value,
                "source_domains": self.source_domains,
                "fixed_auebc": fixed,
                "policy_auebc": policy,
                "oracle_auebc": oracle,
                "equal_domain_effect": effect,
                "oracle_gap_available": gap,
                "oracle_gap_closure": closure,
                "improved_domains": improved,
                "positive_action_observability": positive,
            }
        )
        object.__setattr__(self, "fixed_auebc", fixed)
        object.__setattr__(self, "policy_auebc", policy)
        object.__setattr__(self, "oracle_auebc", oracle)
        object.__setattr__(self, "equal_domain_effect", effect)
        object.__setattr__(self, "oracle_gap_available", gap)
        object.__setattr__(self, "oracle_gap_closure", closure)
        object.__setattr__(self, "improved_domains", improved)
        object.__setattr__(self, "positive_action_observability", positive)
        object.__setattr__(self, "state_sha256", state)


@dataclass(frozen=True, slots=True)
class AAWRAuthorization:
    outer_target: str
    status: str
    prerequisite_satisfied: bool
    authorized_tasks: tuple[InspectionTask, ...]
    evidence: tuple[AAWRSourceEvidence, ...]
    state_sha256: str


def authorize_conditional_aawr(
    evidence: tuple[AAWRSourceEvidence, ...],
    *,
    soft_distillation_with_dagger: bool = True,
) -> AAWRAuthorization:
    if (
        type(evidence) is not tuple
        or not 1 <= len(evidence) <= 2
        or any(type(row) is not AAWRSourceEvidence for row in evidence)
        or len({row.task for row in evidence}) != len(evidence)
        or len({row.outer_target for row in evidence}) != 1
        or type(soft_distillation_with_dagger) is not bool
    ):
        raise PrivilegedAWRError("AAWR authorization evidence is invalid")
    evidence = tuple(
        sorted(
            evidence,
            key=lambda row: (InspectionTask.FIELD, InspectionTask.CAI).index(row.task),
        )
    )
    authorized = tuple(
        row.task
        for row in evidence
        if soft_distillation_with_dagger
        and row.positive_action_observability
        and row.oracle_gap_closure is not None
        and 0.0 < row.oracle_gap_closure < AAWR_GAP_CLOSURE_LIMIT
    )
    status = "AUTHORIZED_SOURCE_ONLY" if authorized else "NOT_RUN_NOT_AUTHORIZED"
    outer_target = evidence[0].outer_target
    payload = {
        "schema": 1,
        "kind": "g1-aawr-authorization",
        "outer_target": outer_target,
        "status": status,
        "prerequisite_satisfied": soft_distillation_with_dagger,
        "authorized_tasks": tuple(task.value for task in authorized),
        "evidence": tuple(row.state_sha256 for row in evidence),
    }
    return AAWRAuthorization(
        outer_target=outer_target,
        status=status,
        prerequisite_satisfied=soft_distillation_with_dagger,
        authorized_tasks=authorized,
        evidence=evidence,
        state_sha256=_json_sha(payload),
    )


def aawr_policy_hyperparameters(
    base: PolicyTrainingHyperparameters,
    *,
    authorization_sha256: str,
    authorized_tasks: tuple[InspectionTask, ...],
    expectile: float,
    beta: float,
) -> PolicyTrainingHyperparameters:
    tasks = tuple(
        task
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        if task in authorized_tasks
    )
    if (
        type(base) is not PolicyTrainingHyperparameters
        or base.route is not TrainingRoute.SOFT_UTILITY_DISTILL
        or base.dagger_iterations <= 0
        or not _valid_sha256(authorization_sha256)
        or type(authorized_tasks) is not tuple
        or not authorized_tasks
        or tasks != authorized_tasks
        or len(set(tasks)) != len(tasks)
        or float(expectile) not in REGISTERED_EXPECTILES
        or float(beta) not in REGISTERED_BETAS
    ):
        raise PrivilegedAWRError(
            "AAWR requires an authorized soft-distillation + DAgger source policy"
        )
    return PolicyTrainingHyperparameters(
        model_name=base.model_name,
        route=TrainingRoute.PRIVILEGED_AAWR,
        cai_context_mode=base.cai_context_mode,
        task_token_mode=base.task_token_mode,
        tau=base.tau,
        learning_rate=base.learning_rate,
        weight_decay=base.weight_decay,
        dagger_iterations=base.dagger_iterations,
        aawr_expectile=float(expectile),
        aawr_beta=float(beta),
        aawr_authorized_tasks=tasks,
        aawr_authorization_sha256=authorization_sha256,
        base_hyperparameters_sha256=base.state_sha256,
    )


def advantage_weights(
    advantages: torch.Tensor,
    *,
    beta: float,
    maximum_weight: float,
) -> torch.Tensor:
    temperature = float(beta)
    limit = float(maximum_weight)
    if (
        not isinstance(advantages, torch.Tensor)
        or advantages.ndim != 1
        or not advantages.is_floating_point()
        or not torch.isfinite(advantages).all()
        or temperature not in REGISTERED_BETAS
        or limit != REGISTERED_MAXIMUM_ADVANTAGE_WEIGHT
    ):
        raise PrivilegedAWRError("AAWR advantage-weight request is invalid")
    return torch.exp(advantages / temperature).clamp(max=limit)


def expectile_value_loss(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
    *,
    expectile: float,
) -> torch.Tensor:
    level = float(expectile)
    if (
        not isinstance(predicted_values, torch.Tensor)
        or not isinstance(target_values, torch.Tensor)
        or predicted_values.ndim != 1
        or target_values.shape != predicted_values.shape
        or not predicted_values.is_floating_point()
        or not target_values.is_floating_point()
        or not torch.isfinite(predicted_values).all()
        or not torch.isfinite(target_values).all()
        or level not in REGISTERED_EXPECTILES
    ):
        raise PrivilegedAWRError("AAWR expectile tensors are invalid")
    residual = target_values - predicted_values
    weights = torch.where(residual > 0.0, level, 1.0 - level)
    return torch.mean(weights * residual.square())


def advantage_weighted_actor_loss(
    action_logits: torch.Tensor,
    behavior_slots: torch.Tensor,
    legal_action_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    beta: float,
    maximum_weight: float,
) -> torch.Tensor:
    if (
        not isinstance(action_logits, torch.Tensor)
        or not isinstance(behavior_slots, torch.Tensor)
        or not isinstance(legal_action_mask, torch.Tensor)
        or not isinstance(advantages, torch.Tensor)
        or action_logits.ndim != 2
        or action_logits.shape[1] != ACTION_SLOT_COUNT
        or behavior_slots.shape != (action_logits.shape[0],)
        or behavior_slots.dtype is not torch.long
        or legal_action_mask.shape != action_logits.shape
        or legal_action_mask.dtype is not torch.bool
        or advantages.shape != (action_logits.shape[0],)
        or not action_logits.is_floating_point()
        or len(
            {
                action_logits.device,
                behavior_slots.device,
                legal_action_mask.device,
                advantages.device,
            }
        )
        != 1
        or torch.any(behavior_slots < 0)
        or torch.any(behavior_slots >= ACTION_SLOT_COUNT)
        or not torch.isfinite(action_logits[legal_action_mask]).all()
    ):
        raise PrivilegedAWRError("AAWR actor tensors are invalid")
    rows = torch.arange(action_logits.shape[0], device=action_logits.device)
    if not torch.all(legal_action_mask[rows, behavior_slots]):
        raise PrivilegedAWRError("AAWR behavior action must be legal")
    log_probabilities = torch.log_softmax(
        action_logits.masked_fill(~legal_action_mask, -torch.inf), dim=1
    )
    losses = -log_probabilities[rows, behavior_slots]
    weights = advantage_weights(
        advantages,
        beta=beta,
        maximum_weight=maximum_weight,
    ).detach()
    return torch.sum(weights * losses) / torch.sum(weights)


def privileged_critic_features(
    full_scan_embedding: object,
    *,
    task: InspectionTask,
    current_task_loss: float,
    true_cai: float | None,
) -> np.ndarray:
    embedding = np.asarray(full_scan_embedding, dtype=np.float64)
    loss = float(current_task_loss)
    cai = 0.0 if true_cai is None else float(true_cai)
    if (
        embedding.shape != (512,)
        or not np.all(np.isfinite(embedding))
        or task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or not math.isfinite(loss)
        or loss < 0.0
        or (task is InspectionTask.CAI and true_cai is None)
        or (task is InspectionTask.FIELD and true_cai is not None)
        or not math.isfinite(cai)
    ):
        raise PrivilegedAWRError("privileged critic feature request is invalid")
    values = np.concatenate(
        (
            embedding,
            np.asarray((loss, cai, float(true_cai is not None)), dtype=np.float64),
        )
    )
    values.setflags(write=False)
    return values


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class G1AAWRTrainingRecord:
    """One source state with queried one-step transitions and privileged critic data."""

    example: G1PolicyTrainingExample
    state_source: str
    source_record_sha256: str
    full_scan_embedding: np.ndarray
    true_cai: float | None
    current_task_loss: float = field(init=False)
    privileged_features: np.ndarray = field(init=False)
    transition_slots: np.ndarray = field(init=False)
    transition_rewards: np.ndarray = field(init=False)
    transition_exact_added_costs: np.ndarray = field(init=False)
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        embedding = np.ascontiguousarray(
            self.full_scan_embedding, dtype=np.float64
        )
        cai = None if self.true_cai is None else float(self.true_cai)
        if (
            type(self.example) is not G1PolicyTrainingExample
            or self.state_source not in AAWR_TRAJECTORY_SOURCES
            or not _valid_sha256(self.source_record_sha256)
            or embedding.shape != (512,)
            or not np.all(np.isfinite(embedding))
            or (self.example.task is InspectionTask.FIELD and cai is not None)
            or (
                self.example.task is InspectionTask.CAI
                and (cai is None or not math.isfinite(cai))
            )
        ):
            raise PrivilegedAWRError("AAWR source transition record is invalid")
        candidates = self.example.teacher_label.candidates
        losses = tuple(
            float(candidate.raw_value + candidate.task_loss_after)
            for candidate in candidates
        )
        current_loss = losses[0]
        if (
            not math.isfinite(current_loss)
            or current_loss < 0.0
            or any(
                not math.isclose(
                    value,
                    current_loss,
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-12,
                )
                for value in losses[1:]
            )
        ):
            raise PrivilegedAWRError("AAWR candidate rewards do not share one state loss")
        slots = np.ascontiguousarray(
            [candidate.slot for candidate in candidates], dtype=np.int64
        )
        rewards = np.ascontiguousarray(
            [candidate.raw_value for candidate in candidates], dtype=np.float64
        )
        costs = np.ascontiguousarray(
            [candidate.exact_added_cost for candidate in candidates], dtype=np.int64
        )
        for array in (embedding, slots, rewards, costs):
            array.setflags(write=False)
        features = privileged_critic_features(
            embedding,
            task=self.example.task,
            current_task_loss=current_loss,
            true_cai=cai,
        )
        payload = {
            "schema": 1,
            "kind": "g1-aawr-source-transition-record",
            "example": self.example.state_sha256,
            "state_source": self.state_source,
            "source_record": self.source_record_sha256,
            "full_scan_embedding": _array_sha(embedding),
            "true_cai": cai,
            "current_task_loss": current_loss,
            "slots": tuple(int(value) for value in slots),
            "rewards": tuple(float(value) for value in rewards),
            "exact_added_costs": tuple(int(value) for value in costs),
            "privileged_features": _array_sha(features),
            "target_outcomes_opened": False,
        }
        object.__setattr__(self, "full_scan_embedding", embedding)
        object.__setattr__(self, "true_cai", cai)
        object.__setattr__(self, "current_task_loss", current_loss)
        object.__setattr__(self, "privileged_features", features)
        object.__setattr__(self, "transition_slots", slots)
        object.__setattr__(self, "transition_rewards", rewards)
        object.__setattr__(self, "transition_exact_added_costs", costs)
        object.__setattr__(self, "state_sha256", _json_sha(payload))


class PrivilegedValueNetwork(nn.Module):
    """Small source-training critic kept physically separate from the actor."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128 + PRIVILEGED_CRITIC_FEATURE_DIMENSION, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        observable_context: torch.Tensor,
        privileged_features: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not isinstance(observable_context, torch.Tensor)
            or not isinstance(privileged_features, torch.Tensor)
            or observable_context.ndim != 2
            or observable_context.shape[1] != 128
            or privileged_features.shape
            != (observable_context.shape[0], PRIVILEGED_CRITIC_FEATURE_DIMENSION)
            or not observable_context.is_floating_point()
            or not privileged_features.is_floating_point()
            or observable_context.device != privileged_features.device
            or not torch.isfinite(observable_context).all()
            or not torch.isfinite(privileged_features).all()
        ):
            raise PrivilegedAWRError("privileged critic tensors are invalid")
        return self.network(torch.cat((observable_context, privileged_features), dim=1)).squeeze(-1)


@dataclass(frozen=True, slots=True)
class AAWRPolicyFitEvidence:
    outer_target: str
    validation_domain: str | None
    fit_domains: tuple[str, ...]
    hyperparameters_sha256: str
    base_model_sha256: str
    actor_model_sha256: str
    critic_model_sha256: str
    privileged_normalizer_sha256: str
    source_record_sha256s: tuple[str, ...]
    trajectory_source_counts: tuple[tuple[str, int], ...]
    critic_epochs_run: int
    critic_selected_epoch: int
    critic_objectives: tuple[float, ...]
    best_validation_expectile_loss: float | None
    target_outcomes_opened: bool = False
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or (
                self.validation_domain is not None
                and (
                    type(self.validation_domain) is not str
                    or self.validation_domain in self.fit_domains
                )
            )
            or type(self.fit_domains) is not tuple
            or not self.fit_domains
            or self.outer_target in self.fit_domains
            or not all(
                _valid_sha256(value)
                for value in (
                    self.hyperparameters_sha256,
                    self.base_model_sha256,
                    self.actor_model_sha256,
                    self.critic_model_sha256,
                    self.privileged_normalizer_sha256,
                    *self.source_record_sha256s,
                )
            )
            or not self.source_record_sha256s
            or type(self.trajectory_source_counts) is not tuple
            or not self.trajectory_source_counts
            or any(
                source not in AAWR_TRAJECTORY_SOURCES
                or type(count) is not int
                or count <= 0
                for source, count in self.trajectory_source_counts
            )
            or type(self.critic_epochs_run) is not int
            or type(self.critic_selected_epoch) is not int
            or not 1 <= self.critic_selected_epoch <= self.critic_epochs_run
            or self.critic_epochs_run != len(self.critic_objectives)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in self.critic_objectives
            )
            or (
                self.validation_domain is None
                and self.best_validation_expectile_loss is not None
            )
            or (
                self.validation_domain is not None
                and (
                    self.best_validation_expectile_loss is None
                    or not math.isfinite(self.best_validation_expectile_loss)
                    or self.best_validation_expectile_loss < 0.0
                )
            )
            or self.target_outcomes_opened
        ):
            raise PrivilegedAWRError("AAWR fit evidence is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-aawr-policy-fit-evidence",
                    "outer_target": self.outer_target,
                    "validation_domain": self.validation_domain,
                    "fit_domains": self.fit_domains,
                    "hyperparameters": self.hyperparameters_sha256,
                    "base_model": self.base_model_sha256,
                    "actor_model": self.actor_model_sha256,
                    "critic_model": self.critic_model_sha256,
                    "privileged_normalizer": self.privileged_normalizer_sha256,
                    "source_records": self.source_record_sha256s,
                    "trajectory_sources": self.trajectory_source_counts,
                    "critic_epochs_run": self.critic_epochs_run,
                    "critic_selected_epoch": self.critic_selected_epoch,
                    "critic_objectives": self.critic_objectives,
                    "best_validation_expectile_loss": (
                        self.best_validation_expectile_loss
                    ),
                    "target_outcomes_opened": False,
                }
            ),
        )


def _aawr_fit_evidence_payload(
    evidence: AAWRPolicyFitEvidence,
) -> dict[str, object]:
    if type(evidence) is not AAWRPolicyFitEvidence:
        raise PrivilegedAWRError("issued AAWR fit evidence is required")
    return {
        "schema_version": 1,
        "scope": "inspection_agent_g1_aawr_fit_evidence",
        "outer_target": evidence.outer_target,
        "validation_domain": evidence.validation_domain,
        "fit_domains": list(evidence.fit_domains),
        "hyperparameters_sha256": evidence.hyperparameters_sha256,
        "base_model_sha256": evidence.base_model_sha256,
        "actor_model_sha256": evidence.actor_model_sha256,
        "critic_model_sha256": evidence.critic_model_sha256,
        "privileged_normalizer_sha256": (
            evidence.privileged_normalizer_sha256
        ),
        "source_record_sha256s": list(evidence.source_record_sha256s),
        "trajectory_source_counts": [
            [source, count] for source, count in evidence.trajectory_source_counts
        ],
        "critic_epochs_run": evidence.critic_epochs_run,
        "critic_selected_epoch": evidence.critic_selected_epoch,
        "critic_objectives": list(evidence.critic_objectives),
        "best_validation_expectile_loss": (
            evidence.best_validation_expectile_loss
        ),
        "target_outcomes_opened": False,
        "state_sha256": evidence.state_sha256,
    }


def write_aawr_fit_evidence(
    path: str | Path,
    evidence: AAWRPolicyFitEvidence,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(
            (
                json.dumps(
                    _aawr_fit_evidence_payload(evidence),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("ascii")
        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    replay = read_aawr_fit_evidence(destination)
    if replay.state_sha256 != evidence.state_sha256:
        raise PrivilegedAWRError("written AAWR fit evidence did not replay")
    return destination


def read_aawr_fit_evidence(path: str | Path) -> AAWRPolicyFitEvidence:
    try:
        payload = json.loads(Path(path).read_bytes())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("scope")
            != "inspection_agent_g1_aawr_fit_evidence"
            or payload.get("target_outcomes_opened") is not False
        ):
            raise PrivilegedAWRError("AAWR fit evidence schema changed")
        evidence = AAWRPolicyFitEvidence(
            outer_target=str(payload["outer_target"]),
            validation_domain=(
                None
                if payload["validation_domain"] is None
                else str(payload["validation_domain"])
            ),
            fit_domains=tuple(str(value) for value in payload["fit_domains"]),
            hyperparameters_sha256=str(payload["hyperparameters_sha256"]),
            base_model_sha256=str(payload["base_model_sha256"]),
            actor_model_sha256=str(payload["actor_model_sha256"]),
            critic_model_sha256=str(payload["critic_model_sha256"]),
            privileged_normalizer_sha256=str(
                payload["privileged_normalizer_sha256"]
            ),
            source_record_sha256s=tuple(
                str(value) for value in payload["source_record_sha256s"]
            ),
            trajectory_source_counts=tuple(
                (str(source), int(count))
                for source, count in payload["trajectory_source_counts"]
            ),
            critic_epochs_run=int(payload["critic_epochs_run"]),
            critic_selected_epoch=int(payload["critic_selected_epoch"]),
            critic_objectives=tuple(
                float(value) for value in payload["critic_objectives"]
            ),
            best_validation_expectile_loss=(
                None
                if payload["best_validation_expectile_loss"] is None
                else float(payload["best_validation_expectile_loss"])
            ),
            target_outcomes_opened=False,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        if isinstance(error, PrivilegedAWRError):
            raise
        raise PrivilegedAWRError("AAWR fit evidence is unreadable") from error
    if evidence.state_sha256 != payload.get("state_sha256"):
        raise PrivilegedAWRError("AAWR fit evidence hash changed")
    return evidence


@dataclass(frozen=True, slots=True)
class _AAWRPolicyArrays:
    reconstruction_embedding: np.ndarray
    global_scalars: np.ndarray
    task_token: np.ndarray
    cell_features: np.ndarray
    candidate_features: np.ndarray
    legal_action_mask: np.ndarray
    rewards: np.ndarray
    utilities: np.ndarray
    privileged_features: np.ndarray


def _ordered_aawr_records(
    records: tuple[G1AAWRTrainingRecord, ...],
) -> tuple[G1AAWRTrainingRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1AAWRTrainingRecord for row in records)
        or len({row.state_sha256 for row in records}) != len(records)
        or len({row.example.outer_target for row in records}) != 1
    ):
        raise PrivilegedAWRError("AAWR training roster is invalid")
    return tuple(
        sorted(
            records,
            key=lambda row: (
                row.example.source_domain,
                row.example.specimen_sha256,
                row.example.task.value,
                row.example.dagger_iteration,
                row.example.policy_state.state_sha256,
            ),
        )
    )


def _state_weights(
    records: tuple[G1AAWRTrainingRecord, ...],
    indices: np.ndarray,
) -> np.ndarray:
    selected = tuple(records[int(index)] for index in indices)
    tasks_by_specimen: dict[str, set[InspectionTask]] = {}
    states = Counter(
        (row.example.specimen_sha256, row.example.task) for row in selected
    )
    for row in selected:
        tasks_by_specimen.setdefault(row.example.specimen_sha256, set()).add(
            row.example.task
        )
    task_sets = {tuple(sorted(tasks, key=lambda task: task.value)) for tasks in tasks_by_specimen.values()}
    if len(task_sets) != 1:
        raise PrivilegedAWRError("AAWR specimens do not share one authorized task roster")
    specimen_count = len(tasks_by_specimen)
    task_count = len(next(iter(task_sets)))
    weights = np.zeros(len(records), dtype=np.float64)
    for index in indices:
        row = records[int(index)]
        weights[int(index)] = (
            1.0
            / specimen_count
            / task_count
            / states[(row.example.specimen_sha256, row.example.task)]
        )
    if not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1.0e-12):
        raise PrivilegedAWRError("AAWR state weights do not sum to one")
    return weights


def _privileged_normalizer(
    records: tuple[G1AAWRTrainingRecord, ...],
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    values = np.stack(
        [records[int(index)].privileged_features for index in fit_indices]
    )
    mean = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, dtype=np.float64)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "kind": "g1-aawr-privileged-normalizer",
                "fit_records": tuple(
                    records[int(index)].state_sha256 for index in fit_indices
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name, array in (("mean", mean), ("scale", scale)):
        value = np.ascontiguousarray(array, dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return mean, scale, digest.hexdigest()


def _aawr_arrays(
    records: tuple[G1AAWRTrainingRecord, ...],
    normalizer: G1ActorNormalizer,
    *,
    privileged_mean: np.ndarray,
    privileged_scale: np.ndarray,
) -> _AAWRPolicyArrays:
    transformed = tuple(normalizer.transform(row.example.policy_state) for row in records)
    rewards = np.zeros((len(records), ACTION_SLOT_COUNT), dtype=np.float32)
    utilities = np.zeros_like(rewards)
    for row_index, row in enumerate(records):
        rewards[row_index, row.transition_slots] = row.transition_rewards
        utilities[row_index, row.transition_slots] = np.asarray(
            [candidate.objective_value for candidate in row.example.teacher_label.candidates],
            dtype=np.float32,
        )
    return _AAWRPolicyArrays(
        reconstruction_embedding=np.stack(
            [value.reconstruction_embedding for value in transformed]
        ),
        global_scalars=np.stack([value.global_scalars for value in transformed]),
        task_token=np.stack([value.task_token for value in transformed]),
        cell_features=np.stack([value.cell_features for value in transformed]),
        candidate_features=np.stack(
            [value.candidate_features for value in transformed]
        ),
        legal_action_mask=np.stack(
            [value.legal_action_mask for value in transformed]
        ),
        rewards=rewards,
        utilities=utilities,
        privileged_features=np.asarray(
            [
                (row.privileged_features - privileged_mean) / privileged_scale
                for row in records
            ],
            dtype=np.float32,
        ),
    )


def _tensor(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.from_numpy(values[indices]).to(device)


def _actor_output(
    model: nn.Module,
    arrays: _AAWRPolicyArrays,
    indices: np.ndarray,
    *,
    device: torch.device,
):
    return model(
        _tensor(arrays.reconstruction_embedding, indices, device=device),
        _tensor(arrays.global_scalars, indices, device=device),
        _tensor(arrays.task_token, indices, device=device),
        _tensor(arrays.cell_features, indices, device=device),
        _tensor(arrays.candidate_features, indices, device=device),
        _tensor(arrays.legal_action_mask, indices, device=device),
    )


def _specimen_batches(
    records: tuple[G1AAWRTrainingRecord, ...],
    indices: np.ndarray,
    *,
    epoch: int,
    seed: int,
    batch_specimens: int,
) -> tuple[np.ndarray, ...]:
    specimens = np.asarray(
        sorted({records[int(index)].example.specimen_sha256 for index in indices}),
        dtype=object,
    )
    order = np.random.Generator(np.random.PCG64(seed + epoch)).permutation(specimens)
    specimen_by_row = np.asarray(
        [row.example.specimen_sha256 for row in records], dtype=object
    )
    available = np.zeros(len(records), dtype=np.bool_)
    available[indices] = True
    return tuple(
        np.flatnonzero(
            available
            & np.isin(
                specimen_by_row,
                tuple(order[start : start + batch_specimens]),
            )
        )
        for start in range(0, len(order), batch_specimens)
    )


def _critic_loss(
    values: torch.Tensor,
    rewards: torch.Tensor,
    legal_mask: torch.Tensor,
    state_weights: torch.Tensor,
    *,
    expectile: float,
) -> torch.Tensor:
    residual = rewards - values.unsqueeze(1)
    expectile_weights = torch.where(
        residual > 0.0,
        float(expectile),
        1.0 - float(expectile),
    )
    per_state = torch.sum(
        torch.where(legal_mask, expectile_weights * residual.square(), 0.0),
        dim=1,
    ) / torch.sum(legal_mask, dim=1)
    return torch.sum(per_state * state_weights) / torch.sum(state_weights)


def _actor_loss(
    action_logits: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    legal_mask: torch.Tensor,
    state_weights: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    advantages = rewards - values.unsqueeze(1)
    weights = torch.exp(advantages / float(beta)).clamp(
        max=REGISTERED_MAXIMUM_ADVANTAGE_WEIGHT
    )
    log_probabilities = torch.log_softmax(
        action_logits.masked_fill(~legal_mask, -torch.inf), dim=1
    )
    legal_log_probabilities = torch.where(
        legal_mask, log_probabilities, 0.0
    )
    weighted_nll = -weights * legal_log_probabilities
    per_state = torch.sum(weighted_nll, dim=1) / torch.sum(
        torch.where(legal_mask, weights, 0.0), dim=1
    )
    return torch.sum(per_state * state_weights) / torch.sum(state_weights)


def _validation_regret(
    model: nn.Module,
    arrays: _AAWRPolicyArrays,
    indices: np.ndarray,
    weights: np.ndarray,
    *,
    device: torch.device,
) -> float:
    total = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), 64):
            batch = indices[start : start + 64]
            output = _actor_output(model, arrays, batch, device=device)
            mask = _tensor(arrays.legal_action_mask, batch, device=device)
            utilities = _tensor(arrays.utilities, batch, device=device)
            probabilities = torch.softmax(
                output.action_logits.masked_fill(~mask, -torch.inf), dim=1
            )
            best = utilities.masked_fill(~mask, -torch.inf).max(dim=1).values
            regret = (
                probabilities
                * torch.where(mask, best.unsqueeze(1) - utilities, 0.0)
            ).sum(dim=1)
            total += float(
                torch.sum(
                    regret
                    * torch.from_numpy(weights[batch]).to(
                        device=device, dtype=regret.dtype
                    )
                ).cpu()
            )
    if not math.isfinite(total) or total < -1.0e-9:
        raise PrivilegedAWRError("AAWR validation regret is invalid")
    return max(total, 0.0)


def _critic_validation_loss(
    model: nn.Module,
    critic: PrivilegedValueNetwork,
    arrays: _AAWRPolicyArrays,
    indices: np.ndarray,
    weights: np.ndarray,
    *,
    expectile: float,
    device: torch.device,
) -> float:
    model.eval()
    critic.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(indices), 64):
            batch = indices[start : start + 64]
            output = _actor_output(model, arrays, batch, device=device)
            values = critic(
                output.global_context,
                _tensor(arrays.privileged_features, batch, device=device),
            )
            loss = _critic_loss(
                values,
                _tensor(arrays.rewards, batch, device=device),
                _tensor(arrays.legal_action_mask, batch, device=device),
                torch.from_numpy(weights[batch]).to(
                    device=device, dtype=values.dtype
                ),
                expectile=expectile,
            )
            total += float(loss.cpu()) * float(np.sum(weights[batch]))
    if not math.isfinite(total) or total < 0.0:
        raise PrivilegedAWRError("AAWR validation expectile loss is invalid")
    return total


def _critic_model_sha256(
    critic: PrivilegedValueNetwork,
    *,
    hyperparameters_sha256: str,
    privileged_normalizer_sha256: str,
    training_roster_sha256: str,
) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "schema": 1,
                "kind": "g1-aawr-privileged-critic",
                "hyperparameters": hyperparameters_sha256,
                "privileged_normalizer": privileged_normalizer_sha256,
                "training_roster": training_roster_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name, value in sorted(critic.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _aawr_actor_model_sha256(
    model: nn.Module,
    normalizer: G1ActorNormalizer,
    hyperparameters: PolicyTrainingHyperparameters,
    audit: PolicyFitAudit,
    *,
    critic_model_sha256: str,
    privileged_normalizer_sha256: str,
) -> str:
    if not _valid_sha256(critic_model_sha256) or not _valid_sha256(
        privileged_normalizer_sha256
    ):
        raise PrivilegedAWRError("AAWR actor provenance is invalid")
    observable_actor_sha256 = _model_hash(
        model,
        normalizer,
        hyperparameters,
        audit,
    )
    return _json_sha(
        {
            "schema": 1,
            "kind": "g1-aawr-observable-actor",
            "observable_actor": observable_actor_sha256,
            "privileged_critic": critic_model_sha256,
            "privileged_normalizer": privileged_normalizer_sha256,
        }
    )


def _fit_aawr_policy(
    records: tuple[G1AAWRTrainingRecord, ...],
    *,
    base_actor: TrainedObservablePolicy,
    hyperparameters: PolicyTrainingHyperparameters,
    validation_domain: str | None,
    max_epochs: int,
    patience: int | None,
    batch_specimens: int,
    gradient_clip_norm: float,
    seed: int,
    device: str,
) -> tuple[TrainedObservablePolicy, AAWRPolicyFitEvidence]:
    ordered = _ordered_aawr_records(records)
    source_domains = tuple(sorted({row.example.source_domain for row in ordered}))
    tasks = tuple(
        task
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        if any(row.example.task is task for row in ordered)
    )
    expected_fit_domains = tuple(
        domain for domain in source_domains if domain != validation_domain
    )
    base_hyperparameters = getattr(base_actor, "hyperparameters", None)
    base_audit = getattr(base_actor, "audit", None)
    if (
        type(base_actor) is not TrainedObservablePolicy
        or type(hyperparameters) is not PolicyTrainingHyperparameters
        or hyperparameters.route is not TrainingRoute.PRIVILEGED_AAWR
        or hyperparameters.base_hyperparameters_sha256
        != getattr(base_hyperparameters, "state_sha256", None)
        or getattr(base_hyperparameters, "route", None)
        is not TrainingRoute.SOFT_UTILITY_DISTILL
        or tuple(hyperparameters.aawr_authorized_tasks) != tasks
        or len(source_domains) != 5
        or ordered[0].example.outer_target in source_domains
        or (
            validation_domain is not None
            and validation_domain not in source_domains
        )
        or getattr(base_audit, "outer_target", None)
        != ordered[0].example.outer_target
        or getattr(base_audit, "validation_domain", None) != validation_domain
        or tuple(getattr(base_audit, "fit_domains", ())) != expected_fit_domains
        or base_actor.normalizer.fit_domains != expected_fit_domains
        or any(
            row.example.task not in hyperparameters.aawr_authorized_tasks
            or row.example.dagger_iteration > hyperparameters.dagger_iterations
            or row.example.policy_state.cai_context_mode
            is not hyperparameters.cai_context_mode
            or row.example.policy_state.task_token_mode
            is not hyperparameters.task_token_mode
            for row in ordered
        )
        or type(max_epochs) is not int
        or not 1 <= max_epochs <= REGISTERED_MAX_EPOCHS
        or (
            validation_domain is not None
            and (
                type(patience) is not int
                or not 1 <= patience <= REGISTERED_EARLY_STOPPING_PATIENCE
            )
        )
        or (validation_domain is None and patience is not None)
        or type(batch_specimens) is not int
        or not 1 <= batch_specimens <= REGISTERED_BATCH_SPECIMENS
        or float(gradient_clip_norm) != REGISTERED_GRADIENT_CLIP_NORM
        or seed != REGISTERED_INITIALIZATION_SEED
        or type(device) is not str
        or not device
    ):
        raise PrivilegedAWRError("AAWR fit request leaves the source-only protocol")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise PrivilegedAWRError("registered CUDA AAWR device is unavailable")
    fit_indices = np.asarray(
        [
            index
            for index, row in enumerate(ordered)
            if row.example.source_domain in expected_fit_domains
        ],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [
            index
            for index, row in enumerate(ordered)
            if row.example.source_domain == validation_domain
        ],
        dtype=np.int64,
    )
    if fit_indices.size == 0 or (
        validation_domain is not None and validation_indices.size == 0
    ):
        raise PrivilegedAWRError("AAWR fit or validation roster is empty")
    fit_weights = _state_weights(ordered, fit_indices)
    validation_weights = (
        None
        if validation_domain is None
        else _state_weights(ordered, validation_indices)
    )
    privileged_mean, privileged_scale, privileged_normalizer_sha256 = (
        _privileged_normalizer(ordered, fit_indices)
    )
    arrays = _aawr_arrays(
        ordered,
        base_actor.normalizer,
        privileged_mean=privileged_mean,
        privileged_scale=privileged_scale,
    )
    expectile = float(hyperparameters.aawr_expectile)
    beta = float(hyperparameters.aawr_beta)
    critic_trace: list[float] = []
    actor_trace: list[float] = []
    best_critic_state: dict[str, torch.Tensor] | None = None
    best_actor_state: dict[str, torch.Tensor] | None = None
    best_critic_loss = math.inf
    best_actor_regret = math.inf
    critic_selected_epoch = max_epochs
    actor_selected_epoch = max_epochs
    critic_stale = 0
    actor_stale = 0
    with _deterministic_torch(seed, cpu=torch_device.type == "cpu"):
        model = deepcopy(base_actor.model).to(torch_device)
        critic = PrivilegedValueNetwork().to(torch_device)
        critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=hyperparameters.learning_rate,
            weight_decay=hyperparameters.weight_decay,
        )
        for epoch in range(1, max_epochs + 1):
            model.eval()
            critic.train()
            critic_optimizer.zero_grad(set_to_none=True)
            objective = 0.0
            for batch in _specimen_batches(
                ordered,
                fit_indices,
                epoch=epoch,
                seed=seed,
                batch_specimens=batch_specimens,
            ):
                with torch.no_grad():
                    context = _actor_output(
                        model, arrays, batch, device=torch_device
                    ).global_context
                values = critic(
                    context,
                    _tensor(
                        arrays.privileged_features, batch, device=torch_device
                    ),
                )
                weights = torch.from_numpy(fit_weights[batch]).to(
                    device=torch_device, dtype=values.dtype
                )
                loss = _critic_loss(
                    values,
                    _tensor(arrays.rewards, batch, device=torch_device),
                    _tensor(
                        arrays.legal_action_mask, batch, device=torch_device
                    ),
                    weights,
                    expectile=expectile,
                )
                mass = float(np.sum(fit_weights[batch]))
                (loss * mass).backward()
                objective += float(loss.detach().cpu()) * mass
            nn.utils.clip_grad_norm_(
                critic.parameters(), max_norm=gradient_clip_norm
            )
            critic_optimizer.step()
            critic_trace.append(objective)
            if validation_domain is None:
                continue
            assert validation_weights is not None
            validation_loss = _critic_validation_loss(
                model,
                critic,
                arrays,
                validation_indices,
                validation_weights,
                expectile=expectile,
                device=torch_device,
            )
            if validation_loss < best_critic_loss - 1.0e-12:
                best_critic_loss = validation_loss
                critic_selected_epoch = epoch
                best_critic_state = {
                    name: value.detach().cpu().clone()
                    for name, value in critic.state_dict().items()
                }
                critic_stale = 0
            else:
                critic_stale += 1
                if critic_stale >= int(patience):
                    break
        if validation_domain is not None:
            if best_critic_state is None:
                raise PrivilegedAWRError("AAWR selected no privileged critic")
            critic.load_state_dict(best_critic_state, strict=True)
        critic.eval()
        actor_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hyperparameters.learning_rate,
            weight_decay=hyperparameters.weight_decay,
        )
        for epoch in range(1, max_epochs + 1):
            model.train()
            actor_optimizer.zero_grad(set_to_none=True)
            objective = 0.0
            for batch in _specimen_batches(
                ordered,
                fit_indices,
                epoch=epoch,
                seed=seed,
                batch_specimens=batch_specimens,
            ):
                output = _actor_output(model, arrays, batch, device=torch_device)
                with torch.no_grad():
                    values = critic(
                        output.global_context.detach(),
                        _tensor(
                            arrays.privileged_features,
                            batch,
                            device=torch_device,
                        ),
                    )
                weights = torch.from_numpy(fit_weights[batch]).to(
                    device=torch_device, dtype=output.action_logits.dtype
                )
                loss = _actor_loss(
                    output.action_logits,
                    values,
                    _tensor(arrays.rewards, batch, device=torch_device),
                    _tensor(
                        arrays.legal_action_mask, batch, device=torch_device
                    ),
                    weights,
                    beta=beta,
                )
                mass = float(np.sum(fit_weights[batch]))
                (loss * mass).backward()
                objective += float(loss.detach().cpu()) * mass
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            actor_optimizer.step()
            actor_trace.append(objective)
            if validation_domain is None:
                continue
            assert validation_weights is not None
            regret = _validation_regret(
                model,
                arrays,
                validation_indices,
                validation_weights,
                device=torch_device,
            )
            if regret < best_actor_regret - 1.0e-12:
                best_actor_regret = regret
                actor_selected_epoch = epoch
                best_actor_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                actor_stale = 0
            else:
                actor_stale += 1
                if actor_stale >= int(patience):
                    break
        if validation_domain is not None:
            if best_actor_state is None:
                raise PrivilegedAWRError("AAWR selected no observable actor")
            model.load_state_dict(best_actor_state, strict=True)
        model.cpu().eval()
        critic.cpu().eval()
    training_roster_sha256 = _json_sha(
        tuple(ordered[int(index)].state_sha256 for index in fit_indices)
    )
    audit = PolicyFitAudit(
        outer_target=ordered[0].example.outer_target,
        validation_domain=validation_domain,
        fit_domains=expected_fit_domains,
        fit_specimen_sha256s=tuple(
            sorted(
                {
                    ordered[int(index)].example.specimen_sha256
                    for index in fit_indices
                }
            )
        ),
        validation_specimen_sha256s=tuple(
            sorted(
                {
                    ordered[int(index)].example.specimen_sha256
                    for index in validation_indices
                }
            )
        ),
        epochs_run=len(actor_trace),
        selected_epoch=actor_selected_epoch,
        best_validation_regret=(
            None if validation_domain is None else float(best_actor_regret)
        ),
        training_objectives=tuple(actor_trace),
        normalizer_state_sha256=base_actor.normalizer.state_sha256,
        training_roster_sha256=training_roster_sha256,
    )
    critic_sha256 = _critic_model_sha256(
        critic,
        hyperparameters_sha256=hyperparameters.state_sha256,
        privileged_normalizer_sha256=privileged_normalizer_sha256,
        training_roster_sha256=training_roster_sha256,
    )
    model_state_sha256 = _aawr_actor_model_sha256(
        model,
        base_actor.normalizer,
        hyperparameters,
        audit,
        critic_model_sha256=critic_sha256,
        privileged_normalizer_sha256=privileged_normalizer_sha256,
    )
    actor = TrainedObservablePolicy(
        model=model,
        normalizer=base_actor.normalizer,
        hyperparameters=hyperparameters,
        audit=audit,
        model_state_sha256=model_state_sha256,
    )
    counts = Counter(
        ordered[int(index)].state_source for index in fit_indices
    )
    evidence = AAWRPolicyFitEvidence(
        outer_target=ordered[0].example.outer_target,
        validation_domain=validation_domain,
        fit_domains=expected_fit_domains,
        hyperparameters_sha256=hyperparameters.state_sha256,
        base_model_sha256=base_actor.model_state_sha256,
        actor_model_sha256=model_state_sha256,
        critic_model_sha256=critic_sha256,
        privileged_normalizer_sha256=privileged_normalizer_sha256,
        source_record_sha256s=tuple(
            ordered[int(index)].state_sha256 for index in fit_indices
        ),
        trajectory_source_counts=tuple(sorted(counts.items())),
        critic_epochs_run=len(critic_trace),
        critic_selected_epoch=critic_selected_epoch,
        critic_objectives=tuple(critic_trace),
        best_validation_expectile_loss=(
            None if validation_domain is None else float(best_critic_loss)
        ),
        target_outcomes_opened=False,
    )
    return actor, evidence


def fit_inner_aawr_policy(
    records: tuple[G1AAWRTrainingRecord, ...],
    *,
    base_actor: TrainedObservablePolicy,
    hyperparameters: PolicyTrainingHyperparameters,
    validation_domain: str,
    max_epochs: int = REGISTERED_MAX_EPOCHS,
    patience: int = REGISTERED_EARLY_STOPPING_PATIENCE,
    batch_specimens: int = REGISTERED_BATCH_SPECIMENS,
    gradient_clip_norm: float = REGISTERED_GRADIENT_CLIP_NORM,
    seed: int = REGISTERED_INITIALIZATION_SEED,
    device: str = "cpu",
) -> tuple[TrainedObservablePolicy, AAWRPolicyFitEvidence]:
    return _fit_aawr_policy(
        records,
        base_actor=base_actor,
        hyperparameters=hyperparameters,
        validation_domain=validation_domain,
        max_epochs=max_epochs,
        patience=patience,
        batch_specimens=batch_specimens,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
        device=device,
    )


def fit_final_aawr_policy(
    records: tuple[G1AAWRTrainingRecord, ...],
    *,
    base_actor: TrainedObservablePolicy,
    hyperparameters: PolicyTrainingHyperparameters,
    selected_epochs: int,
    batch_specimens: int = REGISTERED_BATCH_SPECIMENS,
    gradient_clip_norm: float = REGISTERED_GRADIENT_CLIP_NORM,
    seed: int = REGISTERED_INITIALIZATION_SEED,
    device: str = "cpu",
) -> tuple[TrainedObservablePolicy, AAWRPolicyFitEvidence]:
    return _fit_aawr_policy(
        records,
        base_actor=base_actor,
        hyperparameters=hyperparameters,
        validation_domain=None,
        max_epochs=selected_epochs,
        patience=None,
        batch_specimens=batch_specimens,
        gradient_clip_norm=gradient_clip_norm,
        seed=seed,
        device=device,
    )


__all__ = [
    "AAWR_GAP_CLOSURE_LIMIT",
    "AAWR_TRAJECTORY_SOURCES",
    "PRIVILEGED_CRITIC_FEATURE_DIMENSION",
    "REGISTERED_BETAS",
    "REGISTERED_EXPECTILES",
    "REGISTERED_MAXIMUM_ADVANTAGE_WEIGHT",
    "AAWRAuthorization",
    "AAWRPolicyFitEvidence",
    "AAWRSourceEvidence",
    "G1AAWRTrainingRecord",
    "PrivilegedAWRError",
    "PrivilegedValueNetwork",
    "aawr_policy_hyperparameters",
    "advantage_weighted_actor_loss",
    "advantage_weights",
    "authorize_conditional_aawr",
    "expectile_value_loss",
    "fit_final_aawr_policy",
    "fit_inner_aawr_policy",
    "privileged_critic_features",
    "read_aawr_fit_evidence",
    "write_aawr_fit_evidence",
]
