"""Conditionally authorized source-only privileged advantage training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .contracts import ACTION_SLOT_COUNT

REGISTERED_EXPECTILES = (0.7, 0.8)
REGISTERED_BETAS = (1.0, 3.0)
REGISTERED_MAXIMUM_ADVANTAGE_WEIGHT = 100.0
AAWR_GAP_CLOSURE_LIMIT = 0.20
PRIVILEGED_CRITIC_FEATURE_DIMENSION = 515


class PrivilegedAWRError(ValueError):
    """Raised when AAWR evidence or training tensors violate the source-only gate."""


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _finite_nonnegative(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) and value >= 0.0 for value in values)


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
    authorized_tasks: tuple[InspectionTask, ...]
    evidence: tuple[AAWRSourceEvidence, ...]
    state_sha256: str


def authorize_conditional_aawr(
    evidence: tuple[AAWRSourceEvidence, ...],
) -> AAWRAuthorization:
    if (
        type(evidence) is not tuple
        or not 1 <= len(evidence) <= 2
        or any(type(row) is not AAWRSourceEvidence for row in evidence)
        or len({row.task for row in evidence}) != len(evidence)
        or len({row.outer_target for row in evidence}) != 1
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
        if row.positive_action_observability
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
        "authorized_tasks": tuple(task.value for task in authorized),
        "evidence": tuple(row.state_sha256 for row in evidence),
    }
    return AAWRAuthorization(
        outer_target=outer_target,
        status=status,
        authorized_tasks=authorized,
        evidence=evidence,
        state_sha256=_json_sha(payload),
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


__all__ = [
    "AAWR_GAP_CLOSURE_LIMIT",
    "PRIVILEGED_CRITIC_FEATURE_DIMENSION",
    "REGISTERED_BETAS",
    "REGISTERED_EXPECTILES",
    "REGISTERED_MAXIMUM_ADVANTAGE_WEIGHT",
    "AAWRAuthorization",
    "AAWRSourceEvidence",
    "PrivilegedAWRError",
    "PrivilegedValueNetwork",
    "advantage_weighted_actor_loss",
    "advantage_weights",
    "authorize_conditional_aawr",
    "expectile_value_loss",
    "privileged_critic_features",
]
