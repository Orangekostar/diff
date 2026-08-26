"""Source-only on-policy state aggregation with fail-closed target isolation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import InspectionState
from .dynamic_data import DynamicStateGroup
from .dynamic_voi import CandidateDescriptor, conditional_teacher_value
from .state_candidates import StateCandidateBatch
from .teacher import FoldStateLabels


class MAVISAggregationError(ValueError):
    """Raised when on-policy aggregation crosses its source-only boundary."""


def _is_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class AggregationModel:
    model: Any
    model_state_sha256: str

    def __post_init__(self) -> None:
        if not _is_sha(self.model_state_sha256):
            raise MAVISAggregationError("aggregation model hash is invalid")


@dataclass(frozen=True, slots=True)
class AggregationRoundAudit:
    round_index: int
    outer_domain: str
    source_domains: tuple[str, ...]
    source_specimen_ids: tuple[str, ...]
    model_state_sha256: str
    state_count_before: int
    visited_state_count: int
    duplicate_state_count: int
    appended_state_count: int
    target_state_count: int
    state_count_after: int
    aggregate_state_sha256: str


@dataclass(frozen=True, slots=True)
class SourceAggregationResult:
    outer_domain: str
    groups: tuple[DynamicStateGroup, ...]
    audits: tuple[AggregationRoundAudit, ...]
    final_model: AggregationModel
    state_sha256: str


def _groups(
    value: object,
    *,
    outer_domain: str,
    source_specimens: set[str] | None,
) -> tuple[DynamicStateGroup, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(type(group) is not DynamicStateGroup for group in value)
    ):
        raise MAVISAggregationError("aggregation state groups are invalid")
    for group in value:
        if group.outer_domain != outer_domain:
            raise MAVISAggregationError("aggregation outer fold changed")
        if group.domain_id == outer_domain:
            raise MAVISAggregationError("target state reached source-only aggregation")
        if source_specimens is not None and group.specimen_id not in source_specimens:
            raise MAVISAggregationError("aggregation visited an unregistered specimen")
    return value


def _state_hash(groups: tuple[DynamicStateGroup, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            [(group.state_id, group.state_sha256) for group in groups],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_on_policy_group(
    state: InspectionState,
    candidates: StateCandidateBatch,
    labels: FoldStateLabels,
    *,
    outer_domain: str,
) -> DynamicStateGroup:
    if (
        type(state) is not InspectionState
        or type(candidates) is not StateCandidateBatch
        or type(labels) is not FoldStateLabels
        or type(outer_domain) is not str
        or not outer_domain
        or outer_domain != labels.outer_domain
        or outer_domain == candidates.dataset_id
        or labels.query_domain != candidates.dataset_id
        or candidates.specimen_id != state.specimen_id
        or candidates.inspection_state_sha256 != state.state_sha256
        or not candidates.actions
        or len(labels.action_values) != len(candidates.actions)
        or not _is_sha(labels.teacher_state_sha256)
        or not _is_sha(labels.predictor_state_sha256)
    ):
        raise MAVISAggregationError("on-policy teacher group is invalid")
    remaining = math.floor(candidates.endpoint_budget * state.native_count) - int(
        state.exact_acquired_count
    )
    if remaining <= 0:
        raise MAVISAggregationError("on-policy state has no remaining exact cost")
    descriptors: list[CandidateDescriptor] = []
    predictions: list[float] = []
    stored_values: list[float] = []
    true_values: set[float] = set()
    current_values: set[float] = set()
    for action, cost, label in zip(
        candidates.actions,
        candidates.candidate_costs,
        labels.action_values,
        strict=True,
    ):
        if (
            label.specimen_id != state.specimen_id
            or label.dataset_id != candidates.dataset_id
            or label.action != action
            or label.exact_added_cost != cost
            or label.predictor_state_sha256 != labels.predictor_state_sha256
        ):
            raise MAVISAggregationError("on-policy candidate label linkage changed")
        descriptors.append(
            CandidateDescriptor(
                cell_index=action.cell_index,
                from_level=action.from_level,
                to_level=action.to_level,
                exact_added_cost=cost,
                native_count=state.native_count,
                remaining_cost=remaining,
            )
        )
        predictions.append(float(label.candidate_prediction))
        stored_values.append(float(label.primary_value))
        true_values.add(float(label.true_cai))
        current_values.add(float(label.current_prediction))
    if len(true_values) != 1 or len(current_values) != 1:
        raise MAVISAggregationError("on-policy teacher scalars changed")
    true_cai = true_values.pop()
    current = current_values.pop()
    prediction_array = np.ascontiguousarray(predictions, dtype="<f8")
    value_array = conditional_teacher_value(
        true_cai=true_cai,
        current_prediction=current,
        candidate_predictions=prediction_array,
    )
    if not np.allclose(value_array, stored_values, rtol=0.0, atol=1.0e-12):
        raise MAVISAggregationError("on-policy teacher utility changed")
    prediction_output = np.frombuffer(prediction_array.tobytes(order="C"), dtype="<f8")
    prediction_output.setflags(write=False)
    value_output = np.frombuffer(value_array.tobytes(order="C"), dtype="<f8")
    value_output.setflags(write=False)
    state_id = f"on_policy::{outer_domain}::{state.specimen_id}::{state.state_sha256}"
    payload = {
        "schema": 1,
        "state_id": state_id,
        "state_sha256": state.state_sha256,
        "candidate_batch_state_sha256": candidates.state_sha256,
        "teacher_state_sha256": labels.teacher_state_sha256,
        "predictor_state_sha256": labels.predictor_state_sha256,
        "candidate_predictions": prediction_output.tolist(),
        "teacher_values": value_output.tolist(),
    }
    return DynamicStateGroup(
        state_id=state_id,
        specimen_id=state.specimen_id,
        domain_id=candidates.dataset_id,
        outer_domain=outer_domain,
        candidates=tuple(descriptors),
        true_cai=true_cai,
        current_prediction=current,
        candidate_predictions=prediction_output,
        teacher_values=value_output,
        teacher_outer_domains=(outer_domain,),
        teacher_fold_count=1,
        state_sha256=hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    )


def run_source_only_aggregation(
    initial_groups: tuple[DynamicStateGroup, ...],
    *,
    outer_domain: str,
    rounds: int,
    train_model: Callable[[tuple[DynamicStateGroup, ...], int], AggregationModel],
    collect_source_groups: Callable[
        [Any, tuple[str, ...], int], tuple[DynamicStateGroup, ...]
    ],
) -> SourceAggregationResult:
    if (
        type(outer_domain) is not str
        or not outer_domain
        or type(rounds) is not int
        or rounds <= 0
        or not callable(train_model)
        or not callable(collect_source_groups)
    ):
        raise MAVISAggregationError("aggregation request is invalid")
    initial = _groups(
        initial_groups,
        outer_domain=outer_domain,
        source_specimens=None,
    )
    source_specimens = {group.specimen_id for group in initial}
    source_domains = tuple(sorted({group.domain_id for group in initial}))
    if len(source_domains) < 2:
        raise MAVISAggregationError("aggregation source-domain roster is incomplete")
    current = tuple(sorted(initial, key=lambda group: group.state_id))
    state_by_id: dict[str, str] = {}
    for group in current:
        previous = state_by_id.setdefault(group.state_id, group.state_sha256)
        if previous != group.state_sha256:
            raise MAVISAggregationError("aggregation state identity is ambiguous")
    current = tuple(
        group
        for index, group in enumerate(current)
        if group.state_id not in {item.state_id for item in current[:index]}
    )
    audits: list[AggregationRoundAudit] = []
    for round_index in range(rounds):
        fitted = train_model(current, round_index)
        if type(fitted) is not AggregationModel:
            raise MAVISAggregationError("aggregation trainer returned an invalid model")
        visited = _groups(
            collect_source_groups(
                fitted.model,
                tuple(sorted(source_specimens)),
                round_index,
            ),
            outer_domain=outer_domain,
            source_specimens=source_specimens,
        )
        additions: list[DynamicStateGroup] = []
        duplicate_count = 0
        for group in sorted(visited, key=lambda item: item.state_id):
            previous = state_by_id.get(group.state_id)
            if previous is not None:
                if previous != group.state_sha256:
                    raise MAVISAggregationError(
                        "aggregation duplicate state hash changed"
                    )
                duplicate_count += 1
                continue
            state_by_id[group.state_id] = group.state_sha256
            additions.append(group)
        before = len(current)
        current = tuple(sorted((*current, *additions), key=lambda group: group.state_id))
        audits.append(
            AggregationRoundAudit(
                round_index=round_index,
                outer_domain=outer_domain,
                source_domains=source_domains,
                source_specimen_ids=tuple(sorted(source_specimens)),
                model_state_sha256=fitted.model_state_sha256,
                state_count_before=before,
                visited_state_count=len(visited),
                duplicate_state_count=duplicate_count,
                appended_state_count=len(additions),
                target_state_count=0,
                state_count_after=len(current),
                aggregate_state_sha256=_state_hash(current),
            )
        )
    final_model = train_model(current, rounds)
    if type(final_model) is not AggregationModel:
        raise MAVISAggregationError("aggregation final trainer returned an invalid model")
    payload = {
        "schema": 1,
        "outer_domain": outer_domain,
        "rounds": [audit.aggregate_state_sha256 for audit in audits],
        "groups": [(group.state_id, group.state_sha256) for group in current],
        "final_model_state_sha256": final_model.model_state_sha256,
    }
    return SourceAggregationResult(
        outer_domain=outer_domain,
        groups=current,
        audits=tuple(audits),
        final_model=final_model,
        state_sha256=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )


__all__ = [
    "AggregationModel",
    "AggregationRoundAudit",
    "MAVISAggregationError",
    "SourceAggregationResult",
    "build_on_policy_group",
    "run_source_only_aggregation",
]
