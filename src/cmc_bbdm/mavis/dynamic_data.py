"""Strict-OOF grouped training and evaluation data for dynamic MAVIS VoI."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from .dynamic_voi import CandidateDescriptor, conditional_teacher_value


class MAVISDynamicDataError(ValueError):
    """Raised when P1 state/action rows violate the P3 data boundary."""


_STATE_COLUMNS = {
    "state_id",
    "specimen_id",
    "domain_id",
    "exact_acquired_cost",
    "native_count",
    "remaining_cost_to_endpoint",
    "candidate_cell_indices",
    "candidate_from_levels",
    "candidate_to_levels",
    "candidate_exact_added_costs",
}
_ACTION_COLUMNS = {
    "state_id",
    "specimen_id",
    "domain_id",
    "outer_domain",
    "candidate_index",
    "cell_index",
    "from_level",
    "to_level",
    "exact_added_cost",
    "teacher_true_cai",
    "current_prediction",
    "candidate_prediction",
    "primary_value",
    "teacher_state_sha256",
}


def _readonly(values: object) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype="<f8")
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise MAVISDynamicDataError("dynamic numeric roster is invalid")
    result = np.frombuffer(array.tobytes(order="C"), dtype="<f8")
    result.setflags(write=False)
    return result


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class DynamicStateGroup:
    state_id: str
    specimen_id: str
    domain_id: str
    outer_domain: str
    candidates: tuple[CandidateDescriptor, ...]
    true_cai: float
    current_prediction: float
    candidate_predictions: np.ndarray
    teacher_values: np.ndarray
    teacher_outer_domains: tuple[str, ...]
    teacher_fold_count: int
    state_sha256: str


def _tables(
    states: object,
    actions: object,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, dict[str, object]]]:
    if (
        not isinstance(states, pl.DataFrame)
        or not isinstance(actions, pl.DataFrame)
        or states.height == 0
        or actions.height == 0
        or not _STATE_COLUMNS <= set(states.columns)
        or not _ACTION_COLUMNS <= set(actions.columns)
        or states.get_column("state_id").n_unique() != states.height
    ):
        raise MAVISDynamicDataError("P3 source tables are invalid")
    state_table = states.sort("state_id")
    action_table = actions.sort(["state_id", "outer_domain", "candidate_index"])
    state_lookup = {
        str(row["state_id"]): row for row in state_table.iter_rows(named=True)
    }
    if not set(action_table.get_column("state_id").unique()).issubset(state_lookup):
        raise MAVISDynamicDataError("P3 action rows reference an unknown state")
    return state_table, action_table, state_lookup


def _candidates(state: dict[str, object]) -> tuple[CandidateDescriptor, ...]:
    cells = tuple(state["candidate_cell_indices"])
    from_levels = tuple(state["candidate_from_levels"])
    to_levels = tuple(state["candidate_to_levels"])
    costs = tuple(state["candidate_exact_added_costs"])
    if not cells or not (len(cells) == len(from_levels) == len(to_levels) == len(costs)):
        raise MAVISDynamicDataError("P3 state candidate roster is invalid")
    remaining = int(state["remaining_cost_to_endpoint"])
    return tuple(
        CandidateDescriptor(
            cell_index=int(cell),
            from_level=int(from_level),
            to_level=int(to_level),
            exact_added_cost=int(cost),
            native_count=int(state["native_count"]),
            remaining_cost=remaining,
        )
        for cell, from_level, to_level, cost in zip(
            cells,
            from_levels,
            to_levels,
            costs,
            strict=True,
        )
    )


def _fold_rows(
    state: dict[str, object],
    rows: list[dict[str, object]],
) -> tuple[tuple[CandidateDescriptor, ...], float, float, np.ndarray, np.ndarray]:
    candidates = _candidates(state)
    if len(rows) != len(candidates):
        raise MAVISDynamicDataError("P3 action roster is incomplete")
    rows.sort(key=lambda row: int(row["candidate_index"]))
    if tuple(int(row["candidate_index"]) for row in rows) != tuple(
        range(len(candidates))
    ):
        raise MAVISDynamicDataError("P3 candidate indices changed")
    for candidate, row in zip(candidates, rows, strict=True):
        if (
            str(row["state_id"]) != str(state["state_id"])
            or str(row["specimen_id"]) != str(state["specimen_id"])
            or str(row["domain_id"]) != str(state["domain_id"])
            or int(row["cell_index"]) != candidate.cell_index
            or int(row["from_level"]) != candidate.from_level
            or int(row["to_level"]) != candidate.to_level
            or int(row["exact_added_cost"]) != candidate.exact_added_cost
            or not _is_sha(row["teacher_state_sha256"])
        ):
            raise MAVISDynamicDataError("P3 state/action candidate linkage changed")
    true_values = {float(row["teacher_true_cai"]) for row in rows}
    current_values = {float(row["current_prediction"]) for row in rows}
    if (
        len(true_values) != 1
        or len(current_values) != 1
        or not all(math.isfinite(value) for value in (*true_values, *current_values))
    ):
        raise MAVISDynamicDataError("P3 state teacher scalars are inconsistent")
    true_cai = true_values.pop()
    current = current_values.pop()
    predictions = _readonly([row["candidate_prediction"] for row in rows])
    values = conditional_teacher_value(
        true_cai=true_cai,
        current_prediction=current,
        candidate_predictions=predictions,
    )
    stored = np.asarray([row["primary_value"] for row in rows], dtype=np.float64)
    if not np.allclose(values, stored, rtol=0.0, atol=1.0e-12):
        raise MAVISDynamicDataError("P3 teacher utility changed")
    return candidates, true_cai, current, predictions, values


def _group(
    state: dict[str, object],
    *,
    outer_domain: str,
    rows_by_outer: dict[str, list[dict[str, object]]],
) -> DynamicStateGroup:
    folds = tuple(sorted(rows_by_outer))
    fold_outputs = [_fold_rows(state, rows_by_outer[fold]) for fold in folds]
    candidate_rosters = [output[0] for output in fold_outputs]
    if any(roster != candidate_rosters[0] for roster in candidate_rosters[1:]):
        raise MAVISDynamicDataError("P3 teacher folds use different candidates")
    true_values = {output[1] for output in fold_outputs}
    if len(true_values) != 1:
        raise MAVISDynamicDataError("P3 teacher folds disagree on evaluation CAI")
    true_cai = true_values.pop()
    current = float(np.mean([output[2] for output in fold_outputs], dtype=np.float64))
    predictions = _readonly(
        np.mean(
            np.stack([output[3] for output in fold_outputs]),
            axis=0,
            dtype=np.float64,
        )
    )
    values = conditional_teacher_value(
        true_cai=true_cai,
        current_prediction=current,
        candidate_predictions=predictions,
    )
    payload = {
        "schema": 1,
        "state_id": state["state_id"],
        "specimen_id": state["specimen_id"],
        "domain_id": state["domain_id"],
        "outer_domain": outer_domain,
        "teacher_outer_domains": folds,
        "candidates": [candidate.features().tolist() for candidate in candidate_rosters[0]],
        "true_cai": true_cai,
        "current_prediction": current,
        "candidate_predictions": predictions.tolist(),
        "teacher_values": values.tolist(),
    }
    return DynamicStateGroup(
        state_id=str(state["state_id"]),
        specimen_id=str(state["specimen_id"]),
        domain_id=str(state["domain_id"]),
        outer_domain=outer_domain,
        candidates=candidate_rosters[0],
        true_cai=true_cai,
        current_prediction=current,
        candidate_predictions=predictions,
        teacher_values=values,
        teacher_outer_domains=folds,
        teacher_fold_count=len(folds),
        state_sha256=_sha(payload),
    )


def build_dynamic_training_groups(
    states: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    outer_domain: str,
) -> tuple[DynamicStateGroup, ...]:
    """Issue source groups whose teacher fold excludes the outer target."""

    state_table, action_table, state_lookup = _tables(states, actions)
    domains = tuple(sorted(state_table.get_column("domain_id").unique()))
    if outer_domain not in domains:
        raise MAVISDynamicDataError("P3 outer domain is invalid")
    selected = action_table.filter(pl.col("outer_domain") == outer_domain)
    if selected.height == 0 or selected.filter(pl.col("domain_id") == outer_domain).height:
        raise MAVISDynamicDataError("P3 training target isolation failed")
    grouped: list[DynamicStateGroup] = []
    for state_id, rows in selected.group_by("state_id", maintain_order=True):
        key = str(state_id[0])
        records = rows.iter_rows(named=True)
        grouped.append(
            _group(
                state_lookup[key],
                outer_domain=outer_domain,
                rows_by_outer={outer_domain: list(records)},
            )
        )
    result = tuple(sorted(grouped, key=lambda group: group.state_id))
    if not result or {group.domain_id for group in result} != set(domains) - {outer_domain}:
        raise MAVISDynamicDataError("P3 training domain roster is incomplete")
    return result


def build_target_evaluation_groups(
    states: pl.DataFrame,
    actions: pl.DataFrame,
    *,
    target_domain: str,
) -> tuple[DynamicStateGroup, ...]:
    """Build post-hoc target utility from models that all exclude the target query."""

    state_table, action_table, state_lookup = _tables(states, actions)
    domains = tuple(sorted(state_table.get_column("domain_id").unique()))
    expected_folds = set(domains) - {target_domain}
    if target_domain not in domains or not expected_folds:
        raise MAVISDynamicDataError("P3 target domain is invalid")
    selected = action_table.filter(pl.col("domain_id") == target_domain)
    if selected.height == 0 or selected.filter(pl.col("outer_domain") == target_domain).height:
        raise MAVISDynamicDataError("P3 target evaluation fold boundary failed")
    grouped: list[DynamicStateGroup] = []
    for state_id, rows in selected.group_by("state_id", maintain_order=True):
        key = str(state_id[0])
        by_outer: dict[str, list[dict[str, object]]] = {}
        for row in rows.iter_rows(named=True):
            by_outer.setdefault(str(row["outer_domain"]), []).append(row)
        if set(by_outer) != expected_folds:
            raise MAVISDynamicDataError("P3 target strict-OOF fold roster is incomplete")
        grouped.append(
            _group(
                state_lookup[key],
                outer_domain=target_domain,
                rows_by_outer=by_outer,
            )
        )
    result = tuple(sorted(grouped, key=lambda group: group.state_id))
    if not result or {group.domain_id for group in result} != {target_domain}:
        raise MAVISDynamicDataError("P3 target evaluation roster is incomplete")
    return result


__all__ = [
    "DynamicStateGroup",
    "MAVISDynamicDataError",
    "build_dynamic_training_groups",
    "build_target_evaluation_groups",
]
