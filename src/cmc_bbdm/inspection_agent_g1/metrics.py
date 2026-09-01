"""Common-geometry G1 engineering curves and oracle-gap closure."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.evaluation import zero_inclusive_auebc

NOMINAL_CHECKPOINTS = (0.0, 0.0625, 0.125, 0.1875, 0.25)


class G1MetricError(ValueError):
    """Raised when a G1 curve or same-geometry comparison is invalid."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _readonly(value: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f8")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise G1MetricError("engineering curve array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(shape)
    output.setflags(write=False)
    return output


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class EngineeringCurve:
    method: str
    target_domain: str
    specimen_sha256: str
    task: InspectionTask
    grid_sha256: str
    evaluator_sha256: str
    warm_start_sha256: str
    nominal_budgets: np.ndarray
    exact_budgets: np.ndarray
    task_losses: np.ndarray
    projected_state_sha256: tuple[str, ...]
    auebc: float
    state_sha256: str


def build_engineering_curve(
    *,
    method: str,
    target_domain: str,
    specimen_sha256: str,
    task: InspectionTask,
    grid_sha256: str,
    evaluator_sha256: str,
    warm_start_sha256: str,
    state_budgets: tuple[float, ...],
    state_losses: tuple[float, ...],
    state_sha256: tuple[str, ...],
) -> EngineeringCurve:
    budgets = tuple(float(value) for value in state_budgets)
    losses = tuple(float(value) for value in state_losses)
    if (
        type(method) is not str
        or not method
        or type(target_domain) is not str
        or not target_domain
        or not _valid_sha256(specimen_sha256)
        or task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or not all(
            _valid_sha256(value)
            for value in (grid_sha256, evaluator_sha256, warm_start_sha256)
        )
        or type(state_budgets) is not tuple
        or not budgets
        or budgets[0] != 0.0
        or budgets[-1] > NOMINAL_CHECKPOINTS[-1] + 1.0e-15
        or any(not math.isfinite(value) or value < 0.0 for value in budgets)
        or any(right <= left for left, right in pairwise(budgets))
        or type(state_losses) is not tuple
        or len(losses) != len(budgets)
        or any(not math.isfinite(value) or value < 0.0 for value in losses)
        or type(state_sha256) is not tuple
        or len(state_sha256) != len(budgets)
        or any(not _valid_sha256(value) for value in state_sha256)
    ):
        raise G1MetricError("engineering curve request is invalid")
    projected_budget = []
    projected_loss = []
    projected_state = []
    for checkpoint in NOMINAL_CHECKPOINTS:
        index = max(
            position
            for position, budget in enumerate(budgets)
            if budget <= checkpoint + 1.0e-15
        )
        projected_budget.append(budgets[index])
        projected_loss.append(losses[index])
        projected_state.append(state_sha256[index])
    nominal = _readonly(NOMINAL_CHECKPOINTS, (len(NOMINAL_CHECKPOINTS),))
    exact = _readonly(projected_budget, nominal.shape)
    task_values = _readonly(projected_loss, nominal.shape)
    area = zero_inclusive_auebc(nominal, task_values)
    payload = {
        "schema": 1,
        "kind": "g1-engineering-curve",
        "method": method,
        "target_domain": target_domain,
        "specimen_sha256": specimen_sha256,
        "task": task.value,
        "grid_sha256": grid_sha256,
        "evaluator_sha256": evaluator_sha256,
        "warm_start_sha256": warm_start_sha256,
        "nominal_budgets": tuple(float(value) for value in nominal),
        "exact_budgets": tuple(float(value) for value in exact),
        "task_losses": tuple(float(value) for value in task_values),
        "projected_state_sha256": tuple(projected_state),
        "auebc": area,
    }
    return EngineeringCurve(
        method=method,
        target_domain=target_domain,
        specimen_sha256=specimen_sha256,
        task=task,
        grid_sha256=grid_sha256,
        evaluator_sha256=evaluator_sha256,
        warm_start_sha256=warm_start_sha256,
        nominal_budgets=nominal,
        exact_budgets=exact,
        task_losses=task_values,
        projected_state_sha256=tuple(projected_state),
        auebc=area,
        state_sha256=_json_sha(payload),
    )


@dataclass(frozen=True, slots=True)
class OracleGapClosure:
    fixed_minus_learned: float
    fixed_minus_oracle: float
    available: bool
    value: float | None
    fixed_curve_sha256: str
    learned_curve_sha256: str
    oracle_curve_sha256: str
    state_sha256: str


def oracle_gap_closure(
    *,
    fixed: EngineeringCurve,
    learned: EngineeringCurve,
    oracle: EngineeringCurve,
) -> OracleGapClosure:
    if any(type(curve) is not EngineeringCurve for curve in (fixed, learned, oracle)):
        raise G1MetricError("oracle-gap closure requires issued engineering curves")
    identities = {
        (
            curve.target_domain,
            curve.specimen_sha256,
            curve.task,
            curve.grid_sha256,
            curve.evaluator_sha256,
            curve.warm_start_sha256,
            tuple(curve.nominal_budgets),
        )
        for curve in (fixed, learned, oracle)
    }
    if len(identities) != 1:
        raise G1MetricError("oracle-gap curves do not share evaluation geometry")
    numerator = fixed.auebc - learned.auebc
    denominator = fixed.auebc - oracle.auebc
    available = denominator > 0.0
    value = numerator / denominator if available else None
    payload = {
        "schema": 1,
        "kind": "g1-oracle-gap-closure",
        "fixed": fixed.state_sha256,
        "learned": learned.state_sha256,
        "oracle": oracle.state_sha256,
        "fixed_minus_learned": numerator,
        "fixed_minus_oracle": denominator,
        "available": available,
        "value": value,
    }
    return OracleGapClosure(
        fixed_minus_learned=numerator,
        fixed_minus_oracle=denominator,
        available=available,
        value=value,
        fixed_curve_sha256=fixed.state_sha256,
        learned_curve_sha256=learned.state_sha256,
        oracle_curve_sha256=oracle.state_sha256,
        state_sha256=_json_sha(payload),
    )


__all__ = [
    "NOMINAL_CHECKPOINTS",
    "EngineeringCurve",
    "G1MetricError",
    "OracleGapClosure",
    "build_engineering_curve",
    "oracle_gap_closure",
]
