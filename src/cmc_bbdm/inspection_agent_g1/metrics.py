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


def engineering_curve_payload(curve: EngineeringCurve) -> dict[str, object]:
    if type(curve) is not EngineeringCurve:
        raise G1MetricError("issued engineering curve is required")
    return {
        "schema": 1,
        "kind": "g1-engineering-curve",
        "method": curve.method,
        "target_domain": curve.target_domain,
        "specimen_sha256": curve.specimen_sha256,
        "task": curve.task.value,
        "grid_sha256": curve.grid_sha256,
        "evaluator_sha256": curve.evaluator_sha256,
        "warm_start_sha256": curve.warm_start_sha256,
        "nominal_budgets": tuple(float(value) for value in curve.nominal_budgets),
        "exact_budgets": tuple(float(value) for value in curve.exact_budgets),
        "task_losses": tuple(float(value) for value in curve.task_losses),
        "projected_state_sha256": curve.projected_state_sha256,
        "auebc": float(curve.auebc),
    }


def validate_engineering_curve(curve: EngineeringCurve) -> None:
    if type(curve) is not EngineeringCurve:
        raise G1MetricError("issued engineering curve is required")
    nominal = np.asarray(curve.nominal_budgets, dtype=np.float64)
    exact = np.asarray(curve.exact_budgets, dtype=np.float64)
    losses = np.asarray(curve.task_losses, dtype=np.float64)
    if (
        type(curve.method) is not str
        or not curve.method
        or type(curve.target_domain) is not str
        or not curve.target_domain
        or curve.task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or nominal.shape != (5,)
        or exact.shape != nominal.shape
        or losses.shape != nominal.shape
        or tuple(float(value) for value in nominal) != NOMINAL_CHECKPOINTS
        or not np.all(np.isfinite(exact))
        or not np.all(np.isfinite(losses))
        or np.any(exact < 0.0)
        or np.any(exact - nominal > 1.0e-15)
        or any(float(right) < float(left) for left, right in pairwise(exact))
        or np.any(losses < 0.0)
        or len(curve.projected_state_sha256) != 5
        or not all(_valid_sha256(value) for value in curve.projected_state_sha256)
        or not all(
            _valid_sha256(value)
            for value in (
                curve.specimen_sha256,
                curve.grid_sha256,
                curve.evaluator_sha256,
                curve.warm_start_sha256,
                curve.state_sha256,
            )
        )
        or not math.isclose(
            float(curve.auebc),
            zero_inclusive_auebc(nominal, losses),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or curve.state_sha256 != _json_sha(engineering_curve_payload(curve))
    ):
        raise G1MetricError("engineering curve identity changed")


def replay_engineering_curve(
    *,
    method: str,
    target_domain: str,
    specimen_sha256: str,
    task: InspectionTask,
    grid_sha256: str,
    evaluator_sha256: str,
    warm_start_sha256: str,
    exact_budgets: object,
    task_losses: object,
    projected_state_sha256: tuple[str, ...],
    state_sha256: str | None = None,
) -> EngineeringCurve:
    nominal = _readonly(NOMINAL_CHECKPOINTS, (len(NOMINAL_CHECKPOINTS),))
    exact = _readonly(exact_budgets, nominal.shape)
    losses = _readonly(task_losses, nominal.shape)
    projected = tuple(projected_state_sha256)
    area = zero_inclusive_auebc(nominal, losses)
    curve = EngineeringCurve(
        method=method,
        target_domain=target_domain,
        specimen_sha256=specimen_sha256,
        task=task,
        grid_sha256=grid_sha256,
        evaluator_sha256=evaluator_sha256,
        warm_start_sha256=warm_start_sha256,
        nominal_budgets=nominal,
        exact_budgets=exact,
        task_losses=losses,
        projected_state_sha256=projected,
        auebc=area,
        state_sha256="",
    )
    issued = EngineeringCurve(
        method=curve.method,
        target_domain=curve.target_domain,
        specimen_sha256=curve.specimen_sha256,
        task=curve.task,
        grid_sha256=curve.grid_sha256,
        evaluator_sha256=curve.evaluator_sha256,
        warm_start_sha256=curve.warm_start_sha256,
        nominal_budgets=curve.nominal_budgets,
        exact_budgets=curve.exact_budgets,
        task_losses=curve.task_losses,
        projected_state_sha256=curve.projected_state_sha256,
        auebc=curve.auebc,
        state_sha256=_json_sha(engineering_curve_payload(curve)),
    )
    if state_sha256 is not None and state_sha256 != issued.state_sha256:
        raise G1MetricError("replayed engineering curve hash changed")
    validate_engineering_curve(issued)
    return issued


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
    return replay_engineering_curve(
        method=method,
        target_domain=target_domain,
        specimen_sha256=specimen_sha256,
        task=task,
        grid_sha256=grid_sha256,
        evaluator_sha256=evaluator_sha256,
        warm_start_sha256=warm_start_sha256,
        exact_budgets=projected_budget,
        task_losses=projected_loss,
        projected_state_sha256=tuple(projected_state),
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
    "engineering_curve_payload",
    "oracle_gap_closure",
    "replay_engineering_curve",
    "validate_engineering_curve",
]
