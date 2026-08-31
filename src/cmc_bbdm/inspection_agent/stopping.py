"""Source-selected fixed references and specimen-specific sufficiency stopping."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class InspectionStoppingError(ValueError):
    """Raised when a stopping reference or task-loss curve is invalid."""


@dataclass(frozen=True, slots=True)
class ReferenceEndpoint:
    method: str
    dataset_id: str
    specimen_id: str
    task_loss: float

    def __post_init__(self) -> None:
        loss = float(self.task_loss)
        if (
            any(type(value) is not str or not value for value in (
                self.method,
                self.dataset_id,
                self.specimen_id,
            ))
            or not math.isfinite(loss)
            or loss < 0.0
        ):
            raise InspectionStoppingError("reference endpoint is invalid")
        object.__setattr__(self, "task_loss", loss)


@dataclass(frozen=True, slots=True)
class FixedReferenceSelection:
    method: str
    equal_domain_loss: float
    domain_losses: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class StoppingResult:
    reached: bool
    stop_index: int
    budget_to_sufficiency: float
    normalized_measurement_saving: float
    final_task_loss: float
    reference_budget: float
    reference_loss: float
    threshold_loss: float


def select_strongest_fixed_reference(
    rows: tuple[ReferenceEndpoint, ...],
    *,
    allowed_methods: tuple[str, ...],
) -> FixedReferenceSelection:
    if (
        type(rows) is not tuple
        or not rows
        or any(type(row) is not ReferenceEndpoint for row in rows)
        or type(allowed_methods) is not tuple
        or not allowed_methods
        or len(set(allowed_methods)) != len(allowed_methods)
        or any(type(method) is not str or not method for method in allowed_methods)
    ):
        raise InspectionStoppingError("fixed reference request is invalid")
    selected_rows = tuple(row for row in rows if row.method in allowed_methods)
    if not selected_rows or {row.method for row in selected_rows} != set(allowed_methods):
        raise InspectionStoppingError("fixed reference method roster is incomplete")
    rosters = {
        method: {
            (row.dataset_id, row.specimen_id)
            for row in selected_rows
            if row.method == method
        }
        for method in allowed_methods
    }
    if len({frozenset(roster) for roster in rosters.values()}) != 1:
        raise InspectionStoppingError("fixed reference specimen rosters differ")
    candidates: list[FixedReferenceSelection] = []
    for method in allowed_methods:
        method_rows = tuple(row for row in selected_rows if row.method == method)
        domains = tuple(sorted({row.dataset_id for row in method_rows}))
        domain_losses = tuple(
            (
                domain,
                float(
                    np.mean(
                        [row.task_loss for row in method_rows if row.dataset_id == domain],
                        dtype=np.float64,
                    )
                ),
            )
            for domain in domains
        )
        candidates.append(
            FixedReferenceSelection(
                method=method,
                equal_domain_loss=float(
                    np.mean([value for _domain, value in domain_losses])
                ),
                domain_losses=domain_losses,
            )
        )
    return min(candidates, key=lambda item: (item.equal_domain_loss, item.method))


def earliest_sufficient_state(
    *,
    budgets: object,
    losses: object,
    reference_budget: float,
    reference_loss: float,
    tolerance: float,
) -> StoppingResult:
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    budget = float(reference_budget)
    loss = float(reference_loss)
    margin = float(tolerance)
    if (
        x.ndim != 1
        or x.size == 0
        or y.shape != x.shape
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(x < 0.0)
        or np.any(y < 0.0)
        or (x.size > 1 and np.any(np.diff(x) <= 0.0))
        or not math.isfinite(budget)
        or budget <= 0.0
        or not math.isfinite(loss)
        or loss < 0.0
        or not math.isfinite(margin)
        or margin < 0.0
        or x[-1] > budget + 1.0e-15
    ):
        raise InspectionStoppingError("sufficiency curve is invalid")
    threshold = (1.0 + margin) * loss
    passing = np.flatnonzero(y <= threshold)
    reached = bool(passing.size)
    index = int(passing[0]) if reached else len(x) - 1
    stop_budget = float(x[index])
    saving = float(max(0.0, 1.0 - stop_budget / budget)) if reached else 0.0
    return StoppingResult(
        reached=reached,
        stop_index=index,
        budget_to_sufficiency=stop_budget,
        normalized_measurement_saving=saving,
        final_task_loss=float(y[index]),
        reference_budget=budget,
        reference_loss=loss,
        threshold_loss=threshold,
    )


__all__ = [
    "FixedReferenceSelection",
    "InspectionStoppingError",
    "ReferenceEndpoint",
    "StoppingResult",
    "earliest_sufficient_state",
    "select_strongest_fixed_reference",
]
