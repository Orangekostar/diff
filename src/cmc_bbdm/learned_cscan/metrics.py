"""Exact action-step metrics and physical-specimen statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np

from .contracts import ReferenceEvidence, Task


@dataclass(frozen=True, slots=True)
class StepSnapshot:
    cost: float
    success: bool
    task_loss: float
    report_digest: str

    def __post_init__(self) -> None:
        cost = float(self.cost)
        loss = float(self.task_loss)
        if (
            isinstance(self.cost, bool)
            or not math.isfinite(cost)
            or not 0.0 <= cost <= 1.0
            or type(self.success) is not bool
            or isinstance(self.task_loss, bool)
            or not math.isfinite(loss)
            or not 0.0 <= loss <= 1.0
            or not self.report_digest
        ):
            raise ValueError("step snapshot is invalid")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "task_loss", loss)


@dataclass(frozen=True, slots=True)
class MetricRecord:
    specimen_key: str
    domain: str
    method: str
    task: Task
    seed: int
    value: float

    def __post_init__(self) -> None:
        value = float(self.value)
        if (
            not self.specimen_key
            or not self.domain
            or not self.method
            or type(self.task) is not Task
            or type(self.seed) is not int
            or not math.isfinite(value)
        ):
            raise ValueError("metric record is invalid")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    physical_specimen_count: int
    domain_count: int
    replicates: int


@dataclass(frozen=True, slots=True)
class ScopedEffect:
    proxy_effect: PairedBootstrapResult
    formal_effect: PairedBootstrapResult | None


def exact_step_integral(
    snapshots: tuple[StepSnapshot, ...],
    *,
    field: Literal["success", "failure", "task_loss"],
    start_cost: float,
    end_cost: float,
) -> float:
    checked = _checked_snapshots(snapshots)
    start = float(start_cost)
    end = float(end_cost)
    if (
        isinstance(start_cost, bool)
        or isinstance(end_cost, bool)
        or not math.isfinite(start)
        or not math.isfinite(end)
        or not 0.0 <= start < end <= 1.0
        or field not in {"success", "failure", "task_loss"}
    ):
        raise ValueError("step integral interval is invalid")
    collapsed: list[StepSnapshot] = []
    for snapshot in checked:
        if collapsed and math.isclose(snapshot.cost, collapsed[-1].cost, abs_tol=1e-15):
            collapsed[-1] = snapshot
        else:
            collapsed.append(snapshot)
    preceding = [snapshot for snapshot in collapsed if snapshot.cost <= start + 1e-15]
    if not preceding:
        raise ValueError("step integral lacks a report at the start cost")
    current = preceding[-1]
    cursor = start
    area = 0.0
    for snapshot in collapsed:
        if snapshot.cost <= start + 1e-15:
            continue
        boundary = min(snapshot.cost, end)
        area += (boundary - cursor) * _snapshot_value(current, field)
        cursor = boundary
        if snapshot.cost >= end - 1e-15:
            break
        current = snapshot
    if cursor < end:
        area += (end - cursor) * _snapshot_value(current, field)
    return float(area)


def value_at_cost(
    snapshots: tuple[StepSnapshot, ...],
    *,
    cost: float,
) -> StepSnapshot:
    checked = _checked_snapshots(snapshots)
    query = float(cost)
    if isinstance(cost, bool) or not math.isfinite(query) or not 0.0 <= query <= 1.0:
        raise ValueError("snapshot query cost is invalid")
    available = [snapshot for snapshot in checked if snapshot.cost <= query + 1e-15]
    if not available:
        raise ValueError("no snapshot is available at the query cost")
    return available[-1]


def failure_penalized_cost(terminal_cost: float, *, success: bool) -> float:
    cost = float(terminal_cost)
    if (
        isinstance(terminal_cost, bool)
        or not math.isfinite(cost)
        or not 0.0 <= cost <= 1.0
        or type(success) is not bool
    ):
        raise ValueError("failure-penalized cost request is invalid")
    return cost if success else 1.0


def paired_physical_specimen_bootstrap(
    records: tuple[MetricRecord, ...],
    *,
    treatment: str,
    comparator: str,
    replicates: int,
    seed: int,
) -> PairedBootstrapResult:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not MetricRecord for row in records)
        or not treatment
        or not comparator
        or treatment == comparator
        or type(replicates) is not int
        or replicates < 1
        or type(seed) is not int
    ):
        raise ValueError("paired bootstrap request is invalid")
    specimen_domains: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in records:
        previous = specimen_domains.setdefault(row.specimen_key, row.domain)
        if previous != row.domain:
            raise ValueError("one physical specimen appears in multiple domains")
        grouped.setdefault((row.specimen_key, row.method), []).append(row.value)
    effects_by_domain: dict[str, list[float]] = {}
    for specimen, domain in specimen_domains.items():
        treatment_values = grouped.get((specimen, treatment))
        comparator_values = grouped.get((specimen, comparator))
        if not treatment_values or not comparator_values:
            raise ValueError("paired method values are incomplete")
        effect = float(np.mean(treatment_values) - np.mean(comparator_values))
        effects_by_domain.setdefault(domain, []).append(effect)
    domains = tuple(sorted(effects_by_domain))
    estimate = float(
        np.mean([np.mean(effects_by_domain[domain]) for domain in domains])
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        domain_means = []
        for domain in domains:
            values = np.asarray(effects_by_domain[domain], dtype=np.float64)
            indices = rng.integers(0, len(values), size=len(values))
            domain_means.append(float(np.mean(values[indices])))
        bootstrap[replicate] = float(np.mean(domain_means))
    return PairedBootstrapResult(
        estimate=estimate,
        ci_lower=float(np.quantile(bootstrap, 0.025)),
        ci_upper=float(np.quantile(bootstrap, 0.975)),
        physical_specimen_count=len(specimen_domains),
        domain_count=len(domains),
        replicates=replicates,
    )


def scope_effect(
    effect: PairedBootstrapResult,
    *,
    references: tuple[ReferenceEvidence, ...],
) -> ScopedEffect:
    if (
        type(effect) is not PairedBootstrapResult
        or type(references) is not tuple
        or not references
        or any(type(reference) is not ReferenceEvidence for reference in references)
    ):
        raise ValueError("effect scope request is invalid")
    return ScopedEffect(
        proxy_effect=effect,
        formal_effect=(effect if all(row.formal_eligible for row in references) else None),
    )


def _checked_snapshots(
    snapshots: tuple[StepSnapshot, ...],
) -> tuple[StepSnapshot, ...]:
    if (
        type(snapshots) is not tuple
        or not snapshots
        or any(type(snapshot) is not StepSnapshot for snapshot in snapshots)
        or any(
            right.cost < left.cost - 1e-15
            for left, right in pairwise(snapshots)
        )
    ):
        raise ValueError("ordered step snapshots are required")
    return snapshots


def _snapshot_value(
    snapshot: StepSnapshot, field: Literal["success", "failure", "task_loss"]
) -> float:
    if field == "success":
        return float(snapshot.success)
    if field == "failure":
        return float(not snapshot.success)
    return snapshot.task_loss


__all__ = [
    "MetricRecord",
    "PairedBootstrapResult",
    "ScopedEffect",
    "StepSnapshot",
    "exact_step_integral",
    "failure_penalized_cost",
    "paired_physical_specimen_bootstrap",
    "scope_effect",
    "value_at_cost",
]
