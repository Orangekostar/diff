"""Completion-rate and paired-domain metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .contracts import EvaluationMode


@dataclass(frozen=True, slots=True)
class CurveSnapshot:
    cost: float
    success: bool
    stopped: bool


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float


def success_curve(
    snapshots: tuple[CurveSnapshot, ...],
    *,
    mode: EvaluationMode,
    checkpoints: tuple[float, ...],
) -> tuple[float, ...]:
    if (
        type(mode) is not EvaluationMode
        or not checkpoints
        or any(not 0.0 <= float(value) <= 1.0 for value in checkpoints)
        or tuple(sorted(checkpoints)) != checkpoints
    ):
        raise ValueError("success curve request is invalid")
    ordered = tuple(sorted(snapshots, key=lambda item: item.cost))
    if any(
        type(item) is not CurveSnapshot
        or not math.isfinite(item.cost)
        or not 0.0 <= item.cost <= 1.0
        for item in ordered
    ):
        raise ValueError("curve snapshots are invalid")
    output = []
    stopped = next((item for item in ordered if item.stopped), None)
    for checkpoint in checkpoints:
        available = [item for item in ordered if item.cost <= checkpoint]
        if mode is EvaluationMode.ANYTIME_REPORT:
            output.append(float(available[-1].success) if available else 0.0)
        else:
            output.append(
                float(stopped.success)
                if stopped is not None and stopped.cost <= checkpoint
                else 0.0
            )
    return tuple(output)


def paired_domain_bootstrap(
    differences_by_domain: dict[str, np.ndarray], *, replicates: int, seed: int
) -> BootstrapResult:
    if type(replicates) is not int or replicates < 2 or type(seed) is not int:
        raise ValueError("bootstrap configuration is invalid")
    arrays: list[np.ndarray] = []
    for domain in sorted(differences_by_domain):
        values = np.asarray(differences_by_domain[domain], dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("bootstrap domain values are invalid")
        arrays.append(values)
    if not arrays:
        raise ValueError("bootstrap requires domains")
    estimate = float(np.mean([float(values.mean()) for values in arrays]))
    generator = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        domain_means = []
        for values in arrays:
            sampled = generator.choice(values, size=len(values), replace=True)
            domain_means.append(float(sampled.mean()))
        draws[index] = float(np.mean(domain_means))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return BootstrapResult(estimate, float(lower), float(upper))


__all__ = [
    "BootstrapResult",
    "CurveSnapshot",
    "paired_domain_bootstrap",
    "success_curve",
]
