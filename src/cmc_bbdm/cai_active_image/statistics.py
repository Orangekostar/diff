"""Registered CAI trajectory metrics and equal-domain paired inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalized_error_area_mpa(
    *,
    costs: np.ndarray,
    predictions_mpa: np.ndarray,
    target_mpa: float,
    endpoint_budget: float,
) -> float:
    x = np.asarray(costs, dtype=np.float64)
    predictions = np.asarray(predictions_mpa, dtype=np.float64)
    if (
        x.ndim != 1
        or predictions.shape != x.shape
        or len(x) < 2
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(predictions))
        or not np.isfinite(target_mpa)
        or endpoint_budget <= 0.0
        or x[0] != 0.0
        or np.any(np.diff(x) <= 0.0)
        or x[-1] > endpoint_budget + 1e-12
    ):
        raise ValueError("CAI error-area trajectory is invalid")
    if x[-1] < endpoint_budget:
        x = np.append(x, endpoint_budget)
        predictions = np.append(predictions, predictions[-1])
    errors = np.abs(predictions - float(target_mpa))
    return float(np.trapezoid(errors, x) / endpoint_budget)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    replicates: int
    seed: int


def domain_clustered_paired_bootstrap(
    rows: list[tuple[str, str, float]],
    *,
    replicates: int,
    seed: int,
) -> BootstrapInterval:
    if replicates < 2 or not rows:
        raise ValueError("bootstrap request is invalid")
    grouped: dict[str, dict[str, float]] = {}
    for domain, specimen, difference in rows:
        if not domain or not specimen or not np.isfinite(difference):
            raise ValueError("bootstrap row is invalid")
        specimens = grouped.setdefault(domain, {})
        if specimen in specimens:
            raise ValueError("bootstrap specimen is duplicated within domain")
        specimens[specimen] = float(difference)
    domains = tuple(sorted(grouped))
    estimate = float(
        np.mean([np.mean(tuple(grouped[domain].values())) for domain in domains])
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_domains = rng.choice(domains, size=len(domains), replace=True)
        means: list[float] = []
        for domain in sampled_domains:
            values = np.asarray(tuple(grouped[str(domain)].values()), dtype=np.float64)
            means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        samples[index] = float(np.mean(means))
    confidence = 1.0 - 0.05 / 3.0
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=estimate,
        lower=float(np.quantile(samples, tail)),
        upper=float(np.quantile(samples, 1.0 - tail)),
        confidence_level=confidence,
        replicates=replicates,
        seed=seed,
    )


__all__ = [
    "BootstrapInterval",
    "domain_clustered_paired_bootstrap",
    "normalized_error_area_mpa",
]
