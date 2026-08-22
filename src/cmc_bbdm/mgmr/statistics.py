"""Prediction-derived metrics, paired domain bootstrap, and M0 gate logic."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .evaluation import PredictionRecord


class MGMRStatisticsError(ValueError):
    """Raised when metrics cannot be derived from aligned prediction evidence."""


@dataclass(frozen=True, slots=True)
class DomainMetric:
    method: str
    domain: str
    specimen_count: int
    mae: float
    pearson: float
    spearman: float


@dataclass(frozen=True, slots=True)
class MetricSummary:
    method: str
    specimen_count: int
    specimen_mae: float
    equal_domain_mae: float
    worst_domain_mae: float
    pearson: float
    spearman: float
    domain_metrics: tuple[DomainMetric, ...]


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    effect: str
    estimate: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class DomainBootstrap:
    domain_order: tuple[str, ...]
    seed: int
    resamples: int
    quantiles: tuple[float, float]
    draw_sha256: str
    intervals: Mapping[str, BootstrapInterval]


@dataclass(frozen=True, slots=True)
class M0Decision:
    status: str
    gates: Mapping[str, bool]
    required_gates: tuple[str, ...]
    improved_domains: Mapping[str, int]
    benefits: Mapping[str, float]


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return 0.0
    left_centered = left - np.mean(left, dtype=np.float64)
    right_centered = right - np.mean(right, dtype=np.float64)
    denominator = float(
        np.sqrt(
            np.sum(left_centered * left_centered, dtype=np.float64)
            * np.sum(right_centered * right_centered, dtype=np.float64)
        )
    )
    if denominator == 0.0:
        return 0.0
    value = float(
        np.sum(left_centered * right_centered, dtype=np.float64) / denominator
    )
    return min(1.0, max(-1.0, value))


def prediction_metrics(
    records: Sequence[PredictionRecord], *, domain_order: Sequence[str]
) -> MetricSummary:
    """Aggregate metrics only from immutable-style outer prediction rows."""

    rows = tuple(records)
    domains = tuple(domain_order)
    if (
        not rows
        or not domains
        or len(set(domains)) != len(domains)
        or any(type(row) is not PredictionRecord for row in rows)
    ):
        raise MGMRStatisticsError("prediction records or domain order are invalid")
    method = rows[0].method
    if (
        any(row.method != method for row in rows)
        or len({row.specimen_id for row in rows}) != len(rows)
        or {row.dataset_id for row in rows} != set(domains)
    ):
        raise MGMRStatisticsError("prediction record roster is inconsistent")
    targets = np.asarray([row.target for row in rows], dtype=np.float64)
    predictions = np.asarray([row.prediction for row in rows], dtype=np.float64)
    dataset_ids = np.asarray([row.dataset_id for row in rows], dtype=object)
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(predictions)):
        raise MGMRStatisticsError("prediction records contain non-finite values")
    errors = np.abs(targets - predictions)
    domain_rows: list[DomainMetric] = []
    for domain in domains:
        mask = dataset_ids == domain
        domain_target = targets[mask]
        domain_prediction = predictions[mask]
        if not domain_target.size:
            raise MGMRStatisticsError("prediction records omit a registered domain")
        domain_rows.append(
            DomainMetric(
                method=method,
                domain=domain,
                specimen_count=int(domain_target.size),
                mae=float(np.mean(np.abs(domain_target - domain_prediction))),
                pearson=_correlation(domain_target, domain_prediction),
                spearman=_correlation(_ranks(domain_target), _ranks(domain_prediction)),
            )
        )
    equal = float(math.fsum(row.mae for row in domain_rows) / len(domain_rows))
    return MetricSummary(
        method=method,
        specimen_count=len(rows),
        specimen_mae=float(np.mean(errors, dtype=np.float64)),
        equal_domain_mae=equal,
        worst_domain_mae=max(row.mae for row in domain_rows),
        pearson=_correlation(targets, predictions),
        spearman=_correlation(_ranks(targets), _ranks(predictions)),
        domain_metrics=tuple(domain_rows),
    )


def paired_domain_bootstrap(
    effects: Mapping[str, Sequence[float]],
    *,
    domain_order: Sequence[str],
    seed: int,
    resamples: int,
    quantiles: tuple[float, float],
) -> DomainBootstrap:
    """Bootstrap every six-domain effect with one synchronized draw matrix."""

    domains = tuple(domain_order)
    if (
        not isinstance(effects, Mapping)
        or not effects
        or len(domains) < 2
        or len(set(domains)) != len(domains)
        or type(seed) is not int
        or type(resamples) is not int
        or resamples <= 0
        or len(quantiles) != 2
        or not 0.0 <= quantiles[0] < quantiles[1] <= 1.0
    ):
        raise MGMRStatisticsError("domain bootstrap request is invalid")
    arrays: dict[str, np.ndarray] = {}
    for name, value in effects.items():
        array = np.asarray(value, dtype=np.float64)
        if (
            type(name) is not str
            or not name
            or array.shape != (len(domains),)
            or not np.all(np.isfinite(array))
        ):
            raise MGMRStatisticsError("domain bootstrap effect is invalid")
        arrays[name] = array
    draws = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(domains), size=(resamples, len(domains)), dtype=np.int64
    )
    draw_sha = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
    intervals: dict[str, BootstrapInterval] = {}
    for name, values in arrays.items():
        distribution = np.mean(values[draws], axis=1, dtype=np.float64)
        low, high = np.quantile(distribution, quantiles)
        intervals[name] = BootstrapInterval(
            effect=name,
            estimate=float(np.mean(values, dtype=np.float64)),
            low=float(low),
            high=float(high),
        )
    return DomainBootstrap(
        domain_order=domains,
        seed=seed,
        resamples=resamples,
        quantiles=(float(quantiles[0]), float(quantiles[1])),
        draw_sha256=draw_sha,
        intervals=MappingProxyType(intervals),
    )


def _domain_mae(metric: MetricSummary) -> tuple[float, ...]:
    if type(metric) is not MetricSummary or not metric.domain_metrics:
        raise MGMRStatisticsError("gate metric is invalid")
    return tuple(row.mae for row in metric.domain_metrics)


def _benefit(reference: MetricSummary, candidate: MetricSummary) -> tuple[float, ...]:
    reference_values = _domain_mae(reference)
    candidate_values = _domain_mae(candidate)
    reference_domains = tuple(row.domain for row in reference.domain_metrics)
    candidate_domains = tuple(row.domain for row in candidate.domain_metrics)
    if reference_domains != candidate_domains:
        raise MGMRStatisticsError("gate metric domains are not aligned")
    return tuple(
        left - right
        for left, right in zip(reference_values, candidate_values, strict=True)
    )


def decide_m0(
    *,
    direct: Mapping[str, MetricSummary],
    coarse_baseline: MetricSummary,
    coarse_corrected: MetricSummary,
    full_baseline: MetricSummary,
    full_corrected: MetricSummary,
    shuffled: Mapping[int, MetricSummary],
    required_gates: Sequence[str],
    minimum_positive_domains: int,
) -> M0Decision:
    """Apply the frozen strict A/B/C/D decision without rounding."""

    if set(direct) != {"B1", "B2", "B3"} or not shuffled:
        raise MGMRStatisticsError("gate method roster is incomplete")
    required = tuple(required_gates)
    if required != ("A", "B", "D") or type(minimum_positive_domains) is not int:
        raise MGMRStatisticsError("gate authority changed")
    b1, b2, b3 = direct["B1"], direct["B2"], direct["B3"]
    a_effect = _benefit(b1, b3)
    coarse_effect = _benefit(coarse_baseline, coarse_corrected)
    full_effect = _benefit(full_baseline, full_corrected)
    shuffle_effects = {
        seed: _benefit(coarse_baseline, metric) for seed, metric in shuffled.items()
    }
    real_benefit = float(math.fsum(coarse_effect) / len(coarse_effect))
    shuffle_benefits = {
        seed: float(math.fsum(values) / len(values))
        for seed, values in shuffle_effects.items()
    }
    mean_shuffle_by_domain = tuple(
        math.fsum(values[index] for values in shuffle_effects.values())
        / len(shuffle_effects)
        for index in range(len(coarse_effect))
    )
    improved = {
        "A": sum(value > 0.0 for value in a_effect),
        "B": sum(value > 0.0 for value in coarse_effect),
        "C": sum(value > 0.0 for value in full_effect),
        "D": sum(
            real > shuffled_value
            for real, shuffled_value in zip(
                coarse_effect, mean_shuffle_by_domain, strict=True
            )
        ),
    }
    gates = {
        "A": (
            b3.equal_domain_mae < b1.equal_domain_mae
            and b3.equal_domain_mae < b2.equal_domain_mae
            and improved["A"] >= minimum_positive_domains
        ),
        "B": real_benefit > 0.0 and improved["B"] >= minimum_positive_domains,
        "C": (
            full_baseline.equal_domain_mae - full_corrected.equal_domain_mae > 0.0
            and improved["C"] >= minimum_positive_domains
        ),
        "D": (
            all(real_benefit > value for value in shuffle_benefits.values())
            and improved["D"] >= minimum_positive_domains
        ),
    }
    benefits = {"coarse_real": real_benefit}
    benefits.update(
        {f"coarse_shuffle_{seed}": value for seed, value in shuffle_benefits.items()}
    )
    benefits["full_real"] = float(math.fsum(full_effect) / len(full_effect))
    status = "MGMR_GO" if all(gates[name] for name in required) else "MGMR_NO_GO"
    return M0Decision(
        status=status,
        gates=MappingProxyType(gates),
        required_gates=required,
        improved_domains=MappingProxyType(improved),
        benefits=MappingProxyType(benefits),
    )


__all__ = [
    "BootstrapInterval",
    "DomainBootstrap",
    "DomainMetric",
    "M0Decision",
    "MGMRStatisticsError",
    "MetricSummary",
    "decide_m0",
    "paired_domain_bootstrap",
    "prediction_metrics",
]
