"""Specimen-first synchronized bootstrap for six-domain G0 contrasts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


class InspectionStatisticsError(ValueError):
    """Raised when a paired specimen/domain contrast is invalid."""


@dataclass(frozen=True, slots=True)
class PairedBootstrapSummary:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    improved_domains: int
    domain_effects: tuple[tuple[str, float], ...]
    replicates: int
    seed: int
    distribution_sha256: str


def synchronized_paired_bootstrap(
    *,
    dataset_ids: tuple[str, ...],
    specimen_ids: tuple[str, ...],
    effects: object,
    replicates: int,
    seed: int,
) -> PairedBootstrapSummary:
    values = np.asarray(effects, dtype=np.float64)
    count = len(dataset_ids)
    if (
        type(dataset_ids) is not tuple
        or type(specimen_ids) is not tuple
        or not dataset_ids
        or len(specimen_ids) != count
        or values.shape != (count,)
        or not np.all(np.isfinite(values))
        or any(type(value) is not str or not value for value in (*dataset_ids, *specimen_ids))
        or type(replicates) is not int
        or replicates < 100
        or type(seed) is not int
    ):
        raise InspectionStatisticsError("paired bootstrap request is invalid")
    keys = tuple(zip(dataset_ids, specimen_ids, strict=True))
    if len(set(keys)) != count:
        raise InspectionStatisticsError("each specimen must have one paired effect")
    ordered = sorted(
        zip(dataset_ids, specimen_ids, values, strict=True),
        key=lambda row: (row[0], row[1]),
    )
    domains = tuple(sorted(set(dataset_ids)))
    by_domain: dict[str, np.ndarray] = {}
    for domain in domains:
        domain_values = np.asarray(
            [float(value) for row_domain, _specimen, value in ordered if row_domain == domain],
            dtype=np.float64,
        )
        if domain_values.size == 0:
            raise InspectionStatisticsError("paired bootstrap domain is empty")
        by_domain[domain] = domain_values
    domain_effects = tuple(
        (domain, float(np.mean(by_domain[domain], dtype=np.float64)))
        for domain in domains
    )
    point = float(np.mean([value for _domain, value in domain_effects]))
    distribution = np.empty(replicates, dtype="<f8")
    generator = np.random.Generator(np.random.PCG64(seed))
    chunk_size = 2048
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        aggregate = np.zeros(size, dtype=np.float64)
        for domain in domains:
            domain_values = by_domain[domain]
            indices = generator.integers(
                0,
                len(domain_values),
                size=(size, len(domain_values)),
            )
            aggregate += np.mean(domain_values[indices], axis=1, dtype=np.float64)
        distribution[start : start + size] = aggregate / len(domains)
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    if not all(math.isfinite(value) for value in (point, float(lower), float(upper))):
        raise InspectionStatisticsError("paired bootstrap result is nonfinite")
    return PairedBootstrapSummary(
        point_estimate=point,
        ci_lower=float(lower),
        ci_upper=float(upper),
        improved_domains=sum(value > 0.0 for _domain, value in domain_effects),
        domain_effects=domain_effects,
        replicates=replicates,
        seed=seed,
        distribution_sha256=hashlib.sha256(
            distribution.tobytes(order="C")
        ).hexdigest(),
    )


__all__ = [
    "InspectionStatisticsError",
    "PairedBootstrapSummary",
    "synchronized_paired_bootstrap",
]
