"""Formal six-domain specimen-paired bootstrap for G1 effects."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

FORMAL_BOOTSTRAP_REPLICATES = 100_000
FORMAL_BOOTSTRAP_SEED = 2026090104


class G1StatisticsError(ValueError):
    """Raised when formal G1 inference violates its paired-domain contract."""


@dataclass(frozen=True, slots=True, eq=False)
class G1PairedBootstrap:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    improved_domains: int
    domain_effects: tuple[tuple[str, float], ...]
    replicates: int
    seed: int
    distribution: np.ndarray
    distribution_sha256: str


def formal_synchronized_bootstrap(
    *,
    dataset_ids: tuple[str, ...],
    specimen_ids: tuple[str, ...],
    baseline_values: object,
    learned_values: object,
    seed: int,
) -> G1PairedBootstrap:
    baseline = np.asarray(baseline_values, dtype=np.float64)
    learned = np.asarray(learned_values, dtype=np.float64)
    count = len(dataset_ids)
    if (
        type(dataset_ids) is not tuple
        or type(specimen_ids) is not tuple
        or not dataset_ids
        or len(specimen_ids) != count
        or baseline.shape != (count,)
        or learned.shape != (count,)
        or not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(learned))
        or np.any(baseline < 0.0)
        or np.any(learned < 0.0)
        or any(
            type(value) is not str or not value
            for value in (*dataset_ids, *specimen_ids)
        )
        or len(set(dataset_ids)) != 6
        or type(seed) is not int
        or seed != FORMAL_BOOTSTRAP_SEED
    ):
        raise G1StatisticsError("formal paired-bootstrap request is invalid")
    keys = tuple(zip(dataset_ids, specimen_ids, strict=True))
    if len(set(keys)) != count:
        raise G1StatisticsError("formal effect requires one row per physical specimen")
    effects = baseline - learned
    ordered = sorted(
        zip(dataset_ids, specimen_ids, effects, strict=True),
        key=lambda row: (row[0], row[1]),
    )
    domains = tuple(sorted(set(dataset_ids)))
    by_domain = {
        domain: np.asarray(
            [float(effect) for row_domain, _specimen, effect in ordered if row_domain == domain],
            dtype=np.float64,
        )
        for domain in domains
    }
    if any(values.size == 0 for values in by_domain.values()):
        raise G1StatisticsError("formal paired-bootstrap domain is empty")
    domain_effects = tuple(
        (domain, float(np.mean(by_domain[domain], dtype=np.float64)))
        for domain in domains
    )
    point = float(np.mean([effect for _domain, effect in domain_effects]))
    distribution = np.empty(FORMAL_BOOTSTRAP_REPLICATES, dtype="<f8")
    generator = np.random.Generator(np.random.PCG64(seed))
    chunk_size = 2_048
    for start in range(0, FORMAL_BOOTSTRAP_REPLICATES, chunk_size):
        size = min(chunk_size, FORMAL_BOOTSTRAP_REPLICATES - start)
        aggregate = np.zeros(size, dtype=np.float64)
        for domain in domains:
            values = by_domain[domain]
            indices = generator.integers(0, len(values), size=(size, len(values)))
            aggregate += np.mean(values[indices], axis=1, dtype=np.float64)
        distribution[start : start + size] = aggregate / len(domains)
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    if not all(math.isfinite(value) for value in (point, float(lower), float(upper))):
        raise G1StatisticsError("formal paired-bootstrap result is nonfinite")
    frozen = np.frombuffer(
        distribution.tobytes(order="C"), dtype="<f8"
    ).reshape((FORMAL_BOOTSTRAP_REPLICATES,))
    frozen.setflags(write=False)
    return G1PairedBootstrap(
        point_estimate=point,
        ci_lower=float(lower),
        ci_upper=float(upper),
        improved_domains=sum(effect > 0.0 for _domain, effect in domain_effects),
        domain_effects=domain_effects,
        replicates=FORMAL_BOOTSTRAP_REPLICATES,
        seed=seed,
        distribution=frozen,
        distribution_sha256=hashlib.sha256(frozen.tobytes(order="C")).hexdigest(),
    )


__all__ = [
    "FORMAL_BOOTSTRAP_REPLICATES",
    "FORMAL_BOOTSTRAP_SEED",
    "G1PairedBootstrap",
    "G1StatisticsError",
    "formal_synchronized_bootstrap",
]
