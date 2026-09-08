"""Physical-specimen statistics and result assembly for the supplement."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .metrics import MetricRecord


@dataclass(frozen=True, slots=True, eq=False)
class DomainBootstrapDraws:
    domains: tuple[str, ...]
    specimens_by_domain: tuple[tuple[str, ...], ...]
    indices_by_domain: tuple[np.ndarray, ...]
    replicates: int
    seed: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PairedDomainBootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    physical_specimen_count: int
    domain_count: int
    replicates: int
    draws_sha256: str


def make_domain_bootstrap_draws(
    specimen_domains: Mapping[str, str], *, replicates: int, seed: int
) -> DomainBootstrapDraws:
    """Create one reusable domain-balanced physical-specimen draw set."""

    if (
        not isinstance(specimen_domains, Mapping)
        or not specimen_domains
        or any(
            type(specimen) is not str
            or not specimen
            or type(domain) is not str
            or not domain
            for specimen, domain in specimen_domains.items()
        )
        or type(replicates) is not int
        or replicates < 1
        or type(seed) is not int
    ):
        raise ValueError("domain bootstrap draw request is invalid")
    domains = tuple(sorted(set(specimen_domains.values())))
    specimens = tuple(
        tuple(
            sorted(
                specimen
                for specimen, candidate in specimen_domains.items()
                if candidate == domain
            )
        )
        for domain in domains
    )
    if any(not group for group in specimens):
        raise ValueError("domain bootstrap draw request is invalid")
    rng = np.random.default_rng(seed)
    indices = []
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"domains": domains, "specimens": specimens, "replicates": replicates, "seed": seed},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for group in specimens:
        values = rng.integers(0, len(group), size=(replicates, len(group)))
        values.setflags(write=False)
        digest.update(values.tobytes(order="C"))
        indices.append(values)
    return DomainBootstrapDraws(
        domains=domains,
        specimens_by_domain=specimens,
        indices_by_domain=tuple(indices),
        replicates=replicates,
        seed=seed,
        sha256=digest.hexdigest(),
    )


def paired_domain_bootstrap(
    records: tuple[MetricRecord, ...],
    *,
    treatment: str,
    comparator: str,
    confidence_level: float,
    draws: DomainBootstrapDraws,
) -> PairedDomainBootstrapResult:
    """Average seeds within specimen, then bootstrap paired domain effects."""

    confidence = float(confidence_level)
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not MetricRecord for row in records)
        or not treatment
        or not comparator
        or treatment == comparator
        or isinstance(confidence_level, bool)
        or not 0.0 < confidence < 1.0
        or type(draws) is not DomainBootstrapDraws
    ):
        raise ValueError("paired domain bootstrap request is invalid")
    tasks = {row.task for row in records}
    if len(tasks) != 1:
        raise ValueError("paired bootstrap requires one task")
    domain_by_specimen: dict[str, str] = {}
    grouped: dict[tuple[str, str, int], float] = {}
    seeds_by_specimen_method: dict[tuple[str, str], set[int]] = {}
    for row in records:
        previous = domain_by_specimen.setdefault(row.specimen_key, row.domain)
        if previous != row.domain:
            raise ValueError("one physical specimen appears in multiple domains")
        key = (row.specimen_key, row.method, row.seed)
        if key in grouped:
            raise ValueError("duplicate specimen/method/seed metric")
        grouped[key] = row.value
        seeds_by_specimen_method.setdefault(
            (row.specimen_key, row.method), set()
        ).add(row.seed)
    expected_specimens = {
        specimen
        for group in draws.specimens_by_domain
        for specimen in group
    }
    if set(domain_by_specimen) != expected_specimens:
        raise ValueError("bootstrap draws and metric specimens differ")
    for domain, group in zip(
        draws.domains, draws.specimens_by_domain, strict=True
    ):
        if any(domain_by_specimen[specimen] != domain for specimen in group):
            raise ValueError("bootstrap draw domain identity changed")

    method_seed_sets: dict[str, set[int]] = {}
    for specimen in sorted(expected_specimens):
        for method in (treatment, comparator):
            seeds = seeds_by_specimen_method.get((specimen, method))
            if not seeds:
                raise ValueError("paired method values are incomplete")
            previous = method_seed_sets.setdefault(method, set(seeds))
            if previous != seeds:
                raise ValueError("method seed coverage differs across specimens")

    effects: list[np.ndarray] = []
    for group in draws.specimens_by_domain:
        domain_effects = []
        for specimen in group:
            treatment_values = [
                grouped[(specimen, treatment, seed)]
                for seed in sorted(method_seed_sets[treatment])
            ]
            comparator_values = [
                grouped[(specimen, comparator, seed)]
                for seed in sorted(method_seed_sets[comparator])
            ]
            domain_effects.append(
                float(np.mean(treatment_values) - np.mean(comparator_values))
            )
        effects.append(np.asarray(domain_effects, dtype=np.float64))
    estimate = float(np.mean([float(values.mean()) for values in effects]))
    bootstrap = np.empty(draws.replicates, dtype=np.float64)
    for replicate in range(draws.replicates):
        bootstrap[replicate] = float(
            np.mean(
                [
                    float(values[indices[replicate]].mean())
                    for values, indices in zip(
                        effects, draws.indices_by_domain, strict=True
                    )
                ]
            )
        )
    alpha = (1.0 - confidence) / 2.0
    return PairedDomainBootstrapResult(
        estimate=estimate,
        ci_lower=float(np.quantile(bootstrap, alpha)),
        ci_upper=float(np.quantile(bootstrap, 1.0 - alpha)),
        confidence_level=confidence,
        physical_specimen_count=len(expected_specimens),
        domain_count=len(draws.domains),
        replicates=draws.replicates,
        draws_sha256=draws.sha256,
    )


def cost_at_success_rate(
    rows: Sequence[Mapping[str, object]], *, target: float
) -> float | None:
    """Return the first cost attaining a domain-balanced current-report rate."""

    threshold = float(target)
    if (
        not rows
        or isinstance(target, bool)
        or not math.isfinite(threshold)
        or not 0.0 < threshold <= 1.0
    ):
        raise ValueError("cost-at-success-rate request is invalid")
    required = {"dataset_id", "specimen_key", "seed", "step", "cost", "success"}
    if any(not isinstance(row, Mapping) or not required <= set(row) for row in rows):
        raise ValueError("cost-at-success-rate rows are invalid")
    domain_by_specimen: dict[str, str] = {}
    seeds_by_specimen: dict[str, set[int]] = {}
    events: dict[float, dict[tuple[str, int], bool]] = {}
    ordered = sorted(rows, key=lambda row: (float(row["cost"]), int(row["step"])))
    for row in ordered:
        specimen = str(row["specimen_key"])
        domain = str(row["dataset_id"])
        seed = row["seed"]
        step = row["step"]
        cost = row["cost"]
        success = row["success"]
        if (
            not specimen
            or not domain
            or type(seed) is not int
            or type(step) is not int
            or step < 0
            or isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or not 0.0 <= float(cost) <= 1.0
            or type(success) is not bool
            or domain_by_specimen.setdefault(specimen, domain) != domain
        ):
            raise ValueError("cost-at-success-rate rows are invalid")
        seeds_by_specimen.setdefault(specimen, set()).add(seed)
        events.setdefault(float(cost), {})[(specimen, seed)] = success
    states = {
        (specimen, seed): False
        for specimen, seeds in seeds_by_specimen.items()
        for seed in seeds
    }
    domains = tuple(sorted(set(domain_by_specimen.values())))
    for cost in sorted(events):
        states.update(events[cost])
        domain_rates = []
        for domain in domains:
            specimen_rates = []
            for specimen in sorted(
                candidate
                for candidate, value in domain_by_specimen.items()
                if value == domain
            ):
                specimen_rates.append(
                    float(
                        np.mean(
                            [
                                states[(specimen, seed)]
                                for seed in sorted(seeds_by_specimen[specimen])
                            ]
                        )
                    )
                )
            domain_rates.append(float(np.mean(specimen_rates)))
        if float(np.mean(domain_rates)) + 1e-15 >= threshold:
            return cost
    return None


__all__ = [
    "DomainBootstrapDraws",
    "PairedDomainBootstrapResult",
    "cost_at_success_rate",
    "make_domain_bootstrap_draws",
    "paired_domain_bootstrap",
]
