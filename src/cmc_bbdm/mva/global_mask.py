"""Leakage-safe aggregation for source-learned global acquisition masks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


class GlobalMaskError(ValueError):
    """Raised when source-value rows cannot define a valid static ranking."""


@dataclass(frozen=True, slots=True)
class GlobalMaskRanking:
    outer_domain: str
    method: str
    cell_order: tuple[int, ...]
    cell_scores: tuple[float, ...]
    mean_raw_values: tuple[float, ...]
    mean_value_per_measurement: tuple[float, ...]
    source_domains: tuple[str, ...]
    source_specimen_count: int


@dataclass(frozen=True, slots=True)
class SourceDomainStability:
    outer_domain: str
    method: str
    removed_domain: str
    top1_agreement: bool
    top10_overlap: float
    spearman: float
    rbo_p0_9: float


def normalized_candidate_ranks(values: Sequence[float]) -> np.ndarray:
    """Map a deterministic descending candidate order onto the interval [0, 1]."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise GlobalMaskError("candidate values must be a finite one-dimensional set")
    order = sorted(range(array.size), key=lambda index: (-float(array[index]), index))
    output = np.empty(array.size, dtype=np.float64)
    denominator = float(array.size - 1)
    for position, index in enumerate(order):
        output[index] = 1.0 - position / denominator
    output.setflags(write=False)
    return output


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise GlobalMaskError(f"{label} is invalid")
    return value


def _cell(value: object, cell_count: int) -> int:
    if type(value) is not int or not 0 <= value < cell_count:
        raise GlobalMaskError("candidate cell index is invalid")
    return value


def _value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GlobalMaskError("candidate value must be finite")
    output = float(value)
    if not math.isfinite(output):
        raise GlobalMaskError("candidate value must be finite")
    return output


def _measurement_count(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise GlobalMaskError("added measurement count must be positive")
    return value


def aggregate_global_ranking(
    rows: Sequence[Mapping[str, object]],
    *,
    outer_domain: str,
    method: str,
    domain_order: Sequence[str],
    cell_count: int = 64,
) -> GlobalMaskRanking:
    """Aggregate specimen ranks within domain and then weight source domains equally."""

    if type(cell_count) is not int or cell_count < 2:
        raise GlobalMaskError("cell count must be an integer of at least two")
    domains = tuple(_identity(domain, "domain") for domain in domain_order)
    if len(domains) < 2 or len(set(domains)) != len(domains):
        raise GlobalMaskError("domain order is invalid")
    outer = _identity(outer_domain, "outer domain")
    selected_method = _identity(method, "method")
    if outer not in domains:
        raise GlobalMaskError("outer domain is not registered")
    source_domains = tuple(domain for domain in domains if domain != outer)
    grouped: dict[tuple[str, str], dict[int, tuple[float, int]]] = {}
    for row in rows:
        specimen_id = _identity(row.get("specimen_id"), "specimen ID")
        dataset_id = _identity(row.get("dataset_id"), "dataset ID")
        row_method = _identity(row.get("method"), "method")
        if dataset_id == outer:
            raise GlobalMaskError("target rows cannot train a global mask")
        if dataset_id not in source_domains:
            raise GlobalMaskError("source row has an unregistered domain")
        if row_method != selected_method:
            raise GlobalMaskError("source rows contain a different method")
        cell_index = _cell(row.get("cell_index"), cell_count)
        primary_value = _value(row.get("primary_value"))
        added = _measurement_count(row.get("added_measurements"))
        key = (dataset_id, specimen_id)
        specimen = grouped.setdefault(key, {})
        if cell_index in specimen:
            raise GlobalMaskError("candidate cell is duplicated")
        specimen[cell_index] = (primary_value, added)
    if not grouped:
        raise GlobalMaskError("source candidate rows are empty")
    observed_domains = {domain for domain, _ in grouped}
    if observed_domains != set(source_domains):
        raise GlobalMaskError("source domain roster changed")

    expected_cells = set(range(cell_count))
    domain_rank_sums = {
        domain: np.zeros(cell_count, dtype=np.float64) for domain in source_domains
    }
    domain_raw_sums = {
        domain: np.zeros(cell_count, dtype=np.float64) for domain in source_domains
    }
    domain_cost_sums = {
        domain: np.zeros(cell_count, dtype=np.float64) for domain in source_domains
    }
    domain_counts = {domain: 0 for domain in source_domains}
    for (domain, _specimen_id), candidates in sorted(grouped.items()):
        if set(candidates) != expected_cells:
            raise GlobalMaskError("specimen candidate roster is incomplete")
        values = np.asarray(
            [candidates[index][0] for index in range(cell_count)], dtype=np.float64
        )
        costs = np.asarray(
            [candidates[index][1] for index in range(cell_count)], dtype=np.float64
        )
        domain_rank_sums[domain] += normalized_candidate_ranks(values)
        domain_raw_sums[domain] += values
        domain_cost_sums[domain] += values / costs
        domain_counts[domain] += 1

    domain_rank_means = np.vstack(
        [domain_rank_sums[domain] / domain_counts[domain] for domain in source_domains]
    )
    domain_raw_means = np.vstack(
        [domain_raw_sums[domain] / domain_counts[domain] for domain in source_domains]
    )
    domain_cost_means = np.vstack(
        [domain_cost_sums[domain] / domain_counts[domain] for domain in source_domains]
    )
    scores = np.mean(domain_rank_means, axis=0, dtype=np.float64)
    raw = np.mean(domain_raw_means, axis=0, dtype=np.float64)
    per_measurement = np.mean(domain_cost_means, axis=0, dtype=np.float64)
    order = tuple(
        sorted(range(cell_count), key=lambda index: (-float(scores[index]), index))
    )
    return GlobalMaskRanking(
        outer_domain=outer,
        method=selected_method,
        cell_order=order,
        cell_scores=tuple(float(value) for value in scores),
        mean_raw_values=tuple(float(value) for value in raw),
        mean_value_per_measurement=tuple(float(value) for value in per_measurement),
        source_domains=source_domains,
        source_specimen_count=len(grouped),
    )


def _rbo(first: Sequence[int], second: Sequence[int], persistence: float = 0.9) -> float:
    first_seen: set[int] = set()
    second_seen: set[int] = set()
    overlap = 0.0
    score = 0.0
    for depth, (left, right) in enumerate(zip(first, second, strict=True), start=1):
        first_seen.add(left)
        second_seen.add(right)
        overlap = len(first_seen & second_seen) / depth
        score += (1.0 - persistence) * overlap * persistence ** (depth - 1)
    return float(score + overlap * persistence ** len(first))


def leave_one_source_domain_out_stability(
    rows: Sequence[Mapping[str, object]],
    *,
    primary: GlobalMaskRanking,
    domain_order: Sequence[str],
    cell_count: int = 64,
) -> tuple[SourceDomainStability, ...]:
    """Recompute nonselecting rankings after removing each source domain."""

    output: list[SourceDomainStability] = []
    primary_scores = np.asarray(primary.cell_scores, dtype=np.float64)
    top = max(1, math.ceil(0.1 * cell_count))
    for removed in primary.source_domains:
        reduced_rows = [row for row in rows if row.get("dataset_id") != removed]
        reduced_order = tuple(domain for domain in domain_order if domain != removed)
        reduced = aggregate_global_ranking(
            reduced_rows,
            outer_domain=primary.outer_domain,
            method=primary.method,
            domain_order=reduced_order,
            cell_count=cell_count,
        )
        reduced_scores = np.asarray(reduced.cell_scores, dtype=np.float64)
        correlation = float(spearmanr(primary_scores, reduced_scores).statistic)
        if not math.isfinite(correlation):
            correlation = 1.0 if np.array_equal(primary_scores, reduced_scores) else 0.0
        output.append(
            SourceDomainStability(
                outer_domain=primary.outer_domain,
                method=primary.method,
                removed_domain=removed,
                top1_agreement=primary.cell_order[0] == reduced.cell_order[0],
                top10_overlap=float(
                    len(set(primary.cell_order[:top]) & set(reduced.cell_order[:top]))
                    / top
                ),
                spearman=correlation,
                rbo_p0_9=_rbo(primary.cell_order, reduced.cell_order),
            )
        )
    return tuple(output)


__all__ = [
    "GlobalMaskError",
    "GlobalMaskRanking",
    "SourceDomainStability",
    "aggregate_global_ranking",
    "leave_one_source_domain_out_stability",
    "normalized_candidate_ranks",
]
