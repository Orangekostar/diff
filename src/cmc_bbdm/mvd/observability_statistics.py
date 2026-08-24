"""Equal-domain M1 aggregation and the frozen observability gate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from cmc_bbdm.mva.statistics import (
    BootstrapEffect,
    paired_domain_bootstrap,
    synchronized_bootstrap_indices,
)

_PRIMARY = "o2_global_candidate"
_GLOBAL = "global_mechanical"
_RANDOM = "random_median"
_METRICS = (
    "spearman",
    "ndcg_5",
    "ndcg_10",
    "recall_5",
    "recall_10",
    "regret_1",
    "mean_budgeted_regret",
)


@dataclass(frozen=True, slots=True)
class ObservabilityGate:
    status: str
    go: bool
    spearman_pass: bool
    ndcg_global_pass: bool
    regret_global_pass: bool
    regret_random_pass: bool


@dataclass(frozen=True, slots=True)
class ObservabilityAggregation:
    domain_order: tuple[str, ...]
    domain_metrics: tuple[dict[str, object], ...]
    model_metrics: tuple[dict[str, object], ...]
    bootstrap_effects: tuple[BootstrapEffect, ...]
    gate: ObservabilityGate
    state_sha256: str


def _vector(
    rows: tuple[dict[str, object], ...],
    *,
    domain_order: tuple[str, ...],
    method: str,
    metric: str,
) -> np.ndarray:
    mapping = {
        str(row["outer_domain"]): float(row[metric])
        for row in rows
        if row["method"] == method
    }
    if set(mapping) != set(domain_order):
        raise ValueError("M1 domain metric roster changed")
    return np.asarray([mapping[domain] for domain in domain_order], dtype=np.float64)


def aggregate_observability_metrics(
    metrics: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    bootstrap_seed: int,
    bootstrap_resamples: int,
    minimum_improved_domains: int,
) -> ObservabilityAggregation:
    """Aggregate specimen metrics with equal domain mass and issue M1 GO/NO-GO."""

    required = {"outer_domain", "specimen_id", "method", *_METRICS}
    if (
        type(metrics) is not pl.DataFrame
        or not required <= set(metrics.columns)
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or set(metrics["outer_domain"]) != set(domain_order)
        or not {_PRIMARY, _GLOBAL, _RANDOM} <= set(metrics["method"])
        or minimum_improved_domains != 4
        or bootstrap_resamples <= 0
    ):
        raise ValueError("M1 aggregation authority changed")
    if metrics.unique(subset=["outer_domain", "specimen_id", "method"]).height != metrics.height:
        raise ValueError("M1 specimen metric duplicated")
    for name in _METRICS:
        if not bool(metrics.select(pl.col(name).is_finite().all()).item()):
            raise ValueError("M1 metric is nonfinite")
    grouped = metrics.group_by(["outer_domain", "method"]).agg(
        pl.len().alias("specimen_count"),
        *(pl.col(name).mean().alias(name) for name in _METRICS),
    ).sort(["outer_domain", "method"])
    domain_rows = tuple(grouped.iter_rows(named=True))
    methods = tuple(sorted(set(metrics["method"])))
    model_rows: list[dict[str, object]] = []
    for method in methods:
        selected = grouped.filter(pl.col("method") == method)
        if selected.height != len(domain_order):
            raise ValueError("M1 method domain coverage changed")
        model_rows.append(
            {
                "method": method,
                "domain_count": len(domain_order),
                **{
                    name: float(np.mean(selected[name].to_numpy(), dtype=np.float64))
                    for name in _METRICS
                },
            }
        )
    indices = synchronized_bootstrap_indices(
        seed=bootstrap_seed, resamples=bootstrap_resamples, domains=len(domain_order)
    )
    primary_spearman = _vector(
        domain_rows, domain_order=domain_order, method=_PRIMARY, metric="spearman"
    )
    primary_ndcg = _vector(
        domain_rows, domain_order=domain_order, method=_PRIMARY, metric="ndcg_10"
    )
    global_ndcg = _vector(
        domain_rows, domain_order=domain_order, method=_GLOBAL, metric="ndcg_10"
    )
    primary_regret = _vector(
        domain_rows,
        domain_order=domain_order,
        method=_PRIMARY,
        metric="mean_budgeted_regret",
    )
    global_regret = _vector(
        domain_rows,
        domain_order=domain_order,
        method=_GLOBAL,
        metric="mean_budgeted_regret",
    )
    random_regret = _vector(
        domain_rows,
        domain_order=domain_order,
        method=_RANDOM,
        metric="mean_budgeted_regret",
    )
    effects = (
        paired_domain_bootstrap(
            primary_spearman,
            np.zeros(6, dtype=np.float64),
            indices=indices,
            effect_id="o2_spearman_positive",
        ),
        paired_domain_bootstrap(
            primary_ndcg,
            global_ndcg,
            indices=indices,
            effect_id="o2_minus_global_ndcg10",
        ),
        paired_domain_bootstrap(
            global_regret,
            primary_regret,
            indices=indices,
            effect_id="global_minus_o2_budgeted_regret",
        ),
        paired_domain_bootstrap(
            random_regret,
            primary_regret,
            indices=indices,
            effect_id="random_minus_o2_budgeted_regret",
        ),
    )
    by_id = {value.effect_id: value for value in effects}

    def positive(effect_id: str, *, require_domains: bool) -> bool:
        effect = by_id[effect_id]
        return (
            effect.point_estimate > 0.0
            and effect.lower > 0.0
            and (
                not require_domains
                or effect.improved_domains >= minimum_improved_domains
            )
        )

    passes = (
        positive("o2_spearman_positive", require_domains=False),
        positive("o2_minus_global_ndcg10", require_domains=True),
        positive("global_minus_o2_budgeted_regret", require_domains=True),
        positive("random_minus_o2_budgeted_regret", require_domains=True),
    )
    gate = ObservabilityGate(
        status=("MVD_OBSERVABILITY_GO" if all(passes) else "MVD_OBSERVABILITY_NO_GO"),
        go=all(passes),
        spearman_pass=passes[0],
        ndcg_global_pass=passes[1],
        regret_global_pass=passes[2],
        regret_random_pass=passes[3],
    )
    payload = {
        "bootstrap_effects": [asdict(value) for value in effects],
        "domain_metrics": domain_rows,
        "domain_order": domain_order,
        "gate": asdict(gate),
        "model_metrics": model_rows,
    }
    state = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()
    return ObservabilityAggregation(
        domain_order=domain_order,
        domain_metrics=domain_rows,
        model_metrics=tuple(model_rows),
        bootstrap_effects=effects,
        gate=gate,
        state_sha256=state,
    )


__all__ = [
    "ObservabilityAggregation",
    "ObservabilityGate",
    "aggregate_observability_metrics",
]
