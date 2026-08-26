"""Source-selected confidence fallback for engineering-safe MAVIS rollout."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from .policy import PolicySelection


class MAVISFallbackError(ValueError):
    """Raised when safe-policy selection accesses invalid or target outcomes."""


_COLUMNS = {
    "domain_id",
    "specimen_id",
    "confidence",
    "mavis_auebc",
    "uniform_auebc",
    "reconstruction_auebc",
}


@dataclass(frozen=True, slots=True)
class SafePolicySelection:
    outer_domain: str
    baseline: str
    threshold: float
    source_domains: tuple[str, ...]
    source_specimen_ids: tuple[str, ...]
    target_outcomes_used: bool
    audit: pl.DataFrame
    state_sha256: str


@dataclass(frozen=True, slots=True)
class SafeAction:
    selection: PolicySelection
    used_fallback: bool
    confidence: float
    threshold: float
    baseline: str


def _domain_metrics(table: pl.DataFrame, value_column: str) -> tuple[float, float]:
    values = (
        table.group_by("domain_id")
        .agg(pl.col(value_column).mean().alias("value"))
        .sort("domain_id")
        .get_column("value")
        .to_numpy()
    )
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise MAVISFallbackError("safe-policy domain metric is invalid")
    return float(np.mean(values, dtype=np.float64)), float(np.max(values))


def select_source_safe_policy(
    source_metrics: pl.DataFrame,
    *,
    outer_domain: str,
    thresholds: tuple[float, ...],
) -> SafePolicySelection:
    if (
        not isinstance(source_metrics, pl.DataFrame)
        or source_metrics.height == 0
        or not _COLUMNS <= set(source_metrics.columns)
        or type(outer_domain) is not str
        or not outer_domain
        or type(thresholds) is not tuple
        or not thresholds
    ):
        raise MAVISFallbackError("safe-policy selection request is invalid")
    values = tuple(float(value) for value in thresholds)
    if (
        tuple(sorted(values)) != values
        or len(set(values)) != len(values)
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values)
        or source_metrics.get_column("specimen_id").n_unique()
        != source_metrics.height
        or outer_domain in source_metrics.get_column("domain_id").unique()
    ):
        raise MAVISFallbackError("target or duplicate outcome reached safe selection")
    numeric = source_metrics.select(
        "confidence",
        "mavis_auebc",
        "uniform_auebc",
        "reconstruction_auebc",
    )
    if (
        numeric.select(pl.any_horizontal(pl.all().is_nan().any())).item()
        or source_metrics.filter(
            (pl.col("confidence") < 0.0) | (pl.col("confidence") > 1.0)
        ).height
    ):
        raise MAVISFallbackError("safe-policy source values are invalid")
    baseline_values = {
        baseline: _domain_metrics(source_metrics, f"{baseline}_auebc")[0]
        for baseline in ("uniform", "reconstruction")
    }
    baseline = min(baseline_values, key=lambda name: (baseline_values[name], name))
    baseline_column = f"{baseline}_auebc"
    audit_rows: list[dict[str, object]] = []
    for threshold in values:
        candidate = source_metrics.with_columns(
            pl.when(pl.col("confidence") >= threshold)
            .then(pl.col("mavis_auebc"))
            .otherwise(pl.col(baseline_column))
            .alias("safe_auebc"),
            (pl.col("confidence") < threshold).alias("used_fallback"),
        )
        aggregate, worst = _domain_metrics(candidate, "safe_auebc")
        domain_comparison = candidate.group_by("domain_id").agg(
            pl.col("safe_auebc").mean().alias("safe"),
            pl.col(baseline_column).mean().alias("baseline"),
        )
        improved = domain_comparison.filter(pl.col("safe") < pl.col("baseline")).height
        audit_rows.append(
            {
                "threshold": threshold,
                "baseline": baseline,
                "domain_balanced_auebc": aggregate,
                "improved_domain_count": improved,
                "worst_domain_auebc": worst,
                "fallback_count": candidate.get_column("used_fallback").sum(),
                "fallback_frequency": candidate.get_column("used_fallback").mean(),
                "target_outcomes_used": False,
            }
        )
    audit = pl.DataFrame(audit_rows).sort("threshold")
    selected = min(
        audit.iter_rows(named=True),
        key=lambda row: (
            float(row["domain_balanced_auebc"]),
            -int(row["improved_domain_count"]),
            float(row["worst_domain_auebc"]),
            -float(row["threshold"]),
        ),
    )
    source_domains = tuple(sorted(source_metrics.get_column("domain_id").unique()))
    specimen_ids = tuple(sorted(source_metrics.get_column("specimen_id").unique()))
    payload = {
        "schema": 1,
        "outer_domain": outer_domain,
        "baseline": baseline,
        "threshold": selected["threshold"],
        "source_domains": source_domains,
        "source_specimen_ids": specimen_ids,
        "audit": audit.to_dicts(),
    }
    return SafePolicySelection(
        outer_domain=outer_domain,
        baseline=baseline,
        threshold=float(selected["threshold"]),
        source_domains=source_domains,
        source_specimen_ids=specimen_ids,
        target_outcomes_used=False,
        audit=audit,
        state_sha256=hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    )


def apply_safe_action(
    mavis: PolicySelection,
    fallback: PolicySelection,
    *,
    confidence: float,
    safe_policy: SafePolicySelection,
) -> SafeAction:
    value = float(confidence)
    if (
        type(mavis) is not PolicySelection
        or type(fallback) is not PolicySelection
        or type(safe_policy) is not SafePolicySelection
        or isinstance(confidence, bool)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise MAVISFallbackError("safe action request is invalid")
    used_fallback = value < safe_policy.threshold
    return SafeAction(
        selection=fallback if used_fallback else mavis,
        used_fallback=used_fallback,
        confidence=value,
        threshold=safe_policy.threshold,
        baseline=safe_policy.baseline,
    )


__all__ = [
    "MAVISFallbackError",
    "SafeAction",
    "SafePolicySelection",
    "apply_safe_action",
    "select_source_safe_policy",
]
