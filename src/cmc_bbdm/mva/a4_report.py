"""Deterministic evidence report for the formal MVA A4 package."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from .a4_config import load_a4_config
from .a4_evaluation import AGGREGATED_METHODS


class A4ReportError(ValueError):
    """Raised when the A4 report inputs are incomplete or inconsistent."""


EFFECTS = (
    "uniform_minus_global_mechanical_auebc",
    "global_reconstruction_minus_global_mechanical_auebc",
    "global_appearance_minus_global_mechanical_auebc",
    "global_mechanical_minus_mechanical_oracle_auebc",
)
EFFECT_LABEL = {
    "uniform_minus_global_mechanical_auebc": "Uniform - global mechanical",
    "global_reconstruction_minus_global_mechanical_auebc": (
        "Global reconstruction - global mechanical"
    ),
    "global_appearance_minus_global_mechanical_auebc": (
        "Global appearance - global mechanical"
    ),
    "global_mechanical_minus_mechanical_oracle_auebc": (
        "Global mechanical - mechanical oracle"
    ),
}
METHOD_LABEL = {
    "uniform": "Uniform",
    "global_appearance_mask": "Global appearance",
    "global_reconstruction_mask": "Global reconstruction",
    "global_mechanical_mask": "Global mechanical",
    "mechanical_oracle": "Mechanical oracle",
    "random_median": "Random median",
}
_BOOTSTRAP_COLUMNS = {
    "effect_id",
    "point_estimate",
    "lower",
    "upper",
    "improved_domains",
    "domain_effects",
    "seed",
    "resamples",
    "indices_sha256",
}
_BUDGET_COLUMNS = {
    "method",
    "auebc",
    "b_2p5",
    "b_5",
    "b_7p5",
    "saving_vs_uniform_b5",
}
_STABILITY_COLUMNS = {
    "outer_domain",
    "method",
    "removed_domain",
    "top1_agreement",
    "top10_overlap",
    "spearman",
    "rbo_p0_9",
}


def _read_csv(path: Path, label: str) -> pl.DataFrame:
    try:
        return pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise A4ReportError(f"{label} cannot be read") from error


def _finite(table: pl.DataFrame, columns: tuple[str, ...], label: str) -> None:
    try:
        valid = all(
            bool(table.select(pl.col(column).is_finite().all()).item())
            for column in columns
        )
    except pl.exceptions.PolarsError as error:
        raise A4ReportError(f"{label} numeric schema changed") from error
    if not valid:
        raise A4ReportError(f"{label} contains nonfinite values")


def _load_summary(path: Path, a4_statuses: tuple[str, ...], a5_statuses: tuple[str, ...]):
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        gate = summary["gate"]
        relative_gap = float(gate["relative_adaptive_gap"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise A4ReportError("summary cannot be read") from error
    if (
        summary.get("global_mask_status") not in a4_statuses
        or summary.get("a5_status") not in a5_statuses
        or not math.isfinite(relative_gap)
        or relative_gap < 0.0
    ):
        raise A4ReportError("summary status changed")
    return summary, relative_gap


def _validate_bootstrap(table: pl.DataFrame, *, seed: int, resamples: int):
    if (
        set(table.columns) != _BOOTSTRAP_COLUMNS
        or table.height != len(EFFECTS)
        or set(table["effect_id"]) != set(EFFECTS)
        or table.unique(subset=["effect_id"]).height != table.height
        or set(table["seed"]) != {seed}
        or set(table["resamples"]) != {resamples}
        or table["indices_sha256"].n_unique() != 1
    ):
        raise A4ReportError("bootstrap roster changed")
    _finite(
        table,
        ("point_estimate", "lower", "upper", "improved_domains"),
        "bootstrap",
    )
    if table.filter(
        (pl.col("lower") > pl.col("upper"))
        | (pl.col("improved_domains") < 0)
        | (pl.col("improved_domains") > 6)
    ).height:
        raise A4ReportError("bootstrap values changed")
    return table.sort("effect_id")


def _validate_budget(table: pl.DataFrame) -> pl.DataFrame:
    if (
        set(table.columns) != _BUDGET_COLUMNS
        or table.height != len(AGGREGATED_METHODS)
        or set(table["method"]) != set(AGGREGATED_METHODS)
        or table.unique(subset=["method"]).height != table.height
    ):
        raise A4ReportError("budget roster changed")
    _finite(table, ("auebc",), "budget metrics")
    if table.filter(pl.col("auebc") < 0.0).height:
        raise A4ReportError("budget values changed")
    for column in ("b_2p5", "b_5", "b_7p5", "saving_vs_uniform_b5"):
        values = table[column].drop_nulls()
        if values.len() and not bool(values.is_finite().all()):
            raise A4ReportError("budget values changed")
    return table.sort("method")


def _validate_stability(
    table: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    methods: tuple[str, ...],
) -> pl.DataFrame:
    expected_rows = len(domain_order) * len(methods) * (len(domain_order) - 1)
    if (
        set(table.columns) != _STABILITY_COLUMNS
        or table.height != expected_rows
        or set(table["outer_domain"]) != set(domain_order)
        or set(table["removed_domain"]) != set(domain_order)
        or set(table["method"]) != set(methods)
        or table.unique(subset=["outer_domain", "method", "removed_domain"]).height
        != table.height
        or table.filter(pl.col("outer_domain") == pl.col("removed_domain")).height
    ):
        raise A4ReportError("ranking stability roster changed")
    _finite(table, ("top10_overlap", "spearman", "rbo_p0_9"), "ranking stability")
    if table.filter(
        (pl.col("top10_overlap") < 0.0)
        | (pl.col("top10_overlap") > 1.0)
        | (pl.col("spearman") < -1.0)
        | (pl.col("spearman") > 1.0)
        | (pl.col("rbo_p0_9") < 0.0)
        | (pl.col("rbo_p0_9") > 1.0)
    ).height:
        raise A4ReportError("ranking stability values changed")
    return table


def _budget(value: object) -> str:
    if value is None:
        return "not reached"
    return f"{100.0 * float(value):.3f}%"


def render_a4_report(
    evidence_dir: str | Path,
    *,
    config_path: str | Path,
    project_root: str | Path,
) -> Path:
    """Write REPORT.md solely from registered aggregate evidence tables."""

    root = Path(project_root).resolve(strict=True)
    evidence = Path(evidence_dir).resolve(strict=True)
    config = load_a4_config(config_path, project_root=root)
    summary, relative_gap = _load_summary(
        evidence / "summary.json", config.a4_statuses, config.a5_statuses
    )
    bootstrap = _validate_bootstrap(
        _read_csv(evidence / "bootstrap.csv", "bootstrap"),
        seed=config.bootstrap_seed,
        resamples=config.bootstrap_resamples,
    )
    budget = _validate_budget(
        _read_csv(evidence / "budget_metrics.csv", "budget metrics")
    )
    stability = _validate_stability(
        _read_csv(evidence / "ranking_stability.csv", "ranking stability"),
        domain_order=config.domain_order,
        methods=config.methods,
    )

    lines = [
        "# MVA A4 Global Task-Aware Static Acquisition",
        "",
        "## Scope",
        "",
        (
            "This retrospective six-domain evaluation compares three fixed "
            "source-only acquisition rankings with the registered A2 controls. "
            "Each outer-domain ranking excludes all target-domain images, CAI "
            "targets, metadata, and fitted target-domain predictors."
        ),
        "",
        "## Registered decisions",
        "",
        f"- Global-mask decision: `{summary['global_mask_status']}`",
        f"- A5 authorization decision: `{summary['a5_status']}`",
        f"- Relative global-to-oracle AUEBC gap: {100.0 * relative_gap:.3f}%",
        "",
        "The two decisions are independent and use the frozen preregistered gates.",
        "",
        "## Synchronized domain-bootstrap effects",
        "",
        "| Contrast | Point | 95% interval | Improved domains |",
        "|---|---:|---:|---:|",
    ]
    rows_by_effect = {
        str(row["effect_id"]): row for row in bootstrap.iter_rows(named=True)
    }
    for effect in EFFECTS:
        row = rows_by_effect[effect]
        lines.append(
            f"| {EFFECT_LABEL[effect]} | {float(row['point_estimate']):.6f} | "
            f"[{float(row['lower']):.6f}, {float(row['upper']):.6f}] | "
            f"{int(row['improved_domains'])}/6 |"
        )

    lines.extend(
        [
            "",
            "Positive values favor the second method named in each contrast.",
            "",
            "## Equal-domain budget metrics",
            "",
            "| Method | AUEBC | B2.5 | B5 | B7.5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    rows_by_method = {
        str(row["method"]): row for row in budget.iter_rows(named=True)
    }
    for method in AGGREGATED_METHODS:
        row = rows_by_method[method]
        lines.append(
            f"| {METHOD_LABEL[method]} | {float(row['auebc']):.6f} | "
            f"{_budget(row['b_2p5'])} | {_budget(row['b_5'])} | "
            f"{_budget(row['b_7p5'])} |"
        )

    lines.extend(
        [
            "",
            "## Ranking stability diagnostics",
            "",
            "| Source objective | Mean top-10 overlap | Minimum Spearman | Mean RBO |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in config.methods:
        selected = stability.filter(pl.col("method") == method)
        top10 = float(
            np.mean(np.sort(selected["top10_overlap"].to_numpy()), dtype=np.float64)
        )
        spearman = float(selected["spearman"].min())
        rbo = float(
            np.mean(np.sort(selected["rbo_p0_9"].to_numpy()), dtype=np.float64)
        )
        lines.append(
            f"| {METHOD_LABEL[method]} | {top10:.4f} | {spearman:.4f} | {rbo:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The findings describe retrospective simulation under the frozen "
                "interpolation, encoder, CAI estimators, and domain roster. RGB "
                "reconstruction fidelity is reported as a mechanism diagnostic, "
                "not as evidence of mechanical validity. No adaptive policy is "
                "trained or evaluated in A4."
            ),
            "",
        ]
    )
    report = evidence / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


__all__ = ["A4ReportError", "render_a4_report"]
