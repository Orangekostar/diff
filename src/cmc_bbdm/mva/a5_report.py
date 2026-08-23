"""Deterministic evidence report for formal MVA A5."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .a5_config import load_a5_config
from .a5_evaluation import AGGREGATED_METHODS


class A5ReportError(ValueError):
    """Raised when A5 report evidence is incomplete or inconsistent."""


LABEL = {
    "center_first": "Center-first",
    "observed_gradient": "Observed gradient",
    "observed_uncertainty": "Observed uncertainty",
    "imitation_policy": "Imitation policy",
    "uniform": "Uniform",
    "random_median": "Random median",
    "global_mechanical_mask": "Global mechanical",
    "mechanical_oracle": "Mechanical oracle",
}


def _budget(value: object) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.3f}%"


def render_a5_report(
    evidence_dir: str | Path,
    *,
    config_path: str | Path,
    project_root: str | Path,
) -> Path:
    """Render the registered A5 gate, metrics, and interpretation boundary."""

    evidence = Path(evidence_dir).resolve(strict=True)
    config = load_a5_config(config_path, project_root=project_root)
    try:
        summary = json.loads((evidence / "summary.json").read_text(encoding="utf-8"))
        budget = pl.read_csv(evidence / "budget_metrics.csv")
        bootstrap = pl.read_csv(evidence / "bootstrap.csv")
        domains = pl.read_csv(evidence / "domain_metrics.csv")
    except (OSError, UnicodeError, json.JSONDecodeError, pl.exceptions.PolarsError) as error:
        raise A5ReportError("A5 report evidence cannot be read") from error
    if (
        summary.get("a5_status") not in config.a5_statuses
        or summary.get("a6_status") not in config.a6_statuses
        or set(budget["method"]) != set(AGGREGATED_METHODS)
        or set(domains["dataset_id"]) != set(config.domain_order)
    ):
        raise A5ReportError("A5 report evidence roster changed")
    gate = summary["gate"]
    lines = [
        "# MVA A5 Oracle-Imitation Policy Report",
        "",
        f"- A5 policy status: `{summary['a5_status']}`",
        f"- A6 authorization: `{summary['a6_status']}`",
        f"- Oracle gap closure: {100.0 * float(gate['gap_closure']):.2f}%",
        "",
        "## Synchronized held-out-domain effects",
        "",
        "| Contrast | Point | 95% interval | Improved domains |",
        "|---|---:|---:|---:|",
    ]
    effect_label = {
        "global_minus_policy_auebc": "Global mechanical - policy",
        "uniform_minus_policy_auebc": "Uniform - policy",
        "policy_minus_oracle_auebc": "Policy - oracle",
    }
    for row in bootstrap.sort("effect_id").iter_rows(named=True):
        lines.append(
            f"| {effect_label[str(row['effect_id'])]} | "
            f"{float(row['point_estimate']):.6f} | "
            f"[{float(row['lower']):.6f}, {float(row['upper']):.6f}] | "
            f"{int(row['improved_domains'])}/6 |"
        )
    lines.extend(
        [
            "",
            "Positive values favor the second method in each contrast.",
            "",
            "## Equal-domain P-B budget metrics",
            "",
            "| Method | AUEBC | B2.5 | B5 | B7.5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    rows = {str(row["method"]): row for row in budget.iter_rows(named=True)}
    for method in AGGREGATED_METHODS:
        row = rows[method]
        lines.append(
            f"| {LABEL[method]} | {float(row['auebc']):.6f} | "
            f"{_budget(row['b_2p5'])} | {_budget(row['b_5'])} | "
            f"{_budget(row['b_7p5'])} |"
        )
    policy_domains = domains.filter(pl.col("method") == "imitation_policy")
    global_domains = domains.filter(pl.col("method") == "global_mechanical_mask")
    effects = {
        str(row["dataset_id"]): float(row["auebc"])
        for row in global_domains.iter_rows(named=True)
    }
    improved = sum(
        effects[str(row["dataset_id"])] - float(row["auebc"]) > 0.0
        for row in policy_domains.iter_rows(named=True)
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                f"The imitation policy improves AUEBC over global mechanical in "
                f"{improved}/6 held-out domains. The preregistered decision above "
                "also requires improvement over uniform and at least 20% point "
                "oracle-gap closure. No B5 comparison changes the gate."
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "This is retrospective normalized-raster acquisition simulation. "
                "The deployable selectors never receive true CAI, unmeasured RGB "
                "values, full-image features, teacher values, or oracle actions. "
                "Mechanical oracle remains a diagnostic upper bound. These results "
                "do not establish physical scan-time or inspection-time savings."
            ),
            "",
        ]
    )
    path = evidence / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


__all__ = ["A5ReportError", "render_a5_report"]
