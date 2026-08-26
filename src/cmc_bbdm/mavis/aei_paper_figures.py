"""Deterministic, evidence-bound figures for the AEI information-hierarchy paper."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from cmc_bbdm.mavis.aei_paper_evidence import PaperMetric, build_canonical_metrics

matplotlib.use("Agg", force=True)

_COLUMNS = (
    "panel",
    "series",
    "metric",
    "value",
    "ci95_lower",
    "ci95_upper",
    "status",
    "source_claim_id",
    "source_artifact",
    "source_hash",
)
_USEFUL = "#E69F00"
_OBSERVABLE = "#0072B2"
_ACTIONABLE = "#009E73"
_ADVERSE = "#D55E00"
_REFERENCE = "#666666"
_UNCERTAINTY = "#A6A6A6"
_TEXT = "#222222"
_GRID = "#D9D9D9"
_FIXED_TIME = datetime(2026, 8, 26, tzinfo=UTC)


@dataclass(frozen=True)
class FigureArtifact:
    """Paths belonging to one rendered paper figure."""

    svg: Path
    pdf: Path
    png: Path
    source_csv: Path
    caption: Path

    @property
    def outputs(self) -> tuple[Path, Path, Path]:
        return (self.svg, self.pdf, self.png)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: float | None) -> str:
    return "" if value is None else format(value, ".17g")


def _metric_row(
    metric: PaperMetric,
    *,
    panel: str,
    series: str,
    value: float | None = None,
    ci95_lower: float | None = None,
    ci95_upper: float | None = None,
    status: str | None = None,
) -> dict[str, str]:
    return {
        "panel": panel,
        "series": series,
        "metric": metric.metric,
        "value": _number(metric.estimate if value is None else value),
        "ci95_lower": _number(ci95_lower),
        "ci95_upper": _number(ci95_upper),
        "status": metric.status if status is None else status,
        "source_claim_id": metric.claim_id,
        "source_artifact": metric.source_artifact,
        "source_hash": metric.source_hash,
    }


def _effect_row(metric: PaperMetric, *, panel: str, series: str) -> dict[str, str]:
    return _metric_row(
        metric,
        panel=panel,
        series=series,
        ci95_lower=metric.ci95_lower,
        ci95_upper=metric.ci95_upper,
    )


def _derived_row(
    metric: PaperMetric,
    *,
    panel: str,
    series: str,
    value: float,
    metric_name: str,
    ci95_lower: float | None = None,
    ci95_upper: float | None = None,
    status: str | None = None,
) -> dict[str, str]:
    row = _metric_row(
        metric,
        panel=panel,
        series=series,
        value=value,
        ci95_lower=ci95_lower,
        ci95_upper=ci95_upper,
        status=status,
    )
    row["metric"] = metric_name
    return row


def _load_json(root: Path, metric: PaperMetric) -> dict[str, Any]:
    path = root / metric.source_artifact
    if _sha256(path) != metric.source_hash:
        raise ValueError(f"figure source hash changed: {metric.source_artifact}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError(
            f"figure source must be a JSON object: {metric.source_artifact}"
        )
    return payload


def _load_csv(root: Path, metric: PaperMetric) -> list[dict[str, str]]:
    path = root / metric.source_artifact
    if _sha256(path) != metric.source_hash:
        raise ValueError(f"figure source hash changed: {metric.source_artifact}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _figure1_rows(root: Path) -> list[dict[str, str]]:
    source = "artifacts/aei_information_hierarchy/AEI_SCOPE_AND_STRUCTURE_LEDGER.md"
    source_path = root / source
    source_hash = _sha256(source_path)
    definitions = (
        (
            "hierarchy",
            "Useful",
            "Does an information source improve the engineering task?",
            "H1_USEFUL",
        ),
        (
            "hierarchy",
            "Observable",
            "Can legal inspection state identify future measurement value?",
            "H2_OBSERVABLE",
        ),
        (
            "hierarchy",
            "Actionable",
            "Can the estimate improve a cost-constrained sensing decision?",
            "H3_ACTIONABLE",
        ),
        (
            "evidence boundary",
            "Retrospective teacher or oracle",
            "Diagnostic evidence; unavailable to a deployable policy",
            "H4_RETROSPECTIVE",
        ),
        (
            "evidence boundary",
            "Legal deployable state",
            "Metadata, acquired positions, measured content, and exact cost",
            "H5_LEGAL_STATE",
        ),
        (
            "evidence boundary",
            "Policy and planner",
            "Selection under the frozen information and cost boundary",
            "H6_POLICY",
        ),
    )
    return [
        {
            "panel": panel,
            "series": series,
            "metric": description,
            "value": "",
            "ci95_lower": "",
            "ci95_upper": "",
            "status": "CONCEPTUAL",
            "source_claim_id": claim_id,
            "source_artifact": source,
            "source_hash": source_hash,
        }
        for panel, series, description, claim_id in definitions
    ]


def _figure2_rows(metrics: dict[str, PaperMetric]) -> list[dict[str, str]]:
    matched = metrics["U1_MATCHED_FIELD"]
    retention = metrics["U2_SPARSE_RETENTION"]
    sparse_gain = metrics["U2_SPARSE_GAIN"]
    sparse_gap = metrics["U2_SPARSE_FULL_GAP"]
    oracle_cai = metrics["U4_ORACLE_CAI_SPECIFICITY"]
    oracle_image = metrics["U4_ORACLE_IMAGE_SPECIFICITY"]
    learned = metrics["U4_LEARNED_SPECIFICITY_BOUNDARY"]
    return [
        _derived_row(
            matched,
            panel="a",
            series="Matched scalar",
            value=float(matched.reference_value),
            metric_name="equal_domain_cai_ratio_mae",
        ),
        _derived_row(
            matched,
            panel="a",
            series="Matched spatial field",
            value=float(matched.candidate_value),
            metric_name="equal_domain_cai_ratio_mae",
        ),
        _effect_row(matched, panel="a", series="Registered MAE reduction"),
        _derived_row(
            sparse_gain,
            panel="b",
            series="Surface reference",
            value=float(sparse_gain.reference_value),
            metric_name="equal_domain_cai_ratio_mae",
        ),
        _derived_row(
            retention,
            panel="b",
            series="Full spatial field",
            value=float(retention.reference_value),
            metric_name="equal_domain_cai_ratio_mae",
        ),
        _derived_row(
            retention,
            panel="b",
            series="Sparse spatial field",
            value=float(retention.candidate_value),
            metric_name="equal_domain_cai_ratio_mae",
        ),
        _effect_row(retention, panel="b", series="Registered gain retained"),
        _effect_row(sparse_gain, panel="b", series="Surface-to-sparse reduction"),
        _effect_row(sparse_gap, panel="b", series="Sparse-to-full gap"),
        _effect_row(oracle_cai, panel="c", series="CAI-specific oracle contrast"),
        _effect_row(oracle_image, panel="c", series="Image-specific oracle contrast"),
        _effect_row(learned, panel="c", series="Learned global-mask separation"),
    ]


def _figure3_rows(root: Path, metrics: dict[str, PaperMetric]) -> list[dict[str, str]]:
    rows = [
        _effect_row(
            metrics["O1_STATIC_SPEARMAN"], panel="a", series="Static value rank"
        ),
        _effect_row(metrics["O1_STATIC_SET_REGRET"], panel="a", series="Static scorer"),
        _effect_row(
            metrics["O1_GLOBAL_SET_REGRET"], panel="a", series="Global reference"
        ),
        _effect_row(
            metrics["O1_RANDOM_SET_REGRET"], panel="a", series="Random reference"
        ),
    ]
    teacher_metric = metrics["O2_TEACHER_TURNOVER"]
    p9 = _load_json(root, teacher_metric)
    claim_by_key = {
        "best_action_turnover": metrics["O2_TEACHER_TURNOVER"],
        "rank_spearman": metrics["O2_TEACHER_RANK"],
        "top_k_jaccard": metrics["O2_TEACHER_TOPK"],
    }
    for checkpoint in p9["teacher_by_checkpoint"]:
        cost = float(checkpoint["current_checkpoint"])
        for key, claim in claim_by_key.items():
            rows.append(
                _derived_row(
                    claim,
                    panel="b",
                    series=f"{key}@{format(cost, '.6g')}",
                    value=float(checkpoint[key]),
                    metric_name=key,
                )
            )

    positions = metrics["O3_REAL_MINUS_POSITIONS"]
    reconstruction = metrics["O3_REAL_MINUS_RECONSTRUCTION"]
    endpoint = next(
        row
        for row in _load_csv(root, positions)
        if row["nominal_checkpoint"] == "0.25"
        and row["control_mode"] == "positions_only"
    )
    endpoint_reconstruction = next(
        row
        for row in _load_csv(root, reconstruction)
        if row["nominal_checkpoint"] == "0.25"
        and row["control_mode"] == "reconstruction"
    )
    rows.extend(
        (
            _derived_row(
                positions,
                panel="c",
                series="Measured content",
                value=float(endpoint["real_equal_domain_mae"]),
                metric_name="endpoint_equal_domain_cai_mae",
            ),
            _derived_row(
                positions,
                panel="c",
                series="Matched positions",
                value=float(endpoint["control_equal_domain_mae"]),
                metric_name="endpoint_equal_domain_cai_mae",
            ),
            _derived_row(
                reconstruction,
                panel="c",
                series="Reconstruction control",
                value=float(endpoint_reconstruction["control_equal_domain_mae"]),
                metric_name="endpoint_equal_domain_cai_mae",
            ),
            _effect_row(positions, panel="c", series="Measured minus positions"),
            _effect_row(
                reconstruction, panel="c", series="Measured minus reconstruction"
            ),
            _effect_row(
                metrics["O4_DYNAMIC_MINUS_STATIC"],
                panel="c",
                series="Conditional minus static regret",
            ),
            _effect_row(
                metrics["O4_DYNAMIC_MINUS_SHUFFLED"],
                panel="c",
                series="Conditional minus shuffled regret",
            ),
        )
    )
    return rows


def _figure4_rows(root: Path, metrics: dict[str, PaperMetric]) -> list[dict[str, str]]:
    rows = [
        _effect_row(
            metrics["A1_VALUATION_SUBSTITUTION"], panel="a", series="Valuation"
        ),
        _effect_row(
            metrics["A1_LEARNED_PLANNING_SUBSTITUTION"],
            panel="a",
            series="Bounded learned planning",
        ),
        _effect_row(
            metrics["A1_TRUE_VALUE_PLANNING_SUBSTITUTION"],
            panel="a",
            series="True-value stronger planning",
        ),
    ]
    planning = metrics["A2_GREEDY_PLANNING_REGRET"]
    p13 = _load_json(root, planning)
    planning_specs = (
        ("current_greedy", "Current greedy", planning),
        ("beam_width_2", "Beam width 2", planning),
        ("beam_width_4", "Beam width 4", metrics["A2_BEAM4_PLANNING_REGRET"]),
        ("two_step_lookahead", "Two-step lookahead", planning),
        (
            "retrospective_joint_near_oracle_reachable_pool",
            "Bounded near-oracle",
            planning,
        ),
    )
    for key, label, claim in planning_specs:
        interval = p13["planning_regret_intervals"].get(key, (None, None))
        rows.append(
            _derived_row(
                claim,
                panel="b",
                series=label,
                value=float(p13["aggregate"][key]["planning_regret"]),
                metric_name="bounded_set_planning_regret",
                ci95_lower=None if interval[0] is None else float(interval[0]),
                ci95_upper=None if interval[1] is None else float(interval[1]),
                status="RETROSPECTIVE_BOUND",
            )
        )
    rows.extend(
        (
            _effect_row(
                metrics["A3_FEEDBACK_BENEFIT"], panel="c", series="Feedback benefit"
            ),
            _effect_row(
                metrics["A4_BASELINE_MINUS_MAVIS"],
                panel="c",
                series="Baseline minus frozen policy",
            ),
        )
    )
    return rows


def _write_rows(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_figure_sources(root: Path, output_root: Path) -> dict[str, Path]:
    """Generate the four source CSVs from hash-bound paper evidence."""
    root = root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    canonical = {metric.claim_id: metric for metric in build_canonical_metrics(root)}
    rows_by_figure = {
        "figure1": _figure1_rows(root),
        "figure2": _figure2_rows(canonical),
        "figure3": _figure3_rows(root, canonical),
        "figure4": _figure4_rows(root, canonical),
    }
    names = {
        "figure1": "figure1_hierarchy.csv",
        "figure2": "figure2_usefulness.csv",
        "figure3": "figure3_observability.csv",
        "figure4": "figure4_actionability.csv",
    }
    paths: dict[str, Path] = {}
    for figure_id, rows in rows_by_figure.items():
        path = output_root / names[figure_id]
        _write_rows(path, rows)
        paths[figure_id] = path
    return paths


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _find(rows: list[dict[str, str]], series: str) -> dict[str, str]:
    selected = [row for row in rows if row["series"] == series]
    if len(selected) != 1:
        raise ValueError(f"expected one figure row for {series}, found {len(selected)}")
    return selected[0]


def _panel_title(ax: Axes, letter: str, title: str) -> None:
    ax.set_title(
        f"({letter})  {title}", loc="left", fontsize=8.5, fontweight="bold", pad=7
    )


def _clean_axis(ax: Axes, *, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=_GRID, linewidth=0.55, zorder=0)
    ax.tick_params(labelsize=7, colors=_TEXT, length=2.5)


def _render_figure1(rows: list[dict[str, str]]) -> Figure:
    del rows
    fig, ax = plt.subplots(figsize=(7.2, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.95,
        "Task-relevant information hierarchy",
        fontsize=10,
        fontweight="bold",
        color=_TEXT,
        va="top",
    )
    stages = (
        (
            0.02,
            _USEFUL,
            "USEFUL",
            "Improves the\nengineering task",
            "Retrospective evidence",
        ),
        (
            0.35,
            _OBSERVABLE,
            "OBSERVABLE",
            "Identifiable from\nlegal inspection state",
            "Conditional evidence",
        ),
        (
            0.68,
            _ACTIONABLE,
            "ACTIONABLE",
            "Improves a bounded\nsensing decision",
            "Deployable endpoint",
        ),
    )
    for x, color, heading, question, evidence in stages:
        ax.add_patch(
            Rectangle((x, 0.52), 0.27, 0.28, facecolor="white", edgecolor=color, lw=2)
        )
        ax.add_patch(Rectangle((x, 0.74), 0.27, 0.06, facecolor=color, edgecolor=color))
        ax.text(
            x + 0.135,
            0.77,
            heading,
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            x + 0.135,
            0.645,
            question,
            ha="center",
            va="center",
            color=_TEXT,
            fontsize=8,
            linespacing=1.35,
        )
        ax.text(
            x + 0.135,
            0.55,
            evidence,
            ha="center",
            va="center",
            color=_REFERENCE,
            fontsize=6.8,
        )
    for start in (0.29, 0.62):
        ax.annotate(
            "",
            xy=(start + 0.055, 0.66),
            xytext=(start, 0.66),
            arrowprops={"arrowstyle": "-|>", "color": _REFERENCE, "lw": 1.4},
        )
        ax.text(
            start + 0.0275,
            0.825,
            "must be tested",
            ha="center",
            fontsize=5.8,
            color=_REFERENCE,
        )
    lanes = (
        (
            0.02,
            0.26,
            "Retrospective teacher\n/ oracle",
            "future outcome and unmeasured\nfield available",
            _USEFUL,
        ),
        (
            0.35,
            0.26,
            "Legal deployable state",
            "metadata + acquired positions/content\n+ exact cost",
            _OBSERVABLE,
        ),
        (
            0.68,
            0.26,
            "Frozen policy / planner",
            "selection within information and\nbudget constraints",
            _ACTIONABLE,
        ),
    )
    for x, y, title, detail, color in lanes:
        ax.add_patch(
            Rectangle(
                (x, y),
                0.27,
                0.14,
                facecolor="#F7F7F7",
                edgecolor=color,
                lw=1.1,
                linestyle="--",
            )
        )
        ax.text(
            x + 0.135,
            y + 0.105,
            title,
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=_TEXT,
            linespacing=0.9,
        )
        ax.text(
            x + 0.135,
            y + 0.038,
            detail,
            ha="center",
            va="center",
            fontsize=5.3,
            color=_REFERENCE,
            linespacing=1.1,
        )
    ax.text(
        0.02, 0.12, "Evidence boundary", fontsize=7.4, fontweight="bold", color=_TEXT
    )
    ax.plot([0.23, 0.98], [0.135, 0.135], color=_REFERENCE, lw=0.8)
    ax.text(
        0.5,
        0.055,
        "Passing an earlier question does not imply observability or actionability.",
        ha="center",
        va="center",
        fontsize=7.4,
        color=_ADVERSE,
        fontweight="bold",
    )
    return fig


def _render_figure2(rows: list[dict[str, str]]) -> Figure:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.05),
        gridspec_kw={"wspace": 0.40, "width_ratios": [1.0, 1.0, 1.12]},
    )
    ax = axes[0]
    labels = ["Matched scalar", "Matched spatial field"]
    values = [_float(_find(rows, label), "value") for label in labels]
    ax.barh(
        [1, 0],
        values,
        color=[_REFERENCE, _USEFUL],
        edgecolor=_TEXT,
        linewidth=0.5,
        height=0.56,
        hatch=["//", ""],
    )
    for y, value in zip((1, 0), values, strict=True):
        ax.text(
            value - 0.004,
            y,
            f"{value:.3f}",
            ha="right",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
    effect = _find(rows, "Registered MAE reduction")
    ax.text(
        0.004,
        -0.72,
        f"Reduction {float(effect['value']):.3f}\n95% CI [{float(effect['ci95_lower']):.3f}, {float(effect['ci95_upper']):.3f}]",
        fontsize=6.5,
        color=_TEXT,
        va="center",
    )
    ax.set_yticks([1, 0], labels, fontsize=7)
    ax.set_xlim(0, 0.21)
    ax.set_ylim(-1.0, 1.55)
    ax.set_xlabel("Equal-domain CAI-ratio MAE", fontsize=7)
    _panel_title(ax, "a", "Matched morphology")
    _clean_axis(ax, grid_axis="x")

    ax = axes[1]
    labels = ["Surface reference", "Full spatial field", "Sparse spatial field"]
    values = [_float(_find(rows, label), "value") for label in labels]
    colors = [_REFERENCE, _USEFUL, _USEFUL]
    bars = ax.bar(
        [0, 1, 2], values, color=colors, edgecolor=_TEXT, linewidth=0.5, width=0.65
    )
    bars[2].set_hatch("..")
    for x, value in enumerate(values):
        ax.text(x, value + 0.003, f"{value:.3f}", ha="center", fontsize=6.6)
    retention = float(_find(rows, "Registered gain retained")["value"])
    gap = _find(rows, "Sparse-to-full gap")
    ax.text(
        1.5,
        0.198,
        f"{retention * 100:.1f}% gain retained",
        ha="center",
        fontsize=6.6,
        color=_USEFUL,
        fontweight="bold",
    )
    ax.text(
        1.5,
        0.153,
        f"Residual gap {float(gap['value']):.4f}\nCI [{float(gap['ci95_lower']):.4f}, {float(gap['ci95_upper']):.4f}]",
        ha="center",
        fontsize=5.8,
        color=_ADVERSE,
    )
    ax.set_xticks([0, 1, 2], ["Surface", "Full", "Sparse"], fontsize=7)
    ax.set_ylim(0.10, 0.205)
    ax.set_ylabel("Equal-domain CAI-ratio MAE", fontsize=7)
    _panel_title(ax, "b", "Sparse retention")
    _clean_axis(ax, grid_axis="y")

    ax = axes[2]
    ax.axis("off")
    _panel_title(ax, "c", "Task specificity")
    cai = _find(rows, "CAI-specific oracle contrast")
    image = _find(rows, "Image-specific oracle contrast")
    learned = _find(rows, "Learned global-mask separation")
    blocks = (
        (
            0.86,
            _USEFUL,
            "CAI objective",
            f"Mechanics oracle benefit\n{float(cai['value']):.4f}  [{float(cai['ci95_lower']):.4f}, {float(cai['ci95_upper']):.4f}]",
        ),
        (
            0.55,
            _OBSERVABLE,
            "Image objective",
            f"Reconstruction oracle benefit\n{float(image['value']):.6f}  [{float(image['ci95_lower']):.6f}, {float(image['ci95_upper']):.6f}]",
        ),
        (
            0.29,
            _ADVERSE,
            "Learned global masks",
            "Oracle separation not reproduced"
            if float(learned["value"]) == 0
            else "Separation reproduced",
        ),
    )
    for y, color, title, detail in blocks:
        ax.add_patch(
            Rectangle(
                (0.02, y - 0.19),
                0.96,
                0.22,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor=color,
                lw=1.3,
                hatch="//" if color == _ADVERSE else None,
            )
        )
        ax.text(
            0.06,
            y - 0.04,
            title,
            transform=ax.transAxes,
            fontsize=7.2,
            fontweight="bold",
            color=color,
            va="center",
        )
        ax.text(
            0.06,
            y - 0.13,
            detail,
            transform=ax.transAxes,
            fontsize=5.6,
            color=_TEXT,
            va="center",
        )
    ax.text(
        0.04,
        0.0,
        "Oracle rows are retrospective, not deployable.",
        transform=ax.transAxes,
        fontsize=6.3,
        color=_REFERENCE,
    )
    return fig


def _render_figure3(rows: list[dict[str, str]]) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), gridspec_kw={"wspace": 0.48})
    ax = axes[0]
    regrets = [
        (label, float(_find(rows, label)["value"]))
        for label in ("Static scorer", "Global reference", "Random reference")
    ]
    ax.barh(
        [2, 1, 0],
        [value for _, value in regrets],
        color=[_OBSERVABLE, _REFERENCE, _REFERENCE],
        edgecolor=_TEXT,
        linewidth=0.4,
        height=0.55,
        hatch=["", "//", ".."],
    )
    ax.set_yticks(
        [2, 1, 0],
        [label.replace(" reference", "") for label, _ in regrets],
        fontsize=6.2,
    )
    ax.set_xlim(0, 0.09)
    ax.set_ylim(-0.5, 5.0)
    ax.set_xlabel("Exact-budget set regret", fontsize=7)
    _panel_title(ax, "a", "Static observability")
    _clean_axis(ax, grid_axis="x")

    rank = _find(rows, "Static value rank")
    estimate = float(rank["value"])
    low = float(rank["ci95_lower"])
    high = float(rank["ci95_upper"])
    inset = ax.inset_axes([0.04, 0.67, 0.92, 0.20])
    inset.errorbar(
        estimate,
        0,
        xerr=[[estimate - low], [high - estimate]],
        fmt="s",
        color=_OBSERVABLE,
        ecolor=_UNCERTAINTY,
        capsize=3,
        ms=4,
    )
    inset.axvline(0, color=_REFERENCE, lw=0.7)
    inset.set_xlim(-0.12, 0.12)
    inset.set_ylim(-0.5, 0.5)
    inset.set_yticks([0], ["Strict-OOF rank"], fontsize=5.5)
    inset.set_xlabel("Spearman correlation", fontsize=5.8, labelpad=1)
    inset.tick_params(labelsize=5.5, length=2)
    inset.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    metric_specs = (
        ("best_action_turnover", "Turnover", _OBSERVABLE, "s", "--"),
        ("rank_spearman", "Rank agreement", _REFERENCE, "o", "-"),
        ("top_k_jaccard", "Top-5 overlap", _USEFUL, "^", ":"),
    )
    for metric_name, label, color, marker, linestyle in metric_specs:
        selected = [
            row for row in rows if row["panel"] == "b" and row["metric"] == metric_name
        ]
        x = np.asarray([float(row["series"].split("@")[1]) * 100 for row in selected])
        y = np.asarray([float(row["value"]) for row in selected])
        ax.plot(
            x,
            y,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.2,
            markersize=4,
        )
    ax.set_xlabel("Acquired normalized cost (%)", fontsize=7)
    ax.set_ylabel("Teacher-state diagnostic", fontsize=7)
    ax.set_ylim(0.25, 0.76)
    ax.legend(frameon=False, fontsize=5.8, loc="best", handlelength=2.3)
    _panel_title(ax, "b", "Conditional value evolves")
    _clean_axis(ax, grid_axis="y")

    ax = axes[2]
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    _panel_title(ax, "c", "Content controls")
    labels = ["Measured content", "Matched positions", "Reconstruction control"]
    values = [float(_find(rows, label)["value"]) for label in labels]
    mae_ax = ax.inset_axes([0.02, 0.46, 0.96, 0.48])
    bars = mae_ax.bar(
        [0, 1, 2],
        values,
        color=[_OBSERVABLE, _REFERENCE, _REFERENCE],
        edgecolor=_TEXT,
        linewidth=0.5,
        width=0.66,
    )
    bars[1].set_hatch("//")
    bars[2].set_hatch("..")
    mae_ax.set_xticks([0, 1, 2], ["Real", "Position", "Recon."], fontsize=5.8)
    mae_ax.set_ylim(0.08, 0.14)
    mae_ax.set_ylabel("Endpoint CAI MAE", fontsize=6)
    _clean_axis(mae_ax, grid_axis="y")
    inset = ax.inset_axes([0.12, 0.05, 0.84, 0.25])
    controls = [
        ("Conditional minus static regret", _OBSERVABLE, "s"),
        ("Conditional minus shuffled regret", _ADVERSE, "x"),
    ]
    for y, (label, color, marker) in enumerate(controls):
        row = _find(rows, label)
        value = float(row["value"]) * 1000
        low = float(row["ci95_lower"]) * 1000
        high = float(row["ci95_upper"]) * 1000
        inset.errorbar(
            value,
            y,
            xerr=[[value - low], [high - value]],
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2,
            ms=4,
        )
    inset.axvline(0, color=_REFERENCE, lw=0.7)
    inset.set_yticks([0, 1], ["vs static", "vs shuffled"], fontsize=5.2)
    inset.set_xlabel("Regret contrast (×10³)", fontsize=5.5)
    inset.tick_params(labelsize=5.2, length=1.5)
    inset.spines[["top", "right"]].set_visible(False)
    return fig


def _forest(
    ax: Axes,
    rows: list[dict[str, str]],
    labels: list[str],
    *,
    scale: float,
    colors: list[str],
    markers: list[str],
    display_labels: list[str] | None = None,
) -> None:
    positions = np.arange(len(labels))[::-1]
    for y, label, color, marker in zip(positions, labels, colors, markers, strict=True):
        row = _find(rows, label)
        value = float(row["value"]) * scale
        if row["ci95_lower"] and row["ci95_upper"]:
            low = float(row["ci95_lower"]) * scale
            high = float(row["ci95_upper"]) * scale
            xerr: Any = [[value - low], [high - value]]
        else:
            xerr = None
        ax.errorbar(
            value,
            y,
            xerr=xerr,
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2.5,
            ms=5,
            lw=1,
        )
    ax.set_yticks(
        positions,
        labels if display_labels is None else display_labels,
        fontsize=6.2,
    )
    ax.axvline(0, color=_REFERENCE, lw=0.8)


def _render_figure4(rows: list[dict[str, str]]) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), gridspec_kw={"wspace": 0.55})
    ax = axes[0]
    labels = ["Valuation", "Bounded learned planning", "True-value stronger planning"]
    _forest(
        ax,
        rows,
        labels,
        scale=10000,
        colors=[_USEFUL, _ACTIONABLE, _USEFUL],
        markers=["o", "s", "D"],
    )
    ax.set_xlabel("Retrospective CAI-AUEBC improvement (×10⁴)", fontsize=6.5)
    _panel_title(ax, "a", "Component substitutions")
    _clean_axis(ax, grid_axis="x")

    ax = axes[1]
    labels = [
        "Current greedy",
        "Beam width 2",
        "Beam width 4",
        "Two-step lookahead",
        "Bounded near-oracle",
    ]
    _forest(
        ax,
        rows,
        labels,
        scale=10000,
        colors=[_REFERENCE, _ACTIONABLE, _ACTIONABLE, _ACTIONABLE, _USEFUL],
        markers=["o", "s", "D", "^", "*"],
        display_labels=[
            "Current greedy",
            "Beam 2",
            "Beam 4",
            "Two-step",
            "Near-oracle*",
        ],
    )
    ax.set_xlabel("Bounded set-planning regret (×10⁴)", fontsize=6.5)
    _panel_title(ax, "b", "Planning boundary")
    _clean_axis(ax, grid_axis="x")
    ax.text(
        0.02,
        -0.20,
        "Near-oracle is retrospective.",
        transform=ax.transAxes,
        fontsize=5.8,
        color=_REFERENCE,
    )

    ax = axes[2]
    labels = ["Feedback benefit", "Baseline minus frozen policy"]
    _forest(
        ax, rows, labels, scale=10000, colors=[_ADVERSE, _ADVERSE], markers=["x", "D"]
    )
    ax.set_xlabel("CAI-AUEBC effect (×10⁴)", fontsize=6.5)
    _panel_title(ax, "c", "Deployable boundary")
    _clean_axis(ax, grid_axis="x")
    ax.text(
        0.02,
        -0.22,
        "Positive favors feedback / frozen policy",
        transform=ax.transAxes,
        fontsize=5.5,
        color=_REFERENCE,
    )
    return fig


_CAPTIONS = {
    "figure1": (
        "**Figure 1. Task-relevant information hierarchy.** Useful information improves "
        "the engineering task, observable information can be inferred from legally "
        "available inspection state, and actionable information improves a bounded "
        "sensing decision. Retrospective teacher/oracle evidence is separated from the "
        "deployable state and policy; passing one question does not imply the next.\n"
    ),
    "figure2": (
        "**Figure 2. Usefulness evidence and boundaries.** (a) The registered matched "
        "B-family spatial field reduces equal-domain CAI-ratio MAE relative to its "
        "scalar counterpart. (b) The selected sparse condition retains 89.9% of the "
        "registered full-field gain but remains worse than the full field. (c) Oracle "
        "acquisitions separate CAI and reconstruction objectives, whereas learned global "
        "masks do not reproduce that separation. Oracle and sparse-design rows are "
        "retrospective and do not establish scanner-time savings or deployability.\n"
    ),
    "figure3": (
        "**Figure 3. Observability evidence and adverse controls.** (a) Strict-OOF static "
        "value ranking is unsupported and its exact-budget regret is comparable with "
        "global and random references. (b) Retrospective conditional values change with "
        "acquisition state. (c) Measured-content prediction is worse than matched-position "
        "and reconstruction controls; although conditional scoring narrowly improves on "
        "the static scorer, shuffled content remains better. Intervals are synchronized "
        "specimen-bootstrap intervals where shown.\n"
    ),
    "figure4": (
        "**Figure 4. Actionability evidence and final boundary.** (a) Retrospective "
        "component substitutions expose valuation and bounded planning gaps. (b) A "
        "positive two-action reachable-pool planning gap remains; the near-oracle is "
        "non-deployable. (c) Feedback is adverse and the frozen learned policy does not "
        "outperform the strongest deployable baseline. Effects use specimen-first, "
        "equal-domain aggregation and synchronized intervals where available.\n"
    ),
}


_STEMS = {
    "figure1": "figure1_information_hierarchy",
    "figure2": "figure2_usefulness",
    "figure3": "figure3_observability",
    "figure4": "figure4_actionability",
}


def _save_figure(fig: Figure, output_root: Path, figure_id: str) -> FigureArtifact:
    stem = _STEMS[figure_id]
    svg = output_root / f"{stem}.svg"
    pdf = output_root / f"{stem}.pdf"
    png = output_root / f"{stem}.png"
    source = (
        output_root
        / {
            "figure1": "figure1_hierarchy.csv",
            "figure2": "figure2_usefulness.csv",
            "figure3": "figure3_observability.csv",
            "figure4": "figure4_actionability.csv",
        }[figure_id]
    )
    caption = output_root / f"{stem}_caption.md"
    caption.write_text(_CAPTIONS[figure_id], encoding="utf-8", newline="\n")
    common = {"bbox_inches": "tight", "facecolor": "white"}
    fig.savefig(
        svg,
        format="svg",
        metadata={
            "Title": stem,
            "Creator": "cmc_bbdm AEI paper evidence renderer",
            "Description": _CAPTIONS[figure_id].replace("**", "").strip(),
            "Date": "2026-08-26",
        },
        **common,
    )
    fig.savefig(
        pdf,
        format="pdf",
        metadata={
            "Title": stem,
            "Author": "AEI information-hierarchy paper package",
            "Subject": "Hash-bound paper evidence figure",
            "Keywords": "ultrasonic inspection, task-relevant information",
            "Creator": "cmc_bbdm AEI paper evidence renderer",
            "Producer": "Matplotlib",
            "CreationDate": _FIXED_TIME,
            "ModDate": _FIXED_TIME,
        },
        **common,
    )
    fig.savefig(
        png,
        format="png",
        dpi=300,
        metadata={"Software": "cmc_bbdm AEI paper evidence renderer"},
        **common,
    )
    return FigureArtifact(svg=svg, pdf=pdf, png=png, source_csv=source, caption=caption)


def _write_figure_manifest(
    output_root: Path, artifacts: dict[str, FigureArtifact]
) -> Path:
    path = output_root / "FIGURE_CHECKSUMS.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("figure_id", "artifact_type", "path", "sha256", "bytes"),
            lineterminator="\n",
        )
        writer.writeheader()
        for figure_id in sorted(artifacts):
            artifact = artifacts[figure_id]
            entries = (
                ("svg", artifact.svg),
                ("pdf", artifact.pdf),
                ("png", artifact.png),
                ("source_csv", artifact.source_csv),
                ("caption", artifact.caption),
            )
            for artifact_type, deliverable in entries:
                writer.writerow(
                    {
                        "figure_id": figure_id,
                        "artifact_type": artifact_type,
                        "path": deliverable.name,
                        "sha256": _sha256(deliverable),
                        "bytes": deliverable.stat().st_size,
                    }
                )
    return path


def render_paper_figures(root: Path, output_root: Path) -> dict[str, FigureArtifact]:
    """Render four deterministic figures and their captions."""
    output_root.mkdir(parents=True, exist_ok=True)
    sources = build_figure_sources(root, output_root)
    renderers = {
        "figure1": _render_figure1,
        "figure2": _render_figure2,
        "figure3": _render_figure3,
        "figure4": _render_figure4,
    }
    artifacts: dict[str, FigureArtifact] = {}
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "text.color": _TEXT,
        "axes.labelcolor": _TEXT,
        "axes.edgecolor": _TEXT,
        "axes.linewidth": 0.65,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "aei-information-hierarchy-20260826",
    }
    with plt.rc_context(style):
        for figure_id, source in sources.items():
            fig = renderers[figure_id](_read_rows(source))
            artifacts[figure_id] = _save_figure(fig, output_root, figure_id)
            plt.close(fig)
    _write_figure_manifest(output_root, artifacts)
    return artifacts
