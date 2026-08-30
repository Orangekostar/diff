"""Deterministic figures for the AEI task-relevant acquisition paper."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from cmc_bbdm.mavis.aei_paper_evidence import PaperMetric, build_canonical_metrics
from cmc_bbdm.mavis.aei_paper_visual_assets import (
    INITIAL_CHECKPOINT,
    LATER_CHECKPOINT,
    REPRESENTATIVE_METHOD,
    REPRESENTATIVE_SPECIMEN,
    PriorityState,
    ReconstructedState,
    gallery_specimen_roster,
    load_gallery_states,
    load_priority_state,
    load_task_priority_maps,
)
from cmc_bbdm.mavis.nature_figure_alignment import (
    require_matplotlib_panel_alignment,
)

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
_USEFUL = "#0F4D92"
_OBSERVABLE = "#2A8C82"
_ACTIONABLE = "#658B3A"
_ADVERSE = "#B64342"
_REFERENCE = "#606060"
_UNCERTAINTY = "#A8A8A8"
_TEXT = "#272727"
_GRID = "#DCDCDC"
_PAPER_WHITE = "#FFFFFF"
_FULL_WIDTH_IN = 180.0 / 25.4
_FIXED_TIME = datetime(2026, 8, 27, tzinfo=UTC)


@dataclass(frozen=True)
class FigureArtifact:
    """Paths belonging to one rendered paper figure."""

    svg: Path
    pdf: Path
    png: Path
    source_csv: Path
    caption: Path
    alignment_json: Path | None = None
    alignment_svg: Path | None = None

    @property
    def outputs(self) -> tuple[Path, Path, Path]:
        return (self.svg, self.pdf, self.png)

    @property
    def qa_outputs(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in (self.alignment_json, self.alignment_svg)
            if path is not None
        )


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


def _source_row(
    root: Path,
    *,
    panel: str,
    series: str,
    metric: str,
    source_artifact: str,
    status: str = "REPRESENTATIVE_VISUAL",
) -> dict[str, str]:
    source = root / source_artifact
    if not source.is_file():
        raise ValueError(f"figure source is unavailable: {source_artifact}")
    return {
        "panel": panel,
        "series": series,
        "metric": metric,
        "value": "",
        "ci95_lower": "",
        "ci95_upper": "",
        "status": status,
        "source_claim_id": "",
        "source_artifact": source_artifact,
        "source_hash": _sha256(source),
    }


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
    source = "artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.csv"
    source_path = root / source
    source_hash = _sha256(source_path)
    definitions = (
        (
            "Part I",
            "Information characterization",
            "Spatial structure, sparse recoverability, objective and state conditioning",
            "H1_INFORMATION_CHARACTERIZATION",
        ),
        (
            "Part I",
            "Retrospective task value",
            "Predictor-conditioned teacher and oracle evidence",
            "H2_RETROSPECTIVE_VALUE",
        ),
        (
            "Part II",
            "Evidence-calibrated realization",
            "Legal-state valuation, source controls, planning, and frozen endpoint",
            "H3_DECISION_REALIZATION",
        ),
        (
            "Part II",
            "Legal partial state",
            "Metadata, acquired-position history, measured content, and exact cost",
            "H4_LEGAL_STATE",
        ),
        (
            "validation criteria",
            "Usefulness and task-value observability",
            "Information enrichment and inference from legal state",
            "H5_VALIDATION_CRITERIA",
        ),
        (
            "validation criteria",
            "Actionability and deployment calibration",
            "Bounded decision consequence under the frozen protocol",
            "H6_DEPLOYMENT_CALIBRATION",
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


def _figure2_rows(root: Path, metrics: dict[str, PaperMetric]) -> list[dict[str, str]]:
    matched = metrics["U1_MATCHED_FIELD"]
    retention = metrics["U2_SPARSE_RETENTION"]
    sparse_gain = metrics["U2_SPARSE_GAIN"]
    sparse_gap = metrics["U2_SPARSE_FULL_GAP"]
    uniform = metrics["U3_UNIFORM_ORACLE"]
    reconstruction_oracle = metrics["U3_RECONSTRUCTION_ORACLE"]
    headroom = metrics["U3_HEADROOM_RETENTION"]
    oracle_cai = metrics["U4_ORACLE_CAI_SPECIFICITY"]
    oracle_image = metrics["U4_ORACLE_IMAGE_SPECIFICITY"]
    rows = [
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
        _effect_row(
            metrics["U1_SURFACE_FIELD"],
            panel="a",
            series="Surface-to-field reduction",
        ),
        _derived_row(
            retention,
            panel="a",
            series="Sparse spatial field",
            value=float(retention.candidate_value),
            metric_name="equal_domain_cai_ratio_mae",
        ),
        _effect_row(retention, panel="a", series="Registered gain retained"),
        _effect_row(sparse_gain, panel="a", series="Surface-to-sparse reduction"),
        _effect_row(sparse_gap, panel="a", series="Sparse-to-full gap"),
        _effect_row(uniform, panel="c", series="Mechanical vs uniform"),
        _effect_row(
            reconstruction_oracle,
            panel="c",
            series="Mechanical vs field-content reference",
        ),
        _effect_row(headroom, panel="c", series="Sequential headroom retained"),
        _effect_row(oracle_cai, panel="d", series="CAI-task priority specificity"),
        _effect_row(
            oracle_image,
            panel="e",
            series="Field-content priority specificity",
        ),
    ]
    rows.extend(
        (
            _source_row(
                root,
                panel="b",
                series="c8-2 initial reconstruction",
                metric="registered_state_reconstruction",
                source_artifact=("results/mavis/p1_state_bank/state_manifest.parquet"),
            ),
            _source_row(
                root,
                panel="d-f",
                series="c8-2 paired CAI and field-content priority maps",
                metric="paired_oracle_priority_percentile",
                source_artifact="results/mva/a2_oracle_value/oracle_values.parquet",
            ),
        )
    )
    return rows


def _figure3_rows(root: Path, metrics: dict[str, PaperMetric]) -> list[dict[str, str]]:
    teacher_metric = metrics["O2_TEACHER_TURNOVER"]
    payload = _load_json(root, teacher_metric)
    claim_by_key = {
        "best_action_turnover": metrics["O2_TEACHER_TURNOVER"],
        "rank_spearman": metrics["O2_TEACHER_RANK"],
        "top_k_jaccard": metrics["O2_TEACHER_TOPK"],
    }
    rows: list[dict[str, str]] = []
    for checkpoint in payload["teacher_by_checkpoint"]:
        cost = float(checkpoint["current_checkpoint"])
        for key, claim in claim_by_key.items():
            rows.append(
                _derived_row(
                    claim,
                    panel="d",
                    series=f"{key}@{format(cost, '.6g')}",
                    value=float(checkpoint[key]),
                    metric_name=key,
                )
            )
    rows.extend(
        (
            _effect_row(
                metrics["O2_TEACHER_OPPORTUNITY"],
                panel="d",
                series="Final-state opportunity",
            ),
            _effect_row(
                metrics["O4_DYNAMIC_MINUS_STATIC"],
                panel="e",
                series="Dynamic minus static regret",
            ),
            _effect_row(
                metrics["O1_STATIC_SPEARMAN"], panel="e", series="Static value rank"
            ),
            _effect_row(
                metrics["U5_RIDGE_HUBER_SPEARMAN"],
                panel="f",
                series="Ridge-Huber rank agreement",
            ),
            _effect_row(
                metrics["U5_RIDGE_MLP_SPEARMAN"],
                panel="f",
                series="Ridge-MLP rank agreement",
            ),
            _source_row(
                root,
                panel="a-b",
                series="c8-2 initial and updated teacher priority",
                metric="strict_oof_teacher_value_map",
                source_artifact="results/mavis/p3_dynamic_voi/action_scores.parquet",
            ),
            _source_row(
                root,
                panel="c",
                series="c8-2 acquired-cell path",
                metric="registered_acquisition_history",
                source_artifact=("results/mavis/p1_state_bank/state_manifest.parquet"),
            ),
        )
    )
    return rows


def _figure4_rows(root: Path, metrics: dict[str, PaperMetric]) -> list[dict[str, str]]:
    positions = metrics["O3_REAL_MINUS_POSITIONS"]
    field_content = metrics["O3_REAL_MINUS_RECONSTRUCTION"]
    endpoint = next(
        row
        for row in _load_csv(root, positions)
        if row["nominal_checkpoint"] == "0.25"
        and row["control_mode"] == "positions_only"
    )
    field_endpoint = next(
        row
        for row in _load_csv(root, field_content)
        if row["nominal_checkpoint"] == "0.25"
        and row["control_mode"] == "reconstruction"
    )
    return [
        _effect_row(metrics["O3_REAL_CHANGE"], panel="a", series="Real-state change"),
        _derived_row(
            positions,
            panel="a",
            series="Measured state",
            value=float(endpoint["real_equal_domain_mae"]),
            metric_name="endpoint_equal_domain_cai_mae",
        ),
        _derived_row(
            positions,
            panel="a",
            series="Acquired-position/history control",
            value=float(endpoint["control_equal_domain_mae"]),
            metric_name="endpoint_equal_domain_cai_mae",
        ),
        _derived_row(
            field_content,
            panel="a",
            series="Field-content control",
            value=float(field_endpoint["control_equal_domain_mae"]),
            metric_name="endpoint_equal_domain_cai_mae",
        ),
        _effect_row(
            positions,
            panel="a",
            series="Measured minus acquired-position/history",
        ),
        _effect_row(field_content, panel="a", series="Measured minus field-content"),
        _effect_row(
            metrics["O4_DYNAMIC_MINUS_SHUFFLED"],
            panel="a",
            series="Dynamic minus shuffled regret",
        ),
        _effect_row(
            metrics["A1_VALUATION_SUBSTITUTION"], panel="b", series="Valuation"
        ),
        _effect_row(
            metrics["A1_LEARNED_PLANNING_SUBSTITUTION"],
            panel="b",
            series="Bounded learned planning",
        ),
        _effect_row(
            metrics["A1_TRUE_VALUE_PLANNING_SUBSTITUTION"],
            panel="b",
            series="True-value stronger planning",
        ),
        _effect_row(
            metrics["A2_GREEDY_PLANNING_REGRET"],
            panel="c",
            series="Current greedy",
        ),
        _effect_row(
            metrics["A2_BEAM4_PLANNING_REGRET"],
            panel="c",
            series="Beam width 4",
        ),
        _effect_row(
            metrics["A4_BASELINE_MINUS_MAVIS"],
            panel="d",
            series="Static reference minus learned implementation",
        ),
    ]


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
        "figure2": _figure2_rows(root, canonical),
        "figure3": _figure3_rows(root, canonical),
        "figure4": _figure4_rows(root, canonical),
    }
    names = {
        "figure1": "figure1_task_relevant_acquisition_framework.csv",
        "figure2": "figure2_information_characterization.csv",
        "figure3": "figure3_state_conditioned_value.csv",
        "figure4": "figure4_valuation_planning_realization.csv",
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
    from matplotlib.transforms import ScaledTranslation

    offset = ScaledTranslation(-9.0 / 72.0, 4.0 / 72.0, ax.figure.dpi_scale_trans)
    ax.text(
        0,
        1,
        letter,
        transform=ax.transAxes + offset,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=_TEXT,
    )
    ax.set_title(title, loc="left", x=0.04, fontsize=7.4, fontweight="bold", pad=6)


def _clean_axis(ax: Axes, *, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=_GRID, linewidth=0.55, zorder=0)
    ax.tick_params(labelsize=7, colors=_TEXT, length=2.5)


def _attach_alignment_contract(
    fig: Figure,
    axes: list[Axes],
    panel_ids: list[str],
    *,
    row_groups: list[list[str]],
    column_groups: list[list[str]] | None = None,
) -> None:
    if len(axes) != len(panel_ids) or len(axes) < 2:
        raise ValueError("multi-panel alignment contract is invalid")
    fig._aei_alignment_options = {  # type: ignore[attr-defined]
        "axes": axes,
        "panel_ids": panel_ids,
        "row_groups": row_groups,
        "column_groups": [] if column_groups is None else column_groups,
    }


def _style_image_axis(ax: Axes) -> None:
    ax.autoscale(enable=True, axis="both", tight=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _measurement_overlay(state: ReconstructedState, color: str) -> np.ndarray:
    rgba = np.zeros((*state.measurement_mask.shape, 4), dtype=np.float64)
    rgba[state.measurement_mask] = matplotlib.colors.to_rgba(color, alpha=0.34)
    return rgba


def _draw_reconstruction(
    ax: Axes,
    state: ReconstructedState,
    *,
    show_measurements: bool,
) -> None:
    ax.imshow(state.image, interpolation="nearest", aspect="auto")
    if show_measurements:
        ax.imshow(
            _measurement_overlay(state, _OBSERVABLE),
            interpolation="nearest",
            aspect="auto",
        )
    ax.set_aspect("equal", adjustable="datalim")
    _style_image_axis(ax)


def _draw_priority_overlay(
    ax: Axes,
    state: PriorityState | ReconstructedState,
    percentiles: np.ndarray | None = None,
    *,
    alpha: float = 0.56,
) -> Any:
    reconstruction = state.reconstruction if isinstance(state, PriorityState) else state
    values = state.percentiles if isinstance(state, PriorityState) else percentiles
    if values is None or values.shape != (8, 8):
        raise ValueError("priority overlay requires one 8x8 map")
    ax.imshow(reconstruction.image, interpolation="nearest", aspect="auto")
    image = ax.pcolormesh(
        np.asarray(reconstruction.column_boundaries, dtype=np.float64),
        np.asarray(reconstruction.row_boundaries, dtype=np.float64),
        values,
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        alpha=alpha,
        edgecolors=(1.0, 1.0, 1.0, 0.26),
        linewidth=0.25,
    )
    ax.set_xlim(0, reconstruction.image.shape[1] - 1)
    ax.set_ylim(reconstruction.image.shape[0] - 1, 0)
    ax.set_aspect("equal", adjustable="datalim")
    _style_image_axis(ax)
    return image


def _draw_acquisition_path(ax: Axes, state: ReconstructedState) -> None:
    _draw_reconstruction(ax, state, show_measurements=False)
    cells = state.acquired_cell_indices
    if not cells:
        raise ValueError("acquisition path is empty")
    rows = np.asarray(state.row_boundaries, dtype=np.float64)
    columns = np.asarray(state.column_boundaries, dtype=np.float64)
    x = np.asarray([(columns[cell % 8] + columns[cell % 8 + 1]) / 2 for cell in cells])
    y = np.asarray([(rows[cell // 8] + rows[cell // 8 + 1]) / 2 for cell in cells])
    ax.plot(x, y, color=_PAPER_WHITE, lw=1.4, alpha=0.9, zorder=3)
    ax.scatter(
        x,
        y,
        c=np.arange(len(cells)),
        cmap="viridis",
        s=9,
        linewidths=0.25,
        edgecolors=_PAPER_WHITE,
        zorder=4,
    )
    ax.scatter(
        [x[0], x[-1]],
        [y[0], y[-1]],
        s=22,
        marker="o",
        c=[_PAPER_WHITE, _ADVERSE],
        edgecolors=_TEXT,
        linewidths=0.45,
        zorder=5,
    )


def _outline_top_cells(
    ax: Axes,
    state: ReconstructedState,
    values: np.ndarray,
    *,
    color: str = _PAPER_WHITE,
    count: int = 5,
) -> None:
    rows = state.row_boundaries
    columns = state.column_boundaries
    for cell in np.argsort(values.ravel())[-count:]:
        row = int(cell) // 8
        column = int(cell) % 8
        ax.add_patch(
            Rectangle(
                (columns[column], rows[row]),
                columns[column + 1] - columns[column],
                rows[row + 1] - rows[row],
                fill=False,
                edgecolor=color,
                linewidth=0.9,
                zorder=6,
            )
        )


def _render_figure1(rows: list[dict[str, str]]) -> Figure:
    del rows
    fig, ax = plt.subplots(figsize=(_FULL_WIDTH_IN, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.95,
        "Task-relevant information acquisition",
        fontsize=10,
        fontweight="bold",
        color=_TEXT,
        va="top",
    )
    bands = (
        (0.025, 0.690, _USEFUL, "PART I - INFORMATION CHARACTERIZATION"),
        (0.745, 0.230, _ACTIONABLE, "PART II\nSTATE-CONDITIONED ACQUISITION"),
    )
    for x, width, color, label in bands:
        ax.add_patch(
            Rectangle((x, 0.82), width, 0.07, facecolor=color, edgecolor=color)
        )
        ax.text(
            x + width / 2,
            0.855,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=6.2 if width > 0.3 else 5.1,
            fontweight="bold",
            linespacing=1.25,
        )
    stages = (
        (
            0.025,
            _USEFUL,
            "a",
            "Complete sensing field",
            "Information-rich field\n!= engineering decision",
        ),
        (
            0.265,
            _OBSERVABLE,
            "b",
            "Limited sensing under exact cost",
            "Only a legal subset\ncan be measured",
        ),
        (
            0.505,
            _USEFUL,
            "c",
            "Task-relevant value",
            "Candidates differ in\ndownstream CAI value",
        ),
        (
            0.745,
            _ACTIONABLE,
            "d",
            "State-conditioned\nacquisition loop",
            "I_t -> value -> legal action\n-> reveal -> I_{t+1}",
        ),
    )
    for x, color, panel, heading, detail in stages:
        ax.add_patch(
            Rectangle(
                (x, 0.43),
                0.21,
                0.31,
                facecolor="#FAFAFA",
                edgecolor=color,
                lw=1.5,
            )
        )
        ax.text(
            x + 0.014,
            0.715,
            panel,
            ha="left",
            va="center",
            color=color,
            fontsize=7.2,
            fontweight="bold",
        )
        ax.text(
            x + 0.105,
            0.655,
            heading,
            ha="center",
            va="center",
            color=_TEXT,
            fontsize=5.35 if panel == "b" else 5.7 if panel == "d" else 6.0,
            fontweight="bold",
            linespacing=1.25,
        )
        ax.text(
            x + 0.105,
            0.535,
            detail,
            ha="center",
            va="center",
            color=_REFERENCE,
            fontsize=5.55,
            linespacing=1.25,
        )
    for left, right in ((0.235, 0.265), (0.475, 0.505), (0.715, 0.745)):
        ax.annotate(
            "",
            xy=(right - 0.003, 0.585),
            xytext=(left + 0.003, 0.585),
            arrowprops={"arrowstyle": "-|>", "color": _REFERENCE, "lw": 1.2},
        )
    ax.axvline(0.73, ymin=0.15, ymax=0.78, color=_OBSERVABLE, lw=0.9, ls="--")
    ax.text(
        0.730,
        0.115,
        "legal-state boundary",
        ha="center",
        va="center",
        fontsize=5.0,
        color=_OBSERVABLE,
    )
    ax.add_patch(
        Rectangle(
            (0.025, 0.19),
            0.690,
            0.12,
            facecolor="white",
            edgecolor=_USEFUL,
            lw=1.0,
            linestyle="--",
        )
    )
    ax.text(
        0.370,
        0.250,
        "Retrospective teacher/oracle evidence | characterization only",
        ha="center",
        va="center",
        fontsize=5.7,
        color=_USEFUL,
        fontweight="bold",
    )
    ax.add_patch(
        Rectangle(
            (0.745, 0.19),
            0.230,
            0.12,
            facecolor="white",
            edgecolor=_ACTIONABLE,
            lw=1.0,
            linestyle="--",
        )
    )
    ax.text(
        0.860,
        0.250,
        "Deployable decision uses\nlegal state and exact cost only",
        ha="center",
        va="center",
        fontsize=5.45,
        color=_ACTIONABLE,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.085,
        "WHY: complete-field information is not the same as decision-relevant measurement value",
        ha="center",
        va="center",
        fontsize=6.6,
        color=_TEXT,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.98)
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


def _render_figure2_nature(root: Path, rows: list[dict[str, str]]) -> Figure:
    priorities = load_task_priority_maps(root, specimen_id=REPRESENTATIVE_SPECIMEN)
    state = priorities.reconstruction
    fig, axes_grid = plt.subplots(2, 3, figsize=(_FULL_WIDTH_IN, 5.15))
    axes = list(axes_grid.ravel())

    ax = axes[0]
    labels = ["Matched scalar", "Matched spatial field", "Sparse spatial field"]
    values = [_float(_find(rows, label), "value") for label in labels]
    y = np.arange(3)[::-1]
    colors = [_REFERENCE, _USEFUL, _OBSERVABLE]
    ax.hlines(y, 0.12, values, colors=colors, linewidth=1.5)
    ax.scatter(values, y, c=colors, s=30, zorder=3, edgecolors=_PAPER_WHITE, lw=0.5)
    for yi, value in zip(y, values, strict=True):
        ax.text(value + 0.002, yi, f"{value:.3f}", va="center", fontsize=6.2)
    reduction = 100 * (values[0] - values[1]) / values[0]
    retention = 100 * float(_find(rows, "Registered gain retained")["value"])
    gap = float(_find(rows, "Sparse-to-full gap")["value"])
    ax.text(
        0.98,
        0.58,
        f"{reduction:.1f}% MAE reduction\n{retention:.1f}% of full-field gain retained\n"
        f"Sparse-to-full gap {gap:.5f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=5.35,
        color=_OBSERVABLE,
        fontweight="bold",
    )
    ax.set_yticks(y, ["Scalar", "Full field", "25% sparse"], fontsize=6.1)
    ax.set_xlim(0.118, 0.208)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("Equal-domain CAI-ratio MAE (lower is better)", fontsize=6.1)
    _panel_title(ax, "a", "Spatial gain and sparse retention")
    _clean_axis(ax)

    ax = axes[1]
    _draw_reconstruction(ax, state, show_measurements=True)
    _panel_title(ax, "b", "Initial legal state | 3.13%")

    ax = axes[2]
    task_labels = ["Mechanical vs uniform", "Mechanical vs field-content reference"]
    _forest(
        ax,
        rows,
        task_labels,
        scale=1000,
        colors=[_USEFUL, _USEFUL],
        markers=["o", "s"],
        display_labels=["vs uniform", "field-content"],
    )
    ax.set_xlim(2.2, 5.5)
    ax.set_ylim(-0.8, 1.6)
    ax.set_xlabel("CAI-AUEBC improvement (x10^3)", fontsize=6.4)
    headroom = 100 * float(_find(rows, "Sequential headroom retained")["value"])
    ax.text(
        0.02,
        0.06,
        f"{headroom:.1f}% of sequential-oracle headroom\nRetrospective; non-deployable",
        transform=ax.transAxes,
        fontsize=5.2,
        color=_REFERENCE,
    )
    _panel_title(ax, "c", "Heterogeneous\nspatial opportunity")
    _clean_axis(ax)

    ax = axes[3]
    priority_image = _draw_priority_overlay(
        ax, state, priorities.mechanical_percentiles
    )
    _outline_top_cells(ax, state, priorities.mechanical_percentiles)
    _panel_title(ax, "d", "CAI-task priority")

    ax = axes[4]
    _draw_priority_overlay(ax, state, priorities.reconstruction_percentiles)
    _outline_top_cells(ax, state, priorities.reconstruction_percentiles)
    _panel_title(ax, "e", "C-scan-content priority")
    ax.text(
        0.5,
        -0.10,
        "registered normalized-RGB-MSE field-content reference",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.0,
        color=_REFERENCE,
    )

    ax = axes[5]
    ax.imshow(state.image, interpolation="nearest", aspect="auto")
    difference = ax.pcolormesh(
        np.asarray(state.column_boundaries, dtype=np.float64),
        np.asarray(state.row_boundaries, dtype=np.float64),
        priorities.percentile_difference,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        shading="flat",
        alpha=0.66,
        edgecolors=(1.0, 1.0, 1.0, 0.24),
        linewidth=0.25,
    )
    ax.set_xlim(0, state.image.shape[1] - 1)
    ax.set_ylim(state.image.shape[0] - 1, 0)
    ax.set_aspect("equal", adjustable="datalim")
    _style_image_axis(ax)
    _panel_title(ax, "f", "CAI-specific excess priority")

    fig.subplots_adjust(
        left=0.125, right=0.985, bottom=0.14, top=0.94, wspace=0.5, hspace=0.52
    )
    value_axis = fig.add_axes([0.15, 0.045, 0.48, 0.012])
    value_bar = fig.colorbar(priority_image, cax=value_axis, orientation="horizontal")
    value_bar.set_label("Within-map priority percentile", fontsize=5.2, labelpad=1)
    value_bar.ax.tick_params(labelsize=5.0, length=1.5)
    difference_axis = fig.add_axes([0.74, 0.045, 0.20, 0.012])
    difference_bar = fig.colorbar(
        difference, cax=difference_axis, orientation="horizontal"
    )
    difference_bar.set_label("CAI - field-content percentile", fontsize=5.0, labelpad=1)
    difference_bar.ax.tick_params(labelsize=5.0, length=1.5)
    _attach_alignment_contract(
        fig,
        axes,
        list("abcdef"),
        row_groups=[["a", "b", "c"], ["d", "e", "f"]],
        column_groups=[["a", "d"], ["b", "e"], ["c", "f"]],
    )
    return fig


def _render_figure3_nature(root: Path, rows: list[dict[str, str]]) -> Figure:
    initial = load_priority_state(
        root,
        specimen_id=REPRESENTATIVE_SPECIMEN,
        method=REPRESENTATIVE_METHOD,
        checkpoint=INITIAL_CHECKPOINT,
    )
    later = load_priority_state(
        root,
        specimen_id=REPRESENTATIVE_SPECIMEN,
        method=REPRESENTATIVE_METHOD,
        checkpoint=LATER_CHECKPOINT,
    )
    fig, axes_grid = plt.subplots(2, 3, figsize=(_FULL_WIDTH_IN, 5.0))
    axes = list(axes_grid.ravel())

    ax = axes[0]
    image = _draw_priority_overlay(ax, initial)
    _outline_top_cells(ax, initial.reconstruction, initial.percentiles)
    _panel_title(ax, "a", "Initial priority | 3.13%")

    ax = axes[1]
    _draw_priority_overlay(ax, later)
    _outline_top_cells(ax, later.reconstruction, later.percentiles)
    _panel_title(ax, "b", "Updated priority | 18.75%")

    ax = axes[2]
    _draw_acquisition_path(ax, later.reconstruction)
    _panel_title(ax, "c", "Acquisition history")

    ax = axes[3]
    metric_specs = (
        ("best_action_turnover", "Best-action turnover", _OBSERVABLE, "s", "--"),
        ("rank_spearman", "Rank agreement", _REFERENCE, "o", "-"),
        ("top_k_jaccard", "Top-5 overlap", _USEFUL, "^", ":"),
    )
    for metric_name, label, color, marker, linestyle in metric_specs:
        selected = [
            row for row in rows if row["panel"] == "d" and row["metric"] == metric_name
        ]
        x = np.asarray([float(row["series"].split("@")[1]) * 100 for row in selected])
        values = np.asarray([float(row["value"]) for row in selected])
        ax.plot(
            x,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.15,
            markersize=3.5,
        )
    for y_position, label, color in (
        (0.735, "Turnover 70.4%", _OBSERVABLE),
        (0.435, "Rank 0.405", _REFERENCE),
        (0.275, "Top-5 0.307", _USEFUL),
    ):
        ax.text(
            16.2,
            y_position,
            label,
            fontsize=5.0,
            color=color,
            va="center",
        )
    ax.set_xlabel("Acquired normalized cost (%)", fontsize=6.2)
    ax.set_ylabel("Teacher-state diagnostic", fontsize=6.2)
    ax.set_ylim(0.25, 0.82)
    ax.set_xlim(5.5, 23.5)
    _panel_title(ax, "d", "Value changes with\nacquired evidence")
    _clean_axis(ax)

    ax = axes[4]
    dynamic = _find(rows, "Dynamic minus static regret")
    value = float(dynamic["value"]) * 1000
    low = float(dynamic["ci95_lower"]) * 1000
    high = float(dynamic["ci95_upper"]) * 1000
    ax.errorbar(
        value,
        0,
        xerr=[[value - low], [high - value]],
        fmt="o",
        color=_USEFUL,
        ecolor=_USEFUL,
        capsize=3,
        ms=5,
    )
    ax.axvline(0, color=_REFERENCE, lw=0.8, linestyle="--")
    ax.set_yticks([0], ["Dynamic\n- static"], fontsize=6.0)
    ax.set_xlim(-2.6, 0.45)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel("One-step value regret (x10^3)", fontsize=6.4)
    rank = _find(rows, "Static value rank")
    ax.text(
        0.02,
        0.08,
        f"Static rank Spearman {float(rank['value']):.4f}\n"
        f"95% CI [{float(rank['ci95_lower']):.4f}, {float(rank['ci95_upper']):.4f}]",
        transform=ax.transAxes,
        fontsize=5.4,
        color=_REFERENCE,
    )
    ax.text(
        0.02,
        0.90,
        "Negative favors\ndynamic valuation",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color=_USEFUL,
    )
    _panel_title(ax, "e", "Dynamic versus static valuation")
    _clean_axis(ax)

    ax = axes[5]
    _forest(
        ax,
        rows,
        [
            "Ridge-Huber rank agreement",
            "Ridge-MLP rank agreement",
        ],
        scale=1,
        colors=[_ACTIONABLE, _ADVERSE],
        markers=["o", "s"],
        display_labels=["Ridge-Huber", "Ridge-MLP"],
    )
    ax.set_xlim(0, 0.9)
    ax.set_ylim(-0.65, 1.55)
    ax.set_xlabel("Strict-OOF action-value Spearman", fontsize=6.1)
    ax.text(
        0.02,
        0.06,
        "Full-state MAE\nRidge 0.08964 | Huber 0.08618\nMLP 0.15067",
        transform=ax.transAxes,
        fontsize=5.1,
        color=_REFERENCE,
    )
    _panel_title(ax, "f", "Predictor dependence")
    _clean_axis(ax)

    fig.subplots_adjust(
        left=0.125, right=0.985, bottom=0.09, top=0.94, wspace=0.5, hspace=0.52
    )
    color_axis = fig.add_axes([0.43, 0.505, 0.24, 0.012])
    colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Priority percentile", fontsize=5.4, labelpad=1)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(labelsize=5.0, length=1.5)
    _attach_alignment_contract(
        fig,
        axes,
        list("abcdef"),
        row_groups=[["a", "b", "c"], ["d", "e", "f"]],
        column_groups=[["a", "d"], ["b", "e"], ["c", "f"]],
    )
    return fig


def _render_figure4_reframed(rows: list[dict[str, str]]) -> Figure:
    fig, axes_grid = plt.subplots(2, 2, figsize=(_FULL_WIDTH_IN, 5.05))
    axes = list(axes_grid.ravel())

    ax = axes[0]
    labels = [
        "Measured state",
        "Acquired-position/history control",
        "Field-content control",
    ]
    values = np.asarray([float(_find(rows, label)["value"]) for label in labels])
    positions = np.arange(3)[::-1]
    colors = [_USEFUL, _REFERENCE, _UNCERTAINTY]
    ax.hlines(positions, 0.07, values, colors=colors, linewidth=1.5)
    ax.scatter(values, positions, c=colors, s=30, zorder=3, edgecolors="white")
    ax.set_yticks(
        positions,
        ["Measured state", "Position/history", "Field-content"],
        fontsize=5.8,
    )
    ax.set_xlim(0.07, 0.14)
    ax.set_ylim(-1.35, 2.55)
    ax.set_xlabel("Endpoint CAI MAE (lower is better)", fontsize=6.2)
    history = _find(rows, "Measured minus acquired-position/history")
    field = _find(rows, "Measured minus field-content")
    shuffled = _find(rows, "Dynamic minus shuffled regret")
    ax.text(
        0.02,
        0.10,
        f"Real - history {float(history['value']):+.5f}\n"
        f"Real - field-content {float(field['value']):+.5f}\n"
        f"Dynamic real - shuffled {float(shuffled['value']):+.3e}",
        transform=ax.transAxes,
        fontsize=5.15,
        color=_ADVERSE,
    )
    _panel_title(ax, "a", "Matched source controls")
    _clean_axis(ax)

    ax = axes[1]
    _forest(
        ax,
        rows,
        ["Valuation", "Bounded learned planning", "True-value stronger planning"],
        scale=10000,
        colors=[_USEFUL, _ACTIONABLE, _USEFUL],
        markers=["o", "s", "D"],
        display_labels=["Valuation", "Learned planning", "True-value planning"],
    )
    ax.set_xlabel("Retrospective CAI-AUEBC improvement (x10^4)", fontsize=6.2)
    _panel_title(ax, "b", "Valuation and planning substitutions")
    _clean_axis(ax, grid_axis="x")

    ax = axes[2]
    _forest(
        ax,
        rows,
        ["Current greedy", "Beam width 4"],
        scale=10000,
        colors=[_REFERENCE, _ACTIONABLE],
        markers=["o", "D"],
        display_labels=["Current greedy", "Beam width 4"],
    )
    ax.set_xlabel("Reachable-set planning regret (x10^4)", fontsize=6.2)
    ax.text(
        0.10,
        0.52,
        "Reference: retrospective joint near-oracle set\n"
        "Two-action reachable pool at 6.25%",
        transform=ax.transAxes,
        fontsize=5.2,
        color=_REFERENCE,
        va="center",
    )
    _panel_title(ax, "c", "Cost-constrained set realization")
    _clean_axis(ax)

    ax = axes[3]
    _forest(
        ax,
        rows,
        ["Static reference minus learned implementation"],
        scale=10000,
        colors=[_ADVERSE],
        markers=["D"],
        display_labels=["Reference - learned"],
    )
    ax.set_xlim(-1.05, 0.20)
    ax.set_ylim(-0.75, 0.75)
    ax.set_xlabel("CAI-AUEBC contrast (x10^4)", fontsize=6.2)
    ax.text(
        0.04,
        0.90,
        "Negative favors the static reference\nLearned 0.125053 | static 0.124992",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color=_ADVERSE,
    )
    _panel_title(ax, "d", "Deployment calibration")
    _clean_axis(ax)

    fig.subplots_adjust(
        left=0.18, right=0.985, bottom=0.10, top=0.94, wspace=0.52, hspace=0.50
    )
    _attach_alignment_contract(
        fig,
        axes,
        list("abcd"),
        row_groups=[["a", "b"], ["c", "d"]],
        column_groups=[["a", "c"], ["b", "d"]],
    )
    return fig


def _render_supplementary_gallery(root: Path) -> Figure:
    pairs = load_gallery_states(root)
    fig, axes_grid = plt.subplots(6, 2, figsize=(_FULL_WIDTH_IN, 8.75))
    axes = list(axes_grid.ravel())
    panel_ids = list("abcdefghijkl")
    image = None
    for row_index, pair in enumerate(pairs):
        for column_index, state in enumerate((pair.initial, pair.later)):
            index = row_index * 2 + column_index
            ax = axes[index]
            image = _draw_priority_overlay(ax, state, alpha=0.54)
            checkpoint = "3.13%" if column_index == 0 else "18.75%"
            _panel_title(
                ax,
                panel_ids[index],
                f"{pair.specimen.specimen_id} | {checkpoint}",
            )
    if image is None:
        raise ValueError("supplementary gallery is empty")
    fig.subplots_adjust(
        left=0.045, right=0.99, bottom=0.075, top=0.98, wspace=0.08, hspace=0.34
    )
    color_axis = fig.add_axes([0.33, 0.025, 0.36, 0.012])
    colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        "Within-state teacher-value percentile", fontsize=5.4, labelpad=1
    )
    colorbar.ax.tick_params(labelsize=5.0, length=1.5)
    _attach_alignment_contract(
        fig,
        axes,
        panel_ids,
        row_groups=[
            panel_ids[index : index + 2] for index in range(0, len(panel_ids), 2)
        ],
        column_groups=[panel_ids[0::2], panel_ids[1::2]],
    )
    return fig


_CAPTIONS = {
    "figure1": (
        "**Figure 1. Task-Relevant Information Acquisition.** Part I characterizes "
        "spatial, sparse, objective-conditioned, and state-conditioned task "
        "information. Part II estimates value from legal partial state and realizes "
        "measurement sets under exact native-raster cost. Retrospective teachers and "
        "oracles characterize value; the acquisition rule uses only legal state.\n"
    ),
    "figure2": (
        "**Figure 2. What ultrasonic information matters for CAI?** (a) Equal-domain "
        "CAI-ratio MAE for matched scalar, full spatial field, and registered 25% "
        "sparse field. (b) Hash-verified initial legal state for specimen c8-2; teal "
        "marks measured native-raster positions. (c) Retrospective mechanical-oracle "
        "opportunity relative to uniform acquisition and the registered field-content "
        "reference. (d,e) CAI-task and C-scan-content within-map priority percentiles "
        "on the same initial state and legal 8x8 action grid; white outlines mark the "
        "five highest-priority cells. The field-content reference is operationalized "
        "by the registered normalized-RGB-MSE reconstruction objective. (f) Paired "
        "CAI-minus-field-content percentile difference, not raw utility or a causal "
        "material map. Oracles are retrospective and non-deployable.\n"
    ),
    "figure3": (
        "**Figure 3. Why must measurement value be state-conditioned?** (a,b) "
        "Within-state strict-OOF teacher-value percentiles for all 64 legal next-cell "
        "actions on specimen c8-2 at 3.13% and 18.75%; white outlines mark the five "
        "highest-priority cells. (c) Stored acquisition history for the same trajectory, "
        "from the initial white marker to the latest red marker. (d) Best-action "
        "turnover, rank agreement, and top-five overlap over acquired normalized cost. "
        "(e) Dynamic real-state versus static next-action regret; negative favors "
        "dynamic valuation. (f) Predictor-conditioned rank agreement with the unequal "
        "shallow-MLP accuracy boundary. Priority maps are state- and "
        "predictor-conditioned, not universal material maps.\n"
    ),
    "figure4": (
        "**Figure 4. From state-conditioned value to a cost-constrained decision.** "
        "(a) Matched measured-state, acquired-position/history, field-content, and "
        "shuffled-content controls retain their observed adverse directions; the "
        "field-content control uses the registered reconstruction control. (b) "
        "Retrospective substitutions separate valuation and bounded-planning effects. "
        "(c) Greedy and beam-4 selection retain positive regret relative to a "
        "retrospective joint near-oracle set in the registered two-action pool. (d) "
        "Signed deployment calibration for one registered supervised implementation; "
        "negative favors the static reference. This endpoint does not define the "
        "performance of the full Task-Relevant Information Acquisition framework.\n"
    ),
    "supplementary_figure_s1": (
        "**Supplementary Figure S1. Cross-domain state-conditioned priority "
        "gallery.** Rows show one specimen from each of six held-out domains, selected "
        "as the lexicographically first specimen in the sorted domain roster without "
        "using outcomes or effects. Left and right panels show the registered 3.13% "
        "initial and 18.75% states for the same one-shot mechanical-oracle trajectory. "
        "Overlays are within-state percentiles of strict-OOF teacher value on the "
        "registered 8x8 legal action grid. The gallery demonstrates qualitative "
        "breadth and does not define cross-panel absolute value magnitudes.\n"
    ),
}


_STEMS = {
    "figure1": "figure1_task_relevant_acquisition_framework",
    "figure2": "figure2_information_characterization",
    "figure3": "figure3_state_conditioned_value",
    "figure4": "figure4_valuation_planning_realization",
    "supplementary_figure_s1": (
        "supplementary_figure_s1_cross_domain_state_priority_gallery"
    ),
}


_SOURCE_NAMES = {
    "figure1": "figure1_task_relevant_acquisition_framework.csv",
    "figure2": "figure2_information_characterization.csv",
    "figure3": "figure3_state_conditioned_value.csv",
    "figure4": "figure4_valuation_planning_realization.csv",
    "supplementary_figure_s1": (
        "supplementary_figure_s1_cross_domain_state_priority_gallery.csv"
    ),
}


def _save_figure(fig: Figure, output_root: Path, figure_id: str) -> FigureArtifact:
    stem = _STEMS[figure_id]
    svg = output_root / f"{stem}.svg"
    pdf = output_root / f"{stem}.pdf"
    png = output_root / f"{stem}.png"
    source = output_root / _SOURCE_NAMES[figure_id]
    caption = output_root / f"{stem}_caption.md"
    caption.write_text(_CAPTIONS[figure_id], encoding="utf-8", newline="\n")
    alignment_json = None
    alignment_svg = None
    alignment_options = getattr(fig, "_aei_alignment_options", None)
    if alignment_options is not None:
        alignment_json = output_root / f"{stem}.alignment.json"
        alignment_svg = output_root / f"{stem}.alignment.svg"
        require_matplotlib_panel_alignment(
            fig,
            json_out=alignment_json,
            overlay_svg=alignment_svg,
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
            **alignment_options,
        )
    common = {"facecolor": "white"}
    fig.savefig(
        svg,
        format="svg",
        metadata={
            "Title": stem,
            "Creator": "cmc_bbdm AEI paper evidence renderer",
            "Description": _CAPTIONS[figure_id].replace("**", "").strip(),
            "Date": "2026-08-27",
        },
        **common,
    )
    svg.write_text(
        "\n".join(
            line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fig.savefig(
        pdf,
        format="pdf",
        metadata={
            "Title": stem,
            "Author": "AEI task-relevant acquisition paper package",
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
    return FigureArtifact(
        svg=svg,
        pdf=pdf,
        png=png,
        source_csv=source,
        caption=caption,
        alignment_json=alignment_json,
        alignment_svg=alignment_svg,
    )


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
                *(("alignment_json", path) for path in artifact.qa_outputs[:1]),
                *(("alignment_svg", path) for path in artifact.qa_outputs[1:]),
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


def _supplementary_source_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    panel_ids = list("abcdefghijkl")
    for row_index, specimen in enumerate(gallery_specimen_roster(root)):
        for column_index, checkpoint in enumerate(
            (INITIAL_CHECKPOINT, LATER_CHECKPOINT)
        ):
            panel = panel_ids[row_index * 2 + column_index]
            label = f"{specimen.specimen_id}@{checkpoint:.6g}"
            rows.extend(
                (
                    _source_row(
                        root,
                        panel=panel,
                        series=f"{label} reconstruction",
                        metric="registered_state_reconstruction",
                        source_artifact=(
                            "results/mavis/p1_state_bank/state_manifest.parquet"
                        ),
                    ),
                    _source_row(
                        root,
                        panel=panel,
                        series=f"{label} teacher priority",
                        metric="strict_oof_teacher_value_map",
                        source_artifact=(
                            "results/mavis/p3_dynamic_voi/action_scores.parquet"
                        ),
                    ),
                )
            )
    return rows


def build_supplementary_figure_sources(
    root: Path, output_root: Path
) -> dict[str, Path]:
    """Write source indices for supplementary qualitative figures."""

    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / _SOURCE_NAMES["supplementary_figure_s1"]
    _write_rows(path, _supplementary_source_rows(root.resolve()))
    return {"supplementary_figure_s1": path}


def _figure_style() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7,
        "text.color": _TEXT,
        "axes.labelcolor": _TEXT,
        "axes.edgecolor": _TEXT,
        "axes.linewidth": 0.75,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "aei-task-relevant-acquisition-20260828",
    }


def render_paper_figures(root: Path, output_root: Path) -> dict[str, FigureArtifact]:
    """Render four deterministic main-paper figures and their captions."""
    output_root.mkdir(parents=True, exist_ok=True)
    sources = build_figure_sources(root, output_root)
    renderers = {
        "figure1": _render_figure1,
        "figure2": lambda rows: _render_figure2_nature(root, rows),
        "figure3": lambda rows: _render_figure3_nature(root, rows),
        "figure4": _render_figure4_reframed,
    }
    artifacts: dict[str, FigureArtifact] = {}
    with plt.rc_context(_figure_style()):
        for figure_id, source in sources.items():
            fig = renderers[figure_id](_read_rows(source))
            artifacts[figure_id] = _save_figure(fig, output_root, figure_id)
            plt.close(fig)
    _write_figure_manifest(output_root, artifacts)
    return artifacts


def render_supplementary_figures(
    root: Path, output_root: Path
) -> dict[str, FigureArtifact]:
    """Render the deterministic six-domain supplementary gallery."""

    output_root.mkdir(parents=True, exist_ok=True)
    build_supplementary_figure_sources(root, output_root)
    with plt.rc_context(_figure_style()):
        fig = _render_supplementary_gallery(root)
        artifact = _save_figure(fig, output_root, "supplementary_figure_s1")
        plt.close(fig)
    artifacts = {"supplementary_figure_s1": artifact}
    _write_figure_manifest(output_root, artifacts)
    return artifacts


def _panel_crop_boxes(
    alignment: dict[str, Any],
    image: Image.Image,
    *,
    axes_only: bool = False,
) -> dict[str, tuple[int, int, int, int]]:
    layout = alignment.get("layout", {})
    figure = layout.get("figure", {})
    panels = layout.get("panels", [])
    width_pt = float(figure.get("width_pt", 0))
    height_pt = float(figure.get("height_pt", 0))
    if width_pt <= 0 or height_pt <= 0 or not panels:
        raise ValueError("panel alignment geometry is incomplete")

    columns: dict[int, list[dict[str, Any]]] = {}
    for panel in panels:
        columns.setdefault(int(panel["col_start"]), []).append(panel)
    column_ids = sorted(columns)
    x_extents = {
        column: (
            float(np.mean([item["bbox_pt"][0] for item in columns[column]])),
            float(np.mean([item["bbox_pt"][2] for item in columns[column]])),
        )
        for column in column_ids
    }
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for panel in panels:
        column = int(panel["col_start"])
        column_index = column_ids.index(column)
        left_pt = (
            0.0
            if column_index == 0
            else (x_extents[column_ids[column_index - 1]][1] + x_extents[column][0]) / 2
        )
        right_pt = (
            width_pt
            if column_index == len(column_ids) - 1
            else (x_extents[column][1] + x_extents[column_ids[column_index + 1]][0]) / 2
        )
        panel_bbox = [float(value) for value in panel["bbox_pt"]]
        if axes_only:
            left_pt, bottom_pt, right_pt, top_pt = panel_bbox
        else:
            top_pt = min(height_pt, panel_bbox[3] + 16.0)
            bottom_pt = max(0.0, panel_bbox[1] - 28.0)
        left = max(0, int(np.floor(left_pt / width_pt * image.width)))
        right = min(image.width, int(np.ceil(right_pt / width_pt * image.width)))
        upper = max(0, int(np.floor((height_pt - top_pt) / height_pt * image.height)))
        lower = min(
            image.height,
            int(np.ceil((height_pt - bottom_pt) / height_pt * image.height)),
        )
        if right <= left or lower <= upper:
            raise ValueError(f"panel {panel['id']} has invalid crop geometry")
        boxes[str(panel["id"])] = (left, upper, right, lower)
    return boxes


def _trim_white_margin(image: Image.Image, *, padding: int = 12) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    content = np.any(pixels < 250, axis=2)
    y, x = np.nonzero(content)
    if not len(x):
        raise ValueError("panel crop contains no visible content")
    left = max(0, int(x.min()) - padding)
    upper = max(0, int(y.min()) - padding)
    right = min(image.width, int(x.max()) + padding + 1)
    lower = min(image.height, int(y.max()) + padding + 1)
    return image.crop((left, upper, right, lower))


def export_panel_pngs(
    artifacts: dict[str, FigureArtifact], output_root: Path
) -> dict[str, Path]:
    """Export formal figure panels as deterministic, unscaled PNG crops."""

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    exports: dict[str, Path] = {}
    manifest_rows: list[dict[str, str | int]] = []
    for figure_id in sorted(artifacts):
        artifact = artifacts[figure_id]
        figure_dir = output_root / figure_id
        figure_dir.mkdir()
        with Image.open(artifact.png) as source:
            image = source.convert("RGB")
            if artifact.alignment_json is None:
                if figure_id != "figure1":
                    raise ValueError(f"{figure_id} has no panel alignment geometry")
                panel_images = {"full": image.copy()}
            else:
                alignment = json.loads(
                    artifact.alignment_json.read_text(encoding="utf-8")
                )
                if alignment.get("verdict") != "PASS":
                    raise ValueError(f"{figure_id} panel alignment did not pass")
                axes_only = figure_id == "supplementary_figure_s1"
                panel_images = {
                    panel_id: _trim_white_margin(image.crop(box))
                    for panel_id, box in _panel_crop_boxes(
                        alignment, image, axes_only=axes_only
                    ).items()
                }
        for panel_id, panel_image in panel_images.items():
            extrema = panel_image.getextrema()
            if not any(low != high for low, high in extrema):
                raise ValueError(f"{figure_id} panel {panel_id} is blank")
            destination = figure_dir / f"{figure_id}_panel_{panel_id}.png"
            panel_image.save(
                destination,
                format="PNG",
                dpi=(300, 300),
                compress_level=9,
                optimize=False,
            )
            key = f"{figure_id}_{panel_id}"
            exports[key] = destination
            manifest_rows.append(
                {
                    "figure_id": figure_id,
                    "panel_id": panel_id,
                    "path": destination.relative_to(output_root).as_posix(),
                    "source_png": artifact.png.name,
                    "width_px": panel_image.width,
                    "height_px": panel_image.height,
                    "aspect_ratio": format(
                        panel_image.width / panel_image.height, ".8f"
                    ),
                    "sha256": _sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )
    manifest = output_root / "PANEL_PNG_MANIFEST.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "figure_id",
                "panel_id",
                "path",
                "source_png",
                "width_px",
                "height_px",
                "aspect_ratio",
                "sha256",
                "bytes",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_root / "README.md").write_text(
        "# AEI figure panels\n\n"
        "These PNG files are aspect-preserving crops of the final 300-dpi figures "
        "for manual composition. They are not part of the submission package.\n\n"
        "Main-panel crops retain their local titles and labels. Shared color bars "
        "remain in the composed figures and are not duplicated into individual "
        "panels. Supplementary Figure S1 crops contain only the correctly "
        "proportioned scan panels; use the panel IDs in the full figure or source "
        "CSV when rebuilding labels.\n",
        encoding="ascii",
        newline="\n",
    )
    return exports
