"""Deterministic figures for the AEI task-relevant acquisition paper."""

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
import numpy as np

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

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
    load_reconstructed_state,
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
        _effect_row(uniform, panel="b", series="Mechanical vs uniform"),
        _effect_row(
            reconstruction_oracle,
            panel="b",
            series="Mechanical vs reconstruction",
        ),
        _effect_row(headroom, panel="b", series="Sequential headroom retained"),
        _effect_row(oracle_cai, panel="b", series="CAI-specific oracle contrast"),
        _effect_row(oracle_image, panel="b", series="Image-specific oracle contrast"),
    ]
    teacher_metric = metrics["O2_TEACHER_TURNOVER"]
    payload = _load_json(root, teacher_metric)
    claim_by_key = {
        "best_action_turnover": metrics["O2_TEACHER_TURNOVER"],
        "rank_spearman": metrics["O2_TEACHER_RANK"],
        "top_k_jaccard": metrics["O2_TEACHER_TOPK"],
    }
    for checkpoint in payload["teacher_by_checkpoint"]:
        cost = float(checkpoint["current_checkpoint"])
        for key, claim in claim_by_key.items():
            rows.append(
                _derived_row(
                    claim,
                    panel="c",
                    series=f"{key}@{format(cost, '.6g')}",
                    value=float(checkpoint[key]),
                    metric_name=key,
                )
            )
    rows.extend(
        (
            _effect_row(
                metrics["O2_TEACHER_OPPORTUNITY"],
                panel="c",
                series="Final-state opportunity",
            ),
            _effect_row(
                metrics["U5_RIDGE_HUBER_SPEARMAN"],
                panel="d",
                series="Ridge-Huber rank agreement",
            ),
            _effect_row(
                metrics["U5_RIDGE_MLP_SPEARMAN"],
                panel="d",
                series="Ridge-MLP rank agreement",
            ),
        )
    )
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
                panel="c",
                series="c8-2 uniform 25 percent reconstruction",
                metric="registered_state_reconstruction",
                source_artifact=("results/mavis/p1_state_bank/state_manifest.parquet"),
            ),
        )
    )
    return rows


def _figure3_rows(root: Path, metrics: dict[str, PaperMetric]) -> list[dict[str, str]]:
    rows = [
        _effect_row(
            metrics["O4_DYNAMIC_MINUS_STATIC"],
            panel="a",
            series="Conditional minus static regret",
        ),
        _effect_row(
            metrics["O1_STATIC_SPEARMAN"], panel="a", series="Static value rank"
        ),
        _effect_row(metrics["O3_REAL_CHANGE"], panel="b", series="Real-state change"),
    ]
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
                panel="b",
                series="Measured content",
                value=float(endpoint["real_equal_domain_mae"]),
                metric_name="endpoint_equal_domain_cai_mae",
            ),
            _derived_row(
                positions,
                panel="b",
                series="Acquired-position/history control",
                value=float(endpoint["control_equal_domain_mae"]),
                metric_name="endpoint_equal_domain_cai_mae",
            ),
            _derived_row(
                reconstruction,
                panel="b",
                series="Reconstruction control",
                value=float(endpoint_reconstruction["control_equal_domain_mae"]),
                metric_name="endpoint_equal_domain_cai_mae",
            ),
            _effect_row(
                positions,
                panel="c",
                series="Measured minus acquired-position/history",
            ),
            _effect_row(
                reconstruction, panel="c", series="Measured minus reconstruction"
            ),
            _effect_row(
                metrics["O4_DYNAMIC_MINUS_SHUFFLED"],
                panel="c",
                series="Conditional minus shuffled regret",
            ),
        )
    )
    rows.extend(
        (
            _source_row(
                root,
                panel="b",
                series="c8-2 initial teacher priority",
                metric="strict_oof_teacher_value_map",
                source_artifact="results/mavis/p3_dynamic_voi/action_scores.parquet",
            ),
            _source_row(
                root,
                panel="c",
                series="c8-2 later teacher priority",
                metric="strict_oof_teacher_value_map",
                source_artifact="results/mavis/p3_dynamic_voi/action_scores.parquet",
            ),
            _source_row(
                root,
                panel="d",
                series="c8-2 acquired-cell path",
                metric="registered_acquisition_history",
                source_artifact=("results/mavis/p1_state_bank/state_manifest.parquet"),
            ),
        )
    )
    return rows


def _figure4_rows(metrics: dict[str, PaperMetric]) -> list[dict[str, str]]:
    return [
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
        _effect_row(
            metrics["A2_GREEDY_PLANNING_REGRET"],
            panel="b",
            series="Current greedy",
        ),
        _effect_row(
            metrics["A2_BEAM4_PLANNING_REGRET"],
            panel="b",
            series="Beam width 4",
        ),
    ]


def _figure5_rows(root: Path, metrics: dict[str, PaperMetric]) -> list[dict[str, str]]:
    return [
        _source_row(
            root,
            panel="a",
            series="c8-2 initial reconstruction",
            metric="registered_state_reconstruction",
            source_artifact="results/mavis/p1_state_bank/state_manifest.parquet",
        ),
        _source_row(
            root,
            panel="b-d",
            series="c8-2 paired task-priority maps",
            metric="paired_oracle_priority_percentile",
            source_artifact="results/mva/a2_oracle_value/oracle_values.parquet",
        ),
        _effect_row(
            metrics["U4_ORACLE_CAI_SPECIFICITY"],
            panel="b",
            series="CAI-specific oracle contrast",
        ),
        _effect_row(
            metrics["U4_ORACLE_IMAGE_SPECIFICITY"],
            panel="c",
            series="Image-specific oracle contrast",
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
        "figure4": _figure4_rows(canonical),
        "figure5": _figure5_rows(root, canonical),
    }
    names = {
        "figure1": "figure1_task_relevant_acquisition_framework.csv",
        "figure2": "figure2_information_characterization.csv",
        "figure3": "figure3_state_conditioned_value.csv",
        "figure4": "figure4_valuation_planning_realization.csv",
        "figure5": "figure5_task_specific_measurement_priorities.csv",
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
    stages = (
        (
            0.03,
            _USEFUL,
            "PART I  INFORMATION CHARACTERIZATION",
            "Spatial structure  |  sparse retention\nobjective and state conditioning",
            "What information is valuable to acquire?",
        ),
        (
            0.54,
            _ACTIONABLE,
            "PART II  STATE-CONDITIONED ACQUISITION",
            "Legal-state value  |  matched controls\ncost-constrained set realization",
            "How is value converted into a measurement set?",
        ),
    )
    for x, color, heading, question, evidence in stages:
        ax.add_patch(
            Rectangle((x, 0.52), 0.43, 0.28, facecolor="white", edgecolor=color, lw=2)
        )
        ax.add_patch(Rectangle((x, 0.74), 0.43, 0.06, facecolor=color, edgecolor=color))
        ax.text(
            x + 0.215,
            0.77,
            heading,
            ha="center",
            va="center",
            color="white",
            fontsize=6.0 if x > 0.5 else 6.6,
            fontweight="bold",
        )
        ax.text(
            x + 0.215,
            0.645,
            question,
            ha="center",
            va="center",
            color=_TEXT,
            fontsize=8,
            linespacing=1.35,
        )
        ax.text(
            x + 0.215,
            0.565,
            evidence,
            ha="center",
            va="center",
            color=_REFERENCE,
            fontsize=6.8,
        )
    ax.annotate(
        "",
        xy=(0.535, 0.66),
        xytext=(0.47, 0.66),
        arrowprops={"arrowstyle": "-|>", "color": _REFERENCE, "lw": 1.4},
    )
    ax.text(
        0.502,
        0.825,
        "state-conditioned transition",
        ha="center",
        fontsize=6.1,
        color=_REFERENCE,
    )
    lanes = (
        (
            0.02,
            0.26,
            "Task-value characterization",
            "registered outcomes and counterfactual\nmeasurement comparisons",
            _USEFUL,
        ),
        (
            0.35,
            0.26,
            "Legal partial state",
            "metadata + acquired positions/content\n+ exact cost",
            _OBSERVABLE,
        ),
        (
            0.68,
            0.26,
            "Cost-aware acquisition rule",
            "measurement-set selection within\nlegal state and exact cost",
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
            y + 0.100,
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
            y + 0.032,
            detail,
            ha="center",
            va="center",
            fontsize=5.3,
            color=_REFERENCE,
            linespacing=1.1,
        )
    ax.text(
        0.5,
        0.095,
        "spatial information  ->  state-conditioned value  ->  cost-constrained sensing",
        ha="center",
        va="center",
        fontsize=7.4,
        color=_USEFUL,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.98)
    return fig


def _render_figure2(rows: list[dict[str, str]]) -> Figure:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.7),
        gridspec_kw={"wspace": 0.38, "hspace": 0.42},
    )
    axes = axes.ravel()
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
    _clean_axis(ax)

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
    oracle_labels = ["Mechanical vs uniform", "Mechanical vs reconstruction"]
    for y, (label, marker) in enumerate(zip(oracle_labels, ("o", "s"), strict=True)):
        row = _find(rows, label)
        value = float(row["value"])
        low = float(row["ci95_lower"])
        high = float(row["ci95_upper"])
        ax.errorbar(
            value,
            y,
            xerr=[[value - low], [high - value]],
            fmt=marker,
            color=_USEFUL,
            ecolor=_USEFUL,
            capsize=3,
            ms=5,
        )
    ax.axvline(0, color=_REFERENCE, lw=0.7)
    ax.set_yticks([0, 1], ["vs uniform", "vs reconstruction"], fontsize=6.5)
    ax.set_xlabel("Mechanical-oracle CAI-AUEBC improvement", fontsize=6.5)
    retained = float(_find(rows, "Sequential headroom retained")["value"])
    ax.text(
        0.98,
        0.08,
        f"{retained * 100:.1f}% of sequential headroom retained",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.2,
        color=_REFERENCE,
    )
    _panel_title(ax, "c", "Spatial heterogeneity")
    _clean_axis(ax)

    ax = axes[3]
    ax.axis("off")
    _panel_title(ax, "d", "Objective conditioning")
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
    ax = axes[1]
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
    dynamic = _find(rows, "Conditional minus static regret")
    ax.set_xlabel(
        f"Exact-budget set regret\nDynamic - static: {float(dynamic['value']):.4f}",
        fontsize=6.5,
    )
    _panel_title(ax, "b", "Static to dynamic")
    _clean_axis(ax)

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
    ax = axes[0]
    metric_specs = (
        ("best_action_turnover", "Turnover", _OBSERVABLE, "s", "--"),
        ("rank_spearman", "Rank agreement", _REFERENCE, "o", "-"),
        ("top_k_jaccard", "Top-5 overlap", _USEFUL, "^", ":"),
    )
    for metric_name, label, color, marker, linestyle in metric_specs:
        selected = [
            row for row in rows if row["panel"] == "a" and row["metric"] == metric_name
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
    _panel_title(ax, "a", "Value evolves with state")
    _clean_axis(ax, grid_axis="y")

    ax = axes[2]
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    _panel_title(ax, "c", "Content controls")
    labels = [
        "Measured content",
        "Acquired-position/history control",
        "Reconstruction control",
    ]
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
    mae_ax.set_xticks([0, 1, 2], ["Real", "History", "Recon."], fontsize=5.8)
    mae_ax.set_ylim(0.08, 0.14)
    mae_ax.set_ylabel("Endpoint CAI MAE", fontsize=6)
    _clean_axis(mae_ax, grid_axis="y")
    inset = ax.inset_axes([0.12, 0.05, 0.84, 0.25])
    controls = [("Conditional minus shuffled regret", _ADVERSE, "x")]
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
    inset.set_yticks([0], ["real - shuffled"], fontsize=5.2)
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
    labels = ["Feedback", "Baseline - learned"]
    _forest(
        ax, rows, labels, scale=10000, colors=[_ADVERSE, _ADVERSE], markers=["x", "D"]
    )
    ax.set_xlabel("CAI-AUEBC effect (×10⁴)", fontsize=6.5)
    _panel_title(ax, "c", "Deployable boundary")
    _clean_axis(ax, grid_axis="x")
    ax.text(
        0.02,
        -0.22,
        "Negative favors no-feedback / reference",
        transform=ax.transAxes,
        fontsize=5.5,
        color=_REFERENCE,
    )
    return fig


def _render_figure2_reframed(rows: list[dict[str, str]]) -> Figure:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.7),
        gridspec_kw={"wspace": 0.42, "hspace": 0.58},
    )
    axes = axes.ravel()

    ax = axes[0]
    labels = ["Matched scalar", "Matched spatial field", "Sparse spatial field"]
    values = [_float(_find(rows, label), "value") for label in labels]
    bars = ax.barh(
        [2, 1, 0],
        values,
        color=[_REFERENCE, _USEFUL, _USEFUL],
        edgecolor=_TEXT,
        linewidth=0.5,
        height=0.58,
    )
    bars[0].set_hatch("//")
    bars[2].set_hatch("..")
    for y, value in zip((2, 1, 0), values, strict=True):
        ax.text(
            value - 0.004,
            y,
            f"{value:.3f}",
            ha="right",
            va="center",
            fontsize=6.5,
            color="white",
            fontweight="bold",
        )
    retention = float(_find(rows, "Registered gain retained")["value"])
    ax.text(
        0.98,
        0.06,
        f"{retention * 100:.1f}% of full-field gain retained",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.2,
        color=_TEXT,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    ax.set_yticks([2, 1, 0], ["Scalar", "Full field", "25% sparse"], fontsize=6.5)
    ax.set_xlim(0, 0.21)
    ax.set_xlabel("Equal-domain CAI-ratio MAE", fontsize=7)
    _panel_title(ax, "a", "Spatial information and sparse recovery")
    _clean_axis(ax, grid_axis="x")

    ax = axes[1]
    ax.axis("off")
    _panel_title(ax, "b", "Task-conditioned spatial value")
    task_rows = (
        ("Mechanical vs uniform", "Mechanical vs uniform", _USEFUL, ".4f"),
        (
            "Mechanical vs reconstruction",
            "Mechanical vs reconstruction",
            _USEFUL,
            ".4f",
        ),
        ("CAI-specific oracle contrast", "CAI task contrast", _OBSERVABLE, ".4f"),
        ("Image-specific oracle contrast", "Image task contrast", _ACTIONABLE, ".6f"),
    )
    for index, (series, label, color, number_format) in enumerate(task_rows):
        row = _find(rows, series)
        y = 0.84 - index * 0.22
        ax.add_patch(
            Rectangle(
                (0.02, y - 0.13),
                0.96,
                0.17,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor=color,
                lw=1.1,
            )
        )
        ax.text(
            0.06,
            y - 0.01,
            label,
            transform=ax.transAxes,
            fontsize=6.8,
            fontweight="bold",
            color=color,
            va="center",
        )
        value = format(float(row["value"]), number_format)
        ax.text(
            0.94,
            y - 0.01,
            value,
            transform=ax.transAxes,
            fontsize=7.2,
            color=_TEXT,
            ha="right",
            va="center",
        )
        ax.text(
            0.94,
            y - 0.08,
            f"95% CI [{format(float(row['ci95_lower']), number_format)}, {format(float(row['ci95_upper']), number_format)}]",
            transform=ax.transAxes,
            fontsize=5.3,
            color=_REFERENCE,
            ha="right",
            va="center",
        )
    headroom = float(_find(rows, "Sequential headroom retained")["value"])
    ax.text(
        0.04,
        -0.09,
        f"One-shot mechanics retains {headroom * 100:.1f}% of sequential headroom; all oracle rows are retrospective.",
        transform=ax.transAxes,
        fontsize=5.8,
        color=_REFERENCE,
        clip_on=False,
    )

    ax = axes[2]
    metric_specs = (
        ("best_action_turnover", "Best-action turnover", _OBSERVABLE, "s", "--"),
        ("rank_spearman", "Rank agreement", _REFERENCE, "o", "-"),
        ("top_k_jaccard", "Top-5 overlap", _USEFUL, "^", ":"),
    )
    for metric_name, label, color, marker, linestyle in metric_specs:
        selected = [
            row for row in rows if row["panel"] == "c" and row["metric"] == metric_name
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
    opportunity = float(_find(rows, "Final-state opportunity")["value"])
    ax.text(
        0.98,
        0.06,
        f"Final opportunity {opportunity:.5f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.0,
        color=_OBSERVABLE,
    )
    ax.set_xlabel("Acquired normalized cost (%)", fontsize=7)
    ax.set_ylabel("Teacher-state diagnostic", fontsize=7)
    ax.set_ylim(0.25, 0.76)
    ax.legend(frameon=False, fontsize=5.6, loc="best", handlelength=2.2)
    _panel_title(ax, "c", "Value changes with evidence")
    _clean_axis(ax, grid_axis="y")

    ax = axes[3]
    labels = ["Ridge-Huber rank agreement", "Ridge-MLP rank agreement"]
    _forest(
        ax,
        rows,
        labels,
        scale=1,
        colors=[_ACTIONABLE, _ADVERSE],
        markers=["o", "s"],
        display_labels=["Ridge-Huber", "Ridge-MLP"],
    )
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Strict-OOF action-value Spearman", fontsize=6.7)
    ax.text(
        0.02,
        0.48,
        "Full-state MAE: Ridge 0.08964 | Huber 0.08618 | MLP 0.15067",
        transform=ax.transAxes,
        fontsize=5.6,
        color=_REFERENCE,
    )
    _panel_title(ax, "d", "Predictor-conditioned value")
    _clean_axis(ax, grid_axis="x")
    return fig


def _render_figure3_reframed(rows: list[dict[str, str]]) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), gridspec_kw={"wspace": 0.58})

    ax = axes[0]
    ax.axis("off")
    _panel_title(ax, "a", "State-conditioned value")

    dynamic = _find(rows, "Conditional minus static regret")
    value = float(dynamic["value"]) * 1000
    low = float(dynamic["ci95_lower"]) * 1000
    high = float(dynamic["ci95_upper"]) * 1000
    dynamic_ax = ax.inset_axes([0.05, 0.58, 0.92, 0.27])
    dynamic_ax.errorbar(
        value,
        0,
        xerr=[[value - low], [high - value]],
        fmt="o",
        color=_OBSERVABLE,
        ecolor=_OBSERVABLE,
        capsize=3,
        ms=5,
    )
    dynamic_ax.axvline(0, color=_REFERENCE, lw=0.7)
    dynamic_ax.set_yticks([0], ["Dynamic - static"], fontsize=5.2)
    dynamic_ax.set_xlim(-2.6, 0.4)
    dynamic_ax.set_xlabel("One-step regret (x10^3)", fontsize=5.4, labelpad=1)
    dynamic_ax.tick_params(labelsize=5.0, length=2)
    dynamic_ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.03,
        0.41,
        "Negative regret favors dynamic valuation",
        transform=ax.transAxes,
        fontsize=5.6,
        color=_OBSERVABLE,
    )

    rank = _find(rows, "Static value rank")
    rvalue = float(rank["value"])
    rlow = float(rank["ci95_lower"])
    rhigh = float(rank["ci95_upper"])
    rank_ax = ax.inset_axes([0.05, 0.12, 0.92, 0.24])
    rank_ax.errorbar(
        rvalue,
        0,
        xerr=[[rvalue - rlow], [rhigh - rvalue]],
        fmt="s",
        color=_REFERENCE,
        ecolor=_UNCERTAINTY,
        capsize=3,
        ms=4,
    )
    rank_ax.axvline(0, color=_REFERENCE, lw=0.7)
    rank_ax.set_xlim(-0.12, 0.12)
    rank_ax.set_yticks([0], ["Static scorer"], fontsize=5.2)
    rank_ax.set_xlabel("Action-value Spearman", fontsize=5.4, labelpad=1)
    rank_ax.tick_params(labelsize=5.0, length=2)
    rank_ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    labels = [
        "Measured content",
        "Acquired-position/history control",
        "Reconstruction control",
    ]
    values = [float(_find(rows, label)["value"]) for label in labels]
    bars = ax.bar(
        [0, 1, 2],
        values,
        color=[_OBSERVABLE, _REFERENCE, _REFERENCE],
        edgecolor=_TEXT,
        linewidth=0.5,
        width=0.66,
    )
    bars[1].set_hatch("//")
    bars[2].set_hatch("..")
    ax.set_xticks([0, 1, 2], ["Real", "History", "Recon."], fontsize=5.9)
    ax.set_ylim(0.08, 0.14)
    ax.set_ylabel("Endpoint CAI MAE", fontsize=6.5)
    change = _find(rows, "Real-state change")
    ax.text(
        0.98,
        0.95,
        f"Real-state change {float(change['value']):.6f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=_OBSERVABLE,
    )
    _panel_title(ax, "b", "Matched source controls")
    _clean_axis(ax, grid_axis="y")

    ax = axes[2]
    labels = [
        "Measured minus acquired-position/history",
        "Measured minus reconstruction",
    ]
    _forest(
        ax,
        rows,
        labels,
        scale=1,
        colors=[_ADVERSE, _ADVERSE],
        markers=["o", "s"],
        display_labels=["Real - history", "Real - recon."],
    )
    ax.axvline(0, color=_REFERENCE, lw=0.7)
    ax.set_xlabel("Endpoint CAI-MAE contrast", fontsize=6.5)
    shuffled = _find(rows, "Conditional minus shuffled regret")
    ax.text(
        0.03,
        -0.23,
        f"Real - shuffled regret: {float(shuffled['value']):.6f}\n95% CI [{float(shuffled['ci95_lower']):.6f}, {float(shuffled['ci95_upper']):.6f}]",
        transform=ax.transAxes,
        fontsize=5.4,
        color=_ADVERSE,
    )
    _panel_title(ax, "c", "Attribution boundary")
    _clean_axis(ax, grid_axis="x")
    return fig


def _render_figure4_reframed(rows: list[dict[str, str]]) -> Figure:
    fig, axes = plt.subplots(
        1, 2, figsize=(_FULL_WIDTH_IN, 2.9), gridspec_kw={"wspace": 0.48}
    )
    ax = axes[0]
    labels = ["Valuation", "Bounded learned planning", "True-value stronger planning"]
    _forest(
        ax,
        rows,
        labels,
        scale=10000,
        colors=[_USEFUL, _ACTIONABLE, _USEFUL],
        markers=["o", "s", "D"],
        display_labels=["Valuation", "Learned planning", "True-value planning"],
    )
    ax.set_xlabel("Retrospective CAI-AUEBC improvement (x10^4)", fontsize=6.8)
    _panel_title(ax, "a", "Valuation and planning substitutions")
    _clean_axis(ax, grid_axis="x")

    ax = axes[1]
    labels = ["Current greedy", "Beam width 4"]
    _forest(
        ax,
        rows,
        labels,
        scale=10000,
        colors=[_REFERENCE, _ACTIONABLE],
        markers=["o", "D"],
        display_labels=["Current greedy", "Beam width 4"],
    )
    ax.set_xlabel("Reachable-set planning regret (x10^4)", fontsize=6.8)
    ax.text(
        0.08,
        0.48,
        "Reference: retrospective\njoint near-oracle set",
        transform=ax.transAxes,
        fontsize=5.5,
        color=_REFERENCE,
    )
    _panel_title(ax, "b", "Cost-constrained set realization")
    _clean_axis(ax)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.22, top=0.88, wspace=0.52)
    _attach_alignment_contract(
        fig,
        list(axes),
        ["a", "b"],
        row_groups=[["a", "b"]],
    )
    return fig


def _render_figure2_nature(root: Path, rows: list[dict[str, str]]) -> Figure:
    initial = load_reconstructed_state(
        root,
        specimen_id=REPRESENTATIVE_SPECIMEN,
        method=REPRESENTATIVE_METHOD,
        checkpoint=INITIAL_CHECKPOINT,
    )
    sparse = load_reconstructed_state(
        root,
        specimen_id=REPRESENTATIVE_SPECIMEN,
        method="uniform",
        checkpoint=0.25,
    )
    fig, axes_grid = plt.subplots(2, 3, figsize=(_FULL_WIDTH_IN, 5.05))
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
    retention = float(_find(rows, "Registered gain retained")["value"])
    ax.text(
        0.98,
        0.08,
        f"{100 * retention:.1f}% full-field gain retained",
        transform=ax.transAxes,
        ha="right",
        fontsize=5.8,
        color=_OBSERVABLE,
        fontweight="bold",
    )
    ax.set_yticks(y, ["Scalar", "Full field", "25% sparse"], fontsize=6.1)
    ax.set_xlim(0.118, 0.208)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("Equal-domain CAI-ratio MAE", fontsize=6.6)
    _panel_title(ax, "a", "Spatial gain and retention")
    _clean_axis(ax)

    ax = axes[1]
    _draw_reconstruction(ax, initial, show_measurements=True)
    ax.text(
        0.03,
        0.04,
        f"{initial.exact_acquired_cost:,} measured pixels",
        transform=ax.transAxes,
        color=_PAPER_WHITE,
        fontsize=5.8,
        fontweight="bold",
    )
    _panel_title(ax, "b", "Initial scout | 3.13%")

    ax = axes[2]
    _draw_reconstruction(ax, sparse, show_measurements=True)
    ax.text(
        0.03,
        0.04,
        f"{sparse.exact_acquired_cost:,} measured pixels",
        transform=ax.transAxes,
        color=_PAPER_WHITE,
        fontsize=5.8,
        fontweight="bold",
    )
    _panel_title(ax, "c", "Uniform sparse | 25%")

    ax = axes[3]
    task_labels = ["Mechanical vs uniform", "Mechanical vs reconstruction"]
    _forest(
        ax,
        rows,
        task_labels,
        scale=1000,
        colors=[_USEFUL, _USEFUL],
        markers=["o", "s"],
        display_labels=["Uniform", "Reconstruction"],
    )
    ax.set_xlim(2.2, 5.5)
    ax.set_ylim(-0.8, 1.6)
    ax.set_xlabel("CAI-AUEBC improvement (x10^3)", fontsize=6.4)
    cai = float(_find(rows, "CAI-specific oracle contrast")["value"])
    image = float(_find(rows, "Image-specific oracle contrast")["value"])
    ax.text(
        0.02,
        0.06,
        f"Cross-objective\nCAI {cai:.5f} | RGB MSE {image:.3e}",
        transform=ax.transAxes,
        fontsize=5.25,
        color=_REFERENCE,
    )
    _panel_title(ax, "d", "Task-conditioned opportunity")
    _clean_axis(ax)

    ax = axes[4]
    metric_specs = (
        ("best_action_turnover", "Best-action turnover", _OBSERVABLE, "s", "--"),
        ("rank_spearman", "Rank agreement", _REFERENCE, "o", "-"),
        ("top_k_jaccard", "Top-5 overlap", _USEFUL, "^", ":"),
    )
    direct_labels = {
        "best_action_turnover": "Turnover",
        "rank_spearman": "Rank",
        "top_k_jaccard": "Top-5",
    }
    for metric_name, label, color, marker, linestyle in metric_specs:
        selected = [
            row for row in rows if row["panel"] == "c" and row["metric"] == metric_name
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
        ax.text(
            x[-1] + 0.45,
            values[-1],
            direct_labels[metric_name],
            color=color,
            fontsize=5.1,
            va="center",
        )
    ax.set_xlabel("Acquired normalized cost (%)", fontsize=6.5)
    ax.set_ylabel("Teacher-state diagnostic", fontsize=6.5)
    ax.set_ylim(0.25, 0.76)
    ax.set_xlim(5.5, 23.5)
    _panel_title(ax, "e", "Value evolves with state")
    _clean_axis(ax)

    ax = axes[5]
    _forest(
        ax,
        rows,
        ["Ridge-Huber rank agreement", "Ridge-MLP rank agreement"],
        scale=1,
        colors=[_ACTIONABLE, _ADVERSE],
        markers=["o", "s"],
        display_labels=["Ridge-Huber", "Ridge-MLP"],
    )
    ax.set_xlim(0, 0.9)
    ax.set_ylim(-0.65, 1.55)
    ax.set_xlabel("Strict-OOF action-value Spearman", fontsize=6.3)
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
    dynamic = _find(rows, "Conditional minus static regret")
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
    ax.set_yticks([0], ["Dynamic - static"], fontsize=6.0)
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
    _panel_title(ax, "a", "Dynamic versus static")
    _clean_axis(ax)

    ax = axes[1]
    image = _draw_priority_overlay(ax, initial)
    _outline_top_cells(ax, initial.reconstruction, initial.percentiles)
    _panel_title(ax, "b", "Initial priority | 3.13%")

    ax = axes[2]
    _draw_priority_overlay(ax, later)
    _outline_top_cells(ax, later.reconstruction, later.percentiles)
    _panel_title(ax, "c", "Updated priority | 18.75%")

    ax = axes[3]
    _draw_acquisition_path(ax, later.reconstruction)
    _panel_title(ax, "d", "Acquisition history")

    ax = axes[4]
    labels = [
        "Measured content",
        "Acquired-position/history control",
        "Reconstruction control",
    ]
    values = [float(_find(rows, label)["value"]) for label in labels]
    ax.bar(
        [0, 1, 2],
        values,
        color=[_USEFUL, _REFERENCE, _UNCERTAINTY],
        edgecolor="none",
        linewidth=0,
        width=0.62,
    )
    ax.set_xticks([0, 1, 2], ["Real", "History", "Recon."], fontsize=5.8)
    ax.set_ylim(0.08, 0.14)
    ax.set_ylabel("Endpoint CAI MAE", fontsize=6.4)
    _panel_title(ax, "e", "Matched state controls")
    _clean_axis(ax, grid_axis="y")

    ax = axes[5]
    _forest(
        ax,
        rows,
        [
            "Measured minus acquired-position/history",
            "Measured minus reconstruction",
        ],
        scale=1,
        colors=[_ADVERSE, _ADVERSE],
        markers=["o", "s"],
        display_labels=["Real - history", "Real - recon."],
    )
    ax.set_xlim(-0.005, 0.05)
    ax.set_ylim(-0.75, 1.55)
    ax.set_xlabel("Endpoint CAI-MAE contrast", fontsize=6.3)
    shuffled = _find(rows, "Conditional minus shuffled regret")
    ax.text(
        0.12,
        0.05,
        f"Real - shuffled {float(shuffled['value']):.3e}\n"
        f"95% CI [{float(shuffled['ci95_lower']):.3e}, {float(shuffled['ci95_upper']):.3e}]",
        transform=ax.transAxes,
        fontsize=5.1,
        color=_ADVERSE,
    )
    _panel_title(ax, "f", "Source-control contrasts")
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


def _render_figure5_nature(root: Path, rows: list[dict[str, str]]) -> Figure:
    del rows
    priorities = load_task_priority_maps(root, specimen_id=REPRESENTATIVE_SPECIMEN)
    state = priorities.reconstruction
    fig, axes_grid = plt.subplots(1, 4, figsize=(_FULL_WIDTH_IN, 2.65))
    axes = list(axes_grid.ravel())

    ax = axes[0]
    _draw_reconstruction(ax, state, show_measurements=True)
    _panel_title(ax, "a", "Initial state")

    ax = axes[1]
    image = _draw_priority_overlay(ax, state, priorities.mechanical_percentiles)
    _outline_top_cells(ax, state, priorities.mechanical_percentiles)
    _panel_title(ax, "b", "CAI priority")

    ax = axes[2]
    _draw_priority_overlay(ax, state, priorities.reconstruction_percentiles)
    _outline_top_cells(ax, state, priorities.reconstruction_percentiles)
    _panel_title(ax, "c", "RGB priority")

    ax = axes[3]
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
    _style_image_axis(ax)
    _panel_title(ax, "d", "CAI - RGB priority")

    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.22, top=0.88, wspace=0.12)
    value_axis = fig.add_axes([0.295, 0.105, 0.39, 0.018])
    value_bar = fig.colorbar(image, cax=value_axis, orientation="horizontal")
    value_bar.set_label("Within-map priority percentile", fontsize=5.4, labelpad=1)
    value_bar.ax.tick_params(labelsize=5.0, length=1.5)
    difference_axis = fig.add_axes([0.79, 0.105, 0.18, 0.018])
    difference_bar = fig.colorbar(
        difference, cax=difference_axis, orientation="horizontal"
    )
    difference_bar.set_label("Task-priority difference", fontsize=5.4, labelpad=1)
    difference_bar.ax.tick_params(labelsize=5.0, length=1.5)
    _attach_alignment_contract(
        fig,
        axes,
        list("abcd"),
        row_groups=[["a", "b", "c", "d"]],
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
        "**Figure 2. Spatial task-relevant ultrasonic information and sparse "
        "retention.** (a) Equal-domain CAI-ratio MAE for matched scalar, full-field, "
        "and registered 25% sparse representations; the sparse representation retains "
        "89.9% of the full-field gain. (b,c) Hash-verified reconstructions of specimen "
        "c8-2 at the 3.13% initial scout state and 25% uniform state; teal pixels mark "
        "measured native-raster positions. (d) Retrospective exact-cost mechanical "
        "oracle improvements over uniform and reconstruction acquisition. (e) "
        "Strict-OOF teacher priorities evolve over normalized cost. (f) Value-rank "
        "agreement varies with the downstream predictor. Aggregate intervals are "
        "synchronized specimen-bootstrap contrasts where shown.\n"
    ),
    "figure3": (
        "**Figure 3. State-conditioned measurement value evolves with acquired "
        "evidence.** (a) Dynamic real-state valuation has lower one-step regret than "
        "the static reference at 18.75%. (b,c) Within-state percentiles of strict-OOF "
        "teacher values for all 64 legal next-cell actions on specimen c8-2 at 3.13% "
        "and 18.75%; white outlines mark the five highest-priority cells. (d) Stored "
        "acquired-cell history for the same trajectory, from the initial white marker "
        "to the latest red marker. (e) Endpoint CAI MAE under "
        "real, acquired-position/history, and reconstruction states. (f) Matched source "
        "contrasts retain their observed adverse directions. Priority maps are "
        "predictor- and state-conditioned, not universal material maps.\n"
    ),
    "figure4": (
        "**Figure 4. Valuation, planning, and set realization.** (a) Retrospective "
        "substitutions expose valuation and bounded planning effects. (b) Greedy and "
        "beam-width-four selection retain positive regret relative to the retrospective "
        "joint set within the registered two-action reachable pool.\n"
    ),
    "figure5": (
        "**Figure 5. Task-specific measurement priorities on a CFRP specimen.** "
        "(a) Hash-verified initial reconstruction of specimen c8-2; teal marks the "
        "registered measured positions. (b,c) Retrospective CAI-mechanical and "
        "normalized-RGB-reconstruction oracle priorities on the same initial state and "
        "8x8 action grid; white outlines mark the five highest-priority cells. (d) "
        "Paired within-map percentile difference, with red favoring CAI priority and "
        "blue favoring reconstruction priority. Across all six domains, the CAI "
        "contrast is 0.04862 (95% CI 0.04527-0.05205) and the registered normalized-"
        "RGB-MSE contrast is 5.503e-4 (95% CI 5.006e-4-6.063e-4). These panels "
        "visualize retrospective oracle opportunity, not learned-policy performance.\n"
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
    "figure5": "figure5_task_specific_measurement_priorities",
    "supplementary_figure_s1": (
        "supplementary_figure_s1_cross_domain_state_priority_gallery"
    ),
}


_SOURCE_NAMES = {
    "figure1": "figure1_task_relevant_acquisition_framework.csv",
    "figure2": "figure2_information_characterization.csv",
    "figure3": "figure3_state_conditioned_value.csv",
    "figure4": "figure4_valuation_planning_realization.csv",
    "figure5": "figure5_task_specific_measurement_priorities.csv",
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
    """Render five deterministic main-paper figures and their captions."""
    output_root.mkdir(parents=True, exist_ok=True)
    sources = build_figure_sources(root, output_root)
    renderers = {
        "figure1": _render_figure1,
        "figure2": lambda rows: _render_figure2_nature(root, rows),
        "figure3": lambda rows: _render_figure3_nature(root, rows),
        "figure4": _render_figure4_reframed,
        "figure5": lambda rows: _render_figure5_nature(root, rows),
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
