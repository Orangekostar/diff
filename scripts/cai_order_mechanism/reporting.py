"""Offline figures, report, and manuscript-facing artifacts."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Rectangle

from scripts.cai_order_mechanism.inputs import (
    TaskContext,
    atomic_json,
    atomic_text,
    read_csv_rows,
    read_gzip_csv_rows,
)

COLORS = {
    "native": "#006D77",
    "reverse": "#D1495B",
    "permuted": "#7A5195",
    "raw": "#5B6573",
    "accent": "#E09F3E",
    "ink": "#20242A",
    "grid": "#D7DCE2",
}
CASE_KEYS = (
    "74t7kcdgkr:c8-16",
    "cgtnjyggtm:q24-48",
    "w68dtmpfyf:q16-29",
)


def _float(row: Mapping[str, str], name: str) -> float:
    return float(row[name])


def _values(text: str, cast: type = float) -> np.ndarray:
    return np.asarray([cast(value) for value in text.split(";") if value != ""])


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Noto Sans CJK SC", "Arial"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _alignment_qa(fig: plt.Figure, qa_root: Path, stem: str) -> Path:
    qa_root.mkdir(parents=True, exist_ok=True)
    output = qa_root / f"{stem}.alignment.json"
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        return atomic_json(
            output,
            {
                "status": "NOT_APPLICABLE",
                "reason": "single-axis figure",
                "axes_count": len(fig.axes),
            },
        )
    require_matplotlib_panel_alignment(
        fig,
        json_out=output,
        overlay_svg=qa_root / f"{stem}.alignment.svg",
        strict=True,
    )
    return output


def _save_figure(fig: plt.Figure, root: Path, qa_root: Path, stem: str) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    qa = _alignment_qa(fig, qa_root, stem)
    png = root / f"{stem}.png"
    svg = root / f"{stem}.svg"
    pdf = root / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return [qa, png, svg, pdf]


def _archived_stage_figure(results: Path, figures: Path, qa: Path) -> list[Path]:
    rows = read_csv_rows(results / "stage_paired_contrasts.csv")
    controls = list(dict.fromkeys(row["comparator"] for row in rows))
    labels = {
        row["comparator"]: row["comparator_display_label"] for row in rows
    }
    palette = ["#006D77", "#D1495B", "#7A5195", "#3D5A80", "#E09F3E"]
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    for index, control in enumerate(controls):
        for metric, linestyle, marker, alpha in (
            ("weighted_net_mpa", "-", "o", 1.0),
            ("raw_net_mpa", "--", "x", 0.72),
        ):
            local = [
                row
                for row in rows
                if row["comparator"] == control and row["metric"] == metric
            ]
            local.sort(key=lambda row: int(row["stage"]))
            ax.plot(
                [int(row["stage"]) for row in local],
                [_float(row, "estimate_mpa") for row in local],
                color=palette[index],
                linestyle=linestyle,
                marker=marker,
                linewidth=1.25,
                markersize=4,
                alpha=alpha,
                label=(labels[control] if metric == "weighted_net_mpa" else None),
            )
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xticks([1, 2, 3, 4], ["(0,.0625]", "(.0625,.125]", "(.125,.1875]", "(.1875,.25]"])
    ax.set_xlabel("Completed-cost stage")
    ax.set_ylabel("Proposed - control contribution (MPa)")
    ax.set_title("Archived trajectories: when the proposed policy changes error")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    return _save_figure(fig, figures, qa, "F1_archived_stage_contrasts")


def _fixed_curve_figure(results: Path, figures: Path, qa: Path) -> list[Path]:
    rows = [
        row
        for row in read_gzip_csv_rows(results / "cost_error_curves.csv.gz")
        if row["identity_family"] == "FIXED_SET_ORDERS"
        and row["grid_type"] == "FULL_EVENT_UNION"
    ]
    styles = {
        "NATIVE_REPLAY": (COLORS["native"], "Native order", "-"),
        "REVERSE": (COLORS["reverse"], "Reverse order", "--"),
        "PERMUTED_MEAN": (COLORS["permuted"], "Permuted mean", "-."),
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    for series, (color, label, line) in styles.items():
        local = [row for row in rows if row["series"] == series]
        local.sort(key=lambda row: _float(row, "requested_cost"))
        ax.step(
            [_float(row, "requested_cost") for row in local],
            [_float(row, "mae_mpa") for row in local],
            where="post",
            color=color,
            linestyle=line,
            linewidth=1.45,
            label=label,
        )
    for cap in (0.0625, 0.125, 0.1875, 0.25):
        ax.axvline(cap, color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.axhline(41.69001007080078, color=COLORS["accent"], linewidth=0.9, linestyle=":")
    ax.set_xlim(0.0, 0.25)
    ax.set_xlabel("Requested acquisition cost")
    ax.set_ylabel("Cohort MAE (MPa)")
    ax.set_title("Fixed final set: prediction error depends on acquisition order")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    for x, label, color in (
        (0.085, "Native order", COLORS["native"]),
        (0.145, "Reverse order", COLORS["reverse"]),
        (0.210, "Permuted mean", COLORS["permuted"]),
    ):
        ax.text(x, 58.35, label, color=color, fontsize=7, ha="center", va="center")
    return _save_figure(fig, figures, qa, "F2_fixed_order_curves")


def _order_effect_figure(results: Path, figures: Path, qa: Path) -> list[Path]:
    rows = read_csv_rows(results / "order_specimen_metrics.csv")
    selected = {
        (row["variant"], row["specimen_key"]): row
        for row in rows
        if row["variant"] in {"NATIVE_REPLAY", "REVERSE", "PERMUTED_MEAN"}
    }
    keys = sorted({key for _variant, key in selected})
    effects = {
        "PERMUTED_MEAN": np.asarray(
            [
                _float(selected[("PERMUTED_MEAN", key)], "area_mpa")
                - _float(selected[("NATIVE_REPLAY", key)], "area_mpa")
                for key in keys
            ]
        ),
        "REVERSE": np.asarray(
            [
                _float(selected[("REVERSE", key)], "area_mpa")
                - _float(selected[("NATIVE_REPLAY", key)], "area_mpa")
                for key in keys
            ]
        ),
    }
    contrast_rows = read_csv_rows(results / "order_paired_contrasts.csv")
    summary = {
        row["contrast"]: row
        for row in contrast_rows
        if row["metric"] == "area_mpa"
    }
    specs = (
        ("PERMUTED_MEAN", "PRIMARY_PERMUTED_MEAN_MINUS_NATIVE", "Permuted mean - native", COLORS["permuted"]),
        ("REVERSE", "SECONDARY_REVERSE_MINUS_NATIVE", "Reverse - native", COLORS["reverse"]),
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    for index, (variant, contrast, label, color) in enumerate(specs, start=1):
        jitter = np.linspace(-0.16, 0.16, len(effects[variant]))
        ax.scatter(
            index + jitter,
            effects[variant],
            s=15,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.58,
            zorder=2,
        )
        row = summary[contrast]
        estimate = _float(row, "estimate_mpa")
        low = _float(row, "ci_low_mpa")
        high = _float(row, "ci_high_mpa")
        ax.errorbar(
            index,
            estimate,
            yerr=[[estimate - low], [high - estimate]],
            fmt="D",
            color=COLORS["ink"],
            markerfacecolor="white",
            markersize=5,
            linewidth=1.3,
            capsize=3,
            zorder=4,
        )
        ax.text(index + 0.2, estimate, f"{estimate:+.2f} [{low:+.2f}, {high:+.2f}]", va="center", fontsize=7)
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.7)
    ax.set_xticks([1, 2], [spec[2] for spec in specs])
    ax.set_xlim(0.5, 2.8)
    ax.set_ylabel("Within-specimen area difference (MPa)")
    ax.set_title("Order intervention: specimen effects and paired 95% bootstrap intervals")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax.text(0.01, 0.02, "positive values favor native order", transform=ax.transAxes, fontsize=7, color=COLORS["raw"])
    return _save_figure(fig, figures, qa, "F3_order_effects")


def _matched_quality_figure(results: Path, figures: Path, qa: Path) -> list[Path]:
    rows = [
        row
        for row in read_csv_rows(results / "matched_quality.csv")
        if row["identity_family"] == "FIXED_SET_ORDERS"
        and row["grid_type"] == "FULL_EVENT_UNION"
    ]
    specs = (
        ("REVERSE", COLORS["reverse"], "Reverse vs native"),
        ("PERMUTED_MEAN", COLORS["permuted"], "Permuted mean vs native"),
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    for comparator, color, label in specs:
        local = [row for row in rows if row["comparator_series"] == comparator]
        defined = [row for row in local if row["saving_status"] == "DEFINED"]
        missing = [row for row in local if row["saving_status"] != "DEFINED"]
        ax.plot(
            [_float(row, "target_mpa") for row in defined],
            [_float(row, "absolute_saving") for row in defined],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label=label,
        )
        ax.scatter(
            [_float(row, "target_mpa") for row in missing],
            [-0.012] * len(missing),
            marker="x",
            s=20,
            color=color,
        )
    ax.axhline(0.0, color=COLORS["ink"], linewidth=0.7)
    ax.axvline(41.69001007080078, color=COLORS["accent"], linewidth=0.9, linestyle=":")
    ax.set_xlabel("Target cohort MAE (MPa)")
    ax.set_ylabel("Comparator cost - native cost")
    ax.set_title("Matched-quality acquisition savings preserve crossings and unreached targets")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    ax.legend(frameon=False)
    return _save_figure(fig, figures, qa, "F4_matched_quality")


def _draw_order_grid(
    ax: plt.Axes, cells: Sequence[int], x0: float, y0: float, size: float, title: str
) -> None:
    rank = {cell: index + 1 for index, cell in enumerate(cells)}
    norm = colors.Normalize(vmin=1, vmax=max(rank.values()))
    cmap = matplotlib.colormaps["viridis"]
    cell_size = size / 8.0
    for cell in range(64):
        row, column = divmod(cell, 8)
        x = x0 + column * cell_size
        y = y0 + (7 - row) * cell_size
        if cell in rank:
            face = cmap(norm(rank[cell]))
            label = str(rank[cell])
            text_color = "white" if rank[cell] < 10 else COLORS["ink"]
        else:
            face = "#F2F4F6"
            label = ""
            text_color = COLORS["ink"]
        ax.add_patch(Rectangle((x, y), cell_size, cell_size, facecolor=face, edgecolor="white", linewidth=0.45))
        if label:
            ax.text(x + cell_size / 2, y + cell_size / 2, label, ha="center", va="center", fontsize=5.2, color=text_color)
    ax.add_patch(Rectangle((x0, y0), size, size, fill=False, edgecolor=COLORS["ink"], linewidth=0.7))
    ax.text(x0 + size / 2, y0 + size + 0.018, title, ha="center", va="bottom", fontsize=7)


def _case_figure(results: Path, cases: Path, qa: Path, specimen_key: str) -> list[Path]:
    rows = [
        row
        for row in read_gzip_csv_rows(results / "reorder_trajectories.csv.gz")
        if row["specimen_key"] == specimen_key
    ]
    lookup = {(row["variant"], int(row["repeat"])): row for row in rows}
    selected = [
        ("NATIVE_REPLAY", 0, "Native", COLORS["native"]),
        ("REVERSE", 0, "Reverse", COLORS["reverse"]),
        ("PERMUTED", 0, "Preset permutation 0", COLORS["permuted"]),
    ]
    if any((variant, repeat) not in lookup for variant, repeat, _label, _color in selected):
        raise ValueError(f"fixed case is incomplete: {specimen_key}")
    fig, ax = plt.subplots(figsize=(7.2, 5.1), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    for (variant, repeat, label, _color), x0 in zip(selected, (0.07, 0.38, 0.69), strict=True):
        _draw_order_grid(
            ax,
            [int(value) for value in _values(lookup[(variant, repeat)]["cells"], int)],
            x0,
            0.56,
            0.24,
            label,
        )
    predictions = [
        value
        for variant, repeat, _label, _color in selected
        for value in _values(lookup[(variant, repeat)]["predictions_mpa"])
    ]
    target = _float(rows[0], "target_mpa")
    y_min = min(min(predictions), target)
    y_max = max(max(predictions), target)
    margin = max(1.0, (y_max - y_min) * 0.08)
    y_min -= margin
    y_max += margin
    left, right, bottom, top = 0.09, 0.96, 0.10, 0.43
    ax.add_patch(Rectangle((left, bottom), right - left, top - bottom, fill=False, edgecolor=COLORS["ink"], linewidth=0.7))
    for fraction in (0.0, 0.0625, 0.125, 0.1875, 0.25):
        x = left + (right - left) * fraction / 0.25
        ax.plot([x, x], [bottom, top], color=COLORS["grid"], linewidth=0.45, zorder=0)
        ax.text(x, bottom - 0.025, f"{fraction:.4g}", ha="center", va="top", fontsize=6.2)
    for variant, repeat, label, color in selected:
        row = lookup[(variant, repeat)]
        x = left + (right - left) * _values(row["costs"]) / 0.25
        y = bottom + (top - bottom) * (_values(row["predictions_mpa"]) - y_min) / (y_max - y_min)
        ax.plot(x, y, color=color, linewidth=1.25, marker="o", markersize=2.3, label=label)
    target_y = bottom + (top - bottom) * (target - y_min) / (y_max - y_min)
    ax.plot([left, right], [target_y, target_y], color=COLORS["accent"], linestyle=":", linewidth=1.0)
    for value in np.linspace(y_min, y_max, 4):
        y = bottom + (top - bottom) * (value - y_min) / (y_max - y_min)
        ax.text(left - 0.012, y, f"{value:.0f}", ha="right", va="center", fontsize=6.2)
    ax.text((left + right) / 2, bottom - 0.065, "Acquisition cost", ha="center", fontsize=7)
    ax.text(0.012, (bottom + top) / 2, "Prediction (MPa)", rotation=90, rotation_mode="anchor", va="center", fontsize=7)
    ax.text(0.02, 0.98, specimen_key, ha="left", va="top", fontsize=9, weight="bold")
    ax.text(0.98, 0.98, f"target {target:.2f} MPa; identical final set", ha="right", va="top", fontsize=7, color=COLORS["raw"])
    for x, (_variant, _repeat, label, color) in zip(
        (0.33, 0.52, 0.74), selected, strict=True
    ):
        ax.text(x, 0.495, label, color=color, ha="center", va="center", fontsize=7)
    stem = specimen_key.replace(":", "_")
    return _save_figure(fig, cases, qa, stem)


def _table(rows: Sequence[Mapping[str, str]], fields: Sequence[tuple[str, str]], limit: int | None = None) -> str:
    shown = rows if limit is None else rows[:limit]
    head = "".join(f"<th>{html.escape(label)}</th>" for _field, label in fields)
    body = []
    for row in shown:
        cells = []
        for field, _label in fields:
            value = row.get(field, "")
            try:
                value = f"{float(value):.3f}"
            except (TypeError, ValueError):
                pass
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _render_html(results: Path, artifacts: Path) -> Path:
    summary = read_csv_rows(results / "order_summary.csv")
    contrasts = [
        row
        for row in read_csv_rows(results / "order_paired_contrasts.csv")
        if row["metric"] in {"area_mpa", "early_area_mpa", "final_error_mpa"}
    ]
    stage = [
        row
        for row in read_csv_rows(results / "stage_paired_contrasts.csv")
        if row["metric"] == "weighted_net_mpa"
    ]
    resource = json.loads((results / "resource_usage.json").read_text(encoding="utf-8"))
    figure_cards = "".join(
        f"<figure><img src='figures/{stem}.svg' alt='{alt}' loading='lazy'><figcaption>{caption}</figcaption></figure>"
        for stem, alt, caption in (
            ("F2_fixed_order_curves", "Three fixed-set order MAE curves over acquisition cost", "Fixed final set, different order trajectories."),
            ("F3_order_effects", "Paired specimen area effects with bootstrap intervals", "Primary and secondary paired effects."),
            ("F4_matched_quality", "Matched-quality cost savings by error target", "First crossings retain recrossings and missing targets."),
        )
    )
    archived_card = (
        "<figure><img src='figures/F1_archived_stage_contrasts.svg' "
        "alt='Stage-wise archived contribution contrasts'><figcaption>Archived-only timing decomposition; no new model forwards.</figcaption></figure>"
    )
    case_cards = "".join(
        f"<figure><img src='cases/{key.replace(':', '_')}.svg' alt='Order grids and trajectories for {key}' loading='lazy'><figcaption>{key}</figcaption></figure>"
        for key in CASE_KEYS
    )
    links = " ".join(
        f"<a href='{name}'>{label}</a>"
        for name, label in (
            ("order_summary.csv", "order summary"),
            ("order_paired_contrasts.csv", "paired contrasts"),
            ("stage_paired_contrasts.csv", "stage contrasts"),
            ("matched_quality.csv", "matched quality"),
            ("resource_usage.json", "resource ledger"),
            ("release_manifest.json", "release manifest"),
        )
    )
    artifact_links = " ".join(
        f"<a href='../../../../artifacts/cai_agent_v3/order_mechanism/r1_37b3c404/{name}'>{label}</a>"
        for name, label in (
            ("FINDINGS_ZH.md", "中文结论"),
            ("METHODS_FACTS.md", "methods facts"),
            ("FIGURE_CAPTIONS.md", "captions"),
            ("REPRODUCE.md", "reproduce"),
        )
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAI order-mechanism analysis</title>
<style>
:root{{--ink:#20242a;--muted:#5b6573;--line:#d7dce2;--paper:#fff;--band:#f3f6f7;--a:#006d77;--b:#d1495b}}*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--paper);font:14px/1.55 system-ui,sans-serif}}header{{border-bottom:1px solid var(--line);background:var(--band)}}.wrap{{max-width:1120px;margin:auto;padding:24px}}h1{{font-size:26px;margin:0 0 7px}}h2{{font-size:18px;margin:25px 0 10px}}p{{max-width:78ch}}.meta{{color:var(--muted)}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}}button{{border:1px solid var(--line);background:white;padding:8px 12px;cursor:pointer}}button[aria-selected=true]{{background:var(--ink);color:white}}[data-panel]{{display:none}}[data-panel].active{{display:block}}.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}}.kpis div{{background:white;padding:14px}}.kpis strong{{display:block;font-size:19px}}.plots{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}figure{{margin:0;border-top:1px solid var(--line);padding-top:10px}}img{{width:100%;height:auto;display:block}}figcaption{{color:var(--muted);font-size:12px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid var(--line);padding:7px;text-align:left}}th{{background:var(--band)}}a{{color:var(--a)}}footer{{border-top:1px solid var(--line);margin-top:30px}}@media(max-width:760px){{.kpis,.plots{{grid-template-columns:1fr}}.wrap{{padding:16px}}}}
</style></head><body><header><div class="wrap"><h1>CAI order-mechanism analysis</h1><div class="meta">Frozen VALID50 cohort · source 37b3c404 · fixed predictor MEAN_SC update 1750</div><nav aria-label="Analysis track"><button type="button" data-target="fixed" aria-selected="true">Track B · fixed final set</button><button type="button" data-target="archived" aria-selected="false">Track A · archived timing</button></nav></div></header>
<main class="wrap"><section id="fixed" data-panel class="active"><h2>Fixed final set order intervention</h2><p>The native, reverse, and five preset hash permutations contain exactly the same terminal cells. Positive comparator-minus-native area differences favor native order; scientific significance is reported, not used as a completion gate.</p><div class="kpis"><div><span>Native area</span><strong>43.60 MPa</strong></div><div><span>Permuted - native</span><strong>+1.49 MPa</strong><small>95% CI -0.66 to 3.38</small></div><div><span>Reverse - native</span><strong>+3.93 MPa</strong><small>95% CI 0.14 to 7.40</small></div><div><span>Endpoint difference</span><strong>0.00 MPa</strong><small>same final set</small></div></div><div class="plots">{figure_cards}</div><h2>Fixed cases</h2><div class="plots">{case_cards}</div><h2>Summary table</h2>{_table(summary, (("variant_display_label", "Order"),("area_mpa", "Area, MPa"),("early_area_mpa", "Early area, MPa"),("endpoint_mae_mpa", "Endpoint MAE, MPa")))}<h2>Paired contrasts</h2>{_table(contrasts, (("contrast", "Contrast"),("metric", "Metric"),("estimate_mpa", "Estimate"),("ci_low_mpa", "CI low"),("ci_high_mpa", "CI high")))}</section>
<section id="archived" data-panel><h2>Archived trajectory timing decomposition</h2><p>Five controls are compared with Proposed / 本文方法 using saved trajectories only. Weighted and raw event identities are preserved; random-policy repeats are averaged within specimen before six-domain-equal aggregation.</p>{archived_card}<h2>Weighted stage contrasts</h2>{_table(stage, (("comparator_display_label", "Control"),("stage", "Stage"),("estimate_mpa", "Estimate"),("ci_low_mpa", "CI low"),("ci_high_mpa", "CI high")))}</section><h2>Data and audit</h2><p>{links}</p><p>{artifact_links}</p><p class="meta">Nominal prefixes {resource['nominal_prefix_requests']}; unique cached prefixes {resource['unique_cached_prefixes']}; evaluated rows including QA {resource['predictor_evaluated_state_rows_including_retries_qa']}/{resource['predictor_evaluated_state_row_cap']}; actor, training, Qwen, CNN, OOF, autograd, and test forwards: 0.</p></main><footer><div class="wrap meta">Self-contained offline report. No remote styles, scripts, fonts, or images.</div></footer>
<script>document.querySelectorAll('button[data-target]').forEach(function(b){{b.addEventListener('click',function(){{document.querySelectorAll('button[data-target]').forEach(function(x){{x.setAttribute('aria-selected','false')}});document.querySelectorAll('[data-panel]').forEach(function(x){{x.classList.remove('active')}});b.setAttribute('aria-selected','true');document.getElementById(b.dataset.target).classList.add('active') }})}})</script></body></html>"""
    del artifacts
    return atomic_text(results / "index.html", document)


def _write_artifacts(context: TaskContext) -> list[Path]:
    root = context.artifacts_root
    root.mkdir(parents=True, exist_ok=True)
    findings = """# 获取顺序机制分析结论

本任务在冻结的 VALID50 队列（50 个物理试样、48 个采集组、6 个域）上区分两类证据。Track A 只重算既有 500 条轨迹的事件时序，不执行新模型前向；对每次采集分别保留原始误差变化与按剩余预算加权的贡献，并先在试样内平均随机策略的 5 次重复，再对 6 个域等权汇总。加权分段结果逐项复现既有证据，成对恒等式最大残差为 4.26e-14 MPa，说明分解与总面积一致。该轨道回答“优势何时出现”，但不能单独证明顺序因果。

Track B 将本文方法每个试样的最终 16 个单元集合固定，只改变获取次序；原次序、逆序和 5 个预注册 SHA256 排序均使用同一个冻结的 MEAN_SC predictor update 1750 重新计算所有前缀，actor 未加载也未前向。原次序的域等权面积为 43.60 MPa，逆序为 47.53 MPa，五次置换均值为 45.08 MPa。主要对比“置换均值减原次序”为 +1.49 MPa，95% 成对 bootstrap 区间为 [-0.66, 3.38]；区间跨零，因此不能声称一般随机次序显著更差。次要对比“逆序减原次序”为 +3.93 MPa，区间 [0.14, 7.40]，在本冻结队列和预注册干预下支持原次序优于完全逆序。早期面积对应差值为 +1.63 [-2.19, 5.18] 与 +3.33 [-1.53, 7.98] MPa，均保留不确定性。

三种次序的终点 MAE 均为 42.38 MPa，成对终点差严格为 0；这与最终集合相同相符，也把差异定位到获取过程而非终点信息量。完整输入参考 MAE 为 41.69 MPa，任一次序曲线在观测成本范围内均未达到该锚点，故不报告伪造的节省；其他质量阈值按首次穿越计算，并保留后续反穿、未达到和负节省。推理共 5,894 个名义前缀请求，对应 5,152 个唯一缓存状态与 742 次复用，另含 12 条非缓存终点核验；累计评估 5,164 行，低于 7,000 行上限。训练、优化器更新、actor、Qwen、CNN、OOF、autograd 与 TEST 推理均为零。结论仅适用于冻结模型、固定 VALID 队列及指定排序规则，不外推为新策略训练收益。
"""
    methods = """# Methods facts

- Cohort: VALID only; 50 physical specimens, 48 capture groups, six domains. No TEST inference or label join.
- Track A: 500 archived rows across six methods. Random has five repeats; all other methods have one. Repeats are averaged within specimen before domain-equal aggregation.
- Track B: the 16-cell terminal set from each proposed-policy trajectory is held fixed. Native, reverse, and five preset SHA256 permutations are scored without actor decisions.
- Clock: Track A preserves archived float64 costs. Track B recomputes cumulative native-cell pixels divided by native image pixels, with float32 cost only at predictor input.
- Predictor: frozen MEAN_SC, selected update 1750. Every original and reordered prefix is evaluated through one common engine and a fully bound cache key.
- Primary estimand: permuted five-repeat mean area minus native area. Secondary estimand: reverse area minus native area. Positive differences favor native.
- Aggregation: area metrics use repeat mean, within-domain mean, then six-domain equal mean. Endpoint MAE/RMSE average losses, never predictions.
- Uncertainty: 5,000 fixed capture-group-within-domain paired bootstrap draws; no fallback bootstrap.
- Quality comparison: first queue-level MAE crossing on separate event-union grids; recrossings, unreached targets, and negative savings remain explicit.
- Completion is technical. Scientific effect size or confidence-interval sign is not a gate.
"""
    captions = """# Figure captions

## F1
**EN.** Archived trajectory timing contrasts. Solid circles show time-weighted contribution differences (Proposed minus control); dashed crosses show raw error-change differences over four left-open, right-closed cost stages. Positive values favor Proposed / 本文方法.

**中文。** 既有轨迹的时序贡献对比。实线圆点为时间加权贡献差（本文方法减对照），虚线叉号为四个左开右闭成本阶段的原始误差变化差；正值有利于本文方法。

## F2
**EN.** Cohort MAE under native, reverse, and five-repeat mean preset permutations of identical terminal sets. Curves use the full union of observed event costs; the dotted line is the frozen full-input reference.

**中文。** 固定相同终点集合后，原次序、逆序与五次预设置换均值的队列 MAE。曲线使用全部事件成本并集；点线为冻结的完整输入参考。

## F3
**EN.** Within-specimen area effects for preset permutation mean and reverse order relative to native order. Points are physical specimens; diamonds and whiskers are domain-equal estimates and 95% paired bootstrap intervals.

**中文。** 置换均值及逆序相对原次序的试样内面积效应。散点代表物理试样，菱形与误差线为域等权估计及 95% 成对 bootstrap 区间。

## F4
**EN.** Matched-quality cost differences at first MAE crossing. Positive values favor native order; crosses mark thresholds not reached by one or both series. Recrossings are not removed.

**中文。** 首次达到 MAE 阈值时的匹配质量成本差。正值有利于原次序；叉号表示至少一条曲线未达到阈值，后续反穿不被删除。

## Fixed cases
**EN.** Predeclared cases show acquisition ranks on the same 8x8 terminal set and their corresponding prediction trajectories. Permutation 0 is displayed by fixed rule, not selected by outcome.

**中文。** 预先固定的三个案例展示相同 8x8 终点集合上的采集排名及预测轨迹。置换 0 按固定规则展示，未依据结果挑选。
"""
    reproduce = """# Reproduce

From the repository worktree, with the locked Python environment available:

```bash
python -m scripts.cai_order_mechanism.run all --root . --scope docs/cai/order_mechanism/ORDER_MECHANISM_SCOPE.json
```

The phase state hashes inputs and outputs. Completed phases are reused; the single permitted recovery was already recorded. The report reads cached tables only and never invokes a research model. For a focused local acceptance run:

```bash
pytest -q tests/test_cai_order_mechanism.py
python -m scripts.cai_order_mechanism.run verify --root . --scope docs/cai/order_mechanism/ORDER_MECHANISM_SCOPE.json
```
"""
    handoff = """# Codex handoff: CAI order mechanism

- Task: `CAI_ORDER_MECHANISM_R1_37B3C404`
- Source: `37b3c40414c00c6633b64656c4bbb2b178ef9848`
- Branch: `research/cai-vlm-agent-v3-controlled-reuse`
- Scientific status: primary preset-permutation contrast is directionally positive but its 95% interval crosses zero; reverse order is worse than native in the prespecified secondary contrast. Endpoint equality is exact.
- Technical status: inspect `acceptance_results.json` and `PUBLICATION_RECEIPT.json`; the latter is written only after remote publication checks.
- Audit anchors: `ORDER_MECHANISM_SCOPE.json`, `input_bindings.json`, `runtime_lock.json`, `resource_usage.json`, `order_plan.sha256`, and `release_manifest.json`.
- No model selection, actor decision, optimization, TEST access, manuscript rewrite, pull request, merge, force push, or tag is part of this delivery.
"""
    return [
        atomic_text(root / "FINDINGS_ZH.md", findings),
        atomic_text(root / "METHODS_FACTS.md", methods),
        atomic_text(root / "FIGURE_CAPTIONS.md", captions),
        atomic_text(root / "REPRODUCE.md", reproduce),
        atomic_text(root / "CODEX_HANDOFF_CAI_ORDER_MECHANISM.md", handoff),
    ]


def _charge_render(results: Path) -> Path:
    path = results / "resource_usage.json"
    usage: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    count = int(usage.get("report_render_count", 0)) + 1
    if count > 2:
        raise ValueError("cache-only report-render limit exceeded")
    usage["report_render_count"] = count
    return atomic_json(path, usage)


def render_report(context: TaskContext) -> list[Path]:
    _style()
    results = context.results_root
    figures = results / "figures"
    cases = results / "cases"
    qa = results / "figure_qa"
    outputs: list[Path] = []
    outputs.extend(_archived_stage_figure(results, figures, qa))
    outputs.extend(_fixed_curve_figure(results, figures, qa))
    outputs.extend(_order_effect_figure(results, figures, qa))
    outputs.extend(_matched_quality_figure(results, figures, qa))
    for key in CASE_KEYS:
        outputs.extend(_case_figure(results, cases, qa, key))
    outputs.extend(_write_artifacts(context))
    outputs.append(_render_html(results, context.artifacts_root))
    outputs.append(_charge_render(results))
    return outputs


__all__ = ["CASE_KEYS", "render_report"]
