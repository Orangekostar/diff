"""Render cached diagnostic arrays into independent figures and offline HTML."""

from __future__ import annotations

import ast
import csv
import html
import importlib.util
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

from scripts.cai_c_retrain.vlm import render_c_inputs

from .context import TaskContext, atomic_json, sha256_file


class _AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.assets: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for name in ("src", "href"):
            value = values.get(name)
            if value and not value.startswith("#"):
                self.assets.add(value)


def audit_local_html_links(path: str | Path) -> list[str]:
    document = Path(path)
    parser = _AssetParser()
    parser.feed(document.read_text(encoding="utf-8"))
    for asset in parser.assets:
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", asset) or asset.startswith("//"):
            raise ValueError(f"remote HTML asset is forbidden: {asset}")
        target = (document.parent / asset.split("#", 1)[0]).resolve()
        if not target.exists():
            raise ValueError(f"missing HTML asset: {asset}")
    return sorted(parser.assets)


def nearest_rgba_layer(
    values: np.ndarray,
    *,
    width: int,
    height: int,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    alpha: int = 150,
) -> Image.Image:
    grid = np.asarray(values, dtype=np.float64)
    if grid.shape != (8, 8) or not np.isfinite(grid).all() or vmax < vmin:
        raise ValueError("heatmap layer requires a finite 8x8 grid")
    span = vmax - vmin
    normalized = np.zeros_like(grid) if span == 0 else np.clip((grid - vmin) / span, 0, 1)
    rgba = (colormaps[cmap](normalized) * 255).astype(np.uint8)
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode="RGBA").resize((width, height), Image.Resampling.NEAREST)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _slug(key: str) -> str:
    return key.replace(":", "_")


def _scores(path: Path) -> dict[str, np.ndarray]:
    rows = _read_csv(path)
    if len(rows) != 64:
        raise ValueError(f"score map is not 64 rows: {path}")
    return {
        "logits": np.asarray([float(row["raw_logit"]) for row in rows]),
        "legal": np.asarray([row["env_legal"] == "True" for row in rows]),
        "proposal": np.asarray([row["proposal_legal"] == "True" for row in rows]),
        "p_env": np.asarray([float(row["p_env"]) for row in rows]),
        "p_policy": np.asarray([float(row["p_policy"]) for row in rows]),
        "measured": np.asarray([row["measured"] == "True" for row in rows]),
        "indicator": np.asarray([float(row["vlm_indicator"]) for row in rows]),
        "selected": np.asarray([row["selected"] == "True" for row in rows]),
    }


def _alignment_tool():
    codex_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    path = codex_root / "skills/nature-figure/scripts/audit_panel_alignment.py"
    spec = importlib.util.spec_from_file_location("nature_panel_alignment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Nature panel-alignment tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_figure(
    fig: Any,
    png_path: Path,
    records: list[dict[str, Any]],
    *,
    qa_root: Path,
    inputs: list[str],
    vector: bool = True,
    force: bool = False,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path = qa_root / "panel_alignment" / png_path.parent.name / f"{png_path.stem}.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    cached_report = (
        json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else None
    )
    if (
        not force
        and png_path.exists()
        and (not vector or pdf_path.exists())
        and cached_report is not None
        and cached_report.get("verdict") == "PASS"
    ):
        report = cached_report
    else:
        fig.canvas.draw()
        report = _alignment_tool().require_matplotlib_panel_alignment(
            fig, json_out=qa_path, strict=True
        )
        fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
        if vector:
            fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    def repository_relative(path: Path) -> str:
        resolved = path.resolve()
        for parent in resolved.parents:
            if (parent / ".git").exists():
                return resolved.relative_to(parent).as_posix()
        raise ValueError(f"figure path is outside a Git worktree: {path}")

    records.append(
        {
            "path": repository_relative(png_path),
            "sha256": sha256_file(png_path),
            "pdf": repository_relative(pdf_path) if vector else "",
            "pdf_sha256": sha256_file(pdf_path) if vector else "",
            "inputs": [repository_relative(Path(value)) for value in inputs],
            "panel_alignment_verdict": report["verdict"],
        }
    )


def _axis_map(
    axis: Any,
    clean: Image.Image,
    grid: np.ndarray | None,
    title: str,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    measured: np.ndarray | None = None,
    candidates: np.ndarray | None = None,
    selected: np.ndarray | None = None,
) -> None:
    axis.imshow(clean, extent=(0, 8, 8, 0), interpolation="nearest")
    if grid is not None:
        image = axis.imshow(
            np.asarray(grid).reshape(8, 8),
            extent=(0, 8, 8, 0),
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.62,
        )
        axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
    for cell in range(64):
        x, y = cell % 8, cell // 8
        if measured is not None and measured[cell]:
            axis.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor="#f7f7f7", linewidth=1.0))
        if candidates is not None and candidates[cell]:
            axis.add_patch(Rectangle((x + 0.06, y + 0.06), 0.88, 0.88, fill=False, edgecolor="#d62728", linewidth=1.4))
        if selected is not None and selected[cell]:
            axis.add_patch(Rectangle((x + 0.14, y + 0.14), 0.72, 0.72, fill=False, edgecolor="#00bcd4", linewidth=2.1))
    axis.set_xlim(0, 8)
    axis.set_ylim(8, 0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=8, pad=4)


def _outline_layer(
    width: int,
    height: int,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int, int],
    inset_fraction: float,
    line_width: int,
) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for cell in np.flatnonzero(mask):
        row, col = divmod(int(cell), 8)
        x0, x1 = round(col * width / 8), round((col + 1) * width / 8)
        y0, y1 = round(row * height / 8), round((row + 1) * height / 8)
        inset = round(min(x1 - x0, y1 - y0) * inset_fraction)
        draw.rectangle((x0 + inset, y0 + inset, x1 - inset - 1, y1 - inset - 1), outline=color, width=line_width)
    return image


def _number_layer(width: int, height: int) -> Image.Image:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for cell in range(64):
        row, col = divmod(cell, 8)
        x, y = round(col * width / 8) + 3, round(row * height / 8) + 3
        draw.text((x, y), str(cell), font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 220))
    return image


def _parse_list(value: str) -> list[Any]:
    parsed = ast.literal_eval(value)
    return parsed if isinstance(parsed, list) else [parsed]


def _state_for_prefix(states: list[dict[str, str]], key: str, model: str, prefix: int) -> dict[str, str]:
    for row in states:
        if row["specimen_key"] != key:
            continue
        for source_model, source_prefix in zip(
            _parse_list(row["source_models"]), _parse_list(row["prefix_lengths"])
        ):
            if source_model == model and int(source_prefix) == prefix:
                return row
    raise ValueError(f"missing {model} prefix {prefix} for {key}")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _html_document(
    output: Path,
    selected: list[dict[str, str]],
    states: list[dict[str, str]],
    figures: dict[str, dict[str, str]],
) -> str:
    case_sections = []
    for case in selected:
        key = case["specimen_key"]
        slug = _slug(key)
        links = figures[key]
        state_links = []
        for row in states:
            if row["specimen_key"] != key:
                continue
            base = f"states/{slug}/{row['state_id']}"
            state_links.append(
                f"<tr><td>{html.escape(row['state_id'])}</td><td>{html.escape(row['sources'])}</td>"
                f"<td>{html.escape(row['action_count'])}</td><td>{float(row['exact_cost']):.4f}</td>"
                f"<td><a href='{base}/C/scores.csv'>C scores</a> | <a href='{base}/N/scores.csv'>N scores</a> | "
                f"<a href='{base}/C/attention.npz'>C attention</a> | <a href='{base}/N/attention.npz'>N attention</a> | "
                f"<a href='{base}/C/surface_attribution.npz'>C attribution</a> | "
                f"<a href='{base}/N/surface_attribution.npz'>N attribution</a></td></tr>"
            )
        case_sections.append(
            f"""
<section id="{slug}">
  <h2>{html.escape(key)}</h2>
  <div class="layer-controls">
    <label><input type="checkbox" data-layer="heatmap" checked> Heatmap</label>
    <label><input type="checkbox" data-layer="candidates" checked> VLM candidates</label>
    <label><input type="checkbox" data-layer="selection" checked> Selection</label>
    <label><input type="checkbox" data-layer="numbers"> Numbering</label>
  </div>
  <div class="stack" data-stack>
    <img src="{links['clean']}" alt="clean surface">
    <img class="heatmap" src="{links['heatmap_layer']}" alt="C environment probability">
    <img class="candidates" src="{links['candidate_layer']}" alt="C0 candidates">
    <img class="selection" src="{links['selection_layer']}" alt="selected cells">
    <img class="numbers hidden" src="{links['number_layer']}" alt="cell numbering">
  </div>
  <nav class="group-links">
    <a href="{links['A']}">A C0 gate</a><a href="{links['B']}">B attribution</a>
    <a href="{links['D']}">D native process</a><a href="{links['E']}">E Actor attention</a>
    <a href="{links['F']}">F intervention</a>
  </nav>
  <div class="plate"><img src="{links['A']}" alt="group A"><img src="{links['B']}" alt="group B"></div>
  <div class="plate"><img src="{links['D']}" alt="group D"><img src="{links['E']}" alt="group E"></div>
  <img class="wide" src="{links['F']}" alt="group F">
  <details><summary>Common physical states ({sum(row['specimen_key'] == key for row in states)})</summary>
    <table><thead><tr><th>State</th><th>Source</th><th>t</th><th>Cost</th><th>Raw data</th></tr></thead>
    <tbody>{''.join(state_links)}</tbody></table>
  </details>
</section>"""
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Actor C0 机制诊断</title>
<style>
:root{{--ink:#20252b;--muted:#69737d;--line:#d9dee3;--paper:#fff;--accent:#007c83}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);font:14px/1.55 system-ui,sans-serif;background:#f5f7f8}}
header,main{{max-width:1260px;margin:auto}} header{{padding:28px 24px 14px}} h1{{font-size:26px;margin:0 0 8px;letter-spacing:0}}
header p{{color:var(--muted);margin:0}} section{{background:var(--paper);border-top:1px solid var(--line);padding:24px;margin-bottom:16px}}
h2{{font-size:18px;margin:0 0 14px;letter-spacing:0}} .layer-controls{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px}}
.stack{{position:relative;width:min(100%,620px);aspect-ratio:1/1;overflow:hidden;background:#111}}
.stack img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}} .hidden{{display:none!important}}
.group-links{{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}} a{{color:var(--accent)}} .group-links a{{border:1px solid var(--line);padding:6px 9px;text-decoration:none}}
.plate{{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}} .plate img,.wide{{width:100%;display:block;border:1px solid var(--line)}}
table{{width:100%;border-collapse:collapse;margin-top:12px;font-size:12px}} th,td{{border-bottom:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}}
details{{margin-top:18px}} @media(max-width:760px){{.plate{{grid-template-columns:1fr}} section{{padding:16px}}}}
</style></head><body><header><h1>Actor C0 机制诊断</h1>
<p>六件固定 VALID 案例。区分外部门控、同状态先验通道敏感性、表面特征局部敏感性与 Actor attention；不包含训练、TEST 或论文改写。</p>
<p><a href="first_action_c0_audit.csv">首步 C0 审计</a> · <a href="c0_intervention_results.csv">冻结策略干预</a> · <a href="fixed_state_prior_sensitivity.csv">共同状态先验敏感性</a> · <a href="surface_probe_results.csv">特征扰动</a></p>
</header><main>{''.join(case_sections)}</main>
<script>document.querySelectorAll('.layer-controls').forEach(c=>c.addEventListener('change',e=>{{const s=c.nextElementSibling;s.querySelector('.'+e.target.dataset.layer).classList.toggle('hidden',!e.target.checked)}}));</script>
</body></html>"""


def render_stage(context: TaskContext) -> dict[str, Any]:
    output = context.path("output")
    signature = context.phase_signature(
        "render",
        (
            context.config_path,
            output / "state_manifest.csv",
            output / "first_action_c0_audit.csv",
            output / "attribution_checks.json",
            output / "attention_checks.json",
            Path(__file__),
        ),
    )
    if context.stage_complete("render", signature):
        return {"status": "RENDER_REUSED"}
    selected = _read_csv(output / "selected_cases.csv")
    states = _read_csv(output / "state_manifest.csv")
    figure_records: list[dict[str, Any]] = []
    html_figures: dict[str, dict[str, str]] = {}
    qa_root = context.path("artifacts") / "figure_qa"
    source_root = json.loads(
        (context.path("data") / "feature_bank_manifest.json").read_text(encoding="utf-8")
    )["encoder_execution_root"]
    preflight_rows = []

    plt.rcParams.update(
        {"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6,
         "ytick.labelsize": 6, "legend.fontsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none"}
    )
    for case in selected:
        key, slug = case["specimen_key"], _slug(case["specimen_key"])
        source_path = context.source_path(case["source_path"], additional_roots=(source_root,))
        rendered = render_c_inputs(Image.open(source_path))
        if rendered.clean_sha256 != case["clean_sha256"]:
            raise ValueError(f"clean surface hash changed for {key}")
        overlay_root = output / "overlays" / slug
        panel_root = output / "panels" / slug
        overlay_root.mkdir(parents=True, exist_ok=True)
        panel_root.mkdir(parents=True, exist_ok=True)
        clean_path = overlay_root / "clean.png"
        rendered.clean.save(clean_path)
        width, height = rendered.clean.size
        t0 = _state_for_prefix(states, key, "C", 0)
        state_root = output / "states" / slug / t0["state_id"]
        c = _scores(state_root / "C/scores.csv")
        n = _scores(state_root / "N/scores.csv")
        for name, values in (("C p_env", c["p_env"]), ("C p_policy", c["p_policy"]), ("N p_env", n["p_env"])):
            if np.any(values < -1e-12) or np.any(values > 1 + 1e-12) or abs(values.sum() - 1) > 1e-6:
                raise ValueError(f"invalid cached probability map: {key} {name}")
        heatmap_layer = nearest_rgba_layer(c["p_env"].reshape(8, 8), width=width, height=height, vmin=0, vmax=1)
        heatmap_layer_path = overlay_root / "initial_c_env_heatmap_layer.png"
        heatmap_layer.save(heatmap_layer_path)
        candidate_layer_path = overlay_root / "initial_c0_candidates_layer.png"
        _outline_layer(width, height, c["proposal"], color=(220, 39, 39, 255), inset_fraction=0.06, line_width=max(2, width // 300)).save(candidate_layer_path)
        selection_layer_path = overlay_root / "initial_selection_layer.png"
        _outline_layer(width, height, c["selected"], color=(0, 220, 235, 255), inset_fraction=0.16, line_width=max(2, width // 220)).save(selection_layer_path)
        number_layer_path = overlay_root / "cell_numbers_layer.png"
        _number_layer(width, height).save(number_layer_path)

        fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.05), constrained_layout=True)
        _axis_map(axes[0], rendered.clean, c["p_env"], "C: p over environment L", vmin=0, vmax=1, measured=c["measured"], selected=np.zeros(64, bool))
        _axis_map(axes[1], rendered.clean, c["proposal"].astype(float), "External C0 eligible cells", cmap="Reds", vmin=0, vmax=1, candidates=c["proposal"], selected=c["selected"])
        _axis_map(axes[2], rendered.clean, c["p_policy"], "C: p after C0", vmin=0, vmax=1, candidates=c["proposal"], selected=c["selected"])
        _axis_map(axes[3], rendered.clean, n["p_env"], "N: p over same L", vmin=0, vmax=1, selected=n["selected"])
        fig.suptitle(f"A | Initial external gate | {key}", fontsize=10)
        a_path = panel_root / "group_A_initial_gate.png"
        _save_figure(fig, a_path, figure_records, qa_root=qa_root, inputs=[str(state_root / "C/scores.csv"), str(state_root / "N/scores.csv")])

        with np.load(state_root / "C/surface_attribution.npz", allow_pickle=False) as payload:
            c_attr = {name: np.asarray(payload[name]) for name in payload.files}
        with np.load(state_root / "N/surface_attribution.npz", allow_pickle=False) as payload:
            n_attr = {name: np.asarray(payload[name]) for name in payload.files}
        fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.7), constrained_layout=True)
        for column, name in enumerate(("direct_signed", "total_signed", "via_signed")):
            maximum = max(float(np.max(np.abs(c_attr[name][0]))), float(np.max(np.abs(n_attr[name][0]))))
            maximum = maximum if maximum > 0 else 1.0
            _axis_map(axes[0, column], rendered.clean, c_attr[name][0], f"C own | {name.replace('_signed','')}", cmap="coolwarm", vmin=-maximum, vmax=maximum, measured=c["measured"], selected=c["selected"])
            _axis_map(axes[1, column], rendered.clean, n_attr[name][0], f"N own | {name.replace('_signed','')}", cmap="coolwarm", vmin=-maximum, vmax=maximum, measured=n["measured"], selected=n["selected"])
        fig.suptitle(f"B | Local feature sensitivity (Gradient x delta-S) | {key}", fontsize=10)
        b_path = panel_root / "group_B_surface_attribution.png"
        _save_figure(fig, b_path, figure_records, qa_root=qa_root, inputs=[str(state_root / "C/surface_attribution.npz"), str(state_root / "N/surface_attribution.npz")])
        if len(n_attr["target_actions"]) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.95), constrained_layout=True)
            for column, name in enumerate(("direct_signed", "total_signed", "via_signed")):
                maximum = float(np.max(np.abs(n_attr[name][1]))) or 1.0
                _axis_map(axes[column], rendered.clean, n_attr[name][1], f"N on common C action | {name.replace('_signed','')}", cmap="coolwarm", vmin=-maximum, vmax=maximum, measured=n["measured"])
            fig.suptitle(f"B-common | N sensitivity for C target action | {key}", fontsize=10)
            _save_figure(fig, panel_root / "group_B_common_c_action.png", figure_records, qa_root=qa_root, inputs=[str(state_root / "N/surface_attribution.npz")])

        case_states = [row for row in states if row["specimen_key"] == key]
        for state_row in case_states:
            sid = state_row["state_id"]
            current_root = output / "states" / slug / sid
            cs, ns = _scores(current_root / "C/scores.csv"), _scores(current_root / "N/scores.csv")
            fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.9), constrained_layout=True)
            _axis_map(axes[0], rendered.clean, cs["p_env"], "C on fixed physical state", vmin=0, vmax=1, measured=cs["measured"], selected=cs["selected"])
            _axis_map(axes[1], rendered.clean, ns["p_env"], "N on same physical state", vmin=0, vmax=1, measured=ns["measured"], selected=ns["selected"])
            fig.suptitle(f"C | {key} | {sid} | sources {state_row['sources']}", fontsize=9)
            _save_figure(fig, panel_root / f"group_C_{sid}.png", figure_records, qa_root=qa_root, inputs=[str(current_root / "C/scores.csv"), str(current_root / "N/scores.csv")], vector=False)

        fig, axes = plt.subplots(2, 6, figsize=(12.0, 4.3), constrained_layout=True)
        for route_index, model in enumerate(("C", "N")):
            prefixes = sorted(
                {int(prefix) for row in case_states for source_model, prefix in zip(_parse_list(row["source_models"]), _parse_list(row["prefix_lengths"])) if source_model == model}
            )
            decision_prefixes = [value for value in (0, 1, 2, 8, max(prefixes)) if value in prefixes]
            for column, prefix in enumerate(decision_prefixes):
                state_row = _state_for_prefix(states, key, model, prefix)
                values = _scores(output / "states" / slug / state_row["state_id"] / model / "scores.csv")
                _axis_map(axes[route_index, column], rendered.clean, values["p_policy"], f"{model} native t={prefix}", vmin=0, vmax=1, measured=values["measured"], selected=values["selected"])
            terminal_root = output / "states" / slug / f"terminal_{model.lower()}"
            with np.load(terminal_root / "physical_state.npz", allow_pickle=False) as payload:
                terminal_measured = np.asarray(payload["measured"], dtype=bool)
            _axis_map(axes[route_index, 5], rendered.clean, None, f"{model} terminal T", measured=terminal_measured)
        fig.suptitle(f"D | Native decision process; terminal has no next action | {key}", fontsize=10)
        d_path = panel_root / "group_D_native_process.png"
        _save_figure(fig, d_path, figure_records, qa_root=qa_root, inputs=[str(output / "state_manifest.csv")])

        fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.7), constrained_layout=True)
        attention_maps: dict[tuple[int, int], tuple[np.ndarray, str]] = {}
        for row_index, model in enumerate(("C", "N")):
            with np.load(state_root / model / "attention.npz", allow_pickle=False) as payload:
                query_map = np.asarray(payload["query_to_cells"])[-1]
                action_map = np.asarray(payload["action_to_cells"])[-1]
                rollout_map = np.asarray(payload["rollout_query_to_cells"])
                query_self = float(np.asarray(payload["query_self_mass"])[-1])
            attention_maps[(row_index, 0)] = (query_map, f"{model} last-layer query (self={query_self:.3f})")
            attention_maps[(row_index, 1)] = (action_map, f"{model} target-token attention")
            attention_maps[(row_index, 2)] = (rollout_map, f"{model} approximate rollout")
        attention_vmax = max(float(values.max()) for values, _title in attention_maps.values()) or 1.0
        for (row_index, column), (values, title) in attention_maps.items():
            _axis_map(axes[row_index, column], rendered.clean, values, title, cmap="magma", vmin=0, vmax=attention_vmax)
        fig.suptitle(f"E | Actor fusion-token attention (not surface attribution) | {key}", fontsize=10)
        e_path = panel_root / "group_E_actor_attention.png"
        _save_figure(fig, e_path, figure_records, qa_root=qa_root, inputs=[str(state_root / "C/attention.npz"), str(state_root / "N/attention.npz")])

        fig, axis = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
        colors = {"C_NATIVE": "#007c83", "N_NATIVE": "#5f6368", "C_NO_C0": "#c43c39"}
        for condition in ("C_NATIVE", "N_NATIVE", "C_NO_C0"):
            trajectory = json.loads((output / "trajectories" / slug / f"{condition}.json").read_text(encoding="utf-8"))
            axis.step(trajectory["costs"], trajectory["predictions_mpa"], where="post", label=condition, color=colors[condition], linewidth=1.8)
        axis.axhline(float(case["target_mpa"]), color="#111111", linestyle="--", linewidth=1, label="Target")
        axis.set(xlabel="Acquisition cost", ylabel="Current prediction (MPa)", title=f"F | Frozen-policy intervention | {key}")
        axis.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
        axis.grid(axis="y", color="#e1e5e8", linewidth=0.6)
        f_path = panel_root / "group_F_intervention_curves.png"
        _save_figure(
            fig,
            f_path,
            figure_records,
            qa_root=qa_root,
            inputs=[str(output / "trajectories" / slug)],
            force=True,
        )

        html_figures[key] = {
            "clean": _relative(clean_path, output),
            "heatmap_layer": _relative(heatmap_layer_path, output),
            "candidate_layer": _relative(candidate_layer_path, output),
            "selection_layer": _relative(selection_layer_path, output),
            "number_layer": _relative(number_layer_path, output),
            "A": _relative(a_path, output), "B": _relative(b_path, output),
            "D": _relative(d_path, output), "E": _relative(e_path, output), "F": _relative(f_path, output),
        }
        preflight_rows.append(
            {"specimen_key": key, "source_sha256": sha256_file(source_path),
             "clean_sha256": rendered.clean_sha256, "state_count": len(case_states), "status": "PASS"}
        )

    index_path = output / "index.html"
    index_path.write_text(_html_document(output, selected, states, html_figures), encoding="utf-8")
    links = audit_local_html_links(index_path)
    atomic_json(output / "figure_source_preflight.json", {"status": "PASS", "cases": preflight_rows})
    atomic_json(
        output / "figure_manifest.json",
        {"status": "PASS", "backend": "python-matplotlib", "interpolation": "nearest",
         "figure_count": len(figure_records), "figures": figure_records, "html_asset_count": len(links)},
    )
    result = {"status": "RENDER_COMPLETE", "figure_count": len(figure_records), "html_asset_count": len(links)}
    context.complete_stage("render", signature, result)
    return result


__all__ = ["audit_local_html_links", "nearest_rgba_layer", "render_stage"]
