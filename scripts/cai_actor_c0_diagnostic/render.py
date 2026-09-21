"""Render cached diagnostic arrays into independent figures and offline HTML."""

from __future__ import annotations

import ast
import csv
import html
import importlib.util
import json
import math
import os
import re
import time
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
from .reporting import enrich_cached_report, trajectory_difference_rows


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty render table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


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
        "rank_env": np.asarray([int(row["rank_env"]) if row["rank_env"] else 0 for row in rows]),
        "rank_policy": np.asarray([int(row["rank_policy"]) if row["rank_policy"] else 0 for row in rows]),
        "measured": np.asarray([row["measured"] == "True" for row in rows]),
        "indicator": np.asarray([float(row["vlm_indicator"]) for row in rows]),
        "confidence": np.asarray([float(row["vlm_confidence"]) for row in rows]),
        "selected": np.asarray([row["selected"] == "True" for row in rows]),
    }


def centered_legal_logits(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(legal, dtype=bool)
    if values.shape != mask.shape or values.ndim != 1 or not np.any(mask):
        raise ValueError("centered logits require matching non-empty legal cells")
    output = np.full(values.shape, np.nan, dtype=np.float64)
    output[mask] = values[mask] - float(values[mask].mean())
    return output


def _candidate_masks(
    indicator: np.ndarray,
    proposal: np.ndarray,
    *,
    c0_reason: str,
) -> tuple[np.ndarray, np.ndarray]:
    vlm_candidates = np.asarray(indicator) > 0
    proposal_mask = np.asarray(proposal, dtype=bool)
    if vlm_candidates.shape != proposal_mask.shape or vlm_candidates.ndim != 1:
        raise ValueError("candidate masks require matching one-dimensional arrays")
    effective_c0 = (
        proposal_mask.copy()
        if c0_reason == "HIGHEST_RELIABLE_CONFIDENCE_C0"
        else np.zeros_like(proposal_mask)
    )
    return vlm_candidates, effective_c0


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
    del output

    def state_article(row: dict[str, str]) -> str:
        slug = _slug(row["specimen_key"])
        state_id = row["state_id"]
        base = f"states/{slug}/{state_id}"
        panel = f"panels/{slug}/group_C_{state_id}.png"
        raw_links = (
            ("metadata", f"{base}/metadata.json"),
            ("physical state", f"{base}/physical_state.npz"),
            ("C scores", f"{base}/C/scores.csv"),
            ("N scores", f"{base}/N/scores.csv"),
            ("C attention", f"{base}/C/attention.npz"),
            ("N attention", f"{base}/N/attention.npz"),
            ("C attribution", f"{base}/C/surface_attribution.npz"),
            ("N attribution", f"{base}/N/surface_attribution.npz"),
            ("C zero-prior scores", f"{base}/C/zero_prior_scores.npz"),
            ("zero-prior summary", f"{base}/C/zero_prior_sensitivity.json"),
            ("C query checks", f"{base}/C/query_checks.json"),
            ("N query checks", f"{base}/N/query_checks.json"),
        )
        links = " · ".join(
            f'<a href="{href}">{html.escape(label)}</a>' for label, href in raw_links
        )
        return f"""
<article class="state-record" id="{slug}-{html.escape(state_id)}">
  <div class="state-head"><strong>{html.escape(state_id)}</strong>
    <span>来源 {html.escape(row['sources'])}</span><span>t={html.escape(row['t'])}</span>
    <span>cost={float(row['exact_cost']):.6f}</span>
    <span>history={html.escape(row['prefix_cells_in_order'])}</span>
  </div>
  <a class="figure-link" href="{panel}"><img src="{panel}" alt="C 与 N 在相同物理状态的评分对照"></a>
  <div class="raw-links">{links}</div>
</article>"""

    case_sections: list[str] = []
    for case_index, case in enumerate(selected):
        key = case["specimen_key"]
        slug = _slug(key)
        links = figures[key]
        decision_states = [
            row
            for row in states
            if row["specimen_key"] == key and row["terminal_view_only"] == "False"
        ]
        c_states = "".join(
            state_article(row)
            for row in decision_states
            if "C" in _parse_list(row["source_models"])
        )
        n_states = "".join(
            state_article(row)
            for row in decision_states
            if "N" in _parse_list(row["source_models"])
        )
        hidden = " hidden" if case_index else ""
        case_sections.append(
            f"""
<section id="{slug}" class="case-panel{hidden}" data-case-panel="{slug}">
  <div class="case-heading"><h2>{html.escape(key)}</h2><span>target={float(case['target_mpa']):.4f} MPa</span></div>
  <div class="layer-controls">
    <label><input type="checkbox" data-layer="heatmap" checked> 概率热图</label>
    <label><input type="checkbox" data-layer="candidates" checked> VLM全部候选</label>
    <label><input type="checkbox" data-layer="selection" checked> 原生选中</label>
    <label><input type="checkbox" data-layer="numbers"> 格子编号</label>
  </div>
  <div class="stack" data-stack>
    <img src="{links['clean']}" alt="clean surface">
    <img class="heatmap" src="{links['heatmap_layer']}" alt="C environment probability">
    <img class="candidates" src="{links['candidate_layer']}" alt="all VLM candidates">
    <img class="selection" src="{links['selection_layer']}" alt="selected cells">
    <img class="numbers hidden" src="{links['number_layer']}" alt="cell numbering">
  </div>
  <nav class="question-links" aria-label="问题导航">
    <a href="#{slug}-gate">外部门控改变首步了吗？</a>
    <a href="#{slug}-path">表面通过哪条路径影响决策？</a>
    <a href="#{slug}-process">两条原生路径怎样演化？</a>
    <a href="#{slug}-attention">Actor关注了哪些token？</a>
    <a href="#{slug}-intervention">关闭C0后怎样变化？</a>
  </nav>
  <div class="view-tabs" role="tablist">
    <button type="button" class="active" data-view-tab="native">原生轨迹对照</button>
    <button type="button" data-view-tab="fixed">相同状态对照</button>
  </div>
  <div class="view-pane" data-view-pane="native">
    <div class="plate" id="{slug}-gate"><a class="figure-link" href="{links['A']}"><img src="{links['A']}" alt="首步门控"></a></div>
    <div class="plate" id="{slug}-path"><a class="figure-link" href="{links['B']}"><img src="{links['B']}" alt="表面归因"></a></div>
    <div class="plate" id="{slug}-process"><a class="figure-link" href="{links['D']}"><img src="{links['D']}" alt="原生决策进程"></a></div>
    <div class="plate" id="{slug}-attention"><a class="figure-link" href="{links['E']}"><img src="{links['E']}" alt="Actor attention"></a></div>
    <div class="plate" id="{slug}-intervention"><a class="figure-link" href="{links['F']}"><img src="{links['F']}" alt="冻结策略干预"></a></div>
    <p><a href="{links['F_csv']}">动作顺序及差值表</a> · <a href="c0_intervention_results.csv">三路径结果表</a></p>
  </div>
  <div class="view-pane hidden" data-view-pane="fixed">
    <div class="source-tabs" role="tablist">
      <button type="button" class="active" data-source-tab="C">C来源状态</button>
      <button type="button" data-source-tab="N">N来源状态</button>
    </div>
    <div class="source-pane" data-source-pane="C">{c_states}</div>
    <div class="source-pane hidden" data-source-pane="N">{n_states}</div>
  </div>
</section>"""
        )
    options = "".join(
        f'<option value="{_slug(row["specimen_key"])}">{html.escape(row["specimen_key"])}</option>'
        for row in selected
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Actor C0 机制诊断</title>
<style>
:root{{--ink:#20252b;--muted:#69737d;--line:#d9dee3;--paper:#fff;--accent:#006f75;--danger:#a93632}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);font:14px/1.55 system-ui,sans-serif;background:#f5f7f8}}
header,main{{max-width:1260px;margin:auto}} header{{padding:24px}} h1{{font-size:26px;margin:0 0 10px;letter-spacing:0}}
header p{{color:var(--muted);margin:7px 0}} select{{min-width:min(100%,360px);padding:7px;border:1px solid var(--line);background:white}}
section{{background:var(--paper);border-top:1px solid var(--line);padding:24px;margin-bottom:16px}}
h2{{font-size:18px;margin:0;letter-spacing:0}} .case-heading{{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;margin-bottom:14px}}
.case-heading span,.state-head span{{color:var(--muted)}} .layer-controls{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px}}
.stack{{position:relative;width:min(100%,620px);aspect-ratio:1/1;overflow:hidden;background:#111}}
.stack img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}} .hidden{{display:none!important}}
a{{color:var(--accent)}} .question-links{{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}}
.question-links a{{border-bottom:2px solid var(--line);padding:6px 2px;text-decoration:none}}
.view-tabs,.source-tabs{{display:flex;border-bottom:1px solid var(--line);margin:16px 0 12px}}
.view-tabs button,.source-tabs button{{border:0;border-bottom:3px solid transparent;background:transparent;padding:9px 13px;color:var(--muted);cursor:pointer}}
.view-tabs button.active,.source-tabs button.active{{border-bottom-color:var(--accent);color:var(--ink);font-weight:650}}
.plate{{margin:0 0 18px}} .plate img,.state-record img,.summary img{{width:100%;display:block;border:1px solid var(--line)}}
.figure-link{{display:block}} .state-record{{border-top:1px solid var(--line);padding:16px 0 20px}}
.state-head{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:9px}} .raw-links{{margin-top:8px;font-size:12px;overflow-wrap:anywhere}}
.summary{{padding:0 24px 24px}} .summary h2{{margin-bottom:10px}}
@media(max-width:760px){{section{{padding:16px}} header{{padding:18px 16px}} .summary{{padding:0 16px 16px}} .view-tabs button,.source-tabs button{{padding:8px}}}}
</style></head><body><header><h1>Actor C0 机制诊断</h1>
<p>六件固定 VALID 案例。区分外部门控、同状态先验通道敏感性、表面特征局部敏感性与 Actor attention；不包含训练、TEST 或论文改写。</p>
<p><a href="first_action_c0_audit.csv">首步 C0 审计</a> · <a href="c0_intervention_results.csv">冻结策略干预</a> · <a href="fixed_state_prior_sensitivity.csv">共同状态先验敏感性</a> · <a href="surface_probe_results.csv">特征扰动</a></p>
<label for="case-selector">案例</label> <select id="case-selector">{options}</select>
</header><main>
<section class="summary"><h2>六案例 t0 等权空间概览</h2>
<a class="figure-link" href="panels/summary/group_G_t0_equal_weight_summary.png"><img src="panels/summary/group_G_t0_equal_weight_summary.png" alt="六案例 t0 等权空间概览"></a>
<p><a href="t0_equal_weight_summary.csv">t0_equal_weight_summary.csv</a> · 规范化图格频率，不是物理毫米损伤图。</p></section>
{''.join(case_sections)}</main>
<script>
const select=document.getElementById('case-selector');
select.addEventListener('change',()=>document.querySelectorAll('[data-case-panel]').forEach(p=>p.classList.toggle('hidden',p.dataset.casePanel!==select.value)));
document.querySelectorAll('.layer-controls').forEach(c=>c.addEventListener('change',e=>{{const s=c.parentElement.querySelector('[data-stack]');s.querySelector('.'+e.target.dataset.layer).classList.toggle('hidden',!e.target.checked)}}));
document.querySelectorAll('[data-view-tab]').forEach(b=>b.addEventListener('click',()=>{{const root=b.closest('[data-case-panel]');root.querySelectorAll('[data-view-tab]').forEach(x=>x.classList.toggle('active',x===b));root.querySelectorAll('[data-view-pane]').forEach(x=>x.classList.toggle('hidden',x.dataset.viewPane!==b.dataset.viewTab));}}));
document.querySelectorAll('[data-source-tab]').forEach(b=>b.addEventListener('click',()=>{{const root=b.closest('[data-view-pane]');root.querySelectorAll('[data-source-tab]').forEach(x=>x.classList.toggle('active',x===b));root.querySelectorAll('[data-source-pane]').forEach(x=>x.classList.toggle('hidden',x.dataset.sourcePane!==b.dataset.sourceTab));}}));
</script>
</body></html>"""


def render_stage(context: TaskContext) -> dict[str, Any]:
    output = context.path("output")
    provenance = enrich_cached_report(context)
    trajectory_inputs = sorted((output / "trajectories").glob("*/*.json"))
    signature = context.phase_signature(
        "render",
        (
            context.config_path,
            output / "state_manifest.csv",
            output / "selected_cases.csv",
            output / "first_action_c0_audit.csv",
            output / "fixed_state_prior_sensitivity.csv",
            output / "attribution_checks.json",
            output / "attention_checks.json",
            Path(__file__),
            Path(__file__).with_name("reporting.py"),
            *trajectory_inputs,
        ),
    )
    if context.stage_complete("render", signature):
        return {"status": "RENDER_REUSED"}
    render_started = time.monotonic()
    selected = _read_csv(output / "selected_cases.csv")
    states = _read_csv(output / "state_manifest.csv")
    first_actions = {
        row["specimen_key"]: row for row in _read_csv(output / "first_action_c0_audit.csv")
    }
    figure_records: list[dict[str, Any]] = []
    html_figures: dict[str, dict[str, str]] = {}
    qa_root = context.path("artifacts") / "figure_qa"
    source_root = json.loads(
        (context.path("data") / "feature_bank_manifest.json").read_text(encoding="utf-8")
    )["encoder_execution_root"]
    preflight_rows = []
    t0_summary: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "C mean p_env",
            "N mean p_env",
            "VLM candidate frequency",
            "C selected frequency",
            "C unrestricted argmax frequency",
            "N selected frequency",
        )
    }

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
        first = first_actions[key]
        vlm_candidates, effective_c0 = _candidate_masks(
            c["indicator"], c["proposal"], c0_reason=first["c0_reason"]
        )
        for name, values in (("C p_env", c["p_env"]), ("C p_policy", c["p_policy"]), ("N p_env", n["p_env"])):
            if np.any(values < -1e-12) or np.any(values > 1 + 1e-12) or abs(values.sum() - 1) > 1e-6:
                raise ValueError(f"invalid cached probability map: {key} {name}")
        heatmap_layer = nearest_rgba_layer(c["p_env"].reshape(8, 8), width=width, height=height, vmin=0, vmax=1)
        heatmap_layer_path = overlay_root / "initial_c_env_heatmap_layer.png"
        heatmap_layer.save(heatmap_layer_path)
        candidate_layer_path = overlay_root / "initial_c0_candidates_layer.png"
        _outline_layer(width, height, vlm_candidates, color=(220, 39, 39, 255), inset_fraction=0.06, line_width=max(2, width // 300)).save(candidate_layer_path)
        selection_layer_path = overlay_root / "initial_selection_layer.png"
        _outline_layer(width, height, c["selected"], color=(0, 220, 235, 255), inset_fraction=0.16, line_width=max(2, width // 220)).save(selection_layer_path)
        number_layer_path = overlay_root / "cell_numbers_layer.png"
        _number_layer(width, height).save(number_layer_path)

        c_free = np.zeros(64, dtype=np.float64)
        c_free[int(first["a_free"])] = 1.0
        t0_summary["C mean p_env"].append(c["p_env"])
        t0_summary["N mean p_env"].append(n["p_env"])
        t0_summary["VLM candidate frequency"].append(vlm_candidates.astype(np.float64))
        t0_summary["C selected frequency"].append(c["selected"].astype(np.float64))
        t0_summary["C unrestricted argmax frequency"].append(c_free)
        t0_summary["N selected frequency"].append(n["selected"].astype(np.float64))

        fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.05), constrained_layout=True)
        _axis_map(axes[0], rendered.clean, c["p_env"], "C: p over environment L", vmin=0, vmax=1, measured=c["measured"], selected=np.zeros(64, bool))
        c0_title = (
            "Effective C0; red=all VLM candidates"
            if np.any(effective_c0)
            else "C0 inactive; red=all VLM candidates"
        )
        policy_title = (
            "C: p after effective C0"
            if np.any(effective_c0)
            else "C: p after environment-L fallback"
        )
        _axis_map(axes[1], rendered.clean, effective_c0.astype(float), c0_title, cmap="Reds", vmin=0, vmax=1, candidates=vlm_candidates, selected=c["selected"])
        _axis_map(axes[2], rendered.clean, c["p_policy"], policy_title, vmin=0, vmax=1, candidates=vlm_candidates, selected=c["selected"])
        _axis_map(axes[3], rendered.clean, n["p_env"], "N: p over same L", vmin=0, vmax=1, selected=n["selected"])
        fig.suptitle(
            f"A | {key} | source=C/N shared | t=0 | cost={float(t0['exact_cost']):.6f} | "
            f"a_C0={first['a_C0']}  a_free={first['a_free']}  a_N={first['a_N']}  "
            f"blocked_mass={float(first['blocked_mass']):.4f}",
            fontsize=9,
        )
        a_path = panel_root / "group_A_initial_gate.png"
        _save_figure(
            fig,
            a_path,
            figure_records,
            qa_root=qa_root,
            inputs=[str(state_root / "C/scores.csv"), str(state_root / "N/scores.csv")],
            force=True,
        )

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
        fig.suptitle(
            f"B | feature-level local sensitivity | {key} | source=C/N shared | "
            f"t=0 | cost={float(t0['exact_cost']):.6f} | "
            f"targets C={int(c_attr['target_actions'][0])}, N={int(n_attr['target_actions'][0])}",
            fontsize=9,
        )
        b_path = panel_root / "group_B_surface_attribution.png"
        _save_figure(
            fig,
            b_path,
            figure_records,
            qa_root=qa_root,
            inputs=[str(state_root / "C/surface_attribution.npz"), str(state_root / "N/surface_attribution.npz")],
            force=True,
        )
        if len(n_attr["target_actions"]) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.95), constrained_layout=True)
            for column, name in enumerate(("direct_signed", "total_signed", "via_signed")):
                maximum = float(np.max(np.abs(n_attr[name][1]))) or 1.0
                _axis_map(axes[column], rendered.clean, n_attr[name][1], f"N on common C action | {name.replace('_signed','')}", cmap="coolwarm", vmin=-maximum, vmax=maximum, measured=n["measured"])
            fig.suptitle(
                f"B-common | {key} | source=C/N shared | t=0 | "
                f"cost={float(t0['exact_cost']):.6f} | N target={int(n_attr['target_actions'][1])}",
                fontsize=9,
            )
            _save_figure(
                fig,
                panel_root / "group_B_common_c_action.png",
                figure_records,
                qa_root=qa_root,
                inputs=[str(state_root / "N/surface_attribution.npz")],
                force=True,
            )

        case_states = [row for row in states if row["specimen_key"] == key]
        for state_row in case_states:
            if state_row["terminal_view_only"] == "True":
                continue
            sid = state_row["state_id"]
            current_root = output / "states" / slug / sid
            cs, ns = _scores(current_root / "C/scores.csv"), _scores(current_root / "N/scores.csv")
            c_centered = centered_legal_logits(cs["logits"], cs["legal"])
            n_centered = centered_legal_logits(ns["logits"], ns["legal"])
            maximum = max(
                float(np.nanmax(np.abs(c_centered))),
                float(np.nanmax(np.abs(n_centered))),
            ) or 1.0
            c_action = int(np.flatnonzero(cs["selected"])[0])
            n_action = int(np.flatnonzero(ns["selected"])[0])
            fig, axes = plt.subplots(2, 2, figsize=(6.1, 5.8), constrained_layout=True)
            _axis_map(
                axes[0, 0], rendered.clean, c_centered, "C legal centered logits",
                cmap="coolwarm", vmin=-maximum, vmax=maximum,
                measured=cs["measured"], selected=cs["selected"],
            )
            _axis_map(
                axes[0, 1], rendered.clean, cs["p_env"], "C full probability over L",
                vmin=0, vmax=1, measured=cs["measured"], selected=cs["selected"],
            )
            _axis_map(
                axes[1, 0], rendered.clean, n_centered, "N legal centered logits",
                cmap="coolwarm", vmin=-maximum, vmax=maximum,
                measured=ns["measured"], selected=ns["selected"],
            )
            _axis_map(
                axes[1, 1], rendered.clean, ns["p_env"], "N full probability over same L",
                vmin=0, vmax=1, measured=ns["measured"], selected=ns["selected"],
            )
            fig.suptitle(
                f"C | {key} | {sid} | source={state_row['sources']} | "
                f"t={state_row['t']} | cost={float(state_row['exact_cost']):.6f} | "
                f"targets C={c_action}, N={n_action}",
                fontsize=9,
            )
            _save_figure(
                fig,
                panel_root / f"group_C_{sid}.png",
                figure_records,
                qa_root=qa_root,
                inputs=[str(current_root / "C/scores.csv"), str(current_root / "N/scores.csv")],
                vector=False,
                force=True,
            )

        fig, axes = plt.subplots(2, 6, figsize=(12.0, 4.3), constrained_layout=True)
        for route_index, model in enumerate(("C", "N")):
            prefixes = sorted(
                {int(prefix) for row in case_states for source_model, prefix in zip(_parse_list(row["source_models"]), _parse_list(row["prefix_lengths"])) if source_model == model}
            )
            decision_prefixes = [value for value in (0, 1, 2, 8, max(prefixes)) if value in prefixes]
            for column, prefix in enumerate(decision_prefixes):
                state_row = _state_for_prefix(states, key, model, prefix)
                values = _scores(output / "states" / slug / state_row["state_id"] / model / "scores.csv")
                target_action = int(np.flatnonzero(values["selected"])[0])
                _axis_map(
                    axes[route_index, column],
                    rendered.clean,
                    values["p_policy"],
                    f"{model} native | t={prefix} | cost={float(state_row['exact_cost']):.4f}\na={target_action}",
                    vmin=0,
                    vmax=1,
                    measured=values["measured"],
                    selected=values["selected"],
                )
            terminal_root = output / "states" / slug / f"terminal_{model.lower()}"
            with np.load(terminal_root / "physical_state.npz", allow_pickle=False) as payload:
                terminal_measured = np.asarray(payload["measured"], dtype=bool)
                terminal_cost = float(payload["exact_cost"])
            _axis_map(
                axes[route_index, 5], rendered.clean, None,
                f"{model} terminal T | cost={terminal_cost:.4f}\nno next action",
                measured=terminal_measured,
            )
        fig.suptitle(
            f"D | native decision process | {key} | target={float(case['target_mpa']):.4f} MPa",
            fontsize=10,
        )
        d_path = panel_root / "group_D_native_process.png"
        _save_figure(
            fig, d_path, figure_records, qa_root=qa_root,
            inputs=[str(output / "state_manifest.csv")], force=True,
        )

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
        fig.suptitle(
            f"E | Actor fusion-token attention | {key} | source=C/N shared | t=0 | "
            f"cost={float(t0['exact_cost']):.6f} | targets C={first['a_C0']}, N={first['a_N']}",
            fontsize=9,
        )
        e_path = panel_root / "group_E_actor_attention.png"
        _save_figure(
            fig,
            e_path,
            figure_records,
            qa_root=qa_root,
            inputs=[str(state_root / "C/attention.npz"), str(state_root / "N/attention.npz")],
            force=True,
        )

        case_trajectories = {
            condition: json.loads(
                (output / "trajectories" / slug / f"{condition}.json").read_text(
                    encoding="utf-8"
                )
            )
            for condition in ("C_NATIVE", "N_NATIVE", "C_NO_C0")
        }
        difference_rows = trajectory_difference_rows(case_trajectories)
        f_csv_path = panel_root / "group_F_action_difference.csv"
        _write_csv(f_csv_path, difference_rows)
        fig, (axis, table_axis) = plt.subplots(
            2,
            1,
            figsize=(8.8, 7.2),
            gridspec_kw={"height_ratios": (3.2, 2.8)},
            constrained_layout=True,
        )
        colors = {"C_NATIVE": "#007c83", "N_NATIVE": "#5f6368", "C_NO_C0": "#c43c39"}
        for condition in ("C_NATIVE", "N_NATIVE", "C_NO_C0"):
            trajectory = case_trajectories[condition]
            axis.step(trajectory["costs"], trajectory["predictions_mpa"], where="post", label=condition, color=colors[condition], linewidth=1.8)
        axis.axhline(float(case["target_mpa"]), color="#111111", linestyle="--", linewidth=1, label="Target")
        axis.set(
            xlabel="Acquisition cost",
            ylabel="Current prediction (MPa)",
            title=f"F | frozen-policy intervention | {key} | target={float(case['target_mpa']):.4f} MPa",
        )
        axis.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
        axis.grid(axis="y", color="#e1e5e8", linewidth=0.6)
        table_axis.axis("off")
        table = table_axis.table(
            cellText=[
                [
                    row["step"],
                    row["c_native_action"],
                    row["n_native_action"],
                    row["c_no_c0_action"],
                    row["n_native_action_differs_from_c"],
                    row["c_no_c0_action_differs_from_c"],
                    f"{float(row['c_no_c0_cost_minus_c']):.6f}"
                    if row["c_no_c0_cost_minus_c"] != ""
                    else "",
                ]
                for row in difference_rows
            ],
            colLabels=("step", "C action", "N action", "C-no-C0 action", "N != C", "off != C", "off cost - C"),
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(5.2)
        table.scale(1.0, 0.78)
        table_axis.set_title("Action sequence and per-step difference table", fontsize=8, pad=4)
        f_path = panel_root / "group_F_intervention_curves.png"
        _save_figure(
            fig,
            f_path,
            figure_records,
            qa_root=qa_root,
            inputs=[
                str(output / "trajectories" / slug / "C_NATIVE.json"),
                str(output / "trajectories" / slug / "N_NATIVE.json"),
                str(output / "trajectories" / slug / "C_NO_C0.json"),
                str(f_csv_path),
            ],
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
            "F_csv": _relative(f_csv_path, output),
        }
        preflight_rows.append(
            {"specimen_key": key, "source_sha256": sha256_file(source_path),
             "clean_sha256": rendered.clean_sha256, "state_count": len(case_states), "status": "PASS"}
        )

    if any(len(values) != 6 for values in t0_summary.values()):
        raise ValueError("t0 spatial summary must weight exactly six specimens equally")
    summary_arrays = {
        name: np.mean(np.stack(values, axis=0), axis=0)
        for name, values in t0_summary.items()
    }
    summary_fields = {
        "C mean p_env": "mean_c_p_env",
        "N mean p_env": "mean_n_p_env",
        "VLM candidate frequency": "vlm_candidate_frequency",
        "C selected frequency": "c_selected_frequency",
        "C unrestricted argmax frequency": "c_unrestricted_argmax_frequency",
        "N selected frequency": "n_selected_frequency",
    }
    summary_rows = [
        {
            "cell": cell,
            "row": cell // 8,
            "col": cell % 8,
            **{
                summary_fields[name]: float(values[cell])
                for name, values in summary_arrays.items()
            },
            "specimen_weight": "1/6",
            "interpretation": "NORMALIZED_GRID_FREQUENCY_NOT_PHYSICAL_MM_DAMAGE",
        }
        for cell in range(64)
    ]
    summary_csv_path = output / "t0_equal_weight_summary.csv"
    _write_csv(summary_csv_path, summary_rows)
    blank_surface = Image.new("RGB", (640, 640), "white")
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.8), constrained_layout=True)
    for axis, (name, values) in zip(axes.flat, summary_arrays.items()):
        _axis_map(axis, blank_surface, values, name, vmin=0, vmax=1)
    fig.suptitle(
        "G | Equal-weight t0 summary across six specimens | normalized grid, not physical mm damage",
        fontsize=9,
    )
    summary_png_path = output / "panels/summary/group_G_t0_equal_weight_summary.png"
    _save_figure(
        fig,
        summary_png_path,
        figure_records,
        qa_root=qa_root,
        inputs=[
            str(path)
            for path in sorted(output.glob("states/*/s00_*/[CN]/scores.csv"))
        ] + [str(summary_csv_path)],
        force=True,
    )

    index_path = output / "index.html"
    index_path.write_text(_html_document(output, selected, states, html_figures), encoding="utf-8")
    links = audit_local_html_links(index_path)
    atomic_json(output / "figure_source_preflight.json", {"status": "PASS", "cases": preflight_rows})
    atomic_json(
        output / "figure_manifest.json",
        {"status": "PASS", "backend": "python-matplotlib", "interpolation": "nearest",
         "figure_count": len(figure_records), "figures": figure_records, "html_asset_count": len(links),
         "t0_equal_weight_specimen_count": 6,
         "t0_spatial_interpretation": "NORMALIZED_GRID_FREQUENCY_NOT_PHYSICAL_MM_DAMAGE"},
    )
    provenance["cache_report_render_seconds_upper_bound"] = math.ceil(
        time.monotonic() - render_started
    )
    provenance["timing_measurement"] = "MONOTONIC_WALL_SECONDS_ROUNDED_UP"
    atomic_json(output / "report_provenance.json", provenance)
    result = {
        "status": "RENDER_COMPLETE",
        "figure_count": len(figure_records),
        "html_asset_count": len(links),
        "provenance": provenance["status"],
    }
    context.complete_stage("render", signature, result)
    return result


__all__ = [
    "audit_local_html_links",
    "centered_legal_logits",
    "nearest_rgba_layer",
    "render_stage",
]
