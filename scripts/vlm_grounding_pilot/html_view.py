"""Render the fixed VLM grounding diagnostic payload as a static HTML page."""

from html import escape
from pathlib import Path

_SUMMARY_LINKS = (
    "summary.csv",
    "candidate_changes.csv",
    "historical_replay_comparison.csv",
    "human_review_template.csv",
    "HOW_TO_REVIEW_ZH.md",
    "experiment_lock.json",
)

_PANEL_LABELS = {
    "A": "A",
    "A_P0_R0": "A",
    "B": "B",
    "B_P1_R0": "B",
    "C": "C",
    "C_P0_R1": "C",
    "D": "D",
    "D_P1_R1": "D",
    "H00": "H00",
    "H00_CACHED": "H00",
}


def _safe(value):
    """Escape a source value for either HTML text or an attribute."""

    return escape("" if value is None else str(value), quote=True)


def _bool_text(value):
    return "true" if value else "false"


def _cells(values):
    return ", ".join(_safe(value) for value in values)


def _panel_label(variant):
    variant = "" if variant is None else str(variant)
    if variant in _PANEL_LABELS:
        return _PANEL_LABELS[variant]
    prefix = variant.split("_", 1)[0]
    return prefix if prefix in {"A", "B", "C", "D", "H00"} else variant


def _asset_link(path, label, *, download=False, class_name="asset-link"):
    if not path:
        return '<span class="muted">未提供</span>'
    download_attr = " download" if download else ""
    return (
        f'<a class="{_safe(class_name)}" href="{_safe(path)}"'
        f' target="_blank" rel="noopener"{download_attr}>{_safe(label)}</a>'
    )


def _image_figure(path, alt, caption):
    if not path:
        return (
            '<figure class="image-figure missing">'
            f'<figcaption>{_safe(caption)}</figcaption>'
            '<p class="muted">未提供图像</p></figure>'
        )
    return (
        '<figure class="image-figure">'
        f'<a href="{_safe(path)}" target="_blank" rel="noopener">'
        f'<img src="{_safe(path)}" alt="{_safe(alt)}" loading="lazy"></a>'
        f'<figcaption>{_safe(caption)}</figcaption>'
        f'<div class="image-links">{_asset_link(path, "打开原尺寸")}'
        f'{_asset_link(path, "下载 PNG（原尺寸）", download=True)}</div>'
        '</figure>'
    )


def _region_html(region, index):
    region = region or {}
    return (
        '<article class="region">'
        f'<h4>Region {_safe(index)}</h4>'
        '<dl class="facts">'
        f'<dt>actual cells</dt><dd>{_cells(region.get("cells", []))}</dd>'
        f'<dt>cue</dt><dd>{_safe(region.get("cue", ""))}</dd>'
        f'<dt>alternative</dt><dd>{_safe(region.get("alternative", ""))}</dd>'
        f'<dt>confidence</dt><dd>{_safe(region.get("confidence", ""))}</dd>'
        '</dl></article>'
    )


def _panel_html(case, panel, case_index, panel_index):
    panel = panel or {}
    variant = panel.get("variant", "")
    label = _panel_label(variant)
    panel_id = f"case-{case_index}-panel-{panel_index}"
    image_id = f"{panel_id}-image"
    link_id = f"{panel_id}-native-link"
    empty_id = f"{panel_id}-empty"
    overlay = panel.get("overlay", "")
    numbered = panel.get("numbered", "")
    clean = case.get("clean", "")
    initial = overlay or numbered or clean
    regions = panel.get("regions", [])
    raw_links = panel.get("raw_links", [])
    raw_html = []
    for raw_index, raw_path in enumerate(raw_links):
        raw_label = ("H00 historical final raw" if label == "H00" else "raw first-pass") if raw_index == 0 else "raw repair"
        if raw_index > 1:
            raw_label = f"raw #{raw_index + 1}"
        raw_html.append(f'<li>{_asset_link(raw_path, raw_label)}</li>')
    if not raw_html:
        raw_html.append('<li class="muted">未提供</li>')

    regions_html = "".join(_region_html(region, index) for index, region in enumerate(regions, 1))
    if not regions_html:
        regions_html = '<p class="muted">regions: 0</p>'

    selected_source = "overlay" if overlay else "numbered" if numbered else "clean"
    selected_option = {
        "overlay": "selected" if selected_source == "overlay" else "",
        "numbered": "selected" if selected_source == "numbered" else "",
        "clean": "selected" if selected_source == "clean" else "",
    }
    source_select = (
        f'<label class="source-control" for="{panel_id}-source">'
        "显示图像："
        f'<select id="{panel_id}-source" data-source-select="1"'
        f' data-target-image="{image_id}" data-target-link="{link_id}"'
        f' data-target-empty="{empty_id}" data-overlay="{_safe(overlay)}"'
        f' data-numbered="{_safe(numbered)}" data-clean="{_safe(clean)}">'
        f'<option value="overlay" {selected_option["overlay"]}>overlay（候选叠图）</option>'
        f'<option value="numbered" {selected_option["numbered"]}>actual numbered input（实际编号输入）</option>'
        f'<option value="clean" {selected_option["clean"]}>clean（clean 输入）</option>'
        "</select></label>"
    )
    visual = (
        '<div class="panel-visual">'
        f"{source_select}"
        '<div class="panel-image-frame">'
        f'<img id="{image_id}" src="{_safe(initial)}" alt="{_safe(label)} selected image" loading="lazy">'
        f'<p id="{empty_id}" class="muted" hidden>未提供所选图像</p>'
        "</div>"
        f'<p class="selected-image-links"><a id="{link_id}" href="{_safe(initial)}"'
        ' target="_blank" rel="noopener" download>下载所选图像（原尺寸）</a></p>'
        "</div>"
    )
    overlay_link = _asset_link(overlay, "下载 overlay PNG（独立、原尺寸）", download=True)
    ordinal = panel.get("ordinal", "")
    ordinal_html = _image_figure(
        ordinal,
        f"{label} decoded ordinal confidence",
        "decoded ordinal confidence, NOT attention / damage probability（解码序数置信度，不是注意力 / 损伤概率）",
    )
    return (
        f'<section class="panel" id="{panel_id}">'
        '<header class="panel-header">'
        f'<h3><span class="panel-label">{_safe(label)}</span>'
        f'<span class="variant">{_safe(variant)}</span></h3>'
        f'<p class="panel-state"><span>state / no-cue state</span>: '
        f'{_safe(panel.get("status", ""))}</p>'
        "</header>"
        f"{visual}"
        '<div class="panel-content">'
        '<dl class="facts status-facts">'
        f'<dt>first-pass/repair status</dt><dd>{_safe(panel.get("first_pass_status", ""))}</dd>'
        f'<dt>no_reliable_cue</dt><dd>{_safe(panel.get("no_reliable_cue", "UNAVAILABLE"))}</dd>'
        f'<dt>repair-dependent</dt><dd>{_bool_text(panel.get("repair_dependent", False))}</dd>'
        f'<dt>C0 cells</dt><dd>{_cells(panel.get("c0", []))}</dd>'
        '</dl>'
        '<p class="c0-note"><strong>C0 diagnostic only/no action</strong>'
        f'（C0 仅诊断，不执行动作）: {_safe(panel.get("c0_reason", ""))}</p>'
        '<div class="regions">'
        f'<h4>Regions ({_safe(len(regions))})</h4>{regions_html}'
        '</div>'
        '<div class="panel-files">'
        f'<p><span>Overlay PNG:</span> {overlay_link}</p>'
        f'<div class="file-row"><span>Raw first/repair:</span>'
        f'<ul class="raw-links">{"".join(raw_html)}</ul></div>'
        f'<p><span>Features CSV:</span> {_asset_link(panel.get("features", ""), "打开 features CSV")}</p>'
        '</div>'
        f'<details><summary>首次回答解析内容（与修复后分开）</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">{_safe(panel.get("first_pass_detail", "H00仅保存历史最终回答"))}</pre></details>'
        f'{ordinal_html}'
        '</div></section>'
    )


def _case_html(case, case_index):
    case = case or {}
    case_key = case.get("specimen_key", "")
    visuals = (
        _image_figure(case.get("clean", ""), f"{case_key} clean input", "clean（模型输入）")
        + _image_figure(case.get("R0", ""), f"{case_key} R0 numbered input", "R0（原编号模型输入）")
        + _image_figure(case.get("R1", ""), f"{case_key} R1 numbered input", "R1（新编号模型输入）")
    )
    panels = "".join(
        _panel_html(case, panel, case_index, panel_index)
        for panel_index, panel in enumerate(case.get("panels", []))
    )
    return (
        f'<section class="case" id="case-{case_index}">'
        '<header class="case-header">'
        f'<h2>{_safe(case_key)}</h2>'
        f'<p>split: {_safe(case.get("split", ""))}</p>'
        '</header>'
        f'<div class="case-visuals">{visuals}</div>'
        f'<div class="panels">{panels}</div>'
        '</section>'
    )


def _page_html(payload):
    status = _safe(payload.get("status", ""))
    cases = payload.get("cases", [])
    summary_links = " ".join(
        _asset_link(path, path) for path in _SUMMARY_LINKS
    )
    case_html = "".join(_case_html(case, index) for index, case in enumerate(cases))
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VLM grounding pilot diagnostic</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: #1f2933; background: #f4f6f8; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; min-width: 0; line-height: 1.45; overflow-x: hidden; }}
main {{ width: min(1500px, 100%); margin: 0 auto; padding: 18px; }}
h1, h2, h3, h4, p {{ margin-top: 0; }}
h1 {{ font-size: clamp(1.35rem, 4vw, 2rem); margin-bottom: 8px; }}
h2 {{ font-size: 1.25rem; margin-bottom: 2px; overflow-wrap: anywhere; }}
h3 {{ display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 4px; }}
h4 {{ font-size: 0.98rem; margin-bottom: 6px; }}
.top, .case {{ background: #fff; border: 1px solid #d7dde3; border-radius: 6px; }}
.top {{ padding: 18px; margin-bottom: 18px; }}
.note {{ border-left: 4px solid #c2410c; background: #fff7ed; padding: 10px 12px; overflow-wrap: anywhere; }}
.note p:last-child {{ margin-bottom: 0; }}
.links {{ display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 12px 0 0; overflow-wrap: anywhere; }}
.links a, .asset-link {{ color: #075985; }}
.case {{ padding: 14px; margin-bottom: 18px; }}
.case-header {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; margin-bottom: 12px; }}
.case-header p {{ margin-bottom: 0; color: #52606d; }}
.case-visuals {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-bottom: 16px; }}
.image-figure {{ min-width: 0; margin: 0; padding: 8px; border: 1px solid #e5e7eb; background: #fafbfc; }}
.image-figure > a:first-child {{ display: block; min-width: 0; }}
.image-figure img {{ display: block; width: 100%; max-width: 100%; height: auto; max-height: 360px; object-fit: contain; background: #eef1f4; }}
.image-figure figcaption {{ margin-top: 6px; font-weight: 600; overflow-wrap: anywhere; }}
.image-links {{ display: flex; flex-wrap: wrap; gap: 4px 10px; margin-top: 4px; font-size: 0.88rem; overflow-wrap: anywhere; }}
.panels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 290px), 1fr)); gap: 12px; align-items: start; }}
.panel {{ min-width: 0; border: 1px solid #cfd6dd; border-top: 4px solid #64748b; background: #fff; padding: 10px; }}
.panel-header {{ border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; margin-bottom: 9px; }}
.panel-label {{ display: inline-grid; place-items: center; min-width: 2rem; min-height: 2rem; padding: 2px 6px; background: #e2e8f0; color: #0f172a; font-weight: 700; }}
.variant {{ color: #52606d; font-size: 0.86rem; overflow-wrap: anywhere; }}
.panel-state {{ margin-bottom: 0; overflow-wrap: anywhere; }}
.panel-state span, .panel-files span {{ font-weight: 600; }}
.panel-visual {{ min-width: 0; }}
.source-control {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 7px; font-weight: 600; }}
select {{ max-width: 100%; min-height: 2rem; padding: 4px 6px; border: 1px solid #9aa5b1; background: #fff; color: inherit; }}
.panel-image-frame {{ min-height: 150px; display: grid; place-items: center; overflow: hidden; background: #eef1f4; }}
.panel-image-frame img {{ display: block; max-width: 100%; width: 100%; height: auto; max-height: 300px; object-fit: contain; }}
.selected-image-links {{ margin: 6px 0 0; overflow-wrap: anywhere; font-size: 0.88rem; }}
.panel-content {{ min-width: 0; }}
.facts {{ display: grid; grid-template-columns: minmax(0, max-content) minmax(0, 1fr); gap: 4px 8px; margin: 10px 0; }}
.facts dt {{ color: #52606d; overflow-wrap: anywhere; }}
.facts dd {{ margin: 0; overflow-wrap: anywhere; word-break: break-word; }}
.status-facts {{ border-top: 1px solid #e5e7eb; padding-top: 8px; }}
.c0-note {{ padding: 8px; margin: 10px 0; background: #fefce8; border-left: 3px solid #ca8a04; overflow-wrap: anywhere; }}
.regions {{ margin-top: 12px; }}
.region {{ border-top: 1px solid #e5e7eb; padding-top: 7px; margin-top: 7px; }}
.region .facts {{ margin-bottom: 0; }}
.panel-files {{ border-top: 1px solid #e5e7eb; margin-top: 12px; padding-top: 8px; overflow-wrap: anywhere; }}
.panel-files p {{ margin-bottom: 6px; }}
.raw-links {{ margin: 3px 0 0; padding-left: 18px; }}
.ordinal-figure {{ margin-top: 12px; }}
.ordinal-figure img {{ max-height: 220px; }}
.ordinal-figure figcaption {{ font-size: 0.86rem; font-weight: 600; }}
.muted {{ color: #68737d; }}
.missing {{ min-height: 72px; }}
[hidden] {{ display: none !important; }}
@media (max-width: 640px) {{
  main {{ padding: 10px; }}
  .top, .case {{ padding: 10px; }}
  .case-visuals {{ grid-template-columns: minmax(0, 1fr); }}
  .facts {{ grid-template-columns: 1fr; gap: 1px; }}
  .facts dt {{ margin-top: 5px; }}
  .facts dd {{ padding-left: 6px; }}
}}
</style>
</head>
<body>
<main>
<section class="top">
<h1>VLM grounding pilot：静态诊断查看页</h1>
<p>状态：{status}</p>
<div class="note">
<p><strong>边界说明：</strong>输出变化是敏感性，不是准确率；output changes are sensitivity not accuracy。human review PENDING。</p>
<p>six development cases not independent TEST。training / Actor / CAI / attention: 0。</p>
<p>C0 diagnostic only/no action；页面只呈现诊断输出，不推断 GT、赢家或新动作。</p>
</div>
<nav class="links" aria-label="报告文件">{summary_links}</nav>
</section>
{case_html}
</main>
<script>
(function () {{
  document.querySelectorAll('[data-source-select]').forEach(function (select) {{
    var image = document.getElementById(select.dataset.targetImage);
    var nativeLink = document.getElementById(select.dataset.targetLink);
    var empty = document.getElementById(select.dataset.targetEmpty);
    var sources = {{
      overlay: select.dataset.overlay || '',
      numbered: select.dataset.numbered || '',
      clean: select.dataset.clean || ''
    }};
    function updateImage() {{
      var source = sources[select.value] || '';
      image.hidden = !source;
      empty.hidden = !!source;
      nativeLink.hidden = !source;
      if (source) {{
        image.src = source;
        nativeLink.href = source;
      }} else {{
        image.removeAttribute('src');
        nativeLink.removeAttribute('href');
      }}
    }}
    select.addEventListener('change', updateImage);
    updateImage();
  }});
}}());
</script>
</body>
</html>
'''


def build_html(payload: dict, out: Path) -> None:
    """Write the static diagnostic page to ``out/index.html``."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(_page_html(payload), encoding="utf-8")
