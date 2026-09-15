"""Build the lightweight, local HTML index for the VLM cell diagnostic."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cai_agent_v3" / "vlm_cell_diagnostic" / "r1_8778aa53"

IMAGE_SPECS = (
    (
        "01_source_surface.png",
        "原始方向（source orientation）",
    ),
    (
        "02_vlm_clean_input.png",
        "历史 clean 输入（historical VLM clean input）",
    ),
    (
        "03_vlm_numbered_input.png",
        "历史编号输入（historical VLM numbered input）",
    ),
    (
        "04_readable_grid_reference.png",
        "诊断参考：清晰编号；不是历史 VLM 输入（diagnostic reference, not historical VLM input）",
    ),
    (
        "05_cached_confidence_8x8.png",
        "解码后的序数置信；不是注意力（confidence after decoding, before Actor selection）",
    ),
    (
        "06_raw_ids_on_numbered_input.png",
        "原始回答格号直接映射到编号输入（raw IDs on numbered input）",
    ),
    (
        "07_parsed_and_feature_cells.png",
        "解析格与特征格逐格对照（parsed and feature cells）",
    ),
    (
        "08_c0_and_historical_action.png",
        "C0候选与历史 Actor 首动作分开展示（historical Actor action）",
    ),
    (
        "09_pre_cell_attention_clean.png",
        "clean 图上的首格前诊断回放注意力（diagnostic replay, pre-first-cell token attention）",
    ),
    (
        "10_pre_cell_attention_numbered.png",
        "编号图上的首格前诊断回放注意力（diagnostic replay, pre-first-cell token attention）",
    ),
    (
        "11_attention_with_raw_cells.png",
        "注意力诊断叠加原始格号与历史 Actor 首动作（diagnostic attention with raw cells and historical Actor action）",
    ),
)

RESOURCE_LABELS = (
    ("raw_response.txt", "原始回答 raw_response.txt"),
    ("prompt.txt", "原始提示词 prompt.txt"),
    ("identity.json", "身份 identity.json"),
    ("stage_a.json", "A阶段状态 stage_a.json"),
    ("attention_status.json", "注意力状态 attention_status.json"),
    ("coordinate_trace.csv", "坐标追踪表 coordinate_trace.csv"),
    ("features_64.csv", "64维特征表 features_64.csv"),
    ("parsed_response.json", "解析回答 parsed_response.json"),
    ("cache_record.json", "缓存记录 cache_record.json"),
    ("attention_token_mapping.csv", "注意力 token 映射 attention_token_mapping.csv"),
    ("attention_mass.json", "注意力权重总量 attention_mass.json"),
    ("attention_clean.npy", "clean 原始35×35注意力数组"),
    ("attention_numbered.npy", "numbered 原始35×35注意力数组"),
    ("attention_headmean_last4.npy", "最后四层各层全head平均向量"),
    ("attention_clean_native_unsmoothed.png", "clean 原尺寸未平滑热图"),
    ("attention_numbered_native_unsmoothed.png", "numbered 原尺寸未平滑热图"),
    ("attention_preflight.json", "前缀、query和图像token参数"),
    ("next_token.json", "真实下一token top-10"),
    ("historical_surface_cues.png", "历史候选图原件"),
    ("historical_first_action.png", "历史首动作图原件"),
    ("synthetic_roundtrip.csv", "64格合成往返检查"),
)


def escape(value: object) -> str:
    """Escape dynamic text before placing it in HTML text or an attribute."""

    return html.escape(str(value), quote=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> tuple[object | None, str | None]:
    if not path.is_file():
        return None, None
    raw = read_text(path)
    try:
        return json.loads(raw), raw
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, raw


def display_value(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def render_details(label: str, raw: str) -> str:
    return (
        '<details class="raw-block">'
        f"<summary>{escape(label)}</summary>"
        f"<pre>{escape(raw)}</pre>"
        "</details>"
    )


def render_state() -> tuple[str, object | None, str]:
    records: list[tuple[str, object | None, str | None]] = []
    for filename in ("identity.json", "stage_a.json", "attention_status.json"):
        data, raw = read_json(OUT / filename)
        if raw is not None:
            records.append((filename, data, raw))

    rows: list[str] = []
    identity = next((data for name, data, _ in records if name == "identity.json"), None)
    if isinstance(identity, dict):
        for key, label in (
            ("specimen_key", "试样 specimen_key"),
            ("source_identity_note", "源身份说明 source_identity_note"),
        ):
            if key in identity:
                rows.append(
                    f"<dt>{escape(label)}</dt><dd>{escape(display_value(identity[key]))}</dd>"
                )

    stage = next((data for name, data, _ in records if name == "stage_a.json"), None)
    if isinstance(stage, dict) and "status" in stage:
        rows.append(
            f"<dt>A阶段 status</dt><dd>{escape(display_value(stage['status']))}</dd>"
        )

    attention = next(
        (data for name, data, _ in records if name == "attention_status.json"), None
    )
    if isinstance(attention, dict):
        if "status" in attention:
            rows.append(
                f"<dt>注意力 status</dt><dd>{escape(display_value(attention['status']))}</dd>"
            )
        if "reason" in attention:
            rows.append(
                f"<dt>注意力 reason</dt><dd>{escape(display_value(attention['reason']))}</dd>"
            )

    if rows:
        summary = f'<dl class="state-summary">{"".join(rows)}</dl>'
    else:
        summary = '<p class="muted">尚未找到可解析的身份或阶段状态字段。</p>'

    details = "".join(
        render_details(filename, raw)
        for filename, _data, raw in records
        if raw is not None
    )
    if not details:
        details = '<p class="muted">identity.json、stage_a.json、attention_status.json 均缺失。</p>'

    return summary + details, attention, "".join(
        raw for filename, _data, raw in records if filename == "attention_status.json" and raw
    )


def attention_reason(attention: object | None) -> str:
    default = "尚未导出，不展示替代热图"
    if not isinstance(attention, dict):
        return default
    reason = attention.get("reason")
    if reason is None:
        return default
    rendered = display_value(reason).strip()
    return rendered or default


def render_image(filename: str, caption: str) -> str:
    href = escape(filename)
    alt = escape(f"{filename}：{caption}")
    return (
        f'<figure class="diagnostic-figure{" wide" if filename[:2] in ("08", "09", "10", "11") else ""}">'
        f'<a href="{href}" target="_blank" rel="noopener">'
        f'<img src="{href}" alt="{alt}" loading="lazy">'
        "</a>"
        f"<figcaption><strong>{escape(filename)}</strong>：{escape(caption)}</figcaption>"
        "</figure>"
    )


def render_resource_links() -> str:
    links: list[str] = []
    for filename, label in RESOURCE_LABELS:
        path = OUT / filename
        if path.is_file():
            href = escape(filename)
            links.append(f'<li><a href="{href}">{escape(label)}</a></li>')

    if not links:
        return '<p class="muted">尚未找到可下载的诊断表格、JSON 或文本输出。</p>'
    return '<ul class="resource-links">' + "".join(links) + "</ul>"


def render_sequence(attention: object | None) -> str:
    parts: list[str] = []
    raw_response = OUT / "raw_response.txt"
    attention_files = [OUT / filename for filename in ("09_pre_cell_attention_clean.png", "10_pre_cell_attention_numbered.png", "11_attention_with_raw_cells.png")]
    attention_complete = all(path.is_file() for path in attention_files)

    for filename, caption in IMAGE_SPECS:
        if filename == "05_cached_confidence_8x8.png":
            if raw_response.is_file():
                parts.append(render_details("原始回答 raw_response.txt（折叠）", read_text(raw_response)))
            else:
                parts.append('<p class="notice">缺少 raw_response.txt，原始回答未展示。</p>')

        if filename == "09_pre_cell_attention_clean.png" and not attention_complete:
            parts.append(
                '<p class="notice attention-notice">'
                f"B阶段注意力图缺失：{escape(attention_reason(attention))}。"
                "页面不会用替代热图伪造注意力结果。</p>"
            )

        path = OUT / filename
        if path.is_file():
            parts.append(render_image(filename, caption))

    if not parts:
        parts.append(
            '<p class="notice">当前诊断输出目录缺少可展示输入；缺失内容不会被替换为伪造结果。</p>'
        )
    return '<div class="diagnostic-grid">' + "".join(parts) + "</div>"


def render_conclusion() -> str:
    stage, _ = read_json(OUT / "stage_a.json")
    attention, _ = read_json(OUT / "attention_status.json")
    mass, _ = read_json(OUT / "attention_mass.json")
    parts = []
    if isinstance(stage, dict) and stage.get("raw_parsed_cache_csv_all_64_equal"):
        parts.append("原回答→解析→64格特征一致；历史候选图和首动作图逐像素复现，未发现该链路坐标错位。")
    if isinstance(attention, dict) and attention.get("status") == "ATTENTION_EXPORTED":
        parts.append(f"历史没有保存选格前注意力；本次新执行{attention.get('actual_qwen_forwards')}次冻结Qwen前向。完整历史解码过程未重放，只检查首个数字token之前的状态。")
    if isinstance(mass, dict) and "clean" in mass and "numbered" in mass:
        parts.append(f"该query的原始注意力总量：clean图{mass['clean']:.2%}、编号图{mass['numbered']:.2%}。请重点查看图11；中央峰值token跨越多个格，不能把它当作GT或最优动作。")
    return "".join(f"<p>{escape(part)}</p>" for part in parts)


def build_html() -> str:
    state_html, attention, _attention_raw = render_state()
    known_outputs = [OUT / filename for filename, _label in RESOURCE_LABELS]
    known_outputs.extend(OUT / filename for filename, _caption in IMAGE_SPECS)
    has_any_output = any(path.is_file() for path in known_outputs)
    empty_notice = "" if has_any_output else '<p class="notice">尚未找到任何已导出的诊断输入或结果文件。</p>'

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VLM选格与坐标诊断</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", sans-serif;
      color: #20252b;
      background: #f4f6f8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; line-height: 1.55; background: #f4f6f8; }}
    main {{ width: min(1180px, calc(100% - 2rem)); margin: 0 auto; padding: 1.5rem 0 3rem; }}
    h1 {{ margin: 0 0 .35rem; font-size: clamp(1.45rem, 3vw, 2.15rem); }}
    h2 {{ margin: 1.8rem 0 .7rem; font-size: 1.18rem; }}
    p {{ margin: .45rem 0; }}
    .lead, .panel, .diagnostic-figure, .raw-block {{
      background: #fff;
      border: 1px solid #d9dee5;
      border-radius: 6px;
    }}
    .lead, .panel {{ padding: 1rem; }}
    .lead {{ margin-bottom: 1rem; }}
    .muted {{ color: #5e6873; }}
    .notice {{
      margin: .75rem 0;
      padding: .7rem .85rem;
      color: #5b4217;
      background: #fff7df;
      border-left: 4px solid #c99423;
    }}
    .attention-notice {{ margin: 0; grid-column: 1 / -1; }}
    .state-summary {{ display: grid; grid-template-columns: minmax(11rem, 18rem) 1fr; margin: 0; }}
    .state-summary dt, .state-summary dd {{ margin: 0; padding: .25rem 0; border-bottom: 1px solid #edf0f3; }}
    .state-summary dt {{ font-weight: 600; }}
    .state-summary dd {{ overflow-wrap: anywhere; }}
    .raw-block {{ margin-top: .7rem; overflow: hidden; }}
    .raw-block summary {{ cursor: pointer; padding: .6rem .8rem; font-weight: 600; }}
    .raw-block pre {{ margin: 0; padding: .8rem; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; background: #f8fafb; }}
    .diagnostic-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; align-items: start; }}
    .diagnostic-figure {{ margin: 0; overflow: hidden; }}
    .diagnostic-figure.wide {{ grid-column: 1 / -1; }}
    .diagnostic-figure a {{ display: block; background: #eef1f4; }}
    .diagnostic-figure img {{ display: block; width: 100%; height: auto; max-height: 72vh; object-fit: contain; }}
    figcaption {{ padding: .7rem .8rem .8rem; font-size: .94rem; overflow-wrap: anywhere; }}
    .resource-links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: .35rem 1.2rem; padding-left: 1.2rem; }}
    a {{ color: #075e9b; }}
    @media (max-width: 600px) {{
      main {{ width: min(100% - 1rem, 1180px); padding-top: 1rem; }}
      .state-summary {{ display: block; }}
      .state-summary dt {{ margin-top: .35rem; border-bottom: 0; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="lead">
      <h1>VLM选格与坐标诊断</h1>
      <p>本页区分历史候选与本次新提取的注意力。图片可点击查看原尺寸；clean与编号图的注意力使用相同色标。注意力不是损伤概率或CAI价值。</p>
      <p>来源：Hasebe等，cgtnjyggtm v1，CC BY 4.0；诊断叠加由本任务添加。原始方向见01，其余表面视图统一为历史VLM输入方向。</p>
      {render_conclusion()}
      {empty_notice}
    </section>

    <section class="panel" aria-labelledby="state-heading">
      <h2 id="state-heading">身份与阶段状态</h2>
      {state_html}
    </section>

    <section aria-labelledby="sequence-heading">
      <h2 id="sequence-heading">诊断输出（按顺序）</h2>
      {render_sequence(attention)}
    </section>

    <section class="panel" aria-labelledby="resources-heading">
      <h2 id="resources-heading">数据与记录链接</h2>
      {render_resource_links()}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text("\n".join(line.rstrip() for line in build_html().splitlines()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
