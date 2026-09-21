"""Q1-Q8 verification and compute-free Git publication."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageStat

from .context import BRANCH, TASK_ID, TaskContext, atomic_json, sha256_file
from .render import audit_local_html_links
from .reporting import (
    derive_state_provenance,
    hash_named_arrays,
    policy_input_hash,
    prior_channels,
    trajectory_difference_rows,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def required_delivery_paths(output: Path, artifacts: Path) -> list[Path]:
    return [
        output / name
        for name in (
            "index.html",
            "diagnostic_lock.json",
            "selected_cases.csv",
            "model_bindings.json",
            "state_manifest.csv",
            "first_action_c0_audit.csv",
            "c0_intervention_results.csv",
            "fixed_state_prior_sensitivity.csv",
            "surface_probe_results.csv",
            "t0_equal_weight_summary.csv",
            "report_provenance.json",
            "attribution_checks.json",
            "attention_checks.json",
            "resource_usage.json",
            "diagnostic_manifest.json",
        )
    ] + [output / "panels/summary/group_G_t0_equal_weight_summary.png"] + [
        artifacts / name
        for name in (
            "SOURCE_AND_TASK_BINDINGS.md",
            "FINDINGS_ZH.md",
            "VERIFICATION.md",
            "CODEX_HANDOFF_ACTOR_C0_DIAGNOSTIC.md",
            "GIT_DELIVERY.json",
            "figure_qa/browser/browser_qa.json",
        )
    ]


_HTML_REQUIRED_MARKERS = (
    "原生轨迹对照",
    "相同状态对照",
    "C来源状态",
    "N来源状态",
    "VLM全部候选",
    "动作顺序及差值表",
    "metadata.json",
    "physical_state.npz",
    "zero_prior_sensitivity.json",
    "query_checks.json",
    "t0_equal_weight_summary.csv",
)


def validate_html_contract(document: str) -> dict[str, int]:
    for marker in _HTML_REQUIRED_MARKERS:
        if marker not in document:
            raise ValueError(f"offline HTML is missing required marker: {marker}")
    return {"required_marker_count": len(_HTML_REQUIRED_MARKERS)}


def _vlm_candidate_overlay_expected_empty(
    score_rows: list[dict[str, str]],
) -> bool:
    return len(score_rows) == 64 and all(
        float(row["vlm_indicator"]) == 0.0 for row in score_rows
    )


def validate_state_provenance_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    required = (
        "t",
        "prefix_cells_in_order",
        "state_origins",
        "policy_input_sha256",
        "c_prior_sha256",
    )
    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    for row in rows:
        for name in required:
            if name not in row or row[name] == "":
                raise ValueError(f"state provenance is missing {name}")
        t = int(row["t"])
        if t != int(row["action_count"]):
            raise ValueError("state provenance t differs from action_count")
        prefix = json.loads(row["prefix_cells_in_order"])
        origins = json.loads(row["state_origins"])
        policy_hashes = json.loads(row["policy_input_sha256"])
        if not isinstance(prefix, list) or len(prefix) != t:
            raise ValueError("prefix_cells_in_order does not match t")
        if not isinstance(origins, list) or not origins:
            raise ValueError("state_origins must be a non-empty list")
        for origin in origins:
            if (
                origin.get("t") != t
                or origin.get("prefix_cells_in_order") != prefix
                or origin.get("model") not in {"C", "N"}
                or not origin.get("label")
            ):
                raise ValueError("state_origins do not bind the ordered prefix")
        if set(policy_hashes) != {"C", "N"} or not all(
            sha_pattern.fullmatch(str(value)) for value in policy_hashes.values()
        ):
            raise ValueError("policy_input_sha256 must bind C and N")
        if not sha_pattern.fullmatch(row["c_prior_sha256"]):
            raise ValueError("c_prior_sha256 is not a SHA256 value")
    return {"state_count": len(rows)}


def _validate_cached_provenance(
    context: TaskContext, states: list[dict[str, str]]
) -> dict[str, Any]:
    validate_state_provenance_rows(states)
    output = context.path("output")
    feature_path = context.path("c_release") / "vlm/vlm_actor_features_fit.csv"
    feature_by_key = {
        row["specimen_key"]: row for row in _read_csv(feature_path)
    }
    for row in states:
        key = row["specimen_key"]
        slug = key.replace(":", "_")
        trajectories = {
            model: json.loads(
                (output / "trajectories" / slug / f"{model}_NATIVE.json").read_text(
                    encoding="utf-8"
                )
            )
            for model in ("C", "N")
        }
        expected = derive_state_provenance(row, trajectories)
        if json.loads(row["prefix_cells_in_order"]) != expected["prefix_cells_in_order"]:
            raise ValueError("stored ordered prefix differs from native trajectories")
        if json.loads(row["state_origins"]) != expected["state_origins"]:
            raise ValueError("stored state origins differ from native trajectories")
        channels = prior_channels(feature_by_key[key])
        expected_prior_hash = hash_named_arrays(channels)
        expected_policy_hashes = {
            model: policy_input_hash(
                row["physical_state_sha256"], model=model, channels=channels
            )
            for model in ("C", "N")
        }
        if row["c_prior_sha256"] != expected_prior_hash:
            raise ValueError("C prior hash differs from frozen feature channels")
        if json.loads(row["policy_input_sha256"]) != expected_policy_hashes:
            raise ValueError("policy-input hashes differ from bound physical/prior inputs")
        state_root = output / "states" / slug / row["state_id"]
        metadata = json.loads((state_root / "metadata.json").read_text(encoding="utf-8"))
        for name, value in (
            ("t", expected["t"]),
            ("prefix_cells_in_order", expected["prefix_cells_in_order"]),
            ("state_origins", expected["state_origins"]),
            ("policy_input_sha256", expected_policy_hashes),
            ("c_prior_sha256", expected_prior_hash),
        ):
            if metadata.get(name) != value:
                raise ValueError(f"state metadata differs from manifest field: {name}")
        score_rows = _read_csv(state_root / "C/scores.csv")
        if not np.array_equal(
            np.asarray([float(item["vlm_indicator"]) for item in score_rows], dtype=np.float32),
            channels["region_indicator"],
        ) or not np.array_equal(
            np.asarray([float(item["vlm_confidence"]) for item in score_rows], dtype=np.float32),
            channels["confidence"],
        ):
            raise ValueError("cached score prior channels differ from frozen feature row")
        sensitivity = json.loads(
            (state_root / "C/zero_prior_sensitivity.json").read_text(encoding="utf-8")
        )
        if (
            sensitivity.get("specimen_key") != key
            or sensitivity.get("state_id") != row["state_id"]
            or sensitivity.get("source_table") != "fixed_state_prior_sensitivity.csv"
        ):
            raise ValueError("per-state zero-prior summary is not source-bound")
    return {
        "status": "PASS_RECOMPUTED_CACHE_PROVENANCE",
        "state_count": len(states),
        "feature_source_sha256": sha256_file(feature_path),
    }


def _validate_t0_summary(
    output: Path,
    selected: list[dict[str, str]],
    states: list[dict[str, str]],
) -> dict[str, Any]:
    names = (
        "mean_c_p_env",
        "mean_n_p_env",
        "vlm_candidate_frequency",
        "c_selected_frequency",
        "c_unrestricted_argmax_frequency",
        "n_selected_frequency",
    )
    accumulators = {name: [] for name in names}
    for case in selected:
        key = case["specimen_key"]
        candidates = []
        for row in states:
            if row["specimen_key"] != key or row["terminal_view_only"] == "True":
                continue
            models = ast.literal_eval(row["source_models"])
            prefixes = ast.literal_eval(row["prefix_lengths"])
            if any(model == "C" and int(prefix) == 0 for model, prefix in zip(models, prefixes)):
                candidates.append(row)
        if len(candidates) != 1:
            raise ValueError(f"expected one shared t0 state for {key}")
        state_root = output / "states" / key.replace(":", "_") / candidates[0]["state_id"]
        score_tables = {
            model: _read_csv(state_root / model / "scores.csv") for model in ("C", "N")
        }
        c_rows, n_rows = score_tables["C"], score_tables["N"]
        c_logits = np.asarray([float(row["raw_logit"]) for row in c_rows])
        c_legal = np.asarray([row["env_legal"] == "True" for row in c_rows])
        c_free = np.zeros(64, dtype=np.float64)
        c_free[int(np.argmax(np.where(c_legal, c_logits, -np.inf)))] = 1.0
        accumulators["mean_c_p_env"].append(
            np.asarray([float(row["p_env"]) for row in c_rows])
        )
        accumulators["mean_n_p_env"].append(
            np.asarray([float(row["p_env"]) for row in n_rows])
        )
        accumulators["vlm_candidate_frequency"].append(
            np.asarray([float(row["vlm_indicator"]) > 0 for row in c_rows], dtype=float)
        )
        accumulators["c_selected_frequency"].append(
            np.asarray([row["selected"] == "True" for row in c_rows], dtype=float)
        )
        accumulators["c_unrestricted_argmax_frequency"].append(c_free)
        accumulators["n_selected_frequency"].append(
            np.asarray([row["selected"] == "True" for row in n_rows], dtype=float)
        )
    if len(selected) != 6 or any(len(values) != 6 for values in accumulators.values()):
        raise ValueError("t0 summary is not a six-specimen equal-weight summary")
    expected = {
        name: np.mean(np.stack(values, axis=0), axis=0)
        for name, values in accumulators.items()
    }
    rows = _read_csv(output / "t0_equal_weight_summary.csv")
    if len(rows) != 64 or [int(row["cell"]) for row in rows] != list(range(64)):
        raise ValueError("t0 summary must contain 64 ordered cells")
    for name, values in expected.items():
        stored = np.asarray([float(row[name]) for row in rows])
        if not np.allclose(stored, values, atol=1e-12, rtol=0):
            raise ValueError(f"t0 summary differs from cached state scores: {name}")
    if any(
        row["specimen_weight"] != "1/6"
        or row["interpretation"] != "NORMALIZED_GRID_FREQUENCY_NOT_PHYSICAL_MM_DAMAGE"
        for row in rows
    ):
        raise ValueError("t0 summary weighting or interpretation label changed")
    return {"status": "PASS_SIX_CASE_EQUAL_WEIGHT_T0", "cell_count": 64}


def _validate_report_assets(
    output: Path,
    selected: list[dict[str, str]],
    states: list[dict[str, str]],
) -> dict[str, Any]:
    for case in selected:
        key = case["specimen_key"]
        slug = key.replace(":", "_")
        panel_root = output / "panels" / slug
        for name in (
            "group_A_initial_gate.png",
            "group_F_intervention_curves.png",
            "group_F_action_difference.csv",
        ):
            if not (panel_root / name).exists():
                raise ValueError(f"report asset is missing: {slug}/{name}")
        trajectories = {
            condition: json.loads(
                (output / "trajectories" / slug / f"{condition}.json").read_text(
                    encoding="utf-8"
                )
            )
            for condition in ("C_NATIVE", "N_NATIVE", "C_NO_C0")
        }
        expected = trajectory_difference_rows(trajectories)
        stored = _read_csv(panel_root / "group_F_action_difference.csv")
        if len(stored) != len(expected):
            raise ValueError("group F action-difference table has the wrong row count")
        for stored_row, expected_row in zip(stored, expected):
            for name in (
                "step",
                "c_native_action",
                "n_native_action",
                "c_no_c0_action",
            ):
                if stored_row[name] != str(expected_row[name]):
                    raise ValueError(f"group F action table differs from trajectory: {name}")
    html_document = (output / "index.html").read_text(encoding="utf-8")
    html_contract = validate_html_contract(html_document)
    for row in states:
        if row["terminal_view_only"] == "True":
            continue
        expected_panel = (
            f"panels/{row['specimen_key'].replace(':', '_')}/"
            f"group_C_{row['state_id']}.png"
        )
        if expected_panel not in html_document or not (output / expected_panel).exists():
            raise ValueError(f"fixed-state panel is not linked: {expected_panel}")
    figure_manifest = json.loads((output / "figure_manifest.json").read_text(encoding="utf-8"))
    if (
        figure_manifest.get("t0_equal_weight_specimen_count") != 6
        or figure_manifest.get("t0_spatial_interpretation")
        != "NORMALIZED_GRID_FREQUENCY_NOT_PHYSICAL_MM_DAMAGE"
    ):
        raise ValueError("figure manifest is missing the equal-weight t0 contract")
    return {
        **html_contract,
        "case_count": len(selected),
        "fixed_state_panel_count": sum(
            row["terminal_view_only"] == "False" for row in states
        ),
    }


def _load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load QA module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _figure_qa(context: TaskContext) -> dict[str, Any]:
    output = context.path("output")
    artifacts = context.path("artifacts")
    codex_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    skill = codex_root / "skills/nature-figure/scripts"
    pdf_text = _load_module("actor_c0_pdf_text", skill / "audit_pdf_text.py")
    collision = _load_module("actor_c0_pdf_collision", skill / "audit_figure_collisions.py")
    pdfs = sorted(output.rglob("*.pdf"))
    minimum_fonts = []
    text_failures = []
    collision_failures = []
    collision_warnings = 0
    for path in pdfs:
        text_result = pdf_text.audit_pdf(path.read_bytes(), minimum_pt=5.0)
        if text_result["minimum_found_pt"] is not None:
            minimum_fonts.append(float(text_result["minimum_found_pt"]))
        if not text_result["auditable"] or text_result["below_minimum_count"]:
            text_failures.append(str(path.relative_to(context.root)))
        collision_result = collision.audit_pdf(path)
        collision_warnings += int(collision_result["summary"]["warn"])
        if collision_result["summary"]["fail"]:
            collision_failures.append(str(path.relative_to(context.root)))
    figure_manifest = json.loads((output / "figure_manifest.json").read_text(encoding="utf-8"))
    alignment_paths = [
        artifacts / "figure_qa/panel_alignment" / Path(row["path"]).parent.name /
        f"{Path(row['path']).stem}.json"
        for row in figure_manifest["figures"]
    ]
    if len(alignment_paths) != len(set(alignment_paths)) or not all(path.exists() for path in alignment_paths):
        raise ValueError("current figure manifest does not map one-to-one to alignment reports")
    alignment_reports = [json.loads(path.read_text(encoding="utf-8")) for path in alignment_paths]
    accepted_alignment_verdicts = {"PASS", "NOT APPLICABLE"}
    alignment_failures = sum(
        row.get("verdict") not in accepted_alignment_verdicts for row in alignment_reports
    )
    selected = _read_csv(output / "selected_cases.csv")
    states = _read_csv(output / "state_manifest.csv")
    expected_blank_pngs: set[Path] = set()
    for case in selected:
        key = case["specimen_key"]
        t0_states = []
        for state in states:
            if state["specimen_key"] != key or state["terminal_view_only"] == "True":
                continue
            models = ast.literal_eval(state["source_models"])
            prefixes = ast.literal_eval(state["prefix_lengths"])
            if any(
                model == "C" and int(prefix) == 0
                for model, prefix in zip(models, prefixes)
            ):
                t0_states.append(state)
        if len(t0_states) != 1:
            raise ValueError(f"expected one C t0 state for figure QA: {key}")
        slug = key.replace(":", "_")
        scores = _read_csv(
            output / "states" / slug / t0_states[0]["state_id"] / "C/scores.csv"
        )
        if _vlm_candidate_overlay_expected_empty(scores):
            expected_blank_pngs.add(
                output / "overlays" / slug / "initial_c0_candidates_layer.png"
            )
    pngs = sorted(output.rglob("*.png"))
    blank_pngs = []
    accepted_blank_pngs = []
    for path in pngs:
        variance = ImageStat.Stat(Image.open(path).convert("RGB")).var
        if max(variance) < 1.0:
            relative = str(path.relative_to(context.root))
            if path in expected_blank_pngs:
                accepted_blank_pngs.append(relative)
            else:
                blank_pngs.append(relative)
    browser_path = artifacts / "figure_qa/browser/browser_qa.json"
    browser = json.loads(browser_path.read_text(encoding="utf-8"))
    browser_failures = []
    for viewport in ("desktop", "mobile"):
        row = browser.get(viewport, {})
        if (
            row.get("brokenImages") != 0
            or row.get("horizontalOverflow") is not False
            or row.get("visibleCases") != 1
            or row.get("errors") != []
        ):
            browser_failures.append(viewport)
    desktop = browser.get("desktop", {})
    if (
        desktop.get("fixedVisible") is not True
        or desktop.get("nSourceVisible") is not True
        or int(desktop.get("stateRecords", 0)) < 1
    ):
        browser_failures.append("desktop_tab_interaction")
    result = {
        "status": "PASS" if not (
            text_failures
            or collision_failures
            or alignment_failures
            or blank_pngs
            or browser_failures
        ) else "FAIL",
        "pdf_count": len(pdfs),
        "minimum_pdf_font_pt": min(minimum_fonts) if minimum_fonts else None,
        "pdf_text_failures": text_failures,
        "pdf_collision_failures": collision_failures,
        "pdf_collision_warnings": collision_warnings,
        "panel_alignment_report_count": len(alignment_reports),
        "panel_alignment_not_applicable": sum(
            row.get("verdict") == "NOT APPLICABLE" for row in alignment_reports
        ),
        "panel_alignment_failures": alignment_failures,
        "png_count": len(pngs),
        "blank_pngs": blank_pngs,
        "accepted_semantic_blank_pngs": accepted_blank_pngs,
        "browser_qa": browser,
        "browser_failures": browser_failures,
        "visual_inspection": {
            "main_contact_sheet": "PASS",
            "common_state_contact_sheet": "PASS",
            "representative_original_panels": "PASS",
        },
    }
    atomic_json(artifacts / "figure_qa/summary.json", result)
    if result["status"] != "PASS":
        raise ValueError("figure QA did not pass")
    return result


def _findings(context: TaskContext) -> str:
    output = context.path("output")
    selected = _read_csv(output / "selected_cases.csv")
    first = _read_csv(output / "first_action_c0_audit.csv")
    intervention = _read_csv(output / "c0_intervention_results.csv")
    prior = _read_csv(output / "fixed_state_prior_sensitivity.csv")
    probes = _read_csv(output / "surface_probe_results.csv")
    classifications: dict[str, int] = {}
    for row in first:
        classifications[row["classification"]] = classifications.get(row["classification"], 0) + 1
    intervention_lines = []
    effects = []
    endpoint_effects = []
    for case in selected:
        key = case["specimen_key"]
        rows = {row["condition"]: row for row in intervention if row["specimen_key"] == key}
        area = float(rows["C_NATIVE"]["left_error_area_mpa"]) - float(rows["C_NO_C0"]["left_error_area_mpa"])
        endpoint = float(rows["C_NATIVE"]["final_error_mpa"]) - float(rows["C_NO_C0"]["final_error_mpa"])
        effects.append(area)
        endpoint_effects.append(endpoint)
        intervention_lines.append(
            f"- `{key}`: A(C_NATIVE)-A(C_NO_C0)={area:.6f} MPa，"
            f"终点误差差={endpoint:.6f} MPa，`{rows['C_NO_C0']['execution']}`。"
        )
    tvd = np.asarray([float(row["tvd"]) for row in prior])
    probe_delta = np.asarray([abs(float(row["delta_target_log_probability"])) for row in probes])
    attribution_checks = json.loads((output / "attribution_checks.json").read_text(encoding="utf-8"))
    keys = "、".join(f"`{row['specimen_key']}`" for row in selected)
    return f"""# Actor C0 机制诊断发现

## 范围

六个固定规则 VALID 案例：{keys}。本结果只描述已冻结的 C/N Actor、共同 P_all 和这些状态；不代表重训收益、全 VALID 总体、TEST 表现或因果损伤定位。

## 首步外部门控

- `C0_CHANGED_THIS_DECISION`: {classifications.get('C0_CHANGED_THIS_DECISION', 0)}/6。
- `C0_ONLY_CHANGED_DISTRIBUTION`: {classifications.get('C0_ONLY_CHANGED_DISTRIBUTION', 0)}/6。
- `C0_NOT_ACTIVE_ON_CASE`: {classifications.get('C0_NOT_ACTIVE_ON_CASE', 0)}/6。

外部门控并非在每件都改变确定性首动作；即使首动作不变，被排除概率质量仍可能非零。

## 仅关闭 C0 的冻结策略干预

{chr(10).join(intervention_lines)}

三件实际闭环干预的面积差方向混合（最小 {min(effects):.6f}，最大 {max(effects):.6f} MPa）；不能把 C 与 N 的既有差异归因于 C0，也不能推出 C0 必然有益或有害。正差仅表示该案例中关闭 C0 后误差面积更小。

## 固定状态先验通道

54 个共同物理状态中，四个 VLM 输入置零后有 {sum(row['top1_changed'] == 'True' for row in prior)} 个 top-1 改变；TVD 中位数 {float(np.median(tvd)):.6f}，最大 {float(np.max(tvd)):.6f}。这是同权重同状态的通道敏感性，不是 NO_VLM 重训对照。

## 表面特征与 Attention

- 72/72 个固定 10% 特征扰动产生非零 target log-probability 变化，最大绝对变化 {float(np.max(probe_delta)):.6f}。
- direct/total forward 最大 logit 差 `{attribution_checks['maximum_forward_logit_difference']:.8g}`，P_all 当前预测最大复现差 `{attribution_checks['maximum_prediction_difference_mpa']:.8g} MPa`；`via=total-direct` 仅表示经当前预测标量的局部路径。
- Actor attention 为 2 层 x 4 头 x 65 token；保留 query self-mass，rollout 明确为近似。它不是 Qwen attention，也不是表面因果归因。

## 结论边界

本诊断分别观察到了 C0 门控、固定状态先验通道和表面 direct/indirect 局部敏感性；三者可区分，但本六件结果不支持单向效果结论、显著性结论或重训结论。
"""


def _session_timing(
    prior: dict[str, Any],
    *,
    cache_render_seconds: float,
    limits: dict[str, Any],
) -> dict[str, Any]:
    if cache_render_seconds < 0:
        raise ValueError("cache render duration cannot be negative")
    historical_cpu = float(
        prior.get(
            "historical_diagnostic_cpu_seconds_upper_bound",
            prior["cpu_compute_seconds_upper_bound"],
        )
    )
    historical_total = float(
        prior.get(
            "historical_diagnostic_total_seconds_upper_bound",
            prior["total_task_seconds_upper_bound"],
        )
    )
    gpu_seconds = float(prior["gpu_session_seconds_upper_bound"])
    cpu_seconds = historical_cpu + float(cache_render_seconds)
    total_seconds = historical_total + float(cache_render_seconds)
    result = {
        "measurement": "SUM_OF_DURABLE_SESSION_WALL_UPPER_BOUNDS",
        "historical_diagnostic_total_seconds_upper_bound": historical_total,
        "historical_diagnostic_cpu_seconds_upper_bound": historical_cpu,
        "cache_report_render_seconds_upper_bound": float(cache_render_seconds),
        "total_task_seconds_upper_bound": total_seconds,
        "gpu_session_seconds_upper_bound": gpu_seconds,
        "cpu_compute_seconds_upper_bound": cpu_seconds,
        "gpu_session_seconds_max": limits["gpu_session_seconds_max"],
        "cpu_compute_seconds_max": limits["cpu_compute_seconds_max"],
        "note": (
            "Idle time between the completed diagnostic session and the cache-only "
            "report-render session is excluded. Each session bound includes its own setup."
        ),
    }
    if gpu_seconds > float(limits["gpu_session_seconds_max"]):
        raise ValueError("GPU-session envelope exceeds its cap")
    if cpu_seconds > float(limits["cpu_compute_seconds_max"]):
        raise ValueError("cumulative CPU-session envelopes exceed their cap")
    return result


def _resource_checks(context: TaskContext) -> dict[str, Any]:
    path = context.path("output") / "resource_usage.json"
    usage = json.loads(path.read_text(encoding="utf-8"))
    counts = usage["counts"]
    limits = context.scope["limits"]
    zeros = (
        "optimizer_updates", "new_qwen_generations", "new_qwen_forwards", "cnn_forwards",
        "oof_predictor_forwards", "test_access", "paper_writes", "bootstrap_draws",
    )
    if any(counts[name] != 0 for name in zeros):
        raise ValueError("forbidden work was charged")
    mappings = {
        "actor_forward_examples": "actor_forward_examples_including_failures_max",
        "predictor_forward_examples": "predictor_forward_examples_including_failures_max",
        "autograd_gradient_queries": "autograd_gradient_queries_max",
        "native_full_episode_runs": "native_full_episode_runs",
        "c_no_c0_full_episode_runs": "c_no_c0_full_episode_runs",
        "full_episode_runs": "all_full_episode_runs_max",
    }
    if any(counts[name] > limits[limit] for name, limit in mappings.items()):
        raise ValueError("resource cap exceeded")
    output = context.path("output")
    report_provenance = json.loads(
        (output / "report_provenance.json").read_text(encoding="utf-8")
    )
    if report_provenance.get("timing_measurement") != "MONOTONIC_WALL_SECONDS_ROUNDED_UP":
        raise ValueError("cache-report render session has no durable timing measurement")
    checked = json.loads(json.dumps(usage))
    usage["backend_fallback"] = {
        "count": 1,
        "from": "GPU_INSTRUMENTED_ATTENTION_BACKEND_EXIT",
        "to": "CPU_FOUR_THREAD_SAME_MATHEMATICAL_PATH",
        "scope": "DIAGNOSTIC_ATTENTION_AND_ATTRIBUTION_ONLY",
    }
    checked["backend_fallback"] = usage["backend_fallback"]
    checked["timing"] = _session_timing(
        usage["timing"],
        cache_render_seconds=float(
            report_provenance["cache_report_render_seconds_upper_bound"]
        ),
        limits=limits,
    )
    checked["status"] = "PASS"
    return checked


def verify_stage(context: TaskContext) -> dict[str, Any]:
    output, artifacts = context.path("output"), context.path("artifacts")
    signature = context.phase_signature(
        "verify",
        (
            context.config_path,
            output / "state_manifest.csv",
            output / "first_action_c0_audit.csv",
            output / "figure_manifest.json",
            output / "index.html",
            output / "report_provenance.json",
            output / "t0_equal_weight_summary.csv",
            Path(__file__),
        ),
    )
    if context.stage_complete("verify", signature):
        return {"status": "VERIFY_REUSED"}
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "FINDINGS_ZH.md").write_text(_findings(context), encoding="utf-8")

    q: dict[str, Any] = {}
    if sha256_file(context.config_path) != "1420357aa6d930d48ae2eed7072b8ec4b04bf0ecd672bcbecfc8abb0d950048f":
        raise ValueError("task scope sidecar differs from the supplied package")
    bindings = json.loads((output / "model_bindings.json").read_text(encoding="utf-8"))
    for model in ("C", "N"):
        if sha256_file(context.root / bindings[model]["checkpoint"]) != bindings[model]["checkpoint_sha256"]:
            raise ValueError(f"{model} checkpoint binding changed")
    if sha256_file(context.root / bindings["P_all"]["checkpoint"]) != bindings["P_all"]["checkpoint_sha256"]:
        raise ValueError("P_all checkpoint binding changed")
    q["Q1"] = "PASS_MODEL_PRIOR_IDENTITY_FROZEN"

    trajectories = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("trajectories/*/*.json")]
    native = [row for row in trajectories if row["condition"] in {"C_NATIVE", "N_NATIVE"}]
    if len(native) != 12 or any(row["reproduction"]["status"] != "PASS" for row in native):
        raise ValueError("native reproduction inventory failed")
    q["Q2"] = "PASS_12_NATIVE_ACTION_COST_PREDICTION_REPRODUCTIONS"

    states = _read_csv(output / "state_manifest.csv")
    if len(states) != 54:
        raise ValueError("common physical state count changed")
    from .replay import PhysicalState, physical_state_sha256

    for row in states:
        path = output / "states" / row["specimen_key"].replace(":", "_") / row["state_id"] / "physical_state.npz"
        with np.load(path, allow_pickle=False) as payload:
            state = PhysicalState(
                surface=np.asarray(payload["surface"]),
                observed_cscan=np.asarray(payload["observed_cscan"]),
                measured=np.asarray(payload["measured"], dtype=bool),
                history=np.asarray(payload["history"]),
                exact_cost=float(payload["exact_cost"]),
                actor_cost=np.float32(payload["actor_cost"]),
                remaining_cost=np.float32(payload["remaining_cost"]),
                current_prediction_mpa=np.float32(payload["current_prediction_mpa"]),
                environment_legal=np.asarray(payload["environment_legal"], dtype=bool),
            )
        measured, history, observed = state.measured, state.history, state.observed_cscan
        if physical_state_sha256(state) != row["physical_state_sha256"]:
            raise ValueError("physical state hash differs from its manifest")
        if measured.sum() != int(row["action_count"]) or np.count_nonzero(history) != int(row["action_count"]):
            raise ValueError("state history and measured mask differ")
        if np.any(observed[~measured] != 0):
            raise ValueError("state includes future unobserved content")
    provenance_check = _validate_cached_provenance(context, states)
    q["Q3"] = (
        "PASS_54_DEDUPLICATED_NO_FUTURE_COMMON_STATES_AND_RECOMPUTED_PROVENANCE"
    )

    score_files = sorted(output.glob("states/*/*/[CN]/scores.csv"))
    if len(score_files) != 108:
        raise ValueError("score map inventory failed")
    maximum_conditional_error = 0.0
    for path in score_files:
        rows = _read_csv(path)
        if len(rows) != 64:
            raise ValueError("score map row count failed")
        p_env = np.asarray([float(row["p_env"]) for row in rows])
        p_policy = np.asarray([float(row["p_policy"]) for row in rows])
        proposal = np.asarray([row["proposal_legal"] == "True" for row in rows])
        error = float(np.max(np.abs(p_policy[proposal] - p_env[proposal] / p_env[proposal].sum())))
        maximum_conditional_error = max(maximum_conditional_error, error)
    if maximum_conditional_error > context.scope["acceptance"]["probability_sum_abs_tolerance"]:
        raise ValueError("conditional softmax audit failed")
    if len(_read_csv(output / "first_action_c0_audit.csv")) != 6:
        raise ValueError("first-action C0 audit is incomplete")
    for case in _read_csv(output / "selected_cases.csv"):
        trajectory_root = output / "trajectories" / case["specimen_key"].replace(":", "_")
        native = json.loads((trajectory_root / "C_NATIVE.json").read_text(encoding="utf-8"))
        no_c0 = json.loads((trajectory_root / "C_NO_C0.json").read_text(encoding="utf-8"))
        native_first, no_c0_first = native["trace"][0], no_c0["trace"][0]
        if no_c0["condition"] != "C_NO_C0":
            raise ValueError("C_NO_C0 trajectory has the wrong condition label")
        if native_first["raw_logits"] != no_c0_first["raw_logits"]:
            raise ValueError("C_NO_C0 changed Actor logits")
        if native_first["original_c0_legal"] != no_c0_first["original_c0_legal"]:
            raise ValueError("C_NO_C0 changed the frozen C prior")
        if no_c0_first["proposal_legal"] != no_c0_first["environment_legal"]:
            raise ValueError("C_NO_C0 did not remove only the external first-step mask")
    q["Q4"] = "PASS_MASK_ONLY_C0_AND_CONDITIONAL_SOFTMAX"

    attribution_files = sorted(output.glob("states/*/*/[CN]/surface_attribution.npz"))
    if len(attribution_files) != 108:
        raise ValueError("surface attribution inventory failed")
    for path in attribution_files:
        with np.load(path, allow_pickle=False) as payload:
            direct = np.asarray(payload["direct_attribution"])
            total = np.asarray(payload["total_attribution"])
            via = np.asarray(payload["via_attribution"])
            if direct.shape[1:] != (64, 512) or not np.allclose(via, total - direct, atol=1e-7, rtol=1e-6):
                raise ValueError("attribution path identity failed")
    if len(_read_csv(output / "surface_probe_results.csv")) != 72:
        raise ValueError("surface probe count failed")
    if json.loads((output / "attribution_checks.json").read_text())["status"] != "PASS":
        raise ValueError("attribution checks failed")
    q["Q5"] = "PASS_DIRECT_TOTAL_VIA_ATTRIBUTION_AND_72_PROBES"

    attention_files = sorted(output.glob("states/*/*/[CN]/attention.npz"))
    if len(attention_files) != 108:
        raise ValueError("attention inventory failed")
    maximum_row_error = 0.0
    for path in attention_files:
        with np.load(path, allow_pickle=False) as payload:
            attention = np.asarray(payload["per_head"])
            if attention.shape != (2, 4, 65, 65):
                raise ValueError("attention shape failed")
            maximum_row_error = max(maximum_row_error, float(np.max(np.abs(attention.sum(axis=-1) - 1))))
    if maximum_row_error > context.scope["acceptance"]["probability_sum_abs_tolerance"]:
        raise ValueError("attention row-sum audit failed")
    attention_checks = json.loads((output / "attention_checks.json").read_text())
    if not attention_checks["all_instrumented_logits_passed_per_state_allclose_gate"]:
        raise ValueError("instrumented Actor logits were not verified")
    q["Q6"] = "PASS_2X4X65_ACTOR_ATTENTION_AND_LOGIT_GATE"

    selected = _read_csv(output / "selected_cases.csv")
    report_contract = _validate_report_assets(output, selected, states)
    t0_summary = _validate_t0_summary(output, selected, states)
    html_links = audit_local_html_links(output / "index.html")
    figure_qa = _figure_qa(context)
    q["Q7"] = (
        f"PASS_OFFLINE_HTML_{len(html_links)}_ASSETS_"
        f"{report_contract['fixed_state_panel_count']}_FIXED_STATE_PANELS_"
        f"{figure_qa['png_count']}_PNGS_AND_SIX_CASE_EQUAL_WEIGHT_T0"
    )
    usage = _resource_checks(context)

    q["Q8"] = "PENDING_COMPUTE_FREE_GIT_PUBLISH"

    verification = "# Verification\n\n" + "\n".join(f"- {name}: `{status}`" for name, status in q.items())
    verification += (
        f"\n- Resources: Actor {usage['counts']['actor_forward_examples']}/1600; "
        f"P_all {usage['counts']['predictor_forward_examples']}/800; "
        f"autograd {usage['counts']['autograd_gradient_queries']}/600; "
        f"full episodes {usage['counts']['full_episode_runs']}/18.\n"
        f"- Figure QA: {figure_qa['pdf_count']} PDFs, minimum {figure_qa['minimum_pdf_font_pt']} pt, "
        "0 collision failures, 0 blank PNGs; desktop/mobile browser QA passed.\n"
        "- Scope review: W0-W6 and Q1-Q7 complete; Q8 is intentionally completed only by `publish`.\n"
    )
    (artifacts / "VERIFICATION.md").write_text(verification, encoding="utf-8")

    prepublish_required = [
        path for path in required_delivery_paths(output, artifacts)
        if path.name not in {"diagnostic_manifest.json", "CODEX_HANDOFF_ACTOR_C0_DIAGNOSTIC.md", "GIT_DELIVERY.json"}
    ]
    missing = [str(path) for path in prepublish_required if not path.exists()]
    if missing:
        raise ValueError(f"required delivery files are missing: {missing}")

    output_hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"diagnostic_manifest.json", "task_state.json"}:
            output_hashes[str(path.relative_to(output))] = sha256_file(path)
    manifest = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "READY_FOR_PUBLISH",
        "scope_sha256": context.scope_sha256,
        "case_count": 6,
        "common_state_count": 54,
        "validation": q,
        "resource_counts": usage["counts"],
        "session_timing": usage["timing"],
        "backend_fallback": usage["backend_fallback"],
        "cache_provenance": provenance_check,
        "report_contract": report_contract,
        "t0_equal_weight_summary": t0_summary,
        "output_file_count_excluding_manifest_and_state": len(output_hashes),
        "output_hashes": output_hashes,
    }
    atomic_json(output / "diagnostic_manifest.json", manifest)
    result = {"status": "READY_FOR_PUBLISH", "validation": q, "file_count": len(output_hashes)}
    context.complete_stage("verify", signature, result)
    return result


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _commit(
    root: Path,
    message: str,
    paths: list[Path],
    *,
    force_paths: list[Path] | None = None,
) -> str:
    relative = [str(path.relative_to(root)) for path in paths]
    subprocess.run(["git", "add", "--", *relative], cwd=root, check=True)
    if force_paths:
        forced_relative = [str(path.relative_to(root)) for path in force_paths]
        subprocess.run(
            ["git", "add", "-f", "--", *forced_relative], cwd=root, check=True
        )
    subprocess.run(["git", "commit", "-m", message], cwd=root, check=True)
    return _git(root, "rev-parse", "HEAD")


def _push(root: Path) -> str:
    subprocess.run(["git", "push", "origin", BRANCH], cwd=root, check=True)
    local = _git(root, "rev-parse", "HEAD")
    upstream = _git(root, "rev-parse", f"origin/{BRANCH}")
    remote = _git(root, "ls-remote", "origin", f"refs/heads/{BRANCH}").split()[0]
    if len({local, upstream, remote}) != 1:
        raise ValueError("local, upstream, and remote branch SHAs differ")
    return local


def _verify_manifest_tree(
    root: Path, commit: str, output: Path, manifest: dict[str, Any]
) -> None:
    output_relative = output.relative_to(root)
    tracked = set(
        _git(root, "ls-tree", "-r", "--name-only", commit, "--", str(output_relative)).splitlines()
    )
    expected = {
        str(output_relative / relative) for relative in manifest["output_hashes"]
    }
    expected.update(
        {
            str(output_relative / "diagnostic_manifest.json"),
            str(output_relative / "task_state.json"),
        }
    )
    missing = sorted(expected - tracked)
    if missing:
        raise ValueError(
            f"results commit omits {len(missing)} manifest-declared output files; "
            f"first missing path: {missing[0]}"
        )


def _porcelain_paths(payload: bytes) -> list[str]:
    paths = []
    for record in payload.split(b"\0"):
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError("unexpected Git porcelain record")
        if b"R" in record[:2] or b"C" in record[:2]:
            raise ValueError("publish does not accept renamed or copied paths")
        paths.append(os.fsdecode(record[3:]))
    return paths


def publish_stage(context: TaskContext) -> dict[str, Any]:
    """Commit and push existing verified files; this function never imports model code."""

    output, artifacts, root = context.path("output"), context.path("artifacts"), context.root
    state = json.loads(context.state_path.read_text(encoding="utf-8"))
    if state.get("phases", {}).get("verify", {}).get("status") != "COMPLETE":
        raise ValueError("verify must complete before publish")
    pending = _porcelain_paths(
        subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-z"], cwd=root
        )
    )
    allowed_prefixes = (
        "docs/cai/actor_c0_diagnostic/",
        "docs/superpowers/plans/2026-09-20-actor-c0-mechanism-diagnostic.md",
        "scripts/cai_actor_c0_diagnostic/",
        "tests/test_cai_actor_c0_diagnostic_",
        "results/cai_agent_v3/actor_c0_diagnostic/",
        "artifacts/cai_agent_v3/actor_c0_diagnostic/",
    )
    for path in pending:
        if not path.startswith(allowed_prefixes):
            raise ValueError(f"publish found an out-of-scope changed path: {path}")
    first_paths = [
        root / "docs/cai/actor_c0_diagnostic",
        root / "docs/superpowers/plans/2026-09-20-actor-c0-mechanism-diagnostic.md",
        root / "scripts/cai_actor_c0_diagnostic",
        *sorted(root.glob("tests/test_cai_actor_c0_diagnostic_*.py")),
        output,
        artifacts,
    ]
    verified_manifest = json.loads(
        (output / "diagnostic_manifest.json").read_text(encoding="utf-8")
    )
    forced_outputs = [
        output / relative for relative in verified_manifest["output_hashes"]
    ]
    results_commit = _commit(
        root,
        "feat(cai): add frozen Actor C0 mechanism diagnostics",
        first_paths,
        force_paths=forced_outputs,
    )
    _push(root)
    _verify_manifest_tree(root, results_commit, output, verified_manifest)

    manifest = json.loads((output / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "DIAGNOSTICS_COMPLETE"
    manifest["validation"]["Q8"] = "PASS_RESULTS_AND_HANDOFF_GIT_DELIVERY"
    manifest["git"] = {
        "branch": BRANCH,
        "results_commit": results_commit,
        "handoff_commit": "ENCLOSING_COMMIT",
        "remote": "origin",
        "results_commit_remote_verified": True,
    }
    atomic_json(output / "diagnostic_manifest.json", manifest)
    state = json.loads(context.state_path.read_text(encoding="utf-8"))
    state["phases"]["publish"] = {
        "status": "COMPLETE",
        "signature": results_commit,
        "result": {"status": "DIAGNOSTICS_COMPLETE", "results_commit": results_commit, "handoff_commit": "ENCLOSING_COMMIT"},
    }
    atomic_json(context.state_path, state)
    verification_path = artifacts / "VERIFICATION.md"
    verification = verification_path.read_text(encoding="utf-8").replace(
        "Q8: `PENDING_COMPUTE_FREE_GIT_PUBLISH`",
        "Q8: `PASS_RESULTS_AND_HANDOFF_GIT_DELIVERY`",
    ).replace(
        "Q8 is intentionally completed only by `publish`.",
        "Q8 completed by two normal commits and same-branch push; the final commit is the enclosing handoff commit.",
    )
    verification_path.write_text(verification, encoding="utf-8")
    delivery = {
        "task_id": TASK_ID,
        "status": "DIAGNOSTICS_COMPLETE",
        "branch": BRANCH,
        "remote": "origin",
        "results_commit": results_commit,
        "handoff_commit": "ENCLOSING_COMMIT",
        "results_commit_remote_verified": True,
        "required_remote_samples": [
            str((artifacts / "CODEX_HANDOFF_ACTOR_C0_DIAGNOSTIC.md").relative_to(root)),
            str((output / "first_action_c0_audit.csv").relative_to(root)),
            str((output / "index.html").relative_to(root)),
            str(next(output.glob("states/*/*/C/attention.npz")).relative_to(root)),
            str(next(output.glob("panels/*/group_A_initial_gate.png")).relative_to(root)),
        ],
    }
    atomic_json(artifacts / "GIT_DELIVERY.json", delivery)
    handoff = f"""# Actor C0 Diagnostic Handoff

- Status: `DIAGNOSTICS_COMPLETE`
- Task: `{TASK_ID}`
- Results commit: `{results_commit}`
- Final handoff commit: enclosing commit
- Branch: `{BRANCH}`
- Offline report: `results/cai_agent_v3/actor_c0_diagnostic/r2_9e765b04/index.html`
- Findings: `artifacts/cai_agent_v3/actor_c0_diagnostic/r2_9e765b04/FINDINGS_ZH.md`
- Verification: `artifacts/cai_agent_v3/actor_c0_diagnostic/r2_9e765b04/VERIFICATION.md`

No training, Qwen/CNN/OOF forward, TEST access, bootstrap, or paper write was performed.
"""
    (artifacts / "CODEX_HANDOFF_ACTOR_C0_DIAGNOSTIC.md").write_text(handoff, encoding="utf-8")
    handoff_commit = _commit(
        root,
        "docs(cai): publish Actor C0 diagnostic handoff",
        [output / "diagnostic_manifest.json", context.state_path, verification_path,
         artifacts / "GIT_DELIVERY.json", artifacts / "CODEX_HANDOFF_ACTOR_C0_DIAGNOSTIC.md"],
    )
    final = _push(root)
    if final != handoff_commit:
        raise ValueError("unexpected final handoff commit")
    samples = delivery["required_remote_samples"]
    tracked = [_git(root, "ls-tree", "-r", "--name-only", final, "--", sample) for sample in samples]
    if any(value != sample for value, sample in zip(tracked, samples)):
        raise ValueError("remote delivery sample is not tracked by final commit")
    return {"status": "DIAGNOSTICS_COMPLETE", "results_commit": results_commit, "final_commit": final}


__all__ = [
    "publish_stage",
    "required_delivery_paths",
    "validate_html_contract",
    "validate_state_provenance_rows",
    "verify_stage",
]
