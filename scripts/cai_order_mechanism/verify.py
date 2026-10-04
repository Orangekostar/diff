"""Local acceptance and delivery-inventory verification."""

from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.cai_order_mechanism.inputs import (
    ARCHIVED_METHODS,
    EXPECTED_DOMAIN_COUNTS,
    TaskContext,
    atomic_json,
    read_csv_rows,
    read_gzip_csv_rows,
    sha256_file,
)
from scripts.cai_order_mechanism.orders import NAMESPACE, fixed_permutation

REQUIRED_RESULT_FILES = (
    "ORDER_MECHANISM_SCOPE.json",
    "input_bindings.json",
    "runtime_lock.json",
    "MODEL_SELECTION_LOCK.json",
    "cohort_manifest.csv",
    "order_plan.json",
    "order_plan.sha256",
    "quality_targets.csv",
    "archived_cohort_episodes.csv.gz",
    "archived_events.csv.gz",
    "stage_episode_metrics.csv",
    "stage_specimen_metrics.csv",
    "stage_summary.csv",
    "stage_paired_contrasts.csv",
    "terminal_set_comparison.csv",
    "native_reproduction.csv",
    "clock_boundary_audit.csv",
    "prefix_request_map.csv.gz",
    "predictor_cache.npz",
    "reorder_trajectories.csv.gz",
    "endpoint_invariance.csv",
    "endpoint_uncached_checks.csv",
    "order_specimen_metrics.csv",
    "order_summary.csv",
    "order_paired_contrasts.csv",
    "order_by_domain.csv",
    "same_cap_metrics.csv",
    "cost_error_curves.csv.gz",
    "matched_quality.csv",
    "bootstrap_group_weights.npz",
    "resource_events.jsonl",
    "resource_usage.json",
    "acceptance_results.json",
    "release_manifest.json",
    "index.html",
)

REQUIRED_ARTIFACT_FILES = (
    "FINDINGS_ZH.md",
    "METHODS_FACTS.md",
    "FIGURE_CAPTIONS.md",
    "REPRODUCE.md",
    "CODEX_HANDOFF_CAI_ORDER_MECHANISM.md",
)

CASE_STEMS = {
    "74t7kcdgkr_c8-16",
    "cgtnjyggtm_q24-48",
    "w68dtmpfyf_q16-29",
}


def _paired_stems(directory: Path, *, prefix: str | None = None) -> set[str]:
    png = {
        path.stem
        for path in directory.glob("*.png")
        if prefix is None or path.stem.startswith(prefix)
    }
    svg = {
        path.stem
        for path in directory.glob("*.svg")
        if prefix is None or path.stem.startswith(prefix)
    }
    if png != svg:
        raise ValueError(f"PNG/SVG figure pairs differ: {sorted(png ^ svg)}")
    return png


def verify_delivery_inventory(
    results_root: str | Path, artifacts_root: str | Path
) -> dict[str, int]:
    results = Path(results_root)
    artifacts = Path(artifacts_root)
    missing_results = [
        name for name in REQUIRED_RESULT_FILES if not (results / name).is_file()
    ]
    missing_artifacts = [
        name for name in REQUIRED_ARTIFACT_FILES if not (artifacts / name).is_file()
    ]
    if missing_results or missing_artifacts:
        raise ValueError(
            f"delivery inventory incomplete: results={missing_results}, artifacts={missing_artifacts}"
        )
    html = (results / "index.html").read_text(encoding="utf-8")
    if re.search(r"(?:https?:)?//", html, flags=re.IGNORECASE):
        raise ValueError("offline HTML contains a remote dependency")
    figures = _paired_stems(results / "figures", prefix="F")
    if not figures or len(figures) > 5:
        raise ValueError("main figure-family count must be in [1, 5]")
    cases = _paired_stems(results / "cases")
    if cases != CASE_STEMS:
        raise ValueError(f"case figures differ from the fixed cases: {sorted(cases)}")
    findings = (artifacts / "FINDINGS_ZH.md").read_text(encoding="utf-8")
    findings_length = len(re.sub(r"\s+", "", findings))
    if not 600 <= findings_length <= 1000:
        raise ValueError("FINDINGS_ZH must contain 600-1000 non-whitespace characters")
    return {
        "result_files": len(REQUIRED_RESULT_FILES),
        "artifact_files": len(REQUIRED_ARTIFACT_FILES),
        "figure_families": len(figures),
        "case_figures": len(cases),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bool(value: object) -> bool:
    return str(value).lower() == "true"


def _q1_identity(context: TaskContext) -> dict[str, object]:
    results = context.results_root
    _require(
        context.scope_path.read_bytes() == (results / "ORDER_MECHANISM_SCOPE.json").read_bytes(),
        "result scope differs from the packaged scope",
    )
    cohort = read_csv_rows(results / "cohort_manifest.csv")
    _require(len(cohort) == 50 and len({row["specimen_key"] for row in cohort}) == 50, "cohort is not 50 unique specimens")
    _require(len({row["capture_group_id"] for row in cohort}) == 48, "capture-group count changed")
    domains = Counter(row["dataset_id"] for row in cohort)
    _require(dict(domains) == EXPECTED_DOMAIN_COUNTS, "domain counts changed")
    _require(any(row["specimen_key"] == "ykhs7s2dck:q8-22" for row in cohort), "q8-22 is missing")
    archive = read_gzip_csv_rows(results / "archived_cohort_episodes.csv.gz")
    counts = Counter(row["method"] for row in archive)
    expected = {method: (250 if method == "RANDOM" else 50) for method in ARCHIVED_METHODS}
    _require(len(archive) == 500 and dict(counts) == expected, "archive method panel changed")
    lock = json.loads((results / "MODEL_SELECTION_LOCK.json").read_text(encoding="utf-8"))
    _require(lock["actor"]["selected_update"] == 1000 and not lock["actor"]["loaded"], "actor lock changed")
    _require(lock["predictor"]["selected_update"] == 1750 and lock["predictor"]["model"] == "MEAN_SC", "predictor lock changed")
    _require(lock["new_model_selection"] == 0, "unexpected model selection")
    bindings = json.loads((results / "input_bindings.json").read_text(encoding="utf-8"))
    for name, scope_key in (
        ("assembled_episodes", "assembled_episodes"),
        ("feature_index", "feature_index"),
    ):
        _require(bindings[name]["sha256"] == context.scope["sources"][scope_key]["sha256"], f"{name} hash changed")
    for name, model in (("actor_checkpoint", "actor"), ("predictor_checkpoint", "predictor")):
        _require(bindings[name]["sha256"] == context.scope["frozen_models"][model]["sha256"], f"{name} hash changed")
    _require(bindings["test_access"] == 0, "TEST access is nonzero")
    return {"physical_n": 50, "capture_groups": 48, "domains": dict(domains), "archived_rows": 500}


def _q2_plan(context: TaskContext) -> dict[str, object]:
    results = context.results_root
    plan_path = results / "order_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    records = plan["records"]
    _require(plan["namespace"] == NAMESPACE and plan["label_inputs_used"] is False, "plan binding changed")
    _require(len(records) == 350 and plan["reordered_scenario_count"] == 300, "plan counts changed")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["specimen_key"]].append(row)
        _require(row["label_inputs_used"] is False, "plan row used a label")
        _require(sorted(row["order"]) == row["sorted_final_set"], "order changed final set")
        if row["variant"] == "PERMUTED":
            expected = list(fixed_permutation(row["specimen_key"], int(row["repeat"]), row["sorted_final_set"], namespace=NAMESPACE))
            _require(row["order"] == expected, "permutation differs from fixed SHA256 order")
    _require(len(grouped) == 50 and all(len(rows) == 7 for rows in grouped.values()), "per-specimen plan is incomplete")
    variants = Counter(row["variant"] for row in records)
    _require(variants == {"NATIVE_REPLAY": 50, "REVERSE": 50, "PERMUTED": 250}, "variant counts changed")
    sidecar = (results / "order_plan.sha256").read_text(encoding="utf-8").split()[0]
    _require(sidecar == sha256_file(plan_path), "order-plan sidecar hash changed")
    return {"scenarios": 350, "reordered": 300, "variants": dict(variants), "plan_sha256": sidecar}


def _q3_clocks(context: TaskContext) -> dict[str, object]:
    results = context.results_root
    audit = read_csv_rows(results / "clock_boundary_audit.csv")
    _require(len(audit) == 250, "clock audit must contain 50 by five caps")
    maximum = max(float(row["max_state_cost_abs_difference"]) for row in audit)
    _require(maximum <= 1e-12, "integer and archived native clocks exceed tolerance")
    plan = json.loads((results / "order_plan.json").read_text(encoding="utf-8"))["records"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plan:
        grouped[row["specimen_key"]].append(row)
    _require(
        all(
            len({row["final_pixels"] for row in rows}) == 1
            and len({row["final_cost32_hex"] for row in rows}) == 1
            and len({row["final_mask_hex"] for row in rows}) == 1
            for rows in grouped.values()
        ),
        "fixed-set terminal clock or mask differs by order",
    )
    changed = sum(_bool(row["step_membership_changed"]) for row in audit)
    return {"clock_rows": len(audit), "max_native_cost_difference": maximum, "cap_membership_changes": changed}


def _q4_prefix_recomputation(context: TaskContext) -> dict[str, object]:
    rows = read_gzip_csv_rows(context.results_root / "reorder_trajectories.csv.gz")
    _require(len(rows) == 350, "reorder trajectories are incomplete")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["specimen_key"]].append(row)
    changed_prefix = 0
    for local in grouped.values():
        native = next(row for row in local if row["variant"] == "NATIVE_REPLAY")
        reverse = next(row for row in local if row["variant"] == "REVERSE")
        native_values = [float(value) for value in native["predictions_mpa"].split(";")]
        reverse_values = [float(value) for value in reverse["predictions_mpa"].split(";")]
        if any(abs(a - b) > 1e-6 for a, b in zip(native_values[1:-1], reverse_values[1:-1], strict=True)):
            changed_prefix += 1
        _require(abs(native_values[-1] - reverse_values[-1]) <= 1e-4, "fixed-set endpoints differ")
    usage = json.loads((context.results_root / "resource_usage.json").read_text(encoding="utf-8"))
    _require(changed_prefix > 0 and usage["unique_cached_prefixes"] > 0, "prefix recomputation has no evidence")
    return {"trajectory_rows": 350, "specimens_with_changed_native_reverse_prefix": changed_prefix}


def _q5_reproduction_and_endpoints(context: TaskContext) -> dict[str, object]:
    results = context.results_root
    reproduction = read_csv_rows(results / "native_reproduction.csv")
    states = [row for row in reproduction if row["record_level"] == "STATE"]
    trajectories = [row for row in reproduction if row["record_level"] == "TRAJECTORY"]
    _require(len(trajectories) == 50 and all(_bool(row["passed"]) for row in reproduction), "native reproduction failed")
    pred_max = max(float(row["prediction_abs_difference_mpa"] or 0.0) for row in states)
    cost_max = max(float(row["cost_abs_difference"] or 0.0) for row in states)
    area_max = max(float(row["area_abs_difference_mpa"] or 0.0) for row in trajectories)
    _require(pred_max <= 1e-4 and area_max <= 1e-4 and cost_max <= 1e-12, "native reproduction exceeds tolerance")
    endpoints = read_csv_rows(results / "endpoint_invariance.csv")
    qa = read_csv_rows(results / "endpoint_uncached_checks.csv")
    _require(len(endpoints) == 350 and all(_bool(row["passed"]) for row in endpoints), "endpoint invariance failed")
    _require(len(qa) == 12 and all(_bool(row["passed"]) and _bool(row["cache_bypassed"]) for row in qa), "uncached endpoint QA failed")
    _require(len({row["dataset_id"] for row in qa}) == 6, "uncached endpoint QA is not one pair per domain")
    return {"native_state_rows": len(states), "prediction_max_abs_mpa": pred_max, "cost_max_abs": cost_max, "area_max_abs_mpa": area_max, "endpoint_rows": 350, "uncached_qa_rows": 12}


def _q6_mathematics(context: TaskContext) -> dict[str, object]:
    results = context.results_root
    episodes = read_csv_rows(results / "stage_episode_metrics.csv")
    raw = max(abs(float(row["raw_identity_residual_mpa"])) for row in episodes)
    weighted = max(abs(float(row["weighted_identity_residual_mpa"])) for row in episodes)
    _require(raw <= 1e-8 and weighted <= 1e-8, "episode event identity failed")
    stage = read_csv_rows(results / "stage_paired_contrasts.csv")
    stage_weighted = max(abs(float(row["max_weighted_identity_residual_mpa"])) for row in stage)
    stage_raw = max(abs(float(row["max_raw_identity_residual_mpa"])) for row in stage)
    _require(stage_weighted <= 1e-8 and stage_raw <= 1e-8, "paired stage identity failed")
    order = read_csv_rows(results / "order_paired_contrasts.csv")
    structure = max(abs(float(row["max_endpoint_and_j_minus_a_residual_mpa"])) for row in order)
    _require(structure <= 1e-4, "fixed-set endpoint or J-minus-A identity failed")
    _require(any(float(row["raw_negative_mpa"]) < 0 for row in episodes), "negative events were discarded")
    return {"raw_identity_max_mpa": raw, "weighted_identity_max_mpa": weighted, "paired_stage_max_mpa": max(stage_weighted, stage_raw), "j_minus_a_and_endpoint_max_mpa": structure}


def _q7_statistics(context: TaskContext) -> dict[str, object]:
    results = context.results_root
    cohort = read_csv_rows(results / "cohort_manifest.csv")
    keys = sorted(row["specimen_key"] for row in cohort)
    with np.load(results / "bootstrap_group_weights.npz", allow_pickle=False) as values:
        weights = np.asarray(values["weights"])
        stored_keys = list(map(str, values["specimen_keys"]))
        groups = list(map(str, values["capture_groups"]))
    _require(weights.shape == (5000, 50) and stored_keys == keys, "bootstrap weights are not fixed and reindexed")
    group_members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_members[group].append(index)
    _require(
        all(np.all(weights[:, members] == weights[:, members[:1]]) for members in group_members.values()),
        "capture-group members do not share bootstrap multiplicities",
    )
    summary = read_csv_rows(results / "order_summary.csv")
    _require(len(summary) == 3 and all("DOMAIN_MEAN_THEN_SIX_DOMAIN_EQUAL" in row["aggregation"] for row in summary), "order summary estimand changed")
    primary = next(row for row in read_csv_rows(results / "order_paired_contrasts.csv") if row["contrast"] == "PRIMARY_PERMUTED_MEAN_MINUS_NATIVE" and row["metric"] == "area_mpa")
    _require(int(primary["bootstrap_replicates"]) == 5000 and primary["estimand"] == "DOMAIN_EQUAL", "primary bootstrap estimand changed")
    return {"bootstrap_shape": list(weights.shape), "capture_groups": len(group_members), "primary_area_mpa": float(primary["estimate_mpa"]), "primary_ci_mpa": [float(primary["ci_low_mpa"]), float(primary["ci_high_mpa"])]}


def _q8_efficiency(context: TaskContext) -> dict[str, object]:
    rows = read_csv_rows(context.results_root / "matched_quality.csv")
    statuses = Counter(row["saving_status"] for row in rows)
    recrossings = sum(_bool(row["main_later_recrosses_target"]) or _bool(row["comparator_later_recrosses_target"]) for row in rows)
    negative = sum(bool(row["absolute_saving"]) and float(row["absolute_saving"]) < 0 for row in rows)
    _require(statuses["DEFINED"] > 0 and statuses["ONE_OR_BOTH_NOT_REACHED"] > 0 and statuses["ZERO_COMPARATOR_COST"] > 0, "matched-quality null states were lost")
    _require(recrossings > 0 and negative > 0, "recrossings or negative savings were lost")
    _require(all(row["aggregation"] == "QUEUE_MAE_FIRST_CROSSING_NOT_SPECIMEN_ORACLE" for row in rows), "matched-quality aggregation changed")
    return {"rows": len(rows), "statuses": dict(statuses), "recrossings": recrossings, "negative_savings": negative}


def _q9_resources(context: TaskContext) -> dict[str, object]:
    results = context.results_root
    usage = json.loads((results / "resource_usage.json").read_text(encoding="utf-8"))
    zero_fields = ("optimizer_updates", "actor_forwards", "qwen_forwards", "cnn_forwards", "oof_predictor_forwards", "autograd", "test_forwards", "test_access", "research_checkpoint_files_created")
    _require(all(int(usage[field]) == 0 for field in zero_fields), "a forbidden resource count is nonzero")
    _require(usage["nominal_prefix_requests"] <= 5950, "nominal prefix cap exceeded")
    _require(usage["predictor_evaluated_state_rows_including_retries_qa"] <= 7000, "evaluated-row cap exceeded")
    _require(usage["report_render_count"] <= 2 and usage["scenario_count"] == 350, "render or scenario cap changed")
    events = [json.loads(line) for line in (results / "resource_events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    intended = [row for row in events if row["state"] == "INTENDED"]
    complete = [row for row in events if row["state"] == "COMPLETE"]
    _require([(row["batch_id"], row["evaluated_rows"]) for row in intended] == [(row["batch_id"], row["evaluated_rows"]) for row in complete], "resource intents and completions differ")
    charged = sum(int(row["evaluated_rows"]) for row in intended)
    _require(charged == usage["predictor_evaluated_state_rows_including_retries_qa"], "resource charge does not reconcile")
    state = json.loads((results / "task_state.json").read_text(encoding="utf-8"))
    _require(state["resume_attempts"] == 1, "resume count changed")
    _require(float(usage["elapsed_seconds"]) <= 1800.0, "GPU session cap exceeded")
    return {"nominal_prefix_requests": usage["nominal_prefix_requests"], "unique_cached_prefixes": usage["unique_cached_prefixes"], "cache_hits": usage["cache_hits_for_nominal_requests"], "evaluated_rows": charged, "forward_calls": usage["predictor_top_level_forward_calls"], "report_renders": usage["report_render_count"], "resume_attempts": 1, "gpu_session_seconds": usage["elapsed_seconds"]}


def _q10_delivery(context: TaskContext) -> dict[str, object]:
    results = context.results_root
    figures = _paired_stems(results / "figures", prefix="F")
    cases = _paired_stems(results / "cases")
    _require(1 <= len(figures) <= 5 and cases == CASE_STEMS, "figure families or fixed cases changed")
    for directory, stems in ((results / "figures", figures), (results / "cases", cases)):
        for stem in stems:
            with Image.open(directory / f"{stem}.png") as image:
                dpi = image.info.get("dpi", (0.0, 0.0))
                _require(min(dpi) >= 299.0, f"{stem} is below 300 dpi")
            font = json.loads((results / "figure_qa" / f"{stem}.fonts.json").read_text(encoding="utf-8"))
            collision = json.loads((results / "figure_qa" / f"{stem}.collisions.json").read_text(encoding="utf-8"))
            alignment = json.loads((results / "figure_qa" / f"{stem}.alignment.json").read_text(encoding="utf-8"))
            _require(font["below_minimum_count"] == 0 and font["minimum_found_pt"] >= 5.0, f"{stem} font audit failed")
            _require(collision["verdict"] == "PASS", f"{stem} collision audit failed")
            _require(alignment["verdict"] in {"PASS", "NOT APPLICABLE"}, f"{stem} alignment audit failed")
    contact_sheet = results / "figure_qa" / "report_equivalent_contact_sheet.png"
    _require(contact_sheet.is_file() and contact_sheet.stat().st_size > 10_000, "equivalent visual contact sheet is missing")
    html_text = (results / "index.html").read_text(encoding="utf-8")
    _require(not re.search(r"(?:https?:)?//", html_text, flags=re.IGNORECASE), "report contains a remote dependency")
    _require("data-target=\"fixed\"" in html_text and "data-target=\"archived\"" in html_text, "track-switch controls are missing")
    _require("@media(max-width:760px)" in html_text and "classList.add('active')" in html_text, "responsive or track-switch behavior is missing")
    local_references = re.findall(r"(?:src|href)=['\"]([^'\"]+)['\"]", html_text)
    missing = []
    for reference in local_references:
        path = (results / html.unescape(reference)).resolve()
        if path.name == "release_manifest.json":
            continue
        if not path.is_file():
            missing.append(reference)
    _require(not missing, f"offline report references missing local files: {missing}")
    return {
        "figure_families": len(figures),
        "case_figures": len(cases),
        "pdf_collision_audits": len(figures) + len(cases),
        "html_open_check": "EQUIVALENT_STATIC_RESOURCE_AND_BEHAVIOR_CHECK",
        "browser_binary_available": False,
        "contact_sheet": str(contact_sheet.relative_to(context.root)),
    }


def _release_entries(context: TaskContext) -> list[dict[str, object]]:
    excluded = {"release_manifest.json", "task_state.json", "PUBLICATION_RECEIPT.json"}
    entries = []
    roots = (
        context.root_path("code"),
        context.root_path("tests"),
        context.root_path("scope"),
        context.results_root,
        context.artifacts_root,
    )
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if (
                not path.is_file()
                or path.name in excluded
                or path.name.endswith((".tmp", ".pyc"))
                or "__pycache__" in path.parts
            ):
                continue
            entries.append(
                {
                    "path": str(path.relative_to(context.root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return entries


def verify_release(context: TaskContext) -> list[Path]:
    checks = []
    for name, function in (
        ("Q1_IDENTITY_AND_BINDINGS", _q1_identity),
        ("Q2_PERMUTATION_DESIGN", _q2_plan),
        ("Q3_ARCHIVE_AND_INTEGER_CLOCKS", _q3_clocks),
        ("Q4_PREFIX_RECOMPUTATION", _q4_prefix_recomputation),
        ("Q5_NATIVE_AND_ENDPOINT_REPRODUCTION", _q5_reproduction_and_endpoints),
        ("Q6_MATHEMATICAL_IDENTITIES", _q6_mathematics),
        ("Q7_STATISTICAL_ESTIMANDS", _q7_statistics),
        ("Q8_MATCHED_QUALITY_EFFICIENCY", _q8_efficiency),
        ("Q9_RESOURCES_AND_REUSE", _q9_resources),
        ("Q10_LOCAL_DELIVERY", _q10_delivery),
    ):
        checks.append({"check": name, "status": "PASS", "detail": function(context)})
    acceptance_path = context.results_root / "acceptance_results.json"
    acceptance = {
        "schema_version": 1,
        "task_id": context.scope["task_id"],
        "source_commit": context.scope["source_commit"],
        "status": context.scope["publication"]["local_verification_status"],
        "publication_status": "PENDING_GIT_PUBLICATION",
        "target_technical_complete": context.scope["publication"]["technical_complete"],
        "scientific_effect_is_not_completion_gate": True,
        "passed": len(checks),
        "failed": 0,
        "checks": checks,
    }
    atomic_json(acceptance_path, acceptance)
    entries = _release_entries(context)
    manifest_path = context.results_root / "release_manifest.json"
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "task_id": context.scope["task_id"],
            "status": context.scope["publication"]["local_verification_status"],
            "publication_status": "PENDING_GIT_PUBLICATION",
            "source_commit": context.scope["source_commit"],
            "scope_sha256": context.scope_sha256,
            "order_plan_sha256": (context.results_root / "order_plan.sha256").read_text(encoding="utf-8").split()[0],
            "entry_count": len(entries),
            "total_bytes": sum(int(row["bytes"]) for row in entries),
            "excluded_from_self_reference": [
                str(manifest_path.relative_to(context.root)),
                str((context.results_root / "task_state.json").relative_to(context.root)),
                str((context.artifacts_root / "PUBLICATION_RECEIPT.json").relative_to(context.root)),
            ],
            "entries": entries,
        },
    )
    verify_delivery_inventory(context.results_root, context.artifacts_root)
    return [acceptance_path, manifest_path]


__all__ = [
    "CASE_STEMS",
    "REQUIRED_ARTIFACT_FILES",
    "REQUIRED_RESULT_FILES",
    "verify_delivery_inventory",
    "verify_release",
]
