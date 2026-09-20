"""Build the independent C manuscript from recomputed evidence."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

from scripts.cai_c_retrain.context import TaskContext, atomic_json, sha256_file

MAIN = "VLM_SPATIAL_FEEDBACK"
METHOD_ORDER = (
    "CENTER_FIRST",
    "GEOMETRY_SPREAD",
    "SERPENTINE",
    "RANDOM",
    "LEARNED_STATIC_TRUE",
    "NO_VLM_SPATIAL_FEEDBACK",
    "VLM_MEAN_FEEDBACK",
    "VLM_SPATIAL_FEEDBACK",
    "VLM_SPATIAL_OPEN_LOOP",
)
LABELS = {
    "CENTER_FIRST": "Center-first",
    "GEOMETRY_SPREAD": "Geometry-spread",
    "SERPENTINE": "Serpentine",
    "RANDOM": "Random",
    "LEARNED_STATIC_TRUE": "Learned-static",
    "NO_VLM_SPATIAL_FEEDBACK": "No-VLM spatial feedback",
    "VLM_MEAN_FEEDBACK": "C mean feedback",
    "VLM_SPATIAL_FEEDBACK": "C spatial feedback (main)",
    "VLM_SPATIAL_OPEN_LOOP": "C spatial open-loop",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        if not value.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _f(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _effect_sentence(name: str, row: dict[str, str]) -> str:
    gain = float(row["gain_mpa"])
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    direction = "lower" if gain >= 0 else "higher"
    crossing = "included zero" if low <= 0 <= high else "did not include zero"
    return (
        f"Against {name}, the main C policy had {direction} error by "
        f"{abs(gain):.3f} MPa for {row['metric']}; the 95% exploratory interval "
        f"was [{low:.3f}, {high:.3f}] MPa and {crossing}."
    )


def _markdown_table(
    rows: list[dict[str, Any]], fields: tuple[tuple[str, str], ...]
) -> str:
    header = "| " + " | ".join(label for _, label in fields) + " |"
    rule = (
        "|"
        + "|".join("---" if index == 0 else "---:" for index in range(len(fields)))
        + "|"
    )
    body = []
    for row in rows:
        values = []
        for key, _label in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(_f(value))
            else:
                try:
                    values.append(
                        _f(value)
                        if key not in {"method", "metric", "dataset_id", "effect"}
                        else str(value)
                    )
                except (TypeError, ValueError):
                    values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, rule, *body])


def _evidence(context: TaskContext) -> dict[str, Any]:
    root = context.path("evidence")
    names = (
        "method_summary.csv",
        "same_cost_metrics.csv",
        "same_cost_metrics_by_domain.csv",
        "mechanism_effects.csv",
        "same_cost_paired_summary.csv",
        "version_A_C_comparison.csv",
        "version_A_C_by_domain.csv",
        "timing_contributions.csv",
        "quality_targets.csv",
        "equal_quality_anchors.csv",
        "case_same_cost_states.csv",
        "acquisition_events.csv",
    )
    tables = {name: _read_csv(root / name) for name in names}
    tables["full_scan_reference.json"] = json.loads(
        (root / "full_scan_reference.json").read_text(encoding="utf-8")
    )
    tables["analysis_manifest.json"] = json.loads(
        (root / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    tables["actor_manifests.json"] = json.loads(
        (context.path("w3") / "actor_manifests.json").read_text(encoding="utf-8")
    )
    tables["vlm_manifest_fit.json"] = json.loads(
        (context.path("vlm") / "vlm_manifest_fit.json").read_text(encoding="utf-8")
    )
    return tables


def _source_sections(context: TaskContext, paper: Path) -> tuple[str, str, str, str]:
    old = context.path("old_paper")
    introduction = (old / "sections/01_introduction.md").read_text(encoding="utf-8")
    related = (old / "sections/02_related_work.md").read_text(encoding="utf-8")
    framework = (old / "sections/03_framework.md").read_text(encoding="utf-8")
    experimental = (old / "sections/04_experimental_design.md").read_text(
        encoding="utf-8"
    )
    r1_paragraph = (
        "\n\nThe current C prior keeps the original P0 prompt and changes only the "
        "numbered input rendering. The clean view is converted to RGB, rotated clockwise "
        "once and resized to a maximum edge of 1024 pixels. R1 overlays the same 8x8 grid "
        "with readable row-major labels 0-63 using the frozen font and inset contract. The "
        "model receives the clean view first and R1 second. A strict parser accepts only the "
        "registered region schema; one format-repair request is allowed after an invalid "
        "response. These operations alter neither P0 semantics nor the downstream 64-cell "
        "feature representation.\n"
    )
    marker = "## 3.3 Partial-observation CAI assessment"
    if marker not in framework:
        raise ValueError("old framework section structure changed")
    framework = framework.replace(marker, r1_paragraph + "\n" + marker)
    split = experimental.split("## 4.3 Learning and evaluation protocol")
    if len(split) != 2:
        raise ValueError("old experimental section structure changed")
    tail = split[1].split("## 4.4 Cost, quality and statistical analysis")
    if len(tail) != 2:
        raise ValueError("old statistical section structure changed")
    return introduction, related, framework, tail[1]


def _build_experimental(
    context: TaskContext, statistical_tail: str, evidence: dict[str, Any]
) -> str:
    old = context.path("old_paper") / "sections/04_experimental_design.md"
    prefix = old.read_text(encoding="utf-8").split(
        "## 4.3 Learning and evaluation protocol"
    )[0]
    actors = evidence["actor_manifests.json"]["actor_manifests"]
    selected = ", ".join(
        f"{LABELS[row['method']]} at update {row['selected_update']}" for row in actors
    )
    vlm = evidence["vlm_manifest_fit.json"]
    status_counts: dict[str, int] = {}
    for row in vlm["states"]:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    repairs = status_counts.get("VALID_AFTER_REPAIR", 0)
    unavailable = sum(
        count
        for status, count in status_counts.items()
        if status.startswith("UNAVAILABLE")
    )
    protocol = f"""## 4.3 Learning and evaluation protocol

Perception and assessment were frozen before the current actor runs. The C prior contains 211 terminal TRAIN/VALID records, including {vlm["reused_pilot_rows"]} exact pilot reuses and {vlm["new_unique_primary_jobs"]} newly generated primary jobs. The strict parser accepted {repairs} records after the single allowed format repair; {unavailable} records remained unavailable and contributed zero prior features. The reserved TEST images and labels were not accessed. The frozen MEAN_SC checkpoint at update 1750 served as the common evaluation predictor, and the three original capture-group cross-fitted predictors supplied training feedback.

Only three C policies were retrained: spatial feedback, spatial open-loop and mean feedback. Each used its registered seed, 1250 logical AdamW updates, batches of 16, learning rate 3x$10^{{-4}}$, weight decay $10^{{-4}}$, gradient-norm clipping at one, critic weight 0.5 and terminal-error weight 0.25. Entropy decreased from 0.01 to zero by logical update. Candidate checkpoints at updates 250, 500, 750, 1000 and 1250 were evaluated on the same 50 VALID specimens. Selection minimized six-domain-equal A(B), retained the earliest checkpoint unless improvement exceeded $10^{{-12}}$, and selected {selected}. All 15 candidate weights and trajectories were retained.

The six controls were copied byte-for-byte from the frozen W3 trajectory matrix after their W2 checkpoint, feature-bank shards, cost files and specimen identities matched the current C bindings. The three historical A policies were retained only in a separate version-comparison sidecar. The primary matrix therefore contains 500 reused control rows and 150 new C rows. Random contributes five repeats per specimen; every other method contributes one, for 650 rows over the same 50 physical validation specimens.

One initialization was used for each current C policy, with seeds 2026091301, 2026091303 and 2026091305 for spatial feedback, open-loop and mean feedback. These are method-specific initializations rather than repeated seeds. The validation cohort also selected the reported actor checkpoints, so intervals and comparisons remain exploratory selected-validation evidence rather than independent confirmation.

## 4.4 Cost, quality and statistical analysis
"""
    return prefix + protocol + statistical_tail.lstrip()


def _build_results(evidence: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    summary = {row["method"]: row for row in evidence["method_summary.csv"]}
    analysis = evidence["analysis_manifest.json"]
    best = analysis["best_nonadaptive"]
    main = summary[MAIN]
    control = summary[best]
    full = evidence["full_scan_reference.json"]
    paired = {
        (row["comparator"], float(row["budget"])): row
        for row in evidence["same_cost_paired_summary.csv"]
    }
    endpoint = paired[(best, 0.25)]
    gain = float(endpoint["mae_gain_mpa"])
    main_word = "lower" if gain >= 0 else "higher"
    effects = {row["effect"]: row for row in evidence["mechanism_effects.csv"]}
    versions = evidence["version_A_C_comparison.csv"]
    version_area = {
        row["method"]: row for row in versions if row["metric"] == "area_mpa"
    }
    timing = evidence["timing_contributions.csv"]
    main_timing = {
        int(row["stage"]): float(row["contribution_mpa"])
        for row in timing
        if row["method"] == MAIN
    }
    open_timing = {
        int(row["stage"]): float(row["contribution_mpa"])
        for row in timing
        if row["method"] == "VLM_SPATIAL_OPEN_LOOP"
    }
    timing_delta = {
        stage: main_timing[stage] - open_timing[stage] for stage in range(1, 5)
    }
    largest_stage = max(timing_delta, key=timing_delta.get)
    increases = sum(
        float(row["error_reduction_mpa"]) < 0
        for row in evidence["acquisition_events.csv"]
        if row["method"] == MAIN
    )
    main_events = sum(
        row["method"] == MAIN for row in evidence["acquisition_events.csv"]
    )
    gap = float(main["endpoint_mae_mpa"]) - float(full["mae"])

    table_rows = []
    for method in METHOD_ORDER:
        row = summary[method]
        table_rows.append(
            {
                "method": LABELS[method],
                "area": float(row["area_mpa"]),
                "early": float(row["early_area_mpa"]),
                "mae": float(row["endpoint_mae_mpa"]),
                "rmse": float(row["endpoint_rmse_mpa"]),
                "r2": float(row["endpoint_r2"]),
            }
        )
    table_rows.append(
        {
            "method": "Complete input, fraction 1.0",
            "area": "NA",
            "early": "NA",
            "mae": float(full["mae"]),
            "rmse": float(full["rmse"]),
            "r2": float(full["r2"]),
        }
    )
    main_table = _markdown_table(
        table_rows,
        (
            ("method", "Acquisition policy"),
            ("area", "A (MPa)"),
            ("early", "Early A (MPa)"),
            ("mae", "MAE at cap (MPa)"),
            ("rmse", "RMSE at cap (MPa)"),
            ("r2", "R2 at cap"),
        ),
    )
    contrast_rows = [
        {
            "effect": row["effect"],
            "comparator": LABELS.get(row["comparator"], row["comparator"]),
            "metric": row["metric"],
            "gain": float(row["gain_mpa"]),
            "interval": f"[{_f(row['ci_low'])}, {_f(row['ci_high'])}]",
        }
        for row in evidence["mechanism_effects.csv"]
    ]
    contrast_table = _markdown_table(
        contrast_rows,
        (
            ("effect", "Contrast"),
            ("comparator", "Comparator"),
            ("metric", "Measure"),
            ("gain", "Control - C main (MPa)"),
            ("interval", "95% exploratory interval"),
        ),
    )
    version_text = " ".join(
        (
            f"For {LABELS[method]}, historical A minus current C was "
            f"{float(row['A_minus_C']):.3f} MPa "
            f"([{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}])."
        )
        for method, row in version_area.items()
    )
    results = f"""# 5 Results and discussion

## 5.1 Prediction quality under matched acquisition caps

Under the common 25% cap, C spatial feedback achieved a pooled MAE of {_f(main["endpoint_mae_mpa"])} MPa, RMSE of {_f(main["endpoint_rmse_mpa"])} MPa and R2 of {_f(main["endpoint_r2"])} on 50 selected validation specimens. The best non-adaptive strategy by six-domain-equal area was {LABELS[best]}, with endpoint MAE {_f(control["endpoint_mae_mpa"])} MPa. The main policy's endpoint MAE was {main_word} by {abs(gain):.3f} MPa, with a 95% exploratory paired interval of [{_f(endpoint["ci_low"])}, {_f(endpoint["ci_high"])}] MPa. This fixed-cohort interval quantifies uncertainty but does not undo validation-based checkpoint selection.

Across the full observed range, C spatial feedback had A={_f(main["area_mpa"])} MPa and early A={_f(main["early_area_mpa"])} MPa. {LABELS[best]} had A={_f(control["area_mpa"])} MPa. Area and endpoint error need not rank strategies identically because area values improvements by how long they remain available. Figure 2 preserves all nine held-error paths, including reversals, and the source tables report actual acquired fractions and ranges at each cap.

{main_table}

**Table 3. Current C common-predictor comparison.** Partial policies use cap 0.25. Complete input is a separate observed point at cost 1.0 and has no partial-range area. Areas weight domains equally; endpoint metrics pool physical specimens after within-specimen repeat-loss averaging.

![Current C MAE and RMSE along the five frozen acquisition caps. Curves retain observed reversals and use no smoothing.](figures/F1_cost_error_curves.pdf){{width=100%}}

## 5.2 Feedback, VLM prior and spatial interaction

{_effect_sentence(LABELS[best], effects["best_nonadaptive"])} {_effect_sentence("C spatial open-loop", effects["feedback"])} {_effect_sentence("no-VLM spatial feedback", effects["vlm_early"])} {_effect_sentence("C mean feedback", effects["spatial_vs_mean"])}

{contrast_table}

**Table 4. Prespecified C contrasts.** Positive differences favour C spatial feedback for the stated measure. Intervals use 5000 paired capture-group bootstrap draws within domains and remain conditional on the selected checkpoints.

The component comparisons are descriptive rather than causal allocations. The policies were trained separately and can visit different states. A negative estimate is retained as evidence against a uniform benefit from the added component. Figure 3 reports the endpoint and area contrasts without filtering by direction.

![Paired endpoint and trajectory-area contrasts. Positive values favour current C spatial feedback; all signs and intervals are retained.](figures/F2_paired_effects.pdf){{width=100%}}

## 5.3 What changed from historical A to current C

Current C changes only the numbered rendering and the three retrained VLM policies; P0, predictors, feature bank, cost definition and six primary controls remain fixed. {version_text} These comparisons isolate the observed version change within each policy name but remain selected-validation comparisons rather than a second independent experiment.

Domain-level A-minus-C values varied across the six source domains. This heterogeneity is retained in Figure 6 and the source table. The data do not identify a material-specific cause because domain, image source and predictor quality are not experimentally separated.

![Endpoint performance by domain and historical-A-minus-current-C trajectory area for the three retrained policies.](figures/F5_domain_results.pdf){{width=100%}}

## 5.4 Timing and executed acquisition cases

The largest signed main-minus-open-loop timing difference occurred in completion stage {largest_stage}, where it was {timing_delta[largest_stage]:.3f} MPa. The four stage differences were {", ".join(f"{timing_delta[stage]:+.3f}" for stage in range(1, 5))} MPa. These terms satisfy A=e0-sum(g) and preserve negative contributions. They describe when saved prediction changes affected the remaining budget interval, not a causal percentage attributable to feedback.

Individual acquisitions did not guarantee monotonic improvement. {increases} of {main_events} current main-policy events increased absolute prediction error. This adverse-event count is compatible with a favourable overall area when earlier or larger reductions dominate. Figure 5 shows all signed stage contributions and the identity components.

![Signed timing-weighted contributions for all nine methods. Negative values and unequal initial-error terms are retained.](figures/F4_timing_contributions.pdf){{width=100%}}

The three prespecified cases were regenerated from the selected current C main-policy trajectories and current C priors. Surface candidates, the real first action, measured states after 1, 4 and 8 acquisitions, the endpoint and prediction process are new outputs. The measured set grows monotonically on the strict 8x8 grid, while predictions may reverse.

![Current C acquisition progression for the three fixed validation cases. Panels show the C prior, acquired states and saved prediction paths.](figures/F6_case_progression.pdf){{width=100%}}

## 5.5 Equal-quality and full-input boundaries

Equal-quality comparisons use the earliest observed population-curve crossing on both the five-cap grid and the union of saved event costs. Unreached targets, negative savings, zero denominators and later recrossings remain in the source tables. Figure 4 displays the complete integer target range rather than selecting a favourable threshold. These population first passages do not define a label-free stopping policy for individual specimens.

![Earliest observed costs and current-main savings across the complete empirical MAE target grid.](figures/F3_equal_quality.pdf){{width=100%}}

The complete-input predictor achieved MAE {_f(full["mae"])} MPa, RMSE {_f(full["rmse"])} MPa and R2 {_f(full["r2"])} at cost 1.0. Current C spatial feedback retained an endpoint MAE gap of {gap:.3f} MPa at the 25% cap. No unobserved segment between 0.25 and 1.0 is interpolated. The comparison therefore bounds the observed image-acquisition trade-off without converting native-pixel fraction into physical inspection time.

Taken together, the current evidence supports an offline task-driven acquisition analysis on a selected validation cohort. It does not establish independent generalization, causal component shares or deployment-time savings. Those questions require new specimens, repeated policy seeds and a physical acquisition-cost model.
"""
    facts = {
        "main": main,
        "best_nonadaptive": best,
        "endpoint_gain": gain,
        "full": full,
        "gap": gap,
        "timing_delta": timing_delta,
        "increasing_events": increases,
        "main_events": main_events,
    }
    return results, facts


def _build_abstract(facts: dict[str, Any]) -> str:
    main = facts["main"]
    best = LABELS[facts["best_nonadaptive"]]
    direction = "lower" if facts["endpoint_gain"] >= 0 else "higher"
    return f"""# Task-driven partial C-scan acquisition for compression-after-impact assessment

Image-based compression-after-impact assessment commonly assumes that a complete internal scan is available, although acquisition cost depends on which regions are measured and when they inform the estimate. We couple a frozen CAI predictor to learned acquisition policies that select cells from an 8x8 internal C-scan grid using surface information, a cached vision-language prior and acquired internal evidence. The current C version retains the original P0 prompt, introduces an exact readable R1 numbering render and retrains three VLM-conditioned actors while freezing six controls and all predictors. On 50 selected validation specimens, C spatial feedback achieved MAE {_f(main["endpoint_mae_mpa"])} MPa and six-domain-equal trajectory area {_f(main["area_mpa"])} MPa at a 25% native-pixel cap. Its endpoint MAE was {direction} than {best} by {abs(facts["endpoint_gain"]):.3f} MPa, while component and historical-version contrasts remained mixed across methods and domains. The complete-input MAE was {_f(facts["full"]["mae"])} MPa, leaving a {facts["gap"]:.3f} MPa gap at the partial cap. Signed timing terms and three executed cases show that useful trajectories can still contain prediction reversals. The study provides a reproducible offline framework for comparing assessment quality throughout acquisition, bounded to validation-selected image replay rather than physical inspection-time savings.

**Keywords:** active acquisition; compression after impact; C-scan; multimodal learning; sequential decision making; vision-language model
"""


def _build_conclusion(facts: dict[str, Any]) -> str:
    main = facts["main"]
    return f"""# 6 Conclusions

Task-driven acquisition links the choice of internal observations to the evolving compression-after-impact estimate. The current C implementation keeps the original surface-perception prompt, uses an exact readable 8x8 numbering render and retrains the three VLM-conditioned actors against the same frozen predictor and data bindings.

On the 50 selected validation specimens, C spatial feedback achieved {_f(main["endpoint_mae_mpa"])} MPa endpoint MAE and {_f(main["area_mpa"])} MPa six-domain-equal trajectory area under a 25% native-pixel cap. Comparisons with non-adaptive acquisition, open-loop feedback, no-VLM feedback, mean feedback and historical A were mixed rather than uniformly favourable. Signed timing contributions and {facts["increasing_events"]} adverse prediction changes among {facts["main_events"]} acquisitions show why an informative sequence need not improve after every action.

The framework supports reproducible same-cap, equal-quality and timing analyses under a common assessment mapping. Its evidence remains conditional on one policy seed per method, validation-based checkpoint selection and offline image replay. Independent specimens, repeated training seeds and physical travel, coupling and latency costs are required before inferring deployment savings.
"""


def _build_supplement(evidence: dict[str, Any]) -> str:
    actors = evidence["actor_manifests.json"]["actor_manifests"]
    actor_rows = [
        {
            "method": LABELS[row["method"]],
            "seed": row["training_seed"],
            "selected": row["selected_update"],
            "area": float(row["validation_area_mpa"]),
            "logical": row["logical_updates"],
            "actual": row["actual_optimizer_updates"],
        }
        for row in actors
    ]
    same_cost = []
    for row in evidence["same_cost_metrics.csv"]:
        same_cost.append(
            {
                "method": LABELS[row["method"]],
                "budget": float(row["budget"]),
                "mae": float(row["mae_mpa"]),
                "rmse": float(row["rmse_mpa"]),
                "r2": float(row["r2"]),
                "actual": float(row["actual_cost_mean"]),
                "range": f"{float(row['actual_cost_min']):.6f}-{float(row['actual_cost_max']):.6f}",
            }
        )
    version = [
        {
            "method": LABELS[row["method"]],
            "metric": row["metric"],
            "a": float(row["historical_A"]),
            "c": float(row["current_C"]),
            "difference": float(row["A_minus_C"]),
            "interval": f"[{_f(row['ci_low'])}, {_f(row['ci_high'])}]",
        }
        for row in evidence["version_A_C_comparison.csv"]
    ]
    return f"""# Supplementary Information

## S1 Current C protocol and selected actors

The C prior uses Qwen2.5-VL-7B-Instruct revision cc594898137f460bfe9f0759e9844b3ce807cfb5, P0 without a new system message, clean-plus-R1 image order, bfloat16, SDPA, deterministic generation and at most 500 new tokens. R1 rotates the RGB source clockwise once, limits the longest edge to 1024 pixels and applies readable row-major labels 0-63. A strict parser permits one format repair. The 211-row TRAIN/VALID prior is complete; TEST was not accessed.

{_markdown_table(actor_rows, (("method", "Method"), ("seed", "Seed"), ("selected", "Selected update"), ("area", "VALID A (MPa)"), ("logical", "Logical updates"), ("actual", "Actual updates")))}

All five candidate checkpoints per C actor remain archived. Selection retained the earliest candidate unless a later domain-equal area improved by more than 1e-12. The six controls are frozen historical rows, while the three A policies are stored separately from the 650-row primary matrix.

## S2 Statistical estimands

Same-cost endpoint losses first average Random repeat losses within each physical specimen. Pooled MAE and RMSE then aggregate 50 physical specimens. A and early A instead average repeats within specimen, specimens within domain and the six domains equally. Capture-group bootstrap intervals use 5000 paired draws within domain with seed 2026091401. These intervals are exploratory and conditional on checkpoint selection.

Timing uses $g_t=(1-c_t/0.25)(e_{{t-1}}-e_t)$ in four right-closed completion intervals. Negative terms are retained and the machine-readable evidence verifies $A=e_0-\\sum_t g_t$. Equal-quality analysis searches the earliest supported crossing on the five-cap and event-union grids; it retains unreached targets, recrossings and negative or undefined savings.

## S3 Complete same-cost results

{_markdown_table(same_cost, (("method", "Method"), ("budget", "Cap"), ("mae", "MAE"), ("rmse", "RMSE"), ("r2", "R2"), ("actual", "Mean actual cost"), ("range", "Actual range")))}

## S4 Historical A and current C

{_markdown_table(version, (("method", "Method"), ("metric", "Metric"), ("a", "Historical A"), ("c", "Current C"), ("difference", "A - C"), ("interval", "95% interval")))}

Positive A-minus-C values favour current C. All domain-specific values, endpoint paired contrasts and target grids are retained in the accompanying CSV files.

## S5 Current C case illustrations

The three cases were fixed before this retraining release. Every panel below was regenerated from the selected current C main-policy trajectory and current C prior. They are process illustrations rather than independent replications.

![Current C candidate surfaces, acquired states and prediction processes for c8-16, q24-48 and q16-29.](figures/F6_case_progression.pdf){{width=100%}}

## S6 Source-data index

The tables directory contains the recomputed method summary, all five-cap metrics, domain metrics, paired contrasts, A/C version comparisons, event curves, timing contributions, quality targets, case states and full-input reference. The evidence manifest records their hashes. The figures directory contains vector PDF/SVG files and 300-dpi PNG previews, plus all new case PNGs.

# References
"""


def _copy_sources_and_outputs(
    context: TaskContext, paper: Path, evidence: dict[str, Any]
) -> None:
    old = context.path("old_paper")
    for relative in (
        "references.bib",
        "declarations.md",
        "build_manuscript.py",
        "references/reference_ledger.csv",
        "references/verification_notes.md",
        "references/closest_work_matrix.md",
        "figures/Fig1_framework.pdf",
        "figures/Fig1_framework.svg",
        "figures/Fig1_framework.png",
    ):
        _copy(old / relative, paper / relative)
    evidence_root = context.path("evidence")
    for path in sorted((evidence_root / "figures").iterdir()):
        if path.is_file() and path.suffix in {".png", ".svg", ".pdf"}:
            _copy(path, paper / "figures" / path.name)
    for case in evidence_root.glob("cases/*/*.png"):
        _copy(case, paper / "figures" / f"{case.parent.name}_{case.name}")
    for path in sorted(evidence_root.iterdir()):
        if path.is_file() and path.suffix in {".csv", ".json", ".npz"}:
            _copy(path, paper / "tables" / path.name)
    timing = evidence_root / "timing_contributions.csv"
    _copy(timing, paper / "analysis" / timing.name)


def _write_indexes(context: TaskContext, paper: Path, evidence: dict[str, Any]) -> None:
    figures = [
        {
            "figure": "Fig1",
            "path": "figures/Fig1_framework.pdf",
            "evidence": "frozen architecture and current Methods",
            "status": "REUSED_VERIFIED_STRUCTURE",
        }
    ]
    for index, name in enumerate(
        (
            "F1_cost_error_curves",
            "F2_paired_effects",
            "F3_equal_quality",
            "F4_timing_contributions",
            "F5_domain_results",
            "F6_case_progression",
        ),
        start=2,
    ):
        figures.append(
            {
                "figure": f"Fig{index}",
                "path": f"figures/{name}.pdf",
                "evidence": f"tables and manifest for {name}",
                "status": "CURRENT_C_RECOMPUTED",
            }
        )
    _write_csv(paper / "figures/FIGURE_INDEX.csv", figures)
    evidence_map = [
        {
            "location": "abstract and sections/05_results.md",
            "claim": "current C same-cost performance and boundary",
            "evidence": "tables/method_summary.csv; tables/same_cost_paired_summary.csv",
            "status": "SUPPORTED_SELECTED_VALID",
        },
        {
            "location": "sections/04_experimental_design.md",
            "claim": "211 priors, three retrained actors, six frozen controls",
            "evidence": "vlm_manifest_fit.json; actor_manifests.json; assembly_manifest.json",
            "status": "SUPPORTED_EXECUTION_RECORD",
        },
        {
            "location": "sections/05_results.md historical comparison",
            "claim": "historical A versus current C",
            "evidence": "tables/version_A_C_comparison.csv; tables/version_A_C_by_domain.csv",
            "status": "SUPPORTED_SELECTED_VALID",
        },
        {
            "location": "sections/05_results.md timing and cases",
            "claim": "signed timing and executed current C cases",
            "evidence": "analysis/timing_contributions.csv; tables/case_same_cost_states.csv; figures/F6_case_progression.pdf",
            "status": "SUPPORTED_EXECUTION_RECORD",
        },
    ]
    _write_csv(paper / "EVIDENCE_MAP.csv", evidence_map)
    allocation = [
        {
            "result": "same-cap main comparison",
            "class": "core_discovery",
            "destination": "main",
            "reason": "central quality-cost result",
        },
        {
            "result": "component contrasts",
            "class": "qualification",
            "destination": "main",
            "reason": "mixed directions bound contribution",
        },
        {
            "result": "historical A versus C",
            "class": "necessary_support",
            "destination": "main",
            "reason": "defines current release change",
        },
        {
            "result": "all cap and domain values",
            "class": "heterogeneity",
            "destination": "SI/source data",
            "reason": "complete record without main-text repetition",
        },
        {
            "result": "211 state and provenance details",
            "class": "provenance_detail",
            "destination": "repository/evidence index",
            "reason": "auditability",
        },
    ]
    _write_csv(paper / "RESULT_ALLOCATION.csv", allocation)
    actor_rows = evidence["actor_manifests.json"]["actor_manifests"]
    selected = ", ".join(
        f"{row['method']}={row['selected_update']}" for row in actor_rows
    )
    readme = f"""# Current C author-review manuscript

This directory is the default manuscript entry for `C_P0_R1_GLOBAL_V1`. It was generated from the recomputed C evidence and does not overwrite historical A.

- Main source: `manuscript.md`
- HTML: `manuscript.html`
- TeX: `main.tex` and `supplementary.tex`
- PDFs: `build/main.pdf` and `build/supplementary.pdf`
- Evidence map: `EVIDENCE_MAP.csv`
- Selected C updates: {selected}

Scientific boundary: fixed selected VALID cohort, one seed per current C actor, offline replay, no TEST evaluation and no physical-time saving claim.
"""
    _atomic_text(paper / "README.md", readme)
    _atomic_text(
        paper / "TERMINOLOGY_AND_NOTATION.md",
        """# Terminology ledger

| Canonical term | Definition |
|---|---|
| current C | P0 plus exact R1 numbered render and current three-actor retraining |
| historical A | prior P0/R0-derived actor version used only for comparison |
| C spatial feedback | current main actor using surface, C prior and acquired internal/prediction feedback |
| native-pixel fraction | unique acquired pixels divided by registered C-scan pixels |
| A | six-domain-equal normalized left-constant absolute-error area over cost 0-0.25 |
| selected VALID | the 50-specimen validation cohort used for checkpoint selection and reporting |
""",
    )
    _atomic_text(
        paper / "AUTHOR_READING_GUIDE_ZH.md",
        """# 作者导读

本稿默认主线是 current C，不以效果方向作为发布门槛。请优先审阅摘要、5.1 同成本主结果、5.2 不利/混合组件结果、5.3 A/C 版本差异及结论边界。作者姓名、单位、基金、利益冲突、许可和最终 AI 披露仍需作者确认。
""",
    )
    _atomic_text(
        paper / "AI_USE_RECORD.md",
        """# AI use record

OpenAI Codex assisted with deterministic analysis code, figure composition and evidence-bound language revision. All numerical claims are generated from archived trajectories and machine-readable tables. Human authors remain responsible for scientific interpretation, authorship, disclosure wording and submission decisions. The Qwen model used by the method is separately identified in Methods.
""",
    )
    _atomic_text(
        paper / "AUTHOR_INPUTS.md",
        """# Author inputs still required

- Author names, affiliations and corresponding-author details
- Funding and acknowledgements
- Final conflict-of-interest and data/code availability wording
- Target-journal formatting confirmation
- Human scientific approval of the current C interpretation and AI-use disclosure
""",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _find_pandoc(paper: Path) -> str:
    candidates = [
        os.environ.get("PANDOC"),
        shutil.which("pandoc"),
        str(paper / "tools/pandoc"),
        str(Path.home() / ".local/bin/pandoc"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("Pandoc is required to rebuild the manuscript")


def _run_build(paper: Path) -> dict[str, Any]:
    build = paper / "build"
    build.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PANDOC"] = _find_pandoc(paper)
    subprocess.run(
        [sys.executable, "build_manuscript.py"],
        cwd=paper,
        env=environment,
        check=True,
    )
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise FileNotFoundError("latexmk is required to compile the manuscript")
    logs = {}
    for source, name in (("main.tex", "main"), ("supplementary.tex", "supplementary")):
        completed = subprocess.run(
            [
                latexmk,
                "-lualatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-outdir=build",
                source,
            ],
            cwd=paper,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path = build / f"{name}_compile.log"
        _atomic_text(log_path, completed.stdout)
        if completed.returncode:
            raise RuntimeError(f"{source} compilation failed; see {log_path}")
        pdf = build / f"{name}.pdf"
        if not pdf.is_file() or pdf.stat().st_size < 10_000:
            raise ValueError(f"compiled PDF is missing or undersized: {pdf}")
        logs[name] = {"pdf_sha256": sha256_file(pdf), "bytes": pdf.stat().st_size}
    return logs


def _page_count(pdf: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    match = re.search(r"^Pages:\s+(\d+)", output, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"cannot read PDF page count: {pdf}")
    return int(match.group(1))


def _term_pages(extracted: str, required_terms: tuple[str, ...]) -> dict[str, int]:
    page_texts = extracted.split("\f")
    missing = [term for term in required_terms if term not in extracted]
    if missing:
        raise ValueError(f"required PDF content is missing: {missing}")
    return {
        term: next(
            index
            for index, page_text in enumerate(page_texts, start=1)
            if term in page_text
        )
        for term in required_terms
    }


def _visual_qa(paper: Path) -> dict[str, Any]:
    build = paper / "build"
    qa_root = build / "visual_qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    report = {"status": "PASS", "documents": {}}
    for name, required_terms in (
        (
            "main",
            (
                "Abstract",
                "Algorithm 1",
                "Table 3",
                "Figure 2",
                "Results and discussion",
            ),
        ),
        (
            "supplementary",
            ("Supplementary Information", "Table S1", "Complete same-cost results"),
        ),
    ):
        pdf = build / f"{name}.pdf"
        pages = _page_count(pdf)
        extracted = subprocess.check_output(
            ["pdftotext", "-layout", str(pdf), "-"], text=True
        )
        try:
            term_pages = _term_pages(extracted, required_terms)
        except ValueError as error:
            raise ValueError(
                f"required PDF content missing from {name}: {error}"
            ) from error
        selected = sorted({1, pages, *term_pages.values()})
        page_records = []
        for page in selected:
            output = qa_root / f"{name}_page_{page}"
            subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-singlefile",
                    "-png",
                    "-r",
                    "110",
                    str(pdf),
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            image_path = output.with_suffix(".png")
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                difference = ImageChops.difference(
                    rgb, Image.new("RGB", rgb.size, "white")
                )
                bbox = difference.getbbox()
                if bbox is None:
                    raise ValueError(f"blank PDF page: {name} page {page}")
                coverage = (
                    (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (rgb.width * rgb.height)
                )
                if coverage < 0.05:
                    raise ValueError(
                        f"suspiciously sparse PDF page: {name} page {page}"
                    )
                page_records.append(
                    {
                        "page": page,
                        "preview": image_path.relative_to(paper).as_posix(),
                        "width": rgb.width,
                        "height": rgb.height,
                        "content_bbox": list(bbox),
                        "bbox_coverage": coverage,
                    }
                )
        report["documents"][name] = {
            "pages": pages,
            "required_terms": list(required_terms),
            "term_pages": term_pages,
            "checked_pages": page_records,
        }
    atomic_json(build / "visual_qa.json", report)
    return report


def _output_hashes(root: Path, manifest_path: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path != manifest_path
        and not path.name.endswith(
            (".aux", ".fdb_latexmk", ".fls", ".log", ".out", ".blg", ".bbl")
        )
    }


def paper_stage(context: TaskContext) -> dict[str, Any]:
    analysis_path = context.path("evidence") / "analysis_manifest.json"
    if not analysis_path.is_file():
        raise RuntimeError(
            "current C evidence must be complete before manuscript generation"
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("status") != "C_EVIDENCE_COMPLETE":
        raise ValueError("current C evidence is incomplete")
    signature_inputs = [
        analysis_path,
        context.path("w3") / "actor_manifests.json",
        context.path("vlm") / "vlm_manifest_fit.json",
        context.config_path,
    ]
    signature = context.phase_signature("paper", signature_inputs)
    paper = context.path("paper")
    manifest_path = paper / "paper_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "C_MANUSCRIPT_COMPLETE":
            if existing.get("input_signature") != signature:
                raise ValueError("completed manuscript input signature changed")
            for relative, digest in existing["outputs"].items():
                path = paper / relative
                if not path.is_file() or sha256_file(path) != digest:
                    raise ValueError(f"completed manuscript output changed: {relative}")
            return existing
    context.transition("paper", "RUNNING")
    paper.mkdir(parents=True, exist_ok=True)
    evidence = _evidence(context)
    introduction, related, framework, statistics = _source_sections(context, paper)
    results, facts = _build_results(evidence)
    sections = {
        "01_introduction.md": introduction,
        "02_related_work.md": related,
        "03_framework.md": framework,
        "04_experimental_design.md": _build_experimental(context, statistics, evidence),
        "05_results.md": results,
        "06_conclusions.md": _build_conclusion(facts),
    }
    _atomic_text(paper / "abstract.md", _build_abstract(facts))
    for name, content in sections.items():
        _atomic_text(paper / "sections" / name, content)
    _atomic_text(paper / "supplementary.md", _build_supplement(evidence))
    _copy_sources_and_outputs(context, paper, evidence)
    _write_indexes(context, paper, evidence)
    build = _run_build(paper)
    visual_qa = _visual_qa(paper)
    required = (
        "abstract.md",
        "manuscript.md",
        "manuscript.html",
        "main.tex",
        "supplementary.md",
        "supplementary.tex",
        "references.bib",
        "EVIDENCE_MAP.csv",
        "README.md",
        "build/main.pdf",
        "build/supplementary.pdf",
    )
    missing = [relative for relative in required if not (paper / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"required manuscript outputs are missing: {missing}")
    manifest = {
        "schema_version": 1,
        "task_id": context.task_id,
        "status": "C_MANUSCRIPT_COMPLETE",
        "prior_version": context.scope["prior_version"],
        "execution_code_sha256": sha256_file(Path(__file__)),
        "input_signature": signature,
        "source_paper": context.path("old_paper").relative_to(context.root).as_posix(),
        "source_sections": 6,
        "current_C_default": True,
        "build": build,
        "visual_qa": visual_qa,
        "author_review_required": True,
        "outputs": {},
    }
    manifest["outputs"] = _output_hashes(paper, manifest_path)
    atomic_json(manifest_path, manifest)
    context.transition(
        "paper",
        "COMPLETE",
        main_pdf="build/main.pdf",
        supplementary_pdf="build/supplementary.pdf",
        author_review_required=True,
    )
    return manifest


__all__ = ["paper_stage"]
