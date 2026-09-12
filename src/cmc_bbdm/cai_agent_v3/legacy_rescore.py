"""Corrected statistical rescore of immutable CAI active-image v2 trajectories."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .cohort import build_capture_groups
from .files import read_csv, write_csv
from .metrics import (
    left_error_area_mpa,
    prediction_at_budget,
    trajectory_objective_mpa,
)

_ENDPOINT_BUDGET = 0.25
_EARLY_BUDGET = 0.0625
_CONFIDENCE_LEVEL = 1.0 - 0.05 / 3.0
_BOOTSTRAP_REPLICATES = 5000
_BOOTSTRAP_SEED = 2026091001

_COMPARISONS = (
    (
        "VLM_EARLY_GUIDANCE",
        "early_left_error_area_mpa",
        "NO_VLM_FEEDBACK",
        "VLM_CAI_FEEDBACK_AGENT",
    ),
    (
        "FEEDBACK_ACQUISITION",
        "left_error_area_mpa",
        "VLM_OPEN_LOOP",
        "VLM_CAI_FEEDBACK_AGENT",
    ),
    (
        "BEST_NONADAPTIVE_ACQUISITION",
        "left_error_area_mpa",
        "CENTER_FIRST",
        "VLM_CAI_FEEDBACK_AGENT",
    ),
)


def reconstruct_state_trajectory(
    rows: Sequence[Mapping[str, str]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Restore the exact zero-state and action-after states from stored v2 rows."""

    ordered = sorted(rows, key=lambda row: int(row["actor_call_index"]))
    if not ordered:
        raise ValueError("legacy trajectory is empty")
    costs = [float(ordered[0]["before_cost"])]
    predictions = [float(ordered[0]["prediction_before_mpa"])]
    if costs[0] != 0.0:
        raise ValueError("legacy trajectory does not begin at zero cost")
    for expected_index, row in enumerate(ordered):
        if int(row["actor_call_index"]) != expected_index:
            raise ValueError("legacy actor call indices are not contiguous")
        before = float(row["before_cost"])
        after = float(row["after_cost"])
        prediction_before = float(row["prediction_before_mpa"])
        prediction_after = float(row["prediction_mpa"])
        if not math.isclose(before, costs[-1], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("legacy trajectory cost boundary is discontinuous")
        if not math.isclose(
            prediction_before, predictions[-1], rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("legacy trajectory prediction boundary is discontinuous")
        if after <= before or after > _ENDPOINT_BUDGET + 1e-12:
            raise ValueError("legacy trajectory action cost is invalid")
        costs.append(after)
        predictions.append(prediction_after)
    return tuple(costs), tuple(predictions)


def _v2_labels(test_rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    labels: dict[str, float] = {}
    for row in test_rows:
        key = row["specimen_key"]
        target = float(row["target_mpa"])
        if key in labels and not math.isclose(
            labels[key], target, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("v2 test labels differ across stored runs")
        labels[key] = target
    return labels


def _author_labels(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    labels: dict[str, float] = {}
    for row in rows:
        if (
            row["measurement_semantics"] != "CAI_STRENGTH"
            or row["unit_normalized"] != "MPa"
        ):
            continue
        key = row["specimen_key"]
        if key in labels:
            raise ValueError("author CAI label is duplicated")
        labels[key] = float(row["value_numeric"])
    return labels


def _capture_groups(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    return build_capture_groups(rows)


def _mean_by_specimen(
    rows: Sequence[Mapping[str, object]], metric: str
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["specimen_key"]))].append(
            float(row[metric])
        )
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def _capture_bootstrap(
    rows: Sequence[tuple[str, str, float]],
) -> tuple[float, float, float]:
    domains: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for domain, capture_group, difference in rows:
        domains[domain][capture_group].append(float(difference))
    ordered = {
        domain: tuple(sorted(groups.items()))
        for domain, groups in sorted(domains.items())
    }
    estimate = float(
        np.mean(
            [
                np.mean([value for _, values in groups for value in values])
                for groups in ordered.values()
            ]
        )
    )
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = np.empty(_BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(_BOOTSTRAP_REPLICATES):
        domain_means: list[float] = []
        for groups in ordered.values():
            indices = rng.integers(0, len(groups), size=len(groups))
            values = [value for index in indices for value in groups[index][1]]
            domain_means.append(float(np.mean(values)))
        samples[replicate] = float(np.mean(domain_means))
    tail = (1.0 - _CONFIDENCE_LEVEL) / 2.0
    return (
        estimate,
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1.0 - tail)),
    )


def rescore_v2(*, project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    source = root / "results/cai_active_image/v2"
    output = root / "results/cai_agent_v3/legacy_v2_rescore"
    trajectories = read_csv(source / "trajectories.csv")
    old_runs = read_csv(source / "test_episodes.csv")
    old_effects = read_csv(source / "vlm_ablation_effects.csv")
    p0r = read_csv(
        root
        / "results/agentic_task_driven_nde/p0r_author_registration/surface_manifest.csv"
    )
    author_rows = read_csv(
        root / "results/hasebe_reference_evidence/v1/author_measurements.csv"
    )
    v2_labels = _v2_labels(old_runs)
    author_labels = _author_labels(author_rows)
    capture_groups = _capture_groups(p0r)
    if not set(v2_labels) <= set(capture_groups):
        raise ValueError("P0R provenance does not cover every v2 TEST specimen")

    old_by_run = {
        (row["specimen_key"], row["method"], int(row["seed"])): row for row in old_runs
    }
    trajectory_groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(
        list
    )
    for row in trajectories:
        trajectory_groups[
            (row["specimen_key"], row["method"], int(row["seed"]))
        ].append(row)
    if set(trajectory_groups) != set(old_by_run):
        raise ValueError("stored v2 trajectory and episode identities differ")

    state_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for identity in sorted(trajectory_groups):
        specimen_key, method, seed = identity
        source_rows = trajectory_groups[identity]
        costs, predictions = reconstruct_state_trajectory(source_rows)
        target = v2_labels[specimen_key]
        dataset_id = specimen_key.split(":", 1)[0]
        for state_index, (cost, prediction) in enumerate(zip(costs, predictions)):
            state_rows.append(
                {
                    "specimen_key": specimen_key,
                    "dataset_id": dataset_id,
                    "capture_group_id": capture_groups[specimen_key],
                    "method": method,
                    "seed": seed,
                    "state_index": state_index,
                    "cost": cost,
                    "prediction_mpa": prediction,
                    "target_mpa": target,
                    "absolute_error_mpa": abs(prediction - target),
                    "state_role": (
                        "ZERO_CSCAN"
                        if state_index == 0
                        else "FINAL"
                        if state_index == len(costs) - 1
                        else "INTERMEDIATE"
                    ),
                }
            )
        old = old_by_run[identity]
        area = left_error_area_mpa(costs, predictions, target, end=_ENDPOINT_BUDGET)
        early_area = left_error_area_mpa(costs, predictions, target, end=_EARLY_BUDGET)
        run_rows.append(
            {
                "specimen_key": specimen_key,
                "dataset_id": dataset_id,
                "capture_group_id": capture_groups[specimen_key],
                "method": method,
                "method_alias": (
                    "V2_SURFACE_DEPENDENT_LEARNED_CONTROL"
                    if method == "LEARNED_STATIC"
                    else method
                ),
                "seed": seed,
                "target_mpa": target,
                "author_target_mpa": author_labels[specimen_key],
                "target_difference_mpa": target - author_labels[specimen_key],
                "state_count": len(costs),
                "action_count": len(costs) - 1,
                "final_cost": costs[-1],
                "unused_budget": _ENDPOINT_BUDGET - costs[-1],
                "zero_prediction_mpa": predictions[0],
                "final_prediction_mpa": predictions[-1],
                "final_absolute_error_mpa": abs(predictions[-1] - target),
                "prediction_at_early_budget_mpa": prediction_at_budget(
                    costs, predictions, _EARLY_BUDGET
                ),
                "early_left_error_area_mpa": early_area,
                "left_error_area_mpa": area,
                "trajectory_objective_mpa": trajectory_objective_mpa(
                    costs, predictions, target, budget=_ENDPOINT_BUDGET
                ),
                "old_trapezoid_area_mpa": float(old["normalized_error_area_mpa"]),
                "corrected_minus_old_area_mpa": area
                - float(old["normalized_error_area_mpa"]),
            }
        )

    specimen_means = {
        metric: _mean_by_specimen(run_rows, metric)
        for metric in ("early_left_error_area_mpa", "left_error_area_mpa")
    }
    effect_rows: list[dict[str, object]] = []
    for claim, metric, comparator, main in _COMPARISONS:
        paired: list[tuple[str, str, float]] = []
        keys = sorted(
            specimen for method, specimen in specimen_means[metric] if method == main
        )
        for specimen in keys:
            difference = (
                specimen_means[metric][(comparator, specimen)]
                - specimen_means[metric][(main, specimen)]
            )
            paired.append(
                (
                    specimen.split(":", 1)[0],
                    capture_groups[specimen],
                    difference,
                )
            )
        estimate, lower, upper = _capture_bootstrap(paired)
        effect_rows.append(
            {
                "claim": claim,
                "metric": metric,
                "comparison": f"{comparator}_minus_{main}",
                "estimate_mpa": estimate,
                "ci_lower_mpa": lower,
                "ci_upper_mpa": upper,
                "confidence_level": _CONFIDENCE_LEVEL,
                "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
                "bootstrap_seed": _BOOTSTRAP_SEED,
                "capture_group_count": len({row[1] for row in paired}),
                "physical_specimen_count": len(paired),
                "direction": "POSITIVE_FAVORS_MAIN",
                "evidence_status": "SUPPORTED" if lower > 0.0 else "NOT_SUPPORTED",
            }
        )

    old_effect_by_claim = {row["claim"]: row for row in old_effects}
    comparison_rows: list[dict[str, object]] = []
    for row in effect_rows:
        old = old_effect_by_claim[str(row["claim"])]
        comparison_rows.append(
            {
                "claim": row["claim"],
                "old_metric": old["metric"],
                "corrected_metric": row["metric"],
                "old_estimate_mpa": float(old["estimate_mpa"]),
                "corrected_estimate_mpa": row["estimate_mpa"],
                "old_ci_lower_mpa": float(old["ci_lower_mpa"]),
                "old_ci_upper_mpa": float(old["ci_upper_mpa"]),
                "corrected_ci_lower_mpa": row["ci_lower_mpa"],
                "corrected_ci_upper_mpa": row["ci_upper_mpa"],
                "old_evidence_status": old["evidence_status"],
                "corrected_evidence_status": row["evidence_status"],
                "conclusion_changed": old["evidence_status"] != row["evidence_status"],
            }
        )

    write_csv(output / "state_predictions.csv", state_rows)
    write_csv(output / "per_run_metrics.csv", run_rows)
    write_csv(output / "paired_effects.csv", effect_rows)
    write_csv(output / "old_vs_corrected.csv", comparison_rows)
    max_target_difference = max(
        abs(float(row["target_difference_mpa"])) for row in run_rows
    )
    scope = f"""# Legacy v2 Rescore Scope Deviations

- Status: `LEGACY_RESCORE_COMPLETE`.
- Source models and trajectories were not rerun or modified.
- The corrected area uses a left-constant state trajectory with a last-state tail; the legacy area used trapezoidal integration.
- Each seed/run is scored before seed losses are averaged by physical specimen. Stored ensemble-prediction rows remain `ENSEMBLED_PREDICTIONS_UNPRICED` and are not treated as a paid run.
- The original stored v2 target is retained. Its maximum absolute difference from the direct author-workbook MPa extraction is `{max_target_difference:.9g}` MPa, attributable to the stored float32 feature-bank value.
- `LEARNED_STATIC` is not reinterpreted as a true static policy: it consumed surface-image features. Its reporting alias is `V2_SURFACE_DEPENDENT_LEARNED_CONTROL`.
- Corrected evaluation does not retroactively correct the v2 Actor training objective, checkpoint selection, or batch-termination behavior.
- Capture groups come only from P0R specimen/source-image identities; no independence relation was inferred from predictions or labels.
"""
    (output / "scope_deviations.md").write_text(scope, encoding="utf-8")
    return {
        "status": "LEGACY_RESCORE_COMPLETE",
        "run_count": len(run_rows),
        "state_count": len(state_rows),
        "specimen_count": len(v2_labels),
        "capture_group_count": len(
            {capture_groups[specimen] for specimen in v2_labels}
        ),
        "effects": effect_rows,
        "output_dir": output.relative_to(root).as_posix(),
    }


__all__ = ["reconstruct_state_trajectory", "rescore_v2"]
