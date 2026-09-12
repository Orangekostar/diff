"""One conditionally unlocked internal TEST evaluation for CAI Agent v3."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .actor_training import (
    _cell_costs,
    _domain_equal_episode_score,
    _VLMFeatures,
    evaluate_policy,
    load_actor_checkpoint,
)
from .feature_bank import V3FeatureBank, load_feature_bank
from .files import read_csv, write_csv, write_json
from .metrics import prediction_at_budget
from .predictor_training import load_predictor_checkpoint

_REPORT_BUDGETS = (0.0, 0.0625, 0.125, 0.1875, 0.25)
_BOOTSTRAP_REPLICATES = 5000
_CONFIDENCE = 0.9833333333333333


def _test_target_array(output: Path, bank: V3FeatureBank) -> np.ndarray:
    rows = {row["specimen_key"]: row for row in read_csv(output / "scoring_labels.csv")}
    targets = bank.targets_mpa.astype(np.float64, copy=True)
    for index in bank.indices("TEST"):
        row = rows[bank.specimen_keys[int(index)]]
        if row["split"] != "TEST" or row["scoring_only_for_test"] != "True":
            raise ValueError("TEST scoring-label boundary is invalid")
        targets[int(index)] = float(row["author_cai_mpa"])
    if not np.all(np.isfinite(targets[bank.indices("TEST")])):
        raise ValueError("TEST scoring labels are incomplete")
    return targets


def _metric_rows(
    episodes: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in episodes:
        evaluation_run = row["run"] if row["method"] == "RANDOM" else 0
        grouped[
            (str(row["method"]), str(row["seed_panel"]), str(evaluation_run))
        ].append(row)
    metrics = []
    domains = []
    for (method, seed_panel, evaluation_run), rows in sorted(grouped.items()):
        for budget in _REPORT_BUDGETS:
            values = []
            by_domain: dict[str, list[float]] = defaultdict(list)
            targets = []
            predictions = []
            for row in rows:
                prediction = prediction_at_budget(
                    [float(value) for value in str(row["costs"]).split(";")],
                    [float(value) for value in str(row["predictions_mpa"]).split(";")],
                    budget,
                )
                target = float(row["target_mpa"])
                error = abs(prediction - target)
                values.append(error)
                targets.append(target)
                predictions.append(prediction)
                by_domain[str(row["dataset_id"])].append(error)
            target_array = np.asarray(targets)
            prediction_array = np.asarray(predictions)
            denominator = float(np.sum(np.square(target_array - np.mean(target_array))))
            r2 = (
                float(
                    1.0
                    - np.sum(np.square(prediction_array - target_array)) / denominator
                )
                if denominator > 0.0
                else math.nan
            )
            metrics.append(
                {
                    "method": method,
                    "seed_panel": seed_panel,
                    "evaluation_run": evaluation_run,
                    "state": f"COST_{budget:g}",
                    "cost": budget,
                    "physical_n": len(values),
                    "mae_mpa": float(np.mean(values)),
                    "rmse_mpa": float(np.sqrt(np.mean(np.square(values)))),
                    "r2": r2,
                }
            )
            for domain, errors in sorted(by_domain.items()):
                domains.append(
                    {
                        "method": method,
                        "seed_panel": seed_panel,
                        "evaluation_run": evaluation_run,
                        "state": f"COST_{budget:g}",
                        "dataset_id": domain,
                        "physical_n": len(errors),
                        "mae_mpa": float(np.mean(errors)),
                    }
                )
    return metrics, domains


def _full_input_rows(
    predictor: torch.nn.Module,
    bank: V3FeatureBank,
    targets: np.ndarray,
    *,
    device: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    indices = bank.indices("TEST")
    predictions = []
    batch_size = 32
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            prediction = predictor(
                torch.from_numpy(bank.surface_tokens[batch]).to(device),
                torch.from_numpy(bank.cscan_tokens[batch]).to(device),
                torch.ones((len(batch), 64), dtype=torch.bool, device=device),
                cost=torch.ones(len(batch), dtype=torch.float32, device=device),
            )
            predictions.extend(float(value) for value in prediction.cpu())
    truth = targets[indices]
    values = np.asarray(predictions)
    errors = np.abs(values - truth)
    denominator = float(np.sum(np.square(truth - np.mean(truth))))
    overall = {
        "method": "P_ALL_FULL_64_DIAGNOSTIC",
        "seed_panel": 0,
        "evaluation_run": 0,
        "state": "FULL_64",
        "cost": 1.0,
        "physical_n": len(indices),
        "mae_mpa": float(np.mean(errors)),
        "rmse_mpa": float(np.sqrt(np.mean(np.square(errors)))),
        "r2": float(1.0 - np.sum(np.square(values - truth)) / denominator),
    }
    by_domain: dict[str, list[float]] = defaultdict(list)
    for offset, index in enumerate(indices):
        by_domain[bank.dataset_ids[int(index)]].append(float(errors[offset]))
    domain_rows = [
        {
            "method": "P_ALL_FULL_64_DIAGNOSTIC",
            "seed_panel": 0,
            "evaluation_run": 0,
            "state": "FULL_64",
            "dataset_id": domain,
            "physical_n": len(domain_errors),
            "mae_mpa": float(np.mean(domain_errors)),
        }
        for domain, domain_errors in sorted(by_domain.items())
    ]
    return overall, domain_rows


def _per_specimen(
    rows: list[dict[str, object]], method: str, metric: str
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            values[str(row["specimen_key"])].append(float(row[metric]))
    return {key: float(np.mean(items)) for key, items in values.items()}


def _bootstrap_effects(
    episodes: list[dict[str, object]], *, best_nonadaptive: str
) -> list[dict[str, object]]:
    main_full = _per_specimen(episodes, "VLM_SPATIAL_FEEDBACK", "left_error_area_mpa")
    main_early = _per_specimen(
        episodes, "VLM_SPATIAL_FEEDBACK", "early_left_error_area_mpa"
    )
    comparisons = (
        (
            "BEST_NONADAPTIVE_MINUS_MAIN_FULL_A",
            _per_specimen(episodes, best_nonadaptive, "left_error_area_mpa"),
            main_full,
        ),
        (
            "VLM_OPEN_LOOP_MINUS_MAIN_FULL_A",
            _per_specimen(episodes, "VLM_SPATIAL_OPEN_LOOP", "left_error_area_mpa"),
            main_full,
        ),
        (
            "NO_VLM_MINUS_MAIN_EARLY_A",
            _per_specimen(
                episodes, "NO_VLM_SPATIAL_FEEDBACK", "early_left_error_area_mpa"
            ),
            main_early,
        ),
    )
    identity = {
        str(row["specimen_key"]): (
            str(row["dataset_id"]),
            str(row["capture_group_id"]),
        )
        for row in episodes
    }
    expected_keys = set(main_full)
    if any(set(control) != expected_keys for _, control, _ in comparisons):
        raise ValueError("TEST comparison methods do not cover identical specimens")
    grouped_keys: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key in sorted(expected_keys):
        domain, group = identity[key]
        grouped_keys[domain][group].append(key)
    rng = np.random.default_rng(2026091501)
    draw_indices = {
        domain: rng.integers(
            0,
            len(groups),
            size=(_BOOTSTRAP_REPLICATES, len(groups)),
        )
        for domain, groups in (
            (domain, sorted(values)) for domain, values in sorted(grouped_keys.items())
        )
    }
    alpha = (1.0 - _CONFIDENCE) / 2.0
    output = []
    for name, control, main in comparisons:
        effects = {key: control[key] - main[key] for key in expected_keys}
        domain_points = defaultdict(list)
        for key, value in effects.items():
            domain_points[identity[key][0]].append(value)
        point = float(np.mean([np.mean(values) for values in domain_points.values()]))
        replicate_domains = []
        for domain, groups_by_id in sorted(grouped_keys.items()):
            groups = sorted(groups_by_id)
            sums = np.asarray(
                [sum(effects[key] for key in groups_by_id[group]) for group in groups]
            )
            counts = np.asarray([len(groups_by_id[group]) for group in groups])
            draws = draw_indices[domain]
            replicate_domains.append(
                np.sum(sums[draws], axis=1) / np.sum(counts[draws], axis=1)
            )
        replicates = np.mean(np.stack(replicate_domains, axis=1), axis=1)
        lower, upper = np.quantile(replicates, [alpha, 1.0 - alpha])
        output.append(
            {
                "comparison": name,
                "positive_means_main_lower_error": True,
                "effect_mpa": point,
                "confidence_percent": 100.0 * _CONFIDENCE,
                "ci_lower_mpa": float(lower),
                "ci_upper_mpa": float(upper),
                "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
                "domain_stratified": True,
                "capture_group_resampled": True,
                "physical_specimen_weighted": True,
                "supported": bool(lower > 0.0),
            }
        )
    return output


def run_internal_test_evaluation(
    *, project_root: str | Path, device: str
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    expansion = json.loads((output / "policy_expansion.json").read_text())
    if expansion.get("status") != "POLICY_SEEDS_1_TO_3_LOCKED":
        payload = {
            "status": "NOT_EXECUTED_POLICY_EXPANSION_NOT_LOCKED",
            "test_labels_accessed": False,
        }
        write_json(output / "policy_test_summary.json", payload)
        return payload
    vlm_test = json.loads((output / "vlm_manifest_test.json").read_text())
    if vlm_test.get("status") != "REAL_FROZEN_VLM_PERCEPTION_COMPLETE":
        raise ValueError("TEST VLM perception is incomplete")
    bank = load_feature_bank(project_root=root)
    targets = _test_target_array(output, bank)
    features = _VLMFeatures(
        bank,
        output / "vlm_actor_features_test.csv",
        required_splits=("TEST",),
    )
    predictor_gate = json.loads((output / "predictor_gate.json").read_text())
    predictor_manifest = next(
        row
        for row in predictor_gate["candidate_manifests"]
        if row["model"] == predictor_gate["selected_p_all"]
    )
    predictor, _ = load_predictor_checkpoint(
        root / predictor_manifest["checkpoint_path"], device=device
    )
    cell_costs = _cell_costs(bank)
    actor_manifests = [
        row
        for row in (
            *expansion["seed1_actor_manifests"],
            *expansion["new_actor_manifests"],
        )
        if row["method"]
        in {
            "VLM_SPATIAL_FEEDBACK",
            "NO_VLM_SPATIAL_FEEDBACK",
            "VLM_SPATIAL_OPEN_LOOP",
            "LEARNED_STATIC_TRUE",
        }
    ]
    started = time.perf_counter()
    episodes = []
    for manifest in actor_manifests:
        actor, _ = load_actor_checkpoint(
            root / manifest["checkpoint_path"], bank, device=device
        )
        _, rows = evaluate_policy(
            actor,
            str(manifest["method"]),
            predictor,
            bank,
            features,
            cell_costs,
            device=device,
            split="TEST",
            targets_mpa=targets,
        )
        for row in rows:
            row["seed_panel"] = manifest["seed_panel"]
            row["training_seed"] = manifest["training_seed"]
        episodes.extend(rows)
    for method in ("CENTER_FIRST", "GEOMETRY_SPREAD", "SERPENTINE", "RANDOM"):
        _, rows = evaluate_policy(
            None,
            method,
            predictor,
            bank,
            features,
            cell_costs,
            device=device,
            split="TEST",
            targets_mpa=targets,
        )
        for row in rows:
            row["seed_panel"] = 0
            row["training_seed"] = 0
        episodes.extend(rows)
    write_csv(output / "policy_test_episodes.csv", episodes)
    metric_rows, domain_rows = _metric_rows(episodes)
    full_row, full_domains = _full_input_rows(predictor, bank, targets, device=device)
    metric_rows.append(full_row)
    domain_rows.extend(full_domains)
    write_csv(output / "policy_test_absolute_metrics.csv", metric_rows)
    write_csv(output / "policy_test_domain_mae.csv", domain_rows)
    effects = _bootstrap_effects(
        episodes, best_nonadaptive=expansion["locked_best_nonadaptive"]
    )
    write_csv(output / "policy_test_paired_effects.csv", effects)
    method_areas = {
        method: _domain_equal_episode_score(
            [row for row in episodes if row["method"] == method],
            "left_error_area_mpa",
        )
        for method in sorted({str(row["method"]) for row in episodes})
    }
    payload = {
        "status": "INTERNAL_REUSED_TEST_EVALUATION_COMPLETE",
        "protocol_scope": "INTERNAL_REUSED_COHORT_V3_NOT_UNTOUCHED_CONFIRMATION",
        "test_physical_n": len(bank.indices("TEST")),
        "test_capture_group_n": len(
            {bank.capture_group_ids[int(index)] for index in bank.indices("TEST")}
        ),
        "method_area_mpa": method_areas,
        "locked_best_nonadaptive": expansion["locked_best_nonadaptive"],
        "paired_effects": effects,
        "engineering_accuracy_status": "ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED",
        "test_labels_accessed": True,
        "test_evaluation_count": 1,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "policy_test_summary.json", payload)
    with (root / "results/cai_agent_v3/compute_ledger.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "job": "internal_reused_test_evaluation",
                    "stage": "W5",
                    "device": device,
                    "actual_optimizer_updates": 0,
                    "status": "COMPLETED",
                    "elapsed_seconds": payload["elapsed_seconds"],
                    "test_evaluation_count": 1,
                },
                sort_keys=True,
            )
            + "\n"
        )
    return payload


__all__ = ["run_internal_test_evaluation"]
