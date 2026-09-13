"""W3 metrics and figures from saved episodes only; no model inference."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .actor_selection import (
    episode_metrics,
    read_episodes,
    write_atomic,
    write_episodes,
)
from .files import read_csv, sha256_file, write_csv
from .gates import policy_pilot_gate
from .metrics import prediction_at_budget


def values(text, cast=float):
    return [cast(v) for v in str(text).split(";") if v != ""]


def pooled_errors(rows, *, budget):
    by_specimen = defaultdict(list)
    by_repeat = defaultdict(list)
    for row in rows:
        y = float(row["target_mpa"])
        p = prediction_at_budget(
            values(row["costs"]), values(row["predictions_mpa"]), budget
        )
        by_specimen[row["specimen_key"]].append(p - y)
        by_repeat[int(row["run"])].append((p, y))
    mae = float(np.mean([np.mean(np.abs(v)) for v in by_specimen.values()]))
    mse = float(np.mean([np.mean(np.square(v)) for v in by_specimen.values()]))
    r2 = []
    for pairs in by_repeat.values():
        p, y = np.asarray(pairs).T
        denominator = np.sum((y - y.mean()) ** 2)
        r2.append(
            float(1 - np.sum((p - y) ** 2) / denominator)
            if denominator > 0
            else float("nan")
        )
    return {
        "mae_mpa": mae,
        "mse_mpa2": mse,
        "rmse_mpa": float(np.sqrt(mse)),
        "r2_repeat_mean": float(np.mean(r2)),
        "physical_n": len(by_specimen),
    }


def method_metrics(rows):
    metrics = episode_metrics(rows)
    endpoint = pooled_errors(rows, budget=0.25)
    return {
        "method": rows[0]["method"],
        "scope": "VALID_PILOT",
        "episode_count": len(rows),
        "validation_area_mpa": metrics["left_error_area_mpa"],
        "validation_early_area_mpa": metrics["early_left_error_area_mpa"],
        "validation_objective_mpa": metrics["trajectory_objective_mpa"],
        "domain_equal_endpoint_mae_mpa": metrics["final_error_mpa"],
        "endpoint_pooled_mae_mpa": endpoint["mae_mpa"],
        "endpoint_pooled_rmse_mpa": endpoint["rmse_mpa"],
        "endpoint_r2_repeat_mean": endpoint["r2_repeat_mean"],
        "physical_n": endpoint["physical_n"],
    }


def summarize(root):
    from .w2_replay import update_usage
    from .w3_pilot import ART, RUN, TASK_ID
    from .w3_pilot import rows as ledger_rows

    out = root / RUN
    art = root / ART
    fixed = (
        read_episodes(out / "fixed_episodes.csv.gz")
        if (out / "fixed_episodes.csv.gz").exists()
        else []
    )
    jobs = [json.loads(p.read_text()) for p in sorted((out / "jobs").glob("*.json"))]
    episodes = fixed.copy()
    manifests = []
    initial = []
    progress = []
    for job in jobs:
        m = job["manifest"]
        manifests.append(m)
        progress.extend(job["progress"])
        episodes.extend(read_episodes(root / m["selected_episodes"]))
        initial.append(
            {
                k: m[k]
                for k in (
                    "method",
                    "training_seed",
                    "initial_state_dict_sha256",
                    "parameter_count",
                    "architecture",
                )
            }
        )
    write_episodes(out / "policy_validation_episodes.csv.gz", episodes)
    if progress:
        write_csv(out / "policy_training_progress.csv", progress)
    if initial:
        write_csv(out / "initialization_summary.csv", initial)
    grouped = defaultdict(list)
    for row in episodes:
        grouped[row["method"]].append(row)
    metrics = []
    absolute = []
    domains = []
    for method, group in sorted(grouped.items()):
        result = method_metrics(group)
        manifest = next((m for m in manifests if m["method"] == method), None)
        result.update(
            selected_update=manifest["selected_update"] if manifest else 0,
            actual_updates=manifest["updates_completed"] if manifest else 0,
        )
        metrics.append(result)
        for budget in (0.0, 0.0625, 0.125, 0.1875, 0.25):
            absolute.append(
                dict(
                    method=method,
                    budget=budget,
                    scope="VALID_PILOT",
                    **pooled_errors(group, budget=budget),
                )
            )
        for domain in sorted({r["dataset_id"] for r in group}):
            local = [r for r in group if r["dataset_id"] == domain]
            domains.append(dict(dataset_id=domain, **method_metrics(local)))
    write_csv(out / "policy_pilot_metrics.csv", metrics)
    write_csv(out / "absolute_cai_performance.csv", absolute)
    write_csv(out / "per_domain_metrics.csv", domains)
    complete = len(jobs) == 5 and len(episodes) == 650
    gate = {
        "status": "PILOT_INCOMPLETE",
        "pilot_expansion_condition_met": False,
        "automatically_execute_next_stage": False,
    }
    effects = []
    paired = []
    if complete:
        scores = {r["method"]: r for r in metrics}
        fixed_names = [
            "CENTER_FIRST",
            "GEOMETRY_SPREAD",
            "SERPENTINE",
            "RANDOM",
            "LEARNED_STATIC_TRUE",
        ]
        best = min(fixed_names, key=lambda n: (scores[n]["validation_area_mpa"], n))
        main = scores["VLM_SPATIAL_FEEDBACK"]
        decision = policy_pilot_gate(
            main_area_mpa=main["validation_area_mpa"],
            best_nonadaptive_area_mpa=scores[best]["validation_area_mpa"],
            open_loop_area_mpa=scores["VLM_SPATIAL_OPEN_LOOP"]["validation_area_mpa"],
        )
        gate.update(
            status=decision.status,
            pilot_expansion_condition_met=decision.passed,
            reasons=decision.reasons,
            best_nonadaptive=best,
        )
        contrasts = [
            ("fixed", best, "left_error_area_mpa", "validation_area_mpa"),
            (
                "feedback",
                "VLM_SPATIAL_OPEN_LOOP",
                "left_error_area_mpa",
                "validation_area_mpa",
            ),
            (
                "vlm_early",
                "NO_VLM_SPATIAL_FEEDBACK",
                "early_left_error_area_mpa",
                "validation_early_area_mpa",
            ),
            (
                "spatial",
                "VLM_MEAN_FEEDBACK",
                "left_error_area_mpa",
                "validation_area_mpa",
            ),
        ]
        for name, control, metric, field in contrasts:
            effects.append(
                {
                    "effect": name,
                    "comparator": control,
                    "positive_favors_main": True,
                    "effect_mpa": scores[control][field] - main[field],
                }
            )
            control_by_key = defaultdict(list)
            for row in grouped[control]:
                control_by_key[row["specimen_key"]].append(float(row[metric]))
            for row in grouped["VLM_SPATIAL_FEEDBACK"]:
                paired.append(
                    {
                        "effect": name,
                        "specimen_key": row["specimen_key"],
                        "dataset_id": row["dataset_id"],
                        "capture_group_id": row["capture_group_id"],
                        "paired_gain_mpa": float(
                            np.mean(control_by_key[row["specimen_key"]])
                        )
                        - float(row[metric]),
                    }
                )
        write_csv(out / "pilot_effects.csv", effects)
        write_csv(out / "paired_effects.csv", paired)
        direction = []
        for name in [r["effect"] for r in effects]:
            for domain in sorted({r["dataset_id"] for r in paired}):
                vals = [
                    r["paired_gain_mpa"]
                    for r in paired
                    if r["effect"] == name and r["dataset_id"] == domain
                ]
                direction.append(
                    {
                        "effect": name,
                        "dataset_id": domain,
                        "physical_n": len(vals),
                        "gain_mpa": float(np.mean(vals)),
                    }
                )
        write_csv(out / "per_domain_effects.csv", direction)
    write_atomic(out / "pilot_gate.json", gate)
    ledger = ledger_rows(root)
    local = [r for r in ledger if r.get("task_id") == TASK_ID]
    (out / "ledger_view.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in local)
    )
    final = {
        "task_id": TASK_ID,
        "implementation_status": "EXECUTION_COMPLETE"
        if complete
        else "EXECUTION_INCOMPLETE",
        "protocol_conformance": "PENDING_FINAL_REVIEW",
        "final_episodes": len(episodes),
        "VALID_physical_n": 50,
        "completed_jobs": len(jobs),
        "actual_updates": sum(r.get("actual_optimizer_updates", 0) or 0 for r in local),
        "task_used_upper_bound": update_usage(local),
        "cumulative_used_upper_bound": update_usage(ledger),
        "cumulative_cap": 40014,
        "new_gpu_seconds": sum(
            r.get("elapsed_gpu_seconds", 0)
            for r in local
            if r.get("status") == "GPU_SESSION_ENDED"
        ),
        "gpu_seconds_cap": 21600,
        "pilot_expansion_condition_met": gate["pilot_expansion_condition_met"],
        "scientific_evidence": "VALID_ONLY_PILOT_NOT_INDEPENDENT_CONFIRMATION",
        "engineering_status": "ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED",
        "prohibited_stage_counts": {
            "W2_training": 0,
            "new_VLM_calls": 0,
            "CNN_encoding": 0,
            "GDFS": 0,
            "seed2_or_3": 0,
            "TEST_prediction_or_label_join": 0,
            "STOP": 0,
        },
        "historical_invalidated_results": "UNCHANGED",
        "downstream_authorization": "NOT_AUTHORIZED_BEYOND_W3_SEED1_VALID",
    }
    write_atomic(out / "final_manifest.json", final)
    write_atomic(
        art / "RESULT_POINTER.json",
        {
            "final_manifest": f"{RUN}/final_manifest.json",
            "actor_manifests": f"{RUN}/actor_manifests.json",
            "downstream_authorization": final["downstream_authorization"],
        },
    )
    return final


def export_figures(root):
    import matplotlib
    from PIL import Image

    from cmc_bbdm.cai_active_image.environment import NativeCellGrid
    from cmc_bbdm.vlm_cscan.runtime import render_surface_inputs

    from .diagnostics import _draw_cells, _measured_image
    from .w3_pilot import DATA, RUN

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out = root / RUN
    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)
    cases = read_csv(out / "case_manifest.csv")
    assert len(cases) <= 3
    keys = {r["specimen_key"] for r in cases}
    # Only the frozen VALID keys and image columns are selected; target columns are not joined.
    queue = {
        r["specimen_key"]: {
            k: r[k]
            for k in (
                "impacted_surface_path",
                "registered_cscan_crop_path",
                "surface_sha256",
                "registered_cscan_crop_sha256",
                "split",
            )
        }
        for r in read_csv(root / DATA / "candidate_queue.csv")
        if r["specimen_key"] in keys
    }
    features = {
        r["specimen_key"]: r
        for r in read_csv(root / DATA / "vlm_actor_features_fit.csv")
        if r["specimen_key"] in keys
    }
    external = Path(
        json.loads((root / DATA / "feature_bank_manifest.json").read_text())[
            "encoder_execution_root"
        ]
    )
    all_episodes = read_episodes(out / "policy_validation_episodes.csv.gz")
    selected = {
        r["specimen_key"]: r
        for r in all_episodes
        if r["method"] == "VLM_SPATIAL_FEEDBACK" and r["specimen_key"] in keys
    }
    outputs = []
    for key in [r["specimen_key"] for r in cases]:
        if key not in selected:
            outputs.append(
                {"specimen_key": key, "status": "BLOCKED_MAIN_EPISODE_MISSING"}
            )
            continue
        source = queue[key]
        assert source["split"] == "VALID"
        surface_path = external / source["impacted_surface_path"]
        cscan_path = external / source["registered_cscan_crop_path"]
        if not surface_path.exists() or not cscan_path.exists():
            outputs.append(
                {
                    "specimen_key": key,
                    "status": "BLOCKED_SOURCE_IMAGE_MISSING",
                    "paths": [str(surface_path), str(cscan_path)],
                }
            )
            continue
        assert sha256_file(surface_path) == source["surface_sha256"]
        assert sha256_file(cscan_path) == source["registered_cscan_crop_sha256"]
        with Image.open(surface_path) as image:
            surface = np.asarray(
                render_surface_inputs(image, max_edge=1024).clean, dtype=np.uint8
            )
        with Image.open(cscan_path) as image:
            cscan = np.asarray(image.convert("RGB"), dtype=np.uint8)
        grid = NativeCellGrid.from_shape(cscan.shape[:2])
        surface_grid = NativeCellGrid.from_shape(surface.shape[:2])
        row = selected[key]
        cells = values(row["cells"], int)
        costs = values(row["costs"])
        preds = values(row["predictions_mpa"])
        trace = json.loads(row["execution_trace"])
        assert [t["cell"] for t in trace] == cells

        def draw_grid(ax, g):
            for x in sorted({c.col_start for c in g.cells} | {g.cells[-1].col_stop}):
                ax.axvline(x - 0.5, color="gray", lw=0.4)
            for y in sorted({c.row_start for c in g.cells} | {g.cells[-1].row_stop}):
                ax.axhline(y - 0.5, color="gray", lw=0.4)

        def save(fig, name, case_key=key):
            path = figure_dir / f"{case_key.replace(':', '_')}_{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return str(path.relative_to(root))

        generated = []
        for mode in ("surface_cues", "first_action"):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(surface)
            draw_grid(ax, surface_grid)
            cue_cells = [
                i
                for i, v in enumerate(values(features[key]["region_indicator"]))
                if v > 0
            ]
            show = cue_cells if mode == "surface_cues" else cells[:1]
            _draw_cells(ax, surface_grid, show, color="#D55E00", fill=True)
            ax.set_title(
                "VALID surface + frozen VLM cues"
                if mode == "surface_cues"
                else f"VALID real first action: cell {cells[0]}"
            )
            ax.axis("off")
            generated.append(save(fig, mode))
        for count in sorted({min(k, len(cells)) for k in (1, 4, 8, len(cells))}):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(_measured_image(cscan, grid, cells[:count]))
            draw_grid(ax, grid)
            _draw_cells(ax, grid, cells[:count], color="#009E73", fill=False)
            suffix = "budget endpoint"
            if count < len(cells):
                _draw_cells(
                    ax, grid, [trace[count]["cell"]], color="#D55E00", fill=False
                )
                suffix = f"next cell {trace[count]['cell']} (call {trace[count]['actor_call_index']})"
            ax.set_title(
                f"VALID k={count}, cost={costs[count]:.4f}\n{suffix}; gray = unobserved"
            )
            ax.axis("off")
            generated.append(save(fig, f"measured_{count}"))
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.step(
            [*costs, 0.25],
            [*preds, preds[-1]],
            where="post",
            label="Current prediction",
        )
        ax.axhline(
            float(row["target_mpa"]),
            ls="--",
            color="black",
            label="VALID target (scoring only)",
        )
        ax.set(
            xlim=(0, 0.25),
            xlabel="Native pixel cost",
            ylabel="CAI (MPa)",
            title=f"VALID pilot: {key}",
        )
        ax.legend(fontsize=8)
        generated.append(save(fig, "prediction"))
        outputs.append(
            {
                "specimen_key": key,
                "status": "FIGURES_EXPORTED",
                "paths": generated,
                "source": "SAVED_REAL_EXECUTION_TRACE",
                "new_model_forwards": 0,
            }
        )
    result = {
        "status": "FIGURES_COMPLETE"
        if all(r["status"] == "FIGURES_EXPORTED" for r in outputs)
        else "FIGURES_PARTIALLY_BLOCKED",
        "cases": outputs,
        "new_model_forwards": 0,
    }
    write_atomic(out / "figure_manifest.json", result)
    return result
