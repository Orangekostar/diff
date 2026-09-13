"""Summarize saved W2 predictions only: no model loading or inference."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .files import read_csv, write_csv, write_json
from .w2_replay import ART, REPLAY_ID, RUN, _rows, update_usage


def summarize(root):
    output = root / RUN
    art = root / ART
    gate = json.loads((output / "predictor_gate.json").read_text())
    oof_path = output / "oof_readiness.json"
    if not oof_path.exists():
        oof = {
            "status": "NOT_EXECUTED_PREDICTOR_NOT_READY"
            if gate["status"] != "PREDICTOR_READY"
            else "NOT_EXECUTED_YET",
            "actual_optimizer_updates": 0,
        }
        if gate["status"] != "PREDICTOR_READY":
            write_json(oof_path, oof)
    else:
        oof = json.loads(oof_path.read_text())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    indices = read_csv(
        root / "results/cai_agent_v3/new_protocol/feature_bank_index.csv"
    )
    cost_rows = []
    fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
    for manifest, color in zip(
        gate["candidate_manifests"], ["#0072B2", "#D55E00", "#009E73"], strict=True
    ):
        name = manifest["model"]
        path = (
            output
            / "candidate_state_predictions"
            / f"A_{name}"
            / f"update_{manifest['selected_update']:06d}.npz"
        )
        with np.load(path) as data:
            grouped = defaultdict(list)
            for j, (i, route) in enumerate(
                zip(data["specimen_indices"], data["route_names"], strict=True)
            ):
                grouped[(int(i), str(route))].append(j)
            for budget in (0.0, 0.0625, 0.125, 0.1875, 0.25):
                states = []
                for (i, _), rows in grouped.items():
                    eligible = [j for j in rows if data["costs"][j] <= budget]
                    j = eligible[-1]
                    states.append(
                        (
                            i,
                            float(data["predictions_mpa"][j]),
                            float(data["targets_mpa"][j]),
                        )
                    )
                domains = sorted({indices[i]["dataset_id"] for i, _, _ in states})
                for domain in ["POOLED", *domains]:
                    subset = [
                        r
                        for r in states
                        if domain == "POOLED" or indices[r[0]]["dataset_id"] == domain
                    ]
                    errors = np.array([p - y for _, p, y in subset])
                    cost_rows.append(
                        {
                            "model": name,
                            "scope": "VALID_FIXED_PREFIX",
                            "domain": domain,
                            "budget": budget,
                            "physical_n": len({i for i, _, _ in subset}),
                            "mae_mpa": float(np.abs(errors).mean()),
                            "rmse_mpa": float(np.sqrt((errors**2).mean())),
                        }
                    )
        selected = [
            r for r in cost_rows if r["model"] == name and r["domain"] == "POOLED"
        ]
        ax.plot(
            [r["budget"] for r in selected],
            [r["mae_mpa"] for r in selected],
            marker="o",
            label=name,
            color=color,
        )
    ax.set(
        xlabel="Native-pixel acquisition cost",
        ylabel="Pooled VALID MAE (MPa)",
        title="VALID fixed prefixes · 50 specimens",
    )
    ax.legend(fontsize=8)
    fig.savefig(output / "valid_error_cost.png", dpi=200)
    plt.close(fig)
    write_csv(output / "predictor_cost_metrics.csv", cost_rows)
    oof_rows = []
    if oof.get("fold_manifests"):
        states = read_csv(output / "oof_state_predictions.csv")
        full = [r for r in states if r["state_role"] == "FULL_INPUT_DIAGNOSTIC"]
        fig, ax = plt.subplots(figsize=(4.4, 4.2), layout="constrained")
        for fold, color in zip(
            range(3), ["#0072B2", "#D55E00", "#009E73"], strict=True
        ):
            rows = [r for r in full if int(r["fold"]) == fold]
            target = np.array([float(r["target_mpa"]) for r in rows])
            prediction = np.array([float(r["prediction_mpa"]) for r in rows])
            ax.scatter(
                target,
                prediction,
                s=13,
                alpha=0.7,
                color=color,
                label=f"Fold {fold} (n={len(rows)})",
            )
            oof_rows.append(
                {
                    "fold": fold,
                    "scope": "TRAIN_OOF_QUERY_FULL_64",
                    "physical_n": len(rows),
                    "mae_mpa": float(np.abs(target - prediction).mean()),
                    "rmse_mpa": float(np.sqrt(((target - prediction) ** 2).mean())),
                }
            )
        values = [float(r[k]) for r in full for k in ("target_mpa", "prediction_mpa")]
        lo, hi = min(values) - 10, max(values) + 10
        ax.plot([lo, hi], [lo, hi], color="0.5", linestyle="--", linewidth=1)
        ax.set(
            xlim=(lo, hi),
            ylim=(lo, hi),
            xlabel="Observed CAI (MPa)",
            ylabel="OOF predicted CAI (MPa)",
            title="TRAIN OOF query · full 64 cells",
        )
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        fig.savefig(output / "oof_full_predictions.png", dpi=200)
        plt.close(fig)
        write_csv(output / "oof_query_metrics.csv", oof_rows)
        comparison = []
        for m in oof["fold_manifests"]:
            comparison.append(
                {
                    k: m[k]
                    for k in [
                        "fold",
                        "model",
                        "fit_physical_n",
                        "query_physical_n",
                        "fit_capture_group_n",
                        "query_capture_group_n",
                        "selected_update",
                        "updates_completed",
                        "readiness_status",
                    ]
                }
                | m["metrics"]
            )
        write_csv(output / "oof_comparison.csv", comparison)
    rows = _rows(root)
    local = [r for r in rows if r.get("replay_id") == REPLAY_ID]
    result = {
        "replay_id": REPLAY_ID,
        "implementation_status": "W2_EXECUTION_COMPLETE",
        "predictor_readiness": gate["status"],
        "selected_p_all": gate["selected_p_all"],
        "reward_readiness": oof["status"],
        "scientific_scope": "W2 TRAIN/VALID only; historical W2-W4 remain invalidated",
        "downstream_authorization": "NOT_AUTHORIZED_W2_ONLY",
        "engineering_status": "ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED",
        "new_actual_updates": sum(
            r.get("actual_optimizer_updates", 0) or 0 for r in local
        ),
        "new_used_upper_bound": update_usage(local),
        "cumulative_used_upper_bound": update_usage(rows),
        "new_gpu_seconds": sum(
            r.get("elapsed_gpu_seconds", 0)
            for r in local
            if r.get("status") == "GPU_SESSION_ENDED"
        ),
        "cumulative_cap": 34264,
        "new_updates_cap": 12000,
        "new_gpu_seconds_cap": 21600,
        "new_vlm_calls": 0,
        "actor_forward_or_training": 0,
        "new_test_predictions_or_scores": 0,
        "predictor_gate_path": (output / "predictor_gate.json")
        .relative_to(root)
        .as_posix(),
        "oof_readiness_path": oof_path.relative_to(root).as_posix(),
    }
    result["status"] = (
        "W2_READY_FOR_FUTURE_POLICY_TASK"
        if gate["status"] == "PREDICTOR_READY"
        and oof["status"] == "REWARD_MODELS_READY"
        else "W2_COMPLETED_READINESS_NOT_SUPPORTED"
    )
    if oof["status"] == "NOT_EXECUTED_YET":
        result["implementation_status"] = "W2_B_PENDING"
        result["status"] = "W2_B_PENDING"
    result["checkpoint_selection_status"] = {
        "common_predictors": "COMPLETE",
        "oof_predictors": "COMPLETE" if oof.get("fold_manifests") else oof["status"],
    }
    write_json(output / "final_manifest.json", result)
    write_json(
        art / "RESULT_POINTER.json",
        {
            "final_manifest": (output / "final_manifest.json")
            .relative_to(root)
            .as_posix(),
            "downstream_authorization": "NOT_AUTHORIZED_W2_ONLY",
        },
    )
    return result
