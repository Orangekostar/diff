"""Read only bounded W2 model/ledger evidence; no inference or optimization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch


def audit(root: Path) -> dict:
    output = root / "results/cai_agent_v3/new_protocol"
    ledger = root / "results/cai_agent_v3/compute_ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    by_stage = defaultdict(lambda: {"known_updates": 0, "unknown_upper_bound": 0})
    for row in rows:
        actual = row.get("actual_optimizer_updates")
        stage = by_stage[row.get("stage", "UNSPECIFIED")]
        if isinstance(actual, int):
            stage["known_updates"] += actual
        elif actual is None:
            stage["unknown_upper_bound"] += row.get(
                "actual_optimizer_updates_upper_bound", 0
            )
    completed = {
        r["run_id"] for r in rows if r.get("run_id") and r.get("status") == "COMPLETED"
    }
    pending = {
        r["run_id"]: r
        for r in rows
        if r.get("run_id")
        and "optimizer_update_reservation" in r
        and r["run_id"] not in completed
    }
    for row in pending.values():
        by_stage[row.get("stage", "UNSPECIFIED")]["unknown_upper_bound"] += row[
            "optimizer_update_reservation"
        ]
    known = sum(s["known_updates"] for s in by_stage.values())
    unknown = sum(s["unknown_upper_bound"] for s in by_stage.values())
    # Only this study's explicit models directory, including any selection_history.
    paths = sorted((output / "models").rglob("*.pt"))
    checkpoints = []
    for path in paths:
        if not (path.name.startswith(("predictor_", "reward_predictor_", "update_"))):
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        manifest = payload["manifest"]
        checkpoints.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model": payload["model_name"],
                "update": manifest.get("update", manifest.get("selected_update")),
                "updates_completed": manifest.get("updates_completed"),
                "query_fold": payload.get("query_fold"),
                "validation_identity_present": "validation" in manifest
                or "validation_identity" in manifest,
                "selection_archive_present": "selection_archive" in manifest,
            }
        )
    progress = []
    for filename in [
        "predictor_training_progress.csv",
        "oof_predictor_training_progress.csv",
    ]:
        with (output / filename).open() as handle:
            progress.extend(
                dict(row, source=filename) for row in csv.DictReader(handle)
            )
    groups = defaultdict(list)
    for row in progress:
        groups[(row["model"], row.get("fold"))].append(int(row["update"]))
    coverage = []
    for (name, fold), updates in groups.items():
        stored = [
            p["update"]
            for p in checkpoints
            if p["model"] == name
            and p["query_fold"] == (None if fold is None else int(fold))
        ]
        coverage.append(
            {
                "model": name,
                "fold": fold,
                "actual_selection_updates": updates,
                "retained_updates": stored,
                "missing_updates": sorted(set(updates) - set(stored)),
            }
        )
    precision = json.loads((output / "cost_precision_audit.json").read_text())
    time_seconds = sum(
        r.get("elapsed_seconds", 0)
        for r in rows
        if str(r.get("device", "")).startswith("cuda")
    )
    return {
        "scope": "metadata and files only; no model inference, training, VLM or TEST scoring",
        "ledger_path": ledger.relative_to(root).as_posix(),
        "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "requested_new_protocol_ledger_exists": (
            output / "compute_ledger.jsonl"
        ).exists(),
        "checked_model_directory": str(output / "models"),
        "all_model_file_paths": [p.relative_to(root).as_posix() for p in paths],
        "w2_checkpoints": checkpoints,
        "selection_coverage": coverage,
        "actual_selection_point_count": len(progress),
        "missing_selection_point_count": sum(
            len(r["missing_updates"]) for r in coverage
        ),
        "historical_selection_recoverable": all(
            not r["missing_updates"] for r in coverage
        ),
        "historical_validation_mask_changes": precision[
            "validation_rows_with_mask_change"
        ],
        "ledger_by_stage": dict(by_stage),
        "known_updates": known,
        "unknown_updates_upper_bound": unknown,
        "used_upper_bound": known + unknown,
        "cumulative_limit": 28100,
        "remaining_lower_bound": 28100 - known - unknown,
        "w2_replay_registered_upper_bound": 12000,
        "w2_additional_update_capacity_needed": max(0, known + unknown + 12000 - 28100),
        "ledger_recorded_gpu_seconds_lower_bound": time_seconds,
        "gpu_time_note": "incomplete durations: interrupted static and some prechecks; not an upper bound",
        "this_task_real_optimizer_updates": 0,
        "this_task_new_vlm_calls": 0,
        "this_task_new_test_perception_or_scores": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(args.project_root.resolve(strict=True))
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
