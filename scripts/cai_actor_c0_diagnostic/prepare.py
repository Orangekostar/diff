"""Validate immutable inputs and select the six outcome-independent VALID cases."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .context import ResourceLedger, TaskContext, atomic_json, sha256_file


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError("cannot write an empty CSV")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(destination)


def _selection_score(key: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"ACTOR_C0_VIS_V2|{key}".encode()).hexdigest()
    return digest, key


def select_cases(
    index_path: str | Path,
    *,
    fixed: Iterable[str],
    remaining_domains: Iterable[str],
) -> list[dict[str, str]]:
    rows = _read_csv(index_path)
    by_key = {row["specimen_key"]: row for row in rows}
    output: list[dict[str, str]] = []
    for key in fixed:
        row = by_key.get(key)
        if row is None or row.get("split") != "VALID":
            raise ValueError(f"missing fixed VALID case: {key}")
        output.append({**row, "selection_rule": "FIXED_CASE", "selection_sha256": ""})
    for domain in remaining_domains:
        candidates = [
            row
            for row in rows
            if row.get("dataset_id") == domain and row.get("split") == "VALID"
        ]
        if not candidates:
            raise ValueError(f"no VALID case for remaining domain: {domain}")
        selected = min(candidates, key=lambda row: _selection_score(row["specimen_key"]))
        digest, _ = _selection_score(selected["specimen_key"])
        output.append(
            {**selected, "selection_rule": "HASH_MIN_VALID", "selection_sha256": digest}
        )
    if len(output) != 6 or len({row["specimen_key"] for row in output}) != 6:
        raise ValueError("case selection must yield six unique specimens")
    return output


def _require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash changed: {actual}")


def _markdown_bindings(context: TaskContext, rows: list[dict[str, Any]]) -> str:
    keys = "\n".join(f"- `{row['specimen_key']}` ({row['selection_rule']})" for row in rows)
    return f"""# Source and Task Bindings

- Task: `{context.scope['task_id']}`
- Branch: `{context.branch}`
- Source commit: `{context.scope['source_commit']}`
- Scope SHA-256: `{context.scope_sha256}`
- Mode: frozen VALID-only policy diagnostics
- Forbidden work: training, Qwen/CNN/OOF forward, TEST, bootstrap, paper edits

## Selected Cases

{keys}
"""


def prepare_stage(context: TaskContext) -> dict[str, Any]:
    scope = context.scope
    data = context.path("data")
    c_root = context.path("c_release")
    n_root = context.path("historical_w3")
    output = context.path("output")
    artifacts = context.path("artifacts")
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    index_path = data / "feature_bank_index.csv"
    signature = context.phase_signature("prepare", (context.config_path, index_path))
    if context.stage_complete("prepare", signature):
        return {"status": "PREPARE_REUSED", "case_count": 6}

    index_rows = _read_csv(index_path)
    bank_manifest = json.loads(
        (data / "feature_bank_manifest.json").read_text(encoding="utf-8")
    )
    immutable_source_roots = (bank_manifest["encoder_execution_root"],)
    split_counts = {
        split: sum(row["split"] == split for row in index_rows)
        for split in ("TRAIN", "VALID", "TEST")
    }
    if split_counts != {"TRAIN": 161, "VALID": 50, "TEST": 65}:
        raise ValueError(f"feature-bank split counts changed: {split_counts}")
    cases = select_cases(
        index_path,
        fixed=scope["cases"]["fixed"],
        remaining_domains=scope["cases"]["remaining_domains"],
    )

    queue = {row["specimen_key"]: row for row in _read_csv(data / "candidate_queue.csv")}
    c_inputs = {
        row["specimen_key"]: row
        for row in _read_csv(c_root / "vlm/input_manifest.csv")
    }
    selected_rows = []
    index_by_key = {row["specimen_key"]: index for index, row in enumerate(index_rows)}
    for row in cases:
        key = row["specimen_key"]
        source = queue[key]
        rendered = c_inputs[key]
        source_path = context.source_path(
            source["impacted_surface_path"], additional_roots=immutable_source_roots
        )
        _require_hash(source_path, source["surface_sha256"], f"surface {key}")
        if rendered["source_sha256"] != source["surface_sha256"]:
            raise ValueError(f"C input source binding changed: {key}")
        selected_rows.append(
            {
                "order": len(selected_rows),
                "specimen_key": key,
                "dataset_id": row["dataset_id"],
                "split": row["split"],
                "feature_bank_index": index_by_key[key],
                "capture_group_id": source["capture_group_id"],
                "target_mpa": source["author_cai_mpa"],
                "selection_rule": row["selection_rule"],
                "selection_sha256": row["selection_sha256"],
                "source_path": source["impacted_surface_path"],
                "source_sha256": source["surface_sha256"],
                "clean_sha256": rendered["clean_sha256"],
                "c_prior_status": rendered["status"],
            }
        )

    c_model = scope["models"]["C"]
    n_model = scope["models"]["N"]
    predictor = context.path("predictor")
    c_checkpoint = context.root / c_model["checkpoint"]
    n_checkpoint = context.root / n_model["checkpoint"]
    _require_hash(c_checkpoint, c_model["checkpoint_sha256"], "C checkpoint")
    _require_hash(n_checkpoint, n_model["checkpoint_sha256"], "N checkpoint")
    _require_hash(predictor, scope["models"]["P_all_sha256"], "P_all checkpoint")
    c_manifest = c_root / "w3/models/vlm_spatial_feedback/manifest.json"
    n_selection = n_root / "models/selection_history/no_vlm_spatial_feedback_seed1/selection.json"
    c_meta = json.loads(c_manifest.read_text(encoding="utf-8"))
    n_meta = json.loads(n_selection.read_text(encoding="utf-8"))
    if c_meta["selected_update"] != c_model["selected_update"]:
        raise ValueError("C selected update changed")
    if n_meta["selected_update"] != n_model["selected_update"]:
        raise ValueError("N selected update changed")

    runtime = c_root / "runtime_lock.json"
    runtime_value = json.loads(runtime.read_text(encoding="utf-8"))
    model_bindings = {
        "task_id": scope["task_id"],
        "C": {**c_model, "manifest": str(c_manifest.relative_to(context.root))},
        "N": {**n_model, "selection": str(n_selection.relative_to(context.root))},
        "P_all": {
            "checkpoint": str(predictor.relative_to(context.root)),
            "checkpoint_sha256": scope["models"]["P_all_sha256"],
        },
        "python": runtime_value["actor_python"],
        "python_version": runtime_value["python_version"],
        "source_root": "src",
    }
    _write_csv(output / "selected_cases.csv", selected_rows)
    atomic_json(output / "model_bindings.json", model_bindings)
    ResourceLedger(output / "resource_usage.json", scope["limits"])
    atomic_json(
        output / "diagnostic_lock.json",
        {
            "task_id": scope["task_id"],
            "scope_sha256": context.scope_sha256,
            "branch": context.branch,
            "head_at_prepare": context.head,
            "source_commit": scope["source_commit"],
            "config": str(context.config_path.relative_to(context.root)),
            "config_sha256": sha256_file(context.config_path),
            "feature_bank_index_sha256": sha256_file(index_path),
            "split_counts": split_counts,
            "python": sys.executable,
            "python_runtime": platform.python_version(),
            "status": "INPUTS_LOCKED",
        },
    )
    (artifacts / "SOURCE_AND_TASK_BINDINGS.md").write_text(
        _markdown_bindings(context, selected_rows), encoding="utf-8"
    )
    result = {"status": "INPUTS_LOCKED", "case_count": 6, "split_counts": split_counts}
    context.complete_stage("prepare", signature, result)
    return result


__all__ = ["prepare_stage", "select_cases"]
