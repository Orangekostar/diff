"""Validate and publish the bounded C retraining release."""

from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.cai_c_retrain.context import (
    BRANCH,
    METHOD_SEEDS,
    SOURCE_COMMIT,
    TaskContext,
    atomic_json,
    sha256_file,
)

EXPECTED_METHOD_COUNTS = {
    "CENTER_FIRST": 50,
    "GEOMETRY_SPREAD": 50,
    "SERPENTINE": 50,
    "RANDOM": 250,
    "LEARNED_STATIC_TRUE": 50,
    "NO_VLM_SPATIAL_FEEDBACK": 50,
    "VLM_MEAN_FEEDBACK": 50,
    "VLM_SPATIAL_FEEDBACK": 50,
    "VLM_SPATIAL_OPEN_LOOP": 50,
}
PLAN_PATH = "docs/superpowers/plans/2026-09-20-c-render-retrain-release.md"
RESULTS_COMMIT_MESSAGE = "results: complete CAI C retraining release"
RESULT_PATHS = (
    "results/cai_agent_v3/c_render_retrain/r1_331f5295",
    "artifacts/cai_agent_v3/c_render_retrain/r1_331f5295",
    "paper_cai_aei/r2_c_331f5295",
    "results/cai_agent_v3/compute_ledger.jsonl",
    "scripts/cai_c_retrain",
    "tests/test_cai_c_retrain_evidence.py",
    "tests/test_cai_c_retrain_paper.py",
    "tests/test_cai_c_retrain_validate.py",
    "tests/test_cai_c_retrain_vlm.py",
    PLAN_PATH,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
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


def _require_hashes(root: Path, records: dict[str, str], *, label: str) -> None:
    for relative, digest in records.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"{label} output changed: {relative}")


def _vlm_checks(context: TaskContext) -> dict[str, Any]:
    root = context.path("vlm")
    manifest_path = root / "vlm_manifest_fit.json"
    features_path = root / "vlm_actor_features_fit.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _read_csv(features_path)
    input_manifest_path = root / "input_manifest.csv"
    config_lock_path = root / "config_lock.json"
    input_rows = _read_csv(input_manifest_path)
    config_lock = json.loads(config_lock_path.read_text(encoding="utf-8"))
    protocol = json.loads(
        (context.path("output") / "protocol_lock.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("status") != "C_PRIOR_COMPLETE"
        or manifest.get("rows") != 211
        or manifest.get("terminal_rows") != 211
        or manifest.get("train_rows") != 161
        or manifest.get("valid_rows") != 50
        or manifest.get("test_rows") != 0
        or manifest.get("feature_csv_sha256") != sha256_file(features_path)
        or manifest.get("first_new_case_wiring", {}).get("status") != "PASS"
        or len(rows) != 211
        or len({row["specimen_key"] for row in rows}) != 211
        or any(row["split"] not in {"TRAIN", "VALID"} for row in rows)
        or manifest.get("input_manifest_sha256")
        != sha256_file(input_manifest_path)
        or manifest.get("config_lock_sha256") != sha256_file(config_lock_path)
        or len(input_rows) != 211
        or len({row["specimen_key"] for row in input_rows}) != 211
        or any(row["split"] not in {"TRAIN", "VALID"} for row in input_rows)
        or config_lock.get("prompt_sha256") != protocol["prompt_sha256"]
        or config_lock.get("model_config") != protocol["model_config"]
        or config_lock.get("render_id") != "R1"
        or config_lock.get("image_order") != ["clean", "numbered"]
        or config_lock.get("cohort")
        != {"rows": 211, "train": 161, "valid": 50, "test": 0}
        or config_lock.get("test_accessed") is not False
    ):
        raise ValueError("Q1/Q2 failed: current C prior is incomplete or includes TEST")
    state_counts = Counter(row["status"] for row in manifest["states"])
    input_status = {row["specimen_key"]: row["status"] for row in input_rows}
    for state in manifest["states"]:
        run = root / "runs" / state["specimen_key"].replace(":", "__") / "state.json"
        if not run.is_file():
            raise FileNotFoundError(run)
        payload = json.loads(run.read_text(encoding="utf-8"))
        if payload.get("status") != state["status"]:
            raise ValueError(
                f"VLM terminal state differs from manifest: {state['specimen_key']}"
            )
        if input_status.get(state["specimen_key"]) != state["status"]:
            raise ValueError(
                f"VLM input manifest differs from state: {state['specimen_key']}"
            )
        if len(payload.get("attempts", [])) > 2:
            raise ValueError(
                f"VLM per-case attempt cap exceeded: {state['specimen_key']}"
            )
    if (
        manifest["reused_pilot_rows"] != 6
        or manifest["new_unique_primary_jobs"] > 211
        or manifest["new_generation_attempts"] > 422
        or manifest["new_output_tokens"] > 211000
    ):
        raise ValueError("VLM resource or exact-reuse contract failed")
    return {
        "status": "PASS",
        "states": dict(sorted(state_counts.items())),
        "reused_pilot_rows": manifest["reused_pilot_rows"],
        "new_unique_primary_jobs": manifest["new_unique_primary_jobs"],
        "new_generation_attempts": manifest["new_generation_attempts"],
        "new_output_tokens": manifest["new_output_tokens"],
        "new_qwen_top_level_forward_calls": manifest[
            "new_qwen_top_level_forward_calls"
        ],
        "feature_sha256": sha256_file(features_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "config_lock_sha256": sha256_file(config_lock_path),
    }


def _actor_checks(context: TaskContext) -> dict[str, Any]:
    w3 = context.path("w3")
    aggregate_path = w3 / "actor_manifests.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if (
        aggregate.get("status") != "THREE_C_ACTORS_COMPLETE"
        or aggregate.get("method_count") != 3
        or aggregate.get("candidate_count") != 15
        or aggregate.get("candidate_episode_count") != 750
        or aggregate.get("selected_episode_count") != 150
        or aggregate.get("logical_updates") != 3750
        or aggregate.get("actual_update_upper_bound", 4501) > 4500
    ):
        raise ValueError("Q3/Q4 failed: C actor aggregate is incomplete or over budget")
    records = []
    signatures = set()
    for manifest in aggregate["actor_manifests"]:
        method = manifest["method"]
        if (
            method not in METHOD_SEEDS
            or manifest.get("status") != "COMPLETE"
            or manifest.get("training_seed") != METHOD_SEEDS[method]
            or manifest.get("logical_updates") != 1250
            or manifest.get("candidate_count") != 5
            or manifest.get("candidate_episode_count") != 250
            or manifest.get("charged_upper_bound", 1501) > 1500
            or len(manifest.get("first_batch_specimen_keys", [])) != 16
            or manifest.get("release_smoke", {}).get("status") != "PASS"
        ):
            raise ValueError(f"actor contract failed: {method}")
        signatures.add(manifest["input_signature"])
        for candidate in manifest["candidates"]:
            method_root = w3 / "models" / method.lower()
            for field, hash_field in (
                ("checkpoint", "checkpoint_sha256"),
                ("episodes", "episodes_sha256"),
                ("snapshot", "snapshot_sha256"),
            ):
                path = method_root / candidate[field]
                if not path.is_file() or sha256_file(path) != candidate[hash_field]:
                    raise ValueError(f"candidate artifact changed: {path}")
        selected = context.root / manifest["selected_checkpoint"]
        if sha256_file(selected) != manifest["selected_checkpoint_sha256"]:
            raise ValueError(f"selected actor changed: {method}")
        for required in ("initial_state.pt", "latest_training_state.pt"):
            if not (w3 / "models" / method.lower() / required).is_file():
                raise FileNotFoundError(f"missing actor state: {method}/{required}")
        records.append(
            {
                "method": method,
                "seed": manifest["training_seed"],
                "selected_update": manifest["selected_update"],
                "logical_updates": manifest["logical_updates"],
                "actual_optimizer_updates": manifest["actual_optimizer_updates"],
                "lost_updates_upper_bound": manifest["lost_updates_upper_bound"],
                "charged_upper_bound": manifest["charged_upper_bound"],
                "selected_checkpoint": manifest["selected_checkpoint"],
                "selected_checkpoint_sha256": manifest["selected_checkpoint_sha256"],
            }
        )
    if len(signatures) != 1:
        raise ValueError("the three C actors do not share one input binding")
    return {
        "status": "PASS",
        "candidate_weights": 15,
        "candidate_episode_rows": 750,
        "selected_models": 3,
        "selected_episode_rows": 150,
        "logical_updates": aggregate["logical_updates"],
        "actual_update_upper_bound": aggregate["actual_update_upper_bound"],
        "input_signature": next(iter(signatures)),
        "actors": records,
    }


def _matrix_checks(context: TaskContext) -> dict[str, Any]:
    w3 = context.path("w3")
    assembly_path = w3 / "assembly_manifest.json"
    assembly = json.loads(assembly_path.read_text(encoding="utf-8"))
    primary_path = w3 / "policy_validation_episodes.csv.gz"
    provenance_path = w3 / "row_provenance.csv"
    historical_path = w3 / "historical_A_comparison/policy_validation_episodes.csv.gz"
    primary = _read_gzip_csv(primary_path)
    provenance = _read_csv(provenance_path)
    historical = _read_gzip_csv(historical_path)
    if (
        assembly.get("status") != "PRIMARY_MATRIX_COMPLETE"
        or len(primary) != 650
        or len(provenance) != 650
        or len(historical) != 150
        or Counter(row["method"] for row in primary) != Counter(EXPECTED_METHOD_COUNTS)
        or Counter(row["provenance"] for row in provenance)
        != Counter({"REUSED_FROZEN_CONTROL": 500, "NEW_C_RETRAIN": 150})
        or assembly.get("binding_audit", {}).get("status")
        != "FROZEN_W2_FEATURE_COST_AND_C_PRIOR_BINDINGS_MATCH"
    ):
        raise ValueError("Q5 failed: nine-method matrix or provenance is invalid")
    return {
        "status": "PASS",
        "primary_rows": 650,
        "reused_control_rows": 500,
        "new_C_rows": 150,
        "historical_A_rows": 150,
        "method_counts": dict(Counter(row["method"] for row in primary)),
        "primary_sha256": sha256_file(primary_path),
        "provenance_sha256": sha256_file(provenance_path),
        "historical_A_sha256": sha256_file(historical_path),
    }


def _evidence_paper_checks(context: TaskContext) -> dict[str, Any]:
    evidence_root = context.path("evidence")
    evidence_path = evidence_root / "analysis_manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "C_EVIDENCE_COMPLETE":
        raise ValueError("Q6 failed: current C evidence is incomplete")
    _require_hashes(evidence_root, evidence["outputs"], label="evidence")
    paper_root = context.path("paper")
    paper_path = paper_root / "paper_manifest.json"
    paper = json.loads(paper_path.read_text(encoding="utf-8"))
    if (
        paper.get("status") != "C_MANUSCRIPT_COMPLETE"
        or paper.get("source_sections") != 6
        or paper.get("visual_qa", {}).get("status") != "PASS"
    ):
        raise ValueError("Q7 failed: current C manuscript or visual QA is incomplete")
    _require_hashes(paper_root, paper["outputs"], label="paper")
    required = (
        "manuscript.md",
        "manuscript.html",
        "main.tex",
        "supplementary.tex",
        "build/main.pdf",
        "build/supplementary.pdf",
    )
    if any(not (paper_root / relative).is_file() for relative in required):
        raise FileNotFoundError("required current C manuscript format is missing")
    return {
        "status": "PASS",
        "best_nonadaptive": evidence["best_nonadaptive"],
        "event_grid_points": evidence["event_grid_points"],
        "figure_count": evidence["figure_count"],
        "case_count": evidence["case_count"],
        "evidence_manifest_sha256": sha256_file(evidence_path),
        "paper_manifest_sha256": sha256_file(paper_path),
        "main_pdf_sha256": sha256_file(paper_root / "build/main.pdf"),
        "supplementary_pdf_sha256": sha256_file(paper_root / "build/supplementary.pdf"),
    }


def _resource_checks(context: TaskContext) -> dict[str, Any]:
    path = context.path("output") / "resource_usage.json"
    resource = json.loads(path.read_text(encoding="utf-8"))
    historical = int(resource["historical_usage_declared"])
    actor_upper = int(resource["new_actor_update_upper_bound"])
    if (
        resource["new_actor_updates"] != 3750
        or actor_upper > 4500
        or historical + actor_upper > 44514
        or resource["new_vlm_primary_jobs"] > 211
        or resource["new_vlm_attempts"] > 422
        or resource["new_vlm_output_tokens"] > 211000
        or float(resource["vlm_gpu_seconds"]) > 5400
        or float(resource["actor_gpu_seconds"]) > 10800
        or float(resource["vlm_gpu_seconds"]) + float(resource["actor_gpu_seconds"])
        > 16200
    ):
        raise ValueError("authorized resource envelope was exceeded")
    protocol = json.loads(
        (context.path("output") / "protocol_lock.json").read_text(encoding="utf-8")
    )
    if (
        protocol.get("test_accessed") is not False
        or protocol["cohort"].get("test") != 0
    ):
        raise ValueError("TEST-access prohibition failed")
    return {
        "status": "PASS",
        **resource,
        "cumulative_update_upper_bound": historical + actor_upper,
        "test_accessed": False,
    }


def _task_ledger_summary(
    rows: list[dict[str, Any]], task_id: str
) -> dict[str, Any]:
    task_rows = [row for row in rows if row.get("task_id") == task_id]
    event_ids = [row.get("event_id") for row in task_rows]
    reservation_ids = [row.get("reservation_id") for row in task_rows]
    if (
        len(task_rows) != 15
        or len(set(event_ids)) != len(task_rows)
        or None in event_ids
        or len(set(reservation_ids)) != len(task_rows)
        or None in reservation_ids
    ):
        raise ValueError("task ledger must contain 15 unique segment events")
    expected = {
        (method, start, start + 250)
        for method in METHOD_SEEDS
        for start in range(0, 1250, 250)
    }
    observed = {
        (row.get("method"), row.get("start_update"), row.get("end_update"))
        for row in task_rows
    }
    if observed != expected or any(row.get("status") != "COMPLETED" for row in task_rows):
        raise ValueError("task ledger segment coverage is incomplete")
    actual = sum(int(row.get("actual_optimizer_updates", -1)) for row in task_rows)
    lost = sum(
        int(row.get("actual_optimizer_updates_upper_bound", -1))
        for row in task_rows
    )
    if actual != 3750 or lost != 0:
        raise ValueError("task ledger optimizer accounting changed")
    return {
        "status": "PASS",
        "segment_events": len(task_rows),
        "actual_optimizer_updates": actual,
        "lost_or_replayed_upper_bound": lost,
        "unique_event_ids": len(set(event_ids)),
    }


def _ledger_checks(context: TaskContext) -> dict[str, Any]:
    ledger_path = context.path("global_ledger")
    relative = ledger_path.relative_to(context.root).as_posix()
    historical = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=context.root
    )
    current = ledger_path.read_bytes()
    if not current.startswith(historical):
        raise ValueError("global compute ledger historical prefix changed")
    rows = [json.loads(line) for line in current.decode("utf-8").splitlines()]
    summary = _task_ledger_summary(rows, context.task_id)
    resource = json.loads(
        (context.path("output") / "resource_usage.json").read_text(encoding="utf-8")
    )
    if (
        summary["actual_optimizer_updates"] != resource["new_actor_updates"]
        or summary["actual_optimizer_updates"]
        + summary["lost_or_replayed_upper_bound"]
        > resource["new_actor_update_upper_bound"]
    ):
        raise ValueError("global ledger and resource summary disagree")
    return {
        **summary,
        "historical_prefix_lines": len(historical.splitlines()),
        "current_lines": len(rows),
        "historical_prefix_sha256": hashlib.sha256(historical).hexdigest(),
    }


def _scope_checks(context: TaskContext) -> dict[str, Any]:
    immutable = tuple(
        Path(context.scope["roots"][name])
        for name in (
            "data",
            "predictors",
            "old_w3",
            "old_w3_artifacts",
            "old_evidence",
            "grounding_pilot",
            "old_paper",
        )
    )
    changed = set(
        filter(
            None,
            _git(
                context.root, "diff", "--name-only", f"{SOURCE_COMMIT}...HEAD"
            ).splitlines(),
        )
    )
    changed.update(
        filter(
            None,
            _git(
                context.root, "ls-files", "--others", "--exclude-standard"
            ).splitlines(),
        )
    )
    violations = [
        path
        for path in sorted(changed)
        if any(
            Path(path) == root or Path(path).is_relative_to(root) for root in immutable
        )
    ]
    if violations:
        raise ValueError(f"immutable source roots were changed: {violations}")
    return {
        "status": "PASS",
        "branch": context.branch,
        "source_commit": SOURCE_COMMIT,
        "immutable_root_violations": [],
        "changed_path_count": len(changed),
    }


def _release_index(context: TaskContext) -> None:
    output = context.path("output")
    paper_relative = Path(os.path.relpath(context.path("paper"), output))
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAI C retrain release</title><style>body{{font:15px/1.5 system-ui,sans-serif;max-width:850px;margin:35px auto;padding:0 20px;color:#182126}}h1{{border-bottom:3px solid #a73434;padding-bottom:8px}}li{{margin:9px 0}}code{{background:#eef1f1;padding:2px 4px}}</style></head>
<body><h1>CAI C retrain release</h1><p><code>C_P0_R1_GLOBAL_V1</code>, TRAIN/VALID only, author-review manuscript.</p>
<ul><li><a href="evidence/index.html">Evidence index</a></li><li><a href="{html.escape(paper_relative.as_posix())}/manuscript.html">Manuscript HTML</a></li><li><a href="{html.escape(paper_relative.as_posix())}/build/main.pdf">Main PDF</a></li><li><a href="{html.escape(paper_relative.as_posix())}/build/supplementary.pdf">Supplementary PDF</a></li><li><a href="release_manifest.json">Release manifest</a></li></ul></body></html>
"""
    _atomic_text(output / "index.html", document)


def _write_review_and_handoff(
    context: TaskContext, release: dict[str, Any], *, results_commit: str | None = None
) -> None:
    artifacts = context.path("artifacts")
    actors = release["validation"]["Q3_Q4_actors"]["actors"]
    actor_text = "\n".join(
        f"- `{row['method']}`: seed {row['seed']}, selected update {row['selected_update']}, "
        f"logical {row['logical_updates']}, charged upper bound {row['charged_upper_bound']}."
        for row in actors
    )
    review = f"""# Execution Review

The C release satisfies Q1-Q7 and is ready for the bounded Q8 Git delivery. Scientific effect direction was not used as a completion gate.

- Q1-Q2: 211 terminal TRAIN/VALID priors, six exact pilot reuses, no TEST access, per-case attempt cap preserved.
- Q3-Q4: 15 candidate weights, 750 candidate trajectories, three selected actors and 150 selected trajectories. All disk-reload smoke checks passed.
- Q5: 500 frozen controls plus 150 new C rows form the 650-row primary matrix; historical A remains separate.
- Q6: nine-method same-cost, paired, A/C, equal-quality, timing, domain and three-case evidence were recomputed.
- Q7: six-section C manuscript, HTML, TeX, main PDF and SI PDF were rebuilt and visually checked.
- Q8: {"results commit " + results_commit if results_commit else "pending publish-stage result commit and push"}.

Actor selection:
{actor_text}

Remaining author inputs are names, affiliations, funding, declarations, final availability wording and submission approval. These do not block the scientific author-review draft.
"""
    _atomic_text(artifacts / "EXECUTION_REVIEW.md", review)
    handoff = f"""# CODEX Handoff: CAI C Retrain Release

Status: {release["status"]}

- Definition: exact P0 plus exact readable R1, `C_P0_R1_GLOBAL_V1`.
- Cohort: 161 TRAIN and 50 VALID priors; reserved TEST was not accessed.
- Actors: three current C methods, one registered seed each, 1250 logical updates each.
- Primary evaluation: 650 rows, comprising 500 frozen controls and 150 current C rows.
- Evidence: `{context.path("evidence").relative_to(context.root)}/index.html`.
- Manuscript: `{context.path("paper").relative_to(context.root)}/manuscript.md` and two PDFs under `build/`.
- Release manifest: `{context.path("output").relative_to(context.root)}/release_manifest.json`.
- Git results commit: {results_commit or "PENDING_PUBLISH"}.

The effect report retains favourable, null and adverse directions. Do not extend this handoff to TEST, additional seeds, a new prompt or physical deployment claims without new authorization.
"""
    _atomic_text(artifacts / "CODEX_HANDOFF_CAI_C_RETRAIN_RELEASE.md", handoff)


def verify_stage(context: TaskContext) -> dict[str, Any]:
    context.transition("verify", "RUNNING")
    checks = {
        "Q1_Q2_vlm": _vlm_checks(context),
        "Q3_Q4_actors": _actor_checks(context),
        "Q5_matrix": _matrix_checks(context),
        "Q6_Q7_evidence_paper": _evidence_paper_checks(context),
        "resources": _resource_checks(context),
        "ledger": _ledger_checks(context),
        "scope": _scope_checks(context),
    }
    _release_index(context)
    release = {
        "schema_version": 1,
        "task_id": context.task_id,
        "status": "C_RETRAIN_RESULTS_COMPLETE",
        "prior_version": context.scope["prior_version"],
        "branch": BRANCH,
        "source_commit": SOURCE_COMMIT,
        "scope_sha256": context.scope_sha256,
        "validation": checks,
        "bindings": {
            "protocol_lock": {
                "path": "protocol_lock.json",
                "sha256": sha256_file(context.path("output") / "protocol_lock.json"),
            },
            "vlm_manifest": {
                "path": "vlm/vlm_manifest_fit.json",
                "sha256": sha256_file(context.path("vlm") / "vlm_manifest_fit.json"),
            },
            "actor_manifests": {
                "path": "w3/actor_manifests.json",
                "sha256": sha256_file(context.path("w3") / "actor_manifests.json"),
            },
            "assembly_manifest": {
                "path": "w3/assembly_manifest.json",
                "sha256": sha256_file(context.path("w3") / "assembly_manifest.json"),
            },
            "evidence_manifest": {
                "path": "evidence/analysis_manifest.json",
                "sha256": sha256_file(
                    context.path("evidence") / "analysis_manifest.json"
                ),
            },
            "paper_manifest": {
                "path": context.path("paper").relative_to(context.root).as_posix()
                + "/paper_manifest.json",
                "sha256": sha256_file(context.path("paper") / "paper_manifest.json"),
            },
        },
        "completion_gate": "ENGINEERING_AND_DATA_INTEGRITY_NOT_EFFECT_DIRECTION",
        "author_review_required": True,
        "verified_unix": time.time(),
    }
    manifest_path = context.path("output") / "release_manifest.json"
    atomic_json(manifest_path, release)
    _write_review_and_handoff(context, release)
    context.transition(
        "verify",
        "COMPLETE",
        release_status=release["status"],
        qa_categories=8,
        test_accessed=False,
    )
    return release


def _pending_paths(root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
        stderr=subprocess.STDOUT,
    )
    paths = []
    for line in output.splitlines():
        raw = line[3:]
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw)
    return paths


def _allowed_result_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(prefix + "/") for prefix in RESULT_PATHS
    )


def _mark_plan_publish_complete(root: Path) -> None:
    path = root / PLAN_PATH
    pending = "- [ ] **Step 6: Commit results/handoff and publish**"
    complete = "- [x] **Step 6: Commit results/handoff and publish**"
    source = path.read_text(encoding="utf-8")
    if pending in source:
        _atomic_text(path, source.replace(pending, complete, 1))
    elif complete not in source:
        raise ValueError("publish plan Step 6 marker changed")


def _commit(root: Path, message: str, paths: tuple[str, ...]) -> str:
    subprocess.run(
        [
            "git",
            "add",
            "-f",
            "--",
            *paths,
            ":(exclude,glob)**/__pycache__/**",
            ":(exclude,glob)**/*.pyc",
        ],
        cwd=root,
        check=True,
    )
    staged = _git(root, "diff", "--cached", "--name-only")
    if not staged:
        raise RuntimeError("publish stage has no files to commit")
    subprocess.run(["git", "commit", "-m", message], cwd=root, check=True)
    return _git(root, "rev-parse", "HEAD")


def _interrupted_results_commit(root: Path) -> str:
    if _git(root, "show", "-s", "--format=%s", "HEAD") != RESULTS_COMMIT_MESSAGE:
        raise RuntimeError("publish has no result changes or resumable results commit")
    return _git(root, "rev-parse", "HEAD")


def _push_and_verify(root: Path, expected: str) -> str:
    subprocess.run(["git", "push", "origin", f"HEAD:{BRANCH}"], cwd=root, check=True)
    upstream = _git(root, "rev-parse", "@{upstream}")
    remote_line = _git(root, "ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}")
    remote = remote_line.split()[0] if remote_line else ""
    local = _git(root, "rev-parse", "HEAD")
    if local != expected or upstream != expected or remote != expected:
        raise RuntimeError(
            f"Git identity mismatch: local={local}, upstream={upstream}, remote={remote}"
        )
    return remote


def _delivery_required_paths(context: TaskContext) -> list[str]:
    aggregate = json.loads(
        (context.path("w3") / "actor_manifests.json").read_text(encoding="utf-8")
    )
    paths = {
        "scripts/cai_c_retrain/cli.py",
        (context.path("vlm") / "vlm_manifest_fit.json")
        .relative_to(context.root)
        .as_posix(),
        (context.path("w3") / "policy_validation_episodes.csv.gz")
        .relative_to(context.root)
        .as_posix(),
        (context.path("output") / "release_manifest.json")
        .relative_to(context.root)
        .as_posix(),
        (context.path("artifacts") / "CODEX_HANDOFF_CAI_C_RETRAIN_RELEASE.md")
        .relative_to(context.root)
        .as_posix(),
        (context.path("paper") / "build/main.pdf").relative_to(context.root).as_posix(),
        (context.path("paper") / "build/supplementary.pdf")
        .relative_to(context.root)
        .as_posix(),
    }
    candidate_weights = set()
    selected_weights = set()
    for manifest in aggregate["actor_manifests"]:
        method_root = context.path("w3") / "models" / manifest["method"].lower()
        for candidate in manifest["candidates"]:
            path = (method_root / candidate["checkpoint"]).relative_to(context.root)
            candidate_weights.add(path.as_posix())
        selected_weights.add(manifest["selected_checkpoint"])
    if len(candidate_weights) != 15 or len(selected_weights) != 3:
        raise ValueError(
            "delivery does not identify 15 candidate and 3 selected weights"
        )
    return sorted(paths | candidate_weights | selected_weights)


def _verify_commit_paths(root: Path, commit: str, paths: list[str]) -> int:
    missing = []
    for path in paths:
        completed = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{path}"],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode:
            missing.append(path)
    if missing:
        raise ValueError(f"required delivery paths are absent from {commit}: {missing}")
    return len(paths)


def publish_stage(context: TaskContext) -> dict[str, Any]:
    release_path = context.path("output") / "release_manifest.json"
    if not release_path.is_file():
        raise RuntimeError("verify must complete before publish")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("status") not in {
        "C_RETRAIN_RESULTS_COMPLETE",
        "C_RETRAIN_RELEASE_COMPLETE",
    }:
        raise ValueError("release manifest is not publishable")
    if context.branch != BRANCH:
        raise ValueError(f"publish must run on {BRANCH}")
    pending = _pending_paths(context.root)
    unexpected = [path for path in pending if not _allowed_result_path(path)]
    if unexpected:
        raise ValueError(
            f"publish found changes outside bounded result roots: {unexpected}"
        )
    results_commit = (
        _commit(context.root, RESULTS_COMMIT_MESSAGE, RESULT_PATHS)
        if pending
        else _interrupted_results_commit(context.root)
    )
    required_paths = _delivery_required_paths(context)
    tracked_required_count = _verify_commit_paths(
        context.root, results_commit, required_paths
    )
    _push_and_verify(context.root, results_commit)

    _mark_plan_publish_complete(context.root)
    release["status"] = "C_RETRAIN_RELEASE_COMPLETE"
    release["git"] = {
        "results_commit": results_commit,
        "results_commit_remote_verified": True,
        "delivery_commit": "ENCLOSING_COMMIT",
        "branch": BRANCH,
        "remote": "origin",
        "tracked_required_count": tracked_required_count,
    }
    atomic_json(release_path, release)
    delivery = {
        "schema_version": 1,
        "task_id": context.task_id,
        "status": "RESULTS_COMMIT_REMOTE_VERIFIED_DELIVERY_IN_ENCLOSING_COMMIT",
        "branch": BRANCH,
        "remote": "origin",
        "results_commit": results_commit,
        "results_commit_local_upstream_remote_match": True,
        "tracked_required_paths": required_paths,
        "tracked_required_count": tracked_required_count,
        "delivery_commit": "ENCLOSING_COMMIT",
        "self_reference_note": "The final delivery SHA is the commit containing this file and is verified after push; it is intentionally not embedded in itself.",
    }
    atomic_json(context.path("artifacts") / "GIT_DELIVERY.json", delivery)
    _write_review_and_handoff(context, release, results_commit=results_commit)
    context.transition(
        "publish",
        "COMPLETE",
        release_status="C_RETRAIN_RELEASE_COMPLETE",
        results_commit=results_commit,
        delivery_commit="ENCLOSING_COMMIT",
    )
    delivery_commit = _commit(
        context.root,
        "docs: record CAI C retraining delivery",
        (RESULT_PATHS[0], RESULT_PATHS[1], PLAN_PATH),
    )
    _verify_commit_paths(
        context.root,
        delivery_commit,
        [
            *required_paths,
            (context.path("artifacts") / "GIT_DELIVERY.json")
            .relative_to(context.root)
            .as_posix(),
        ],
    )
    remote = _push_and_verify(context.root, delivery_commit)
    if _pending_paths(context.root):
        raise RuntimeError("publish left a dirty worktree")
    return {
        "status": "C_RETRAIN_RELEASE_COMPLETE",
        "branch": BRANCH,
        "results_commit": results_commit,
        "final_commit": delivery_commit,
        "local_sha": delivery_commit,
        "upstream_sha": delivery_commit,
        "remote_sha": remote,
        "tracked_required_count": tracked_required_count,
    }


__all__ = ["publish_stage", "verify_stage"]
