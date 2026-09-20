"""Freeze the authorized C retraining inputs before any model execution."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

from scripts.cai_c_retrain.context import TaskContext, atomic_json, sha256_file
from scripts.cai_c_retrain.vlm import load_roster, render_c_inputs


def _package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "Pillow", "numpy")
    }


def _production_prompt_sha256(context: TaskContext) -> str:
    source_package = context.root / "src/cmc_bbdm"
    command = f"""
import hashlib
import sys
import types
package = types.ModuleType("cmc_bbdm")
package.__path__ = [{str(source_package)!r}]
sys.modules["cmc_bbdm"] = package
from cmc_bbdm.learned_cscan.perception import SURFACE_PERCEPT_PROMPT
print(hashlib.sha256(SURFACE_PERCEPT_PROMPT.encode("utf-8")).hexdigest())
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(context.root / "src")
    return subprocess.check_output(
        [sys.executable, "-c", command],
        cwd=context.root,
        env=environment,
        text=True,
    ).strip()


def _gpu_inventory() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return [{"status": "UNAVAILABLE", "detail": str(error)}]
    rows = []
    for line in output.splitlines():
        index, uuid, name, free, total = (part.strip() for part in line.split(",", 4))
        rows.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_free_mib": free,
                "memory_total_mib": total,
            }
        )
    return rows


def _source_hashes(context: TaskContext) -> dict[str, str]:
    paths = {
        "scope": context.config_path,
        "candidate_queue": context.path("data") / "candidate_queue.csv",
        "feature_bank_manifest": context.path("data") / "feature_bank_manifest.json",
        "predictor_mean_sc": context.root
        / context.scope["predictors"]["common_checkpoint"],
        "pilot_lock": context.path("grounding_pilot") / "experiment_lock.json",
        "pilot_p0": context.root / context.scope["vlm"]["prompt_source"],
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required immutable inputs are missing: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def _validate_six_pilot_inputs(
    context: TaskContext,
    roster: list[dict[str, str]],
    pilot_lock: dict[str, object],
) -> list[dict[str, str]]:
    jobs = [job for job in pilot_lock["jobs"] if job["variant"] == "C_P0_R1"]
    if len(jobs) != 6:
        raise ValueError("grounding pilot must contain exactly six C jobs")
    queue = {row["specimen_key"]: row for row in roster}
    feature_manifest = json.loads(
        (context.path("data") / "feature_bank_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_root = Path(feature_manifest["encoder_execution_root"])
    reusable = []
    for job in jobs:
        key = job["specimen_key"]
        if key not in queue:
            raise ValueError(f"pilot C case is outside the authorized roster: {key}")
        row = queue[key]
        source_path = source_root / row["impacted_surface_path"]
        if sha256_file(source_path) != row["surface_sha256"]:
            raise ValueError(f"pilot source hash changed: {key}")
        with Image.open(source_path) as source:
            rendered = render_c_inputs(source)
        signature_data = job["signature_data"]
        if (
            rendered.clean_sha256 != signature_data["clean_sha256"]
            or rendered.numbered_sha256 != signature_data["numbered_sha256"]
            or rendered.render_config != signature_data["render_config"]
        ):
            raise ValueError(f"pilot C rendering changed: {key}")
        state_path = (
            context.path("grounding_pilot")
            / "runs"
            / key.replace(":", "__")
            / "C_P0_R1/state.json"
        )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != job["signature"] or not state["status"].startswith(
            "VALID"
        ):
            raise ValueError(f"pilot C state is not exactly reusable: {key}")
        reusable.append(
            {
                "specimen_key": key,
                "signature": job["signature"],
                "status": state["status"],
                "attempts": str(len(state["attempts"])),
            }
        )
    return reusable


def _write_author_documents(context: TaskContext) -> None:
    artifacts = context.path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    bindings = """# Source And Requirement Bindings

- Authority: `docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json`
- Frozen source: `331f52952b93bd9442ad2e44243cc71c7d966b4d`
- Prior C: exact P0 prompt plus exact pilot R1 renderer.
- Cohort: 161 TRAIN plus 50 VALID; TEST image, labels and scoring are forbidden.
- Predictors: frozen W2 MEAN_SC P_all and original three-fold OOF routing.
- Actors: only the three registered C-dependent methods and registered seeds.
- Evidence and manuscript: rebuilt in isolated C release roots.
"""
    decision = """# Author C Decision

The author-selected C definition is treated as the release configuration:
original P0 surface-perception semantics with the readable R1 numbered grid.
This choice authorizes controlled retraining and reporting; it is not itself
evidence that C improves predictive performance. All favorable and unfavorable
results will be retained.
"""
    (artifacts / "SOURCE_AND_REQUIREMENT_BINDINGS.md").write_text(
        bindings, encoding="utf-8"
    )
    (artifacts / "AUTHOR_C_DECISION.md").write_text(decision, encoding="utf-8")


def prepare_stage(context: TaskContext) -> dict[str, object]:
    output = context.path("output")
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_hashes(context)
    if (
        source_hashes["predictor_mean_sc"]
        != context.scope["predictors"]["common_checkpoint_sha256"]
    ):
        raise ValueError("frozen W2 predictor hash changed")

    production_prompt_sha256 = _production_prompt_sha256(context)
    if source_hashes["pilot_p0"] != production_prompt_sha256:
        raise ValueError("pilot P0 no longer equals the production surface prompt")
    roster = load_roster(context.path("data") / "candidate_queue.csv")
    pilot_lock = json.loads(
        (context.path("grounding_pilot") / "experiment_lock.json").read_text(
            encoding="utf-8"
        )
    )
    reusable = _validate_six_pilot_inputs(context, roster, pilot_lock)
    python = str(Path(sys.executable).resolve())
    runtime_lock = {
        "task_id": context.task_id,
        "created_unix": time.time(),
        "branch": context.branch,
        "head_at_prepare": context.head,
        "source_commit": context.scope["source_commit"],
        "vlm_python": python,
        "actor_python": python,
        "report_python": python,
        "python_version": platform.python_version(),
        "packages": _package_versions(),
        "source_root": str(context.root / "src"),
        "max_cpu_threads": 4,
        "max_visible_gpus": 1,
        "gpu_inventory": _gpu_inventory(),
    }
    protocol_lock = {
        "task_id": context.task_id,
        "scope_sha256": context.scope_sha256,
        "source_hashes": source_hashes,
        "prompt_sha256": source_hashes["pilot_p0"],
        "repair_sha256": pilot_lock["config"]["repair_sha256"],
        "model_config": pilot_lock["config"],
        "cohort": {
            "rows": len(roster),
            "train": sum(row["split"] == "TRAIN" for row in roster),
            "valid": sum(row["split"] == "VALID" for row in roster),
            "test": 0,
            "specimen_keys_sha256": hashlib_sha256_text(
                "\n".join(row["specimen_key"] for row in roster) + "\n"
            ),
        },
        "exact_pilot_reuse": reusable,
        "expected_new_primary_jobs": 205,
        "maximum_attempts": 422,
        "test_accessed": False,
    }
    authorization = {
        "task_id": context.task_id,
        "scope_sha256": context.scope_sha256,
        "scope": context.scope,
        "captured_unix": time.time(),
    }
    resource_usage = {
        "task_id": context.task_id,
        "status": "PREPARED",
        "historical_usage_declared": 39014,
        "global_authorized_cap": 44514,
        "new_actor_updates": 0,
        "new_actor_update_upper_bound": 0,
        "new_vlm_primary_jobs": 0,
        "new_vlm_attempts": 0,
        "new_vlm_output_tokens": 0,
        "vlm_gpu_seconds": 0.0,
        "actor_gpu_seconds": 0.0,
    }
    atomic_json(output / "runtime_lock.json", runtime_lock)
    atomic_json(output / "protocol_lock.json", protocol_lock)
    atomic_json(output / "authorization_snapshot.json", authorization)
    atomic_json(output / "resource_usage.json", resource_usage)
    _write_author_documents(context)
    context.transition(
        "prepare",
        "COMPLETE",
        roster_rows=211,
        exact_pilot_reuse=6,
        test_accessed=False,
        prompt_sha256=source_hashes["pilot_p0"],
    )
    return {
        "status": "COMPLETE",
        "roster_rows": 211,
        "exact_pilot_reuse": 6,
        "expected_new_primary_jobs": 205,
        "model_calls": 0,
        "test_accessed": False,
    }


def hashlib_sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
