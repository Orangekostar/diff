"""Extend the frozen Qwen surface-percept cache to the v3 cohort."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from cmc_bbdm.learned_cscan.perception import (
    FORMAT_REPAIR_CONTEXT_PREFIX,
    FORMAT_REPAIR_CONTEXT_SUFFIX,
    FORMAT_REPAIR_PROMPT,
    SURFACE_PERCEPT_PROMPT,
    SurfacePerceptCache,
    SurfacePerceptRequest,
)
from cmc_bbdm.vlm_cscan.runtime import render_surface_inputs
from cmc_bbdm.vlm_cscan.vlm import QwenVLBackend

from .files import read_csv, sha256_file, write_csv, write_json

_MODEL_PATH = Path(
    "/mnt/shared/ww/models/huggingface/"
    "Qwen--Qwen2.5-VL-7B-Instruct/"
    "cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
_MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
_RENDER_VERSION = "b341e99fb74d9f815be810b2e8b1d91e59f236540072577c244313c584281f40"


def _prompt_sha256() -> str:
    payload = (
        f"{SURFACE_PERCEPT_PROMPT}\n---FORMAT_REPAIR---\n"
        f"{FORMAT_REPAIR_PROMPT}\n{FORMAT_REPAIR_CONTEXT_PREFIX}"
        f"{{RAW_RESPONSE}}{FORMAT_REPAIR_CONTEXT_SUFFIX}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _features(percept) -> tuple[np.ndarray, np.ndarray]:
    indicator = np.zeros(64, dtype=np.float32)
    confidence = np.zeros(64, dtype=np.float32)
    levels = {"unknown": 0.0, "low": 1.0 / 3.0, "medium": 2.0 / 3.0, "high": 1.0}
    for region in percept.regions:
        for cell in region.cells:
            indicator[cell] = 1.0
            confidence[cell] = max(confidence[cell], levels[region.confidence])
    return indicator, confidence


def _prior_failed_specimens(path: Path, *, scope: str) -> dict[str, int]:
    """Recover terminal schema failures so a resume never resamples them."""

    if not path.is_file():
        return {}
    prefix = f"vlm_perception_{scope}_"
    failures: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                job = row.get("job", "")
                calls = int(row.get("actual_vlm_calls", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if (
                row.get("stage") == "W3_INPUT"
                and row.get("status") == "FAILED"
                and isinstance(job, str)
                and job.startswith(prefix)
                and calls >= 2
            ):
                key = job[len(prefix) :]
                failures[key] = max(failures.get(key, 0), calls)
    return failures


def _cache_call_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[str(row["cache_key"])] = int(row["call_count"])
    return output


def run_vlm_perception(
    *,
    project_root: str | Path,
    source_root: str | Path,
    scope: str,
) -> dict[str, object]:
    root = Path(project_root).resolve(strict=True)
    external = Path(source_root).resolve(strict=True)
    if scope not in {"fit", "test"}:
        raise ValueError("VLM scope must be fit or test")
    output = root / "results/cai_agent_v3/new_protocol"
    gate = json.loads((output / "predictor_gate.json").read_text(encoding="utf-8"))
    if gate.get("status") != "PREDICTOR_READY":
        payload = {
            "status": "NOT_EXECUTED_PREDICTOR_NOT_READY",
            "scope": scope,
            "actual_calls": 0,
        }
        write_json(output / f"vlm_manifest_{scope}.json", payload)
        return payload
    if scope == "test":
        policy_path = output / "policy_pilot_gate.json"
        policy = (
            json.loads(policy_path.read_text(encoding="utf-8"))
            if policy_path.is_file()
            else {}
        )
        if policy.get("status") != "POLICY_PILOT_SUPPORTED":
            payload = {
                "status": "NOT_EXECUTED_POLICY_PILOT_NOT_SUPPORTED",
                "scope": scope,
                "actual_calls": 0,
            }
            write_json(output / f"vlm_manifest_{scope}.json", payload)
            return payload
        expansion_path = output / "policy_expansion.json"
        expansion = (
            json.loads(expansion_path.read_text(encoding="utf-8"))
            if expansion_path.is_file()
            else {}
        )
        if expansion.get("status") != "POLICY_SEEDS_1_TO_3_LOCKED":
            payload = {
                "status": "NOT_EXECUTED_POLICY_EXPANSION_NOT_LOCKED",
                "scope": scope,
                "actual_calls": 0,
            }
            write_json(output / f"vlm_manifest_{scope}.json", payload)
            return payload
    candidates = read_csv(output / "candidate_queue.csv")
    selected = [
        row
        for row in candidates
        if (scope == "fit" and row["split"] in {"TRAIN", "VALID"})
        or (scope == "test" and row["split"] == "TEST")
    ]
    cache_path = output / "vlm_surface_percepts.jsonl"
    old_cache = root / "results/learned_cscan_same_perception/surface_percepts.jsonl"
    if not cache_path.exists():
        shutil.copyfile(old_cache, cache_path)
    cache = SurfacePerceptCache(cache_path)
    backend = QwenVLBackend(
        _MODEL_PATH,
        device="cuda:0",
        dtype="bfloat16",
        max_new_tokens=500,
    )
    actual_calls = 0
    cache_hits = 0
    repairs = 0
    ledger = root / "results/cai_agent_v3/compute_ledger.jsonl"
    prior_failures = _prior_failed_specimens(ledger, scope=scope)
    unavailable_keys: list[str] = []
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    for position, row in enumerate(selected, start=1):
        key = row["specimen_key"]
        surface_path = external / row["impacted_surface_path"]
        if sha256_file(surface_path) != row["surface_sha256"]:
            raise ValueError(f"VLM surface source hash changed: {key}")
        with Image.open(surface_path) as image:
            render = render_surface_inputs(image, max_edge=1024)
        request = SurfacePerceptRequest(
            model_revision=_MODEL_REVISION,
            clean_image_sha256=render.clean_sha256,
            gridded_image_sha256=render.gridded_sha256,
            render_version=_RENDER_VERSION,
            prompt_sha256=_prompt_sha256(),
            schema_version=2,
        )
        call_counter = 0
        input_images = (render.clean, render.gridded)

        def infer(prompt: str, images=input_images):
            nonlocal call_counter
            call_counter += 1
            return backend.infer(images, prompt)

        result = None
        if key not in prior_failures:
            try:
                result = cache.resolve(request, infer)
            except Exception as error:
                is_terminal_schema_failure = (
                    call_counter >= 2
                    and isinstance(error, ValueError)
                    and "remained invalid after one repair" in str(error)
                )
                if not is_terminal_schema_failure:
                    raise
                prior_failures[key] = call_counter
                actual_calls += call_counter
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "job": f"vlm_perception_{scope}_{key}",
                                "stage": "W3_INPUT",
                                "device": "cuda:0",
                                "actual_optimizer_updates": 0,
                                "actual_vlm_calls": call_counter,
                                "status": "FAILED",
                                "reason": "SCHEMA_INVALID_AFTER_ONE_REPAIR",
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        if result is None:
            unavailable_keys.append(key)
            zeros = np.zeros(64, dtype=np.float32)
            rows.append(
                {
                    "specimen_key": key,
                    "dataset_id": row["dataset_id"],
                    "split": row["split"],
                    "cache_key": request.cache_key,
                    "cache_hit": False,
                    "call_count": prior_failures[key],
                    "repaired": True,
                    "clean_image_sha256": render.clean_sha256,
                    "gridded_image_sha256": render.gridded_sha256,
                    "vlm_available": False,
                    "no_reliable_cue": False,
                    "region_count": 0,
                    "region_indicator": ";".join(str(float(value)) for value in zeros),
                    "confidence": ";".join(str(float(value)) for value in zeros),
                    "failure_reason": "SCHEMA_INVALID_AFTER_ONE_REPAIR",
                }
            )
            print(
                f"v3 surface percept {position}/{len(selected)} {key} "
                "unavailable=SCHEMA_INVALID_AFTER_ONE_REPAIR",
                flush=True,
            )
            continue
        if call_counter > 0 and result.actual_call_count == 0:
            raise ValueError("VLM cache accounting is inconsistent")
        if call_counter == 0 and not result.cache_hit:
            raise ValueError("VLM cache accounting is inconsistent")
        if call_counter > 0 and result.actual_call_count != call_counter:
            raise ValueError("VLM backend call accounting is inconsistent")
        if result.actual_call_count > 2:
            raise ValueError("VLM repair call limit was exceeded")
        if result.actual_call_count == 2 and not result.repaired:
            raise ValueError("VLM repair identity is inconsistent")
        if result.actual_call_count < 0:
            raise ValueError("VLM call count is invalid")
        if result.cache_hit and result.actual_call_count != 0:
            raise ValueError("VLM cache hit made an actual call")
        if not result.cache_hit and result.actual_call_count == 0:
            raise ValueError("VLM cache miss made no actual call")
        if result.original_call_count not in {1, 2}:
            raise ValueError("VLM original call count is invalid")
        if result.repaired != (result.original_call_count == 2):
            raise ValueError("VLM repair flag is inconsistent")
        if result.cache_key != request.cache_key:
            raise ValueError("VLM cache identity changed")
        if result.percept.no_reliable_cue and result.percept.regions:
            raise ValueError("VLM no-cue response contains regions")
        if result.actual_call_count > 0:
            actual_calls += result.actual_call_count
        cache_hits += int(result.cache_hit)
        repairs += int(result.repaired and not result.cache_hit)
        indicator, confidence = _features(result.percept)
        rows.append(
            {
                "specimen_key": key,
                "dataset_id": row["dataset_id"],
                "split": row["split"],
                "cache_key": result.cache_key,
                "cache_hit": result.cache_hit,
                "call_count": result.original_call_count,
                "repaired": result.repaired,
                "clean_image_sha256": render.clean_sha256,
                "gridded_image_sha256": render.gridded_sha256,
                "vlm_available": True,
                "no_reliable_cue": result.percept.no_reliable_cue,
                "region_count": len(result.percept.regions),
                "region_indicator": ";".join(str(float(value)) for value in indicator),
                "confidence": ";".join(str(float(value)) for value in confidence),
                "failure_reason": "",
            }
        )
        print(
            f"v3 surface percept {position}/{len(selected)} {key} "
            f"calls={actual_calls} hits={cache_hits}",
            flush=True,
        )
    feature_path = output / f"vlm_actor_features_{scope}.csv"
    write_csv(feature_path, rows)
    elapsed = time.perf_counter() - started
    old_calls = _cache_call_counts(old_cache)
    current_calls = _cache_call_counts(cache_path)
    cumulative_valid_calls = sum(
        calls for key, calls in current_calls.items() if key not in old_calls
    )
    cumulative_failed_calls = sum(prior_failures.values())
    payload = {
        "status": "REAL_FROZEN_VLM_PERCEPTION_COMPLETE",
        "scope": scope,
        "model_repository": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": _MODEL_REVISION,
        "prompt_sha256": _prompt_sha256(),
        "render_version": _RENDER_VERSION,
        "schema_version_surface_percept": 2,
        "specimen_count": len(rows),
        "vlm_available_count": len(rows) - len(unavailable_keys),
        "vlm_unavailable_count": len(unavailable_keys),
        "vlm_unavailable_specimen_keys": unavailable_keys,
        "actual_calls_this_command": actual_calls,
        "cumulative_new_valid_calls": cumulative_valid_calls,
        "cumulative_failed_calls": cumulative_failed_calls,
        "cumulative_v3_calls": cumulative_valid_calls + cumulative_failed_calls,
        "cache_hits_this_command": cache_hits,
        "repair_calls_this_command": repairs,
        "elapsed_seconds": elapsed,
        "confidence_cell_counts": dict(
            sorted(
                Counter(
                    value
                    for row in rows
                    for value in row["confidence"].split(";")
                    if float(value) > 0.0
                ).items()
            )
        ),
        "no_reliable_cue_count": sum(
            row["vlm_available"] and bool(row["no_reliable_cue"]) for row in rows
        ),
        "cache_path": cache_path.relative_to(root).as_posix(),
        "cache_sha256": sha256_file(cache_path),
        "actor_features_path": feature_path.relative_to(root).as_posix(),
        "actor_features_sha256": sha256_file(feature_path),
        "test_outcomes_used": False,
    }
    write_json(output / f"vlm_manifest_{scope}.json", payload)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "job": f"vlm_perception_{scope}",
                    "stage": "W3_INPUT",
                    "device": "cuda:0",
                    "actual_optimizer_updates": 0,
                    "actual_vlm_calls": actual_calls,
                    "cache_hits": cache_hits,
                    "repair_calls": repairs,
                    "vlm_unavailable_count": len(unavailable_keys),
                    "cumulative_v3_calls": cumulative_valid_calls
                    + cumulative_failed_calls,
                    "elapsed_seconds": elapsed,
                    "status": "COMPLETED",
                },
                sort_keys=True,
            )
            + "\n"
        )
    return payload


__all__ = ["run_vlm_perception"]
