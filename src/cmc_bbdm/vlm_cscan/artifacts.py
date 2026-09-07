"""Preparation, execution, aggregation, and integrity artifacts for the pilot."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import platform
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from PIL import Image, ImageDraw

from .benchmark import TrajectoryResult, run_method_trajectory
from .contracts import (
    BenchmarkTask,
    CScanReference,
    EvaluationMode,
    MethodId,
    ReferenceType,
    ReviewState,
)
from .metrics import paired_domain_bootstrap
from .reader import read_sparse_evidence
from .references import (
    component_bbox_polygons,
    derive_proxy_reference,
    reference_from_payload,
)
from .reporting import build_task_report
from .runtime import (
    BenchmarkConfig,
    InputSpecimen,
    load_benchmark_config,
    load_input_records,
    load_runtime_context,
    open_specimen_runtime,
    render_surface_inputs,
    select_pilot_records,
)
from .vlm import (
    INITIAL_SURFACE_PROMPT,
    QwenVLBackend,
    SurfacePlanCache,
    SurfacePlanInference,
    SurfacePlanRequest,
)

_FIXED_METHODS = (MethodId.B0, MethodId.B1, MethodId.B2, MethodId.B3)
_RULE_METHODS = (MethodId.B4, MethodId.B5, MethodId.B7)
_TASKS = (BenchmarkTask.LOCATE, BenchmarkTask.CHARACTERIZE)
_QWEN_RESIZE_FACTOR = 28
_QWEN_PATCH_SIZE = 14
_QWEN_MERGE_SIZE = 2
_QWEN_MIN_VISUAL_TOKENS = 256
_QWEN_MAX_VISUAL_TOKENS = 1280
_QWEN_PIXELS_PER_VISUAL_TOKEN = _QWEN_RESIZE_FACTOR**2
_REQUIRED_RESULT_FILES = (
    "input_manifest.csv",
    "reference_manifest.csv",
    "split_manifest.csv",
    "model_and_prompt_manifest.json",
    "surface_plans.jsonl",
    "episodes.parquet",
    "reports.parquet",
    "aggregate_metrics.csv",
    "comparisons.csv",
    "failure_types.csv",
    "summary.json",
)


def prepare_benchmark(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_root: str | Path,
) -> dict[str, object]:
    config = load_benchmark_config(config_path, project_root=project_root)
    roster = load_input_records(config, source_root=source_root)
    output = _result_root(config)
    output.mkdir(parents=True, exist_ok=True)
    pilot_keys = {record.specimen_key for record in roster.pilot_records}
    smoke_keys = {record.specimen_key for record in roster.smoke_records}
    render_metadata: dict[str, tuple[str, str, int, int, dict[str, object]]] = {}
    for record in roster.pilot_records:
        with Image.open(record.surface_path) as image:
            rendered = render_surface_inputs(
                image,
                max_edge=int(config.values["surface"]["max_edge"]),
            )
        surface_geometry = _qwen_processor_geometry(
            rendered.clean.height,
            rendered.clean.width,
        )
        render_metadata[record.specimen_key] = (
            rendered.clean_sha256,
            rendered.gridded_sha256,
            rendered.clean.height,
            rendered.clean.width,
            surface_geometry,
        )
    input_rows = []
    for record in roster.records:
        clean_sha, gridded_sha, render_height, render_width, surface_geometry = (
            render_metadata.get(
                record.specimen_key,
                ("", "", "", "", {}),
            )
        )
        evidence_geometry = _qwen_processor_geometry(*record.native_shape)
        input_rows.append(
            {
                "schema_version": 1,
                "dataset_id": record.dataset_id,
                "specimen_id": record.specimen_id,
                "specimen_key": record.specimen_key,
                "surface_path": record.surface_path.relative_to(
                    Path(source_root).resolve()
                ).as_posix(),
                "surface_sha256": record.surface_sha256,
                "registered_cscan_path": record.cscan_path.relative_to(
                    Path(source_root).resolve()
                ).as_posix(),
                "registered_cscan_sha256": record.cscan_sha256,
                "native_height": record.native_shape[0],
                "native_width": record.native_shape[1],
                "registration_transform_sha256": record.transform_sha256,
                "surface_frame_transform": "ROT90_CLOCKWISE",
                "pilot_selected": record.specimen_key in pilot_keys,
                "smoke_selected": record.specimen_key in smoke_keys,
                "clean_render_sha256": clean_sha,
                "gridded_render_sha256": gridded_sha,
                "surface_render_height_px": render_height,
                "surface_render_width_px": render_width,
                "model_surface_height_px": surface_geometry.get("height_px", ""),
                "model_surface_width_px": surface_geometry.get("width_px", ""),
                "model_surface_image_grid_thw": json.dumps(
                    surface_geometry.get("image_grid_thw", []),
                    separators=(",", ":"),
                ),
                "model_surface_visual_tokens": surface_geometry.get(
                    "visual_token_count", ""
                ),
                "model_evidence_height_px": evidence_geometry["height_px"],
                "model_evidence_width_px": evidence_geometry["width_px"],
                "model_evidence_image_grid_thw": json.dumps(
                    evidence_geometry["image_grid_thw"],
                    separators=(",", ":"),
                ),
                "model_evidence_visual_tokens": evidence_geometry[
                    "visual_token_count"
                ],
            }
        )
    _write_csv(output / "input_manifest.csv", input_rows)
    split_rows = []
    for outer_domain in config.domain_order:
        for record in roster.records:
            split_rows.append(
                {
                    "schema_version": 1,
                    "outer_domain": outer_domain,
                    "dataset_id": record.dataset_id,
                    "specimen_id": record.specimen_id,
                    "specimen_key": record.specimen_key,
                    "role": (
                        "TARGET" if record.dataset_id == outer_domain else "SOURCE"
                    ),
                    "pilot_selected": record.specimen_key in pilot_keys,
                    "evidence_scope": "INTERNAL_REUSED_COHORT_EVALUATION",
                }
            )
    _write_csv(output / "split_manifest.csv", split_rows)
    _write_reference_manifest(config, roster.records)
    previous_execution = _existing_model_execution(config)
    _write_model_manifest(
        config,
        execution=previous_execution
        or {
            "state": "MODEL_AVAILABLE_INFERENCE_PENDING",
            "cohort": None,
            "unique_initial_plans": _initial_cache_count(
                output / "surface_plans.jsonl"
            ),
        },
    )
    return {
        "input_count": len(roster.records),
        "pilot_count": len(roster.pilot_records),
        "smoke_count": len(roster.smoke_records),
        "reviewed_reference_count": _reviewed_reference_count(config, roster.records),
        "output": output.as_posix(),
    }


def export_annotation_queue(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_root: str | Path,
) -> dict[str, object]:
    config = load_benchmark_config(config_path, project_root=project_root)
    roster = load_input_records(config, source_root=source_root)
    output = _result_root(config)
    queue = output / "annotation_queue"
    surface_queue = output / "surface_annotation_queue"
    visualizations = output / "visualizations"
    queue.mkdir(parents=True, exist_ok=True)
    surface_queue.mkdir(parents=True, exist_ok=True)
    visualizations.mkdir(parents=True, exist_ok=True)
    selected_visualizations = {
        record.specimen_key
        for domain in config.domain_order
        for record in tuple(
            item for item in roster.pilot_records if item.dataset_id == domain
        )[:2]
    }
    visualization_rows = []
    for record in roster.pilot_records:
        reference, full_scan = _derive_proxy(config, record)
        with Image.open(record.surface_path) as image:
            rendered_surface = render_surface_inputs(
                image,
                max_edge=int(config.values["surface"]["max_edge"]),
            )
        surface_template = surface_queue / _queue_name(record)
        if not surface_template.exists():
            _write_json(
                surface_template,
                {
                    "schema_version": 1,
                    "specimen_key": record.specimen_key,
                    "frame": "registered_surface_rot90",
                    "source_image_sha256": record.surface_sha256,
                    "rendered_image_sha256": rendered_surface.clean_sha256,
                    "source_path_from_source_root": record.surface_path.relative_to(
                        Path(source_root).resolve()
                    ).as_posix(),
                    "review_state": "pending",
                    "reviewer_alias": None,
                    "visible_regions": [],
                    "interference_regions": [],
                    "unable_to_determine": True,
                    "allowed_visible_cues": [
                        "indentation_like",
                        "crack_like",
                        "other_suspicious",
                    ],
                    "allowed_interference_cues": [
                        "reflection",
                        "contamination",
                        "texture_variation",
                    ],
                    "blinding_contract": "SURFACE_ONLY_NO_CSCAN_NO_CAI",
                    "notes": "",
                },
            )
        payload = {
            "schema_version": 1,
            "specimen_key": record.specimen_key,
            "frame": "registered_cscan",
            "source_image_sha256": record.cscan_sha256,
            "reference_type": "ALGORITHM_DERIVED_NOT_REVIEWED",
            "review_state": "pending",
            "reviewer_alias": None,
            "regions": [
                {
                    "id": f"proposal-{index + 1}",
                    "polygon": polygon,
                    "certainty": "certain",
                }
                for index, polygon in enumerate(
                    component_bbox_polygons(reference.certain_mask)
                )
            ],
            "uncertain_regions": [
                {
                    "id": f"uncertain-{index + 1}",
                    "polygon": polygon,
                    "certainty": "uncertain",
                }
                for index, polygon in enumerate(
                    component_bbox_polygons(reference.uncertain_mask)
                )
            ],
            "proposal_provenance": {
                "status": "ALGORITHM_DERIVED_NOT_REVIEWED",
                "algorithm": "FULL_CSCAN_INNER_BORDER_RGB_DISTANCE",
                "background": "SAME_SPECIMEN_INNER_BORDER_MEDIAN",
                "parameters": {
                    key: config.values["reader"][key]
                    for key in (
                        "distance_threshold",
                        "uncertainty_band",
                        "border_exclusion_fraction",
                        "minimum_component_pixels",
                    )
                },
            },
            "review_instructions": (
                "A qualified reviewer must inspect and edit polygons, then set "
                "reference_type to EXPERT_REVIEWED or AUTHOR_PROVIDED, review_state "
                "to reviewed, and reviewer_alias to a non-empty attribution."
            ),
            "notes": "",
        }
        template = queue / _queue_name(record)
        if not template.exists():
            _write_json(template, payload)
        else:
            existing = json.loads(template.read_text(encoding="utf-8"))
            if (
                existing.get("review_state") == "pending"
                and existing.get("reference_type")
                == "ALGORITHM_DERIVED_NOT_REVIEWED"
                and existing.get("reviewer_alias") is None
            ):
                existing["proposal_provenance"] = payload["proposal_provenance"]
                _write_json(template, existing)
        if record.specimen_key in selected_visualizations:
            target = visualizations / f"{record.dataset_id}__{record.specimen_id}.png"
            _write_reference_visualization(record, reference, full_scan, target)
            visualization_rows.append(
                {
                    "specimen_key": record.specimen_key,
                    "path": target.relative_to(output).as_posix(),
                    "surface_sha256": record.surface_sha256,
                    "registered_cscan_sha256": record.cscan_sha256,
                    "reference_status": "ALGORITHM_DERIVED_NOT_REVIEWED",
                }
            )
    _write_csv(visualizations / "index.csv", visualization_rows)
    _write_reference_manifest(config, roster.records)
    return {
        "annotation_count": len(roster.pilot_records),
        "surface_annotation_count": len(roster.pilot_records),
        "visualization_count": len(visualization_rows),
        "reviewed_reference_count": _reviewed_reference_count(config, roster.records),
    }


def infer_surface_plans(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_root: str | Path,
    cohort: str,
) -> dict[str, object]:
    started = time.perf_counter()
    config = load_benchmark_config(config_path, project_root=project_root)
    roster = load_input_records(config, source_root=source_root)
    records = _cohort_records(roster, cohort, allow_all=False)
    output = _result_root(config)
    output.mkdir(parents=True, exist_ok=True)
    cache = SurfacePlanCache(output / "surface_plans.jsonl")
    model = config.values["model"]
    backend = QwenVLBackend(
        model["local_path"],
        device=model["device"],
        dtype=model["dtype"],
        max_new_tokens=int(model["max_new_tokens"]),
    )
    cache_hits = 0
    actual_calls = 0
    fallbacks = 0
    for record in records:
        with Image.open(record.surface_path) as image:
            rendered = render_surface_inputs(
                image,
                max_edge=int(config.values["surface"]["max_edge"]),
            )
        request = _surface_request(config, rendered)
        result = cache.resolve(
            request,
            lambda prompt, current=rendered: backend.infer(
                (current.clean, current.gridded), prompt
            ),
        )
        cache_hits += int(result.cache_hit)
        actual_calls += result.actual_call_count
        fallbacks += int(result.parse_status == "FALLBACK")
    elapsed = time.perf_counter() - started
    execution = {
        "state": "REAL_VLM_INITIAL_PLANS_EXECUTED",
        "cohort": cohort,
        "specimen_count": len(records),
        "cache_hits_this_command": cache_hits,
        "actual_calls_this_command": actual_calls,
        "fallback_count": fallbacks,
        "elapsed_seconds": elapsed,
        "unique_initial_plans": _initial_cache_count(output / "surface_plans.jsonl"),
    }
    _write_model_manifest(config, execution=execution)
    return execution


def run_benchmark(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_root: str | Path,
    cohort: str,
    workers: int = 4,
) -> dict[str, object]:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be between one and four")
    started = time.perf_counter()
    config = load_benchmark_config(config_path, project_root=project_root)
    roster = load_input_records(config, source_root=source_root)
    records = _cohort_records(roster, cohort, allow_all=True)
    if cohort == "all" and _reviewed_reference_count(config, records) != len(records):
        raise RuntimeError("full cohort is blocked until all selected references are reviewed")
    output = _result_root(config)
    cache_path = output / "surface_plans.jsonl"
    _require_surface_plans(config, records, cache_path)
    work = _work_root(config)
    (work / "episodes").mkdir(parents=True, exist_ok=True)
    (work / "reports").mkdir(parents=True, exist_ok=True)
    pending_cheap = [
        record.specimen_key
        for record in records
        if _record_needs_methods(work, record, (*_FIXED_METHODS, *_RULE_METHODS))
    ]
    chunks = [pending_cheap[index::workers] for index in range(workers)]
    chunks = [chunk for chunk in chunks if chunk]
    if chunks:
        arguments = [
            (
                str(Path(config_path).resolve()),
                str(Path(project_root).resolve()),
                str(Path(source_root).resolve()),
                tuple(chunk),
                str(cache_path),
                str(work),
            )
            for chunk in chunks
        ]
        if len(arguments) == 1:
            _run_cheap_chunk(arguments[0])
        else:
            with ProcessPoolExecutor(
                max_workers=len(arguments),
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                tuple(executor.map(_run_cheap_chunk, arguments))
    context = load_runtime_context(
        config,
        source_root=source_root,
        verify_pilot_hashes=False,
    )
    model = config.values["model"]
    backend = QwenVLBackend(
        model["local_path"],
        device=model["device"],
        dtype=model["dtype"],
        max_new_tokens=int(model["max_new_tokens"]),
    )
    b6_executed = 0
    b6_actual_calls = 0
    for record in records:
        for task in _TASKS:
            if _shard_exists(work, record, MethodId.B6, task):
                continue
            runtime = open_specimen_runtime(context, record)
            surface_inference = _get_surface_inference(config, runtime, cache_path)
            reference = _load_evaluation_reference(config, record, runtime)
            result = run_method_trajectory(
                runtime,
                method=MethodId.B6,
                tasks=(task,),
                surface_inference=surface_inference,
                proxy_reference=reference,
                config=config,
                cache_path=cache_path,
                replan_infer=backend.infer,
            )
            _write_result_shards(work, record, MethodId.B6, result)
            b6_executed += 1
            b6_actual_calls += result.replan_actual_calls
    _materialize_parquet(config, records)
    elapsed = time.perf_counter() - started
    metadata = {
        "schema_version": 1,
        "cohort": cohort,
        "specimen_count": len(records),
        "method_count": len(MethodId),
        "task_count": len(_TASKS),
        "trajectory_count": len(records) * len(MethodId) * len(_TASKS),
        "b6_trajectories_executed_this_command": b6_executed,
        "b6_actual_replan_calls_this_command": b6_actual_calls,
        "workers": workers,
        "elapsed_seconds": elapsed,
    }
    _write_json(output / "run_metadata.json", metadata)
    return metadata


def evaluate_benchmark(
    config_path: str | Path,
    *,
    project_root: str | Path,
    source_root: str | Path,
    cohort: str,
) -> dict[str, object]:
    config = load_benchmark_config(config_path, project_root=project_root)
    roster = load_input_records(
        config,
        source_root=source_root,
        verify_pilot_hashes=False,
    )
    records = _cohort_records(roster, cohort, allow_all=True)
    _materialize_parquet(config, records)
    output = _result_root(config)
    reports = pl.read_parquet(output / "reports.parquet").sort(
        "method", "task", "dataset_id", "specimen_id", "step"
    )
    report_rows = reports.to_dicts()
    by_method_task: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for row in report_rows:
        key = (str(row["method"]), str(row["task"]))
        by_method_task.setdefault(key, {}).setdefault(
            str(row["specimen_key"]), []
        ).append(row)
    aggregate_rows = _aggregate_rows(config, by_method_task)
    _write_csv(output / "aggregate_metrics.csv", aggregate_rows)
    comparisons = _comparison_rows(config, by_method_task)
    _write_csv(output / "comparisons.csv", comparisons)
    _write_csv(output / "failure_types.csv", _failure_type_rows(by_method_task))
    _write_curve_figures(output, aggregate_rows)
    cache_summary = _cache_summary(output / "surface_plans.jsonl")
    reviewed_count = _reviewed_reference_count(config, records)
    summary = {
        "schema_version": 1,
        "status": (
            "REVIEWED_BENCHMARK_EVALUATED"
            if reviewed_count == len(records)
            else "PILOT_EXECUTED_PROXY_ONLY"
        ),
        "evidence_scope": "INTERNAL_REUSED_COHORT_EVALUATION",
        "cohort": cohort,
        "specimen_count": len(records),
        "domain_count": len(config.domain_order),
        "methods": [method.value for method in MethodId],
        "tasks": [task.value for task in _TASKS],
        "reviewed_reference_count": reviewed_count,
        "reference_coverage": reviewed_count / len(records),
        "formal_success_metrics": None if not reviewed_count else "SEE_AGGREGATES",
        "proxy_metrics_role": "DIAGNOSTIC_ONLY_NOT_FORMAL_TASK_EVIDENCE",
        "surface_plan_benefit": "INCONCLUSIVE",
        "ultrasound_feedback_benefit": "INCONCLUSIVE",
        "vlm_replanning_benefit": "INCONCLUSIVE",
        "autonomous_completion_efficiency": "INCONCLUSIVE",
        "initial_surface_plan_records": cache_summary["initial_records"],
        "replan_choice_records": cache_summary["replan_records"],
        "vlm_deployment_call_count": cache_summary["deployment_calls"],
        "vlm_fallback_count": cache_summary["fallback_count"],
        "vlm_total_latency_seconds": cache_summary["latency_seconds"],
        "trajectory_rows": pl.read_parquet(output / "episodes.parquet").height,
        "report_rows": reports.height,
        "full_cohort_executed": cohort == "all",
        "stress_test_executed": False,
        "new_training": False,
        "proxy_final_snapshot": _proxy_final_snapshot(aggregate_rows),
    }
    if (output / "run_metadata.json").is_file():
        summary["run_metadata"] = json.loads(
            (output / "run_metadata.json").read_text(encoding="utf-8")
        )
    _write_json(output / "summary.json", summary)
    _write_model_manifest(
        config,
        execution=_existing_model_execution(config)
        or {"state": "MODEL_AVAILABLE_INFERENCE_PENDING"},
    )
    _write_checksums(output)
    return summary


def verify_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    output = _result_root(config)
    missing = [name for name in _REQUIRED_RESULT_FILES if not (output / name).is_file()]
    if missing:
        raise ValueError(f"benchmark results are incomplete: {missing}")
    expected: dict[str, str] = {}
    for line in (output / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and path.name != "CHECKSUMS.sha256"
        and ".work" not in path.relative_to(output).parts
    }
    if set(expected) != actual_files:
        raise ValueError("benchmark checksum roster differs")
    for relative, digest in expected.items():
        if _file_sha256(output / relative) != digest:
            raise ValueError(f"benchmark checksum mismatch: {relative}")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    return {
        "status": summary["status"],
        "verified_file_count": len(expected),
        "reference_coverage": summary["reference_coverage"],
    }


def _run_cheap_chunk(arguments: tuple[str, str, str, tuple[str, ...], str, str]) -> int:
    config_path, project_root, source_root, specimen_keys, cache_path, work_path = arguments
    config = load_benchmark_config(config_path, project_root=project_root)
    context = load_runtime_context(
        config,
        source_root=source_root,
        verify_pilot_hashes=False,
    )
    by_key = {record.specimen_key: record for record in context.roster.records}
    work = Path(work_path)
    executed = 0
    for specimen_key in specimen_keys:
        record = by_key[specimen_key]
        runtime = open_specimen_runtime(context, record)
        surface_inference = _get_surface_inference(config, runtime, Path(cache_path))
        reference = _load_evaluation_reference(config, record, runtime)
        for method in _FIXED_METHODS:
            if all(_shard_exists(work, record, method, task) for task in _TASKS):
                continue
            result = run_method_trajectory(
                runtime,
                method=method,
                tasks=_TASKS,
                surface_inference=surface_inference,
                proxy_reference=reference,
                config=config,
                cache_path=cache_path,
            )
            _write_result_shards(work, record, method, result)
            executed += 1
        for method in _RULE_METHODS:
            for task in _TASKS:
                if _shard_exists(work, record, method, task):
                    continue
                result = run_method_trajectory(
                    runtime,
                    method=method,
                    tasks=(task,),
                    surface_inference=surface_inference,
                    proxy_reference=reference,
                    config=config,
                    cache_path=cache_path,
                )
                _write_result_shards(work, record, method, result)
                executed += 1
    return executed


def _write_result_shards(
    work: Path,
    record: InputSpecimen,
    method: MethodId,
    result: TrajectoryResult,
) -> None:
    for task in _TASKS:
        episodes = [row for row in result.episode_rows if row["task"] == task.value]
        reports = [row for row in result.report_rows if row["task"] == task.value]
        if not episodes and not reports:
            continue
        stem = _shard_stem(record, method, task)
        _write_parquet_atomic(work / "episodes" / f"{stem}.parquet", episodes)
        _write_parquet_atomic(work / "reports" / f"{stem}.parquet", reports)


def _materialize_parquet(
    config: BenchmarkConfig, records: tuple[InputSpecimen, ...]
) -> None:
    output = _result_root(config)
    work = _work_root(config)
    episode_paths = []
    report_paths = []
    for record in records:
        for method in MethodId:
            for task in _TASKS:
                stem = _shard_stem(record, method, task)
                episode = work / "episodes" / f"{stem}.parquet"
                report = work / "reports" / f"{stem}.parquet"
                if not episode.is_file() or not report.is_file():
                    raise RuntimeError(
                        f"trajectory shard is missing: {record.specimen_key}/{method.value}/{task.value}"
                    )
                episode_paths.append(episode)
                report_paths.append(report)
    episode_table = pl.concat(
        [pl.read_parquet(path) for path in episode_paths], how="diagonal_relaxed"
    ).sort("method", "task", "dataset_id", "specimen_id", "step")
    report_table = pl.concat(
        [pl.read_parquet(path) for path in report_paths], how="diagonal_relaxed"
    ).sort("method", "task", "dataset_id", "specimen_id", "step")
    _write_table_atomic(output / "episodes.parquet", episode_table)
    _write_table_atomic(output / "reports.parquet", report_table)


def _aggregate_rows(
    config: BenchmarkConfig,
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
) -> list[dict[str, object]]:
    checkpoints = tuple(float(value) for value in config.values["acquisition"]["display_checkpoints"])
    rows = []
    for (method, task), specimens in sorted(groups.items()):
        scopes: list[tuple[str, dict[str, list[dict[str, Any]]]]] = [("ALL", specimens)]
        scopes.extend(
            (
                domain,
                {
                    key: value
                    for key, value in specimens.items()
                    if value[0]["dataset_id"] == domain
                },
            )
            for domain in config.domain_order
        )
        for scope, selected in scopes:
            for mode in EvaluationMode:
                for checkpoint in checkpoints:
                    formal = []
                    proxy = []
                    for specimen_rows in selected.values():
                        formal_value, proxy_value = _snapshot(
                            specimen_rows,
                            mode=mode,
                            checkpoint=checkpoint,
                        )
                        if formal_value is not None:
                            formal.append(float(formal_value))
                        proxy.append(float(proxy_value))
                    rows.append(
                        _metric_row(
                            scope,
                            method,
                            task,
                            mode.value,
                            "SUCCESS_RATE",
                            checkpoint,
                            "FORMAL",
                            float(np.mean(formal)) if formal else None,
                            int(sum(formal)),
                            len(formal),
                        )
                    )
                    rows.append(
                        _metric_row(
                            scope,
                            method,
                            task,
                            mode.value,
                            "SUCCESS_RATE",
                            checkpoint,
                            "PROXY_DIAGNOSTIC",
                            float(np.mean(proxy)),
                            int(sum(proxy)),
                            len(proxy),
                        )
                    )
                proxy_ausc = [
                    _specimen_ausc(specimen_rows, mode=mode, formal=False)
                    for specimen_rows in selected.values()
                ]
                formal_ausc = [
                    value
                    for specimen_rows in selected.values()
                    if (
                        value := _specimen_ausc(
                            specimen_rows,
                            mode=mode,
                            formal=True,
                        )
                    )
                    is not None
                ]
                rows.append(
                    _metric_row(
                        scope,
                        method,
                        task,
                        mode.value,
                        "AUSC",
                        None,
                        "FORMAL",
                        float(np.mean(formal_ausc)) if formal_ausc else None,
                        None,
                        len(formal_ausc),
                    )
                )
                rows.append(
                    _metric_row(
                        scope,
                        method,
                        task,
                        mode.value,
                        "AUSC",
                        None,
                        "PROXY_DIAGNOSTIC",
                        float(np.mean(proxy_ausc)),
                        None,
                        len(proxy_ausc),
                    )
                )
                if scope == "ALL":
                    for target in config.values["evaluation"]["target_success_rates"]:
                        rows.append(
                            _metric_row(
                                scope,
                                method,
                                task,
                                mode.value,
                                f"C_AT_SR_{float(target):.2f}",
                                None,
                                "PROXY_DIAGNOSTIC",
                                _cost_at_success_rate(
                                    selected,
                                    mode=mode,
                                    target=float(target),
                                ),
                                None,
                                len(selected),
                            )
                        )
                    terminal_rows = [value[-1] for value in selected.values()]
                    terminal_route = [
                        _route_cost_for_mode(specimen_rows, mode)
                        for specimen_rows in selected.values()
                    ]
                    rows.append(
                        _metric_row(
                            scope,
                            method,
                            task,
                            mode.value,
                            "FINAL_NORMALIZED_ROUTE_COST",
                            None,
                            "PROXY_DIAGNOSTIC",
                            float(np.mean(terminal_route)),
                            None,
                            len(terminal_route),
                        )
                    )
                    if mode is EvaluationMode.AUTONOMOUS_REPORT:
                        early_wrong = [
                            row["autonomous_stop_cost"] is not None
                            and not row["autonomous_proxy_diagnostic_success"]
                            for row in terminal_rows
                        ]
                        incomplete = [
                            row["autonomous_stop_cost"] is None
                            for row in terminal_rows
                        ]
                        for name, values in (
                            ("EARLY_WRONG_STOP_RATE", early_wrong),
                            ("INCOMPLETE_RATE", incomplete),
                        ):
                            rows.append(
                                _metric_row(
                                    scope,
                                    method,
                                    task,
                                    mode.value,
                                    name,
                                    None,
                                    "PROXY_DIAGNOSTIC",
                                    float(np.mean(values)),
                                    int(sum(values)),
                                    len(values),
                                )
                            )
                        proxy_completion_costs = [
                            (
                                float(row["autonomous_stop_cost"])
                                if row["autonomous_stop_cost"] is not None
                                and row["autonomous_proxy_diagnostic_success"]
                                else 1.0
                            )
                            for row in terminal_rows
                        ]
                        formal_completion_costs = [
                            (
                                float(row["autonomous_stop_cost"])
                                if row["autonomous_stop_cost"] is not None
                                and row["autonomous_formal_success"]
                                else 1.0
                            )
                            for row in terminal_rows
                            if row["reference_eligible"]
                        ]
                        for evidence_kind, values in (
                            ("FORMAL", formal_completion_costs),
                            ("PROXY_DIAGNOSTIC", proxy_completion_costs),
                        ):
                            rows.append(
                                _metric_row(
                                    scope,
                                    method,
                                    task,
                                    mode.value,
                                    "FAILURE_PENALIZED_COMPLETION_COST",
                                    None,
                                    evidence_kind,
                                    float(np.mean(values)) if values else None,
                                    None,
                                    len(values),
                                )
                            )
    return rows


def _comparison_rows(
    config: BenchmarkConfig,
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
) -> list[dict[str, object]]:
    pairs = (
        ("H1", "B3", "B0"),
        ("H1", "B3", "B1"),
        ("H1", "B3", "B2"),
        ("H2", "B5", "B3"),
        ("H2_CONTROL", "B4", "B0"),
        ("H3", "B6", "B5"),
        ("SYSTEM", "B5", "B4"),
        ("SYSTEM", "B5", "B7"),
        ("SYSTEM", "B6", "B4"),
        ("SYSTEM", "B6", "B7"),
    )
    output = []
    replicates = int(config.values["evaluation"]["bootstrap_replicates"])
    seed = int(config.values["evaluation"]["bootstrap_seed"])
    for task in (task.value for task in _TASKS):
        for mode in EvaluationMode:
            for hypothesis, treatment, comparator in pairs:
                treatment_rows = groups[(treatment, task)]
                comparator_rows = groups[(comparator, task)]
                for metric in ("FINAL_SUCCESS_RATE", "AUSC"):
                    proxy_differences: dict[str, list[float]] = defaultdict(list)
                    formal_differences: dict[str, list[float]] = defaultdict(list)
                    for specimen_key in sorted(treatment_rows):
                        left = treatment_rows[specimen_key]
                        right = comparator_rows[specimen_key]
                        if metric == "FINAL_SUCCESS_RATE":
                            left_formal, left_proxy = _snapshot(
                                left,
                                mode=mode,
                                checkpoint=1.0,
                            )
                            right_formal, right_proxy = _snapshot(
                                right,
                                mode=mode,
                                checkpoint=1.0,
                            )
                        else:
                            left_proxy = _specimen_ausc(
                                left,
                                mode=mode,
                                formal=False,
                            )
                            right_proxy = _specimen_ausc(
                                right,
                                mode=mode,
                                formal=False,
                            )
                            left_formal = _specimen_ausc(
                                left,
                                mode=mode,
                                formal=True,
                            )
                            right_formal = _specimen_ausc(
                                right,
                                mode=mode,
                                formal=True,
                            )
                        domain = str(left[0]["dataset_id"])
                        proxy_differences[domain].append(
                            float(left_proxy) - float(right_proxy)
                        )
                        if left_formal is not None and right_formal is not None:
                            formal_differences[domain].append(
                                float(left_formal) - float(right_formal)
                            )
                    proxy_result = paired_domain_bootstrap(
                        {
                            domain: np.asarray(values, dtype=np.float64)
                            for domain, values in proxy_differences.items()
                        },
                        replicates=replicates,
                        seed=seed,
                    )
                    formal_result = (
                        paired_domain_bootstrap(
                            {
                                domain: np.asarray(values, dtype=np.float64)
                                for domain, values in formal_differences.items()
                            },
                            replicates=replicates,
                            seed=seed,
                        )
                        if formal_differences
                        else None
                    )
                    output.append(
                        {
                            "schema_version": 1,
                            "hypothesis": hypothesis,
                            "task": task,
                            "evaluation_mode": mode.value,
                            "metric": metric,
                            "treatment": treatment,
                            "comparator": comparator,
                            "formal_estimate": (
                                formal_result.estimate if formal_result else None
                            ),
                            "formal_ci_lower": (
                                formal_result.ci_lower if formal_result else None
                            ),
                            "formal_ci_upper": (
                                formal_result.ci_upper if formal_result else None
                            ),
                            "proxy_diagnostic_estimate": proxy_result.estimate,
                            "proxy_diagnostic_ci_lower": proxy_result.ci_lower,
                            "proxy_diagnostic_ci_upper": proxy_result.ci_upper,
                            "physical_specimen_n": sum(
                                len(value) for value in proxy_differences.values()
                            ),
                            "domain_n": len(proxy_differences),
                            "formal_specimen_n": sum(
                                len(value) for value in formal_differences.values()
                            ),
                            "formal_domain_n": len(formal_differences),
                            "multiple_comparison": (
                                "HOLM_PENDING"
                                if formal_result
                                else "NOT_APPLICABLE_REFERENCE_PENDING"
                            ),
                            "conclusion": "INCONCLUSIVE",
                            "evidence_status": (
                                "FORMAL_ESTIMATE_AVAILABLE_CONCLUSION_PENDING"
                                if formal_result
                                else "PROXY_ONLY_NOT_FORMAL_TASK_EVIDENCE"
                            ),
                        }
                    )
    return output


def _failure_type_rows(
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
) -> list[dict[str, object]]:
    output = []
    for (method, task), specimens in sorted(groups.items()):
        for mode in EvaluationMode:
            counts: Counter[str] = Counter()
            for rows in specimens.values():
                terminal = rows[-1]
                if mode is EvaluationMode.ANYTIME_REPORT:
                    success = bool(terminal["proxy_diagnostic_success"])
                    failures = json.loads(terminal["proxy_failure_types_json"])
                    score_row = terminal
                elif terminal["autonomous_stop_cost"] is None:
                    success = False
                    failures = ["NO_AUTONOMOUS_STOP"]
                    score_row = terminal
                else:
                    success = bool(terminal["autonomous_proxy_diagnostic_success"])
                    failures = json.loads(
                        terminal["autonomous_proxy_failure_types_json"]
                    )
                    stop_step = int(terminal["autonomous_stop_step"])
                    score_row = next(
                        row for row in rows if int(row["step"]) == stop_step
                    )
                if not success and not failures:
                    failures = _task_threshold_failures(score_row, task)
                if success:
                    counts["NO_FAILURE"] += 1
                elif failures:
                    counts.update(str(value) for value in failures)
                else:
                    counts["TASK_CRITERIA_NOT_MET"] += 1
            denominator = len(specimens)
            for failure_type, count in sorted(counts.items()):
                output.append(
                    {
                        "schema_version": 1,
                        "scope": "ALL",
                        "method": method,
                        "task": task,
                        "evaluation_mode": mode.value,
                        "evidence_kind": "PROXY_DIAGNOSTIC",
                        "failure_type": failure_type,
                        "affected_specimen_count": count,
                        "denominator": denominator,
                        "rate": float(count / denominator),
                        "status": "DIAGNOSTIC_ONLY_REFERENCE_PENDING",
                    }
                )
    return output


def _task_threshold_failures(
    row: dict[str, Any], task: str
) -> list[str]:
    if task not in {item.value for item in _TASKS}:
        raise ValueError("threshold failure task is invalid")
    if task == BenchmarkTask.LOCATE.value:
        return (
            ["BBOX_IOU_BELOW_0_50"]
            if float(row["proxy_iou"]) < 0.50
            else []
        )
    failures = []
    if float(row["proxy_iou"]) < 0.70:
        failures.append("MASK_IOU_BELOW_0_70")
    if float(row["proxy_recall"]) < 0.90:
        failures.append("CERTAIN_RECALL_BELOW_0_90")
    if float(row["proxy_relative_area_error"]) > 0.10:
        failures.append("AREA_ERROR_ABOVE_0_10")
    return failures


def _snapshot(
    rows: list[dict[str, Any]],
    *,
    mode: EvaluationMode,
    checkpoint: float,
) -> tuple[bool | None, bool]:
    available = [row for row in rows if float(row["acquisition_cost"]) <= checkpoint + 1e-15]
    row = available[-1] if available else rows[0]
    if mode is EvaluationMode.ANYTIME_REPORT:
        return row["formal_success"], bool(row["proxy_diagnostic_success"])
    stop_cost = row["autonomous_stop_cost"]
    completed = stop_cost is not None and float(stop_cost) <= checkpoint + 1e-15
    if not completed:
        return False if row["reference_eligible"] else None, False
    return row["autonomous_formal_success"], bool(
        row["autonomous_proxy_diagnostic_success"]
    )


def _route_cost_for_mode(
    rows: list[dict[str, Any]], mode: EvaluationMode
) -> float:
    if not rows or type(mode) is not EvaluationMode:
        raise ValueError("route-cost rows or evaluation mode are invalid")
    if mode is EvaluationMode.ANYTIME_REPORT:
        return float(rows[-1]["normalized_route_cost"])
    stop_step = rows[-1]["autonomous_stop_step"]
    if stop_step is None:
        return float(rows[-1]["normalized_route_cost"])
    for row in rows:
        if int(row["step"]) == int(stop_step):
            return float(row["normalized_route_cost"])
    raise ValueError("autonomous stop step is absent from report rows")


def _specimen_ausc(
    rows: list[dict[str, Any]], *, mode: EvaluationMode, formal: bool
) -> float | None:
    if formal and not rows[0]["reference_eligible"]:
        return None
    if mode is EvaluationMode.AUTONOMOUS_REPORT:
        terminal = rows[-1]
        stop_cost = terminal["autonomous_stop_cost"]
        success_key = (
            "autonomous_formal_success"
            if formal
            else "autonomous_proxy_diagnostic_success"
        )
        if stop_cost is None or not terminal[success_key]:
            return 0.0
        return 1.0 - float(stop_cost)
    success_key = "formal_success" if formal else "proxy_diagnostic_success"
    area = 0.0
    for current, following in pairwise(rows):
        area += float(bool(current[success_key])) * (
            float(following["acquisition_cost"]) - float(current["acquisition_cost"])
        )
    return float(area)


def _cost_at_success_rate(
    specimens: dict[str, list[dict[str, Any]]],
    *,
    mode: EvaluationMode,
    target: float,
) -> float | None:
    costs = sorted(
        {
            float(row["acquisition_cost"])
            for specimen_rows in specimens.values()
            for row in specimen_rows
        }
    )
    for cost in costs:
        rate = float(
            np.mean(
                [
                    _snapshot(rows, mode=mode, checkpoint=cost)[1]
                    for rows in specimens.values()
                ]
            )
        )
        if rate >= target:
            return cost
    return None


def _metric_row(
    scope: str,
    method: str,
    task: str,
    mode: str,
    metric: str,
    checkpoint: float | None,
    evidence_kind: str,
    value: float | None,
    numerator: int | None,
    denominator: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": scope,
        "method": method,
        "task": task,
        "evaluation_mode": mode,
        "metric": metric,
        "checkpoint": checkpoint,
        "evidence_kind": evidence_kind,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "status": "ESTIMATED" if evidence_kind == "PROXY_DIAGNOSTIC" else (
            "ESTIMATED" if denominator else "REFERENCE_PENDING"
        ),
    }


def _proxy_final_snapshot(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "method": row["method"],
            "task": row["task"],
            "evaluation_mode": row["evaluation_mode"],
            "proxy_diagnostic_success_rate": row["value"],
            "n": row["denominator"],
        }
        for row in rows
        if row["scope"] == "ALL"
        and row["metric"] == "SUCCESS_RATE"
        and row["checkpoint"] == 1.0
        and row["evidence_kind"] == "PROXY_DIAGNOSTIC"
    ]


def _write_curve_figures(output: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visualizations = output / "visualizations"
    visualizations.mkdir(parents=True, exist_ok=True)
    palette = ("#1b6ca8", "#d1495b", "#2a9d8f", "#f4a261", "#6a4c93", "#577590", "#8f5d2f", "#3a7d44")
    for task in (task.value for task in _TASKS):
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
        for axis, mode in zip(axes, EvaluationMode, strict=True):
            for color, method in zip(palette, (item.value for item in MethodId), strict=True):
                selected = [
                    row
                    for row in rows
                    if row["scope"] == "ALL"
                    and row["task"] == task
                    and row["method"] == method
                    and row["evaluation_mode"] == mode.value
                    and row["metric"] == "SUCCESS_RATE"
                    and row["evidence_kind"] == "PROXY_DIAGNOSTIC"
                ]
                axis.step(
                    [float(row["checkpoint"]) for row in selected],
                    [float(row["value"]) for row in selected],
                    where="post",
                    label=method,
                    color=color,
                    linewidth=1.4,
                    marker="o",
                    markevery=[len(selected) - 1],
                    markersize=3.0,
                    clip_on=False,
                )
            axis.set_title(mode.value)
            axis.set_xlabel("Exact acquisition cost")
            axis.grid(alpha=0.25)
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
        axes[0].set_ylabel("Proxy diagnostic success rate")
        axes[1].legend(ncol=2, fontsize=8, frameon=False)
        figure.suptitle(f"{task}: proxy diagnostic only (reviewed reference pending)")
        figure.tight_layout()
        figure.savefig(
            visualizations / f"proxy_success_cost_{task.lower()}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)


def _write_reference_manifest(
    config: BenchmarkConfig, records: tuple[InputSpecimen, ...]
) -> None:
    output = _result_root(config)
    pilot_keys = {
        record.specimen_key
        for record in select_pilot_records(
            records,
            domain_order=config.domain_order,
            per_domain=config.pilot_per_domain,
            seed=config.pilot_seed,
        )
    }
    rows = []
    for record in records:
        queue_path = output / "annotation_queue" / _queue_name(record)
        review_state = "pending"
        reference_type = "ALGORITHM_DERIVED_NOT_REVIEWED"
        reviewer_alias = None
        formal_eligible = False
        if queue_path.is_file():
            payload = json.loads(queue_path.read_text(encoding="utf-8"))
            review_state = payload.get("review_state", "pending")
            reference_type = payload.get(
                "reference_type", "ALGORITHM_DERIVED_NOT_REVIEWED"
            )
            reviewer_alias = payload.get("reviewer_alias")
            if review_state == "reviewed":
                reference = reference_from_payload(
                    payload,
                    native_shape=record.native_shape,
                )
                formal_eligible = _task_reference_eligible(reference)
        rows.append(
            {
                "schema_version": 1,
                "dataset_id": record.dataset_id,
                "specimen_id": record.specimen_id,
                "specimen_key": record.specimen_key,
                "registered_cscan_sha256": record.cscan_sha256,
                "reference_type": reference_type,
                "review_state": review_state,
                "reviewer_alias": reviewer_alias,
                "formal_eligible": formal_eligible,
                "pilot_selected": record.specimen_key in pilot_keys,
                "annotation_path": (
                    queue_path.relative_to(output).as_posix()
                    if queue_path.is_file()
                    else ""
                ),
                "formal_success_available": formal_eligible,
            }
        )
    _write_csv(output / "reference_manifest.csv", rows)


def _reviewed_reference_count(
    config: BenchmarkConfig, records: tuple[InputSpecimen, ...]
) -> int:
    count = 0
    queue = _result_root(config) / "annotation_queue"
    for record in records:
        path = queue / _queue_name(record)
        if not path.is_file():
            continue
        try:
            reference = reference_from_payload(
                json.loads(path.read_text(encoding="utf-8")),
                native_shape=record.native_shape,
            )
        except (json.JSONDecodeError, ValueError):
            continue
        count += int(_task_reference_eligible(reference))
    return count


def _load_evaluation_reference(
    config: BenchmarkConfig, record: InputSpecimen, runtime
) -> CScanReference:
    path = _result_root(config) / "annotation_queue" / _queue_name(record)
    if path.is_file():
        try:
            reference = reference_from_payload(
                json.loads(path.read_text(encoding="utf-8")),
                native_shape=record.native_shape,
            )
        except (json.JSONDecodeError, ValueError):
            reference = None
        if reference is not None and _task_reference_eligible(reference):
            if (
                reference.specimen_key != record.specimen_key
                or reference.source_image_sha256 != record.cscan_sha256
            ):
                raise ValueError("reviewed reference identity differs from the input")
            return reference
    return _full_input_proxy(config, record, runtime)


def _task_reference_eligible(reference: CScanReference) -> bool:
    return reference.formal_eligible and bool(np.any(reference.certain_mask))


def _full_input_proxy(
    config: BenchmarkConfig,
    record: InputSpecimen,
    runtime,
) -> CScanReference:
    with Image.open(record.cscan_path) as image:
        full_scan = np.asarray(image.convert("RGB"), dtype=np.uint8)
    positions = np.argwhere(np.ones(record.native_shape, dtype=np.bool_)).astype(
        np.int64,
        copy=False,
    )
    reader = config.values["reader"]
    evidence = read_sparse_evidence(
        native_shape=record.native_shape,
        positions=positions,
        values=full_scan.reshape(-1, 3),
        background_rgb=runtime.background_rgb,
        distance_threshold=float(reader["distance_threshold"]),
    )
    full_report = build_task_report(
        evidence,
        runtime.grid,
        cell_levels=(2,) * 64,
        task=BenchmarkTask.CHARACTERIZE,
        cell_indication_fraction=float(reader["cell_indication_fraction"]),
        minimum_component_pixels=int(reader["minimum_component_pixels"]),
    )
    lower = float(reader["distance_threshold"]) - float(reader["uncertainty_band"])
    uncertain = (evidence.scores >= lower) & ~full_report.predicted_mask
    return CScanReference(
        specimen_key=record.specimen_key,
        source_image_sha256=record.cscan_sha256,
        reference_type=ReferenceType.ALGORITHM_DERIVED_NOT_REVIEWED,
        review_state=ReviewState.PENDING,
        reviewer_alias=None,
        certain_mask=full_report.predicted_mask,
        uncertain_mask=uncertain,
    )


def _derive_proxy(
    config: BenchmarkConfig, record: InputSpecimen
) -> tuple[CScanReference, np.ndarray]:
    with Image.open(record.cscan_path) as image:
        full_scan = np.asarray(image.convert("RGB"), dtype=np.uint8)
    reader = config.values["reader"]
    reference = derive_proxy_reference(
        specimen_key=record.specimen_key,
        source_image_sha256=record.cscan_sha256,
        full_scan=full_scan,
        distance_threshold=float(reader["distance_threshold"]),
        minimum_component_pixels=int(reader["minimum_component_pixels"]),
        uncertainty_band=float(reader["uncertainty_band"]),
        border_exclusion_fraction=float(reader["border_exclusion_fraction"]),
    )
    return reference, full_scan


def _write_reference_visualization(
    record: InputSpecimen,
    reference: CScanReference,
    full_scan: np.ndarray,
    target: Path,
) -> None:
    with Image.open(record.surface_path) as source:
        surface = render_surface_inputs(source, max_edge=512).clean
    cscan = Image.fromarray(full_scan, mode="RGB")
    overlay = np.array(full_scan, copy=True)
    overlay[reference.certain_mask] = (
        0.45 * overlay[reference.certain_mask] + 0.55 * np.asarray([220, 35, 45])
    ).astype(np.uint8)
    overlay_image = Image.fromarray(overlay, mode="RGB")
    panels = []
    for label, image in (
        ("Surface (ROT90 scan frame)", surface),
        ("Registered C-scan", cscan),
        ("Algorithm proposal - NOT REVIEWED", overlay_image),
    ):
        panel = image.copy()
        panel.thumbnail((512, 260), Image.Resampling.LANCZOS)
        framed = Image.new("RGB", (512, panel.height + 24), "white")
        framed.paste(panel, ((512 - panel.width) // 2, 24))
        ImageDraw.Draw(framed).text((5, 5), label, fill="black")
        panels.append(framed)
    canvas = Image.new("RGB", (512, sum(panel.height for panel in panels)), "white")
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG", optimize=True)


def _surface_request(config: BenchmarkConfig, rendered) -> SurfacePlanRequest:
    model = config.values["model"]
    return SurfacePlanRequest(
        model_repository=model["repository"],
        model_revision=model["revision"],
        prompt_sha256=hashlib.sha256(INITIAL_SURFACE_PROMPT.encode("utf-8")).hexdigest(),
        clean_image_sha256=rendered.clean_sha256,
        gridded_image_sha256=rendered.gridded_sha256,
        preprocessing_sha256=config.values["surface"]["preprocessing_sha256"],
    )


def _get_surface_inference(
    config: BenchmarkConfig,
    runtime,
    cache_path: Path,
) -> SurfacePlanInference:
    request = _surface_request(config, runtime.render)
    result = SurfacePlanCache(cache_path).get(request)
    if result is None:
        raise RuntimeError(f"surface plan is missing: {runtime.record.specimen_key}")
    return result


def _require_surface_plans(
    config: BenchmarkConfig,
    records: tuple[InputSpecimen, ...],
    cache_path: Path,
) -> None:
    cache = SurfacePlanCache(cache_path)
    for record in records:
        with Image.open(record.surface_path) as image:
            rendered = render_surface_inputs(
                image,
                max_edge=int(config.values["surface"]["max_edge"]),
            )
        if cache.get(_surface_request(config, rendered)) is None:
            raise RuntimeError(f"surface plan is missing: {record.specimen_key}")


def _cohort_records(roster, cohort: str, *, allow_all: bool) -> tuple[InputSpecimen, ...]:
    if cohort == "smoke":
        return roster.smoke_records
    if cohort == "pilot":
        return roster.pilot_records
    if cohort == "all" and allow_all:
        return roster.records
    raise ValueError("cohort must be smoke or pilot for this reference-pending stage")


def _record_needs_methods(
    work: Path,
    record: InputSpecimen,
    methods: tuple[MethodId, ...],
) -> bool:
    return any(
        not _shard_exists(work, record, method, task)
        for method in methods
        for task in _TASKS
    )


def _shard_exists(
    work: Path, record: InputSpecimen, method: MethodId, task: BenchmarkTask
) -> bool:
    stem = _shard_stem(record, method, task)
    return (work / "episodes" / f"{stem}.parquet").is_file() and (
        work / "reports" / f"{stem}.parquet"
    ).is_file()


def _shard_stem(
    record: InputSpecimen, method: MethodId, task: BenchmarkTask
) -> str:
    identity = hashlib.sha256(record.specimen_key.encode("utf-8")).hexdigest()[:12]
    return f"{record.dataset_id}__{identity}__{method.value}__{task.value}"


def _queue_name(record: InputSpecimen) -> str:
    return f"{record.dataset_id}__{record.specimen_id}.json"


def _result_root(config: BenchmarkConfig) -> Path:
    relative = Path(config.values["outputs"]["root"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("benchmark output path is invalid")
    return config.project_root / relative


def _work_root(config: BenchmarkConfig) -> Path:
    digest = hashlib.sha256(config.config_sha256.encode("ascii"))
    module_root = Path(__file__).parent
    for name in (
        "benchmark.py",
        "contracts.py",
        "planning.py",
        "reader.py",
        "references.py",
        "reporting.py",
        "route.py",
        "runtime.py",
        "vlm.py",
    ):
        digest.update(name.encode("ascii"))
        digest.update((module_root / name).read_bytes())
    reviewed_digest = _reviewed_reference_cache_digest(config)
    if reviewed_digest is not None:
        digest.update(b"reviewed_reference_set")
        digest.update(reviewed_digest.encode("ascii"))
    return _result_root(config) / ".work" / digest.hexdigest()[:16]


def _reviewed_reference_cache_digest(config: BenchmarkConfig) -> str | None:
    queue = _result_root(config) / "annotation_queue"
    reviewed = []
    if queue.is_dir():
        for path in sorted(queue.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if (
                payload.get("review_state") == "reviewed"
                and payload.get("reference_type")
                in {"EXPERT_REVIEWED", "AUTHOR_PROVIDED"}
                and payload.get("regions")
            ):
                reviewed.append((path.name, path.read_bytes()))
    if not reviewed:
        return None
    digest = hashlib.sha256()
    for name, content in reviewed:
        digest.update(name.encode("utf-8"))
        digest.update(content)
    return digest.hexdigest()


def _write_model_manifest(
    config: BenchmarkConfig, *, execution: dict[str, object]
) -> None:
    model = config.values["model"]
    path = Path(model["local_path"])
    payload = {
        "schema_version": 1,
        "model_repository": model["repository"],
        "model_revision": model["revision"],
        "model_license": model["license"],
        "local_model_path": path.as_posix(),
        "local_model_available": path.is_dir(),
        "frozen_model_count": 1,
        "dtype": model["dtype"],
        "device": model["device"],
        "do_sample": model["do_sample"],
        "max_new_tokens": model["max_new_tokens"],
        "processor_visual_tokens": {
            "minimum": _QWEN_MIN_VISUAL_TOKENS,
            "maximum": _QWEN_MAX_VISUAL_TOKENS,
        },
        "processor_pixels_per_visual_token": _QWEN_PIXELS_PER_VISUAL_TOKEN,
        "processor_image_transform": {
            "implementation": "Qwen2VLImageProcessor.smart_resize",
            "resize_factor_px": _QWEN_RESIZE_FACTOR,
            "minimum_pixels": (
                _QWEN_MIN_VISUAL_TOKENS * _QWEN_PIXELS_PER_VISUAL_TOKEN
            ),
            "maximum_pixels": (
                _QWEN_MAX_VISUAL_TOKENS * _QWEN_PIXELS_PER_VISUAL_TOKEN
            ),
            "patch_size_px": _QWEN_PATCH_SIZE,
            "temporal_patch_size": 2,
            "merge_size": _QWEN_MERGE_SIZE,
            "spatial_padding": "NONE",
            "text_batch_padding": True,
            "request_batch_size": 1,
            "images_per_request": 2,
            "actual_dimensions": "SEE_INPUT_MANIFEST",
        },
        "surface_render": {
            "orientation": config.values["surface"]["orientation"],
            "maximum_edge_px": config.values["surface"]["max_edge"],
            "preserve_aspect_ratio": config.values["surface"][
                "preserve_aspect_ratio"
            ],
        },
        "format_retries": model["format_retries"],
        "max_replans_per_episode": model["max_replans_per_episode"],
        "initial_prompt": INITIAL_SURFACE_PROMPT,
        "initial_prompt_sha256": hashlib.sha256(
            INITIAL_SURFACE_PROMPT.encode("utf-8")
        ).hexdigest(),
        "surface_preprocessing_sha256": config.values["surface"][
            "preprocessing_sha256"
        ],
        "model_config_sha256": (
            _file_sha256(path / "config.json")
            if (path / "config.json").is_file()
            else None
        ),
        "model_preprocessor_config_sha256": (
            _file_sha256(path / "preprocessor_config.json")
            if (path / "preprocessor_config.json").is_file()
            else None
        ),
        "software_versions": {
            "python": platform.python_version(),
            "torch": _installed_version("torch"),
            "transformers": _installed_version("transformers"),
            "Pillow": _installed_version("Pillow"),
        },
        "config_sha256": config.config_sha256,
        "execution": execution,
        "cumulative_response_cache": _cache_summary(
            _result_root(config) / "surface_plans.jsonl"
        ),
    }
    _write_json(_result_root(config) / "model_and_prompt_manifest.json", payload)


def _existing_model_execution(config: BenchmarkConfig) -> dict[str, object] | None:
    path = _result_root(config) / "model_and_prompt_manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        execution = payload["execution"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    model = config.values["model"]
    if (
        payload.get("model_repository") != model["repository"]
        or payload.get("model_revision") != model["revision"]
        or type(execution) is not dict
    ):
        return None
    return execution


def _qwen_processor_geometry(height: int, width: int) -> dict[str, object]:
    if (
        type(height) is not int
        or type(width) is not int
        or min(height, width) < _QWEN_RESIZE_FACTOR
        or max(height, width) / min(height, width) > 200
    ):
        raise ValueError("Qwen image dimensions are invalid")
    resized_height = round(height / _QWEN_RESIZE_FACTOR) * _QWEN_RESIZE_FACTOR
    resized_width = round(width / _QWEN_RESIZE_FACTOR) * _QWEN_RESIZE_FACTOR
    minimum_pixels = _QWEN_MIN_VISUAL_TOKENS * _QWEN_PIXELS_PER_VISUAL_TOKEN
    maximum_pixels = _QWEN_MAX_VISUAL_TOKENS * _QWEN_PIXELS_PER_VISUAL_TOKEN
    if resized_height * resized_width > maximum_pixels:
        scale = math.sqrt((height * width) / maximum_pixels)
        resized_height = (
            math.floor(height / scale / _QWEN_RESIZE_FACTOR) * _QWEN_RESIZE_FACTOR
        )
        resized_width = (
            math.floor(width / scale / _QWEN_RESIZE_FACTOR) * _QWEN_RESIZE_FACTOR
        )
    elif resized_height * resized_width < minimum_pixels:
        scale = math.sqrt(minimum_pixels / (height * width))
        resized_height = (
            math.ceil(height * scale / _QWEN_RESIZE_FACTOR) * _QWEN_RESIZE_FACTOR
        )
        resized_width = (
            math.ceil(width * scale / _QWEN_RESIZE_FACTOR) * _QWEN_RESIZE_FACTOR
        )
    grid_height = resized_height // _QWEN_PATCH_SIZE
    grid_width = resized_width // _QWEN_PATCH_SIZE
    return {
        "height_px": resized_height,
        "width_px": resized_width,
        "image_grid_thw": [1, grid_height, grid_width],
        "visual_token_count": (
            grid_height * grid_width // (_QWEN_MERGE_SIZE**2)
        ),
    }


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _cache_summary(path: Path) -> dict[str, object]:
    initial = 0
    replans = 0
    calls = 0
    fallback = 0
    latency = 0.0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            record_type = row.get("record_type", "initial_surface_plan")
            initial += int(record_type == "initial_surface_plan")
            replans += int(record_type == "replan_choice")
            calls += int(row["call_count"])
            latency += float(row["latency_seconds"])
            fallback += int(row["parse_status"] == "FALLBACK")
    return {
        "initial_records": initial,
        "replan_records": replans,
        "deployment_calls": calls,
        "fallback_count": fallback,
        "latency_seconds": latency,
    }


def _initial_cache_count(path: Path) -> int:
    return int(_cache_summary(path)["initial_records"])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError(f"CSV rows have inconsistent columns: {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty trajectory shard")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(rows, strict=False, infer_schema_length=None)
    _write_table_atomic(path, table)


def _write_table_atomic(path: Path, table: pl.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def _write_checksums(output: Path) -> None:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name != "CHECKSUMS.sha256"
        and ".work" not in path.relative_to(output).parts
    )
    payload = "".join(
        f"{_file_sha256(path)}  {path.relative_to(output).as_posix()}\n"
        for path in paths
    )
    (output / "CHECKSUMS.sha256").write_text(payload, encoding="ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "evaluate_benchmark",
    "export_annotation_queue",
    "infer_surface_plans",
    "prepare_benchmark",
    "run_benchmark",
    "verify_benchmark",
]
