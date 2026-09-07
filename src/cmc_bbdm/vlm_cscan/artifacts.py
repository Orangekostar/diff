"""Preparation, execution, aggregation, and integrity artifacts for the pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import time
from collections import defaultdict
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
    render_hashes: dict[str, tuple[str, str]] = {}
    for record in roster.pilot_records:
        with Image.open(record.surface_path) as image:
            rendered = render_surface_inputs(
                image,
                max_edge=int(config.values["surface"]["max_edge"]),
            )
        render_hashes[record.specimen_key] = (
            rendered.clean_sha256,
            rendered.gridded_sha256,
        )
    input_rows = []
    for record in roster.records:
        clean_sha, gridded_sha = render_hashes.get(record.specimen_key, ("", ""))
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
    _write_model_manifest(
        config,
        execution={
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
    visualizations = output / "visualizations"
    queue.mkdir(parents=True, exist_ok=True)
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
                "reader": config.values["reader"],
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
                    early_wrong = [
                        row["autonomous_stop_cost"] is not None
                        and not row["autonomous_proxy_diagnostic_success"]
                        for row in terminal_rows
                    ]
                    incomplete = [
                        row["autonomous_stop_cost"] is None for row in terminal_rows
                    ]
                    terminal_route = [
                        float(row["normalized_route_cost"]) for row in terminal_rows
                    ]
                    for name, values in (
                        ("EARLY_WRONG_STOP_RATE", early_wrong),
                        ("INCOMPLETE_RATE", incomplete),
                        ("FINAL_NORMALIZED_ROUTE_COST", terminal_route),
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
                                int(sum(values)) if name.endswith("RATE") else None,
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
                    differences: dict[str, list[float]] = defaultdict(list)
                    for specimen_key in sorted(treatment_rows):
                        left = treatment_rows[specimen_key]
                        right = comparator_rows[specimen_key]
                        if metric == "FINAL_SUCCESS_RATE":
                            left_value = _snapshot(left, mode=mode, checkpoint=1.0)[1]
                            right_value = _snapshot(right, mode=mode, checkpoint=1.0)[1]
                        else:
                            left_value = _specimen_ausc(left, mode=mode, formal=False)
                            right_value = _specimen_ausc(right, mode=mode, formal=False)
                        differences[str(left[0]["dataset_id"])].append(
                            float(left_value) - float(right_value)
                        )
                    result = paired_domain_bootstrap(
                        {
                            domain: np.asarray(values, dtype=np.float64)
                            for domain, values in differences.items()
                        },
                        replicates=replicates,
                        seed=seed,
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
                            "formal_estimate": None,
                            "formal_ci_lower": None,
                            "formal_ci_upper": None,
                            "proxy_diagnostic_estimate": result.estimate,
                            "proxy_diagnostic_ci_lower": result.ci_lower,
                            "proxy_diagnostic_ci_upper": result.ci_upper,
                            "physical_specimen_n": sum(len(value) for value in differences.values()),
                            "domain_n": len(differences),
                            "multiple_comparison": "NOT_APPLICABLE_REFERENCE_PENDING",
                            "conclusion": "INCONCLUSIVE",
                            "evidence_status": "PROXY_ONLY_NOT_FORMAL_TASK_EVIDENCE",
                        }
                    )
    return output


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
                formal_eligible = reference.formal_eligible
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
        count += int(reference.formal_eligible)
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
        if reference is not None and reference.formal_eligible:
            if (
                reference.specimen_key != record.specimen_key
                or reference.source_image_sha256 != record.cscan_sha256
            ):
                raise ValueError("reviewed reference identity differs from the input")
            return reference
    return _full_input_proxy(config, record, runtime)


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
    return _result_root(config) / ".work" / digest.hexdigest()[:16]


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
        "format_retries": model["format_retries"],
        "max_replans_per_episode": model["max_replans_per_episode"],
        "initial_prompt": INITIAL_SURFACE_PROMPT,
        "initial_prompt_sha256": hashlib.sha256(
            INITIAL_SURFACE_PROMPT.encode("utf-8")
        ).hexdigest(),
        "surface_preprocessing_sha256": config.values["surface"][
            "preprocessing_sha256"
        ],
        "config_sha256": config.config_sha256,
        "execution": execution,
    }
    _write_json(_result_root(config) / "model_and_prompt_manifest.json", payload)


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
