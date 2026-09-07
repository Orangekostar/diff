"""Study-stage orchestration for the same-perception learned C-scan pilot."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from cmc_bbdm.inspection_agent.state import (
    GeneralizedMeasurementState,
    InspectionCellAction,
    action_added_positions_from_mask,
    candidate_budget_record,
)
from cmc_bbdm.vlm_cscan.route import compile_route, reference_full_raster_length
from cmc_bbdm.vlm_cscan.vlm import QwenVLBackend

from .artifacts import (
    verify_checksums,
    write_checksums,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)
from .contracts import Split, Task
from .metrics import (
    MetricRecord,
    StepSnapshot,
    exact_step_integral,
    failure_penalized_cost,
    paired_physical_specimen_bootstrap,
    value_at_cost,
)
from .observation import ObservationPacket, build_observation_packet
from .perception import (
    FORMAT_REPAIR_CONTEXT_PREFIX,
    FORMAT_REPAIR_CONTEXT_SUFFIX,
    FORMAT_REPAIR_PROMPT,
    SURFACE_PERCEPT_PROMPT,
    SurfacePercept,
    SurfacePerceptCache,
    SurfacePerceptRequest,
    display_cell_id,
    map_percept_to_physical,
)
from .policies import (
    LearnedCellActor,
    RuleMethod,
    select_balanced_visible_action,
    select_rule_action,
)
from .readout import (
    TaskReportV2,
    build_task_report_v2,
    read_visible_evidence,
    read_visible_task_report,
)
from .rollouts import compile_route_cost, cost_to_go_targets
from .runtime import (
    SplitAssignment,
    StudyConfig,
    StudyContext,
    StudySpecimenRuntime,
    load_study_config,
    load_study_context,
    open_study_specimen,
    render_registered_surface,
)
from .stopping import (
    LearnedStopAuthorization,
    LearnedStopHead,
    LearnedStopStatus,
    RuleStopController,
    StopValidationRow,
    calibrate_learned_stop,
)
from .training import (
    PolicyTrainingExample,
    StopTrainingExample,
    TrainingRoute,
    behavior_cloning_targets,
    fit_actor,
    fit_stop_head,
)

SPLIT_FIELDS = (
    "schema_version",
    "dataset_id",
    "specimen_id",
    "specimen_key",
    "split",
    "domain_rank",
    "surface_sha256",
    "registered_cscan_sha256",
    "native_height",
    "native_width",
    "transform_sha256",
)
REFERENCE_FIELDS = (
    "schema_version",
    "dataset_id",
    "specimen_id",
    "specimen_key",
    "split",
    "reference_type",
    "review_state",
    "reviewer_alias",
    "formal_eligible",
    "proxy_scope",
    "formal_success_available",
)
READOUT_FIELDS = (
    "schema_version",
    "dataset_id",
    "specimen_id",
    "specimen_key",
    "split",
    "task",
    "native_count",
    "candidate_pixels",
    "candidate_fraction",
    "candidate_cells",
    "support_count",
    "signal_strength",
    "nondegenerate",
)
MODEL_METHODS = ("L_BC", "L_CTG_0", "L_CTG_1", "S_LEARN")
RULE_CONFIGS = (
    ("R_GEOM", RuleMethod.R_GEOM, 4),
    ("R_CENTER", RuleMethod.R_CENTER, 4),
    ("R_VLM_OPEN", RuleMethod.R_VLM_OPEN, 4),
    ("R_LEGACY", RuleMethod.R_LEGACY, 4),
    ("R_BALANCED_P4", RuleMethod.R_BALANCED, 4),
    ("R_BALANCED_P8", RuleMethod.R_BALANCED, 8),
)
_BRANCH_WORKER_CONTEXT: StudyContext | None = None
_EVALUATION_WORKER_CONTEXT: StudyContext | None = None
_EVALUATION_WORKER_PERCEPTS: dict[str, SurfacePercept] | None = None
_EVALUATION_WORKER_PLANNERS: tuple[
    tuple[str, RuleMethod | LearnedCellActor, int, int], ...
] = ()
_EVALUATION_WORKER_STOP: LearnedStopHead | None = None
_EVALUATION_WORKER_DEVICE = "cpu"
_EVALUATION_WORKER_THRESHOLD: float | None = None


def prepare_study(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    started = time.perf_counter()
    config, context = _load(config_path, project_root, source_root)
    output = _output_root(config)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.path, output / "config.yaml")
    split_rows = tuple(_split_row(row) for row in context.roster.assignments)
    write_csv_atomic(output / "split_manifest.csv", split_rows, SPLIT_FIELDS)
    reference_rows = tuple(_reference_row(row) for row in context.roster.assignments)
    write_csv_atomic(
        output / "reference_manifest.csv", reference_rows, REFERENCE_FIELDS
    )
    counts = {
        split.value: sum(row.split is split for row in context.roster.assignments)
        for split in Split
    }
    domains = {
        domain: {
            split.value: sum(
                row.record.dataset_id == domain and row.split is split
                for row in context.roster.assignments
            )
            for split in Split
        }
        for domain in config.domain_order
    }
    model = config.values["surface_model"]
    inventory = {
        "schema_version": 1,
        "stage": "W0_PREPARED",
        "repository_base_sha": config.values["repository_base_sha"],
        "config_sha256": config.config_sha256,
        "evidence_scope": config.values["cohort"]["evidence_scope"],
        "source_root": str(context.source_root),
        "source_root_exists": context.source_root.is_dir(),
        "total_source_specimens": len(context.roster.records),
        "pilot_specimens": len(context.roster.pilot_records),
        "split_counts": counts,
        "domain_split_counts": domains,
        "reference": {
            "state": "PROXY_ONLY",
            "reviewed_count": 0,
            "pending_algorithmic_count": len(reference_rows),
            "formal_effect": None,
            "proxy_scope": "SAME_READER_SELF_CONSISTENCY",
        },
        "background_prior": {
            "fit_split": context.background_prior.fit_split,
            "fit_specimen_count": len(context.background_prior.fit_specimen_keys),
            "fit_specimen_keys_sha256": _json_sha(
                context.background_prior.fit_specimen_keys
            ),
            "rgb": context.background_prior.rgb.tolist(),
        },
        "surface_model": {
            "repository": model["repository"],
            "revision": model["revision"],
            "local_path": model["local_path"],
            "available": Path(model["local_path"]).is_dir(),
            "device": model["device"],
            "dtype": model["dtype"],
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "no_new_training": True,
        "test_opened": False,
    }
    write_json_atomic(output / "inventory.json", inventory)
    return {
        "stage": "W0_PREPARED",
        "output": str(output),
        "pilot_specimens": len(split_rows),
        "split_counts": counts,
        "reference": "PROXY_ONLY",
    }


def run_perception(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    started = time.perf_counter()
    config, context = _load(config_path, project_root, source_root)
    output = _output_root(config)
    _require_files(output, ("inventory.json", "split_manifest.csv"))
    cache = SurfacePerceptCache(output / "surface_percepts.jsonl")
    backend = _surface_backend(config)
    actual_calls = 0
    cache_hits = 0
    repairs = 0
    mapped_rows: list[dict[str, object]] = []
    originals: dict[str, SurfacePercept] = {}
    assignments = tuple(
        sorted(context.roster.assignments, key=lambda row: row.record.specimen_key)
    )
    for index, assignment in enumerate(assignments, start=1):
        registered = render_registered_surface(
            assignment.record,
            max_edge=int(config.legacy_config.values["surface"]["max_edge"]),
        )
        request = _surface_request(config, registered.render)
        result = cache.resolve(
            request,
            lambda prompt, render=registered.render: backend.infer(
                (render.clean, render.gridded), prompt
            ),
        )
        actual_calls += result.actual_call_count
        cache_hits += int(result.cache_hit)
        repairs += int(result.repaired and not result.cache_hit)
        originals[assignment.record.specimen_key] = result.percept
        mapped_rows.append(
            {
                "specimen_key": assignment.record.specimen_key,
                "split": assignment.split.value,
                "cache_key": result.cache_key,
                "cache_hit": result.cache_hit,
                "call_count": result.original_call_count,
                "repaired": result.repaired,
                "clean_image_sha256": registered.render.clean_sha256,
                "gridded_image_sha256": registered.render.gridded_sha256,
                "region_count": len(result.percept.regions),
                "cell_count": sum(len(row.cells) for row in result.percept.regions),
                "no_reliable_cue": result.percept.no_reliable_cue,
            }
        )
        print(
            f"surface percept {index}/{len(assignments)} "
            f"calls={actual_calls} cache_hits={cache_hits}",
            flush=True,
        )
    if actual_calls > int(config.values["surface_model"]["main_new_call_cap"]) + repairs:
        raise RuntimeError("main surface-call cap exceeded")
    diagnostics = _perception_diagnostics(
        config=config,
        context=context,
        backend=backend,
        originals=originals,
        output=output,
    )
    readout_rows = _full_input_readout_validation(context)
    write_csv_atomic(output / "readout_validation.csv", readout_rows, READOUT_FIELDS)
    nondegenerate = sum(row["nondegenerate"] for row in readout_rows)
    manifest = {
        "schema_version": 1,
        "state": "REAL_FROZEN_VLM_PERCEPTION_COMPLETE",
        "model_repository": config.values["surface_model"]["repository"],
        "model_revision": config.values["surface_model"]["revision"],
        "prompt_sha256": _surface_prompt_sha256(),
        "schema_version_surface_percept": 2,
        "render_version": _render_version(config),
        "specimen_count": len(assignments),
        "records": mapped_rows,
        "execution": {
            "actual_calls_this_command": actual_calls,
            "cache_hits_this_command": cache_hits,
            "repair_calls_this_command": repairs,
            "cumulative_deployment_calls": sum(
                int(row["call_count"]) for row in mapped_rows
            ),
            "main_call_cap": config.values["surface_model"]["main_new_call_cap"],
            "elapsed_seconds": time.perf_counter() - started,
        },
        "diagnostics": diagnostics,
        "reader_v2": {
            "record_count": len(readout_rows),
            "nondegenerate_count": nondegenerate,
            "status": (
                "FUNCTIONAL" if nondegenerate == len(readout_rows) else "READOUT_LIMITED"
            ),
        },
        "task_independent": True,
        "action_free": True,
        "test_outcomes_opened": False,
    }
    write_json_atomic(output / "perception_manifest.json", manifest)
    return {
        "stage": "W1_PERCEPTION_COMPLETE",
        "specimens": len(assignments),
        "actual_calls": actual_calls,
        "cache_hits": cache_hits,
        "repairs": repairs,
        "diagnostic_calls": diagnostics["cumulative_diagnostic_calls"],
        "readout_status": manifest["reader_v2"]["status"],
    }


def build_training_bank(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    started = time.perf_counter()
    config, context = _load(config_path, project_root, source_root)
    output = _output_root(config)
    _require_files(
        output,
        ("inventory.json", "surface_percepts.jsonl", "perception_manifest.json"),
    )
    percepts = _load_percepts(config, context, output)
    base_rows: list[dict[str, object]] = []
    bc_rows: list[dict[str, object]] = []
    stop_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    transition_count = 0
    training = config.values["training"]
    targets = tuple(float(value) for value in training["base_state_costs"])
    cap = int(training["logical_transition_cap"])
    assignments = tuple(
        row for row in context.roster.assignments if row.split is Split.TRAIN
    )
    branch_executor = _new_branch_executor(config, context)
    for specimen_index, assignment in enumerate(assignments, start=1):
        runtime = open_study_specimen(context, assignment.record)
        percept = percepts[assignment.record.specimen_key]
        states, trajectory_transitions = _behavior_states(
            runtime,
            percept=percept,
            task=Task.LOCATE,
            prior=context.background_prior,
            distance_threshold=float(config.values["reader"]["distance_threshold"]),
            target_costs=targets,
        )
        for task in Task:
            reference = _full_reference(context, runtime, task)
            transition_count += trajectory_transitions
            if transition_count > cap:
                raise RuntimeError("training transition cap exceeded")
            for state_index, state in enumerate(states):
                observation = runtime.world.replay(state["history"])
                packet = _packet(
                    observation,
                    runtime=runtime,
                    percept=percept,
                    task=task,
                    context=context,
                    probe_position=state["probe_position"],
                    route_cost=state["route_cost"],
                )
                mu_action = select_rule_action(
                    RuleMethod.R_BALANCED,
                    packet,
                    grid=runtime.grid,
                    coverage_period=4,
                )
                candidates = _candidate_actions(
                    packet,
                    runtime=runtime,
                    mu_action=mu_action,
                    seed_material=(
                        f"{assignment.record.specimen_key}|{task.value}|{state_index}"
                    ),
                    limit=int(training["max_candidates_per_state"]),
                )
                branches, branch_transitions = _candidate_branches(
                    runtime,
                    candidates=candidates,
                    base_history=state["history"],
                    base_probe=state["probe_position"],
                    base_route_cost=state["route_cost"],
                    percept=percept,
                    task=task,
                    reference=reference,
                    context=context,
                    executor=branch_executor,
                )
                transition_count += branch_transitions
                if transition_count > cap:
                    raise RuntimeError("training transition cap exceeded")
                labels = cost_to_go_targets(
                    current_cost=observation.effective_budget,
                    candidate_snapshots=branches,
                    temperature=float(training["cost_temperature"]),
                    auxiliary_weight=float(training["auxiliary_loss_weight"]),
                )
                base_rows.append(
                    _policy_row(
                        packet,
                        specimen_key=assignment.record.specimen_key,
                        task=task,
                        queried_cells=tuple(int(value) for value in labels.queried_actions),
                        target_probabilities=labels.probabilities,
                        state_origin="BEHAVIOR_BASE",
                        state_cost=observation.effective_budget,
                        main_costs=labels.main_costs,
                        auxiliary_costs=labels.auxiliary_costs,
                    )
                )
                bc_cells, bc_targets = behavior_cloning_targets(
                    packet.legal_mask, mu_action.cell_index
                )
                bc_rows.append(
                    _policy_row(
                        packet,
                        specimen_key=assignment.record.specimen_key,
                        task=task,
                        queried_cells=bc_cells,
                        target_probabilities=bc_targets,
                        state_origin="BEHAVIOR_BASE",
                        state_cost=observation.effective_budget,
                        main_costs=np.zeros(
                            np.count_nonzero(packet.legal_mask), dtype=np.float64
                        ),
                        auxiliary_costs=np.zeros(
                            np.count_nonzero(packet.legal_mask), dtype=np.float64
                        ),
                    )
                )
                score = _proxy_score(packet.report, reference)
                stop_rows.append(
                    _stop_row(
                        packet,
                        specimen_key=assignment.record.specimen_key,
                        task=task,
                        label=score["success"],
                    )
                )
                manifest_rows.append(
                    {
                        "specimen_key": assignment.record.specimen_key,
                        "domain": assignment.record.dataset_id,
                        "task": task.value,
                        "state_index": state_index,
                        "state_cost": observation.effective_budget,
                        "queried_cells": list(labels.queried_actions),
                        "candidate_count": len(labels.queried_actions),
                        "suffix_transitions": branch_transitions,
                    }
                )
        print(
            f"training bank {specimen_index}/{len(assignments)} "
            f"states={len(base_rows)} transitions={transition_count}",
            flush=True,
        )
    branch_executor.shutdown(wait=True)
    work = output / "_work"
    work.mkdir(parents=True, exist_ok=True)
    bank_path = work / "training_bank.pt"
    _torch_save_atomic(
        {
            "schema_version": 1,
            "config_sha256": config.config_sha256,
            "base_ctg": base_rows,
            "base_bc": bc_rows,
            "stop": stop_rows,
        },
        bank_path,
    )
    manifest = {
        "schema_version": 1,
        "state": "TRAIN_BASE_BANK_COMPLETE",
        "fit_split": Split.TRAIN.value,
        "physical_specimens": len(assignments),
        "tasks": [task.value for task in Task],
        "base_state_count": len(base_rows),
        "bc_state_count": len(bc_rows),
        "bc_target_space": "ALL_LEGAL_CELLS",
        "stop_state_count": len(stop_rows),
        "candidate_suffix_count": sum(row["candidate_count"] for row in manifest_rows),
        "logical_transition_count": transition_count,
        "logical_transition_cap": cap,
        "cpu_workers": int(training["cpu_workers"]),
        "aggregation_rounds_complete": 0,
        "bank_sha256": _file_sha256(bank_path),
        "elapsed_seconds": time.perf_counter() - started,
        "records": manifest_rows,
        "test_opened": False,
    }
    write_json_atomic(output / "training_bank_manifest.json", manifest)
    return {
        "stage": "W3_BASE_BANK_COMPLETE",
        "states": len(base_rows),
        "candidate_suffixes": manifest["candidate_suffix_count"],
        "logical_transitions": transition_count,
        "bank_sha256": manifest["bank_sha256"],
    }


def train_models(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    started = time.perf_counter()
    config, context = _load(config_path, project_root, source_root)
    output = _output_root(config)
    _require_files(
        output,
        ("training_bank_manifest.json", "surface_percepts.jsonl"),
    )
    bank_path = output / "_work/training_bank.pt"
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    if bank.get("config_sha256") != config.config_sha256:
        raise RuntimeError("training bank config identity changed")
    bank["stop"] = list(bank["stop"][: len(bank["base_ctg"])])
    bank.pop("aggregate_ctg", None)
    bank.pop("aggregate_stop", None)
    original_bank_sha256 = _file_sha256(bank_path)
    normalized_bc_rows = []
    bc_targets_changed = False
    for row in bank["base_bc"]:
        queried_cells = tuple(int(cell) for cell in row["queried_cells"])
        probabilities = np.asarray(row["target_probabilities"], dtype=np.float32)
        if (
            probabilities.shape != (len(queried_cells),)
            or np.count_nonzero(probabilities > 1e-8) != 1
        ):
            raise RuntimeError("stored behavior-cloning target is invalid")
        chosen_cell = queried_cells[int(np.argmax(probabilities))]
        complete_cells, complete_probabilities = behavior_cloning_targets(
            np.asarray(row["legal_mask"], dtype=np.bool_), chosen_cell
        )
        normalized = dict(row)
        normalized["queried_cells"] = complete_cells
        normalized["target_probabilities"] = complete_probabilities
        normalized["main_costs"] = np.zeros(len(complete_cells), dtype=np.float64)
        normalized["auxiliary_costs"] = np.zeros(
            len(complete_cells), dtype=np.float64
        )
        normalized_bc_rows.append(normalized)
        bc_targets_changed |= queried_cells != complete_cells or not np.array_equal(
            probabilities, complete_probabilities
        )
    bank["base_bc"] = normalized_bc_rows
    if bc_targets_changed:
        _torch_save_atomic(bank, bank_path)
        bank_manifest_path = output / "training_bank_manifest.json"
        bank_manifest = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
        bank_manifest["bc_target_space"] = "ALL_LEGAL_CELLS"
        bank_manifest["bc_target_migration"] = {
            "source": "RECORDED_RULE_ACTION_AND_LEGAL_MASK",
            "row_count": len(normalized_bc_rows),
            "pre_migration_bank_sha256": original_bank_sha256,
        }
        bank_manifest["bank_sha256"] = _file_sha256(bank_path)
        write_json_atomic(bank_manifest_path, bank_manifest)
    bc_examples = tuple(_policy_example(row) for row in bank["base_bc"])
    ctg_examples = tuple(_policy_example(row) for row in bank["base_ctg"])
    training = config.values["training"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda:0":
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(0)
    overfit_subset = ctg_examples[: min(32, len(ctg_examples))]
    torch.manual_seed(int(training["seed"]))
    overfit_model = LearnedCellActor()
    overfit = fit_actor(
        overfit_model,
        overfit_subset,
        route=TrainingRoute.COST_TO_GO,
        max_steps=64,
        batch_size=min(16, len(overfit_subset)),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        seed=int(training["seed"]),
        device=device,
    )
    if overfit.final_loss >= overfit.initial_loss:
        raise RuntimeError("fixed small-set fit check did not lower loss")
    models: dict[str, torch.nn.Module] = {}
    fits: dict[str, Any] = {}
    logs: list[dict[str, object]] = []
    for method, route, examples in (
        ("L_BC", TrainingRoute.BEHAVIOR_CLONING, bc_examples),
        ("L_CTG_0", TrainingRoute.COST_TO_GO, ctg_examples),
    ):
        fit_examples, valid_examples = _internal_fit_split(examples)
        torch.manual_seed(int(training["seed"]))
        model = LearnedCellActor()
        fit = fit_actor(
            model,
            fit_examples,
            route=route,
            max_steps=int(training["max_optimizer_steps"]),
            batch_size=min(32, len(fit_examples)),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            gradient_clip=float(training["gradient_clip"]),
            seed=int(training["seed"]),
            device=device,
            validation_examples=valid_examples,
            validation_interval=int(training["validation_interval"]),
            patience=int(training["validation_patience"]),
        )
        models[method] = model
        fits[method] = fit
        logs.extend(_fit_log_rows(method, int(training["seed"]), fit))
        print(
            f"trained {method} steps={fit.optimizer_steps} "
            f"loss={fit.initial_loss:.6f}->{fit.final_loss:.6f}",
            flush=True,
        )
    percepts = _load_percepts(config, context, output)
    aggregate_rows, aggregate_stop, aggregate_transitions = _aggregate_states(
        config=config,
        context=context,
        percepts=percepts,
        actor=models["L_CTG_0"],
        device=device,
    )
    bank["aggregate_ctg"] = aggregate_rows
    bank["aggregate_stop"] = aggregate_stop
    bank["stop"].extend(aggregate_stop)
    _torch_save_atomic(bank, bank_path)
    all_ctg = tuple(
        _policy_example(row) for row in (*bank["base_ctg"], *aggregate_rows)
    )
    fit_examples, valid_examples = _internal_fit_split(all_ctg)
    torch.manual_seed(int(training["seed"]))
    ctg1 = LearnedCellActor()
    ctg1_fit = fit_actor(
        ctg1,
        fit_examples,
        route=TrainingRoute.COST_TO_GO,
        max_steps=int(training["max_optimizer_steps"]),
        batch_size=min(32, len(fit_examples)),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        seed=int(training["seed"]),
        device=device,
        validation_examples=valid_examples,
        validation_interval=int(training["validation_interval"]),
        patience=int(training["validation_patience"]),
    )
    models["L_CTG_1"] = ctg1
    fits["L_CTG_1"] = ctg1_fit
    logs.extend(_fit_log_rows("L_CTG_1", int(training["seed"]), ctg1_fit))
    stop_all = tuple(_stop_example(row) for row in bank["stop"])
    stop_fit_examples, stop_valid_examples = _internal_stop_split(stop_all)
    torch.manual_seed(int(training["seed"]))
    stop_model = LearnedStopHead()
    stop_fit = fit_stop_head(
        stop_model,
        stop_fit_examples,
        max_steps=int(training["max_optimizer_steps"]),
        batch_size=min(32, len(stop_fit_examples)),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        seed=int(training["seed"]),
        device=device,
        validation_examples=stop_valid_examples,
        validation_interval=int(training["validation_interval"]),
        patience=int(training["validation_patience"]),
    )
    models["S_LEARN"] = stop_model
    fits["S_LEARN"] = stop_fit
    logs.extend(_fit_log_rows("S_LEARN", int(training["seed"]), stop_fit))
    aggregate_candidate_suffixes = sum(
        len(row["queried_cells"]) for row in aggregate_rows
    )
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_rows = []
    for method in MODEL_METHODS:
        path = model_dir / f"{method.lower()}.pt"
        _save_model(
            models[method],
            path,
            method=method,
            seed=int(training["seed"]),
            config_sha256=config.config_sha256,
        )
        fit = fits[method]
        model_rows.append(
            {
                "method": method,
                "seed": int(training["seed"]),
                "path": path.relative_to(output).as_posix(),
                "sha256": _file_sha256(path),
                "parameter_count": sum(
                    parameter.numel() for parameter in models[method].parameters()
                ),
                "use_surface_features": bool(
                    getattr(models[method], "use_surface_features", True)
                ),
                "optimizer_steps": fit.optimizer_steps,
                "initial_loss": fit.initial_loss,
                "final_loss": fit.final_loss,
                "best_valid_loss": fit.best_valid_loss,
                "stopped_early": fit.stopped_early,
            }
        )
    log_fields = (
        "schema_version",
        "method",
        "seed",
        "optimizer_step",
        "train_loss",
        "valid_loss",
    )
    write_csv_atomic(output / "training_log.csv", tuple(logs), log_fields)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(0))
        if device == "cuda:0"
        else 0
    )
    manifest = {
        "schema_version": 1,
        "state": "ACTORS_AND_STOP_TRAINED",
        "device": device,
        "models": model_rows,
        "overfit_check": {
            "state_count": len(overfit_subset),
            "optimizer_steps": overfit.optimizer_steps,
            "initial_loss": overfit.initial_loss,
            "final_loss": overfit.final_loss,
            "passed": overfit.final_loss < overfit.initial_loss,
        },
        "training_bank": {
            "base_ctg_states": len(ctg_examples),
            "aggregate_ctg_states": len(aggregate_rows),
            "aggregate_candidate_suffixes": aggregate_candidate_suffixes,
            "aggregation_rounds": 1,
            "aggregation_transitions": aggregate_transitions,
            "bank_sha256": _file_sha256(bank_path),
        },
        "peak_gpu_memory_bytes": peak_memory,
        "elapsed_seconds": time.perf_counter() - started,
        "test_opened": False,
    }
    write_json_atomic(output / "model_manifest.json", manifest)
    bank_manifest_path = output / "training_bank_manifest.json"
    bank_manifest = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    bank_manifest["aggregation_rounds_complete"] = 1
    bank_manifest["aggregation_state_count"] = len(aggregate_rows)
    bank_manifest["aggregation_candidate_suffix_count"] = (
        aggregate_candidate_suffixes
    )
    bank_manifest["aggregation_transition_count"] = aggregate_transitions
    bank_manifest["logical_transition_count"] += aggregate_transitions
    bank_manifest["bank_sha256"] = _file_sha256(bank_path)
    if bank_manifest["logical_transition_count"] > int(training["logical_transition_cap"]):
        raise RuntimeError("training transition cap exceeded")
    write_json_atomic(bank_manifest_path, bank_manifest)
    return {
        "stage": "W3_TRAINING_COMPLETE",
        "device": device,
        "model_count": len(model_rows),
        "models": model_rows,
        "aggregation_states": len(aggregate_rows),
        "aggregation_candidate_suffixes": aggregate_candidate_suffixes,
        "elapsed_seconds": manifest["elapsed_seconds"],
    }


def _train_validation_confirmations(
    *,
    config: StudyConfig,
    output: Path,
    selected_method: str,
    device: str,
) -> tuple[tuple[str, str, str, int, int], ...]:
    bank = torch.load(
        output / "_work/training_bank.pt", map_location="cpu", weights_only=False
    )
    rows = list(bank["base_ctg"])
    if selected_method == "L_CTG_1":
        rows.extend(bank.get("aggregate_ctg", ()))
    elif selected_method != "L_CTG_0":
        raise RuntimeError("selected learned checkpoint is invalid")
    examples = tuple(_policy_example(row) for row in rows)
    fit_examples, valid_examples = _internal_fit_split(examples)
    training = config.values["training"]
    trained: list[tuple[str, str, str, int, int]] = []
    model_rows = []
    log_rows = []
    specifications = (
        (selected_method, 2, True),
        (selected_method, 3, True),
        ("L_NO_VLM", 1, False),
    )
    for method, seed, use_surface_features in specifications:
        torch.manual_seed(seed)
        model = LearnedCellActor(use_surface_features=use_surface_features)
        fit = fit_actor(
            model,
            fit_examples,
            route=TrainingRoute.COST_TO_GO,
            max_steps=int(training["max_optimizer_steps"]),
            batch_size=min(32, len(fit_examples)),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            gradient_clip=float(training["gradient_clip"]),
            seed=seed,
            device=device,
            validation_examples=valid_examples,
            validation_interval=int(training["validation_interval"]),
            patience=int(training["validation_patience"]),
        )
        filename = (
            f"{selected_method.lower()}_seed{seed}.pt"
            if use_surface_features
            else "l_no_vlm.pt"
        )
        path = output / "models" / filename
        _save_model(
            model,
            path,
            method=method,
            seed=seed,
            config_sha256=config.config_sha256,
        )
        model_rows.append(
            {
                "method": method,
                "seed": seed,
                "path": path.relative_to(output).as_posix(),
                "sha256": _file_sha256(path),
                "parameter_count": model.parameter_count,
                "optimizer_steps": fit.optimizer_steps,
                "initial_loss": fit.initial_loss,
                "final_loss": fit.final_loss,
                "best_valid_loss": fit.best_valid_loss,
                "stopped_early": fit.stopped_early,
                "use_surface_features": use_surface_features,
            }
        )
        log_rows.extend(_fit_log_rows(method, seed, fit))
        trained.append(
            (method, "ACTOR", path.relative_to(output).as_posix(), 4, seed)
        )
        print(
            f"trained validation-triggered {method} seed={seed} "
            f"steps={fit.optimizer_steps} loss={fit.initial_loss:.6f}->"
            f"{fit.final_loss:.6f}",
            flush=True,
        )
    log_fields = (
        "schema_version",
        "method",
        "seed",
        "optimizer_step",
        "train_loss",
        "valid_loss",
    )
    with (output / "training_log.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        existing_logs = list(csv.DictReader(handle))
    write_csv_atomic(
        output / "training_log.csv", (*existing_logs, *log_rows), log_fields
    )
    manifest_path = output / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_paths = {row["path"] for row in model_rows}
    manifest["models"] = [
        row for row in manifest["models"] if row["path"] not in new_paths
    ] + model_rows
    manifest["validation_triggered_training"] = {
        "selected_checkpoint": selected_method,
        "confirmation_seeds": [2, 3],
        "no_vlm_seed": 1,
    }
    write_json_atomic(manifest_path, manifest)
    return tuple(trained)


def _evaluate_assignments_parallel(
    *,
    config: StudyConfig,
    context: StudyContext,
    assignments: tuple[SplitAssignment, ...],
    planner_specs: tuple[tuple[str, str, str, int, int], ...],
    learned_threshold: float | None,
    store_trajectory: bool,
    label: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[StopValidationRow]],
]:
    episode_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    stop_rows_by_method: dict[str, list[StopValidationRow]] = {}
    with ProcessPoolExecutor(
        max_workers=min(int(config.values["training"]["cpu_workers"]), len(assignments)),
        mp_context=get_context("spawn"),
        initializer=_initialize_evaluation_worker,
        initargs=(
            str(config.path),
            str(config.project_root),
            str(context.source_root),
            planner_specs,
            learned_threshold,
        ),
    ) as executor:
        futures = tuple(
            executor.submit(
                _evaluate_specimen_worker,
                assignment.record.specimen_key,
                store_trajectory=store_trajectory,
            )
            for assignment in assignments
        )
        for specimen_index, future in enumerate(futures, start=1):
            episodes, trajectories, stop_rows = future.result()
            episode_rows.extend(episodes)
            trajectory_rows.extend(trajectories)
            for method, rows in stop_rows.items():
                stop_rows_by_method.setdefault(method, []).extend(rows)
            print(
                f"{label} {specimen_index}/{len(assignments)} "
                f"episodes={len(episode_rows)} trajectory_rows={len(trajectory_rows)}",
                flush=True,
            )
    return episode_rows, trajectory_rows, stop_rows_by_method


def _initialize_evaluation_worker(
    config_path: str,
    project_root: str,
    source_root: str,
    planner_specs: tuple[tuple[str, str, str, int, int], ...],
    learned_threshold: float | None,
) -> None:
    global _EVALUATION_WORKER_CONTEXT
    global _EVALUATION_WORKER_DEVICE
    global _EVALUATION_WORKER_PERCEPTS
    global _EVALUATION_WORKER_PLANNERS
    global _EVALUATION_WORKER_STOP
    global _EVALUATION_WORKER_THRESHOLD
    config, context = _load(
        Path(config_path), Path(project_root), Path(source_root)
    )
    output = _output_root(config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    planners = []
    for method, kind, reference, period, seed in planner_specs:
        if kind == "RULE":
            planner: RuleMethod | LearnedCellActor = RuleMethod(reference)
        elif kind == "ACTOR":
            planner = _load_actor(output / reference, config, device)
        else:
            raise RuntimeError("evaluation planner specification is invalid")
        planners.append((method, planner, period, seed))
    _EVALUATION_WORKER_CONTEXT = context
    _EVALUATION_WORKER_DEVICE = device
    _EVALUATION_WORKER_PERCEPTS = _load_percepts(config, context, output)
    _EVALUATION_WORKER_PLANNERS = tuple(planners)
    _EVALUATION_WORKER_STOP = _load_stop(output / "models/s_learn.pt", config, device)
    _EVALUATION_WORKER_THRESHOLD = learned_threshold


def _evaluate_specimen_worker(
    specimen_key: str, *, store_trajectory: bool
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[StopValidationRow]],
]:
    if (
        _EVALUATION_WORKER_CONTEXT is None
        or _EVALUATION_WORKER_PERCEPTS is None
        or _EVALUATION_WORKER_STOP is None
        or not _EVALUATION_WORKER_PLANNERS
    ):
        raise RuntimeError("evaluation worker is not initialized")
    context = _EVALUATION_WORKER_CONTEXT
    record = next(
        row for row in context.roster.pilot_records if row.specimen_key == specimen_key
    )
    runtime = open_study_specimen(context, record)
    percept = _EVALUATION_WORKER_PERCEPTS[specimen_key]
    episodes = []
    trajectories = []
    stop_rows_by_method: dict[str, list[StopValidationRow]] = {}
    for task in Task:
        reference = _full_reference(context, runtime, task)
        for method, planner, period, seed in _EVALUATION_WORKER_PLANNERS:
            result, trajectory, stop_rows = _run_planner_episode(
                runtime,
                percept=percept,
                task=task,
                reference=reference,
                context=context,
                method_name=method,
                planner=planner,
                coverage_period=period,
                seed=seed,
                device=_EVALUATION_WORKER_DEVICE,
                stop_model=_EVALUATION_WORKER_STOP,
                learned_threshold=_EVALUATION_WORKER_THRESHOLD,
                store_trajectory=store_trajectory,
            )
            episodes.append(result)
            trajectories.extend(trajectory)
            if not store_trajectory:
                stop_rows_by_method.setdefault(method, []).extend(stop_rows)
    return episodes, trajectories, stop_rows_by_method


def validate_models(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    started = time.perf_counter()
    config, context = _load(config_path, project_root, source_root)
    output = _output_root(config)
    _require_files(output, ("model_manifest.json", "surface_percepts.jsonl"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assignments = tuple(
        row for row in context.roster.assignments if row.split is Split.VALID
    )
    planner_specs = tuple(
        (name, "RULE", method.value, period, 1)
        for name, method, period in RULE_CONFIGS
    ) + tuple(
        (method, "ACTOR", f"models/{method.lower()}.pt", 4, 1)
        for method in ("L_BC", "L_CTG_0", "L_CTG_1")
    )
    episode_rows, _trajectories, stop_rows_by_method = (
        _evaluate_assignments_parallel(
            config=config,
            context=context,
            assignments=assignments,
            planner_specs=planner_specs,
            learned_threshold=None,
            store_trajectory=False,
            label="validation",
        )
    )
    rule_names = tuple(name for name, _method, _period in RULE_CONFIGS)
    rule_selected = max(
        rule_names,
        key=lambda name: (
            _mean_metric(episode_rows, name, "ausc_any"),
            name == "R_BALANCED_P4",
            name,
        ),
    )
    learned_selected = max(
        ("L_CTG_0", "L_CTG_1"),
        key=lambda name: (_mean_metric(episode_rows, name, "ausc_any"), name),
    )
    stop_validation = (
        *stop_rows_by_method[rule_selected],
        *stop_rows_by_method[learned_selected],
    )
    authorization = calibrate_learned_stop(
        stop_validation,
        thresholds=tuple(
            float(value) for value in config.values["stopping"]["learned_thresholds"]
        ),
    )
    rule_score = _mean_metric(episode_rows, rule_selected, "ausc_any")
    learned_score = _mean_metric(episode_rows, learned_selected, "ausc_any")
    positive = learned_score > rule_score + 1e-12
    confirmation_seed_scores = {"1": learned_score}
    if positive:
        confirmation_methods = _train_validation_confirmations(
            config=config,
            output=output,
            selected_method=learned_selected,
            device=device,
        )
        planner_specs += confirmation_methods
        confirmation_rows, _trajectories, _stop_rows = (
            _evaluate_assignments_parallel(
                config=config,
                context=context,
                assignments=assignments,
                planner_specs=confirmation_methods,
                learned_threshold=None,
                store_trajectory=False,
                label="validation confirmation",
            )
        )
        episode_rows.extend(confirmation_rows)
        confirmation_seed_scores.update(
            {
                str(seed): _mean_metric(
                    [row for row in episode_rows if int(row["seed"]) == seed],
                    learned_selected,
                    "ausc_any",
                )
                for seed in (2, 3)
            }
        )
        learned_score = _mean_metric(episode_rows, learned_selected, "ausc_any")
    summaries = {
        name: {
            task.value: _mean_metric(
                episode_rows, name, "ausc_any", task=task
            )
            for task in Task
        }
        | {"overall": _mean_metric(episode_rows, name, "ausc_any")}
        for name in dict.fromkeys(row[0] for row in planner_specs)
    }
    selected_rule_spec = next(row for row in planner_specs if row[0] == rule_selected)
    selection = {
        "schema_version": 1,
        "state": "VALID_SELECTION_LOCKED",
        "config_sha256": config.config_sha256,
        "validation_physical_specimens": len(assignments),
        "validation_episode_count": len(episode_rows),
        "selected_rule": {
            "method": rule_selected,
            "coverage_period": selected_rule_spec[3],
            "mean_ausc_any": rule_score,
        },
        "selected_learned": {
            "method": learned_selected,
            "seeds": [1, 2, 3] if positive else [1],
            "mean_ausc_any": learned_score,
        },
        "validation_action_signal": {
            "difference": learned_score - rule_score,
            "positive_initial_seed": positive,
            "positive_after_confirmation": learned_score > rule_score + 1e-12,
        },
        "method_metrics": summaries,
        "learned_stop": _authorization_payload(authorization),
        "confirmation_seeds": {
            "run": positive,
            "reason": (
                "RUN_VALID_POSITIVE_SIGNAL"
                if positive
                else "NOT_RUN_VALID_SIGNAL_NOT_POSITIVE"
            ),
            "seeds": [1, 2, 3] if positive else [1],
            "mean_ausc_any_by_seed": confirmation_seed_scores,
        },
        "l_no_vlm": {
            "run": positive,
            "reason": (
                "RUN_VALID_POSITIVE_SIGNAL"
                if positive
                else "NOT_RUN_VALID_SIGNAL_NOT_POSITIVE"
            ),
            "seed": 1 if positive else None,
            "mean_ausc_any": (
                _mean_metric(episode_rows, "L_NO_VLM", "ausc_any")
                if positive
                else None
            ),
        },
        "formal_effect": None,
        "reference": "PROXY_ONLY",
        "test_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    selection["selection_sha256"] = _json_sha(selection)
    write_json_atomic(output / "validation_selection.json", selection)
    return {
        "stage": selection["state"],
        "selected_rule": rule_selected,
        "selected_learned": learned_selected,
        "learned_minus_rule_ausc": learned_score - rule_score,
        "learned_stop": authorization.status.value,
        "validation_specimens": len(assignments),
    }


def evaluate_study(
    *, config_path: Path, project_root: Path, source_root: Path, split: str
) -> dict[str, object]:
    if split != "test":
        raise ValueError("only the explicit untouched TEST split may be evaluated")
    started = time.perf_counter()
    config, context = _load(config_path, project_root, source_root)
    output = _output_root(config)
    _require_files(
        output,
        (
            "validation_selection.json",
            "model_manifest.json",
            "surface_percepts.jsonl",
        ),
    )
    if (output / "_work/test_evaluation_complete.json").is_file():
        metadata = json.loads(
            (output / "_work/test_evaluation_complete.json").read_text(
                encoding="utf-8"
            )
        )
        return metadata | {"idempotent_reuse": True}
    selection = json.loads(
        (output / "validation_selection.json").read_text(encoding="utf-8")
    )
    stored_selection_sha = selection.pop("selection_sha256")
    if stored_selection_sha != _json_sha(selection):
        raise RuntimeError("VALID selection identity changed")
    selection["selection_sha256"] = stored_selection_sha
    if selection["state"] != "VALID_SELECTION_LOCKED" or selection["test_opened"]:
        raise RuntimeError("TEST evaluation is blocked until VALID choices are locked")
    selected_learned = selection["selected_learned"]["method"]
    selected_seeds = tuple(int(seed) for seed in selection["selected_learned"]["seeds"])
    stop_threshold = (
        float(selection["learned_stop"]["threshold"])
        if selection["learned_stop"]["status"]
        == LearnedStopStatus.AUTHORIZED.value
        else None
    )
    balanced_name = max(
        ("R_BALANCED_P4", "R_BALANCED_P8"),
        key=lambda name: selection["method_metrics"][name]["overall"],
    )
    balanced_period = 4 if balanced_name.endswith("P4") else 8
    planner_specs = (
        ("R_GEOM", "RULE", RuleMethod.R_GEOM.value, 4, 1),
        ("R_CENTER", "RULE", RuleMethod.R_CENTER.value, 4, 1),
        ("R_VLM_OPEN", "RULE", RuleMethod.R_VLM_OPEN.value, 4, 1),
        ("R_LEGACY", "RULE", RuleMethod.R_LEGACY.value, 4, 1),
        (
            "R_BALANCED",
            "RULE",
            RuleMethod.R_BALANCED.value,
            balanced_period,
            1,
        ),
        ("L_BC", "ACTOR", "models/l_bc.pt", 4, 1),
    ) + tuple(
        (
            "L_CTG",
            "ACTOR",
            (
                f"models/{selected_learned.lower()}.pt"
                if seed == 1
                else f"models/{selected_learned.lower()}_seed{seed}.pt"
            ),
            4,
            seed,
        )
        for seed in selected_seeds
    )
    if selection["l_no_vlm"]["run"]:
        planner_specs += (
            (
                "L_NO_VLM",
                "ACTOR",
                "models/l_no_vlm.pt",
                4,
                int(selection["l_no_vlm"]["seed"]),
            ),
        )
    assignments = tuple(
        row for row in context.roster.assignments if row.split is Split.TEST
    )
    episode_rows, trajectory_rows, _stop_rows = _evaluate_assignments_parallel(
        config=config,
        context=context,
        assignments=assignments,
        planner_specs=planner_specs,
        learned_threshold=stop_threshold,
        store_trajectory=True,
        label="TEST evaluation",
    )
    episode_fields = tuple(episode_rows[0])
    write_csv_atomic(
        output / "per_episode_metrics.csv", tuple(episode_rows), episode_fields
    )
    write_parquet_atomic(output / "trajectories.parquet", tuple(trajectory_rows))
    selected_rule_validation = selection["selected_rule"]["method"]
    selected_rule = (
        "R_BALANCED"
        if selected_rule_validation.startswith("R_BALANCED")
        else selected_rule_validation
    )
    comparisons = _comparison_rows(
        episode_rows,
        treatment="L_CTG",
        comparators=(
            selected_rule,
            "R_BALANCED",
            "L_BC",
            *(("L_NO_VLM",) if selection["l_no_vlm"]["run"] else ()),
        ),
        replicates=int(config.values["evaluation"]["bootstrap_replicates"]),
        seed=int(config.values["evaluation"]["bootstrap_seed"]),
    )
    comparison_fields = (
        "schema_version",
        "task",
        "metric",
        "treatment",
        "comparator",
        "estimate",
        "ci_lower",
        "ci_upper",
        "physical_specimen_count",
        "domain_count",
        "bootstrap_replicates",
        "proxy_scope",
        "formal_estimate",
    )
    write_csv_atomic(output / "comparisons.csv", comparisons, comparison_fields)
    failures = _failure_rows(episode_rows)
    failure_fields = (
        "schema_version",
        "dataset_id",
        "specimen_id",
        "specimen_key",
        "task",
        "method",
        "seed",
        "evaluation_mode",
        "failure_type",
        "terminal_cost",
    )
    write_csv_atomic(output / "failure_cases.csv", failures, failure_fields)
    metadata = {
        "stage": "W5_TEST_EVALUATION_COMPLETE",
        "test_physical_specimens": len(assignments),
        "episode_count": len(episode_rows),
        "trajectory_row_count": len(trajectory_rows),
        "method_count": len({row[0] for row in planner_specs}),
        "selected_rule": selected_rule,
        "selected_learned_checkpoint": selected_learned,
        "selected_learned_seeds": list(selected_seeds),
        "learned_stop_status": selection["learned_stop"]["status"],
        "elapsed_seconds": time.perf_counter() - started,
        "selection_sha256": stored_selection_sha,
        "formal_effect": None,
    }
    write_json_atomic(output / "_work/test_evaluation_complete.json", metadata)
    return metadata


def summarize_study(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    config, _context = _load(config_path, project_root, source_root)
    output = _output_root(config)
    _require_files(
        output,
        (
            "inventory.json",
            "perception_manifest.json",
            "training_bank_manifest.json",
            "model_manifest.json",
            "validation_selection.json",
            "per_episode_metrics.csv",
            "trajectories.parquet",
            "comparisons.csv",
            "failure_cases.csv",
        ),
    )
    with (output / "per_episode_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        episodes = tuple(csv.DictReader(handle))
    with (output / "comparisons.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        comparisons = tuple(csv.DictReader(handle))
    inventory = json.loads((output / "inventory.json").read_text(encoding="utf-8"))
    perception = json.loads(
        (output / "perception_manifest.json").read_text(encoding="utf-8")
    )
    bank = json.loads(
        (output / "training_bank_manifest.json").read_text(encoding="utf-8")
    )
    models = json.loads(
        (output / "model_manifest.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (output / "validation_selection.json").read_text(encoding="utf-8")
    )
    method_metrics = {}
    for method in sorted({row["method"] for row in episodes}):
        method_metrics[method] = {}
        for task in Task:
            rows = [
                row
                for row in episodes
                if row["method"] == method and row["task"] == task.value
            ]
            method_metrics[method][task.value] = {
                "physical_specimens": len({row["specimen_key"] for row in rows}),
                "ausc_any": float(np.mean([float(row["ausc_any"]) for row in rows])),
                "success_c010": float(
                    np.mean([_csv_bool(row["success_c010"]) for row in rows])
                ),
                "success_c040": float(
                    np.mean([_csv_bool(row["success_c040"]) for row in rows])
                ),
                "full_input_success": float(
                    np.mean([_csv_bool(row["success_c100"]) for row in rows])
                ),
                "rule_stop_success": float(
                    np.mean([_csv_bool(row["rule_stop_success"]) for row in rows])
                ),
                "learned_stop_success": float(
                    np.mean([_csv_bool(row["learned_stop_success"]) for row in rows])
                ),
                "mean_route_cost": float(
                    np.mean([float(row["route_cost"]) for row in rows])
                ),
                "mean_inference_seconds": float(
                    np.mean([float(row["inference_seconds"]) for row in rows])
                ),
            }
    primary = [
        row
        for row in comparisons
        if row["treatment"] == "L_CTG"
        and row["comparator"]
        == (
            "R_BALANCED"
            if selection["selected_rule"]["method"].startswith("R_BALANCED")
            else selection["selected_rule"]["method"]
        )
        and row["metric"] == "ausc_any"
    ]
    proxy_supported = bool(
        len(primary) == len(Task)
        and all(float(row["ci_lower"]) > 0.0 for row in primary)
    )
    vlm_rows = [
        row
        for row in comparisons
        if row["treatment"] == "L_CTG"
        and row["comparator"] == "L_NO_VLM"
        and row["metric"] == "ausc_any"
    ]
    if not vlm_rows:
        vlm_increment = "NOT_TESTED"
    elif all(float(row["ci_lower"]) > 0.0 for row in vlm_rows):
        vlm_increment = "SUPPORTED"
    elif all(float(row["ci_upper"]) < 0.0 for row in vlm_rows):
        vlm_increment = "NOT_SUPPORTED"
    else:
        vlm_increment = "INCONCLUSIVE"
    not_run = {
        "all_276": "RESOURCE_SCOPED_PILOT",
        "leave_domain_out": "RESOURCE_SCOPED_PILOT",
        "new_vlm_models": "FROZEN_SINGLE_BACKEND",
        "l_open_init": "NOT_RUN_RESOURCE_SCOPED",
    }
    if not selection["l_no_vlm"]["run"]:
        not_run["l_no_vlm"] = selection["l_no_vlm"]["reason"]
    if not selection["confirmation_seeds"]["run"]:
        not_run["extra_policy_seeds"] = selection["confirmation_seeds"]["reason"]
    summary = {
        "schema_version": 1,
        "execution": "TRAINED_AND_EVALUATED",
        "model_trained": True,
        "reference": "PROXY_ONLY",
        "formal_planner_effect": "INCONCLUSIVE",
        "formal_effect": None,
        "proxy_planner_effect": "SUPPORTED" if proxy_supported else "NOT_SUPPORTED",
        "stop_effect": (
            "NOT_AUTHORIZED"
            if selection["learned_stop"]["status"]
            == LearnedStopStatus.NOT_AUTHORIZED.value
            else "INCONCLUSIVE"
        ),
        "vlm_increment": vlm_increment,
        "repository_base_sha": config.values["repository_base_sha"],
        "branch": "research/learned-cscan-same-perception",
        "evidence_scope": config.values["cohort"]["evidence_scope"],
        "data_roster": {
            "pilot": inventory["pilot_specimens"],
            "split_counts": inventory["split_counts"],
            "domains": len(config.domain_order),
        },
        "surface_backend": {
            "repository": perception["model_repository"],
            "revision": perception["model_revision"],
            "prompt_sha256": perception["prompt_sha256"],
            "schema_version": perception["schema_version_surface_percept"],
        },
        "surface_perception": perception["execution"],
        "reader_status": perception["reader_v2"]["status"],
        "training": {
            "models": models["models"],
            "base_bank_states": bank["base_state_count"],
            "base_candidate_suffixes": bank["candidate_suffix_count"],
            "aggregation_states": bank["aggregation_state_count"],
            "aggregation_candidate_suffixes": bank[
                "aggregation_candidate_suffix_count"
            ],
            "aggregation_rounds": bank["aggregation_rounds_complete"],
            "logical_transitions": bank["logical_transition_count"],
        },
        "validation_selection": selection,
        "test": {
            "physical_specimens": len({row["specimen_key"] for row in episodes}),
            "episode_count": len(episodes),
            "method_metrics": method_metrics,
            "primary_proxy_comparisons": primary,
        },
        "proxy_effect": primary,
        "resource_use": {
            "training_logical_transitions": bank["logical_transition_count"],
            "validation_action_transitions": selection[
                "validation_episode_count"
            ]
            * 192,
            "test_action_transitions": sum(int(row["action_count"]) for row in episodes),
            "total_logical_transitions": bank["logical_transition_count"]
            + selection["validation_episode_count"] * 192
            + sum(int(row["action_count"]) for row in episodes),
            "logical_transition_cap": config.values["training"][
                "logical_transition_cap"
            ],
        },
        "claims": {
            "learned_planning_supported_proxy": proxy_supported,
            "learned_planning_supported_formal": None,
            "vlm_increment_supported": (
                True
                if vlm_increment == "SUPPORTED"
                else False if vlm_increment == "NOT_SUPPORTED" else None
            ),
            "stop_increment_supported": None,
        },
        "limitations": [
            "All task outcomes are same-reader self-consistency proxies; none is independently reviewed.",
            "The retrospective 60-specimen cohort is reused and is not a prospective scanner or robot experiment.",
            "The pilot TEST split has 24 physical specimens and does not support new-material transfer claims.",
            "Route values are normalized image-plane diagnostics, not physical travel time or path optimality.",
        ],
        "not_run": not_run,
    }
    write_json_atomic(output / "summary.json", summary)
    write_checksums(output, output / "CHECKSUMS.sha256")
    verified = verify_checksums(output, output / "CHECKSUMS.sha256")
    return {
        "stage": "W6_SUMMARY_COMPLETE",
        "execution": summary["execution"],
        "reference": summary["reference"],
        "proxy_planner_effect": summary["proxy_planner_effect"],
        "formal_effect": None,
        "checksum_files": len(verified),
    }


def _load(
    config_path: Path, project_root: Path, source_root: Path
) -> tuple[StudyConfig, StudyContext]:
    config = load_study_config(config_path, project_root=project_root)
    context = load_study_context(config, source_root=source_root)
    return config, context


def _output_root(config: StudyConfig) -> Path:
    relative = Path(config.values["outputs"]["root"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("study output root is invalid")
    return config.project_root / relative


def _split_row(assignment: SplitAssignment) -> dict[str, object]:
    record = assignment.record
    return {
        "schema_version": 1,
        "dataset_id": record.dataset_id,
        "specimen_id": record.specimen_id,
        "specimen_key": record.specimen_key,
        "split": assignment.split.value,
        "domain_rank": assignment.domain_rank,
        "surface_sha256": record.surface_sha256,
        "registered_cscan_sha256": record.cscan_sha256,
        "native_height": record.native_shape[0],
        "native_width": record.native_shape[1],
        "transform_sha256": record.transform_sha256,
    }


def _reference_row(assignment: SplitAssignment) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": assignment.record.dataset_id,
        "specimen_id": assignment.record.specimen_id,
        "specimen_key": assignment.record.specimen_key,
        "split": assignment.split.value,
        "reference_type": "ALGORITHM_DERIVED_NOT_REVIEWED",
        "review_state": "pending",
        "reviewer_alias": "",
        "formal_eligible": False,
        "proxy_scope": "SAME_READER_SELF_CONSISTENCY",
        "formal_success_available": False,
    }


def _surface_backend(config: StudyConfig) -> QwenVLBackend:
    model = config.values["surface_model"]
    return QwenVLBackend(
        model["local_path"],
        device=model["device"],
        dtype=model["dtype"],
        max_new_tokens=int(model["max_new_tokens"]),
    )


def _surface_request(config: StudyConfig, render: Any) -> SurfacePerceptRequest:
    return SurfacePerceptRequest(
        model_revision=config.values["surface_model"]["revision"],
        clean_image_sha256=render.clean_sha256,
        gridded_image_sha256=render.gridded_sha256,
        render_version=_render_version(config),
        prompt_sha256=_surface_prompt_sha256(),
        schema_version=2,
    )


def _render_version(config: StudyConfig) -> str:
    return str(config.legacy_config.values["surface"]["preprocessing_sha256"])


def _surface_prompt_sha256() -> str:
    return hashlib.sha256(
        (
            f"{SURFACE_PERCEPT_PROMPT}\n---FORMAT_REPAIR---\n"
            f"{FORMAT_REPAIR_PROMPT}\n{FORMAT_REPAIR_CONTEXT_PREFIX}"
            f"{{RAW_RESPONSE}}{FORMAT_REPAIR_CONTEXT_SUFFIX}"
        ).encode()
    ).hexdigest()


def _perception_diagnostics(
    *,
    config: StudyConfig,
    context: StudyContext,
    backend: QwenVLBackend,
    originals: dict[str, SurfacePercept],
    output: Path,
) -> dict[str, object]:
    cache = SurfacePerceptCache(output / "_work/diagnostic_percepts.jsonl")
    preflight_calls = 24
    rows: list[dict[str, object]] = [
        {
            "kind": "PREFLIGHT_SCHEMA_DEBUG",
            "actual_calls": preflight_calls,
            "finding": (
                "Cross-region cell-count and nonempty-string constraints were absent "
                "from the machine-readable repair schema; both were fixed before the "
                "final deployment. A final diagnostic showed overlapping cells across "
                "regions, which are schema-valid and collapse without duplicate weight "
                "in the downstream per-cell features."
            ),
        }
    ]
    actual_calls = 0
    deployment_calls = 0
    checked_domains: tuple[str, ...] = ()
    for domain in checked_domains:
        assignment = next(
            row
            for row in context.roster.assignments
            if row.split is Split.TRAIN and row.record.dataset_id == domain
        )
        registered = render_registered_surface(
            assignment.record,
            max_edge=int(config.legacy_config.values["surface"]["max_edge"]),
        )
        permuted = _permuted_grid(registered.render.clean)
        request = SurfacePerceptRequest(
            model_revision=config.values["surface_model"]["revision"],
            clean_image_sha256=registered.render.clean_sha256,
            gridded_image_sha256=_image_sha256(permuted),
            render_version=f"{_render_version(config)}:PERMUTED_17X_PLUS_13",
            prompt_sha256=_surface_prompt_sha256(),
            schema_version=2,
        )
        result = cache.resolve(
            request,
            lambda prompt, clean=registered.render.clean, grid=permuted: backend.infer(
                (clean, grid), prompt
            ),
        )
        actual_calls += result.actual_call_count
        deployment_calls += result.original_call_count
        physical = map_percept_to_physical(result.percept)
        original_cells = sorted(
            cell for region in originals[assignment.record.specimen_key].regions for cell in region.cells
        )
        permuted_cells = sorted(cell for region in physical.regions for cell in region.cells)
        rows.append(
            {
                "kind": "DISPLAY_NUMBER_PERMUTATION",
                "domain": domain,
                "specimen_key": assignment.record.specimen_key,
                "cache_hit": result.cache_hit,
                "call_count": result.original_call_count,
                "original_cells": original_cells,
                "mapped_permuted_cells": permuted_cells,
                "cell_jaccard": _set_jaccard(original_cells, permuted_cells),
                "original_no_reliable_cue": originals[
                    assignment.record.specimen_key
                ].no_reliable_cue,
                "permuted_no_reliable_cue": physical.no_reliable_cue,
            }
        )
    rows.append(
        {
            "kind": "BLANK_SURFACE_NOT_RUN",
            "reason": "DIAGNOSTIC_CALL_CAP_RESERVED_FOR_FINAL_MAIN_DEPLOYMENT",
        }
    )
    rows.extend(
        {
            "kind": "DISPLAY_NUMBER_PERMUTATION_NOT_RUN",
            "domain": domain,
            "reason": "DIAGNOSTIC_CALL_CAP_RESERVED_FOR_FINAL_MAIN_DEPLOYMENT",
        }
        for domain in config.domain_order
        if domain not in checked_domains
    )
    cap = int(config.values["surface_model"]["diagnostic_extra_call_cap"])
    cumulative_calls = preflight_calls + deployment_calls
    if cumulative_calls > cap:
        raise RuntimeError("surface diagnostic-call cap exceeded")
    return {
        "actual_calls_this_command": actual_calls,
        "recorded_preflight_debug_calls": preflight_calls,
        "diagnostic_deployment_calls": deployment_calls,
        "cumulative_diagnostic_calls": cumulative_calls,
        "call_cap": cap,
        "records": rows,
    }


def _permuted_grid(clean: Image.Image) -> Image.Image:
    gridded = clean.copy()
    draw = ImageDraw.Draw(gridded, "RGBA")
    width, height = gridded.size
    line_width = max(1, round(max(width, height) / 512))
    for index in range(1, 8):
        x = round(index * width / 8)
        y = round(index * height / 8)
        draw.line((x, 0, x, height - 1), fill=(0, 255, 255, 110), width=line_width)
        draw.line((0, y, width - 1, y), fill=(0, 255, 255, 110), width=line_width)
    for row in range(8):
        for column in range(8):
            physical = row * 8 + column
            draw.text(
                (round(column * width / 8) + 2, round(row * height / 8) + 1),
                str(display_cell_id(physical)),
                fill=(255, 255, 255, 230),
                stroke_width=1,
                stroke_fill=(0, 0, 0, 220),
            )
    return gridded


def _full_input_readout_validation(
    context: StudyContext,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    split_by_key = {
        row.record.specimen_key: row.split for row in context.roster.assignments
    }
    threshold = float(context.config.values["reader"]["distance_threshold"])
    for record in sorted(context.roster.pilot_records, key=lambda row: row.specimen_key):
        runtime = open_study_specimen(context, record)
        full_scan = context.authority.source_teacher_view(record.specimen_id).full_scan
        positions = np.argwhere(np.ones(record.native_shape, dtype=np.bool_))
        readout = read_visible_evidence(
            grid=runtime.grid,
            positions=positions,
            values=full_scan[positions[:, 0], positions[:, 1]],
            cell_levels=(2,) * 64,
            prior=context.background_prior,
            distance_threshold=threshold,
        )
        for task in Task:
            report = build_task_report_v2(readout, task=task)
            count = int(np.count_nonzero(report.predicted_mask))
            rows.append(
                {
                    "schema_version": 1,
                    "dataset_id": record.dataset_id,
                    "specimen_id": record.specimen_id,
                    "specimen_key": record.specimen_key,
                    "split": split_by_key[record.specimen_key].value,
                    "task": task.value,
                    "native_count": int(report.predicted_mask.size),
                    "candidate_pixels": count,
                    "candidate_fraction": count / report.predicted_mask.size,
                    "candidate_cells": len(report.candidate_cells),
                    "support_count": len(report.support_positions),
                    "signal_strength": report.signal_strength,
                    "nondegenerate": bool(0 < count < report.predicted_mask.size),
                }
            )
    return tuple(rows)


def _load_percepts(
    config: StudyConfig, context: StudyContext, output: Path
) -> dict[str, SurfacePercept]:
    cache = SurfacePerceptCache(output / "surface_percepts.jsonl")
    percepts = {}
    for record in context.roster.pilot_records:
        registered = render_registered_surface(
            record,
            max_edge=int(config.legacy_config.values["surface"]["max_edge"]),
        )
        result = cache.get(_surface_request(config, registered.render))
        if result is None:
            raise RuntimeError(f"surface percept is missing: {record.specimen_key}")
        percepts[record.specimen_key] = result.percept
    return percepts


def _packet(
    observation: Any,
    *,
    runtime: StudySpecimenRuntime,
    percept: SurfacePercept,
    task: Task,
    context: StudyContext,
    probe_position: tuple[float, float],
    route_cost: float,
    include_actor_subblocks: bool = True,
) -> ObservationPacket:
    return build_observation_packet(
        observation,
        grid=runtime.grid,
        percept=percept,
        task=task,
        prior=context.background_prior,
        distance_threshold=float(context.config.values["reader"]["distance_threshold"]),
        probe_position=probe_position,
        route_cost=route_cost,
        include_actor_subblocks=include_actor_subblocks,
    )


def _full_reference(
    context: StudyContext, runtime: StudySpecimenRuntime, task: Task
) -> TaskReportV2:
    full_scan = context.authority.source_teacher_view(
        runtime.record.specimen_id
    ).full_scan
    positions = np.argwhere(np.ones(runtime.grid.native_shape, dtype=np.bool_))
    readout = read_visible_evidence(
        grid=runtime.grid,
        positions=positions,
        values=full_scan[positions[:, 0], positions[:, 1]],
        cell_levels=(2,) * 64,
        prior=context.background_prior,
        distance_threshold=float(context.config.values["reader"]["distance_threshold"]),
    )
    return build_task_report_v2(readout, task=task)


def _behavior_states(
    runtime: StudySpecimenRuntime,
    *,
    percept: SurfacePercept,
    task: Task,
    prior: Any,
    distance_threshold: float,
    target_costs: tuple[float, ...],
) -> tuple[tuple[dict[str, object], ...], int]:
    observation = runtime.world.reset()
    probe = (0.0, 0.0)
    route_cost = 0.0
    states: list[dict[str, object]] = []
    remaining = list(target_costs)
    transitions = 0
    while True:
        while remaining and observation.effective_budget + 1e-15 >= remaining[0]:
            if not states or states[-1]["levels"] != observation.measurement_state.levels:
                states.append(
                    {
                        "history": observation.action_history,
                        "levels": observation.measurement_state.levels,
                        "cost": observation.effective_budget,
                        "probe_position": probe,
                        "route_cost": route_cost,
                    }
                )
            remaining.pop(0)
        if not remaining or all(level == 2 for level in observation.measurement_state.levels):
            break
        packet = build_observation_packet(
            observation,
            grid=runtime.grid,
            percept=percept,
            task=task,
            prior=prior,
            distance_threshold=distance_threshold,
            probe_position=probe,
            route_cost=route_cost,
            include_actor_subblocks=False,
        )
        action = select_rule_action(
            RuleMethod.R_BALANCED,
            packet,
            grid=runtime.grid,
            coverage_period=4,
        )
        observation, probe, route_cost = _advance(
            runtime, observation, action, probe, route_cost
        )
        transitions += 1
    return tuple(states), transitions


def _candidate_actions(
    packet: ObservationPacket,
    *,
    runtime: StudySpecimenRuntime,
    mu_action: InspectionCellAction,
    seed_material: str,
    limit: int,
) -> tuple[InspectionCellAction, ...]:
    candidates: list[InspectionCellAction] = []

    def add(action: InspectionCellAction) -> None:
        if action.cell_index not in {row.cell_index for row in candidates}:
            candidates.append(action)

    add(mu_action)
    add(select_rule_action(RuleMethod.R_GEOM, packet, grid=runtime.grid))
    add(select_rule_action(RuleMethod.R_LEGACY, packet, grid=runtime.grid))
    state = GeneralizedMeasurementState(runtime.grid.state_sha256, packet.cell_levels)
    legal = tuple(
        InspectionCellAction(cell, level, level + 1)
        for cell, level in enumerate(packet.cell_levels)
        if level < 2
    )
    low_cost = min(
        legal,
        key=lambda action: (
            candidate_budget_record(runtime.grid, state, action).measured_count,
            action.cell_index,
        ),
    )
    add(low_cost)
    remaining = [
        action
        for action in legal
        if action.cell_index not in {row.cell_index for row in candidates}
    ]
    for random_index in range(2):
        if not remaining:
            break
        seed = int.from_bytes(
            hashlib.sha256(
                f"{seed_material}|random-{random_index}".encode()
            ).digest()[:8],
            "big",
        )
        selected = remaining[int(np.random.default_rng(seed).integers(len(remaining)))]
        add(selected)
        remaining = [
            action for action in remaining if action.cell_index != selected.cell_index
        ]
    return tuple(candidates[:limit])


def _candidate_suffix(
    runtime: StudySpecimenRuntime,
    *,
    base_history: tuple[InspectionCellAction, ...],
    base_probe: tuple[float, float],
    base_route_cost: float,
    first_action: InspectionCellAction,
    percept: SurfacePercept,
    task: Task,
    reference: TaskReportV2,
    context: StudyContext,
    route_normalizer: float | None = None,
) -> tuple[tuple[StepSnapshot, ...], int]:
    observation = runtime.world.replay(base_history)
    probe = base_probe
    route_cost = base_route_cost
    visible = read_visible_task_report(
        grid=runtime.grid,
        positions=observation.acquired_positions,
        values=observation.measurement_values,
        cell_levels=observation.measurement_state.levels,
        prior=context.background_prior,
        distance_threshold=float(context.config.values["reader"]["distance_threshold"]),
        task=task,
    )
    score = _proxy_score(visible.report, reference)
    snapshots = [_snapshot(observation.effective_budget, visible.report, score)]
    transitions = 0
    action = first_action
    while True:
        observation, probe, route_cost = _advance(
            runtime,
            observation,
            action,
            probe,
            route_cost,
            route_normalizer=route_normalizer,
        )
        transitions += 1
        visible = read_visible_task_report(
            grid=runtime.grid,
            positions=observation.acquired_positions,
            values=observation.measurement_values,
            cell_levels=observation.measurement_state.levels,
            prior=context.background_prior,
            distance_threshold=float(
                context.config.values["reader"]["distance_threshold"]
            ),
            task=task,
        )
        score = _proxy_score(visible.report, reference)
        snapshots.append(
            _snapshot(observation.effective_budget, visible.report, score)
        )
        if all(level == 2 for level in observation.measurement_state.levels):
            break
        action = select_balanced_visible_action(
            cell_levels=observation.measurement_state.levels,
            candidate_cells=visible.report.candidate_cells,
            percept=percept,
            measured_mask=visible.measured_mask,
            probe_position=probe,
            grid=runtime.grid,
            coverage_period=4,
        )
    return tuple(snapshots), transitions


def _candidate_branches(
    runtime: StudySpecimenRuntime,
    *,
    candidates: tuple[InspectionCellAction, ...],
    base_history: tuple[InspectionCellAction, ...],
    base_probe: tuple[float, float],
    base_route_cost: float,
    percept: SurfacePercept,
    task: Task,
    reference: TaskReportV2,
    context: StudyContext,
    executor: ProcessPoolExecutor,
) -> tuple[dict[int, tuple[StepSnapshot, ...]], int]:
    if not isinstance(executor, ProcessPoolExecutor):
        raise TypeError("candidate branch executor is invalid")
    futures = tuple(
        executor.submit(
            _candidate_suffix_worker,
            runtime.record.specimen_key,
            base_history=base_history,
            base_probe=base_probe,
            base_route_cost=base_route_cost,
            first_action=action,
            percept=percept,
            task=task,
            reference=reference,
            route_normalizer=_full_route_length(runtime.grid.native_shape),
        )
        for action in candidates
    )
    results = tuple(future.result() for future in futures)
    return (
        {
            action.cell_index: snapshots
            for action, (snapshots, _used) in zip(
                candidates, results, strict=True
            )
        },
        sum(used for _snapshots, used in results),
    )


def _new_branch_executor(
    config: StudyConfig, context: StudyContext
) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=int(config.values["training"]["cpu_workers"]),
        mp_context=get_context("spawn"),
        initializer=_initialize_branch_worker,
        initargs=(str(config.path), str(config.project_root), str(context.source_root)),
    )


def _initialize_branch_worker(
    config_path: str, project_root: str, source_root: str
) -> None:
    global _BRANCH_WORKER_CONTEXT
    _config, _BRANCH_WORKER_CONTEXT = _load(
        Path(config_path), Path(project_root), Path(source_root)
    )


def _candidate_suffix_worker(
    specimen_key: str,
    *,
    base_history: tuple[InspectionCellAction, ...],
    base_probe: tuple[float, float],
    base_route_cost: float,
    first_action: InspectionCellAction,
    percept: SurfacePercept,
    task: Task,
    reference: TaskReportV2,
    route_normalizer: float,
) -> tuple[tuple[StepSnapshot, ...], int]:
    if _BRANCH_WORKER_CONTEXT is None:
        raise RuntimeError("candidate branch worker is not initialized")
    record = next(
        row
        for row in _BRANCH_WORKER_CONTEXT.roster.pilot_records
        if row.specimen_key == specimen_key
    )
    return _candidate_suffix(
        open_study_specimen(_BRANCH_WORKER_CONTEXT, record),
        base_history=base_history,
        base_probe=base_probe,
        base_route_cost=base_route_cost,
        first_action=first_action,
        percept=percept,
        task=task,
        reference=reference,
        context=_BRANCH_WORKER_CONTEXT,
        route_normalizer=route_normalizer,
    )


def _advance(
    runtime: StudySpecimenRuntime,
    observation: Any,
    action: InspectionCellAction,
    probe: tuple[float, float],
    route_cost: float,
    *,
    route_normalizer: float | None = None,
) -> tuple[Any, tuple[float, float], float]:
    mask = np.zeros(runtime.grid.native_shape, dtype=np.bool_)
    if len(observation.acquired_positions):
        mask[
            observation.acquired_positions[:, 0], observation.acquired_positions[:, 1]
        ] = True
    added = action_added_positions_from_mask(
        runtime.grid, observation.measurement_state, action, mask
    )
    route_length, next_probe = compile_route_cost(
        added, native_shape=runtime.grid.native_shape, start_position=probe
    )
    next_observation = runtime.world.step(observation, action)
    denominator = (
        _full_route_length(runtime.grid.native_shape)
        if route_normalizer is None
        else float(route_normalizer)
    )
    normalized = route_length / denominator
    return next_observation, next_probe, route_cost + normalized


def _advance_detailed(
    runtime: StudySpecimenRuntime,
    observation: Any,
    action: InspectionCellAction,
    probe: tuple[float, float],
    route_cost: float,
) -> tuple[Any, tuple[float, float], float, int]:
    mask = np.zeros(runtime.grid.native_shape, dtype=np.bool_)
    if len(observation.acquired_positions):
        mask[
            observation.acquired_positions[:, 0], observation.acquired_positions[:, 1]
        ] = True
    added = action_added_positions_from_mask(
        runtime.grid, observation.measurement_state, action, mask
    )
    route = compile_route(
        added, native_shape=runtime.grid.native_shape, start_position=probe
    )
    next_observation = runtime.world.step(observation, action)
    normalized = route.total_length / _full_route_length(runtime.grid.native_shape)
    return (
        next_observation,
        route.end_position,
        route_cost + normalized,
        route.turn_count,
    )


@lru_cache(maxsize=4)
def _full_route_length(native_shape: tuple[int, int]) -> float:
    return reference_full_raster_length(native_shape, (0.0, 0.0))


def _proxy_score(
    report: TaskReportV2, reference: TaskReportV2
) -> dict[str, object]:
    if report.task is not reference.task or report.predicted_mask.shape != reference.predicted_mask.shape:
        raise ValueError("proxy report and reference differ")
    prediction = report.predicted_mask
    target = reference.predicted_mask
    if report.task is Task.LOCATE:
        prediction_box = _bbox(_largest_component(prediction))
        target_component = _largest_component(target)
        iou = _bbox_iou(prediction_box, _bbox(target_component))
        supports = report.support_positions
        inside = (
            target_component[supports[:, 0], supports[:, 1]]
            if len(supports)
            else np.empty(0, dtype=np.bool_)
        )
        accepted = supports[inside]
        support_quality = min(
            1.0,
            len(accepted) / 3.0,
            len(set(accepted[:, 0])) / 2.0 if len(accepted) else 0.0,
            len(set(accepted[:, 1])) / 2.0 if len(accepted) else 0.0,
        )
        success = bool(np.any(target_component) and iou >= 0.50 and support_quality >= 1.0)
        loss = 1.0 - iou * support_quality
        return {
            "success": success,
            "task_loss": float(np.clip(loss, 0.0, 1.0)),
            "iou": iou,
            "recall": float(iou >= 0.50),
            "relative_area_error": 0.0,
        }
    intersection = int(np.count_nonzero(prediction & target))
    union = int(np.count_nonzero(prediction | target))
    target_area = int(np.count_nonzero(target))
    predicted_area = int(np.count_nonzero(prediction))
    iou = float(intersection / union) if union else 0.0
    recall = float(intersection / target_area) if target_area else 0.0
    area_error = (
        float(abs(predicted_area - target_area) / target_area)
        if target_area
        else float(predicted_area > 0)
    )
    success = bool(
        target_area and iou >= 0.70 and recall >= 0.90 and area_error <= 0.10
    )
    quality = (iou + recall + max(0.0, 1.0 - min(area_error, 1.0))) / 3.0
    return {
        "success": success,
        "task_loss": float(np.clip(1.0 - quality, 0.0, 1.0)),
        "iou": iou,
        "recall": recall,
        "relative_area_error": area_error,
    }


def _snapshot(
    cost: float, report: TaskReportV2, score: dict[str, object]
) -> StepSnapshot:
    return StepSnapshot(
        cost=cost,
        success=bool(score["success"]),
        task_loss=float(score["task_loss"]),
        report_digest=_report_digest(report),
    )


def _report_digest(report: TaskReportV2) -> str:
    digest = hashlib.sha256(report.task.value.encode())
    digest.update(np.packbits(report.predicted_mask, bitorder="little").tobytes())
    digest.update(np.asarray(report.support_positions, dtype="<i8").tobytes())
    return digest.hexdigest()


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if not count:
        return np.zeros_like(mask, dtype=np.bool_)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.argwhere(mask)
    if not len(points):
        return None
    minimum = points.min(axis=0)
    maximum = points.max(axis=0) + 1
    return int(minimum[0]), int(minimum[1]), int(maximum[0]), int(maximum[1])


def _bbox_iou(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int] | None,
) -> float:
    if left is None or right is None:
        return 0.0
    ly0, lx0, ly1, lx1 = left
    ry0, rx0, ry1, rx1 = right
    intersection = max(0, min(ly1, ry1) - max(ly0, ry0)) * max(
        0, min(lx1, rx1) - max(lx0, rx0)
    )
    union = (ly1 - ly0) * (lx1 - lx0) + (ry1 - ry0) * (rx1 - rx0) - intersection
    return float(intersection / union) if union else 0.0


def _policy_row(
    packet: ObservationPacket,
    *,
    specimen_key: str,
    task: Task,
    queried_cells: tuple[int, ...],
    target_probabilities: np.ndarray,
    state_origin: str,
    state_cost: float,
    main_costs: np.ndarray,
    auxiliary_costs: np.ndarray,
) -> dict[str, object]:
    return {
        "specimen_key": specimen_key,
        "task": task.value,
        "cell_features": np.asarray(packet.cell_features),
        "subblock_features": np.asarray(packet.subblock_features),
        "global_features": np.asarray(packet.global_features),
        "history_features": np.asarray(packet.history_features),
        "legal_mask": np.asarray(packet.legal_mask),
        "queried_cells": queried_cells,
        "target_probabilities": np.asarray(target_probabilities, dtype=np.float32),
        "state_origin": state_origin,
        "state_cost": float(state_cost),
        "main_costs": np.asarray(main_costs, dtype=np.float64),
        "auxiliary_costs": np.asarray(auxiliary_costs, dtype=np.float64),
    }


def _stop_row(
    packet: ObservationPacket,
    *,
    specimen_key: str,
    task: Task,
    label: bool,
) -> dict[str, object]:
    return {
        "specimen_key": specimen_key,
        "task": task.value,
        "cell_features": np.asarray(packet.cell_features),
        "subblock_features": np.asarray(packet.subblock_features),
        "global_features": np.asarray(packet.global_features),
        "history_features": np.asarray(packet.history_features),
        "label": bool(label),
    }


def _torch_save_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _policy_example(row: dict[str, object]) -> PolicyTrainingExample:
    return PolicyTrainingExample(
        specimen_key=str(row["specimen_key"]),
        task=Task(str(row["task"])),
        cell_features=np.asarray(row["cell_features"], dtype=np.float32),
        subblock_features=np.asarray(row["subblock_features"], dtype=np.float32),
        global_features=np.asarray(row["global_features"], dtype=np.float32),
        history_features=np.asarray(row["history_features"], dtype=np.float32),
        legal_mask=np.asarray(row["legal_mask"], dtype=np.bool_),
        queried_cells=tuple(int(value) for value in row["queried_cells"]),
        target_probabilities=np.asarray(
            row["target_probabilities"], dtype=np.float32
        ),
    )


def _stop_example(row: dict[str, object]) -> StopTrainingExample:
    return StopTrainingExample(
        specimen_key=str(row["specimen_key"]),
        task=Task(str(row["task"])),
        cell_features=np.asarray(row["cell_features"], dtype=np.float32),
        subblock_features=np.asarray(row["subblock_features"], dtype=np.float32),
        global_features=np.asarray(row["global_features"], dtype=np.float32),
        history_features=np.asarray(row["history_features"], dtype=np.float32),
        label=bool(row["label"]),
    )


def _internal_fit_split(
    examples: tuple[PolicyTrainingExample, ...],
) -> tuple[tuple[PolicyTrainingExample, ...], tuple[PolicyTrainingExample, ...]]:
    validation_keys = _one_specimen_per_domain(
        tuple(row.specimen_key for row in examples)
    )
    fit = tuple(row for row in examples if row.specimen_key not in validation_keys)
    valid = tuple(row for row in examples if row.specimen_key in validation_keys)
    if not fit or not valid:
        raise RuntimeError("internal TRAIN split is empty")
    return fit, valid


def _internal_stop_split(
    examples: tuple[StopTrainingExample, ...],
) -> tuple[tuple[StopTrainingExample, ...], tuple[StopTrainingExample, ...]]:
    validation_keys = _one_specimen_per_domain(
        tuple(row.specimen_key for row in examples)
    )
    fit = tuple(row for row in examples if row.specimen_key not in validation_keys)
    valid = tuple(row for row in examples if row.specimen_key in validation_keys)
    if not fit or not valid:
        raise RuntimeError("internal stop TRAIN split is empty")
    return fit, valid


def _one_specimen_per_domain(keys: tuple[str, ...]) -> set[str]:
    output = set()
    for domain in sorted({key.split(":", 1)[0] for key in keys}):
        candidates = sorted({key for key in keys if key.startswith(f"{domain}:")})
        output.add(candidates[-1])
    return output


def _fit_log_rows(method: str, seed: int, fit: Any) -> list[dict[str, object]]:
    return [
        {
            "schema_version": 1,
            "method": method,
            "seed": seed,
            "optimizer_step": row.optimizer_step,
            "train_loss": row.train_loss,
            "valid_loss": "" if row.valid_loss is None else row.valid_loss,
        }
        for row in fit.log
    ]


def _aggregate_states(
    *,
    config: StudyConfig,
    context: StudyContext,
    percepts: dict[str, SurfacePercept],
    actor: LearnedCellActor,
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    actor.eval()
    actor.to(device)
    ctg_rows: list[dict[str, object]] = []
    stop_rows: list[dict[str, object]] = []
    transitions = 0
    training = config.values["training"]
    assignments = tuple(
        row for row in context.roster.assignments if row.split is Split.TRAIN
    )
    branch_executor = _new_branch_executor(config, context)
    for specimen_index, assignment in enumerate(assignments, start=1):
        runtime = open_study_specimen(context, assignment.record)
        percept = percepts[assignment.record.specimen_key]
        for task in Task:
            reference = _full_reference(context, runtime, task)
            states, used = _learned_states(
                runtime,
                percept=percept,
                task=task,
                context=context,
                actor=actor,
                device=device,
                action_indices=(64, 128),
            )
            transitions += used
            for state_index, state in enumerate(states):
                observation = runtime.world.replay(state["history"])
                packet = _packet(
                    observation,
                    runtime=runtime,
                    percept=percept,
                    task=task,
                    context=context,
                    probe_position=state["probe_position"],
                    route_cost=state["route_cost"],
                )
                mu_action = select_rule_action(
                    RuleMethod.R_BALANCED,
                    packet,
                    grid=runtime.grid,
                    coverage_period=4,
                )
                candidates = _candidate_actions(
                    packet,
                    runtime=runtime,
                    mu_action=mu_action,
                    seed_material=(
                        f"aggregate|{assignment.record.specimen_key}|"
                        f"{task.value}|{state_index}"
                    ),
                    limit=int(training["max_candidates_per_state"]),
                )
                branches, branch_used = _candidate_branches(
                    runtime,
                    candidates=candidates,
                    base_history=state["history"],
                    base_probe=state["probe_position"],
                    base_route_cost=state["route_cost"],
                    percept=percept,
                    task=task,
                    reference=reference,
                    context=context,
                    executor=branch_executor,
                )
                transitions += branch_used
                labels = cost_to_go_targets(
                    current_cost=observation.effective_budget,
                    candidate_snapshots=branches,
                    temperature=float(training["cost_temperature"]),
                    auxiliary_weight=float(training["auxiliary_loss_weight"]),
                )
                ctg_rows.append(
                    _policy_row(
                        packet,
                        specimen_key=assignment.record.specimen_key,
                        task=task,
                        queried_cells=tuple(int(value) for value in labels.queried_actions),
                        target_probabilities=labels.probabilities,
                        state_origin="LEARNED_VISITED_AGGREGATION_1",
                        state_cost=observation.effective_budget,
                        main_costs=labels.main_costs,
                        auxiliary_costs=labels.auxiliary_costs,
                    )
                )
                score = _proxy_score(packet.report, reference)
                stop_rows.append(
                    _stop_row(
                        packet,
                        specimen_key=assignment.record.specimen_key,
                        task=task,
                        label=bool(score["success"]),
                    )
                )
        print(
            f"aggregation {specimen_index}/{len(assignments)} "
            f"states={len(ctg_rows)} transitions={transitions}",
            flush=True,
        )
    branch_executor.shutdown(wait=True)
    return ctg_rows, stop_rows, transitions


def _learned_states(
    runtime: StudySpecimenRuntime,
    *,
    percept: SurfacePercept,
    task: Task,
    context: StudyContext,
    actor: LearnedCellActor,
    device: str,
    action_indices: tuple[int, ...],
) -> tuple[tuple[dict[str, object], ...], int]:
    observation = runtime.world.reset()
    probe = (0.0, 0.0)
    route_cost = 0.0
    states = []
    transitions = 0
    targets = list(action_indices)
    while targets:
        packet = _packet(
            observation,
            runtime=runtime,
            percept=percept,
            task=task,
            context=context,
            probe_position=probe,
            route_cost=route_cost,
        )
        action = _actor_action(actor, packet, runtime, device)
        observation, probe, route_cost = _advance(
            runtime, observation, action, probe, route_cost
        )
        transitions += 1
        if transitions == targets[0]:
            states.append(
                {
                    "history": observation.action_history,
                    "levels": observation.measurement_state.levels,
                    "cost": observation.effective_budget,
                    "probe_position": probe,
                    "route_cost": route_cost,
                }
            )
            targets.pop(0)
    return tuple(states), transitions


def _actor_action(
    actor: LearnedCellActor,
    packet: ObservationPacket,
    runtime: StudySpecimenRuntime,
    device: str,
) -> InspectionCellAction:
    cell = actor.select_cell(packet, device=device)
    level = packet.cell_levels[cell]
    return InspectionCellAction(cell, level, level + 1)


def _save_model(
    model: torch.nn.Module,
    path: Path,
    *,
    method: str,
    seed: int,
    config_sha256: str,
) -> None:
    model.to("cpu")
    _torch_save_atomic(
        {
            "schema_version": 1,
            "method": method,
            "seed": seed,
            "config_sha256": config_sha256,
            "use_surface_features": bool(
                getattr(model, "use_surface_features", True)
            ),
            "state_dict": model.state_dict(),
        },
        path,
    )


def _load_actor(
    path: Path, config: StudyConfig, device: str
) -> LearnedCellActor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    method = payload.get("method")
    use_surface_features = bool(payload.get("use_surface_features", True))
    if (
        payload.get("config_sha256") != config.config_sha256
        or method not in {"L_BC", "L_CTG_0", "L_CTG_1", "L_NO_VLM"}
        or use_surface_features != (method != "L_NO_VLM")
    ):
        raise RuntimeError("learned actor model identity changed")
    model = LearnedCellActor(use_surface_features=use_surface_features)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def _load_stop(
    path: Path, config: StudyConfig, device: str
) -> LearnedStopHead:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("config_sha256") != config.config_sha256 or payload.get("method") != "S_LEARN":
        raise RuntimeError("learned stop model identity changed")
    model = LearnedStopHead()
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def _run_planner_episode(
    runtime: StudySpecimenRuntime,
    *,
    percept: SurfacePercept,
    task: Task,
    reference: TaskReportV2,
    context: StudyContext,
    method_name: str,
    planner: RuleMethod | LearnedCellActor,
    coverage_period: int,
    seed: int,
    device: str,
    stop_model: LearnedStopHead,
    learned_threshold: float | None,
    store_trajectory: bool,
) -> tuple[dict[str, object], list[dict[str, object]], list[StopValidationRow]]:
    observation = runtime.world.reset()
    probe = (0.0, 0.0)
    route_cost = 0.0
    route_turns = 0
    last_action: InspectionCellAction | None = None
    controller = RuleStopController(task)
    snapshots: list[StepSnapshot] = []
    trajectory: list[dict[str, object]] = []
    stop_rows: list[StopValidationRow] = []
    rule_stop: tuple[float, bool] | None = None
    learned_stop: tuple[float, bool] | None = None
    inference_total = 0.0
    step = 0
    while True:
        packet = _packet(
            observation,
            runtime=runtime,
            percept=percept,
            task=task,
            context=context,
            probe_position=probe,
            route_cost=route_cost,
        )
        score = _proxy_score(packet.report, reference)
        snapshot = _snapshot(observation.effective_budget, packet.report, score)
        snapshots.append(snapshot)
        rule_decision = controller.update(
            packet.report,
            cell_levels=packet.cell_levels,
            last_action=last_action,
        )
        if rule_decision.should_stop and rule_stop is None:
            rule_stop = (observation.effective_budget, bool(score["success"]))
        stop_probability = _stop_probability(stop_model, packet, device)
        eligible = _mechanically_stop_eligible(packet)
        stop_rows.append(
            StopValidationRow(
                specimen_key=runtime.record.specimen_key,
                task=task,
                probability=stop_probability,
                label=bool(score["success"]),
                mechanically_eligible=eligible,
            )
        )
        if (
            learned_threshold is not None
            and eligible
            and stop_probability >= learned_threshold
            and learned_stop is None
        ):
            learned_stop = (observation.effective_budget, bool(score["success"]))
        terminal = all(level == 2 for level in packet.cell_levels)
        action: InspectionCellAction | None = None
        inference_seconds = 0.0
        if not terminal:
            before = time.perf_counter()
            if isinstance(planner, RuleMethod):
                action = select_rule_action(
                    planner,
                    packet,
                    grid=runtime.grid,
                    coverage_period=coverage_period,
                )
            else:
                action = _actor_action(planner, packet, runtime, device)
            inference_seconds = time.perf_counter() - before
            inference_total += inference_seconds
        if store_trajectory:
            trajectory.append(
                {
                    "schema_version": 1,
                    "dataset_id": runtime.record.dataset_id,
                    "specimen_id": runtime.record.specimen_id,
                    "specimen_key": runtime.record.specimen_key,
                    "task": task.value,
                    "method": method_name,
                    "seed": seed,
                    "step": step,
                    "cost": observation.effective_budget,
                    "success": bool(score["success"]),
                    "task_loss": float(score["task_loss"]),
                    "iou": float(score["iou"]),
                    "recall": float(score["recall"]),
                    "relative_area_error": float(score["relative_area_error"]),
                    "report_sha256": snapshot.report_digest,
                    "packet_sha256": packet.feature_sha256,
                    "candidate_cell_count": len(packet.report.candidate_cells),
                    "support_count": len(packet.report.support_positions),
                    "action_cell": -1 if action is None else action.cell_index,
                    "action_from_level": -2 if action is None else action.from_level,
                    "action_to_level": -2 if action is None else action.to_level,
                    "rule_stop": rule_decision.should_stop,
                    "learned_stop_probability": stop_probability,
                    "learned_stop_eligible": eligible,
                    "cumulative_route_cost": route_cost,
                    "cumulative_route_turns": route_turns,
                    "action_inference_seconds": inference_seconds,
                }
            )
        if terminal:
            break
        assert action is not None
        observation, probe, route_cost, turns = _advance_detailed(
            runtime, observation, action, probe, route_cost
        )
        route_turns += turns
        last_action = action
        step += 1
        if step > 192:
            raise RuntimeError("progressive episode exceeded 192 actions")
    snapshots_tuple = tuple(snapshots)
    ausc = exact_step_integral(
        snapshots_tuple, field="success", start_cost=0.0, end_cost=1.0
    )
    rule_completed = bool(rule_stop is not None and rule_stop[1])
    learned_completed = bool(learned_stop is not None and learned_stop[1])
    result = {
        "schema_version": 1,
        "dataset_id": runtime.record.dataset_id,
        "specimen_id": runtime.record.specimen_id,
        "specimen_key": runtime.record.specimen_key,
        "task": task.value,
        "method": method_name,
        "seed": seed,
        "action_count": step,
        "ausc_any": ausc,
        "success_c0025": value_at_cost(snapshots_tuple, cost=0.025).success,
        "success_c005": value_at_cost(snapshots_tuple, cost=0.05).success,
        "success_c010": value_at_cost(snapshots_tuple, cost=0.10).success,
        "success_c020": value_at_cost(snapshots_tuple, cost=0.20).success,
        "success_c040": value_at_cost(snapshots_tuple, cost=0.40).success,
        "success_c060": value_at_cost(snapshots_tuple, cost=0.60).success,
        "success_c080": value_at_cost(snapshots_tuple, cost=0.80).success,
        "success_c100": snapshots_tuple[-1].success,
        "full_input_task_loss": snapshots_tuple[-1].task_loss,
        "route_cost": route_cost,
        "route_turns": route_turns,
        "inference_seconds": inference_total,
        "rule_stopped": rule_stop is not None,
        "rule_stop_cost": "" if rule_stop is None else rule_stop[0],
        "rule_stop_success": rule_completed,
        "rule_false_stop": bool(rule_stop is not None and not rule_stop[1]),
        "rule_resource_exhausted": rule_stop is None,
        "rule_autonomous_ausc": (
            1.0 - rule_stop[0] if rule_completed and rule_stop is not None else 0.0
        ),
        "rule_failure_penalized_cost": failure_penalized_cost(
            1.0 if rule_stop is None else rule_stop[0], success=rule_completed
        ),
        "learned_stopped": learned_stop is not None,
        "learned_stop_cost": "" if learned_stop is None else learned_stop[0],
        "learned_stop_success": learned_completed,
        "learned_false_stop": bool(learned_stop is not None and not learned_stop[1]),
        "learned_resource_exhausted": learned_stop is None,
        "learned_autonomous_ausc": (
            1.0 - learned_stop[0]
            if learned_completed and learned_stop is not None
            else 0.0
        ),
        "learned_failure_penalized_cost": failure_penalized_cost(
            1.0 if learned_stop is None else learned_stop[0],
            success=learned_completed,
        ),
    }
    return result, trajectory, stop_rows


def _stop_probability(
    model: LearnedStopHead, packet: ObservationPacket, device: str
) -> float:
    tensors = packet.actor_tensors()
    target = torch.device(device)
    with torch.no_grad():
        logit = model(
            torch.tensor(tensors["cell_features"]).unsqueeze(0).to(target),
            torch.tensor(tensors["subblock_features"]).unsqueeze(0).to(target),
            torch.tensor(tensors["global_features"]).unsqueeze(0).to(target),
            torch.tensor(tensors["history_features"]).unsqueeze(0).to(target),
        )
    return float(torch.sigmoid(logit[0]).cpu())


def _mechanically_stop_eligible(packet: ObservationPacket) -> bool:
    supports = packet.report.support_positions
    return bool(
        len(packet.measured_positions)
        and packet.report.candidate_cells
        and len(supports) >= 3
        and len({int(value) for value in supports[:, 0]}) >= 2
        and len({int(value) for value in supports[:, 1]}) >= 2
    )


def _mean_metric(
    rows: list[dict[str, object]],
    method: str,
    field: str,
    *,
    task: Task | None = None,
) -> float:
    values = [
        float(row[field])
        for row in rows
        if row["method"] == method and (task is None or row["task"] == task.value)
    ]
    if not values:
        raise RuntimeError(f"validation metric is empty: {method}/{field}")
    return float(np.mean(values))


def _authorization_payload(
    authorization: LearnedStopAuthorization,
) -> dict[str, object]:
    return {
        "status": authorization.status.value,
        "threshold": authorization.threshold,
        "validation_specimen_count": authorization.validation_specimen_count,
        "diagnostics": [asdict(row) for row in authorization.diagnostics],
    }


def _comparison_rows(
    episodes: list[dict[str, object]],
    *,
    treatment: str,
    comparators: tuple[str, ...],
    replicates: int,
    seed: int,
) -> tuple[dict[str, object], ...]:
    rows = []
    for task_index, task in enumerate(Task):
        task_episodes = [row for row in episodes if row["task"] == task.value]
        for comparator_index, comparator in enumerate(dict.fromkeys(comparators)):
            metric_rows = tuple(
                MetricRecord(
                    specimen_key=str(row["specimen_key"]),
                    domain=str(row["dataset_id"]),
                    method=str(row["method"]),
                    task=task,
                    seed=int(row["seed"]),
                    value=float(row["ausc_any"]),
                )
                for row in task_episodes
                if row["method"] in {treatment, comparator}
            )
            effect = paired_physical_specimen_bootstrap(
                metric_rows,
                treatment=treatment,
                comparator=comparator,
                replicates=replicates,
                seed=seed + 100 * task_index + comparator_index,
            )
            rows.append(
                {
                    "schema_version": 1,
                    "task": task.value,
                    "metric": "ausc_any",
                    "treatment": treatment,
                    "comparator": comparator,
                    "estimate": effect.estimate,
                    "ci_lower": effect.ci_lower,
                    "ci_upper": effect.ci_upper,
                    "physical_specimen_count": effect.physical_specimen_count,
                    "domain_count": effect.domain_count,
                    "bootstrap_replicates": effect.replicates,
                    "proxy_scope": "SAME_READER_SELF_CONSISTENCY",
                    "formal_estimate": "",
                }
            )
    return tuple(rows)


def _failure_rows(
    episodes: list[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    rows = []
    for episode in episodes:
        if not bool(episode["success_c100"]):
            rows.append(
                _failure_row(
                    episode,
                    mode="PLANNER_ONLY",
                    failure="FULL_INPUT_PROXY_FAILURE",
                    cost=1.0,
                )
            )
        for prefix, mode in (("rule", "S_RULE"), ("learned", "S_LEARN")):
            if bool(episode[f"{prefix}_stop_success"]):
                continue
            stopped = bool(episode[f"{prefix}_stopped"])
            cost_value = episode[f"{prefix}_stop_cost"]
            rows.append(
                _failure_row(
                    episode,
                    mode=mode,
                    failure="FALSE_STOP" if stopped else "RESOURCE_EXHAUSTED",
                    cost=1.0 if cost_value == "" else float(cost_value),
                )
            )
    return tuple(rows)


def _failure_row(
    episode: dict[str, object], *, mode: str, failure: str, cost: float
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": episode["dataset_id"],
        "specimen_id": episode["specimen_id"],
        "specimen_key": episode["specimen_key"],
        "task": episode["task"],
        "method": episode["method"],
        "seed": episode["seed"],
        "evaluation_mode": mode,
        "failure_type": failure,
        "terminal_cost": cost,
    }


def _csv_bool(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError("CSV boolean is invalid")
    return value == "True"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_sha256(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _set_jaccard(left: list[int], right: list[int]) -> float:
    union = set(left) | set(right)
    return float(len(set(left) & set(right)) / len(union)) if union else 1.0


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_files(root: Path, names: tuple[str, ...]) -> None:
    missing = tuple(name for name in names if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"required study artifacts are missing: {missing}")


__all__ = [
    "build_training_bank",
    "evaluate_study",
    "prepare_study",
    "run_perception",
    "summarize_study",
    "train_models",
    "validate_models",
]
