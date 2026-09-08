"""Finalize frozen C-scan evidence from optional user-supplied inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.vlm_cscan.references import reference_from_payload

from .contracts import Split, Task
from .frozen_process_analysis import (
    STOP_SYSTEMS,
    FrozenProcessConfig,
    _wait_components,
    planning_stop_accounting,
    report_stability,
    stop_decomposition,
)
from .frozen_process_recovery import recover_reviewed_report_scores
from .metrics import MetricRecord
from .runtime import load_study_config, load_study_context
from .supplement_analysis import make_domain_bootstrap_draws, paired_domain_bootstrap

CLAIM_STATUSES = (
    "SUPPORTED",
    "NOT_SUPPORTED",
    "INSUFFICIENT_PRECISION",
    "PENDING_INPUT",
    "NOT_EVALUATED",
)
BLIND_REVIEW_DECISIONS = (
    "DELIVERABLE",
    "NEEDS_FURTHER_INSPECTION",
    "UNABLE_TO_JUDGE",
)
REVIEWED_METHODS = (
    "R_BALANCED_P8",
    "BC_S1",
    "BC_S2",
    "BC_S3",
    "BC_NO_VLM_S1",
    "BC_NO_US_FEEDBACK_S1",
)
REVIEWED_RECOVERY_TABLES = (
    "report_scores",
    "full_input_readout",
    "surface_reference_agreement",
    "spatial_action_statistics",
    "spatial_episode_summary",
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_reviewed_recoveries(
    config: FrozenProcessConfig,
    *,
    reference_version: str,
    cached: Mapping[str, object],
    fresh: Mapping[str, object],
    total_specimen_count: int,
    cached_specimen_count: int,
) -> dict[str, object]:
    """Merge identity-checked cached specimens with newly recovered specimens."""

    if (
        type(config) is not FrozenProcessConfig
        or not reference_version.startswith("REVIEWED_")
        or not 0 <= cached_specimen_count <= total_specimen_count
    ):
        raise ValueError("reviewed recovery merge request is invalid")
    merged: dict[str, object] = {}
    for table in REVIEWED_RECOVERY_TABLES:
        rows = []
        for source in (cached, fresh):
            values = source.get(table, ())
            if not isinstance(values, tuple):
                raise TypeError("reviewed recovery table is invalid")
            rows.extend(
                {**row, "reference_version": reference_version}
                for row in values
            )
        merged[table] = tuple(
            sorted(
                rows,
                key=lambda row: (
                    str(row.get("specimen_key", "")),
                    str(row.get("task", "")),
                    str(row.get("method", "")),
                    int(row.get("seed", -1)),
                    str(row.get("stop_system", "")),
                    int(row.get("step", -1)),
                ),
            )
        )
    fresh_recovery = fresh.get("recovery", {})
    if not isinstance(fresh_recovery, Mapping):
        raise TypeError("reviewed recovery metadata is invalid")
    episodes_per_specimen = len(config.tasks) * len(REVIEWED_METHODS)
    total_episodes = total_specimen_count * episodes_per_specimen
    cached_episodes = cached_specimen_count * episodes_per_specimen
    source_transitions = total_episodes * config.actions_per_episode
    cached_transitions = cached_episodes * config.actions_per_episode
    current_transitions = int(
        fresh_recovery.get("stored_action_transition_count", 0)
    )
    if (
        current_transitions + cached_transitions != source_transitions
        or current_transitions > config.reviewed_recovery_step_cap
    ):
        raise ValueError("reviewed recovery merge counts are invalid")
    merged["recovery"] = {
        "status": "FROZEN_REVIEWED_RECOVERY_COMPLETE",
        "reference_version": reference_version,
        "reviewed_specimen_count": total_specimen_count,
        "episode_count": total_episodes,
        "report_state_count": total_episodes * config.states_per_episode,
        "cache_hit_specimen_count": cached_specimen_count,
        "cache_hit_episode_count": cached_episodes,
        "stored_action_transition_count": current_transitions,
        "cached_action_transition_count": cached_transitions,
        "source_action_transition_count": source_transitions,
        "conditional_transition_cap": config.reviewed_recovery_step_cap,
        "world_step_count": 0,
        "cache_reuse": "IDENTITY_BOUND_RESULT_ROWS_AND_STORED_ACTIONS",
        "actor_forward_calls": 0,
        "stop_forward_calls": 0,
        "vlm_calls": 0,
        "training_updates": 0,
    }
    return merged


def _reviewed_cache_identity(config: FrozenProcessConfig) -> dict[str, object]:
    digest = hashlib.sha256()
    for path in sorted(
        (Path(__file__).resolve(), Path(recover_reviewed_report_scores.__code__.co_filename))
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return {
        "cache_schema_version": 1,
        "repository_base_sha": config.repository_base_sha,
        "config_sha256": config.config_sha256,
        "implementation_sha256": digest.hexdigest(),
        "frozen_input_hashes": dict(sorted(config.expected_input_hashes.items())),
        "external_frozen_input_hashes": dict(
            sorted(config.expected_external_hashes.items())
        ),
    }


def _empty_reviewed_cache() -> dict[str, object]:
    return {
        **{table: () for table in REVIEWED_RECOVERY_TABLES},
        "cached_specimen_keys": (),
    }


def _load_cached_reviewed_recovery(
    config: FrozenProcessConfig,
    *,
    current_coverage: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    """Load only per-specimen scores bound to unchanged inputs and code."""

    reviewed = config.output_root / "reviewed"
    manifest_path = reviewed / "recovery_manifest.json"
    coverage_path = reviewed / "reference_coverage.csv"
    required_paths = {
        "report_scores": reviewed / "report_scores.parquet",
        "full_input_readout": reviewed / "full_input_readout.csv",
        "surface_reference_agreement": reviewed
        / "surface_reference_agreement.csv",
        "spatial_action_statistics": reviewed
        / "spatial_action_statistics.parquet",
        "spatial_episode_summary": reviewed / "spatial_episode_summary.csv",
    }
    if not manifest_path.is_file() or not coverage_path.is_file() or any(
        not path.is_file() for path in required_paths.values()
    ):
        return _empty_reviewed_cache()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = _reviewed_cache_identity(config)
    if any(manifest.get(key) != value for key, value in identity.items()):
        return _empty_reviewed_cache()
    reference_files = manifest.get("reference_files")
    if not isinstance(reference_files, dict):
        return _empty_reviewed_cache()
    current_files = {
        str(row["specimen_key"]): str(row["reference_sha256"])
        for row in current_coverage
        if row.get("status") == "REVIEWED_REFERENCE_MATCHED"
    }
    cached_coverage = pl.read_csv(coverage_path, infer_schema_length=None).to_dicts()
    cached_files = {
        str(row["specimen_key"]): str(row["reference_sha256"])
        for row in cached_coverage
        if row.get("status") == "REVIEWED_REFERENCE_MATCHED"
    }
    matched = tuple(
        sorted(
            key
            for key, sha256 in current_files.items()
            if cached_files.get(key) == sha256
            and reference_files.get(key) == sha256
        )
    )
    if not matched:
        return _empty_reviewed_cache()
    tables = {
        name: (
            pl.read_parquet(path)
            if path.suffix == ".parquet"
            else pl.read_csv(path, infer_schema_length=None)
        )
        .filter(pl.col("specimen_key").is_in(matched))
        .to_dicts()
        for name, path in required_paths.items()
    }
    expected_counts = {
        "report_scores": (
            len(matched)
            * len(config.tasks)
            * len(REVIEWED_METHODS)
            * config.states_per_episode
        ),
        "full_input_readout": len(matched) * len(config.tasks),
        "surface_reference_agreement": len(matched) * len(config.tasks),
        "spatial_action_statistics": (
            len(matched)
            * len(config.tasks)
            * len(config.spatial_methods)
            * config.actions_per_episode
        ),
        "spatial_episode_summary": (
            len(matched) * len(config.tasks) * len(config.spatial_methods) * 3
        ),
    }
    if any(len(tables[name]) != count for name, count in expected_counts.items()):
        return _empty_reviewed_cache()
    return {
        **{name: tuple(rows) for name, rows in tables.items()},
        "cached_specimen_keys": matched,
    }


def rescore_frozen_episode(
    rows: tuple[Mapping[str, object], ...],
    *,
    formal_success_by_report: Mapping[str, bool],
    stop_system: str,
) -> dict[str, object]:
    """Replace only success labels while preserving the frozen trajectory/STOP."""

    if not rows or not isinstance(formal_success_by_report, Mapping):
        raise ValueError("reviewed episode rescore request is invalid")
    reviewed_rows = []
    for row in rows:
        digest = str(row.get("report_sha256", ""))
        success = formal_success_by_report.get(digest)
        if type(success) is not bool:
            raise ValueError(f"reviewed success is missing for report {digest}")
        reviewed_rows.append({**row, "success": success})
    reviewed = tuple(reviewed_rows)
    stop = stop_decomposition(reviewed, stop_system)
    accounting = planning_stop_accounting(
        reviewed, stop_step=stop["stop_step"]
    )
    return {**stop, **accounting}


def evaluate_path_b_decision(
    completion_effect: Mapping[str, object],
    cost_effect: Mapping[str, object],
    *,
    physical_n: int,
    expected_physical_n: int,
) -> dict[str, object]:
    """Apply the frozen joint 97.5% completion/cost acceptance rule."""

    if (
        not isinstance(completion_effect, Mapping)
        or not isinstance(cost_effect, Mapping)
        or type(physical_n) is not int
        or type(expected_physical_n) is not int
        or physical_n < 0
        or expected_physical_n < 1
        or physical_n > expected_physical_n
    ):
        raise ValueError("Path B decision request is invalid")
    for effect in (completion_effect, cost_effect):
        if any(
            isinstance(effect.get(field), bool)
            or not isinstance(effect.get(field), (int, float))
            or not math.isfinite(float(effect[field]))
            for field in ("estimate", "ci_lower", "ci_upper")
        ):
            raise ValueError("Path B effect is invalid")
    completion_pass = float(completion_effect["ci_lower"]) >= -0.05 - 1e-15
    cost_pass = float(cost_effect["ci_lower"]) > 0.0
    if physical_n < expected_physical_n:
        status = "INSUFFICIENT_PRECISION"
        joint: bool | None = None
    else:
        joint = completion_pass and cost_pass
        status = "SUPPORTED" if joint else "NOT_SUPPORTED"
    return {
        "status": status,
        "joint_path_b_pass": joint,
        "completion_noninferiority_pass": completion_pass,
        "failure_cost_improvement_pass": cost_pass,
        "completion_noninferiority_margin": -0.05,
        "confidence_level": 0.975,
        "physical_specimen_count": physical_n,
        "expected_physical_specimen_count": expected_physical_n,
    }


def summarize_blind_reviews(
    reviews: tuple[Mapping[str, object], ...],
    *,
    report_index: Mapping[str, Mapping[str, object]],
) -> dict[str, tuple[dict[str, object], ...]]:
    """Preserve reviewer rows and summarize without inventing consensus."""

    if not isinstance(report_index, Mapping) or any(
        not isinstance(row, Mapping) for row in reviews
    ):
        raise ValueError("blind review input is invalid")
    decisions = []
    seen_reviews = set()
    for row in reviews:
        report_id = str(row.get("report_id", ""))
        reviewer_id = str(row.get("reviewer_id", ""))
        decision = str(row.get("decision", ""))
        report = report_index.get(report_id)
        if (
            not report_id
            or not reviewer_id
            or decision not in BLIND_REVIEW_DECISIONS
            or report is None
        ):
            raise ValueError("blind review row cannot be matched")
        review_key = (report_id, reviewer_id)
        if review_key in seen_reviews:
            raise ValueError("duplicate reviewer/report decision")
        seen_reviews.add(review_key)
        objective = report.get("objective_success")
        human_positive = decision == "DELIVERABLE"
        decisions.append(
            {
                **report,
                "analysis_scope": "BLIND_REPORT_REVIEW",
                "reviewer_id": reviewer_id,
                "decision": decision,
                "problem_type": str(row.get("problem_type") or "NOT_RECORDED"),
                "review_basis": str(row.get("review_basis") or "NOT_RECORDED"),
                "blinding_condition": str(
                    row.get("blinding_condition") or "NOT_RECORDED"
                ),
                "objective_human_disagreement": (
                    None
                    if type(objective) is not bool
                    or decision == "UNABLE_TO_JUDGE"
                    else bool(objective) != human_positive
                ),
            }
        )
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in decisions:
        groups[
            (
                str(row["task"]),
                str(row["method"]),
                str(row["stop_system"]),
                str(row["reference_version"]),
            )
        ].append(row)
    summaries = []
    for (task, method, stop_system, reference_version), group in sorted(
        groups.items()
    ):
        summaries.append(
            {
                "task": task,
                "method": method,
                "stop_system": stop_system,
                "reference_version": reference_version,
                "analysis_scope": "DESCRIPTIVE_BLIND_REPORT_REVIEW",
                "review_count": len(group),
                "report_count": len({str(row["report_id"]) for row in group}),
                "physical_specimen_count": len(
                    {str(row["specimen_key"]) for row in group}
                ),
                "deliverable_count": sum(
                    row["decision"] == "DELIVERABLE" for row in group
                ),
                "needs_further_inspection_count": sum(
                    row["decision"] == "NEEDS_FURTHER_INSPECTION" for row in group
                ),
                "unable_to_judge_count": sum(
                    row["decision"] == "UNABLE_TO_JUDGE" for row in group
                ),
                "consensus": "NOT_COMPUTED",
            }
        )
    disagreements = tuple(
        row for row in decisions if row["objective_human_disagreement"] is not None
    )
    paired_groups: dict[
        tuple[str, str, str, str, str], list[dict[str, object]]
    ] = defaultdict(list)
    for row in decisions:
        if row["method"] == "R_BALANCED_P8" or row["method"] in {
            "BC_S1",
            "BC_S2",
            "BC_S3",
        }:
            paired_groups[
                (
                    str(row["specimen_key"]),
                    str(row["task"]),
                    str(row["stop_system"]),
                    str(row["reviewer_id"]),
                    str(row["reference_version"]),
                )
            ].append(row)
    paired = []
    for (
        specimen_key,
        task,
        stop_system,
        reviewer_id,
        reference_version,
    ), group in sorted(paired_groups.items()):
        p8 = [row for row in group if row["method"] == "R_BALANCED_P8"]
        bc = [row for row in group if row["method"] in {"BC_S1", "BC_S2", "BC_S3"}]
        p8_scores = [
            float(row["decision"] == "DELIVERABLE")
            for row in p8
            if row["decision"] != "UNABLE_TO_JUDGE"
        ]
        bc_scores = [
            float(row["decision"] == "DELIVERABLE")
            for row in bc
            if row["decision"] != "UNABLE_TO_JUDGE"
        ]
        if not p8_scores or not bc_scores:
            continue
        p8_fraction = sum(p8_scores) / len(p8_scores)
        bc_fraction = sum(bc_scores) / len(bc_scores)
        paired.append(
            {
                "specimen_key": specimen_key,
                "task": task,
                "stop_system": stop_system,
                "reviewer_id": reviewer_id,
                "reference_version": reference_version,
                "bc_report_count": len(bc_scores),
                "p8_report_count": len(p8_scores),
                "bc_deliverable_fraction": bc_fraction,
                "p8_deliverable_fraction": p8_fraction,
                "bc_minus_p8_deliverable_fraction": bc_fraction - p8_fraction,
                "analysis_scope": "DESCRIPTIVE_BLIND_REVIEW_PAIRED_SPECIMEN",
            }
        )
    return {
        "report_decisions": tuple(decisions),
        "method_summary": tuple(summaries),
        "paired_specimen_differences": tuple(paired),
        "objective_review_disagreements": disagreements,
    }


def match_human_sessions(
    sessions: tuple[Mapping[str, object], ...],
    *,
    model_rows: tuple[Mapping[str, object], ...],
    expected_geometry: str,
) -> dict[str, tuple[dict[str, object], ...]]:
    """Match only sessions with the same specimen/task/geometry/scoring version."""

    if (
        not expected_geometry
        or any(not isinstance(row, Mapping) for row in sessions)
        or any(not isinstance(row, Mapping) for row in model_rows)
    ):
        raise ValueError("human session matching request is invalid")
    manifest = []
    per_session = []
    comparisons = []
    seen_sessions = set()
    for session in sessions:
        session_id = str(session.get("session_id", ""))
        specimen_key = str(session.get("specimen_key", ""))
        task = str(session.get("task", ""))
        operator_id = str(session.get("operator_id", ""))
        geometry = str(session.get("geometry_version", ""))
        reference_version = str(session.get("reference_version", ""))
        if (
            not session_id
            or session_id in seen_sessions
            or not specimen_key
            or not task
            or not operator_id
            or not reference_version
        ):
            raise ValueError("human session identity is invalid")
        seen_sessions.add(session_id)
        candidates = tuple(
            row
            for row in model_rows
            if row.get("specimen_key") == specimen_key
            and row.get("task") == task
            and row.get("reference_version") == reference_version
            and row.get("method")
            in {"R_BALANCED_P8", "BC_S1", "BC_S2", "BC_S3"}
        )
        reasons = []
        if geometry != expected_geometry:
            reasons.append("GEOMETRY_MISMATCH")
        if not candidates:
            reasons.append("NO_MATCHING_FROZEN_EPISODE")
        status = "COMPARABLE" if not reasons else "UNMATCHED_DATA"
        cost_unit = str(session.get("cost_unit") or "NOT_RECORDED")
        cost_comparable = cost_unit == "NATIVE_RASTER_FRACTION"
        route_cost_unit = str(
            session.get("route_cost_unit") or "NOT_RECORDED"
        )
        route_cost_comparable = (
            status == "COMPARABLE"
            and route_cost_unit == "FROZEN_GRID_ROUTE_COST"
        )
        information_permission = str(
            session.get("information_permission") or "NOT_RECORDED"
        )
        report_production = str(
            session.get("report_production") or "NOT_RECORDED"
        )
        action_sequence = session.get("action_sequence")
        action_sequence_recorded = action_sequence not in (None, "", [], ())
        first_stop_or_handoff = session.get("first_stop_or_handoff")
        first_stop_or_handoff_recorded = first_stop_or_handoff not in (
            None,
            "",
        )
        final_report_identity = (
            session.get("final_report_id")
            or session.get("final_report_sha256")
            or "NOT_RECORDED"
        )
        final_report_recorded = final_report_identity != "NOT_RECORDED"
        process_trace_comparable = (
            status == "COMPARABLE"
            and action_sequence_recorded
            and first_stop_or_handoff_recorded
        )
        pure_planner_comparable = (
            status == "COMPARABLE"
            and information_permission == "MATCHED_TO_FROZEN_MODEL"
            and report_production == "COMMON_READER_UNEDITED"
        )
        manifest.append(
            {
                "analysis_scope": "HUMAN_SESSION_COMPARABILITY",
                "session_id": session_id,
                "specimen_key": specimen_key,
                "task": task,
                "operator_id": operator_id,
                "reference_version": reference_version,
                "geometry_version": geometry,
                "status": status,
                "reasons": "|".join(reasons),
                "outcome_comparable": status == "COMPARABLE",
                "cost_comparable": status == "COMPARABLE" and cost_comparable,
                "cost_limitation": "" if cost_comparable else "COST_UNIT_NOT_COMPARABLE",
                "route_cost_comparable": route_cost_comparable,
                "route_cost_limitation": (
                    ""
                    if route_cost_comparable
                    else "ROUTE_COST_UNIT_NOT_COMPARABLE"
                ),
                "information_permission": information_permission,
                "report_production": report_production,
                "action_sequence_recorded": action_sequence_recorded,
                "first_stop_or_handoff_recorded": (
                    first_stop_or_handoff_recorded
                ),
                "final_report_recorded": final_report_recorded,
                "process_trace_comparable": process_trace_comparable,
                "pure_planner_comparable": pure_planner_comparable,
            }
        )
        per_session.append(
            {
                **session,
                "analysis_scope": "HUMAN_SESSION_RECORDED_RESULT",
                "status": status,
                "matched_frozen_method_count": len(candidates) if not reasons else 0,
                "information_permission": information_permission,
                "report_production": report_production,
                "action_sequence_recorded": action_sequence_recorded,
                "first_stop_or_handoff_recorded": (
                    first_stop_or_handoff_recorded
                ),
                "final_report_identity": final_report_identity,
                "process_trace_comparable": process_trace_comparable,
                "pure_planner_comparable": pure_planner_comparable,
            }
        )
        if status == "COMPARABLE":
            for model in candidates:
                comparisons.append(
                    {
                        "session_id": session_id,
                        "specimen_key": specimen_key,
                        "task": task,
                        "operator_id": operator_id,
                        "reference_version": reference_version,
                        "model_method": model["method"],
                        "model_seed": model["seed"],
                        "model_stop_system": model["stop_system"],
                        "analysis_scope": (
                            "MATCHED_HUMAN_PLANNER_COMPARISON"
                            if pure_planner_comparable
                            else "MATCHED_HUMAN_FULL_SYSTEM_COMPARISON"
                        ),
                        "information_permission": information_permission,
                        "report_production": report_production,
                        "process_trace_comparable": process_trace_comparable,
                        "pure_planner_comparable": pure_planner_comparable,
                        "final_report_identity": final_report_identity,
                        "human_completion": session.get("completion"),
                        "model_completion": model.get("completion"),
                        "human_false_stop": session.get("false_stop"),
                        "model_false_stop": model.get("false_stop"),
                        "human_measurement_cost": (
                            session.get("measurement_cost") if cost_comparable else None
                        ),
                        "model_measurement_cost": (
                            model.get("prefix_measurement_cost")
                            if cost_comparable
                            else None
                        ),
                        "model_failure_penalized_cost": (
                            model.get("failure_penalized_cost") if cost_comparable else None
                        ),
                        "human_route_cost": (
                            session.get("route_cost")
                            if route_cost_comparable
                            else None
                        ),
                        "model_route_cost": (
                            model.get("prefix_route_cost")
                            if route_cost_comparable
                            else None
                        ),
                        "comparison_scope": (
                            "OUTCOME_MEASUREMENT_AND_ROUTE_COST"
                            if cost_comparable and route_cost_comparable
                            else (
                                "OUTCOME_AND_MEASUREMENT_COST"
                                if cost_comparable
                                else (
                                    "OUTCOME_AND_ROUTE_COST"
                                    if route_cost_comparable
                                    else "OUTCOME_ONLY_COST_NOT_COMPARABLE"
                                )
                            )
                        ),
                    }
                )
    return {
        "comparability_manifest": tuple(manifest),
        "per_session_results": tuple(per_session),
        "matched_method_comparison": tuple(comparisons),
    }


def _human_model_rows(
    proxy_stop_rows: tuple[Mapping[str, object], ...],
    reviewed_episode_rows: tuple[Mapping[str, object], ...],
) -> tuple[Mapping[str, object], ...]:
    if any(
        not isinstance(row, Mapping)
        for row in (*proxy_stop_rows, *reviewed_episode_rows)
    ):
        raise TypeError("human model row pool is invalid")
    return proxy_stop_rows + reviewed_episode_rows


def _reference_inputs(
    config: FrozenProcessConfig,
    *,
    source_root: str | Path,
    references_path: str | Path | None,
) -> dict[str, object]:
    cohort = pl.read_csv(
        config.source_result_root / "cohort_and_reference_coverage.csv",
        infer_schema_length=None,
    ).filter(pl.col("split") == "TEST")
    cohort_rows = cohort.sort("specimen_key").to_dicts()
    if references_path is None:
        return {
            "references": {},
            "reference_version": "REVIEWED_PENDING",
            "coverage": tuple(
                {
                    "schema_version": 1,
                    "dataset_id": row["dataset_id"],
                    "specimen_id": row["specimen_id"],
                    "specimen_key": row["specimen_key"],
                    "split": "TEST",
                    "reference_version": "REVIEWED_PENDING",
                    "reference_path": "",
                    "reference_sha256": "",
                    "reviewer_alias": "",
                    "formal_eligible": False,
                    "status": "PENDING_USER_INPUT",
                }
                for row in cohort_rows
            ),
            "input_status": "PENDING_USER_INPUT",
            "input_path": "",
            "input_sha256": "",
        }
    root = Path(references_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("reviewed reference input must be a directory")
    parent = load_study_config(config.parent_config_path, project_root=config.project_root)
    context = load_study_context(parent, source_root=source_root)
    records = {
        assignment.record.specimen_key: assignment.record
        for assignment in context.roster.assignments
        if assignment.split is Split.TEST
    }
    all_records = {
        assignment.record.specimen_key: assignment.record
        for assignment in context.roster.assignments
    }
    references = {}
    file_by_key = {}
    no_certain_by_key = {}
    unmatched = []
    seen_keys = set()
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.json")):
        raw = path.read_bytes()
        file_sha = hashlib.sha256(raw).hexdigest()
        digest.update(path.name.encode())
        digest.update(raw)
        payload = json.loads(raw)
        specimen_key = str(payload.get("specimen_key", ""))
        if not specimen_key or specimen_key in seen_keys:
            raise ValueError("reviewed reference specimen identity is missing or duplicated")
        seen_keys.add(specimen_key)
        record = all_records.get(specimen_key)
        if record is None:
            dataset_id, _, specimen_id = specimen_key.partition(":")
            unmatched.append(
                {
                    "schema_version": 1,
                    "dataset_id": dataset_id,
                    "specimen_id": specimen_id,
                    "specimen_key": specimen_key,
                    "split": "UNMATCHED",
                    "reference_version": "",
                    "reference_path": path.name,
                    "reference_sha256": file_sha,
                    "reviewer_alias": str(payload.get("reviewer_alias") or ""),
                    "formal_eligible": False,
                    "status": "UNMATCHED_SPECIMEN_KEY",
                }
            )
            continue
        reference = reference_from_payload(payload, native_shape=record.native_shape)
        if (
            not reference.formal_eligible
            or reference.specimen_key != specimen_key
            or reference.source_image_sha256 != record.cscan_sha256
        ):
            raise ValueError(f"reviewed reference identity changed: {specimen_key}")
        if specimen_key in records:
            file_by_key[specimen_key] = (path, file_sha)
            if np.any(reference.certain_mask):
                references[specimen_key] = reference
            else:
                no_certain_by_key[specimen_key] = reference
        else:
            unmatched.append(
                {
                    "schema_version": 1,
                    "dataset_id": record.dataset_id,
                    "specimen_id": record.specimen_id,
                    "specimen_key": specimen_key,
                    "split": next(
                        assignment.split.value
                        for assignment in context.roster.assignments
                        if assignment.record.specimen_key == specimen_key
                    ),
                    "reference_version": "",
                    "reference_path": path.name,
                    "reference_sha256": file_sha,
                    "reviewer_alias": str(reference.reviewer_alias),
                    "formal_eligible": True,
                    "status": "UNMATCHED_FROZEN_EPISODES",
                }
            )
    version = "REVIEWED_" + digest.hexdigest()[:16]
    coverage = []
    for row in cohort_rows:
        key = str(row["specimen_key"])
        reference = references.get(key)
        no_certain = no_certain_by_key.get(key)
        file_info = file_by_key.get(key)
        coverage.append(
            {
                "schema_version": 1,
                "dataset_id": row["dataset_id"],
                "specimen_id": row["specimen_id"],
                "specimen_key": key,
                "split": "TEST",
                "reference_version": version,
                "reference_path": (
                    "" if file_info is None else file_info[0].name
                ),
                "reference_sha256": (
                    "" if file_info is None else file_info[1]
                ),
                "reviewer_alias": (
                    ""
                    if reference is None and no_certain is None
                    else str((reference or no_certain).reviewer_alias)
                ),
                "formal_eligible": reference is not None,
                "status": (
                    "REVIEWED_REFERENCE_MATCHED"
                    if reference is not None
                    else (
                        "REVIEWED_NO_CERTAIN_REGION"
                        if no_certain is not None
                        else "MISSING_REVIEWED_REFERENCE"
                    )
                ),
            }
        )
    coverage.extend({**row, "reference_version": version} for row in unmatched)
    return {
        "references": references,
        "reference_version": version,
        "coverage": tuple(coverage),
        "input_status": (
            "FULL_TEST_REVIEWED_REFERENCE"
            if len(references) == len(records)
            else (
                "PARTIAL_TEST_REVIEWED_REFERENCE"
                if references
                else (
                    "NO_EVALUABLE_REVIEWED_REFERENCE"
                    if no_certain_by_key
                    else (
                        "UNMATCHED_FROZEN_EPISODES"
                        if unmatched
                        else "PENDING_USER_INPUT"
                    )
                )
            )
        ),
        "input_path": str(root),
        "input_sha256": digest.hexdigest(),
    }


def _report_id(row: Mapping[str, object], stop_system: str, stop_step: int) -> str:
    payload = {
        "specimen_key": row["specimen_key"],
        "task": row["task"],
        "method": row["method"],
        "seed": row["seed"],
        "stop_system": stop_system,
        "stop_step": stop_step,
        "report_sha256": row["report_sha256"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reviewed_episode_rows(
    trajectories: pl.DataFrame,
    report_scores: tuple[Mapping[str, object], ...],
    reference_version: str,
) -> tuple[dict[str, object], ...]:
    score_frames = pl.DataFrame(report_scores).partition_by(
        ["specimen_key", "task", "method", "seed"], maintain_order=True
    )
    trajectory_index = {
        (
            str(frame["specimen_key"][0]),
            str(frame["task"][0]),
            str(frame["method"][0]),
            int(frame["seed"][0]),
        ): tuple(frame.sort("step").to_dicts())
        for frame in trajectories.partition_by(
            ["specimen_key", "task", "method", "seed"], maintain_order=True
        )
    }
    output = []
    for score_frame in score_frames:
        scores = tuple(score_frame.sort("step").to_dicts())
        first = scores[0]
        key = (
            str(first["specimen_key"]),
            str(first["task"]),
            str(first["method"]),
            int(first["seed"]),
        )
        source_rows = trajectory_index[key]
        formal = {
            str(row["report_sha256"]): bool(row["formal_success"])
            for row in scores
        }
        reviewed_rows = tuple(
            {
                **source,
                "success": bool(score["formal_success"]),
                "iou": float(score["iou"]),
                "recall": float(score["recall"]),
                "relative_area_error": float(score["relative_area_error"]),
                "reference_version": reference_version,
            }
            for source, score in zip(source_rows, scores, strict=True)
        )
        for stop_system in STOP_SYSTEMS:
            result = rescore_frozen_episode(
                reviewed_rows,
                formal_success_by_report=formal,
                stop_system=stop_system,
            )
            stop_step = result["stop_step"]
            stop_source = (
                None if stop_step is None else reviewed_rows[int(stop_step)]
            )
            output.append(
                {
                    "schema_version": 1,
                    "dataset_id": first["dataset_id"],
                    "specimen_id": first["specimen_id"],
                    "specimen_key": first["specimen_key"],
                    "task": first["task"],
                    "method": first["method"],
                    "seed": int(first["seed"]),
                    "stop_system": stop_system,
                    "analysis_scope": "FROZEN_REPORT_REVIEWED_RESCORE",
                    "reference_version": reference_version,
                    "report_id": (
                        None
                        if stop_source is None
                        else _report_id(stop_source, stop_system, int(stop_step))
                    ),
                    **result,
                }
            )
    return tuple(output)


def _reviewed_process_diagnostics(
    trajectories: pl.DataFrame,
    report_scores: tuple[Mapping[str, object], ...],
    *,
    reference_version: str,
    thresholds: Mapping[str, float],
) -> dict[str, tuple[dict[str, object], ...]]:
    trajectory_index = {
        (
            str(frame["specimen_key"][0]),
            str(frame["task"][0]),
            str(frame["method"][0]),
            int(frame["seed"][0]),
        ): tuple(frame.sort("step").to_dicts())
        for frame in trajectories.partition_by(
            ["specimen_key", "task", "method", "seed"], maintain_order=True
        )
    }
    stability_rows = []
    wait_rows = []
    for score_frame in pl.DataFrame(report_scores).partition_by(
        ["specimen_key", "task", "method", "seed"], maintain_order=True
    ):
        scores = tuple(score_frame.sort("step").to_dicts())
        first = scores[0]
        key = (
            str(first["specimen_key"]),
            str(first["task"]),
            str(first["method"]),
            int(first["seed"]),
        )
        source_rows = trajectory_index[key]
        reviewed_rows = tuple(
            {
                **source,
                "success": bool(score["formal_success"]),
                "iou": float(score["iou"]),
                "recall": float(score["recall"]),
                "relative_area_error": float(score["relative_area_error"]),
                "reference_version": reference_version,
            }
            for source, score in zip(source_rows, scores, strict=True)
        )
        identity = {
            "schema_version": 1,
            "dataset_id": first["dataset_id"],
            "specimen_id": first["specimen_id"],
            "specimen_key": first["specimen_key"],
            "task": first["task"],
            "method": first["method"],
            "seed": int(first["seed"]),
            "reference_version": reference_version,
        }
        for stop_system in STOP_SYSTEMS:
            stop = stop_decomposition(reviewed_rows, stop_system)
            stability = report_stability(
                reviewed_rows, stop_step=stop["stop_step"]
            )
            stability_rows.append(
                {
                    **identity,
                    "analysis_scope": "REVIEWED_POST_STOP_SUFFIX_DIAGNOSTIC",
                    "stop_system": stop_system,
                    **stability,
                }
            )
            wait_rows.append(
                {
                    **identity,
                    "analysis_scope": "REVIEWED_AUTONOMOUS_PREFIX",
                    "stop_system": stop_system,
                    **_wait_components(
                        reviewed_rows,
                        stop_system=stop_system,
                        stop=stop,
                        stability=stability,
                        threshold=float(thresholds[str(first["task"])]),
                    ),
                }
            )
    return {
        "report_stability_and_delay": tuple(stability_rows),
        "stop_wait_components": tuple(wait_rows),
    }


def _paired_effect(
    rows: list[dict[str, object]],
    *,
    metric: str,
    task: str,
    treatment: str,
    comparator: str,
    confidence: float,
    seed: int,
    replicates: int,
    method_map: Mapping[str, str],
    negate: bool = False,
) -> dict[str, object]:
    selected = [row for row in rows if row["task"] == task and row["method"] in method_map]
    domains = {str(row["specimen_key"]): str(row["dataset_id"]) for row in selected}
    draws = make_domain_bootstrap_draws(domains, replicates=replicates, seed=seed)
    records = tuple(
        MetricRecord(
            specimen_key=str(row["specimen_key"]),
            domain=str(row["dataset_id"]),
            method=method_map[str(row["method"])],
            task=Task(task),
            seed=int(row["seed"]),
            value=(-float(row[metric]) if negate else float(row[metric])),
        )
        for row in selected
    )
    effect = paired_domain_bootstrap(
        records,
        treatment=treatment,
        comparator=comparator,
        confidence_level=confidence,
        draws=draws,
    )
    return {
        "estimate": effect.estimate,
        "ci_lower": effect.ci_lower,
        "ci_upper": effect.ci_upper,
        "confidence_level": effect.confidence_level,
        "physical_specimen_count": effect.physical_specimen_count,
        "domain_count": effect.domain_count,
        "bootstrap_replicates": effect.replicates,
        "bootstrap_draws_sha256": effect.draws_sha256,
    }


def _reviewed_summaries(
    config: FrozenProcessConfig,
    episode_rows: tuple[dict[str, object], ...],
    *,
    reference_version: str,
) -> dict[str, object]:
    rows = list(episode_rows)
    main_map = {
        "R_BALANCED_P8": "R_BALANCED_P8",
        "BC_S1": "BC_3SEED",
        "BC_S2": "BC_3SEED",
        "BC_S3": "BC_3SEED",
    }
    planner = []
    autonomous = []
    decisions = {}
    for task in config.tasks:
        planner_source = [row for row in rows if row["stop_system"] == "S_BC_CAL"]
        for confidence in (0.95, 0.975):
            effect = _paired_effect(
                planner_source,
                metric="planner_ausc",
                task=task,
                treatment="BC_3SEED",
                comparator="R_BALANCED_P8",
                confidence=confidence,
                seed=config.bootstrap_seed,
                replicates=config.bootstrap_replicates,
                method_map=main_map,
            )
            planner.append(
                {
                    "schema_version": 1,
                    "task": task,
                    "analysis_scope": "REVIEWED_PLANNER_EFFECT",
                    "reference_version": reference_version,
                    "metric": "planner_ausc",
                    "treatment": "BC_3SEED",
                    "comparator": "R_BALANCED_P8",
                    **effect,
                }
            )
        for stop_system in STOP_SYSTEMS:
            source = [row for row in rows if row["stop_system"] == stop_system]
            for confidence in (0.95, 0.975):
                completion = _paired_effect(
                    source,
                    metric="completion",
                    task=task,
                    treatment="BC_3SEED",
                    comparator="R_BALANCED_P8",
                    confidence=confidence,
                    seed=config.bootstrap_seed,
                    replicates=config.bootstrap_replicates,
                    method_map=main_map,
                )
                cost = _paired_effect(
                    source,
                    metric="failure_penalized_cost",
                    task=task,
                    treatment="BC_3SEED",
                    comparator="R_BALANCED_P8",
                    confidence=confidence,
                    seed=config.bootstrap_seed,
                    replicates=config.bootstrap_replicates,
                    method_map=main_map,
                    negate=True,
                )
                for metric, effect, direction in (
                    ("completion_rate_difference", completion, "BC_MINUS_P8"),
                    ("failure_cost_reduction", cost, "P8_MINUS_BC"),
                ):
                    autonomous.append(
                        {
                            "schema_version": 1,
                            "task": task,
                            "stop_system": stop_system,
                            "analysis_scope": "REVIEWED_AUTONOMOUS_EFFECT",
                            "reference_version": reference_version,
                            "metric": metric,
                            "effect_direction": direction,
                            "treatment": "BC_3SEED",
                            "comparator": "R_BALANCED_P8",
                            **effect,
                        }
                    )
        key_rows = [
            row
            for row in autonomous
            if row["task"] == task
            and row["stop_system"] == "S_BC_CAL"
            and row["confidence_level"] == 0.975
        ]
        completion = next(
            row for row in key_rows if row["metric"] == "completion_rate_difference"
        )
        cost = next(row for row in key_rows if row["metric"] == "failure_cost_reduction")
        decisions[task] = evaluate_path_b_decision(
            completion,
            cost,
            physical_n=int(completion["physical_specimen_count"]),
            expected_physical_n=config.physical_specimens,
        )
    risk = []
    groups: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["task"]),
                str(row["method"]),
                int(row["seed"]),
                str(row["stop_system"]),
            )
        ].append(row)
    for (task, method, seed, stop_system), group in sorted(groups.items()):
        risk.append(
            {
                "schema_version": 1,
                "task": task,
                "method": method,
                "seed": seed,
                "stop_system": stop_system,
                "analysis_scope": "REVIEWED_RISK_AND_COST",
                "reference_version": reference_version,
                "physical_specimen_count": len(group),
                "completion_rate": sum(bool(row["completion"]) for row in group)
                / len(group),
                "false_stop_episode_rate": sum(bool(row["false_stop"]) for row in group)
                / len(group),
                "exhaustion_rate": sum(bool(row["exhausted"]) for row in group)
                / len(group),
                "autonomous_ausc": sum(float(row["autonomous_ausc"]) for row in group)
                / len(group),
                "failure_penalized_cost": sum(
                    float(row["failure_penalized_cost"]) for row in group
                )
                / len(group),
            }
        )
    ablations = []
    ablation_maps = (
        (
            "ACTOR_SURFACE_CUES",
            {"BC_S1": "BC_S1", "BC_NO_VLM_S1": "BC_NO_VLM_S1"},
            "BC_NO_VLM_S1",
        ),
        (
            "ACTOR_US_FEEDBACK_CONTENT",
            {
                "BC_S1": "BC_S1",
                "BC_NO_US_FEEDBACK_S1": "BC_NO_US_FEEDBACK_S1",
            },
            "BC_NO_US_FEEDBACK_S1",
        ),
    )
    planner_source = [row for row in rows if row["stop_system"] == "S_BC_CAL"]
    for isolated_input, method_map, comparator in ablation_maps:
        for task in config.tasks:
            for confidence in (0.95, 0.975):
                effect = _paired_effect(
                    planner_source,
                    metric="planner_ausc",
                    task=task,
                    treatment="BC_S1",
                    comparator=comparator,
                    confidence=confidence,
                    seed=config.bootstrap_seed,
                    replicates=config.bootstrap_replicates,
                    method_map=method_map,
                )
                ablations.append(
                    {
                        "schema_version": 1,
                        "task": task,
                        "analysis_scope": "REVIEWED_ACTOR_INPUT_INCREMENT_ONLY",
                        "reference_version": reference_version,
                        "isolated_input": isolated_input,
                        "full_actor": "BC_S1",
                        "ablated_actor": comparator,
                        "metric": "planner_ausc",
                        **effect,
                    }
                )
    return {
        "planner_effects": tuple(planner),
        "autonomous_effects": tuple(autonomous),
        "risk_and_cost": tuple(risk),
        "actor_input_ablations": tuple(ablations),
        "task_decisions": decisions,
    }


def _read_csv_rows(path: str | Path) -> tuple[dict[str, str], ...]:
    source = Path(path).resolve(strict=True)
    with source.open(encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _parse_bool(value: object, *, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"external {field} is not boolean")


def _human_sessions(path: str | Path) -> tuple[dict[str, object], ...]:
    source = Path(path).resolve(strict=True)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if type(payload) is dict:
            payload = payload.get("sessions")
        if type(payload) is not list or any(type(row) is not dict for row in payload):
            raise ValueError("human session JSON is invalid")
        rows = tuple(dict(row) for row in payload)
    else:
        rows = _read_csv_rows(source)
    output = []
    for row in rows:
        converted = dict(row)
        for field in ("completion", "false_stop"):
            converted[field] = _parse_bool(row.get(field), field=field)
        for field in ("measurement_cost", "route_cost"):
            value = row.get(field)
            converted[field] = None if value in (None, "") else float(value)
            if converted[field] is not None and (
                not math.isfinite(converted[field]) or converted[field] < 0.0
            ):
                raise ValueError(f"external {field} is invalid")
        if converted["completion"] and converted["false_stop"]:
            raise ValueError("human completion and false_stop are mutually exclusive")
        output.append(converted)
    return tuple(output)


def _claim_rows(
    config: FrozenProcessConfig,
    *,
    reviewed_status: str,
    reviewed_decisions: Mapping[str, object],
    blind: Mapping[str, object],
    human: Mapping[str, object],
    blind_status: str,
    human_status: str,
) -> tuple[dict[str, object], ...]:
    source = json.loads((config.source_result_root / "summary.json").read_text())
    blind_rows = tuple(blind["report_decisions"])  # type: ignore[arg-type]
    human_rows = tuple(human["matched_method_comparison"])  # type: ignore[arg-type]
    blind_versions = {str(row["reference_version"]) for row in blind_rows}
    human_versions = {str(row["reference_version"]) for row in human_rows}
    claims = [
        {
            "claim_id": "C1_PLANNER_VS_P8",
            "task": "BOTH",
            "exact_claim": "Under the shared proxy Reader, BC planning has higher AUsC than P8 in both tasks.",
            "planned_evidence": "Paired physical-specimen planner AUsC",
            "existing_evidence": "paired_effects.csv",
            "new_evidence": "paired_diagnostic_effects.csv",
            "reference_version": "PROXY_LEGACY",
            "physical_n": 24,
            "seed_count": 3,
            "human_coverage": 0,
            "estimate": "LOCATE +0.097059; CHARACTERIZE +0.040324",
            "interval": "97.5% lower bounds +0.012096 and +0.015609",
            "source_table": "results/bc_cscan_path_b_supplement/paired_effects.csv",
            "status": "SUPPORTED" if source["planner_supported_proxy"] else "NOT_SUPPORTED",
            "permitted_wording": "Positive planning signal under the shared proxy Reader.",
            "prohibited_overstatement": "Independently validated diagnostic superiority.",
        },
        {
            "claim_id": "C2_SURFACE_INPUT",
            "task": "BOTH",
            "exact_claim": "Surface cues add planner AUsC for the frozen seed-1 actor under proxy scoring.",
            "planned_evidence": "Seed-1 frozen input ablation",
            "existing_evidence": "ablation_effects.csv",
            "new_evidence": "reviewed/actor_input_ablations.csv when references arrive",
            "reference_version": "PROXY_LEGACY",
            "physical_n": 24,
            "seed_count": 1,
            "human_coverage": 0,
            "estimate": "LOCATE +0.080795; CHARACTERIZE +0.058830",
            "interval": "97.5% intervals positive",
            "source_table": "results/bc_cscan_path_b_supplement/ablation_effects.csv",
            "status": "SUPPORTED",
            "permitted_wording": "Seed-1 actor input increment under proxy scoring.",
            "prohibited_overstatement": "Whole-system causal or multi-seed proof.",
        },
        {
            "claim_id": "C3_US_FEEDBACK",
            "task": "BOTH",
            "exact_claim": "Dynamic ultrasound content adds planner AUsC for the frozen seed-1 actor under proxy scoring.",
            "planned_evidence": "Seed-1 frozen input ablation",
            "existing_evidence": "ablation_effects.csv",
            "new_evidence": "reviewed/actor_input_ablations.csv when references arrive",
            "reference_version": "PROXY_LEGACY",
            "physical_n": 24,
            "seed_count": 1,
            "human_coverage": 0,
            "estimate": "LOCATE +0.459773; CHARACTERIZE +0.441817",
            "interval": "97.5% intervals positive",
            "source_table": "results/bc_cscan_path_b_supplement/ablation_effects.csv",
            "status": "SUPPORTED",
            "permitted_wording": "Seed-1 actor input increment under proxy scoring.",
            "prohibited_overstatement": "Natural-language reasoning or physical-causal proof.",
        },
    ]
    for task in config.tasks:
        reviewed = reviewed_decisions.get(task)
        pending = reviewed is None or reviewed.get("status") == "PENDING_INPUT"
        claims.append(
            {
                "claim_id": f"C4_PATH_B_{task}",
                "task": task,
                "exact_claim": f"Frozen BC satisfies the completion-noninferiority and cost-improvement Path B criterion for {task}.",
                "planned_evidence": "Reviewed first-STOP paired 97.5% joint criterion",
                "existing_evidence": "Proxy criterion not supported",
                "new_evidence": "reviewed/autonomous_effects.csv",
                "reference_version": (
                    "REVIEWED_PENDING" if pending else reviewed_status
                ),
                "physical_n": 0 if pending else reviewed["physical_specimen_count"],
                "seed_count": 3,
                "human_coverage": 0,
                "estimate": "" if pending else reviewed["joint_path_b_pass"],
                "interval": "97.5% paired domain bootstrap",
                "source_table": "reviewed/task_decisions.json",
                "status": "PENDING_INPUT" if pending else reviewed["status"],
                "permitted_wording": "Report the task-specific decision and reference coverage.",
                "prohibited_overstatement": "Transfer one task's decision to the other or choose a 95% interval.",
            }
        )
    claims.extend(
        (
            {
                "claim_id": "C5_BLIND_DELIVERABILITY",
                "task": "BOTH",
                "exact_claim": "Blind reviewers judge frozen BC reports more deliverable than P8 reports.",
                "planned_evidence": "User-supplied anonymous report reviews",
                "existing_evidence": "None",
                "new_evidence": "human_review/method_summary.csv",
                "reference_version": (
                    next(iter(blind_versions))
                    if len(blind_versions) == 1
                    else ("MIXED" if blind_versions else reviewed_status)
                ),
                "physical_n": len(
                    {str(row["specimen_key"]) for row in blind_rows}
                ),
                "seed_count": 3,
                "human_coverage": len(
                    {str(row["reviewer_id"]) for row in blind_rows}
                ),
                "estimate": (
                    "See paired_specimen_differences.csv"
                    if blind_rows
                    else ""
                ),
                "interval": "",
                "source_table": "human_review/paired_specimen_differences.csv",
                "status": blind_status,
                "permitted_wording": "Only the observed reviewer distribution and coverage.",
                "prohibited_overstatement": "Invented consensus or independent physical sample inflation.",
            },
            {
                "claim_id": "C6_HUMAN_COMPARISON",
                "task": "BOTH",
                "exact_claim": "BC outperforms human planning in comparable replay sessions.",
                "planned_evidence": "Matched user-supplied human sessions",
                "existing_evidence": "None",
                "new_evidence": "human_planning/matched_method_comparison.csv",
                "reference_version": (
                    next(iter(human_versions))
                    if len(human_versions) == 1
                    else ("MIXED" if human_versions else reviewed_status)
                ),
                "physical_n": len(
                    {str(row["specimen_key"]) for row in human_rows}
                ),
                "seed_count": 3,
                "human_coverage": len(
                    {str(row["operator_id"]) for row in human_rows}
                ),
                "estimate": (
                    "See matched_method_comparison.csv" if human_rows else ""
                ),
                "interval": "",
                "source_table": "human_planning/matched_method_comparison.csv",
                "status": human_status,
                "permitted_wording": "Only matched-session descriptive results.",
                "prohibited_overstatement": "Superhuman performance or unmatched cost conversion.",
            },
            {
                "claim_id": "C7_PROCESS_ASSOCIATION",
                "task": "BOTH",
                "exact_claim": "Frozen process diagnostics quantify report instability, STOP delay, cost allocation, and surface-proxy conflict.",
                "planned_evidence": "W2-W5 frozen cohort diagnostics",
                "existing_evidence": "Frozen trajectories",
                "new_evidence": "W2-W5 result tables",
                "reference_version": "PROXY_LEGACY",
                "physical_n": 24,
                "seed_count": 3,
                "human_coverage": 0,
                "estimate": "See analysis_summary.json",
                "interval": "95% exploratory intervals for selected paired effects",
                "source_table": "analysis_summary.json",
                "status": "SUPPORTED",
                "permitted_wording": "Descriptive frozen-path associations.",
                "prohibited_overstatement": "Causal mechanism or guaranteed performance gain.",
            },
            {
                "claim_id": "C8_SCOPE_BOUNDARIES",
                "task": "BOTH",
                "exact_claim": "Current evidence does not establish material generalization, hardware benefit, or natural-language reasoning.",
                "planned_evidence": "External material/hardware/causal studies",
                "existing_evidence": "None",
                "new_evidence": "None in this frozen analysis",
                "reference_version": "NOT_APPLICABLE",
                "physical_n": 0,
                "seed_count": 0,
                "human_coverage": 0,
                "estimate": "",
                "interval": "",
                "source_table": "FINAL_RESULTS_AND_CLAIM_BOUNDARIES.md",
                "status": "NOT_EVALUATED",
                "permitted_wording": "Explicitly state these boundaries.",
                "prohibited_overstatement": "Generalization, hardware efficiency, or language reasoning claims.",
            },
        )
    )
    return tuple(claims)


def finalize_frozen_evidence(
    config: FrozenProcessConfig,
    *,
    source_root: str | Path,
    trajectories: pl.DataFrame,
    proxy_stop_rows: tuple[Mapping[str, object], ...],
    references_path: str | Path | None = None,
    blind_reviews_path: str | Path | None = None,
    human_sessions_path: str | Path | None = None,
) -> dict[str, object]:
    """Assemble all supplied external tracks and explicit pending states."""

    inputs = _reference_inputs(
        config, source_root=source_root, references_path=references_path
    )
    references = inputs["references"]
    reference_version = str(inputs["reference_version"])
    if references:
        cached = _load_cached_reviewed_recovery(
            config,
            current_coverage=inputs["coverage"],  # type: ignore[arg-type]
        )
        cached_keys = set(cached["cached_specimen_keys"])
        fresh_references = {
            key: value for key, value in references.items() if key not in cached_keys
        }
        fresh = (
            recover_reviewed_report_scores(
                config,
                source_root=source_root,
                trajectories=trajectories,
                references=fresh_references,
                reference_version=reference_version,
            )
            if fresh_references
            else {
                **{table: () for table in REVIEWED_RECOVERY_TABLES},
                "recovery": {"stored_action_transition_count": 0},
            }
        )
        recovered = _merge_reviewed_recoveries(
            config,
            reference_version=reference_version,
            cached=cached,
            fresh=fresh,
            total_specimen_count=len(references),
            cached_specimen_count=len(cached_keys),
        )
        episode_rows = _reviewed_episode_rows(
            trajectories,
            recovered["report_scores"],  # type: ignore[arg-type]
            reference_version,
        )
        summaries = _reviewed_summaries(
            config, episode_rows, reference_version=reference_version
        )
        process = _reviewed_process_diagnostics(
            trajectories,
            recovered["report_scores"],  # type: ignore[arg-type]
            reference_version=reference_version,
            thresholds=config.stop_thresholds,
        )
    else:
        recovered = {
            "report_scores": (),
            "full_input_readout": (),
            "surface_reference_agreement": (),
            "spatial_action_statistics": (),
            "spatial_episode_summary": (),
            "recovery": {
                "status": "PENDING_USER_INPUT",
                "world_step_count": 0,
                "actor_forward_calls": 0,
                "stop_forward_calls": 0,
                "vlm_calls": 0,
                "training_updates": 0,
            },
        }
        episode_rows = ()
        summaries = {
            "planner_effects": (),
            "autonomous_effects": (),
            "risk_and_cost": (),
            "actor_input_ablations": (),
            "task_decisions": {
                task: {"status": "PENDING_INPUT"} for task in config.tasks
            },
        }
        process = {
            "report_stability_and_delay": (),
            "stop_wait_components": (),
        }
    report_index = {}
    for row in proxy_stop_rows:
        if row.get("report_id"):
            report_index[str(row["report_id"])] = {
                **row,
                "objective_success": row.get("success_at_stop"),
            }
    for row in episode_rows:
        if row.get("report_id"):
            report_index[str(row["report_id"])] = {
                **row,
                "objective_success": row.get("completion"),
            }
    if blind_reviews_path is None:
        blind_input_identity = {
            "status": "PENDING_USER_INPUT",
            "path": "",
            "sha256": "",
        }
        blind = {
            "input_coverage": (
                {
                    "analysis_scope": "HUMAN_SESSION_COMPARABILITY",
                    "status": "PENDING_USER_INPUT",
                    "input_path": "",
                    "review_count": 0,
                    "matched_review_count": 0,
                },
            ),
            "report_decisions": (),
            "method_summary": (),
            "paired_specimen_differences": (),
            "objective_review_disagreements": (),
            "status": "PENDING_INPUT",
        }
    else:
        blind_source = Path(blind_reviews_path).resolve(strict=True)
        blind_input_identity = {
            "status": "PROVIDED",
            "path": str(blind_source),
            "sha256": _file_sha256(blind_source),
        }
        reviews = _read_csv_rows(blind_reviews_path)
        blind_summary = summarize_blind_reviews(reviews, report_index=report_index)
        blind = {
            "input_coverage": (
                {
                    "status": "PROCESSED",
                    "input_path": str(Path(blind_reviews_path).resolve()),
                    "review_count": len(reviews),
                    "matched_review_count": len(blind_summary["report_decisions"]),
                },
            ),
            **blind_summary,
            "status": "INSUFFICIENT_PRECISION",
        }
    model_rows = _human_model_rows(proxy_stop_rows, episode_rows)
    if human_sessions_path is None:
        human_input_identity = {
            "status": "PENDING_USER_INPUT",
            "path": "",
            "sha256": "",
        }
        human = {
            "comparability_manifest": (
                {
                    "status": "PENDING_USER_INPUT",
                    "session_id": "",
                    "specimen_key": "",
                    "task": "",
                    "operator_id": "",
                    "reference_version": reference_version,
                    "geometry_version": "",
                    "reasons": "",
                    "outcome_comparable": False,
                    "cost_comparable": False,
                    "cost_limitation": "",
                    "route_cost_comparable": False,
                    "route_cost_limitation": "",
                    "information_permission": "NOT_RECORDED",
                    "report_production": "NOT_RECORDED",
                    "action_sequence_recorded": False,
                    "first_stop_or_handoff_recorded": False,
                    "final_report_recorded": False,
                    "process_trace_comparable": False,
                    "pure_planner_comparable": False,
                },
            ),
            "per_session_results": (),
            "matched_method_comparison": (),
            "status": "PENDING_INPUT",
        }
    else:
        human_source = Path(human_sessions_path).resolve(strict=True)
        human_input_identity = {
            "status": "PROVIDED",
            "path": str(human_source),
            "sha256": _file_sha256(human_source),
        }
        sessions = _human_sessions(human_sessions_path)
        matched = match_human_sessions(
            sessions,
            model_rows=tuple(model_rows),
            expected_geometry="FROZEN_NATIVE_8X8_THREE_LEVEL",
        )
        human = {
            **matched,
            "status": "INSUFFICIENT_PRECISION",
        }
    claims = _claim_rows(
        config,
        reviewed_status=reference_version,
        reviewed_decisions=summaries["task_decisions"],  # type: ignore[arg-type]
        blind=blind,
        human=human,
        blind_status=str(blind["status"]),
        human_status=str(human["status"]),
    )
    all_inputs = bool(references) and blind_reviews_path is not None and human_sessions_path is not None
    return {
        "reference_coverage": inputs["coverage"],
        "reference_input_status": inputs["input_status"],
        "reference_version": reference_version,
        "reviewed_report_scores": recovered["report_scores"],
        "reviewed_full_input_readout": recovered["full_input_readout"],
        "reviewed_surface_reference_agreement": recovered[
            "surface_reference_agreement"
        ],
        "reviewed_spatial_action_statistics": recovered[
            "spatial_action_statistics"
        ],
        "reviewed_spatial_episode_summary": recovered[
            "spatial_episode_summary"
        ],
        "reviewed_per_episode_metrics": episode_rows,
        "reviewed_report_stability_and_delay": process[
            "report_stability_and_delay"
        ],
        "reviewed_stop_wait_components": process["stop_wait_components"],
        "reviewed_planner_effects": summaries["planner_effects"],
        "reviewed_autonomous_effects": summaries["autonomous_effects"],
        "reviewed_risk_and_cost": summaries["risk_and_cost"],
        "reviewed_actor_input_ablations": summaries["actor_input_ablations"],
        "reviewed_task_decisions": summaries["task_decisions"],
        "reviewed_recovery": recovered["recovery"],
        "human_review": blind,
        "human_planning": human,
        "external_input_identities": {
            "reviewed_references": {
                "status": inputs["input_status"],
                "path": inputs["input_path"],
                "sha256": inputs["input_sha256"],
            },
            "blind_reviews": blind_input_identity,
            "human_sessions": human_input_identity,
        },
        "claim_rows": claims,
        "final_evidence_manifest": {
            "schema_version": 1,
            "stage": (
                "EVIDENCE_ASSEMBLY_COMPLETE"
                if all_inputs
                else "COMPUTATION_COMPLETE_INPUT_PENDING"
            ),
            "computational_implementation": "W0_W8_IMPLEMENTED",
            "external_evidence": {
                "reviewed_references": inputs["input_status"],
                "blind_reviews": blind["status"],
                "human_sessions": human["status"],
            },
            "external_input_identities": {
                "reviewed_references": {
                    "status": inputs["input_status"],
                    "path": inputs["input_path"],
                    "sha256": inputs["input_sha256"],
                },
                "blind_reviews": blind_input_identity,
                "human_sessions": human_input_identity,
            },
            "scientific_claim_statuses": {
                str(row["claim_id"]): row["status"] for row in claims
            },
            "training_updates": 0,
            "vlm_calls": 0,
            "actor_forward_calls": 0,
            "stop_forward_calls": 0,
        },
    }


__all__ = [
    "BLIND_REVIEW_DECISIONS",
    "CLAIM_STATUSES",
    "evaluate_path_b_decision",
    "finalize_frozen_evidence",
    "match_human_sessions",
    "rescore_frozen_episode",
    "summarize_blind_reviews",
]
