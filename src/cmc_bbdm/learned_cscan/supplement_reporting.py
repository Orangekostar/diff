"""Reference, analysis, and reporting stages for the BC supplement."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.vlm_cscan.references import (
    evaluate_task_report,
    reference_from_payload,
)

from .artifacts import write_csv_atomic, write_json_atomic
from .bc_supplement import (
    SupplementConfig,
    load_supplement_config,
)
from .benchmark import (
    _advance_detailed,
    _load_percepts,
    _packet,
    _report_digest,
)
from .contracts import Task
from .metrics import MetricRecord
from .runtime import (
    load_study_config,
    load_study_context,
    open_study_specimen,
)
from .supplement_adapters import adapt_task_report_v2
from .supplement_analysis import (
    make_domain_bootstrap_draws,
    paired_domain_bootstrap,
)
from .supplement_figures import render_supplement_figures

ANNOTATION_QUEUE_FIELDS = (
    "schema_version",
    "cohort",
    "dataset_id",
    "specimen_id",
    "specimen_key",
    "split",
    "priority",
    "reference_version",
    "reference_type",
    "review_state",
    "reviewer_alias",
    "formal_eligible",
    "annotation_path",
    "annotation_exists",
    "blinding_contract",
    "status",
)

CONFIRMATION_FIELDS = (
    "schema_version",
    "cohort",
    "dataset_id",
    "specimen_id",
    "specimen_key",
    "domain_rank",
    "selection_seed",
    "selection_sha256",
    "prior_pilot_member",
    "registered_cscan_sha256",
    "surface_sha256",
    "annotation_path",
    "reference_version",
    "review_state",
    "formal_eligible",
    "evaluation_status",
)

REVIEWED_RESCORE_FIELDS = (
    "schema_version",
    "dataset_id",
    "specimen_id",
    "specimen_key",
    "task",
    "method",
    "seed",
    "step",
    "cost",
    "report_sha256",
    "rule_stop",
    "calibrated_stop_trigger",
    "reference_version",
    "formal_success",
    "iou",
    "recall",
    "relative_area_error",
    "status",
)

PAIRED_EFFECT_FIELDS = (
    "schema_version",
    "analysis",
    "task",
    "metric",
    "treatment",
    "comparator",
    "effect_direction",
    "estimate",
    "ci_lower",
    "ci_upper",
    "confidence_level",
    "physical_specimen_count",
    "domain_count",
    "treatment_seed_count",
    "comparator_seed_count",
    "bootstrap_replicates",
    "bootstrap_draws_sha256",
    "reference_version",
    "formal_estimate",
)

PER_DOMAIN_FIELDS = (
    "schema_version",
    "analysis",
    "task",
    "metric",
    "treatment",
    "comparator",
    "effect_direction",
    "dataset_id",
    "estimate",
    "physical_specimen_count",
    "wins",
    "ties",
    "losses",
    "seed_effect_min",
    "seed_effect_max",
    "reference_version",
    "formal_estimate",
)

ABLATION_FIELDS = (
    "schema_version",
    "task",
    "metric",
    "full_actor",
    "ablated_actor",
    "isolated_input",
    "effect_direction",
    "estimate",
    "ci_lower",
    "ci_upper",
    "confidence_level",
    "physical_specimen_count",
    "domain_count",
    "seed",
    "bootstrap_replicates",
    "bootstrap_draws_sha256",
    "reference_version",
    "formal_estimate",
    "claim_scope",
)

RISK_COVERAGE_FIELDS = (
    "schema_version",
    "task",
    "planner",
    "seed",
    "stop_system",
    "physical_specimen_count",
    "execution_episode_count",
    "stop_count",
    "completion_count",
    "false_stop_count",
    "exhaustion_count",
    "completion_rate",
    "false_stop_episode_rate",
    "false_stop_ci_lower",
    "false_stop_ci_upper",
    "wrong_among_stops",
    "wrong_ci_lower",
    "wrong_ci_upper",
    "exhaustion_rate",
    "autonomous_ausc",
    "failure_penalized_cost",
    "mean_stop_cost_when_stopped",
    "mean_successful_stop_cost",
    "mean_route_cost_at_stop",
    "mean_route_turns_at_stop",
    "latency_mode",
    "mean_planner_inference_seconds",
    "mean_stop_inference_seconds",
    "reference_version",
    "formal_metrics_available",
)


def _config(
    config_path: Path, *, project_root: Path
) -> SupplementConfig:
    return load_supplement_config(config_path, project_root=project_root)


def _bootstrap_settings(config: SupplementConfig) -> tuple[int, int, tuple[float, ...]]:
    evaluation = config.values["evaluation"]
    return (
        int(evaluation["bootstrap_replicates"]),
        int(evaluation["bootstrap_seed"]),
        tuple(float(value) for value in evaluation["confidence_levels"]),
    )


def _method_records(
    rows: list[dict[str, object]],
    *,
    task: Task,
    method: str,
    source_methods: tuple[str, ...],
    field: str,
) -> tuple[MetricRecord, ...]:
    records = []
    for row in rows:
        if row["task"] != task.value or row["method"] not in source_methods:
            continue
        records.append(
            MetricRecord(
                specimen_key=str(row["specimen_key"]),
                domain=str(row["dataset_id"]),
                method=method,
                task=task,
                seed=int(row["seed"]),
                value=float(row[field]),
            )
        )
    if not records:
        raise RuntimeError(f"analysis rows are missing: {task.value}/{method}/{field}")
    return tuple(records)


def _paired_rows(
    records: tuple[MetricRecord, ...],
    *,
    analysis: str,
    task: Task,
    metric: str,
    treatment: str,
    comparator: str,
    effect_direction: str,
    confidence_levels: tuple[float, ...],
    draws: object,
) -> list[dict[str, object]]:
    treatment_seeds = {row.seed for row in records if row.method == treatment}
    comparator_seeds = {row.seed for row in records if row.method == comparator}
    output = []
    for confidence in confidence_levels:
        result = paired_domain_bootstrap(
            records,
            treatment=treatment,
            comparator=comparator,
            confidence_level=confidence,
            draws=draws,
        )
        output.append(
            {
                "schema_version": 1,
                "analysis": analysis,
                "task": task.value,
                "metric": metric,
                "treatment": treatment,
                "comparator": comparator,
                "effect_direction": effect_direction,
                "estimate": result.estimate,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "confidence_level": result.confidence_level,
                "physical_specimen_count": result.physical_specimen_count,
                "domain_count": result.domain_count,
                "treatment_seed_count": len(treatment_seeds),
                "comparator_seed_count": len(comparator_seeds),
                "bootstrap_replicates": result.replicates,
                "bootstrap_draws_sha256": result.draws_sha256,
                "reference_version": "PROXY_LEGACY",
                "formal_estimate": "",
            }
        )
    return output


def analyze_existing(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Compute direct paired effects from the already frozen TEST table."""

    config = _config(config_path, project_root=project_root)
    rows = pl.read_csv(
        config.source_result_root / "per_episode_metrics.csv"
    ).to_dicts()
    specimens = {
        str(row["specimen_key"]): str(row["dataset_id"])
        for row in rows
        if row["method"] == "L_BC"
    }
    replicates, seed, confidence_levels = _bootstrap_settings(config)
    draws = make_domain_bootstrap_draws(
        specimens, replicates=replicates, seed=seed
    )
    effects = []
    comparators = (
        "R_BALANCED",
        "R_CENTER",
        "R_GEOM",
        "R_LEGACY",
        "R_VLM_OPEN",
    )
    for task in Task:
        bc = _method_records(
            rows,
            task=task,
            method="BC_S1",
            source_methods=("L_BC",),
            field="ausc_any",
        )
        for comparator in comparators:
            rule = _method_records(
                rows,
                task=task,
                method=comparator,
                source_methods=(comparator,),
                field="ausc_any",
            )
            effects.extend(
                _paired_rows(
                    (*bc, *rule),
                    analysis="HISTORICAL_TEST_DIRECT_PAIRED",
                    task=task,
                    metric="planner_ausc_any",
                    treatment="BC_S1",
                    comparator=comparator,
                    effect_direction="BC_MINUS_RULE_POSITIVE_IS_BETTER",
                    confidence_levels=confidence_levels,
                    draws=draws,
                )
            )
    payload = {
        "schema_version": 1,
        "stage": "HISTORICAL_TEST_DIRECT_ANALYSIS_COMPLETE",
        "historical_test_reuse": True,
        "physical_specimens": 24,
        "domains": 6,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bootstrap_draws_sha256": draws.sha256,
        "effects": effects,
        "reference_version": "PROXY_LEGACY",
        "formal_effects": None,
    }
    write_json_atomic(
        config.output_root / "_work/existing_paired_effects.json", payload
    )
    return {
        "stage": payload["stage"],
        "comparisons": len(effects),
        "physical_specimens": 24,
        "formal_effects": None,
    }


def _annotation_file(
    config: SupplementConfig, annotation_path: str
) -> Path:
    relative = Path(annotation_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("annotation path escapes its declared root")
    if relative.parts and relative.parts[0] == "confirmation_annotation_queue":
        return config.output_root / relative
    return config.annotation_root / relative.name


def import_references(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Inventory attributable review coverage without promoting proposals."""

    del source_root
    config = _config(config_path, project_root=project_root)
    coverage = pl.read_csv(
        config.output_root / "cohort_and_reference_coverage.csv"
    ).to_dicts()
    rows = []
    priority = {"TEST": 1, "VALID": 2, "TRAIN": 3}
    reviewed = 0
    for row in coverage:
        annotation_path = str(row["annotation_path"])
        source = _annotation_file(config, annotation_path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        review_state = str(payload.get("review_state", "pending"))
        reviewer_alias = payload.get("reviewer_alias")
        reference_type = str(
            payload.get("reference_type", "ALGORITHM_DERIVED_NOT_REVIEWED")
        )
        formal_eligible = bool(
            review_state == "reviewed"
            and reviewer_alias
            and reference_type in {"EXPERT_REVIEWED", "AUTHOR_PROVIDED"}
        )
        if formal_eligible:
            reviewed += 1
        split = str(row["split"])
        rows.append(
            {
                "schema_version": 1,
                "cohort": "RETROSPECTIVE_PILOT_60",
                "dataset_id": row["dataset_id"],
                "specimen_id": row["specimen_id"],
                "specimen_key": row["specimen_key"],
                "split": split,
                "priority": priority[split],
                "reference_version": (
                    "REVIEWED_V1" if formal_eligible else "PROXY_LEGACY"
                ),
                "reference_type": reference_type,
                "review_state": review_state,
                "reviewer_alias": "" if reviewer_alias is None else reviewer_alias,
                "formal_eligible": formal_eligible,
                "annotation_path": (
                    "results/vlm_cscan_efficiency/annotation_queue/"
                    f"{source.name}"
                ),
                "annotation_exists": source.is_file(),
                "blinding_contract": "FULL_CSCAN_ONLY_NO_MODEL_OUTPUTS",
                "status": (
                    "ATTRIBUTABLE_REVIEW_AVAILABLE"
                    if formal_eligible
                    else "INDEPENDENT_REVIEW_PENDING"
                ),
            }
        )
    confirmation_roster = config.output_root / "confirmation_roster.csv"
    if confirmation_roster.is_file():
        for confirmation in pl.read_csv(confirmation_roster).to_dicts():
            annotation_path = str(confirmation["annotation_path"])
            source = _annotation_file(config, annotation_path)
            payload = json.loads(source.read_text(encoding="utf-8"))
            review_state = str(payload.get("review_state", "pending"))
            reviewer_alias = payload.get("reviewer_alias")
            reference_type = str(
                payload.get(
                    "reference_type", "ALGORITHM_DERIVED_NOT_REVIEWED"
                )
            )
            formal_eligible = bool(
                review_state == "reviewed"
                and reviewer_alias
                and reference_type in {"EXPERT_REVIEWED", "AUTHOR_PROVIDED"}
            )
            reviewed += int(formal_eligible)
            rows.append(
                {
                    "schema_version": 1,
                    "cohort": "ADDITIONAL_BC_CONFIRM_24",
                    "dataset_id": confirmation["dataset_id"],
                    "specimen_id": confirmation["specimen_id"],
                    "specimen_key": confirmation["specimen_key"],
                    "split": "CONFIRM",
                    "priority": 4,
                    "reference_version": (
                        "REVIEWED_V1"
                        if formal_eligible
                        else "REVIEWED_V1_PENDING"
                    ),
                    "reference_type": reference_type,
                    "review_state": review_state,
                    "reviewer_alias": (
                        "" if reviewer_alias is None else reviewer_alias
                    ),
                    "formal_eligible": formal_eligible,
                    "annotation_path": annotation_path,
                    "annotation_exists": source.is_file(),
                    "blinding_contract": "FULL_CSCAN_ONLY_NO_MODEL_OUTPUTS",
                    "status": (
                        "ATTRIBUTABLE_REVIEW_AVAILABLE"
                        if formal_eligible
                        else "INDEPENDENT_REVIEW_PENDING"
                    ),
                }
            )
    rows.sort(key=lambda row: (int(row["priority"]), str(row["specimen_key"])))
    write_csv_atomic(
        config.output_root / "annotation_queue.csv",
        tuple(rows),
        ANNOTATION_QUEUE_FIELDS,
    )
    return {
        "stage": "REFERENCE_COVERAGE_IMPORTED",
        "cohort": len(rows),
        "reviewed_references": reviewed,
        "formal_effects_available": reviewed > 0,
        "proxy_scope": "SAME_READER_SELF_CONSISTENCY",
    }


def _confirmation_candidates(
    config: SupplementConfig, *, source_root: Path
) -> tuple[object, ...]:
    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    context = load_study_context(parent, source_root=source_root)
    pilot = {record.specimen_key for record in context.roster.pilot_records}
    seed = str(config.values["confirmation"]["roster_seed"])
    per_domain = int(config.values["confirmation"]["specimens_per_domain"])
    selected = []
    for domain in sorted({row.dataset_id for row in context.roster.records}):
        candidates = tuple(
            row
            for row in context.roster.records
            if row.dataset_id == domain and row.specimen_key not in pilot
        )
        ranked = sorted(
            candidates,
            key=lambda row: (
                hashlib.sha256(f"{seed}|{row.specimen_key}".encode()).hexdigest(),
                row.specimen_key,
            ),
        )
        selected.extend(ranked[:per_domain])
    if len(selected) != 24 or len({row.specimen_key for row in selected}) != 24:
        raise RuntimeError("confirmation roster is not six domains by four")
    return tuple(selected)


def _write_confirmation_template(
    config: SupplementConfig, record: object, *, source_root: Path
) -> str:
    name = f"{record.dataset_id}__{record.specimen_id}.json"
    relative = Path("confirmation_annotation_queue") / name
    path = config.output_root / relative
    payload = {
        "schema_version": 1,
        "specimen_key": record.specimen_key,
        "frame": "registered_cscan",
        "source_image_sha256": record.cscan_sha256,
        "source_path_from_source_root": record.cscan_path.relative_to(
            source_root.resolve(strict=True)
        ).as_posix(),
        "reference_type": "ALGORITHM_DERIVED_NOT_REVIEWED",
        "review_state": "pending",
        "reviewer_alias": None,
        "regions": [],
        "uncertain_regions": [],
        "proposal_provenance": {
            "status": "BLANK_INDEPENDENT_REVIEW_TEMPLATE",
            "algorithm": None,
            "parameters": {},
        },
        "blinding_contract": "FULL_CSCAN_ONLY_NO_MODEL_OUTPUTS",
        "review_instructions": (
            "Inspect the registered C-scan without model trajectories or method "
            "outcomes. Edit regions, preserve uncertain boundaries, set an "
            "attributable reviewer_alias, and only then mark reviewed."
        ),
        "notes": "",
    }
    write_json_atomic(path, payload)
    return relative.as_posix()


def prepare_confirmation(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Prepare a blinded 24-specimen queue without running any model."""

    config = _config(config_path, project_root=project_root)
    selected = _confirmation_candidates(config, source_root=source_root)
    seed = str(config.values["confirmation"]["roster_seed"])
    rows = []
    queue_rows = [
        row
        for row in (
            pl.read_csv(config.output_root / "annotation_queue.csv").to_dicts()
            if (config.output_root / "annotation_queue.csv").is_file()
            else []
        )
        if row["cohort"] != "ADDITIONAL_BC_CONFIRM_24"
    ]
    by_domain: dict[str, int] = {}
    for record in selected:
        rank = by_domain.get(record.dataset_id, 0)
        by_domain[record.dataset_id] = rank + 1
        selection_sha = hashlib.sha256(
            f"{seed}|{record.specimen_key}".encode()
        ).hexdigest()
        annotation = _write_confirmation_template(
            config, record, source_root=source_root
        )
        rows.append(
            {
                "schema_version": 1,
                "cohort": "ADDITIONAL_BC_CONFIRM_24",
                "dataset_id": record.dataset_id,
                "specimen_id": record.specimen_id,
                "specimen_key": record.specimen_key,
                "domain_rank": rank,
                "selection_seed": seed,
                "selection_sha256": selection_sha,
                "prior_pilot_member": False,
                "registered_cscan_sha256": record.cscan_sha256,
                "surface_sha256": record.surface_sha256,
                "annotation_path": annotation,
                "reference_version": "REVIEWED_V1_PENDING",
                "review_state": "pending",
                "formal_eligible": False,
                "evaluation_status": (
                    "ADDITIONAL_CONFIRMATION_PENDING_REFERENCE"
                ),
            }
        )
        queue_rows.append(
            {
                "schema_version": 1,
                "cohort": "ADDITIONAL_BC_CONFIRM_24",
                "dataset_id": record.dataset_id,
                "specimen_id": record.specimen_id,
                "specimen_key": record.specimen_key,
                "split": "CONFIRM",
                "priority": 4,
                "reference_version": "REVIEWED_V1_PENDING",
                "reference_type": "ALGORITHM_DERIVED_NOT_REVIEWED",
                "review_state": "pending",
                "reviewer_alias": "",
                "formal_eligible": False,
                "annotation_path": annotation,
                "annotation_exists": True,
                "blinding_contract": "FULL_CSCAN_ONLY_NO_MODEL_OUTPUTS",
                "status": "INDEPENDENT_REVIEW_PENDING",
            }
        )
    rows.sort(key=lambda row: (str(row["dataset_id"]), int(row["domain_rank"])))
    queue_rows.sort(
        key=lambda row: (int(row["priority"]), str(row["specimen_key"]))
    )
    write_csv_atomic(
        config.output_root / "confirmation_roster.csv",
        tuple(rows),
        CONFIRMATION_FIELDS,
    )
    write_csv_atomic(
        config.output_root / "annotation_queue.csv",
        tuple(queue_rows),
        ANNOTATION_QUEUE_FIELDS,
    )
    return {
        "stage": "ADDITIONAL_CONFIRMATION_PREPARED",
        "cohort": "ADDITIONAL_BC_CONFIRM_24",
        "specimens": len(rows),
        "domains": len(by_domain),
        "reviewed_references": 0,
        "world_transitions": 0,
        "vlm_calls": 0,
        "evaluation_status": "ADDITIONAL_CONFIRMATION_PENDING_REFERENCE",
    }


def _replay_reviewed_reports(
    config: SupplementConfig,
    *,
    source_root: Path,
    references: dict[str, object],
    trajectories: pl.DataFrame,
) -> list[dict[str, object]]:
    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    context = load_study_context(parent, source_root=source_root)
    percepts = _load_percepts(parent, context, config.source_result_root)
    records = {record.specimen_key: record for record in context.roster.pilot_records}
    output = []
    for specimen_key in sorted(references):
        record = records.get(specimen_key)
        if record is None:
            continue
        runtime = open_study_specimen(context, record)
        specimen = trajectories.filter(pl.col("specimen_key") == specimen_key)
        episodes = specimen.partition_by(
            ["task", "method", "seed"], maintain_order=True
        )
        if len(episodes) != 14:
            raise RuntimeError(
                f"stored TEST episode matrix changed for {specimen_key}"
            )
        for episode in episodes:
            episode = episode.sort("step")
            rows = episode.to_dicts()
            if len(rows) != 193:
                raise RuntimeError(
                    f"stored TEST trajectory length changed for {specimen_key}"
                )
            task = Task(str(rows[0]["task"]))
            observation = runtime.world.reset()
            probe = (0.0, 0.0)
            route_cost = 0.0
            for index, row in enumerate(rows):
                if int(row["step"]) != index:
                    raise RuntimeError("stored TEST trajectory order changed")
                packet = _packet(
                    observation,
                    runtime=runtime,
                    percept=percepts[specimen_key],
                    task=task,
                    context=context,
                    probe_position=probe,
                    route_cost=route_cost,
                )
                if (
                    _report_digest(packet.report) != row["report_sha256"]
                    or packet.feature_sha256 != row["packet_sha256"]
                    or abs(float(observation.effective_budget) - float(row["cost"]))
                    > 1e-12
                ):
                    raise RuntimeError("stored report replay identity changed")
                score = evaluate_task_report(
                    adapt_task_report_v2(packet.report), references[specimen_key]
                )
                if score.formal_success is None:
                    raise RuntimeError("reviewed rescore lost formal eligibility")
                output.append(
                    {
                        "schema_version": 1,
                        "dataset_id": row["dataset_id"],
                        "specimen_id": row["specimen_id"],
                        "specimen_key": specimen_key,
                        "task": task.value,
                        "method": row["method"],
                        "seed": int(row["seed"]),
                        "step": index,
                        "cost": float(row["cost"]),
                        "report_sha256": row["report_sha256"],
                        "rule_stop": bool(row["rule_stop"]),
                        "calibrated_stop_trigger": bool(
                            row["calibrated_stop_trigger"]
                        ),
                        "reference_version": "REVIEWED_V1",
                        "formal_success": bool(score.formal_success),
                        "iou": score.iou,
                        "recall": score.recall,
                        "relative_area_error": score.relative_area_error,
                        "status": "FROZEN_REPORT_PREFIX_RESCORED",
                    }
                )
                action_cell = int(row["action_cell"])
                if action_cell < 0:
                    if index != len(rows) - 1:
                        raise RuntimeError("stored terminal action is misplaced")
                    continue
                action = InspectionCellAction(
                    action_cell,
                    int(row["action_from_level"]),
                    int(row["action_to_level"]),
                )
                observation, probe, route_cost, _turns = _advance_detailed(
                    runtime, observation, action, probe, route_cost
                )
    return output


def rescore_references(
    *,
    config_path: Path,
    project_root: Path,
    source_root: Path,
    references: Path,
) -> dict[str, object]:
    """Rescore frozen TEST report prefixes when reviewed references arrive."""

    config = _config(config_path, project_root=project_root)
    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    context = load_study_context(parent, source_root=source_root)
    records = {record.specimen_key: record for record in context.roster.records}
    trajectory_path = config.output_root / "trajectories.parquet"
    if not trajectory_path.is_file():
        raise RuntimeError("run frozen TEST evaluation before reference rescoring")
    trajectories = pl.read_parquet(trajectory_path)
    test_keys = set(trajectories["specimen_key"].unique().to_list())
    root = references.resolve(strict=True)
    reviewed: dict[str, object] = {}
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = str(payload.get("specimen_key", ""))
        record = records.get(key)
        if record is None or payload.get("review_state") != "reviewed":
            continue
        reference = reference_from_payload(
            payload, native_shape=record.native_shape
        )
        if not reference.formal_eligible:
            continue
        if (
            reference.specimen_key != key
            or reference.source_image_sha256 != record.cscan_sha256
        ):
            raise RuntimeError(f"reviewed reference identity changed: {key}")
        if key in reviewed:
            raise RuntimeError(f"duplicate reviewed reference: {key}")
        reviewed[key] = reference
    reviewed_test = {key: value for key, value in reviewed.items() if key in test_keys}
    rows = _replay_reviewed_reports(
        config,
        source_root=source_root,
        references=reviewed_test,
        trajectories=trajectories,
    )
    write_csv_atomic(
        config.output_root / "reviewed_rescore.csv",
        tuple(rows),
        REVIEWED_RESCORE_FIELDS,
    )
    test_count = len(reviewed_test)
    if test_count == 24:
        state = "FULL_TEST_REVIEWED_RESCORE_AVAILABLE"
    elif test_count:
        state = "PARTIAL_TEST_REVIEWED_RESCORE_AVAILABLE"
    elif reviewed:
        state = "REVIEWED_REFERENCE_OUTSIDE_FROZEN_TEST"
    else:
        state = "INDEPENDENT_REVIEW_PENDING"
    status = {
        "schema_version": 1,
        "stage": "REVIEWED_RESCORE_FROZEN",
        "reference_root": "USER_SUPPLIED_REFERENCE_ROOT",
        "reviewed_reference_count": len(reviewed),
        "reviewed_test_reference_count": test_count,
        "rescored_report_count": len(rows),
        "formal_report_scores_available": bool(rows),
        "formal_effects": None,
        "proxy_values_unchanged": True,
        "model_retrained": False,
        "threshold_recalibrated": False,
        "status": state,
    }
    write_json_atomic(config.output_root / "reviewed_rescore_status.json", status)
    return status


def _wilson(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Wilson interval counts are invalid")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    fraction = successes / total
    denominator = 1.0 + z * z / total
    center = (fraction + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            fraction * (1.0 - fraction) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _risk_fields(stop_system: str) -> dict[str, str]:
    if stop_system == "S_RULE":
        return {
            "stopped": "rule_stopped",
            "success": "rule_stop_success",
            "false": "rule_false_stop",
            "exhausted": "rule_resource_exhausted",
            "cost": "rule_stop_cost",
            "ausc": "rule_autonomous_ausc",
            "failure": "rule_failure_penalized_cost",
            "route": "rule_route_cost_at_stop",
            "turns": "rule_route_turns_at_stop",
            "planner_latency": "rule_planner_inference_seconds",
        }
    if stop_system == "S_BC_CAL":
        return {
            "stopped": "calibrated_stopped",
            "success": "calibrated_stop_success",
            "false": "calibrated_false_stop",
            "exhausted": "calibrated_resource_exhausted",
            "cost": "calibrated_stop_cost",
            "ausc": "calibrated_autonomous_ausc",
            "failure": "calibrated_failure_penalized_cost",
            "route": "calibrated_route_cost",
            "turns": "calibrated_route_turns",
            "planner_latency": "calibrated_planner_inference_seconds",
        }
    if stop_system == "S_OLD_090":
        return {
            "stopped": "learned_stopped",
            "success": "learned_stop_success",
            "false": "learned_false_stop",
            "exhausted": "learned_resource_exhausted",
            "cost": "learned_stop_cost",
            "ausc": "learned_autonomous_ausc",
            "failure": "learned_failure_penalized_cost",
            "route": "",
            "turns": "",
            "planner_latency": "",
        }
    raise ValueError("unknown STOP system")


def _risk_row(
    rows: list[dict[str, object]],
    *,
    task: Task,
    planner: str,
    source_method: str,
    stop_system: str,
    seed: int,
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["task"] == task.value
        and row["method"] == source_method
        and int(row["seed"]) == seed
    ]
    if len(selected) != 24:
        raise RuntimeError(
            f"risk cohort changed: {task.value}/{source_method}/{stop_system}"
        )
    fields = _risk_fields(stop_system)
    stopped = sum(bool(row[fields["stopped"]]) for row in selected)
    completed = sum(bool(row[fields["success"]]) for row in selected)
    false = sum(bool(row[fields["false"]]) for row in selected)
    exhausted = sum(bool(row[fields["exhausted"]]) for row in selected)
    if stopped != completed + false or len(selected) != stopped + exhausted:
        raise RuntimeError("STOP episode accounting is inconsistent")
    false_interval = _wilson(false, len(selected))
    wrong_interval = _wilson(false, stopped) if stopped else (None, None)
    stopped_costs = [
        float(row[fields["cost"]])
        for row in selected
        if row[fields["cost"]] not in {"", None}
    ]
    successful_costs = [
        float(row[fields["cost"]])
        for row in selected
        if bool(row[fields["success"]])
    ]
    route = fields["route"]
    turns = fields["turns"]
    planner_latency = fields["planner_latency"]
    return {
        "schema_version": 1,
        "task": task.value,
        "planner": planner,
        "seed": seed,
        "stop_system": stop_system,
        "physical_specimen_count": 24,
        "execution_episode_count": 24,
        "stop_count": stopped,
        "completion_count": completed,
        "false_stop_count": false,
        "exhaustion_count": exhausted,
        "completion_rate": completed / len(selected),
        "false_stop_episode_rate": false / len(selected),
        "false_stop_ci_lower": false_interval[0],
        "false_stop_ci_upper": false_interval[1],
        "wrong_among_stops": "" if not stopped else false / stopped,
        "wrong_ci_lower": "" if not stopped else wrong_interval[0],
        "wrong_ci_upper": "" if not stopped else wrong_interval[1],
        "exhaustion_rate": exhausted / len(selected),
        "autonomous_ausc": float(
            np.mean([float(row[fields["ausc"]]) for row in selected])
        ),
        "failure_penalized_cost": float(
            np.mean([float(row[fields["failure"]]) for row in selected])
        ),
        "mean_stop_cost_when_stopped": (
            "" if not stopped_costs else float(np.mean(stopped_costs))
        ),
        "mean_successful_stop_cost": (
            "" if not successful_costs else float(np.mean(successful_costs))
        ),
        "mean_route_cost_at_stop": (
            ""
            if not route
            else float(np.mean([float(row[route]) for row in selected]))
        ),
        "mean_route_turns_at_stop": (
            ""
            if not turns
            else float(np.mean([float(row[turns]) for row in selected]))
        ),
        "latency_mode": (
            "NOT_MEASURED_HISTORICAL"
            if not planner_latency
            else "FULL_TRAJECTORY_PREFIX_PLANNER_ONLY_STOP_LATENCY_NOT_MEASURED"
        ),
        "mean_planner_inference_seconds": (
            ""
            if not planner_latency
            else float(
                np.mean([float(row[planner_latency]) for row in selected])
            )
        ),
        "mean_stop_inference_seconds": "",
        "reference_version": "PROXY_LEGACY",
        "formal_metrics_available": False,
    }


def _records_from_field(
    rows: list[dict[str, object]],
    *,
    task: Task,
    alias: str,
    source_methods: tuple[str, ...],
    field: str,
) -> tuple[MetricRecord, ...]:
    return _method_records(
        rows,
        task=task,
        method=alias,
        source_methods=source_methods,
        field=field,
    )


def _domain_rows(
    records: tuple[MetricRecord, ...],
    *,
    analysis: str,
    task: Task,
    metric: str,
    treatment: str,
    comparator: str,
    effect_direction: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], float] = {
        (row.specimen_key, row.method, row.seed): row.value for row in records
    }
    domains = {row.specimen_key: row.domain for row in records}
    treatment_seeds = sorted(
        {row.seed for row in records if row.method == treatment}
    )
    comparator_seeds = sorted(
        {row.seed for row in records if row.method == comparator}
    )
    output = []
    for domain in sorted(set(domains.values())):
        specimens = sorted(
            specimen for specimen, value in domains.items() if value == domain
        )
        effects = []
        seed_effects = []
        for specimen in specimens:
            treatment_value = float(
                np.mean(
                    [grouped[(specimen, treatment, seed)] for seed in treatment_seeds]
                )
            )
            comparator_value = float(
                np.mean(
                    [grouped[(specimen, comparator, seed)] for seed in comparator_seeds]
                )
            )
            effects.append(treatment_value - comparator_value)
        for seed in treatment_seeds:
            seed_effects.append(
                float(
                    np.mean(
                        [
                            grouped[(specimen, treatment, seed)]
                            - float(
                                np.mean(
                                    [
                                        grouped[(specimen, comparator, candidate)]
                                        for candidate in comparator_seeds
                                    ]
                                )
                            )
                            for specimen in specimens
                        ]
                    )
                )
            )
        output.append(
            {
                "schema_version": 1,
                "analysis": analysis,
                "task": task.value,
                "metric": metric,
                "treatment": treatment,
                "comparator": comparator,
                "effect_direction": effect_direction,
                "dataset_id": domain,
                "estimate": float(np.mean(effects)),
                "physical_specimen_count": len(specimens),
                "wins": sum(value > 1e-12 for value in effects),
                "ties": sum(abs(value) <= 1e-12 for value in effects),
                "losses": sum(value < -1e-12 for value in effects),
                "seed_effect_min": min(seed_effects),
                "seed_effect_max": max(seed_effects),
                "reference_version": "PROXY_LEGACY",
                "formal_estimate": "",
            }
        )
    return output


def _append_effect(
    destination: list[dict[str, object]],
    domain_destination: list[dict[str, object]],
    *,
    records: tuple[MetricRecord, ...],
    analysis: str,
    task: Task,
    metric: str,
    treatment: str,
    comparator: str,
    effect_direction: str,
    confidence_levels: tuple[float, ...],
    draws: object,
    include_domains: bool = True,
) -> None:
    destination.extend(
        _paired_rows(
            records,
            analysis=analysis,
            task=task,
            metric=metric,
            treatment=treatment,
            comparator=comparator,
            effect_direction=effect_direction,
            confidence_levels=confidence_levels,
            draws=draws,
        )
    )
    if include_domains:
        domain_destination.extend(
            _domain_rows(
                records,
                analysis=analysis,
                task=task,
                metric=metric,
                treatment=treatment,
                comparator=comparator,
                effect_direction=effect_direction,
            )
        )


def _find_effect(
    effects: list[dict[str, object]],
    *,
    analysis: str,
    task: str,
    metric: str,
    confidence_level: float,
) -> dict[str, object]:
    matches = [
        row
        for row in effects
        if row["analysis"] == analysis
        and row["task"] == task
        and row["metric"] == metric
        and math.isclose(float(row["confidence_level"]), confidence_level)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"summary effect is not unique: {analysis}/{task}/{metric}")
    return matches[0]


def _assemble_statistics(
    config: SupplementConfig,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    new_rows = pl.read_csv(config.output_root / "per_episode_metrics.csv").to_dicts()
    historical_rows = pl.read_csv(
        config.source_result_root / "per_episode_metrics.csv"
    ).to_dicts()
    specimen_domains = {
        str(row["specimen_key"]): str(row["dataset_id"])
        for row in new_rows
        if row["method"] == "BC_S1"
    }
    if len(specimen_domains) != 24 or len(set(specimen_domains.values())) != 6:
        raise RuntimeError("TEST statistics cohort changed")
    replicates, seed, confidence_levels = _bootstrap_settings(config)
    draws = make_domain_bootstrap_draws(
        specimen_domains, replicates=replicates, seed=seed
    )
    paired: list[dict[str, object]] = []
    domains: list[dict[str, object]] = []
    ablations: list[dict[str, object]] = []

    for task in Task:
        bc = _records_from_field(
            new_rows,
            task=task,
            alias="BC_3SEED",
            source_methods=("BC_S1", "BC_S2", "BC_S3"),
            field="ausc_any",
        )
        planner_comparators = (
            ("R_BALANCED_P4", new_rows, "R_BALANCED_P4"),
            ("R_BALANCED_P8", new_rows, "R_BALANCED_P8"),
            ("R_CENTER", historical_rows, "R_CENTER"),
            ("R_GEOM", historical_rows, "R_GEOM"),
            ("R_LEGACY", historical_rows, "R_LEGACY"),
            ("R_VLM_OPEN", historical_rows, "R_VLM_OPEN"),
        )
        for comparator, source, source_method in planner_comparators:
            rule = _records_from_field(
                source,
                task=task,
                alias=comparator,
                source_methods=(source_method,),
                field="ausc_any",
            )
            _append_effect(
                paired,
                domains,
                records=(*bc, *rule),
                analysis="H_PLAN_BC_3SEED_VS_REGISTERED_RULE",
                task=task,
                metric="planner_ausc_any",
                treatment="BC_3SEED",
                comparator=comparator,
                effect_direction="BC_MINUS_RULE_POSITIVE_IS_BETTER",
                confidence_levels=confidence_levels,
                draws=draws,
            )
        for bc_method in ("BC_S1", "BC_S2", "BC_S3"):
            seed_bc = _records_from_field(
                new_rows,
                task=task,
                alias=bc_method,
                source_methods=(bc_method,),
                field="ausc_any",
            )
            for comparator in ("R_BALANCED_P4", "R_BALANCED_P8"):
                rule = _records_from_field(
                    new_rows,
                    task=task,
                    alias=comparator,
                    source_methods=(comparator,),
                    field="ausc_any",
                )
                _append_effect(
                    paired,
                    domains,
                    records=(*seed_bc, *rule),
                    analysis="H_PLAN_SEED_SPECIFIC",
                    task=task,
                    metric="planner_ausc_any",
                    treatment=bc_method,
                    comparator=comparator,
                    effect_direction="BC_MINUS_RULE_POSITIVE_IS_BETTER",
                    confidence_levels=confidence_levels,
                    draws=draws,
                    include_domains=False,
                )

        for stop_system, prefix in (
            ("S_RULE", "rule"),
            ("S_BC_CAL", "calibrated"),
        ):
            bc_completion = _records_from_field(
                new_rows,
                task=task,
                alias="BC_3SEED",
                source_methods=("BC_S1", "BC_S2", "BC_S3"),
                field=f"{prefix}_stop_success",
            )
            rule_completion = _records_from_field(
                new_rows,
                task=task,
                alias="R_BALANCED_P8",
                source_methods=("R_BALANCED_P8",),
                field=f"{prefix}_stop_success",
            )
            _append_effect(
                paired,
                domains,
                records=(*bc_completion, *rule_completion),
                analysis=f"H_AUTO_BC_VS_P8_{stop_system}",
                task=task,
                metric="completion_rate_difference",
                treatment="BC_3SEED",
                comparator="R_BALANCED_P8",
                effect_direction="BC_MINUS_RULE_POSITIVE_IS_BETTER",
                confidence_levels=confidence_levels,
                draws=draws,
            )
            bc_failure = _records_from_field(
                new_rows,
                task=task,
                alias="BC_3SEED",
                source_methods=("BC_S1", "BC_S2", "BC_S3"),
                field=f"{prefix}_failure_penalized_cost",
            )
            rule_failure = _records_from_field(
                new_rows,
                task=task,
                alias="R_BALANCED_P8",
                source_methods=("R_BALANCED_P8",),
                field=f"{prefix}_failure_penalized_cost",
            )
            _append_effect(
                paired,
                domains,
                records=(*rule_failure, *bc_failure),
                analysis=f"H_AUTO_BC_VS_P8_{stop_system}",
                task=task,
                metric="failure_cost_reduction",
                treatment="R_BALANCED_P8",
                comparator="BC_3SEED",
                effect_direction="RULE_MINUS_BC_POSITIVE_IS_BETTER",
                confidence_levels=confidence_levels,
                draws=draws,
            )
            bc_auto = _records_from_field(
                new_rows,
                task=task,
                alias="BC_3SEED",
                source_methods=("BC_S1", "BC_S2", "BC_S3"),
                field=f"{prefix}_autonomous_ausc",
            )
            rule_auto = _records_from_field(
                new_rows,
                task=task,
                alias="R_BALANCED_P8",
                source_methods=("R_BALANCED_P8",),
                field=f"{prefix}_autonomous_ausc",
            )
            _append_effect(
                paired,
                domains,
                records=(*bc_auto, *rule_auto),
                analysis=f"H_AUTO_BC_VS_P8_{stop_system}",
                task=task,
                metric="autonomous_ausc",
                treatment="BC_3SEED",
                comparator="R_BALANCED_P8",
                effect_direction="BC_MINUS_RULE_POSITIVE_IS_BETTER",
                confidence_levels=confidence_levels,
                draws=draws,
                include_domains=False,
            )

        for planner, source_methods in (
            ("BC_3SEED", ("BC_S1", "BC_S2", "BC_S3")),
            ("R_BALANCED_P8", ("R_BALANCED_P8",)),
        ):
            calibrated_completion = _records_from_field(
                new_rows,
                task=task,
                alias=f"{planner}+S_BC_CAL",
                source_methods=source_methods,
                field="calibrated_stop_success",
            )
            rule_completion = _records_from_field(
                new_rows,
                task=task,
                alias=f"{planner}+S_RULE",
                source_methods=source_methods,
                field="rule_stop_success",
            )
            _append_effect(
                paired,
                domains,
                records=(*calibrated_completion, *rule_completion),
                analysis="H_STOP_CALIBRATED_VS_RULE",
                task=task,
                metric="completion_rate_difference",
                treatment=f"{planner}+S_BC_CAL",
                comparator=f"{planner}+S_RULE",
                effect_direction="CALIBRATED_MINUS_RULE_POSITIVE_IS_BETTER",
                confidence_levels=confidence_levels,
                draws=draws,
                include_domains=False,
            )
            calibrated_failure = _records_from_field(
                new_rows,
                task=task,
                alias=f"{planner}+S_BC_CAL",
                source_methods=source_methods,
                field="calibrated_failure_penalized_cost",
            )
            rule_failure = _records_from_field(
                new_rows,
                task=task,
                alias=f"{planner}+S_RULE",
                source_methods=source_methods,
                field="rule_failure_penalized_cost",
            )
            _append_effect(
                paired,
                domains,
                records=(*rule_failure, *calibrated_failure),
                analysis="H_STOP_CALIBRATED_VS_RULE",
                task=task,
                metric="failure_cost_reduction",
                treatment=f"{planner}+S_RULE",
                comparator=f"{planner}+S_BC_CAL",
                effect_direction="RULE_MINUS_CALIBRATED_POSITIVE_IS_BETTER",
                confidence_levels=confidence_levels,
                draws=draws,
                include_domains=False,
            )

        for ablated, isolated in (
            ("BC_NO_VLM_S1", "ACTOR_SURFACE_CUES"),
            ("BC_NO_US_FEEDBACK_S1", "ACTOR_US_FEEDBACK_CONTENT"),
        ):
            full = _records_from_field(
                new_rows,
                task=task,
                alias="BC_S1",
                source_methods=("BC_S1",),
                field="ausc_any",
            )
            ablation = _records_from_field(
                new_rows,
                task=task,
                alias=ablated,
                source_methods=(ablated,),
                field="ausc_any",
            )
            records = (*full, *ablation)
            for confidence in confidence_levels:
                result = paired_domain_bootstrap(
                    records,
                    treatment="BC_S1",
                    comparator=ablated,
                    confidence_level=confidence,
                    draws=draws,
                )
                ablations.append(
                    {
                        "schema_version": 1,
                        "task": task.value,
                        "metric": "planner_ausc_any",
                        "full_actor": "BC_S1",
                        "ablated_actor": ablated,
                        "isolated_input": isolated,
                        "effect_direction": "FULL_MINUS_ABLATION_POSITIVE_IS_BETTER",
                        "estimate": result.estimate,
                        "ci_lower": result.ci_lower,
                        "ci_upper": result.ci_upper,
                        "confidence_level": result.confidence_level,
                        "physical_specimen_count": result.physical_specimen_count,
                        "domain_count": result.domain_count,
                        "seed": 1,
                        "bootstrap_replicates": result.replicates,
                        "bootstrap_draws_sha256": result.draws_sha256,
                        "reference_version": "PROXY_LEGACY",
                        "formal_estimate": "",
                        "claim_scope": "ACTOR_INPUT_INCREMENT_ONLY",
                    }
                )

    risk = []
    for task in Task:
        for planner in ("R_BALANCED_P8", "BC_S1", "BC_S2", "BC_S3"):
            seed_value = 1 if planner in {"R_BALANCED_P8", "BC_S1"} else int(planner[-1])
            for stop_system in ("S_RULE", "S_BC_CAL"):
                risk.append(
                    _risk_row(
                        new_rows,
                        task=task,
                        planner=planner,
                        source_method=planner,
                        stop_system=stop_system,
                        seed=seed_value,
                    )
                )
        for planner, source in (
            ("R_BALANCED_P8", "R_BALANCED"),
            ("BC_S1", "L_BC"),
        ):
            risk.append(
                _risk_row(
                    historical_rows,
                    task=task,
                    planner=planner,
                    source_method=source,
                    stop_system="S_OLD_090",
                    seed=1,
                )
            )

    path_b = {}
    for task in Task:
        completion = _find_effect(
            paired,
            analysis="H_AUTO_BC_VS_P8_S_BC_CAL",
            task=task.value,
            metric="completion_rate_difference",
            confidence_level=0.975,
        )
        failure = _find_effect(
            paired,
            analysis="H_AUTO_BC_VS_P8_S_BC_CAL",
            task=task.value,
            metric="failure_cost_reduction",
            confidence_level=0.975,
        )
        completion_pass = float(completion["ci_lower"]) >= -0.05
        cost_pass = float(failure["ci_lower"]) > 0.0
        path_b[task.value] = {
            "completion_bc_minus_p8": completion,
            "failure_cost_p8_minus_bc": failure,
            "completion_noninferiority_margin": -0.05,
            "completion_noninferiority_pass": completion_pass,
            "failure_cost_improvement_pass": cost_pass,
            "joint_proxy_path_b_pass": completion_pass and cost_pass,
        }
    planner_means = {}
    autonomous_means = {}
    for task in Task:
        per_method = {
            method: float(
                np.mean(
                    [
                        float(row["ausc_any"])
                        for row in new_rows
                        if row["task"] == task.value and row["method"] == method
                    ]
                )
            )
            for method in (
                "R_BALANCED_P4",
                "R_BALANCED_P8",
                "BC_S1",
                "BC_S2",
                "BC_S3",
                "BC_NO_VLM_S1",
                "BC_NO_US_FEEDBACK_S1",
            )
        }
        bc_seeds = [per_method[f"BC_S{seed_value}"] for seed_value in (1, 2, 3)]
        planner_means[task.value] = {
            **per_method,
            "BC_3SEED_MEAN": float(np.mean(bc_seeds)),
            "BC_3SEED_MIN": min(bc_seeds),
            "BC_3SEED_MAX": max(bc_seeds),
        }
        autonomous_means[task.value] = {}
        for stop_system in ("S_RULE", "S_BC_CAL"):
            selected_risk = [
                row
                for row in risk
                if row["task"] == task.value
                and row["stop_system"] == stop_system
            ]
            p8 = next(
                row for row in selected_risk if row["planner"] == "R_BALANCED_P8"
            )
            bc_rows = [
                row
                for row in selected_risk
                if row["planner"] in {"BC_S1", "BC_S2", "BC_S3"}
            ]
            autonomous_means[task.value][stop_system] = {
                "R_BALANCED_P8": {
                    "completion_rate": p8["completion_rate"],
                    "false_stop_episode_rate": p8["false_stop_episode_rate"],
                    "exhaustion_rate": p8["exhaustion_rate"],
                    "failure_penalized_cost": p8["failure_penalized_cost"],
                },
                "BC_3SEED_MEAN": {
                    field: float(np.mean([float(row[field]) for row in bc_rows]))
                    for field in (
                        "completion_rate",
                        "false_stop_episode_rate",
                        "exhaustion_rate",
                        "failure_penalized_cost",
                    )
                },
            }
    report_manifest = json.loads(
        (config.output_root / "report_manifest.json").read_text(encoding="utf-8")
    )
    model_manifest = json.loads(
        (config.output_root / "model_manifest.json").read_text(encoding="utf-8")
    )
    rescore_status = json.loads(
        (config.output_root / "reviewed_rescore_status.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "schema_version": 1,
        "stage": "BC_SUPPLEMENT_COMPLETED_PROXY_ONLY",
        "scientific_status": (
            "BC_PLANNING_SUPPORTED_STOP_NOT_SUPPORTED_PROXY_ONLY"
        ),
        "planner_supported_proxy": True,
        "autonomous_path_b_supported_proxy": False,
        "bc_planner_signal_proxy": {
            "effects": [
                row
                for row in paired
                if row["analysis"] == "H_PLAN_BC_3SEED_VS_REGISTERED_RULE"
                and row["comparator"] in {"R_BALANCED_P4", "R_BALANCED_P8"}
                and math.isclose(float(row["confidence_level"]), 0.975)
            ],
            "scope": "SAME_READER_SELF_CONSISTENCY",
        },
        "bc_planner_signal_reviewed": None,
        "bc_autonomous_effect_proxy": path_b,
        "bc_autonomous_effect_reviewed": None,
        "actor_surface_increment": {
            "rows": [
                row
                for row in ablations
                if row["isolated_input"] == "ACTOR_SURFACE_CUES"
                and math.isclose(float(row["confidence_level"]), 0.975)
            ],
            "scope": "ACTOR_INPUT_INCREMENT_ONLY",
        },
        "actor_us_feedback_increment": {
            "rows": [
                row
                for row in ablations
                if row["isolated_input"] == "ACTOR_US_FEEDBACK_CONTENT"
                and math.isclose(float(row["confidence_level"]), 0.975)
            ],
            "scope": "ACTOR_INPUT_INCREMENT_ONLY",
        },
        "stop_increment": {
            "rows": [
                row
                for row in paired
                if row["analysis"] == "H_STOP_CALIBRATED_VS_RULE"
                and math.isclose(float(row["confidence_level"]), 0.975)
            ],
            "old_stop_retained_as": "S_OLD_090_HISTORICAL_REFERENCE",
        },
        "reference_status": {
            "reviewed_count": rescore_status["reviewed_reference_count"],
            "reviewed_test_count": rescore_status[
                "reviewed_test_reference_count"
            ],
            "rescored_report_count": rescore_status[
                "rescored_report_count"
            ],
            "reviewed_version": (
                "REVIEWED_V1"
                if rescore_status["formal_report_scores_available"]
                else "REVIEWED_V1_PENDING"
            ),
            "proxy_version": "PROXY_LEGACY",
            "formal_effects": None,
        },
        "historical_test_reuse": True,
        "additional_confirmation_status": (
            "ADDITIONAL_CONFIRMATION_PENDING_REFERENCE"
        ),
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "draws_sha256": draws.sha256,
            "confidence_levels": list(confidence_levels),
            "physical_specimens": 24,
            "domains": 6,
            "seeds_averaged_within_physical_specimen": True,
        },
        "planner_means_proxy": planner_means,
        "autonomous_means_proxy": autonomous_means,
        "true_break_evidence": {
            "actual_episode_count": report_manifest["true_break"]["episode_count"],
            "physical_specimen_count": report_manifest["true_break"][
                "specimen_count"
            ],
            "all_cached_prefixes_match": report_manifest["true_break"][
                "all_cached_prefixes_match"
            ],
            "post_stop_world_steps": report_manifest["true_break"][
                "post_stop_world_steps"
            ],
        },
        "resource_use": {
            **report_manifest["resource_use"],
            **model_manifest["resource_use"],
            "confirmation_world_transitions": 0,
            "confirmation_vlm_calls": 0,
        },
    }
    return paired, domains, ablations, risk, summary


def summarize_supplement(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Assemble the frozen proxy statistics after TEST evaluation."""

    config = _config(config_path, project_root=project_root)
    required = (
        "per_episode_metrics.csv",
        "trajectories.parquet",
        "report_manifest.json",
        "annotation_queue.csv",
        "confirmation_roster.csv",
        "reviewed_rescore_status.json",
    )
    missing = [name for name in required if not (config.output_root / name).is_file()]
    if missing:
        raise RuntimeError(f"supplement summary inputs are missing: {missing}")
    paired, domains, ablations, risk, summary = _assemble_statistics(config)
    write_csv_atomic(
        config.output_root / "paired_effects.csv",
        tuple(paired),
        PAIRED_EFFECT_FIELDS,
    )
    write_csv_atomic(
        config.output_root / "per_domain_effects.csv",
        tuple(domains),
        PER_DOMAIN_FIELDS,
    )
    write_csv_atomic(
        config.output_root / "ablation_effects.csv",
        tuple(ablations),
        ABLATION_FIELDS,
    )
    write_csv_atomic(
        config.output_root / "risk_coverage.csv",
        tuple(risk),
        RISK_COVERAGE_FIELDS,
    )
    write_json_atomic(config.output_root / "summary.json", summary)
    figures = render_supplement_figures(
        config_path=config_path,
        project_root=project_root,
        source_root=source_root,
    )
    summary["figures"] = figures
    write_json_atomic(config.output_root / "summary.json", summary)
    return {
        "stage": summary["stage"],
        "paired_effect_rows": len(paired),
        "per_domain_rows": len(domains),
        "ablation_rows": len(ablations),
        "risk_rows": len(risk),
        "path_b_proxy": {
            task: value["joint_proxy_path_b_pass"]
            for task, value in summary["bc_autonomous_effect_proxy"].items()
        },
        "figures": len(figures["files"]),
        "formal_effects": None,
    }


__all__ = [
    "analyze_existing",
    "import_references",
    "prepare_confirmation",
    "rescore_references",
    "summarize_supplement",
]
