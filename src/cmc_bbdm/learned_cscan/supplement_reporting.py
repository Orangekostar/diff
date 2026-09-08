"""Reference, analysis, and reporting stages for the BC supplement."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl
from PIL import Image

from cmc_bbdm.vlm_cscan.references import (
    component_bbox_polygons,
    derive_proxy_reference,
    reference_from_payload,
)

from .artifacts import write_csv_atomic, write_json_atomic
from .bc_supplement import SupplementConfig, load_supplement_config
from .contracts import Task
from .metrics import MetricRecord
from .runtime import load_study_config, load_study_context
from .supplement_analysis import (
    make_domain_bootstrap_draws,
    paired_domain_bootstrap,
)

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
    "specimen_key",
    "task",
    "method",
    "seed",
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

    del source_root
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
    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    reader = parent.legacy_config.values["reader"]
    with Image.open(record.cscan_path) as image:
        full_scan = np.asarray(image.convert("RGB"), dtype=np.uint8)
    reference = derive_proxy_reference(
        specimen_key=record.specimen_key,
        source_image_sha256=record.cscan_sha256,
        full_scan=full_scan,
        distance_threshold=float(reader["distance_threshold"]),
        minimum_component_pixels=int(reader["minimum_component_pixels"]),
        uncertainty_band=float(reader["uncertainty_band"]),
        border_exclusion_fraction=float(reader["border_exclusion_fraction"]),
    )
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
        "regions": [
            {
                "id": f"proposal-{index}",
                "polygon": polygon,
                "certainty": "certain",
            }
            for index, polygon in enumerate(
                component_bbox_polygons(reference.certain_mask), start=1
            )
        ],
        "uncertain_regions": [
            {
                "id": f"uncertain-{index}",
                "polygon": polygon,
                "certainty": "uncertain",
            }
            for index, polygon in enumerate(
                component_bbox_polygons(reference.uncertain_mask), start=1
            )
        ],
        "proposal_provenance": {
            "status": "ALGORITHM_DERIVED_NOT_REVIEWED",
            "algorithm": "FULL_CSCAN_INNER_BORDER_RGB_DISTANCE",
            "parameters": {
                key: reader[key]
                for key in (
                    "distance_threshold",
                    "uncertainty_band",
                    "border_exclusion_fraction",
                    "minimum_component_pixels",
                )
            },
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
    queue_rows = (
        pl.read_csv(config.output_root / "annotation_queue.csv").to_dicts()
        if (config.output_root / "annotation_queue.csv").is_file()
        else []
    )
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


def rescore_references(
    *,
    config_path: Path,
    project_root: Path,
    source_root: Path,
    references: Path,
) -> dict[str, object]:
    """Freeze the current reviewed-reference availability and score status."""

    config = _config(config_path, project_root=project_root)
    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    context = load_study_context(parent, source_root=source_root)
    shapes = {
        record.specimen_key: record.native_shape for record in context.roster.records
    }
    root = references.resolve(strict=True)
    reviewed = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = payload.get("specimen_key")
        if key not in shapes or payload.get("review_state") != "reviewed":
            continue
        reference = reference_from_payload(payload, native_shape=shapes[str(key)])
        if reference.formal_eligible:
            reviewed.append((path, reference))
    if reviewed:
        raise RuntimeError(
            "reviewed references arrived; reconstruct stored report prefixes "
            "before writing formal scores"
        )
    write_csv_atomic(
        config.output_root / "reviewed_rescore.csv",
        (),
        REVIEWED_RESCORE_FIELDS,
    )
    status = {
        "schema_version": 1,
        "stage": "REVIEWED_RESCORE_FROZEN",
        "reference_root": "USER_SUPPLIED_REFERENCE_ROOT",
        "reviewed_reference_count": 0,
        "formal_effects": None,
        "proxy_values_unchanged": True,
        "model_retrained": False,
        "threshold_recalibrated": False,
        "status": "INDEPENDENT_REVIEW_PENDING",
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
    summary = {
        "schema_version": 1,
        "stage": "BC_SUPPLEMENT_COMPLETED_PROXY_ONLY",
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
            "reviewed_count": 0,
            "reviewed_version": "REVIEWED_V1_PENDING",
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
    }
    return paired, domains, ablations, risk, summary


def summarize_supplement(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Assemble the frozen proxy statistics after TEST evaluation."""

    del source_root
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
        "formal_effects": None,
    }


__all__ = [
    "analyze_existing",
    "import_references",
    "prepare_confirmation",
    "rescore_references",
    "summarize_supplement",
]
