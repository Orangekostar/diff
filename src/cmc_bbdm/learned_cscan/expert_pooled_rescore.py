"""Thin orchestration and reporting for the pooled expert C-scan rescore."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from cmc_bbdm.vlm_cscan.references import reference_from_payload

REFERENCE_SET_LABEL = "EXPERT_POOL_V1"
INHERITED_PROXY_FILES = (
    "analysis_summary.json",
    "recovery_manifest.json",
    "report_stability_and_delay.csv",
)
MAIN_METHODS = ("R_BALANCED_P8", "BC_S1", "BC_S2", "BC_S3")
REVIEWED_METHODS = MAIN_METHODS + (
    "BC_NO_VLM_S1",
    "BC_NO_US_FEEDBACK_S1",
)


@dataclass(frozen=True, slots=True)
class ExpectedReference:
    dataset_id: str
    source_image_sha256: str
    native_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class ReferenceCandidate:
    source_path: Path
    source_packet_id: str
    reviewer_alias: str
    application_state: str
    source_session_path: Path | None = None


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    materialized = tuple(dict(row) for row in rows)
    if not materialized and not fieldnames:
        raise ValueError("CSV output rows are empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames or tuple(materialized[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def pool_reference_files(
    candidates: tuple[ReferenceCandidate, ...],
    destination: str | Path,
    expected: Mapping[str, ExpectedReference],
) -> dict[str, object]:
    """Validate and byte-preserve one reviewed reference per TEST specimen."""

    if not candidates or not expected:
        raise ValueError("reference pool input is empty")
    entries: dict[str, dict[str, object]] = {}
    exact_duplicates = 0
    for candidate in candidates:
        source = Path(candidate.source_path).resolve(strict=True)
        raw = source.read_bytes()
        payload = json.loads(raw)
        if type(payload) is not dict:
            raise ValueError("reference payload is invalid")
        specimen_key = str(payload.get("specimen_key") or "")
        if specimen_key not in expected:
            raise ValueError(f"unknown specimen: {specimen_key}")
        prior = entries.get(specimen_key)
        if prior is not None:
            if prior["raw"] == raw:
                exact_duplicates += 1
                continue
            raise ValueError(
                f"duplicate specimen has conflicting references: {specimen_key}"
            )
        identity = expected[specimen_key]
        if (
            candidate.application_state != "CONFIRMED"
            or payload.get("reviewer_alias") != candidate.reviewer_alias
            or payload.get("source_image_sha256") != identity.source_image_sha256
        ):
            raise ValueError(f"reference provenance changed: {specimen_key}")
        reference = reference_from_payload(payload, native_shape=identity.native_shape)
        certain_pixels = int(reference.certain_mask.sum())
        if not reference.formal_eligible or certain_pixels < 1:
            raise ValueError(f"reference is not formally eligible: {specimen_key}")
        entries[specimen_key] = {
            "candidate": candidate,
            "payload": payload,
            "raw": raw,
            "reference_sha256": _sha256_bytes(raw),
            "certain_pixel_count": certain_pixels,
            "uncertain_pixel_count": int(reference.uncertain_mask.sum()),
            "native_height": identity.native_shape[0],
            "native_width": identity.native_shape[1],
        }
    missing = sorted(set(expected) - set(entries))
    if missing:
        raise ValueError("missing expected specimens: " + ", ".join(missing))

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    provenance = []
    for specimen_key, entry in sorted(entries.items()):
        candidate = entry["candidate"]
        assert isinstance(candidate, ReferenceCandidate)
        raw = entry["raw"]
        assert isinstance(raw, bytes)
        name = candidate.source_path.name
        if name in used_names:
            name = f"reference-{hashlib.sha256(specimen_key.encode()).hexdigest()[:12]}.json"
        used_names.add(name)
        pooled = root / name
        if pooled.exists():
            if pooled.read_bytes() != raw:
                raise ValueError(f"pooled reference would be overwritten: {name}")
        else:
            shutil.copyfile(candidate.source_path, pooled)
        payload = entry["payload"]
        assert isinstance(payload, dict)
        identity = expected[specimen_key]
        provenance.append(
            {
                "specimen_key": specimen_key,
                "dataset_id": identity.dataset_id,
                "source_json_path": str(candidate.source_path.resolve()),
                "pooled_json_path": str(pooled),
                "source_image_sha256": identity.source_image_sha256,
                "reference_sha256": entry["reference_sha256"],
                "source_packet_id": candidate.source_packet_id,
                "reviewer_alias": candidate.reviewer_alias,
                "reference_type": payload["reference_type"],
                "application_state": candidate.application_state,
                "native_height": entry["native_height"],
                "native_width": entry["native_width"],
                "certain_pixel_count": entry["certain_pixel_count"],
                "uncertain_pixel_count": entry["uncertain_pixel_count"],
            }
        )
    stale = sorted(
        path.name for path in root.glob("*.json") if path.name not in used_names
    )
    if stale:
        raise ValueError("unexpected pooled references: " + ", ".join(stale))
    return {
        "reference_set_label": REFERENCE_SET_LABEL,
        "physical_specimen_count": len(provenance),
        "source_expert_count": len({row["reviewer_alias"] for row in provenance}),
        "exact_duplicate_count": exact_duplicates,
        "provenance": tuple(provenance),
    }


def verify_output_only_config(
    source_config: str | Path, derived_config: str | Path
) -> dict[str, str]:
    """Require the derived YAML to differ only in the two output roots."""

    source = yaml.safe_load(Path(source_config).read_text(encoding="utf-8"))
    derived = yaml.safe_load(Path(derived_config).read_text(encoding="utf-8"))
    if type(source) is not dict or type(derived) is not dict:
        raise ValueError("rescore configuration is invalid")
    expected = json.loads(json.dumps(source))
    expected["paths"]["output_root"] = "results/bc_cscan_expert_pooled_rescore/v1"
    expected["paths"]["artifact_root"] = "artifacts/bc_cscan_expert_pooled_rescore/v1"
    if derived != expected:
        raise ValueError("derived config changed frozen scientific fields")
    return {
        "paths.output_root": derived["paths"]["output_root"],
        "paths.artifact_root": derived["paths"]["artifact_root"],
    }


def inherit_proxy_inputs(
    source_root: str | Path, output_root: str | Path
) -> dict[str, object]:
    """Copy only the three legacy files required by the existing finalize path."""

    source = Path(source_root).resolve(strict=True)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for name in INHERITED_PROXY_FILES:
        source_path = source / name
        if not source_path.is_file():
            raise ValueError(f"required inherited proxy input is missing: {name}")
        target = destination / name
        raw = source_path.read_bytes()
        if target.exists() and target.read_bytes() != raw:
            raise ValueError(f"inherited proxy input changed: {name}")
        if not target.exists():
            shutil.copyfile(source_path, target)
        files.append(
            {
                "name": name,
                "source_path": str(source_path),
                "output_path": str(target),
                "sha256": _sha256_bytes(raw),
                "role": "FINALIZE_CONTEXT_NOT_REVIEWED_RESULT",
            }
        )
    manifest = {
        "schema_version": 1,
        "analysis_scope": "PROXY_LEGACY_CONTEXT_ONLY",
        "reference_version": "PROXY_LEGACY",
        "files": files,
    }
    _write_json(destination / "inherited_proxy_inputs.json", manifest)
    return manifest


def discover_reference_candidates(
    export_root: str | Path,
) -> tuple[ReferenceCandidate, ...]:
    """Discover validated per-specimen files from existing session exports."""

    root = Path(export_root).resolve(strict=True)
    candidates = []
    for references_root in sorted(root.glob("*/*/references")):
        session_root = references_root.parent
        session_path = session_root / "session.json"
        audit_path = session_root / "annotation_audit.csv"
        if not session_path.is_file() or not audit_path.is_file():
            raise ValueError(f"reference export metadata is incomplete: {session_root}")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        reviewer = session.get("reviewer") if type(session) is dict else None
        if type(reviewer) is not dict:
            raise ValueError(f"reference export session is invalid: {session_path}")
        alias = str(reviewer.get("reviewer_alias") or "")
        packet_id = str(session.get("packet_id") or "")
        with audit_path.open(encoding="utf-8", newline="") as handle:
            audit = {str(row["item_id"]): row for row in csv.DictReader(handle)}
        for source_path in sorted(references_root.glob("*.json")):
            row = audit.get(source_path.stem)
            if row is None:
                raise ValueError(
                    f"reference export audit row is missing: {source_path}"
                )
            if row.get("packet_id") != packet_id or row.get("reviewer_alias") != alias:
                raise ValueError(
                    f"reference export audit identity changed: {source_path}"
                )
            candidates.append(
                ReferenceCandidate(
                    source_path=source_path,
                    source_packet_id=packet_id,
                    reviewer_alias=alias,
                    application_state=str(row.get("application_state") or ""),
                    source_session_path=session_path,
                )
            )
    if not candidates:
        raise ValueError(f"no exported reference JSONs found below {root}")
    return tuple(candidates)


def _expected_test_references(
    config: object, source_root: str | Path
) -> dict[str, ExpectedReference]:
    from .contracts import Split
    from .runtime import load_study_config, load_study_context

    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    context = load_study_context(parent, source_root=source_root)
    return {
        assignment.record.specimen_key: ExpectedReference(
            dataset_id=assignment.record.dataset_id,
            source_image_sha256=assignment.record.cscan_sha256,
            native_shape=assignment.record.native_shape,
        )
        for assignment in context.roster.assignments
        if assignment.split is Split.TEST
    }


def prepare_expert_pool(
    *,
    config_path: str | Path,
    source_config_path: str | Path,
    project_root: str | Path,
    source_root: str | Path,
    export_root: str | Path,
    legacy_output_root: str | Path,
    original_session_root: str | Path | None = None,
) -> dict[str, object]:
    """Prepare the isolated reviewed input tree without running recovery."""

    from .frozen_evidence_finalize import _reference_inputs
    from .frozen_process_analysis import load_frozen_process_config

    project = Path(project_root).resolve(strict=True)
    changes = verify_output_only_config(source_config_path, config_path)
    config = load_frozen_process_config(config_path, project_root=project)
    expected = _expected_test_references(config, source_root)
    if len(expected) != config.physical_specimens:
        raise ValueError("frozen TEST specimen count changed")
    candidates = discover_reference_candidates(export_root)
    references_root = config.output_root / "inputs/references"
    pooled = pool_reference_files(candidates, references_root, expected)
    inherited = inherit_proxy_inputs(legacy_output_root, config.output_root)
    imported = _reference_inputs(
        config, source_root=source_root, references_path=references_root
    )
    if (
        imported["input_status"] != "FULL_TEST_REVIEWED_REFERENCE"
        or len(imported["references"]) != config.physical_specimens
    ):
        raise ValueError("pooled reference coverage is incomplete")

    inputs_root = config.output_root / "inputs"
    provenance = tuple(pooled["provenance"])
    _write_csv(inputs_root / "reference_provenance.csv", provenance)
    sessions = []
    for path in sorted(
        {
            candidate.source_session_path
            for candidate in candidates
            if candidate.source_session_path is not None
        }
    ):
        assert path is not None
        payload = json.loads(path.read_text(encoding="utf-8"))
        sessions.append(
            {
                "export_id": payload["export_id"],
                "packet_id": payload["packet_id"],
                "reviewer_alias": payload["reviewer"]["reviewer_alias"],
                "session_path": str(path),
                "session_sha256": _sha256_bytes(path.read_bytes()),
                "exported_reference_count": sum(
                    candidate.source_session_path == path for candidate in candidates
                ),
            }
        )
    original_sessions = []
    if original_session_root is not None:
        original_root = Path(original_session_root).resolve(strict=True)
        for path in sorted(original_root.glob("reference-*.json")):
            original_sessions.append(
                {"path": str(path), "sha256": _sha256_bytes(path.read_bytes())}
            )
    manifest = {
        "schema_version": 1,
        "reference_set_label": REFERENCE_SET_LABEL,
        "reference_version": imported["reference_version"],
        "reference_input_status": imported["input_status"],
        "analysis_grouping": "POOLED_NOT_BY_REVIEWER",
        "physical_specimen_count": pooled["physical_specimen_count"],
        "domain_count": len({row["dataset_id"] for row in provenance}),
        "source_expert_count": pooled["source_expert_count"],
        "source_export_count": len(sessions),
        "certain_reference_count": len(provenance),
        "uncertain_region_reference_count": sum(
            int(row["uncertain_pixel_count"]) > 0 for row in provenance
        ),
        "configuration_changes": changes,
        "exported_sessions": sessions,
        "original_sessions": original_sessions,
        "inherited_proxy_inputs": inherited["files"],
        "pooled_reference_root": str(references_root),
        "source_root": str(Path(source_root).resolve(strict=True)),
        "new_training": 0,
        "new_vlm_calls": 0,
        "new_actor_forward_calls": 0,
        "new_stop_forward_calls": 0,
    }
    _write_json(inputs_root / "reference_set_manifest.json", manifest)
    bindings = "\n".join(
        (
            "# Input and Execution Bindings",
            "",
            "- Development base: `0e7f15b5c4cb59b05f19b2c598cdc9a9278cc15e`",
            f"- Derived configuration: `{Path(config_path)}`",
            "- Frozen configuration change: output and artifact roots only",
            f"- Original session directory: `{original_session_root or 'not provided'}`",
            f"- Existing export directory: `{Path(export_root).resolve()}`",
            f"- Pooled references: `{references_root}`",
            f"- Coverage: {pooled['physical_specimen_count']} TEST specimens across {manifest['domain_count']} domains",
            f"- Source experts: {pooled['source_expert_count']} (pooled; not a grouping variable)",
            f"- Reference set label: `{REFERENCE_SET_LABEL}`",
            f"- Scoring reference version: `{imported['reference_version']}`",
            "- All 24 references contain a confirmed certain region.",
            "- Original polygons, aliases, identities, and JSON bytes were preserved.",
            "- Historical proxy files are inherited only as finalize context.",
            "- No training or new VLM/Actor/STOP forward pass is part of preparation.",
            "",
        )
    )
    artifact = config.artifact_root / "INPUT_AND_EXECUTION_BINDINGS.md"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(bindings, encoding="utf-8")
    return manifest


def run_expert_rescore(
    *,
    config_path: str | Path,
    source_config_path: str | Path,
    project_root: str | Path,
    source_root: str | Path,
    references_path: str | Path,
) -> dict[str, object]:
    """Run the existing frozen finalize path once for the pooled references."""

    from .frozen_process_analysis import execute_frozen_finalize

    verify_output_only_config(source_config_path, config_path)
    from .frozen_process_analysis import load_frozen_process_config

    config = load_frozen_process_config(config_path, project_root=project_root)
    formal_run_path = config.output_root / "formal_run_result.json"
    if formal_run_path.exists():
        raise ValueError("formal pooled expert rescore was already completed")
    result = execute_frozen_finalize(
        config_path=config_path,
        project_root=project_root,
        source_root=source_root,
        references_path=references_path,
    )
    recorded = {
        "schema_version": 1,
        "formal_run_count": 1,
        "entrypoint": "execute_frozen_finalize",
        "result": result,
    }
    _write_json(formal_run_path, recorded)
    return recorded


def _bc_seed_mean_rows(
    rows: Iterable[Mapping[str, object]], fields: tuple[str, ...]
) -> tuple[dict[str, object], ...]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row["method"] in {"BC_S1", "BC_S2", "BC_S3"}:
            groups[str(row["specimen_key"])].append(row)
    output = []
    for specimen_key, group in sorted(groups.items()):
        if {str(row["method"]) for row in group} != {"BC_S1", "BC_S2", "BC_S3"}:
            raise ValueError(f"BC seed coverage is incomplete: {specimen_key}")
        output.append(
            {
                "specimen_key": specimen_key,
                "dataset_id": group[0]["dataset_id"],
                **{
                    field: sum(float(row[field]) for row in group) / len(group)
                    for field in fields
                },
            }
        )
    return tuple(output)


def autonomous_absolute_rows(
    episode_rows: tuple[Mapping[str, object], ...],
) -> tuple[dict[str, object], ...]:
    metrics = (
        "completion",
        "false_stop",
        "exhausted",
        "failure_penalized_cost",
        "autonomous_ausc",
        "prefix_measurement_cost",
    )
    output = []
    for stop_system in ("S_BC_CAL", "S_RULE"):
        for task in sorted({str(row["task"]) for row in episode_rows}):
            selected = [
                row
                for row in episode_rows
                if row["stop_system"] == stop_system
                and row["task"] == task
                and row["method"] in MAIN_METHODS
            ]
            for method in MAIN_METHODS:
                group = [row for row in selected if row["method"] == method]
                if not group:
                    continue
                output.append(
                    {
                        "task": task,
                        "stop_system": stop_system,
                        "method": method,
                        **{
                            metric: _domain_equal_mean(group, metric)
                            for metric in metrics
                        },
                        "physical_specimen_count": len(
                            {str(row["specimen_key"]) for row in group}
                        ),
                        "domain_count": len({str(row["dataset_id"]) for row in group}),
                    }
                )
            bc_rows = _bc_seed_mean_rows(selected, metrics)
            output.append(
                {
                    "task": task,
                    "stop_system": stop_system,
                    "method": "BC_3SEED",
                    **{
                        metric: _domain_equal_mean(bc_rows, metric)
                        for metric in metrics
                    },
                    "physical_specimen_count": len(bc_rows),
                    "domain_count": len({str(row["dataset_id"]) for row in bc_rows}),
                }
            )
    return tuple(output)


def _outcome(row: Mapping[str, object], prefix: str = "") -> str:
    if bool(row[f"{prefix}completion"]):
        return "COMPLETION"
    if bool(row[f"{prefix}false_stop"]):
        return "FALSE_STOP"
    if bool(row[f"{prefix}exhausted"]):
        return "EXHAUSTED"
    raise ValueError("episode outcome partition is invalid")


def _same_optional_float(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), abs_tol=1e-12)


def _proxy_reviewed_episode_rows(
    legacy_output_root: Path,
    reviewed_rows: tuple[Mapping[str, object], ...],
) -> tuple[dict[str, object], ...]:
    keys = ("specimen_key", "task", "method", "seed", "stop_system")
    proxy = pl.read_csv(
        legacy_output_root / "planning_to_stop_accounting.csv",
        infer_schema_length=None,
    ).filter(pl.col("method").is_in(REVIEWED_METHODS))
    proxy_index = {tuple(row[key] for key in keys): row for row in proxy.to_dicts()}
    if len(proxy_index) != len(reviewed_rows):
        raise ValueError("proxy/reviewed episode coverage differs")
    output = []
    for reviewed in reviewed_rows:
        key = tuple(reviewed[field] for field in keys)
        proxy_row = proxy_index.get(key)
        if proxy_row is None:
            raise ValueError(f"proxy episode is missing: {key}")
        stop_identity_unchanged = (
            bool(proxy_row["stopped"]) == bool(reviewed["stopped"])
            and proxy_row["stop_step"] == reviewed["stop_step"]
            and _same_optional_float(proxy_row["stop_cost"], reviewed["stop_cost"])
        )
        if not stop_identity_unchanged:
            raise ValueError(f"frozen STOP identity changed: {key}")
        proxy_outcome = _outcome(proxy_row)
        reviewed_outcome = _outcome(reviewed)
        output.append(
            {
                **{field: reviewed[field] for field in keys},
                "dataset_id": reviewed["dataset_id"],
                "reference_version_proxy": "PROXY_LEGACY",
                "reference_version_reviewed": reviewed["reference_version"],
                "stop_identity_unchanged": True,
                "proxy_outcome": proxy_outcome,
                "reviewed_outcome": reviewed_outcome,
                "outcome_flip": (
                    "UNCHANGED"
                    if proxy_outcome == reviewed_outcome
                    else f"{proxy_outcome}_TO_{reviewed_outcome}"
                ),
                "proxy_planner_ausc": proxy_row["planner_ausc"],
                "reviewed_planner_ausc": reviewed["planner_ausc"],
                "planner_ausc_difference": float(reviewed["planner_ausc"])
                - float(proxy_row["planner_ausc"]),
                "proxy_autonomous_ausc": proxy_row["autonomous_ausc"],
                "reviewed_autonomous_ausc": reviewed["autonomous_ausc"],
                "proxy_failure_penalized_cost": 1.0
                - float(proxy_row["autonomous_ausc"]),
                "reviewed_failure_penalized_cost": reviewed["failure_penalized_cost"],
            }
        )
    return tuple(output)


def _state_pairing(
    config: object,
) -> tuple[pl.DataFrame, tuple[dict[str, object], ...]]:
    proxy = (
        pl.read_parquet(config.source_result_root / "trajectories.parquet")
        .filter(pl.col("method").is_in(REVIEWED_METHODS))
        .select(
            "dataset_id",
            "specimen_key",
            "task",
            "method",
            "seed",
            "step",
            "cost",
            "report_sha256",
            pl.col("success").alias("proxy_success"),
        )
    )
    reviewed = pl.read_parquet(
        config.output_root / "reviewed/report_scores.parquet"
    ).select(
        "specimen_key",
        "task",
        "method",
        "seed",
        "step",
        pl.col("formal_success").alias("reviewed_success"),
        pl.col("report_sha256").alias("reviewed_report_sha256"),
    )
    keys = ["specimen_key", "task", "method", "seed", "step"]
    paired = proxy.join(reviewed, on=keys, how="inner", validate="1:1")
    if paired.height != proxy.height or paired.height != reviewed.height:
        raise ValueError("proxy/reviewed state coverage differs")
    if paired.filter(
        pl.col("report_sha256") != pl.col("reviewed_report_sha256")
    ).height:
        raise ValueError("proxy/reviewed report identity changed")
    summary = (
        paired.group_by("task", "method")
        .agg(
            pl.len().alias("state_count"),
            pl.col("specimen_key").n_unique().alias("physical_specimen_count"),
            pl.col("proxy_success").mean().alias("proxy_success_fraction"),
            pl.col("reviewed_success").mean().alias("reviewed_success_fraction"),
            (pl.col("proxy_success") != pl.col("reviewed_success"))
            .sum()
            .alias("status_flip_count"),
            ((~pl.col("proxy_success")) & pl.col("reviewed_success"))
            .sum()
            .alias("failure_to_success_count"),
            (pl.col("proxy_success") & (~pl.col("reviewed_success")))
            .sum()
            .alias("success_to_failure_count"),
        )
        .with_columns(
            (
                pl.col("reviewed_success_fraction") - pl.col("proxy_success_fraction")
            ).alias("reviewed_minus_proxy_success_fraction")
        )
        .sort("task", "method")
        .to_dicts()
    )
    return paired, tuple(summary)


def _full_input_tables(
    rows: tuple[Mapping[str, object], ...],
    provenance: tuple[Mapping[str, object], ...],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    total_pixels = sum(
        int(row["native_height"]) * int(row["native_width"]) for row in provenance
    )
    uncertain_pixels = sum(int(row["uncertain_pixel_count"]) for row in provenance)
    summary = []
    failures = []
    for task in sorted({str(row["task"]) for row in rows}):
        group = [row for row in rows if row["task"] == task]
        summary.append(
            {
                "task": task,
                "physical_specimen_count": len(group),
                "formal_success_count": sum(
                    bool(row["formal_success"]) for row in group
                ),
                "formal_success_fraction": sum(
                    bool(row["formal_success"]) for row in group
                )
                / len(group),
                "mean_iou": sum(float(row["iou"]) for row in group) / len(group),
                "mean_recall": sum(float(row["recall"]) for row in group) / len(group),
                "mean_relative_area_error": sum(
                    float(row["relative_area_error"]) for row in group
                )
                / len(group),
                "uncertain_pixel_fraction": uncertain_pixels / total_pixels,
                "references_with_uncertain_regions": sum(
                    int(row["uncertain_pixel_count"]) > 0 for row in provenance
                ),
            }
        )
        failures.extend(
            {
                "dataset_id": row["dataset_id"],
                "specimen_key": row["specimen_key"],
                "task": task,
                "iou": row["iou"],
                "recall": row["recall"],
                "relative_area_error": row["relative_area_error"],
                "failure_types": row["failure_types"],
            }
            for row in group
            if not bool(row["formal_success"])
        )
    return tuple(summary), tuple(failures)


def _method_absolute(
    rows: tuple[Mapping[str, object], ...], task: str, method: str
) -> float:
    selected = [
        row
        for row in rows
        if row["task"] == task
        and row["method"] == method
        and row["stop_system"] == "S_BC_CAL"
    ]
    return _domain_equal_mean(selected, "planner_ausc")


def _ablation_summary_rows(
    episode_rows: tuple[Mapping[str, object], ...],
    effect_rows: tuple[Mapping[str, object], ...],
) -> tuple[dict[str, object], ...]:
    output = []
    for effect in effect_rows:
        if float(effect["confidence_level"]) != 0.975:
            continue
        task = str(effect["task"])
        full = str(effect["full_actor"])
        ablated = str(effect["ablated_actor"])
        output.append(
            {
                "task": task,
                "isolated_input": effect["isolated_input"],
                "full_actor": full,
                "ablated_actor": ablated,
                "full_actor_planner_ausc": _method_absolute(episode_rows, task, full),
                "ablated_actor_planner_ausc": _method_absolute(
                    episode_rows, task, ablated
                ),
                "estimate": effect["estimate"],
                "ci_lower_97_5": effect["ci_lower"],
                "ci_upper_97_5": effect["ci_upper"],
                "physical_specimen_count": effect["physical_specimen_count"],
                "domain_count": effect["domain_count"],
            }
        )
    return tuple(output)


def _reviewed_claim_rows(
    *,
    reference_version: str,
    physical_n: int,
    expected_n: int,
    planner_effects: tuple[Mapping[str, object], ...],
    ablation_effects: tuple[Mapping[str, object], ...],
    decisions: Mapping[str, Mapping[str, object]],
    full_input: tuple[Mapping[str, object], ...],
) -> tuple[dict[str, object], ...]:
    claims = []
    for row in planner_effects:
        if float(row["confidence_level"]) == 0.975:
            claims.append(
                {
                    "claim_id": f"R1_{row['task']}",
                    "task": row["task"],
                    "claim": "BC three-seed mean improves reviewed planner AUSC over P8",
                    "status": classify_positive_effect(
                        float(row["ci_lower"]), physical_n, expected_n
                    ),
                    "estimate": row["estimate"],
                    "ci_lower_97_5": row["ci_lower"],
                    "ci_upper_97_5": row["ci_upper"],
                    "evidence": "reviewed/planner_effects.csv",
                    "reference_version": reference_version,
                }
            )
    claim_by_input = {
        "ACTOR_SURFACE_CUES": "R2",
        "ACTOR_US_FEEDBACK_CONTENT": "R3",
    }
    for row in ablation_effects:
        if float(row["confidence_level"]) == 0.975:
            claim_id = claim_by_input[str(row["isolated_input"])]
            claims.append(
                {
                    "claim_id": f"{claim_id}_{row['task']}",
                    "task": row["task"],
                    "claim": f"{row['isolated_input']} contributes to seed-1 Actor planner AUSC",
                    "status": classify_positive_effect(
                        float(row["ci_lower"]), physical_n, expected_n
                    ),
                    "estimate": row["estimate"],
                    "ci_lower_97_5": row["ci_lower"],
                    "ci_upper_97_5": row["ci_upper"],
                    "evidence": "reviewed/actor_input_ablations.csv",
                    "reference_version": reference_version,
                }
            )
    for task, decision in sorted(decisions.items()):
        claims.append(
            {
                "claim_id": f"R4_{task}",
                "task": task,
                "claim": "Frozen Path B joint completion/cost condition",
                "status": decision["status"],
                "joint_path_b_pass": decision["joint_path_b_pass"],
                "evidence": "reviewed/task_decisions.json",
                "reference_version": reference_version,
            }
        )
    for row in full_input:
        claims.append(
            {
                "claim_id": f"R5_{row['task']}",
                "task": row["task"],
                "claim": "Full-input Reader agreement with pooled expert references",
                "status": "DESCRIPTIVE_RESULT",
                "formal_success_count": row["formal_success_count"],
                "physical_specimen_count": row["physical_specimen_count"],
                "evidence": "tables/full_input_reader_summary.csv",
                "reference_version": reference_version,
            }
        )
    claims.extend(
        (
            {
                "claim_id": "R6_PROCESS",
                "task": "BOTH",
                "claim": "Frozen report and STOP process description",
                "status": "DESCRIPTIVE_RESULT",
                "evidence": "tables/reviewed_process_summary.csv",
                "reference_version": reference_version,
            },
            {
                "claim_id": "R7_BLIND_REVIEW",
                "task": "BOTH",
                "claim": "Blinded report review",
                "status": "PENDING_INPUT",
                "evidence": "human_review/input_coverage.csv",
                "reference_version": reference_version,
            },
            {
                "claim_id": "R8_HUMAN_PLANNING",
                "task": "BOTH",
                "claim": "Human planning comparison",
                "status": "PENDING_INPUT",
                "evidence": "human_planning/comparability_manifest.csv",
                "reference_version": reference_version,
            },
        )
    )
    return tuple(claims)


def _process_summary_rows(
    rows: tuple[Mapping[str, object], ...],
) -> tuple[dict[str, object], ...]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["task"]), str(row["method"]), str(row["stop_system"]))].append(
            row
        )
    output = []
    for (task, method, stop_system), group in sorted(groups.items()):
        outcomes: dict[str, int] = defaultdict(int)
        for row in group:
            outcomes[str(row["outcome_category"])] += 1
        output.append(
            {
                "task": task,
                "method": method,
                "stop_system": stop_system,
                "episode_count": len(group),
                "first_success_observed_count": sum(
                    row["first_success_step"] is not None for row in group
                ),
                "sustained_success_observed_count": sum(
                    row["sustained_success_step"] is not None for row in group
                ),
                "regressed_after_first_success_count": sum(
                    bool(row["regressed_after_first_success"]) for row in group
                ),
                "outcome_category_counts": json.dumps(
                    dict(sorted(outcomes.items())), sort_keys=True
                ),
            }
        )
    return tuple(output)


def _fixed_case_rows(
    paired_states: pl.DataFrame, case_manifest_path: Path
) -> tuple[dict[str, object], ...]:
    cases = pl.read_csv(case_manifest_path, infer_schema_length=None).to_dicts()
    output = []
    for case in cases:
        specimen_key = str(case["specimen_key"])
        bc_method = str(case["bc_method"])
        for method in ("R_BALANCED_P8", bc_method):
            selected = paired_states.filter(
                (pl.col("specimen_key") == specimen_key) & (pl.col("method") == method)
            ).sort("task", "step")
            if selected.height != 2 * 193:
                raise ValueError(
                    f"fixed-case state coverage is incomplete: {specimen_key}"
                )
            for row in selected.to_dicts():
                output.append(
                    {
                        "dataset_id": row["dataset_id"],
                        "specimen_key": specimen_key,
                        "task": row["task"],
                        "method": method,
                        "method_group": "P8"
                        if method == "R_BALANCED_P8"
                        else "BC_FIXED_SEED",
                        "seed": row["seed"],
                        "step": row["step"],
                        "cost": row["cost"],
                        "proxy_success": row["proxy_success"],
                        "reviewed_success": row["reviewed_success"],
                        "selection_rule": case["selection_rule"],
                    }
                )
    return tuple(output)


def _format_float(value: object) -> str:
    return f"{float(value):.4f}"


def _plot_figures(
    *,
    artifact_root: Path,
    planner_absolute: tuple[Mapping[str, object], ...],
    planner_effects: tuple[Mapping[str, object], ...],
    autonomous_absolute: tuple[Mapping[str, object], ...],
    fixed_cases: tuple[Mapping[str, object], ...],
    reference_version: str,
) -> tuple[str, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    figures = artifact_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = []
    tasks = ("LOCATE", "CHARACTERIZE")
    colors = {"R_BALANCED_P8": "#31745B", "BC_3SEED": "#C74B3C"}

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), constrained_layout=True)
    for axis, task in zip(axes, tasks, strict=True):
        values = {
            str(row["method"]): float(row["planner_ausc"])
            for row in planner_absolute
            if row["task"] == task and row["method"] in {"R_BALANCED_P8", "BC_3SEED"}
        }
        methods = ("R_BALANCED_P8", "BC_3SEED")
        axis.bar(
            range(2),
            [values[method] for method in methods],
            color=[colors[method] for method in methods],
            width=0.62,
        )
        effect = next(
            row
            for row in planner_effects
            if row["task"] == task and float(row["confidence_level"]) == 0.975
        )
        axis.text(
            0.5,
            0.97,
            "BC-P8 = "
            + _format_float(effect["estimate"])
            + "\n97.5% CI ["
            + _format_float(effect["ci_lower"])
            + ", "
            + _format_float(effect["ci_upper"])
            + "]",
            ha="center",
            va="top",
            transform=axis.transAxes,
            fontsize=8,
        )
        axis.set_xticks(range(2), ("P8", "BC 3-seed"))
        axis.set_ylim(0.0, 1.0)
        axis.set_title(task.title())
        axis.set_ylabel("Reviewed planner AUSC")
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    figure.suptitle(f"Pooled expert reference: {reference_version}", fontsize=10)
    path = figures / "figure_1_reviewed_planner_effects.png"
    figure.savefig(path, dpi=240)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    for axis, task in zip(axes, tasks, strict=True):
        rows = {
            str(row["method"]): row
            for row in autonomous_absolute
            if row["task"] == task
            and row["stop_system"] == "S_BC_CAL"
            and row["method"] in {"R_BALANCED_P8", "BC_3SEED"}
        }
        methods = ("R_BALANCED_P8", "BC_3SEED")
        x = np.arange(2)
        bottom = np.zeros(2)
        for metric, label, color in (
            ("completion", "Correct completion", "#31745B"),
            ("false_stop", "False stop", "#C74B3C"),
            ("exhausted", "Exhausted", "#D7A23D"),
        ):
            values = np.asarray([float(rows[method][metric]) for method in methods])
            axis.bar(x, values, bottom=bottom, label=label, color=color, width=0.62)
            bottom += values
        costs = [float(rows[method]["failure_penalized_cost"]) for method in methods]
        axis.plot(x, costs, "ko", label="Failure-penalized cost", markersize=4)
        axis.set_xticks(x, ("P8", "BC 3-seed"))
        axis.set_ylim(0.0, 1.04)
        axis.set_title(task.title())
        axis.set_ylabel("Episode fraction / cost")
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    figure.suptitle("Original calibrated STOP under expert scoring", fontsize=10)
    path = figures / "figure_2_reviewed_stop_outcomes.png"
    figure.savefig(path, dpi=240)
    plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    case_keys = sorted({str(row["specimen_key"]) for row in fixed_cases})
    for axis, task in zip(axes, tasks, strict=True):
        for specimen_key in case_keys:
            for group, color, style in (
                ("P8", "#31745B", "-"),
                ("BC_FIXED_SEED", "#C74B3C", "--"),
            ):
                rows = sorted(
                    (
                        row
                        for row in fixed_cases
                        if row["task"] == task
                        and row["specimen_key"] == specimen_key
                        and row["method_group"] == group
                    ),
                    key=lambda row: int(row["step"]),
                )
                axis.step(
                    [float(row["cost"]) for row in rows],
                    [int(bool(row["reviewed_success"])) for row in rows],
                    where="post",
                    color=color,
                    linestyle=style,
                    linewidth=0.8,
                    alpha=0.5,
                )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.04, 1.04)
        axis.set_title(task.title())
        axis.set_xlabel("Exact native-raster acquisition cost")
        axis.set_ylabel("Reviewed current-report success")
        axis.grid(alpha=0.2, linewidth=0.7)
    figure.legend(
        handles=(
            Line2D([0], [0], color="#31745B", linewidth=1.4, label="P8"),
            Line2D(
                [0],
                [0],
                color="#C74B3C",
                linestyle="--",
                linewidth=1.4,
                label="Preselected BC seed",
            ),
        ),
        loc="outside lower center",
        ncol=2,
        fontsize=8,
    )
    figure.suptitle("Six preselected physical cases; frozen trajectories", fontsize=10)
    path = figures / "figure_3_fixed_case_reviewed_process.png"
    figure.savefig(path, dpi=240)
    plt.close(figure)
    paths.append(str(path))
    return tuple(paths)


def _results_document(
    *,
    reference_version: str,
    full_input: tuple[Mapping[str, object], ...],
    planner_absolute: tuple[Mapping[str, object], ...],
    planner_effects: tuple[Mapping[str, object], ...],
    autonomous_absolute: tuple[Mapping[str, object], ...],
    ablations: tuple[Mapping[str, object], ...],
    decisions: Mapping[str, Mapping[str, object]],
    claims: tuple[Mapping[str, object], ...],
) -> str:
    lines = [
        "# Reviewed Results and Claim Boundaries",
        "",
        f"Reference version: `{reference_version}`. The 24 references are pooled across two source experts and are not grouped by reviewer.",
        "",
        "## Full-input Reader",
        "",
        "| Task | Success | Mean IoU | Mean recall | Mean area error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in full_input:
        lines.append(
            f"| {row['task']} | {row['formal_success_count']}/{row['physical_specimen_count']} | "
            f"{_format_float(row['mean_iou'])} | {_format_float(row['mean_recall'])} | "
            f"{_format_float(row['mean_relative_area_error'])} |"
        )
    lines.extend(
        (
            "",
            "LOCATE recall and area-error fields retain their task-specific bbox/support semantics; CHARACTERIZE uses mask IoU, recall, and the registered area tolerance.",
            "",
            "## Reviewed planner AUSC",
            "",
            "| Task | P8 | BC S1 | BC S2 | BC S3 | BC 3-seed | BC-P8 (97.5% CI) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for task in ("LOCATE", "CHARACTERIZE"):
        values = {
            str(row["method"]): row["planner_ausc"]
            for row in planner_absolute
            if row["task"] == task
        }
        effect = next(
            row
            for row in planner_effects
            if row["task"] == task and float(row["confidence_level"]) == 0.975
        )
        lines.append(
            f"| {task} | {_format_float(values['R_BALANCED_P8'])} | "
            f"{_format_float(values['BC_S1'])} | {_format_float(values['BC_S2'])} | "
            f"{_format_float(values['BC_S3'])} | {_format_float(values['BC_3SEED'])} | "
            f"{_format_float(effect['estimate'])} [{_format_float(effect['ci_lower'])}, {_format_float(effect['ci_upper'])}] |"
        )
    lines.extend(
        (
            "",
            "BC 3-seed is a statistical seed average, not an inference-time ensemble.",
            "",
            "## Original calibrated STOP and Path B",
            "",
            "| Task | Method | Completion | False stop | Exhausted | Failure cost | Prefix cost | Path B |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        )
    )
    for task in ("LOCATE", "CHARACTERIZE"):
        for method in ("R_BALANCED_P8", "BC_3SEED"):
            row = next(
                item
                for item in autonomous_absolute
                if item["task"] == task
                and item["stop_system"] == "S_BC_CAL"
                and item["method"] == method
            )
            lines.append(
                f"| {task} | {method} | {_format_float(row['completion'])} | "
                f"{_format_float(row['false_stop'])} | {_format_float(row['exhausted'])} | "
                f"{_format_float(row['failure_penalized_cost'])} | "
                f"{_format_float(row['prefix_measurement_cost'])} | "
                f"{decisions[task]['status']} |"
            )
    lines.extend(
        (
            "",
            "Path B is reported separately by task and uses the frozen 97.5% joint completion/cost rule. S_RULE remains an auxiliary diagnostic in the machine-readable tables.",
            "",
            "## Existing seed-1 Actor input ablations",
            "",
            "| Task | Input | Full | Ablated | Delta (97.5% CI) |",
            "|---|---|---:|---:|---:|",
        )
    )
    for row in ablations:
        lines.append(
            f"| {row['task']} | {row['isolated_input']} | "
            f"{_format_float(row['full_actor_planner_ausc'])} | "
            f"{_format_float(row['ablated_actor_planner_ausc'])} | "
            f"{_format_float(row['estimate'])} [{_format_float(row['ci_lower_97_5'])}, {_format_float(row['ci_upper_97_5'])}] |"
        )
    lines.extend(
        (
            "",
            "These ablations isolate the two registered seed-1 Actor inputs; they are not whole-system no-VLM/no-ultrasound experiments.",
            "",
            "## Claim status",
            "",
            "| Claim | Task | Status | Evidence |",
            "|---|---|---|---|",
        )
    )
    for row in claims:
        lines.append(
            f"| {row['claim_id']} | {row['task']} | {row['status']} | `{row['evidence']}` |"
        )
    lines.extend(
        (
            "",
            "Historical C1-C3 remain `PROXY_LEGACY`; this document and `reviewed_claims.json` are the reviewed claim authority. Blind report review and human planning comparison remain pending real input.",
            "",
        )
    )
    return "\n".join(lines)


def _evidence_matrix_document(claims: tuple[Mapping[str, object], ...]) -> str:
    lines = [
        "# Reviewed Submission Evidence Matrix",
        "",
        "| Claim | Task | Status | Evidence | Boundary |",
        "|---|---|---|---|---|",
    ]
    for row in claims:
        claim_id = str(row["claim_id"])
        if claim_id.startswith(("R1_", "R2_", "R3_")):
            boundary = (
                "Positive support requires complete coverage and 97.5% CI lower > 0."
            )
        elif claim_id.startswith("R4_"):
            boundary = "Uses frozen Path B joint rule; tasks are not pooled."
        elif claim_id.startswith("R5_"):
            boundary = "Descriptive Reader agreement; no new safety threshold."
        elif claim_id == "R6_PROCESS":
            boundary = "Post hoc process description; no new STOP policy."
        else:
            boundary = "Pending real human input; no proxy substitution."
        lines.append(
            f"| {claim_id} | {row['task']} | {row['status']} | `{row['evidence']}` | {boundary} |"
        )
    lines.append("")
    return "\n".join(lines)


def summarize_expert_rescore(
    *,
    config_path: str | Path,
    source_config_path: str | Path,
    project_root: str | Path,
    legacy_output_root: str | Path,
) -> dict[str, object]:
    """Read completed reviewed outputs and build claim-facing tables and figures."""

    from .frozen_process_analysis import load_frozen_process_config

    verify_output_only_config(source_config_path, config_path)
    config = load_frozen_process_config(config_path, project_root=project_root)
    legacy = Path(legacy_output_root).resolve(strict=True)
    formal_run = json.loads(
        (config.output_root / "formal_run_result.json").read_text(encoding="utf-8")
    )
    input_manifest = json.loads(
        (config.output_root / "inputs/reference_set_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = tuple(
        pl.read_csv(
            config.output_root / "inputs/reference_provenance.csv",
            infer_schema_length=None,
        ).to_dicts()
    )
    recovery_manifest = json.loads(
        (config.output_root / "reviewed/recovery_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    reference_version = str(recovery_manifest["reference_version"])
    if reference_version != input_manifest["reference_version"]:
        raise ValueError("prepared and recovered reference versions differ")
    episode_rows = tuple(
        pl.read_csv(
            config.output_root / "reviewed/per_episode_metrics.csv",
            infer_schema_length=None,
        ).to_dicts()
    )
    full_input_rows = tuple(
        pl.read_csv(
            config.output_root / "reviewed/full_input_readout.csv",
            infer_schema_length=None,
        ).to_dicts()
    )
    planner_effects = tuple(
        pl.read_csv(
            config.output_root / "reviewed/planner_effects.csv",
            infer_schema_length=None,
        ).to_dicts()
    )
    ablation_effects = tuple(
        pl.read_csv(
            config.output_root / "reviewed/actor_input_ablations.csv",
            infer_schema_length=None,
        ).to_dicts()
    )
    decisions = json.loads(
        (config.output_root / "reviewed/task_decisions.json").read_text(
            encoding="utf-8"
        )
    )
    stability_rows = tuple(
        pl.read_csv(
            config.output_root / "reviewed/report_stability_and_delay.csv",
            infer_schema_length=None,
        ).to_dicts()
    )
    invariants = validate_episode_invariants(episode_rows)
    if (
        not invariants["outcome_partition_valid"]
        or not invariants["cost_identity_valid"]
    ):
        raise ValueError("reviewed episode invariants failed")

    tables = config.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    planner_absolute = planner_absolute_rows(episode_rows)
    autonomous_absolute = autonomous_absolute_rows(episode_rows)
    full_input, full_input_failures = _full_input_tables(full_input_rows, provenance)
    ablations = _ablation_summary_rows(episode_rows, ablation_effects)
    episode_comparison = _proxy_reviewed_episode_rows(legacy, episode_rows)
    paired_states, state_comparison = _state_pairing(config)
    process_summary = _process_summary_rows(stability_rows)
    fixed_cases = _fixed_case_rows(paired_states, legacy / "case_manifest.csv")
    claims = _reviewed_claim_rows(
        reference_version=reference_version,
        physical_n=int(input_manifest["physical_specimen_count"]),
        expected_n=config.physical_specimens,
        planner_effects=planner_effects,
        ablation_effects=ablation_effects,
        decisions=decisions,
        full_input=full_input,
    )

    _write_csv(tables / "full_input_reader_summary.csv", full_input)
    _write_csv(
        tables / "full_input_reader_failures.csv",
        full_input_failures,
        (
            "dataset_id",
            "specimen_key",
            "task",
            "iou",
            "recall",
            "relative_area_error",
            "failure_types",
        ),
    )
    _write_csv(tables / "planner_absolute_means.csv", planner_absolute)
    _write_csv(tables / "autonomous_absolute_means.csv", autonomous_absolute)
    _write_csv(tables / "actor_input_ablation_summary.csv", ablations)
    _write_csv(tables / "proxy_reviewed_episode_comparison.csv", episode_comparison)
    flips = tuple(
        row for row in episode_comparison if row["outcome_flip"] != "UNCHANGED"
    )
    _write_csv(
        tables / "stop_status_flips.csv",
        flips,
        tuple(episode_comparison[0]),
    )
    _write_csv(tables / "proxy_reviewed_state_summary.csv", state_comparison)
    _write_csv(tables / "reviewed_process_summary.csv", process_summary)
    _write_csv(tables / "fixed_case_process.csv", fixed_cases)
    claims_payload = {
        "schema_version": 1,
        "reference_set_label": REFERENCE_SET_LABEL,
        "reference_version": reference_version,
        "claims": claims,
    }
    _write_json(config.output_root / "reviewed_claims.json", claims_payload)

    figures = _plot_figures(
        artifact_root=config.artifact_root,
        planner_absolute=planner_absolute,
        planner_effects=planner_effects,
        autonomous_absolute=autonomous_absolute,
        fixed_cases=fixed_cases,
        reference_version=reference_version,
    )
    from PIL import Image

    figure_checks = []
    for figure_path in figures:
        image = np.asarray(Image.open(figure_path).convert("L"))
        figure_checks.append(
            {
                "path": figure_path,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "grayscale_standard_deviation": float(image.std()),
                "nonblank": bool(image.std() > 1.0),
            }
        )
    results_doc = _results_document(
        reference_version=reference_version,
        full_input=full_input,
        planner_absolute=planner_absolute,
        planner_effects=planner_effects,
        autonomous_absolute=autonomous_absolute,
        ablations=ablations,
        decisions=decisions,
        claims=claims,
    )
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    (config.artifact_root / "REVIEWED_RESULTS_AND_CLAIM_BOUNDARIES.md").write_text(
        results_doc, encoding="utf-8"
    )
    (config.artifact_root / "REVIEWED_SUBMISSION_EVIDENCE_MATRIX.md").write_text(
        _evidence_matrix_document(claims), encoding="utf-8"
    )
    uncertain_count = int(input_manifest["uncertain_region_reference_count"])
    protocol_note = "\n".join(
        (
            "# Scoring Protocol Note",
            "",
            "The proxy and reviewed evaluations reuse identical frozen methods, actions, reports, and STOP events, but they do not share an identical scoring protocol.",
            "",
            "LOCATE reviewed scoring uses the expert certain-region bounding box, rejects a full-frame prediction, and enforces the registered measured-support rule. CHARACTERIZE reviewed scoring uses expert certain/uncertain regions, an uncertainty-aware area interval, and per-predicted-component measured support. Historical proxy CHARACTERIZE scoring instead compared with the full-input Reader target and did not apply that component-support check.",
            "",
            f"This pooled set contains {uncertain_count} references with non-empty uncertain regions. Differences in the proxy/reviewed comparison are therefore reported as changes under both reference and protocol, not as a causal effect of annotation alone. All BC/P8 reviewed comparisons use the same reviewed evaluator.",
            "",
        )
    )
    (config.artifact_root / "SCORING_PROTOCOL_NOTE.md").write_text(
        protocol_note, encoding="utf-8"
    )
    next_inputs = """# Next Human Inputs

- R7 blinded report review: `PENDING_INPUT`. The existing HTML tool has 187 actual-STOP reports and records 5 no-STOP episodes; no real returned blind-review CSV was supplied to this run.
- R8 human planning comparison: `PENDING_INPUT`. No real operator scan-session file was supplied.
- These missing tracks do not invalidate the completed pooled-reference rescore and are not replaced by proxy data.
"""
    (config.artifact_root / "NEXT_HUMAN_INPUTS.md").write_text(
        next_inputs, encoding="utf-8"
    )
    readme = """# Pooled Expert C-scan Rescore

Authoritative reviewed results:

1. `REVIEWED_RESULTS_AND_CLAIM_BOUNDARIES.md`
2. `REVIEWED_SUBMISSION_EVIDENCE_MATRIX.md`
3. `SCORING_PROTOCOL_NOTE.md`
4. `../../../results/bc_cscan_expert_pooled_rescore/v1/reviewed_claims.json`

Historical C1-C3 documents generated by the legacy finalize template remain proxy-only and are not the reviewed claim authority.
"""
    (config.artifact_root / "README.md").write_text(readme, encoding="utf-8")

    main_stop_rows = [
        row
        for row in episode_rows
        if row["stop_system"] == "S_BC_CAL" and row["method"] in MAIN_METHODS
    ]
    stopped = sum(bool(row["stopped"]) for row in main_stop_rows)
    recovery = recovery_manifest["recovery"]
    acceptance_checks = {
        "reference_count_24": len(provenance) == 24,
        "domain_count_6": len({row["dataset_id"] for row in provenance}) == 6,
        "source_expert_count_2": input_manifest["source_expert_count"] == 2,
        "full_input_rows_48": len(full_input_rows) == 48,
        "reviewed_episode_rows_576": len(episode_rows) == 576,
        "reviewed_state_rows_55584": paired_states.height == 55_584,
        "stored_action_transitions_55296": recovery["stored_action_transition_count"]
        == 55_296,
        "main_stop_coverage_187_5": len(main_stop_rows) == 192
        and stopped == 187
        and len(main_stop_rows) - stopped == 5,
        "stop_identity_unchanged": all(
            bool(row["stop_identity_unchanged"]) for row in episode_comparison
        ),
        "episode_outcome_partition": bool(invariants["outcome_partition_valid"]),
        "episode_cost_identity": bool(invariants["cost_identity_valid"]),
        "zero_training": recovery["training_updates"] == 0,
        "zero_vlm_calls": recovery["vlm_calls"] == 0,
        "zero_actor_forward_calls": recovery["actor_forward_calls"] == 0,
        "zero_stop_forward_calls": recovery["stop_forward_calls"] == 0,
        "formal_run_once": formal_run["formal_run_count"] == 1,
        "figures_nonblank": len(figure_checks) == 3
        and all(check["nonblank"] for check in figure_checks),
        "pooled_reference_bytes_preserved": all(
            _sha256_bytes(Path(row["pooled_json_path"]).read_bytes())
            == row["reference_sha256"]
            for row in provenance
        ),
    }
    acceptance = {
        "schema_version": 1,
        "status": "PASSED" if all(acceptance_checks.values()) else "FAILED",
        "checks": acceptance_checks,
        "counts": {
            "physical_specimens": len(provenance),
            "reviewed_episodes": len(episode_rows),
            "reviewed_states": paired_states.height,
            "stored_action_transitions": recovery["stored_action_transition_count"],
            "main_actual_stop": stopped,
            "main_no_stop": len(main_stop_rows) - stopped,
            "full_input_reports": len(full_input_rows),
        },
        "figure_checks": figure_checks,
    }
    _write_json(config.output_root / "acceptance_results.json", acceptance)
    if acceptance["status"] != "PASSED":
        raise ValueError("pooled expert rescore acceptance failed")

    manifest = {
        "schema_version": 1,
        "execution_status": "REVIEWED_COMPUTATION_COMPLETE_OTHER_HUMAN_INPUTS_PENDING",
        "reference_input_status": "FULL_TEST_REVIEWED_REFERENCE",
        "reference_set_label": REFERENCE_SET_LABEL,
        "reference_version": reference_version,
        "physical_specimen_count": len(provenance),
        "source_expert_count": input_manifest["source_expert_count"],
        "analysis_grouping": "POOLED_NOT_BY_REVIEWER",
        "source_versions": {
            "development_base_sha": "0e7f15b5c4cb59b05f19b2c598cdc9a9278cc15e",
            "frozen_science_sha": "2102cc4a1726910931dfaaf20e29ad29a20eaf2e",
            "frozen_config_repository_base_sha": config.repository_base_sha,
            "derived_config_sha256": config.config_sha256,
            "proxy_reference_version": "PROXY_LEGACY",
        },
        "reviewed_scientific_claim_statuses": {
            str(row["claim_id"]): row["status"] for row in claims
        },
        "blind_review_status": "PENDING_INPUT",
        "human_planning_status": "PENDING_INPUT",
        "resource_use": recovery,
        "primary_result_files": [
            "reviewed/full_input_readout.csv",
            "reviewed/planner_effects.csv",
            "reviewed/autonomous_effects.csv",
            "reviewed/actor_input_ablations.csv",
            "reviewed/task_decisions.json",
            "tables/planner_absolute_means.csv",
            "tables/autonomous_absolute_means.csv",
            "tables/proxy_reviewed_episode_comparison.csv",
            "reviewed_claims.json",
        ],
        "figure_checks": figure_checks,
    }
    _write_json(config.output_root / "expert_rescore_manifest.json", manifest)
    reproduce = """# Reproduce pooled expert rescore

Run from the repository worktree:

```bash
PYTHONPATH=src python scripts/cscan_human_review.py export-references \\
  --session /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference-20260909070923.json \\
  --packet /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference_packet_001.json \\
  --output-root .local/cscan_human_review_html/expert_reference_exports
PYTHONPATH=src python scripts/cscan_human_review.py export-references \\
  --session /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference-20260909073915.json \\
  --packet /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference_packet_002.json \\
  --output-root .local/cscan_human_review_html/expert_reference_exports
PYTHONPATH=src python scripts/cscan_human_review.py export-references \\
  --session /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference-20260909081350.json \\
  --packet /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference_packet_003.json \\
  --output-root .local/cscan_human_review_html/expert_reference_exports
PYTHONPATH=src python scripts/cscan_human_review.py export-references \\
  --session /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference-20260909082011.json \\
  --packet /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets/reference_packet_004.json \\
  --output-root .local/cscan_human_review_html/expert_reference_exports
PYTHONPATH=src python scripts/rescore_cscan_expert_pool.py prepare \\
  --source-root /home/ww/paper3/cmc_damage_inference \\
  --export-root .local/cscan_human_review_html/expert_reference_exports \\
  --original-session-root /home/ww/diff/.worktrees/cscan-human-review-html/.local/cscan_human_review_html/reference_packets
PYTHONPATH=src python scripts/rescore_cscan_expert_pool.py run \\
  --source-root /home/ww/paper3/cmc_damage_inference
PYTHONPATH=src python scripts/rescore_cscan_expert_pool.py summarize
```

`run` is the sole reviewed recovery command. `summarize` only reads completed reviewed outputs and must not replay recovery.
"""
    (config.output_root / "reproduce.md").write_text(reproduce, encoding="utf-8")
    from .artifacts import write_checksums

    write_checksums(config.output_root, config.output_root / "CHECKSUMS.sha256")
    return {
        "status": acceptance["status"],
        "reference_version": reference_version,
        "physical_specimen_count": len(provenance),
        "reviewed_state_count": paired_states.height,
        "reviewed_episode_count": len(episode_rows),
        "figures": figures,
        "claim_statuses": manifest["reviewed_scientific_claim_statuses"],
    }


def _domain_equal_mean(rows: Iterable[Mapping[str, object]], field: str) -> float:
    domains: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        domains[str(row["dataset_id"])].append(float(row[field]))
    if not domains:
        raise ValueError("summary group is empty")
    return sum(sum(values) / len(values) for values in domains.values()) / len(domains)


def planner_absolute_rows(
    episode_rows: tuple[Mapping[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """Summarize planner AUSC once per episode, including the BC seed mean."""

    selected = [
        row
        for row in episode_rows
        if row.get("stop_system") == "S_BC_CAL" and row.get("method") in MAIN_METHODS
    ]
    keys = [
        (row["specimen_key"], row["task"], row["method"], int(row["seed"]))
        for row in selected
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("planner rows duplicate an episode")
    output = []
    for task in sorted({str(row["task"]) for row in selected}):
        task_rows = [row for row in selected if row["task"] == task]
        for method in MAIN_METHODS:
            group = [row for row in task_rows if row["method"] == method]
            if not group:
                continue
            output.append(
                {
                    "task": task,
                    "method": method,
                    "planner_ausc": _domain_equal_mean(group, "planner_ausc"),
                    "physical_specimen_count": len(
                        {str(row["specimen_key"]) for row in group}
                    ),
                    "domain_count": len({str(row["dataset_id"]) for row in group}),
                }
            )
        by_specimen: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in task_rows:
            if str(row["method"]).startswith("BC_S"):
                by_specimen[str(row["specimen_key"])].append(row)
        bc_rows = []
        for specimen_key, group in by_specimen.items():
            if {str(row["method"]) for row in group} != {"BC_S1", "BC_S2", "BC_S3"}:
                raise ValueError(f"BC seed coverage is incomplete: {specimen_key}")
            bc_rows.append(
                {
                    "specimen_key": specimen_key,
                    "dataset_id": group[0]["dataset_id"],
                    "planner_ausc": sum(float(row["planner_ausc"]) for row in group)
                    / 3.0,
                }
            )
        output.append(
            {
                "task": task,
                "method": "BC_3SEED",
                "planner_ausc": _domain_equal_mean(bc_rows, "planner_ausc"),
                "physical_specimen_count": len(bc_rows),
                "domain_count": len({str(row["dataset_id"]) for row in bc_rows}),
            }
        )
    return tuple(output)


def classify_positive_effect(
    ci_lower: float, physical_n: int, expected_physical_n: int
) -> str:
    if physical_n < expected_physical_n:
        return "INSUFFICIENT_PRECISION"
    return "SUPPORTED" if float(ci_lower) > 0.0 else "NOT_SUPPORTED"


def validate_episode_invariants(
    rows: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    outcome_valid = True
    cost_valid = True
    for row in rows:
        outcome_valid &= (
            int(bool(row["completion"]))
            + int(bool(row["false_stop"]))
            + int(bool(row["exhausted"]))
            == 1
        )
        cost_valid &= math.isclose(
            float(row["autonomous_ausc"]) + float(row["failure_penalized_cost"]),
            1.0,
            abs_tol=1e-12,
        )
    return {
        "episode_count": len(rows),
        "outcome_partition_valid": outcome_valid,
        "cost_identity_valid": cost_valid,
    }


__all__ = [
    "ExpectedReference",
    "ReferenceCandidate",
    "classify_positive_effect",
    "discover_reference_candidates",
    "inherit_proxy_inputs",
    "planner_absolute_rows",
    "pool_reference_files",
    "prepare_expert_pool",
    "run_expert_rescore",
    "summarize_expert_rescore",
    "validate_episode_invariants",
    "verify_output_only_config",
]
