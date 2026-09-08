"""Bounded orchestration for the BC C-scan Path-B supplement."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import polars as pl
import torch
import yaml

from .artifacts import (
    _atomic_write,
    write_csv_atomic,
    write_json_atomic,
)
from .benchmark import _internal_fit_split, _policy_example, _torch_save_atomic
from .runtime import load_study_config, load_study_context

SUPPLEMENT_BASE_SHA = "d8b5b090891fc030931c6dc81e3619a80966f739"
SUPPLEMENT_PROMPT_SHA256 = (
    "ff4e2986d904127225a9fa0c834b38fddcb55860a995d4378031cde2d43a06c9"
)
PARENT_CONFIG_SHA256 = (
    "12268dcaf470f769c326007702f7b0b6f4a13cf326447eef652a6dda972b5662"
)
FROZEN_BC_SHA256 = (
    "c95185aec70515f2578d671e520ebe9b57e600e91e70e5f6752567ae2f87fbe5"
)
FROZEN_STOP_SHA256 = (
    "c6af5c383ef4c586a4ac85c7dc9afbcd5bc71dc07558c692fb295d0f5386892d"
)
FROZEN_BANK_SHA256 = (
    "52167731b60bbd6db759a1e74a85bd0eb0bc8ea208bd51e981bff44d841281c7"
)
FROZEN_BASE_BC_CONTENT_SHA256 = (
    "9d73c3e27de235ffe8337738517aec48f3f4a3267eb5f687ef0f6eeaae3566d1"
)

COHORT_FIELDS = (
    "schema_version",
    "dataset_id",
    "specimen_id",
    "specimen_key",
    "split",
    "domain_rank",
    "reference_type",
    "review_state",
    "reviewer_alias",
    "formal_eligible",
    "formal_success_available",
    "annotation_path",
    "annotation_exists",
    "surface_cache_available",
)


@dataclass(frozen=True, slots=True)
class SupplementConfig:
    """Strict identity and path bindings for supplement-only outputs."""

    path: Path
    project_root: Path
    config_sha256: str
    parent_config_path: Path
    parent_config_sha256: str
    source_result_root: Path
    source_artifact_root: Path
    output_root: Path
    artifact_root: Path
    frozen_bc_path: Path
    frozen_stop_path: Path
    bank_candidates: tuple[Path, ...]
    reference_manifest_path: Path
    annotation_root: Path
    actor_update_cap: int
    total_update_cap: int
    base_episode_transition_cap: int
    confirmation_transition_cap: int
    values: MappingProxyType[str, Any]


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of one regular file."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"hash input is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"hash input cannot be read: {source}") from error
    return digest.hexdigest()


def verify_frozen_file(
    path: str | Path, expected_sha256: str, *, label: str
) -> str:
    """Reject a missing or changed frozen source file."""

    if len(expected_sha256) != 64 or not label:
        raise ValueError("frozen file binding is invalid")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} hash changed: expected {expected_sha256}, got {actual}"
        )
    return actual


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"supplement {label} is invalid")
    return value


def _relative_path(root: Path, value: object, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"supplement {label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"supplement {label} path must stay in the project")
    return (root / relative).resolve(strict=False)


def _source_candidate(root: Path, value: object) -> Path:
    if type(value) is not str or not value:
        raise ValueError("supplement bank candidate is invalid")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError("supplement bank candidate must be workspace-relative")
    resolved = (root / candidate).resolve(strict=False)
    worktrees_root = root.parent.resolve(strict=True)
    if resolved != worktrees_root and worktrees_root not in resolved.parents:
        raise ValueError("supplement bank candidate escapes the worktree area")
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def load_supplement_config(
    path: str | Path, *, project_root: str | Path
) -> SupplementConfig:
    """Load and validate the supplement config without opening evidence splits."""

    root = Path(project_root).resolve(strict=True)
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError("supplement config is invalid YAML") from error
    if type(payload) is not dict:
        raise ValueError("supplement config is invalid")
    parent = _mapping(payload.get("parent"), label="parent")
    references = _mapping(payload.get("references"), label="references")
    training = _mapping(payload.get("training"), label="training")
    stopping = _mapping(payload.get("stopping"), label="stopping")
    evaluation = _mapping(payload.get("evaluation"), label="evaluation")
    outputs = _mapping(payload.get("outputs"), label="outputs")
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "BC_CSCAN_PATH_B_SUPPLEMENT"
        or payload.get("mode") != "retrospective_supplement"
        or payload.get("repository_base_sha") != SUPPLEMENT_BASE_SHA
        or payload.get("controlling_prompt_sha256")
        != SUPPLEMENT_PROMPT_SHA256
        or payload.get("configuration_frozen") is not True
        or parent.get("config_sha256") != PARENT_CONFIG_SHA256
        or parent.get("frozen_bc_sha256") != FROZEN_BC_SHA256
        or parent.get("frozen_stop_sha256") != FROZEN_STOP_SHA256
        or parent.get("bank_sha256") != FROZEN_BANK_SHA256
        or parent.get("base_bc_content_sha256")
        != FROZEN_BASE_BC_CONTENT_SHA256
        or training.get("actor_update_cap") != 16_000
        or training.get("total_update_cap") != 20_000
        or training.get("base_episode_transition_cap") != 180_000
        or training.get("confirmation_transition_cap") != 60_000
        or training.get("original_60_vlm_call_cap") != 0
        or stopping.get("candidate_thresholds") != [0.90, 0.95, 0.99]
        or evaluation.get("tasks") != ["LOCATE", "CHARACTERIZE"]
        or evaluation.get("bootstrap_replicates") != 5000
    ):
        raise ValueError("supplement config identity is invalid")
    candidates_raw = parent.get("bank_candidates")
    if type(candidates_raw) is not list or not candidates_raw:
        raise ValueError("supplement bank candidates are invalid")
    parent_config = _relative_path(
        root, parent.get("config_path"), label="parent config"
    )
    verify_frozen_file(
        parent_config, PARENT_CONFIG_SHA256, label="parent config"
    )
    source_result = _relative_path(
        root, parent.get("result_root"), label="source result"
    )
    source_artifact = _relative_path(
        root, parent.get("artifact_root"), label="source artifact"
    )
    output = _relative_path(root, outputs.get("root"), label="output")
    artifact = _relative_path(root, outputs.get("artifacts"), label="artifact")
    if _paths_overlap(source_result, output) or _paths_overlap(
        source_artifact, artifact
    ):
        raise ValueError("supplement source and destination overlap")
    return SupplementConfig(
        path=source,
        project_root=root,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        parent_config_path=parent_config,
        parent_config_sha256=PARENT_CONFIG_SHA256,
        source_result_root=source_result,
        source_artifact_root=source_artifact,
        output_root=output,
        artifact_root=artifact,
        frozen_bc_path=_relative_path(
            root, parent.get("frozen_bc_path"), label="frozen BC"
        ),
        frozen_stop_path=_relative_path(
            root, parent.get("frozen_stop_path"), label="frozen STOP"
        ),
        bank_candidates=tuple(
            _source_candidate(root, value) for value in candidates_raw
        ),
        reference_manifest_path=_relative_path(
            root, references.get("source_manifest"), label="reference manifest"
        ),
        annotation_root=_relative_path(
            root, references.get("annotation_root"), label="annotation root"
        ),
        actor_update_cap=int(training["actor_update_cap"]),
        total_update_cap=int(training["total_update_cap"]),
        base_episode_transition_cap=int(
            training["base_episode_transition_cap"]
        ),
        confirmation_transition_cap=int(
            training["confirmation_transition_cap"]
        ),
        values=MappingProxyType(payload),
    )


def _selected_bank(config: SupplementConfig) -> Path:
    available = tuple(path for path in config.bank_candidates if path.is_file())
    if not available:
        raise RuntimeError("frozen base-BC bank is unavailable")
    return available[0]


def _git_sha(root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _source_entry(root: Path, path: Path, *, role: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    try:
        label = resolved.relative_to(root).as_posix()
    except ValueError:
        label = str(resolved)
    return {
        "path": label,
        "role": role,
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
        "read_only_source": True,
    }


def _baseline_metrics(source: Path) -> list[dict[str, object]]:
    table = pl.read_csv(source)
    expected = {
        ("LOCATE", "R_BALANCED_P8", "planner_ausc"): 0.6298188483710804,
        ("CHARACTERIZE", "R_BALANCED_P8", "planner_ausc"): 0.4679300987311570,
        ("LOCATE", "L_BC", "planner_ausc"): 0.8099545385296550,
        ("CHARACTERIZE", "L_BC", "planner_ausc"): 0.5488189459859377,
        ("LOCATE", "R_BALANCED_P8", "rule_completion"): 0.875,
        ("CHARACTERIZE", "R_BALANCED_P8", "rule_completion"): 0.875,
        ("LOCATE", "L_BC", "rule_completion"): 0.7083333333333334,
        ("CHARACTERIZE", "L_BC", "rule_completion"): 1.0,
        ("LOCATE", "R_BALANCED_P8", "old_stop_completion"): 0.5833333333333334,
        ("CHARACTERIZE", "R_BALANCED_P8", "old_stop_completion"): 0.6666666666666666,
        ("LOCATE", "L_BC", "old_stop_completion"): 0.625,
        ("CHARACTERIZE", "L_BC", "old_stop_completion"): 0.6666666666666666,
    }
    output: list[dict[str, object]] = []
    source_methods = {
        "R_BALANCED_P8": "R_BALANCED",
        "L_BC": "L_BC",
    }
    specs = (
        ("planner_ausc", "ausc_any"),
        ("rule_completion", "rule_stop_success"),
        ("rule_autonomous_ausc", "rule_autonomous_ausc"),
        ("rule_failure_penalized_cost", "rule_failure_penalized_cost"),
        ("old_stop_completion", "learned_stop_success"),
        ("old_stop_autonomous_ausc", "learned_autonomous_ausc"),
        ("old_stop_failure_penalized_cost", "learned_failure_penalized_cost"),
    )
    for task in ("LOCATE", "CHARACTERIZE"):
        for method in ("R_BALANCED_P8", "L_BC"):
            source_method = source_methods[method]
            rows = table.filter(
                (pl.col("task") == task)
                & (pl.col("method") == source_method)
            )
            if rows.height != 24:
                raise RuntimeError(
                    f"historical metric cohort changed for {task}/{method}"
                )
            for metric, field in specs:
                value = float(rows[field].mean())
                expected_value = expected.get((task, method, metric))
                if expected_value is not None and abs(value - expected_value) > 1e-12:
                    raise RuntimeError(
                        f"historical metric changed for {task}/{method}/{metric}"
                    )
                output.append(
                    {
                        "task": task,
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "physical_specimens": rows.height,
                        "matches_frozen_prompt_value": (
                            None
                            if expected_value is None
                            else abs(value - expected_value) <= 1e-12
                        ),
                    }
                )
    return output


def _cohort_rows(
    split: pl.DataFrame,
    references: pl.DataFrame,
    perception: dict[str, object],
    annotation_root: Path,
) -> tuple[dict[str, object], ...]:
    if split.height != 60 or references.height != 276:
        raise RuntimeError("frozen cohort or reference manifest row count changed")
    reference_by_key = {
        str(row["specimen_key"]): row
        for row in references.iter_rows(named=True)
    }
    cached_keys = {
        str(row["specimen_key"])
        for row in perception.get("records", [])
        if type(row) is dict and row.get("cache_hit") is True
    }
    rows: list[dict[str, object]] = []
    for assignment in split.sort(["dataset_id", "domain_rank"]).iter_rows(
        named=True
    ):
        key = str(assignment["specimen_key"])
        reference = reference_by_key.get(key)
        if reference is None:
            raise RuntimeError(f"reference manifest is missing {key}")
        annotation = str(reference.get("annotation_path") or "")
        annotation_path = annotation_root / Path(annotation).name if annotation else None
        rows.append(
            {
                "schema_version": 1,
                "dataset_id": str(assignment["dataset_id"]),
                "specimen_id": str(assignment["specimen_id"]),
                "specimen_key": key,
                "split": str(assignment["split"]),
                "domain_rank": int(assignment["domain_rank"]),
                "reference_type": str(reference["reference_type"]),
                "review_state": str(reference["review_state"]),
                "reviewer_alias": str(reference.get("reviewer_alias") or ""),
                "formal_eligible": bool(reference["formal_eligible"]),
                "formal_success_available": bool(
                    reference["formal_success_available"]
                ),
                "annotation_path": annotation,
                "annotation_exists": bool(
                    annotation_path is not None and annotation_path.is_file()
                ),
                "surface_cache_available": key in cached_keys,
            }
        )
    if len({row["specimen_key"] for row in rows}) != 60:
        raise RuntimeError("supplement cohort join is not one-to-one")
    return tuple(rows)


def _write_text_atomic(path: Path, text: str) -> None:
    def writer(temporary: Path) -> None:
        temporary.write_text(text, encoding="utf-8", newline="")

    _atomic_write(path, writer)


def audit_supplement(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Verify frozen inputs and materialize the supplement source audit."""

    config = load_supplement_config(config_path, project_root=project_root)
    verify_frozen_file(config.frozen_bc_path, FROZEN_BC_SHA256, label="frozen BC")
    verify_frozen_file(
        config.frozen_stop_path, FROZEN_STOP_SHA256, label="frozen STOP"
    )
    bank_path = _selected_bank(config)
    verify_frozen_file(bank_path, FROZEN_BANK_SHA256, label="frozen BC bank")

    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    context = load_study_context(parent, source_root=source_root)
    source = config.source_result_root
    split_path = source / "split_manifest.csv"
    perception_path = source / "perception_manifest.json"
    reference_path = config.reference_manifest_path
    split = pl.read_csv(split_path)
    references = pl.read_csv(reference_path)
    perception = json.loads(perception_path.read_text(encoding="utf-8"))
    cohort_rows = _cohort_rows(
        split, references, perception, config.annotation_root
    )

    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    if type(bank) is not dict or type(bank.get("base_bc")) is not list:
        raise RuntimeError("frozen bank does not contain base_bc rows")
    base_bc = bank["base_bc"]
    if len(base_bc) != 192:
        raise RuntimeError("frozen base_bc row count changed")
    examples = tuple(_policy_example(row) for row in base_bc)
    fit_examples, valid_examples = _internal_fit_split(examples)
    fit_keys = tuple(sorted({row.specimen_key for row in fit_examples}))
    valid_keys = tuple(sorted({row.specimen_key for row in valid_examples}))
    if (len(fit_examples), len(valid_examples), len(fit_keys), len(valid_keys)) != (
        144,
        48,
        18,
        6,
    ):
        raise RuntimeError("frozen base_bc internal split changed")

    destination_work = config.output_root / "_work"
    _torch_save_atomic(
        {
            "schema_version": 1,
            "parent_config_sha256": PARENT_CONFIG_SHA256,
            "source_bank_sha256": FROZEN_BANK_SHA256,
            "base_bc_content_sha256": FROZEN_BASE_BC_CONTENT_SHA256,
            "base_bc": base_bc,
        },
        destination_work / "base_bc.pt",
    )

    config.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.path, config.output_root / "config.yaml")
    write_csv_atomic(
        config.output_root / "cohort_and_reference_coverage.csv",
        cohort_rows,
        COHORT_FIELDS,
    )
    trajectories = pl.read_parquet(source / "trajectories.parquet")
    trajectory_fields = set(trajectories.columns)
    required_trajectory_fields = {
        "action_cell",
        "action_from_level",
        "action_to_level",
        "report_sha256",
        "rule_stop",
        "learned_stop_probability",
        "learned_stop_eligible",
        "cumulative_route_cost",
    }
    if not required_trajectory_fields <= trajectory_fields:
        raise RuntimeError("frozen trajectories are missing required replay fields")

    baseline_metrics = _baseline_metrics(source / "per_episode_metrics.csv")
    reviewed_counts = {
        split_name: sum(
            row["split"] == split_name
            and row["formal_eligible"]
            and bool(row["reviewer_alias"])
            for row in cohort_rows
        )
        for split_name in ("TRAIN", "VALID", "TEST")
    }
    cache_counts = {
        split_name: sum(
            row["split"] == split_name and row["surface_cache_available"]
            for row in cohort_rows
        )
        for split_name in ("TRAIN", "VALID", "TEST")
    }
    inventory = {
        "schema_version": 1,
        "stage": "E0_SOURCE_AUDIT_COMPLETE",
        "repository_base_sha": SUPPLEMENT_BASE_SHA,
        "execution_commit_sha": _git_sha(config.project_root),
        "config_sha256": config.config_sha256,
        "controlling_prompt_sha256": SUPPLEMENT_PROMPT_SHA256,
        "source_root": str(context.source_root),
        "data_paths": {
            "hasebe": str(context.source_root / "data/public/hasebe"),
            "hasebe_cai": str(context.source_root / "data/public/hasebe_cai"),
        },
        "cohort": {
            "physical_specimens": len(cohort_rows),
            "split_counts": {
                split_name: sum(row["split"] == split_name for row in cohort_rows)
                for split_name in ("TRAIN", "VALID", "TEST")
            },
            "domains": len({row["dataset_id"] for row in cohort_rows}),
        },
        "base_bc": {
            "source_bank_path": str(bank_path),
            "source_bank_sha256": FROZEN_BANK_SHA256,
            "content_sha256": FROZEN_BASE_BC_CONTENT_SHA256,
            "row_count": len(base_bc),
            "fit_rows": len(fit_examples),
            "internal_valid_rows": len(valid_examples),
            "fit_specimens": len(fit_keys),
            "internal_valid_specimens": len(valid_keys),
            "internal_valid_keys": list(valid_keys),
            "destination_cache": "_work/base_bc.pt",
            "only_base_bc_copied": True,
        },
        "frozen_models": {
            "bc_seed_1_sha256": FROZEN_BC_SHA256,
            "old_stop_sha256": FROZEN_STOP_SHA256,
        },
        "surface_cache": {
            "coverage_by_split": cache_counts,
            "original_60_vlm_calls_required": 0,
        },
        "references": {
            "source_manifest_rows": references.height,
            "algorithmic_annotation_proposals": sum(
                row["annotation_exists"] for row in cohort_rows
            ),
            "reviewed_by_split": reviewed_counts,
            "reviewed_total": sum(reviewed_counts.values()),
            "formal_effect": None,
            "proxy_scope": "SAME_READER_SELF_CONSISTENCY",
        },
        "trajectories": {
            "rows": trajectories.height,
            "columns": trajectories.width,
            "required_replay_fields_available": True,
            "predicted_mask_saved": "predicted_mask" in trajectory_fields,
            "support_positions_saved": "support_positions" in trajectory_fields,
            "reviewed_rescore_recovery": "REPLAY_STORED_ACTION_PREFIXES",
        },
        "historical_metrics": baseline_metrics,
        "test_opened_by_supplement": False,
        "new_training_started": False,
    }
    write_json_atomic(config.output_root / "inventory.json", inventory)

    source_files = (
        (config.parent_config_path, "parent_config"),
        (split_path, "cohort_split"),
        (source / "surface_percepts.jsonl", "surface_cache"),
        (perception_path, "surface_cache_manifest"),
        (source / "model_manifest.json", "model_manifest"),
        (config.frozen_bc_path, "frozen_bc_seed_1"),
        (config.frozen_stop_path, "frozen_old_stop"),
        (source / "per_episode_metrics.csv", "historical_episode_metrics"),
        (source / "trajectories.parquet", "historical_trajectories"),
        (reference_path, "reference_manifest_276"),
        (bank_path, "frozen_training_bank"),
    )
    annotation_files = tuple(
        sorted(
            {
                config.annotation_root / Path(str(row["annotation_path"])).name
                for row in cohort_rows
                if row["annotation_exists"]
            }
        )
    )
    manifest = {
        "schema_version": 1,
        "stage": "E0_SOURCE_AUDIT_COMPLETE",
        "source_files": [
            _source_entry(config.project_root, path, role=role)
            for path, role in source_files
        ]
        + [
            _source_entry(
                config.project_root,
                path,
                role="algorithmic_annotation_proposal_not_reviewed",
            )
            for path in annotation_files
        ],
        "source_roots_read_only": [
            str(config.source_result_root),
            str(config.source_artifact_root),
        ],
    }
    write_json_atomic(
        config.output_root / "source_artifact_manifest.json", manifest
    )

    metrics_lines = "\n".join(
        "| {task} | {method} | {metric} | {value:.16g} |".format(**row)
        for row in baseline_metrics
    )
    markdown = f"""# BC C-scan Path-B Inventory and Bindings

## Frozen identity

- Evidence base: `{SUPPLEMENT_BASE_SHA}`
- Execution commit at audit: `{inventory['execution_commit_sha']}`
- Parent config SHA-256: `{PARENT_CONFIG_SHA256}`
- BC seed-1 SHA-256: `{FROZEN_BC_SHA256}`
- Old STOP SHA-256: `{FROZEN_STOP_SHA256}`
- Source bank SHA-256: `{FROZEN_BANK_SHA256}`
- Canonical base-BC content SHA-256: `{FROZEN_BASE_BC_CONTENT_SHA256}`

## Cohort and fit split

- Reused cohort: 60 physical specimens across six domains; TRAIN/VALID/TEST = 24/12/24.
- BC bank: 192 rows; internal fit/validation = 144/48 rows from 18/6 specimens.
- Internal-validation keys: `{', '.join(valid_keys)}`.
- Hasebe data: `{context.source_root / 'data/public/hasebe'}`.
- Hasebe CAI data: `{context.source_root / 'data/public/hasebe_cai'}`; no CAI training is performed.

## Cache and reference coverage

- Frozen surface-cache coverage by TRAIN/VALID/TEST: {cache_counts['TRAIN']}/{cache_counts['VALID']}/{cache_counts['TEST']}.
- Independently reviewed reference coverage by TRAIN/VALID/TEST: {reviewed_counts['TRAIN']}/{reviewed_counts['VALID']}/{reviewed_counts['TEST']}.
- The 276-row reference manifest remains algorithm-derived and unreviewed; formal effects are null.

## Replay sufficiency

- Historical trajectories contain actions, report hashes, rule STOP, learned STOP probability and eligibility, cumulative route cost, and turn counts.
- Predicted masks and support positions are not stored. Reviewed rescoring must reconstruct reports by replaying stored action prefixes; a report hash alone is not rescored.
- The source bank is read-only. Only its 192 `base_bc` rows are copied to the ignored destination `_work/base_bc.pt` cache.

## Frozen historical metric replay

| Task | Method | Metric | Mean over 24 TEST specimens |
|---|---|---|---:|
{metrics_lines}

These values are same-Reader proxy evidence. This audit performs no fitting, no new TEST evaluation, and no VLM calls.
"""
    _write_text_atomic(
        config.artifact_root / "inventory_and_bindings.md", markdown
    )
    return {
        "stage": "E0_SOURCE_AUDIT_COMPLETE",
        "output": str(config.output_root),
        "cohort": len(cohort_rows),
        "base_bc_rows": len(base_bc),
        "reviewed_references": sum(reviewed_counts.values()),
        "surface_cache_coverage": sum(cache_counts.values()),
        "training_started": False,
        "test_opened": False,
    }


__all__ = [
    "FROZEN_BC_SHA256",
    "SupplementConfig",
    "audit_supplement",
    "file_sha256",
    "load_supplement_config",
    "verify_frozen_file",
]
