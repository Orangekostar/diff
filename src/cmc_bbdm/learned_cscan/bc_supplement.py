"""Bounded orchestration for the BC C-scan Path-B supplement."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml

from .artifacts import (
    _atomic_write,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)
from .benchmark import (
    _actor_action_and_scores,
    _advance_detailed,
    _full_reference,
    _internal_fit_split,
    _internal_stop_split,
    _load_actor,
    _load_percepts,
    _load_stop,
    _packet,
    _policy_example,
    _proxy_score,
    _run_planner_episode,
    _stop_example,
    _torch_save_atomic,
)
from .contracts import Split, Task
from .episode_stop_calibration import calibrate_episode_stop
from .policies import LearnedCellActor, RuleMethod, select_rule_action
from .runtime import (
    StudyContext,
    load_study_config,
    load_study_context,
    open_study_specimen,
)
from .stopping import LearnedStopHead
from .supplement_adapters import (
    ActorInputMode,
    load_supplement_actor,
    save_supplement_actor,
    transform_policy_example,
)
from .training import (
    ActorFitResult,
    TrainingRoute,
    fit_actor,
    fit_stop_head,
)

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

TRAINING_LOG_FIELDS = (
    "schema_version",
    "method",
    "seed",
    "actor_input_mode",
    "optimizer_step",
    "train_loss",
    "valid_loss",
)

_ACTOR_SPECS = (
    ("BC_S2", 2, ActorInputMode.FULL, "bc_seed2.pt"),
    ("BC_S3", 3, ActorInputMode.FULL, "bc_seed3.pt"),
    ("BC_NO_VLM_S1", 1, ActorInputMode.NO_VLM, "bc_no_vlm_seed1.pt"),
    (
        "BC_NO_US_FEEDBACK_S1",
        1,
        ActorInputMode.NO_US_FEEDBACK,
        "bc_no_us_feedback_seed1.pt",
    ),
)

VALIDATION_TRAJECTORY_FIELDS = (
    "schema_version",
    "dataset_id",
    "specimen_id",
    "specimen_key",
    "task",
    "planner",
    "seed",
    "step",
    "cost",
    "stop_model",
    "stop_probability",
    "mechanically_eligible",
    "rule_stop",
    "report_id",
    "reference_score_for_calibration_only",
    "task_loss_for_calibration_only",
    "action_cell",
    "action_from_level",
    "action_to_level",
    "cumulative_route_cost",
    "cumulative_route_turns",
)

_SUPPLEMENT_WORKER_CONTEXT: StudyContext | None = None
_SUPPLEMENT_WORKER_PERCEPTS: dict[str, object] | None = None
_SUPPLEMENT_WORKER_PLANNERS: tuple[tuple[str, object, int, int], ...] = ()
_SUPPLEMENT_WORKER_STOP: LearnedStopHead | None = None
_SUPPLEMENT_WORKER_THRESHOLDS: dict[str, float | None] = {}
_SUPPLEMENT_WORKER_DEVICE = "cpu"

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


@dataclass(frozen=True, slots=True)
class TrueBreakResult:
    terminal_step: int
    stopped: bool
    exhausted: bool
    state: object
    packet: object


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


def planned_checkpoint_paths(config: SupplementConfig) -> tuple[Path, ...]:
    """Return all new Actor destinations under the supplement result root."""

    if type(config) is not SupplementConfig:
        raise TypeError("issued supplement config is required")
    paths = tuple(config.output_root / "models" / spec[3] for spec in _ACTOR_SPECS)
    if any(
        config.output_root not in path.parents
        or config.source_result_root in path.parents
        or path in {config.frozen_bc_path, config.frozen_stop_path}
        for path in paths
    ):
        raise RuntimeError("supplement checkpoint destination is unsafe")
    return paths


def require_calibration_split(split: Split) -> None:
    """Restrict STOP threshold selection to the frozen VALID split."""

    if type(split) is not Split:
        raise TypeError("typed split is required")
    if split is not Split.VALID:
        raise ValueError("STOP calibration requires VALID")


def run_true_break(
    *,
    initial_state: object,
    packet_for_state: Callable[[object], object],
    should_stop: Callable[[object], bool],
    select_action: Callable[[object], object | None],
    advance: Callable[[object, object], object],
    exhausted: Callable[[object], bool],
    max_actions: int,
) -> TrueBreakResult:
    """Execute actions only until the first visible STOP or exhaustion."""

    if (
        not callable(packet_for_state)
        or not callable(should_stop)
        or not callable(select_action)
        or not callable(advance)
        or not callable(exhausted)
        or type(max_actions) is not int
        or max_actions < 0
    ):
        raise ValueError("true-break execution request is invalid")
    state = initial_state
    for step in range(max_actions + 1):
        packet = packet_for_state(state)
        decision = should_stop(packet)
        terminal = exhausted(packet)
        if type(decision) is not bool or type(terminal) is not bool:
            raise TypeError("true-break predicates must return booleans")
        if decision:
            return TrueBreakResult(step, True, False, state, packet)
        if terminal:
            return TrueBreakResult(step, False, True, state, packet)
        if step == max_actions:
            raise RuntimeError("true-break action cap exceeded")
        action = select_action(packet)
        if action is None:
            raise RuntimeError("true-break planner returned no legal action")
        state = advance(state, action)
    raise AssertionError("unreachable true-break state")


def _base_bc_examples(
    config: SupplementConfig,
) -> tuple[Any, ...]:
    source_bank = _selected_bank(config)
    verify_frozen_file(source_bank, FROZEN_BANK_SHA256, label="frozen BC bank")
    cache_path = config.output_root / "_work/base_bc.pt"
    if not cache_path.is_file():
        raise RuntimeError("run supplement audit before training")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if (
        type(payload) is not dict
        or set(payload)
        != {
            "schema_version",
            "parent_config_sha256",
            "source_bank_sha256",
            "base_bc_content_sha256",
            "base_bc",
        }
        or payload.get("schema_version") != 1
        or payload.get("parent_config_sha256") != PARENT_CONFIG_SHA256
        or payload.get("source_bank_sha256") != FROZEN_BANK_SHA256
        or payload.get("base_bc_content_sha256")
        != FROZEN_BASE_BC_CONTENT_SHA256
        or type(payload.get("base_bc")) is not list
        or len(payload["base_bc"]) != 192
    ):
        raise RuntimeError("audited base_bc cache identity changed")
    return tuple(_policy_example(row) for row in payload["base_bc"])


def _existing_training_state(
    config: SupplementConfig,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    manifest_path = config.output_root / "model_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            type(manifest) is not dict
            or manifest.get("schema_version") != 1
            or manifest.get("parent_config_sha256") != PARENT_CONFIG_SHA256
            or manifest.get("source_bank_sha256") != FROZEN_BANK_SHA256
            or type(manifest.get("models")) is not list
        ):
            raise RuntimeError("supplement model manifest identity changed")
    else:
        manifest = {
            "schema_version": 1,
            "state": "ACTOR_SUPPLEMENT_TRAINING_STARTED",
            "parent_config_sha256": PARENT_CONFIG_SHA256,
            "source_bank_sha256": FROZEN_BANK_SHA256,
            "base_bc_content_sha256": FROZEN_BASE_BC_CONTENT_SHA256,
            "frozen_bc_seed_1": {
                "path": config.frozen_bc_path.relative_to(
                    config.project_root
                ).as_posix(),
                "sha256": FROZEN_BC_SHA256,
                "retrained": False,
            },
            "models": [],
            "resource_use": {
                "actor_optimizer_updates": 0,
                "actor_update_cap": config.actor_update_cap,
                "base_episode_transitions": 0,
                "base_episode_transition_cap": config.base_episode_transition_cap,
                "original_60_vlm_calls": 0,
            },
            "test_opened": False,
        }
    log_path = config.output_root / "training_log.csv"
    logs = pl.read_csv(log_path).to_dicts() if log_path.is_file() else []
    return manifest, logs


def _model_record(
    config: SupplementConfig,
    *,
    method: str,
    seed: int,
    mode: ActorInputMode,
    path: Path,
    fit: ActorFitResult,
    parameter_count: int,
    fit_examples: tuple[Any, ...],
    valid_examples: tuple[Any, ...],
) -> dict[str, object]:
    return {
        "method": method,
        "seed": seed,
        "actor_input_mode": mode.value,
        "path": path.relative_to(config.output_root).as_posix(),
        "sha256": file_sha256(path),
        "parameter_count": parameter_count,
        "optimizer_steps": fit.optimizer_steps,
        "initial_loss": fit.initial_loss,
        "final_loss": fit.final_loss,
        "best_valid_loss": fit.best_valid_loss,
        "stopped_early": fit.stopped_early,
        "fit_rows": len(fit_examples),
        "internal_valid_rows": len(valid_examples),
        "fit_specimens": len({row.specimen_key for row in fit_examples}),
        "internal_valid_specimens": len(
            {row.specimen_key for row in valid_examples}
        ),
        "training_route": TrainingRoute.BEHAVIOR_CLONING.value,
    }


def _verify_recovered_model(
    config: SupplementConfig,
    record: dict[str, object],
    *,
    method: str,
    seed: int,
    mode: ActorInputMode,
) -> bool:
    if (
        record.get("method") != method
        or record.get("seed") != seed
        or record.get("actor_input_mode") != mode.value
        or type(record.get("path")) is not str
        or type(record.get("sha256")) is not str
    ):
        return False
    path = (config.output_root / str(record["path"])).resolve(strict=True)
    if config.output_root not in path.parents:
        raise RuntimeError("supplement checkpoint escaped the result root")
    verify_frozen_file(path, str(record["sha256"]), label=method)
    loaded = load_supplement_actor(
        path, parent_config_sha256=PARENT_CONFIG_SHA256
    )
    if (
        loaded.method != method
        or loaded.seed != seed
        or loaded.view.mode is not mode
        or loaded.parameter_count != 329_505
    ):
        raise RuntimeError(f"recovered supplement checkpoint changed: {method}")
    return True


def _train_actor_stage(
    *,
    config_path: Path,
    project_root: Path,
    source_root: Path,
    requested_methods: tuple[str, ...],
) -> dict[str, object]:
    started = time.perf_counter()
    config = load_supplement_config(config_path, project_root=project_root)
    verify_frozen_file(config.frozen_bc_path, FROZEN_BC_SHA256, label="frozen BC")
    verify_frozen_file(
        config.frozen_stop_path, FROZEN_STOP_SHA256, label="frozen STOP"
    )
    audited = json.loads(
        (config.output_root / "inventory.json").read_text(encoding="utf-8")
    )
    actual_source = str(Path(source_root).resolve(strict=True))
    if (
        audited.get("stage") != "E0_SOURCE_AUDIT_COMPLETE"
        or audited.get("source_root") != actual_source
        or audited.get("test_opened_by_supplement") is not False
    ):
        raise RuntimeError("supplement source audit is missing or changed")
    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    training = parent.values["training"]
    if (
        int(training["max_optimizer_steps"]) != 4000
        or float(training["learning_rate"]) != 0.0003
        or float(training["weight_decay"]) != 0.0001
        or float(training["gradient_clip"]) != 1.0
        or int(training["validation_interval"]) != 250
        or int(training["validation_patience"]) != 4
    ):
        raise RuntimeError("parent BC training contract changed")
    base_examples = _base_bc_examples(config)
    manifest, logs = _existing_training_state(config)
    existing = {
        str(row["method"]): row
        for row in manifest["models"]
        if type(row) is dict and "method" in row
    }
    specifications = tuple(
        spec for spec in _ACTOR_SPECS if spec[0] in requested_methods
    )
    if {spec[0] for spec in specifications} != set(requested_methods):
        raise ValueError("requested supplement Actor set is invalid")
    recovered = []
    trained = []
    device = str(config.values["training"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("configured CUDA device is unavailable")
    if device.startswith("cuda"):
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    for method, seed, mode, filename in specifications:
        old_record = existing.get(method)
        if old_record is not None:
            if not _verify_recovered_model(
                config,
                old_record,
                method=method,
                seed=seed,
                mode=mode,
            ):
                raise RuntimeError(f"supplement checkpoint identity changed: {method}")
            recovered.append(method)
            continue
        transformed = tuple(
            transform_policy_example(example, mode) for example in base_examples
        )
        fit_examples, valid_examples = _internal_fit_split(transformed)
        if (len(fit_examples), len(valid_examples)) != (144, 48):
            raise RuntimeError("supplement Actor fit split changed")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        actor = LearnedCellActor(
            use_surface_features=mode is not ActorInputMode.NO_VLM
        )
        fit = fit_actor(
            actor,
            fit_examples,
            route=TrainingRoute.BEHAVIOR_CLONING,
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
        path = config.output_root / "models" / filename
        save_supplement_actor(
            path,
            actor,
            method=method,
            seed=seed,
            mode=mode,
            parent_config_sha256=PARENT_CONFIG_SHA256,
            optimizer_steps=fit.optimizer_steps,
        )
        record = _model_record(
            config,
            method=method,
            seed=seed,
            mode=mode,
            path=path,
            fit=fit,
            parameter_count=actor.parameter_count,
            fit_examples=fit_examples,
            valid_examples=valid_examples,
        )
        manifest["models"].append(record)
        existing[method] = record
        logs.extend(
            {
                "schema_version": 1,
                "method": method,
                "seed": seed,
                "actor_input_mode": mode.value,
                "optimizer_step": row.optimizer_step,
                "train_loss": row.train_loss,
                "valid_loss": row.valid_loss,
            }
            for row in fit.log
        )
        trained.append(method)
        print(
            f"trained {method} seed={seed} mode={mode.value} "
            f"steps={fit.optimizer_steps} "
            f"loss={fit.initial_loss:.6f}->{fit.final_loss:.6f}",
            flush=True,
        )
        actor.to("cpu")
        del actor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    manifest["models"] = sorted(
        manifest["models"], key=lambda row: str(row["method"])
    )
    total_updates = sum(int(row["optimizer_steps"]) for row in manifest["models"])
    if total_updates > config.actor_update_cap:
        raise RuntimeError("supplement Actor update cap exceeded")
    complete = {row["method"] for row in manifest["models"]} == {
        spec[0] for spec in _ACTOR_SPECS
    }
    manifest["state"] = (
        "ACTOR_SUPPLEMENT_TRAINING_COMPLETE"
        if complete
        else "ACTOR_SUPPLEMENT_TRAINING_PARTIAL"
    )
    manifest["device"] = device
    manifest["resource_use"] = {
        "actor_optimizer_updates": total_updates,
        "actor_update_cap": config.actor_update_cap,
        "base_episode_transitions": 0,
        "base_episode_transition_cap": config.base_episode_transition_cap,
        "original_60_vlm_calls": 0,
    }
    current_peak = (
        int(torch.cuda.max_memory_allocated(torch.device(device)))
        if device.startswith("cuda")
        else 0
    )
    manifest["peak_gpu_memory_bytes"] = max(
        int(manifest.get("peak_gpu_memory_bytes", 0)), current_peak
    )
    manifest["last_stage_elapsed_seconds"] = time.perf_counter() - started
    manifest["test_opened"] = False
    write_csv_atomic(
        config.output_root / "training_log.csv",
        tuple(
            sorted(
                logs,
                key=lambda row: (
                    str(row["method"]),
                    int(row["seed"]),
                    int(row["optimizer_step"]),
                ),
            )
        ),
        TRAINING_LOG_FIELDS,
    )
    write_json_atomic(config.output_root / "model_manifest.json", manifest)
    return {
        "stage": manifest["state"],
        "trained": trained,
        "recovered": recovered,
        "actor_optimizer_updates": total_updates,
        "actor_update_cap": config.actor_update_cap,
        "test_opened": False,
        "elapsed_seconds": manifest["last_stage_elapsed_seconds"],
    }


def train_replicas(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Train the fixed BC replicas for seeds two and three."""

    return _train_actor_stage(
        config_path=config_path,
        project_root=project_root,
        source_root=source_root,
        requested_methods=("BC_S2", "BC_S3"),
    )


def train_ablations(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Independently train the two fixed seed-one Actor-input ablations."""

    return _train_actor_stage(
        config_path=config_path,
        project_root=project_root,
        source_root=source_root,
        requested_methods=("BC_NO_VLM_S1", "BC_NO_US_FEEDBACK_S1"),
    )


def _load_supplement_stop(path: Path, *, device: str) -> LearnedStopHead:
    payload = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("stage") != "BC_CSCAN_PATH_B_SUPPLEMENT"
        or payload.get("method") != "S_BC_CAL"
        or payload.get("parent_config_sha256") != PARENT_CONFIG_SHA256
        or payload.get("parameter_count") != 13_697
        or type(payload.get("state_dict")) is not dict
    ):
        raise RuntimeError("supplement STOP checkpoint identity changed")
    model = LearnedStopHead()
    model.load_state_dict(payload["state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 13_697:
        raise RuntimeError("supplement STOP parameter count changed")
    model.to(torch.device(device))
    model.eval()
    return model


def _initialize_supplement_evaluation_worker(
    config_path: str,
    project_root: str,
    source_root: str,
    planner_specs: tuple[tuple[str, str, str, int, int], ...],
    stop_kind: str,
    stop_path: str,
    thresholds: dict[str, float | None],
) -> None:
    global _SUPPLEMENT_WORKER_CONTEXT
    global _SUPPLEMENT_WORKER_DEVICE
    global _SUPPLEMENT_WORKER_PERCEPTS
    global _SUPPLEMENT_WORKER_PLANNERS
    global _SUPPLEMENT_WORKER_STOP
    global _SUPPLEMENT_WORKER_THRESHOLDS
    supplement = load_supplement_config(
        Path(config_path), project_root=Path(project_root)
    )
    parent = load_study_config(
        supplement.parent_config_path, project_root=supplement.project_root
    )
    context = load_study_context(parent, source_root=Path(source_root))
    device = str(supplement.values["training"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("configured evaluation CUDA device is unavailable")
    planners = []
    for method, kind, reference, period, seed in planner_specs:
        if kind == "RULE":
            planner: object = RuleMethod(reference)
        elif kind == "FROZEN_BC":
            planner = _load_actor(supplement.frozen_bc_path, parent, device)
        elif kind == "SUPPLEMENT_ACTOR":
            checkpoint = load_supplement_actor(
                supplement.output_root / reference,
                parent_config_sha256=PARENT_CONFIG_SHA256,
                device=device,
            )
            if checkpoint.method != method or checkpoint.seed != seed:
                raise RuntimeError("supplement evaluation Actor identity changed")
            planner = checkpoint.view
        else:
            raise RuntimeError("supplement planner specification is invalid")
        planners.append((method, planner, period, seed))
    if stop_kind == "FROZEN_STOP":
        stop = _load_stop(Path(stop_path), parent, device)
    elif stop_kind == "SUPPLEMENT_STOP":
        stop = _load_supplement_stop(Path(stop_path), device=device)
    else:
        raise RuntimeError("supplement STOP specification is invalid")
    _SUPPLEMENT_WORKER_CONTEXT = context
    _SUPPLEMENT_WORKER_DEVICE = device
    _SUPPLEMENT_WORKER_PERCEPTS = _load_percepts(
        parent, context, supplement.source_result_root
    )
    _SUPPLEMENT_WORKER_PLANNERS = tuple(planners)
    _SUPPLEMENT_WORKER_STOP = stop
    _SUPPLEMENT_WORKER_THRESHOLDS = thresholds


def _supplement_evaluation_worker(
    specimen_key: str, *, store_trajectory: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if (
        _SUPPLEMENT_WORKER_CONTEXT is None
        or _SUPPLEMENT_WORKER_PERCEPTS is None
        or _SUPPLEMENT_WORKER_STOP is None
        or not _SUPPLEMENT_WORKER_PLANNERS
    ):
        raise RuntimeError("supplement evaluation worker is not initialized")
    context = _SUPPLEMENT_WORKER_CONTEXT
    record = next(
        row for row in context.roster.pilot_records if row.specimen_key == specimen_key
    )
    runtime = open_study_specimen(context, record)
    percept = _SUPPLEMENT_WORKER_PERCEPTS[specimen_key]
    episodes = []
    trajectories = []
    for task in Task:
        reference = _full_reference(context, runtime, task)
        for method, planner, period, seed in _SUPPLEMENT_WORKER_PLANNERS:
            result, trajectory, _stop_rows = _run_planner_episode(
                runtime,
                percept=percept,
                task=task,
                reference=reference,
                context=context,
                method_name=method,
                planner=planner,
                coverage_period=period,
                seed=seed,
                device=_SUPPLEMENT_WORKER_DEVICE,
                stop_model=_SUPPLEMENT_WORKER_STOP,
                learned_threshold=_SUPPLEMENT_WORKER_THRESHOLDS.get(task.value),
                store_trajectory=store_trajectory,
            )
            episodes.append(result)
            trajectories.extend(trajectory)
    return episodes, trajectories


def _collect_stop_training_worker(
    specimen_key: str,
) -> tuple[list[dict[str, object]], int]:
    if (
        _SUPPLEMENT_WORKER_CONTEXT is None
        or _SUPPLEMENT_WORKER_PERCEPTS is None
        or not _SUPPLEMENT_WORKER_PLANNERS
    ):
        raise RuntimeError("supplement training worker is not initialized")
    context = _SUPPLEMENT_WORKER_CONTEXT
    record = next(
        row for row in context.roster.pilot_records if row.specimen_key == specimen_key
    )
    runtime = open_study_specimen(context, record)
    percept = _SUPPLEMENT_WORKER_PERCEPTS[specimen_key]
    selected_steps = tuple(
        int(value) for value in np.rint(np.linspace(0, 192, 16)).astype(np.int64)
    )
    if len(set(selected_steps)) != 16:
        raise RuntimeError("predetermined STOP training states are not unique")
    rows: list[dict[str, object]] = []
    transitions = 0
    for task in Task:
        reference = _full_reference(context, runtime, task)
        for planner_name, planner, period, seed in _SUPPLEMENT_WORKER_PLANNERS:
            observation = runtime.world.reset()
            probe = (0.0, 0.0)
            route_cost = 0.0
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
                if step in selected_steps:
                    score = _proxy_score(packet.report, reference)
                    rows.append(
                        {
                            "specimen_key": specimen_key,
                            "task": task.value,
                            "planner": planner_name,
                            "planner_seed": seed,
                            "step": step,
                            "cost": float(observation.effective_budget),
                            "cell_features": np.asarray(packet.cell_features),
                            "subblock_features": np.asarray(
                                packet.subblock_features
                            ),
                            "global_features": np.asarray(
                                packet.global_features
                            ),
                            "history_features": np.asarray(
                                packet.history_features
                            ),
                            "label": bool(score["success"]),
                        }
                    )
                terminal = all(level == 2 for level in packet.cell_levels)
                if terminal:
                    break
                if isinstance(planner, RuleMethod):
                    action = select_rule_action(
                        planner,
                        packet,
                        grid=runtime.grid,
                        coverage_period=period,
                    )
                else:
                    action, _scores = _actor_action_and_scores(
                        planner, packet, runtime, _SUPPLEMENT_WORKER_DEVICE
                    )
                observation, probe, route_cost, _turns = _advance_detailed(
                    runtime, observation, action, probe, route_cost
                )
                transitions += 1
                step += 1
                if step > 192:
                    raise RuntimeError("STOP training trajectory exceeded 192 actions")
            selected = [
                row
                for row in rows
                if row["task"] == task.value
                and row["planner"] == planner_name
            ]
            if len(selected) != 16:
                raise RuntimeError("STOP training state count changed")
    return rows, transitions


def _collect_stop_training_rows(
    config: SupplementConfig, *, source_root: Path
) -> tuple[list[dict[str, object]], int]:
    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    context = load_study_context(parent, source_root=source_root)
    assignments = tuple(
        row for row in context.roster.assignments if row.split is Split.TRAIN
    )
    planner_specs = (
        ("BC_S1", "FROZEN_BC", "", 4, 1),
        (
            "R_BALANCED_P8",
            "RULE",
            RuleMethod.R_BALANCED.value,
            8,
            1,
        ),
    )
    rows: list[dict[str, object]] = []
    transitions = 0
    with ProcessPoolExecutor(
        max_workers=min(
            int(parent.values["training"]["cpu_workers"]), len(assignments)
        ),
        mp_context=get_context("spawn"),
        initializer=_initialize_supplement_evaluation_worker,
        initargs=(
            str(config.path),
            str(config.project_root),
            str(Path(source_root).resolve(strict=True)),
            planner_specs,
            "FROZEN_STOP",
            str(config.frozen_stop_path),
            {},
        ),
    ) as executor:
        futures = tuple(
            executor.submit(
                _collect_stop_training_worker, assignment.record.specimen_key
            )
            for assignment in assignments
        )
        for index, future in enumerate(futures, start=1):
            specimen_rows, specimen_transitions = future.result()
            rows.extend(specimen_rows)
            transitions += specimen_transitions
            print(
                f"TRAIN STOP states {index}/{len(assignments)} "
                f"rows={len(rows)} transitions={transitions}",
                flush=True,
            )
    if len(rows) != 24 * 2 * 2 * 16 or transitions != 24 * 2 * 2 * 192:
        raise RuntimeError("conditional STOP training roster changed")
    return rows, transitions


def _save_supplement_stop(
    path: Path,
    model: LearnedStopHead,
    *,
    optimizer_steps: int,
) -> None:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 13_697 or not 0 < optimizer_steps <= 4000:
        raise RuntimeError("supplement STOP fit identity is invalid")
    payload = {
        "schema_version": 1,
        "stage": "BC_CSCAN_PATH_B_SUPPLEMENT",
        "method": "S_BC_CAL",
        "seed": 1,
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "source_bank_sha256": FROZEN_BANK_SHA256,
        "reference_version": "PROXY_LEGACY",
        "parameter_count": parameter_count,
        "optimizer_steps": optimizer_steps,
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
    }
    _atomic_write(path, lambda temporary: torch.save(payload, temporary))


def _fit_conditional_stop(
    config: SupplementConfig, *, source_root: Path
) -> tuple[Path, dict[str, object], int]:
    model_path = config.output_root / "models/s_bc_cal.pt"
    manifest_path = config.output_root / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior_record = manifest.get("conditional_stop")
    bank_path = config.output_root / "_work/stop_training_bank.pt"
    if prior_record is not None:
        if (
            type(prior_record) is not dict
            or prior_record.get("method") != "S_BC_CAL"
            or prior_record.get("seed") != 1
            or prior_record.get("reference_version") != "PROXY_LEGACY"
            or prior_record.get("parameter_count") != 13_697
            or type(prior_record.get("sha256")) is not str
        ):
            raise RuntimeError("conditional STOP manifest identity changed")
        verify_frozen_file(
            model_path, str(prior_record["sha256"]), label="S_BC_CAL"
        )
        _load_supplement_stop(model_path, device="cpu")
        return model_path, prior_record, int(
            prior_record["training_world_transitions"]
        )
    if bank_path.is_file():
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        if (
            type(bank) is not dict
            or bank.get("schema_version") != 1
            or bank.get("parent_config_sha256") != PARENT_CONFIG_SHA256
            or bank.get("source")
            != "TRAIN_NATURAL_BC_S1_AND_R_BALANCED_P8"
            or type(bank.get("rows")) is not list
            or len(bank["rows"]) != 1536
            or bank.get("world_transitions") != 18_432
        ):
            raise RuntimeError("conditional STOP training cache changed")
        rows = bank["rows"]
        transitions = int(bank["world_transitions"])
    else:
        rows, transitions = _collect_stop_training_rows(
            config, source_root=source_root
        )
        _torch_save_atomic(
            {
                "schema_version": 1,
                "parent_config_sha256": PARENT_CONFIG_SHA256,
                "source": "TRAIN_NATURAL_BC_S1_AND_R_BALANCED_P8",
                "states_per_specimen_task_planner": 16,
                "selection": "ROUND_LINSPACE_ACTION_INDEX_0_TO_192",
                "reference_version": "PROXY_LEGACY",
                "world_transitions": transitions,
                "rows": rows,
            },
            bank_path,
        )
    examples = tuple(_stop_example(row) for row in rows)
    fit_examples, valid_examples = _internal_stop_split(examples)
    if (
        len(examples) != 1536
        or len(fit_examples) != 1152
        or len(valid_examples) != 384
        or len({row.specimen_key for row in fit_examples}) != 18
        or len({row.specimen_key for row in valid_examples}) != 6
    ):
        raise RuntimeError("conditional STOP internal split changed")
    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    training = parent.values["training"]
    device = str(config.values["training"]["device"])
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    model = LearnedStopHead()
    fit = fit_stop_head(
        model,
        fit_examples,
        max_steps=int(training["max_optimizer_steps"]),
        batch_size=min(32, len(fit_examples)),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        seed=1,
        device=device,
        validation_examples=valid_examples,
        validation_interval=int(training["validation_interval"]),
        patience=int(training["validation_patience"]),
    )
    _save_supplement_stop(model_path, model, optimizer_steps=fit.optimizer_steps)
    record: dict[str, object] = {
        "method": "S_BC_CAL",
        "seed": 1,
        "path": model_path.relative_to(config.output_root).as_posix(),
        "sha256": file_sha256(model_path),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "optimizer_steps": fit.optimizer_steps,
        "initial_loss": fit.initial_loss,
        "final_loss": fit.final_loss,
        "best_valid_loss": fit.best_valid_loss,
        "stopped_early": fit.stopped_early,
        "fit_rows": len(fit_examples),
        "internal_valid_rows": len(valid_examples),
        "fit_specimens": 18,
        "internal_valid_specimens": 6,
        "states_per_specimen_task_planner": 16,
        "reference_version": "PROXY_LEGACY",
        "training_world_transitions": transitions,
        "retrained_once": True,
        "test_opened": False,
    }
    logs = pl.read_csv(config.output_root / "training_log.csv").to_dicts()
    logs.extend(
        {
            "schema_version": 1,
            "method": "S_BC_CAL",
            "seed": 1,
            "actor_input_mode": "STOP_FULL_VISIBLE",
            "optimizer_step": row.optimizer_step,
            "train_loss": row.train_loss,
            "valid_loss": row.valid_loss,
        }
        for row in fit.log
    )
    write_csv_atomic(
        config.output_root / "training_log.csv",
        tuple(
            sorted(
                logs,
                key=lambda row: (
                    str(row["method"]),
                    int(row["seed"]),
                    int(row["optimizer_step"]),
                ),
            )
        ),
        TRAINING_LOG_FIELDS,
    )
    actor_updates = int(
        manifest["resource_use"]["actor_optimizer_updates"]
    )
    if actor_updates + fit.optimizer_steps > config.total_update_cap:
        raise RuntimeError("supplement total optimizer update cap exceeded")
    manifest["conditional_stop"] = record
    manifest["resource_use"]["conditional_stop_optimizer_updates"] = (
        fit.optimizer_steps
    )
    manifest["resource_use"]["total_optimizer_updates"] = (
        actor_updates + fit.optimizer_steps
    )
    manifest["resource_use"]["total_update_cap"] = config.total_update_cap
    manifest["resource_use"]["stop_training_world_transitions"] = transitions
    write_json_atomic(manifest_path, manifest)
    print(
        f"trained S_BC_CAL steps={fit.optimizer_steps} "
        f"loss={fit.initial_loss:.6f}->{fit.final_loss:.6f}",
        flush=True,
    )
    model.to("cpu")
    return model_path, record, transitions


def _evaluate_supplement_assignments(
    config: SupplementConfig,
    *,
    source_root: Path,
    split: Split,
    planner_specs: tuple[tuple[str, str, str, int, int], ...],
    stop_kind: str,
    stop_path: Path,
    thresholds: dict[str, float | None],
    store_trajectory: bool,
    label: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    parent = load_study_config(
        config.parent_config_path, project_root=config.project_root
    )
    context = load_study_context(parent, source_root=source_root)
    assignments = tuple(
        row for row in context.roster.assignments if row.split is split
    )
    if not assignments:
        raise RuntimeError("supplement evaluation split is empty")
    episodes: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=min(
            int(parent.values["training"]["cpu_workers"]), len(assignments)
        ),
        mp_context=get_context("spawn"),
        initializer=_initialize_supplement_evaluation_worker,
        initargs=(
            str(config.path),
            str(config.project_root),
            str(Path(source_root).resolve(strict=True)),
            planner_specs,
            stop_kind,
            str(stop_path),
            thresholds,
        ),
    ) as executor:
        futures = tuple(
            executor.submit(
                _supplement_evaluation_worker,
                assignment.record.specimen_key,
                store_trajectory=store_trajectory,
            )
            for assignment in assignments
        )
        for index, future in enumerate(futures, start=1):
            specimen_episodes, specimen_trajectories = future.result()
            episodes.extend(specimen_episodes)
            trajectories.extend(specimen_trajectories)
            print(
                f"{label} {index}/{len(assignments)} "
                f"episodes={len(episodes)} trajectories={len(trajectories)}",
                flush=True,
            )
    return episodes, trajectories


def _validation_rows(
    trajectories: list[dict[str, object]], *, stop_model: str
) -> tuple[dict[str, object], ...]:
    rows = tuple(
        {
            "schema_version": 1,
            "dataset_id": str(row["dataset_id"]),
            "specimen_id": str(row["specimen_id"]),
            "specimen_key": str(row["specimen_key"]),
            "task": str(row["task"]),
            "planner": str(row["method"]),
            "seed": int(row["seed"]),
            "step": int(row["step"]),
            "cost": float(row["cost"]),
            "stop_model": stop_model,
            "stop_probability": float(row["learned_stop_probability"]),
            "mechanically_eligible": bool(row["learned_stop_eligible"]),
            "rule_stop": bool(row["rule_stop"]),
            "report_id": str(row["report_sha256"]),
            "reference_score_for_calibration_only": bool(row["success"]),
            "task_loss_for_calibration_only": float(row["task_loss"]),
            "action_cell": int(row["action_cell"]),
            "action_from_level": int(row["action_from_level"]),
            "action_to_level": int(row["action_to_level"]),
            "cumulative_route_cost": float(row["cumulative_route_cost"]),
            "cumulative_route_turns": int(row["cumulative_route_turns"]),
        }
        for row in trajectories
    )
    if not rows or any(set(row) != set(VALIDATION_TRAJECTORY_FIELDS) for row in rows):
        raise RuntimeError("VALID trajectory export is invalid")
    return rows


def _rule_completion_from_trajectories(
    rows: tuple[dict[str, object], ...], *, task: str, planner: str
) -> float:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["task"] == task and row["planner"] == planner:
            grouped.setdefault(str(row["specimen_key"]), []).append(row)
    if len(grouped) != 12:
        raise RuntimeError("VALID rule completion cohort changed")
    completed = []
    for episode in grouped.values():
        first = next((row for row in episode if row["rule_stop"]), None)
        completed.append(
            bool(first is not None and first["reference_score_for_calibration_only"])
        )
    return float(np.mean(completed))


def _calibration_payload(
    rows: tuple[dict[str, object], ...], *, head_name: str, head_sha256: str
) -> dict[str, object]:
    planners = ("BC_S1", "R_BALANCED_P8")
    task_results = {}
    selected = {}
    for task in ("LOCATE", "CHARACTERIZE"):
        rule_completion = {
            planner: _rule_completion_from_trajectories(
                rows, task=task, planner=planner
            )
            for planner in planners
        }
        task_rows = tuple(
            {
                **row,
                "success": bool(row["reference_score_for_calibration_only"]),
            }
            for row in rows
            if row["task"] == task
        )
        result = calibrate_episode_stop(
            task_rows,
            task=task,
            planners=planners,
            rule_completion=rule_completion,
        )
        task_results[task] = asdict(result)
        selected[task] = result.selected_threshold
    qualified = all(task_results[task]["qualified"] for task in task_results)
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "VALID_EPISODE_FIRST_STOP_CALIBRATED",
        "fit_split": "NONE_FROZEN_HEAD",
        "calibration_split": Split.VALID.value,
        "reference_version": "PROXY_LEGACY",
        "head": head_name,
        "head_sha256": head_sha256,
        "candidate_thresholds": [0.90, 0.95, 0.99],
        "planners": list(planners),
        "task_results": task_results,
        "selected_thresholds": selected,
        "all_tasks_qualified": qualified,
        "status": (
            "S_EP_QUALIFIED" if qualified else "S_EP_NOT_QUALIFIED"
        ),
        "test_opened": False,
    }
    identity = dict(payload)
    payload["calibration_id"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def calibrate_stop(
    *, config_path: Path, project_root: Path, source_root: Path
) -> dict[str, object]:
    """Collect VALID trajectories and calibrate the frozen STOP by episode."""

    config = load_supplement_config(config_path, project_root=project_root)
    require_calibration_split(Split.VALID)
    if (config.output_root / "per_episode_metrics.csv").exists():
        raise RuntimeError("STOP calibration cannot run after supplement TEST")
    calibration_path = config.output_root / "stop_calibration.json"
    if calibration_path.is_file():
        previous = json.loads(calibration_path.read_text(encoding="utf-8"))
        if previous.get("locked_for_test") is True:
            stored_id = previous.pop("calibration_id", None)
            computed_id = hashlib.sha256(
                json.dumps(previous, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            previous["calibration_id"] = stored_id
            if (
                stored_id != computed_id
                or file_sha256(
                    config.output_root / "validation_episode_scores.parquet"
                )
                != previous.get("validation_trajectory_sha256")
            ):
                raise RuntimeError("locked STOP calibration identity changed")
            return {
                "stage": previous["stage"],
                "head": previous["head"],
                "status": previous["status"],
                "selected_thresholds": previous["selected_thresholds"],
                "conditional_refit_required": False,
                "locked_for_test": True,
                "validation_trajectory_rows": previous[
                    "validation_trajectory_rows"
                ],
                "validation_world_transitions": previous[
                    "validation_world_transitions"
                ],
                "test_opened": False,
                "recovered": True,
            }
    verify_frozen_file(
        config.frozen_stop_path, FROZEN_STOP_SHA256, label="frozen STOP"
    )
    manifest = json.loads(
        (config.output_root / "model_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("state") != "ACTOR_SUPPLEMENT_TRAINING_COMPLETE":
        raise RuntimeError("Actor supplement training is incomplete")
    trajectory_path = config.output_root / "validation_episode_scores.parquet"
    if trajectory_path.is_file():
        all_rows = tuple(pl.read_parquet(trajectory_path).to_dicts())
        rows = tuple(row for row in all_rows if row.get("stop_model") == "S_OLD")
        if len(rows) != 9264:
            raise RuntimeError("stored VALID STOP trajectory identity changed")
        transitions = 12 * 2 * 2 * 192
        recovered = True
    else:
        episodes, trajectories = _evaluate_supplement_assignments(
            config,
            source_root=source_root,
            split=Split.VALID,
            planner_specs=(
                ("BC_S1", "FROZEN_BC", "", 4, 1),
                (
                    "R_BALANCED_P8",
                    "RULE",
                    RuleMethod.R_BALANCED.value,
                    8,
                    1,
                ),
            ),
            stop_kind="FROZEN_STOP",
            stop_path=config.frozen_stop_path,
            thresholds={},
            store_trajectory=True,
            label="VALID STOP calibration",
        )
        if len(episodes) != 48:
            raise RuntimeError("VALID episode count changed")
        transitions = sum(int(row["action_count"]) for row in episodes)
        rows = _validation_rows(trajectories, stop_model="S_OLD")
        write_parquet_atomic(trajectory_path, rows)
        recovered = False
    if transitions > config.base_episode_transition_cap:
        raise RuntimeError("supplement transition cap exceeded during VALID")
    initial_payload = _calibration_payload(
        rows, head_name="S_OLD", head_sha256=FROZEN_STOP_SHA256
    )
    if initial_payload["all_tasks_qualified"]:
        final_rows = rows
        payload = initial_payload
        payload["conditional_refit_required"] = False
        payload["conditional_refit_performed"] = False
        payload["locked_for_test"] = True
        total_transitions = transitions
    else:
        stop_path, fit_record, training_transitions = _fit_conditional_stop(
            config, source_root=source_root
        )
        new_cache = config.output_root / "_work/validation_s_bc_cal.parquet"
        if new_cache.is_file():
            calibrated_rows = tuple(pl.read_parquet(new_cache).to_dicts())
            if (
                len(calibrated_rows) != 9264
                or any(
                    row.get("stop_model") != "S_BC_CAL"
                    for row in calibrated_rows
                )
            ):
                raise RuntimeError("stored S_BC_CAL VALID trajectories changed")
            calibrated_transitions = 12 * 2 * 2 * 192
        else:
            calibrated_episodes, calibrated_trajectories = (
                _evaluate_supplement_assignments(
                    config,
                    source_root=source_root,
                    split=Split.VALID,
                    planner_specs=(
                        ("BC_S1", "FROZEN_BC", "", 4, 1),
                        (
                            "R_BALANCED_P8",
                            "RULE",
                            RuleMethod.R_BALANCED.value,
                            8,
                            1,
                        ),
                    ),
                    stop_kind="SUPPLEMENT_STOP",
                    stop_path=stop_path,
                    thresholds={},
                    store_trajectory=True,
                    label="VALID S_BC_CAL calibration",
                )
            )
            if len(calibrated_episodes) != 48:
                raise RuntimeError("S_BC_CAL VALID episode count changed")
            calibrated_transitions = sum(
                int(row["action_count"]) for row in calibrated_episodes
            )
            calibrated_rows = _validation_rows(
                calibrated_trajectories, stop_model="S_BC_CAL"
            )
            write_parquet_atomic(new_cache, calibrated_rows)
        old_by_key = {
            (
                row["specimen_key"],
                row["task"],
                row["planner"],
                row["seed"],
                row["step"],
            ): row
            for row in rows
        }
        calibrated_by_key = {
            (
                row["specimen_key"],
                row["task"],
                row["planner"],
                row["seed"],
                row["step"],
            ): row
            for row in calibrated_rows
        }
        if set(old_by_key) != set(calibrated_by_key):
            raise RuntimeError("STOP recalibration trajectories are not paired")
        frozen_fields = (
            "dataset_id",
            "specimen_id",
            "cost",
            "rule_stop",
            "report_id",
            "reference_score_for_calibration_only",
            "task_loss_for_calibration_only",
            "action_cell",
            "action_from_level",
            "action_to_level",
            "cumulative_route_cost",
            "cumulative_route_turns",
        )
        if any(
            old_by_key[key][field] != calibrated_by_key[key][field]
            for key in old_by_key
            for field in frozen_fields
        ):
            raise RuntimeError("STOP head changed the frozen planner trajectory")
        final_rows = (*rows, *calibrated_rows)
        write_parquet_atomic(trajectory_path, final_rows)
        payload = _calibration_payload(
            calibrated_rows,
            head_name="S_BC_CAL",
            head_sha256=str(fit_record["sha256"]),
        )
        payload["fit_split"] = "TRAIN_INTERNAL_18_6"
        payload["status"] = (
            "S_BC_CAL_QUALIFIED"
            if payload["all_tasks_qualified"]
            else "S_BC_CAL_NOT_QUALIFIED"
        )
        payload["initial_frozen_head_calibration"] = initial_payload
        payload["conditional_stop_fit"] = fit_record
        payload["conditional_refit_required"] = False
        payload["conditional_refit_performed"] = True
        payload["locked_for_test"] = True
        total_transitions = (
            transitions + training_transitions + calibrated_transitions
        )
    if total_transitions > config.base_episode_transition_cap:
        raise RuntimeError("supplement transition cap exceeded during calibration")
    payload["validation_physical_specimens"] = 12
    payload["validation_episode_count"] = 48
    payload["validation_trajectory_rows"] = len(final_rows)
    payload["validation_world_transitions"] = total_transitions
    payload["validation_trajectory_sha256"] = file_sha256(trajectory_path)
    payload["recovered_trajectory"] = recovered
    payload.pop("calibration_id")
    payload["calibration_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json_atomic(calibration_path, payload)
    return {
        "stage": payload["stage"],
        "head": payload["head"],
        "status": payload["status"],
        "selected_thresholds": payload["selected_thresholds"],
        "conditional_refit_required": payload["conditional_refit_required"],
        "locked_for_test": payload["locked_for_test"],
        "validation_trajectory_rows": len(final_rows),
        "validation_world_transitions": total_transitions,
        "test_opened": False,
        "recovered": False,
    }


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
    "TrueBreakResult",
    "audit_supplement",
    "calibrate_stop",
    "file_sha256",
    "load_supplement_config",
    "planned_checkpoint_paths",
    "require_calibration_split",
    "run_true_break",
    "train_ablations",
    "train_replicas",
    "verify_frozen_file",
]
