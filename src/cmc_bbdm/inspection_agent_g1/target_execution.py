"""Hash-bound target policy trajectories frozen before target-truth access."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.state import InspectionCellAction

from .contracts import ACTION_SLOT_COUNT, TaskTokenMode
from .rollout import (
    ClosedLoopTrajectory,
    ObservablePolicyScores,
    RolloutStep,
    SurfaceVariant,
)


class G1TargetExecutionError(ValueError):
    """Raised when a frozen target trajectory cannot be verified exactly."""


class TargetPolicyVariant(str, Enum):
    PROPOSED = "PROPOSED"
    PROPOSED_STOP = "PROPOSED_STOP"
    NO_TASK = "NO_TASK"
    WRONG_TASK = "WRONG_TASK"
    NO_SURFACE = "NO_SURFACE"
    SHUFFLED_SURFACE = "SHUFFLED_SURFACE"


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


_VARIANT_MODES = {
    TargetPolicyVariant.PROPOSED: (
        TaskTokenMode.CORRECT,
        SurfaceVariant.CORRECT_SURFACE,
        False,
    ),
    TargetPolicyVariant.PROPOSED_STOP: (
        TaskTokenMode.CORRECT,
        SurfaceVariant.CORRECT_SURFACE,
        True,
    ),
    TargetPolicyVariant.NO_TASK: (
        TaskTokenMode.NO_TASK,
        SurfaceVariant.CORRECT_SURFACE,
        False,
    ),
    TargetPolicyVariant.WRONG_TASK: (
        TaskTokenMode.WRONG_TASK,
        SurfaceVariant.CORRECT_SURFACE,
        False,
    ),
    TargetPolicyVariant.NO_SURFACE: (
        TaskTokenMode.CORRECT,
        SurfaceVariant.NO_SURFACE,
        False,
    ),
    TargetPolicyVariant.SHUFFLED_SURFACE: (
        TaskTokenMode.CORRECT,
        SurfaceVariant.SHUFFLED_SURFACE,
        False,
    ),
}


@dataclass(frozen=True, slots=True)
class G1TargetTrajectoryRecord:
    outer_target: str
    specimen_id: str
    specimen_sha256: str
    task: InspectionTask
    variant: TargetPolicyVariant
    task_token_mode: TaskTokenMode
    surface_variant: SurfaceVariant
    donor_surface_sha256: str | None
    final_dependency_sha256: str
    action_selection_sha256: str
    action_model_sha256: str
    stop_model_sha256: str
    trajectory: ClosedLoopTrajectory
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        expected = _VARIANT_MODES.get(self.variant)
        has_donor = self.donor_surface_sha256 is not None
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.specimen_id) is not str
            or not self.specimen_id
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or type(self.variant) is not TargetPolicyVariant
            or type(self.task_token_mode) is not TaskTokenMode
            or type(self.surface_variant) is not SurfaceVariant
            or expected is None
            or (self.task_token_mode, self.surface_variant) != expected[:2]
            or has_donor
            != (self.surface_variant is SurfaceVariant.SHUFFLED_SURFACE)
            or (has_donor and not _valid_sha256(self.donor_surface_sha256))
            or not all(
                _valid_sha256(value)
                for value in (
                    self.final_dependency_sha256,
                    self.action_selection_sha256,
                    self.action_model_sha256,
                    self.stop_model_sha256,
                )
            )
            or type(self.trajectory) is not ClosedLoopTrajectory
            or self.trajectory.target_domain != self.outer_target
            or self.trajectory.specimen_sha256 != self.specimen_sha256
            or self.trajectory.task is not self.task
            or self.trajectory.model_sha256 != self.stop_model_sha256
            or (self.trajectory.stop_threshold is not None) != expected[2]
        ):
            raise G1TargetExecutionError("target trajectory record is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-trajectory-record",
                    "outer_target": self.outer_target,
                    "specimen_id": self.specimen_id,
                    "specimen_sha256": self.specimen_sha256,
                    "task": self.task.value,
                    "variant": self.variant.value,
                    "task_token_mode": self.task_token_mode.value,
                    "surface_variant": self.surface_variant.value,
                    "donor_surface": self.donor_surface_sha256,
                    "final_dependency": self.final_dependency_sha256,
                    "action_selection": self.action_selection_sha256,
                    "action_model": self.action_model_sha256,
                    "stop_model": self.stop_model_sha256,
                    "trajectory": self.trajectory.state_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1TargetTrajectoryBankFile:
    row_count: int
    outer_target: str
    parquet_sha256: str
    records_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.row_count) is not int
            or self.row_count <= 0
            or type(self.outer_target) is not str
            or not self.outer_target
            or not all(
                _valid_sha256(value)
                for value in (
                    self.parquet_sha256,
                    self.records_sha256,
                    self.manifest_sha256,
                )
            )
        ):
            raise G1TargetExecutionError("target trajectory bank identity is invalid")


def _record_key(record: G1TargetTrajectoryRecord) -> tuple[str, str, str]:
    return record.specimen_id, record.task.value, record.variant.value


def _ordered_records(
    records: tuple[G1TargetTrajectoryRecord, ...],
) -> tuple[G1TargetTrajectoryRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1TargetTrajectoryRecord for row in records)
        or len({_record_key(row) for row in records}) != len(records)
        or len({row.state_sha256 for row in records}) != len(records)
        or len({row.trajectory.state_sha256 for row in records}) != len(records)
        or len({row.outer_target for row in records}) != 1
        or len({row.final_dependency_sha256 for row in records}) != 1
        or len({row.action_selection_sha256 for row in records}) != 1
        or len({row.action_model_sha256 for row in records}) != 1
        or len({row.stop_model_sha256 for row in records}) != 1
    ):
        raise G1TargetExecutionError("target trajectory bank roster is invalid")
    return tuple(sorted(records, key=_record_key))


def _records_sha(records: tuple[G1TargetTrajectoryRecord, ...]) -> str:
    return _json_sha(tuple(row.state_sha256 for row in records))


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.manifest.json")


def _action_payload(action: InspectionCellAction) -> list[int]:
    return [action.cell_index, action.from_level, action.to_level]


def _step_payload(step: RolloutStep) -> dict[str, object]:
    return {
        "step_index": step.step_index,
        "observation_sha256": step.observation_sha256,
        "policy_state_sha256": step.policy_state_sha256,
        "model_sha256": step.scores.model_sha256,
        "stop_probability": step.scores.stop_probability,
        "selected_slot": step.selected_slot,
        "scores_sha256": step.scores.state_sha256,
        "step_sha256": step.state_sha256,
    }


def _record_row(record: G1TargetTrajectoryRecord) -> dict[str, object]:
    trajectory = record.trajectory
    logits = np.ascontiguousarray(
        np.stack([step.scores.action_logits for step in trajectory.steps]),
        dtype="<f8",
    )
    positions = np.ascontiguousarray(trajectory.acquired_positions, dtype="<i8")
    values = np.ascontiguousarray(trajectory.acquired_values, dtype=np.uint8)
    return {
        "outer_target": record.outer_target,
        "specimen_id": record.specimen_id,
        "specimen_sha256": record.specimen_sha256,
        "task": record.task.value,
        "variant": record.variant.value,
        "task_token_mode": record.task_token_mode.value,
        "surface_variant": record.surface_variant.value,
        "donor_surface_sha256": record.donor_surface_sha256,
        "final_dependency_sha256": record.final_dependency_sha256,
        "action_selection_sha256": record.action_selection_sha256,
        "action_model_sha256": record.action_model_sha256,
        "stop_model_sha256": record.stop_model_sha256,
        "stop_threshold": trajectory.stop_threshold,
        "stopped": trajectory.stopped,
        "termination_reason": trajectory.termination_reason,
        "action_history_json": json.dumps(
            [_action_payload(action) for action in trajectory.action_history],
            separators=(",", ":"),
        ),
        "steps_json": json.dumps(
            [_step_payload(step) for step in trajectory.steps],
            separators=(",", ":"),
        ),
        "action_logits_le_f64": logits.tobytes(order="C"),
        "final_observation_sha256": trajectory.final_observation_sha256,
        "acquired_count": len(positions),
        "acquired_positions_le_i64": positions.tobytes(order="C"),
        "acquired_values_u8": values.tobytes(order="C"),
        "native_count": trajectory.native_count,
        "effective_budget": trajectory.effective_budget,
        "acquired_positions_sha256": trajectory.acquired_positions_sha256,
        "acquired_values_sha256": trajectory.acquired_values_sha256,
        "trajectory_sha256": trajectory.state_sha256,
        "record_sha256": record.state_sha256,
    }


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_g1_target_trajectory_bank(
    path: str | Path,
    records: tuple[G1TargetTrajectoryRecord, ...],
) -> G1TargetTrajectoryBankFile:
    destination = Path(path)
    ordered = _ordered_records(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        pl.DataFrame(
            [_record_row(record) for record in ordered], infer_schema_length=None
        ).write_parquet(
            temporary,
            compression="zstd",
            statistics=False,
            row_group_size=128,
        )
        parquet_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        records_sha = _records_sha(ordered)
        manifest = {
            "schema_version": 1,
            "scope": "inspection_agent_g1_target_trajectory_bank",
            "row_count": len(ordered),
            "outer_target": ordered[0].outer_target,
            "final_dependency_sha256": ordered[0].final_dependency_sha256,
            "action_selection_sha256": ordered[0].action_selection_sha256,
            "action_model_sha256": ordered[0].action_model_sha256,
            "stop_model_sha256": ordered[0].stop_model_sha256,
            "parquet_sha256": parquet_sha,
            "records_sha256": records_sha,
        }
        manifest_payload = (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("ascii")
        os.replace(temporary, destination)
        _atomic_replace_bytes(_manifest_path(destination), manifest_payload)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    identity, replay = read_g1_target_trajectory_bank(destination)
    if tuple(row.state_sha256 for row in replay) != tuple(
        row.state_sha256 for row in ordered
    ):
        raise G1TargetExecutionError(
            "written target trajectory bank did not replay exactly"
        )
    return identity


def _decode_actions(raw: object) -> tuple[InspectionCellAction, ...]:
    parsed = json.loads(str(raw))
    if not isinstance(parsed, list):
        raise TypeError
    return tuple(
        InspectionCellAction(int(value[0]), int(value[1]), int(value[2]))
        for value in parsed
        if isinstance(value, list) and len(value) == 3
    )


def _decode_steps(row: dict[str, object]) -> tuple[RolloutStep, ...]:
    parsed = json.loads(str(row["steps_json"]))
    if not isinstance(parsed, list) or not parsed:
        raise TypeError
    payload = bytes(row["action_logits_le_f64"])
    logits = np.frombuffer(payload, dtype="<f8")
    if logits.size != len(parsed) * ACTION_SLOT_COUNT:
        raise TypeError
    logits = logits.reshape(len(parsed), ACTION_SLOT_COUNT)
    steps = []
    for value, action_logits in zip(parsed, logits, strict=True):
        if not isinstance(value, dict):
            raise TypeError
        scores = ObservablePolicyScores(
            policy_state_sha256=str(value["policy_state_sha256"]),
            model_sha256=str(value["model_sha256"]),
            action_logits=action_logits,
            stop_probability=float(value["stop_probability"]),
            state_sha256=str(value["scores_sha256"]),
        )
        selected_raw = value["selected_slot"]
        step = RolloutStep(
            step_index=int(value["step_index"]),
            observation_sha256=str(value["observation_sha256"]),
            policy_state_sha256=str(value["policy_state_sha256"]),
            scores=scores,
            selected_slot=None if selected_raw is None else int(selected_raw),
            state_sha256=str(value["step_sha256"]),
        )
        steps.append(step)
    return tuple(steps)


def _record_from_row(row: dict[str, object]) -> G1TargetTrajectoryRecord:
    try:
        count = int(row["acquired_count"])
        positions = np.frombuffer(
            bytes(row["acquired_positions_le_i64"]), dtype="<i8"
        )
        values = np.frombuffer(bytes(row["acquired_values_u8"]), dtype=np.uint8)
        if count <= 0 or positions.size != count * 2 or values.size != count * 3:
            raise TypeError
        threshold_raw = row["stop_threshold"]
        trajectory = ClosedLoopTrajectory(
            target_domain=str(row["outer_target"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            model_sha256=str(row["stop_model_sha256"]),
            stop_threshold=(
                None if threshold_raw is None else float(threshold_raw)
            ),
            stopped=bool(row["stopped"]),
            termination_reason=str(row["termination_reason"]),
            action_history=_decode_actions(row["action_history_json"]),
            steps=_decode_steps(row),
            final_observation_sha256=str(row["final_observation_sha256"]),
            acquired_positions=positions.reshape(count, 2),
            acquired_values=values.reshape(count, 3),
            native_count=int(row["native_count"]),
            effective_budget=float(row["effective_budget"]),
        )
        if (
            trajectory.acquired_positions_sha256
            != str(row["acquired_positions_sha256"])
            or trajectory.acquired_values_sha256
            != str(row["acquired_values_sha256"])
            or trajectory.state_sha256 != str(row["trajectory_sha256"])
        ):
            raise G1TargetExecutionError("target trajectory hash changed")
        record = G1TargetTrajectoryRecord(
            outer_target=str(row["outer_target"]),
            specimen_id=str(row["specimen_id"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            variant=TargetPolicyVariant(str(row["variant"])),
            task_token_mode=TaskTokenMode(str(row["task_token_mode"])),
            surface_variant=SurfaceVariant(str(row["surface_variant"])),
            donor_surface_sha256=(
                None
                if row["donor_surface_sha256"] is None
                else str(row["donor_surface_sha256"])
            ),
            final_dependency_sha256=str(row["final_dependency_sha256"]),
            action_selection_sha256=str(row["action_selection_sha256"]),
            action_model_sha256=str(row["action_model_sha256"]),
            stop_model_sha256=str(row["stop_model_sha256"]),
            trajectory=trajectory,
        )
        if record.state_sha256 != str(row["record_sha256"]):
            raise G1TargetExecutionError("target trajectory record hash changed")
        return record
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ) as error:
        if isinstance(error, G1TargetExecutionError):
            raise
        raise G1TargetExecutionError(
            "target trajectory row cannot be reconstructed"
        ) from error


def read_g1_target_trajectory_bank(
    path: str | Path,
) -> tuple[G1TargetTrajectoryBankFile, tuple[G1TargetTrajectoryRecord, ...]]:
    source = Path(path)
    try:
        parquet_payload = source.read_bytes()
        manifest_payload = _manifest_path(source).read_bytes()
        manifest = json.loads(manifest_payload)
        rows = pl.read_parquet(source).to_dicts()
    except (OSError, UnicodeError, json.JSONDecodeError, pl.exceptions.PolarsError) as error:
        raise G1TargetExecutionError(
            "target trajectory bank cannot be read"
        ) from error
    expected_keys = {
        "schema_version",
        "scope",
        "row_count",
        "outer_target",
        "final_dependency_sha256",
        "action_selection_sha256",
        "action_model_sha256",
        "stop_model_sha256",
        "parquet_sha256",
        "records_sha256",
    }
    parquet_sha = hashlib.sha256(parquet_payload).hexdigest()
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("scope") != "inspection_agent_g1_target_trajectory_bank"
        or manifest.get("parquet_sha256") != parquet_sha
    ):
        raise G1TargetExecutionError("target trajectory bank manifest changed")
    records = tuple(_record_from_row(row) for row in rows)
    ordered = _ordered_records(records)
    records_sha = _records_sha(ordered)
    first = ordered[0]
    if (
        records != ordered
        or manifest.get("row_count") != len(records)
        or manifest.get("outer_target") != first.outer_target
        or manifest.get("final_dependency_sha256")
        != first.final_dependency_sha256
        or manifest.get("action_selection_sha256")
        != first.action_selection_sha256
        or manifest.get("action_model_sha256") != first.action_model_sha256
        or manifest.get("stop_model_sha256") != first.stop_model_sha256
        or manifest.get("records_sha256") != records_sha
    ):
        raise G1TargetExecutionError("target trajectory bank evidence changed")
    identity = G1TargetTrajectoryBankFile(
        row_count=len(records),
        outer_target=first.outer_target,
        parquet_sha256=parquet_sha,
        records_sha256=records_sha,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )
    return identity, records


__all__ = [
    "G1TargetExecutionError",
    "G1TargetTrajectoryBankFile",
    "G1TargetTrajectoryRecord",
    "TargetPolicyVariant",
    "read_g1_target_trajectory_bank",
    "write_g1_target_trajectory_bank",
]
