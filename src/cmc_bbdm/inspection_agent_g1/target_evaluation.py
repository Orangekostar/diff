"""Evaluation-only target curves issued after the complete trajectory-bank seal."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.mavis.contracts import EvaluationView

from .formal import evaluate_g1_action_history
from .g1 import G1Runtime, build_g1_world
from .metrics import (
    EngineeringCurve,
    replay_engineering_curve,
    validate_engineering_curve,
)
from .rollout import TargetTruthVault
from .target_execution import (
    G1TargetExecutionError,
    G1TargetTrajectoryBankFile,
    G1TargetTrajectoryBankSeal,
    G1TargetTrajectoryRecord,
    TargetPolicyVariant,
    validate_g1_target_trajectory_bank_seal,
)


class G1TargetEvaluationError(ValueError):
    """Raised when target truth is requested without a matching frozen bank."""


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


@dataclass(frozen=True, slots=True)
class G1TargetCurveRecord:
    outer_target: str
    specimen_id: str
    specimen_sha256: str
    task: InspectionTask
    variant: TargetPolicyVariant
    target_record_sha256: str
    trajectory_sha256: str
    bank_seal_sha256: str
    trajectory_seal_sha256: str
    truth_view_sha256: str
    curve: EngineeringCurve
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            validate_engineering_curve(self.curve)
        except ValueError as error:
            raise G1TargetEvaluationError("target engineering curve changed") from error
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.specimen_id) is not str
            or not self.specimen_id
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or type(self.variant) is not TargetPolicyVariant
            or not all(
                _valid_sha256(value)
                for value in (
                    self.target_record_sha256,
                    self.trajectory_sha256,
                    self.bank_seal_sha256,
                    self.trajectory_seal_sha256,
                    self.truth_view_sha256,
                )
            )
            or self.curve.method != self.variant.value
            or self.curve.target_domain != self.outer_target
            or self.curve.specimen_sha256 != self.specimen_sha256
            or self.curve.task is not self.task
        ):
            raise G1TargetEvaluationError("target curve record is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-curve-record",
                    "outer_target": self.outer_target,
                    "specimen_id": self.specimen_id,
                    "specimen_sha256": self.specimen_sha256,
                    "task": self.task.value,
                    "variant": self.variant.value,
                    "target_record": self.target_record_sha256,
                    "trajectory": self.trajectory_sha256,
                    "bank_seal": self.bank_seal_sha256,
                    "trajectory_seal": self.trajectory_seal_sha256,
                    "truth_view": self.truth_view_sha256,
                    "curve": self.curve.state_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1TargetCurveBankFile:
    row_count: int
    outer_target: str
    trajectory_bank_seal_sha256: str
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
                    self.trajectory_bank_seal_sha256,
                    self.parquet_sha256,
                    self.records_sha256,
                    self.manifest_sha256,
                )
            )
        ):
            raise G1TargetEvaluationError("target curve bank identity is invalid")


@dataclass(frozen=True, slots=True)
class G1OuterTargetCurveBuild:
    path: Path
    outer_target: str
    specimen_count: int
    record_count: int
    trajectory_bank_manifest_sha256: str
    trajectory_bank_seal_sha256: str
    fragment_manifest_sha256s: tuple[str, ...]
    bank: G1TargetCurveBankFile
    target_outcomes_opened: bool = True
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, Path)
            or type(self.outer_target) is not str
            or not self.outer_target
            or type(self.specimen_count) is not int
            or self.specimen_count <= 0
            or type(self.record_count) is not int
            or self.record_count < self.specimen_count
            or not _valid_sha256(self.trajectory_bank_manifest_sha256)
            or not _valid_sha256(self.trajectory_bank_seal_sha256)
            or type(self.fragment_manifest_sha256s) is not tuple
            or len(self.fragment_manifest_sha256s) != self.specimen_count
            or not all(
                _valid_sha256(value) for value in self.fragment_manifest_sha256s
            )
            or type(self.bank) is not G1TargetCurveBankFile
            or self.bank.outer_target != self.outer_target
            or self.bank.row_count != self.record_count
            or self.bank.trajectory_bank_seal_sha256
            != self.trajectory_bank_seal_sha256
            or self.target_outcomes_opened is not True
        ):
            raise G1TargetEvaluationError("outer target curve build is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-outer-target-curve-build",
                    "path": self.path.as_posix(),
                    "outer_target": self.outer_target,
                    "specimen_count": self.specimen_count,
                    "record_count": self.record_count,
                    "trajectory_bank_manifest": self.trajectory_bank_manifest_sha256,
                    "trajectory_bank_seal": self.trajectory_bank_seal_sha256,
                    "fragments": self.fragment_manifest_sha256s,
                    "bank": self.bank.manifest_sha256,
                    "target_outcomes_opened": True,
                }
            ),
        )


def _ordered_records(
    records: tuple[G1TargetTrajectoryRecord, ...],
) -> tuple[G1TargetTrajectoryRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1TargetTrajectoryRecord for row in records)
    ):
        raise G1TargetEvaluationError("target trajectory records are invalid")
    return tuple(
        sorted(
            records,
            key=lambda row: (row.specimen_id, row.task.value, row.variant.value),
        )
    )


def _curve_key(record: G1TargetCurveRecord) -> tuple[str, str, str]:
    return record.specimen_id, record.task.value, record.variant.value


def _ordered_curve_records(
    records: tuple[G1TargetCurveRecord, ...],
) -> tuple[G1TargetCurveRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1TargetCurveRecord for row in records)
        or len({_curve_key(row) for row in records}) != len(records)
        or len({row.state_sha256 for row in records}) != len(records)
        or len({row.outer_target for row in records}) != 1
        or len({row.bank_seal_sha256 for row in records}) != 1
    ):
        raise G1TargetEvaluationError("target curve bank roster is invalid")
    return tuple(sorted(records, key=_curve_key))


def _curve_records_sha(records: tuple[G1TargetCurveRecord, ...]) -> str:
    return _json_sha(tuple(row.state_sha256 for row in records))


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.manifest.json")


def target_curve_bank_path(work_root: str | Path, outer_target: str) -> Path:
    if (
        type(outer_target) is not str
        or not outer_target
        or "/" in outer_target
        or "\\" in outer_target
        or outer_target in {".", ".."}
    ):
        raise G1TargetEvaluationError("target curve path identity is invalid")
    return Path(work_root) / outer_target / "target_curves.parquet"


def target_curve_fragment_path(
    work_root: str | Path,
    outer_target: str,
    specimen_id: str,
) -> Path:
    if (
        type(specimen_id) is not str
        or not specimen_id
        or "/" in specimen_id
        or "\\" in specimen_id
        or specimen_id in {".", ".."}
    ):
        raise G1TargetEvaluationError("target curve fragment identity is invalid")
    return (
        target_curve_bank_path(work_root, outer_target).parent
        / "fragments"
        / f"{specimen_id}.parquet"
    )


def _evaluation_views(
    runtime: G1Runtime,
    records: tuple[G1TargetTrajectoryRecord, ...],
) -> dict[str, EvaluationView]:
    views = {}
    for record in records:
        if record.specimen_id in views:
            continue
        view = runtime.mavis.evaluation_view(record.specimen_id)
        if (
            type(view) is not EvaluationView
            or view.specimen_id != record.specimen_id
            or view.dataset_id != record.outer_target
        ):
            raise G1TargetEvaluationError("target evaluation view identity changed")
        views[record.specimen_id] = view
    return views


def _validate_evaluation_request(
    runtime: G1Runtime,
    bank: G1TargetTrajectoryBankFile,
    seal: G1TargetTrajectoryBankSeal,
    *,
    prior: SourceBackgroundPrior,
    assessor: object,
    encoder: object,
) -> None:
    if (
        type(runtime) is not G1Runtime
        or type(bank) is not G1TargetTrajectoryBankFile
        or type(seal) is not G1TargetTrajectoryBankSeal
        or type(prior) is not SourceBackgroundPrior
        or prior.outer_domain != bank.outer_target
        or not callable(getattr(assessor, "predict", None))
        or not _valid_sha256(getattr(assessor, "model_state_sha256", None))
        or not callable(getattr(encoder, "encode", None))
    ):
        raise G1TargetEvaluationError("target evaluation request is invalid")


def _evaluate_records_with_views(
    runtime: G1Runtime,
    records: tuple[G1TargetTrajectoryRecord, ...],
    views: dict[str, EvaluationView],
    seal: G1TargetTrajectoryBankSeal,
    *,
    prior: SourceBackgroundPrior,
    assessor: object,
    encoder: object,
) -> tuple[G1TargetCurveRecord, ...]:
    output = []
    for record in records:
        view = views.get(record.specimen_id)
        if view is None:
            raise G1TargetEvaluationError("target evaluation view is absent")
        vault = TargetTruthVault(
            target_domain=record.outer_target,
            specimen_sha256=record.specimen_sha256,
            full_scan=view.full_scan,
            true_cai=view.true_cai,
        )
        trajectory_seal = vault.seal_trajectory(record.trajectory)
        truth = vault.open_truth(trajectory_seal)
        world, grid, _surface = build_g1_world(
            runtime,
            dataset_id=record.outer_target,
            specimen_id=record.specimen_id,
            task=record.task,
            endpoint_budget=0.25,
        )
        curve = evaluate_g1_action_history(
            world,
            grid,
            prior,
            assessor=assessor,
            encoder=encoder,
            method=record.variant.value,
            target_domain=record.outer_target,
            specimen_sha256=record.specimen_sha256,
            actions=record.trajectory.action_history,
            full_scan=truth.full_scan,
            true_cai=truth.true_cai,
        )
        output.append(
            G1TargetCurveRecord(
                outer_target=record.outer_target,
                specimen_id=record.specimen_id,
                specimen_sha256=record.specimen_sha256,
                task=record.task,
                variant=record.variant,
                target_record_sha256=record.state_sha256,
                trajectory_sha256=record.trajectory.state_sha256,
                bank_seal_sha256=seal.state_sha256,
                trajectory_seal_sha256=trajectory_seal.state_sha256,
                truth_view_sha256=truth.state_sha256,
                curve=curve,
            )
        )
    return _ordered_curve_records(tuple(output))


def evaluate_g1_target_trajectory_bank(
    runtime: G1Runtime,
    bank: G1TargetTrajectoryBankFile,
    records: tuple[G1TargetTrajectoryRecord, ...],
    seal: G1TargetTrajectoryBankSeal,
    *,
    prior: SourceBackgroundPrior,
    assessor: object,
    encoder: object,
) -> tuple[G1TargetCurveRecord, ...]:
    _validate_evaluation_request(
        runtime,
        bank,
        seal,
        prior=prior,
        assessor=assessor,
        encoder=encoder,
    )
    ordered = _ordered_records(records)
    try:
        validate_g1_target_trajectory_bank_seal(bank, ordered, seal)
    except G1TargetExecutionError as error:
        raise G1TargetEvaluationError("target trajectory bank seal changed") from error

    # This is the first permitted hidden-target access in the formal state machine.
    views = _evaluation_views(runtime, ordered)
    return _evaluate_records_with_views(
        runtime,
        ordered,
        views,
        seal,
        prior=prior,
        assessor=assessor,
        encoder=encoder,
    )


def _curve_row(record: G1TargetCurveRecord) -> dict[str, object]:
    curve = record.curve
    return {
        "outer_target": record.outer_target,
        "specimen_id": record.specimen_id,
        "specimen_sha256": record.specimen_sha256,
        "task": record.task.value,
        "variant": record.variant.value,
        "target_record_sha256": record.target_record_sha256,
        "trajectory_sha256": record.trajectory_sha256,
        "bank_seal_sha256": record.bank_seal_sha256,
        "trajectory_seal_sha256": record.trajectory_seal_sha256,
        "truth_view_sha256": record.truth_view_sha256,
        "grid_sha256": curve.grid_sha256,
        "evaluator_sha256": curve.evaluator_sha256,
        "warm_start_sha256": curve.warm_start_sha256,
        "nominal_budgets_le_f64": np.ascontiguousarray(
            curve.nominal_budgets, dtype="<f8"
        ).tobytes(order="C"),
        "exact_budgets_le_f64": np.ascontiguousarray(
            curve.exact_budgets, dtype="<f8"
        ).tobytes(order="C"),
        "task_losses_le_f64": np.ascontiguousarray(
            curve.task_losses, dtype="<f8"
        ).tobytes(order="C"),
        "projected_state_sha256_json": json.dumps(
            curve.projected_state_sha256, separators=(",", ":")
        ),
        "auebc": curve.auebc,
        "curve_sha256": curve.state_sha256,
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


def write_g1_target_curve_bank(
    path: str | Path,
    records: tuple[G1TargetCurveRecord, ...],
) -> G1TargetCurveBankFile:
    destination = Path(path)
    ordered = _ordered_curve_records(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        pl.DataFrame(
            [_curve_row(record) for record in ordered], infer_schema_length=None
        ).write_parquet(
            temporary,
            compression="zstd",
            statistics=False,
            row_group_size=128,
        )
        parquet_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        records_sha = _curve_records_sha(ordered)
        manifest = {
            "schema_version": 1,
            "scope": "inspection_agent_g1_target_curve_bank",
            "row_count": len(ordered),
            "outer_target": ordered[0].outer_target,
            "trajectory_bank_seal_sha256": ordered[0].bank_seal_sha256,
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
    identity, replay = read_g1_target_curve_bank(destination)
    if tuple(row.state_sha256 for row in replay) != tuple(
        row.state_sha256 for row in ordered
    ):
        raise G1TargetEvaluationError("written target curve bank did not replay exactly")
    return identity


def _curve_from_row(row: dict[str, object]) -> G1TargetCurveRecord:
    try:
        projected = json.loads(str(row["projected_state_sha256_json"]))
        if not isinstance(projected, list):
            raise TypeError
        curve = replay_engineering_curve(
            method=str(row["variant"]),
            target_domain=str(row["outer_target"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            grid_sha256=str(row["grid_sha256"]),
            evaluator_sha256=str(row["evaluator_sha256"]),
            warm_start_sha256=str(row["warm_start_sha256"]),
            exact_budgets=np.frombuffer(
                bytes(row["exact_budgets_le_f64"]), dtype="<f8"
            ),
            task_losses=np.frombuffer(bytes(row["task_losses_le_f64"]), dtype="<f8"),
            projected_state_sha256=tuple(str(value) for value in projected),
            state_sha256=str(row["curve_sha256"]),
        )
        nominal = np.frombuffer(bytes(row["nominal_budgets_le_f64"]), dtype="<f8")
        if (
            nominal.tobytes(order="C") != curve.nominal_budgets.tobytes(order="C")
            or float(row["auebc"]) != curve.auebc
        ):
            raise G1TargetEvaluationError("target curve hash changed")
        record = G1TargetCurveRecord(
            outer_target=str(row["outer_target"]),
            specimen_id=str(row["specimen_id"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            variant=TargetPolicyVariant(str(row["variant"])),
            target_record_sha256=str(row["target_record_sha256"]),
            trajectory_sha256=str(row["trajectory_sha256"]),
            bank_seal_sha256=str(row["bank_seal_sha256"]),
            trajectory_seal_sha256=str(row["trajectory_seal_sha256"]),
            truth_view_sha256=str(row["truth_view_sha256"]),
            curve=curve,
        )
        if record.state_sha256 != str(row["record_sha256"]):
            raise G1TargetEvaluationError("target curve record hash changed")
        return record
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ) as error:
        if isinstance(error, G1TargetEvaluationError):
            raise
        raise G1TargetEvaluationError("target curve row cannot be reconstructed") from error


def read_g1_target_curve_bank(
    path: str | Path,
) -> tuple[G1TargetCurveBankFile, tuple[G1TargetCurveRecord, ...]]:
    source = Path(path)
    try:
        parquet_payload = source.read_bytes()
        manifest_payload = _manifest_path(source).read_bytes()
        manifest = json.loads(manifest_payload)
        rows = pl.read_parquet(source).to_dicts()
    except (OSError, UnicodeError, json.JSONDecodeError, pl.exceptions.PolarsError) as error:
        raise G1TargetEvaluationError("target curve bank cannot be read") from error
    expected_keys = {
        "schema_version",
        "scope",
        "row_count",
        "outer_target",
        "trajectory_bank_seal_sha256",
        "parquet_sha256",
        "records_sha256",
    }
    parquet_sha = hashlib.sha256(parquet_payload).hexdigest()
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("scope") != "inspection_agent_g1_target_curve_bank"
        or manifest.get("parquet_sha256") != parquet_sha
    ):
        raise G1TargetEvaluationError("target curve bank manifest changed")
    records = tuple(_curve_from_row(row) for row in rows)
    ordered = _ordered_curve_records(records)
    records_sha = _curve_records_sha(ordered)
    first = ordered[0]
    if (
        records != ordered
        or manifest.get("row_count") != len(records)
        or manifest.get("outer_target") != first.outer_target
        or manifest.get("trajectory_bank_seal_sha256") != first.bank_seal_sha256
        or manifest.get("records_sha256") != records_sha
    ):
        raise G1TargetEvaluationError("target curve bank evidence changed")
    return (
        G1TargetCurveBankFile(
            row_count=len(records),
            outer_target=first.outer_target,
            trajectory_bank_seal_sha256=first.bank_seal_sha256,
            parquet_sha256=parquet_sha,
            records_sha256=records_sha,
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        ),
        records,
    )


def _validate_curve_roster(
    curves: tuple[G1TargetCurveRecord, ...],
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
    seal: G1TargetTrajectoryBankSeal,
) -> None:
    by_key = {_curve_key(row): row for row in trajectories}
    if (
        {_curve_key(row) for row in curves} != set(by_key)
        or any(
            row.target_record_sha256 != by_key[_curve_key(row)].state_sha256
            or row.trajectory_sha256
            != by_key[_curve_key(row)].trajectory.state_sha256
            or row.bank_seal_sha256 != seal.state_sha256
            for row in curves
        )
    ):
        raise G1TargetEvaluationError("target curve roster changed")


def build_g1_outer_target_curve_bank(
    runtime: G1Runtime,
    trajectory_bank: G1TargetTrajectoryBankFile,
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
    seal: G1TargetTrajectoryBankSeal,
    *,
    prior: SourceBackgroundPrior,
    assessor: object,
    encoder: object,
    work_root: str | Path,
    progress: object | None = None,
) -> G1OuterTargetCurveBuild:
    _validate_evaluation_request(
        runtime,
        trajectory_bank,
        seal,
        prior=prior,
        assessor=assessor,
        encoder=encoder,
    )
    if progress is not None and not callable(progress):
        raise G1TargetEvaluationError("target curve progress callback is invalid")
    ordered = _ordered_records(trajectories)
    try:
        validate_g1_target_trajectory_bank_seal(trajectory_bank, ordered, seal)
    except G1TargetExecutionError as error:
        raise G1TargetEvaluationError("target trajectory bank seal changed") from error
    outer_target = ordered[0].outer_target
    specimen_ids = tuple(sorted({row.specimen_id for row in ordered}))
    combined = []
    fragment_manifests = []
    for specimen_id in specimen_ids:
        specimen_rows = tuple(row for row in ordered if row.specimen_id == specimen_id)
        path = target_curve_fragment_path(work_root, outer_target, specimen_id)
        present = path.exists(), _manifest_path(path).exists()
        if present == (True, True):
            fragment_bank, curves = read_g1_target_curve_bank(path)
            if progress is not None:
                progress(f"G1 target curve fragment replay {outer_target}/{specimen_id}")
        elif present == (False, False):
            views = _evaluation_views(runtime, specimen_rows)
            curves = _evaluate_records_with_views(
                runtime,
                specimen_rows,
                views,
                seal,
                prior=prior,
                assessor=assessor,
                encoder=encoder,
            )
            fragment_bank = write_g1_target_curve_bank(path, curves)
            if progress is not None:
                progress(f"G1 target curves {outer_target}/{specimen_id}: {len(curves)}")
        else:
            raise G1TargetEvaluationError("target curve fragment is incomplete")
        _validate_curve_roster(curves, specimen_rows, seal)
        combined.extend(curves)
        fragment_manifests.append(fragment_bank.manifest_sha256)
    curves = _ordered_curve_records(tuple(combined))
    _validate_curve_roster(curves, ordered, seal)
    destination = target_curve_bank_path(work_root, outer_target)
    present = destination.exists(), _manifest_path(destination).exists()
    if present == (True, True):
        bank, replay = read_g1_target_curve_bank(destination)
        if tuple(row.state_sha256 for row in replay) != tuple(
            row.state_sha256 for row in curves
        ):
            raise G1TargetEvaluationError("outer target curve bank changed")
    elif present == (False, False):
        bank = write_g1_target_curve_bank(destination, curves)
    else:
        raise G1TargetEvaluationError("outer target curve bank is incomplete")
    result = G1OuterTargetCurveBuild(
        path=destination,
        outer_target=outer_target,
        specimen_count=len(specimen_ids),
        record_count=len(curves),
        trajectory_bank_manifest_sha256=trajectory_bank.manifest_sha256,
        trajectory_bank_seal_sha256=seal.state_sha256,
        fragment_manifest_sha256s=tuple(fragment_manifests),
        bank=bank,
    )
    if progress is not None:
        progress(f"G1 target curve bank complete {outer_target}: {len(curves)} records")
    return result


__all__ = [
    "G1OuterTargetCurveBuild",
    "G1TargetCurveBankFile",
    "G1TargetCurveRecord",
    "G1TargetEvaluationError",
    "build_g1_outer_target_curve_bank",
    "evaluate_g1_target_trajectory_bank",
    "read_g1_target_curve_bank",
    "target_curve_bank_path",
    "target_curve_fragment_path",
    "write_g1_target_curve_bank",
]
