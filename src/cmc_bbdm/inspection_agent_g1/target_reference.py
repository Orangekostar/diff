"""Evaluation-only fixed and privileged-oracle curves for frozen G1 targets."""

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
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.mavis.contracts import EvaluationView

from .formal import (
    FIXED_BASELINE_METHODS,
    evaluate_g1_action_history,
    plan_g1_fixed_actions,
    run_g1_warm_started_oracle_actions,
)
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
    validate_g1_target_trajectory_bank_seal,
)

TARGET_ORACLE_METHODS = {
    InspectionTask.FIELD: "ORACLE_FIELD",
    InspectionTask.CAI: "ORACLE_CAI",
}


class G1TargetReferenceError(ValueError):
    """Raised when target reference evaluation violates the frozen protocol."""


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


def _action_history_sha(actions: tuple[InspectionCellAction, ...]) -> str:
    return _json_sha(
        tuple(
            (action.cell_index, action.from_level, action.to_level)
            for action in actions
        )
    )


@dataclass(frozen=True, slots=True)
class G1TargetReferenceCurveRecord:
    outer_target: str
    specimen_id: str
    specimen_sha256: str
    task: InspectionTask
    method: str
    final_dependency_sha256: str
    prior_sha256: str
    assessor_sha256: str
    bank_seal_sha256: str
    authorization_trajectory_sha256: str
    truth_view_sha256: str
    action_history_sha256: str
    curve: EngineeringCurve
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            validate_engineering_curve(self.curve)
        except ValueError as error:
            raise G1TargetReferenceError("target reference curve changed") from error
        allowed = {*FIXED_BASELINE_METHODS, TARGET_ORACLE_METHODS.get(self.task)}
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.specimen_id) is not str
            or not self.specimen_id
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or self.method not in allowed
            or not all(
                _valid_sha256(value)
                for value in (
                    self.final_dependency_sha256,
                    self.prior_sha256,
                    self.assessor_sha256,
                    self.bank_seal_sha256,
                    self.authorization_trajectory_sha256,
                    self.truth_view_sha256,
                    self.action_history_sha256,
                )
            )
            or self.curve.method != self.method
            or self.curve.target_domain != self.outer_target
            or self.curve.specimen_sha256 != self.specimen_sha256
            or self.curve.task is not self.task
        ):
            raise G1TargetReferenceError("target reference record is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-target-reference-curve-record",
                    "outer_target": self.outer_target,
                    "specimen_id": self.specimen_id,
                    "specimen_sha256": self.specimen_sha256,
                    "task": self.task.value,
                    "method": self.method,
                    "final_dependency": self.final_dependency_sha256,
                    "prior": self.prior_sha256,
                    "assessor": self.assessor_sha256,
                    "bank_seal": self.bank_seal_sha256,
                    "authorization_trajectory": (
                        self.authorization_trajectory_sha256
                    ),
                    "truth_view": self.truth_view_sha256,
                    "action_history": self.action_history_sha256,
                    "curve": self.curve.state_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1TargetReferenceBankFile:
    row_count: int
    outer_target: str
    trajectory_bank_seal_sha256: str
    final_dependency_sha256: str
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
                    self.final_dependency_sha256,
                    self.parquet_sha256,
                    self.records_sha256,
                    self.manifest_sha256,
                )
            )
        ):
            raise G1TargetReferenceError("target reference bank identity is invalid")


@dataclass(frozen=True, slots=True)
class G1OuterTargetReferenceBuild:
    path: Path
    outer_target: str
    specimen_count: int
    record_count: int
    trajectory_bank_manifest_sha256: str
    trajectory_bank_seal_sha256: str
    fragment_manifest_sha256s: tuple[str, ...]
    bank: G1TargetReferenceBankFile
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
            or self.record_count != 12 * self.specimen_count
            or not _valid_sha256(self.trajectory_bank_manifest_sha256)
            or not _valid_sha256(self.trajectory_bank_seal_sha256)
            or type(self.fragment_manifest_sha256s) is not tuple
            or len(self.fragment_manifest_sha256s) != self.specimen_count
            or not all(
                _valid_sha256(value) for value in self.fragment_manifest_sha256s
            )
            or type(self.bank) is not G1TargetReferenceBankFile
            or self.bank.outer_target != self.outer_target
            or self.bank.row_count != self.record_count
            or self.bank.trajectory_bank_seal_sha256
            != self.trajectory_bank_seal_sha256
            or self.target_outcomes_opened is not True
        ):
            raise G1TargetReferenceError("outer target reference build is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-outer-target-reference-build",
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


def _record_key(
    record: G1TargetReferenceCurveRecord,
) -> tuple[str, str, str]:
    return record.specimen_id, record.task.value, record.method


def _ordered_trajectories(
    records: tuple[G1TargetTrajectoryRecord, ...],
) -> tuple[G1TargetTrajectoryRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1TargetTrajectoryRecord for row in records)
        or len({row.outer_target for row in records}) != 1
        or len({row.final_dependency_sha256 for row in records}) != 1
    ):
        raise G1TargetReferenceError("target reference trajectory roster is invalid")
    return tuple(
        sorted(
            records,
            key=lambda row: (row.specimen_id, row.task.value, row.variant.value),
        )
    )


def _ordered_reference_records(
    records: tuple[G1TargetReferenceCurveRecord, ...],
) -> tuple[G1TargetReferenceCurveRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1TargetReferenceCurveRecord for row in records)
        or len({_record_key(row) for row in records}) != len(records)
        or len({row.state_sha256 for row in records}) != len(records)
        or len({row.outer_target for row in records}) != 1
        or len({row.bank_seal_sha256 for row in records}) != 1
        or len({row.final_dependency_sha256 for row in records}) != 1
        or len({row.prior_sha256 for row in records}) != 1
        or len({row.assessor_sha256 for row in records}) != 1
    ):
        raise G1TargetReferenceError("target reference bank roster is invalid")
    return tuple(sorted(records, key=_record_key))


def _records_sha(records: tuple[G1TargetReferenceCurveRecord, ...]) -> str:
    return _json_sha(tuple(row.state_sha256 for row in records))


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.manifest.json")


def target_reference_bank_path(work_root: str | Path, outer_target: str) -> Path:
    if (
        type(outer_target) is not str
        or not outer_target
        or "/" in outer_target
        or "\\" in outer_target
        or outer_target in {".", ".."}
    ):
        raise G1TargetReferenceError("target reference path identity is invalid")
    return Path(work_root) / outer_target / "target_references.parquet"


def target_reference_fragment_path(
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
        raise G1TargetReferenceError("target reference fragment identity is invalid")
    return (
        target_reference_bank_path(work_root, outer_target).parent
        / "fragments"
        / f"{specimen_id}.parquet"
    )


def _validate_reference_request(
    runtime: G1Runtime,
    bank: G1TargetTrajectoryBankFile,
    seal: G1TargetTrajectoryBankSeal,
    *,
    prior: SourceBackgroundPrior,
    assessor: object,
    encoder: object,
    random_seed: int,
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
        or type(random_seed) is not int
    ):
        raise G1TargetReferenceError("target reference request is invalid")


def _evaluate_reference_specimens(
    runtime: G1Runtime,
    bank: G1TargetTrajectoryBankFile,
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
    seal: G1TargetTrajectoryBankSeal,
    *,
    prior: SourceBackgroundPrior,
    assessor: object,
    encoder: object,
    random_seed: int,
) -> tuple[G1TargetReferenceCurveRecord, ...]:
    output = []
    specimen_ids = tuple(sorted({row.specimen_id for row in trajectories}))
    for specimen_id in specimen_ids:
        specimen_rows = tuple(
            row for row in trajectories if row.specimen_id == specimen_id
        )
        authorizer = specimen_rows[0]
        view = runtime.mavis.evaluation_view(specimen_id)
        if (
            type(view) is not EvaluationView
            or view.specimen_id != specimen_id
            or view.dataset_id != bank.outer_target
        ):
            raise G1TargetReferenceError("target reference view identity changed")
        vault = TargetTruthVault(
            target_domain=bank.outer_target,
            specimen_sha256=authorizer.specimen_sha256,
            full_scan=view.full_scan,
            true_cai=view.true_cai,
        )
        trajectory_seal = vault.seal_trajectory(authorizer.trajectory)
        truth = vault.open_truth(trajectory_seal)
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            world, grid, surface = build_g1_world(
                runtime,
                dataset_id=bank.outer_target,
                specimen_id=specimen_id,
                task=task,
                endpoint_budget=0.25,
            )
            initial = world.reset()
            methods = []
            for method in FIXED_BASELINE_METHODS:
                actions = plan_g1_fixed_actions(
                    grid,
                    surface.hypothesis,
                    surface_sha256=initial.surface_sha256,
                    specimen_sha256=authorizer.specimen_sha256,
                    method=method,
                    random_seed=random_seed,
                    endpoint_budget=initial.endpoint_budget,
                )
                methods.append((method, actions))
            oracle_actions = run_g1_warm_started_oracle_actions(
                world,
                grid,
                prior,
                surface_hypothesis=surface.hypothesis,
                full_scan=truth.full_scan,
                true_cai=truth.true_cai,
                assessor=assessor,
                encoder=encoder,
            )
            methods.append((TARGET_ORACLE_METHODS[task], oracle_actions))
            for method, actions in methods:
                curve = evaluate_g1_action_history(
                    world,
                    grid,
                    prior,
                    assessor=assessor,
                    encoder=encoder,
                    method=method,
                    target_domain=bank.outer_target,
                    specimen_sha256=authorizer.specimen_sha256,
                    actions=actions,
                    full_scan=truth.full_scan,
                    true_cai=truth.true_cai,
                )
                output.append(
                    G1TargetReferenceCurveRecord(
                        outer_target=bank.outer_target,
                        specimen_id=specimen_id,
                        specimen_sha256=authorizer.specimen_sha256,
                        task=task,
                        method=method,
                        final_dependency_sha256=(
                            authorizer.final_dependency_sha256
                        ),
                        prior_sha256=prior.state_sha256,
                        assessor_sha256=assessor.model_state_sha256,
                        bank_seal_sha256=seal.state_sha256,
                        authorization_trajectory_sha256=(
                            authorizer.trajectory.state_sha256
                        ),
                        truth_view_sha256=truth.state_sha256,
                        action_history_sha256=_action_history_sha(actions),
                        curve=curve,
                    )
                )
    records = tuple(sorted(output, key=_record_key))
    if len(records) != 12 * len(specimen_ids):
        raise G1TargetReferenceError("target reference record count changed")
    return records


def evaluate_g1_target_reference_curves(
    runtime: G1Runtime,
    bank: G1TargetTrajectoryBankFile,
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
    seal: G1TargetTrajectoryBankSeal,
    *,
    prior: SourceBackgroundPrior,
    assessor: object,
    encoder: object,
    random_seed: int,
) -> tuple[G1TargetReferenceCurveRecord, ...]:
    _validate_reference_request(
        runtime,
        bank,
        seal,
        prior=prior,
        assessor=assessor,
        encoder=encoder,
        random_seed=random_seed,
    )
    ordered = _ordered_trajectories(trajectories)
    try:
        validate_g1_target_trajectory_bank_seal(bank, ordered, seal)
    except G1TargetExecutionError as error:
        raise G1TargetReferenceError("target trajectory bank seal changed") from error
    return _evaluate_reference_specimens(
        runtime,
        bank,
        ordered,
        seal,
        prior=prior,
        assessor=assessor,
        encoder=encoder,
        random_seed=random_seed,
    )


def _record_row(record: G1TargetReferenceCurveRecord) -> dict[str, object]:
    curve = record.curve
    return {
        "outer_target": record.outer_target,
        "specimen_id": record.specimen_id,
        "specimen_sha256": record.specimen_sha256,
        "task": record.task.value,
        "method": record.method,
        "final_dependency_sha256": record.final_dependency_sha256,
        "prior_sha256": record.prior_sha256,
        "assessor_sha256": record.assessor_sha256,
        "bank_seal_sha256": record.bank_seal_sha256,
        "authorization_trajectory_sha256": (
            record.authorization_trajectory_sha256
        ),
        "truth_view_sha256": record.truth_view_sha256,
        "action_history_sha256": record.action_history_sha256,
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


def write_g1_target_reference_bank(
    path: str | Path,
    records: tuple[G1TargetReferenceCurveRecord, ...],
) -> G1TargetReferenceBankFile:
    destination = Path(path)
    ordered = _ordered_reference_records(records)
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
        first = ordered[0]
        manifest = {
            "schema_version": 1,
            "scope": "inspection_agent_g1_target_reference_bank",
            "row_count": len(ordered),
            "outer_target": first.outer_target,
            "trajectory_bank_seal_sha256": first.bank_seal_sha256,
            "final_dependency_sha256": first.final_dependency_sha256,
            "prior_sha256": first.prior_sha256,
            "assessor_sha256": first.assessor_sha256,
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
    identity, replay = read_g1_target_reference_bank(destination)
    if tuple(row.state_sha256 for row in replay) != tuple(
        row.state_sha256 for row in ordered
    ):
        raise G1TargetReferenceError(
            "written target reference bank did not replay exactly"
        )
    return identity


def _record_from_row(row: dict[str, object]) -> G1TargetReferenceCurveRecord:
    try:
        projected = json.loads(str(row["projected_state_sha256_json"]))
        if not isinstance(projected, list):
            raise TypeError
        curve = replay_engineering_curve(
            method=str(row["method"]),
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
            raise G1TargetReferenceError("target reference curve hash changed")
        record = G1TargetReferenceCurveRecord(
            outer_target=str(row["outer_target"]),
            specimen_id=str(row["specimen_id"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            method=str(row["method"]),
            final_dependency_sha256=str(row["final_dependency_sha256"]),
            prior_sha256=str(row["prior_sha256"]),
            assessor_sha256=str(row["assessor_sha256"]),
            bank_seal_sha256=str(row["bank_seal_sha256"]),
            authorization_trajectory_sha256=str(
                row["authorization_trajectory_sha256"]
            ),
            truth_view_sha256=str(row["truth_view_sha256"]),
            action_history_sha256=str(row["action_history_sha256"]),
            curve=curve,
        )
        if record.state_sha256 != str(row["record_sha256"]):
            raise G1TargetReferenceError("target reference record hash changed")
        return record
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ) as error:
        if isinstance(error, G1TargetReferenceError):
            raise
        raise G1TargetReferenceError(
            "target reference row cannot be reconstructed"
        ) from error


def read_g1_target_reference_bank(
    path: str | Path,
) -> tuple[G1TargetReferenceBankFile, tuple[G1TargetReferenceCurveRecord, ...]]:
    source = Path(path)
    try:
        parquet_payload = source.read_bytes()
        manifest_payload = _manifest_path(source).read_bytes()
        manifest = json.loads(manifest_payload)
        rows = pl.read_parquet(source).to_dicts()
    except (OSError, UnicodeError, json.JSONDecodeError, pl.exceptions.PolarsError) as error:
        raise G1TargetReferenceError("target reference bank cannot be read") from error
    expected_keys = {
        "schema_version",
        "scope",
        "row_count",
        "outer_target",
        "trajectory_bank_seal_sha256",
        "final_dependency_sha256",
        "prior_sha256",
        "assessor_sha256",
        "parquet_sha256",
        "records_sha256",
    }
    parquet_sha = hashlib.sha256(parquet_payload).hexdigest()
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("scope") != "inspection_agent_g1_target_reference_bank"
        or manifest.get("parquet_sha256") != parquet_sha
    ):
        raise G1TargetReferenceError("target reference bank manifest changed")
    records = tuple(_record_from_row(row) for row in rows)
    ordered = _ordered_reference_records(records)
    records_sha = _records_sha(ordered)
    first = ordered[0]
    if (
        records != ordered
        or manifest.get("row_count") != len(records)
        or manifest.get("outer_target") != first.outer_target
        or manifest.get("trajectory_bank_seal_sha256") != first.bank_seal_sha256
        or manifest.get("final_dependency_sha256")
        != first.final_dependency_sha256
        or manifest.get("prior_sha256") != first.prior_sha256
        or manifest.get("assessor_sha256") != first.assessor_sha256
        or manifest.get("records_sha256") != records_sha
    ):
        raise G1TargetReferenceError("target reference bank evidence changed")
    return (
        G1TargetReferenceBankFile(
            row_count=len(records),
            outer_target=first.outer_target,
            trajectory_bank_seal_sha256=first.bank_seal_sha256,
            final_dependency_sha256=first.final_dependency_sha256,
            parquet_sha256=parquet_sha,
            records_sha256=records_sha,
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        ),
        records,
    )


def _validate_reference_roster(
    records: tuple[G1TargetReferenceCurveRecord, ...],
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
    seal: G1TargetTrajectoryBankSeal,
    *,
    prior: SourceBackgroundPrior,
    assessor: object,
) -> None:
    specimen_rows = {
        specimen_id: tuple(row for row in trajectories if row.specimen_id == specimen_id)
        for specimen_id in {row.specimen_id for row in trajectories}
    }
    expected = {
        (specimen_id, task.value, method)
        for specimen_id in specimen_rows
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
        for method in (*FIXED_BASELINE_METHODS, TARGET_ORACLE_METHODS[task])
    }
    if (
        {_record_key(row) for row in records} != expected
        or any(
            row.specimen_sha256 != specimen_rows[row.specimen_id][0].specimen_sha256
            or row.final_dependency_sha256
            != specimen_rows[row.specimen_id][0].final_dependency_sha256
            or row.authorization_trajectory_sha256
            not in {
                trajectory.trajectory.state_sha256
                for trajectory in specimen_rows[row.specimen_id]
            }
            or row.bank_seal_sha256 != seal.state_sha256
            or row.prior_sha256 != prior.state_sha256
            or row.assessor_sha256 != assessor.model_state_sha256
            for row in records
        )
    ):
        raise G1TargetReferenceError("target reference roster changed")


def build_g1_outer_target_reference_bank(
    runtime: G1Runtime,
    trajectory_bank: G1TargetTrajectoryBankFile,
    trajectories: tuple[G1TargetTrajectoryRecord, ...],
    seal: G1TargetTrajectoryBankSeal,
    *,
    prior: SourceBackgroundPrior,
    assessor: object,
    encoder: object,
    random_seed: int,
    work_root: str | Path,
    progress: object | None = None,
) -> G1OuterTargetReferenceBuild:
    _validate_reference_request(
        runtime,
        trajectory_bank,
        seal,
        prior=prior,
        assessor=assessor,
        encoder=encoder,
        random_seed=random_seed,
    )
    if progress is not None and not callable(progress):
        raise G1TargetReferenceError("target reference progress callback is invalid")
    ordered = _ordered_trajectories(trajectories)
    try:
        validate_g1_target_trajectory_bank_seal(trajectory_bank, ordered, seal)
    except G1TargetExecutionError as error:
        raise G1TargetReferenceError("target trajectory bank seal changed") from error
    outer_target = ordered[0].outer_target
    specimen_ids = tuple(sorted({row.specimen_id for row in ordered}))
    combined = []
    fragment_manifests = []
    for specimen_id in specimen_ids:
        specimen_rows = tuple(row for row in ordered if row.specimen_id == specimen_id)
        path = target_reference_fragment_path(work_root, outer_target, specimen_id)
        present = path.exists(), _manifest_path(path).exists()
        if present == (True, True):
            fragment_bank, records = read_g1_target_reference_bank(path)
            if progress is not None:
                progress(f"G1 target reference fragment replay {outer_target}/{specimen_id}")
        elif present == (False, False):
            records = _evaluate_reference_specimens(
                runtime,
                trajectory_bank,
                specimen_rows,
                seal,
                prior=prior,
                assessor=assessor,
                encoder=encoder,
                random_seed=random_seed,
            )
            fragment_bank = write_g1_target_reference_bank(path, records)
            if progress is not None:
                progress(f"G1 target references {outer_target}/{specimen_id}: 12")
        else:
            raise G1TargetReferenceError("target reference fragment is incomplete")
        _validate_reference_roster(
            records,
            specimen_rows,
            seal,
            prior=prior,
            assessor=assessor,
        )
        combined.extend(records)
        fragment_manifests.append(fragment_bank.manifest_sha256)
    records = _ordered_reference_records(tuple(combined))
    _validate_reference_roster(
        records,
        ordered,
        seal,
        prior=prior,
        assessor=assessor,
    )
    destination = target_reference_bank_path(work_root, outer_target)
    present = destination.exists(), _manifest_path(destination).exists()
    if present == (True, True):
        bank, replay = read_g1_target_reference_bank(destination)
        if tuple(row.state_sha256 for row in replay) != tuple(
            row.state_sha256 for row in records
        ):
            raise G1TargetReferenceError("outer target reference bank changed")
    elif present == (False, False):
        bank = write_g1_target_reference_bank(destination, records)
    else:
        raise G1TargetReferenceError("outer target reference bank is incomplete")
    result = G1OuterTargetReferenceBuild(
        path=destination,
        outer_target=outer_target,
        specimen_count=len(specimen_ids),
        record_count=len(records),
        trajectory_bank_manifest_sha256=trajectory_bank.manifest_sha256,
        trajectory_bank_seal_sha256=seal.state_sha256,
        fragment_manifest_sha256s=tuple(fragment_manifests),
        bank=bank,
    )
    if progress is not None:
        progress(f"G1 target reference bank complete {outer_target}: {len(records)}")
    return result


__all__ = [
    "TARGET_ORACLE_METHODS",
    "G1OuterTargetReferenceBuild",
    "G1TargetReferenceBankFile",
    "G1TargetReferenceCurveRecord",
    "G1TargetReferenceError",
    "build_g1_outer_target_reference_bank",
    "evaluate_g1_target_reference_curves",
    "read_g1_target_reference_bank",
    "target_reference_bank_path",
    "target_reference_fragment_path",
    "write_g1_target_reference_bank",
]
