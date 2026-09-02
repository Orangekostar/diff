"""Fold-safe fixed endpoints and source STOP-reference execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.cai_assessor import state_scalars
from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.field_task import field_loss
from cmc_bbdm.inspection_agent.generalized_reconstruction import reconstruct_observation
from cmc_bbdm.inspection_agent.stopping import ReferenceEndpoint

from .formal import FIXED_BASELINE_METHODS, plan_g1_fixed_actions
from .g1 import (
    G1ExecutionError,
    G1Protocol,
    G1Runtime,
    G1SourceDependencies,
    build_g1_source_dependencies,
    build_g1_world,
)
from .stopping_policy import SourceFixedReference, select_source_fixed_reference
from .teacher import SourceTeacherAuthorization


class G1StopExecutionError(ValueError):
    """Raised when source STOP execution leaves its fold-safe authority."""


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
class G1FixedEndpointRecord:
    outer_target: str
    source_domain: str
    specimen_id: str
    specimen_sha256: str
    task: InspectionTask
    method: str
    fit_domains: tuple[str, ...]
    dependency_sha256: str
    action_history_sha256: str
    endpoint_observation_sha256: str
    effective_budget: float
    exact_acquired_count: int
    native_count: int
    task_loss: float
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        budget = float(self.effective_budget)
        loss = float(self.task_loss)
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or self.source_domain == self.outer_target
            or type(self.specimen_id) is not str
            or not self.specimen_id
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or self.method not in FIXED_BASELINE_METHODS
            or type(self.fit_domains) is not tuple
            or len(self.fit_domains) != 4
            or self.fit_domains != tuple(sorted(set(self.fit_domains)))
            or {self.outer_target, self.source_domain} & set(self.fit_domains)
            or not _valid_sha256(self.dependency_sha256)
            or not _valid_sha256(self.action_history_sha256)
            or not _valid_sha256(self.endpoint_observation_sha256)
            or not math.isfinite(budget)
            or not 0.0 < budget <= 0.25
            or type(self.exact_acquired_count) is not int
            or self.exact_acquired_count <= 0
            or type(self.native_count) is not int
            or self.native_count < self.exact_acquired_count
            or budget != self.exact_acquired_count / self.native_count
            or not math.isfinite(loss)
            or loss < 0.0
        ):
            raise G1StopExecutionError("fixed endpoint record is invalid")
        object.__setattr__(self, "effective_budget", budget)
        object.__setattr__(self, "task_loss", loss)
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-fixed-endpoint-record",
                    "outer_target": self.outer_target,
                    "source_domain": self.source_domain,
                    "specimen_id": self.specimen_id,
                    "specimen_sha256": self.specimen_sha256,
                    "task": self.task.value,
                    "method": self.method,
                    "fit_domains": self.fit_domains,
                    "dependency_sha256": self.dependency_sha256,
                    "action_history_sha256": self.action_history_sha256,
                    "endpoint_observation_sha256": self.endpoint_observation_sha256,
                    "effective_budget": budget,
                    "exact_acquired_count": self.exact_acquired_count,
                    "native_count": self.native_count,
                    "task_loss": loss,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1FixedEndpointBankFile:
    row_count: int
    parquet_sha256: str
    records_sha256: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.row_count) is not int
            or self.row_count <= 0
            or not all(
                _valid_sha256(value)
                for value in (
                    self.parquet_sha256,
                    self.records_sha256,
                    self.manifest_sha256,
                )
            )
        ):
            raise G1StopExecutionError("fixed endpoint bank identity is invalid")


@dataclass(frozen=True, slots=True)
class G1FixedEndpointBuild:
    path: Path
    outer_target: str
    source_domain: str
    specimen_count: int
    dependency_sha256: str
    bank: G1FixedEndpointBankFile


def _ordered_records(
    records: tuple[G1FixedEndpointRecord, ...],
) -> tuple[G1FixedEndpointRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1FixedEndpointRecord for row in records)
        or len({row.state_sha256 for row in records}) != len(records)
        or len({row.outer_target for row in records}) != 1
        or len({row.source_domain for row in records}) != 1
        or len({row.fit_domains for row in records}) != 1
    ):
        raise G1StopExecutionError("fixed endpoint bank roster is invalid")
    keys = tuple(
        (row.specimen_sha256, row.task, row.method) for row in records
    )
    if len(set(keys)) != len(keys):
        raise G1StopExecutionError("fixed endpoint bank key is duplicated")
    return tuple(
        sorted(
            records,
            key=lambda row: (
                row.specimen_sha256,
                row.task.value,
                FIXED_BASELINE_METHODS.index(row.method),
            ),
        )
    )


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.manifest.json")


def _records_sha(records: tuple[G1FixedEndpointRecord, ...]) -> str:
    return _json_sha(tuple(row.state_sha256 for row in records))


def _row(record: G1FixedEndpointRecord) -> dict[str, object]:
    return {
        "outer_target": record.outer_target,
        "source_domain": record.source_domain,
        "specimen_id": record.specimen_id,
        "specimen_sha256": record.specimen_sha256,
        "task": record.task.value,
        "method": record.method,
        "fit_domains_json": json.dumps(record.fit_domains, separators=(",", ":")),
        "dependency_sha256": record.dependency_sha256,
        "action_history_sha256": record.action_history_sha256,
        "endpoint_observation_sha256": record.endpoint_observation_sha256,
        "effective_budget": record.effective_budget,
        "exact_acquired_count": record.exact_acquired_count,
        "native_count": record.native_count,
        "task_loss": record.task_loss,
        "record_sha256": record.state_sha256,
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


def write_fixed_endpoint_bank(
    path: str | Path,
    records: tuple[G1FixedEndpointRecord, ...],
) -> G1FixedEndpointBankFile:
    destination = Path(path)
    ordered = _ordered_records(records)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        pl.DataFrame([_row(row) for row in ordered]).write_parquet(
            temporary,
            compression="zstd",
            statistics=False,
            row_group_size=512,
        )
        parquet_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        records_sha = _records_sha(ordered)
        manifest = {
            "schema_version": 1,
            "scope": "inspection_agent_g1_source_fixed_endpoints",
            "row_count": len(ordered),
            "outer_target": ordered[0].outer_target,
            "source_domain": ordered[0].source_domain,
            "fit_domains": list(ordered[0].fit_domains),
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
        _atomic_bytes(_manifest_path(destination), manifest_payload)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    identity, loaded = read_fixed_endpoint_bank(destination)
    if loaded != ordered:
        raise G1StopExecutionError("written fixed endpoint bank did not replay exactly")
    return identity


def _record_from_row(row: dict[str, object]) -> G1FixedEndpointRecord:
    try:
        fit_raw = json.loads(str(row["fit_domains_json"]))
        if not isinstance(fit_raw, list):
            raise TypeError
        result = G1FixedEndpointRecord(
            outer_target=str(row["outer_target"]),
            source_domain=str(row["source_domain"]),
            specimen_id=str(row["specimen_id"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            method=str(row["method"]),
            fit_domains=tuple(str(value) for value in fit_raw),
            dependency_sha256=str(row["dependency_sha256"]),
            action_history_sha256=str(row["action_history_sha256"]),
            endpoint_observation_sha256=str(row["endpoint_observation_sha256"]),
            effective_budget=float(row["effective_budget"]),
            exact_acquired_count=int(row["exact_acquired_count"]),
            native_count=int(row["native_count"]),
            task_loss=float(row["task_loss"]),
        )
        if result.state_sha256 != str(row["record_sha256"]):
            raise G1StopExecutionError("fixed endpoint record hash changed")
        return result
    except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
        if isinstance(error, G1StopExecutionError):
            raise
        raise G1StopExecutionError("fixed endpoint row cannot be reconstructed") from error


def read_fixed_endpoint_bank(
    path: str | Path,
) -> tuple[G1FixedEndpointBankFile, tuple[G1FixedEndpointRecord, ...]]:
    source = Path(path)
    try:
        parquet_payload = source.read_bytes()
        manifest_payload = _manifest_path(source).read_bytes()
        manifest = json.loads(manifest_payload)
        rows = pl.read_parquet(source).to_dicts()
    except (OSError, UnicodeError, json.JSONDecodeError, pl.exceptions.PolarsError) as error:
        raise G1StopExecutionError("fixed endpoint bank cannot be read") from error
    parquet_sha = hashlib.sha256(parquet_payload).hexdigest()
    expected_keys = {
        "schema_version",
        "scope",
        "row_count",
        "outer_target",
        "source_domain",
        "fit_domains",
        "parquet_sha256",
        "records_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("scope") != "inspection_agent_g1_source_fixed_endpoints"
        or manifest.get("parquet_sha256") != parquet_sha
    ):
        raise G1StopExecutionError("fixed endpoint bank SHA-256 mismatch")
    records = tuple(_record_from_row(row) for row in rows)
    ordered = _ordered_records(records)
    records_sha = _records_sha(ordered)
    if (
        records != ordered
        or manifest.get("row_count") != len(records)
        or manifest.get("outer_target") != records[0].outer_target
        or manifest.get("source_domain") != records[0].source_domain
        or manifest.get("fit_domains") != list(records[0].fit_domains)
        or manifest.get("records_sha256") != records_sha
    ):
        raise G1StopExecutionError("fixed endpoint bank manifest changed")
    return (
        G1FixedEndpointBankFile(
            row_count=len(records),
            parquet_sha256=parquet_sha,
            records_sha256=records_sha,
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        ),
        records,
    )


def select_g1_source_fixed_reference(
    authorization: SourceTeacherAuthorization,
    records: tuple[G1FixedEndpointRecord, ...],
    *,
    task: InspectionTask,
) -> SourceFixedReference:
    if (
        type(authorization) is not SourceTeacherAuthorization
        or type(records) is not tuple
        or not records
        or any(type(row) is not G1FixedEndpointRecord for row in records)
        or task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or any(row.outer_target != authorization.outer_target for row in records)
        or any(row.source_domain == authorization.outer_target for row in records)
    ):
        raise G1StopExecutionError("source fixed-reference evidence is invalid")
    selected = tuple(
        ReferenceEndpoint(
            method=row.method,
            dataset_id=row.source_domain,
            specimen_id=row.specimen_id,
            task_loss=row.task_loss,
        )
        for row in records
        if row.task is task and row.source_domain in authorization.fit_domains
    )
    return select_source_fixed_reference(authorization, selected)


def fixed_endpoint_bank_path(
    work_root: str | Path,
    outer_target: str,
    source_domain: str,
) -> Path:
    if (
        type(outer_target) is not str
        or not outer_target
        or type(source_domain) is not str
        or not source_domain
        or outer_target == source_domain
        or any(
            "/" in value or "\\" in value or value in {".", ".."}
            for value in (outer_target, source_domain)
        )
    ):
        raise G1StopExecutionError("fixed endpoint fold identity is invalid")
    return Path(work_root) / outer_target / f"{source_domain}.parquet"


def _action_history_sha(actions: tuple[object, ...]) -> str:
    try:
        payload = tuple(
            (action.cell_index, action.from_level, action.to_level) for action in actions
        )
    except (AttributeError, TypeError) as error:
        raise G1StopExecutionError("fixed endpoint action history is invalid") from error
    return _json_sha(payload)


def _endpoint_task_loss(
    observation: object,
    grid: object,
    dependencies: G1SourceDependencies,
    *,
    full_scan: np.ndarray,
    true_cai: float,
    encoder: object,
) -> float:
    reconstruction = reconstruct_observation(
        observation,
        grid,
        dependencies.prior_fit.prior,
    )
    if observation.task is InspectionTask.FIELD:
        return float(field_loss(full_scan, reconstruction.image))
    if observation.task is not InspectionTask.CAI:
        raise G1StopExecutionError("fixed endpoint task is invalid")
    embedding = np.asarray(encoder.encode((reconstruction.image,)), dtype=np.float64)
    scalars = np.asarray(state_scalars(observation), dtype=np.float64)[None]
    prediction = np.asarray(
        dependencies.assessor_fit.assessor.predict(embedding, scalars),
        dtype=np.float64,
    )
    if (
        embedding.shape != (1, 512)
        or prediction.shape != (1,)
        or not np.all(np.isfinite(embedding))
        or not np.all(np.isfinite(prediction))
        or not math.isfinite(float(true_cai))
    ):
        raise G1StopExecutionError("fixed endpoint CAI evaluation is invalid")
    return abs(float(true_cai) - float(prediction[0]))


def materialize_g1_source_fixed_endpoint_records(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    specimen_ids: tuple[str, ...],
    progress: object = None,
) -> tuple[G1FixedEndpointRecord, ...]:
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or type(dependencies) is not G1SourceDependencies
        or not callable(getattr(encoder, "encode", None))
        or type(specimen_ids) is not tuple
        or not specimen_ids
        or len(set(specimen_ids)) != len(specimen_ids)
        or (progress is not None and not callable(progress))
    ):
        raise G1StopExecutionError("fixed endpoint materialization request is invalid")
    roster = dependencies.roster
    available = {
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == roster.labeled_domain
    }
    if not set(specimen_ids) <= available:
        raise G1StopExecutionError("fixed endpoint specimen is outside the source fold")
    output: list[G1FixedEndpointRecord] = []
    for specimen_index, specimen in enumerate(specimen_ids, start=1):
        teacher_view = runtime.mavis.source_teacher_view(specimen)
        specimen_sha = runtime.specimen_sha256(roster.labeled_domain, specimen)
        planned: dict[str, tuple[object, ...]] = {}
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            world, grid, surface = build_g1_world(
                runtime,
                dataset_id=roster.labeled_domain,
                specimen_id=specimen,
                task=task,
                endpoint_budget=protocol.endpoint_budget,
            )
            for method in FIXED_BASELINE_METHODS:
                if method not in planned:
                    planned[method] = plan_g1_fixed_actions(
                        grid,
                        surface.hypothesis,
                        surface_sha256=surface.surface_sha256,
                        specimen_sha256=specimen_sha,
                        method=method,
                        random_seed=protocol.teacher_bank_seed,
                        endpoint_budget=protocol.endpoint_budget,
                    )
                actions = planned[method]
                observation = world.replay(actions)
                output.append(
                    G1FixedEndpointRecord(
                        outer_target=roster.outer_target,
                        source_domain=roster.labeled_domain,
                        specimen_id=specimen,
                        specimen_sha256=specimen_sha,
                        task=task,
                        method=method,
                        fit_domains=roster.fit_domains,
                        dependency_sha256=dependencies.state_sha256,
                        action_history_sha256=_action_history_sha(actions),
                        endpoint_observation_sha256=observation.state_sha256,
                        effective_budget=observation.effective_budget,
                        exact_acquired_count=observation.exact_acquired_count,
                        native_count=observation.native_count,
                        task_loss=_endpoint_task_loss(
                            observation,
                            grid,
                            dependencies,
                            full_scan=teacher_view.full_scan,
                            true_cai=teacher_view.true_cai,
                            encoder=encoder,
                        ),
                    )
                )
        if progress is not None and (
            specimen_index % 10 == 0 or specimen_index == len(specimen_ids)
        ):
            progress(
                f"G1 fixed endpoints {roster.outer_target}/{roster.labeled_domain}: "
                f"{specimen_index}/{len(specimen_ids)} specimens"
            )
    return _ordered_records(tuple(output))


def build_g1_source_fixed_endpoint_bank(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    work_root: str | Path,
    progress: object = None,
) -> G1FixedEndpointBuild:
    source = dependencies.roster.labeled_domain
    specimen_ids = tuple(
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == source
    )
    records = materialize_g1_source_fixed_endpoint_records(
        runtime,
        protocol,
        dependencies,
        encoder=encoder,
        specimen_ids=specimen_ids,
        progress=progress,
    )
    path = fixed_endpoint_bank_path(
        work_root,
        dependencies.roster.outer_target,
        source,
    )
    bank = write_fixed_endpoint_bank(path, records)
    return G1FixedEndpointBuild(
        path=path,
        outer_target=dependencies.roster.outer_target,
        source_domain=source,
        specimen_count=len(specimen_ids),
        dependency_sha256=dependencies.state_sha256,
        bank=bank,
    )


def build_g1_all_source_fixed_endpoint_banks(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    encoder: object,
    work_root: str | Path,
    start_fold: int = 1,
    progress: object = None,
) -> tuple[G1FixedEndpointBuild, ...]:
    pairs = tuple(
        (outer, source)
        for outer in protocol.domain_order
        for source in protocol.domain_order
        if source != outer
    )
    if type(start_fold) is not int or not 1 <= start_fold <= len(pairs):
        raise G1StopExecutionError("fixed endpoint start fold is invalid")
    results = []
    for fold_index, (outer, source) in enumerate(
        pairs[start_fold - 1 :], start=start_fold
    ):
        if progress is not None:
            progress(f"G1 fixed endpoint fold {fold_index}/{len(pairs)}: {outer}/{source}")
        try:
            dependencies = build_g1_source_dependencies(
                runtime,
                protocol,
                outer_target=outer,
                labeled_domain=source,
                encoder=encoder,
                progress=progress,
            )
        except G1ExecutionError as error:
            raise G1StopExecutionError("fixed endpoint dependency fit failed") from error
        results.append(
            build_g1_source_fixed_endpoint_bank(
                runtime,
                protocol,
                dependencies,
                encoder=encoder,
                work_root=work_root,
                progress=progress,
            )
        )
    return tuple(results)


__all__ = [
    "G1FixedEndpointBankFile",
    "G1FixedEndpointBuild",
    "G1FixedEndpointRecord",
    "G1StopExecutionError",
    "build_g1_all_source_fixed_endpoint_banks",
    "build_g1_source_fixed_endpoint_bank",
    "fixed_endpoint_bank_path",
    "materialize_g1_source_fixed_endpoint_records",
    "read_fixed_endpoint_bank",
    "select_g1_source_fixed_reference",
    "write_fixed_endpoint_bank",
]
