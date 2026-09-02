"""Hash-bound source engineering curves and fixed-baseline selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid

from .formal import (
    FIXED_BASELINE_METHODS,
    evaluate_g1_action_history,
    plan_g1_fixed_actions,
    run_g1_warm_started_oracle_actions,
)
from .g1 import (
    G1Protocol,
    G1Runtime,
    G1SourceDependencies,
    build_g1_source_dependencies,
    build_g1_world,
)
from .metrics import (
    EngineeringCurve,
    G1MetricError,
    replay_engineering_curve,
    validate_engineering_curve,
)
from .teacher import (
    SourceTeacherAuthorization,
    validate_source_teacher_dependencies,
)

SOURCE_ORACLE_METHODS = {
    InspectionTask.FIELD: "ORACLE_FIELD",
    InspectionTask.CAI: "ORACLE_CAI",
}


class G1SourceBridgeError(ValueError):
    """Raised when source engineering evidence is incomplete or crosses a fold."""


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
class G1SourceBridgeRecord:
    outer_target: str
    source_domain: str
    specimen_id: str
    fit_domains: tuple[str, ...]
    dependency_sha256: str
    action_history_sha256: str
    curve: EngineeringCurve
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            validate_engineering_curve(self.curve)
        except G1MetricError as error:
            raise G1SourceBridgeError(
                "source engineering curve identity changed"
            ) from error
        allowed_methods = (
            *FIXED_BASELINE_METHODS,
            SOURCE_ORACLE_METHODS[self.curve.task],
        )
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or self.source_domain == self.outer_target
            or type(self.specimen_id) is not str
            or not self.specimen_id
            or type(self.fit_domains) is not tuple
            or len(self.fit_domains) != 4
            or len(set(self.fit_domains)) != 4
            or self.outer_target in self.fit_domains
            or self.source_domain in self.fit_domains
            or not _valid_sha256(self.dependency_sha256)
            or not _valid_sha256(self.action_history_sha256)
            or self.curve.target_domain != self.source_domain
            or self.curve.method not in allowed_methods
        ):
            raise G1SourceBridgeError("source bridge record is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-source-bridge-record",
                    "outer_target": self.outer_target,
                    "source_domain": self.source_domain,
                    "specimen_id": self.specimen_id,
                    "fit_domains": self.fit_domains,
                    "dependency": self.dependency_sha256,
                    "action_history": self.action_history_sha256,
                    "curve": self.curve.state_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1SourceBridgeBankFile:
    row_count: int
    parquet_sha256: str
    records_sha256: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class G1SourceBridgeBuild:
    path: Path
    outer_target: str
    source_domain: str
    specimen_count: int
    dependency_sha256: str
    bank: G1SourceBridgeBankFile


@dataclass(frozen=True, slots=True)
class SourceFixedBridgeSelection:
    outer_target: str
    validation_domain: str
    task: InspectionTask
    method: str
    fit_domains: tuple[str, ...]
    equal_domain_auebc: float
    domain_auebc: tuple[tuple[str, float], ...]
    evidence_sha256: str
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        domain_values = tuple(
            (str(domain), float(value)) for domain, value in self.domain_auebc
        )
        mean = float(self.equal_domain_auebc)
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.validation_domain) is not str
            or not self.validation_domain
            or self.validation_domain == self.outer_target
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or self.method not in FIXED_BASELINE_METHODS
            or type(self.fit_domains) is not tuple
            or len(self.fit_domains) != 4
            or len(set(self.fit_domains)) != 4
            or self.outer_target in self.fit_domains
            or self.validation_domain in self.fit_domains
            or type(self.domain_auebc) is not tuple
            or tuple(domain for domain, _value in domain_values) != self.fit_domains
            or any(not math.isfinite(value) or value < 0.0 for _, value in domain_values)
            or not math.isfinite(mean)
            or mean < 0.0
            or not math.isclose(
                mean,
                float(np.mean([value for _domain, value in domain_values])),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not _valid_sha256(self.evidence_sha256)
        ):
            raise G1SourceBridgeError("fixed source bridge selection is invalid")
        object.__setattr__(self, "domain_auebc", domain_values)
        object.__setattr__(self, "equal_domain_auebc", mean)
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-source-fixed-bridge-selection",
                    "outer_target": self.outer_target,
                    "validation_domain": self.validation_domain,
                    "task": self.task.value,
                    "method": self.method,
                    "fit_domains": self.fit_domains,
                    "equal_domain_auebc": mean,
                    "domain_auebc": domain_values,
                    "evidence": self.evidence_sha256,
                }
            ),
        )


def _record_key(record: G1SourceBridgeRecord) -> tuple[object, ...]:
    return (
        record.outer_target,
        record.source_domain,
        record.specimen_id,
        record.curve.task.value,
        record.curve.method,
    )


def _action_history_sha(actions: tuple[object, ...]) -> str:
    try:
        tokens = tuple(
            (action.cell_index, action.from_level, action.to_level)
            for action in actions
        )
    except AttributeError as error:
        raise G1SourceBridgeError("source bridge action history is invalid") from error
    if type(actions) is not tuple or not actions:
        raise G1SourceBridgeError("source bridge action history is invalid")
    return _json_sha(
        {
            "schema": 1,
            "kind": "g1-source-bridge-action-history",
            "actions": tokens,
        }
    )


def materialize_source_bridge_for_world(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    surface_hypothesis: SurfaceHypothesis,
    prior: SourceBackgroundPrior,
    authorization: SourceTeacherAuthorization,
    *,
    assessor: object,
    encoder: object,
    outer_target: str,
    source_domain: str,
    specimen_id: str,
    specimen_sha256: str,
    dependency_sha256: str,
    full_scan: np.ndarray,
    true_cai: float,
    random_seed: int,
) -> tuple[G1SourceBridgeRecord, ...]:
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(surface_hypothesis) is not SurfaceHypothesis
        or type(prior) is not SourceBackgroundPrior
        or type(authorization) is not SourceTeacherAuthorization
        or not callable(getattr(assessor, "predict", None))
        or not _valid_sha256(getattr(assessor, "model_state_sha256", None))
        or not callable(getattr(encoder, "encode", None))
        or outer_target != authorization.outer_target
        or source_domain != authorization.labeled_domain
        or source_domain == outer_target
        or type(specimen_id) is not str
        or not specimen_id
        or not _valid_sha256(specimen_sha256)
        or not _valid_sha256(dependency_sha256)
        or type(random_seed) is not int
    ):
        raise G1SourceBridgeError("source bridge world request is invalid")
    validate_source_teacher_dependencies(
        authorization,
        prior,
        assessor=assessor,
    )
    initial = world.reset()
    if initial.grid_sha256 != grid.state_sha256:
        raise G1SourceBridgeError("source bridge world grid changed")
    task = initial.task
    output = []
    for method in FIXED_BASELINE_METHODS:
        actions = plan_g1_fixed_actions(
            grid,
            surface_hypothesis,
            surface_sha256=initial.surface_sha256,
            specimen_sha256=specimen_sha256,
            method=method,
            random_seed=random_seed,
            endpoint_budget=initial.endpoint_budget,
        )
        curve = evaluate_g1_action_history(
            world,
            grid,
            prior,
            assessor=assessor,
            encoder=encoder,
            method=method,
            target_domain=source_domain,
            specimen_sha256=specimen_sha256,
            actions=actions,
            full_scan=full_scan,
            true_cai=true_cai,
        )
        output.append(
            G1SourceBridgeRecord(
                outer_target=outer_target,
                source_domain=source_domain,
                specimen_id=specimen_id,
                fit_domains=authorization.fit_domains,
                dependency_sha256=dependency_sha256,
                action_history_sha256=_action_history_sha(actions),
                curve=curve,
            )
        )
    oracle_actions = run_g1_warm_started_oracle_actions(
        world,
        grid,
        prior,
        surface_hypothesis=surface_hypothesis,
        full_scan=full_scan,
        true_cai=true_cai,
        assessor=assessor,
        encoder=encoder,
    )
    oracle_method = SOURCE_ORACLE_METHODS[task]
    oracle_curve = evaluate_g1_action_history(
        world,
        grid,
        prior,
        assessor=assessor,
        encoder=encoder,
        method=oracle_method,
        target_domain=source_domain,
        specimen_sha256=specimen_sha256,
        actions=oracle_actions,
        full_scan=full_scan,
        true_cai=true_cai,
    )
    output.append(
        G1SourceBridgeRecord(
            outer_target=outer_target,
            source_domain=source_domain,
            specimen_id=specimen_id,
            fit_domains=authorization.fit_domains,
            dependency_sha256=dependency_sha256,
            action_history_sha256=_action_history_sha(oracle_actions),
            curve=oracle_curve,
        )
    )
    return tuple(output)


def source_bridge_bank_path(
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
        raise G1SourceBridgeError("source bridge fold identity is invalid")
    return Path(work_root) / outer_target / f"{source_domain}.parquet"


def materialize_g1_source_bridge_records(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    specimen_ids: tuple[str, ...],
    progress: Callable[[str], None] | None = None,
) -> tuple[G1SourceBridgeRecord, ...]:
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
        raise G1SourceBridgeError("source bridge materialization request is invalid")
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
        raise G1SourceBridgeError("source bridge specimen is outside its source fold")
    output = []
    prior = dependencies.prior_fit.prior
    assessor = dependencies.assessor_fit.assessor
    for specimen_index, specimen in enumerate(specimen_ids, start=1):
        teacher_view = runtime.mavis.source_teacher_view(specimen)
        specimen_sha = runtime.specimen_sha256(roster.labeled_domain, specimen)
        for task in (InspectionTask.FIELD, InspectionTask.CAI):
            world, grid, surface = build_g1_world(
                runtime,
                dataset_id=roster.labeled_domain,
                specimen_id=specimen,
                task=task,
                endpoint_budget=protocol.endpoint_budget,
            )
            output.extend(
                materialize_source_bridge_for_world(
                    world,
                    grid,
                    surface.hypothesis,
                    prior,
                    dependencies.authorization,
                    assessor=assessor,
                    encoder=encoder,
                    outer_target=roster.outer_target,
                    source_domain=roster.labeled_domain,
                    specimen_id=specimen,
                    specimen_sha256=specimen_sha,
                    dependency_sha256=dependencies.state_sha256,
                    full_scan=teacher_view.full_scan,
                    true_cai=teacher_view.true_cai,
                    random_seed=protocol.teacher_bank_seed,
                )
            )
        if progress is not None and (
            specimen_index % 10 == 0 or specimen_index == len(specimen_ids)
        ):
            progress(
                f"G1 source bridge {roster.outer_target}/{roster.labeled_domain}: "
                f"{specimen_index}/{len(specimen_ids)} specimens"
            )
    records = tuple(output)
    if len(records) != 12 * len(specimen_ids):
        raise G1SourceBridgeError("source bridge row count changed")
    return records


def build_g1_source_bridge_bank(
    runtime: G1Runtime,
    protocol: G1Protocol,
    dependencies: G1SourceDependencies,
    *,
    encoder: object,
    work_root: str | Path,
    progress: Callable[[str], None] | None = None,
) -> G1SourceBridgeBuild:
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or type(dependencies) is not G1SourceDependencies
        or not callable(getattr(encoder, "encode", None))
    ):
        raise G1SourceBridgeError("source bridge build request is invalid")
    roster = dependencies.roster
    specimen_ids = tuple(
        specimen
        for specimen, domain in zip(
            runtime.mavis.specimen_ids,
            runtime.mavis.dataset_ids,
            strict=True,
        )
        if domain == roster.labeled_domain
    )
    if len(specimen_ids) != int(protocol.domain_counts[roster.labeled_domain]):
        raise G1SourceBridgeError("formal source bridge specimen roster changed")
    path = source_bridge_bank_path(
        work_root,
        roster.outer_target,
        roster.labeled_domain,
    )
    manifest_path = _manifest_path(path)
    if path.exists() or manifest_path.exists():
        bank, records = read_source_bridge_bank(path)
        expected_specimens = {
            runtime.specimen_sha256(roster.labeled_domain, specimen)
            for specimen in specimen_ids
        }
        if (
            {row.curve.specimen_sha256 for row in records} != expected_specimens
            or len(records) != 12 * len(specimen_ids)
            or any(
                row.fit_domains != roster.fit_domains
                or row.dependency_sha256 != dependencies.state_sha256
                for row in records
            )
        ):
            raise G1SourceBridgeError("existing source bridge has stale evidence")
        if progress is not None:
            progress(f"G1 source bridge reused: {path}")
    else:
        records = materialize_g1_source_bridge_records(
            runtime,
            protocol,
            dependencies,
            encoder=encoder,
            specimen_ids=specimen_ids,
            progress=progress,
        )
        bank = write_source_bridge_bank(path, records)
    return G1SourceBridgeBuild(
        path=path,
        outer_target=roster.outer_target,
        source_domain=roster.labeled_domain,
        specimen_count=len(specimen_ids),
        dependency_sha256=dependencies.state_sha256,
        bank=bank,
    )


def build_g1_all_source_bridge_banks(
    runtime: G1Runtime,
    protocol: G1Protocol,
    *,
    encoder: object,
    work_root: str | Path,
    start_fold: int = 1,
    progress: Callable[[str], None] | None = None,
) -> tuple[G1SourceBridgeBuild, ...]:
    pairs = tuple(
        (outer, source)
        for outer in protocol.domain_order
        for source in protocol.domain_order
        if source != outer
    )
    if (
        type(runtime) is not G1Runtime
        or type(protocol) is not G1Protocol
        or runtime.domain_order != protocol.domain_order
        or not callable(getattr(encoder, "encode", None))
        or type(start_fold) is not int
        or not 1 <= start_fold <= len(pairs)
        or (progress is not None and not callable(progress))
    ):
        raise G1SourceBridgeError("all-source bridge build request is invalid")
    selected = pairs[start_fold - 1 :]
    output = []
    for fold_index, (outer, source) in enumerate(selected, start=start_fold):
        if progress is not None:
            progress(
                f"G1 source bridge fold {fold_index}/{len(pairs)}: {outer}/{source}"
            )
        dependencies = build_g1_source_dependencies(
            runtime,
            protocol,
            outer_target=outer,
            labeled_domain=source,
            encoder=encoder,
            progress=progress,
        )
        result = build_g1_source_bridge_bank(
            runtime,
            protocol,
            dependencies,
            encoder=encoder,
            work_root=work_root,
            progress=progress,
        )
        if (result.outer_target, result.source_domain) != (outer, source):
            raise G1SourceBridgeError("source bridge fold identity changed")
        output.append(result)
    if tuple((row.outer_target, row.source_domain) for row in output) != selected:
        raise G1SourceBridgeError("source bridge directed roster changed")
    return tuple(output)


def _ordered_records(
    records: tuple[G1SourceBridgeRecord, ...],
) -> tuple[G1SourceBridgeRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1SourceBridgeRecord for row in records)
        or len({row.state_sha256 for row in records}) != len(records)
        or len({_record_key(row) for row in records}) != len(records)
        or len({row.outer_target for row in records}) != 1
        or len({row.source_domain for row in records}) != 1
        or len({row.fit_domains for row in records}) != 1
        or len({row.dependency_sha256 for row in records}) != 1
    ):
        raise G1SourceBridgeError("source bridge bank roster is invalid")
    expected_by_task = {
        task: {*FIXED_BASELINE_METHODS, SOURCE_ORACLE_METHODS[task]}
        for task in (InspectionTask.FIELD, InspectionTask.CAI)
    }
    specimens = {row.specimen_id for row in records}
    for specimen in specimens:
        specimen_rows = tuple(row for row in records if row.specimen_id == specimen)
        if (
            len({row.curve.specimen_sha256 for row in specimen_rows}) != 1
            or len({row.curve.grid_sha256 for row in specimen_rows}) != 1
        ):
            raise G1SourceBridgeError("source bridge specimen identity changed")
        if {row.curve.task for row in specimen_rows} != set(expected_by_task):
            raise G1SourceBridgeError("source bridge task roster is incomplete")
        for task, expected in expected_by_task.items():
            methods = {
                row.curve.method for row in specimen_rows if row.curve.task is task
            }
            if methods != expected:
                raise G1SourceBridgeError("source bridge method roster is incomplete")
    return tuple(sorted(records, key=_record_key))


def _records_sha(records: tuple[G1SourceBridgeRecord, ...]) -> str:
    return _json_sha(tuple(row.state_sha256 for row in records))


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.manifest.json")


def _row(record: G1SourceBridgeRecord) -> dict[str, object]:
    curve = record.curve
    return {
        "outer_target": record.outer_target,
        "source_domain": record.source_domain,
        "specimen_id": record.specimen_id,
        "specimen_sha256": curve.specimen_sha256,
        "task": curve.task.value,
        "method": curve.method,
        "fit_domains_json": json.dumps(record.fit_domains, separators=(",", ":")),
        "dependency_sha256": record.dependency_sha256,
        "action_history_sha256": record.action_history_sha256,
        "grid_sha256": curve.grid_sha256,
        "evaluator_sha256": curve.evaluator_sha256,
        "warm_start_sha256": curve.warm_start_sha256,
        "nominal_budgets": curve.nominal_budgets.tolist(),
        "exact_budgets": curve.exact_budgets.tolist(),
        "task_losses": curve.task_losses.tolist(),
        "projected_state_sha256_json": json.dumps(
            curve.projected_state_sha256, separators=(",", ":")
        ),
        "auebc": curve.auebc,
        "curve_sha256": curve.state_sha256,
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


def write_source_bridge_bank(
    path: str | Path,
    records: tuple[G1SourceBridgeRecord, ...],
) -> G1SourceBridgeBankFile:
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
            "scope": "inspection_agent_g1_source_engineering_bridge",
            "row_count": len(ordered),
            "outer_target": ordered[0].outer_target,
            "source_domain": ordered[0].source_domain,
            "fit_domains": list(ordered[0].fit_domains),
            "dependency_sha256": ordered[0].dependency_sha256,
            "parquet_sha256": parquet_sha,
            "records_sha256": records_sha,
        }
        manifest_payload = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        os.replace(temporary, destination)
        _atomic_bytes(_manifest_path(destination), manifest_payload)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    identity, replay = read_source_bridge_bank(destination)
    if tuple(row.state_sha256 for row in replay) != tuple(
        row.state_sha256 for row in ordered
    ):
        raise G1SourceBridgeError("written source bridge bank did not replay exactly")
    return identity


def _curve_from_row(row: dict[str, object]) -> EngineeringCurve:
    try:
        projected_raw = json.loads(str(row["projected_state_sha256_json"]))
        if not isinstance(projected_raw, list):
            raise TypeError
        curve = replay_engineering_curve(
            method=str(row["method"]),
            target_domain=str(row["source_domain"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            grid_sha256=str(row["grid_sha256"]),
            evaluator_sha256=str(row["evaluator_sha256"]),
            warm_start_sha256=str(row["warm_start_sha256"]),
            exact_budgets=row["exact_budgets"],
            task_losses=row["task_losses"],
            projected_state_sha256=tuple(str(value) for value in projected_raw),
            state_sha256=str(row["curve_sha256"]),
        )
        nominal = tuple(float(value) for value in row["nominal_budgets"])
        if (
            nominal != tuple(float(value) for value in curve.nominal_budgets)
            or float(row["auebc"]) != curve.auebc
        ):
            raise G1SourceBridgeError("source bridge curve summary changed")
        return curve
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, G1SourceBridgeError):
            raise
        raise G1SourceBridgeError("source bridge curve cannot be decoded") from error


def _record_from_row(row: dict[str, object]) -> G1SourceBridgeRecord:
    try:
        fit_raw = json.loads(str(row["fit_domains_json"]))
        if not isinstance(fit_raw, list):
            raise TypeError
        record = G1SourceBridgeRecord(
            outer_target=str(row["outer_target"]),
            source_domain=str(row["source_domain"]),
            specimen_id=str(row["specimen_id"]),
            fit_domains=tuple(str(value) for value in fit_raw),
            dependency_sha256=str(row["dependency_sha256"]),
            action_history_sha256=str(row["action_history_sha256"]),
            curve=_curve_from_row(row),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, G1SourceBridgeError):
            raise
        raise G1SourceBridgeError("source bridge record cannot be decoded") from error
    if record.state_sha256 != str(row.get("record_sha256", "")):
        raise G1SourceBridgeError("source bridge record hash changed")
    return record


def read_source_bridge_bank(
    path: str | Path,
) -> tuple[G1SourceBridgeBankFile, tuple[G1SourceBridgeRecord, ...]]:
    source = Path(path)
    manifest_path = _manifest_path(source)
    try:
        parquet_payload = source.read_bytes()
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        rows = pl.read_parquet(source).to_dicts()
    except (OSError, json.JSONDecodeError, UnicodeError, pl.exceptions.PolarsError) as error:
        raise G1SourceBridgeError("source bridge bank cannot be read") from error
    expected_keys = {
        "schema_version",
        "scope",
        "row_count",
        "outer_target",
        "source_domain",
        "fit_domains",
        "dependency_sha256",
        "parquet_sha256",
        "records_sha256",
    }
    parquet_sha = hashlib.sha256(parquet_payload).hexdigest()
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("scope") != "inspection_agent_g1_source_engineering_bridge"
        or manifest.get("parquet_sha256") != parquet_sha
    ):
        raise G1SourceBridgeError("source bridge bank manifest changed")
    records = tuple(_record_from_row(row) for row in rows)
    ordered = _ordered_records(records)
    records_sha = _records_sha(ordered)
    if (
        records != ordered
        or manifest.get("row_count") != len(records)
        or manifest.get("outer_target") != records[0].outer_target
        or manifest.get("source_domain") != records[0].source_domain
        or manifest.get("fit_domains") != list(records[0].fit_domains)
        or manifest.get("dependency_sha256") != records[0].dependency_sha256
        or manifest.get("records_sha256") != records_sha
    ):
        raise G1SourceBridgeError("source bridge bank evidence changed")
    return (
        G1SourceBridgeBankFile(
            row_count=len(records),
            parquet_sha256=parquet_sha,
            records_sha256=records_sha,
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        ),
        records,
    )


def select_source_fixed_bridge(
    records: tuple[G1SourceBridgeRecord, ...],
    *,
    validation_domain: str,
    task: InspectionTask,
) -> SourceFixedBridgeSelection:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1SourceBridgeRecord for row in records)
        or type(validation_domain) is not str
        or not validation_domain
        or task not in (InspectionTask.FIELD, InspectionTask.CAI)
        or len({row.outer_target for row in records}) != 1
    ):
        raise G1SourceBridgeError("fixed source bridge request is invalid")
    outer = records[0].outer_target
    fixed = tuple(
        row
        for row in records
        if row.curve.task is task and row.curve.method in FIXED_BASELINE_METHODS
    )
    sources = tuple(dict.fromkeys(row.source_domain for row in fixed))
    validation_rows = tuple(row for row in fixed if row.source_domain == validation_domain)
    if not validation_rows:
        raise G1SourceBridgeError("source validation bridge is absent")
    fit_domains = validation_rows[0].fit_domains
    if (
        len(sources) != 5
        or set(sources) != {validation_domain, *fit_domains}
        or outer in sources
        or any(
            set(row.fit_domains) != set(sources) - {row.source_domain}
            for row in fixed
        )
    ):
        raise G1SourceBridgeError("source bridge crossfit roster changed")
    for source in sources:
        source_rows = tuple(row for row in fixed if row.source_domain == source)
        by_method = {
            method: tuple(row for row in source_rows if row.curve.method == method)
            for method in FIXED_BASELINE_METHODS
        }
        specimen_sets = {
            tuple(row.curve.specimen_sha256 for row in values)
            for values in by_method.values()
        }
        if (
            any(not values for values in by_method.values())
            or len(specimen_sets) != 1
            or len(next(iter(specimen_sets)))
            != len(set(next(iter(specimen_sets))))
        ):
            raise G1SourceBridgeError("fixed source bridge method roster changed")
    means: dict[str, tuple[tuple[str, float], ...]] = {}
    for method in FIXED_BASELINE_METHODS:
        means[method] = tuple(
            (
                domain,
                float(
                    np.mean(
                        [
                            row.curve.auebc
                            for row in fixed
                            if row.source_domain == domain
                            and row.curve.method == method
                        ],
                        dtype=np.float64,
                    )
                ),
            )
            for domain in fit_domains
        )
    selected = min(
        FIXED_BASELINE_METHODS,
        key=lambda method: (
            float(np.mean([value for _domain, value in means[method]])),
            FIXED_BASELINE_METHODS.index(method),
        ),
    )
    domain_auebc = means[selected]
    evidence = _json_sha(
        tuple(
            row.state_sha256
            for row in sorted(fixed, key=_record_key)
            if row.source_domain in fit_domains
        )
    )
    return SourceFixedBridgeSelection(
        outer_target=outer,
        validation_domain=validation_domain,
        task=task,
        method=selected,
        fit_domains=fit_domains,
        equal_domain_auebc=float(
            np.mean([value for _domain, value in domain_auebc], dtype=np.float64)
        ),
        domain_auebc=domain_auebc,
        evidence_sha256=evidence,
    )


__all__ = [
    "SOURCE_ORACLE_METHODS",
    "G1SourceBridgeBankFile",
    "G1SourceBridgeBuild",
    "G1SourceBridgeError",
    "G1SourceBridgeRecord",
    "SourceFixedBridgeSelection",
    "build_g1_all_source_bridge_banks",
    "build_g1_source_bridge_bank",
    "materialize_g1_source_bridge_records",
    "materialize_source_bridge_for_world",
    "read_source_bridge_bank",
    "select_source_fixed_bridge",
    "source_bridge_bank_path",
    "write_source_bridge_bank",
]
