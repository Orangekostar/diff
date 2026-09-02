"""Hash-bound source engineering curves and fixed-baseline selection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.evaluation import zero_inclusive_auebc

from .formal import FIXED_BASELINE_METHODS
from .metrics import NOMINAL_CHECKPOINTS, EngineeringCurve

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


def _curve_payload(curve: EngineeringCurve) -> dict[str, object]:
    return {
        "schema": 1,
        "kind": "g1-engineering-curve",
        "method": curve.method,
        "target_domain": curve.target_domain,
        "specimen_sha256": curve.specimen_sha256,
        "task": curve.task.value,
        "grid_sha256": curve.grid_sha256,
        "evaluator_sha256": curve.evaluator_sha256,
        "warm_start_sha256": curve.warm_start_sha256,
        "nominal_budgets": tuple(float(value) for value in curve.nominal_budgets),
        "exact_budgets": tuple(float(value) for value in curve.exact_budgets),
        "task_losses": tuple(float(value) for value in curve.task_losses),
        "projected_state_sha256": curve.projected_state_sha256,
        "auebc": float(curve.auebc),
    }


def _validate_curve(curve: EngineeringCurve) -> None:
    if type(curve) is not EngineeringCurve:
        raise G1SourceBridgeError("issued engineering curve is required")
    nominal = np.asarray(curve.nominal_budgets, dtype=np.float64)
    exact = np.asarray(curve.exact_budgets, dtype=np.float64)
    losses = np.asarray(curve.task_losses, dtype=np.float64)
    if (
        nominal.shape != (5,)
        or exact.shape != nominal.shape
        or losses.shape != nominal.shape
        or tuple(float(value) for value in nominal) != NOMINAL_CHECKPOINTS
        or not np.all(np.isfinite(exact))
        or not np.all(np.isfinite(losses))
        or np.any(exact < 0.0)
        or np.any(exact - nominal > 1.0e-15)
        or any(float(right) < float(left) for left, right in pairwise(exact))
        or np.any(losses < 0.0)
        or len(curve.projected_state_sha256) != 5
        or not all(_valid_sha256(value) for value in curve.projected_state_sha256)
        or not all(
            _valid_sha256(value)
            for value in (
                curve.specimen_sha256,
                curve.grid_sha256,
                curve.evaluator_sha256,
                curve.warm_start_sha256,
                curve.state_sha256,
            )
        )
        or not math.isclose(
            float(curve.auebc),
            zero_inclusive_auebc(nominal, losses),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or curve.state_sha256 != _json_sha(_curve_payload(curve))
    ):
        raise G1SourceBridgeError("source engineering curve identity changed")


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
        _validate_curve(self.curve)
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
        nominal = np.asarray(row["nominal_budgets"], dtype="<f8")
        exact = np.asarray(row["exact_budgets"], dtype="<f8")
        losses = np.asarray(row["task_losses"], dtype="<f8")
        projected_raw = json.loads(str(row["projected_state_sha256_json"]))
        if not isinstance(projected_raw, list):
            raise TypeError
        for value in (nominal, exact, losses):
            value.setflags(write=False)
        return EngineeringCurve(
            method=str(row["method"]),
            target_domain=str(row["source_domain"]),
            specimen_sha256=str(row["specimen_sha256"]),
            task=InspectionTask(str(row["task"])),
            grid_sha256=str(row["grid_sha256"]),
            evaluator_sha256=str(row["evaluator_sha256"]),
            warm_start_sha256=str(row["warm_start_sha256"]),
            nominal_budgets=nominal,
            exact_budgets=exact,
            task_losses=losses,
            projected_state_sha256=tuple(str(value) for value in projected_raw),
            auebc=float(row["auebc"]),
            state_sha256=str(row["curve_sha256"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
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
    "G1SourceBridgeError",
    "G1SourceBridgeRecord",
    "SourceFixedBridgeSelection",
    "read_source_bridge_bank",
    "select_source_fixed_bridge",
    "write_source_bridge_bank",
]
