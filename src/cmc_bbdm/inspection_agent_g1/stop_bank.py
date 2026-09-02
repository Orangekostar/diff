"""Hash-bound source STOP labels keyed to observable action-bank states."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from cmc_bbdm.inspection_agent.contracts import InspectionTask

from .stopping_policy import SourceStopLabel


class G1StopBankError(ValueError):
    """Raised when a STOP bank cannot be joined or replayed exactly."""


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
class G1StopBankRecord:
    outer_target: str
    source_domain: str
    specimen_sha256: str
    task: InspectionTask
    fit_domains: tuple[str, ...]
    state_source: str
    source_state_sha256: str
    action_example_sha256: str
    policy_state_sha256: str
    label: SourceStopLabel
    state_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.outer_target) is not str
            or not self.outer_target
            or type(self.source_domain) is not str
            or not self.source_domain
            or self.source_domain == self.outer_target
            or not _valid_sha256(self.specimen_sha256)
            or self.task not in (InspectionTask.FIELD, InspectionTask.CAI)
            or type(self.fit_domains) is not tuple
            or len(self.fit_domains) != 4
            or self.fit_domains != tuple(sorted(set(self.fit_domains)))
            or {self.outer_target, self.source_domain} & set(self.fit_domains)
            or type(self.state_source) is not str
            or not self.state_source
            or not _valid_sha256(self.source_state_sha256)
            or not _valid_sha256(self.action_example_sha256)
            or not _valid_sha256(self.policy_state_sha256)
            or type(self.label) is not SourceStopLabel
            or self.label.source_domain != self.source_domain
            or self.label.specimen_sha256 != self.specimen_sha256
            or self.label.task is not self.task
            or self.label.policy_state_sha256 != self.policy_state_sha256
        ):
            raise G1StopBankError("STOP-bank record is invalid")
        object.__setattr__(
            self,
            "state_sha256",
            _json_sha(
                {
                    "schema": 1,
                    "kind": "g1-stop-bank-record",
                    "outer_target": self.outer_target,
                    "source_domain": self.source_domain,
                    "specimen_sha256": self.specimen_sha256,
                    "task": self.task.value,
                    "fit_domains": self.fit_domains,
                    "state_source": self.state_source,
                    "source_state_sha256": self.source_state_sha256,
                    "action_example_sha256": self.action_example_sha256,
                    "policy_state_sha256": self.policy_state_sha256,
                    "label": self.label.state_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class G1StopBankFile:
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
            raise G1StopBankError("STOP-bank file identity is invalid")


def _ordered_records(
    records: tuple[G1StopBankRecord, ...],
) -> tuple[G1StopBankRecord, ...]:
    if (
        type(records) is not tuple
        or not records
        or any(type(row) is not G1StopBankRecord for row in records)
        or len({row.policy_state_sha256 for row in records}) != len(records)
        or len({row.state_sha256 for row in records}) != len(records)
        or len({row.outer_target for row in records}) != 1
        or len({row.source_domain for row in records}) != 1
        or len({row.fit_domains for row in records}) != 1
    ):
        message = (
            "STOP-bank policy-state join key is duplicated"
            if records
            and len({row.policy_state_sha256 for row in records}) != len(records)
            else "STOP-bank record roster is invalid"
        )
        raise G1StopBankError(message)
    return tuple(
        sorted(
            records,
            key=lambda row: (
                row.specimen_sha256,
                row.task.value,
                row.state_source,
                row.policy_state_sha256,
            ),
        )
    )


def _record_row(record: G1StopBankRecord) -> dict[str, object]:
    label = record.label
    return {
        "integrity_outer_target": record.outer_target,
        "integrity_source_domain": record.source_domain,
        "integrity_specimen_sha256": record.specimen_sha256,
        "integrity_task": record.task.value,
        "integrity_fit_domains_json": json.dumps(
            record.fit_domains, separators=(",", ":")
        ),
        "integrity_state_source": record.state_source,
        "integrity_source_state_sha256": record.source_state_sha256,
        "integrity_action_example_sha256": record.action_example_sha256,
        "integrity_record_sha256": record.state_sha256,
        "policy_visible_state_sha256": record.policy_state_sha256,
        "privileged_stop_authorization_sha256": label.authorization_sha256,
        "privileged_stop_fixed_reference_sha256": label.fixed_reference_sha256,
        "privileged_stop_reference_method": label.reference_method,
        "privileged_stop_current_true_loss": label.current_true_loss,
        "privileged_stop_reference_true_loss": label.reference_true_loss,
        "privileged_stop_tolerance": label.tolerance,
        "privileged_stop_is_sufficient": label.is_sufficient,
        "privileged_stop_label_sha256": label.state_sha256,
    }


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.manifest.json")


def _records_sha(records: tuple[G1StopBankRecord, ...]) -> str:
    return _json_sha(tuple(row.state_sha256 for row in records))


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


def write_stop_bank(
    path: str | Path,
    records: tuple[G1StopBankRecord, ...],
) -> G1StopBankFile:
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
            [_record_row(row) for row in ordered], infer_schema_length=None
        ).write_parquet(
            temporary,
            compression="zstd",
            statistics=False,
            row_group_size=512,
        )
        payload = temporary.read_bytes()
        parquet_sha = hashlib.sha256(payload).hexdigest()
        records_sha = _records_sha(ordered)
        manifest = {
            "schema_version": 1,
            "scope": "inspection_agent_g1_source_stop_bank",
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
        _atomic_replace_bytes(_manifest_path(destination), manifest_payload)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    identity, loaded = read_stop_bank(destination)
    if loaded != ordered:
        raise G1StopBankError("written STOP bank did not replay exactly")
    return identity


def _record_from_row(row: dict[str, object]) -> G1StopBankRecord:
    try:
        task = InspectionTask(str(row["integrity_task"]))
        fit_raw = json.loads(str(row["integrity_fit_domains_json"]))
        if not isinstance(fit_raw, list):
            raise TypeError
        label = SourceStopLabel(
            authorization_sha256=str(row["privileged_stop_authorization_sha256"]),
            fixed_reference_sha256=str(
                row["privileged_stop_fixed_reference_sha256"]
            ),
            source_domain=str(row["integrity_source_domain"]),
            specimen_sha256=str(row["integrity_specimen_sha256"]),
            task=task,
            policy_state_sha256=str(row["policy_visible_state_sha256"]),
            reference_method=str(row["privileged_stop_reference_method"]),
            current_true_loss=float(row["privileged_stop_current_true_loss"]),
            reference_true_loss=float(row["privileged_stop_reference_true_loss"]),
            tolerance=float(row["privileged_stop_tolerance"]),
            is_sufficient=bool(row["privileged_stop_is_sufficient"]),
            state_sha256=str(row["privileged_stop_label_sha256"]),
        )
        result = G1StopBankRecord(
            outer_target=str(row["integrity_outer_target"]),
            source_domain=str(row["integrity_source_domain"]),
            specimen_sha256=str(row["integrity_specimen_sha256"]),
            task=task,
            fit_domains=tuple(str(value) for value in fit_raw),
            state_source=str(row["integrity_state_source"]),
            source_state_sha256=str(row["integrity_source_state_sha256"]),
            action_example_sha256=str(row["integrity_action_example_sha256"]),
            policy_state_sha256=str(row["policy_visible_state_sha256"]),
            label=label,
        )
        if result.state_sha256 != str(row["integrity_record_sha256"]):
            raise G1StopBankError("STOP-bank record hash changed")
        return result
    except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
        if isinstance(error, G1StopBankError):
            raise
        raise G1StopBankError("STOP-bank row cannot be reconstructed") from error


def read_stop_bank(
    path: str | Path,
) -> tuple[G1StopBankFile, tuple[G1StopBankRecord, ...]]:
    source = Path(path)
    manifest_path = _manifest_path(source)
    try:
        parquet_payload = source.read_bytes()
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        rows = pl.read_parquet(source).to_dicts()
    except (OSError, UnicodeError, json.JSONDecodeError, pl.exceptions.PolarsError) as error:
        raise G1StopBankError("STOP-bank package cannot be read") from error
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
        or manifest.get("scope") != "inspection_agent_g1_source_stop_bank"
        or manifest.get("parquet_sha256") != parquet_sha
    ):
        raise G1StopBankError("STOP-bank Parquet SHA-256 mismatch")
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
        raise G1StopBankError("STOP-bank manifest or row order changed")
    identity = G1StopBankFile(
        row_count=len(records),
        parquet_sha256=parquet_sha,
        records_sha256=records_sha,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
    )
    return identity, records


__all__ = [
    "G1StopBankError",
    "G1StopBankFile",
    "G1StopBankRecord",
    "read_stop_bank",
    "write_stop_bank",
]
