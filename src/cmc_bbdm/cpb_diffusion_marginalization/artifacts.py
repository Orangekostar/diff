"""Validated and transactional publication for the D8 pilot search package."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from .config import DOMAIN_ORDER, D8Config, SourceRecord, load_d8_config
from .pilot import D8PilotStudyEvidence, decide_pilot_escalation
from .search import D8Candidate, robust_inner_objective
from .selection import validate_selection_evidence
from .tracking import TRIAL_INDEX_FIELDS

_SOURCE_FILES = frozenset(
    {
        "trial_index.csv",
        "study.db",
        "residual_bank_manifest.json",
        "search_summary.csv",
        "selected_configs.json",
        "escalation_evidence.json",
        "pilot_report.md",
    }
)
_ROOT_FILES = frozenset({*_SOURCE_FILES, "artifact_manifest.json", "CHECKSUMS.sha256"})
_SUMMARY_FIELDS = (
    "outer_domain",
    "initial_trial_count",
    "trial_count",
    "completed_count",
    "pruned_count",
    "failed_count",
    "best_objective",
    "best_candidate_sha256",
    "selection_state_sha256",
    "escalation_evidence_sha256",
)
_DECISIONS = frozenset(
    {
        "TRAIN_RESIDUAL_DIFFUSION",
        "FREEZE_PILOT_FOR_OUTER_EVALUATION",
        "CLOSE_DIFFUSION_SPECIFIC_ROUTE",
    }
)
_DIFFUSION_CONTROLS = frozenset({"B5", "B6", "B7", "B8"})
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_MAX_FILE_BYTES = 1024 * 1024 * 1024
_P1_PREDICTION_FIELDS = (
    "method",
    "specimen_id",
    "dataset_id",
    "target",
    "prediction",
    "seed",
)


class D8ArtifactError(ValueError):
    """Raised when a D8 search package is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class D8ValidatedSearchPackage:
    outer_domains: tuple[str, ...]
    initial_trial_count: int
    trial_count: int
    outer_evaluation_count: int
    escalation_status: str
    scientific_digest: str
    output_tree_sha256: str
    artifact_manifest_sha256: str
    required_files: frozenset[str]


@dataclass(frozen=True, slots=True)
class _SemanticEvidence:
    initial_trial_count: int
    trial_count: int
    outer_evaluation_count: int
    escalation_status: str
    scientific_digest: str


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - _SHA256_CHARACTERS)
    )


def _read_regular(path: Path, label: str, *, maximum: int = _MAX_FILE_BYTES) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise D8ArtifactError(f"{label} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise D8ArtifactError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise D8ArtifactError(f"{label} is not a bounded regular file")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or total != before.st_size:
        raise D8ArtifactError(f"{label} changed while reading")
    return b"".join(chunks)


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(_read_regular(path, label).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise D8ArtifactError(f"{label} is invalid JSON") from error


def _read_csv(path: Path, fields: Sequence[str]) -> tuple[dict[str, str], ...]:
    try:
        payload = _read_regular(path, path.name, maximum=256 * 1024 * 1024)
        lines = payload.decode("utf-8").splitlines(keepends=True)
        reader = csv.DictReader(lines, strict=True)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise D8ArtifactError(f"{path.name} schema changed")
        rows = tuple(dict(row) for row in reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise D8ArtifactError(f"{path.name} row width changed")
        return rows
    except (UnicodeError, csv.Error) as error:
        raise D8ArtifactError(f"{path.name} is invalid CSV") from error


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is int:
        result = value
    elif type(value) is str:
        try:
            result = int(value)
        except ValueError as error:
            raise D8ArtifactError(f"{label} is not an integer") from error
        if str(result) != value:
            raise D8ArtifactError(f"{label} is not canonical")
    else:
        raise D8ArtifactError(f"{label} is not an integer")
    if result < minimum:
        raise D8ArtifactError(f"{label} is below its minimum")
    return result


def _float(value: object, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in {str, float, int} or type(value) is bool:
        raise D8ArtifactError(f"{label} is not numeric")
    try:
        result = float(value)
    except ValueError as error:
        raise D8ArtifactError(f"{label} is not numeric") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise D8ArtifactError(f"{label} is invalid")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-15)


def _source_authorities(config: D8Config) -> dict[str, dict[str, str]]:
    return {
        name: {"path": record.path, "sha256": record.sha256}
        for name, record in sorted(config.sources.items())
        if type(record) is SourceRecord
    }


def _regular_file_set(root: Path, expected: frozenset[str], label: str) -> None:
    if not root.is_dir() or root.is_symlink():
        raise D8ArtifactError(f"{label} is not a regular directory")
    observed: set[str] = set()
    try:
        entries = tuple(root.iterdir())
    except OSError as error:
        raise D8ArtifactError(f"{label} is unavailable") from error
    for entry in entries:
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise D8ArtifactError(f"{label} entry is unavailable") from error
        if entry.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise D8ArtifactError(f"{label} contains a non-regular file")
        observed.add(entry.name)
    if observed != set(expected):
        raise D8ArtifactError(f"{label} file set changed")


def _output_records(root: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for name in sorted(_SOURCE_FILES):
        payload = _read_regular(root / name, f"D8 output {name}")
        records[name] = {"bytes": len(payload), "sha256": _sha_bytes(payload)}
    return records


def _tree_sha256(records: Mapping[str, object]) -> str:
    return _sha_bytes(_canonical(records))


def _validate_acceptance_evidence(
    row: Mapping[str, str],
    *,
    inner_domains: tuple[str, ...],
    require_eligible: bool,
) -> tuple[int, int, float, dict[str, dict[str, object]]]:
    try:
        acceptance = json.loads(row["acceptance_by_domain"])
    except json.JSONDecodeError as error:
        raise D8ArtifactError("trial acceptance evidence is invalid") from error
    if not isinstance(acceptance, dict) or set(acceptance) != set(inner_domains):
        raise D8ArtifactError("trial acceptance domain roster changed")
    canonical: dict[str, dict[str, object]] = {}
    accepted_total = 0
    proposed_total = 0
    for domain in inner_domains:
        item = acceptance[domain]
        if not isinstance(item, dict) or set(item) != {
            "accepted_proposals",
            "proposed_variants",
            "acceptance_rate",
        }:
            raise D8ArtifactError("trial acceptance evidence schema changed")
        accepted = _integer(item["accepted_proposals"], "domain accepted proposals")
        proposed = _integer(
            item["proposed_variants"], "domain proposed variants", minimum=1
        )
        if accepted > proposed:
            raise D8ArtifactError("trial acceptance counts are invalid")
        rate = accepted / proposed
        if not _close(
            _float(item["acceptance_rate"], "domain acceptance rate"), rate
        ):
            raise D8ArtifactError("trial domain acceptance rate changed")
        canonical[domain] = {
            "accepted_proposals": accepted,
            "proposed_variants": proposed,
            "acceptance_rate": rate,
        }
        accepted_total += accepted
        proposed_total += proposed
    overall = accepted_total / proposed_total
    if (
        _integer(row["accepted_proposals"], "accepted proposals") != accepted_total
        or _integer(row["proposed_variants"], "proposed variants", minimum=1)
        != proposed_total
        or not _close(_float(row["acceptance_rate"], "acceptance rate"), overall)
    ):
        raise D8ArtifactError("trial aggregate acceptance evidence changed")
    eligible = overall >= 0.80 and all(
        item["acceptance_rate"] >= 0.60 for item in canonical.values()
    )
    if eligible is not require_eligible:
        raise D8ArtifactError("trial morphology acceptance state changed")
    return accepted_total, proposed_total, overall, canonical


def _validate_trial_rows(
    rows: tuple[dict[str, str], ...], config: D8Config
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    by_outer: dict[str, list[dict[str, object]]] = {domain: [] for domain in DOMAIN_ORDER}
    canonical_rows: list[dict[str, object]] = []
    identities: set[tuple[str, int]] = set()
    states = {"COMPLETE", "PRUNED", "FAIL"}
    for row in rows:
        outer = row["outer_fold"]
        if outer not in by_outer or row["study_name"] != f"d8::{outer}":
            raise D8ArtifactError("trial study or outer fold changed")
        trial_id = _integer(row["trial_id"], "trial id")
        identity = (row["study_name"], trial_id)
        if identity in identities:
            raise D8ArtifactError("duplicate D8 trial identity")
        identities.add(identity)
        state = row["state"]
        if state not in states:
            raise D8ArtifactError("trial state changed")
        if row["config_sha256"] != config.config_sha256:
            raise D8ArtifactError("trial config hash changed")
        for field in ("search_view_sha256", "candidate_sha256", "evidence_sha256"):
            if state == "COMPLETE" and not _valid_sha256(row[field]):
                raise D8ArtifactError(f"trial {field} changed")
        record: dict[str, object] = dict(row)
        inner_domains = tuple(domain for domain in DOMAIN_ORDER if domain != outer)
        if state == "COMPLETE":
            values: list[float] = []
            for domain in DOMAIN_ORDER:
                cell = row[f"inner_mae__{domain}"]
                if domain == outer:
                    if cell != "":
                        raise D8ArtifactError("trial accessed the outer domain")
                else:
                    values.append(_float(cell, "inner MAE", minimum=0.0))
            array = np.asarray(values, dtype=np.float64)
            mean = math.fsum(values) / len(inner_domains)
            worst = float(np.max(array))
            deviation = float(np.std(array))
            objective = robust_inner_objective(array)
            comparisons = (
                (row["mean_mae"], mean, "mean MAE"),
                (row["worst_mae"], worst, "worst MAE"),
                (row["domain_sd"], deviation, "domain SD"),
                (row["objective"], objective, "objective"),
            )
            for encoded, expected, label in comparisons:
                if not _close(_float(encoded, label, minimum=0.0), expected):
                    raise D8ArtifactError(f"trial {label} differs from inner scores")
            accepted_total, proposed_total, rate, canonical_acceptance = (
                _validate_acceptance_evidence(
                    row, inner_domains=inner_domains, require_eligible=True
                )
            )
            record["inner_mae"] = dict(zip(inner_domains, values, strict=True))
            record["acceptance_by_domain"] = canonical_acceptance
            record["accepted_proposals"] = accepted_total
            record["proposed_variants"] = proposed_total
            record["acceptance_rate"] = rate
            record["mean_mae"] = mean
            record["worst_mae"] = worst
            record["domain_sd"] = deviation
            record["objective"] = objective
        else:
            if not row["failure_reason"]:
                raise D8ArtifactError("non-complete trial lacks a failure reason")
            if row["failure_reason"].startswith("morphology_acceptance:"):
                accepted_total, proposed_total, rate, canonical_acceptance = (
                    _validate_acceptance_evidence(
                        row, inner_domains=inner_domains, require_eligible=False
                    )
                )
                record["acceptance_by_domain"] = canonical_acceptance
                record["accepted_proposals"] = accepted_total
                record["proposed_variants"] = proposed_total
                record["acceptance_rate"] = rate
        record.pop("runtime_seconds", None)
        by_outer[outer].append(record)
        canonical_rows.append(record)
    minimum_per_outer = config.forced_trials + config.optuna_trials
    if any(len(by_outer[domain]) < minimum_per_outer for domain in DOMAIN_ORDER):
        raise D8ArtifactError("registered D8 trial budget is incomplete")
    return by_outer, canonical_rows


def _validate_database(
    path: Path, by_outer: Mapping[str, Sequence[Mapping[str, object]]]
) -> None:
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    try:
        database = sqlite3.connect(uri, uri=True)
        try:
            integrity = database.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise D8ArtifactError("D8 study database integrity check failed")
            rows = database.execute(
                "SELECT s.study_name, t.number, t.state "
                "FROM studies AS s JOIN trials AS t ON t.study_id=s.study_id"
            ).fetchall()
        finally:
            database.close()
    except (sqlite3.Error, OSError) as error:
        raise D8ArtifactError("D8 study database is invalid") from error
    observed = {(str(study), int(number), str(state)) for study, number, state in rows}
    expected = {
        (str(row["study_name"]), int(row["trial_id"]), str(row["state"]))
        for values in by_outer.values()
        for row in values
    }
    if len(rows) != len(expected) or observed != expected:
        raise D8ArtifactError("D8 study database differs from trial index")


def _validate_summaries(
    rows: tuple[dict[str, str], ...],
    by_outer: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    if tuple(row["outer_domain"] for row in rows) != DOMAIN_ORDER:
        raise D8ArtifactError("search summary outer-domain order changed")
    canonical: list[dict[str, object]] = []
    for row in rows:
        outer = row["outer_domain"]
        trials = tuple(by_outer[outer])
        counts = {
            state: sum(item["state"] == state for item in trials)
            for state in ("COMPLETE", "PRUNED", "FAIL")
        }
        initial = _integer(row["initial_trial_count"], "initial trial count", minimum=1)
        if initial != 72:
            raise D8ArtifactError("registered initial trial count changed")
        expected_numbers = {
            "trial_count": len(trials),
            "completed_count": counts["COMPLETE"],
            "pruned_count": counts["PRUNED"],
            "failed_count": counts["FAIL"],
        }
        for field, expected in expected_numbers.items():
            if _integer(row[field], field) != expected:
                raise D8ArtifactError(f"search summary {field} differs from trials")
        completed = tuple(item for item in trials if item["state"] == "COMPLETE")
        if len(completed) < 12:
            raise D8ArtifactError("search summary has fewer than twelve complete trials")
        best = min(
            completed,
            key=lambda item: (float(item["objective"]), str(item["candidate_sha256"])),
        )
        if (
            not _close(_float(row["best_objective"], "best objective"), float(best["objective"]))
            or row["best_candidate_sha256"] != best["candidate_sha256"]
            or not _valid_sha256(row["selection_state_sha256"])
            or not _valid_sha256(row["escalation_evidence_sha256"])
        ):
            raise D8ArtifactError("search summary selection differs from trial evidence")
        canonical.append(
            {
                **row,
                "initial_trial_count": initial,
                **expected_numbers,
                "best_objective": float(best["objective"]),
            }
        )
    return canonical


def _p1_selection_authority(
    project_root: Path, config: D8Config
) -> dict[str, tuple[str, float]]:
    source = config.sources.get("p1_predictions")
    if type(source) is not SourceRecord:
        raise D8ArtifactError("P1 prediction authority is unavailable")
    rows = _read_csv(project_root / source.path, _P1_PREDICTION_FIELDS)
    selected = tuple(
        row for row in rows if row["method"] == "A_surface" and row["seed"] == "0"
    )
    if (
        len(selected) != 276
        or len({row["specimen_id"] for row in selected}) != 276
        or tuple(dict.fromkeys(row["dataset_id"] for row in selected)) != DOMAIN_ORDER
    ):
        raise D8ArtifactError("P1 prediction authority roster changed")
    return {
        row["specimen_id"]: (
            row["dataset_id"],
            _float(row["target"], "P1 target"),
        )
        for row in selected
    }


def _validate_selections(
    payload: object,
    summaries: Sequence[Mapping[str, object]],
    by_outer: Mapping[str, Sequence[Mapping[str, object]]],
    config: D8Config,
    project_root: Path,
) -> tuple[int, list[dict[str, object]]]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "scope",
        "config_sha256",
        "outer_evaluation_count",
        "selections",
        "state_sha256",
    }:
        raise D8ArtifactError("selected configuration schema changed")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 2
        or payload["scope"] != "d8_prospective_outer_selections"
        or payload["config_sha256"] != config.config_sha256
    ):
        raise D8ArtifactError("selected configuration authority changed")
    count = _integer(payload["outer_evaluation_count"], "outer evaluation count")
    if count != 0:
        raise D8ArtifactError("pilot package contains an outer evaluation")
    selections = payload["selections"]
    if not isinstance(selections, list) or len(selections) != len(DOMAIN_ORDER):
        raise D8ArtifactError("selected configuration fold set changed")
    summary_by_outer = {str(row["outer_domain"]): row for row in summaries}
    p1_authority = _p1_selection_authority(project_root, config)
    canonical: list[dict[str, object]] = []
    for expected_outer, item in zip(DOMAIN_ORDER, selections, strict=True):
        if not isinstance(item, dict):
            raise D8ArtifactError("selected configuration record schema changed")
        summary = summary_by_outer[expected_outer]
        trials = tuple(by_outer[expected_outer])
        completed = tuple(row for row in trials if row["state"] == "COMPLETE")
        top = tuple(
            sorted(
                completed,
                key=lambda row: (
                    float(row["objective"]),
                    str(row["candidate_sha256"]),
                ),
            )[:12]
        )
        try:
            rerank = item["rerank"]
            if not isinstance(rerank, Mapping) or not isinstance(
                rerank.get("rows"), list
            ):
                raise TypeError("rerank rows are unavailable")
            candidates = tuple(
                D8Candidate.from_payload(row["candidate"])
                for row in rerank["rows"]
                if isinstance(row, Mapping)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise D8ArtifactError("selected rerank candidates are invalid") from error
        if len(candidates) != 12 or {
            candidate.state_sha256 for candidate in candidates
        } != {str(row["candidate_sha256"]) for row in top}:
            raise D8ArtifactError("selected rerank candidates differ from search top twelve")
        search_states = {str(row["search_view_sha256"]) for row in trials}
        if len(search_states) != 1:
            raise D8ArtifactError("trial search authority changed within an outer fold")
        try:
            search_authority = item["search_authority"]
            if not isinstance(search_authority, Mapping):
                raise TypeError("search authority is unavailable")
            specimen_ids = tuple(search_authority["specimen_ids"])
            expected_ids = {
                specimen_id
                for specimen_id, (domain, _) in p1_authority.items()
                if domain != expected_outer
            }
            if len(specimen_ids) != len(expected_ids) or set(specimen_ids) != expected_ids:
                raise ValueError("search specimen roster changed")
            domain_ids = tuple(p1_authority[specimen_id][0] for specimen_id in specimen_ids)
            targets = np.asarray(
                [p1_authority[specimen_id][1] for specimen_id in specimen_ids],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise D8ArtifactError("selected search authority differs from P1") from error
        try:
            validated = validate_selection_evidence(
                item,
                outer_domain=expected_outer,
                config_sha256=config.config_sha256,
                search_view_sha256=next(iter(search_states)),
                specimen_ids=specimen_ids,
                domain_ids=domain_ids,
                targets=targets,
                search_candidates=candidates,
            )
        except (TypeError, ValueError) as error:
            raise D8ArtifactError("selected configuration evidence is invalid") from error
        if validated.state_sha256 != summary["selection_state_sha256"]:
            raise D8ArtifactError("selected configuration differs from search evidence")
        canonical.append(dict(item))
    if payload["state_sha256"] != _sha_bytes(_canonical(canonical)):
        raise D8ArtifactError("selected configuration state hash changed")
    return count, canonical


def _validate_residual_manifest(payload: object) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "scope",
        "specimen_count",
        "draw_count",
        "record_count",
        "maximum_mean_error",
        "maximum_variance_error",
        "state_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise D8ArtifactError("residual bank manifest schema changed")
    if (
        payload["schema_version"] != 1
        or payload["scope"] != "d8_cross_fitted_p6_residual_bank"
        or _integer(payload["specimen_count"], "residual specimen count") != 276
        or _integer(payload["draw_count"], "residual draw count") != 8
        or _integer(payload["record_count"], "residual record count") != 2208
        or _float(payload["maximum_mean_error"], "residual mean error", minimum=0.0)
        > 1.0e-6
        or _float(payload["maximum_variance_error"], "residual variance error", minimum=0.0)
        > 1.0e-6
        or not _valid_sha256(payload["state_sha256"])
    ):
        raise D8ArtifactError("residual bank manifest differs from registered authority")
    return dict(payload)


def _selection_diffusion_weight(selection: Mapping[str, object]) -> float:
    candidates_value = selection.get("selected_candidates")
    ensemble = selection.get("ensemble")
    if not isinstance(candidates_value, list) or not isinstance(ensemble, Mapping):
        raise D8ArtifactError("escalation selection evidence is incomplete")
    try:
        candidates = tuple(D8Candidate.from_payload(value) for value in candidates_value)
    except (TypeError, ValueError) as error:
        raise D8ArtifactError("escalation selection candidate changed") from error
    if ensemble.get("candidate_sha256") != [
        candidate.state_sha256 for candidate in candidates
    ]:
        raise D8ArtifactError("escalation ensemble candidate roster changed")
    weights_value = ensemble.get("weights")
    if not isinstance(weights_value, list):
        raise D8ArtifactError("escalation ensemble weights are invalid")
    weights = np.asarray(weights_value, dtype=np.float64)
    if (
        weights.shape != (len(candidates),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not _close(float(np.sum(weights)), 1.0)
    ):
        raise D8ArtifactError("escalation ensemble weights are invalid")
    return math.fsum(
        float(weight)
        for candidate, weight in zip(candidates, weights, strict=True)
        if candidate.control_id in _DIFFUSION_CONTROLS
    )


def _validate_escalation_evidence(
    payload: object,
    *,
    summaries: Sequence[Mapping[str, object]],
    by_outer: Mapping[str, Sequence[Mapping[str, object]]],
    selections: Sequence[Mapping[str, object]],
    residual: Mapping[str, object],
    config: D8Config,
) -> dict[str, object]:
    root_keys = {
        "schema_version",
        "scope",
        "config_sha256",
        "residual_bank_sha256",
        "studies",
        "trend_outer_studies",
        "mismatch_outer_studies",
        "freeze_outer_studies",
        "decision",
        "state_sha256",
    }
    study_keys = {
        "outer_domain",
        "baseline_candidate_sha256",
        "diffusion_candidate_sha256",
        "baseline_objective",
        "diffusion_objective",
        "improved_inner_domains",
        "low_band_energy_fraction",
        "maximum_alpha_point_one_acceptance",
        "selected_diffusion_weight",
        "selection_state_sha256",
        "residual_bank_sha256",
        "state_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != root_keys
        or payload["schema_version"] != 1
        or payload["scope"] != "d8_pilot_escalation_evidence"
        or payload["config_sha256"] != config.config_sha256
        or payload["residual_bank_sha256"] != residual["state_sha256"]
    ):
        raise D8ArtifactError("escalation evidence authority changed")
    study_payloads = payload["studies"]
    if not isinstance(study_payloads, list) or len(study_payloads) != len(DOMAIN_ORDER):
        raise D8ArtifactError("escalation study roster changed")
    studies: list[D8PilotStudyEvidence] = []
    for expected_outer, value in zip(DOMAIN_ORDER, study_payloads, strict=True):
        if not isinstance(value, dict) or set(value) != study_keys:
            raise D8ArtifactError("escalation study schema changed")
        item = dict(value)
        state = item.pop("state_sha256")
        improved = item.get("improved_inner_domains")
        if not isinstance(improved, list):
            raise D8ArtifactError("escalation improved-domain roster changed")
        item["improved_inner_domains"] = tuple(improved)
        try:
            study = D8PilotStudyEvidence(**item)
        except (TypeError, ValueError) as error:
            raise D8ArtifactError("escalation study evidence is invalid") from error
        if study.outer_domain != expected_outer or study.state_sha256 != state:
            raise D8ArtifactError("escalation study state changed")
        studies.append(study)
    try:
        decision = decide_pilot_escalation(tuple(studies), config=config)
    except (TypeError, ValueError) as error:
        raise D8ArtifactError("escalation decision is invalid") from error
    if payload != decision.to_payload():
        raise D8ArtifactError("escalation aggregate decision changed")

    summary_by_outer = {str(row["outer_domain"]): row for row in summaries}
    selection_by_outer = {
        str(selection["outer_domain"]): selection for selection in selections
    }
    improvement_threshold = float(
        config.escalation["p6_candidate_minimum_inner_mae_improvement"]
    )
    alpha = float(config.escalation["low_acceptance_alpha"])
    for study in studies:
        outer = study.outer_domain
        if (
            summary_by_outer[outer]["escalation_evidence_sha256"]
            != study.state_sha256
            or selection_by_outer[outer]["state_sha256"]
            != study.selection_state_sha256
            or not _close(
                _selection_diffusion_weight(selection_by_outer[outer]),
                study.selected_diffusion_weight,
            )
        ):
            raise D8ArtifactError("escalation selection binding changed")
        trials = tuple(by_outer[outer])
        complete = tuple(row for row in trials if row["state"] == "COMPLETE")
        baseline_rows = tuple(row for row in complete if row["control_id"] == "B0")
        diffusion_rows = tuple(
            row for row in complete if row["control_id"] in _DIFFUSION_CONTROLS
        )
        if not baseline_rows or not diffusion_rows:
            raise D8ArtifactError("escalation B0 or diffusion trial is missing")
        baseline = min(
            baseline_rows,
            key=lambda row: (float(row["objective"]), str(row["candidate_sha256"])),
        )
        diffusion = min(
            diffusion_rows,
            key=lambda row: (float(row["objective"]), str(row["candidate_sha256"])),
        )
        if (
            baseline["candidate_sha256"] != study.baseline_candidate_sha256
            or diffusion["candidate_sha256"] != study.diffusion_candidate_sha256
            or not _close(float(baseline["objective"]), study.baseline_objective)
            or not _close(float(diffusion["objective"]), study.diffusion_objective)
        ):
            raise D8ArtifactError("escalation best-trial binding changed")
        inner_domains = tuple(domain for domain in DOMAIN_ORDER if domain != outer)
        improved = tuple(
            domain
            for domain in inner_domains
            if float(baseline["inner_mae"][domain]) > 0.0
            and float(baseline["inner_mae"][domain])
            - float(diffusion["inner_mae"][domain])
            >= improvement_threshold * float(baseline["inner_mae"][domain])
        )
        if improved != study.improved_inner_domains:
            raise D8ArtifactError("escalation improved-domain evidence changed")
        alpha_rows = tuple(
            row
            for row in trials
            if row["control_id"] in _DIFFUSION_CONTROLS
            and row["state"] in {"COMPLETE", "PRUNED"}
            and row.get("acceptance_rate") not in {None, ""}
            and _close(_float(row["alpha"], "escalation alpha"), alpha)
        )
        if not alpha_rows:
            raise D8ArtifactError("escalation alpha acceptance evidence is missing")
        maximum_acceptance = max(float(row["acceptance_rate"]) for row in alpha_rows)
        if not _close(
            maximum_acceptance, study.maximum_alpha_point_one_acceptance
        ):
            raise D8ArtifactError("escalation acceptance evidence changed")
    return decision.to_payload()


def _validate_semantics(
    root: Path, config: D8Config, project_root: Path
) -> _SemanticEvidence:
    trial_rows = _read_csv(root / "trial_index.csv", TRIAL_INDEX_FIELDS)
    by_outer, canonical_trials = _validate_trial_rows(trial_rows, config)
    _validate_database(root / "study.db", by_outer)
    summary_rows = _read_csv(root / "search_summary.csv", _SUMMARY_FIELDS)
    summaries = _validate_summaries(summary_rows, by_outer)
    selections_payload = _load_json(root / "selected_configs.json", "selected configs")
    outer_count, selections = _validate_selections(
        selections_payload,
        summaries,
        by_outer,
        config,
        project_root,
    )
    residual = _validate_residual_manifest(
        _load_json(root / "residual_bank_manifest.json", "residual bank manifest")
    )
    escalation = _validate_escalation_evidence(
        _load_json(root / "escalation_evidence.json", "escalation evidence"),
        summaries=summaries,
        by_outer=by_outer,
        selections=selections,
        residual=residual,
        config=config,
    )
    try:
        report = _read_regular(root / "pilot_report.md", "pilot report").decode("utf-8")
    except UnicodeError as error:
        raise D8ArtifactError("pilot report is not UTF-8") from error
    decisions = tuple(status for status in _DECISIONS if f"`{status}`" in report)
    if len(decisions) != 1:
        raise D8ArtifactError("pilot report decision is missing or ambiguous")
    status = decisions[0]
    if status != escalation["decision"]:
        raise D8ArtifactError("pilot report differs from escalation evidence")
    digest = _sha_bytes(
        _canonical(
            {
                "schema_version": 1,
                "scope": "d8_pilot_scientific_evidence",
                "config_sha256": config.config_sha256,
                "trials": canonical_trials,
                "summaries": summaries,
                "selections": selections,
                "residual_bank": residual,
                "escalation": escalation,
                "outer_evaluation_count": outer_count,
            }
        )
    )
    return _SemanticEvidence(
        initial_trial_count=sum(int(row["initial_trial_count"]) for row in summaries),
        trial_count=sum(int(row["trial_count"]) for row in summaries),
        outer_evaluation_count=outer_count,
        escalation_status=status,
        scientific_digest=digest,
    )


def _checksum_payload(root: Path) -> bytes:
    names = (*sorted(_SOURCE_FILES), "artifact_manifest.json")
    return "".join(
        f"{_sha_bytes(_read_regular(root / name, f'D8 checksum source {name}'))}  {name}\n"
        for name in names
    ).encode("ascii")


def _validate_checksums(root: Path) -> None:
    expected = _checksum_payload(root)
    actual = _read_regular(root / "CHECKSUMS.sha256", "D8 checksums")
    if actual != expected:
        raise D8ArtifactError("D8 checksum hash differs from package bytes")


def build_d8_search_package(
    output_dir: str | Path,
    *,
    source_dir: str | Path,
    project_root: str | Path,
    config_path: str | Path,
    escalation_status: str,
) -> D8ValidatedSearchPackage:
    """Build and independently validate one D8 pilot package."""

    if escalation_status not in _DECISIONS:
        raise D8ArtifactError("D8 escalation status is invalid")
    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_d8_config(config_file, project_root=root)
    source = Path(source_dir)
    _regular_file_set(source, _SOURCE_FILES, "D8 package source")
    output = Path(output_dir)
    if output.exists():
        raise D8ArtifactError("D8 package target already exists")
    output.mkdir(parents=True)
    try:
        for name in sorted(_SOURCE_FILES):
            payload = _read_regular(source / name, f"D8 package source {name}")
            (output / name).write_bytes(payload)
        evidence = _validate_semantics(output, config, root)
        if evidence.escalation_status != escalation_status:
            raise D8ArtifactError("D8 escalation status differs from pilot report")
        records = _output_records(output)
        manifest = {
            "schema_version": 1,
            "scope": "cpb_d8_pilot_search_package",
            "outer_domains": list(DOMAIN_ORDER),
            "initial_trial_count": evidence.initial_trial_count,
            "trial_count": evidence.trial_count,
            "outer_evaluation_count": evidence.outer_evaluation_count,
            "escalation_status": evidence.escalation_status,
            "config_sha256": config.config_sha256,
            "source_authorities": _source_authorities(config),
            "outputs": records,
            "output_tree_sha256": _tree_sha256(records),
            "scientific_digest": evidence.scientific_digest,
        }
        _write_json(output / "artifact_manifest.json", manifest)
        (output / "CHECKSUMS.sha256").write_bytes(_checksum_payload(output))
        return validate_d8_search_package(
            output, project_root=root, config_path=config_file
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def validate_d8_search_package(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> D8ValidatedSearchPackage:
    """Reload and independently recompute one published D8 pilot package."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    config = load_d8_config(config_file, project_root=root)
    output = Path(output_dir)
    _regular_file_set(output, _ROOT_FILES, "D8 search package")
    _validate_checksums(output)
    manifest = _load_json(output / "artifact_manifest.json", "D8 artifact manifest")
    expected_keys = {
        "schema_version",
        "scope",
        "outer_domains",
        "initial_trial_count",
        "trial_count",
        "outer_evaluation_count",
        "escalation_status",
        "config_sha256",
        "source_authorities",
        "outputs",
        "output_tree_sha256",
        "scientific_digest",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise D8ArtifactError("D8 artifact manifest schema changed")
    records = _output_records(output)
    if (
        manifest["schema_version"] != 1
        or manifest["scope"] != "cpb_d8_pilot_search_package"
        or manifest["outer_domains"] != list(DOMAIN_ORDER)
        or manifest["config_sha256"] != config.config_sha256
        or manifest["source_authorities"] != _source_authorities(config)
        or manifest["outputs"] != records
        or manifest["output_tree_sha256"] != _tree_sha256(records)
    ):
        raise D8ArtifactError("D8 artifact manifest authority or hash changed")
    evidence = _validate_semantics(output, config, root)
    comparisons = {
        "initial_trial_count": evidence.initial_trial_count,
        "trial_count": evidence.trial_count,
        "outer_evaluation_count": evidence.outer_evaluation_count,
        "escalation_status": evidence.escalation_status,
        "scientific_digest": evidence.scientific_digest,
    }
    if any(manifest[key] != value for key, value in comparisons.items()):
        raise D8ArtifactError("D8 manifest differs from recomputed scientific evidence")
    return D8ValidatedSearchPackage(
        outer_domains=DOMAIN_ORDER,
        initial_trial_count=evidence.initial_trial_count,
        trial_count=evidence.trial_count,
        outer_evaluation_count=evidence.outer_evaluation_count,
        escalation_status=evidence.escalation_status,
        scientific_digest=evidence.scientific_digest,
        output_tree_sha256=str(manifest["output_tree_sha256"]),
        artifact_manifest_sha256=_sha_bytes(
            _read_regular(output / "artifact_manifest.json", "D8 artifact manifest")
        ),
        required_files=_ROOT_FILES,
    )


def _atomic_replace(source: Path, target: Path) -> None:
    source.replace(target)


@contextmanager
def _publication_lock(output: Path):
    lock = output.parent / f".{output.name}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise D8ArtifactError("D8 publication lock is not regular")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise D8ArtifactError("D8 publication is already active") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _owner_payload(output: Path, transaction_uuid: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "cpb_d8_publication_transaction",
        "output": str(output.resolve()),
        "transaction_uuid": transaction_uuid,
    }


def _owned_transaction(transaction: Path, output: Path) -> bool:
    try:
        owner = _load_json(transaction / "transaction_owner.json", "D8 transaction owner")
    except D8ArtifactError:
        return False
    return owner == _owner_payload(output, transaction.name.rsplit("-", 1)[-1])


def _transactions(output: Path) -> tuple[Path, ...]:
    candidates = tuple(sorted(output.parent.glob(f".{output.name}.transaction-*")))
    if any(
        not item.is_dir() or item.is_symlink() or not _owned_transaction(item, output)
        for item in candidates
    ):
        raise D8ArtifactError("foreign D8 publication transaction exists")
    if len(candidates) > 1:
        raise D8ArtifactError("multiple D8 publication transactions exist")
    return candidates


def _recover_unlocked(
    output: Path, *, project_root: Path, config_path: Path
) -> D8ValidatedSearchPackage:
    transactions = _transactions(output)
    if not transactions:
        return validate_d8_search_package(
            output, project_root=project_root, config_path=config_path
        )
    transaction = transactions[0]
    previous = transaction / "previous"
    staged = transaction / "staged"
    if output.exists():
        result = validate_d8_search_package(
            output, project_root=project_root, config_path=config_path
        )
        shutil.rmtree(transaction)
        return result
    candidate = previous if previous.exists() else staged if staged.exists() else None
    if candidate is None:
        raise D8ArtifactError("interrupted D8 publication has no recoverable package")
    validate_d8_search_package(
        candidate, project_root=project_root, config_path=config_path
    )
    _atomic_replace(candidate, output)
    result = validate_d8_search_package(
        output, project_root=project_root, config_path=config_path
    )
    shutil.rmtree(transaction)
    return result


def recover_interrupted_d8_publication(
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> D8ValidatedSearchPackage:
    """Recover one owned interrupted D8 publication transaction."""

    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(output):
        return _recover_unlocked(output, project_root=root, config_path=config_file)


def _commit_staged(
    staged: Path,
    output: Path,
    transaction: Path,
    *,
    project_root: Path,
    config_path: Path,
) -> D8ValidatedSearchPackage:
    previous = transaction / "previous"
    invalid = transaction / "invalid-output"
    moved_previous = False
    committed = False
    rollback_succeeded = False
    try:
        if output.exists():
            validate_d8_search_package(
                output, project_root=project_root, config_path=config_path
            )
            _atomic_replace(output, previous)
            moved_previous = True
        _atomic_replace(staged, output)
        result = validate_d8_search_package(
            output, project_root=project_root, config_path=config_path
        )
        committed = True
        return result
    except Exception:
        if output.exists():
            _atomic_replace(output, invalid)
        if moved_previous and previous.exists():
            _atomic_replace(previous, output)
            validate_d8_search_package(
                output, project_root=project_root, config_path=config_path
            )
            rollback_succeeded = True
        raise
    finally:
        if committed:
            if previous.exists():
                shutil.rmtree(previous)
            if invalid.exists():
                shutil.rmtree(invalid)
            if transaction.exists():
                shutil.rmtree(transaction)
        elif rollback_succeeded:
            if invalid.exists():
                shutil.rmtree(invalid)
            if transaction.exists():
                shutil.rmtree(transaction)


def _publish_built_package(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
    escalation_status: str,
) -> D8ValidatedSearchPackage:
    root = Path(project_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(output):
        transactions = _transactions(output)
        if transactions or output.exists():
            if transactions:
                _recover_unlocked(output, project_root=root, config_path=config_file)
            else:
                validate_d8_search_package(
                    output, project_root=root, config_path=config_file
                )
        transaction_uuid = uuid4().hex
        temporary = Path(tempfile.mkdtemp(prefix=".d8-allocation-", dir=output.parent))
        transaction = output.parent / f".{output.name}.transaction-{transaction_uuid}"
        _atomic_replace(temporary, transaction)
        _write_json(
            transaction / "transaction_owner.json",
            _owner_payload(output, transaction_uuid),
        )
        staged = transaction / "staged"
        try:
            build_d8_search_package(
                staged,
                source_dir=source_dir,
                project_root=root,
                config_path=config_file,
                escalation_status=escalation_status,
            )
            return _commit_staged(
                staged,
                output,
                transaction,
                project_root=root,
                config_path=config_file,
            )
        except Exception:
            if transaction.exists() and not (transaction / "previous").exists():
                shutil.rmtree(transaction)
            raise


def publish_d8_search_package(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
    escalation_status: str,
) -> D8ValidatedSearchPackage:
    """Publish a D8 package only to the registered results/d8_search leaf."""

    root = Path(project_root).resolve(strict=True)
    config = load_d8_config(config_path, project_root=root)
    output = Path(output_dir).resolve()
    registered = (root / config.output_dir).resolve()
    if output != registered or output.name != "d8_search" or output.parent == root:
        raise D8ArtifactError("D8 publication target is not the registered safe leaf")
    return _publish_built_package(
        source_dir,
        output,
        project_root=root,
        config_path=config_path,
        escalation_status=escalation_status,
    )


__all__ = [
    "D8ArtifactError",
    "D8ValidatedSearchPackage",
    "build_d8_search_package",
    "publish_d8_search_package",
    "recover_interrupted_d8_publication",
    "validate_d8_search_package",
]
