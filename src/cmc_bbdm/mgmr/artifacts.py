"""Deterministic publication and independent validation for formal M0 evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .evaluation import PredictionRecord
from .formal_outer import M0FormalResult
from .protocol import MGMRProtocol, load_protocol
from .statistics import (
    MetricSummary,
    decide_m0,
    paired_domain_bootstrap,
    prediction_metrics,
)


class MGMRArtifactError(ValueError):
    """Raised when an M0 package is incomplete, inconsistent, or tampered."""


_PREDICTION_FIELDS = (
    "method",
    "specimen_id",
    "dataset_id",
    "target",
    "prediction",
    "dimensions",
)
_AGGREGATE_FIELDS = (
    "method",
    "specimen_count",
    "specimen_mae",
    "equal_domain_mae",
    "worst_domain_mae",
    "pearson",
    "spearman",
)
_DOMAIN_FIELDS = (
    "method",
    "domain",
    "specimen_count",
    "mae",
    "pearson",
    "spearman",
)
_BOOTSTRAP_FIELDS = (
    "effect",
    "estimate",
    "low",
    "high",
    "seed",
    "resamples",
    "draw_sha256",
)
_SOURCE_FIELDS = (
    "branch",
    "outer_domain",
    "specimen_id",
    "dataset_id",
    "target",
    "baseline_prediction",
    "residual",
    "baseline_fit_domains",
    "baseline_fit_specimen_ids",
)
_SCIENTIFIC_FILES = (
    "config.yaml",
    "predictions.csv",
    "aggregate_metrics.csv",
    "domain_metrics.csv",
    "bootstrap.csv",
    "source_residuals.csv",
    "summary.json",
    "REPORT.md",
)
_ALL_FILES = frozenset(
    (*_SCIENTIFIC_FILES, "artifact_manifest.json", "CHECKSUMS.sha256")
)


@dataclass(frozen=True, slots=True)
class M0PackageValidation:
    status: str
    scientific_digest: str
    output_tree_sha256: str
    file_sha256: Mapping[str, str]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _csv_bytes(fields: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(fields)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _effect_vectors(metrics: Mapping[str, MetricSummary], seeds: Sequence[int]):
    def effect(reference: str, candidate: str) -> tuple[float, ...]:
        left = metrics[reference].domain_metrics
        right = metrics[candidate].domain_metrics
        return tuple(
            first.mae - second.mae
            for first, second in zip(left, right, strict=True)
        )

    output = {
        "B1_minus_B3": effect("B1", "B3"),
        "B2_minus_B3": effect("B2", "B3"),
        "B0_minus_B4": effect("B0", "B4"),
        "B1_minus_R_coarse": effect("B1", "R_coarse"),
        "B0_minus_R_full": effect("B0", "R_full"),
    }
    for seed in seeds:
        name = f"P3_{seed}"
        output[f"B1_minus_{name}"] = effect("B1", name)
        output[f"real_minus_{name}"] = effect(name, "R_coarse")
    return output


def _report(protocol: MGMRProtocol, formal: M0FormalResult) -> bytes:
    lines = [
        "# MGMR M0 Component Gate",
        "",
        f"Status: `{formal.decision.status}`",
        "",
        "## Direct models",
        "",
        "| Method | Equal-domain MAE | Worst-domain MAE |",
        "|---|---:|---:|",
    ]
    for method in ("B0", "B1", "B2", "B3", "B4"):
        metric = formal.metrics[method]
        lines.append(
            f"| {method} | {metric.equal_domain_mae:.12f} | {metric.worst_domain_mae:.12f} |"
        )
    lines.extend(
        [
            "",
            "## Residual audit",
            "",
            "| Method | Equal-domain MAE | Residual Pearson | Residual Spearman |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, signal in (
        ("R_coarse", "S_R_coarse"),
        ("R_full", "S_R_full"),
        *((f"P3_{seed}", f"S_P3_{seed}") for seed in protocol.specificity_seeds),
    ):
        metric = formal.metrics[method]
        signal_metric = formal.metrics[signal]
        lines.append(
            f"| {method} | {metric.equal_domain_mae:.12f} | "
            f"{signal_metric.pearson:.12f} | {signal_metric.spearman:.12f} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            *(
                f"- Gate {name}: {'PASS' if passed else 'FAIL'}"
                for name, passed in formal.decision.gates.items()
            ),
            "",
            (
                "All six domains were exposed in earlier project phases; this is a "
                "registered post-hoc follow-up, not untouched external confirmation."
            ),
            "",
            (
                "M1 remains blocked by the frozen stop rule."
                if formal.decision.status == "MGMR_NO_GO"
                else "M1 requires a separate frozen design before implementation."
            ),
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _render(
    protocol: MGMRProtocol,
    formal: M0FormalResult,
    feature_manifest_sha256: str,
) -> dict[str, bytes]:
    if (
        type(protocol) is not MGMRProtocol
        or type(formal) is not M0FormalResult
        or type(feature_manifest_sha256) is not str
        or len(feature_manifest_sha256) != 64
    ):
        raise MGMRArtifactError("issued M0 package inputs are required")
    try:
        int(feature_manifest_sha256, 16)
    except ValueError as error:
        raise MGMRArtifactError("feature manifest SHA-256 is invalid") from error
    prediction_rows = [
        (
            row.method,
            row.specimen_id,
            row.dataset_id,
            repr(row.target),
            repr(row.prediction),
            json.dumps(row.dimensions, separators=(",", ":")),
        )
        for rows in formal.prediction_records.values()
        for row in rows
    ]
    aggregate_rows = [
        (
            method,
            metric.specimen_count,
            repr(metric.specimen_mae),
            repr(metric.equal_domain_mae),
            repr(metric.worst_domain_mae),
            repr(metric.pearson),
            repr(metric.spearman),
        )
        for method, metric in formal.metrics.items()
    ]
    domain_rows = [
        (
            method,
            row.domain,
            row.specimen_count,
            repr(row.mae),
            repr(row.pearson),
            repr(row.spearman),
        )
        for method, metric in formal.metrics.items()
        for row in metric.domain_metrics
    ]
    bootstrap_rows = [
        (
            effect,
            repr(interval.estimate),
            repr(interval.low),
            repr(interval.high),
            formal.bootstrap.seed,
            formal.bootstrap.resamples,
            formal.bootstrap.draw_sha256,
        )
        for effect, interval in formal.bootstrap.intervals.items()
    ]
    source_rows = [
        (
            row.branch,
            row.outer_domain,
            row.specimen_id,
            row.dataset_id,
            repr(row.target),
            repr(row.baseline_prediction),
            repr(row.residual),
            json.dumps(row.baseline_fit_domains, separators=(",", ":")),
            json.dumps(row.baseline_fit_specimen_ids, separators=(",", ":")),
        )
        for row in formal.source_residuals
    ]
    summary = {
        "schema_version": 1,
        "scope": "mgmr_m0_component_gate",
        "status": formal.decision.status,
        "gates": dict(formal.decision.gates),
        "required_gates": list(formal.decision.required_gates),
        "improved_domains": dict(formal.decision.improved_domains),
        "benefits": dict(formal.decision.benefits),
        "config_sha256": protocol.config_sha256,
        "source_sha256": {
            name: source.sha256 for name, source in protocol.sources.items()
        },
        "feature_manifest_sha256": feature_manifest_sha256,
        "formal_state_sha256": formal.state_sha256,
        "component_state_sha256": formal.component_state_sha256,
        "residual_state_sha256": formal.residual_state_sha256,
        "bootstrap_draw_sha256": formal.bootstrap.draw_sha256,
        "specimen_count": len(formal.specimen_ids),
        "method_order": list(formal.prediction_records),
        "source_residual_record_count": len(formal.source_residuals),
        "historical_outer_exposure": True,
        "external_confirmation": False,
        "m1_action": (
            "stop"
            if formal.decision.status == "MGMR_NO_GO"
            else "freeze_separate_design_before_m1"
        ),
    }
    files = {
        "config.yaml": protocol.config_path.read_bytes(),
        "predictions.csv": _csv_bytes(_PREDICTION_FIELDS, prediction_rows),
        "aggregate_metrics.csv": _csv_bytes(_AGGREGATE_FIELDS, aggregate_rows),
        "domain_metrics.csv": _csv_bytes(_DOMAIN_FIELDS, domain_rows),
        "bootstrap.csv": _csv_bytes(_BOOTSTRAP_FIELDS, bootstrap_rows),
        "source_residuals.csv": _csv_bytes(_SOURCE_FIELDS, source_rows),
        "summary.json": _json_bytes(summary),
        "REPORT.md": _report(protocol, formal),
    }
    manifest = {
        "schema_version": 1,
        "config_sha256": protocol.config_sha256,
        "files": {
            name: {"bytes": len(files[name]), "sha256": _sha(files[name])}
            for name in _SCIENTIFIC_FILES
        },
    }
    files["artifact_manifest.json"] = _json_bytes(manifest)
    checksums = "".join(
        f"{_sha(files[name])}  {name}\n"
        for name in sorted((*_SCIENTIFIC_FILES, "artifact_manifest.json"))
    )
    files["CHECKSUMS.sha256"] = checksums.encode("ascii")
    return files


def _atomic_publish(destination: Path, files: Mapping[str, bytes]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    backup: Path | None = None
    try:
        for name, payload in files.items():
            (staging / name).write_bytes(payload)
        if destination.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent)
            )
            backup.rmdir()
            os.replace(destination, backup)
        os.replace(staging, destination)
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise


def _digests(files: Mapping[str, bytes], status: str) -> M0PackageValidation:
    hashes = {name: _sha(payload) for name, payload in sorted(files.items())}
    scientific = hashlib.sha256()
    for name in _SCIENTIFIC_FILES:
        scientific.update(name.encode("ascii"))
        scientific.update(files[name])
    tree = hashlib.sha256()
    for name in sorted(files):
        tree.update(name.encode("ascii"))
        tree.update(files[name])
    return M0PackageValidation(
        status=status,
        scientific_digest=scientific.hexdigest(),
        output_tree_sha256=tree.hexdigest(),
        file_sha256=MappingProxyType(hashes),
    )


def publish_m0_package(
    output: str | Path,
    *,
    protocol: MGMRProtocol,
    formal: M0FormalResult,
    feature_manifest_sha256: str,
) -> M0PackageValidation:
    """Atomically publish a deterministic, checksum-bound formal package."""

    destination = Path(output).resolve()
    files = _render(protocol, formal, feature_manifest_sha256)
    _atomic_publish(destination, files)
    return validate_m0_package(
        destination,
        project_root=protocol.config_path.parents[2],
        config_path=protocol.config_path,
    )


def _read_csv(payload: bytes, fields: Sequence[str], label: str):
    try:
        reader = csv.DictReader(
            io.StringIO(payload.decode("utf-8"), newline=""), strict=True
        )
        if reader.fieldnames != list(fields):
            raise MGMRArtifactError(f"{label} schema changed")
        return tuple(dict(row) for row in reader)
    except (UnicodeError, csv.Error) as error:
        raise MGMRArtifactError(f"{label} cannot be decoded") from error


def _float(value: str, label: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise MGMRArtifactError(f"{label} is not numeric") from error
    if not __import__("math").isfinite(output):
        raise MGMRArtifactError(f"{label} is not finite")
    return output


def _exact_metric(observed: str, expected: float, label: str) -> None:
    if _float(observed, label) != expected:
        raise MGMRArtifactError(f"{label} does not recalculate from predictions")


def validate_m0_package(
    output: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
) -> M0PackageValidation:
    """Validate checksums, recalculate metrics/bootstrap/gates, and audit leakage."""

    root = Path(project_root).resolve(strict=True)
    protocol = load_protocol(config_path, project_root=root)
    package = Path(output).resolve(strict=True)
    if not package.is_dir() or package.is_symlink():
        raise MGMRArtifactError("M0 package must be a regular directory")
    names = frozenset(path.name for path in package.iterdir() if path.is_file())
    if names != _ALL_FILES or any(path.is_dir() for path in package.iterdir()):
        raise MGMRArtifactError("M0 package file roster changed")
    files = {name: (package / name).read_bytes() for name in _ALL_FILES}
    if files["config.yaml"] != protocol.config_path.read_bytes():
        raise MGMRArtifactError("package config bytes changed")
    try:
        checksum_rows = files["CHECKSUMS.sha256"].decode("ascii").splitlines()
        expected_checksums = {
            name: digest
            for digest, separator, name in (row.partition("  ") for row in checksum_rows)
            if separator
        }
    except UnicodeError as error:
        raise MGMRArtifactError("CHECKSUMS cannot be decoded") from error
    checksum_names = set(_SCIENTIFIC_FILES) | {"artifact_manifest.json"}
    if set(expected_checksums) != checksum_names:
        raise MGMRArtifactError("CHECKSUMS roster changed")
    for name, digest in expected_checksums.items():
        if digest != _sha(files[name]):
            raise MGMRArtifactError(f"{name} checksum changed")
    try:
        manifest = json.loads(files["artifact_manifest.json"])
        summary = json.loads(files["summary.json"])
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MGMRArtifactError("M0 JSON cannot be decoded") from error
    expected_manifest = {
        "schema_version": 1,
        "config_sha256": protocol.config_sha256,
        "files": {
            name: {"bytes": len(files[name]), "sha256": _sha(files[name])}
            for name in _SCIENTIFIC_FILES
        },
    }
    if manifest != expected_manifest:
        raise MGMRArtifactError("artifact manifest SHA-256 registry changed")
    for payload in files.values():
        if str(root).encode("utf-8") in payload:
            raise MGMRArtifactError("package contains an absolute project path")
    if (
        not isinstance(summary, dict)
        or summary.get("config_sha256") != protocol.config_sha256
        or summary.get("source_sha256")
        != {name: source.sha256 for name, source in protocol.sources.items()}
        or summary.get("specimen_count") != protocol.specimen_count
        or summary.get("historical_outer_exposure") is not True
        or summary.get("external_confirmation") is not False
    ):
        raise MGMRArtifactError("summary authority changed")

    prediction_rows = _read_csv(
        files["predictions.csv"], _PREDICTION_FIELDS, "predictions"
    )
    grouped: dict[str, list[PredictionRecord]] = {}
    for row in prediction_rows:
        try:
            raw_dimensions = json.loads(row["dimensions"])
            dimensions = tuple(raw_dimensions)
        except (json.JSONDecodeError, TypeError) as error:
            raise MGMRArtifactError("prediction dimensions are invalid") from error
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise MGMRArtifactError("prediction dimensions are invalid")
        record = PredictionRecord(
            method=row["method"],
            specimen_id=row["specimen_id"],
            dataset_id=row["dataset_id"],
            target=_float(row["target"], "prediction target"),
            prediction=_float(row["prediction"], "prediction"),
            dimensions=dimensions,
        )
        grouped.setdefault(record.method, []).append(record)
    required_methods = {
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "R_coarse",
        "R_full",
        *(f"P3_{seed}" for seed in protocol.specificity_seeds),
        "S_R_coarse",
        "S_R_full",
        *(f"S_P3_{seed}" for seed in protocol.specificity_seeds),
    }
    if set(grouped) != required_methods or any(
        len(rows) != protocol.specimen_count for rows in grouped.values()
    ):
        raise MGMRArtifactError("prediction method or specimen roster changed")
    metrics = {
        method: prediction_metrics(rows, domain_order=protocol.domain_order)
        for method, rows in grouped.items()
    }
    aggregate_rows = _read_csv(
        files["aggregate_metrics.csv"], _AGGREGATE_FIELDS, "aggregate metrics"
    )
    if tuple(row["method"] for row in aggregate_rows) != tuple(summary["method_order"]):
        raise MGMRArtifactError("aggregate metric order changed")
    for row in aggregate_rows:
        metric = metrics[row["method"]]
        if int(row["specimen_count"]) != metric.specimen_count:
            raise MGMRArtifactError("aggregate specimen count changed")
        for field in (
            "specimen_mae",
            "equal_domain_mae",
            "worst_domain_mae",
            "pearson",
            "spearman",
        ):
            _exact_metric(row[field], getattr(metric, field), f"aggregate {field}")
    domain_rows = _read_csv(
        files["domain_metrics.csv"], _DOMAIN_FIELDS, "domain metrics"
    )
    expected_domain = {
        (method, item.domain): item
        for method, metric in metrics.items()
        for item in metric.domain_metrics
    }
    if {(row["method"], row["domain"]) for row in domain_rows} != set(
        expected_domain
    ):
        raise MGMRArtifactError("domain metric roster changed")
    for row in domain_rows:
        metric = expected_domain[(row["method"], row["domain"])]
        if int(row["specimen_count"]) != metric.specimen_count:
            raise MGMRArtifactError("domain specimen count changed")
        for field in ("mae", "pearson", "spearman"):
            _exact_metric(row[field], getattr(metric, field), f"domain {field}")

    decision = decide_m0(
        direct={method: metrics[method] for method in ("B1", "B2", "B3")},
        coarse_baseline=metrics["B1"],
        coarse_corrected=metrics["R_coarse"],
        full_baseline=metrics["B0"],
        full_corrected=metrics["R_full"],
        shuffled={
            seed: metrics[f"P3_{seed}"] for seed in protocol.specificity_seeds
        },
        required_gates=protocol.gate_required,
        minimum_positive_domains=protocol.minimum_positive_domains,
    )
    if (
        summary.get("status") != decision.status
        or summary.get("gates") != dict(decision.gates)
        or summary.get("required_gates") != list(decision.required_gates)
        or summary.get("improved_domains") != dict(decision.improved_domains)
        or summary.get("benefits") != dict(decision.benefits)
    ):
        raise MGMRArtifactError("summary gate does not recalculate")
    bootstrap = paired_domain_bootstrap(
        _effect_vectors(metrics, protocol.specificity_seeds),
        domain_order=protocol.domain_order,
        seed=protocol.bootstrap_seed,
        resamples=protocol.bootstrap_resamples,
        quantiles=protocol.bootstrap_quantiles,
    )
    bootstrap_rows = _read_csv(
        files["bootstrap.csv"], _BOOTSTRAP_FIELDS, "bootstrap"
    )
    if tuple(row["effect"] for row in bootstrap_rows) != tuple(bootstrap.intervals):
        raise MGMRArtifactError("bootstrap effect order changed")
    for row in bootstrap_rows:
        interval = bootstrap.intervals[row["effect"]]
        if (
            int(row["seed"]) != bootstrap.seed
            or int(row["resamples"]) != bootstrap.resamples
            or row["draw_sha256"] != bootstrap.draw_sha256
        ):
            raise MGMRArtifactError("bootstrap authority changed")
        for field in ("estimate", "low", "high"):
            _exact_metric(row[field], getattr(interval, field), f"bootstrap {field}")
    if summary.get("bootstrap_draw_sha256") != bootstrap.draw_sha256:
        raise MGMRArtifactError("summary bootstrap draw changed")

    source_rows = _read_csv(
        files["source_residuals.csv"], _SOURCE_FIELDS, "source residuals"
    )
    if len(source_rows) != summary.get("source_residual_record_count"):
        raise MGMRArtifactError("source residual count changed")
    for row in source_rows:
        try:
            fit_domains = tuple(json.loads(row["baseline_fit_domains"]))
            fit_ids = tuple(json.loads(row["baseline_fit_specimen_ids"]))
        except (json.JSONDecodeError, TypeError) as error:
            raise MGMRArtifactError("source residual fit authority is invalid") from error
        if row["dataset_id"] in fit_domains or row["specimen_id"] in fit_ids:
            raise MGMRArtifactError("source residual baseline leaked query evidence")
        target = _float(row["target"], "source residual target")
        prediction = _float(row["baseline_prediction"], "source baseline prediction")
        residual_value = _float(row["residual"], "source residual")
        if residual_value != target - prediction:
            raise MGMRArtifactError("source residual does not recalculate")
    return _digests(files, decision.status)


__all__ = [
    "M0PackageValidation",
    "MGMRArtifactError",
    "publish_m0_package",
    "validate_m0_package",
]
