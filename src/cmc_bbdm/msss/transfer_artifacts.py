"""Atomic S2 transfer artifacts and prediction-level replay."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .artifacts import S1ArtifactError, validate_s1_package
from .protocol import MSSSProtocol
from .transfer_pipeline import S2Run


class S2ArtifactError(ValueError):
    """Raised when an S2 package is incomplete or inconsistent."""


_METRIC_FIELDS = (
    "task_family",
    "task_id",
    "target_label",
    "comparator",
    "condition_id",
    "specimen_count",
    "domain_count",
    "mae",
    "equal_domain_mae",
    "worst_domain_mae",
    "full_equal_domain_mae",
    "tg",
    "rtg",
    "nonworse",
    "ci_low",
    "ci_high",
)
_SELECTION_FIELDS = (
    "task_family",
    "task_id",
    "target_label",
    "source_domains_json",
    "target_domains_json",
    "full_condition_id",
    "fixed25_condition_id",
    "selected_condition_id",
    "over_coarse_condition_id",
    "boundary_confirmed",
    "sufficient_sets_json",
    "candidate_scores_json",
    "candidate_pca_dimensions_json",
)
_GROUP_FIELDS = (
    "task_family",
    "task_id",
    "target_label",
    "comparator",
    "condition_id",
    "dataset_id",
    "specimen_count",
    "mae",
)
_BOOTSTRAP_FIELDS = (
    "task_id",
    "comparator",
    "estimate",
    "ci_low",
    "ci_high",
    "seed",
    "resamples",
    "draws_sha256",
)
_PREDICTION_FIELDS = (
    "task_family",
    "task_id",
    "target_label",
    "comparator",
    "condition_id",
    "specimen_id",
    "dataset_id",
    "target",
    "prediction",
    "absolute_error",
    "selected_pca_dimension",
    "fit_state_sha256",
)
SCIENTIFIC_S2_FILES = (
    "six_domain_lodo.csv",
    "leave_ply.csv",
    "leave_layup.csv",
    "scale_selection.csv",
    "transfer_gain.csv",
    "group_metrics.csv",
    "bootstrap.csv",
    "transfer_predictions.csv",
)
MANDATORY_S2_FILES = frozenset(
    {
        *SCIENTIFIC_S2_FILES,
        "summary.json",
        "REPORT.md",
        "s1_authorization.json",
        "config.yaml",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
)


@dataclass(frozen=True, slots=True)
class S2PackageValidation:
    gate_status: str
    scientific_digest: str
    output_tree_sha256: str
    s1_scientific_digest: str


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cell(value: object) -> object:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise S2ArtifactError(f"CSV schema mismatch: {path.name}")
            writer.writerow({key: _cell(row[key]) for key in fields})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_rows(run: S2Run) -> list[dict[str, object]]:
    return [
        {
            "task_family": evaluation.task.family,
            "task_id": evaluation.task.task_id,
            "target_label": evaluation.task.target_label,
            "source_domains_json": _json(evaluation.task.source_domains),
            "target_domains_json": _json(evaluation.task.target_domains),
            "full_condition_id": evaluation.selection.full_condition_id,
            "fixed25_condition_id": evaluation.selection.fixed25_condition_id,
            "selected_condition_id": evaluation.selection.selected_condition_id,
            "over_coarse_condition_id": evaluation.selection.over_coarse_condition_id,
            "boundary_confirmed": evaluation.selection.boundary_confirmed,
            "sufficient_sets_json": _json(evaluation.selection.sufficient_sets),
            "candidate_scores_json": _json(evaluation.selection.candidate_scores),
            "candidate_pca_dimensions_json": _json(evaluation.selection.candidate_pca_dimensions),
        }
        for evaluation in run.evaluations
    ]


def _group_rows(run: S2Run) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for evaluation in run.evaluations:
        keys = tuple(
            dict.fromkeys((item.comparator, item.condition_id, item.dataset_id) for item in evaluation.predictions)
        )
        for comparator, condition, dataset in keys:
            values = tuple(
                item.absolute_error
                for item in evaluation.predictions
                if item.comparator == comparator and item.condition_id == condition and item.dataset_id == dataset
            )
            rows.append(
                {
                    "task_family": evaluation.task.family,
                    "task_id": evaluation.task.task_id,
                    "target_label": evaluation.task.target_label,
                    "comparator": comparator,
                    "condition_id": condition,
                    "dataset_id": dataset,
                    "specimen_count": len(values),
                    "mae": float(math.fsum(values) / len(values)),
                }
            )
    return rows


def _write_tables(root: Path, run: S2Run) -> None:
    metrics = [asdict(item) for item in run.metrics]
    _write_csv(root / "transfer_gain.csv", _METRIC_FIELDS, metrics)
    for family, filename in (
        ("domain", "six_domain_lodo.csv"),
        ("ply", "leave_ply.csv"),
        ("layup", "leave_layup.csv"),
    ):
        _write_csv(root / filename, _METRIC_FIELDS, (row for row in metrics if row["task_family"] == family))
    _write_csv(root / "scale_selection.csv", _SELECTION_FIELDS, _selection_rows(run))
    _write_csv(root / "group_metrics.csv", _GROUP_FIELDS, _group_rows(run))
    _write_csv(
        root / "bootstrap.csv",
        _BOOTSTRAP_FIELDS,
        (
            {
                "task_id": task_id,
                "comparator": comparator,
                "estimate": interval.estimate,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "seed": bootstrap.seed,
                "resamples": bootstrap.resamples,
                "draws_sha256": bootstrap.draws_sha256,
            }
            for task_id, bootstrap in run.bootstraps
            for comparator, interval in bootstrap.effects.items()
        ),
    )
    _write_csv(
        root / "transfer_predictions.csv",
        _PREDICTION_FIELDS,
        (asdict(item) for evaluation in run.evaluations for item in evaluation.predictions),
    )


def _scientific_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for name in SCIENTIFIC_S2_FILES:
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in {"artifact_manifest.json", "CHECKSUMS.sha256"}:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256(path).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _summary(protocol: MSSSProtocol, run: S2Run) -> dict[str, object]:
    source = tuple(item for item in run.metrics if item.comparator == "SOURCE_MSSS")
    return {
        "schema_version": 1,
        "stage": "S2_STRUCTURED_TRANSFER",
        "test_only": False,
        "gate_status": run.gate.status,
        "domain_support": run.gate.domain_support,
        "domain_nonworse": run.gate.domain_nonworse,
        "structured_nonworse": run.gate.structured_nonworse,
        "ply_positive": run.gate.ply_positive,
        "layup_positive": run.gate.layup_positive,
        "impact_shift_status": "NOT_RUN_INSUFFICIENT_FROZEN_AUDIT",
        "protocol_sha256": protocol.config_sha256,
        "s1_gate_status": run.authorization.gate_status,
        "s1_scientific_digest": run.authorization.scientific_digest,
        "run_state_sha256": run.state_sha256,
        "source_msss": [
            {
                "task_family": item.task_family,
                "task_id": item.task_id,
                "target_label": item.target_label,
                "condition_id": item.condition_id,
                "tg": item.tg,
                "rtg": item.rtg,
                "nonworse": item.nonworse,
            }
            for item in source
        ],
    }


def _report(root: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# MSSS S2 Structured Transfer Report",
        "",
        f"Decision: **{summary['gate_status']}**.",
        "",
        "| Task | Family | Selected condition | TG | RTG | Non-worse |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in summary["source_msss"]:
        lines.append(
            "| {task_id} | {task_family} | `{condition_id}` | {tg:.6f} | {rtg:.2%} | {nonworse} |".format(**item)
        )
    lines.extend(
        [
            "",
            f"Ordinary six-domain support: `{summary['domain_support']}` ({summary['domain_nonworse']}/6 non-worse).",
            f"Structured shifts: {summary['structured_nonworse']}/5 non-worse; positive ply {summary['ply_positive']}/3; positive layup {summary['layup_positive']}/2.",
            "Impact-condition shift: `NOT_RUN_INSUFFICIENT_FROZEN_AUDIT`.",
            "All scales and PCA dimensions were selected from source domains only.",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manifest(root: Path, protocol: MSSSProtocol, summary: Mapping[str, object]) -> None:
    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol.config_sha256,
        "gate_status": summary["gate_status"],
        "s1_scientific_digest": summary["s1_scientific_digest"],
        "scientific_digest": _scientific_digest(root),
        "output_tree_sha256": _tree_hash(root),
        "files": {
            path.name: _sha256(path)
            for path in sorted(root.iterdir())
            if path.is_file() and path.name not in {"artifact_manifest.json", "CHECKSUMS.sha256"}
        },
    }
    (root / "artifact_manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    paths = tuple(path for path in sorted(root.iterdir()) if path.is_file() and path.name != "CHECKSUMS.sha256")
    (root / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in paths), encoding="ascii"
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise S2ArtifactError(f"invalid JSON artifact: {path.name}") from error
    if type(value) is not dict:
        raise S2ArtifactError(f"invalid JSON artifact: {path.name}")
    return value


def validate_s2_package(
    output: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
    s1_package: str | Path,
) -> S2PackageValidation:
    root = Path(output).resolve(strict=True)
    if not root.is_dir():
        raise S2ArtifactError("S2 package path is invalid")
    names = {path.name for path in root.iterdir() if path.is_file()}
    missing = MANDATORY_S2_FILES - names
    if missing:
        raise S2ArtifactError(f"mandatory S2 artifacts are missing: {sorted(missing)}")
    config = Path(config_path).resolve(strict=True)
    if (root / "config.yaml").read_bytes() != config.read_bytes():
        raise S2ArtifactError("packaged S2 config changed")
    try:
        expected = {}
        for line in (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines():
            digest, name = line.split("  ", 1)
            if name in expected or "/" in name or "\\" in name:
                raise ValueError
            expected[name] = digest
    except (OSError, UnicodeError, ValueError) as error:
        raise S2ArtifactError("S2 checksum registry is invalid") from error
    if set(expected) != names - {"CHECKSUMS.sha256"}:
        raise S2ArtifactError("S2 checksums do not cover the output tree")
    for name, digest in expected.items():
        if _sha256(root / name) != digest:
            raise S2ArtifactError(f"S2 checksum mismatch: {name}")
    try:
        s1 = validate_s1_package(
            s1_package, project_root=project_root, config_path=config
        )
    except S1ArtifactError as error:
        raise S2ArtifactError("S2 authorization package is invalid") from error
    authorization = _read_json(root / "s1_authorization.json")
    summary = _read_json(root / "summary.json")
    manifest = _read_json(root / "artifact_manifest.json")
    if (
        s1.test_only
        or s1.gate_status not in {"GO", "STRONG_GO"}
        or authorization.get("scientific_digest") != s1.scientific_digest
        or authorization.get("output_tree_sha256") != s1.output_tree_sha256
        or summary.get("s1_scientific_digest") != s1.scientific_digest
        or manifest.get("s1_scientific_digest") != s1.scientific_digest
    ):
        raise S2ArtifactError("S2 authorization binding changed")
    scientific = _scientific_digest(root)
    tree = _tree_hash(root)
    if manifest.get("scientific_digest") != scientific or manifest.get("output_tree_sha256") != tree:
        raise S2ArtifactError("S2 manifest digest mismatch")
    if summary.get("gate_status") != manifest.get("gate_status"):
        raise S2ArtifactError("S2 summary and manifest disagree")
    return S2PackageValidation(
        gate_status=str(summary["gate_status"]),
        scientific_digest=scientific,
        output_tree_sha256=tree,
        s1_scientific_digest=s1.scientific_digest,
    )


def publish_s2_package(
    output: str | Path,
    *,
    protocol: MSSSProtocol,
    run: S2Run,
    config_path: str | Path,
    s1_package: str | Path,
) -> S2PackageValidation:
    if type(protocol) is not MSSSProtocol or type(run) is not S2Run:
        raise S2ArtifactError("issued protocol and S2 run are required")
    destination = Path(output).resolve()
    if destination.exists():
        raise S2ArtifactError(f"output already exists: {destination}")
    s1 = validate_s1_package(
        s1_package,
        project_root=Path(config_path).resolve(strict=True).parents[2],
        config_path=config_path,
    )
    if s1 != run.authorization or s1.test_only or s1.gate_status not in {"GO", "STRONG_GO"}:
        raise S2ArtifactError("S2 run lacks current formal S1 authorization")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        _write_tables(staging, run)
        summary = _summary(protocol, run)
        (staging / "summary.json").write_text(_json(summary) + "\n", encoding="utf-8")
        (staging / "s1_authorization.json").write_text(_json(asdict(s1)) + "\n", encoding="utf-8")
        _report(staging, summary)
        shutil.copyfile(Path(config_path).resolve(strict=True), staging / "config.yaml")
        _manifest(staging, protocol, summary)
        validation = validate_s2_package(
            staging,
            project_root=Path(config_path).resolve(strict=True).parents[2],
            config_path=config_path,
            s1_package=s1_package,
        )
        os.replace(staging, destination)
        return validation
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        if isinstance(error, S2ArtifactError):
            raise
        raise S2ArtifactError("S2 package publication failed") from error


def _rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            values = [dict(row) for row in csv.DictReader(handle, strict=True)]
    except (OSError, UnicodeError, csv.Error) as error:
        raise S2ArtifactError(f"S2 replay table is unreadable: {path.name}") from error
    if not values:
        raise S2ArtifactError(f"S2 replay table is empty: {path.name}")
    return values


def _number(value: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise S2ArtifactError("S2 replay numeric value is invalid") from error
    if not math.isfinite(output):
        raise S2ArtifactError("S2 replay numeric value is non-finite")
    return output


def _verify_predictions(root: Path) -> None:
    predictions = _rows(root / "transfer_predictions.csv")
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in predictions:
        grouped.setdefault((row["task_id"], row["comparator"], row["dataset_id"]), []).append(_number(row["absolute_error"]))
    group_rows = _rows(root / "group_metrics.csv")
    reported = {}
    for row in group_rows:
        key = (row["task_id"], row["comparator"], row["dataset_id"])
        if key in reported or key not in grouped:
            raise S2ArtifactError("S2 group roster does not reproduce")
        value = math.fsum(grouped[key]) / len(grouped[key])
        if not math.isclose(value, _number(row["mae"]), rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise S2ArtifactError("S2 group MAE does not reproduce")
        reported[key] = value
    if set(reported) != set(grouped):
        raise S2ArtifactError("S2 prediction roster is incomplete")
    by_task_comparator: dict[tuple[str, str], list[float]] = {}
    for (task, comparator, _dataset), value in reported.items():
        by_task_comparator.setdefault((task, comparator), []).append(value)
    for row in _rows(root / "transfer_gain.csv"):
        key = (row["task_id"], row["comparator"])
        values = by_task_comparator.get(key)
        full = by_task_comparator.get((row["task_id"], "FULL"))
        if not values or not full:
            raise S2ArtifactError("S2 transfer metric roster is incomplete")
        equal = math.fsum(values) / len(values)
        full_equal = math.fsum(full) / len(full)
        tg = full_equal - equal
        if not all(
            (
                math.isclose(equal, _number(row["equal_domain_mae"]), rel_tol=1.0e-12, abs_tol=1.0e-14),
                math.isclose(tg, _number(row["tg"]), rel_tol=1.0e-12, abs_tol=1.0e-14),
                math.isclose(tg / full_equal, _number(row["rtg"]), rel_tol=1.0e-12, abs_tol=1.0e-14),
            )
        ):
            raise S2ArtifactError("S2 transfer gain does not reproduce")


def replay_s2_package(
    source: str | Path,
    destination: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
    s1_package: str | Path,
) -> S2PackageValidation:
    source_root = Path(source).resolve(strict=True)
    output = Path(destination).resolve()
    if output.exists():
        raise S2ArtifactError(f"replay output already exists: {output}")
    source_validation = validate_s2_package(
        source_root,
        project_root=project_root,
        config_path=config_path,
        s1_package=s1_package,
    )
    _verify_predictions(source_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for path in sorted(source_root.iterdir()):
            if not path.is_file():
                raise S2ArtifactError("nested S2 replay content is not allowed")
            shutil.copy2(path, staging / path.name)
        replay_validation = validate_s2_package(
            staging,
            project_root=project_root,
            config_path=config_path,
            s1_package=s1_package,
        )
        if replay_validation != source_validation:
            raise S2ArtifactError("S2 replay package digest changed")
        os.replace(staging, output)
        return replay_validation
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "MANDATORY_S2_FILES",
    "S2ArtifactError",
    "S2PackageValidation",
    "publish_s2_package",
    "replay_s2_package",
    "validate_s2_package",
]
