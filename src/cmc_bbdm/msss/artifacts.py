"""Atomic, checksum-bound publication of S1 MSSS evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .figures import MSSSFigureError, render_s1_figures
from .protocol import MSSSProtocol
from .s1 import S1Run
from .scale_features import ScaleFeatureBank


class S1ArtifactError(ValueError):
    """Raised when an S1 package is incomplete or has lost integrity."""


SCIENTIFIC_CSV_FILES = (
    "sampling_curve.csv",
    "gaussian_curve.csv",
    "wavelet_curve.csv",
    "domain_scale_metrics.csv",
    "spatial_specificity.csv",
    "msss_selection.csv",
    "selection_stability.csv",
    "candidate_predictions.csv",
    "selected_predictions.csv",
    "spatial_predictions.csv",
    "inner_scores.csv",
    "candidate_selection.csv",
)

MANDATORY_S1_FILES = frozenset(
    {
        *SCIENTIFIC_CSV_FILES,
        "summary.json",
        "REPORT.md",
        "feature_index.json",
        "sampling_features.npz",
        "figure_a_sampling_scale.pdf",
        "figure_a_sampling_scale.png",
        "figure_b_gaussian_scale.pdf",
        "figure_b_gaussian_scale.png",
        "figure_c_wavelet_scale.pdf",
        "figure_c_wavelet_scale.png",
        "figure_d_sufficiency_specificity.pdf",
        "figure_d_sufficiency_specificity.png",
        "figure_manifest.json",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
        "config.yaml",
    }
)

_CURVE_FIELDS = (
    "condition_id",
    "value",
    "coarse_rank",
    "primary_eligible",
    "wavelet",
    "level",
    "mode",
    "normalized_retention_index",
    "equal_domain_mae",
    "ci_low",
    "ci_high",
    "full_equal_domain_mae",
    "relative_gap",
    "noninferior_025",
    "noninferior_05",
    "noninferior_075",
)
_DOMAIN_FIELDS = ("axis", "condition_id", "dataset_id", "specimen_count", "mae")
_SPATIAL_FIELDS = (
    "axis",
    "dataset_id",
    "regular_mae",
    "shuffled_mae",
    "ssg",
    "positive_domains",
    "gate_status",
    "ci_low",
    "ci_high",
    "simultaneous_low",
    "simultaneous_high",
)
_SELECTION_FIELDS = (
    "axis",
    "scope",
    "outer_group",
    "selected_condition_id",
    "full_condition_id",
    "over_coarse_condition_id",
    "boundary_confirmed",
    "sufficient_025_json",
    "sufficient_05_json",
    "sufficient_075_json",
    "candidate_scores_json",
)
_STABILITY_FIELDS = (
    "axis",
    "passed",
    "maximum_in_window",
    "window_json",
    "selected_counts_json",
)
_CANDIDATE_PREDICTION_FIELDS = (
    "axis",
    "condition_id",
    "specimen_id",
    "dataset_id",
    "outer_group",
    "target",
    "prediction",
    "absolute_error",
    "selected_pca_dimension",
    "fit_state_sha256",
)
_SELECTED_PREDICTION_FIELDS = (
    "axis",
    "selected_condition_id",
    "specimen_id",
    "dataset_id",
    "outer_group",
    "target",
    "prediction",
    "absolute_error",
    "selected_pca_dimension",
    "fit_state_sha256",
)
_SPATIAL_PREDICTION_FIELDS = (
    "axis",
    "base_condition_id",
    "seed",
    "specimen_id",
    "dataset_id",
    "target",
    "regular_prediction",
    "shuffled_prediction",
    "regular_absolute_error",
    "shuffled_absolute_error",
    "selected_pca_dimension",
)
_INNER_SCORE_FIELDS = (
    "axis",
    "outer_group",
    "inner_group",
    "condition_id",
    "pca_dimension",
    "mae",
    "fit_count",
    "query_count",
    "pca_state_sha256",
    "model_state_sha256",
)
_CANDIDATE_SELECTION_FIELDS = (
    "axis",
    "outer_group",
    "condition_id",
    "selected_pca_dimension",
    "source_equal_group_mae",
    "dimension_scores_json",
)


@dataclass(frozen=True, slots=True)
class S1PackageValidation:
    gate_status: str
    test_only: bool
    scientific_digest: str
    output_tree_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cell(value: object) -> object:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != set(fields):
                raise S1ArtifactError(f"CSV schema mismatch: {path.name}")
            writer.writerow({key: _cell(row[key]) for key in fields})


def _curve_rows(run: S1Run, axis: str) -> list[dict[str, object]]:
    return [
        {key: getattr(metric, key) for key in _CURVE_FIELDS}
        for metric in run.curves
        if metric.axis == axis
    ]


def _spatial_rows(
    protocol: MSSSProtocol, run: S1Run
) -> list[dict[str, object]]:
    summaries = {item.axis: item for item in run.axis_summaries}
    rows: list[dict[str, object]] = []
    for evaluation in run.specificity_evaluations:
        summary = summaries[evaluation.axis]
        for index, dataset in enumerate(protocol.domain_order):
            rows.append(
                {
                    "axis": evaluation.axis,
                    "dataset_id": dataset,
                    "regular_mae": evaluation.regular_domain_mae[index],
                    "shuffled_mae": evaluation.shuffled_domain_mae[index],
                    "ssg": evaluation.result.domain_effects[index],
                    "positive_domains": evaluation.result.positive_domains,
                    "gate_status": evaluation.result.status,
                    "ci_low": "",
                    "ci_high": "",
                    "simultaneous_low": "",
                    "simultaneous_high": "",
                }
            )
        rows.append(
            {
                "axis": evaluation.axis,
                "dataset_id": "EQUAL_DOMAIN",
                "regular_mae": float(np.mean(evaluation.regular_domain_mae)),
                "shuffled_mae": float(np.mean(evaluation.shuffled_domain_mae)),
                "ssg": evaluation.result.estimate,
                "positive_domains": evaluation.result.positive_domains,
                "gate_status": evaluation.result.status,
                "ci_low": summary.specificity_interval.low,
                "ci_high": summary.specificity_interval.high,
                "simultaneous_low": summary.specificity_simultaneous_interval.low,
                "simultaneous_high": summary.specificity_simultaneous_interval.high,
            }
        )
    return rows


def _sufficient(selection: object, margin: float) -> tuple[str, ...]:
    entries = dict(selection.sufficient_sets)
    return tuple(entries.get(margin, ()))


def _selection_rows(run: S1Run) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for evaluation in run.evaluations:
        for selection in evaluation.scale_selections:
            rows.append(
                {
                    "axis": evaluation.axis,
                    "scope": "source_outer_fold",
                    "outer_group": selection.outer_group,
                    "selected_condition_id": selection.selected_condition_id,
                    "full_condition_id": selection.full_condition_id,
                    "over_coarse_condition_id": selection.over_coarse_condition_id,
                    "boundary_confirmed": selection.boundary_confirmed,
                    "sufficient_025_json": _json(_sufficient(selection, 0.025)),
                    "sufficient_05_json": _json(_sufficient(selection, 0.05)),
                    "sufficient_075_json": _json(_sufficient(selection, 0.075)),
                    "candidate_scores_json": _json(selection.candidate_scores),
                }
            )
    for summary in run.axis_summaries:
        ni = summary.global_noninferiority
        rows.append(
            {
                "axis": summary.axis,
                "scope": "global_descriptive",
                "outer_group": "",
                "selected_condition_id": summary.global_selected_condition_id,
                "full_condition_id": summary.full_condition_id,
                "over_coarse_condition_id": summary.global_over_coarse_condition_id,
                "boundary_confirmed": ni.boundary_confirmed,
                "sufficient_025_json": "[]",
                "sufficient_05_json": _json(ni.sufficient_candidates),
                "sufficient_075_json": "[]",
                "candidate_scores_json": _json(tuple(zip(ni.candidates, ni.scores, strict=True))),
            }
        )
    return rows


def _prediction_rows(run: S1Run, selected: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for evaluation in run.evaluations:
        values = evaluation.selected_predictions if selected else evaluation.candidate_predictions
        for item in values:
            row = asdict(item)
            row["axis"] = evaluation.axis
            rows.append(row)
    return rows


def _write_feature_archive(root: Path, bank: ScaleFeatureBank) -> None:
    conditions = tuple(item for item in bank.conditions if item.axis == "sampling")
    archive: dict[str, np.ndarray] = {}
    index: dict[str, object] = {
        "schema_version": 1,
        "bank_state_sha256": bank.state_sha256,
        "specimen_ids": bank.specimen_ids,
        "dataset_ids": bank.dataset_ids,
        "encoder_provenance": bank.encoder_provenance,
        "conditions": {},
    }
    for position, condition in enumerate(conditions):
        key = f"condition_{position:03d}"
        archive[key] = np.asarray(bank.features[condition.condition_id], dtype="<f8")
        index["conditions"][condition.condition_id] = {
            "archive_key": key,
            "feature_sha256": bank.feature_sha256[condition.condition_id],
            "transform_state_sha256": bank.transform_state_sha256[condition.condition_id],
            "value": condition.value,
            "coarse_rank": condition.coarse_rank,
            "is_full_identity": condition.is_full_identity,
        }
    np.savez_compressed(root / "sampling_features.npz", **archive)
    (root / "feature_index.json").write_text(_json(index) + "\n", encoding="utf-8")


def _write_scientific_tables(
    root: Path, protocol: MSSSProtocol, bank: ScaleFeatureBank, run: S1Run
) -> None:
    for axis in ("sampling", "gaussian", "wavelet"):
        _write_csv(root / f"{axis}_curve.csv", _CURVE_FIELDS, _curve_rows(run, axis))
    _write_csv(
        root / "domain_scale_metrics.csv",
        _DOMAIN_FIELDS,
        (asdict(item) for item in run.domain_metrics),
    )
    _write_csv(root / "spatial_specificity.csv", _SPATIAL_FIELDS, _spatial_rows(protocol, run))
    _write_csv(root / "msss_selection.csv", _SELECTION_FIELDS, _selection_rows(run))
    _write_csv(
        root / "selection_stability.csv",
        _STABILITY_FIELDS,
        (
            {
                "axis": item.axis,
                "passed": item.stability.passed,
                "maximum_in_window": item.stability.maximum_in_window,
                "window_json": _json(item.stability.window),
                "selected_counts_json": _json(item.stability.selected_counts),
            }
            for item in run.axis_summaries
        ),
    )
    _write_csv(
        root / "candidate_predictions.csv",
        _CANDIDATE_PREDICTION_FIELDS,
        _prediction_rows(run, selected=False),
    )
    _write_csv(
        root / "selected_predictions.csv",
        _SELECTED_PREDICTION_FIELDS,
        _prediction_rows(run, selected=True),
    )
    _write_csv(
        root / "spatial_predictions.csv",
        _SPATIAL_PREDICTION_FIELDS,
        (
            asdict(row)
            for evaluation in run.specificity_evaluations
            for row in evaluation.predictions
        ),
    )
    _write_csv(
        root / "inner_scores.csv",
        _INNER_SCORE_FIELDS,
        (
            {"axis": evaluation.axis, **asdict(row)}
            for evaluation in run.evaluations
            for row in evaluation.inner_scores
        ),
    )
    _write_csv(
        root / "candidate_selection.csv",
        _CANDIDATE_SELECTION_FIELDS,
        (
            {
                "axis": evaluation.axis,
                "outer_group": row.outer_group,
                "condition_id": row.condition_id,
                "selected_pca_dimension": row.selected_pca_dimension,
                "source_equal_group_mae": row.source_equal_group_mae,
                "dimension_scores_json": _json(row.dimension_scores),
            }
            for evaluation in run.evaluations
            for row in evaluation.candidate_selections
        ),
    )
    _write_feature_archive(root, bank)


def _scientific_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for name in SCIENTIFIC_CSV_FILES:
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update((root / name).read_bytes())
        digest.update(b"\0")
    digest.update((root / "feature_index.json").read_bytes())
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    excluded = {"artifact_manifest.json", "CHECKSUMS.sha256"}
    for path in sorted(item for item in root.iterdir() if item.is_file() and item.name not in excluded):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _summary(protocol: MSSSProtocol, run: S1Run, *, mode: str, test_only: bool) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": "S1_SCALE_DISCOVERY",
        "mode": mode,
        "test_only": test_only,
        "gate_status": run.gate.status,
        "passing_axes": run.gate.passing_axes,
        "total_axes": run.gate.total_axes,
        "fourier_status": "NOT_RUN_NONBLOCKING",
        "protocol_sha256": protocol.config_sha256,
        "feature_bank_state_sha256": run.feature_bank_state_sha256,
        "run_state_sha256": run.state_sha256,
        "bootstrap": {
            "seed": run.curve_bootstrap.seed,
            "resamples": run.curve_bootstrap.resamples,
            "curve_draws_sha256": run.curve_bootstrap.draws_sha256,
            "specificity_draws_sha256": run.specificity_bootstrap.draws_sha256,
        },
        "axes": [
            {
                "axis": item.axis,
                "gate_status": item.gate.status,
                "full_condition_id": item.full_condition_id,
                "selected_condition_id": item.global_selected_condition_id,
                "over_coarse_condition_id": item.global_over_coarse_condition_id,
                "plateau": item.gate.plateau,
                "boundary_confirmed": item.gate.boundary_confirmed,
                "stable": item.gate.stable,
                "mechanically_sufficient": item.gate.mechanically_sufficient,
                "spatially_specific": item.gate.spatially_specific,
                "selected_equal_domain_mae": item.selected_equal_domain_mae,
                "full_equal_domain_mae": item.full_equal_domain_mae,
                "ssg": item.specificity.estimate,
                "ssg_positive_domains": item.specificity.positive_domains,
            }
            for item in run.axis_summaries
        ],
    }


def _write_report(root: Path, summary: Mapping[str, object]) -> None:
    axes = summary["axes"]
    lines = [
        "# MSSS S1 Scale Discovery Report",
        "",
        f"Decision: **{summary['gate_status']}** ({summary['passing_axes']}/{summary['total_axes']} axes passed).",
        "",
        "| Axis | Gate | Selected scale | FULL MAE | Selected MAE | SSG | Positive domains |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for item in axes:
        lines.append(
            "| {axis} | {gate_status} | `{selected_condition_id}` | {full_equal_domain_mae:.6f} | "
            "{selected_equal_domain_mae:.6f} | {ssg:.6f} | {ssg_positive_domains}/6 |".format(**item)
        )
    lines.extend(
        [
            "",
            "Scale selection used source domains only inside each outer fold. Global curves are descriptive.",
            "Fourier sensitivity: `NOT_RUN_NONBLOCKING`.",
            "This package is test-only and cannot support a scientific claim." if summary["test_only"] else "This package is the registered formal S1 result.",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(root: Path, protocol: MSSSProtocol, summary: Mapping[str, object]) -> dict[str, object]:
    scientific = _scientific_digest(root)
    tree = _tree_hash(root)
    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol.config_sha256,
        "source_authorities": {
            item.name: {"path": item.relative_path, "sha256": item.sha256}
            for item in protocol.sources
        },
        "gate_status": summary["gate_status"],
        "test_only": summary["test_only"],
        "scientific_digest": scientific,
        "output_tree_sha256": tree,
        "files": {
            path.name: _sha256(path)
            for path in sorted(root.iterdir())
            if path.is_file() and path.name not in {"artifact_manifest.json", "CHECKSUMS.sha256"}
        },
    }
    (root / "artifact_manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
    checksum_paths = tuple(
        path for path in sorted(root.iterdir()) if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    (root / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="ascii",
    )
    return manifest


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise S1ArtifactError(f"invalid JSON artifact: {path.name}") from error
    if type(value) is not dict:
        raise S1ArtifactError(f"invalid JSON artifact: {path.name}")
    return value


def validate_s1_package(
    output: str | Path, *, project_root: str | Path, config_path: str | Path
) -> S1PackageValidation:
    """Verify exact checksums, registered config, and package-level digests."""

    root = Path(output).resolve(strict=True)
    Path(project_root).resolve(strict=True)
    config = Path(config_path).resolve(strict=True)
    if not root.is_dir():
        raise S1ArtifactError("S1 package path is invalid")
    names = {item.name for item in root.iterdir() if item.is_file()}
    missing = MANDATORY_S1_FILES - names
    if missing:
        raise S1ArtifactError(f"mandatory S1 artifacts are missing: {sorted(missing)}")
    if (root / "config.yaml").read_bytes() != config.read_bytes():
        raise S1ArtifactError("packaged config differs from registered config")
    expected_checksums: dict[str, str] = {}
    try:
        for raw in (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines():
            digest, name = raw.split("  ", 1)
            if name in expected_checksums or "/" in name or "\\" in name:
                raise ValueError
            expected_checksums[name] = digest
    except (OSError, UnicodeError, ValueError) as error:
        raise S1ArtifactError("checksum registry is invalid") from error
    expected_names = names - {"CHECKSUMS.sha256"}
    if set(expected_checksums) != expected_names:
        raise S1ArtifactError("checksum registry does not cover the output tree")
    for name, expected in expected_checksums.items():
        if _sha256(root / name) != expected:
            raise S1ArtifactError(f"checksum mismatch: {name}")
    manifest = _read_json(root / "artifact_manifest.json")
    summary = _read_json(root / "summary.json")
    if manifest.get("protocol_sha256") != _sha256(config):
        raise S1ArtifactError("manifest protocol checksum mismatch")
    scientific = _scientific_digest(root)
    tree = _tree_hash(root)
    if manifest.get("scientific_digest") != scientific:
        raise S1ArtifactError("scientific digest mismatch")
    if manifest.get("output_tree_sha256") != tree:
        raise S1ArtifactError("output tree checksum mismatch")
    if (
        manifest.get("gate_status") != summary.get("gate_status")
        or manifest.get("test_only") != summary.get("test_only")
        or type(summary.get("test_only")) is not bool
    ):
        raise S1ArtifactError("summary and manifest disagree")
    return S1PackageValidation(
        gate_status=str(summary["gate_status"]),
        test_only=bool(summary["test_only"]),
        scientific_digest=scientific,
        output_tree_sha256=tree,
    )


def publish_s1_package(
    output: str | Path,
    *,
    protocol: MSSSProtocol,
    bank: ScaleFeatureBank,
    run: S1Run,
    config_path: str | Path,
    mode: str,
    test_only: bool,
) -> S1PackageValidation:
    """Publish a complete S1 package by one atomic directory rename."""

    if type(protocol) is not MSSSProtocol or type(bank) is not ScaleFeatureBank or type(run) is not S1Run:
        raise S1ArtifactError("issued protocol, bank, and S1 run are required")
    if type(mode) is not str or mode not in {"formal", "smoke"} or type(test_only) is not bool:
        raise S1ArtifactError("publication mode is invalid")
    if (mode == "formal") == test_only:
        raise S1ArtifactError("formal/test-only publication contract is inconsistent")
    destination = Path(output).resolve()
    if destination.exists():
        raise S1ArtifactError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        _write_scientific_tables(staging, protocol, bank, run)
        summary = _summary(protocol, run, mode=mode, test_only=test_only)
        (staging / "summary.json").write_text(_json(summary) + "\n", encoding="utf-8")
        _write_report(staging, summary)
        shutil.copyfile(Path(config_path).resolve(strict=True), staging / "config.yaml")
        render_s1_figures(staging)
        _write_manifest(staging, protocol, summary)
        validation = validate_s1_package(
            staging,
            project_root=Path(config_path).resolve(strict=True).parents[2],
            config_path=config_path,
        )
        os.replace(staging, destination)
        return validation
    except (S1ArtifactError, MSSSFigureError, OSError, ValueError) as error:
        shutil.rmtree(staging, ignore_errors=True)
        if isinstance(error, S1ArtifactError):
            raise
        raise S1ArtifactError("S1 package publication failed") from error


__all__ = [
    "MANDATORY_S1_FILES",
    "SCIENTIFIC_CSV_FILES",
    "S1ArtifactError",
    "S1PackageValidation",
    "publish_s1_package",
    "validate_s1_package",
]
