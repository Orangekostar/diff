from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np
import yaml

from cmc_bbdm.damage_response.artifacts import (
    ArtifactError,
    replay_p0,
    replay_p1,
    replay_p2,
    write_p0_package,
    write_p1_package,
)
from cmc_bbdm.damage_response.authority import AuthorityError, snapshot_file
from cmc_bbdm.damage_response.contracts import (
    DISPLACEMENT_MM_PER_VOLT,
    LOAD_KN_PER_VOLT,
    POST_CAI_IMAGE_INPUT_FORBIDDEN,
    PRIMARY_COUNTS,
    P0GateFacts,
    StageStatus,
    evaluate_p0_gate,
)
from cmc_bbdm.damage_response.feature_views import PRIMARY_TARGET_FIELDS
from cmc_bbdm.damage_response.p1_gate import (
    ENDPOINT_RANGE_TOLERANCES,
    EndpointGateFacts,
    evaluate_p1_gate,
)
from cmc_bbdm.damage_response.pairing import (
    FeatureIdentity,
    PairingError,
    TraceIdentity,
    load_feature_identities,
    pair_exact,
)
from cmc_bbdm.damage_response.post_cai import audit_post_cai_images
from cmc_bbdm.damage_response.raw_cai import (
    RawCaiError,
    RawCaiTrace,
    StrainUnitStatus,
    decode_raw_cai_csv,
)
from cmc_bbdm.damage_response.representative_pairs import (
    CurveRecord,
    select_representative_pairs,
)
from cmc_bbdm.damage_response.response_extraction import (
    ExtractedResponse,
    ExtractionError,
    extract_prepeak_response,
)
from cmc_bbdm.damage_response.sources import (
    OFFICIAL_FOLDER_COUNTS,
    OfficialFileRecord,
    SourceError,
    classify_spatial_expansion,
    load_official_inventory,
    load_spatial_pairs,
    read_lvi_observations,
    read_primary_design_metadata,
    read_published_peaks,
    read_specimen_sizes,
)
from cmc_bbdm.damage_response.targets import (
    PeakReconciliation,
    ResponseTrace,
    TargetError,
    convert_trace_to_response,
    derive_global_absolute_tolerance,
    reconcile_published_peak,
)

_EXPECTED_BASE_SHA = "3951f71f28b6efdf8c74eea0fe274b2a78a9cd57"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RAW_FILENAME_RE = re.compile(
    r"(?P<specimen>(?:[cq](?:8|16|24)|r(?:0|45))-\d+t?)\s+\d{4}\.CSV",
    re.IGNORECASE,
)
_REGISTERED_CHANNEL_NAMES = (
    "Extension",
    "Load",
    "Strain-FL",
    "Strain-FR",
    "Strain-BL",
    "Strain-BR",
)
_DOWNSTREAM_STAGES = ("P1", "P2", "P3", "P4", "P5")
_P0_CONFIG_RELATIVE_PATH = "paper_v3/configs/damage_to_failure_response.yaml"
_EXPECTED_P1_CONFIG = {
    "schema_version": 1,
    "base_sha": _EXPECTED_BASE_SHA,
    "p0": {
        "summary_relative_path": (
            "results/damage_to_failure_response/p0_data_audit/summary.json"
        ),
        "summary_sha256": (
            "9d44ead975119db2181a91efbf14b74165671a9d25b7b576d90f6e104757a633"
        ),
        "required_status": "P0_GO",
    },
    "extraction": {
        "baseline_samples": 50,
        "grid_points": 101,
        "minimum_unique_extension_positions": 50,
        "primary_scope": "pre_peak_inclusive",
        "duplicate_extension_aggregation": "median",
        "interpolation": "linear",
        "smoothing": "none",
        "clipping": "none",
    },
    "primary_endpoints": {
        "extension_peak_mm": {"range_tolerance": 0.001},
        "slope_u20_u60_mpa_per_mm": {
            "range_tolerance": 1.0,
            "fit_u_min": 0.2,
            "fit_u_max": 0.6,
        },
        "normalized_prepeak_auc": {"range_tolerance": 0.0001},
    },
    "redundancy_model": {
        "outer_split": "leave_one_domain_out",
        "strength_polynomial_degree": 3,
        "ridge_alpha": 0.000001,
        "hyperparameter_search": False,
    },
    "design_sensitivity": {
        "numeric_fields": ["ply_count", "width_mm", "thickness_mm"],
        "categorical_fields": {
            "laminate_type": ["cross_ply", "quasi_isotropic"],
            "impactor": [
                "coni120",
                "coni60",
                "flat",
                "hemia",
                "hemib",
                "hemic",
            ],
        },
        "privileged_only_fields": ["impactor"],
        "excluded_fields": [
            "height_mm",
            "impact_energy",
            "energy_per_thickness_j_per_mm",
            "scan_angle",
        ],
    },
    "gate": {
        "minimum_coverage_fraction": 0.9,
        "required_domain_count": 6,
        "near_deterministic_pooled_r2": 0.9,
        "near_deterministic_absolute_domain_spearman": 0.95,
        "nonredundant_domain_count": 4,
        "minimum_range_tolerance_multiplier": 10.0,
        "replay_required": True,
    },
    "representative_pairs": {
        "nearest_strength_neighbors_per_specimen": 1,
        "maximum_pairs": 12,
        "ranking": "normalized_curve_rms_desc_then_ids",
    },
}


class PipelineError(RuntimeError):
    """Raised when P0 orchestration cannot complete its audit package."""


@dataclass(frozen=True)
class SourceSpec:
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class P0Config:
    base_sha: str
    dataset_id: str
    dataset_version: int
    inventory: SourceSpec
    lvi_workbook: SourceSpec
    size_workbook: SourceSpec
    peak_workbook: SourceSpec
    raw_folder: str
    feature_bank: SourceSpec
    legacy_spatial_manifest: SourceSpec
    historical_sources: Mapping[str, str]
    primary_counts: Mapping[str, int]
    load_kn_per_volt: float
    displacement_mm_per_volt: float
    sampling_hz: float
    minimum_exact_pairs_per_domain: int
    maximum_missing_primary_channel_fraction: float


@dataclass(frozen=True)
class P0RunResult:
    status: StageStatus
    output: Path


@dataclass(frozen=True)
class P1Config:
    base_sha: str
    p0_summary_relative_path: str
    p0_summary_sha256: str
    required_p0_status: StageStatus
    baseline_samples: int
    grid_points: int
    minimum_unique_extension_positions: int
    ridge_alpha: float
    minimum_coverage_fraction: float
    near_deterministic_pooled_r2: float
    maximum_representative_pairs: int


@dataclass(frozen=True)
class P1RunResult:
    status: StageStatus
    output: Path
    decision_output: Path
    passing_endpoints: tuple[str, ...]


@dataclass(frozen=True)
class P2RunResult:
    status: StageStatus
    output: Path
    decision_output: Path
    passing_contrasts: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class P1PairAuthority:
    specimen_id: str
    domain_id: str
    raw_relative_path: str
    v3_relative_path: str
    raw_trace_sha256: str


def _mapping(value: object, *, label: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise PipelineError(f"config section {label} has schema drift")
    return value


def _relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PipelineError(f"config {label} must be a relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise PipelineError(f"config {label} must be a safe relative path")
    return path.as_posix()


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise PipelineError(f"config {label} must be a SHA-256")
    result = value.strip().casefold()
    if _SHA256_RE.fullmatch(result) is None:
        raise PipelineError(f"config {label} must be a SHA-256")
    return result


def _source_spec(value: object, *, label: str) -> SourceSpec:
    section = _mapping(
        value, label=label, keys={"relative_path", "sha256"}
    )
    return SourceSpec(
        relative_path=_relative_path(section["relative_path"], label=label),
        sha256=_digest(section["sha256"], label=label),
    )


def load_p0_config(path: Path) -> P0Config:
    """Load and validate the frozen P0 operator configuration."""

    try:
        payload = Path(path).read_text(encoding="utf-8")
        value = yaml.safe_load(payload)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise PipelineError("P0 config cannot be read") from error
    root = _mapping(
        value,
        label="root",
        keys={
            "schema_version",
            "base_sha",
            "dataset",
            "feature_bank",
            "legacy_spatial_manifest",
            "historical_sources",
            "primary_counts",
            "calibration",
            "review_thresholds",
            "input_boundary",
        },
    )
    if root["schema_version"] != 1 or root["base_sha"] != _EXPECTED_BASE_SHA:
        raise PipelineError("P0 config base/schema identity changed")
    dataset = _mapping(
        root["dataset"],
        label="dataset",
        keys={
            "id",
            "version",
            "inventory",
            "lvi_workbook",
            "size_workbook",
            "peak_workbook",
            "raw_folder",
        },
    )
    if dataset["id"] != "8scdmfdcfb" or dataset["version"] != 3:
        raise PipelineError("P0 dataset/version identity changed")
    if dataset["raw_folder"] != "4_Compression after impact testing raw data":
        raise PipelineError("P0 raw folder identity changed")

    historical = root["historical_sources"]
    if not isinstance(historical, dict) or not historical:
        raise PipelineError("P0 historical source registry is empty")
    normalized_historical = {
        _relative_path(name, label="historical source"): _digest(
            digest, label=f"historical source {name}"
        )
        for name, digest in historical.items()
    }
    primary = root["primary_counts"]
    if not isinstance(primary, dict) or dict(primary) != dict(PRIMARY_COUNTS):
        raise PipelineError("P0 primary domain counts changed")
    calibration = _mapping(
        root["calibration"],
        label="calibration",
        keys={"load_kn_per_volt", "displacement_mm_per_volt", "sampling_hz"},
    )
    if calibration != {
        "load_kn_per_volt": LOAD_KN_PER_VOLT,
        "displacement_mm_per_volt": DISPLACEMENT_MM_PER_VOLT,
        "sampling_hz": 50.0,
    }:
        raise PipelineError("P0 physical calibration changed")
    review = _mapping(
        root["review_thresholds"],
        label="review_thresholds",
        keys={
            "minimum_exact_pairs_per_domain",
            "maximum_missing_primary_channel_fraction",
        },
    )
    if review != {
        "minimum_exact_pairs_per_domain": 20,
        "maximum_missing_primary_channel_fraction": 0.2,
    }:
        raise PipelineError("P0 review thresholds changed")
    boundary = _mapping(
        root["input_boundary"],
        label="input_boundary",
        keys={
            "post_cai_image_input_forbidden",
            "true_cai_trace_input_forbidden",
            "true_peak_strength_input_forbidden",
        },
    )
    if boundary != {
        "post_cai_image_input_forbidden": True,
        "true_cai_trace_input_forbidden": True,
        "true_peak_strength_input_forbidden": True,
    }:
        raise PipelineError("P0 input boundary changed")

    return P0Config(
        base_sha=_EXPECTED_BASE_SHA,
        dataset_id="8scdmfdcfb",
        dataset_version=3,
        inventory=_source_spec(dataset["inventory"], label="inventory"),
        lvi_workbook=_source_spec(dataset["lvi_workbook"], label="lvi_workbook"),
        size_workbook=_source_spec(dataset["size_workbook"], label="size_workbook"),
        peak_workbook=_source_spec(dataset["peak_workbook"], label="peak_workbook"),
        raw_folder=str(dataset["raw_folder"]),
        feature_bank=_source_spec(root["feature_bank"], label="feature_bank"),
        legacy_spatial_manifest=_source_spec(
            root["legacy_spatial_manifest"], label="legacy_spatial_manifest"
        ),
        historical_sources=MappingProxyType(normalized_historical),
        primary_counts=MappingProxyType(dict(PRIMARY_COUNTS)),
        load_kn_per_volt=LOAD_KN_PER_VOLT,
        displacement_mm_per_volt=DISPLACEMENT_MM_PER_VOLT,
        sampling_hz=50.0,
        minimum_exact_pairs_per_domain=20,
        maximum_missing_primary_channel_fraction=0.2,
    )


def load_p1_config(path: Path) -> P1Config:
    """Load the exact preregistered P1 response-richness configuration."""

    try:
        payload = Path(path).read_text(encoding="utf-8")
        value = yaml.safe_load(payload)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise PipelineError("P1 config cannot be read") from error
    if value != _EXPECTED_P1_CONFIG:
        raise PipelineError("P1 config differs from the frozen contract")
    p0 = value["p0"]
    extraction = value["extraction"]
    redundancy = value["redundancy_model"]
    gate = value["gate"]
    pairs = value["representative_pairs"]
    return P1Config(
        base_sha=_EXPECTED_BASE_SHA,
        p0_summary_relative_path=_relative_path(
            p0["summary_relative_path"], label="P1 P0 summary"
        ),
        p0_summary_sha256=_digest(
            p0["summary_sha256"], label="P1 P0 summary"
        ),
        required_p0_status=StageStatus.P0_GO,
        baseline_samples=int(extraction["baseline_samples"]),
        grid_points=int(extraction["grid_points"]),
        minimum_unique_extension_positions=int(
            extraction["minimum_unique_extension_positions"]
        ),
        ridge_alpha=float(redundancy["ridge_alpha"]),
        minimum_coverage_fraction=float(gate["minimum_coverage_fraction"]),
        near_deterministic_pooled_r2=float(
            gate["near_deterministic_pooled_r2"]
        ),
        maximum_representative_pairs=int(pairs["maximum_pairs"]),
    )


def validate_p1_p0_authority(config: P1Config, repo_root: Path) -> Path:
    """Require the exact committed P0 package before any external P1 read."""

    repository = Path(repo_root)
    summary_path = repository / config.p0_summary_relative_path
    try:
        snapshot = snapshot_file(
            summary_path,
            max_bytes=16 * 1024 * 1024,
            logical_source="git:p0_data_audit",
            relative_path=config.p0_summary_relative_path,
        )
    except AuthorityError as error:
        raise PipelineError("P0 summary authority is unavailable") from error
    if snapshot.sha256 != config.p0_summary_sha256:
        raise PipelineError(
            "P0 summary SHA-256 mismatch: "
            f"expected {config.p0_summary_sha256}, observed {snapshot.sha256}"
        )
    try:
        payload = summary_path.read_bytes()
    except OSError as error:
        raise PipelineError("P0 summary cannot be read") from error
    if (
        len(payload) != snapshot.size
        or hashlib.sha256(payload).hexdigest() != snapshot.sha256
    ):
        raise PipelineError("P0 summary changed during P1 authorization")
    try:
        replay_p0(summary_path.parent)
    except ArtifactError as error:
        raise PipelineError(f"P0 package replay failed: {error}") from error
    try:
        summary = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PipelineError("P0 summary cannot be decoded") from error
    if (
        not isinstance(summary, dict)
        or summary.get("base_sha") != config.base_sha
        or not isinstance(summary.get("gate"), dict)
        or summary["gate"].get("status") != config.required_p0_status.value
        or not isinstance(summary.get("pairing"), dict)
        or summary["pairing"].get("primary_exact_pairs")
        != sum(PRIMARY_COUNTS.values())
        or summary["pairing"].get("per_domain") != dict(PRIMARY_COUNTS)
    ):
        raise PipelineError("P0 summary does not authorize P1")
    return summary_path.parent


def _regular_directory(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    try:
        info = candidate.lstat()
    except OSError as error:
        raise PipelineError(f"explicit external root is unavailable: {label}") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise PipelineError(f"explicit external root must be a directory: {label}")
    return candidate


def _canonical_raw_id(filename: str) -> str:
    match = _RAW_FILENAME_RE.fullmatch(filename)
    if match is None:
        raise PipelineError(f"official raw filename is not registered: {filename}")
    return match.group("specimen").casefold()


def _snapshot_row(
    path: Path,
    *,
    max_bytes: int,
    logical_source: str,
    relative_path: str,
    expected_sha256: str | None,
    expected_size: int | None = None,
) -> tuple[dict[str, object], bytes | None]:
    try:
        snapshot = snapshot_file(
            path,
            max_bytes=max_bytes,
            logical_source=logical_source,
            relative_path=relative_path,
        )
    except AuthorityError as error:
        raise PipelineError(str(error)) from error
    if expected_sha256 is not None and snapshot.sha256 != expected_sha256:
        raise PipelineError(f"source SHA-256 mismatch: {relative_path}")
    if expected_size is not None and snapshot.size != expected_size:
        raise PipelineError(f"source size mismatch: {relative_path}")
    row = {
        "logical_source": logical_source,
        "relative_path": relative_path,
        "size": snapshot.size,
        "sha256": snapshot.sha256,
        "expected_sha256": expected_sha256 or snapshot.sha256,
        "status": "HASH_MATCH" if expected_sha256 is not None else "OBSERVED",
    }
    return row, None


def _read_bound_raw(
    root: Path, record: OfficialFileRecord
) -> tuple[bytes, dict[str, object]]:
    path = root / record.folder / record.filename
    source_row, _ = _snapshot_row(
        path,
        max_bytes=max(1, record.size),
        logical_source="local:hasebe_v3_root",
        relative_path=record.relative_path,
        expected_sha256=record.sha256,
        expected_size=record.size,
    )
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PipelineError(f"raw trace cannot be read: {record.filename}") from error
    if len(payload) != record.size or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise PipelineError(f"raw trace changed during read: {record.filename}")
    return payload, source_row


def _finite_range(values: np.ndarray) -> tuple[float | str, float | str]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "", ""
    return float(np.min(finite)), float(np.max(finite))


def _csv_value(value: object) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    return value


def _csv_payload(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def _json_payload(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("ascii")


def _raw_qc_row(
    *,
    specimen_id: str,
    domain_id: str,
    primary: bool,
    record: OfficialFileRecord,
    trace: RawCaiTrace | None,
    error: str,
    workbook_crosscheck_count: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "specimen_id": specimen_id,
        "domain_id": domain_id,
        "primary": primary,
        "raw_relative_path": record.relative_path,
        "raw_sha256": record.sha256,
        "raw_size": record.size,
        "parse_status": "PASS" if trace is not None else "FAIL",
        "parse_error": error,
        "internal_title": trace.specimen_id if trace is not None else "",
        "title_matches_canonical": (
            trace.specimen_id == specimen_id if trace is not None else False
        ),
        "canonical_workbook_crosscheck_count": workbook_crosscheck_count,
        "n_rows": trace.n_rows if trace is not None else "",
        "sampling_hz": trace.sampling_hz if trace is not None else "",
        "columns": (
            "|".join(trace.original_channel_names) if trace is not None else ""
        ),
        "units": (
            "|".join(trace.original_channel_units) if trace is not None else ""
        ),
        "peak_row": trace.peak_row if trace is not None else "",
    }
    arrays = (
        trace.extension_volts if trace is not None else None,
        trace.load_volts if trace is not None else None,
        trace.strain_fl if trace is not None else None,
        trace.strain_fr if trace is not None else None,
        trace.strain_bl if trace is not None else None,
        trace.strain_br if trace is not None else None,
    )
    for name, values in zip(_REGISTERED_CHANNEL_NAMES, arrays, strict=True):
        key = name.casefold().replace("-", "_")
        if values is None:
            row[f"{key}_finite_fraction"] = ""
            row[f"{key}_min"] = ""
            row[f"{key}_max"] = ""
        else:
            low, high = _finite_range(values)
            row[f"{key}_finite_fraction"] = float(
                np.count_nonzero(np.isfinite(values)) / len(values)
            )
            row[f"{key}_min"] = low
            row[f"{key}_max"] = high
    return row


def _reconciliation_row(
    *,
    specimen_id: str,
    domain_id: str,
    primary: bool,
    response: ResponseTrace | None,
    reconciliation: PeakReconciliation | None,
    width_mm: float | str,
    thickness_mm: float | str,
    status: str,
    error: str,
) -> dict[str, object]:
    return {
        "specimen_id": specimen_id,
        "domain_id": domain_id,
        "primary": primary,
        "status": status,
        "error": error,
        "width_mm": width_mm,
        "thickness_mm": thickness_mm,
        "peak_row": response.peak_row if response is not None else "",
        "extension_at_peak_mm": (
            float(response.extension_mm[response.peak_row])
            if response is not None
            else ""
        ),
        "raw_peak_mpa": (
            reconciliation.raw_peak_mpa if reconciliation is not None else ""
        ),
        "published_peak_mpa": (
            reconciliation.published_peak_mpa if reconciliation is not None else ""
        ),
        "signed_error_mpa": (
            reconciliation.signed_error_mpa if reconciliation is not None else ""
        ),
        "absolute_error_mpa": (
            reconciliation.absolute_error_mpa if reconciliation is not None else ""
        ),
        "absolute_tolerance_mpa": (
            reconciliation.absolute_tolerance_mpa
            if reconciliation is not None
            else ""
        ),
        "passed": reconciliation.passed if reconciliation is not None else False,
    }


def _execute_p0(
    *, config_path: Path, config: P0Config, repo_root: Path, legacy_root: Path, v3_root: Path
) -> tuple[StageStatus, dict[str, bytes]]:
    source_rows: list[dict[str, object]] = []
    config_row, _ = _snapshot_row(
        config_path,
        max_bytes=1024 * 1024,
        logical_source="git:research_branch",
        relative_path="paper_v3/configs/damage_to_failure_response.yaml",
        expected_sha256=None,
    )
    source_rows.append(config_row)

    feature_path = repo_root / config.feature_bank.relative_path
    feature_identities = load_feature_identities(
        feature_path, expected_sha256=config.feature_bank.sha256
    )
    feature_row, _ = _snapshot_row(
        feature_path,
        max_bytes=1024 * 1024 * 1024,
        logical_source="git:frozen_feature_bank",
        relative_path=config.feature_bank.relative_path,
        expected_sha256=config.feature_bank.sha256,
    )
    source_rows.append(feature_row)

    inventory_path = legacy_root / config.inventory.relative_path
    inventory = load_official_inventory(
        inventory_path,
        expected_sha256=config.inventory.sha256,
        expected_folder_counts=OFFICIAL_FOLDER_COUNTS,
    )
    inventory_row, _ = _snapshot_row(
        inventory_path,
        max_bytes=32 * 1024 * 1024,
        logical_source="local:historical_full_tree",
        relative_path=config.inventory.relative_path,
        expected_sha256=config.inventory.sha256,
    )
    source_rows.append(inventory_row)

    workbook_specs = (
        config.lvi_workbook,
        config.size_workbook,
        config.peak_workbook,
    )
    for spec in workbook_specs:
        row, _ = _snapshot_row(
            v3_root / spec.relative_path,
            max_bytes=128 * 1024 * 1024,
            logical_source="local:hasebe_v3_root",
            relative_path=f"8scdmfdcfb/v3/{spec.relative_path}",
            expected_sha256=spec.sha256,
        )
        source_rows.append(row)

    sizes = read_specimen_sizes(
        v3_root / config.size_workbook.relative_path,
        expected_sha256=config.size_workbook.sha256,
    )
    published_peaks = read_published_peaks(
        v3_root / config.peak_workbook.relative_path,
        expected_sha256=config.peak_workbook.sha256,
    )
    lvi_observations = read_lvi_observations(
        v3_root / config.lvi_workbook.relative_path,
        expected_sha256=config.lvi_workbook.sha256,
    )
    tolerance = derive_global_absolute_tolerance(published_peaks.values())

    spatial_path = legacy_root / config.legacy_spatial_manifest.relative_path
    spatial_pairs = load_spatial_pairs(
        spatial_path, expected_sha256=config.legacy_spatial_manifest.sha256
    )
    spatial_row, _ = _snapshot_row(
        spatial_path,
        max_bytes=32 * 1024 * 1024,
        logical_source="local:historical_full_tree",
        relative_path=config.legacy_spatial_manifest.relative_path,
        expected_sha256=config.legacy_spatial_manifest.sha256,
    )
    source_rows.append(spatial_row)

    for relative_path, expected_hash in config.historical_sources.items():
        row, _ = _snapshot_row(
            legacy_root / relative_path,
            max_bytes=128 * 1024 * 1024,
            logical_source="local:historical_full_tree",
            relative_path=relative_path,
            expected_sha256=expected_hash,
        )
        row["status"] = "DISCOVERY_HASH_MATCH_NO_GIT_AUTHORITY"
        source_rows.append(row)

    raw_records = tuple(
        record for record in inventory if record.folder == config.raw_folder
    )
    image_records = tuple(
        record for record in inventory if record.folder == "3_Specimen image"
    )
    raw_by_id: dict[str, OfficialFileRecord] = {}
    for record in raw_records:
        specimen_id = _canonical_raw_id(record.filename)
        if specimen_id in raw_by_id:
            raise PipelineError(f"duplicate official raw specimen ID: {specimen_id}")
        raw_by_id[specimen_id] = record
    if len(raw_by_id) != 446:
        raise PipelineError(f"official raw identity count changed: {len(raw_by_id)}")

    primary_by_id = {
        item.specimen_id: item.domain_id for item in feature_identities
    }
    traces: dict[str, RawCaiTrace] = {}
    raw_errors: dict[str, str] = {}
    raw_qc_rows: list[dict[str, object]] = []
    for specimen_id in sorted(raw_by_id):
        record = raw_by_id[specimen_id]
        payload, source_row = _read_bound_raw(v3_root, record)
        source_rows.append(source_row)
        try:
            trace = decode_raw_cai_csv(payload)
        except RawCaiError as error:
            trace = None
            raw_errors[specimen_id] = str(error)
        else:
            traces[specimen_id] = trace
        crosscheck_count = sum(
            specimen_id in source
            for source in (sizes, published_peaks, lvi_observations)
        )
        raw_qc_rows.append(
            _raw_qc_row(
                specimen_id=specimen_id,
                domain_id=primary_by_id.get(specimen_id, ""),
                primary=specimen_id in primary_by_id,
                record=record,
                trace=trace,
                error=raw_errors.get(specimen_id, ""),
                workbook_crosscheck_count=crosscheck_count,
            )
        )

    trace_identities = tuple(
        TraceIdentity(
            specimen_id=item.specimen_id,
            domain_id=item.domain_id,
            raw_trace_sha256=raw_by_id[item.specimen_id].sha256,
        )
        for item in feature_identities
        if item.specimen_id in traces
    )
    exact_pairing_possible = len(trace_identities) == len(feature_identities)
    paired = ()
    if exact_pairing_possible:
        try:
            paired = pair_exact(feature_identities, trace_identities)
        except PairingError:
            exact_pairing_possible = False

    valid_pair_counts = Counter(
        item.domain_id
        for item in feature_identities
        if item.specimen_id in traces and item.specimen_id in raw_by_id
    )
    missing_channel_fractions: dict[str, float] = {}
    for domain in PRIMARY_COUNTS:
        members = [item for item in feature_identities if item.domain_id == domain]
        channel_missing_counts = {name: 0 for name in _REGISTERED_CHANNEL_NAMES}
        for item in members:
            trace = traces.get(item.specimen_id)
            if trace is None:
                for name in channel_missing_counts:
                    channel_missing_counts[name] += 1
                continue
            finite_counts = trace.finite_counts
            for name in channel_missing_counts:
                if finite_counts[name] < trace.n_rows:
                    channel_missing_counts[name] += 1
        missing_channel_fractions[domain] = max(channel_missing_counts.values()) / len(
            members
        )

    pairing_rows = [
        {
            "specimen_id": item.specimen_id,
            "domain_id": item.domain_id,
            "raw_relative_path": raw_by_id[item.specimen_id].relative_path,
            "raw_trace_sha256": item.raw_trace_sha256,
            "source_dataset_id": config.dataset_id,
            "source_dataset_version": config.dataset_version,
            "internal_title": traces[item.specimen_id].specimen_id,
            "title_matches_canonical": (
                traces[item.specimen_id].specimen_id == item.specimen_id
            ),
            "identity_rule": "OFFICIAL_FILENAME_PLUS_DOMAIN_PLUS_SHA256",
        }
        for item in paired
    ]

    responses: dict[str, ResponseTrace] = {}
    reconciliations: dict[str, PeakReconciliation] = {}
    reconciliation_rows: list[dict[str, object]] = []
    for specimen_id in sorted(raw_by_id):
        trace = traces.get(specimen_id)
        size = sizes.get(specimen_id)
        published = published_peaks.get(specimen_id)
        domain_id = primary_by_id.get(specimen_id, "")
        primary = specimen_id in primary_by_id
        if trace is None:
            reconciliation_rows.append(
                _reconciliation_row(
                    specimen_id=specimen_id,
                    domain_id=domain_id,
                    primary=primary,
                    response=None,
                    reconciliation=None,
                    width_mm="",
                    thickness_mm="",
                    status="RAW_TRACE_INVALID",
                    error=raw_errors.get(specimen_id, "raw trace unavailable"),
                )
            )
            continue
        if size is None or published is None:
            reconciliation_rows.append(
                _reconciliation_row(
                    specimen_id=specimen_id,
                    domain_id=domain_id,
                    primary=primary,
                    response=None,
                    reconciliation=None,
                    width_mm=size.width_mm if size is not None else "",
                    thickness_mm=size.thickness_mm if size is not None else "",
                    status="WORKBOOK_RECORD_MISSING",
                    error="measured size or published peak is absent",
                )
            )
            continue
        response = convert_trace_to_response(
            trace,
            width_mm=size.width_mm,
            thickness_mm=size.thickness_mm,
            canonical_specimen_id=specimen_id,
        )
        reconciliation = reconcile_published_peak(
            response, published, absolute_tolerance_mpa=tolerance
        )
        responses[specimen_id] = response
        reconciliations[specimen_id] = reconciliation
        reconciliation_rows.append(
            _reconciliation_row(
                specimen_id=specimen_id,
                domain_id=domain_id,
                primary=primary,
                response=response,
                reconciliation=reconciliation,
                width_mm=size.width_mm,
                thickness_mm=size.thickness_mm,
                status="PASS" if reconciliation.passed else "FAIL",
                error="",
            )
        )

    primary_peak_passed = all(
        specimen_id in reconciliations and reconciliations[specimen_id].passed
        for specimen_id in primary_by_id
    )
    primary_title_conflicts = sorted(
        specimen_id
        for specimen_id in primary_by_id
        if specimen_id in traces and traces[specimen_id].specimen_id != specimen_id
    )
    unsupported_title_conflicts = [
        specimen_id
        for specimen_id in primary_title_conflicts
        if sum(
            specimen_id in source
            for source in (sizes, published_peaks, lvi_observations)
        )
        != 3
    ]
    identity_guessed = bool(unsupported_title_conflicts)

    gate = evaluate_p0_gate(
        P0GateFacts(
            exact_identity_pairing_possible=exact_pairing_possible,
            exact_pair_counts={
                domain: valid_pair_counts.get(domain, 0) for domain in PRIMARY_COUNTS
            },
            identity_guessed=identity_guessed,
            all_sources_hash_bound=True,
            peak_reconciliation_passed=primary_peak_passed,
            missing_primary_channel_fractions=missing_channel_fractions,
        )
    )

    spatial_by_id = {record.specimen_id: record for record in spatial_pairs}
    primary_ids = set(primary_by_id)
    spatial_expansion = classify_spatial_expansion(
        raw_identity_ids=set(raw_by_id),
        valid_trace_ids=set(traces),
        spatial_ids=set(spatial_by_id),
        primary_ids=primary_ids,
    )
    spatial_raw_ids = set(spatial_expansion.identity_pair_ids)
    scalar_only_impacted_ids = sorted(
        specimen_id
        for specimen_id in set(raw_by_id) - spatial_raw_ids
        if specimen_id in lvi_observations
        and lvi_observations[specimen_id].has_numeric_damage_observation
    )
    intact_ids = sorted(
        specimen_id
        for specimen_id in raw_by_id
        if specimen_id in lvi_observations
        and lvi_observations[specimen_id].is_intact
    )

    post_rows = audit_post_cai_images(
        image_records, raw_specimen_ids=set(raw_by_id)
    )
    post_manifest_rows = [
        {
            "specimen_id": row.specimen_id,
            "view": row.view,
            "file_id": row.file_id,
            "relative_path": row.relative_path,
            "sha256": row.sha256,
            "size": row.size,
            "integrity_status": row.integrity_status,
            "local_bytes_verified": False,
            "input_forbidden": row.input_forbidden,
        }
        for row in post_rows
    ]

    strain_rows = [
        {
            "specimen_id": specimen_id,
            "domain_id": primary_by_id.get(specimen_id, ""),
            "primary": specimen_id in primary_by_id,
            "raw_parse_status": "PASS" if specimen_id in traces else "FAIL",
            "csv_unit_row": (
                "|".join(traces[specimen_id].original_channel_units)
                if specimen_id in traces
                else ""
            ),
            "csv_strain_unit_label": (
                "microstrain" if specimen_id in traces else ""
            ),
            "article_strain_unit_label": "micrometre_in_prose",
            "cross_source_label_conflict": True,
            "sign_convention_resolved": False,
            "status": StrainUnitStatus.STRAIN_UNIT_UNRESOLVED.value,
            "jis_modulus_authorized": False,
            "maximum_strain_authorized": False,
        }
        for specimen_id in sorted(raw_by_id)
    ]

    primary_absolute_errors = [
        reconciliations[specimen_id].absolute_error_mpa
        for specimen_id in primary_by_id
        if specimen_id in reconciliations
    ]
    legal_p1_endpoints = (
        "extension_at_peak_load",
        "uniform_pre_peak_stress_extension_slope",
        "pre_peak_integrated_stress_extension_index",
        "normalized_pre_peak_stress_extension_shape",
    )
    summary = {
        "base_sha": config.base_sha,
        "dataset": {"id": config.dataset_id, "version": config.dataset_version},
        "downstream_status": {
            stage: StageStatus.NOT_RUN_NOT_AUTHORIZED.value
            for stage in _DOWNSTREAM_STAGES
        },
        "gate": {"reasons": list(gate.reasons), "status": gate.status.value},
        "identity": {
            "identity_guessed": identity_guessed,
            "primary_title_conflicts": primary_title_conflicts,
            "primary_title_conflicts_crosschecked_by_three_workbooks": len(
                primary_title_conflicts
            )
            - len(unsupported_title_conflicts),
            "rule": "official filename + feature-bank domain + raw-file SHA-256",
        },
        "new_training": False,
        "pairing": {
            "primary_exact_pairs": len(paired),
            "primary_expected": sum(PRIMARY_COUNTS.values()),
            "per_domain": {
                domain: valid_pair_counts.get(domain, 0) for domain in PRIMARY_COUNTS
            },
        },
        "peak_reconciliation": {
            "global_absolute_tolerance_mpa": tolerance,
            "maximum_primary_absolute_error_mpa": max(primary_absolute_errors),
            "primary_passed": sum(
                reconciliation.passed
                for specimen_id, reconciliation in reconciliations.items()
                if specimen_id in primary_ids
            ),
            "primary_total": len(primary_ids),
        },
        "post_cai_images": {
            "input_forbidden": POST_CAI_IMAGE_INPUT_FORBIDDEN,
            "official_hash_bound_records": len(post_rows),
            "local_bytes_verified": 0,
        },
        "pre_model_authority_questions": {
            "q1": {
                "answer": config.base_sha,
                "evidence": [
                    "paper_v3/configs/damage_to_failure_response.yaml",
                    "source_hashes.csv",
                ],
                "status": "CLOSED",
            },
            "q2": {
                "answer": {
                    "compact_missing_historical_candidates": sorted(
                        config.historical_sources
                    ),
                    "raw_csvs_verified_outside_git": len(raw_by_id),
                },
                "evidence": [
                    "artifacts/damage_to_failure_response/P0_SOURCE_DISCOVERY.md",
                    "source_hashes.csv",
                ],
                "status": "CLOSED",
            },
            "q3": {
                "answer": "HISTORICAL_FULL_TREE_EXISTS_NO_GIT_AUTHORITY",
                "evidence": [
                    "artifacts/damage_to_failure_response/P0_SOURCE_DISCOVERY.md",
                    "source_hashes.csv",
                ],
                "status": "CLOSED",
            },
            "q4": {
                "answer": {
                    "official_raw_files": len(raw_by_id),
                    "strictly_decoded": len(traces),
                    "strict_decode_failures": sorted(raw_errors),
                },
                "evidence": ["raw_trace_qc.csv"],
                "status": "CLOSED",
            },
            "q5": {
                "answer": {
                    "primary_exact_pairs": len(paired),
                    "per_domain": {
                        domain: valid_pair_counts.get(domain, 0)
                        for domain in PRIMARY_COUNTS
                    },
                },
                "evidence": ["pairing_manifest.csv"],
                "status": "CLOSED",
            },
            "q6": {
                "answer": {
                    "raw_spatial_identity_pairs": len(
                        spatial_expansion.identity_pair_ids
                    ),
                    "raw_spatial_valid_trace_pairs": len(
                        spatial_expansion.valid_trace_pair_ids
                    ),
                    "extra_valid_trace_pairs": len(
                        spatial_expansion.extra_valid_trace_pair_ids
                    ),
                },
                "evidence": [
                    "artifacts/damage_to_failure_response/P0_SOURCE_DISCOVERY.md",
                    "raw_trace_qc.csv",
                ],
                "status": "CLOSED",
            },
            "q7": {
                "answer": StrainUnitStatus.STRAIN_UNIT_UNRESOLVED.value,
                "evidence": ["strain_unit_audit.csv"],
                "status": "CLOSED_AS_STRAIN_UNIT_UNRESOLVED",
            },
            "q8": {
                "answer": {
                    "global_absolute_tolerance_mpa": tolerance,
                    "primary_passed": sum(
                        reconciliations[specimen_id].passed
                        for specimen_id in primary_ids
                    ),
                    "primary_total": len(primary_ids),
                },
                "evidence": ["published_peak_reconciliation.csv"],
                "status": "CLOSED",
            },
            "q9": {
                "answer": {
                    "legal_stress_extension_endpoints": list(legal_p1_endpoints),
                    "strain_dependent_authorized": False,
                },
                "evidence": [
                    "strain_unit_audit.csv",
                    "published_peak_reconciliation.csv",
                ],
                "status": "CLOSED",
            },
            "q10": {
                "answer": "SEARCHED_NOT_ASSUMED",
                "evidence": [
                    "artifacts/damage_to_failure_response/LITERATURE_NOVELTY_LEDGER.md"
                ],
                "status": "CLOSED_SEARCHED_NOT_ASSUMED",
            },
        },
        "raw_trace_qc": {
            "decoded": len(traces),
            "failed": len(raw_errors),
            "failed_specimens": sorted(raw_errors),
            "official_records": len(raw_by_id),
            "primary_decoded": sum(
                specimen_id in traces for specimen_id in primary_ids
            ),
        },
        "response_endpoint_boundary": {
            "legal_for_p1_audit": list(legal_p1_endpoints),
            "strain_dependent_authorized": False,
        },
        "schema_version": 1,
        "spatial_expansion": {
            "extra_exact_spatial_identity_pairs": len(
                spatial_expansion.extra_identity_pair_ids
            ),
            "extra_exact_spatial_identity_specimen_ids": list(
                spatial_expansion.extra_identity_pair_ids
            ),
            "extra_valid_spatial_trace_pairs": len(
                spatial_expansion.extra_valid_trace_pair_ids
            ),
            "extra_valid_spatial_trace_specimen_ids": list(
                spatial_expansion.extra_valid_trace_pair_ids
            ),
            "raw_plus_spatial_identity_pairs": len(
                spatial_expansion.identity_pair_ids
            ),
            "raw_plus_spatial_valid_trace_pairs": len(
                spatial_expansion.valid_trace_pair_ids
            ),
            "scalar_only_impacted_without_spatial_pair": len(
                scalar_only_impacted_ids
            ),
            "intact_raw_specimens": len(intact_ids),
        },
        "strain": {
            "csv_unit_metadata": "microstrain",
            "cross_source_label_conflict": True,
            "sign_convention_resolved": False,
            "status": StrainUnitStatus.STRAIN_UNIT_UNRESOLVED.value,
        },
    }
    per_domain_pairs = {
        domain: valid_pair_counts.get(domain, 0) for domain in PRIMARY_COUNTS
    }
    report = f"""# P0 Damage-to-Failure Response Data Audit

Status: `{gate.status.value}`

## Authority

- Exact Git base: `{config.base_sha}`
- Dataset: `{config.dataset_id}`, version `{config.dataset_version}`
- Official raw records: {len(raw_by_id)}
- Official post-CAI image records: {len(post_rows)} (remote official hashes only;
  local image bytes were not downloaded or used)
- New training: NO

## Primary cohort

- Exact primary pairs: {len(paired)}/276
- Per-domain pairs: {per_domain_pairs}
- Primary raw traces decoded: {sum(specimen_id in traces for specimen_id in primary_ids)}/276
- Internal-title conflicts: {primary_title_conflicts}; each listed conflict is
  retained in QC and canonical identity is supported by official filename,
  dataset version, file SHA-256, and all three workbooks.

## Peak reconciliation

- Formula: `abs(Load[V] * 25 * 1000 / (width_mm * thickness_mm))`
- One workbook-derived absolute tolerance: {tolerance:.17g} MPa
- Primary pass count: {sum(reconciliations[s].passed for s in primary_ids)}/276
- Maximum primary absolute error: {max(primary_absolute_errors):.17g} MPa

## Boundaries and expansion

- Strain status: `{StrainUnitStatus.STRAIN_UNIT_UNRESOLVED.value}`; JIS modulus,
  maximum strain, and all gauge-derived endpoints remain unauthorized.
- Stress-extension endpoints legal for P1 audit: {', '.join(legal_p1_endpoints)}.
- Exact raw-file identity plus existing spatial-observation intersection:
  {len(spatial_expansion.identity_pair_ids)}; primary 276 plus
  {len(spatial_expansion.extra_identity_pair_ids)} additional identities.
- Decodable raw response plus spatial-observation intersection:
  {len(spatial_expansion.valid_trace_pair_ids)}; primary 276 plus
  {len(spatial_expansion.extra_valid_trace_pair_ids)} additional candidates.
- Additional impacted raw identities with scalar damage observations but no
  established spatial pair: {len(scalar_only_impacted_ids)}.
- One non-primary raw source anomaly is retained in QC: `q8-17` declares 15,711
  rows but contains 3,840 data rows.

## Gate

- Decision: `{gate.status.value}`
- Reasons: {list(gate.reasons)}
- P1-P5 execution state in this package: `NOT_RUN_NOT_AUTHORIZED`
"""

    raw_fields = [
        "specimen_id",
        "domain_id",
        "primary",
        "raw_relative_path",
        "raw_sha256",
        "raw_size",
        "parse_status",
        "parse_error",
        "internal_title",
        "title_matches_canonical",
        "canonical_workbook_crosscheck_count",
        "n_rows",
        "sampling_hz",
        "columns",
        "units",
        "peak_row",
    ]
    for name in _REGISTERED_CHANNEL_NAMES:
        key = name.casefold().replace("-", "_")
        raw_fields.extend(
            (f"{key}_finite_fraction", f"{key}_min", f"{key}_max")
        )
    payloads = {
        "REPORT.md": report.encode("utf-8"),
        "pairing_manifest.csv": _csv_payload(
            (
                "specimen_id",
                "domain_id",
                "raw_relative_path",
                "raw_trace_sha256",
                "source_dataset_id",
                "source_dataset_version",
                "internal_title",
                "title_matches_canonical",
                "identity_rule",
            ),
            pairing_rows,
        ),
        "post_cai_image_manifest.csv": _csv_payload(
            (
                "specimen_id",
                "view",
                "file_id",
                "relative_path",
                "sha256",
                "size",
                "integrity_status",
                "local_bytes_verified",
                "input_forbidden",
            ),
            post_manifest_rows,
        ),
        "published_peak_reconciliation.csv": _csv_payload(
            (
                "specimen_id",
                "domain_id",
                "primary",
                "status",
                "error",
                "width_mm",
                "thickness_mm",
                "peak_row",
                "extension_at_peak_mm",
                "raw_peak_mpa",
                "published_peak_mpa",
                "signed_error_mpa",
                "absolute_error_mpa",
                "absolute_tolerance_mpa",
                "passed",
            ),
            reconciliation_rows,
        ),
        "raw_trace_qc.csv": _csv_payload(raw_fields, raw_qc_rows),
        "source_hashes.csv": _csv_payload(
            (
                "logical_source",
                "relative_path",
                "size",
                "sha256",
                "expected_sha256",
                "status",
            ),
            sorted(source_rows, key=lambda row: str(row["relative_path"])),
        ),
        "strain_unit_audit.csv": _csv_payload(
            (
                "specimen_id",
                "domain_id",
                "primary",
                "raw_parse_status",
                "csv_unit_row",
                "csv_strain_unit_label",
                "article_strain_unit_label",
                "cross_source_label_conflict",
                "sign_convention_resolved",
                "status",
                "jis_modulus_authorized",
                "maximum_strain_authorized",
            ),
            strain_rows,
        ),
        "summary.json": _json_payload(summary),
    }
    return gate.status, payloads


def run_p0_audit(
    *,
    config_path: Path,
    repo_root: Path,
    legacy_root: Path,
    hasebe_v3_root: Path,
    output: Path,
) -> P0RunResult:
    """Execute the no-training P0 audit and publish one replayable package."""

    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise PipelineError(f"P0 output already exists: {destination}")
    repository = _regular_directory(repo_root, label="repository")
    historical = _regular_directory(legacy_root, label="legacy_root")
    v3_root = _regular_directory(hasebe_v3_root, label="hasebe_v3_root")
    config = load_p0_config(config_path)
    status, payloads = _execute_p0(
        config_path=Path(config_path),
        config=config,
        repo_root=repository,
        legacy_root=historical,
        v3_root=v3_root,
    )
    try:
        write_p0_package(destination, payloads)
        replay_p0(destination)
    except ArtifactError as error:
        raise PipelineError(str(error)) from error
    return P0RunResult(status=status, output=destination)


def _stable_bound_payload(
    path: Path,
    *,
    expected_sha256: str,
    logical_source: str,
    relative_path: str,
    max_bytes: int,
) -> bytes:
    try:
        snapshot = snapshot_file(
            path,
            max_bytes=max_bytes,
            logical_source=logical_source,
            relative_path=relative_path,
        )
    except AuthorityError as error:
        raise PipelineError(str(error)) from error
    if snapshot.sha256 != expected_sha256:
        raise PipelineError(f"source SHA-256 mismatch: {relative_path}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PipelineError(f"source cannot be read: {relative_path}") from error
    if (
        len(payload) != snapshot.size
        or hashlib.sha256(payload).hexdigest() != snapshot.sha256
    ):
        raise PipelineError(f"source changed during read: {relative_path}")
    return payload


def _read_p1_pair_authority(
    package: Path,
    identities: tuple[FeatureIdentity, ...],
    p0_config: P0Config,
) -> tuple[P1PairAuthority, ...]:
    manifest_path = package / "pairing_manifest.csv"
    try:
        manifest_snapshot = snapshot_file(
            manifest_path,
            max_bytes=16 * 1024 * 1024,
            logical_source="artifact:p0_data_audit",
            relative_path="pairing_manifest.csv",
        )
        payload = manifest_path.read_bytes()
    except (AuthorityError, OSError) as error:
        raise PipelineError("P0 pairing manifest cannot be read") from error
    if (
        len(payload) != manifest_snapshot.size
        or hashlib.sha256(payload).hexdigest() != manifest_snapshot.sha256
    ):
        raise PipelineError("P0 pairing manifest changed during P1 read")
    try:
        reader = csv.DictReader(payload.decode("utf-8").splitlines())
    except UnicodeDecodeError as error:
        raise PipelineError("P0 pairing manifest is not UTF-8") from error
    fields = (
        "specimen_id",
        "domain_id",
        "raw_relative_path",
        "raw_trace_sha256",
        "source_dataset_id",
        "source_dataset_version",
        "internal_title",
        "title_matches_canonical",
        "identity_rule",
    )
    if tuple(reader.fieldnames or ()) != fields:
        raise PipelineError("P0 pairing manifest schema changed")
    rows = tuple(reader)
    if len(rows) != len(identities):
        raise PipelineError("P0 pairing manifest row count changed")

    records: list[P1PairAuthority] = []
    for identity, row in zip(identities, rows, strict=True):
        if set(row) != set(fields):
            raise PipelineError("P0 pairing manifest row schema changed")
        specimen_id = str(row["specimen_id"] or "").strip().casefold()
        domain_id = str(row["domain_id"] or "").strip().casefold()
        digest = str(row["raw_trace_sha256"] or "").strip().casefold()
        raw_relative = str(row["raw_relative_path"] or "").strip()
        path = PurePosixPath(raw_relative)
        expected_prefix = (
            p0_config.dataset_id,
            f"v{p0_config.dataset_version}",
            p0_config.raw_folder,
        )
        if (
            specimen_id != identity.specimen_id
            or domain_id != identity.domain_id
            or _SHA256_RE.fullmatch(digest) is None
            or path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 4
            or path.parts[:3] != expected_prefix
            or row["source_dataset_id"] != p0_config.dataset_id
            or row["source_dataset_version"] != str(p0_config.dataset_version)
            or row["identity_rule"]
            != "OFFICIAL_FILENAME_PLUS_DOMAIN_PLUS_SHA256"
        ):
            raise PipelineError(f"P0 pairing authority changed: {identity.specimen_id}")
        records.append(
            P1PairAuthority(
                specimen_id=specimen_id,
                domain_id=domain_id,
                raw_relative_path=path.as_posix(),
                v3_relative_path=PurePosixPath(*path.parts[2:]).as_posix(),
                raw_trace_sha256=digest,
            )
        )
    return tuple(records)


def _extraction_sha256(record: ExtractedResponse) -> str:
    metadata = {
        "baseline_extension_mm": record.baseline_extension_mm.hex(),
        "baseline_stress_mpa": record.baseline_stress_mpa.hex(),
        "extension_peak_mm": record.extension_peak_mm.hex(),
        "normalized_prepeak_auc": record.normalized_prepeak_auc.hex(),
        "peak_row": record.peak_row,
        "q_midpoint": record.q_midpoint.hex(),
        "slope_u20_u60_mpa_per_mm": record.slope_u20_u60_mpa_per_mm.hex(),
        "specimen_id": record.specimen_id,
        "unique_extension_positions": record.unique_extension_positions,
        "zeroed_peak_stress_mpa": record.zeroed_peak_stress_mpa.hex(),
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    for values in (
        record.u,
        record.extension_mm,
        record.zeroed_stress_mpa,
        record.normalized_stress,
    ):
        digest.update(
            np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")
        )
    return digest.hexdigest()


def _execute_p1(
    *,
    config_path: Path,
    config: P1Config,
    repo_root: Path,
    legacy_root: Path,
    v3_root: Path,
    p0_package: Path,
) -> tuple[
    StageStatus,
    tuple[str, ...],
    dict[str, bytes],
    bytes,
]:
    from cmc_bbdm.damage_response.nested_eval import (
        P1RedundancyRecord,
        evaluate_p1_redundancy,
    )

    config_snapshot = snapshot_file(
        config_path,
        max_bytes=1024 * 1024,
        logical_source="git:research_branch",
        relative_path="paper_v3/configs/damage_to_failure_response_p1.yaml",
    )
    p0_config_path = repo_root / _P0_CONFIG_RELATIVE_PATH
    p0_config = load_p0_config(p0_config_path)
    if p0_config.base_sha != config.base_sha:
        raise PipelineError("P0 and P1 base identities differ")

    feature_identities = load_feature_identities(
        repo_root / p0_config.feature_bank.relative_path,
        expected_sha256=p0_config.feature_bank.sha256,
    )
    pair_authority = _read_p1_pair_authority(
        p0_package, feature_identities, p0_config
    )
    sizes = read_specimen_sizes(
        v3_root / p0_config.size_workbook.relative_path,
        expected_sha256=p0_config.size_workbook.sha256,
    )
    published_peaks = read_published_peaks(
        v3_root / p0_config.peak_workbook.relative_path,
        expected_sha256=p0_config.peak_workbook.sha256,
    )
    design_records = read_primary_design_metadata(
        legacy_root / p0_config.legacy_spatial_manifest.relative_path,
        p0_config.legacy_spatial_manifest.sha256,
        feature_identities,
        sizes,
    )
    design_by_id = {record.specimen_id: record for record in design_records}
    tolerance = derive_global_absolute_tolerance(published_peaks.values())

    extracted_by_id: dict[str, ExtractedResponse] = {}
    extraction_hashes: dict[str, str] = {}
    descriptor_qc_rows: list[dict[str, object]] = []
    descriptor_rows: list[dict[str, object]] = []
    extraction_replay_identical = True
    maximum_peak_error = 0.0

    for identity, authority in zip(
        feature_identities, pair_authority, strict=True
    ):
        specimen_id = identity.specimen_id
        size = sizes.get(specimen_id)
        published = published_peaks.get(specimen_id)
        if size is None or published is None or specimen_id not in design_by_id:
            raise PipelineError(f"P1 source metadata is missing: {specimen_id}")
        raw_payload = _stable_bound_payload(
            v3_root / authority.v3_relative_path,
            expected_sha256=authority.raw_trace_sha256,
            logical_source="local:hasebe_v3_root",
            relative_path=authority.raw_relative_path,
            max_bytes=64 * 1024 * 1024,
        )
        trace = decode_raw_cai_csv(raw_payload)
        response = convert_trace_to_response(
            trace,
            width_mm=size.width_mm,
            thickness_mm=size.thickness_mm,
            canonical_specimen_id=specimen_id,
        )
        reconciliation = reconcile_published_peak(
            response,
            published,
            absolute_tolerance_mpa=tolerance,
        )
        maximum_peak_error = max(
            maximum_peak_error, reconciliation.absolute_error_mpa
        )
        if not reconciliation.passed:
            raise PipelineError(f"P1 peak reconciliation changed: {specimen_id}")
        try:
            first = extract_prepeak_response(
                response,
                baseline_samples=config.baseline_samples,
                grid_points=config.grid_points,
                minimum_unique_extension_positions=(
                    config.minimum_unique_extension_positions
                ),
            )
            second = extract_prepeak_response(
                response,
                baseline_samples=config.baseline_samples,
                grid_points=config.grid_points,
                minimum_unique_extension_positions=(
                    config.minimum_unique_extension_positions
                ),
            )
        except ExtractionError as error:
            descriptor_qc_rows.append(
                {
                    "specimen_id": specimen_id,
                    "domain_id": identity.domain_id,
                    "raw_relative_path": authority.raw_relative_path,
                    "raw_trace_sha256": authority.raw_trace_sha256,
                    "published_cai_strength_mpa": published.value_mpa,
                    "status": "FAIL",
                    "error": str(error),
                    "peak_row": response.peak_row,
                    "raw_peak_mpa": reconciliation.raw_peak_mpa,
                    "peak_absolute_error_mpa": (
                        reconciliation.absolute_error_mpa
                    ),
                    "unique_extension_positions": "",
                    "extraction_sha256": "",
                    "repeat_extraction_byte_identical": False,
                }
            )
            extraction_replay_identical = False
            continue
        first_sha = _extraction_sha256(first)
        second_sha = _extraction_sha256(second)
        repeat_identical = first_sha == second_sha
        extraction_replay_identical &= repeat_identical
        if not repeat_identical:
            raise PipelineError(f"P1 extraction replay differs: {specimen_id}")
        extracted_by_id[specimen_id] = first
        extraction_hashes[specimen_id] = first_sha
        descriptor_qc_rows.append(
            {
                "specimen_id": specimen_id,
                "domain_id": identity.domain_id,
                "raw_relative_path": authority.raw_relative_path,
                "raw_trace_sha256": authority.raw_trace_sha256,
                "published_cai_strength_mpa": published.value_mpa,
                "status": "PASS",
                "error": "",
                "peak_row": first.peak_row,
                "raw_peak_mpa": reconciliation.raw_peak_mpa,
                "peak_absolute_error_mpa": reconciliation.absolute_error_mpa,
                "unique_extension_positions": first.unique_extension_positions,
                "extraction_sha256": first_sha,
                "repeat_extraction_byte_identical": True,
            }
        )
        descriptor_rows.append(
            {
                "specimen_id": specimen_id,
                "domain_id": identity.domain_id,
                "published_cai_strength_mpa": published.value_mpa,
                "extension_peak_mm": first.extension_peak_mm,
                "slope_u20_u60_mpa_per_mm": (
                    first.slope_u20_u60_mpa_per_mm
                ),
                "normalized_prepeak_auc": first.normalized_prepeak_auc,
                "q_midpoint": first.q_midpoint,
                "peak_row": first.peak_row,
                "baseline_extension_mm": first.baseline_extension_mm,
                "baseline_stress_mpa": first.baseline_stress_mpa,
                "zeroed_peak_stress_mpa": first.zeroed_peak_stress_mpa,
                "unique_extension_positions": first.unique_extension_positions,
                "extraction_sha256": first_sha,
            }
        )

    valid_identities = tuple(
        identity
        for identity in feature_identities
        if identity.specimen_id in extracted_by_id
    )
    valid_counts = Counter(identity.domain_id for identity in valid_identities)
    if set(valid_counts) != set(PRIMARY_COUNTS) or min(valid_counts.values()) < 2:
        raise PipelineError(
            "P1 valid extraction cannot support six-domain redundancy evaluation"
        )
    redundancy_records = tuple(
        P1RedundancyRecord(
            specimen_id=identity.specimen_id,
            domain_id=identity.domain_id,
            published_cai_strength_mpa=(
                published_peaks[identity.specimen_id].value_mpa
            ),
            extension_peak_mm=(
                extracted_by_id[identity.specimen_id].extension_peak_mm
            ),
            slope_u20_u60_mpa_per_mm=(
                extracted_by_id[
                    identity.specimen_id
                ].slope_u20_u60_mpa_per_mm
            ),
            normalized_prepeak_auc=(
                extracted_by_id[identity.specimen_id].normalized_prepeak_auc
            ),
            design=design_by_id[identity.specimen_id],
        )
        for identity in valid_identities
    )
    evaluation = evaluate_p1_redundancy(redundancy_records)
    metrics = {
        (metric.endpoint, metric.model): metric for metric in evaluation.metrics
    }

    endpoint_ranges = {
        endpoint: max(float(getattr(record, endpoint)) for record in redundancy_records)
        - min(float(getattr(record, endpoint)) for record in redundancy_records)
        for endpoint in PRIMARY_TARGET_FIELDS
    }
    gate = evaluate_p1_gate(
        tuple(
            EndpointGateFacts(
                endpoint=endpoint,
                valid_count=len(redundancy_records),
                valid_domain_counts={
                    domain: valid_counts.get(domain, 0) for domain in PRIMARY_COUNTS
                },
                strength_only_pooled_r2=metrics[
                    (endpoint, "strength_only")
                ].pooled_r2,
                strength_only_domain_spearman={
                    item.domain_id: item.spearman
                    for item in metrics[
                        (endpoint, "strength_only")
                    ].domain_spearman
                },
                descriptor_range=endpoint_ranges[endpoint],
                range_tolerance=ENDPOINT_RANGE_TOLERANCES[endpoint],
                replay_byte_identical=extraction_replay_identical,
            )
            for endpoint in PRIMARY_TARGET_FIELDS
        )
    )

    curve_records = tuple(
        CurveRecord(
            specimen_id=identity.specimen_id,
            domain_id=identity.domain_id,
            published_cai_strength_mpa=(
                published_peaks[identity.specimen_id].value_mpa
            ),
            normalized_curve=extracted_by_id[
                identity.specimen_id
            ].normalized_stress,
        )
        for identity in valid_identities
    )
    representative_pairs = select_representative_pairs(curve_records)
    if len(representative_pairs) != config.maximum_representative_pairs:
        raise PipelineError("P1 representative-pair count changed")

    oof_rows = [
        {
            "specimen_id": row.specimen_id,
            "domain_id": row.domain_id,
            "held_out_domain": row.held_out_domain,
            "endpoint": row.endpoint,
            "model": row.model,
            "truth": row.truth,
            "prediction": row.prediction,
            "residual": row.truth - row.prediction,
            "fold_state_sha256": row.fold_state_sha256,
        }
        for row in evaluation.predictions
    ]
    domain_rows: list[dict[str, object]] = []
    for endpoint in PRIMARY_TARGET_FIELDS:
        strength_metric = metrics[(endpoint, "strength_only")]
        design_metric = metrics[(endpoint, "strength_plus_design")]
        strength_correlations = {
            item.domain_id: item.spearman
            for item in strength_metric.domain_spearman
        }
        design_correlations = {
            item.domain_id: item.spearman
            for item in design_metric.domain_spearman
        }
        for domain in PRIMARY_COUNTS:
            values = [
                float(getattr(record, endpoint))
                for record in redundancy_records
                if record.domain_id == domain
            ]
            domain_rows.append(
                {
                    "endpoint": endpoint,
                    "domain_id": domain,
                    "valid_count": len(values),
                    "descriptor_min": min(values),
                    "descriptor_max": max(values),
                    "descriptor_range": max(values) - min(values),
                    "strength_only_pooled_r2": strength_metric.pooled_r2,
                    "strength_only_domain_spearman": (
                        strength_correlations[domain]
                    ),
                    "strength_plus_design_pooled_r2": design_metric.pooled_r2,
                    "strength_plus_design_domain_spearman": (
                        design_correlations[domain]
                    ),
                }
            )

    curve_fields = [
        "specimen_id",
        "domain_id",
        "raw_trace_sha256",
        "extraction_sha256",
        "normalized_curve_sha256",
        *(f"q_u{index:03d}" for index in range(config.grid_points)),
    ]
    curve_rows: list[dict[str, object]] = []
    authority_by_id = {record.specimen_id: record for record in pair_authority}
    for identity in valid_identities:
        extracted = extracted_by_id[identity.specimen_id]
        normalized = np.ascontiguousarray(
            extracted.normalized_stress, dtype="<f8"
        )
        row: dict[str, object] = {
            "specimen_id": identity.specimen_id,
            "domain_id": identity.domain_id,
            "raw_trace_sha256": authority_by_id[
                identity.specimen_id
            ].raw_trace_sha256,
            "extraction_sha256": extraction_hashes[identity.specimen_id],
            "normalized_curve_sha256": hashlib.sha256(
                normalized.tobytes(order="C")
            ).hexdigest(),
        }
        row.update(
            {
                f"q_u{index:03d}": float(value)
                for index, value in enumerate(normalized)
            }
        )
        curve_rows.append(row)

    representative_rows = [
        {
            "rank": rank,
            "left_specimen_id": pair.left_specimen_id,
            "right_specimen_id": pair.right_specimen_id,
            "left_domain_id": pair.left_domain_id,
            "right_domain_id": pair.right_domain_id,
            "left_strength_mpa": pair.left_strength_mpa,
            "right_strength_mpa": pair.right_strength_mpa,
            "strength_abs_difference_mpa": pair.strength_abs_difference_mpa,
            "normalized_curve_rms": pair.curve_rms,
            "selection_rule": "NEAREST_STRENGTH_THEN_CURVE_RMS",
        }
        for rank, pair in enumerate(representative_pairs, start=1)
    ]

    endpoint_summaries: dict[str, object] = {}
    for decision in gate.endpoint_decisions:
        model_summaries: dict[str, object] = {}
        for model_name in ("strength_only", "strength_plus_design"):
            metric = metrics[(decision.endpoint, model_name)]
            model_summaries[model_name] = {
                "domain_spearman": {
                    item.domain_id: item.spearman
                    for item in metric.domain_spearman
                },
                "pooled_r2": metric.pooled_r2,
            }
        endpoint_summaries[decision.endpoint] = {
            "coverage_fraction": decision.coverage_fraction,
            "descriptor_range": endpoint_ranges[decision.endpoint],
            "gate_passed": decision.passed,
            "gate_reasons": list(decision.reasons),
            "models": model_summaries,
            "nonredundant_domain_count": decision.nonredundant_domain_count,
            "range_tolerance": ENDPOINT_RANGE_TOLERANCES[decision.endpoint],
            "valid_count": len(redundancy_records),
        }
    downstream_status = {
        "P2": (
            "AUTHORIZED_NOT_RUN"
            if gate.status is StageStatus.P1_GO
            else StageStatus.NOT_RUN_NOT_AUTHORIZED.value
        ),
        "P3": StageStatus.NOT_RUN_NOT_AUTHORIZED.value,
        "P4": StageStatus.NOT_RUN_NOT_AUTHORIZED.value,
        "P5": StageStatus.NOT_RUN_NOT_AUTHORIZED.value,
    }
    summary = {
        "artifact_replay_byte_identical": True,
        "base_sha": config.base_sha,
        "config_sha256": config_snapshot.sha256,
        "downstream_status": downstream_status,
        "endpoints": endpoint_summaries,
        "estimator_contract": {
            "design_sensitivity_is_gate_input": False,
            "fixed_ridge_alpha": config.ridge_alpha,
            "fixed_ridge_fit_count": len(evaluation.fold_states),
            "hyperparameter_search": False,
            "outer_split": "leave_one_domain_out",
            "strength_only_is_gate_input": True,
        },
        "extraction": {
            "baseline_samples": config.baseline_samples,
            "failed": len(feature_identities) - len(redundancy_records),
            "grid_points": config.grid_points,
            "maximum_peak_reconciliation_error_mpa": maximum_peak_error,
            "peak_reconciliation_tolerance_mpa": tolerance,
            "repeat_extraction_byte_identical": extraction_replay_identical,
            "valid": len(redundancy_records),
        },
        "gate": {
            "passing_endpoints": list(gate.passing_endpoints),
            "status": gate.status.value,
        },
        "input_boundary": {
            "post_cai_image_used": False,
            "strain_endpoint_used": False,
            "target_domain_outcome_used_for_fit": False,
            "true_response_used_as_predictor": False,
        },
        "new_damage_to_response_model_training": False,
        "p0_authority": {
            "required_status": config.required_p0_status.value,
            "summary_relative_path": config.p0_summary_relative_path,
            "summary_sha256": config.p0_summary_sha256,
        },
        "primary_cohort": {
            "expected": sum(PRIMARY_COUNTS.values()),
            "per_domain_valid": {
                domain: valid_counts.get(domain, 0) for domain in PRIMARY_COUNTS
            },
            "valid": len(redundancy_records),
        },
        "representative_pairs": {
            "count": len(representative_pairs),
            "manual_selection": False,
            "rule": "nearest strength, deduplicate, curve RMS desc, pair IDs",
        },
        "schema_version": 1,
        "strain_status": "STRAIN_UNIT_UNRESOLVED",
    }

    metric_lines = []
    for endpoint in PRIMARY_TARGET_FIELDS:
        strength = metrics[(endpoint, "strength_only")]
        design = metrics[(endpoint, "strength_plus_design")]
        decision = next(
            item for item in gate.endpoint_decisions if item.endpoint == endpoint
        )
        metric_lines.append(
            f"- `{endpoint}`: range `{endpoint_ranges[endpoint]:.17g}`, "
            f"strength-only pooled R2 `{strength.pooled_r2:.17g}`, "
            f"strength+design pooled R2 `{design.pooled_r2:.17g}`, "
            f"gate `{'PASS' if decision.passed else 'FAIL'}`."
        )
    metric_text = "\n".join(metric_lines)
    strength_gate_text = "\n".join(
        f"  - `{endpoint}`: pooled R2 "
        f"`{metrics[(endpoint, 'strength_only')].pooled_r2:.17g}`"
        for endpoint in PRIMARY_TARGET_FIELDS
    )
    report = f"""# P1 CAI Response Richness Audit

Status: `{gate.status.value}`

## Authority and scope

- Exact Git base: `{config.base_sha}`
- P0 summary SHA-256: `{config.p0_summary_sha256}` (`P0_GO`)
- Primary cohort: {len(redundancy_records)}/276 valid, with all six domains.
- Response axis: stress-extension only; strain remains `STRAIN_UNIT_UNRESOLVED`.
- New damage-to-response model training: NO.
- Fixed redundancy reference fits: {len(evaluation.fold_states)}; no search.

## Endpoint decisions

{metric_text}

## Decision

- Passing endpoints: {list(gate.passing_endpoints)}
- Artifact/extraction replay byte-identical: YES
- P2 status: `{downstream_status['P2']}`
- P3-P5: `NOT_RUN_NOT_AUTHORIZED`
"""
    decision_payload = f"""# P1 Response Richness Decision

Status: `{gate.status.value}`

- Base SHA: `{config.base_sha}`
- P0 authority: `{config.p0_summary_sha256}` / `P0_GO`
- Valid responses: {len(redundancy_records)}/276 across 6/6 domains
- Passing endpoints: {', '.join(gate.passing_endpoints) or 'NONE'}
- Strength-only gate metrics:
{strength_gate_text}
- Extraction repeat: byte-identical
- Artifact package replay: required and verified by the publishing command
- P2: `{downstream_status['P2']}`
- P3-P5: `NOT_RUN_NOT_AUTHORIZED`
- Evidence: `results/damage_to_failure_response/p1_response_richness/`
- Manual specimen/result selection: NO
- Hyperparameter search: NO
""".encode()

    payloads = {
        "descriptor_table.csv": _csv_payload(
            (
                "specimen_id",
                "domain_id",
                "published_cai_strength_mpa",
                "extension_peak_mm",
                "slope_u20_u60_mpa_per_mm",
                "normalized_prepeak_auc",
                "q_midpoint",
                "peak_row",
                "baseline_extension_mm",
                "baseline_stress_mpa",
                "zeroed_peak_stress_mpa",
                "unique_extension_positions",
                "extraction_sha256",
            ),
            descriptor_rows,
        ),
        "descriptor_qc.csv": _csv_payload(
            (
                "specimen_id",
                "domain_id",
                "raw_relative_path",
                "raw_trace_sha256",
                "published_cai_strength_mpa",
                "status",
                "error",
                "peak_row",
                "raw_peak_mpa",
                "peak_absolute_error_mpa",
                "unique_extension_positions",
                "extraction_sha256",
                "repeat_extraction_byte_identical",
            ),
            descriptor_qc_rows,
        ),
        "domain_summary.csv": _csv_payload(
            (
                "endpoint",
                "domain_id",
                "valid_count",
                "descriptor_min",
                "descriptor_max",
                "descriptor_range",
                "strength_only_pooled_r2",
                "strength_only_domain_spearman",
                "strength_plus_design_pooled_r2",
                "strength_plus_design_domain_spearman",
            ),
            domain_rows,
        ),
        "strength_redundancy_oof.csv": _csv_payload(
            (
                "specimen_id",
                "domain_id",
                "held_out_domain",
                "endpoint",
                "model",
                "truth",
                "prediction",
                "residual",
                "fold_state_sha256",
            ),
            oof_rows,
        ),
        "response_curve_manifest.csv": _csv_payload(curve_fields, curve_rows),
        "representative_pair_manifest.csv": _csv_payload(
            (
                "rank",
                "left_specimen_id",
                "right_specimen_id",
                "left_domain_id",
                "right_domain_id",
                "left_strength_mpa",
                "right_strength_mpa",
                "strength_abs_difference_mpa",
                "normalized_curve_rms",
                "selection_rule",
            ),
            representative_rows,
        ),
        "summary.json": _json_payload(summary),
        "REPORT.md": report.encode("utf-8"),
    }
    return gate.status, gate.passing_endpoints, payloads, decision_payload


def _verify_p1_package_determinism(
    parent: Path, payloads: Mapping[str, bytes]
) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".p1-replay-", dir=parent) as directory:
        temporary = Path(directory)
        first = temporary / "first"
        second = temporary / "second"
        write_p1_package(first, payloads)
        write_p1_package(second, payloads)
        first_report = replay_p1(first)
        second_report = replay_p1(second)
        first_names = sorted(path.name for path in first.iterdir())
        second_names = sorted(path.name for path in second.iterdir())
        if (
            not first_report.verified
            or not second_report.verified
            or first_names != second_names
            or any(
                (first / name).read_bytes() != (second / name).read_bytes()
                for name in first_names
            )
        ):
            raise PipelineError("P1 package replay is not byte-identical")


def _write_new_regular_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise PipelineError(f"P1 decision cannot be created: {path}") from error
    completed = False
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
        completed = True
    except OSError as error:
        raise PipelineError(f"P1 decision cannot be written: {path}") from error
    finally:
        os.close(descriptor)
        if not completed:
            try:
                path.unlink()
            except OSError:
                pass


def run_p1_audit(
    *,
    config_path: Path,
    repo_root: Path,
    legacy_root: Path,
    hasebe_v3_root: Path,
    output: Path,
    decision_output: Path,
) -> P1RunResult:
    """Execute the fixed P1 audit after verifying the committed P0 package."""

    destination = Path(output)
    decision_path = Path(decision_output)
    for label, path in (("P1 output", destination), ("P1 decision", decision_path)):
        if path.exists() or path.is_symlink():
            raise PipelineError(f"{label} already exists: {path}")
    repository = _regular_directory(repo_root, label="repository")
    config = load_p1_config(config_path)
    p0_package = validate_p1_p0_authority(config, repository)
    historical = _regular_directory(legacy_root, label="legacy_root")
    v3_root = _regular_directory(hasebe_v3_root, label="hasebe_v3_root")
    status, passing_endpoints, payloads, decision_payload = _execute_p1(
        config_path=Path(config_path),
        config=config,
        repo_root=repository,
        legacy_root=historical,
        v3_root=v3_root,
        p0_package=p0_package,
    )
    try:
        _verify_p1_package_determinism(destination.parent, payloads)
        write_p1_package(destination, payloads)
        replay_p1(destination)
    except ArtifactError as error:
        raise PipelineError(str(error)) from error
    _write_new_regular_file(decision_path, decision_payload)
    return P1RunResult(
        status=status,
        output=destination,
        decision_output=decision_path,
        passing_endpoints=passing_endpoints,
    )


def load_p2_config(path: Path) -> object:
    """Load the P2 protocol without importing its execution stack at startup."""

    from cmc_bbdm.damage_response.p2_pipeline import P2PipelineError
    from cmc_bbdm.damage_response.p2_pipeline import (
        load_p2_config as load_p2_config_core,
    )

    try:
        return load_p2_config_core(path)
    except P2PipelineError as error:
        raise PipelineError(str(error)) from error


def validate_p2_upstream_authority(config: object, repo_root: Path) -> object:
    """Verify committed P0/P1 authority before opening external P2 sources."""

    from cmc_bbdm.damage_response.p2_pipeline import P2PipelineError
    from cmc_bbdm.damage_response.p2_pipeline import (
        validate_p2_upstream_authority as validate_p2_upstream_authority_core,
    )

    try:
        return validate_p2_upstream_authority_core(config, repo_root)
    except P2PipelineError as error:
        raise PipelineError(str(error)) from error


def run_p2_audit(
    *,
    config_path: Path,
    repo_root: Path,
    legacy_root: Path,
    hasebe_v3_root: Path,
    output: Path,
    decision_output: Path,
) -> P2RunResult:
    """Execute and publish the fixed nested-LODO P2 baseline audit."""

    from cmc_bbdm.damage_response.p2_pipeline import P2PipelineError
    from cmc_bbdm.damage_response.p2_pipeline import (
        run_p2_audit as run_p2_audit_core,
    )

    try:
        result = run_p2_audit_core(
            config_path=config_path,
            repo_root=repo_root,
            legacy_root=legacy_root,
            hasebe_v3_root=hasebe_v3_root,
            output=output,
            decision_output=decision_output,
        )
    except P2PipelineError as error:
        raise PipelineError(str(error)) from error
    return P2RunResult(
        status=result.status,
        output=result.output,
        decision_output=result.decision_output,
        passing_contrasts=result.passing_contrasts,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Damage-response stage-gated audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-p0", help="execute the no-training P0 audit")
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--legacy-root", type=Path, required=True)
    audit.add_argument("--hasebe-v3-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    replay = subparsers.add_parser("replay-p0", help="verify a P0 artifact package")
    replay.add_argument("--path", type=Path, required=True)
    audit_p1 = subparsers.add_parser(
        "audit-p1", help="execute the fixed response-richness audit"
    )
    audit_p1.add_argument("--config", type=Path, required=True)
    audit_p1.add_argument("--legacy-root", type=Path, required=True)
    audit_p1.add_argument("--hasebe-v3-root", type=Path, required=True)
    audit_p1.add_argument("--output", type=Path, required=True)
    audit_p1.add_argument("--decision-output", type=Path, required=True)
    replay_p1_parser = subparsers.add_parser(
        "replay-p1", help="verify a P1 artifact package"
    )
    replay_p1_parser.add_argument("--path", type=Path, required=True)
    audit_p2 = subparsers.add_parser(
        "audit-p2", help="execute the fixed damage-response baseline audit"
    )
    audit_p2.add_argument("--config", type=Path, required=True)
    audit_p2.add_argument("--legacy-root", type=Path, required=True)
    audit_p2.add_argument("--hasebe-v3-root", type=Path, required=True)
    audit_p2.add_argument("--output", type=Path, required=True)
    audit_p2.add_argument("--decision-output", type=Path, required=True)
    replay_p2_parser = subparsers.add_parser(
        "replay-p2", help="verify a P2 artifact package"
    )
    replay_p2_parser.add_argument("--path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "audit-p0":
            repository = Path(__file__).resolve().parents[3]
            result = run_p0_audit(
                config_path=arguments.config,
                repo_root=repository,
                legacy_root=arguments.legacy_root,
                hasebe_v3_root=arguments.hasebe_v3_root,
                output=arguments.output,
            )
            print(f"{result.status.value} {result.output}")
        elif arguments.command == "replay-p0":
            report = replay_p0(arguments.path)
            print(f"P0_REPLAY_OK payloads={report.payload_count}")
        elif arguments.command == "audit-p1":
            repository = Path(__file__).resolve().parents[3]
            result = run_p1_audit(
                config_path=arguments.config,
                repo_root=repository,
                legacy_root=arguments.legacy_root,
                hasebe_v3_root=arguments.hasebe_v3_root,
                output=arguments.output,
                decision_output=arguments.decision_output,
            )
            print(f"{result.status.value} {result.output}")
        elif arguments.command == "replay-p1":
            report = replay_p1(arguments.path)
            print(f"P1_REPLAY_OK payloads={report.payload_count}")
        elif arguments.command == "audit-p2":
            repository = Path(__file__).resolve().parents[3]
            result = run_p2_audit(
                config_path=arguments.config,
                repo_root=repository,
                legacy_root=arguments.legacy_root,
                hasebe_v3_root=arguments.hasebe_v3_root,
                output=arguments.output,
                decision_output=arguments.decision_output,
            )
            print(f"{result.status.value} {result.output}")
        else:
            report = replay_p2(arguments.path)
            print(f"P2_REPLAY_OK payloads={report.payload_count}")
    except (
        ArtifactError,
        AuthorityError,
        OSError,
        PairingError,
        PipelineError,
        RawCaiError,
        SourceError,
        TargetError,
        TypeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0
