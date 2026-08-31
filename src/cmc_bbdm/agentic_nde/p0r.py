"""P0R author-registration audit and source-aware replay."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from .artifacts import ArtifactError, replay_p0
from .author_authority import (
    EXPECTED_STATEMENT_SHA256,
    MAPPING_BASIS,
    USER_ATTESTED_SOURCE,
    VERBATIM_STATEMENT,
    build_author_registration_authority,
)
from .contracts import (
    PRIMARY_COUNTS,
    EvidenceClass,
    EvidenceRole,
    FrameGeometry,
    Orientation,
    P0RGateFacts,
    decide_p0r,
)
from .grid import Grid8x8, render_surface_grid
from .p0r_artifacts import write_p0r_package
from .registration import SurfaceToCscanTransform, create_transform
from .scan_frame_provenance import (
    HISTORICAL_CROP_RECIPES,
    ProcessingProvenanceError,
    ScanProcessingProvenance,
    verify_registered_crop,
)


class P0RPipelineError(ValueError):
    """Raised when a P0R authority or deterministic replay changes."""


_CONTROLLING_PROMPT_SHA256 = (
    "37265bb06eef238dca2325b590d1353a8159514ac693e4b4d070a637fb3b8eb8"
)
_REPOSITORY_BASE_SHA = "3cb63b544b6c13047773c0eda045558ff4466afa"
_REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "stage",
    "controlling_prompt_sha256",
    "repository_base_sha",
    "historical_p0",
    "author_authority",
    "external_authorities",
    "primary_counts",
    "processing_recipes",
    "registration",
    "q24_exemplar",
    "qc",
    "status_vocabulary",
}
_HISTORICAL_P0_KEYS = {
    "path",
    "status",
    "checksums_sha256",
    "artifact_manifest_sha256",
    "surface_manifest_sha256",
    "surface_qc_sha256",
    "summary_sha256",
}
_AUTHOR_KEYS = {
    "artifact_path",
    "artifact_sha256",
    "source_type",
    "evidence_status",
    "statement_sha256",
    "original_artifact_sha256",
    "original_archive_status",
    "orientation",
    "outer_frame_crop",
    "mapping_basis",
    "physical_mm_used_for_cross_modal_mapping",
    "example_surface",
    "example_scan",
}
_EXPECTED_STATUS_VOCABULARY = {
    "go": "P0R_AUTHOR_REGISTRATION_GO",
    "no_go": "P0R_AUTHOR_REGISTRATION_NO_GO",
    "evidence_conflict": "P0R_AUTHOR_EVIDENCE_CONFLICT",
    "provenance_unresolved": "P0R_PROCESSING_PROVENANCE_UNRESOLVED",
    "downstream_go": "P0_REGISTRATION_GO",
    "downstream_no_go": "P0_SPATIAL_REGISTRATION_NO_GO",
    "downstream_blocked": "NOT_RUN_NOT_AUTHORIZED",
}
_EXPECTED_FORBIDDEN_EVIDENCE = [
    "HIDDEN_CSCAN_PIXELS",
    "CSCAN_MASK",
    "DAMAGE_CENTROID",
    "CAI",
    "ORACLE_VALUE",
    "TARGET_DOMAIN_LABEL",
    "MANUAL_TARGET_ALIGNMENT",
]
_HEX = frozenset("0123456789abcdef")
_ROUND_TRIP_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class P0RComputation:
    author_authority: dict[str, object]
    surface_manifest: tuple[dict[str, object], ...]
    scan_processing_provenance: tuple[dict[str, object], ...]
    registration: tuple[dict[str, object], ...]
    registration_qc: tuple[dict[str, object], ...]
    grid_mapping_qc: tuple[dict[str, object], ...]
    summary: dict[str, object]
    report: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise P0RPipelineError("P0R authority file cannot be read") from error
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and not set(value) - _HEX


def _safe_relative(value: object, label: str) -> Path:
    if type(value) is not str:
        raise P0RPipelineError(f"{label} path is invalid")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise P0RPipelineError(f"{label} path is unsafe")
    return path


def _resolve_relative_file(
    root: Path, value: object, expected_sha256: object, label: str
) -> Path:
    relative = _safe_relative(value, label)
    if not _is_sha256(expected_sha256):
        raise P0RPipelineError(f"{label} SHA-256 is invalid")
    unresolved = root
    for part in relative.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise P0RPipelineError(f"{label} contains a symlink")
    try:
        path = unresolved.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise P0RPipelineError(f"{label} path escapes its authority root") from error
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise P0RPipelineError(f"{label} SHA-256 changed")
    return path


def _resolve_relative_directory(root: Path, value: object, label: str) -> Path:
    relative = _safe_relative(value, label)
    unresolved = root
    for part in relative.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise P0RPipelineError(f"{label} contains a symlink")
    try:
        path = unresolved.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise P0RPipelineError(f"{label} path escapes its authority root") from error
    if not path.is_dir():
        raise P0RPipelineError(f"{label} is unavailable")
    return path


def _resolve_external_file(root: Path, value: object, label: str) -> tuple[Path, str]:
    if type(value) is not str or not value:
        raise P0RPipelineError(f"{label} path is invalid")
    supplied = Path(value)
    unresolved = supplied if supplied.is_absolute() else root / supplied
    try:
        path = unresolved.resolve(strict=True)
        relative = path.relative_to(root)
    except (OSError, ValueError) as error:
        raise P0RPipelineError(f"{label} escapes the external authority root") from error
    probe = root
    for part in relative.parts:
        probe /= part
        if probe.is_symlink():
            raise P0RPipelineError(f"{label} contains a symlink")
    if not path.is_file():
        raise P0RPipelineError(f"{label} is not a regular file")
    return path, relative.as_posix()


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise P0RPipelineError(f"{label} is unavailable")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if (
                reader.fieldnames is None
                or len(set(reader.fieldnames)) != len(reader.fieldnames)
            ):
                raise P0RPipelineError(f"{label} schema is invalid")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise P0RPipelineError(f"{label} cannot be read") from error
    if not rows:
        raise P0RPipelineError(f"{label} contains no rows")
    return rows


def _index(
    rows: Sequence[Mapping[str, str]],
    *,
    domain_field: str,
    specimen_field: str,
    label: str,
) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for source in rows:
        row = dict(source)
        key = (row.get(domain_field, ""), row.get(specimen_field, ""))
        if not all(key) or key in result:
            raise P0RPipelineError(f"{label} specimen keys are missing or duplicated")
        result[key] = row
    return result


def _integer(row: Mapping[str, str], field: str, label: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise P0RPipelineError(f"{label} {field} is invalid") from error
    return value


def _expected_processing_recipes() -> dict[str, object]:
    normal, astm, dual = HISTORICAL_CROP_RECIPES
    return {
        "single_normal": {
            "screenshot_size": list(normal.screenshot_size),
            "panel_boxes": [list(box) for box in normal.panel_boxes],
        },
        "single_astm": {
            "screenshot_size": list(astm.screenshot_size),
            "panel_boxes": [list(box) for box in astm.panel_boxes],
        },
        "dual": {
            "screenshot_size": list(dual.screenshot_size),
            "panel_boxes": [list(box) for box in dual.panel_boxes],
        },
        "decode_mode": "RGB",
        "operation": "AXIS_ALIGNED_CROP",
        "resize": "NONE",
        "interpolation": "NONE",
        "rotation": "IDENTITY",
        "reflection": "NONE",
    }


def _validate_config(payload: object) -> dict[str, Any]:
    if (
        type(payload) is not dict
        or set(payload) != _REQUIRED_CONFIG_KEYS
        or payload.get("schema_version") != 1
        or payload.get("stage") != "P0R_AUTHOR_SURFACE_CSCAN_REGISTRATION"
        or payload.get("controlling_prompt_sha256")
        != _CONTROLLING_PROMPT_SHA256
        or payload.get("repository_base_sha") != _REPOSITORY_BASE_SHA
        or payload.get("primary_counts") != dict(PRIMARY_COUNTS)
        or payload.get("processing_recipes") != _expected_processing_recipes()
        or payload.get("status_vocabulary") != _EXPECTED_STATUS_VOCABULARY
    ):
        raise P0RPipelineError("P0R config schema or frozen contract changed")

    historical = payload.get("historical_p0")
    author = payload.get("author_authority")
    external = payload.get("external_authorities")
    registration = payload.get("registration")
    q24 = payload.get("q24_exemplar")
    qc = payload.get("qc")
    if (
        type(historical) is not dict
        or set(historical) != _HISTORICAL_P0_KEYS
        or historical.get("status") != "P0_SPATIAL_REGISTRATION_NO_GO"
        or any(
            not _is_sha256(historical.get(field))
            for field in (
                "checksums_sha256",
                "artifact_manifest_sha256",
                "surface_manifest_sha256",
                "surface_qc_sha256",
                "summary_sha256",
            )
        )
        or type(author) is not dict
        or set(author) != _AUTHOR_KEYS
        or type(external) is not dict
        or set(external) != {"paired_manifest", "historical_preprocessor"}
        or type(registration) is not dict
        or type(q24) is not dict
        or type(qc) is not dict
    ):
        raise P0RPipelineError("P0R config authority schema changed")

    expected_author = {
        "source_type": USER_ATTESTED_SOURCE,
        "evidence_status": "USER_ATTESTED",
        "statement_sha256": EXPECTED_STATEMENT_SHA256,
        "original_artifact_sha256": None,
        "original_archive_status": (
            "ORIGINAL_AUTHOR_COMMUNICATION_ARCHIVE_RECOMMENDED"
        ),
        "orientation": "ROT90",
        "outer_frame_crop": "NONE_AT_SPECIMEN_FRAME",
        "mapping_basis": MAPPING_BASIS,
        "physical_mm_used_for_cross_modal_mapping": False,
        "example_surface": "Q24-7astm.png",
        "example_scan": "Q24-7astm.jpg",
    }
    if any(author.get(field) != value for field, value in expected_author.items()):
        raise P0RPipelineError("P0R author authority contract changed")
    if not _is_sha256(author.get("artifact_sha256")):
        raise P0RPipelineError("P0R author artifact SHA-256 is invalid")

    expected_external_keys = {
        "paired_manifest": {"logical_root", "path", "sha256"},
        "historical_preprocessor": {
            "logical_root",
            "path",
            "sha256",
            "symbol",
            "source_git_commit",
        },
    }
    for name, fields in expected_external_keys.items():
        record = external.get(name)
        if (
            type(record) is not dict
            or set(record) != fields
            or record.get("logical_root") != "external_hasebe"
            or not _is_sha256(record.get("sha256"))
        ):
            raise P0RPipelineError(f"P0R {name} authority changed")
    if (
        external["historical_preprocessor"].get("symbol")
        != "crop_cscan_panels"
        or external["historical_preprocessor"].get("source_git_commit")
        != "UNAVAILABLE_NO_GIT_METADATA"
    ):
        raise P0RPipelineError("P0R historical preprocessing identity changed")

    expected_registration = {
        "orientation": "ROT90",
        "mapping_basis": MAPPING_BASIS,
        "normalized_edge_to_edge": True,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "physical_mm_used_for_cross_modal_mapping": False,
        "action_grid_rows": 8,
        "action_grid_columns": 8,
        "minimum_per_domain_fraction": 0.9,
        "minimum_total_registered": 240,
        "forbidden_orientation_evidence": _EXPECTED_FORBIDDEN_EVIDENCE,
    }
    if registration != expected_registration:
        raise P0RPipelineError("P0R registration contract changed")
    if q24 != {
        "specimen_id": "q24-7astm",
        "dataset_id": "6zt73pcnxv",
        "expected_in_frozen_276": False,
    }:
        raise P0RPipelineError("P0R Q24 exemplar contract changed")
    if (
        set(qc) != {
            "selection_seed",
            "specimens_per_domain",
            "panel_width_px",
            "panel_height_px",
        }
        or type(qc.get("selection_seed")) is not str
        or not qc["selection_seed"]
        or qc.get("specimens_per_domain") != 2
        or type(qc.get("panel_width_px")) is not int
        or type(qc.get("panel_height_px")) is not int
        or qc["panel_width_px"] <= 0
        or qc["panel_height_px"] <= 0
    ):
        raise P0RPipelineError("P0R QC contract changed")
    return dict(payload)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise P0RPipelineError("P0R config is unavailable")
    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise P0RPipelineError("P0R config is invalid") from error
    return _validate_config(payload), text


def _historical_p0(
    project_root: Path,
    external_root: Path,
    config: Mapping[str, Any],
) -> tuple[Path, list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    authority = config["historical_p0"]
    root = _resolve_relative_directory(
        project_root, authority["path"], "historical P0"
    )
    configured_files = {
        "CHECKSUMS.sha256": authority["checksums_sha256"],
        "artifact_manifest.json": authority["artifact_manifest_sha256"],
        "surface_manifest.csv": authority["surface_manifest_sha256"],
        "surface_qc.csv": authority["surface_qc_sha256"],
        "summary.json": authority["summary_sha256"],
    }
    if any(_sha256(root / name) != expected for name, expected in configured_files.items()):
        raise P0RPipelineError("historical P0 configured SHA-256 changed")
    try:
        summary = replay_p0(
            root,
            surface_root=external_root,
            project_root=project_root,
        )
    except (ArtifactError, OSError, ValueError) as error:
        raise P0RPipelineError("historical P0 replay failed") from error
    if summary.get("status") != authority["status"]:
        raise P0RPipelineError("historical P0 decision changed")
    return (
        root,
        _read_csv(root / "surface_manifest.csv", "historical P0 surface manifest"),
        _read_csv(root / "surface_qc.csv", "historical P0 surface QC"),
        summary,
    )


def _image_geometry(path: Path, label: str) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            image.load()
            return image.width, image.height, image.mode
    except (OSError, UnidentifiedImageError) as error:
        raise P0RPipelineError(f"{label} cannot be decoded") from error


def _coord(value: float) -> str:
    return f"{value:.12f}"


def _build_transform(
    *,
    surface_width: int,
    surface_height: int,
    crop_width: int,
    crop_height: int,
    evidence_hashes: tuple[str, ...],
) -> SurfaceToCscanTransform:
    try:
        return create_transform(
            source=FrameGeometry(surface_width, surface_height, 1.0, 1.0),
            destination=FrameGeometry(crop_width, crop_height, 1.0, 1.0),
            orientation=Orientation.ROT90,
            evidence_class=EvidenceClass.A_DIRECT_METADATA,
            evidence_roles=(
                EvidenceRole.AUTHOR_CORRESPONDENCE,
                EvidenceRole.DATASET_METADATA,
                EvidenceRole.SURFACE_METADATA,
            ),
            evidence_hashes=evidence_hashes,
            source_only_isolated=True,
            offset_x=0.0,
            offset_y=0.0,
        )
    except ValueError as error:
        raise P0RPipelineError("P0R transform construction failed") from error


def _grid_rows(
    domain: str,
    specimen: str,
    transform: SurfaceToCscanTransform,
) -> tuple[list[dict[str, object]], float]:
    destination = Grid8x8(
        width=float(transform.destination.width_px - 1),
        height=float(transform.destination.height_px - 1),
    )
    surface_records = {row["cell_id"]: row for row in render_surface_grid(transform)}
    rows: list[dict[str, object]] = []
    maximum_error = 0.0
    for cell in destination.cells():
        surface_box = tuple(surface_records[cell.cell_id]["surface_box"])
        round_trip = transform.forward_box(surface_box)
        cscan_box = (cell.x0, cell.y0, cell.x1, cell.y1)
        error = max(abs(left - right) for left, right in zip(cscan_box, round_trip))
        maximum_error = max(maximum_error, error)
        if error > _ROUND_TRIP_TOLERANCE:
            raise P0RPipelineError(
                f"P0R 64-cell round trip failed for {domain}/{specimen}"
            )
        rows.append(
            {
                "dataset_id": domain,
                "specimen_id": specimen,
                "cell_id": cell.cell_id,
                "row": cell.row,
                "column": cell.column,
                "cscan_x0": _coord(cell.x0),
                "cscan_y0": _coord(cell.y0),
                "cscan_x1": _coord(cell.x1),
                "cscan_y1": _coord(cell.y1),
                "surface_x0": _coord(surface_box[0]),
                "surface_y0": _coord(surface_box[1]),
                "surface_x1": _coord(surface_box[2]),
                "surface_y1": _coord(surface_box[3]),
                "round_trip_max_abs_error_px": f"{error:.17g}",
                "round_trip_status": "PASS",
                "transform_sha256": transform.sha256,
            }
        )
    return rows, maximum_error


def _verify_file_identity(
    path: Path, expected_sha256: str, label: str
) -> None:
    if not _is_sha256(expected_sha256) or _sha256(path) != expected_sha256:
        raise P0RPipelineError(f"{label} SHA-256 changed")


def _bind_specimen(
    *,
    key: tuple[str, str],
    historical: Mapping[str, str],
    surface_qc: Mapping[str, str],
    paired: Mapping[str, str],
    external_root: Path,
    evidence_prefix: tuple[str, ...],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    ScanProcessingProvenance,
]:
    domain, specimen = key
    label = f"{domain}/{specimen}"
    surface, surface_relative = _resolve_external_file(
        external_root, paired.get("source_path"), f"surface {label}"
    )
    raw, raw_relative = _resolve_external_file(
        external_root,
        paired.get("target_screenshot_path"),
        f"raw C-scan {label}",
    )
    crop, crop_relative = _resolve_external_file(
        external_root, paired.get("target_path"), f"registered crop {label}"
    )
    expected_surface_hash = historical.get("surface_sha256", "")
    expected_raw_hash = historical.get("cscan_source_sha256", "")
    expected_crop_hash = historical.get("registered_cscan_crop_sha256", "")
    if (
        historical.get("identity_status") != "PASS_EXACT_SPECIMEN_ID_AND_HASH"
        or historical.get("impacted_surface_path") != surface_relative
        or historical.get("cscan_source_path") != raw_relative
        or historical.get("registered_cscan_crop_path") != crop_relative
        or paired.get("source_sha256") != expected_surface_hash
        or paired.get("target_screenshot_sha256") != expected_raw_hash
        or paired.get("target_sha256") != expected_crop_hash
        or surface_qc.get("sha256") != expected_surface_hash
    ):
        raise P0RPipelineError(f"P0R exact identity binding changed for {label}")
    _verify_file_identity(surface, expected_surface_hash, f"surface {label}")
    _verify_file_identity(raw, expected_raw_hash, f"raw C-scan {label}")
    _verify_file_identity(crop, expected_crop_hash, f"registered crop {label}")

    surface_width, surface_height, surface_mode = _image_geometry(
        surface, f"surface {label}"
    )
    if (
        surface_mode != "RGB"
        or surface_qc.get("mode") != "RGB"
        or _integer(surface_qc, "width_px", f"surface QC {label}")
        != surface_width
        or _integer(surface_qc, "height_px", f"surface QC {label}")
        != surface_height
        or _integer(paired, "source_width", f"paired manifest {label}")
        != surface_width
        or _integer(paired, "source_height", f"paired manifest {label}")
        != surface_height
    ):
        raise P0RPipelineError(f"P0R surface geometry changed for {label}")
    panel_index = _integer(paired, "target_panel_index", f"paired manifest {label}")
    if panel_index != _integer(historical, "cscan_panel_index", f"historical P0 {label}"):
        raise P0RPipelineError(f"P0R specimen panel index changed for {label}")
    try:
        provenance = verify_registered_crop(
            raw,
            crop,
            panel_index=panel_index,
            expected_raw_sha256=expected_raw_hash,
            expected_registered_sha256=expected_crop_hash,
        )
    except ProcessingProvenanceError as error:
        raise P0RPipelineError(
            f"P0R scan processing provenance failed for {label}: {error}"
        ) from error
    if (
        _integer(historical, "registered_cscan_width_px", f"historical P0 {label}")
        != provenance.registered_width_px
        or _integer(historical, "registered_cscan_height_px", f"historical P0 {label}")
        != provenance.registered_height_px
        or _integer(paired, "target_width", f"paired manifest {label}")
        != provenance.registered_width_px
        or _integer(paired, "target_height", f"paired manifest {label}")
        != provenance.registered_height_px
    ):
        raise P0RPipelineError(f"P0R registered crop geometry changed for {label}")

    evidence_hashes = (
        *evidence_prefix,
        expected_surface_hash,
        expected_raw_hash,
        expected_crop_hash,
    )
    if len(set(evidence_hashes)) != len(evidence_hashes):
        raise P0RPipelineError(f"P0R evidence hashes collide for {label}")
    transform = _build_transform(
        surface_width=surface_width,
        surface_height=surface_height,
        crop_width=provenance.registered_width_px,
        crop_height=provenance.registered_height_px,
        evidence_hashes=evidence_hashes,
    )
    grid_rows, maximum_error = _grid_rows(domain, specimen, transform)

    surface_record = {
        **historical,
        "p0r_roster_status": "AUTHORIZED",
        "author_statement_sha256": EXPECTED_STATEMENT_SHA256,
    }
    provenance_record = {
        "dataset_id": domain,
        "specimen_id": specimen,
        "surface_path": surface_relative,
        "raw_cscan_path": raw_relative,
        "registered_cscan_crop_path": crop_relative,
        "processing_status": "PASS_EXACT_DECODED_RGB_REPLAY",
        **provenance.as_dict(),
    }
    registration_record = {
        "dataset_id": domain,
        "specimen_id": specimen,
        "status": "AUTHORIZED",
        "mapping_basis": MAPPING_BASIS,
        "orientation": Orientation.ROT90.value,
        "outer_frame_crop": "NONE_AT_SPECIMEN_FRAME",
        "physical_mm_used_for_cross_modal_mapping": "false",
        "coordinate_basis": "NORMALIZED_EDGE_TO_EDGE_PIXEL_FRAME",
        "source_width_px": surface_width,
        "source_height_px": surface_height,
        "destination_width_px": provenance.registered_width_px,
        "destination_height_px": provenance.registered_height_px,
        "scale_x": transform.scale_x,
        "scale_y": transform.scale_y,
        "offset_x": transform.offset_x,
        "offset_y": transform.offset_y,
        "evidence_class": EvidenceClass.A_DIRECT_METADATA.value,
        "evidence_roles": ";".join(role.value for role in transform.evidence_roles),
        "evidence_hashes": ";".join(transform.evidence_hashes),
        "orientation_selection_inputs": "AUTHOR_STATEMENT_ONLY",
        "transform_sha256": transform.sha256,
    }
    registration_qc_record = {
        "dataset_id": domain,
        "specimen_id": specimen,
        "status": "PASS",
        "authorized": "true",
        "invertible": "PASS",
        "boundary_consistency": "PASS_FULL_NORMALIZED_FRAME",
        "global_orientation": "PASS_ROT90_FIXED",
        "panel_resolution": "PASS",
        "decoded_pixel_replay": "PASS",
        "unsupported_rotation_reflection": "PASS_NONE",
        "physical_mm_cross_modal_mapping": "PASS_NOT_USED",
        "cai_access": "PASS_NOT_ACCESSED",
        "oracle_access": "PASS_NOT_ACCESSED",
        "hidden_cscan_orientation_selection": "PASS_NOT_USED",
        "grid_cell_count": 64,
        "maximum_round_trip_error_px": f"{maximum_error:.17g}",
        "deterministic_transform_replay": "PASS",
    }
    return (
        surface_record,
        provenance_record,
        registration_record,
        registration_qc_record,
        grid_rows,
        provenance,
    )


def _bind_q24(
    paired: Mapping[str, str], external_root: Path
) -> dict[str, object]:
    surface, surface_relative = _resolve_external_file(
        external_root, paired.get("source_path"), "Q24-7 surface exemplar"
    )
    raw, raw_relative = _resolve_external_file(
        external_root, paired.get("target_screenshot_path"), "Q24-7 scan exemplar"
    )
    crop, crop_relative = _resolve_external_file(
        external_root, paired.get("target_path"), "Q24-7 registered crop exemplar"
    )
    for path, expected, label in (
        (surface, paired.get("source_sha256", ""), "Q24-7 surface"),
        (raw, paired.get("target_screenshot_sha256", ""), "Q24-7 raw scan"),
        (crop, paired.get("target_sha256", ""), "Q24-7 registered crop"),
    ):
        _verify_file_identity(path, expected, label)
    try:
        provenance = verify_registered_crop(
            raw,
            crop,
            panel_index=_integer(paired, "target_panel_index", "Q24-7 paired manifest"),
            expected_raw_sha256=paired["target_screenshot_sha256"],
            expected_registered_sha256=paired["target_sha256"],
        )
    except ProcessingProvenanceError as error:
        raise P0RPipelineError("Q24-7 processing provenance failed") from error
    width, height, mode = _image_geometry(surface, "Q24-7 surface exemplar")
    if mode != "RGB":
        raise P0RPipelineError("Q24-7 surface exemplar is not RGB")
    return {
        "dataset_id": paired["dataset_id"],
        "specimen_id": paired["sample_id"],
        "in_frozen_276": False,
        "surface_path": surface_relative,
        "surface_sha256": paired["source_sha256"],
        "surface_width_px": width,
        "surface_height_px": height,
        "raw_scan_path": raw_relative,
        "raw_scan_sha256": paired["target_screenshot_sha256"],
        "registered_crop_path": crop_relative,
        "registered_crop_sha256": paired["target_sha256"],
        "registered_crop_width_px": provenance.registered_width_px,
        "registered_crop_height_px": provenance.registered_height_px,
        "panel_index": provenance.panel_index,
        "decoded_pixel_equal": provenance.decoded_pixel_equal,
    }


def _compute(
    *,
    config: Mapping[str, Any],
    external_root: Path,
    project_root: Path,
) -> P0RComputation:
    old_root, old_surface_rows, old_qc_rows, old_summary = _historical_p0(
        project_root, external_root, config
    )
    old_surface = _index(
        old_surface_rows,
        domain_field="dataset_id",
        specimen_field="specimen_id",
        label="historical P0 surface manifest",
    )
    old_qc = _index(
        old_qc_rows,
        domain_field="dataset_id",
        specimen_field="specimen_id",
        label="historical P0 surface QC",
    )
    counts = Counter(domain for domain, _ in old_surface)
    if (
        dict(counts) != dict(PRIMARY_COUNTS)
        or len(old_surface) != 276
        or set(old_surface) != set(old_qc)
        or old_summary.get("exact_identity_bound_count") not in {None, 276}
    ):
        raise P0RPipelineError("historical P0 frozen 276-row authority changed")

    author_config = config["author_authority"]
    author_path = _resolve_relative_file(
        project_root,
        author_config["artifact_path"],
        author_config["artifact_sha256"],
        "P0R author artifact",
    )
    try:
        author_text = author_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise P0RPipelineError("P0R author artifact cannot be read") from error
    if (
        EXPECTED_STATEMENT_SHA256 not in author_text
        or VERBATIM_STATEMENT not in author_text
    ):
        raise P0RPipelineError("P0R author artifact statement binding changed")
    author = build_author_registration_authority(
        original_artifact_sha256=author_config["original_artifact_sha256"]
    ).as_dict()

    paired_config = config["external_authorities"]["paired_manifest"]
    preprocessor_config = config["external_authorities"]["historical_preprocessor"]
    paired_path = _resolve_relative_file(
        external_root,
        paired_config["path"],
        paired_config["sha256"],
        "P0R paired manifest",
    )
    preprocessor_path = _resolve_relative_file(
        external_root,
        preprocessor_config["path"],
        preprocessor_config["sha256"],
        "P0R historical preprocessor",
    )
    try:
        preprocessor_text = preprocessor_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise P0RPipelineError("P0R historical preprocessor cannot be read") from error
    if "def crop_cscan_panels" not in preprocessor_text:
        raise P0RPipelineError("P0R historical preprocessor symbol changed")
    paired_rows = _read_csv(paired_path, "P0R paired manifest")
    paired = _index(
        paired_rows,
        domain_field="dataset_id",
        specimen_field="sample_id",
        label="P0R paired manifest",
    )
    if any(key not in paired for key in old_surface):
        raise P0RPipelineError("P0R paired manifest no longer covers the frozen roster")
    q24_key = (
        config["q24_exemplar"]["dataset_id"],
        config["q24_exemplar"]["specimen_id"],
    )
    if q24_key in old_surface or q24_key not in paired:
        raise P0RPipelineError("P0R Q24 exemplar roster identity changed")
    q24 = _bind_q24(paired[q24_key], external_root)

    evidence_prefix = (
        EXPECTED_STATEMENT_SHA256,
        author_config["artifact_sha256"],
        paired_config["sha256"],
        preprocessor_config["sha256"],
        config["historical_p0"]["surface_manifest_sha256"],
    )
    if len(set(evidence_prefix)) != len(evidence_prefix):
        raise P0RPipelineError("P0R global evidence hashes collide")

    domain_order = {domain: index for index, domain in enumerate(PRIMARY_COUNTS)}
    keys = sorted(old_surface, key=lambda key: (domain_order[key[0]], key[1]))
    surface_manifest: list[dict[str, object]] = []
    scan_provenance: list[dict[str, object]] = []
    registration: list[dict[str, object]] = []
    registration_qc: list[dict[str, object]] = []
    grid_mapping: list[dict[str, object]] = []
    raw_panel_counts: Counter[str] = Counter()
    raw_paths: set[str] = set()
    multi_panel_paths: set[str] = set()

    for key in keys:
        bound = _bind_specimen(
            key=key,
            historical=old_surface[key],
            surface_qc=old_qc[key],
            paired=paired[key],
            external_root=external_root,
            evidence_prefix=evidence_prefix,
        )
        surface_row, provenance_row, registration_row, qc_row, cells, provenance = bound
        surface_manifest.append(surface_row)
        scan_provenance.append(provenance_row)
        registration.append(registration_row)
        registration_qc.append(qc_row)
        grid_mapping.extend(cells)
        raw_path = str(provenance_row["raw_cscan_path"])
        raw_paths.add(raw_path)
        raw_panel_counts[raw_path] += 1
        if provenance.panel_count > 1:
            multi_panel_paths.add(raw_path)

    authorized_counts = Counter(str(row["dataset_id"]) for row in registration)
    facts = P0RGateFacts(
        authorized_by_domain={
            domain: authorized_counts.get(domain, 0) for domain in PRIMARY_COUNTS
        },
        exact_identity_hashes=True,
        author_statement_bound=True,
        global_orientation_rot90=True,
        all_panels_resolved=True,
        processing_provenance_deterministic=True,
        no_unsupported_rotation_reflection=True,
        composed_transform_replayable=True,
        no_result_driven_orientation=True,
        author_evidence_conflict=False,
        processing_provenance_unresolved=False,
    )
    decision = decide_p0r(facts)
    roster_identity = [
        {"dataset_id": row["dataset_id"], "specimen_id": row["specimen_id"]}
        for row in registration
    ]
    registration_identity = [
        {
            "dataset_id": row["dataset_id"],
            "specimen_id": row["specimen_id"],
            "transform_sha256": row["transform_sha256"],
        }
        for row in registration
    ]
    grid_identity = [
        {
            "dataset_id": row["dataset_id"],
            "specimen_id": row["specimen_id"],
            "cell_id": row["cell_id"],
            "surface_box": [
                row["surface_x0"],
                row["surface_y0"],
                row["surface_x1"],
                row["surface_y1"],
            ],
            "transform_sha256": row["transform_sha256"],
        }
        for row in grid_mapping
    ]
    summary: dict[str, object] = {
        "schema_version": 1,
        "stage": "P0R",
        **decision.as_dict(),
        "gate_facts": facts.as_dict(),
        "historical_p0": {
            "logical_path": config["historical_p0"]["path"],
            "status": old_summary["status"],
            "checksums_sha256": config["historical_p0"]["checksums_sha256"],
            "surface_manifest_sha256": config["historical_p0"][
                "surface_manifest_sha256"
            ],
            "preserved_not_reinterpreted": True,
        },
        "author_authority": {
            "logical_path": author_config["artifact_path"],
            "artifact_sha256": author_config["artifact_sha256"],
            "source_type": USER_ATTESTED_SOURCE,
            "statement_sha256": EXPECTED_STATEMENT_SHA256,
            "orientation": "ROT90",
            "mapping_basis": MAPPING_BASIS,
            "original_artifact_sha256": None,
        },
        "processing_provenance": {
            "paired_manifest_path": paired_config["path"],
            "paired_manifest_sha256": paired_config["sha256"],
            "historical_preprocessor_path": preprocessor_config["path"],
            "historical_preprocessor_sha256": preprocessor_config["sha256"],
            "historical_preprocessor_git_commit": "UNAVAILABLE_NO_GIT_METADATA",
            "verified_specimen_count": len(scan_provenance),
            "unique_raw_screenshot_count": len(raw_paths),
            "unique_multi_panel_screenshot_count": len(multi_panel_paths),
            "multi_panel_both_selected_count": sum(
                raw_panel_counts[path] == 2 for path in multi_panel_paths
            ),
            "multi_panel_one_selected_count": sum(
                raw_panel_counts[path] == 1 for path in multi_panel_paths
            ),
            "decoded_rgb_equal_count": sum(
                row["decoded_pixel_equal"] is True for row in scan_provenance
            ),
            "operation": "RGB_DECODE_THEN_AXIS_ALIGNED_CROP",
            "resize": "NONE",
            "interpolation": "NONE",
            "rotation": "IDENTITY",
            "reflection": "NONE",
        },
        "primary_specimen_count": len(old_surface),
        "primary_domain_counts": dict(PRIMARY_COUNTS),
        "authorized_registration_count": len(registration),
        "authorized_by_domain": dict(facts.authorized_by_domain),
        "excluded_specimens": [],
        "authorized_roster_sha256": _canonical_sha256(roster_identity),
        "registration_authority_sha256": _canonical_sha256(
            registration_identity
        ),
        "grid_mapping_sha256": _canonical_sha256(grid_identity),
        "grid_mapping_row_count": len(grid_mapping),
        "action_grid": {"rows": 8, "columns": 8, "cell_ids": list(range(64))},
        "q24_exemplar": q24,
        "physical_mm_used_for_cross_modal_mapping": False,
        "cai_values_accessed": False,
        "oracle_values_accessed": False,
        "damage_targets_used_for_orientation": False,
        "target_domain_labels_used_for_orientation": False,
        "manual_target_alignment_used": False,
        "new_training": False,
        "source_revalidation_required_for_full_replay": True,
    }
    report = (
        "# Agentic NDE P0R Author Registration Audit\n\n"
        f"Decision: `{decision.status.value}`.\n\n"
        f"The historical P0 remains `{old_summary['status']}`. The later "
        "user-attested author statement fixes one global clockwise 90-degree "
        "surface-to-scan orientation and normalized full-frame correspondence.\n\n"
        f"Exact processing replay and registration succeeded for {len(registration)}/"
        f"{len(old_surface)} frozen primary specimens across six domains. All "
        f"{len(grid_mapping)} registered 8x8 cell mappings passed numeric round-trip "
        "checks. The Q24-7 exemplar was verified separately and is not part of the "
        "frozen 276-specimen roster.\n\n"
        "No physical cross-modal calibration, CAI value, oracle value, damage target, "
        "target-domain label, or manual target alignment was used. No model was "
        "trained.\n"
    )
    if old_root != project_root / config["historical_p0"]["path"]:
        raise P0RPipelineError("historical P0 logical path resolution changed")
    return P0RComputation(
        author_authority=author,
        surface_manifest=tuple(surface_manifest),
        scan_processing_provenance=tuple(scan_provenance),
        registration=tuple(registration),
        registration_qc=tuple(registration_qc),
        grid_mapping_qc=tuple(grid_mapping),
        summary=summary,
        report=report,
    )


def _resolved_root(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_dir():
        raise P0RPipelineError(f"{label} must be an explicit existing directory")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise P0RPipelineError(f"{label} is unavailable") from error
    if not resolved.is_dir():
        raise P0RPipelineError(f"{label} is unavailable")
    return resolved


def audit_p0r(
    *,
    config_path: str | Path,
    surface_root: str | Path,
    output: str | Path,
    project_root: str | Path,
) -> Path:
    """Run P0R without training or reading any downstream target."""

    external = _resolved_root(surface_root, "surface root")
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise P0RPipelineError("P0R output must not already exist")
    project = _resolved_root(project_root, "project root")
    config, config_text = _load_config(Path(config_path))
    computation = _compute(
        config=config,
        external_root=external,
        project_root=project,
    )
    result = write_p0r_package(
        destination,
        config_text=config_text,
        author_authority=computation.author_authority,
        surface_manifest=computation.surface_manifest,
        scan_processing_provenance=computation.scan_processing_provenance,
        registration=computation.registration,
        registration_qc=computation.registration_qc,
        grid_mapping_qc=computation.grid_mapping_qc,
        summary=computation.summary,
        report=computation.report,
    )
    revalidate_p0r_sources(
        result,
        surface_root=external,
        project_root=project,
    )
    return result


def _read_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise P0RPipelineError(f"{label} cannot be read") from error


def _normalized_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    result = [{field: str(value) for field, value in row.items()} for row in rows]
    return sorted(
        result,
        key=lambda row: json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ),
    )


def revalidate_p0r_sources(
    path: str | Path,
    *,
    surface_root: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Recompute every P0R row from the bound external and project sources."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise P0RPipelineError("P0R package is unavailable for source replay")
    external = _resolved_root(surface_root, "surface root")
    project = _resolved_root(project_root, "project root")
    config, _ = _load_config(root / "config.yaml")
    expected = _compute(
        config=config,
        external_root=external,
        project_root=project,
    )
    tables = {
        "surface_manifest.csv": expected.surface_manifest,
        "scan_processing_provenance.csv": expected.scan_processing_provenance,
        "registration.csv": expected.registration,
        "registration_qc.csv": expected.registration_qc,
        "grid_mapping_qc.csv": expected.grid_mapping_qc,
    }
    for name, expected_rows in tables.items():
        actual_rows = _read_csv(root / name, f"P0R {name}")
        if _normalized_rows(actual_rows) != _normalized_rows(expected_rows):
            raise P0RPipelineError(f"P0R source replay changed {name}")
    if _read_json(root / "author_authority.json", "P0R author authority") != expected.author_authority:
        raise P0RPipelineError("P0R source replay changed author authority")
    if _read_json(root / "summary.json", "P0R summary") != expected.summary:
        raise P0RPipelineError("P0R source replay changed summary")
    try:
        report = (root / "REPORT.md").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise P0RPipelineError("P0R report cannot be read") from error
    if report != expected.report:
        raise P0RPipelineError("P0R source replay changed report")
    return dict(expected.summary)


__all__ = [
    "P0RPipelineError",
    "audit_p0r",
    "revalidate_p0r_sources",
]
