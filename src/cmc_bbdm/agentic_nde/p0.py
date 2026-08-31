"""P0 surface/C-scan authority and registration audit orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .artifacts import write_p0_package
from .authority import FileSnapshot, snapshot_file
from .contracts import PRIMARY_COUNTS, P0GateFacts, decide_p0
from .frozen_bindings import bind_frozen_a2
from .grid import Grid8x8
from .surface_qc import inspect_surface


class PipelineError(ValueError):
    """Raised when P0 inputs are unavailable, unsafe, or inconsistent."""


_REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "stage",
    "controlling_prompt_sha256",
    "repository_base_sha",
    "dataset",
    "primary_counts",
    "source_authorities",
    "surface_qc",
    "registration",
    "status_vocabulary",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise PipelineError("P0 config is unavailable")
    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PipelineError("P0 config is invalid") from error
    if (
        type(payload) is not dict
        or set(payload) != _REQUIRED_CONFIG_KEYS
        or payload["schema_version"] != 1
        or payload["stage"] != "P0_SURFACE_CSCAN_AUTHORITY_AND_REGISTRATION"
        or payload["primary_counts"] != dict(PRIMARY_COUNTS)
    ):
        raise PipelineError("P0 config schema or frozen roster changed")
    return payload, text


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise PipelineError(f"{label} is unavailable")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if fields is None or len(set(fields)) != len(fields):
                raise PipelineError(f"{label} schema is invalid")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise PipelineError(f"{label} cannot be read") from error
    if not rows:
        raise PipelineError(f"{label} contains no rows")
    return rows


def _index(
    rows: list[dict[str, str]], *, domain: str, specimen: str, label: str
) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get(domain, ""), row.get(specimen, ""))
        if not all(key) or key in result:
            raise PipelineError(f"{label} specimen keys are missing or duplicated")
        result[key] = row
    return result


def _bound_file(root: Path, relative: str, expected_sha256: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PipelineError(f"{label} path is unsafe")
    path = root / candidate
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise PipelineError(f"{label} path escapes its authority root") from error
    if path.is_symlink() or not resolved.is_file() or _sha256(resolved) != expected_sha256:
        raise PipelineError(f"{label} SHA-256 changed")
    return resolved


def _external_file(root: Path, raw_path: str, label: str) -> tuple[Path, str]:
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else root / candidate
    if path.is_symlink():
        raise PipelineError(f"{label} is a symlink")
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise PipelineError(f"{label} escapes the external authority root") from error
    if not resolved.is_file():
        raise PipelineError(f"{label} is not a regular file")
    return resolved, relative


def _snapshot_external(
    root: Path, raw_path: str, *, max_bytes: int, label: str
) -> FileSnapshot:
    path, _ = _external_file(root, raw_path, label)
    try:
        return snapshot_file(
            path,
            root=root,
            logical_root="external_hasebe",
            max_bytes=max_bytes,
        )
    except ValueError as error:
        raise PipelineError(f"{label} cannot be bound") from error


def _snapshot_project(root: Path, path: Path, logical_root: str) -> FileSnapshot:
    try:
        return snapshot_file(
            path,
            root=root,
            logical_root=logical_root,
            max_bytes=max(path.stat().st_size, 1),
        )
    except (OSError, ValueError) as error:
        raise PipelineError("compact source authority cannot be bound") from error


def _source_record(snapshot: FileSnapshot, role: str) -> dict[str, object]:
    return {
        "logical_path": f"{snapshot.logical_root}/{snapshot.relative_path}",
        "logical_root": snapshot.logical_root,
        "relative_path": snapshot.relative_path,
        "role": role,
        "bytes": snapshot.size,
        "sha256": snapshot.sha256,
    }


def _add_source(
    records: dict[str, dict[str, object]], snapshot: FileSnapshot, role: str
) -> None:
    row = _source_record(snapshot, role)
    key = str(row["logical_path"])
    existing = records.get(key)
    if existing is not None:
        if existing["sha256"] != row["sha256"] or existing["bytes"] != row["bytes"]:
            raise PipelineError("one logical source path has conflicting identities")
        roles = set(str(existing["role"]).split(";")) | {role}
        existing["role"] = ";".join(sorted(roles))
        return
    records[key] = row


def _int(row: Mapping[str, str], field: str, label: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise PipelineError(f"{label} has an invalid {field}") from error


def _project_authorities(
    project_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    paths: dict[str, Path] = {}
    records: list[dict[str, object]] = []
    for name, authority in config["source_authorities"].items():
        if authority["logical_root"] != "compact_repository":
            continue
        path = _bound_file(
            project_root,
            authority["path"],
            authority["sha256"],
            name,
        )
        paths[name] = path
        records.append(
            _source_record(
                _snapshot_project(project_root, path, "compact_repository"),
                name,
            )
        )
    return paths, records


def _external_authorities(
    surface_root: Path, config: Mapping[str, Any], maximum: int
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    paths: dict[str, Path] = {}
    records: list[dict[str, object]] = []
    for name, authority in config["source_authorities"].items():
        if authority["logical_root"] != "external_hasebe":
            continue
        path = _bound_file(
            surface_root,
            authority["path"],
            authority["sha256"],
            name,
        )
        paths[name] = path
        records.append(
            _source_record(
                snapshot_file(
                    path,
                    root=surface_root,
                    logical_root="external_hasebe",
                    max_bytes=maximum,
                ),
                name,
            )
        )
    return paths, records


def _grid_sha256() -> str:
    rows = Grid8x8(width=75.0, height=75.0).render_records()
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def audit_p0(
    *,
    config_path: str | Path,
    surface_root: str | Path,
    output: str | Path,
    project_root: str | Path,
) -> Path:
    """Run the complete no-training P0 identity and registration audit."""

    external_root = Path(surface_root)
    destination = Path(output)
    if external_root.is_symlink() or not external_root.is_dir():
        raise PipelineError("surface root must be an explicit existing directory")
    if destination.exists() or destination.is_symlink():
        raise PipelineError("P0 output must not already exist")
    try:
        external_root = external_root.resolve(strict=True)
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise PipelineError("project or surface authority root is unavailable") from error
    if not root.is_dir():
        raise PipelineError("project root is unavailable")

    config, config_text = _load_config(Path(config_path))
    maximum = int(config["surface_qc"]["maximum_file_bytes"])
    external_paths, external_source_rows = _external_authorities(
        external_root, config, maximum
    )
    project_paths, project_source_rows = _project_authorities(root, config)

    p0_rows = _read_csv(external_paths["external_p0_roster"], "external P0 roster")
    paired_rows = _read_csv(
        external_paths["external_paired_manifest"], "external paired manifest"
    )
    cohort_rows = _read_csv(project_paths["compact_cohort"], "compact cohort")
    scan_rows = _read_csv(
        project_paths["compact_scan_manifest"], "compact scan manifest"
    )
    p0_index = _index(
        p0_rows,
        domain="source_dataset",
        specimen="specimen_uid",
        label="external P0 roster",
    )
    paired_index = _index(
        paired_rows,
        domain="dataset_id",
        specimen="sample_id",
        label="external paired manifest",
    )
    cohort_index = _index(
        cohort_rows,
        domain="dataset_id",
        specimen="specimen_id",
        label="compact cohort",
    )
    scan_index = _index(
        scan_rows,
        domain="dataset_id",
        specimen="specimen_id",
        label="compact scan manifest",
    )
    counts = Counter(domain for domain, _ in cohort_index)
    if dict(counts) != dict(PRIMARY_COUNTS) or len(cohort_index) != 276:
        raise PipelineError("compact cohort no longer matches the frozen primary roster")
    cohort_keys = sorted(
        cohort_index,
        key=lambda key: (tuple(PRIMARY_COUNTS).index(key[0]), key[1]),
    )
    if any(key not in p0_index or key not in paired_index or key not in scan_index for key in cohort_keys):
        raise PipelineError("one or more frozen specimens lack exact authority binding")

    source_records = {
        str(row["logical_path"]): row
        for row in (*external_source_rows, *project_source_rows)
    }
    surface_manifest: list[dict[str, object]] = []
    surface_qc: list[dict[str, object]] = []
    registration: list[dict[str, object]] = []
    registration_qc: list[dict[str, object]] = []
    corrected_cscan_semantics = 0

    for key in cohort_keys:
        domain, specimen = key
        p0_row = p0_index[key]
        paired = paired_index[key]
        cohort = cohort_index[key]
        compact_scan = scan_index[key]

        surface_snapshot = _snapshot_external(
            external_root,
            paired["source_path"],
            max_bytes=maximum,
            label=f"surface image {domain}/{specimen}",
        )
        p0_surface_path, p0_surface_relative = _external_file(
            external_root,
            p0_row["surface_source_file"],
            f"P0 surface image {domain}/{specimen}",
        )
        if (
            p0_surface_path != external_root / surface_snapshot.relative_path
            or p0_surface_relative != surface_snapshot.relative_path
            or p0_row["surface_source_sha256"] != surface_snapshot.sha256
            or paired["source_sha256"] != surface_snapshot.sha256
        ):
            raise PipelineError(f"surface authority mismatch for {domain}/{specimen}")

        screenshot_snapshot = _snapshot_external(
            external_root,
            paired["target_screenshot_path"],
            max_bytes=maximum,
            label=f"raw C-scan screenshot {domain}/{specimen}",
        )
        p0_cscan_path, p0_cscan_relative = _external_file(
            external_root,
            p0_row["cscan_source_file"],
            f"P0 C-scan path {domain}/{specimen}",
        )
        if (
            p0_cscan_path != external_root / screenshot_snapshot.relative_path
            or p0_cscan_relative != screenshot_snapshot.relative_path
            or paired["target_screenshot_sha256"] != screenshot_snapshot.sha256
        ):
            raise PipelineError(f"raw C-scan authority mismatch for {domain}/{specimen}")

        crop_snapshot = _snapshot_external(
            external_root,
            paired["target_path"],
            max_bytes=maximum,
            label=f"registered C-scan crop {domain}/{specimen}",
        )
        if (
            paired["target_sha256"] != crop_snapshot.sha256
            or p0_row["cscan_source_sha256"] != crop_snapshot.sha256
            or cohort["image_sha256"] != crop_snapshot.sha256
            or compact_scan["source_image_sha256"] != crop_snapshot.sha256
            or _int(paired, "target_width", "paired manifest")
            != _int(cohort, "native_width", "compact cohort")
            or _int(paired, "target_height", "paired manifest")
            != _int(cohort, "native_height", "compact cohort")
        ):
            raise PipelineError(f"registered C-scan authority mismatch for {domain}/{specimen}")
        if p0_row["cscan_source_sha256"] != screenshot_snapshot.sha256:
            corrected_cscan_semantics += 1

        qc = inspect_surface(
            external_root / surface_snapshot.relative_path,
            expected_sha256=surface_snapshot.sha256,
        )
        _add_source(source_records, surface_snapshot, "impacted_surface")
        _add_source(source_records, screenshot_snapshot, "raw_cscan_screenshot")
        _add_source(source_records, crop_snapshot, "registered_cscan_crop")
        cai_identity = (
            f"{domain}:v{p0_row['source_version']}:"
            f"compression_after_impact_row:{p0_row['cai_source_row']}"
        )
        surface_manifest.append(
            {
                "specimen_id": specimen,
                "dataset_id": domain,
                "dataset_version": p0_row["source_version"],
                "impacted_surface_path": surface_snapshot.relative_path,
                "surface_sha256": surface_snapshot.sha256,
                "surface_bytes": surface_snapshot.size,
                "cscan_source_path": screenshot_snapshot.relative_path,
                "cscan_source_sha256": screenshot_snapshot.sha256,
                "registered_cscan_crop_path": crop_snapshot.relative_path,
                "registered_cscan_crop_sha256": crop_snapshot.sha256,
                "registered_cscan_width_px": cohort["native_width"],
                "registered_cscan_height_px": cohort["native_height"],
                "cscan_panel_index": paired["target_panel_index"],
                "cai_identity": cai_identity,
                "pairing_method": p0_row["pairing_method"],
                "pairing_confidence": p0_row["pairing_confidence"],
                "identity_status": "PASS_EXACT_SPECIMEN_ID_AND_HASH",
            }
        )
        surface_qc.append(
            {
                "specimen_id": specimen,
                "dataset_id": domain,
                **qc.as_dict(),
                "visible_annotations": "UNKNOWN_NOT_MACHINE_VERIFIED",
                "specimen_boundary_visibility": "UNKNOWN_NOT_MACHINE_VERIFIED",
                "published_specimen_width_mm": config["dataset"]["specimen_extent_mm"][0],
                "published_specimen_height_mm": config["dataset"]["specimen_extent_mm"][1],
                "image_frame_physical_extent_status": "UNRESOLVED_NO_EXPORT_TRANSFORM",
            }
        )
        registration.append(
            {
                "specimen_id": specimen,
                "dataset_id": domain,
                "status": "UNRESOLVED",
                "evidence_class": "NONE_AUTHORIZED",
                "source_frame": "IMPACTED_SURFACE_NATIVE_RASTER",
                "cscan_frame": "REGISTERED_CSCAN_CROP_NATIVE_RASTER",
                "orientation": "UNRESOLVED_8_WAY_AMBIGUITY",
                "scale_x": "",
                "scale_y": "",
                "offset_x": "",
                "offset_y": "",
                "transform_sha256": "",
                "reason": "NO_SOURCE_SUPPORTED_CROSS_INSTRUMENT_TRANSFORM",
            }
        )
        registration_qc.append(
            {
                "specimen_id": specimen,
                "dataset_id": domain,
                "authorized": "false",
                "invertible": "NOT_RUN_NO_TRANSFORM",
                "boundary_consistency": "NOT_RUN_NO_TRANSFORM",
                "orientation_ambiguity": "DETECTED_UNRESOLVED",
                "cai_access": "PASS_NOT_ACCESSED",
                "hidden_cscan_content_access": "PASS_NOT_USED_FOR_REGISTRATION",
                "deterministic_transform_replay": "NOT_APPLICABLE",
                "legal_action_grid": "PASS_0_TO_63_ROW_MAJOR",
            }
        )

    if corrected_cscan_semantics != 276:
        raise PipelineError("legacy P0 C-scan hash semantics changed unexpectedly")
    frozen_a2 = bind_frozen_a2(root)
    static_summary = json.loads(
        project_paths["frozen_static_reference_summary"].read_text(encoding="utf-8")
    )
    if static_summary.get("strongest_deployable_baseline") != "mvd_m1_o2":
        raise PipelineError("strongest frozen static reference identity changed")

    facts = P0GateFacts(
        authorized_by_domain={domain: 0 for domain in PRIMARY_COUNTS},
        exact_identity_hashes=True,
        orientation_resolved=False,
        deterministic_transform=False,
        deployable_evidence_only=True,
        replay_verified=False,
    )
    decision = decide_p0(facts)
    summary = {
        "schema_version": 1,
        "stage": "P0",
        **decision.as_dict(),
        "gate_facts": facts.as_dict(),
        "primary_specimen_count": 276,
        "primary_domain_counts": dict(PRIMARY_COUNTS),
        "exact_identity_bound_count": 276,
        "authorized_registration_count": 0,
        "legacy_p0_raw_path_with_crop_hash_count": corrected_cscan_semantics,
        "registration_evidence_classes_audited": [
            "A_DIRECT_METADATA",
            "B_GEOMETRY_ONLY",
            "C_SOURCE_ONLY_LEARNED",
        ],
        "cross_instrument_transform_found": False,
        "normalized_coordinates_accepted_as_correspondence": False,
        "cai_values_accessed": False,
        "hidden_cscan_content_used_for_registration": False,
        "action_grid": {
            "rows": 8,
            "columns": 8,
            "cell_ids": list(range(64)),
            "physical_extent_mm": [75.0, 75.0],
            "grid_sha256": _grid_sha256(),
        },
        "frozen_a2_binding": frozen_a2.as_dict(),
        "strongest_frozen_static_reference": {
            "identity": "mvd_m1_o2",
            "role": "static_reference_not_external_competitor",
            "claim_evidence_path": config["source_authorities"]["frozen_static_reference_claim"]["path"],
            "claim_evidence_sha256": config["source_authorities"]["frozen_static_reference_claim"]["sha256"],
        },
        "new_training": False,
    }
    report = (
        "# Agentic NDE P0 Authority and Registration Audit\n\n"
        f"Decision: `{decision.status.value}`.\n\n"
        "Exact identity and hash binding succeeded for 276 frozen primary specimens "
        "across six domains. The historical P0 roster placed each registered-crop "
        "hash beside a raw screenshot path; the paired manifest was used to bind the "
        "raw screenshot and registered crop separately.\n\n"
        "No source record establishes a shared surface/C-scan coordinate frame, "
        "orientation, offset, or crop transform. Published 80 x 80 mm specimen and "
        "75 x 75 mm scan extents do not resolve the eight-way orientation ambiguity. "
        "Normalized coordinates were not treated as physical correspondence.\n\n"
        "No CAI value, oracle value, hidden C-scan content, target-overlap criterion, "
        "or manual target alignment was used. No model was trained. P1-P4 are "
        "`NOT_RUN_NOT_AUTHORIZED`.\n"
    )
    return write_p0_package(
        destination,
        config_text=config_text,
        surface_manifest=surface_manifest,
        surface_qc=surface_qc,
        registration=registration,
        registration_qc=registration_qc,
        source_hashes=list(source_records.values()),
        summary=summary,
        report=report,
    )


__all__ = ["PipelineError", "audit_p0"]
