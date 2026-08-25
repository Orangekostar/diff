"""Deterministic manifest-only artifacts for the causal MAVIS authority."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from .authority import MAVISAuthority, MAVISAuthorityError, _is_sha256


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_manifest(authority: MAVISAuthority) -> str:
    output = io.StringIO(newline="")
    fields = (
        "specimen_id",
        "dataset_id",
        "height",
        "width",
        "native_count",
        "source_image_sha256",
        "decoded_image_sha256",
        "policy_context_state_sha256",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for index, specimen_id in enumerate(authority.specimen_ids):
        context = authority.policy_context(specimen_id)
        writer.writerow(
            {
                "specimen_id": specimen_id,
                "dataset_id": authority.dataset_ids[index],
                "height": context.native_shape[0],
                "width": context.native_shape[1],
                "native_count": context.native_count,
                "source_image_sha256": authority.source_image_sha256[index],
                "decoded_image_sha256": authority.decoded_image_sha256[index],
                "policy_context_state_sha256": context.state_sha256,
            }
        )
    return output.getvalue()


def write_mavis_authority_package(
    output_directory: str | Path,
    authority: MAVISAuthority,
    *,
    config_sha256: str,
) -> Path:
    if type(authority) is not MAVISAuthority or not _is_sha256(config_sha256):
        raise MAVISAuthorityError("authority package inputs are invalid")
    output = Path(output_directory)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise MAVISAuthorityError("authority output directory is unavailable") from error
    scan_path = output / "scan_manifest.csv"
    report_path = output / "REPORT.md"
    scan_path.write_text(_scan_manifest(authority), encoding="utf-8", newline="")
    domains = tuple(dict.fromkeys(authority.dataset_ids))
    report_path.write_text(
        "# MAVIS Causal Authority\n\n"
        f"- Specimens: `{authority.specimen_count}`\n"
        f"- Domains: `{len(domains)}`\n"
        "- Storage: upstream-bound hash manifest; scan payloads are not duplicated.\n"
        "- Visibility: policy contexts contain 34 surface/context values; privileged "
        "CAI values and complete scans are absent from this package.\n"
        "- Reveal: native RGB values are loaded from the registered source and exposed "
        "only for legal acquired coordinates.\n",
        encoding="utf-8",
    )
    file_records = {
        path.name: {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        for path in (report_path, scan_path)
    }
    manifest = {
        "schema_version": 1,
        "authority_state_sha256": authority.state_sha256,
        "source_authority_sha256": authority.source_authority_sha256,
        "config_sha256": config_sha256,
        "specimen_count": authority.specimen_count,
        "domain_order": list(domains),
        "files": file_records,
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_paths = (report_path, manifest_path, scan_path)
    (output / "CHECKSUMS.sha256").write_text(
        "".join(f"{_file_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="ascii",
    )
    verify_mavis_authority_package(output)
    return output


def verify_mavis_authority_package(directory: str | Path) -> None:
    root = Path(directory)
    expected_names = {
        "CHECKSUMS.sha256",
        "REPORT.md",
        "artifact_manifest.json",
        "scan_manifest.csv",
    }
    try:
        actual_names = {path.name for path in root.iterdir() if path.is_file()}
    except OSError as error:
        raise MAVISAuthorityError("authority package is unavailable") from error
    if actual_names != expected_names:
        raise MAVISAuthorityError("authority package file roster changed")
    checksum_rows = (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    if len(checksum_rows) != 3:
        raise MAVISAuthorityError("authority checksum roster changed")
    checksum_names: list[str] = []
    for row in checksum_rows:
        try:
            expected, name = row.split("  ", maxsplit=1)
        except ValueError as error:
            raise MAVISAuthorityError("authority checksum row is invalid") from error
        if name not in expected_names - {"CHECKSUMS.sha256"} or not _is_sha256(
            expected
        ):
            raise MAVISAuthorityError("authority checksum row is invalid")
        checksum_names.append(name)
        if _file_sha256(root / name) != expected:
            raise MAVISAuthorityError("authority package checksum changed")
    if set(checksum_names) != expected_names - {"CHECKSUMS.sha256"} or len(
        set(checksum_names)
    ) != len(checksum_names):
        raise MAVISAuthorityError("authority checksum roster changed")
    try:
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise MAVISAuthorityError("authority manifest is invalid") from error
    if set(manifest) != {
        "schema_version",
        "authority_state_sha256",
        "source_authority_sha256",
        "config_sha256",
        "specimen_count",
        "domain_order",
        "files",
    } or manifest["schema_version"] != 1:
        raise MAVISAuthorityError("authority manifest schema changed")
    if any(
        not _is_sha256(manifest[name])
        for name in (
            "authority_state_sha256",
            "source_authority_sha256",
            "config_sha256",
        )
    ):
        raise MAVISAuthorityError("authority manifest hash is invalid")
    files = manifest["files"]
    if type(files) is not dict or set(files) != {"REPORT.md", "scan_manifest.csv"}:
        raise MAVISAuthorityError("authority manifest file roster changed")
    for name, record in files.items():
        path = root / name
        if (
            type(record) is not dict
            or set(record) != {"bytes", "sha256"}
            or record["bytes"] != path.stat().st_size
            or record["sha256"] != _file_sha256(path)
        ):
            raise MAVISAuthorityError("authority manifest file binding changed")


__all__ = ["verify_mavis_authority_package", "write_mavis_authority_package"]
