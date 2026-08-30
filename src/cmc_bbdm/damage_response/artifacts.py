from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cmc_bbdm.damage_response.authority import AuthorityError, snapshot_file

REQUIRED_P0_PAYLOADS = (
    "REPORT.md",
    "pairing_manifest.csv",
    "post_cai_image_manifest.csv",
    "published_peak_reconciliation.csv",
    "raw_trace_qc.csv",
    "source_hashes.csv",
    "strain_unit_audit.csv",
    "summary.json",
)
_MANIFEST_NAME = "artifact_manifest.json"
_CHECKSUM_NAME = "CHECKSUMS.sha256"
_PACKAGE_MEMBERS = frozenset(
    {*REQUIRED_P0_PAYLOADS, _MANIFEST_NAME, _CHECKSUM_NAME}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ArtifactError(RuntimeError):
    """Raised when a P0 artifact package is incomplete or has drifted."""


@dataclass(frozen=True)
class ReplayReport:
    payload_count: int
    verified: bool


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("ascii")


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_p0_package(destination: Path, payloads: Mapping[str, bytes]) -> None:
    """Write one deterministic P0 package through an atomic sibling rename."""

    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise ArtifactError(f"P0 output already exists: {output.name}")
    observed_names = set(payloads)
    required_names = set(REQUIRED_P0_PAYLOADS)
    if observed_names != required_names:
        raise ArtifactError(
            "P0 payload membership differs; "
            f"missing={sorted(required_names - observed_names)!r}; "
            f"extra={sorted(observed_names - required_names)!r}"
        )
    normalized: dict[str, bytes] = {}
    for name in REQUIRED_P0_PAYLOADS:
        payload = payloads[name]
        if not isinstance(payload, bytes):
            raise ArtifactError(f"P0 payload must be bytes: {name}")
        normalized[name] = payload

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    renamed = False
    try:
        manifest_files: list[dict[str, object]] = []
        checksums: dict[str, str] = {}
        for name in REQUIRED_P0_PAYLOADS:
            payload = normalized[name]
            digest = hashlib.sha256(payload).hexdigest()
            _write_exclusive(temporary / name, payload)
            manifest_files.append(
                {"path": name, "sha256": digest, "size": len(payload)}
            )
            checksums[name] = digest

        manifest = _canonical_json(
            {"files": manifest_files, "schema_version": 1}
        )
        _write_exclusive(temporary / _MANIFEST_NAME, manifest)
        checksums[_MANIFEST_NAME] = hashlib.sha256(manifest).hexdigest()
        checksum_payload = "".join(
            f"{checksums[name]}  {name}\n" for name in sorted(checksums)
        ).encode("ascii")
        _write_exclusive(temporary / _CHECKSUM_NAME, checksum_payload)
        _fsync_directory(temporary)

        if output.exists() or output.is_symlink():
            raise ArtifactError(f"P0 output already exists: {output.name}")
        try:
            os.rename(temporary, output)
        except OSError as error:
            raise ArtifactError(f"could not atomically publish P0 output: {output.name}") from error
        renamed = True
        _fsync_directory(output.parent)
    finally:
        if not renamed and temporary.exists():
            shutil.rmtree(temporary)


def _regular_package_members(root: Path) -> set[str]:
    try:
        root_info = root.lstat()
    except OSError as error:
        raise ArtifactError("P0 package is not accessible") from error
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise ArtifactError("P0 package root must be a regular directory")

    names: set[str] = set()
    try:
        entries = list(os.scandir(root))
    except OSError as error:
        raise ArtifactError("P0 package directory cannot be listed") from error
    for entry in entries:
        names.add(entry.name)
        if not entry.is_file(follow_symlinks=False):
            raise ArtifactError(f"P0 member must be a regular file: {entry.name}")
    return names


def _snapshot(root: Path, name: str, *, max_bytes: int):
    try:
        return snapshot_file(
            root / name,
            max_bytes=max_bytes,
            logical_source="artifact:p0_data_audit",
            relative_path=name,
        )
    except AuthorityError as error:
        raise ArtifactError(str(error)) from error


def _read_small_regular(root: Path, name: str, *, max_bytes: int) -> bytes:
    record = _snapshot(root, name, max_bytes=max_bytes)
    try:
        payload = (root / name).read_bytes()
    except OSError as error:
        raise ArtifactError(f"P0 member cannot be read: {name}") from error
    if len(payload) != record.size or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise ArtifactError(f"P0 member changed during replay: {name}")
    return payload


def _parse_manifest(payload: bytes) -> dict[str, tuple[int, str]]:
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError("artifact manifest is not canonical JSON") from error
    if not isinstance(value, dict) or set(value) != {"files", "schema_version"}:
        raise ArtifactError("artifact manifest schema drift")
    if value["schema_version"] != 1 or not isinstance(value["files"], list):
        raise ArtifactError("artifact manifest schema drift")

    records: dict[str, tuple[int, str]] = {}
    observed_order: list[str] = []
    for item in value["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            raise ArtifactError("artifact manifest file schema drift")
        name = item["path"]
        size = item["size"]
        digest = item["sha256"]
        if (
            not isinstance(name, str)
            or name not in REQUIRED_P0_PAYLOADS
            or name in records
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise ArtifactError("artifact manifest contains an invalid file record")
        records[name] = (size, digest)
        observed_order.append(name)
    if set(records) != set(REQUIRED_P0_PAYLOADS):
        raise ArtifactError("artifact manifest payload membership differs")
    if observed_order != list(REQUIRED_P0_PAYLOADS):
        raise ArtifactError("artifact manifest ordering drift")
    return records


def replay_p0(path: Path) -> ReplayReport:
    """Verify exact package membership, manifest records, and all checksums."""

    root = Path(path)
    observed_members = _regular_package_members(root)
    if observed_members != _PACKAGE_MEMBERS:
        raise ArtifactError(
            "P0 package membership differs; "
            f"missing={sorted(_PACKAGE_MEMBERS - observed_members)!r}; "
            f"extra={sorted(observed_members - _PACKAGE_MEMBERS)!r}"
        )

    manifest_payload = _read_small_regular(
        root, _MANIFEST_NAME, max_bytes=1024 * 1024
    )
    records = _parse_manifest(manifest_payload)
    observed_hashes: dict[str, str] = {}
    for name in REQUIRED_P0_PAYLOADS:
        expected_size, expected_hash = records[name]
        snapshot = _snapshot(root, name, max_bytes=max(1, expected_size))
        if snapshot.size != expected_size:
            raise ArtifactError(
                f"P0 payload size drift for {name}: "
                f"expected {expected_size}, observed {snapshot.size}"
            )
        if snapshot.sha256 != expected_hash:
            raise ArtifactError(f"P0 payload SHA-256 drift for {name}")
        observed_hashes[name] = snapshot.sha256
    observed_hashes[_MANIFEST_NAME] = hashlib.sha256(manifest_payload).hexdigest()

    expected_checksums = "".join(
        f"{observed_hashes[name]}  {name}\n" for name in sorted(observed_hashes)
    ).encode("ascii")
    checksum_payload = _read_small_regular(
        root, _CHECKSUM_NAME, max_bytes=1024 * 1024
    )
    if checksum_payload != expected_checksums:
        raise ArtifactError("CHECKSUMS.sha256 content drift")
    return ReplayReport(payload_count=len(records), verified=True)
