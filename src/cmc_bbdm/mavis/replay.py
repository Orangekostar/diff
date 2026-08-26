"""Deterministic regeneration and byte audit for frozen MAVIS evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .final_package import finalize_final_package, verify_final_package


class MAVISReplayError(RuntimeError):
    """Raised when replay differs from frozen formal evidence."""


_GIT_BLOB_LIMIT = 100 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _same_bytes(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(1024 * 1024)
            right_chunk = right.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def compare_package_directories(
    formal_path: str | Path,
    replay_path: str | Path,
) -> dict[str, object]:
    try:
        formal = Path(formal_path).resolve(strict=True)
        replay = Path(replay_path).resolve(strict=True)
    except OSError as error:
        raise MAVISReplayError("replay package is unavailable") from error
    if not formal.is_dir() or not replay.is_dir() or formal == replay:
        raise MAVISReplayError("replay directories are invalid")
    formal_files = _files(formal)
    replay_files = _files(replay)
    if set(formal_files) != set(replay_files):
        raise MAVISReplayError("replay file roster changed")
    rows = []
    for name in sorted(formal_files):
        first = formal_files[name]
        second = replay_files[name]
        if not _same_bytes(first, second):
            raise MAVISReplayError(f"replay file content changed: {name}")
        rows.append((name, first.stat().st_size, _sha256(first)))
    state = hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "byte_identical": True,
        "file_count": len(rows),
        "total_bytes": sum(row[1] for row in rows),
        "tree_state_sha256": state,
    }


def run_final_replay(
    config_path: str | Path,
    *,
    project_root: str | Path,
    formal_package: str | Path,
    replay_root: str | Path,
    worker_root: str | Path,
    p1_package: str | Path,
    p4_package: str | Path,
    p5_package: str | Path,
    p6_package: str | Path,
    bootstrap_replicates: int,
) -> Path:
    root = Path(project_root).resolve(strict=True)
    formal = Path(formal_package).resolve(strict=True)
    verify_final_package(formal)
    destination = Path(replay_root)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if root not in destination.parents or destination.exists():
        raise MAVISReplayError("replay destination is invalid or already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".mavis_replay.", dir=destination.parent))
    try:
        replay_package = temporary / "p7_final_frozen_eval"
        finalize_final_package(
            config_path,
            project_root=root,
            worker_root=worker_root,
            p1_package=p1_package,
            p4_package=p4_package,
            p5_package=p5_package,
            p6_package=p6_package,
            bootstrap_replicates=bootstrap_replicates,
            output_path=replay_package,
        )
        comparison = compare_package_directories(formal, replay_package)
        formal_relative = formal.relative_to(root).as_posix()
        replay_relative = (destination / "p7_final_frozen_eval").relative_to(
            root
        ).as_posix()
        summary = {
            "schema_version": 1,
            "status": "COMPLETE",
            "formal_package": formal_relative,
            "replay_package": replay_relative,
            **comparison,
        }
        _write_json(temporary / "summary.json", summary)
        scientific_files = sorted(
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*")
            if path.is_file() and path != temporary / "artifact_manifest.json"
        )
        manifest = {
            "schema_version": 1,
            "artifact": "mavis_final_replay",
            "tree_state_sha256": comparison["tree_state_sha256"],
            "byte_identical": True,
            "files": {
                name: {
                    "bytes": (temporary / name).stat().st_size,
                    "sha256": _sha256(temporary / name),
                }
                for name in scientific_files
            },
        }
        _write_json(temporary / "artifact_manifest.json", manifest)
        ledger_files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "CHECKSUMS.sha256").write_text(
            "".join(
                f"{_sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in ledger_files
            ),
            encoding="ascii",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    verify_replay_package(destination, project_root=root)
    return destination


def verify_replay_package(
    path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    root = Path(path)
    project = Path(project_root).resolve(strict=True)
    try:
        manifest = json.loads(
            (root / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        lines = (root / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MAVISReplayError("replay metadata is invalid") from error
    if (
        type(manifest) is not dict
        or type(manifest.get("files")) is not dict
        or manifest.get("byte_identical") is not True
        or type(summary) is not dict
        or summary.get("status") != "COMPLETE"
        or summary.get("byte_identical") is not True
        or manifest.get("tree_state_sha256") != summary.get("tree_state_sha256")
    ):
        raise MAVISReplayError("replay state is invalid")
    expected = set(manifest["files"]) | {"artifact_manifest.json"}
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item != root / "CHECKSUMS.sha256"
    }
    ledger = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise MAVISReplayError("replay checksum ledger is invalid") from error
        ledger[name] = digest
    if actual != expected or set(ledger) != expected:
        raise MAVISReplayError("replay manifest roster changed")
    for name, digest in ledger.items():
        if not _is_digest(digest) or _sha256(root / name) != digest:
            raise MAVISReplayError(f"replay checksum mismatch: {name}")
    for name, metadata in manifest["files"].items():
        file_path = root / name
        if (
            type(metadata) is not dict
            or metadata.get("bytes") != file_path.stat().st_size
            or metadata.get("sha256") != _sha256(file_path)
            or file_path.stat().st_size >= _GIT_BLOB_LIMIT
        ):
            raise MAVISReplayError(f"replay manifest mismatch: {name}")
    try:
        formal = (project / summary["formal_package"]).resolve(strict=True)
        replay = (project / summary["replay_package"]).resolve(strict=True)
    except (KeyError, OSError, TypeError) as error:
        raise MAVISReplayError("replay package links are invalid") from error
    comparison = compare_package_directories(formal, replay)
    if comparison != {
        key: summary[key]
        for key in ("byte_identical", "file_count", "total_bytes", "tree_state_sha256")
    }:
        raise MAVISReplayError("replay comparison summary changed")
    return manifest


def _is_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "MAVISReplayError",
    "compare_package_directories",
    "run_final_replay",
    "verify_replay_package",
]
