from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class AuthorityError(RuntimeError):
    """Raised when an external source cannot be snapshotted immutably."""


@dataclass(frozen=True)
class FileSnapshot:
    logical_source: str
    relative_path: str
    size: int
    sha256: str


def _validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("manifest identity must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
        raise ValueError("manifest identity must be a safe relative path")
    return path.as_posix()


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def snapshot_file(
    path: Path,
    *,
    max_bytes: int,
    logical_source: str = "local:external",
    relative_path: str | None = None,
) -> FileSnapshot:
    """Hash one external regular file without serializing its runtime path."""

    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if not isinstance(logical_source, str) or not logical_source.strip():
        raise ValueError("logical_source must be a nonempty label")

    runtime_path = Path(path)
    manifest_path = _validate_relative_path(
        runtime_path.name if relative_path is None else relative_path
    )
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise AuthorityError("platform cannot guarantee a regular file without symlinks")
    flags = os.O_RDONLY | no_follow
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    try:
        descriptor = os.open(runtime_path, flags)
    except OSError as error:
        raise AuthorityError(
            f"source must be an accessible regular file: {manifest_path}"
        ) from error

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuthorityError(f"source must be a regular file: {manifest_path}")
        if before.st_size > max_bytes:
            raise AuthorityError(
                f"source exceeds maximum size of {max_bytes} bytes: {manifest_path}"
            )

        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - bytes_read))
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                raise AuthorityError(
                    f"source exceeds maximum size of {max_bytes} bytes: {manifest_path}"
                )
            digest.update(chunk)

        after = os.fstat(descriptor)
        if _identity(before) != _identity(after) or bytes_read != before.st_size:
            raise AuthorityError(f"source changed during snapshot: {manifest_path}")
    finally:
        os.close(descriptor)

    return FileSnapshot(
        logical_source=logical_source.strip(),
        relative_path=manifest_path,
        size=bytes_read,
        sha256=digest.hexdigest(),
    )
