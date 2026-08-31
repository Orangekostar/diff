"""Path-safe file identity records for external P0 authorities."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path


class AuthorityError(ValueError):
    """Raised when an external authority cannot be bound safely."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    logical_root: str
    relative_path: str
    size: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "logical_root": self.logical_root,
            "relative_path": self.relative_path,
            "size": self.size,
            "sha256": self.sha256,
        }


def _descriptor_unchanged(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _reject_symlink_chain(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current /= part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise AuthorityError("authority path must be a regular file under a regular root")
        except FileNotFoundError as error:
            raise AuthorityError("authority path must be an existing regular file") from error


def snapshot_file(
    path: str | Path,
    *,
    root: str | Path,
    logical_root: str,
    max_bytes: int,
    expected_sha256: str | None = None,
) -> FileSnapshot:
    """Bind a regular file to a logical root without leaking an absolute path."""

    candidate = Path(path)
    authority_root = Path(root)
    if (
        type(logical_root) is not str
        or not logical_root
        or Path(logical_root).is_absolute()
        or type(max_bytes) is not int
        or max_bytes < 0
        or (
            expected_sha256 is not None
            and (
                type(expected_sha256) is not str
                or len(expected_sha256) != 64
                or set(expected_sha256) - set("0123456789abcdef")
            )
        )
    ):
        raise AuthorityError("authority snapshot arguments are invalid")
    if authority_root.is_symlink():
        raise AuthorityError("authority path must be a regular file under a regular root")
    try:
        resolved_root = authority_root.resolve(strict=True)
    except OSError as error:
        raise AuthorityError("authority path must be an existing regular file") from error
    if not resolved_root.is_dir():
        raise AuthorityError("authority path must be a regular file under a regular root")
    lexical = candidate if candidate.is_absolute() else resolved_root / candidate
    lexical = Path(os.path.abspath(lexical))
    try:
        relative = lexical.relative_to(resolved_root)
    except ValueError as error:
        raise AuthorityError("authority file is outside the declared root") from error
    _reject_symlink_chain(resolved_root, relative)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lexical, flags)
    except OSError as error:
        raise AuthorityError("authority path must be an existing regular file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AuthorityError("authority path must be a regular file under a regular root")
        if before.st_size > max_bytes:
            raise AuthorityError("authority file exceeds the maximum permitted size")
        digest = hashlib.sha256()
        consumed = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, max_bytes - consumed + 1))
            if not block:
                break
            consumed += len(block)
            if consumed > max_bytes:
                raise AuthorityError("authority file exceeds the maximum permitted size")
            digest.update(block)
        after = os.fstat(descriptor)
        if consumed != before.st_size or not _descriptor_unchanged(before, after):
            raise AuthorityError("authority file changed while it was being read")
    finally:
        os.close(descriptor)
    sha256 = digest.hexdigest()
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise AuthorityError("authority file SHA-256 does not match the expected identity")
    return FileSnapshot(
        logical_root=logical_root,
        relative_path=relative.as_posix(),
        size=consumed,
        sha256=sha256,
    )


__all__ = ["AuthorityError", "FileSnapshot", "snapshot_file"]
