"""Deterministic artifact writers for the learned C-scan study."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import polars as pl


def _atomic_write(path: str | Path, writer: Callable[[Path], None]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        writer(temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_text(path: Path, payload: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload)


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)


def write_json_atomic(path: str | Path, payload: object) -> None:
    """Write one indented, sorted JSON document atomically."""

    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _atomic_write(path, lambda temporary: _write_text(temporary, serialized))


def write_jsonl_atomic(
    path: str | Path, rows: tuple[dict[str, object], ...]
) -> None:
    """Write compact, sorted JSON objects one per line atomically."""

    serialized_rows: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("JSONL rows must be dictionaries")
        serialized_rows.append(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
    serialized = "".join(serialized_rows)
    _atomic_write(path, lambda temporary: _write_text(temporary, serialized))


def write_csv_atomic(
    path: str | Path,
    rows: tuple[dict[str, object], ...],
    fieldnames: tuple[str, ...],
) -> None:
    """Write a header-bearing CSV with an explicit column order atomically."""

    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("CSV fieldnames must be unique")
    if any(not isinstance(fieldname, str) for fieldname in fieldnames):
        raise ValueError("CSV fieldnames must be strings")

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    expected_keys = set(fieldnames)
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ValueError("CSV row keys do not match fieldnames")
        writer.writerow(row)
    serialized = stream.getvalue()
    _atomic_write(path, lambda temporary: _write_text(temporary, serialized))


def _validate_parquet_rows(
    rows: tuple[dict[str, object], ...],
    schema: dict[str, object] | None,
) -> None:
    if not rows and schema is None:
        raise ValueError("Parquet rows cannot be empty without a schema")
    if schema is not None and any(not isinstance(name, str) for name in schema):
        raise ValueError("Parquet schema names must be strings")

    expected_keys = set(schema) if schema is not None else None
    if expected_keys is None and rows:
        first = rows[0]
        if not isinstance(first, dict):
            raise ValueError("Parquet rows must be dictionaries")
        expected_keys = set(first)
    assert expected_keys is not None
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ValueError("Parquet row keys do not match the schema")


def write_parquet_atomic(
    path: str | Path,
    rows: tuple[dict[str, object], ...],
    schema: dict[str, object] | None = None,
) -> None:
    """Write a Polars Parquet table atomically."""

    _validate_parquet_rows(rows, schema)
    table = pl.DataFrame(
        rows,
        schema=schema,
        strict=True,
        infer_schema_length=None,
    )

    def write_table(temporary: Path) -> None:
        table.write_parquet(
            temporary,
            compression="zstd",
            statistics=True,
            use_pyarrow=False,
        )

    _atomic_write(path, write_table)


def _root_directory(root: str | Path) -> Path:
    try:
        resolved = Path(root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("checksum root is missing") from error
    if not resolved.is_dir():
        raise ValueError("checksum root is not a directory")
    return resolved


def _regular_files(root: Path, checksum_path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    checksum_resolved = checksum_path.resolve(strict=False)
    try:
        candidates = root.rglob("*")
        for candidate in candidates:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            relative = candidate.relative_to(root)
            if "_work" in relative.parts:
                continue
            if candidate.resolve(strict=False) == checksum_resolved:
                continue
            entries.append((relative.as_posix(), candidate))
    except (OSError, RuntimeError) as error:
        raise ValueError("checksum files cannot be enumerated") from error
    entries.sort(key=lambda item: item[0])
    return entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot read checksum input: {path}") from error
    return digest.hexdigest()


def write_checksums(root: str | Path, checksum_path: str | Path) -> None:
    """Write a sorted SHA-256 roster for regular files under ``root``."""

    resolved_root = _root_directory(root)
    checksum = Path(checksum_path)
    entries = _regular_files(resolved_root, checksum)
    serialized = "".join(
        f"{_sha256(path)}  {relative}\n" for relative, path in entries
    )
    _atomic_write(
        checksum,
        lambda temporary: _write_bytes(temporary, serialized.encode("ascii")),
    )


def _parse_checksum_name(name: str) -> PurePosixPath:
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or "\r" in name
        or "\n" in name
    ):
        raise ValueError("checksum path is not a relative POSIX path")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or relative.as_posix() != name
        or any(part in ("", ".", "..") for part in relative.parts)
        or "_work" in relative.parts
    ):
        raise ValueError("checksum path is not a relative POSIX path")
    return relative


def verify_checksums(
    root: str | Path, checksum_path: str | Path
) -> tuple[str, ...]:
    """Verify a strict checksum roster and return its sorted relative paths."""

    resolved_root = _root_directory(root)
    checksum = Path(checksum_path)
    try:
        checksum_resolved = checksum.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("checksum file is missing") from error
    if (
        checksum.is_symlink()
        or not checksum.is_file()
        or checksum_resolved.is_dir()
    ):
        raise ValueError("checksum file is not regular")
    try:
        raw = checksum.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("checksum file cannot be read") from error
    if text and not text.endswith("\n"):
        raise ValueError("checksum file must end with a newline")

    names: list[str] = []
    records: list[tuple[str, str]] = []
    lines = text[:-1].split("\n") if text else []
    for line in lines:
        if "\r" in line:
            raise ValueError("checksum file has invalid line endings")
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("checksum file has an invalid line")
        digest = line[:64]
        name = line[66:]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("checksum file has an invalid digest")
        _parse_checksum_name(name)
        names.append(name)
        records.append((digest, name))

    if names != sorted(names) or len(set(names)) != len(names):
        raise ValueError("checksum file paths are not strictly sorted")

    expected_entries = dict(_regular_files(resolved_root, checksum))
    if set(names) != set(expected_entries):
        raise ValueError("checksum file roster changed")

    for digest, name in records:
        candidate = resolved_root.joinpath(*PurePosixPath(name).parts)
        try:
            candidate_resolved = candidate.resolve(strict=True)
            candidate_resolved.relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError("checksum path escaped the root") from error
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate_resolved == checksum_resolved
            or _sha256(candidate) != digest
        ):
            raise ValueError("checksum verification failed")

    return tuple(names)


__all__ = [
    "verify_checksums",
    "write_checksums",
    "write_csv_atomic",
    "write_json_atomic",
    "write_jsonl_atomic",
    "write_parquet_atomic",
]
