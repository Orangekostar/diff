from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cmc_bbdm.damage_response.authority import AuthorityError, snapshot_file


def test_snapshot_rejects_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="ascii")
    link = tmp_path / "link.csv"
    link.symlink_to(source)

    with pytest.raises(AuthorityError, match="regular file"):
        snapshot_file(link, max_bytes=1024)


def test_snapshot_binds_sha_and_size(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    payload = b"a,b\n1,2\n"
    path.write_bytes(payload)

    record = snapshot_file(path, max_bytes=1024)

    assert record.size == len(payload)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert record.relative_path == "source.csv"
    assert record.logical_source == "local:external"


def test_snapshot_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(AuthorityError, match="regular file"):
        snapshot_file(tmp_path, max_bytes=1024)


def test_snapshot_enforces_maximum_bytes(tmp_path: Path) -> None:
    path = tmp_path / "large.csv"
    path.write_bytes(b"x" * 17)

    with pytest.raises(AuthorityError, match="maximum size"):
        snapshot_file(path, max_bytes=16)


@pytest.mark.parametrize("max_bytes", (0, -1, True))
def test_snapshot_rejects_invalid_maximum(max_bytes: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        snapshot_file(Path("unused.csv"), max_bytes=max_bytes)


@pytest.mark.parametrize(
    "relative_path",
    ("/private/source.csv", "../source.csv", "folder/../../source.csv", ""),
)
def test_snapshot_rejects_unsafe_manifest_path(
    tmp_path: Path, relative_path: str
) -> None:
    path = tmp_path / "source.csv"
    path.write_bytes(b"x")

    with pytest.raises(ValueError, match="relative path"):
        snapshot_file(path, max_bytes=10, relative_path=relative_path)


def test_snapshot_serializes_only_explicit_logical_identity(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_bytes(b"payload")

    record = snapshot_file(
        path,
        max_bytes=10,
        logical_source="local:hasebe_v3_root",
        relative_path="raw/c8-2.CSV",
    )

    assert record.logical_source == "local:hasebe_v3_root"
    assert record.relative_path == "raw/c8-2.CSV"
    assert str(tmp_path) not in repr(record)
