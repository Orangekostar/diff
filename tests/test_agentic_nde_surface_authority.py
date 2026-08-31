from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import cmc_bbdm.agentic_nde.authority as authority_module
from cmc_bbdm.agentic_nde.authority import AuthorityError, snapshot_file


def test_snapshot_binds_relative_path_size_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "images" / "s.png"
    path.parent.mkdir()
    path.write_bytes(b"surface")
    record = snapshot_file(path, root=tmp_path, logical_root="hasebe-v1", max_bytes=1024)
    assert record.relative_path == "images/s.png"
    assert record.size == 7
    assert record.sha256 == hashlib.sha256(b"surface").hexdigest()
    assert not record.relative_path.startswith("/")


def test_snapshot_rejects_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"x")
    link = tmp_path / "link.bin"
    link.symlink_to(source)
    with pytest.raises(AuthorityError, match="regular file"):
        snapshot_file(link, root=tmp_path, logical_root="test", max_bytes=10)


def test_snapshot_rejects_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"x")
    try:
        with pytest.raises(AuthorityError, match="root"):
            snapshot_file(outside, root=tmp_path, logical_root="test", max_bytes=10)
    finally:
        outside.unlink()


def test_snapshot_rejects_oversize_file(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"12345")
    with pytest.raises(AuthorityError, match="maximum"):
        snapshot_file(path, root=tmp_path, logical_root="test", max_bytes=4)


def test_snapshot_rejects_non_regular_file(tmp_path: Path) -> None:
    with pytest.raises(AuthorityError, match="regular file"):
        snapshot_file(tmp_path, root=tmp_path, logical_root="test", max_bytes=10)


def test_snapshot_rejects_expected_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "surface.bin"
    path.write_bytes(b"surface")
    with pytest.raises(AuthorityError, match="SHA-256"):
        snapshot_file(
            path,
            root=tmp_path,
            logical_root="test",
            max_bytes=10,
            expected_sha256="0" * 64,
        )


def test_snapshot_rejects_descriptor_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "surface.bin"
    path.write_bytes(b"surface")
    monkeypatch.setattr(
        authority_module,
        "_descriptor_unchanged",
        lambda _before, _after: False,
    )
    with pytest.raises(AuthorityError, match="changed while"):
        snapshot_file(path, root=tmp_path, logical_root="test", max_bytes=10)
