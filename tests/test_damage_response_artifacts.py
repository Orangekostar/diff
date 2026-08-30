from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.damage_response.artifacts import (
    REQUIRED_P0_PAYLOADS,
    ArtifactError,
    replay_p0,
    write_p0_package,
)


def _payloads() -> dict[str, bytes]:
    return {
        name: (
            b'{"status":"P0_GO"}\n'
            if name == "summary.json"
            else f"fixture:{name}\n".encode("ascii")
        )
        for name in REQUIRED_P0_PAYLOADS
    }


@pytest.fixture
def complete_package(tmp_path: Path) -> Path:
    destination = tmp_path / "p0"
    write_p0_package(destination, _payloads())
    return destination


def test_writer_requires_exact_payload_membership(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads.pop("raw_trace_qc.csv")

    with pytest.raises(ArtifactError, match="payload membership"):
        write_p0_package(tmp_path / "p0", payloads)


def test_writer_refuses_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "p0"
    destination.mkdir()

    with pytest.raises(ArtifactError, match="already exists"):
        write_p0_package(destination, _payloads())


def test_replay_accepts_complete_package(complete_package: Path) -> None:
    report = replay_p0(complete_package)
    assert report.payload_count == len(REQUIRED_P0_PAYLOADS)
    assert report.verified is True


def test_replay_rejects_extra_file(complete_package: Path) -> None:
    (complete_package / "extra.txt").write_text("x", encoding="ascii")

    with pytest.raises(ArtifactError, match="membership"):
        replay_p0(complete_package)


def test_replay_rejects_payload_hash_drift(complete_package: Path) -> None:
    path = complete_package / "raw_trace_qc.csv"
    path.write_bytes(path.read_bytes() + b"tampered\n")

    with pytest.raises(ArtifactError, match="size|SHA-256"):
        replay_p0(complete_package)


def test_replay_rejects_symlinked_payload(
    complete_package: Path, tmp_path: Path
) -> None:
    path = complete_package / "REPORT.md"
    target = tmp_path / "outside.md"
    target.write_text("outside\n", encoding="ascii")
    path.unlink()
    path.symlink_to(target)

    with pytest.raises(ArtifactError, match="regular file"):
        replay_p0(complete_package)


def test_identical_inputs_produce_byte_identical_packages(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_p0_package(first, _payloads())
    write_p0_package(second, _payloads())

    first_names = sorted(path.name for path in first.iterdir())
    second_names = sorted(path.name for path in second.iterdir())
    assert first_names == second_names
    for name in first_names:
        assert (first / name).read_bytes() == (second / name).read_bytes()
