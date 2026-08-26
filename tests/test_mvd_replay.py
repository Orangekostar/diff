from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cmc_bbdm.mvd.replay import verify_checksums

ROOT = Path(__file__).resolve().parents[1]


def test_mavis_old_m0_m1_artifacts_unchanged() -> None:
    replay = ROOT / "results/mvd/replay"
    summary = json.loads((replay / "summary.json").read_text())
    assert summary["replay_verified"] is True
    for relative in (
        "../m0_one_shot_oracle",
        "../m1_observability",
        "m0_one_shot_oracle",
        "m1_observability",
    ):
        verify_checksums(replay / relative)

    for package in ("m0_one_shot_oracle", "m1_observability"):
        formal = replay.parent / package
        reproduced = replay / package
        formal_files = sorted(
            path.relative_to(formal)
            for path in formal.rglob("*")
            if path.is_file()
        )
        replay_files = sorted(
            path.relative_to(reproduced)
            for path in reproduced.rglob("*")
            if path.is_file()
        )
        assert replay_files == formal_files
        for relative in formal_files:
            assert (reproduced / relative).read_bytes() == (
                formal / relative
            ).read_bytes()


def test_mvd_checksum_verifier_rejects_unlisted_files(tmp_path: Path) -> None:
    listed = tmp_path / "listed.txt"
    listed.write_text("bound\n", encoding="ascii")
    digest = hashlib.sha256(listed.read_bytes()).hexdigest()
    (tmp_path / "CHECKSUMS.sha256").write_text(
        f"{digest}  listed.txt\n",
        encoding="ascii",
    )
    (tmp_path / "unlisted.txt").write_text("not bound\n", encoding="ascii")

    with pytest.raises(ValueError, match="checksum roster"):
        verify_checksums(tmp_path)


def test_mvd_final_report_answers_all_required_questions() -> None:
    report = (ROOT / "results/mvd/m1_observability/REPORT.md").read_text()
    for required in (
        "Candidate-only versus global-plus-candidate",
        "Regret@1",
        "mean exact-budget set regret",
        "Imperial RSS",
        "Imperial Interlock",
        "TU Delft",
        "Cranfield",
    ):
        assert required in report
