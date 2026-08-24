from __future__ import annotations

import json
from pathlib import Path

from cmc_bbdm.mvd.replay import verify_checksums

ROOT = Path(__file__).resolve().parents[1]


def test_mvd_formal_and_replay_packages_verify() -> None:
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
