from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cmc_bbdm.inspection_agent.artifacts import (
    REQUIRED_OUTPUTS,
    InspectionArtifactError,
    compare_g0_packages,
    publish_g0_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/inspection_agent_g0.yaml"


def _formal_fixture(path: Path) -> None:
    path.mkdir()
    for relative in REQUIRED_OUTPUTS:
        target = path / relative
        if relative == "decision_summary.json":
            target.write_text(
                json.dumps(
                    {"status": "G0_NO_AGENTIC_HEADROOM_NO_GO"},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="ascii",
            )
        else:
            target.write_text(f"fixture:{relative}\n", encoding="ascii")
    shutil.copyfile(CONFIG, path / "config.yaml")
    publish_g0_manifest(path, project_root=ROOT, config_path=CONFIG)


def test_g0_replay_comparison_rejects_one_byte_difference(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    replay = tmp_path / "replay"
    _formal_fixture(formal)
    shutil.copytree(formal, replay)
    with (replay / "REPORT.md").open("ab") as handle:
        handle.write(b"changed\n")
    with pytest.raises(InspectionArtifactError, match="replay|checksum"):
        compare_g0_packages(
            formal,
            replay,
            project_root=ROOT,
            config_path=CONFIG,
        )
