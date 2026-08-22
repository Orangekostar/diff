from __future__ import annotations

from pathlib import Path

import pytest
from test_msss_s1 import ROOT, synthetic_s1_run

from cmc_bbdm.msss.artifacts import publish_s1_package, validate_s1_package
from cmc_bbdm.msss.replay import MSSSReplayError, replay_s1_package


def test_s1_replay_recomputes_tables_and_preserves_scientific_digest(tmp_path: Path) -> None:
    protocol, bank, run = synthetic_s1_run()
    source = tmp_path / "formal"
    replay = tmp_path / "replay"
    publish_s1_package(
        source,
        protocol=protocol,
        bank=bank,
        run=run,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
        mode="smoke",
        test_only=True,
    )
    result = replay_s1_package(
        source,
        replay,
        project_root=ROOT,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
    )
    source_validation = validate_s1_package(
        source,
        project_root=ROOT,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
    )

    assert result.scientific_digest == source_validation.scientific_digest
    assert result.output_tree_sha256 == source_validation.output_tree_sha256


def test_s1_replay_rejects_prediction_curve_mismatch(tmp_path: Path) -> None:
    protocol, bank, run = synthetic_s1_run()
    source = tmp_path / "formal"
    publish_s1_package(
        source,
        protocol=protocol,
        bank=bank,
        run=run,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
        mode="smoke",
        test_only=True,
    )
    curve = source / "gaussian_curve.csv"
    text = curve.read_text(encoding="utf-8").replace("0.102", "0.902", 1)
    curve.write_text(text, encoding="utf-8")

    with pytest.raises(MSSSReplayError):
        replay_s1_package(
            source,
            tmp_path / "replay",
            project_root=ROOT,
            config_path=ROOT / "paper_v3/configs/msss.yaml",
        )
