from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.aei_multiview_regression.protocol import (
    MultiViewProtocolError,
    load_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/aei_multiview_regression.yaml"


def test_multiview_protocol_freezes_views_and_stage_order() -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)

    assert protocol.views == ("FULL", "BILINEAR_50", "BILINEAR_25")
    assert protocol.stage_order == ("E1", "E2", "E3", "E4", "E5")
    assert protocol.baseline_mae == 0.08963580465761432
    assert protocol.pca_dimensions == (8, 16, 32)
    assert protocol.consistency_grid == (
        0.0,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
    )
    assert protocol.target_losses == ("mse", "huber")
    assert protocol.bootstrap_seed == 20260811
    assert protocol.bootstrap_resamples == 100_000
    assert protocol.specimen_count == 276
    assert len(protocol.domain_order) == 6


def test_multiview_protocol_verifies_every_authoritative_source(tmp_path: Path) -> None:
    changed = tmp_path / "changed.yaml"
    changed.write_text(CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(MultiViewProtocolError, match="registered path"):
        load_protocol(changed, project_root=ROOT)
