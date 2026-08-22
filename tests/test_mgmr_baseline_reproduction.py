from __future__ import annotations

from pathlib import Path

import pytest

from cmc_bbdm.mgmr.authority import load_authority
from cmc_bbdm.mgmr.m0_components import load_registered_b0
from cmc_bbdm.mgmr.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mgmr_m0.yaml"


def test_registered_b0_is_exactly_aligned_with_m0_authority() -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)
    authority = load_authority(protocol, project_root=ROOT)

    baseline = load_registered_b0(protocol, authority, project_root=ROOT)

    assert baseline.specimen_count == 276
    assert baseline.specimen_ids == authority.specimen_ids
    assert baseline.dataset_ids == authority.dataset_ids
    assert baseline.pca_dimensions == (8, 32, 8, 8, 8, 8)
    assert baseline.maximum_target_error <= 1.0e-12
    assert baseline.equal_domain_mae == pytest.approx(
        0.08963580465761432, abs=1.0e-12
    )
    assert baseline.predictions.flags.writeable is False
