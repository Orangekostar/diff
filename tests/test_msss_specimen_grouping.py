from __future__ import annotations

from pathlib import Path

import numpy as np

from cmc_bbdm.msss.authority import load_authority
from cmc_bbdm.msss.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/msss.yaml"


def test_msss_authority_binds_specimens_features_and_groups() -> None:
    authority = load_authority(load_protocol(CONFIG, project_root=ROOT), project_root=ROOT)

    assert authority.specimen_count == 276
    assert len(set(authority.specimen_ids)) == 276
    assert authority.full_features.shape == (276, 512)
    assert authority.bilinear50_features.shape == (276, 512)
    assert authority.bilinear25_features.shape == (276, 512)
    assert authority.metadata13.shape == (276, 13)
    assert authority.targets.shape == (276,)
    assert authority.ply_count.shape == (276,)
    assert authority.layup_family.shape == (276,)
    assert not authority.full_features.flags.writeable
    assert not authority.metadata13.flags.writeable
    assert not authority.targets.flags.writeable
    assert np.all(np.isfinite(authority.full_features))
    assert np.all(np.isfinite(authority.targets))
    assert authority.specimen_ids == authority.registered_inputs.specimen_ids
    assert authority.dataset_ids == authority.registered_inputs.dataset_ids
