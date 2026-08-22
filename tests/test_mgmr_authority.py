from __future__ import annotations

from pathlib import Path

import numpy as np

from cmc_bbdm.mgmr.authority import load_authority
from cmc_bbdm.mgmr.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mgmr_m0.yaml"


def test_mgmr_authority_cross_binds_registered_inputs() -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)
    authority = load_authority(protocol, project_root=ROOT)

    assert authority.specimen_count == 276
    assert authority.specimen_ids == tuple(authority.data.sample_ids.tolist())
    assert authority.dataset_ids == tuple(authority.data.dataset_ids.tolist())
    assert tuple(dict.fromkeys(authority.dataset_ids)) == protocol.domain_order
    assert authority.targets.shape == (276,)
    assert authority.metadata13.shape == (276, 13)
    assert authority.full_global.shape == (276, 512)
    assert len(authority.images) == 276
    assert len(authority.image_sha256) == 276
    assert authority.image_sha256 == tuple(
        record.sha256 for record in authority.data.cscan_records
    )
    for image, record in zip(
        authority.images, authority.data.cscan_records, strict=True
    ):
        assert image.shape == (record.height, record.width, 3)
        assert image.dtype == np.uint8
        assert image.flags.writeable is False
    for array in (authority.targets, authority.metadata13, authority.full_global):
        assert np.all(np.isfinite(array))
        assert array.flags.writeable is False
