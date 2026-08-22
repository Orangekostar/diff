from __future__ import annotations

from pathlib import Path

import numpy as np

from cmc_bbdm.cpb_v3.embeddings import FrozenResNet18Encoder, encode_resnet18
from cmc_bbdm.mgmr.spatial_encoder import encode_m0_views

ROOT = Path(__file__).resolve().parents[1]


def test_m0_views_reuse_p5_sampling_and_frozen_layer3() -> None:
    first = np.zeros((33, 35, 3), dtype=np.uint8)
    first[3:19, 7:24] = (200, 80, 10)
    second = np.rot90(first).copy()
    encoder = encode_resnet18(
        weight_path="paper_v3/assets/resnet18-f37072fd.pth",
        project_root=ROOT,
        device="cuda:0",
        batch_size=32,
    )
    assert isinstance(encoder, FrozenResNet18Encoder)

    views = encode_m0_views(
        encoder,
        images=(first, second),
        specimen_ids=("sample-a", "sample-b"),
        dataset_ids=("domain-a", "domain-b"),
        coarse_density=0.25,
    )

    assert views.full_layer3.shape == (2, 256, 14, 14)
    assert views.coarse_layer3.shape == views.full_layer3.shape
    assert views.specimen_ids == ("sample-a", "sample-b")
    assert views.dataset_ids == ("domain-a", "domain-b")
    assert len(views.sampling_records) == 2
    assert all(record.requested_density == 0.25 for record in views.sampling_records)
    assert all(record.interpolation == "bilinear" for record in views.sampling_records)
    assert all(record.measured_points_exact for record in views.sampling_records)
    assert all(record.shape_preserved for record in views.sampling_records)
    assert all(record.dtype_preserved for record in views.sampling_records)
    assert not np.array_equal(views.full_layer3, views.coarse_layer3)
    assert views.full_layer3.flags.writeable is False
    assert views.coarse_layer3.flags.writeable is False
