from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.mavis import aei_paper_visual_assets

ROOT = Path(__file__).resolve().parents[1]


def test_aei_visual_asset_module_exists() -> None:
    assert (
        importlib.util.find_spec("cmc_bbdm.mavis.aei_paper_visual_assets") is not None
    )


def test_registered_state_reconstruction_matches_frozen_hash() -> None:
    initial = aei_paper_visual_assets.load_reconstructed_state(
        ROOT,
        specimen_id="c8-2",
        method="one_shot_mechanical_oracle",
        checkpoint=0.03125,
    )
    sparse = aei_paper_visual_assets.load_reconstructed_state(
        ROOT,
        specimen_id="c8-2",
        method="uniform",
        checkpoint=0.25,
    )

    assert initial.image.shape == sparse.image.shape == (338, 340, 3)
    assert initial.image.dtype == sparse.image.dtype == np.uint8
    assert initial.measurement_mask.shape == sparse.measurement_mask.shape == (338, 340)
    assert int(initial.measurement_mask.sum()) == initial.exact_acquired_cost == 3600
    assert int(sparse.measurement_mask.sum()) == sparse.exact_acquired_cost == 28730
    assert initial.output_sha256 == initial.expected_output_sha256
    assert sparse.output_sha256 == sparse.expected_output_sha256
    assert not initial.image.flags.writeable
    assert not sparse.image.flags.writeable


def test_priority_states_use_all_registered_cells_and_change_with_state() -> None:
    initial = aei_paper_visual_assets.load_priority_state(
        ROOT,
        specimen_id="c8-2",
        method="one_shot_mechanical_oracle",
        checkpoint=0.03125,
    )
    later = aei_paper_visual_assets.load_priority_state(
        ROOT,
        specimen_id="c8-2",
        method="one_shot_mechanical_oracle",
        checkpoint=0.1875,
    )

    for asset in (initial, later):
        assert asset.raw_values.shape == asset.percentiles.shape == (8, 8)
        assert np.isfinite(asset.raw_values).all()
        assert np.isfinite(asset.percentiles).all()
        assert float(asset.percentiles.min()) == pytest.approx(0.0)
        assert float(asset.percentiles.max()) == pytest.approx(1.0)
        assert (
            asset.reconstruction.output_sha256
            == asset.reconstruction.expected_output_sha256
        )
    assert not np.array_equal(initial.percentiles, later.percentiles)
    assert initial.state_id != later.state_id


def test_task_priority_maps_are_paired_on_one_legal_grid() -> None:
    priorities = aei_paper_visual_assets.load_task_priority_maps(
        ROOT, specimen_id="c8-2"
    )

    assert priorities.mechanical_values.shape == (8, 8)
    assert priorities.reconstruction_values.shape == (8, 8)
    assert priorities.mechanical_percentiles.shape == (8, 8)
    assert priorities.reconstruction_percentiles.shape == (8, 8)
    assert priorities.percentile_difference.shape == (8, 8)
    assert priorities.cell_indices == tuple(range(64))
    assert not np.array_equal(
        priorities.mechanical_percentiles, priorities.reconstruction_percentiles
    )
    assert (
        priorities.reconstruction.output_sha256
        == priorities.reconstruction.expected_output_sha256
    )


def test_gallery_roster_is_deterministic_and_domain_complete() -> None:
    roster = aei_paper_visual_assets.gallery_specimen_roster(ROOT)
    assert [(item.domain_id, item.specimen_id) for item in roster] == [
        ("74t7kcdgkr", "c8-10t"),
        ("cgtnjyggtm", "q24-1"),
        ("w68dtmpfyf", "q16-1"),
        ("xcmzfsbd9t", "c24-10"),
        ("yfxyg8jm46", "c16-1"),
        ("ykhs7s2dck", "q8-1"),
    ]


def test_visual_asset_loader_has_no_synthetic_fallback() -> None:
    with pytest.raises(aei_paper_visual_assets.AEIVisualAssetError):
        aei_paper_visual_assets.load_reconstructed_state(
            ROOT,
            specimen_id="not-a-real-specimen",
            method="uniform",
            checkpoint=0.25,
        )
