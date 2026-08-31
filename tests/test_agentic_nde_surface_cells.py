from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cmc_bbdm.agentic_nde.contracts import Orientation
from cmc_bbdm.agentic_nde.surface_cells import (
    crop_rgb_patch,
    integer_crop_box,
    load_surface_cell_authority,
    oriented_surface_boxes,
    shuffled_surface_donors,
    spatial_derangement,
    wrong_orientation,
)

ROOT = Path(__file__).resolve().parents[1]
P0R = ROOT / "results/agentic_task_driven_nde/p0r_author_registration"


def test_integer_crop_box_uses_frozen_half_open_pixel_center_rule() -> None:
    assert integer_crop_box(
        (0.0, 0.0, 209.75, 209.625), width=1679, height=1679
    ) == (0, 0, 210, 210)
    assert integer_crop_box(
        (209.0, 418.1, 419.5, 1678.0), width=1679, height=1679
    ) == (209, 419, 420, 1679)


def test_integer_crop_box_clamps_tiny_boundary_drift_and_rejects_empty() -> None:
    assert integer_crop_box(
        (-1.0e-12, -1.0e-12, 9.0 + 1.0e-12, 9.0 + 1.0e-12),
        width=10,
        height=10,
    ) == (0, 0, 10, 10)
    with pytest.raises(ValueError):
        integer_crop_box((5.0, 5.0, 5.0, 8.0), width=10, height=10)


def test_crop_rgb_patch_preserves_rgb_and_exact_pixel_extent() -> None:
    array = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    image = Image.fromarray(array, mode="RGB")
    patch = crop_rgb_patch(image, (1.1, 2.0, 7.2, 6.0))
    assert patch.mode == "RGB"
    assert patch.size == (6, 5)
    assert np.array_equal(np.asarray(patch), array[2:7, 2:8])


def test_formal_p0r_authority_replays_all_276_by_64_boxes() -> None:
    authority = load_surface_cell_authority(
        P0R / "surface_manifest.csv",
        P0R / "registration.csv",
        P0R / "grid_mapping_qc.csv",
    )
    assert authority.specimen_count == 276
    assert authority.specimen_ids[0] == "c8-10t"
    assert authority.dataset_ids[0] == "74t7kcdgkr"
    assert authority.cell_boxes.shape == (276, 64, 4)
    assert authority.cell_boxes.dtype == np.dtype("<f8")
    assert authority.surface_paths[0] == Path(
        "data/public/hasebe/raw/74t7kcdgkr/v1/impacted surface image/c8-10t.png"
    )
    assert authority.cell_boxes[0, 0] == pytest.approx(
        (0.0, 1468.25, 209.75, 1678.0)
    )
    assert authority.cell_boxes[0, 63] == pytest.approx(
        (1468.25, 0.0, 1678.0, 209.75)
    )
    assert len(authority.state_sha256) == 64


def test_oriented_surface_boxes_excludes_no_cells_and_changes_wrong_mapping() -> None:
    authority = load_surface_cell_authority(
        P0R / "surface_manifest.csv",
        P0R / "registration.csv",
        P0R / "grid_mapping_qc.csv",
    )
    correct = oriented_surface_boxes(authority.records[0], Orientation.ROT90)
    wrong = oriented_surface_boxes(authority.records[0], Orientation.IDENTITY)
    assert np.array_equal(correct, authority.cell_boxes[0])
    assert correct.shape == wrong.shape == (64, 4)
    assert not np.array_equal(correct, wrong)


def test_wrong_orientation_is_hash_selected_and_never_rot90() -> None:
    seed = "agentic-nde-p1-wrong-orientation-v1"
    selected = wrong_orientation(
        "c8-10t", dataset_id="74t7kcdgkr", seed=seed
    )
    assert selected is Orientation.ANTI_TRANSPOSE
    assert selected is not Orientation.ROT90
    assert wrong_orientation(
        "c8-10t", dataset_id="74t7kcdgkr", seed=seed
    ) is selected


def test_shuffled_surface_donors_are_domain_local_bijections_without_self() -> None:
    specimens = ("a", "b", "c", "d", "e")
    domains = ("x", "x", "x", "y", "y")
    donors = shuffled_surface_donors(
        specimens,
        domains,
        seed="agentic-nde-p1-shuffled-surface-v1",
    )
    assert donors == ("c", "a", "b", "e", "d")
    lookup = dict(zip(specimens, domains, strict=True))
    assert all(recipient != donor for recipient, donor in zip(specimens, donors, strict=True))
    assert all(lookup[recipient] == lookup[donor] for recipient, donor in zip(specimens, donors, strict=True))
    assert sorted(donors) == sorted(specimens)


def test_spatial_derangement_is_hash_stable_single_cycle_without_fixed_points() -> None:
    values = spatial_derangement(
        "c8-10t",
        dataset_id="74t7kcdgkr",
        seed="agentic-nde-p1-spatial-derangement-v1",
    )
    assert sorted(values) == list(range(64))
    assert all(index != value for index, value in enumerate(values))
    assert values[:8] == (46, 54, 43, 17, 34, 14, 40, 11)
    payload = ",".join(str(value) for value in values).encode("ascii")
    assert hashlib.sha256(payload).hexdigest() == (
        "5052d228218126412e146ca4994328dd4de21e8de91789234aada6ab09a43fd5"
    )
