from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from cmc_bbdm.vlm_cscan.contracts import EvidenceState
from cmc_bbdm.vlm_cscan.reader import read_sparse_evidence
from cmc_bbdm.vlm_cscan.route import compile_route, reference_full_raster_length
from cmc_bbdm.vlm_cscan.runtime import (
    InputSpecimen,
    _inner_border_median,
    load_benchmark_config,
    load_input_records,
    render_surface_inputs,
    select_pilot_records,
)


def test_surface_render_uses_real_clockwise_rotated_rgb() -> None:
    """Catches feeding a 1x1 carrier or rotating the registered surface backwards."""

    array = np.full((10, 12, 3), 100, dtype=np.uint8)
    array[0, 0] = (1, 2, 3)
    array[0, -1] = (4, 5, 6)
    array[-1, 0] = (7, 8, 9)
    array[-1, -1] = (10, 11, 12)

    rendered = render_surface_inputs(Image.fromarray(array), max_edge=1024)
    clean = np.asarray(rendered.clean)

    assert clean.shape == (12, 10, 3)
    assert tuple(clean[0, 0]) == (7, 8, 9)
    assert tuple(clean[0, -1]) == (1, 2, 3)
    assert tuple(clean[-1, 0]) == (10, 11, 12)
    assert tuple(clean[-1, -1]) == (4, 5, 6)
    assert rendered.clean.size == rendered.gridded.size
    assert not np.array_equal(clean, np.asarray(rendered.gridded))


def test_sparse_reader_never_labels_unmeasured_fill_as_evidence() -> None:
    """Catches converting reconstructed or filled UNKNOWN pixels into observations."""

    evidence = read_sparse_evidence(
        native_shape=(4, 5),
        positions=np.asarray([[0, 0], [2, 3]], dtype=np.int64),
        values=np.asarray([[102, 100, 99], [255, 0, 0]], dtype=np.uint8),
        background_rgb=np.asarray([100, 100, 100], dtype=np.uint8),
        distance_threshold=0.18,
    )

    assert evidence.states[0, 0] == EvidenceState.MEASURED_NO_INDICATION
    assert evidence.states[2, 3] == EvidenceState.MEASURED_INDICATION
    assert np.count_nonzero(evidence.states == EvidenceState.UNKNOWN) == 18
    assert np.array_equal(np.argwhere(evidence.measured_mask), [[0, 0], [2, 3]])
    assert np.array_equal(np.argwhere(evidence.indication_mask), [[2, 3]])


def test_route_deduplicates_shared_points_and_matches_hand_distance() -> None:
    """Catches double charging shared boundaries or using cell-centre distance."""

    route = compile_route(
        np.asarray([[0, 0], [0, 2], [2, 2], [2, 0], [0, 2]], dtype=np.int64),
        native_shape=(3, 3),
        start_position=(0.0, 0.0),
    )

    assert route.unique_revealed_count == 4
    assert route.transit_length == 0.0
    assert route.scan_length == 3.0
    assert route.total_length == 3.0
    assert route.turn_count == 2
    assert route.revisit_count == 0
    assert len({tuple(point) for point in route.pixel_order}) == 4
    assert reference_full_raster_length((3, 3), (0.0, 0.0)) == 4.0


def test_pilot_selection_is_hash_fixed_within_each_domain() -> None:
    """Catches order-dependent or cross-domain pilot selection."""

    records = tuple(
        InputSpecimen(
            dataset_id=domain,
            specimen_id=specimen,
            surface_path=Path(f"{domain}/{specimen}.png"),
            surface_sha256=specimen * 64,
            cscan_path=Path(f"{domain}/{specimen}-cscan.png"),
            cscan_sha256="f" * 64,
            native_shape=(10, 12),
            transform_sha256="e" * 64,
        )
        for domain in ("d1", "d0")
        for specimen in ("c", "a", "b")
    )

    selected = select_pilot_records(
        records,
        domain_order=("d0", "d1"),
        per_domain=2,
        seed="seed",
    )

    assert [(row.dataset_id, row.specimen_id) for row in selected] == [
        ("d0", "b"),
        ("d0", "a"),
        ("d1", "c"),
        ("d1", "a"),
    ]


def test_config_loader_rejects_a_changed_bound_source(tmp_path: Path) -> None:
    """Catches running the pilot against a source other than its frozen manifests."""

    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "paper_v3/configs/vlm_cscan_efficiency.yaml"
    config = load_benchmark_config(source, project_root=project_root)
    assert config.domain_order == (
        "74t7kcdgkr",
        "cgtnjyggtm",
        "w68dtmpfyf",
        "xcmzfsbd9t",
        "yfxyg8jm46",
        "ykhs7s2dck",
    )

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["sources"]["p0r_surface_manifest"]["sha256"] = "0" * 64
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source hash changed"):
        load_benchmark_config(changed, project_root=project_root)


def test_real_roster_resolves_six_domain_hash_fixed_pilot() -> None:
    """Catches losing P0R identity or selecting a different real pilot cohort."""

    source_root = Path("/home/ww/paper3/cmc_damage_inference")
    if not source_root.is_dir():
        pytest.skip("registered external research root is unavailable")
    project_root = Path(__file__).resolve().parents[1]
    config = load_benchmark_config(
        project_root / "paper_v3/configs/vlm_cscan_efficiency.yaml",
        project_root=project_root,
    )

    roster = load_input_records(
        config,
        source_root=source_root,
        verify_pilot_hashes=False,
    )

    assert len(roster.records) == 276
    assert len(roster.pilot_records) == 60
    assert len(roster.smoke_records) == 6
    assert [record.specimen_id for record in roster.smoke_records] == [
        "c8-2",
        "q24-29",
        "q16-22",
        "c24-20",
        "c16-4",
        "q8-14",
    ]
    assert all(record.surface_path.is_file() for record in roster.pilot_records)
    assert all(record.cscan_path.is_file() for record in roster.pilot_records)


def test_inner_border_median_excludes_rendered_frame() -> None:
    image = np.full((40, 50, 3), (70, 80, 150), dtype=np.uint8)
    image[[0, -1], :, :] = 255
    image[:, [0, -1], :] = 255

    background = _inner_border_median(image, border_fraction=0.10)

    assert np.array_equal(background, np.asarray([70.0, 80.0, 150.0]))
