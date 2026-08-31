from __future__ import annotations

import pytest

from cmc_bbdm.agentic_nde.contracts import (
    EvidenceClass,
    EvidenceRole,
    FrameGeometry,
    Orientation,
)
from cmc_bbdm.agentic_nde.grid import (
    Grid8x8,
    render_surface_grid,
    surface_box_to_cscan_cells,
)
from cmc_bbdm.agentic_nde.registration import create_transform


def test_grid_has_exactly_64_row_major_cells() -> None:
    grid = Grid8x8(width=80.0, height=80.0)
    assert [cell.cell_id for cell in grid.cells()] == list(range(64))
    assert grid.cell_box(0) == pytest.approx((0.0, 0.0, 10.0, 10.0))
    assert grid.cell_box(63) == pytest.approx((70.0, 70.0, 80.0, 80.0))


def test_internal_boundary_is_half_open() -> None:
    grid = Grid8x8(width=80.0, height=80.0)
    assert grid.point_to_cell((9.999, 5.0)) == 0
    assert grid.point_to_cell((10.0, 5.0)) == 1
    assert grid.point_to_cell((80.0, 80.0)) == 63


def test_grid_rejects_outside_point() -> None:
    with pytest.raises(ValueError, match="outside"):
        Grid8x8(width=80.0, height=80.0).point_to_cell((80.1, 0.0))


def _identity_transform():
    frame = FrameGeometry(width_px=81, height_px=81, width_mm=75.0, height_mm=75.0)
    return create_transform(
        source=frame,
        destination=frame,
        orientation=Orientation.IDENTITY,
        evidence_class=EvidenceClass.A_DIRECT_METADATA,
        evidence_roles=(EvidenceRole.INSTRUMENT_COORDINATES,),
        evidence_hashes=("c" * 64,),
    )


def test_transformed_surface_box_maps_to_legal_cells() -> None:
    transform = _identity_transform()
    assert surface_box_to_cscan_cells(transform, (0.0, 0.0, 9.9, 9.9)) == (0,)
    assert surface_box_to_cscan_cells(transform, (0.0, 0.0, 20.0, 20.0)) == (
        0,
        1,
        8,
        9,
    )


def test_surface_grid_replay_contains_every_cell_once() -> None:
    records = render_surface_grid(_identity_transform())
    assert tuple(record["cell_id"] for record in records) == tuple(range(64))
    assert records[0]["surface_box"] == pytest.approx((0.0, 0.0, 10.0, 10.0))
    assert records[-1]["surface_box"] == pytest.approx((70.0, 70.0, 80.0, 80.0))


def test_empty_grid_intersection_is_explicit() -> None:
    assert Grid8x8(width=80.0, height=80.0).cells_for_box(
        (10.0, 10.0, 10.0, 20.0)
    ) == ()
