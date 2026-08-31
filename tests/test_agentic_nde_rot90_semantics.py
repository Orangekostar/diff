from __future__ import annotations

import pytest

from cmc_bbdm.agentic_nde.author_authority import (
    EXPECTED_STATEMENT_SHA256,
    build_author_registration_authority,
)
from cmc_bbdm.agentic_nde.contracts import EvidenceClass, FrameGeometry, Orientation
from cmc_bbdm.agentic_nde.grid import Grid8x8, render_surface_grid
from cmc_bbdm.agentic_nde.registration import create_transform


def _author_transform(
    source_size: tuple[int, int], destination_size: tuple[int, int]
):
    authority = build_author_registration_authority()
    return create_transform(
        source=FrameGeometry(*source_size, 80.0, 80.0),
        destination=FrameGeometry(*destination_size, 75.0, 75.0),
        orientation=authority.orientation,
        evidence_class=EvidenceClass.A_DIRECT_METADATA,
        evidence_roles=authority.evidence_roles,
        evidence_hashes=(EXPECTED_STATEMENT_SHA256,),
    )


@pytest.mark.parametrize(
    ("source_size", "destination_size"),
    [
        ((3357, 3357), (675, 674)),
        ((1500, 1500), (340, 338)),
        ((3147, 2084), (430, 675)),
    ],
)
def test_rot90_is_clockwise_under_image_coordinates(
    source_size: tuple[int, int], destination_size: tuple[int, int]
) -> None:
    source_width, source_height = source_size
    destination_width, destination_height = destination_size
    transform = _author_transform(source_size, destination_size)

    assert transform.orientation is Orientation.ROT90
    assert transform.forward_point((0, 0)) == pytest.approx(
        (destination_width - 1, 0)
    )
    assert transform.forward_point((source_width - 1, 0)) == pytest.approx(
        (destination_width - 1, destination_height - 1)
    )
    assert transform.forward_point(
        (source_width - 1, source_height - 1)
    ) == pytest.approx((0, destination_height - 1))
    assert transform.forward_point((0, source_height - 1)) == pytest.approx((0, 0))


@pytest.mark.parametrize(
    ("source_size", "destination_size"),
    [
        ((3357, 3357), (675, 674)),
        ((1500, 1500), (340, 338)),
    ],
)
def test_rot90_full_frame_and_all_grid_cells_round_trip(
    source_size: tuple[int, int], destination_size: tuple[int, int]
) -> None:
    transform = _author_transform(source_size, destination_size)
    destination_width, destination_height = destination_size
    source_width, source_height = source_size

    assert transform.forward_box(
        (0, 0, source_width - 1, source_height - 1)
    ) == pytest.approx((0, 0, destination_width - 1, destination_height - 1))
    records = render_surface_grid(transform)
    assert len(records) == 64
    assert {record["cell_id"] for record in records} == set(range(64))

    grid = Grid8x8(destination_width - 1, destination_height - 1)
    for cell in grid.cells():
        destination_box = (cell.x0, cell.y0, cell.x1, cell.y1)
        surface_box = transform.inverse_box(destination_box)
        assert transform.forward_box(surface_box) == pytest.approx(
            destination_box, abs=1e-9
        )
