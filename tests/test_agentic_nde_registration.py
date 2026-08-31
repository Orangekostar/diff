from __future__ import annotations

import pytest

from cmc_bbdm.agentic_nde.contracts import (
    EvidenceClass,
    EvidenceRole,
    FrameGeometry,
    Orientation,
)
from cmc_bbdm.agentic_nde.registration import create_transform


def _transform(orientation: Orientation = Orientation.IDENTITY):
    frame = FrameGeometry(width_px=101, height_px=101, width_mm=75.0, height_mm=75.0)
    return create_transform(
        source=frame,
        destination=frame,
        orientation=orientation,
        evidence_class=EvidenceClass.B_GEOMETRY_ONLY,
        evidence_roles=(EvidenceRole.SURFACE_METADATA, EvidenceRole.DATASET_METADATA),
        evidence_hashes=("a" * 64,),
    )


def test_identity_transform_is_invertible() -> None:
    transform = _transform()
    point = (12.5, 87.25)
    assert transform.forward_point(point) == pytest.approx(point)
    assert transform.inverse_point(transform.forward_point(point)) == pytest.approx(point)


@pytest.mark.parametrize("orientation", list(Orientation))
def test_all_orientations_keep_corners_in_frame(orientation: Orientation) -> None:
    transform = _transform(orientation)
    corners = ((0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0))
    mapped = [transform.forward_point(point) for point in corners]
    assert all(0.0 <= x <= 100.0 and 0.0 <= y <= 100.0 for x, y in mapped)
    assert all(transform.inverse_point(value) == pytest.approx(source) for source, value in zip(corners, mapped, strict=True))


def test_box_mapping_is_canonical() -> None:
    transform = _transform(Orientation.ROT90)
    x0, y0, x1, y1 = transform.forward_box((10.0, 20.0, 30.0, 40.0))
    assert x0 <= x1 and y0 <= y1
    assert transform.sha256 == _transform(Orientation.ROT90).sha256


def test_out_of_frame_point_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        _transform().forward_point((-1.0, 5.0))


def test_explicit_destination_span_and_offset_round_trip() -> None:
    source = FrameGeometry(width_px=11, height_px=11, width_mm=10.0, height_mm=10.0)
    destination = FrameGeometry(
        width_px=101,
        height_px=101,
        width_mm=100.0,
        height_mm=100.0,
    )
    transform = create_transform(
        source=source,
        destination=destination,
        orientation=Orientation.IDENTITY,
        evidence_class=EvidenceClass.A_DIRECT_METADATA,
        evidence_roles=(EvidenceRole.INSTRUMENT_COORDINATES,),
        evidence_hashes=("b" * 64,),
        scale_x=50.0,
        scale_y=40.0,
        offset_x=10.0,
        offset_y=20.0,
    )
    assert transform.forward_point((0.0, 0.0)) == pytest.approx((10.0, 20.0))
    assert transform.forward_point((10.0, 10.0)) == pytest.approx((60.0, 60.0))
    assert transform.inverse_point((35.0, 40.0)) == pytest.approx((5.0, 5.0))
    assert transform.inverse_box((10.0, 20.0, 60.0, 60.0)) == pytest.approx(
        (0.0, 0.0, 10.0, 10.0)
    )
    assert transform.as_dict()["scale_x"] == 50.0


def test_transform_rejects_span_outside_destination() -> None:
    frame = FrameGeometry(width_px=11, height_px=11, width_mm=10.0, height_mm=10.0)
    with pytest.raises(ValueError, match="destination"):
        create_transform(
            source=frame,
            destination=frame,
            orientation=Orientation.IDENTITY,
            evidence_class=EvidenceClass.A_DIRECT_METADATA,
            evidence_roles=(EvidenceRole.INSTRUMENT_COORDINATES,),
            evidence_hashes=("b" * 64,),
            scale_x=10.0,
            scale_y=10.0,
            offset_x=1.0,
        )
