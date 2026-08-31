"""Hash-bound, target-isolated surface-to-C-scan transforms."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from .contracts import (
    EvidenceClass,
    EvidenceRole,
    FrameGeometry,
    Orientation,
    validate_evidence_roles,
)

_HEX = frozenset("0123456789abcdef")


def _orient(u: float, v: float, orientation: Orientation) -> tuple[float, float]:
    if orientation is Orientation.IDENTITY:
        return u, v
    if orientation is Orientation.ROT90:
        return 1.0 - v, u
    if orientation is Orientation.ROT180:
        return 1.0 - u, 1.0 - v
    if orientation is Orientation.ROT270:
        return v, 1.0 - u
    if orientation is Orientation.FLIP_X:
        return 1.0 - u, v
    if orientation is Orientation.FLIP_Y:
        return u, 1.0 - v
    if orientation is Orientation.TRANSPOSE:
        return v, u
    if orientation is Orientation.ANTI_TRANSPOSE:
        return 1.0 - v, 1.0 - u
    raise ValueError("unsupported orientation")


_INVERSE = {
    Orientation.IDENTITY: Orientation.IDENTITY,
    Orientation.ROT90: Orientation.ROT270,
    Orientation.ROT180: Orientation.ROT180,
    Orientation.ROT270: Orientation.ROT90,
    Orientation.FLIP_X: Orientation.FLIP_X,
    Orientation.FLIP_Y: Orientation.FLIP_Y,
    Orientation.TRANSPOSE: Orientation.TRANSPOSE,
    Orientation.ANTI_TRANSPOSE: Orientation.ANTI_TRANSPOSE,
}


def _validate_point(point: tuple[float, float], frame: FrameGeometry) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError("point must have two coordinates")
    x, y = (float(value) for value in point)
    if (
        not math.isfinite(x)
        or not math.isfinite(y)
        or x < 0.0
        or y < 0.0
        or x > frame.width_px - 1
        or y > frame.height_px - 1
    ):
        raise ValueError("point is outside the declared frame")
    return x, y


@dataclass(frozen=True, slots=True)
class SurfaceToCscanTransform:
    source: FrameGeometry
    destination: FrameGeometry
    orientation: Orientation
    evidence_class: EvidenceClass
    evidence_roles: tuple[EvidenceRole, ...]
    evidence_hashes: tuple[str, ...]
    source_only_isolated: bool
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    sha256: str

    def forward_point(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = _validate_point(point, self.source)
        u = x / (self.source.width_px - 1)
        v = y / (self.source.height_px - 1)
        mapped_u, mapped_v = _orient(u, v, self.orientation)
        return (
            self.offset_x + mapped_u * self.scale_x,
            self.offset_y + mapped_v * self.scale_y,
        )

    def inverse_point(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = _validate_point(point, self.destination)
        if (
            x < self.offset_x
            or x > self.offset_x + self.scale_x
            or y < self.offset_y
            or y > self.offset_y + self.scale_y
        ):
            raise ValueError("point is outside the transformed source field")
        u = (x - self.offset_x) / self.scale_x
        v = (y - self.offset_y) / self.scale_y
        source_u, source_v = _orient(u, v, _INVERSE[self.orientation])
        return (
            source_u * (self.source.width_px - 1),
            source_v * (self.source.height_px - 1),
        )

    def forward_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if len(box) != 4:
            raise ValueError("box must contain x0,y0,x1,y1")
        x0, y0, x1, y1 = (float(value) for value in box)
        if x0 > x1 or y0 > y1:
            raise ValueError("box bounds are not canonical")
        mapped = (
            self.forward_point((x0, y0)),
            self.forward_point((x1, y0)),
            self.forward_point((x0, y1)),
            self.forward_point((x1, y1)),
        )
        xs = [point[0] for point in mapped]
        ys = [point[1] for point in mapped]
        return min(xs), min(ys), max(xs), max(ys)

    def inverse_box(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if len(box) != 4:
            raise ValueError("box must contain x0,y0,x1,y1")
        x0, y0, x1, y1 = (float(value) for value in box)
        if x0 > x1 or y0 > y1:
            raise ValueError("box bounds are not canonical")
        mapped = (
            self.inverse_point((x0, y0)),
            self.inverse_point((x1, y0)),
            self.inverse_point((x0, y1)),
            self.inverse_point((x1, y1)),
        )
        xs = [point[0] for point in mapped]
        ys = [point[1] for point in mapped]
        return min(xs), min(ys), max(xs), max(ys)

    def as_dict(self) -> dict[str, object]:
        return {
            "source_frame": self.source.as_dict(),
            "destination_frame": self.destination.as_dict(),
            "orientation": self.orientation.value,
            "evidence_class": self.evidence_class.value,
            "evidence_roles": [role.value for role in self.evidence_roles],
            "evidence_hashes": list(self.evidence_hashes),
            "source_only_isolated": self.source_only_isolated,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "transform_sha256": self.sha256,
        }


def create_transform(
    *,
    source: FrameGeometry,
    destination: FrameGeometry,
    orientation: Orientation,
    evidence_class: EvidenceClass,
    evidence_roles: tuple[EvidenceRole, ...],
    evidence_hashes: tuple[str, ...],
    source_only_isolated: bool = True,
    scale_x: float | None = None,
    scale_y: float | None = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> SurfaceToCscanTransform:
    """Create a deterministic transform from legal, hash-bound evidence only."""

    if type(source) is not FrameGeometry or type(destination) is not FrameGeometry:
        raise ValueError("source and destination frame geometry are required")
    if type(orientation) is not Orientation or type(evidence_class) is not EvidenceClass:
        raise ValueError("registration orientation or evidence class is invalid")
    validate_evidence_roles(evidence_roles)
    if (
        not evidence_hashes
        or any(
            type(value) is not str
            or len(value) != 64
            or set(value) - _HEX
            for value in evidence_hashes
        )
        or len(set(evidence_hashes)) != len(evidence_hashes)
    ):
        raise ValueError("registration evidence hashes are invalid")
    if type(source_only_isolated) is not bool:
        raise ValueError("source-only isolation flag is invalid")
    if evidence_class is EvidenceClass.C_SOURCE_ONLY_LEARNED and not source_only_isolated:
        raise ValueError("class C registration must be source-only isolated")
    resolved_scale_x = (
        float(destination.width_px - 1) if scale_x is None else float(scale_x)
    )
    resolved_scale_y = (
        float(destination.height_px - 1) if scale_y is None else float(scale_y)
    )
    resolved_offset_x = float(offset_x)
    resolved_offset_y = float(offset_y)
    parameters = (
        resolved_scale_x,
        resolved_scale_y,
        resolved_offset_x,
        resolved_offset_y,
    )
    if (
        any(not math.isfinite(value) for value in parameters)
        or resolved_scale_x <= 0.0
        or resolved_scale_y <= 0.0
        or resolved_offset_x < 0.0
        or resolved_offset_y < 0.0
        or resolved_offset_x + resolved_scale_x > destination.width_px - 1
        or resolved_offset_y + resolved_scale_y > destination.height_px - 1
    ):
        raise ValueError("transform span or offset lies outside the destination frame")
    canonical = {
        "schema_version": 1,
        "source": source.as_dict(),
        "destination": destination.as_dict(),
        "orientation": orientation.value,
        "evidence_class": evidence_class.value,
        "evidence_roles": [role.value for role in evidence_roles],
        "evidence_hashes": list(evidence_hashes),
        "source_only_isolated": source_only_isolated,
        "scale_x": resolved_scale_x,
        "scale_y": resolved_scale_y,
        "offset_x": resolved_offset_x,
        "offset_y": resolved_offset_y,
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return SurfaceToCscanTransform(
        source=source,
        destination=destination,
        orientation=orientation,
        evidence_class=evidence_class,
        evidence_roles=evidence_roles,
        evidence_hashes=evidence_hashes,
        source_only_isolated=source_only_isolated,
        scale_x=resolved_scale_x,
        scale_y=resolved_scale_y,
        offset_x=resolved_offset_x,
        offset_y=resolved_offset_y,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


__all__ = ["SurfaceToCscanTransform", "create_transform"]
