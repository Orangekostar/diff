"""Native-geometry QC for released impacted-surface images."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError


class SurfaceQCError(ValueError):
    """Raised when a surface image fails the frozen P0 format contract."""


@dataclass(frozen=True, slots=True)
class SurfaceQC:
    sha256: str
    width: int
    height: int
    mode: str
    format: str
    channels: int
    exif_orientation: int | None
    orientation_status: str
    metadata_keys: tuple[str, ...]
    metadata_status: str
    annotation_status: str
    specimen_boundary_status: str
    physical_extent_status: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "sha256": self.sha256,
            "width_px": self.width,
            "height_px": self.height,
            "mode": self.mode,
            "format": self.format,
            "channels": self.channels,
            "exif_orientation": self.exif_orientation or "",
            "orientation_status": self.orientation_status,
            "metadata_keys": ";".join(self.metadata_keys),
            "metadata_status": self.metadata_status,
            "annotation_status": self.annotation_status,
            "specimen_boundary_status": self.specimen_boundary_status,
            "physical_extent_status": self.physical_extent_status,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_surface(path: str | Path, *, expected_sha256: str) -> SurfaceQC:
    """Inspect encoded geometry without EXIF autorotation or pixel conversion."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise SurfaceQCError("surface image must be a regular file")
    actual_sha256 = _sha256(source)
    if actual_sha256 != expected_sha256:
        raise SurfaceQCError("surface image SHA-256 does not match authority")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as verifier:
                verifier.verify()
            with Image.open(source) as image:
                width, height = image.size
                mode = image.mode
                image_format = image.format
                orientation = image.getexif().get(274)
                metadata_keys = tuple(sorted(str(key) for key in image.info))
                image.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise SurfaceQCError("surface image exceeds the decompression safety limit") from error
    except (OSError, UnidentifiedImageError) as error:
        raise SurfaceQCError("surface image cannot be decoded") from error
    if mode != "RGB":
        raise SurfaceQCError(f"surface image mode is not permitted: {mode}")
    if image_format not in {"PNG", "JPEG"}:
        raise SurfaceQCError(f"surface image format is not permitted: {image_format}")
    if width <= 0 or height <= 0:
        raise SurfaceQCError("surface image geometry is invalid")
    orientation_status = (
        "UNKNOWN_NO_SOURCE_TRANSFORM"
        if orientation is None
        else "EXIF_PRESENT_NOT_APPLIED_TRANSFORM_UNRESOLVED"
    )
    return SurfaceQC(
        sha256=actual_sha256,
        width=width,
        height=height,
        mode=mode,
        format=image_format,
        channels=3,
        exif_orientation=orientation,
        orientation_status=orientation_status,
        metadata_keys=metadata_keys,
        metadata_status="OBSERVED" if metadata_keys else "UNKNOWN",
        annotation_status="UNKNOWN",
        specimen_boundary_status="UNKNOWN",
        physical_extent_status="UNKNOWN",
    )


__all__ = ["SurfaceQC", "SurfaceQCError", "inspect_surface"]
