from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from cmc_bbdm.agentic_nde.surface_qc import SurfaceQCError, inspect_surface


def test_surface_qc_preserves_native_geometry(tmp_path: Path) -> None:
    path = tmp_path / "surface.png"
    Image.new("RGB", (11, 7), (10, 20, 30)).save(path)
    qc = inspect_surface(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert (qc.width, qc.height, qc.mode, qc.format) == (11, 7, "RGB", "PNG")
    assert qc.exif_orientation is None
    assert qc.orientation_status == "UNKNOWN_NO_SOURCE_TRANSFORM"


def test_surface_qc_does_not_apply_exif_rotation(tmp_path: Path) -> None:
    path = tmp_path / "surface.jpg"
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    qc = inspect_surface(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert (qc.width, qc.height) == (3, 2)
    assert qc.exif_orientation == 6


def test_surface_qc_rejects_alpha_channel(tmp_path: Path) -> None:
    path = tmp_path / "surface.png"
    Image.new("RGBA", (4, 4), (1, 2, 3, 4)).save(path)
    with pytest.raises(SurfaceQCError, match="mode"):
        inspect_surface(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_surface_qc_rejects_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "surface.png"
    Image.new("RGB", (4, 4)).save(path)
    with pytest.raises(SurfaceQCError, match="SHA-256"):
        inspect_surface(path, expected_sha256="0" * 64)


def test_surface_qc_verifies_encoded_image_is_not_truncated(tmp_path: Path) -> None:
    path = tmp_path / "surface.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(path)
    path.write_bytes(path.read_bytes()[:80])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(SurfaceQCError, match="decoded"):
        inspect_surface(path, expected_sha256=digest)


def test_surface_qc_records_embedded_metadata_without_interpreting_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "surface.png"
    metadata = PngInfo()
    metadata.add_text("instrument", "surface-camera")
    Image.new("RGB", (5, 4)).save(path, pnginfo=metadata)
    qc = inspect_surface(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert qc.metadata_keys == ("instrument",)
    assert qc.metadata_status == "OBSERVED"
    assert qc.annotation_status == "UNKNOWN"
    assert qc.specimen_boundary_status == "UNKNOWN"
    assert qc.physical_extent_status == "UNKNOWN"


def test_surface_qc_rejects_decompression_bomb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "surface.png"
    Image.new("RGB", (4, 4)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(SurfaceQCError, match="decompression"):
        inspect_surface(path, expected_sha256=digest)
