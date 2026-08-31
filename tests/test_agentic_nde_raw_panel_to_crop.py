from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw

from cmc_bbdm.agentic_nde.scan_frame_provenance import (
    ProcessingProvenanceError,
    verify_registered_crop,
)


def _write_raw(
    path: Path,
    size: tuple[int, int],
    panels: tuple[tuple[tuple[int, int, int, int], str], ...],
) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for box, color in panels:
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color)
    image.save(path, format="JPEG", quality=100, subsampling=0)


def _decoded_crop(
    raw: Path, box: tuple[int, int, int, int]
) -> Image.Image:
    with Image.open(raw) as decoded:
        return decoded.convert("RGB").crop(box).copy()


def _write_crop(raw: Path, path: Path, box: tuple[int, int, int, int]) -> Image.Image:
    expected = _decoded_crop(raw, box)
    expected.save(path, format="PNG")
    return expected


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _sha256_pixels(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def test_dual_panel_index_zero_is_left_and_one_is_right(tmp_path: Path) -> None:
    raw = tmp_path / "c8-2and3.jpg"
    left_path = tmp_path / "c8-2.png"
    right_path = tmp_path / "c8-3.png"
    left_box = (30, 33, 370, 371)
    right_box = (464, 33, 816, 371)
    _write_raw(
        raw,
        (996, 581),
        ((left_box, "red"), (right_box, "green")),
    )
    expected_left = _write_crop(raw, left_path, left_box)
    expected_right = _write_crop(raw, right_path, right_box)

    left = verify_registered_crop(raw, left_path, panel_index=0)
    right = verify_registered_crop(raw, right_path, panel_index=1)

    actual_left = _read_rgb(left_path)
    actual_right = _read_rgb(right_path)
    assert left.panel_box == left_box
    assert right.panel_box == right_box
    assert actual_left.size == (340, 338)
    assert actual_right.size == (352, 338)
    assert actual_left.tobytes() == expected_left.tobytes()
    assert actual_right.tobytes() == expected_right.tobytes()
    assert actual_left.getpixel((170, 169)) != actual_right.getpixel((176, 169))


@pytest.mark.parametrize(
    ("size", "box", "color"),
    (
        ((891, 891), (31, 33, 706, 707), "blue"),
        ((669, 885), (39, 33, 469, 708), "orange"),
    ),
)
def test_registered_crop_requires_exact_decoded_rgb(
    tmp_path: Path,
    size: tuple[int, int],
    box: tuple[int, int, int, int],
    color: str,
) -> None:
    raw = tmp_path / "source.jpg"
    crop = tmp_path / "registered.png"
    _write_raw(raw, size, ((box, color),))
    expected = _write_crop(raw, crop, box)

    result = verify_registered_crop(raw, crop, panel_index=0)

    assert result.panel_box == box
    assert result.decoded_pixel_equal is True
    assert _read_rgb(crop).tobytes() == expected.tobytes()
    assert result.recovered_panel_rgb_sha256 == _sha256_pixels(expected)
    assert result.registered_crop_rgb_sha256 == _sha256_pixels(expected)


def test_registered_crop_rejects_altered_decoded_pixels(tmp_path: Path) -> None:
    raw = tmp_path / "source.jpg"
    crop = tmp_path / "registered.png"
    box = (31, 33, 706, 707)
    _write_raw(raw, (891, 891), ((box, "blue"),))
    expected = _write_crop(raw, crop, box)
    altered = _read_rgb(crop)
    changed = altered.getpixel((0, 0))
    replacement = (0, 0, 0) if changed != (0, 0, 0) else (255, 255, 255)
    altered.putpixel((0, 0), replacement)
    altered.save(crop, format="PNG")
    assert ImageChops.difference(altered, expected).getbbox() is not None

    with pytest.raises(ProcessingProvenanceError, match="decoded RGB|pixel"):
        verify_registered_crop(raw, crop, panel_index=0)
