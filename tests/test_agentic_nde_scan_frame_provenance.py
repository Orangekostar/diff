from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from cmc_bbdm.agentic_nde.scan_frame_provenance import (
    HISTORICAL_CROP_RECIPES,
    CropRecipe,
    ProcessingProvenanceError,
    ScanProcessingProvenance,
    recipe_for_screenshot_size,
    verify_registered_crop,
)

_RECIPES = (
    ((891, 891), ((31, 33, 706, 707),)),
    ((669, 885), ((39, 33, 469, 708),)),
    ((996, 581), ((30, 33, 370, 371), (464, 33, 816, 371))),
)


def _write_raw(path: Path, size: tuple[int, int], boxes: tuple[tuple[int, int, int, int], ...]) -> None:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    colors = ("red", "green")
    for box, color in zip(boxes, colors[: len(boxes)], strict=True):
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color)
    image.save(path, format="JPEG", quality=100, subsampling=0)


def _write_expected_crop(raw: Path, crop: Path, box: tuple[int, int, int, int]) -> None:
    with Image.open(raw) as decoded:
        decoded.convert("RGB").crop(box).save(crop, format="PNG")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


@pytest.mark.parametrize(("size", "boxes"), _RECIPES)
def test_historical_crop_recipes_are_closed(
    size: tuple[int, int], boxes: tuple[tuple[int, int, int, int], ...]
) -> None:
    recipe = recipe_for_screenshot_size(size)
    assert isinstance(recipe, CropRecipe)
    assert recipe.screenshot_width == size[0]
    assert recipe.screenshot_height == size[1]
    assert recipe.panel_boxes == boxes
    assert len(HISTORICAL_CROP_RECIPES) == len(_RECIPES)


def test_unknown_screenshot_geometry_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "unknown.jpg"
    crop = tmp_path / "unknown.png"
    Image.new("RGB", (100, 100), "white").save(raw, format="JPEG", quality=100)
    Image.new("RGB", (10, 10), "white").save(crop, format="PNG")

    with pytest.raises(ProcessingProvenanceError, match="geometry|recipe|layout|unsupported"):
        recipe_for_screenshot_size((100, 100))
    with pytest.raises(ProcessingProvenanceError, match="geometry|recipe|layout|unsupported"):
        verify_registered_crop(raw, crop, panel_index=0)


@pytest.mark.parametrize("panel_index", (-1, 2, False, True))
def test_panel_index_must_be_a_valid_integer(tmp_path: Path, panel_index: object) -> None:
    raw = tmp_path / "c8-2and3.jpg"
    crop = tmp_path / "panel.png"
    _write_raw(raw, (996, 581), ((30, 33, 370, 371), (464, 33, 816, 371)))
    _write_expected_crop(raw, crop, (30, 33, 370, 371))

    with pytest.raises(ProcessingProvenanceError, match="panel"):
        verify_registered_crop(raw, crop, panel_index=panel_index)  # type: ignore[arg-type]


def test_verifier_rejects_raw_symlink(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jpg"
    raw_link = tmp_path / "raw-link.jpg"
    crop = tmp_path / "crop.png"
    box = (31, 33, 706, 707)
    _write_raw(raw, (891, 891), (box,))
    _write_expected_crop(raw, crop, box)
    raw_link.symlink_to(raw)

    with pytest.raises(ProcessingProvenanceError, match="regular file|symlink"):
        verify_registered_crop(raw_link, crop, panel_index=0)


def test_verifier_rejects_registered_crop_symlink(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jpg"
    crop = tmp_path / "crop.png"
    crop_link = tmp_path / "crop-link.png"
    box = (31, 33, 706, 707)
    _write_raw(raw, (891, 891), (box,))
    _write_expected_crop(raw, crop, box)
    crop_link.symlink_to(crop)

    with pytest.raises(ProcessingProvenanceError, match="regular file|symlink"):
        verify_registered_crop(raw, crop_link, panel_index=0)


@pytest.mark.parametrize("drift", ("raw", "registered"))
def test_verifier_rejects_expected_file_sha_drift(tmp_path: Path, drift: str) -> None:
    raw = tmp_path / "raw.jpg"
    crop = tmp_path / "crop.png"
    box = (31, 33, 706, 707)
    _write_raw(raw, (891, 891), (box,))
    _write_expected_crop(raw, crop, box)
    raw_sha256 = _sha256_bytes(raw.read_bytes())
    crop_sha256 = _sha256_bytes(crop.read_bytes())
    if drift == "raw":
        raw_sha256 = "0" * 64
    else:
        crop_sha256 = "0" * 64

    with pytest.raises(ProcessingProvenanceError, match="SHA-256|hash"):
        verify_registered_crop(
            raw,
            crop,
            panel_index=0,
            expected_raw_sha256=raw_sha256,
            expected_registered_sha256=crop_sha256,
        )


def test_registered_crop_rejects_non_rgb_image(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jpg"
    crop = tmp_path / "crop.png"
    box = (31, 33, 706, 707)
    _write_raw(raw, (891, 891), (box,))
    Image.new("RGBA", (675, 674), (255, 0, 0, 128)).save(crop, format="PNG")

    with pytest.raises(ProcessingProvenanceError, match="RGB|mode"):
        verify_registered_crop(raw, crop, panel_index=0)


def test_verifier_signature_is_closed_to_raw_crop_provenance() -> None:
    parameters = inspect.signature(verify_registered_crop).parameters
    assert tuple(parameters) == (
        "raw_path",
        "registered_crop_path",
        "panel_index",
        "expected_raw_sha256",
        "expected_registered_sha256",
    )
    assert parameters["raw_path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["registered_crop_path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["panel_index"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["expected_raw_sha256"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["expected_registered_sha256"].kind is inspect.Parameter.KEYWORD_ONLY
    forbidden = {
        "surface",
        "surface_path",
        "surface_image",
        "orientation",
        "cai",
        "oracle",
        "oracle_action",
        "damage",
        "damage_mask",
        "damage_centroid",
        "target_domain",
        "target_domain_label",
    }
    assert forbidden.isdisjoint(parameters)


def test_provenance_record_declares_no_pixel_processing(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jpg"
    crop = tmp_path / "crop.png"
    box = (31, 33, 706, 707)
    _write_raw(raw, (891, 891), (box,))
    _write_expected_crop(raw, crop, box)

    result = verify_registered_crop(raw, crop, panel_index=0)

    assert isinstance(result, ScanProcessingProvenance)
    assert result.panel_index == 0
    assert result.panel_box == box
    assert result.decoded_pixel_equal is True
    assert (result.resize, result.interpolation, result.rotation, result.reflection) == (
        "NONE",
        "NONE",
        "IDENTITY",
        "NONE",
    )
