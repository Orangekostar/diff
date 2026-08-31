"""Replay fixed historical raw-screenshot to registered-crop processing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_HEX = frozenset("0123456789abcdef")


class ProcessingProvenanceError(ValueError):
    """Raised when a historical C-scan crop cannot be reproduced exactly."""


@dataclass(frozen=True, slots=True)
class CropRecipe:
    screenshot_width: int
    screenshot_height: int
    panel_boxes: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        if (
            type(self.screenshot_width) is not int
            or type(self.screenshot_height) is not int
            or self.screenshot_width <= 0
            or self.screenshot_height <= 0
            or not self.panel_boxes
        ):
            raise ValueError("historical crop recipe is invalid")
        for box in self.panel_boxes:
            if (
                type(box) is not tuple
                or len(box) != 4
                or any(type(value) is not int for value in box)
            ):
                raise ValueError("historical crop recipe is invalid")
            x0, y0, x1, y1 = box
            if (
                x0 < 0
                or y0 < 0
                or x0 >= x1
                or y0 >= y1
                or x1 > self.screenshot_width
                or y1 > self.screenshot_height
            ):
                raise ValueError("historical crop recipe is invalid")

    @property
    def screenshot_size(self) -> tuple[int, int]:
        return self.screenshot_width, self.screenshot_height


HISTORICAL_CROP_RECIPES = (
    CropRecipe(891, 891, ((31, 33, 706, 707),)),
    CropRecipe(669, 885, ((39, 33, 469, 708),)),
    CropRecipe(
        996,
        581,
        ((30, 33, 370, 371), (464, 33, 816, 371)),
    ),
)
_RECIPES_BY_SIZE = {recipe.screenshot_size: recipe for recipe in HISTORICAL_CROP_RECIPES}


@dataclass(frozen=True, slots=True)
class ScanProcessingProvenance:
    schema_version: int
    screenshot_width_px: int
    screenshot_height_px: int
    panel_count: int
    panel_index: int
    panel_box: tuple[int, int, int, int]
    recovered_width_px: int
    recovered_height_px: int
    registered_width_px: int
    registered_height_px: int
    raw_file_sha256: str
    registered_crop_file_sha256: str
    raw_decoded_rgb_sha256: str
    recovered_panel_rgb_sha256: str
    registered_crop_rgb_sha256: str
    decoded_pixel_equal: bool
    decode_mode: str
    operation: str
    resize: str
    interpolation: str
    rotation: str
    reflection: str

    def as_dict(self) -> dict[str, object]:
        x0, y0, x1, y1 = self.panel_box
        return {
            "schema_version": self.schema_version,
            "screenshot_width_px": self.screenshot_width_px,
            "screenshot_height_px": self.screenshot_height_px,
            "panel_count": self.panel_count,
            "panel_index": self.panel_index,
            "panel_x0": x0,
            "panel_y0": y0,
            "panel_x1": x1,
            "panel_y1": y1,
            "recovered_width_px": self.recovered_width_px,
            "recovered_height_px": self.recovered_height_px,
            "registered_width_px": self.registered_width_px,
            "registered_height_px": self.registered_height_px,
            "raw_file_sha256": self.raw_file_sha256,
            "registered_crop_file_sha256": self.registered_crop_file_sha256,
            "raw_decoded_rgb_sha256": self.raw_decoded_rgb_sha256,
            "recovered_panel_rgb_sha256": self.recovered_panel_rgb_sha256,
            "registered_crop_rgb_sha256": self.registered_crop_rgb_sha256,
            "decoded_pixel_equal": self.decoded_pixel_equal,
            "decode_mode": self.decode_mode,
            "operation": self.operation,
            "resize": self.resize,
            "interpolation": self.interpolation,
            "rotation": self.rotation,
            "reflection": self.reflection,
        }


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and not set(value) - _HEX


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ProcessingProvenanceError("C-scan file cannot be read") from error
    return digest.hexdigest()


def _pixel_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _regular_file(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ProcessingProvenanceError(f"{label} must be a regular file, not a symlink")
    return candidate


def recipe_for_screenshot_size(size: tuple[int, int]) -> CropRecipe:
    if (
        type(size) is not tuple
        or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
    ):
        raise ProcessingProvenanceError("screenshot geometry is invalid")
    recipe = _RECIPES_BY_SIZE.get(size)
    if recipe is None:
        raise ProcessingProvenanceError("unsupported historical screenshot geometry")
    return recipe


def _decode_rgb(path: Path, label: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise ProcessingProvenanceError(f"{label} mode must be RGB")
            image.load()
            return image.copy()
    except ProcessingProvenanceError:
        raise
    except (OSError, UnidentifiedImageError) as error:
        raise ProcessingProvenanceError(f"{label} cannot be decoded") from error


def verify_registered_crop(
    raw_path: str | Path,
    registered_crop_path: str | Path,
    *,
    panel_index: int,
    expected_raw_sha256: str | None = None,
    expected_registered_sha256: str | None = None,
) -> ScanProcessingProvenance:
    """Verify one known panel crop without accepting orientation evidence."""

    if type(panel_index) is not int or panel_index < 0:
        raise ProcessingProvenanceError("panel index must be a nonnegative integer")
    for expected in (expected_raw_sha256, expected_registered_sha256):
        if expected is not None and not _is_sha256(expected):
            raise ProcessingProvenanceError("expected SHA-256 is invalid")
    raw = _regular_file(raw_path, "raw C-scan screenshot")
    registered = _regular_file(registered_crop_path, "registered C-scan crop")
    raw_file_sha256 = _file_sha256(raw)
    registered_file_sha256 = _file_sha256(registered)
    if expected_raw_sha256 is not None and raw_file_sha256 != expected_raw_sha256:
        raise ProcessingProvenanceError("raw C-scan SHA-256 changed")
    if (
        expected_registered_sha256 is not None
        and registered_file_sha256 != expected_registered_sha256
    ):
        raise ProcessingProvenanceError("registered C-scan SHA-256 changed")

    raw_image = _decode_rgb(raw, "raw C-scan screenshot")
    registered_image = _decode_rgb(registered, "registered C-scan crop")
    recipe = recipe_for_screenshot_size(raw_image.size)
    if panel_index >= len(recipe.panel_boxes):
        raise ProcessingProvenanceError("panel index is outside the historical layout")
    panel_box = recipe.panel_boxes[panel_index]
    recovered = raw_image.crop(panel_box)
    recovered_hash = _pixel_sha256(recovered)
    registered_hash = _pixel_sha256(registered_image)
    decoded_equal = (
        recovered.size == registered_image.size
        and recovered_hash == registered_hash
        and recovered.tobytes() == registered_image.tobytes()
    )
    if not decoded_equal:
        raise ProcessingProvenanceError(
            "registered crop decoded RGB pixels differ from the historical panel"
        )
    return ScanProcessingProvenance(
        schema_version=1,
        screenshot_width_px=raw_image.width,
        screenshot_height_px=raw_image.height,
        panel_count=len(recipe.panel_boxes),
        panel_index=panel_index,
        panel_box=panel_box,
        recovered_width_px=recovered.width,
        recovered_height_px=recovered.height,
        registered_width_px=registered_image.width,
        registered_height_px=registered_image.height,
        raw_file_sha256=raw_file_sha256,
        registered_crop_file_sha256=registered_file_sha256,
        raw_decoded_rgb_sha256=_pixel_sha256(raw_image),
        recovered_panel_rgb_sha256=recovered_hash,
        registered_crop_rgb_sha256=registered_hash,
        decoded_pixel_equal=True,
        decode_mode="RGB",
        operation="AXIS_ALIGNED_CROP",
        resize="NONE",
        interpolation="NONE",
        rotation="IDENTITY",
        reflection="NONE",
    )


__all__ = [
    "HISTORICAL_CROP_RECIPES",
    "CropRecipe",
    "ProcessingProvenanceError",
    "ScanProcessingProvenance",
    "recipe_for_screenshot_size",
    "verify_registered_crop",
]
