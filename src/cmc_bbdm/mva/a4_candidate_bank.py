"""Checksum-bound initial-candidate banks for the MVA A4 study."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from .acquisition_grid import INITIAL_BUDGETS, build_acquisition_grid
from .appearance_value import appearance_intensity_value
from .interpolation import (
    RefinementPatchCache,
    reconstruct_measurement_state,
    refine_reconstruction,
)
from .measurement_state import (
    RefinementAction,
    apply_action,
    budget_record,
    initial_state,
    measurement_mask,
)
from .reconstruction_value import normalized_rgb_mse


class CandidateBankError(ValueError):
    """Raised when an A4 candidate bank is incomplete or has drifted."""


class _Encoder(Protocol):
    def encode(self, images: list[np.ndarray]) -> np.ndarray: ...

    def validate(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CandidateBank:
    schema_version: int
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    image_sha256: tuple[str, ...]
    decoded_image_sha256: tuple[str, ...]
    authority_state_sha256: str
    initial_budget: float
    interpolation: str
    native_shapes: tuple[tuple[int, int], ...]
    grid_state_sha256: tuple[str, ...]
    initial_measured_counts: tuple[int, ...]
    native_counts: tuple[int, ...]
    cell_indices: tuple[int, ...]
    from_levels: tuple[int, ...]
    to_levels: tuple[int, ...]
    initial_output_sha256: tuple[str, ...]
    candidate_output_sha256: tuple[tuple[str, ...], ...]
    initial_embeddings: np.ndarray
    embeddings: np.ndarray
    reconstruction_values: np.ndarray
    appearance_values: np.ndarray
    added_measurements: np.ndarray
    state_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)


_ARRAY_NAMES = (
    "added_measurements",
    "appearance_values",
    "embeddings",
    "initial_embeddings",
    "reconstruction_values",
)


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")
    ).hexdigest()


def _readonly(value: object, *, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    if array.shape != shape:
        raise CandidateBankError("candidate bank array shape changed")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _metadata(bank: CandidateBank) -> dict[str, object]:
    return {
        "authority_state_sha256": bank.authority_state_sha256,
        "candidate_output_sha256": bank.candidate_output_sha256,
        "cell_indices": bank.cell_indices,
        "dataset_ids": bank.dataset_ids,
        "decoded_image_sha256": bank.decoded_image_sha256,
        "from_levels": bank.from_levels,
        "grid_state_sha256": bank.grid_state_sha256,
        "image_sha256": bank.image_sha256,
        "initial_budget": bank.initial_budget,
        "initial_measured_counts": bank.initial_measured_counts,
        "initial_output_sha256": bank.initial_output_sha256,
        "interpolation": bank.interpolation,
        "native_counts": bank.native_counts,
        "native_shapes": bank.native_shapes,
        "schema_version": bank.schema_version,
        "specimen_ids": bank.specimen_ids,
        "to_levels": bank.to_levels,
    }


def _state_sha256(bank: CandidateBank) -> str:
    digest = hashlib.sha256(
        json.dumps(
            _metadata(bank), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    )
    for name in _ARRAY_NAMES:
        value = np.asarray(getattr(bank, name))
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def validate_candidate_bank(bank: CandidateBank) -> None:
    """Validate shapes, provenance, numeric contents, and the canonical digest."""

    if type(bank) is not CandidateBank or bank.schema_version != 1:
        raise CandidateBankError("candidate bank schema changed")
    count = bank.specimen_count
    if (
        count < 1
        or len(set(bank.specimen_ids)) != count
        or any(type(value) is not str or not value for value in bank.specimen_ids)
        or len(bank.dataset_ids) != count
        or any(type(value) is not str or not value for value in bank.dataset_ids)
    ):
        raise CandidateBankError("candidate bank specimen roster changed")
    if (
        len(bank.image_sha256) != count
        or not all(_is_sha256(value) for value in bank.image_sha256)
    ):
        raise CandidateBankError("candidate bank image hash changed")
    if (
        len(bank.decoded_image_sha256) != count
        or not all(_is_sha256(value) for value in bank.decoded_image_sha256)
        or not _is_sha256(bank.authority_state_sha256)
    ):
        raise CandidateBankError("candidate bank authority hash changed")
    if (
        bank.initial_budget not in INITIAL_BUDGETS
        or bank.interpolation != "bilinear"
        or bank.cell_indices != tuple(range(64))
        or bank.from_levels != (0,) * 64
        or bank.to_levels != (1,) * 64
    ):
        raise CandidateBankError("candidate bank acquisition contract changed")
    if (
        len(bank.native_shapes) != count
        or len(bank.grid_state_sha256) != count
        or not all(_is_sha256(value) for value in bank.grid_state_sha256)
        or len(bank.initial_measured_counts) != count
        or len(bank.native_counts) != count
        or any(
            type(height) is not int
            or type(width) is not int
            or height < 9
            or width < 9
            for height, width in bank.native_shapes
        )
        or any(
            type(initial) is not int
            or type(native) is not int
            or not 0 < initial < native
            or native != height * width
            for initial, native, (height, width) in zip(
                bank.initial_measured_counts,
                bank.native_counts,
                bank.native_shapes,
                strict=True,
            )
        )
    ):
        raise CandidateBankError("candidate bank geometry changed")
    if (
        len(bank.initial_output_sha256) != count
        or not all(_is_sha256(value) for value in bank.initial_output_sha256)
        or len(bank.candidate_output_sha256) != count
        or any(
            len(row) != 64 or not all(_is_sha256(value) for value in row)
            for row in bank.candidate_output_sha256
        )
    ):
        raise CandidateBankError("candidate reconstruction hash changed")

    expected_shapes = {
        "initial_embeddings": (count, 512),
        "embeddings": (count, 64, 512),
        "reconstruction_values": (count, 64),
        "appearance_values": (count, 64),
        "added_measurements": (count, 64),
    }
    expected_dtypes = {
        "initial_embeddings": np.dtype("<f8"),
        "embeddings": np.dtype("<f8"),
        "reconstruction_values": np.dtype("<f8"),
        "appearance_values": np.dtype("<f8"),
        "added_measurements": np.dtype("<i8"),
    }
    for name in _ARRAY_NAMES:
        value = getattr(bank, name)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != expected_shapes[name]
            or value.dtype != expected_dtypes[name]
        ):
            raise CandidateBankError("candidate bank array contract changed")
        if name == "added_measurements":
            if np.any(value <= 0):
                raise CandidateBankError("candidate bank measurement count changed")
        elif not np.all(np.isfinite(value)):
            raise CandidateBankError("candidate bank contains nonfinite values")
    if np.any(bank.appearance_values < 0.0):
        raise CandidateBankError("candidate bank appearance value changed")
    if not _is_sha256(bank.state_sha256) or _state_sha256(bank) != bank.state_sha256:
        raise CandidateBankError("candidate bank content digest changed")


def build_candidate_bank(
    *,
    specimen_ids: tuple[str, ...],
    dataset_ids: tuple[str, ...],
    images: tuple[np.ndarray, ...],
    image_sha256: tuple[str, ...],
    authority_state_sha256: str,
    initial_budget: float,
    encoder: _Encoder,
) -> CandidateBank:
    """Build all 64 initial level-0 to level-1 candidates per specimen."""

    count = len(specimen_ids)
    if (
        type(specimen_ids) is not tuple
        or type(dataset_ids) is not tuple
        or type(images) is not tuple
        or type(image_sha256) is not tuple
        or count < 1
        or len(dataset_ids) != count
        or len(images) != count
        or len(image_sha256) != count
    ):
        raise CandidateBankError("candidate bank inputs are incomplete")
    if not all(_is_sha256(value) for value in image_sha256):
        raise CandidateBankError("source image hash is invalid")
    if not _is_sha256(authority_state_sha256):
        raise CandidateBankError("authority state hash is invalid")
    if initial_budget not in INITIAL_BUDGETS:
        raise CandidateBankError("initial budget is not registered")
    encode = getattr(encoder, "encode", None)
    encoder_validate = getattr(encoder, "validate", None)
    if not callable(encode) or not callable(encoder_validate):
        raise CandidateBankError("validated encoder is required")

    initial_embeddings = np.empty((count, 512), dtype="<f8")
    embeddings = np.empty((count, 64, 512), dtype="<f8")
    reconstruction_values = np.empty((count, 64), dtype="<f8")
    appearance_values = np.empty((count, 64), dtype="<f8")
    added_measurements = np.empty((count, 64), dtype="<i8")
    decoded_hashes: list[str] = []
    native_shapes: list[tuple[int, int]] = []
    grid_hashes: list[str] = []
    initial_counts: list[int] = []
    native_counts: list[int] = []
    initial_output_hashes: list[str] = []
    candidate_output_hashes: list[tuple[str, ...]] = []

    for index, (specimen_id, dataset_id, image) in enumerate(
        zip(specimen_ids, dataset_ids, images, strict=True)
    ):
        if (
            type(specimen_id) is not str
            or not specimen_id
            or type(dataset_id) is not str
            or not dataset_id
            or not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise CandidateBankError("candidate bank specimen input is invalid")
        height, width = image.shape[:2]
        grid = build_acquisition_grid(height, width, initial_budget=initial_budget)
        state = initial_state(grid)
        current_result = reconstruct_measurement_state(
            image,
            grid,
            state,
            interpolation="bilinear",
            specimen_id=specimen_id,
            dataset_id=dataset_id,
        )
        current = current_result.image
        current_mask = measurement_mask(grid, state)
        current_error = normalized_rgb_mse(image, current)
        patch_cache = RefinementPatchCache(image=image, grid=grid)
        candidates: list[np.ndarray] = []
        output_hashes: list[str] = []
        for cell_index in range(64):
            action = RefinementAction(cell_index, 0, 1)
            candidate = refine_reconstruction(
                image,
                grid,
                state,
                current,
                action,
                interpolation="bilinear",
                current_mask=current_mask,
                patch_cache=patch_cache,
            )
            candidate_state = apply_action(grid, state, action)
            candidate_mask = measurement_mask(grid, candidate_state)
            if not np.array_equal(candidate[candidate_mask], image[candidate_mask]):
                raise CandidateBankError("candidate lost a measured RGB value")
            added = int(np.count_nonzero(candidate_mask & ~current_mask))
            if added <= 0:
                raise CandidateBankError("candidate added no measurement")
            candidates.append(candidate)
            output_hashes.append(_array_sha256(candidate))
            reconstruction_values[index, cell_index] = current_error - (
                normalized_rgb_mse(image, candidate)
            )
            appearance_values[index, cell_index] = appearance_intensity_value(
                image, current_mask, candidate_mask
            )
            added_measurements[index, cell_index] = added

        encoded = np.asarray(encode([current, *candidates]), dtype="<f8")
        if encoded.shape != (65, 512) or not np.all(np.isfinite(encoded)):
            raise CandidateBankError("candidate encoder output is invalid")
        initial_embeddings[index] = encoded[0]
        embeddings[index] = encoded[1:]
        initial_record = budget_record(grid, state)
        decoded_hashes.append(_array_sha256(image))
        native_shapes.append((height, width))
        grid_hashes.append(grid.state_sha256)
        initial_counts.append(initial_record.measured_count)
        native_counts.append(initial_record.native_count)
        initial_output_hashes.append(current_result.output_sha256)
        candidate_output_hashes.append(tuple(output_hashes))

    encoder_validate()
    bank = CandidateBank(
        schema_version=1,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        image_sha256=image_sha256,
        decoded_image_sha256=tuple(decoded_hashes),
        authority_state_sha256=authority_state_sha256,
        initial_budget=float(initial_budget),
        interpolation="bilinear",
        native_shapes=tuple(native_shapes),
        grid_state_sha256=tuple(grid_hashes),
        initial_measured_counts=tuple(initial_counts),
        native_counts=tuple(native_counts),
        cell_indices=tuple(range(64)),
        from_levels=(0,) * 64,
        to_levels=(1,) * 64,
        initial_output_sha256=tuple(initial_output_hashes),
        candidate_output_sha256=tuple(candidate_output_hashes),
        initial_embeddings=_readonly(
            initial_embeddings, dtype="<f8", shape=(count, 512)
        ),
        embeddings=_readonly(embeddings, dtype="<f8", shape=(count, 64, 512)),
        reconstruction_values=_readonly(
            reconstruction_values, dtype="<f8", shape=(count, 64)
        ),
        appearance_values=_readonly(
            appearance_values, dtype="<f8", shape=(count, 64)
        ),
        added_measurements=_readonly(
            added_measurements, dtype="<i8", shape=(count, 64)
        ),
        state_sha256="",
    )
    output = replace(bank, state_sha256=_state_sha256(bank))
    validate_candidate_bank(output)
    return output


def save_candidate_bank(path: str | Path, bank: CandidateBank) -> Path:
    """Atomically save a validated candidate bank."""

    validate_candidate_bank(bank)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": np.asarray([bank.schema_version], dtype="<i8"),
        "specimen_ids": np.asarray(bank.specimen_ids),
        "dataset_ids": np.asarray(bank.dataset_ids),
        "image_sha256": np.asarray(bank.image_sha256),
        "decoded_image_sha256": np.asarray(bank.decoded_image_sha256),
        "authority_state_sha256": np.asarray([bank.authority_state_sha256]),
        "initial_budget": np.asarray([bank.initial_budget], dtype="<f8"),
        "interpolation": np.asarray([bank.interpolation]),
        "native_shapes": np.asarray(bank.native_shapes, dtype="<i8"),
        "grid_state_sha256": np.asarray(bank.grid_state_sha256),
        "initial_measured_counts": np.asarray(
            bank.initial_measured_counts, dtype="<i8"
        ),
        "native_counts": np.asarray(bank.native_counts, dtype="<i8"),
        "cell_indices": np.asarray(bank.cell_indices, dtype="<i8"),
        "from_levels": np.asarray(bank.from_levels, dtype="<i8"),
        "to_levels": np.asarray(bank.to_levels, dtype="<i8"),
        "initial_output_sha256": np.asarray(bank.initial_output_sha256),
        "candidate_output_sha256": np.asarray(bank.candidate_output_sha256),
        "initial_embeddings": bank.initial_embeddings,
        "embeddings": bank.embeddings,
        "reconstruction_values": bank.reconstruction_values,
        "appearance_values": bank.appearance_values,
        "added_measurements": bank.added_measurements,
        "state_sha256": np.asarray([bank.state_sha256]),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return destination


def _strings(value: np.ndarray) -> tuple[str, ...]:
    if value.ndim != 1:
        raise CandidateBankError("candidate bank string array changed")
    return tuple(str(item) for item in value)


def load_candidate_bank(
    path: str | Path,
    *,
    expected_authority_state_sha256: str | None = None,
    expected_specimen_ids: tuple[str, ...] | None = None,
    expected_image_sha256: tuple[str, ...] | None = None,
    expected_initial_budget: float | None = None,
) -> CandidateBank:
    """Load and optionally cross-bind a candidate bank to current authority."""

    expected_keys = {
        "added_measurements",
        "appearance_values",
        "authority_state_sha256",
        "candidate_output_sha256",
        "cell_indices",
        "dataset_ids",
        "decoded_image_sha256",
        "embeddings",
        "from_levels",
        "grid_state_sha256",
        "image_sha256",
        "initial_budget",
        "initial_embeddings",
        "initial_measured_counts",
        "initial_output_sha256",
        "interpolation",
        "native_counts",
        "native_shapes",
        "reconstruction_values",
        "schema_version",
        "specimen_ids",
        "state_sha256",
        "to_levels",
    }
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise CandidateBankError("candidate bank archive keys changed")
            specimen_ids = _strings(archive["specimen_ids"])
            count = len(specimen_ids)
            candidate_hashes_array = np.asarray(archive["candidate_output_sha256"])
            if candidate_hashes_array.shape != (count, 64):
                raise CandidateBankError("candidate reconstruction hash changed")
            bank = CandidateBank(
                schema_version=int(archive["schema_version"][0]),
                specimen_ids=specimen_ids,
                dataset_ids=_strings(archive["dataset_ids"]),
                image_sha256=_strings(archive["image_sha256"]),
                decoded_image_sha256=_strings(archive["decoded_image_sha256"]),
                authority_state_sha256=str(archive["authority_state_sha256"][0]),
                initial_budget=float(archive["initial_budget"][0]),
                interpolation=str(archive["interpolation"][0]),
                native_shapes=tuple(
                    (int(row[0]), int(row[1]))
                    for row in np.asarray(archive["native_shapes"])
                ),
                grid_state_sha256=_strings(archive["grid_state_sha256"]),
                initial_measured_counts=tuple(
                    int(value) for value in archive["initial_measured_counts"]
                ),
                native_counts=tuple(int(value) for value in archive["native_counts"]),
                cell_indices=tuple(int(value) for value in archive["cell_indices"]),
                from_levels=tuple(int(value) for value in archive["from_levels"]),
                to_levels=tuple(int(value) for value in archive["to_levels"]),
                initial_output_sha256=_strings(archive["initial_output_sha256"]),
                candidate_output_sha256=tuple(
                    tuple(str(value) for value in row)
                    for row in candidate_hashes_array
                ),
                initial_embeddings=_readonly(
                    archive["initial_embeddings"], dtype="<f8", shape=(count, 512)
                ),
                embeddings=_readonly(
                    archive["embeddings"], dtype="<f8", shape=(count, 64, 512)
                ),
                reconstruction_values=_readonly(
                    archive["reconstruction_values"],
                    dtype="<f8",
                    shape=(count, 64),
                ),
                appearance_values=_readonly(
                    archive["appearance_values"],
                    dtype="<f8",
                    shape=(count, 64),
                ),
                added_measurements=_readonly(
                    archive["added_measurements"],
                    dtype="<i8",
                    shape=(count, 64),
                ),
                state_sha256=str(archive["state_sha256"][0]),
            )
    except CandidateBankError:
        raise
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        raise CandidateBankError("candidate bank cannot be loaded") from error
    validate_candidate_bank(bank)
    if (
        expected_authority_state_sha256 is not None
        and bank.authority_state_sha256 != expected_authority_state_sha256
    ):
        raise CandidateBankError("candidate bank authority changed")
    if expected_specimen_ids is not None and bank.specimen_ids != expected_specimen_ids:
        raise CandidateBankError("candidate bank specimen roster changed")
    if expected_image_sha256 is not None and bank.image_sha256 != expected_image_sha256:
        raise CandidateBankError("candidate bank image authority changed")
    if (
        expected_initial_budget is not None
        and bank.initial_budget != expected_initial_budget
    ):
        raise CandidateBankError("candidate bank initial budget changed")
    return bank


def candidate_bank_path(project_root: str | Path, initial_budget: float) -> Path:
    """Return the preregistered work-cache path for one initial budget."""

    if initial_budget not in INITIAL_BUDGETS:
        raise CandidateBankError("initial budget is not registered")
    token = str(initial_budget).replace(".", "p")
    return Path(project_root) / "results/mva/.work" / f"a4_candidate_bank_{token}.npz"


__all__ = [
    "CandidateBank",
    "CandidateBankError",
    "build_candidate_bank",
    "candidate_bank_path",
    "load_candidate_bank",
    "save_candidate_bank",
    "validate_candidate_bank",
]
