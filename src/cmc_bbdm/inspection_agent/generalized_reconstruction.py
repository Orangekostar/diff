"""Source-safe reconstruction for cells that may remain unmeasured."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.mva.interpolation import _interpolate_rectilinear

from .contracts import InspectionObservation
from .state import GeneralizedMeasurementState, budget_record, measurement_mask


class GeneralizedReconstructionError(ValueError):
    """Raised when a source prior or observed-only reconstruction is invalid."""


def _readonly(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise GeneralizedReconstructionError("reconstruction array is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _border_median(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise GeneralizedReconstructionError("source C-scan is invalid")
    border = np.concatenate(
        (
            image[0],
            image[-1],
            image[1:-1, 0],
            image[1:-1, -1],
        ),
        axis=0,
    )
    return np.median(border.astype(np.float64), axis=0)


@dataclass(frozen=True, slots=True)
class SourceBackgroundPrior:
    outer_domain: str
    source_domains: tuple[str, ...]
    fit_specimen_ids: tuple[str, ...]
    source_authority_sha256: str
    domain_border_medians: np.ndarray
    background_rgb: np.ndarray
    state_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.outer_domain) is not str
            or not self.outer_domain
            or type(self.source_domains) is not tuple
            or not self.source_domains
            or self.outer_domain in self.source_domains
            or len(set(self.source_domains)) != len(self.source_domains)
            or type(self.fit_specimen_ids) is not tuple
            or not self.fit_specimen_ids
            or len(set(self.fit_specimen_ids)) != len(self.fit_specimen_ids)
            or not _valid_sha256(self.source_authority_sha256)
        ):
            raise GeneralizedReconstructionError("source prior identity is invalid")
        domains = _readonly(
            self.domain_border_medians,
            dtype="<f8",
            shape=(len(self.source_domains), 3),
        )
        background = _readonly(self.background_rgb, dtype=np.uint8, shape=(3,))
        digest = hashlib.sha256()
        digest.update(b"inspection-agent-source-background-prior-v1")
        digest.update(
            json.dumps(
                {
                    "outer_domain": self.outer_domain,
                    "source_domains": self.source_domains,
                    "fit_specimen_ids": self.fit_specimen_ids,
                    "source_authority_sha256": self.source_authority_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(domains.tobytes(order="C"))
        digest.update(background.tobytes(order="C"))
        state = digest.hexdigest()
        if self.state_sha256 not in ("", state):
            raise GeneralizedReconstructionError("source prior hash changed")
        object.__setattr__(self, "domain_border_medians", domains)
        object.__setattr__(self, "background_rgb", background)
        object.__setattr__(self, "state_sha256", state)


@dataclass(frozen=True, slots=True)
class GeneralizedReconstruction:
    image: np.ndarray
    measured_count: int
    native_count: int
    effective_budget: float
    observed_values_exact: bool
    state_sha256: str


def fit_source_background_prior(
    authority: MAVISAuthority,
    *,
    outer_domain: str,
) -> SourceBackgroundPrior:
    if (
        type(authority) is not MAVISAuthority
        or type(outer_domain) is not str
        or outer_domain not in authority.dataset_ids
    ):
        raise GeneralizedReconstructionError("source-prior fit request is invalid")
    source_domains = tuple(
        dict.fromkeys(
            domain for domain in authority.dataset_ids if domain != outer_domain
        )
    )
    if not source_domains:
        raise GeneralizedReconstructionError("source-prior fit has no source domain")
    fit_ids = tuple(
        specimen
        for specimen, domain in zip(
            authority.specimen_ids, authority.dataset_ids, strict=True
        )
        if domain != outer_domain
    )
    domain_rows: list[np.ndarray] = []
    for domain in source_domains:
        medians = [
            _border_median(authority.source_teacher_view(specimen).full_scan)
            for specimen, specimen_domain in zip(
                authority.specimen_ids, authority.dataset_ids, strict=True
            )
            if specimen_domain == domain
        ]
        if not medians:
            raise GeneralizedReconstructionError("source-prior domain is empty")
        domain_rows.append(np.mean(np.asarray(medians), axis=0, dtype=np.float64))
    domain_medians = np.asarray(domain_rows, dtype=np.float64)
    background = np.rint(np.mean(domain_medians, axis=0, dtype=np.float64)).clip(
        0, 255
    ).astype(np.uint8)
    return SourceBackgroundPrior(
        outer_domain=outer_domain,
        source_domains=source_domains,
        fit_specimen_ids=fit_ids,
        source_authority_sha256=authority.state_sha256,
        domain_border_medians=domain_medians,
        background_rgb=background,
    )


def _measurement_arrays(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    acquired_positions: np.ndarray,
    measurement_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected_mask = measurement_mask(grid, state)
    expected_positions = np.argwhere(expected_mask).astype("<i8", copy=False)
    positions = np.asarray(acquired_positions)
    values = np.asarray(measurement_values)
    if (
        positions.dtype.kind not in "iu"
        or positions.shape != expected_positions.shape
        or values.dtype != np.uint8
        or values.shape != (len(expected_positions), 3)
        or not np.array_equal(positions, expected_positions)
    ):
        raise GeneralizedReconstructionError("observations do not match the state mask")
    dense = np.zeros((*grid.native_shape, 3), dtype=np.uint8)
    if len(positions):
        dense[positions[:, 0], positions[:, 1]] = values
    return expected_mask, dense


def reconstruct_measurements(
    grid: AcquisitionGrid,
    state: GeneralizedMeasurementState,
    acquired_positions: np.ndarray,
    measurement_values: np.ndarray,
    prior: SourceBackgroundPrior,
    *,
    interpolation: str = "bilinear",
) -> GeneralizedReconstruction:
    if (
        type(grid) is not AcquisitionGrid
        or type(state) is not GeneralizedMeasurementState
        or type(prior) is not SourceBackgroundPrior
        or state.grid_sha256 != grid.state_sha256
        or interpolation != "bilinear"
    ):
        raise GeneralizedReconstructionError("reconstruction request is invalid")
    observed, dense = _measurement_arrays(
        grid,
        state,
        acquired_positions,
        measurement_values,
    )
    if state.levels == (2,) * 64:
        reconstruction = dense.copy()
    else:
        reconstruction = np.empty((*grid.native_shape, 3), dtype=np.uint8)
        reconstruction[:] = prior.background_rgb
        for cell, level in zip(grid.cells, state.levels, strict=True):
            if level < 0:
                continue
            rows = cell.rows[level]
            columns = cell.columns[level]
            row_start, row_stop = cell.rows[2][0], cell.rows[2][-1]
            column_start, column_stop = cell.columns[2][0], cell.columns[2][-1]
            if not np.all(observed[np.ix_(rows, columns)]):
                raise GeneralizedReconstructionError("cell lattice is not fully observed")
            patch = _interpolate_rectilinear(
                np.ascontiguousarray(dense[np.ix_(rows, columns)]),
                rows,
                columns,
                np.arange(row_start, row_stop + 1, dtype=np.int64),
                np.arange(column_start, column_stop + 1, dtype=np.int64),
                interpolation,
            )
            owned_row_stop = row_stop + 1 if cell.row == 7 else row_stop
            owned_column_stop = column_stop + 1 if cell.column == 7 else column_stop
            patch_row_stop = patch.shape[0] if cell.row == 7 else patch.shape[0] - 1
            patch_column_stop = (
                patch.shape[1] if cell.column == 7 else patch.shape[1] - 1
            )
            reconstruction[
                row_start:owned_row_stop,
                column_start:owned_column_stop,
            ] = patch[:patch_row_stop, :patch_column_stop]
        reconstruction[observed] = dense[observed]
    exact = bool(np.array_equal(reconstruction[observed], dense[observed]))
    if not exact:
        raise GeneralizedReconstructionError("observed pixels were not restored")
    record = budget_record(grid, state)
    frozen = _readonly(
        reconstruction,
        dtype=np.uint8,
        shape=(*grid.native_shape, 3),
    )
    digest = hashlib.sha256()
    digest.update(b"inspection-agent-generalized-reconstruction-v1")
    digest.update(state.state_sha256.encode("ascii"))
    digest.update(prior.state_sha256.encode("ascii"))
    digest.update(interpolation.encode("ascii"))
    digest.update(frozen.tobytes(order="C"))
    return GeneralizedReconstruction(
        image=frozen,
        measured_count=record.measured_count,
        native_count=record.native_count,
        effective_budget=record.effective_budget,
        observed_values_exact=exact,
        state_sha256=digest.hexdigest(),
    )


def reconstruct_observation(
    observation: InspectionObservation,
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    *,
    interpolation: str = "bilinear",
) -> GeneralizedReconstruction:
    if (
        type(observation) is not InspectionObservation
        or observation.grid_sha256 != grid.state_sha256
        or observation.native_shape != grid.native_shape
    ):
        raise GeneralizedReconstructionError("observation does not match the grid")
    return reconstruct_measurements(
        grid,
        observation.measurement_state,
        observation.acquired_positions,
        observation.measurement_values,
        prior,
        interpolation=interpolation,
    )


__all__ = [
    "GeneralizedReconstruction",
    "GeneralizedReconstructionError",
    "SourceBackgroundPrior",
    "fit_source_background_prior",
    "reconstruct_measurements",
    "reconstruct_observation",
]
