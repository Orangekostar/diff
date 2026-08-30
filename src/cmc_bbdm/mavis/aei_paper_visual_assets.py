"""Read-only qualitative assets for the AEI paper figure pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.interpolation import reconstruct_measurement_state
from cmc_bbdm.mva.measurement_state import MeasurementState, measurement_mask

REPRESENTATIVE_SPECIMEN = "c8-2"
REPRESENTATIVE_METHOD = "one_shot_mechanical_oracle"
INITIAL_CHECKPOINT = 0.03125
LATER_CHECKPOINT = 0.1875

_STATE_MANIFEST = Path("results/mavis/p1_state_bank/state_manifest.parquet")
_STATE_PAYLOAD_ROOT = Path("results/mavis/p1_state_bank")
_ACTION_SCORES = Path("results/mavis/p3_dynamic_voi/action_scores.parquet")
_ORACLE_VALUES = Path("results/mva/a2_oracle_value/oracle_values.parquet")
_SCAN_MANIFEST = Path("artifacts/mavis_authority/scan_manifest.csv")


class AEIVisualAssetError(RuntimeError):
    """Raised when a requested paper visual is not bound to frozen evidence."""


@dataclass(frozen=True, slots=True)
class GallerySpecimen:
    """One deterministically selected specimen for a held-out domain."""

    domain_id: str
    specimen_id: str


@dataclass(frozen=True, slots=True, eq=False)
class ReconstructedState:
    """A hash-verified state reconstruction restored from compact measurements."""

    specimen_id: str
    domain_id: str
    method: str
    checkpoint: float
    state_id: str
    exact_acquired_cost: int
    native_count: int
    effective_budget: float
    image: np.ndarray
    measurement_mask: np.ndarray
    measurement_levels: tuple[int, ...]
    acquired_cell_indices: tuple[int, ...]
    row_boundaries: tuple[int, ...]
    column_boundaries: tuple[int, ...]
    output_sha256: str
    expected_output_sha256: str


@dataclass(frozen=True, slots=True, eq=False)
class PriorityState:
    """One registered state and its 8x8 strict-OOF teacher-value map."""

    state_id: str
    reconstruction: ReconstructedState
    raw_values: np.ndarray
    percentiles: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class TaskPriorityMaps:
    """Paired retrospective task-priority maps on one legal action grid."""

    specimen_id: str
    domain_id: str
    reconstruction: ReconstructedState
    cell_indices: tuple[int, ...]
    mechanical_values: np.ndarray
    reconstruction_values: np.ndarray
    mechanical_percentiles: np.ndarray
    reconstruction_percentiles: np.ndarray
    percentile_difference: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class TaskSaliencyMaps:
    """Paired retrospective CAI-task and appearance-saliency maps."""

    specimen_id: str
    domain_id: str
    reconstruction: ReconstructedState
    cell_indices: tuple[int, ...]
    mechanical_values: np.ndarray
    saliency_values: np.ndarray
    mechanical_percentiles: np.ndarray
    saliency_percentiles: np.ndarray
    percentile_difference: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class GalleryStatePair:
    """Initial and later priority states for one deterministic gallery specimen."""

    specimen: GallerySpecimen
    initial: PriorityState
    later: PriorityState


def _resolved_root(root: Path) -> Path:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise AEIVisualAssetError("project root is unavailable")
    return resolved


@lru_cache(maxsize=4)
def _state_manifest(root_text: str) -> pl.DataFrame:
    try:
        manifest = pl.read_parquet(Path(root_text) / _STATE_MANIFEST)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AEIVisualAssetError("state manifest is unavailable") from error
    required = {
        "specimen_id",
        "domain_id",
        "method",
        "state_id",
        "nominal_checkpoint",
        "initial_budget",
        "exact_acquired_cost",
        "native_count",
        "effective_budget",
        "measurement_levels",
        "acquired_action_cell_indices",
        "grid_state_sha256",
        "current_reconstruction_sha256",
        "measurement_payload_file",
    }
    if manifest.height == 0 or not required <= set(manifest.columns):
        raise AEIVisualAssetError("state manifest contract changed")
    return manifest


@lru_cache(maxsize=4)
def _scan_manifest(root_text: str) -> pl.DataFrame:
    try:
        manifest = pl.read_csv(Path(root_text) / _SCAN_MANIFEST)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AEIVisualAssetError("scan manifest is unavailable") from error
    required = {"specimen_id", "dataset_id", "height", "width", "native_count"}
    if manifest.height == 0 or not required <= set(manifest.columns):
        raise AEIVisualAssetError("scan manifest contract changed")
    return manifest


def _unique_row(frame: pl.DataFrame, description: str) -> dict[str, object]:
    if frame.height != 1:
        raise AEIVisualAssetError(f"{description} must resolve to exactly one row")
    return frame.row(0, named=True)


def _state_row(
    root: Path, *, specimen_id: str, method: str, checkpoint: float
) -> dict[str, object]:
    if (
        type(specimen_id) is not str
        or not specimen_id
        or type(method) is not str
        or not method
        or type(checkpoint) not in (float, int)
    ):
        raise AEIVisualAssetError("state request is invalid")
    manifest = _state_manifest(str(root))
    selected = manifest.filter(
        (pl.col("specimen_id") == specimen_id)
        & (pl.col("method") == method)
        & ((pl.col("nominal_checkpoint") - float(checkpoint)).abs() <= 1.0e-12)
    )
    return _unique_row(selected, "registered state")


def _specimen_row(root: Path, specimen_id: str) -> dict[str, object]:
    selected = _scan_manifest(str(root)).filter(pl.col("specimen_id") == specimen_id)
    return _unique_row(selected, "registered specimen")


def _measurement_payload(root: Path, row: dict[str, object]) -> dict[str, object]:
    relative = row["measurement_payload_file"]
    if type(relative) is not str:
        raise AEIVisualAssetError("measurement payload path is invalid")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.parts[:1] != ("revealed_measurements",)
    ):
        raise AEIVisualAssetError("measurement payload path escaped the state package")
    payload_path = root / _STATE_PAYLOAD_ROOT / relative_path
    try:
        selected = (
            pl.scan_parquet(payload_path)
            .filter(pl.col("state_id") == row["state_id"])
            .collect()
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AEIVisualAssetError("measurement payload is unavailable") from error
    return _unique_row(selected, "measurement payload")


def _readonly(array: np.ndarray) -> np.ndarray:
    output = np.ascontiguousarray(array)
    output.setflags(write=False)
    return output


def load_reconstructed_state(
    root: Path,
    *,
    specimen_id: str,
    method: str,
    checkpoint: float,
) -> ReconstructedState:
    """Restore one state solely from versioned revealed RGB measurements."""

    project = _resolved_root(root)
    row = _state_row(
        project, specimen_id=specimen_id, method=method, checkpoint=checkpoint
    )
    specimen = _specimen_row(project, specimen_id)
    if row["domain_id"] != specimen["dataset_id"] or int(row["native_count"]) != int(
        specimen["native_count"]
    ):
        raise AEIVisualAssetError("state and specimen metadata disagree")
    height = int(specimen["height"])
    width = int(specimen["width"])
    payload = _measurement_payload(project, row)
    revealed_rows = np.asarray(payload["revealed_rows"], dtype=np.int64)
    revealed_columns = np.asarray(payload["revealed_columns"], dtype=np.int64)
    rgb = np.column_stack(
        (
            payload["revealed_red"],
            payload["revealed_green"],
            payload["revealed_blue"],
        )
    ).astype(np.uint8, copy=False)
    expected_cost = int(row["exact_acquired_cost"])
    if (
        revealed_rows.shape != (expected_cost,)
        or revealed_columns.shape != (expected_cost,)
        or rgb.shape != (expected_cost, 3)
        or np.any(revealed_rows < 0)
        or np.any(revealed_rows >= height)
        or np.any(revealed_columns < 0)
        or np.any(revealed_columns >= width)
    ):
        raise AEIVisualAssetError("revealed measurement payload is invalid")

    grid = build_acquisition_grid(
        height, width, initial_budget=float(row["initial_budget"])
    )
    if grid.state_sha256 != row["grid_state_sha256"]:
        raise AEIVisualAssetError("registered acquisition grid changed")
    levels = tuple(int(value) for value in row["measurement_levels"])
    state = MeasurementState(grid_sha256=grid.state_sha256, levels=levels)
    mask = measurement_mask(grid, state)
    payload_mask = np.zeros((height, width), dtype=np.bool_)
    payload_mask[revealed_rows, revealed_columns] = True
    if not np.array_equal(mask, payload_mask) or int(mask.sum()) != expected_cost:
        raise AEIVisualAssetError("revealed measurements do not match the legal state")

    sampled_source = np.zeros((height, width, 3), dtype=np.uint8)
    sampled_source[revealed_rows, revealed_columns] = rgb
    reconstruction = reconstruct_measurement_state(
        sampled_source,
        grid,
        state,
        interpolation="bilinear",
        specimen_id=specimen_id,
        dataset_id=str(specimen["dataset_id"]),
    )
    expected_hash = str(row["current_reconstruction_sha256"])
    if reconstruction.output_sha256 != expected_hash:
        raise AEIVisualAssetError("restored reconstruction hash changed")
    return ReconstructedState(
        specimen_id=specimen_id,
        domain_id=str(row["domain_id"]),
        method=method,
        checkpoint=float(row["nominal_checkpoint"]),
        state_id=str(row["state_id"]),
        exact_acquired_cost=expected_cost,
        native_count=int(row["native_count"]),
        effective_budget=float(row["effective_budget"]),
        image=reconstruction.image,
        measurement_mask=_readonly(mask),
        measurement_levels=levels,
        acquired_cell_indices=tuple(
            int(value) for value in row["acquired_action_cell_indices"]
        ),
        row_boundaries=grid.row_boundaries,
        column_boundaries=grid.column_boundaries,
        output_sha256=reconstruction.output_sha256,
        expected_output_sha256=expected_hash,
    )


def _map_from_rows(
    rows: pl.DataFrame, value_column: str
) -> tuple[np.ndarray, np.ndarray]:
    if (
        rows.height != 64
        or rows.get_column("cell_index").n_unique() != 64
        or sorted(rows.get_column("cell_index").to_list()) != list(range(64))
    ):
        raise AEIVisualAssetError("priority rows do not cover the registered 8x8 grid")
    ordered = rows.sort("cell_index")
    values = ordered.get_column(value_column).to_numpy().astype(np.float64, copy=False)
    if values.shape != (64,) or not np.isfinite(values).all():
        raise AEIVisualAssetError("priority values are invalid")
    percentiles = (rankdata(values, method="average") - 1.0) / 63.0
    return _readonly(values.reshape(8, 8)), _readonly(percentiles.reshape(8, 8))


def load_priority_state(
    root: Path,
    *,
    specimen_id: str,
    method: str,
    checkpoint: float,
) -> PriorityState:
    """Load a state reconstruction and its 64-cell strict-OOF teacher values."""

    project = _resolved_root(root)
    reconstruction = load_reconstructed_state(
        project,
        specimen_id=specimen_id,
        method=method,
        checkpoint=checkpoint,
    )
    try:
        rows = (
            pl.scan_parquet(project / _ACTION_SCORES)
            .filter(
                (pl.col("state_id") == reconstruction.state_id)
                & (pl.col("specimen_id") == specimen_id)
                & (pl.col("mode") == "real")
            )
            .select("cell_index", "teacher_value")
            .collect()
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AEIVisualAssetError(
            "dynamic action-score evidence is unavailable"
        ) from error
    raw, percentiles = _map_from_rows(rows, "teacher_value")
    return PriorityState(
        state_id=reconstruction.state_id,
        reconstruction=reconstruction,
        raw_values=raw,
        percentiles=percentiles,
    )


def load_task_priority_maps(
    root: Path, *, specimen_id: str = REPRESENTATIVE_SPECIMEN
) -> TaskPriorityMaps:
    """Load paired CAI and reconstruction oracle maps for one initial state."""

    project = _resolved_root(root)
    reconstruction = load_reconstructed_state(
        project,
        specimen_id=specimen_id,
        method=REPRESENTATIVE_METHOD,
        checkpoint=INITIAL_CHECKPOINT,
    )
    try:
        rows = (
            pl.scan_parquet(project / _ORACLE_VALUES)
            .filter(
                (pl.col("specimen_id") == specimen_id)
                & (pl.col("step") == 0)
                & pl.col("method").is_in(["mechanical_oracle", "reconstruction_oracle"])
            )
            .select("dataset_id", "method", "cell_index", "primary_value")
            .collect()
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AEIVisualAssetError("oracle-value evidence is unavailable") from error
    if rows.get_column("dataset_id").unique().to_list() != [reconstruction.domain_id]:
        raise AEIVisualAssetError("oracle and reconstruction domains disagree")
    mechanical, mechanical_percentiles = _map_from_rows(
        rows.filter(pl.col("method") == "mechanical_oracle"), "primary_value"
    )
    image, image_percentiles = _map_from_rows(
        rows.filter(pl.col("method") == "reconstruction_oracle"), "primary_value"
    )
    difference = _readonly(mechanical_percentiles - image_percentiles)
    return TaskPriorityMaps(
        specimen_id=specimen_id,
        domain_id=reconstruction.domain_id,
        reconstruction=reconstruction,
        cell_indices=tuple(range(64)),
        mechanical_values=mechanical,
        reconstruction_values=image,
        mechanical_percentiles=mechanical_percentiles,
        reconstruction_percentiles=image_percentiles,
        percentile_difference=difference,
    )


def load_task_saliency_maps(
    root: Path, *, specimen_id: str = REPRESENTATIVE_SPECIMEN
) -> TaskSaliencyMaps:
    """Load paired CAI-task and registered appearance-saliency maps."""

    project = _resolved_root(root)
    reconstruction = load_reconstructed_state(
        project,
        specimen_id=specimen_id,
        method=REPRESENTATIVE_METHOD,
        checkpoint=INITIAL_CHECKPOINT,
    )
    try:
        rows = (
            pl.scan_parquet(project / _ORACLE_VALUES)
            .filter(
                (pl.col("specimen_id") == specimen_id)
                & (pl.col("step") == 0)
                & pl.col("method").is_in(["mechanical_oracle", "appearance_oracle"])
            )
            .select(
                "dataset_id",
                "method",
                "nominal_checkpoint",
                "cell_index",
                "from_level",
                "to_level",
                "primary_value",
            )
            .collect()
        )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AEIVisualAssetError("oracle-value evidence is unavailable") from error
    if rows.get_column("dataset_id").unique().to_list() != [reconstruction.domain_id]:
        raise AEIVisualAssetError("saliency oracle and legal-state domains disagree")
    if (
        rows.get_column("nominal_checkpoint").unique().to_list() != [0.0625]
        or rows.get_column("from_level").unique().to_list() != [0]
        or rows.get_column("to_level").unique().to_list() != [1]
    ):
        raise AEIVisualAssetError("registered initial oracle comparison changed")
    mechanical, mechanical_percentiles = _map_from_rows(
        rows.filter(pl.col("method") == "mechanical_oracle"), "primary_value"
    )
    saliency, saliency_percentiles = _map_from_rows(
        rows.filter(pl.col("method") == "appearance_oracle"), "primary_value"
    )
    difference = _readonly(mechanical_percentiles - saliency_percentiles)
    return TaskSaliencyMaps(
        specimen_id=specimen_id,
        domain_id=reconstruction.domain_id,
        reconstruction=reconstruction,
        cell_indices=tuple(range(64)),
        mechanical_values=mechanical,
        saliency_values=saliency,
        mechanical_percentiles=mechanical_percentiles,
        saliency_percentiles=saliency_percentiles,
        percentile_difference=difference,
    )


def gallery_specimen_roster(root: Path) -> tuple[GallerySpecimen, ...]:
    """Return one fixed, result-independent specimen from each sorted domain."""

    project = _resolved_root(root)
    rows = (
        _state_manifest(str(project))
        .select("domain_id", "specimen_id")
        .unique()
        .sort("domain_id", "specimen_id")
        .group_by("domain_id", maintain_order=True)
        .agg(pl.col("specimen_id").first())
    )
    if rows.height != 6 or rows.get_column("domain_id").n_unique() != 6:
        raise AEIVisualAssetError("gallery domain roster changed")
    return tuple(
        GallerySpecimen(domain_id=str(domain), specimen_id=str(specimen))
        for domain, specimen in rows.iter_rows()
    )


def load_gallery_states(root: Path) -> tuple[GalleryStatePair, ...]:
    """Load initial/later priority states for the deterministic six-domain roster."""

    project = _resolved_root(root)
    pairs = []
    for specimen in gallery_specimen_roster(project):
        initial = load_priority_state(
            project,
            specimen_id=specimen.specimen_id,
            method=REPRESENTATIVE_METHOD,
            checkpoint=INITIAL_CHECKPOINT,
        )
        later = load_priority_state(
            project,
            specimen_id=specimen.specimen_id,
            method=REPRESENTATIVE_METHOD,
            checkpoint=LATER_CHECKPOINT,
        )
        if (
            initial.reconstruction.domain_id != specimen.domain_id
            or later.reconstruction.domain_id != specimen.domain_id
        ):
            raise AEIVisualAssetError("gallery state domain changed")
        pairs.append(GalleryStatePair(specimen=specimen, initial=initial, later=later))
    return tuple(pairs)


__all__ = [
    "INITIAL_CHECKPOINT",
    "LATER_CHECKPOINT",
    "REPRESENTATIVE_METHOD",
    "REPRESENTATIVE_SPECIMEN",
    "AEIVisualAssetError",
    "GallerySpecimen",
    "GalleryStatePair",
    "PriorityState",
    "ReconstructedState",
    "TaskPriorityMaps",
    "TaskSaliencyMaps",
    "gallery_specimen_roster",
    "load_gallery_states",
    "load_priority_state",
    "load_reconstructed_state",
    "load_task_priority_maps",
    "load_task_saliency_maps",
]
