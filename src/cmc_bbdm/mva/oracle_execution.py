"""GPU-backed domain workers for the formal MVA A2 oracle audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter

from .acquisition_grid import AcquisitionGrid, build_acquisition_grid
from .authority import MVAAuthority, load_mva_authority
from .cai_evaluator import CAIPredictor
from .config import load_mva_config
from .crossfit import fit_outer_source_predictor
from .encoder_session import MVAEncoderSession
from .interpolation import (
    RefinementPatchCache,
    reconstruct_measurement_state,
    refine_reconstruction,
)
from .measurement_state import (
    MeasurementState,
    RefinementAction,
    apply_action,
    budget_record,
    fitting_actions,
    initial_state,
    measurement_mask,
)
from .mechanical_value import mechanical_value
from .oracle_trajectory import ControlTrajectory, run_control_trajectory
from .pipeline import _encoder
from .reconstruction_value import normalized_rgb_mse

PRIMARY_CHECKPOINTS = (0.0625, 0.09375, 0.125, 0.1875, 0.25)
LOW_CHECKPOINTS = (0.03125,)
UNIFORM_CHECKPOINTS = (*LOW_CHECKPOINTS, *PRIMARY_CHECKPOINTS, 0.5)


@dataclass(frozen=True, slots=True)
class SnapshotImage:
    checkpoint: float
    state: MeasurementState
    image: np.ndarray
    measured_count: int
    native_count: int
    effective_budget: float


@dataclass(frozen=True, slots=True)
class OracleRun:
    values: tuple[dict[str, object], ...]
    actions: tuple[dict[str, object], ...]
    snapshots: tuple[SnapshotImage, ...]
    embeddings: tuple[np.ndarray | None, ...]


def _encode_many(encoder: MVAEncoderSession, images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("at least one image is required for encoding")
    batches = [
        np.asarray(encoder.encode(images[start : start + 256]), dtype=np.float64)
        for start in range(0, len(images), 256)
    ]
    return np.ascontiguousarray(np.vstack(batches), dtype=np.float64)


def _selected_budgets(root: Path) -> dict[str, float]:
    path = root / "results/mva/a1_simulator/summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["initial_survey_selection"]
    return {str(domain): float(budget) for domain, budget in values.items()}


def _token(value: float) -> str:
    return str(value).replace(".", "p")


def _uniform_cache(root: Path, initial_budget: float) -> Path:
    return root / "results/mva/.work" / f"uniform_bank_{_token(initial_budget)}.npz"


def _initial_embeddings(root: Path, initial_budget: float) -> np.ndarray:
    path = (
        root / "results/mva/.work" / f"initial_embeddings_{_token(initial_budget)}.npz"
    )
    with np.load(path, allow_pickle=False) as archive:
        output = np.asarray(archive["embeddings"], dtype=np.float64)
    if output.shape != (276, 512) or not np.all(np.isfinite(output)):
        raise RuntimeError("initial embedding cache is invalid")
    return output


def _materialize_control(
    image: np.ndarray,
    grid: AcquisitionGrid,
    trajectory: ControlTrajectory,
    *,
    specimen_id: str,
    dataset_id: str,
    patch_cache: RefinementPatchCache,
) -> tuple[SnapshotImage, ...]:
    state = initial_state(grid)
    current = reconstruct_measurement_state(
        image,
        grid,
        state,
        interpolation="bilinear",
        specimen_id=specimen_id,
        dataset_id=dataset_id,
    ).image
    current_mask = measurement_mask(grid, state)
    action_index = 0
    output: list[SnapshotImage] = []
    for snapshot in trajectory.snapshots:
        while action_index < snapshot.cumulative_actions:
            action = trajectory.actions[action_index]
            current = refine_reconstruction(
                image,
                grid,
                state,
                current,
                action,
                interpolation="bilinear",
                current_mask=current_mask,
                patch_cache=patch_cache,
            )
            state = apply_action(grid, state, action)
            cell = grid.cells[action.cell_index]
            rows = np.asarray(cell.rows[action.to_level], dtype=np.int64)
            columns = np.asarray(cell.columns[action.to_level], dtype=np.int64)
            current_mask[np.ix_(rows, columns)] = True
            action_index += 1
        if state != snapshot.state:
            raise RuntimeError("materialized control state changed")
        output.append(
            SnapshotImage(
                checkpoint=snapshot.nominal_checkpoint,
                state=state,
                image=current,
                measured_count=snapshot.measured_count,
                native_count=snapshot.native_count,
                effective_budget=snapshot.effective_budget,
            )
        )
    return tuple(output)


def _uniform_archive_valid(archive: object, *, specimen_count: int) -> bool:
    try:
        return all(
            np.asarray(archive[f"embedding_{_token(checkpoint)}"]).shape
            == (specimen_count, 512)
            for checkpoint in UNIFORM_CHECKPOINTS
        )
    except (KeyError, TypeError, ValueError):
        return False


def prepare_uniform_bank(
    config_path: str | Path,
    *,
    project_root: str | Path,
    initial_budget: float,
    device: str,
) -> Path:
    """Build the shared uniform-state embedding bank for one initial budget."""

    root = Path(project_root).resolve(strict=True)
    config = load_mva_config(config_path, project_root=root)
    if initial_budget not in config.initial_budgets:
        raise ValueError("initial budget is not registered")
    authority = load_mva_authority(config, project_root=root)
    output_path = _uniform_cache(root, initial_budget)
    if output_path.is_file():
        with np.load(output_path, allow_pickle=False) as archive:
            valid = _uniform_archive_valid(
                archive, specimen_count=authority.specimen_count
            )
        if valid:
            return output_path
    encoder = MVAEncoderSession(_encoder(root, device))
    embeddings = {
        checkpoint: np.empty((authority.specimen_count, 512), dtype=np.float64)
        for checkpoint in UNIFORM_CHECKPOINTS
    }
    measured = {
        checkpoint: np.empty(authority.specimen_count, dtype=np.int64)
        for checkpoint in UNIFORM_CHECKPOINTS
    }
    effective = {
        checkpoint: np.empty(authority.specimen_count, dtype=np.float64)
        for checkpoint in UNIFORM_CHECKPOINTS
    }
    for start in range(0, authority.specimen_count, 16):
        entries: list[tuple[int, float, SnapshotImage]] = []
        for index in range(start, min(start + 16, authority.specimen_count)):
            image = authority.images[index]
            grid = build_acquisition_grid(
                image.shape[0], image.shape[1], initial_budget=initial_budget
            )
            patch_cache = RefinementPatchCache(image=image, grid=grid)
            trajectory = run_control_trajectory(
                grid,
                initial_state(grid),
                checkpoints=UNIFORM_CHECKPOINTS,
                method="uniform",
            )
            snapshots = _materialize_control(
                image,
                grid,
                trajectory,
                specimen_id=authority.specimen_ids[index],
                dataset_id=authority.dataset_ids[index],
                patch_cache=patch_cache,
            )
            entries.extend(
                (index, snapshot.checkpoint, snapshot) for snapshot in snapshots
            )
        values = _encode_many(encoder, [entry[2].image for entry in entries])
        for row, (index, checkpoint, snapshot) in enumerate(entries):
            embeddings[checkpoint][index] = values[row]
            measured[checkpoint][index] = snapshot.measured_count
            effective[checkpoint][index] = snapshot.effective_budget
    payload: dict[str, np.ndarray] = {
        "specimen_ids": np.asarray(authority.specimen_ids),
        "authority_state": np.asarray([authority.state_sha256]),
        "initial_budget": np.asarray([initial_budget], dtype=np.float64),
    }
    for checkpoint in UNIFORM_CHECKPOINTS:
        token = _token(checkpoint)
        payload[f"embedding_{token}"] = embeddings[checkpoint]
        payload[f"measured_{token}"] = measured[checkpoint]
        payload[f"effective_{token}"] = effective[checkpoint]
    encoder.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def _load_uniform_bank(
    root: Path, authority: MVAAuthority, initial_budget: float
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[float, np.ndarray]]:
    path = _uniform_cache(root, initial_budget)
    with np.load(path, allow_pickle=False) as archive:
        if (
            tuple(str(value) for value in archive["specimen_ids"])
            != authority.specimen_ids
        ):
            raise RuntimeError("uniform bank specimen roster changed")
        embeddings = {
            checkpoint: np.asarray(
                archive[f"embedding_{_token(checkpoint)}"], dtype=np.float64
            )
            for checkpoint in UNIFORM_CHECKPOINTS
        }
        measured = {
            checkpoint: np.asarray(
                archive[f"measured_{_token(checkpoint)}"], dtype=np.int64
            )
            for checkpoint in UNIFORM_CHECKPOINTS
        }
        effective = {
            checkpoint: np.asarray(
                archive[f"effective_{_token(checkpoint)}"], dtype=np.float64
            )
            for checkpoint in UNIFORM_CHECKPOINTS
        }
    return embeddings, measured, effective


def _global_ssim(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    first = reference.astype(np.float64, copy=False)
    second = reconstruction.astype(np.float64, copy=False)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    values: list[float] = []
    for channel in range(3):
        x = first[:, :, channel]
        y = second[:, :, channel]
        mu_x = gaussian_filter(x, sigma=1.5, truncate=3.5, mode="reflect")
        mu_y = gaussian_filter(y, sigma=1.5, truncate=3.5, mode="reflect")
        sigma_x = (
            gaussian_filter(x * x, sigma=1.5, truncate=3.5, mode="reflect")
            - mu_x * mu_x
        )
        sigma_y = (
            gaussian_filter(y * y, sigma=1.5, truncate=3.5, mode="reflect")
            - mu_y * mu_y
        )
        sigma_xy = (
            gaussian_filter(x * y, sigma=1.5, truncate=3.5, mode="reflect")
            - mu_x * mu_y
        )
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        values.append(float(np.mean(numerator / denominator, dtype=np.float64)))
    return float(sum(values) / 3.0)


def _candidate_budget(
    grid: AcquisitionGrid,
    state: MeasurementState,
    current_mask: np.ndarray,
    action: RefinementAction,
) -> tuple[int, int, float]:
    current_count = int(np.count_nonzero(current_mask))
    cell = grid.cells[action.cell_index]
    rows = np.asarray(cell.rows[action.to_level], dtype=np.int64)
    columns = np.asarray(cell.columns[action.to_level], dtype=np.int64)
    added = int(np.count_nonzero(~current_mask[np.ix_(rows, columns)]))
    native = int(current_mask.size)
    measured = current_count + added
    return measured, native, float(measured / native)


def _oracle_trajectory(
    *,
    method: str,
    authority: MVAAuthority,
    specimen_index: int,
    grid: AcquisitionGrid,
    encoder: MVAEncoderSession,
    p_a_model: CAIPredictor,
    initial_embedding: np.ndarray,
    checkpoints: tuple[float, ...] = PRIMARY_CHECKPOINTS,
    patch_cache: RefinementPatchCache,
) -> OracleRun:
    image = authority.images[specimen_index]
    specimen_id = authority.specimen_ids[specimen_index]
    dataset_id = authority.dataset_ids[specimen_index]
    target = float(authority.targets[specimen_index])
    metadata = authority.metadata13[specimen_index : specimen_index + 1]
    state = initial_state(grid)
    current = reconstruct_measurement_state(
        image,
        grid,
        state,
        interpolation="bilinear",
        specimen_id=specimen_id,
        dataset_id=dataset_id,
    ).image
    current_embedding = np.asarray(initial_embedding, dtype=np.float64)
    current_prediction = float(
        p_a_model.predict(metadata, current_embedding.reshape(1, -1))[0]
    )
    border = np.zeros(image.shape[:2], dtype=np.bool_)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    border_median = np.median(image[border].astype(np.float64), axis=0)
    values: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    snapshots: list[SnapshotImage] = []
    snapshot_embeddings: list[np.ndarray | None] = []
    step = 0
    for checkpoint in checkpoints:
        while True:
            actions = fitting_actions(grid, state, checkpoint)
            if not actions:
                break
            budget_before = budget_record(grid, state).effective_budget
            current_mask = measurement_mask(grid, state)
            candidate_images = (
                []
                if method == "appearance_oracle"
                else [
                    refine_reconstruction(
                        image,
                        grid,
                        state,
                        current,
                        action,
                        interpolation="bilinear",
                        current_mask=current_mask,
                        patch_cache=patch_cache,
                    )
                    for action in actions
                ]
            )
            candidate_embeddings: np.ndarray | None = None
            candidate_predictions: np.ndarray | None = None
            primary: list[float] = []
            secondary: list[float | None] = []
            before_values: list[float | None] = []
            after_values: list[float | None] = []
            new_predictions: list[float | None] = []
            if method == "mechanical_oracle":
                candidate_embeddings = _encode_many(encoder, candidate_images)
                candidate_predictions = p_a_model.predict(
                    np.repeat(metadata, len(actions), axis=0), candidate_embeddings
                )
                for prediction in candidate_predictions:
                    score = mechanical_value(
                        target=target,
                        current_prediction=current_prediction,
                        candidate_prediction=float(prediction),
                    )
                    primary.append(score.absolute_error_reduction)
                    secondary.append(score.squared_error_reduction)
                    before_values.append(score.absolute_error_before)
                    after_values.append(score.absolute_error_after)
                    new_predictions.append(float(prediction))
            elif method == "reconstruction_oracle":
                before = normalized_rgb_mse(image, current)
                for candidate in candidate_images:
                    after = normalized_rgb_mse(image, candidate)
                    primary.append(before - after)
                    secondary.append(None)
                    before_values.append(before)
                    after_values.append(after)
                    new_predictions.append(None)
            elif method == "appearance_oracle":
                for action in actions:
                    cell = grid.cells[action.cell_index]
                    rows = np.asarray(cell.rows[action.to_level], dtype=np.int64)
                    columns = np.asarray(cell.columns[action.to_level], dtype=np.int64)
                    local_new = ~current_mask[np.ix_(rows, columns)]
                    revealed = image[np.ix_(rows, columns)][local_new].astype(
                        np.float64
                    )
                    score = float(
                        np.mean(np.abs(revealed - border_median), dtype=np.float64)
                        / 255.0
                    )
                    primary.append(score)
                    secondary.append(None)
                    before_values.append(None)
                    after_values.append(None)
                    new_predictions.append(None)
            else:
                raise ValueError("oracle method is not registered")
            selected_index = max(
                range(len(actions)),
                key=lambda index: (
                    primary[index],
                    -actions[index].cell_index,
                    -actions[index].to_level,
                ),
            )
            for index, action in enumerate(actions):
                measured, native, effective = _candidate_budget(
                    grid, state, current_mask, action
                )
                values.append(
                    {
                        "specimen_id": specimen_id,
                        "dataset_id": dataset_id,
                        "method": method,
                        "step": step,
                        "nominal_checkpoint": checkpoint,
                        "cell_index": action.cell_index,
                        "from_level": action.from_level,
                        "to_level": action.to_level,
                        "measured_count": measured,
                        "native_count": native,
                        "effective_budget": effective,
                        "budget_before": budget_before,
                        "candidate": action.cell_index,
                        "primary_value": primary[index],
                        "value": primary[index],
                        "budget_after": effective,
                        "current_prediction": (
                            current_prediction
                            if method == "mechanical_oracle"
                            else None
                        ),
                        "new_prediction": new_predictions[index],
                        "secondary_value": secondary[index],
                        "error_before": before_values[index],
                        "error_after": after_values[index],
                        "current_error": before_values[index],
                        "new_error": after_values[index],
                        "selected": index == selected_index,
                        "p_a_predictor_state_sha256": p_a_model.state_sha256,
                    }
                )
            action = actions[selected_index]
            selected_image = (
                refine_reconstruction(
                    image,
                    grid,
                    state,
                    current,
                    action,
                    interpolation="bilinear",
                    current_mask=current_mask,
                    patch_cache=patch_cache,
                )
                if method == "appearance_oracle"
                else candidate_images[selected_index]
            )
            state = apply_action(grid, state, action)
            current = selected_image
            if method == "mechanical_oracle":
                assert (
                    candidate_embeddings is not None
                    and candidate_predictions is not None
                )
                current_embedding = candidate_embeddings[selected_index]
                current_prediction = float(candidate_predictions[selected_index])
            measured, native, effective = _candidate_budget(
                grid,
                MeasurementState(
                    grid_sha256=grid.state_sha256,
                    levels=tuple(
                        level - (1 if index == action.cell_index else 0)
                        for index, level in enumerate(state.levels)
                    ),
                ),
                current_mask,
                action,
            )
            selected_rows.append(
                {
                    "record_type": "action",
                    "specimen_id": specimen_id,
                    "dataset_id": dataset_id,
                    "method": method,
                    "seed": None,
                    "step": step,
                    "nominal_checkpoint": checkpoint,
                    "cell_index": action.cell_index,
                    "from_level": action.from_level,
                    "to_level": action.to_level,
                    "measured_count": measured,
                    "native_count": native,
                    "effective_budget": effective,
                }
            )
            step += 1
        budget = budget_record(grid, state)
        snapshots.append(
            SnapshotImage(
                checkpoint=checkpoint,
                state=state,
                image=current,
                measured_count=budget.measured_count,
                native_count=budget.native_count,
                effective_budget=budget.effective_budget,
            )
        )
        snapshot_embeddings.append(
            current_embedding.copy() if method == "mechanical_oracle" else None
        )
    return OracleRun(
        values=tuple(values),
        actions=tuple(selected_rows),
        snapshots=tuple(snapshots),
        embeddings=tuple(snapshot_embeddings),
    )


def _state_row(
    *,
    authority: MVAAuthority,
    specimen_index: int,
    method: str,
    seed: int | None,
    snapshot: SnapshotImage,
    embedding: np.ndarray,
    p_a_model: CAIPredictor,
    p_b_model: CAIPredictor,
    image_metrics: bool = True,
) -> dict[str, object]:
    metadata = authority.metadata13[specimen_index : specimen_index + 1]
    target = float(authority.targets[specimen_index])
    vector = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
    p_a = float(p_a_model.predict(metadata, vector)[0])
    p_b = float(p_b_model.predict(metadata, vector)[0])
    return {
        "specimen_id": authority.specimen_ids[specimen_index],
        "dataset_id": authority.dataset_ids[specimen_index],
        "method": method,
        "seed": seed,
        "nominal_checkpoint": snapshot.checkpoint,
        "measured_count": snapshot.measured_count,
        "native_count": snapshot.native_count,
        "effective_budget": snapshot.effective_budget,
        "target": target,
        "p_a_prediction": p_a,
        "p_a_absolute_error": abs(target - p_a),
        "p_b_prediction": p_b,
        "p_b_absolute_error": abs(target - p_b),
        "normalized_rgb_mse": (
            normalized_rgb_mse(authority.images[specimen_index], snapshot.image)
            if image_metrics
            else None
        ),
        "ssim": (
            _global_ssim(authority.images[specimen_index], snapshot.image)
            if image_metrics
            else None
        ),
        "p_a_predictor_state_sha256": p_a_model.state_sha256,
        "p_b_predictor_state_sha256": p_b_model.state_sha256,
    }


def _control_action_rows(
    authority: MVAAuthority,
    specimen_index: int,
    trajectory: ControlTrajectory,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous = 0
    for snapshot in trajectory.snapshots:
        for step in range(previous, snapshot.cumulative_actions):
            action = trajectory.actions[step]
            rows.append(
                {
                    "record_type": "action",
                    "specimen_id": authority.specimen_ids[specimen_index],
                    "dataset_id": authority.dataset_ids[specimen_index],
                    "method": trajectory.method,
                    "seed": trajectory.seed,
                    "step": step,
                    "nominal_checkpoint": snapshot.nominal_checkpoint,
                    "cell_index": action.cell_index,
                    "from_level": action.from_level,
                    "to_level": action.to_level,
                    "measured_count": None,
                    "native_count": authority.images[specimen_index].shape[0]
                    * authority.images[specimen_index].shape[1],
                    "effective_budget": None,
                }
            )
        previous = snapshot.cumulative_actions
    return rows


def run_domain_worker(
    config_path: str | Path,
    *,
    project_root: str | Path,
    outer_domain: str,
    device: str,
    max_specimens: int | None = None,
    random_seed_count: int = 100,
) -> Path:
    """Run all A2 methods for one isolated target domain and write Parquet shards."""

    root = Path(project_root).resolve(strict=True)
    config = load_mva_config(config_path, project_root=root)
    authority = load_mva_authority(config, project_root=root)
    if outer_domain not in config.domain_order:
        raise ValueError("outer domain is not registered")
    selected_budget = _selected_budgets(root)[outer_domain]
    uniform_embeddings, _uniform_measured, _uniform_effective = _load_uniform_bank(
        root, authority, selected_budget
    )
    p_a_fit = fit_outer_source_predictor(
        method="MVA_P_A",
        outer_domain=outer_domain,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        domain_order=config.domain_order,
        targets=authority.targets,
        metadata=authority.metadata13,
        embeddings=authority.full_embeddings,
        pca_dimensions=config.pca_dimensions,
        ridge_alpha=config.ridge_alpha,
        tie_tolerance=1.0e-12,
    )
    p_b_models = {
        checkpoint: fit_outer_source_predictor(
            method=f"MVA_P_B_{checkpoint}",
            outer_domain=outer_domain,
            specimen_ids=authority.specimen_ids,
            dataset_ids=authority.dataset_ids,
            domain_order=config.domain_order,
            targets=authority.targets,
            metadata=authority.metadata13,
            embeddings=uniform_embeddings[checkpoint],
            pca_dimensions=config.pca_dimensions,
            ridge_alpha=config.ridge_alpha,
            tie_tolerance=1.0e-12,
        ).model
        for checkpoint in PRIMARY_CHECKPOINTS
    }
    encoder = MVAEncoderSession(_encoder(root, device))
    initial_embeddings = _initial_embeddings(root, selected_budget)
    targets = [
        index
        for index, domain in enumerate(authority.dataset_ids)
        if domain == outer_domain
    ]
    if max_specimens is not None:
        targets = targets[:max_specimens]
    state_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []
    random_seeds = config.random_seeds[:random_seed_count]
    for position, specimen_index in enumerate(targets, start=1):
        image = authority.images[specimen_index]
        grid = build_acquisition_grid(
            image.shape[0], image.shape[1], initial_budget=selected_budget
        )
        patch_cache = RefinementPatchCache(image=image, grid=grid)
        uniform_trajectory = run_control_trajectory(
            grid,
            initial_state(grid),
            checkpoints=PRIMARY_CHECKPOINTS,
            method="uniform",
        )
        uniform_snapshots = _materialize_control(
            image,
            grid,
            uniform_trajectory,
            specimen_id=authority.specimen_ids[specimen_index],
            dataset_id=authority.dataset_ids[specimen_index],
            patch_cache=patch_cache,
        )
        trajectory_rows.extend(
            _control_action_rows(authority, specimen_index, uniform_trajectory)
        )
        for snapshot in uniform_snapshots:
            state_rows.append(
                _state_row(
                    authority=authority,
                    specimen_index=specimen_index,
                    method="uniform",
                    seed=None,
                    snapshot=snapshot,
                    embedding=uniform_embeddings[snapshot.checkpoint][specimen_index],
                    p_a_model=p_a_fit.model,
                    p_b_model=p_b_models[snapshot.checkpoint],
                )
            )
        random_entries: list[tuple[int, SnapshotImage]] = []
        for seed in random_seeds:
            trajectory = run_control_trajectory(
                grid,
                initial_state(grid),
                checkpoints=PRIMARY_CHECKPOINTS,
                method="random",
                seed=seed,
            )
            snapshots = _materialize_control(
                image,
                grid,
                trajectory,
                specimen_id=authority.specimen_ids[specimen_index],
                dataset_id=authority.dataset_ids[specimen_index],
                patch_cache=patch_cache,
            )
            trajectory_rows.extend(
                _control_action_rows(authority, specimen_index, trajectory)
            )
            random_entries.extend((seed, snapshot) for snapshot in snapshots)
        random_vectors = _encode_many(
            encoder, [entry[1].image for entry in random_entries]
        )
        for row, (seed, snapshot) in enumerate(random_entries):
            state_rows.append(
                _state_row(
                    authority=authority,
                    specimen_index=specimen_index,
                    method="random",
                    seed=seed,
                    snapshot=snapshot,
                    embedding=random_vectors[row],
                    p_a_model=p_a_fit.model,
                    p_b_model=p_b_models[snapshot.checkpoint],
                    image_metrics=False,
                )
            )
        for method in (
            "appearance_oracle",
            "reconstruction_oracle",
            "mechanical_oracle",
        ):
            oracle = _oracle_trajectory(
                method=method,
                authority=authority,
                specimen_index=specimen_index,
                grid=grid,
                encoder=encoder,
                p_a_model=p_a_fit.model,
                initial_embedding=initial_embeddings[specimen_index],
                patch_cache=patch_cache,
            )
            value_rows.extend(oracle.values)
            trajectory_rows.extend(oracle.actions)
            missing = [
                index
                for index, embedding in enumerate(oracle.embeddings)
                if embedding is None
            ]
            computed: dict[int, np.ndarray] = {}
            if missing:
                vectors = _encode_many(
                    encoder, [oracle.snapshots[index].image for index in missing]
                )
                computed = {index: vectors[row] for row, index in enumerate(missing)}
            for index, snapshot in enumerate(oracle.snapshots):
                embedding = oracle.embeddings[index]
                vector = computed[index] if embedding is None else embedding
                state_rows.append(
                    _state_row(
                        authority=authority,
                        specimen_index=specimen_index,
                        method=method,
                        seed=None,
                        snapshot=snapshot,
                        embedding=vector,
                        p_a_model=p_a_fit.model,
                        p_b_model=p_b_models[snapshot.checkpoint],
                    )
                )
        print(
            f"[{outer_domain}] formal specimen {position}/{len(targets)} complete",
            flush=True,
        )
    encoder.validate()
    output = root / "results/mva/.work/a2_domains" / outer_domain
    output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(state_rows, infer_schema_length=None).write_parquet(
        output / "states.parquet", compression="zstd"
    )
    pl.DataFrame(trajectory_rows, infer_schema_length=None).write_parquet(
        output / "trajectories.parquet", compression="zstd"
    )
    pl.DataFrame(value_rows, infer_schema_length=None).write_parquet(
        output / "oracle_values.parquet", compression="zstd"
    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "outer_domain": outer_domain,
                "specimen_count": len(targets),
                "random_seed_count": len(random_seeds),
                "state_rows": len(state_rows),
                "trajectory_rows": len(trajectory_rows),
                "value_rows": len(value_rows),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def run_low_checkpoint_worker(
    config_path: str | Path,
    *,
    project_root: str | Path,
    outer_domain: str,
    device: str,
    max_specimens: int | None = None,
    random_seed_count: int = 100,
) -> Path:
    """Evaluate the report-only 3.125% checkpoint for one held-out domain."""

    root = Path(project_root).resolve(strict=True)
    config = load_mva_config(config_path, project_root=root)
    authority = load_mva_authority(config, project_root=root)
    if outer_domain not in config.domain_order:
        raise ValueError("outer domain is not registered")
    selected_budget = _selected_budgets(root)[outer_domain]
    uniform_embeddings, _, _ = _load_uniform_bank(root, authority, selected_budget)
    checkpoint = LOW_CHECKPOINTS[0]
    p_a_model = fit_outer_source_predictor(
        method="MVA_P_A",
        outer_domain=outer_domain,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        domain_order=config.domain_order,
        targets=authority.targets,
        metadata=authority.metadata13,
        embeddings=authority.full_embeddings,
        pca_dimensions=config.pca_dimensions,
        ridge_alpha=config.ridge_alpha,
        tie_tolerance=1.0e-12,
    ).model
    p_b_model = fit_outer_source_predictor(
        method=f"MVA_P_B_{checkpoint}",
        outer_domain=outer_domain,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        domain_order=config.domain_order,
        targets=authority.targets,
        metadata=authority.metadata13,
        embeddings=uniform_embeddings[checkpoint],
        pca_dimensions=config.pca_dimensions,
        ridge_alpha=config.ridge_alpha,
        tie_tolerance=1.0e-12,
    ).model
    encoder = MVAEncoderSession(_encoder(root, device))
    initial_embeddings = _initial_embeddings(root, selected_budget)
    targets = [
        index
        for index, domain in enumerate(authority.dataset_ids)
        if domain == outer_domain
    ]
    if max_specimens is not None:
        targets = targets[:max_specimens]
    state_rows: list[dict[str, object]] = []
    random_seeds = config.random_seeds[:random_seed_count]
    for position, specimen_index in enumerate(targets, start=1):
        image = authority.images[specimen_index]
        grid = build_acquisition_grid(
            image.shape[0], image.shape[1], initial_budget=selected_budget
        )
        patch_cache = RefinementPatchCache(image=image, grid=grid)
        start = initial_state(grid)
        uniform = run_control_trajectory(
            grid, start, checkpoints=LOW_CHECKPOINTS, method="uniform"
        )
        uniform_snapshot = _materialize_control(
            image,
            grid,
            uniform,
            specimen_id=authority.specimen_ids[specimen_index],
            dataset_id=authority.dataset_ids[specimen_index],
            patch_cache=patch_cache,
        )[0]
        state_rows.append(
            _state_row(
                authority=authority,
                specimen_index=specimen_index,
                method="uniform",
                seed=None,
                snapshot=uniform_snapshot,
                embedding=uniform_embeddings[checkpoint][specimen_index],
                p_a_model=p_a_model,
                p_b_model=p_b_model,
            )
        )
        random_snapshots: list[tuple[int, SnapshotImage]] = []
        for seed in random_seeds:
            trajectory = run_control_trajectory(
                grid,
                start,
                checkpoints=LOW_CHECKPOINTS,
                method="random",
                seed=seed,
            )
            snapshot = _materialize_control(
                image,
                grid,
                trajectory,
                specimen_id=authority.specimen_ids[specimen_index],
                dataset_id=authority.dataset_ids[specimen_index],
                patch_cache=patch_cache,
            )[0]
            random_snapshots.append((seed, snapshot))
        random_vectors = _encode_many(
            encoder, [snapshot.image for _, snapshot in random_snapshots]
        )
        for row, (seed, snapshot) in enumerate(random_snapshots):
            state_rows.append(
                _state_row(
                    authority=authority,
                    specimen_index=specimen_index,
                    method="random",
                    seed=seed,
                    snapshot=snapshot,
                    embedding=random_vectors[row],
                    p_a_model=p_a_model,
                    p_b_model=p_b_model,
                    image_metrics=False,
                )
            )
        for method in (
            "appearance_oracle",
            "reconstruction_oracle",
            "mechanical_oracle",
        ):
            oracle = _oracle_trajectory(
                method=method,
                authority=authority,
                specimen_index=specimen_index,
                grid=grid,
                encoder=encoder,
                p_a_model=p_a_model,
                initial_embedding=initial_embeddings[specimen_index],
                checkpoints=LOW_CHECKPOINTS,
                patch_cache=patch_cache,
            )
            embedding = oracle.embeddings[0]
            vector = (
                _encode_many(encoder, [oracle.snapshots[0].image])[0]
                if embedding is None
                else embedding
            )
            state_rows.append(
                _state_row(
                    authority=authority,
                    specimen_index=specimen_index,
                    method=method,
                    seed=None,
                    snapshot=oracle.snapshots[0],
                    embedding=vector,
                    p_a_model=p_a_model,
                    p_b_model=p_b_model,
                )
            )
        print(
            f"[{outer_domain}] low-budget specimen {position}/{len(targets)} complete",
            flush=True,
        )
    encoder.validate()
    output = root / "results/mva/.work/a2_low_domains" / outer_domain
    output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(state_rows, infer_schema_length=None).write_parquet(
        output / "states.parquet", compression="zstd"
    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "outer_domain": outer_domain,
                "checkpoint": checkpoint,
                "specimen_count": len(targets),
                "random_seed_count": len(random_seeds),
                "state_rows": len(state_rows),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "LOW_CHECKPOINTS",
    "PRIMARY_CHECKPOINTS",
    "UNIFORM_CHECKPOINTS",
    "prepare_uniform_bank",
    "run_domain_worker",
    "run_low_checkpoint_worker",
]
