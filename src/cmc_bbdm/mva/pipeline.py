"""Ordered execution for the MVA A0-A3 experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmc_bbdm.cpb_sparse_scan.sampling import reconstruct_sparse_rgb
from cmc_bbdm.cpb_v3.embeddings import FrozenResNet18Encoder, encode_resnet18

from .acquisition_grid import build_acquisition_grid
from .authority import (
    FullBaselineReproduction,
    MVAAuthority,
    load_mva_authority,
    reproduce_full_baseline,
)
from .config import MVAConfig, load_mva_config
from .crossfit import fit_outer_source_predictor, select_initial_survey
from .interpolation import reconstruct_measurement_state
from .measurement_state import MeasurementState, initial_state, measurement_mask


@dataclass(frozen=True, slots=True)
class A0A1Run:
    selected_initial_budget: Mapping[str, float]
    initial_embeddings: Mapping[float, np.ndarray]
    baseline: FullBaselineReproduction


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _encoder(root: Path, device: str) -> FrozenResNet18Encoder:
    value = encode_resnet18(
        weight_path=Path("paper_v3/assets/resnet18-f37072fd.pth"),
        project_root=root,
        device=device,
        batch_size=32,
    )
    if not isinstance(value, FrozenResNet18Encoder):
        raise TypeError("frozen encoder construction failed")
    return value


def _embedding_cache_path(root: Path, budget: float) -> Path:
    token = str(budget).replace(".", "p")
    return root / "results/mva/.work" / f"initial_embeddings_{token}.npz"


def _load_embedding_cache(
    path: Path, authority: MVAAuthority, budget: float
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            embeddings = np.asarray(archive["embeddings"], dtype=np.float64)
            specimen_ids = tuple(str(value) for value in archive["specimen_ids"])
            saved_budget = float(archive["budget"][0])
            authority_state = str(archive["authority_state"][0])
    except (OSError, KeyError, ValueError):
        return None
    if (
        embeddings.shape != (authority.specimen_count, 512)
        or not np.all(np.isfinite(embeddings))
        or specimen_ids != authority.specimen_ids
        or saved_budget != budget
        or authority_state != authority.state_sha256
    ):
        return None
    return np.ascontiguousarray(embeddings)


def _save_embedding_cache(
    path: Path,
    authority: MVAAuthority,
    budget: float,
    embeddings: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float64),
        specimen_ids=np.asarray(authority.specimen_ids),
        budget=np.asarray([budget], dtype=np.float64),
        authority_state=np.asarray([authority.state_sha256]),
    )


def _initial_embeddings(
    root: Path,
    authority: MVAAuthority,
    budget: float,
    encoder: FrozenResNet18Encoder,
) -> np.ndarray:
    cache = _embedding_cache_path(root, budget)
    cached = _load_embedding_cache(cache, authority, budget)
    if cached is not None:
        return cached
    rows: list[np.ndarray] = []
    for start in range(0, authority.specimen_count, 32):
        images: list[np.ndarray] = []
        for index in range(start, min(start + 32, authority.specimen_count)):
            source = authority.images[index]
            grid = build_acquisition_grid(
                source.shape[0], source.shape[1], initial_budget=budget
            )
            result = reconstruct_measurement_state(
                source,
                grid,
                initial_state(grid),
                interpolation="bilinear",
                specimen_id=authority.specimen_ids[index],
                dataset_id=authority.dataset_ids[index],
            )
            images.append(result.image)
        rows.append(np.asarray(encoder.encode(images), dtype=np.float64))
    embeddings = np.ascontiguousarray(np.vstack(rows), dtype=np.float64)
    _save_embedding_cache(cache, authority, budget, embeddings)
    return embeddings


def _publish_a0(
    root: Path,
    config: MVAConfig,
    authority: MVAAuthority,
    baseline: FullBaselineReproduction,
) -> None:
    output = root / config.output_dir / "a0_acquisition_audit"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "cohort.csv",
        (
            "specimen_id",
            "dataset_id",
            "native_height",
            "native_width",
            "image_sha256",
            "target",
        ),
        (
            {
                "specimen_id": specimen_id,
                "dataset_id": dataset_id,
                "native_height": image.shape[0],
                "native_width": image.shape[1],
                "image_sha256": image_sha,
                "target": format(float(target), ".17g"),
            }
            for specimen_id, dataset_id, image, image_sha, target in zip(
                authority.specimen_ids,
                authority.dataset_ids,
                authority.images,
                authority.image_sha256,
                authority.targets,
                strict=True,
            )
        ),
    )
    shape_counts: dict[tuple[int, int], int] = {}
    for image in authority.images:
        shape_counts[image.shape[:2]] = shape_counts.get(image.shape[:2], 0) + 1
    _write_csv(
        output / "native_geometry.csv",
        ("native_height", "native_width", "specimen_count", "field_mm", "claim_scope"),
        (
            {
                "native_height": shape[0],
                "native_width": shape[1],
                "specimen_count": count,
                "field_mm": "75x75",
                "claim_scope": "normalized_raster_simulation",
            }
            for shape, count in sorted(shape_counts.items())
        ),
    )
    _write_json(
        output / "baseline.json",
        {
            "authority_state_sha256": authority.state_sha256,
            "equal_domain_mae": baseline.equal_domain_mae,
            "maximum_prediction_delta": baseline.maximum_prediction_delta,
            "registered_mae": config.full_mae,
            "selected_pca_dimensions": list(baseline.selected_pca_dimensions),
            "specimen_count": authority.specimen_count,
            "status": "PASS",
            "tolerance": config.baseline_tolerance,
        },
    )


def _p5_rows(authority: MVAAuthority) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, image in enumerate(authority.images):
        grid = build_acquisition_grid(
            image.shape[0], image.shape[1], initial_budget=0.015625
        )
        state = MeasurementState(grid_sha256=grid.state_sha256, levels=(1,) * 64)
        mva = reconstruct_measurement_state(
            image,
            grid,
            state,
            interpolation="bilinear",
            specimen_id=authority.specimen_ids[index],
            dataset_id=authority.dataset_ids[index],
        )
        p5, p5_record = reconstruct_sparse_rgb(
            image,
            specimen_id=authority.specimen_ids[index],
            dataset_id=authority.dataset_ids[index],
            density=0.25,
            interpolation="bilinear",
        )
        mask = measurement_mask(grid, state)
        equivalent = bool(
            np.array_equal(mva.image, p5)
            and mva.output_sha256 == p5_record.output_sha256
            and int(np.count_nonzero(mask))
            == p5_record.row_count * p5_record.column_count
        )
        if not equivalent:
            raise RuntimeError("MVA level-1 state does not reproduce P5")
        rows.append(
            {
                "specimen_id": authority.specimen_ids[index],
                "dataset_id": authority.dataset_ids[index],
                "measured_count": mva.measured_count,
                "native_count": mva.native_count,
                "effective_budget": format(mva.effective_budget, ".17g"),
                "output_sha256": mva.output_sha256,
                "p5_equivalent": "true",
            }
        )
    return rows


def run_a0_a1(
    config_path: str | Path,
    *,
    project_root: str | Path,
    device: str = "cuda:0",
) -> A0A1Run:
    """Execute authority audit, initial-survey pilot, and P5 simulator checks."""

    root = Path(project_root).resolve(strict=True)
    config = load_mva_config(config_path, project_root=root)
    authority = load_mva_authority(config, project_root=root)
    baseline = reproduce_full_baseline(config, authority)
    _publish_a0(root, config, authority, baseline)
    encoder = _encoder(root, device)
    embeddings = {
        budget: _initial_embeddings(root, authority, budget, encoder)
        for budget in config.initial_budgets
    }
    output = root / config.output_dir / "a1_simulator"
    output.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    selected: dict[str, float] = {}
    for outer_domain in config.domain_order:
        fits = {
            "FULL": fit_outer_source_predictor(
                method="MVA_PILOT_FULL",
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
        }
        for budget in config.initial_budgets:
            fits[str(budget)] = fit_outer_source_predictor(
                method=f"MVA_PILOT_{budget}",
                outer_domain=outer_domain,
                specimen_ids=authority.specimen_ids,
                dataset_ids=authority.dataset_ids,
                domain_order=config.domain_order,
                targets=authority.targets,
                metadata=authority.metadata13,
                embeddings=embeddings[budget],
                pca_dimensions=config.pca_dimensions,
                ridge_alpha=config.ridge_alpha,
                tie_tolerance=1.0e-12,
            )
        full_domain = dict(fits["FULL"].selected_inner_domain_mae)
        candidate_domain = {
            budget: dict(fits[str(budget)].selected_inner_domain_mae)
            for budget in config.initial_budgets
        }
        decision = select_initial_survey(
            outer_domain=outer_domain,
            domain_order=config.domain_order,
            full_domain_mae=full_domain,
            candidate_domain_mae=candidate_domain,
        )
        selected[outer_domain] = decision.selected_budget
        selection_rows.append(
            {
                "outer_domain": outer_domain,
                "selected_budget": format(decision.selected_budget, ".17g"),
                "status": decision.status,
                "source_full_mae": format(decision.source_full_mae, ".17g"),
            }
        )
        for label, fitted in fits.items():
            score_rows.append(
                {
                    "outer_domain": outer_domain,
                    "condition": label,
                    "selected_pca_dimension": fitted.selected_pca_dimension,
                    "inner_equal_domain_mae": format(
                        dict(fitted.inner_dimension_mae)[fitted.selected_pca_dimension],
                        ".17g",
                    ),
                    "predictor_state_sha256": fitted.model.state_sha256,
                }
            )
    _write_csv(
        output / "initial_survey_selection.csv",
        ("outer_domain", "selected_budget", "status", "source_full_mae"),
        selection_rows,
    )
    _write_csv(
        output / "initial_survey_scores.csv",
        (
            "outer_domain",
            "condition",
            "selected_pca_dimension",
            "inner_equal_domain_mae",
            "predictor_state_sha256",
        ),
        score_rows,
    )
    _write_csv(
        output / "p5_equivalence.csv",
        (
            "specimen_id",
            "dataset_id",
            "measured_count",
            "native_count",
            "effective_budget",
            "output_sha256",
            "p5_equivalent",
        ),
        _p5_rows(authority),
    )
    _write_json(
        output / "summary.json",
        {
            "initial_survey_selection": selected,
            "p5_equivalent_specimens": authority.specimen_count,
            "status": "PASS",
        },
    )
    return A0A1Run(
        selected_initial_budget=selected,
        initial_embeddings=embeddings,
        baseline=baseline,
    )


__all__ = ["A0A1Run", "run_a0_a1"]
