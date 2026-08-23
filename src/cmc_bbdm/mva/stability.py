"""Initial-state mechanical-oracle ranking stability diagnostics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from .acquisition_grid import build_acquisition_grid
from .authority import load_mva_authority
from .cai_evaluator import CAIPredictor, fit_sensitivity_cai_predictor
from .config import load_mva_config
from .crossfit import fit_outer_source_predictor
from .encoder_session import MVAEncoderSession
from .interpolation import (
    RefinementPatchCache,
    reconstruct_measurement_state,
    refine_reconstruction,
)
from .measurement_state import fitting_actions, initial_state, measurement_mask
from .oracle_execution import (
    PRIMARY_CHECKPOINTS,
    _encode_many,
    _encoder,
    _initial_embeddings,
    _selected_budgets,
)
from .pipeline import _write_csv


@dataclass(frozen=True, slots=True)
class RankingSimilarity:
    top1_agreement: bool
    top10_overlap: float
    spearman: float
    rbo_p0_9: float


def _rbo(first: list[int], second: list[int], persistence: float = 0.9) -> float:
    first_seen: set[int] = set()
    second_seen: set[int] = set()
    overlap = 0.0
    score = 0.0
    for depth, (left, right) in enumerate(zip(first, second, strict=True), start=1):
        first_seen.add(left)
        second_seen.add(right)
        overlap = len(first_seen & second_seen) / depth
        score += (1.0 - persistence) * overlap * persistence ** (depth - 1)
    return float(score + overlap * persistence ** len(first))


def ranking_similarity(
    reference: Mapping[int, float], candidate: Mapping[int, float]
) -> RankingSimilarity:
    """Compare two aligned candidate-cell rankings with deterministic tie breaks."""

    if set(reference) != set(candidate) or len(reference) < 2:
        raise ValueError("ranking score maps must have aligned candidate keys")
    keys = sorted(reference)
    first_values = np.asarray([reference[key] for key in keys], dtype=np.float64)
    second_values = np.asarray([candidate[key] for key in keys], dtype=np.float64)
    if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
        raise ValueError("ranking scores must be finite")
    first = sorted(keys, key=lambda key: (-float(reference[key]), key))
    second = sorted(keys, key=lambda key: (-float(candidate[key]), key))
    top = max(1, math.ceil(0.1 * len(keys)))
    correlation = float(spearmanr(first_values, second_values).statistic)
    if not math.isfinite(correlation):
        correlation = 1.0 if np.array_equal(first_values, second_values) else 0.0
    return RankingSimilarity(
        top1_agreement=first[0] == second[0],
        top10_overlap=float(len(set(first[:top]) & set(second[:top])) / top),
        spearman=correlation,
        rbo_p0_9=_rbo(first, second),
    )


def _ridge_sensitivity_models(
    *, config, authority, outer_domain: str
) -> dict[float, CAIPredictor]:
    reference = fit_outer_source_predictor(
        method="MVA_STABILITY_RIDGE10",
        outer_domain=outer_domain,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        domain_order=config.domain_order,
        targets=authority.targets,
        metadata=authority.metadata13,
        embeddings=authority.full_embeddings,
        pca_dimensions=config.pca_dimensions,
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    )
    domains = np.asarray(authority.dataset_ids, dtype=object)
    fit_indices = np.flatnonzero(domains != outer_domain)
    models = {10.0: reference.model}
    for alpha in (1.0, 100.0):
        models[alpha] = fit_sensitivity_cai_predictor(
            method=f"MVA_STABILITY_RIDGE{alpha:g}",
            outer_domain=outer_domain,
            specimen_ids=authority.specimen_ids,
            dataset_ids=authority.dataset_ids,
            targets=authority.targets,
            metadata=authority.metadata13,
            embeddings=authority.full_embeddings,
            dimension=reference.selected_pca_dimension,
            fit_indices=fit_indices,
            ridge_alpha=alpha,
        )
    if any(outer_domain in model.fit_domains for model in models.values()):
        raise RuntimeError("stability predictor accessed the outer domain")
    return models


def _mechanical_scores(
    *,
    model: CAIPredictor,
    metadata: np.ndarray,
    target: float,
    cell_indices: tuple[int, ...],
    embeddings: np.ndarray,
) -> dict[int, float]:
    current = float(model.predict(metadata, embeddings[:1])[0])
    predictions = model.predict(
        np.repeat(metadata, len(cell_indices), axis=0), embeddings[1:]
    )
    before = abs(target - current)
    return {
        cell_index: float(before - abs(target - float(prediction)))
        for cell_index, prediction in zip(cell_indices, predictions, strict=True)
    }


def _stability_embedding_set(
    encoder: MVAEncoderSession,
    current: np.ndarray,
    candidates: list[np.ndarray],
    *,
    issued_current: np.ndarray | None = None,
) -> np.ndarray:
    candidate_embeddings = _encode_many(encoder, candidates)
    if issued_current is None:
        current_embedding = _encode_many(encoder, [current])[0]
    else:
        current_embedding = np.asarray(issued_current, dtype=np.float64)
        if current_embedding.shape != (512,) or not np.all(
            np.isfinite(current_embedding)
        ):
            raise ValueError("issued current embedding is invalid")
    return np.ascontiguousarray(
        np.vstack((current_embedding, candidate_embeddings)), dtype=np.float64
    )


def run_stability_domain(
    config_path: str | Path,
    *,
    project_root: str | Path,
    outer_domain: str,
    device: str,
    max_specimens: int | None = None,
) -> Path:
    """Publish initial ranking sensitivity for one strictly held-out domain."""

    root = Path(project_root).resolve(strict=True)
    config = load_mva_config(config_path, project_root=root)
    authority = load_mva_authority(config, project_root=root)
    if outer_domain not in config.domain_order:
        raise ValueError("outer domain is not registered")
    selected_budget = _selected_budgets(root)[outer_domain]
    models = _ridge_sensitivity_models(
        config=config, authority=authority, outer_domain=outer_domain
    )
    encoder = MVAEncoderSession(_encoder(root, device))
    initial_embeddings = _initial_embeddings(root, selected_budget)
    targets = [
        index
        for index, domain in enumerate(authority.dataset_ids)
        if domain == outer_domain
    ]
    if max_specimens is not None:
        targets = targets[:max_specimens]
    formal_values = pl.read_parquet(
        root / "results/mva/.work/a2_domains" / outer_domain / "oracle_values.parquet"
    ).filter((pl.col("method") == "mechanical_oracle") & (pl.col("step") == 0))
    rows: list[dict[str, object]] = []
    maximum_primary_delta = 0.0
    for position, specimen_index in enumerate(targets, start=1):
        image = authority.images[specimen_index]
        specimen_id = authority.specimen_ids[specimen_index]
        grid = build_acquisition_grid(
            image.shape[0], image.shape[1], initial_budget=selected_budget
        )
        state = initial_state(grid)
        patch_cache = RefinementPatchCache(image=image, grid=grid)
        actions = fitting_actions(grid, state, PRIMARY_CHECKPOINTS[0])
        cell_indices = tuple(action.cell_index for action in actions)
        mask = measurement_mask(grid, state)
        embedding_sets: dict[str, np.ndarray] = {}
        for interpolation in ("bilinear", "nearest", "bicubic"):
            current = reconstruct_measurement_state(
                image,
                grid,
                state,
                interpolation=interpolation,
                specimen_id=specimen_id,
                dataset_id=outer_domain,
            ).image
            candidates = [
                refine_reconstruction(
                    image,
                    grid,
                    state,
                    current,
                    action,
                    interpolation=interpolation,
                    current_mask=mask,
                    patch_cache=patch_cache,
                )
                for action in actions
            ]
            embedding_sets[interpolation] = _stability_embedding_set(
                encoder,
                current,
                candidates,
                issued_current=(
                    initial_embeddings[specimen_index]
                    if interpolation == "bilinear"
                    else None
                ),
            )
        metadata = authority.metadata13[specimen_index : specimen_index + 1]
        target = float(authority.targets[specimen_index])
        reference = _mechanical_scores(
            model=models[10.0],
            metadata=metadata,
            target=target,
            cell_indices=cell_indices,
            embeddings=embedding_sets["bilinear"],
        )
        issued = {
            int(row["cell_index"]): float(row["primary_value"])
            for row in formal_values.filter(
                pl.col("specimen_id") == specimen_id
            ).iter_rows(named=True)
        }
        if set(issued) != set(reference):
            raise RuntimeError("issued initial mechanical-value roster changed")
        maximum_primary_delta = max(
            maximum_primary_delta,
            max(abs(reference[key] - issued[key]) for key in reference),
        )
        variants = {
            "nearest_ridge10": _mechanical_scores(
                model=models[10.0],
                metadata=metadata,
                target=target,
                cell_indices=cell_indices,
                embeddings=embedding_sets["nearest"],
            ),
            "bicubic_ridge10": _mechanical_scores(
                model=models[10.0],
                metadata=metadata,
                target=target,
                cell_indices=cell_indices,
                embeddings=embedding_sets["bicubic"],
            ),
            "bilinear_ridge1": _mechanical_scores(
                model=models[1.0],
                metadata=metadata,
                target=target,
                cell_indices=cell_indices,
                embeddings=embedding_sets["bilinear"],
            ),
            "bilinear_ridge100": _mechanical_scores(
                model=models[100.0],
                metadata=metadata,
                target=target,
                cell_indices=cell_indices,
                embeddings=embedding_sets["bilinear"],
            ),
        }
        for variant, scores in variants.items():
            similarity = ranking_similarity(reference, scores)
            rows.append(
                {
                    "specimen_id": specimen_id,
                    "dataset_id": outer_domain,
                    "reference": "bilinear_ridge10",
                    "variant": variant,
                    "candidate_count": len(cell_indices),
                    "top1_agreement": similarity.top1_agreement,
                    "top10_overlap": similarity.top10_overlap,
                    "spearman": similarity.spearman,
                    "rbo_p0_9": similarity.rbo_p0_9,
                }
            )
        print(
            f"[{outer_domain}] stability specimen {position}/{len(targets)} complete",
            flush=True,
        )
    if maximum_primary_delta > 1.0e-12:
        raise RuntimeError(
            "stability reference does not reproduce formal oracle values"
        )
    encoder.validate()
    output = root / "results/mva/.work/a2_stability_domains" / outer_domain
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "stability.csv",
        (
            "specimen_id",
            "dataset_id",
            "reference",
            "variant",
            "candidate_count",
            "top1_agreement",
            "top10_overlap",
            "spearman",
            "rbo_p0_9",
        ),
        rows,
    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "outer_domain": outer_domain,
                "specimen_count": len(targets),
                "row_count": len(rows),
                "maximum_primary_delta": maximum_primary_delta,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


__all__ = ["RankingSimilarity", "ranking_similarity", "run_stability_domain"]
