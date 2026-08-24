"""Strict source-only additivity audit for frozen initial Mechanical Values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

from cmc_bbdm.mva.a4_source_labels import _predict_candidates
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.crossfit import fit_outer_source_predictor
from cmc_bbdm.mva.encoder_session import MVAEncoderSession
from cmc_bbdm.mva.interpolation import RefinementPatchCache
from cmc_bbdm.mva.measurement_state import initial_state
from cmc_bbdm.mva.oracle import uniform_cell_order
from cmc_bbdm.mva.oracle_execution import _encode_many, _materialize_control
from cmc_bbdm.mva.oracle_trajectory import ControlTrajectory
from cmc_bbdm.mva.pipeline import _encoder

from .config import load_mvd_config
from .evaluation import _validate_runtime
from .one_shot_oracle import plan_frozen_ranking, score_initial_ranking


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_interactions(rows: pl.DataFrame) -> pl.DataFrame:
    """Summarize additive versus joint gain without issuing a gate."""

    required = {
        "method",
        "nominal_checkpoint",
        "additive_gain",
        "joint_gain",
    }
    if type(rows) is not pl.DataFrame or not required <= set(rows.columns):
        raise ValueError("interaction rows changed")
    output: list[dict[str, object]] = []
    for key, group in rows.group_by(
        ["method", "nominal_checkpoint"], maintain_order=True
    ):
        method, checkpoint = key
        additive = group["additive_gain"].to_numpy()
        joint = group["joint_gain"].to_numpy()
        if additive.size < 3 or not np.all(np.isfinite(additive)) or not np.all(
            np.isfinite(joint)
        ):
            raise ValueError("interaction group is incomplete")
        pearson = float(pearsonr(additive, joint).statistic)
        spearman = float(spearmanr(additive, joint).statistic)
        difference = additive - joint
        output.append(
            {
                "method": str(method),
                "nominal_checkpoint": float(checkpoint),
                "specimen_fold_count": int(additive.size),
                "pearson": 0.0 if not np.isfinite(pearson) else pearson,
                "spearman": 0.0 if not np.isfinite(spearman) else spearman,
                "signed_bias_additive_minus_joint": float(
                    np.mean(difference, dtype=np.float64)
                ),
                "mean_absolute_error": float(
                    np.mean(np.abs(difference), dtype=np.float64)
                ),
            }
        )
    return pl.DataFrame(output).sort(["method", "nominal_checkpoint"])


def _score_sets(
    mechanical: np.ndarray,
    reconstruction: np.ndarray,
    *,
    random_seed: int,
) -> dict[str, np.ndarray]:
    uniform = np.empty(64, dtype=np.float64)
    for position, cell in enumerate(uniform_cell_order()):
        uniform[cell] = 64.0 - position
    return {
        "one_shot_mechanical": mechanical,
        "one_shot_reconstruction": reconstruction,
        "uniform": uniform,
        "random": np.random.Generator(np.random.PCG64(random_seed)).random(64),
    }


def run_interaction_outer_worker(
    config_path: str | Path,
    *,
    target_root: str | Path,
    research_root: str | Path,
    outer_domain: str,
    device: str,
) -> Path:
    """Evaluate one held-target fold's five strict-OOF source-domain audits."""

    target_root_path = Path(target_root).resolve(strict=True)
    research_root_path = Path(research_root).resolve(strict=True)
    config = load_mvd_config(config_path, project_root=target_root_path)
    if outer_domain not in config.domain_order:
        raise ValueError("interaction outer domain changed")
    compact, _base_config, authority = _validate_runtime(
        target_root_path, research_root_path, config
    )
    initial_budget = config.initial_budgets[outer_domain]
    bank = compact.candidate_banks[initial_budget]
    source_indices = np.flatnonzero(
        np.asarray(authority.dataset_ids, dtype=object) != outer_domain
    )
    source_domains = tuple(
        domain for domain in config.domain_order if domain != outer_domain
    )
    source_ids = tuple(authority.specimen_ids[index] for index in source_indices)
    source_dataset_ids = tuple(
        authority.dataset_ids[index] for index in source_indices
    )
    mechanical_source = compact.source_values.filter(
        (pl.col("outer_domain") == outer_domain)
        & (pl.col("method") == "global_mechanical_mask")
    ).sort(["specimen_id", "cell_index"])
    if mechanical_source.height != len(source_indices) * 64:
        raise ValueError("interaction source value roster changed")
    by_specimen = {
        str(key[0] if isinstance(key, tuple) else key): group.sort("cell_index")
        for key, group in mechanical_source.partition_by(
            "specimen_id", as_dict=True, include_key=False
        ).items()
    }
    checkpoints = tuple(value for value in config.checkpoints if value >= 0.0625)
    encoder = MVAEncoderSession(_encoder(research_root_path, device))
    rows: list[dict[str, object]] = []
    reproduction: list[dict[str, object]] = []
    for query_domain in source_domains:
        fitted = fit_outer_source_predictor(
            method=f"MVA_A4_P_A_{outer_domain}_{query_domain}",
            outer_domain=query_domain,
            specimen_ids=source_ids,
            dataset_ids=source_dataset_ids,
            domain_order=source_domains,
            targets=authority.targets[source_indices],
            metadata=authority.metadata13[source_indices],
            embeddings=bank.initial_embeddings[source_indices],
            pca_dimensions=config.pca_dimensions,
            ridge_alpha=config.ridge_alpha,
            tie_tolerance=1.0e-12,
        )
        query_indices = np.asarray(
            [
                index
                for index in source_indices
                if authority.dataset_ids[index] == query_domain
            ],
            dtype=np.int64,
        )
        initial_predictions = fitted.model.predict(
            authority.metadata13[query_indices], bank.initial_embeddings[query_indices]
        )
        candidate_predictions = _predict_candidates(
            fitted.model,
            authority.metadata13[query_indices],
            bank.embeddings[query_indices],
        )
        initial_deltas: list[float] = []
        candidate_deltas: list[float] = []
        pending: list[tuple[int, str, int | None, object, object, float]] = []
        for local_index, specimen_index in enumerate(query_indices):
            specimen_id = authority.specimen_ids[specimen_index]
            source_rows = by_specimen[specimen_id]
            values = source_rows["primary_value"].to_numpy()
            reference_current = float(source_rows["current_prediction"][0])
            reference_candidates = source_rows["candidate_prediction"].to_numpy()
            initial_deltas.append(
                abs(float(initial_predictions[local_index]) - reference_current)
            )
            candidate_deltas.extend(
                np.abs(candidate_predictions[local_index] - reference_candidates)
            )
            image = authority.images[specimen_index]
            grid = build_acquisition_grid(
                image.shape[0], image.shape[1], initial_budget=initial_budget
            )
            for method, scores in _score_sets(
                values,
                bank.reconstruction_values[specimen_index],
                random_seed=config.random_seed_start,
            ).items():
                ranking = score_initial_ranking(
                    lambda issued=scores: issued, method=method
                )
                plan = plan_frozen_ranking(
                    grid,
                    initial_state(grid),
                    ranking=ranking,
                    checkpoints=checkpoints,
                )
                trajectory = ControlTrajectory(
                    method=method,
                    seed=(config.random_seed_start if method == "random" else None),
                    actions=plan.actions,
                    snapshots=plan.snapshots,
                )
                snapshots = _materialize_control(
                    image,
                    grid,
                    trajectory,
                    specimen_id=specimen_id,
                    dataset_id=query_domain,
                    patch_cache=RefinementPatchCache(image=image, grid=grid),
                )
                for snapshot in snapshots:
                    count = plan.snapshots[
                        plan.checkpoints.index(snapshot.checkpoint)
                    ].cumulative_actions
                    cells = tuple(
                        action.cell_index for action in plan.actions[:count]
                    )
                    additive = float(np.sum(values[list(cells)], dtype=np.float64))
                    pending.append(
                        (
                            int(specimen_index),
                            method,
                            config.random_seed_start if method == "random" else None,
                            plan,
                            snapshot,
                            additive,
                        )
                    )
        maximum_initial_delta = float(np.max(initial_deltas))
        maximum_candidate_delta = float(np.max(candidate_deltas))
        if max(maximum_initial_delta, maximum_candidate_delta) > 1.0e-12:
            raise ValueError("interaction source predictor reproduction failed")
        reproduction.append(
            {
                "outer_domain": outer_domain,
                "query_source_domain": query_domain,
                "fit_domains": "|".join(fitted.model.fit_domains),
                "predictor_state_sha256": fitted.model.state_sha256,
                "maximum_initial_prediction_delta": maximum_initial_delta,
                "maximum_candidate_prediction_delta": maximum_candidate_delta,
                "tolerance": 1.0e-12,
            }
        )
        vectors = _encode_many(encoder, [entry[4].image for entry in pending])
        for row_index, (
            specimen_index,
            method,
            random_seed,
            plan,
            snapshot,
            additive,
        ) in enumerate(pending):
            target = float(authority.targets[specimen_index])
            specimen_id = authority.specimen_ids[specimen_index]
            source_rows = by_specimen[specimen_id]
            current_prediction = float(source_rows["current_prediction"][0])
            initial_error = abs(target - current_prediction)
            final_prediction = float(
                fitted.model.predict(
                    authority.metadata13[specimen_index : specimen_index + 1],
                    vectors[row_index : row_index + 1],
                )[0]
            )
            final_error = abs(target - final_prediction)
            count = plan.snapshots[
                plan.checkpoints.index(snapshot.checkpoint)
            ].cumulative_actions
            cells = tuple(action.cell_index for action in plan.actions[:count])
            rows.append(
                {
                    "outer_domain": outer_domain,
                    "query_source_domain": query_domain,
                    "specimen_id": specimen_id,
                    "method": method,
                    "random_seed": random_seed,
                    "nominal_checkpoint": snapshot.checkpoint,
                    "selected_cell_count": len(cells),
                    "selected_cells": "|".join(str(value) for value in cells),
                    "additive_gain": additive,
                    "joint_gain": initial_error - final_error,
                    "initial_absolute_error": initial_error,
                    "joint_absolute_error": final_error,
                    "joint_prediction": final_prediction,
                    "predictor_state_sha256": fitted.model.state_sha256,
                    "candidate_bank_state_sha256": bank.state_sha256,
                    "ranking_state_sha256": plan.ranking_state_sha256,
                    "plan_state_sha256": plan.state_sha256,
                }
            )
        print(
            f"[{outer_domain}] interaction query={query_domain} "
            f"specimens={len(query_indices)} states={len(pending)}",
            flush=True,
        )
    encoder.validate()
    output = (
        target_root_path
        / config.output_work
        / "interaction_domains"
        / outer_domain
    )
    output.mkdir(parents=True, exist_ok=True)
    table = pl.DataFrame(rows, infer_schema_length=None).sort(
        ["query_source_domain", "specimen_id", "method", "nominal_checkpoint"]
    )
    table.write_parquet(output / "interaction_rows.parquet", compression="zstd")
    pl.DataFrame(reproduction).sort("query_source_domain").write_csv(
        output / "predictor_reproduction.csv"
    )
    files = tuple(path for path in output.iterdir() if path.name != "complete.json")
    complete = {
        "outer_domain": outer_domain,
        "row_count": table.height,
        "source_specimen_count": len(source_indices),
        "methods": sorted(set(table["method"])),
        "checkpoints": checkpoints,
        "files": {path.name: _sha256_file(path) for path in sorted(files)},
    }
    (output / "complete.json").write_text(
        json.dumps(complete, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = ["run_interaction_outer_worker", "summarize_interactions"]
