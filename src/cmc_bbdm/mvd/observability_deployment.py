"""Frozen M1 O2 ranking CAI diagnostic; this is not M2 deployment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.mva.a4_execution import (
    _load_uniform_embeddings,
    fit_outer_evaluation_models,
)
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.encoder_session import MVAEncoderSession
from cmc_bbdm.mva.interpolation import RefinementPatchCache
from cmc_bbdm.mva.measurement_state import initial_state
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


def run_observability_cai_diagnostic_outer_worker(
    config_path: str | Path,
    *,
    target_root: str | Path,
    research_root: str | Path,
    outer_domain: str,
    device: str,
) -> Path:
    """Apply the already-selected O2 scores once, without target-driven tuning."""

    target = Path(target_root).resolve(strict=True)
    research = Path(research_root).resolve(strict=True)
    config = load_mvd_config(config_path, project_root=target)
    if outer_domain not in config.domain_order:
        raise ValueError("M1 CAI diagnostic domain changed")
    compact, _base_config, authority = _validate_runtime(target, research, config)
    initial_budget = config.initial_budgets[outer_domain]
    bank = compact.candidate_banks[initial_budget]
    uniform_embeddings = _load_uniform_embeddings(
        research,
        authority,
        initial_budget=initial_budget,
        checkpoints=config.checkpoints,
    )
    models = fit_outer_evaluation_models(
        outer_domain=outer_domain,
        domain_order=config.domain_order,
        checkpoints=config.checkpoints,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        targets=authority.targets,
        metadata=authority.metadata13,
        full_embeddings=authority.full_embeddings,
        uniform_embeddings=uniform_embeddings,
        pca_dimensions=config.pca_dimensions,
        ridge_alpha=config.ridge_alpha,
        tie_tolerance=1.0e-12,
    )
    prediction_path = (
        target
        / config.output_work
        / "m1_domains"
        / outer_domain
        / "predictions.parquet"
    )
    predictions = (
        pl.scan_parquet(prediction_path)
        .filter(pl.col("method") == "o2_global_candidate")
        .select(
            "outer_domain",
            "specimen_id",
            "dataset_id",
            "cell_index",
            "predicted_value",
        )
        .collect()
        .sort(["specimen_id", "cell_index"])
    )
    target_indices = np.flatnonzero(
        np.asarray(authority.dataset_ids, dtype=object) == outer_domain
    )
    if (
        predictions.height != target_indices.size * 64
        or set(predictions["outer_domain"]) != {outer_domain}
    ):
        raise ValueError("M1 selected O2 prediction roster changed")
    by_specimen = {
        str(key[0] if isinstance(key, tuple) else key): group.sort("cell_index")
        for key, group in predictions.partition_by(
            "specimen_id", as_dict=True, include_key=False
        ).items()
    }
    pending: list[tuple[int, object, object]] = []
    for position, specimen_index in enumerate(target_indices, start=1):
        specimen_id = authority.specimen_ids[specimen_index]
        issued = by_specimen.get(specimen_id)
        if issued is None or tuple(issued["cell_index"]) != tuple(range(64)):
            raise ValueError("M1 target score vector changed")
        scores = issued["predicted_value"].to_numpy()
        ranking = score_initial_ranking(
            lambda values=scores: values, method="predicted_o2_one_shot"
        )
        image = authority.images[specimen_index]
        grid = build_acquisition_grid(
            image.shape[0], image.shape[1], initial_budget=initial_budget
        )
        plan = plan_frozen_ranking(
            grid,
            initial_state(grid),
            ranking=ranking,
            checkpoints=config.checkpoints,
        )
        trajectory = ControlTrajectory(
            method="predicted_o2_one_shot",
            seed=None,
            actions=plan.actions,
            snapshots=plan.snapshots,
        )
        snapshots = _materialize_control(
            image,
            grid,
            trajectory,
            specimen_id=specimen_id,
            dataset_id=outer_domain,
            patch_cache=RefinementPatchCache(image=image, grid=grid),
        )
        pending.extend((int(specimen_index), plan, snapshot) for snapshot in snapshots)
        print(
            f"[{outer_domain}] M1 CAI diagnostic {position}/{len(target_indices)}",
            flush=True,
        )
    encoder = MVAEncoderSession(_encoder(research, device))
    vectors = _encode_many(encoder, [entry[2].image for entry in pending])
    rows: list[dict[str, object]] = []
    for row_index, (specimen_index, plan, snapshot) in enumerate(pending):
        metadata = authority.metadata13[specimen_index : specimen_index + 1]
        vector = vectors[row_index : row_index + 1]
        target_value = float(authority.targets[specimen_index])
        p_a_prediction = float(models.p_a_model.predict(metadata, vector)[0])
        p_b_model = models.p_b_models[snapshot.checkpoint]
        p_b_prediction = float(p_b_model.predict(metadata, vector)[0])
        rows.append(
            {
                "specimen_id": authority.specimen_ids[specimen_index],
                "dataset_id": outer_domain,
                "outer_domain": outer_domain,
                "method": "predicted_o2_one_shot",
                "nominal_checkpoint": snapshot.checkpoint,
                "measured_count": snapshot.measured_count,
                "native_count": snapshot.native_count,
                "effective_budget": snapshot.effective_budget,
                "target": target_value,
                "p_a_prediction": p_a_prediction,
                "p_a_absolute_error": abs(target_value - p_a_prediction),
                "p_b_prediction": p_b_prediction,
                "p_b_absolute_error": abs(target_value - p_b_prediction),
                "p_a_predictor_state_sha256": models.p_a_model.state_sha256,
                "p_b_predictor_state_sha256": p_b_model.state_sha256,
                "candidate_bank_state_sha256": bank.state_sha256,
                "plan_state_sha256": plan.state_sha256,
                "ranking_state_sha256": plan.ranking_state_sha256,
            }
        )
    encoder.validate()
    output = target / config.output_work / "m1_cai_domains" / outer_domain
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "states.parquet"
    pl.DataFrame(rows).sort(
        ["specimen_id", "nominal_checkpoint"]
    ).write_parquet(state_path, compression="zstd")
    (output / "complete.json").write_text(
        json.dumps(
            {
                "outer_domain": outer_domain,
                "specimen_count": int(target_indices.size),
                "state_rows": len(rows),
                "states_sha256": _sha256_file(state_path),
                "evaluator_state_sha256": models.state_sha256,
                "candidate_bank_state_sha256": bank.state_sha256,
                "selection_use": "none_target_outcomes_diagnostic_only",
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


__all__ = ["run_observability_cai_diagnostic_outer_worker"]
