"""Outer-safe target evaluation for frozen initial MVD oracle rankings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl

from cmc_bbdm.mva.a4_execution import (
    _load_uniform_embeddings,
    fit_outer_evaluation_models,
)
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.authority import load_mva_authority
from cmc_bbdm.mva.config import load_mva_config
from cmc_bbdm.mva.crossfit import fit_outer_source_predictor
from cmc_bbdm.mva.encoder_session import MVAEncoderSession
from cmc_bbdm.mva.interpolation import RefinementPatchCache
from cmc_bbdm.mva.measurement_state import apply_action, budget_record, initial_state
from cmc_bbdm.mva.oracle_execution import _encode_many, _materialize_control
from cmc_bbdm.mva.oracle_trajectory import ControlTrajectory
from cmc_bbdm.mva.pipeline import _encoder

from .authority import load_compact_mvd_authority
from .config import MVDConfig, load_mvd_config
from .one_shot_oracle import plan_frozen_ranking, score_initial_ranking


class _Predictor(Protocol):
    outer_domain: str
    fit_domains: tuple[str, ...]
    state_sha256: str

    def predict(self, metadata: object, embeddings: object) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class InitialMechanicalValues:
    current_prediction: float
    candidate_predictions: np.ndarray
    mechanical_values: np.ndarray
    predictor_state_sha256: str


def _readonly(value: object, *, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError("initial Mechanical Value array changed")
    output = np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)
    output.setflags(write=False)
    return output


def initial_mechanical_values(
    predictor: _Predictor,
    *,
    outer_domain: str,
    metadata: object,
    target: float,
    initial_embedding: object,
    candidate_embeddings: object,
) -> InitialMechanicalValues:
    """Compute all 64 privileged target labels once with one frozen outer P-A."""

    meta = np.asarray(metadata, dtype=np.float64)
    initial = np.asarray(initial_embedding, dtype=np.float64)
    candidates = np.asarray(candidate_embeddings, dtype=np.float64)
    fit_domains = tuple(getattr(predictor, "fit_domains", ()))
    if (
        getattr(predictor, "outer_domain", None) != outer_domain
        or outer_domain in fit_domains
        or len(fit_domains) != 5
        or meta.ndim != 2
        or meta.shape[0] != 1
        or initial.shape != (512,)
        or candidates.shape != (64, 512)
        or not np.all(np.isfinite(meta))
        or not np.all(np.isfinite(initial))
        or not np.all(np.isfinite(candidates))
        or not np.isfinite(float(target))
    ):
        raise ValueError("outer-domain Mechanical Value information barrier changed")
    current = float(predictor.predict(meta, initial.reshape(1, 512))[0])
    predictions = np.asarray(
        predictor.predict(np.repeat(meta, 64, axis=0), candidates), dtype=np.float64
    )
    values = abs(float(target) - current) - np.abs(float(target) - predictions)
    return InitialMechanicalValues(
        current_prediction=current,
        candidate_predictions=_readonly(predictions, shape=(64,)),
        mechanical_values=_readonly(values, shape=(64,)),
        predictor_state_sha256=str(predictor.state_sha256),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_runtime(
    target_root: Path, research_root: Path, config: MVDConfig
) -> tuple[object, object, object]:
    compact = load_compact_mvd_authority(config, project_root=target_root)
    source_config_path = research_root / "paper_v3/configs/mva_a0_a3.yaml"
    if _sha256_file(source_config_path) != config.sources["mva_config"].sha256:
        raise ValueError("research MVA config differs from target authority")
    base_config = load_mva_config(source_config_path, project_root=research_root)
    authority = load_mva_authority(base_config, project_root=research_root)
    if (
        authority.state_sha256 != config.authority_state_sha256
        or authority.specimen_ids != compact.specimen_ids
        or authority.dataset_ids != compact.dataset_ids
        or authority.image_sha256 != compact.image_sha256
    ):
        raise ValueError("full runtime authority differs from compact MVD authority")
    return compact, base_config, authority


def _actual_action_rows(
    *,
    specimen_id: str,
    dataset_id: str,
    method: str,
    grid: object,
    plan: object,
    scores: np.ndarray,
) -> list[dict[str, object]]:
    state = initial_state(grid)
    rows: list[dict[str, object]] = []
    for step, (action, position, checkpoint) in enumerate(
        zip(
            plan.actions,
            plan.action_ranking_positions,
            plan.action_checkpoints,
            strict=True,
        )
    ):
        before = budget_record(grid, state)
        state = apply_action(grid, state, action)
        after = budget_record(grid, state)
        rows.append(
            {
                "specimen_id": specimen_id,
                "dataset_id": dataset_id,
                "method": method,
                "step": step,
                "nominal_checkpoint": checkpoint,
                "cell_index": action.cell_index,
                "from_level": action.from_level,
                "to_level": action.to_level,
                "ranking_position": position,
                "ranking_score": float(scores[action.cell_index]),
                "added_measurements_exact": after.measured_count
                - before.measured_count,
                "measured_count_after": after.measured_count,
                "effective_budget_after": after.effective_budget,
                "ranking_state_sha256": plan.ranking_state_sha256,
                "plan_state_sha256": plan.state_sha256,
            }
        )
    return rows


def run_m0_outer_worker(
    config_path: str | Path,
    *,
    target_root: str | Path,
    research_root: str | Path,
    outer_domain: str,
    device: str,
) -> Path:
    """Evaluate both specimen-specific frozen rankings for one outer domain."""

    target = Path(target_root).resolve(strict=True)
    research = Path(research_root).resolve(strict=True)
    config = load_mvd_config(config_path, project_root=target)
    if outer_domain not in config.domain_order:
        raise ValueError("outer domain is not registered")
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
    value_fit = fit_outer_source_predictor(
        method=f"MVD_M0_INITIAL_P_A_{outer_domain}",
        outer_domain=outer_domain,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        domain_order=config.domain_order,
        targets=authority.targets,
        metadata=authority.metadata13,
        embeddings=bank.initial_embeddings,
        pca_dimensions=config.pca_dimensions,
        ridge_alpha=config.ridge_alpha,
        tie_tolerance=1.0e-12,
    )
    target_indices = np.flatnonzero(
        np.asarray(authority.dataset_ids, dtype=object) == outer_domain
    )
    historical = pl.read_parquet(target / config.sources["a2_state_metrics"].path)
    historical_uniform = historical.filter(
        (pl.col("dataset_id") == outer_domain)
        & (pl.col("method") == "uniform")
        & pl.col("nominal_checkpoint").is_in(list(config.checkpoints))
    )
    reproduction_rows: list[dict[str, object]] = []
    for checkpoint in config.checkpoints:
        query_predictions = models.p_b_models[checkpoint].predict(
            authority.metadata13[target_indices],
            uniform_embeddings[checkpoint][target_indices],
        )
        by_specimen = {
            authority.specimen_ids[index]: float(prediction)
            for index, prediction in zip(
                target_indices, query_predictions, strict=True
            )
        }
        reference = historical_uniform.filter(
            pl.col("nominal_checkpoint") == checkpoint
        ).sort("specimen_id")
        reference_hashes = set(reference["p_b_predictor_state_sha256"])
        if (
            reference.height != target_indices.size
            or len(reference_hashes) != 1
            or set(reference["specimen_id"]) != set(by_specimen)
        ):
            raise ValueError("historical P-B reproduction roster changed")
        deltas = np.asarray(
            [
                abs(
                    by_specimen[str(row["specimen_id"])]
                    - float(row["p_b_prediction"])
                )
                for row in reference.iter_rows(named=True)
            ],
            dtype=np.float64,
        )
        if float(np.max(deltas)) > 1.0e-12:
            raise ValueError("historical P-B prediction reproduction failed")
        reproduction_rows.append(
            {
                "outer_domain": outer_domain,
                "nominal_checkpoint": checkpoint,
                "new_predictor_state_sha256": models.p_b_models[
                    checkpoint
                ].state_sha256,
                "reference_predictor_state_sha256": reference_hashes.pop(),
                "maximum_prediction_delta": float(np.max(deltas)),
                "mean_prediction_delta": float(
                    np.mean(deltas, dtype=np.float64)
                ),
                "tolerance": 1.0e-12,
                "target_specimen_count": int(target_indices.size),
                "fit_domains": "|".join(models.p_b_models[checkpoint].fit_domains),
            }
        )
    pending: list[tuple[int, str, object, object]] = []
    ranking_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    value_rows: list[dict[str, object]] = []
    for position, specimen_index in enumerate(target_indices, start=1):
        specimen_id = authority.specimen_ids[specimen_index]
        dataset_id = authority.dataset_ids[specimen_index]
        image = authority.images[specimen_index]
        grid = build_acquisition_grid(
            image.shape[0], image.shape[1], initial_budget=initial_budget
        )
        if (
            grid.state_sha256 != bank.grid_state_sha256[specimen_index]
            or tuple(image.shape[:2]) != bank.native_shapes[specimen_index]
        ):
            raise ValueError("target CandidateBank grid binding changed")
        labels = initial_mechanical_values(
            value_fit.model,
            outer_domain=outer_domain,
            metadata=authority.metadata13[specimen_index : specimen_index + 1],
            target=float(authority.targets[specimen_index]),
            initial_embedding=bank.initial_embeddings[specimen_index],
            candidate_embeddings=bank.embeddings[specimen_index],
        )
        score_sets = {
            "one_shot_reconstruction": bank.reconstruction_values[specimen_index],
            "one_shot_mechanical_oracle": labels.mechanical_values,
        }
        for method, issued_scores in score_sets.items():
            ranking = score_initial_ranking(lambda values=issued_scores: values, method=method)
            plan = plan_frozen_ranking(
                grid,
                initial_state(grid),
                ranking=ranking,
                checkpoints=config.checkpoints,
            )
            trajectory = ControlTrajectory(
                method=method,
                seed=None,
                actions=plan.actions,
                snapshots=plan.snapshots,
            )
            snapshots = _materialize_control(
                image,
                grid,
                trajectory,
                specimen_id=specimen_id,
                dataset_id=dataset_id,
                patch_cache=RefinementPatchCache(image=image, grid=grid),
            )
            pending.extend((specimen_index, method, plan, snapshot) for snapshot in snapshots)
            positions = {
                cell_index: rank_position
                for rank_position, cell_index in enumerate(ranking.cell_order)
            }
            for cell_index in range(64):
                ranking_rows.append(
                    {
                        "specimen_id": specimen_id,
                        "dataset_id": dataset_id,
                        "outer_domain": outer_domain,
                        "method": method,
                        "cell_index": cell_index,
                        "ranking_position": positions[cell_index],
                        "ranking_score": float(ranking.scores[cell_index]),
                        "initial_mechanical_value": float(labels.mechanical_values[cell_index]),
                        "initial_reconstruction_value": float(
                            bank.reconstruction_values[specimen_index, cell_index]
                        ),
                        "candidate_cost_from_initial": int(
                            bank.added_measurements[specimen_index, cell_index]
                        ),
                        "ranking_state_sha256": ranking.state_sha256,
                        "candidate_bank_state_sha256": bank.state_sha256,
                    }
                )
            action_rows.extend(
                _actual_action_rows(
                    specimen_id=specimen_id,
                    dataset_id=dataset_id,
                    method=method,
                    grid=grid,
                    plan=plan,
                    scores=ranking.scores,
                )
            )
        for cell_index in range(64):
            value_rows.append(
                {
                    "specimen_id": specimen_id,
                    "dataset_id": dataset_id,
                    "outer_domain": outer_domain,
                    "cell_index": cell_index,
                    "target": float(authority.targets[specimen_index]),
                    "current_prediction": labels.current_prediction,
                    "candidate_prediction": float(labels.candidate_predictions[cell_index]),
                    "initial_mechanical_value": float(labels.mechanical_values[cell_index]),
                    "initial_reconstruction_value": float(
                        bank.reconstruction_values[specimen_index, cell_index]
                    ),
                    "candidate_cost_from_initial": int(
                        bank.added_measurements[specimen_index, cell_index]
                    ),
                    "p_a_predictor_state_sha256": labels.predictor_state_sha256,
                    "candidate_bank_state_sha256": bank.state_sha256,
                }
            )
        print(
            f"[{outer_domain}] planned specimen {position}/{len(target_indices)}",
            flush=True,
        )
    encoder = MVAEncoderSession(_encoder(research, device))
    vectors = _encode_many(encoder, [entry[3].image for entry in pending])
    state_rows: list[dict[str, object]] = []
    for row_index, (specimen_index, method, plan, snapshot) in enumerate(pending):
        meta = authority.metadata13[specimen_index : specimen_index + 1]
        vector = vectors[row_index : row_index + 1]
        target_value = float(authority.targets[specimen_index])
        p_a_prediction = float(models.p_a_model.predict(meta, vector)[0])
        p_b_model = models.p_b_models[snapshot.checkpoint]
        p_b_prediction = float(p_b_model.predict(meta, vector)[0])
        state_rows.append(
            {
                "specimen_id": authority.specimen_ids[specimen_index],
                "dataset_id": authority.dataset_ids[specimen_index],
                "outer_domain": outer_domain,
                "method": method,
                "initial_budget": initial_budget,
                "nominal_checkpoint": snapshot.checkpoint,
                "measured_count": snapshot.measured_count,
                "native_count": snapshot.native_count,
                "effective_budget": snapshot.effective_budget,
                "cumulative_actions": snapshot.state.levels.count(1),
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
    output = target / config.output_work / "m0_domains" / outer_domain
    output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(state_rows, infer_schema_length=None).sort(
        ["specimen_id", "method", "nominal_checkpoint"]
    ).write_parquet(output / "states.parquet", compression="zstd")
    pl.DataFrame(ranking_rows, infer_schema_length=None).sort(
        ["specimen_id", "method", "cell_index"]
    ).write_parquet(output / "rankings.parquet", compression="zstd")
    pl.DataFrame(action_rows, infer_schema_length=None).sort(
        ["specimen_id", "method", "step"]
    ).write_parquet(output / "actions.parquet", compression="zstd")
    pl.DataFrame(value_rows, infer_schema_length=None).sort(
        ["specimen_id", "cell_index"]
    ).write_parquet(output / "initial_values.parquet", compression="zstd")
    pl.DataFrame(reproduction_rows, infer_schema_length=None).sort(
        ["outer_domain", "nominal_checkpoint"]
    ).write_csv(output / "evaluator_reproduction.csv")
    (output / "fit_audits.json").write_text(
        json.dumps(
            [asdict(value) for value in models.fit_audits],
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "value_fit_audits.json").write_text(
        json.dumps(
            [asdict(value) for value in value_fit.fit_audits],
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    files = tuple(
        sorted(
            path
            for path in output.iterdir()
            if path.is_file() and path.name != "complete.json"
        )
    )
    complete = {
        "outer_domain": outer_domain,
        "initial_budget": initial_budget,
        "specimen_count": int(target_indices.size),
        "state_rows": len(state_rows),
        "ranking_rows": len(ranking_rows),
        "action_rows": len(action_rows),
        "initial_value_rows": len(value_rows),
        "candidate_bank_state_sha256": bank.state_sha256,
        "evaluator_state_sha256": models.state_sha256,
        "value_predictor_state_sha256": value_fit.model.state_sha256,
        "files": {path.name: _sha256_file(path) for path in files},
    }
    (output / "complete.json").write_text(
        json.dumps(complete, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "InitialMechanicalValues",
    "initial_mechanical_values",
    "run_m0_outer_worker",
]
