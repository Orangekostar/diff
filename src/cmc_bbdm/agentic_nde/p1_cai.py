"""Exact-cost P1 CAI evaluation and specimen-first aggregation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cmc_bbdm.mavis.task_specificity import normalized_auebc
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
from cmc_bbdm.mvd.config import load_mvd_config
from cmc_bbdm.mvd.evaluation import _validate_runtime
from cmc_bbdm.mvd.one_shot_oracle import (
    plan_frozen_ranking,
    score_initial_ranking,
)

from .p1 import P1Config
from .p1_execution import P1OuterData, P1OuterScoreEvaluation
from .visual_observability import P1Decision, decide_p1


class P1CAIError(ValueError):
    """Raised when P1 acquisition evidence violates the frozen contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class P1AggregateEvaluation:
    per_specimen_metrics: pl.DataFrame
    domain_metrics: pl.DataFrame
    bootstrap: pl.DataFrame
    control_results: pl.DataFrame
    decision: P1Decision
    state_sha256: str


@dataclass(frozen=True, slots=True)
class P1OuterCAIEvaluation:
    outer_domain: str
    acquisition_curves: pl.DataFrame
    evaluator_state_sha256: str
    state_sha256: str


def run_p1_cai_outer(
    config: P1Config,
    data: P1OuterData,
    score_evaluation: P1OuterScoreEvaluation,
    *,
    research_root: str | Path,
    device: str,
    notify: Callable[[str], None] | None = None,
) -> P1OuterCAIEvaluation:
    """Evaluate specimen-specific frozen rankings with the registered P-B head."""

    if (
        type(config) is not P1Config
        or type(data) is not P1OuterData
        or type(score_evaluation) is not P1OuterScoreEvaluation
        or data.outer_domain != score_evaluation.outer_domain
        or device != "cuda:0"
    ):
        raise P1CAIError("P1 outer CAI request changed")
    target_root = config.project_root
    try:
        research = Path(research_root).resolve(strict=True)
    except OSError as error:
        raise P1CAIError("P1 CAI research root is unavailable") from error
    mvd = load_mvd_config(
        target_root / config.sources["mvd_config"].path,
        project_root=target_root,
    )
    compact, _base_config, authority = _validate_runtime(
        target_root, research, mvd
    )
    outer_domain = data.outer_domain
    if (
        outer_domain not in mvd.domain_order
        or data.initial_budget != mvd.initial_budgets[outer_domain]
        or compact.specimen_ids != authority.specimen_ids
        or compact.dataset_ids != authority.dataset_ids
    ):
        raise P1CAIError("P1 CAI runtime authority changed")
    acquisition = config.raw.get("acquisition")
    if not isinstance(acquisition, Mapping):
        raise P1CAIError("P1 acquisition config changed")
    checkpoints = tuple(float(value) for value in acquisition.get("checkpoints", ()))
    if checkpoints != (0.0625, 0.09375, 0.125, 0.1875, 0.25):
        raise P1CAIError("P1 acquisition checkpoints changed")
    uniform_embeddings = _load_uniform_embeddings(
        research,
        authority,
        initial_budget=data.initial_budget,
        checkpoints=mvd.checkpoints,
    )
    models = fit_outer_evaluation_models(
        outer_domain=outer_domain,
        domain_order=mvd.domain_order,
        checkpoints=mvd.checkpoints,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        targets=authority.targets,
        metadata=authority.metadata13,
        full_embeddings=authority.full_embeddings,
        uniform_embeddings=uniform_embeddings,
        pca_dimensions=mvd.pca_dimensions,
        ridge_alpha=mvd.ridge_alpha,
        tie_tolerance=1.0e-12,
    )
    target_indices = [
        index
        for index, domain in enumerate(authority.dataset_ids)
        if domain == outer_domain
    ]
    expected_ids = tuple(authority.specimen_ids[index] for index in target_indices)
    if (
        expected_ids != data.correct.inference.specimen_ids
        or tuple(sorted(expected_ids))
        != tuple(
            sorted(
                str(value)
                for value in score_evaluation.per_state_scores[
                    "specimen_id"
                ].unique()
            )
        )
        or len(target_indices) != len(data.target_native_shapes)
        or set(score_evaluation.score_matrices)
        != set(score_evaluation.per_specimen_metrics["method"])
    ):
        raise P1CAIError("P1 CAI target score roster changed")
    bank = compact.candidate_banks[data.initial_budget]
    encoder = MVAEncoderSession(_encoder(research, device))
    rows: list[dict[str, object]] = []
    methods = tuple(sorted(score_evaluation.score_matrices))
    for local_index, authority_index in enumerate(target_indices):
        specimen_id = authority.specimen_ids[authority_index]
        image = authority.images[authority_index]
        if tuple(image.shape[:2]) != data.target_native_shapes[local_index]:
            raise P1CAIError("P1 CAI native target shape changed")
        grid = build_acquisition_grid(
            image.shape[0], image.shape[1], initial_budget=data.initial_budget
        )
        if (
            grid.state_sha256 != data.target_grid_state_sha256[local_index]
            or grid.state_sha256 != bank.grid_state_sha256[authority_index]
        ):
            raise P1CAIError("P1 CAI target grid changed")
        patch_cache = RefinementPatchCache(image=image, grid=grid)
        pending: list[tuple[str, object, object]] = []
        for method in methods:
            scores = score_evaluation.score_matrices[method][local_index]
            ranking = score_initial_ranking(
                lambda values=scores: values, method=method
            )
            plan = plan_frozen_ranking(
                grid,
                initial_state(grid),
                ranking=ranking,
                checkpoints=checkpoints,
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
                dataset_id=outer_domain,
                patch_cache=patch_cache,
            )
            pending.extend((method, plan, snapshot) for snapshot in snapshots)
        vectors = _encode_many(encoder, [entry[2].image for entry in pending])
        metadata = authority.metadata13[authority_index : authority_index + 1]
        target_value = float(authority.targets[authority_index])
        for row_index, (method, plan, snapshot) in enumerate(pending):
            vector = vectors[row_index : row_index + 1]
            p_a_prediction = float(models.p_a_model.predict(metadata, vector)[0])
            p_b_model = models.p_b_models[snapshot.checkpoint]
            p_b_prediction = float(p_b_model.predict(metadata, vector)[0])
            rows.append(
                {
                    "candidate_bank_state_sha256": bank.state_sha256,
                    "cumulative_actions": snapshot.state.levels.count(1),
                    "dataset_id": outer_domain,
                    "effective_budget": snapshot.effective_budget,
                    "measured_count": snapshot.measured_count,
                    "method": method,
                    "native_count": snapshot.native_count,
                    "nominal_checkpoint": snapshot.checkpoint,
                    "outer_domain": outer_domain,
                    "p_a_absolute_error": abs(target_value - p_a_prediction),
                    "p_a_prediction": p_a_prediction,
                    "p_a_predictor_state_sha256": models.p_a_model.state_sha256,
                    "p_b_absolute_error": abs(target_value - p_b_prediction),
                    "p_b_prediction": p_b_prediction,
                    "p_b_predictor_state_sha256": p_b_model.state_sha256,
                    "plan_state_sha256": plan.state_sha256,
                    "ranking_state_sha256": plan.ranking_state_sha256,
                    "score_evaluation_state_sha256": score_evaluation.state_sha256,
                    "specimen_id": specimen_id,
                    "target": target_value,
                }
            )
        if notify is not None:
            notify(
                f"[{outer_domain}] P1 CAI {local_index + 1}/{len(target_indices)}"
            )
    encoder.validate()
    curves = pl.DataFrame(rows, infer_schema_length=None).sort(
        ["outer_domain", "specimen_id", "method", "nominal_checkpoint"]
    )
    expected_rows = len(target_indices) * len(methods) * len(checkpoints)
    if curves.height != expected_rows:
        raise P1CAIError("P1 CAI output row count changed")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "checkpoints": checkpoints,
                "data_state_sha256": data.state_sha256,
                "evaluator_state_sha256": models.state_sha256,
                "methods": methods,
                "outer_domain": outer_domain,
                "score_evaluation_state_sha256": score_evaluation.state_sha256,
            }
        )
    )
    for row in curves.iter_rows(named=True):
        digest.update(_canonical_json(row))
    return P1OuterCAIEvaluation(
        outer_domain=outer_domain,
        acquisition_curves=curves,
        evaluator_state_sha256=models.state_sha256,
        state_sha256=digest.hexdigest(),
    )


def _validate_evaluation_tables(
    curves: pl.DataFrame,
    ranking_metrics: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
) -> tuple[tuple[str, ...], int]:
    curve_columns = {
        "outer_domain",
        "dataset_id",
        "specimen_id",
        "method",
        "nominal_checkpoint",
        "effective_budget",
        "p_a_absolute_error",
        "p_b_absolute_error",
    }
    ranking_columns = {
        "outer_domain",
        "dataset_id",
        "specimen_id",
        "method",
        "ndcg_10",
        "next_action_regret",
    }
    if (
        type(curves) is not pl.DataFrame
        or type(ranking_metrics) is not pl.DataFrame
        or not curve_columns <= set(curves.columns)
        or not ranking_columns <= set(ranking_metrics.columns)
        or len(domain_order) != 6
        or len(set(domain_order)) != 6
        or len(checkpoints) != 5
        or tuple(sorted(checkpoints)) != checkpoints
        or checkpoints[0] != 0.0625
        or checkpoints[-1] != 0.25
        or set(curves["outer_domain"]) != set(domain_order)
        or set(ranking_metrics["outer_domain"]) != set(domain_order)
        or curves.filter(pl.col("outer_domain") != pl.col("dataset_id")).height
        or ranking_metrics.filter(
            pl.col("outer_domain") != pl.col("dataset_id")
        ).height
        or {float(value) for value in curves["nominal_checkpoint"].unique()}
        != set(checkpoints)
        or curves.unique(
            subset=["outer_domain", "specimen_id", "method", "nominal_checkpoint"]
        ).height
        != curves.height
        or ranking_metrics.unique(
            subset=["outer_domain", "specimen_id", "method"]
        ).height
        != ranking_metrics.height
    ):
        raise P1CAIError("P1 CAI evaluation roster changed")
    methods = tuple(sorted(str(value) for value in curves["method"].unique()))
    if set(methods) != {str(value) for value in ranking_metrics["method"].unique()}:
        raise P1CAIError("P1 CAI and ranking method rosters differ")
    groups = curves.group_by("outer_domain", "specimen_id", "method").agg(
        pl.len().alias("rows"),
        pl.col("nominal_checkpoint").n_unique().alias("checkpoints"),
    )
    if (
        groups.filter(
            (pl.col("rows") != len(checkpoints))
            | (pl.col("checkpoints") != len(checkpoints))
        ).height
        or groups.height != ranking_metrics.height
        or any(
            not bool(curves.select(pl.col(column).is_finite().all()).item())
            for column in (
                "nominal_checkpoint",
                "effective_budget",
                "p_a_absolute_error",
                "p_b_absolute_error",
            )
        )
        or curves.filter(
            (pl.col("effective_budget") <= 0.0)
            | (pl.col("p_a_absolute_error") < 0.0)
            | (pl.col("p_b_absolute_error") < 0.0)
        ).height
    ):
        raise P1CAIError("P1 CAI curve values changed")
    specimens = curves.select("outer_domain", "specimen_id").unique()
    return methods, specimens.height


def aggregate_p1_evaluation(
    acquisition_curves: pl.DataFrame,
    ranking_metrics: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> P1AggregateEvaluation:
    """Compute normalized specimen AUEBC, equal-domain effects, and P1 status."""

    methods, specimen_count = _validate_evaluation_tables(
        acquisition_curves,
        ranking_metrics,
        domain_order=domain_order,
        checkpoints=checkpoints,
    )
    if bootstrap_seed != 20260831 or bootstrap_resamples < 1:
        raise P1CAIError("P1 bootstrap configuration changed")
    auebc_rows: list[dict[str, object]] = []
    for key, table in acquisition_curves.group_by(
        "outer_domain", "dataset_id", "specimen_id", "method",
        maintain_order=True,
    ):
        outer_domain, dataset_id, specimen_id, method = (
            str(value) for value in key
        )
        ordered = table.sort("nominal_checkpoint")
        if tuple(float(value) for value in ordered["nominal_checkpoint"]) != checkpoints:
            raise P1CAIError("P1 specimen checkpoint sequence changed")
        auebc_rows.append(
            {
                "cai_auebc": normalized_auebc(
                    ordered["nominal_checkpoint"], ordered["p_b_absolute_error"]
                ),
                "dataset_id": dataset_id,
                "method": method,
                "outer_domain": outer_domain,
                "p_a_cai_auebc": normalized_auebc(
                    ordered["nominal_checkpoint"], ordered["p_a_absolute_error"]
                ),
                "specimen_id": specimen_id,
            }
        )
    auebc = pl.DataFrame(auebc_rows, infer_schema_length=None)
    per_specimen = ranking_metrics.join(
        auebc,
        on=["outer_domain", "dataset_id", "specimen_id", "method"],
        how="inner",
        validate="1:1",
    ).sort(["outer_domain", "specimen_id", "method"])
    if per_specimen.height != ranking_metrics.height:
        raise P1CAIError("P1 ranking/CAI specimen join changed")
    numeric_metrics = tuple(
        column
        for column in (
            "cai_auebc",
            "p_a_cai_auebc",
            "next_action_regret",
            "one_step_cai_utility",
            "spearman",
            "ndcg_10",
            "recall_5",
            "top_10_percent_overlap",
            "top_1_oracle_match",
        )
        if column in per_specimen.columns
    )
    domain_metrics = (
        per_specimen.group_by("outer_domain", "method")
        .agg(
            pl.len().alias("specimen_count"),
            *(pl.col(column).mean().alias(column) for column in numeric_metrics),
        )
        .sort(["outer_domain", "method"])
    )
    decision = decide_p1(
        per_specimen.select(
            "outer_domain", "specimen_id", "method", "cai_auebc"
        ),
        per_specimen.select(
            "outer_domain",
            "specimen_id",
            "method",
            "ndcg_10",
            "next_action_regret",
        ),
        domain_order=domain_order,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    bootstrap_rows: list[dict[str, object]] = []
    for name, effect in sorted(decision.effects.items()):
        bootstrap_rows.append(
            {
                "control": effect.control,
                "domain_effects": "|".join(
                    f"{domain}:{value:.17g}"
                    for domain, value in effect.domain_effects
                ),
                "effect_id": effect.effect_id,
                "effect_key": name,
                "improved_domains": effect.improved_domains,
                "lower": effect.lower,
                "point_estimate": effect.point_estimate,
                "proposed": effect.proposed,
                "resamples": effect.resamples,
                "seed": effect.seed,
                "specimen_count": effect.specimen_count,
                "state_sha256": effect.state_sha256,
                "upper": effect.upper,
                "value_column": effect.value_column,
            }
        )
    bootstrap = pl.DataFrame(bootstrap_rows, infer_schema_length=None).sort(
        "effect_key"
    )
    aggregate = (
        domain_metrics.group_by("method")
        .agg(
            pl.col("outer_domain").n_unique().alias("domain_count"),
            pl.col("specimen_count").sum().alias("specimen_count"),
            *(pl.col(column).mean().alias(column) for column in numeric_metrics),
        )
        .sort("method")
    )
    roles = {
        "c0_mvd_m1_o2": "frozen_old_deployable_reference",
        "c1_center_prior": "no_image_center_control",
        "old_refit_diagnostic": "same_target_old_state_diagnostic",
        "proposed": "source_selected_registered_surface_method",
        "c2_global_context": "global_surface_context_control",
        "c3_shuffled_surface": "shuffled_surface_control",
        "c4_wrong_orientation": "wrong_orientation_control",
        "c5_spatial_derangement": "spatial_derangement_control",
        "c3_shuffled_global": "shuffled_global_context_control",
        "mechanical_oracle_diagnostic": "evaluation_only_oracle_diagnostic",
    }
    control_results = aggregate.with_columns(
        pl.col("method")
        .replace_strict(roles, default="registered_diagnostic")
        .alias("role")
    ).select("method", "role", pl.exclude("method", "role"))
    if (
        set(control_results["method"]) != set(methods)
        or control_results.filter(pl.col("domain_count") != len(domain_order)).height
        or int(control_results["specimen_count"].min()) != specimen_count
    ):
        raise P1CAIError("P1 aggregate method coverage changed")
    digest = hashlib.sha256(
        _canonical_json(
            {
                "bootstrap_resamples": bootstrap_resamples,
                "bootstrap_seed": bootstrap_seed,
                "checkpoints": checkpoints,
                "decision_state_sha256": decision.state_sha256,
                "domain_order": domain_order,
                "methods": methods,
                "specimen_count": specimen_count,
            }
        )
    )
    for table in (per_specimen, domain_metrics, bootstrap, control_results):
        for row in table.iter_rows(named=True):
            digest.update(_canonical_json(row))
    return P1AggregateEvaluation(
        per_specimen_metrics=per_specimen,
        domain_metrics=domain_metrics,
        bootstrap=bootstrap,
        control_results=control_results,
        decision=decision,
        state_sha256=digest.hexdigest(),
    )


__all__ = [
    "P1AggregateEvaluation",
    "P1CAIError",
    "P1OuterCAIEvaluation",
    "aggregate_p1_evaluation",
    "run_p1_cai_outer",
]
