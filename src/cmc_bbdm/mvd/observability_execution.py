"""Strict grouped-CV execution for MVD M1 Mechanical Value observability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid

from .authority import load_compact_mvd_authority
from .config import load_mvd_config
from .observability_dataset import (
    ObservedValueExamples,
    build_outer_observability_examples,
    load_observed_candidate_feature_bank,
    subset_observed_examples,
)
from .observability_metrics import build_exact_cost_context, evaluate_ranking
from .observability_models import fit_mlp_scorer, fit_ridge_scorer


def _order(scores: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(range(64), key=lambda cell: (-scores[cell], cell)))


def _ndcg10(truth: np.ndarray, scores: np.ndarray) -> float:
    relevance = truth - float(np.min(truth))
    weights = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    denominator = float(np.sum(relevance[list(_order(truth)[:10])] * weights))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0
    return float(np.sum(relevance[list(_order(scores)[:10])] * weights) / denominator)


def _rank_summary(truth: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    ndcg = np.mean(
        [_ndcg10(truth[index], scores[index]) for index in range(truth.shape[0])],
        dtype=np.float64,
    )
    correlations: list[float] = []
    for index in range(truth.shape[0]):
        value = float(spearmanr(truth[index], scores[index]).statistic)
        correlations.append(0.0 if not np.isfinite(value) else value)
    return float(ndcg), float(np.mean(correlations, dtype=np.float64))


def _configurations(config: object) -> tuple[dict[str, object], ...]:
    return tuple(
        [
            {
                "config_id": f"ridge_alpha_{alpha:g}",
                "family": "ridge",
                "alpha": alpha,
                "loss": "value_regression",
                "ranking_lambda": 0.0,
                "parameter_count": 522,
            }
            for alpha in config.observability_ridge_alphas
        ]
        + [
            {
                "config_id": "mlp_top1",
                "family": "mlp",
                "alpha": None,
                "loss": "top1",
                "ranking_lambda": 0.0,
                "parameter_count": 37_393,
            },
            {
                "config_id": "mlp_huber",
                "family": "mlp",
                "alpha": None,
                "loss": "huber",
                "ranking_lambda": 0.0,
                "parameter_count": 37_393,
            },
            *[
                {
                    "config_id": f"mlp_huber_rank_{value:g}",
                    "family": "mlp",
                    "alpha": None,
                    "loss": "huber_rank",
                    "ranking_lambda": value,
                    "parameter_count": 37_393,
                }
                for value in config.observability_ranking_lambdas
            ],
        ]
    )


def _fit_configuration(
    examples: ObservedValueExamples,
    *,
    outer_domain: str,
    mode: str,
    specification: dict[str, object],
    epochs: int,
):
    if specification["family"] == "ridge":
        return fit_ridge_scorer(
            examples,
            outer_domain=outer_domain,
            mode=mode,
            alpha=float(specification["alpha"]),
        )
    return fit_mlp_scorer(
        examples,
        outer_domain=outer_domain,
        mode=mode,
        loss=str(specification["loss"]),
        ranking_lambda=float(specification["ranking_lambda"]),
        epochs=epochs,
    )


def select_source_configuration(
    examples: ObservedValueExamples,
    *,
    outer_domain: str,
    mode: str,
    specifications: tuple[dict[str, object], ...],
    epochs: int,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Select one configuration using only five source-domain validation folds."""

    source_domains = tuple(dict.fromkeys(examples.dataset_ids))
    if (
        examples.role != "source_train"
        or outer_domain in source_domains
        or len(source_domains) != 5
        or not specifications
    ):
        raise ValueError("source-only observability selection changed")
    rows: list[dict[str, object]] = []
    aggregates: list[dict[str, object]] = []
    for specification in specifications:
        ndcg_values: list[float] = []
        spearman_values: list[float] = []
        for validation_domain in source_domains:
            train_domains = tuple(
                domain for domain in source_domains if domain != validation_domain
            )
            train = subset_observed_examples(
                examples, included_domains=train_domains, role="source_train"
            )
            validation = subset_observed_examples(
                examples,
                included_domains=(validation_domain,),
                role="source_validation",
            )
            model = _fit_configuration(
                train,
                outer_domain=outer_domain,
                mode=mode,
                specification=specification,
                epochs=epochs,
            )
            predictions = model.predict(validation)
            ndcg, association = _rank_summary(
                validation.mechanical_values, predictions
            )
            ndcg_values.append(ndcg)
            spearman_values.append(association)
            rows.append(
                {
                    **specification,
                    "outer_domain": outer_domain,
                    "mode": mode,
                    "validation_domain": validation_domain,
                    "fit_domains": "|".join(train_domains),
                    "validation_specimen_count": validation.specimen_count,
                    "ndcg_10": ndcg,
                    "spearman": association,
                    "model_state_sha256": model.state_sha256,
                }
            )
            print(
                f"[{outer_domain}] select {mode} {specification['config_id']} "
                f"validation={validation_domain}",
                flush=True,
            )
        aggregates.append(
            {
                **specification,
                "mean_ndcg_10": float(np.mean(ndcg_values, dtype=np.float64)),
                "mean_spearman": float(
                    np.mean(spearman_values, dtype=np.float64)
                ),
            }
        )
    selected = min(
        aggregates,
        key=lambda row: (
            -float(row["mean_ndcg_10"]),
            -float(row["mean_spearman"]),
            int(row["parameter_count"]),
            str(row["config_id"]),
        ),
    )
    selection_rows = tuple(
        [
            {
                **row,
                "selected": str(row["config_id"]) == str(selected["config_id"]),
            }
            for row in rows
        ]
        + [
            {
                **row,
                "outer_domain": outer_domain,
                "mode": mode,
                "validation_domain": "EQUAL_SOURCE_MEAN",
                "fit_domains": "|".join(source_domains),
                "validation_specimen_count": examples.specimen_count,
                "ndcg_10": row["mean_ndcg_10"],
                "spearman": row["mean_spearman"],
                "model_state_sha256": None,
                "selected": str(row["config_id"]) == str(selected["config_id"]),
            }
            for row in aggregates
        ]
    )
    return selected, selection_rows


def _ridge_specifications(
    config: object, *, parameter_count: int
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "config_id": f"ridge_alpha_{alpha:g}",
            "family": "ridge",
            "alpha": alpha,
            "loss": "value_regression",
            "ranking_lambda": 0.0,
            "parameter_count": parameter_count,
        }
        for alpha in config.observability_ridge_alphas
    )


def _global_scores(root: Path, config: object, outer_domain: str) -> np.ndarray:
    table = pl.read_csv(root / config.sources["a4_rankings"].path).filter(
        (pl.col("outer_domain") == outer_domain)
        & (pl.col("method") == "global_mechanical_mask")
    ).sort("cell_index")
    if table.height != 64 or tuple(table["cell_index"]) != tuple(range(64)):
        raise ValueError("A4 global ranking roster changed")
    return np.asarray(table["cell_score"], dtype=np.float64)


def run_m1_outer_worker(
    config_path: str | Path,
    *,
    project_root: str | Path,
    outer_domain: str,
) -> Path:
    """Select on five sources, freeze, then evaluate all M1 target rankings."""

    root = Path(project_root).resolve(strict=True)
    config = load_mvd_config(config_path, project_root=root)
    compact = load_compact_mvd_authority(config, project_root=root)
    if outer_domain not in config.domain_order:
        raise ValueError("outer domain is not registered")
    budget = config.initial_budgets[outer_domain]
    token = str(budget).replace(".", "p")
    feature_bank = load_observed_candidate_feature_bank(
        root / config.sources[f"observed_features_{token}"].path,
        compact=compact,
        initial_budget=budget,
    )
    target_values = pl.read_parquet(
        root / config.output_work / "m0_domains" / outer_domain / "initial_values.parquet"
    )
    source, target = build_outer_observability_examples(
        compact, feature_bank, target_values, outer_domain=outer_domain
    )

    selected_o2, o2_audits = select_source_configuration(
        source,
        outer_domain=outer_domain,
        mode="global_candidate",
        specifications=_configurations(config),
        epochs=config.observability_epochs,
    )
    selected_o1, o1_audits = select_source_configuration(
        source,
        outer_domain=outer_domain,
        mode="candidate_only",
        specifications=_ridge_specifications(config, parameter_count=9),
        epochs=config.observability_epochs,
    )
    selected_o3, o3_audits = select_source_configuration(
        source,
        outer_domain=outer_domain,
        mode="a5_initial",
        specifications=_ridge_specifications(config, parameter_count=588),
        epochs=config.observability_epochs,
    )
    # Target labels have not been read by any selection call above.
    fitted = {
        "o1_candidate_ridge": _fit_configuration(
            source,
            outer_domain=outer_domain,
            mode="candidate_only",
            specification=selected_o1,
            epochs=config.observability_epochs,
        ),
        "o1_candidate_mlp_huber": fit_mlp_scorer(
            source,
            outer_domain=outer_domain,
            mode="candidate_only",
            loss="huber",
            epochs=config.observability_epochs,
        ),
        "o2_global_candidate": _fit_configuration(
            source,
            outer_domain=outer_domain,
            mode="global_candidate",
            specification=selected_o2,
            epochs=config.observability_epochs,
        ),
        "o3_a5_ridge": _fit_configuration(
            source,
            outer_domain=outer_domain,
            mode="a5_initial",
            specification=selected_o3,
            epochs=config.observability_epochs,
        ),
        "o3_a5_mlp_huber": fit_mlp_scorer(
            source,
            outer_domain=outer_domain,
            mode="a5_initial",
            loss="huber",
            epochs=config.observability_epochs,
        ),
    }
    deterministic_scores = {
        method: model.predict(target) for method, model in fitted.items()
    }
    deterministic_scores["global_mechanical"] = np.tile(
        _global_scores(root, config, outer_domain), (target.specimen_count, 1)
    )
    deterministic_scores["observed_uncertainty"] = (
        target.candidate_features[:, :, 3]
        * (target.candidate_features[:, :, 6] + target.candidate_features[:, :, 7])
    )
    checkpoints = tuple(
        checkpoint for checkpoint in config.checkpoints if checkpoint >= 0.0625
    )
    bank = compact.candidate_banks[budget]
    global_indices = {
        specimen_id: compact.specimen_ids.index(specimen_id)
        for specimen_id in target.specimen_ids
    }
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    regret_rows: list[dict[str, object]] = []
    for specimen_index, (specimen_id, dataset_id) in enumerate(
        zip(target.specimen_ids, target.dataset_ids, strict=True)
    ):
        global_index = global_indices[specimen_id]
        native_shape = bank.native_shapes[global_index]
        grid = build_acquisition_grid(
            native_shape[0], native_shape[1], initial_budget=budget
        )
        exact_cost_context = build_exact_cost_context(grid)
        truth = target.mechanical_values[specimen_index]
        for method, matrix in deterministic_scores.items():
            scores = matrix[specimen_index]
            metrics = evaluate_ranking(
                truth,
                scores,
                grid=grid,
                checkpoints=checkpoints,
                exact_cost_context=exact_cost_context,
            )
            metric_rows.append(
                {
                    "outer_domain": outer_domain,
                    "specimen_id": specimen_id,
                    "dataset_id": dataset_id,
                    "method": method,
                    "spearman": metrics.spearman,
                    "ndcg_5": metrics.ndcg_5,
                    "ndcg_10": metrics.ndcg_10,
                    "recall_5": metrics.recall_5,
                    "recall_10": metrics.recall_10,
                    "regret_1": metrics.regret_1,
                    "mean_budgeted_regret": float(
                        np.mean(metrics.budgeted_regret, dtype=np.float64)
                    ),
                    "model_state_sha256": getattr(
                        fitted.get(method), "state_sha256", None
                    ),
                }
            )
            for checkpoint, regret, selected_value, oracle_value, capture in zip(
                metrics.checkpoints,
                metrics.budgeted_regret,
                metrics.selected_value,
                metrics.oracle_value,
                metrics.value_capture,
                strict=True,
            ):
                regret_rows.append(
                    {
                        "outer_domain": outer_domain,
                        "specimen_id": specimen_id,
                        "dataset_id": dataset_id,
                        "method": method,
                        "nominal_checkpoint": checkpoint,
                        "budgeted_regret": regret,
                        "selected_mechanical_value": selected_value,
                        "oracle_mechanical_value": oracle_value,
                        "value_capture": capture,
                    }
                )
            for cell_index in range(64):
                prediction_rows.append(
                    {
                        "outer_domain": outer_domain,
                        "specimen_id": specimen_id,
                        "dataset_id": dataset_id,
                        "method": method,
                        "cell_index": cell_index,
                        "predicted_value": float(scores[cell_index]),
                        "teacher_value": float(truth[cell_index]),
                        "candidate_cost": int(
                            target.candidate_costs[specimen_index, cell_index]
                        ),
                    }
                )
        random_metrics: list[object] = []
        for seed in range(
            config.random_seed_start,
            config.random_seed_start + config.random_seed_count,
        ):
            scores = np.random.Generator(np.random.PCG64(seed)).random(64)
            random_metrics.append(
                evaluate_ranking(
                    truth,
                    scores,
                    grid=grid,
                    checkpoints=checkpoints,
                    exact_cost_context=exact_cost_context,
                )
            )
        metric_rows.append(
            {
                "outer_domain": outer_domain,
                "specimen_id": specimen_id,
                "dataset_id": dataset_id,
                "method": "random_median",
                "spearman": float(np.median([value.spearman for value in random_metrics])),
                "ndcg_5": float(np.median([value.ndcg_5 for value in random_metrics])),
                "ndcg_10": float(np.median([value.ndcg_10 for value in random_metrics])),
                "recall_5": float(np.median([value.recall_5 for value in random_metrics])),
                "recall_10": float(np.median([value.recall_10 for value in random_metrics])),
                "regret_1": float(np.median([value.regret_1 for value in random_metrics])),
                "mean_budgeted_regret": float(
                    np.median(
                        [np.mean(value.budgeted_regret) for value in random_metrics]
                    )
                ),
                "model_state_sha256": None,
            }
        )
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            regret_rows.append(
                {
                    "outer_domain": outer_domain,
                    "specimen_id": specimen_id,
                    "dataset_id": dataset_id,
                    "method": "random_median",
                    "nominal_checkpoint": checkpoint,
                    "budgeted_regret": float(
                        np.median(
                            [
                                value.budgeted_regret[checkpoint_index]
                                for value in random_metrics
                            ]
                        )
                    ),
                    "selected_mechanical_value": float(
                        np.median(
                            [
                                value.selected_value[checkpoint_index]
                                for value in random_metrics
                            ]
                        )
                    ),
                    "oracle_mechanical_value": random_metrics[0].oracle_value[
                        checkpoint_index
                    ],
                    "value_capture": float(
                        np.median(
                            [
                                value.value_capture[checkpoint_index]
                                for value in random_metrics
                                if value.value_capture[checkpoint_index] is not None
                            ]
                        )
                    ),
                }
            )
        print(
            f"[{outer_domain}] M1 target {specimen_index + 1}/{target.specimen_count}",
            flush=True,
        )
    output = root / config.output_work / "m1_domains" / outer_domain
    output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(prediction_rows).sort(
        ["specimen_id", "method", "cell_index"]
    ).write_parquet(output / "predictions.parquet", compression="zstd")
    pl.DataFrame(metric_rows).sort(["specimen_id", "method"]).write_csv(
        output / "metrics.csv"
    )
    pl.DataFrame(regret_rows, infer_schema_length=None).sort(
        ["specimen_id", "method", "nominal_checkpoint"]
    ).write_csv(output / "regret.csv")
    selection = (*o2_audits, *o1_audits, *o3_audits)
    pl.DataFrame(selection, infer_schema_length=None).write_csv(
        output / "selection_audit.csv"
    )
    summary = {
        "outer_domain": outer_domain,
        "source_domains": tuple(dict.fromkeys(source.dataset_ids)),
        "source_specimen_count": source.specimen_count,
        "target_specimen_count": target.specimen_count,
        "selected_o2": selected_o2,
        "selected_o1_ridge": selected_o1,
        "selected_o3_ridge": selected_o3,
        "model_states": {
            method: model.state_sha256 for method, model in fitted.items()
        },
        "source_dataset_state_sha256": source.state_sha256,
        "target_dataset_state_sha256": target.state_sha256,
    }
    (output / "complete.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = ["run_m1_outer_worker", "select_source_configuration"]
