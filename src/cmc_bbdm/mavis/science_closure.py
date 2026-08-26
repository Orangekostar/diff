"""Read-only diagnostics over frozen MAVIS evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl


class ValueEvolutionError(ValueError):
    """Raised when frozen state/action evidence cannot support a causal join."""


_STATE_COLUMNS = {
    "domain_id",
    "specimen_id",
    "trajectory_id",
    "method",
    "state_id",
    "inspection_state_sha256",
    "step",
    "nominal_checkpoint",
    "exact_acquired_cost",
    "native_count",
    "effective_budget",
    "teacher_outer_domains",
}
_ACTION_COLUMNS = {
    "outer_domain",
    "domain_id",
    "specimen_id",
    "state_id",
    "mode",
    "candidate_index",
    "cell_index",
    "from_level",
    "to_level",
    "exact_added_cost",
    "predicted_score",
    "teacher_value",
    "teacher_fold_count",
    "dynamic_model_state_sha256",
}
_STATE_KEY = ["domain_id", "specimen_id", "state_id"]
_TRAJECTORY_KEY = ["domain_id", "specimen_id", "trajectory_id", "method"]
_ACTION_KEY = ["cell_index", "from_level", "to_level"]
_EVOLUTION_METRICS = (
    "rank_spearman",
    "top_k_jaccard",
    "best_action_turnover",
    "mean_absolute_value_shift",
    "dynamic_vs_initial_opportunity",
)


@dataclass(frozen=True, slots=True)
class ValueEvolutionMetricTables:
    per_specimen: pl.DataFrame
    per_domain: pl.DataFrame
    aggregate: pl.DataFrame


def _text_roster(value: Sequence[str], label: str) -> tuple[str, ...]:
    if (
        type(value) not in {tuple, list}
        or not value
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueEvolutionError(f"{label} is invalid")
    return tuple(value)


def _validate_inputs(
    states: pl.DataFrame,
    action_scores: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    modes: tuple[str, ...],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    domains = _text_roster(domain_order, "domain order")
    selected_modes = _text_roster(modes, "value mode roster")
    if type(states) is not pl.DataFrame or not _STATE_COLUMNS <= set(states.columns):
        raise ValueEvolutionError("state evidence schema is invalid")
    if type(action_scores) is not pl.DataFrame or not _ACTION_COLUMNS <= set(
        action_scores.columns
    ):
        raise ValueEvolutionError("action-score evidence schema is invalid")
    state_rows = states.select(sorted(_STATE_COLUMNS))
    if state_rows.is_empty() or state_rows.select(pl.struct(_STATE_KEY).n_unique()).item() != state_rows.height:
        raise ValueEvolutionError("causal state roster is empty or duplicated")
    expected_teachers = set(domains)
    for domain_id, teacher_domains in state_rows.select(
        "domain_id", "teacher_outer_domains"
    ).iter_rows():
        if (
            domain_id not in expected_teachers
            or type(teacher_domains) is not list
            or len(teacher_domains) != len(domains) - 1
            or len(set(teacher_domains)) != len(teacher_domains)
            or domain_id in teacher_domains
            or set(teacher_domains) != expected_teachers - {domain_id}
        ):
            raise ValueEvolutionError("state teacher roster is not strict-OOF")
    scores = action_scores.filter(pl.col("mode").is_in(selected_modes)).select(
        sorted(_ACTION_COLUMNS)
    )
    if scores.is_empty() or set(scores.get_column("mode").unique()) != set(
        selected_modes
    ):
        raise ValueEvolutionError("requested action-score modes are incomplete")
    expected_fold_count = len(domains) - 1
    if scores.filter(
        (pl.col("outer_domain") != pl.col("domain_id"))
        | (pl.col("teacher_fold_count") != expected_fold_count)
    ).height:
        raise ValueEvolutionError("action teacher evidence is not strict-OOF")
    return state_rows, scores


def _candidate_table(scores: pl.DataFrame, modes: tuple[str, ...]) -> pl.DataFrame:
    candidate_key = _STATE_KEY + _ACTION_KEY
    roster = scores.group_by(candidate_key).agg(
        pl.len().alias("row_count"),
        pl.col("mode").n_unique().alias("mode_count"),
        pl.col("candidate_index").n_unique().alias("candidate_index_count"),
        pl.col("exact_added_cost").n_unique().alias("cost_count"),
        pl.col("teacher_value").n_unique().alias("teacher_value_count"),
        pl.col("teacher_fold_count").n_unique().alias("teacher_fold_count_count"),
    )
    if roster.filter(
        (pl.col("row_count") != len(modes))
        | (pl.col("mode_count") != len(modes))
        | (pl.col("candidate_index_count") != 1)
        | (pl.col("cost_count") != 1)
        | (pl.col("teacher_value_count") != 1)
        | (pl.col("teacher_fold_count_count") != 1)
    ).height:
        raise ValueEvolutionError("mode action roster or teacher values changed")
    base = scores.group_by(candidate_key).agg(
        pl.col("candidate_index").first(),
        pl.col("exact_added_cost").first(),
        pl.col("teacher_value").first(),
        pl.col("teacher_fold_count").first(),
    )
    wide = scores.pivot(
        on="mode",
        index=candidate_key,
        values="predicted_score",
    ).rename({mode: f"score_{mode}" for mode in modes})
    result = base.join(wide, on=candidate_key, how="inner", validate="1:1")
    if result.height != base.height:
        raise ValueEvolutionError("mode action roster changed during alignment")
    return result


def build_value_evolution(
    states: pl.DataFrame,
    action_scores: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    modes: tuple[str, ...] = ("real", "positions_only", "shuffled", "static"),
) -> pl.DataFrame:
    """Align initial and later values for exact actions legal at both states."""

    state_rows, scores = _validate_inputs(
        states,
        action_scores,
        domain_order=domain_order,
        modes=modes,
    )
    candidates = _candidate_table(scores, modes)
    decision_state_ids = candidates.select(_STATE_KEY).unique()
    decision_states = state_rows.join(
        decision_state_ids,
        on=_STATE_KEY,
        how="inner",
        validate="1:1",
    ).sort(_TRAJECTORY_KEY + ["step", "state_id"])
    if decision_states.is_empty():
        raise ValueEvolutionError("no causal decision states are available")
    duplicated_steps = decision_states.group_by(_TRAJECTORY_KEY + ["step"]).len()
    if duplicated_steps.filter(pl.col("len") != 1).height:
        raise ValueEvolutionError("trajectory decision steps are duplicated")
    pairs = (
        decision_states.with_columns(
            pl.col("state_id").first().over(_TRAJECTORY_KEY).alias("initial_state_id"),
            pl.col("inspection_state_sha256")
            .first()
            .over(_TRAJECTORY_KEY)
            .alias("initial_inspection_state_sha256"),
            pl.col("step").first().over(_TRAJECTORY_KEY).alias("initial_step"),
            pl.col("nominal_checkpoint")
            .first()
            .over(_TRAJECTORY_KEY)
            .alias("initial_checkpoint"),
            pl.col("exact_acquired_cost")
            .first()
            .over(_TRAJECTORY_KEY)
            .alias("initial_acquired_cost"),
            pl.col("effective_budget")
            .first()
            .over(_TRAJECTORY_KEY)
            .alias("initial_effective_budget"),
        )
        .filter(pl.col("state_id") != pl.col("initial_state_id"))
        .select(
            _TRAJECTORY_KEY
            + [
                "initial_state_id",
                "initial_inspection_state_sha256",
                "initial_step",
                "initial_checkpoint",
                "initial_acquired_cost",
                "initial_effective_budget",
                pl.col("state_id").alias("current_state_id"),
                pl.col("inspection_state_sha256").alias(
                    "current_inspection_state_sha256"
                ),
                pl.col("step").alias("current_step"),
                pl.col("nominal_checkpoint").alias("current_checkpoint"),
                pl.col("exact_acquired_cost").alias("current_acquired_cost"),
                pl.col("effective_budget").alias("current_effective_budget"),
                "native_count",
            ]
        )
        .with_columns(
            (pl.col("current_acquired_cost") - pl.col("initial_acquired_cost")).alias(
                "acquired_cost_delta"
            )
        )
    )
    if pairs.is_empty() or pairs.filter(pl.col("acquired_cost_delta") <= 0).height:
        raise ValueEvolutionError("trajectory does not contain increasing causal cost")
    score_columns = [f"score_{mode}" for mode in modes]
    candidate_state = candidates.join(
        decision_states.select(_STATE_KEY + ["trajectory_id", "method"]),
        on=_STATE_KEY,
        how="inner",
        validate="m:1",
    )
    current = pairs.join(
        candidate_state,
        left_on=_TRAJECTORY_KEY + ["current_state_id"],
        right_on=_TRAJECTORY_KEY + ["state_id"],
        how="inner",
        validate="1:m",
    ).rename(
        {
            "candidate_index": "current_candidate_index",
            "exact_added_cost": "current_exact_added_cost",
            "teacher_value": "current_teacher_value",
            "teacher_fold_count": "current_teacher_fold_count",
            **{column: f"current_{column}" for column in score_columns},
        }
    )
    initial = candidate_state.select(
        _TRAJECTORY_KEY
        + [
            pl.col("state_id").alias("initial_state_id"),
            *_ACTION_KEY,
            pl.col("candidate_index").alias("initial_candidate_index"),
            pl.col("exact_added_cost").alias("initial_exact_added_cost"),
            pl.col("teacher_value").alias("initial_teacher_value"),
            pl.col("teacher_fold_count").alias("initial_teacher_fold_count"),
            *[
                pl.col(column).alias(f"initial_{column}")
                for column in score_columns
            ],
        ]
    )
    common = current.join(
        initial,
        on=_TRAJECTORY_KEY + ["initial_state_id"] + _ACTION_KEY,
        how="inner",
        validate="m:1",
    )
    pair_key = _TRAJECTORY_KEY + ["initial_state_id", "current_state_id"]
    if (
        common.select(pair_key).unique().height != pairs.height
        or common.filter(
            pl.col("initial_teacher_fold_count")
            != pl.col("current_teacher_fold_count")
        ).height
    ):
        raise ValueEvolutionError("longitudinal action or teacher roster changed")
    common = common.with_columns(
        pl.col("current_exact_added_cost").alias("exact_added_cost"),
        pl.col("current_teacher_fold_count").alias("teacher_fold_count"),
        pl.col("domain_id").alias("outer_domain"),
    )
    fixed = [
        "outer_domain",
        *_TRAJECTORY_KEY,
        "initial_state_id",
        "current_state_id",
        "initial_inspection_state_sha256",
        "current_inspection_state_sha256",
        "initial_step",
        "current_step",
        "initial_checkpoint",
        "current_checkpoint",
        "initial_acquired_cost",
        "current_acquired_cost",
        "acquired_cost_delta",
        "initial_effective_budget",
        "current_effective_budget",
        "native_count",
        *_ACTION_KEY,
        "initial_exact_added_cost",
        "current_exact_added_cost",
        "exact_added_cost",
        "teacher_fold_count",
        "initial_teacher_value",
        "current_teacher_value",
    ]
    outputs = [
        common.select(
            fixed
            + [
                pl.lit("teacher").alias("value_source"),
                pl.col("initial_teacher_value").alias("initial_value"),
                pl.col("current_teacher_value").alias("current_value"),
            ]
        )
    ]
    for mode in modes:
        outputs.append(
            common.select(
                fixed
                + [
                    pl.lit(mode).alias("value_source"),
                    pl.col(f"initial_score_{mode}").alias("initial_value"),
                    pl.col(f"current_score_{mode}").alias("current_value"),
                ]
            )
        )
    return pl.concat(outputs, how="vertical").sort(
        pair_key + ["value_source"] + _ACTION_KEY
    )


def evaluate_value_evolution(rows: pl.DataFrame, *, top_k: int) -> pl.DataFrame:
    """Compute pre-registered rank, overlap, turnover, shift, and opportunity."""

    required = {
        "outer_domain",
        *_TRAJECTORY_KEY,
        "initial_state_id",
        "current_state_id",
        "initial_step",
        "current_step",
        "initial_checkpoint",
        "current_checkpoint",
        "initial_acquired_cost",
        "current_acquired_cost",
        "acquired_cost_delta",
        "value_source",
        *_ACTION_KEY,
        "initial_value",
        "current_value",
        "current_teacher_value",
    }
    if (
        type(rows) is not pl.DataFrame
        or not required <= set(rows.columns)
        or rows.is_empty()
        or type(top_k) is not int
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueEvolutionError("value-evolution metric request is invalid")
    group = [
        "outer_domain",
        *_TRAJECTORY_KEY,
        "initial_state_id",
        "current_state_id",
        "initial_step",
        "current_step",
        "initial_checkpoint",
        "current_checkpoint",
        "initial_acquired_cost",
        "current_acquired_cost",
        "acquired_cost_delta",
        "value_source",
    ]
    ranked = (
        rows.sort(group + _ACTION_KEY)
        .with_columns(
            pl.col("initial_value")
            .rank(method="ordinal", descending=True)
            .over(group)
            .alias("initial_rank"),
            pl.col("current_value")
            .rank(method="ordinal", descending=True)
            .over(group)
            .alias("current_rank"),
            pl.concat_str(
                [
                    pl.col("cell_index"),
                    pl.col("from_level"),
                    pl.col("to_level"),
                ],
                separator=":",
            ).alias("action_key"),
        )
        .with_columns(
            (pl.col("initial_rank") <= top_k).alias("initial_top_k"),
            (pl.col("current_rank") <= top_k).alias("current_top_k"),
        )
    )
    metrics = ranked.group_by(group, maintain_order=True).agg(
        pl.len().alias("common_candidate_count"),
        pl.col("initial_value").n_unique().alias("initial_unique_value_count"),
        pl.col("current_value").n_unique().alias("current_unique_value_count"),
        pl.corr("initial_value", "current_value", method="spearman").alias(
            "raw_rank_spearman"
        ),
        (pl.col("initial_top_k") & pl.col("current_top_k"))
        .sum()
        .alias("top_k_intersection"),
        (pl.col("initial_top_k") | pl.col("current_top_k"))
        .sum()
        .alias("top_k_union"),
        pl.col("action_key")
        .filter(pl.col("initial_rank") == 1)
        .first()
        .alias("initial_best_action"),
        pl.col("action_key")
        .filter(pl.col("current_rank") == 1)
        .first()
        .alias("current_best_action"),
        pl.col("current_teacher_value")
        .filter(pl.col("initial_rank") == 1)
        .first()
        .alias("initial_best_current_teacher_value"),
        pl.col("current_teacher_value")
        .filter(pl.col("current_rank") == 1)
        .first()
        .alias("current_best_teacher_value"),
        (pl.col("current_value") - pl.col("initial_value"))
        .abs()
        .mean()
        .alias("mean_absolute_value_shift"),
    )
    return (
        metrics.with_columns(
            pl.when(
                (pl.col("initial_unique_value_count") == 1)
                & (pl.col("current_unique_value_count") == 1)
            )
            .then(1.0)
            .when(
                (pl.col("initial_unique_value_count") == 1)
                | (pl.col("current_unique_value_count") == 1)
            )
            .then(0.0)
            .otherwise(pl.col("raw_rank_spearman"))
            .alias("rank_spearman"),
            (
                pl.col("top_k_intersection") / pl.col("top_k_union")
            ).alias("top_k_jaccard"),
            (
                pl.col("initial_best_action") != pl.col("current_best_action")
            ).alias("best_action_changed"),
            (
                pl.col("current_best_teacher_value")
                - pl.col("initial_best_current_teacher_value")
            ).alias("dynamic_vs_initial_opportunity"),
            pl.lit(top_k).alias("top_k"),
        )
        .drop(
            "initial_unique_value_count",
            "current_unique_value_count",
            "raw_rank_spearman",
            "top_k_intersection",
            "top_k_union",
            "initial_best_current_teacher_value",
            "current_best_teacher_value",
        )
        .sort(group)
    )


def aggregate_value_evolution(
    pair_metrics: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
) -> ValueEvolutionMetricTables:
    """Reduce trajectories to specimens before equal-domain aggregation."""

    domains = _text_roster(domain_order, "domain order")
    required = {
        "outer_domain",
        "domain_id",
        "specimen_id",
        "trajectory_id",
        "initial_checkpoint",
        "current_checkpoint",
        "value_source",
        "common_candidate_count",
        "rank_spearman",
        "top_k_jaccard",
        "best_action_changed",
        "mean_absolute_value_shift",
        "dynamic_vs_initial_opportunity",
        "top_k",
    }
    if (
        type(pair_metrics) is not pl.DataFrame
        or not required <= set(pair_metrics.columns)
        or pair_metrics.is_empty()
        or set(pair_metrics.get_column("domain_id").unique()) != set(domains)
        or pair_metrics.filter(pl.col("outer_domain") != pl.col("domain_id")).height
    ):
        raise ValueEvolutionError("value-evolution aggregation evidence is invalid")
    specimen_key = [
        "outer_domain",
        "domain_id",
        "specimen_id",
        "initial_checkpoint",
        "current_checkpoint",
        "value_source",
    ]
    per_specimen = (
        pair_metrics.group_by(specimen_key, maintain_order=True)
        .agg(
            pl.col("trajectory_id").n_unique().alias("trajectory_count"),
            pl.len().alias("state_pair_count"),
            pl.col("common_candidate_count").mean(),
            pl.col("rank_spearman").mean(),
            pl.col("top_k_jaccard").mean(),
            pl.col("best_action_changed")
            .cast(pl.Float64)
            .mean()
            .alias("best_action_turnover"),
            pl.col("mean_absolute_value_shift").mean(),
            pl.col("dynamic_vs_initial_opportunity").mean(),
            pl.col("top_k").first(),
        )
        .sort(specimen_key)
    )
    domain_key = [
        "outer_domain",
        "domain_id",
        "initial_checkpoint",
        "current_checkpoint",
        "value_source",
    ]
    per_domain = (
        per_specimen.group_by(domain_key, maintain_order=True)
        .agg(
            pl.col("specimen_id").n_unique().alias("specimen_count"),
            pl.col("trajectory_count").sum().alias("trajectory_count"),
            pl.col("state_pair_count").sum().alias("state_pair_count"),
            pl.col("common_candidate_count").mean(),
            *[pl.col(metric).mean() for metric in _EVOLUTION_METRICS],
            pl.col("top_k").first(),
        )
        .sort(domain_key)
    )
    comparison_key = ["initial_checkpoint", "current_checkpoint", "value_source"]
    coverage = per_domain.group_by(comparison_key).agg(
        pl.col("domain_id").n_unique().alias("domain_count")
    )
    if coverage.filter(pl.col("domain_count") != len(domains)).height:
        raise ValueEvolutionError("equal-domain value-evolution roster is incomplete")
    aggregate = (
        per_domain.group_by(comparison_key, maintain_order=True)
        .agg(
            pl.col("domain_id").n_unique().alias("domain_count"),
            pl.col("specimen_count").sum().alias("specimen_count"),
            pl.col("trajectory_count").sum().alias("trajectory_count"),
            pl.col("state_pair_count").sum().alias("state_pair_count"),
            pl.col("common_candidate_count").mean(),
            *[pl.col(metric).mean() for metric in _EVOLUTION_METRICS],
            pl.col("top_k").first(),
            pl.lit("equal_domain").alias("statistical_unit"),
        )
        .sort(comparison_key)
    )
    return ValueEvolutionMetricTables(
        per_specimen=per_specimen,
        per_domain=per_domain,
        aggregate=aggregate,
    )


def _bootstrap_summary(
    values: np.ndarray,
    *,
    estimate: float,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    return {
        "estimate": estimate,
        "bootstrap_mean": float(np.mean(values)),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "fraction_above_zero": float(np.mean(values > 0.0)),
        "bootstrap_replicates": replicates,
        "seed": seed,
    }


def bootstrap_value_evolution(
    per_specimen: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    replicates: int,
    seed: int,
) -> pl.DataFrame:
    """Pair modes and resample physical specimens within held-out domains."""

    domains = _text_roster(domain_order, "domain order")
    required = {
        "domain_id",
        "specimen_id",
        "initial_checkpoint",
        "current_checkpoint",
        "value_source",
        *_EVOLUTION_METRICS,
    }
    if (
        type(per_specimen) is not pl.DataFrame
        or not required <= set(per_specimen.columns)
        or per_specimen.is_empty()
        or type(replicates) is not int
        or isinstance(replicates, bool)
        or replicates < 2
        or type(seed) is not int
        or isinstance(seed, bool)
        or set(per_specimen.get_column("domain_id").unique()) != set(domains)
    ):
        raise ValueEvolutionError("value-evolution bootstrap request is invalid")
    sources = tuple(sorted(per_specimen.get_column("value_source").unique()))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    checkpoint_rows = per_specimen.select(
        "initial_checkpoint", "current_checkpoint"
    ).unique().sort(["initial_checkpoint", "current_checkpoint"])
    for initial_checkpoint, current_checkpoint in checkpoint_rows.iter_rows():
        checkpoint = per_specimen.filter(
            (pl.col("initial_checkpoint") == initial_checkpoint)
            & (pl.col("current_checkpoint") == current_checkpoint)
        )
        source_coverage = checkpoint.group_by("domain_id", "specimen_id").agg(
            pl.col("value_source").n_unique().alias("source_count")
        )
        if source_coverage.filter(pl.col("source_count") != len(sources)).height:
            raise ValueEvolutionError("bootstrap mode pairing is incomplete")
        sampled_indices: dict[str, np.ndarray] = {}
        for domain in domains:
            count = checkpoint.filter(pl.col("domain_id") == domain).select(
                pl.col("specimen_id").n_unique()
            ).item()
            if count <= 0:
                raise ValueEvolutionError("bootstrap domain roster is incomplete")
            sampled_indices[domain] = rng.integers(0, count, size=(replicates, count))
        for metric in _EVOLUTION_METRICS:
            domain_estimates: list[np.ndarray] = []
            domain_replicates: list[np.ndarray] = []
            for domain in domains:
                wide = (
                    checkpoint.filter(pl.col("domain_id") == domain)
                    .select("specimen_id", "value_source", metric)
                    .pivot(
                        on="value_source",
                        index="specimen_id",
                        values=metric,
                    )
                    .sort("specimen_id")
                )
                if set(wide.columns[1:]) != set(sources):
                    raise ValueEvolutionError("bootstrap source roster changed")
                values = wide.select(list(sources)).to_numpy().astype(np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueEvolutionError("bootstrap metric contains non-finite values")
                domain_estimates.append(np.mean(values, axis=0))
                domain_replicates.append(
                    np.mean(values[sampled_indices[domain]], axis=1)
                )
            estimates = np.mean(np.stack(domain_estimates, axis=0), axis=0)
            sampled = np.mean(np.stack(domain_replicates, axis=0), axis=0)
            source_index = {source: index for index, source in enumerate(sources)}
            common = {
                "initial_checkpoint": initial_checkpoint,
                "current_checkpoint": current_checkpoint,
                "metric": metric,
                "statistical_unit": "physical_specimen_within_domain",
            }
            for source, index in source_index.items():
                rows.append(
                    {
                        **common,
                        "contrast": source,
                        "left_source": source,
                        "right_source": "zero",
                        **_bootstrap_summary(
                            sampled[:, index],
                            estimate=float(estimates[index]),
                            replicates=replicates,
                            seed=seed,
                        ),
                    }
                )
            if "real" in source_index:
                for control in ("positions_only", "shuffled", "static"):
                    if control not in source_index:
                        continue
                    difference = (
                        sampled[:, source_index["real"]]
                        - sampled[:, source_index[control]]
                    )
                    rows.append(
                        {
                            **common,
                            "contrast": f"real_minus_{control}",
                            "left_source": "real",
                            "right_source": control,
                            **_bootstrap_summary(
                                difference,
                                estimate=float(
                                    estimates[source_index["real"]]
                                    - estimates[source_index[control]]
                                ),
                                replicates=replicates,
                                seed=seed,
                            ),
                        }
                    )
    return pl.DataFrame(rows).sort(
        "initial_checkpoint", "current_checkpoint", "metric", "contrast"
    )


__all__ = [
    "ValueEvolutionError",
    "ValueEvolutionMetricTables",
    "aggregate_value_evolution",
    "bootstrap_value_evolution",
    "build_value_evolution",
    "evaluate_value_evolution",
]
