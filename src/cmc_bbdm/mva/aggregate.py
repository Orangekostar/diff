"""Independent aggregation, statistics, and A3 decision for MVA shards."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

from .authority import load_mva_authority, reproduce_full_baseline
from .budget_metrics import auebc, simulated_saving, sufficiency_budget
from .config import load_mva_config
from .crossfit import fit_outer_source_predictor
from .evaluation import A3GateInputs, evaluate_a3_gate
from .oracle_execution import (
    LOW_CHECKPOINTS,
    PRIMARY_CHECKPOINTS,
    _load_uniform_bank,
    _selected_budgets,
)
from .pipeline import _write_csv, _write_json
from .statistics import paired_domain_bootstrap, synchronized_bootstrap_indices

DETERMINISTIC_METHODS = (
    "uniform",
    "appearance_oracle",
    "reconstruction_oracle",
    "mechanical_oracle",
)
REPORT_CHECKPOINTS = (*LOW_CHECKPOINTS, *PRIMARY_CHECKPOINTS)


def _validate_shards(
    root: Path, domains: tuple[str, ...]
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    states: list[pl.DataFrame] = []
    trajectories: list[pl.DataFrame] = []
    values: list[pl.DataFrame] = []
    for domain in domains:
        leaf = root / "results/mva/.work/a2_domains" / domain
        complete = json.loads((leaf / "complete.json").read_text(encoding="utf-8"))
        if (
            complete["outer_domain"] != domain
            or complete["random_seed_count"] != 100
            or complete["specimen_count"] <= 0
        ):
            raise RuntimeError(f"incomplete formal A2 shard: {domain}")
        state = pl.read_parquet(leaf / "states.parquet")
        trajectory = pl.read_parquet(leaf / "trajectories.parquet")
        value = pl.read_parquet(leaf / "oracle_values.parquet")
        required_value_columns = {
            "budget_before",
            "candidate",
            "value",
            "budget_after",
            "current_prediction",
            "new_prediction",
            "current_error",
            "new_error",
        }
        if (
            state.height != complete["state_rows"]
            or trajectory.height != complete["trajectory_rows"]
            or value.height != complete["value_rows"]
            or not required_value_columns <= set(value.columns)
        ):
            raise RuntimeError(f"A2 shard row count changed: {domain}")
        mechanical = value.filter(pl.col("method") == "mechanical_oracle")
        if (
            mechanical.height == 0
            or mechanical.select(
                pl.any_horizontal(
                    pl.col("current_prediction").is_null(),
                    pl.col("new_prediction").is_null(),
                    pl.col("current_error").is_null(),
                    pl.col("new_error").is_null(),
                    pl.col("budget_before") > pl.col("budget_after"),
                    pl.col("candidate") != pl.col("cell_index"),
                    pl.col("value") != pl.col("primary_value"),
                    pl.col("current_error") != pl.col("error_before"),
                    pl.col("new_error") != pl.col("error_after"),
                ).any()
            ).item()
        ):
            raise RuntimeError(f"A2 oracle transition contract changed: {domain}")
        states.append(state)
        trajectories.append(trajectory)
        values.append(value)
    state_table = pl.concat(states, how="vertical_relaxed").sort(
        ["dataset_id", "specimen_id", "method", "seed", "nominal_checkpoint"],
        nulls_last=False,
    )
    trajectory_table = pl.concat(trajectories, how="vertical_relaxed").sort(
        ["dataset_id", "specimen_id", "method", "seed", "step"],
        nulls_last=False,
    )
    value_table = pl.concat(values, how="vertical_relaxed").sort(
        ["dataset_id", "specimen_id", "method", "step", "cell_index"]
    )
    return state_table, trajectory_table, value_table


def _validate_low_shards(root: Path, domains: tuple[str, ...]) -> pl.DataFrame:
    states: list[pl.DataFrame] = []
    for domain in domains:
        leaf = root / "results/mva/.work/a2_low_domains" / domain
        complete = json.loads((leaf / "complete.json").read_text(encoding="utf-8"))
        if (
            complete["outer_domain"] != domain
            or complete["checkpoint"] != LOW_CHECKPOINTS[0]
            or complete["random_seed_count"] != 100
            or complete["specimen_count"] <= 0
        ):
            raise RuntimeError(f"incomplete formal low-budget shard: {domain}")
        state = pl.read_parquet(leaf / "states.parquet")
        if state.height != complete["state_rows"]:
            raise RuntimeError(f"low-budget shard row count changed: {domain}")
        states.append(state)
    return pl.concat(states, how="vertical_relaxed")


def _validate_stability_shards(root: Path, domains: tuple[str, ...]) -> pl.DataFrame:
    tables: list[pl.DataFrame] = []
    for domain in domains:
        leaf = root / "results/mva/.work/a2_stability_domains" / domain
        complete = json.loads((leaf / "complete.json").read_text(encoding="utf-8"))
        if (
            complete["outer_domain"] != domain
            or complete["specimen_count"] <= 0
            or complete["row_count"] != 4 * complete["specimen_count"]
            or float(complete["maximum_primary_delta"]) > 1.0e-12
        ):
            raise RuntimeError(f"incomplete formal stability shard: {domain}")
        table = pl.read_csv(leaf / "stability.csv")
        if table.height != complete["row_count"]:
            raise RuntimeError(f"stability shard row count changed: {domain}")
        tables.append(table)
    output = pl.concat(tables, how="vertical_relaxed").sort(
        ["dataset_id", "specimen_id", "variant"]
    )
    keys = ["specimen_id", "variant"]
    if output.unique(subset=keys).height != output.height:
        raise RuntimeError("stability roster contains duplicates")
    return output


def _expected_roster(states: pl.DataFrame, specimen_count: int) -> None:
    expected_deterministic = specimen_count * len(REPORT_CHECKPOINTS)
    for method in DETERMINISTIC_METHODS:
        count = states.filter(pl.col("method") == method).height
        if count != expected_deterministic:
            raise RuntimeError(f"state roster incomplete for {method}")
    random = states.filter(pl.col("method") == "random")
    if random.height != specimen_count * 100 * len(REPORT_CHECKPOINTS):
        raise RuntimeError("random state roster is incomplete")
    seeds = random.select(pl.col("seed").n_unique()).item()
    if seeds != 100:
        raise RuntimeError("random seed roster changed")
    keys = ["specimen_id", "method", "seed", "nominal_checkpoint"]
    if states.unique(subset=keys).height != states.height:
        raise RuntimeError("state roster contains duplicate checkpoints")


def _anchor_rows(
    *,
    config,
    authority,
    root: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, dict[float, float]]]]:
    full = reproduce_full_baseline(config, authority)
    domains = np.asarray(authority.dataset_ids, dtype=object)
    selected = _selected_budgets(root)
    rows: list[dict[str, object]] = []
    curves = {protocol: {"uniform": {}, "full": {}} for protocol in ("P-B", "P-A")}
    for outer_domain in config.domain_order:
        indices = np.flatnonzero(domains == outer_domain)
        embeddings, _, effective = _load_uniform_bank(
            root, authority, selected[outer_domain]
        )
        p_a_model = fit_outer_source_predictor(
            method="MVA_P_A_ANCHOR",
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
            method="MVA_P_B_0.5_ANCHOR",
            outer_domain=outer_domain,
            specimen_ids=authority.specimen_ids,
            dataset_ids=authority.dataset_ids,
            domain_order=config.domain_order,
            targets=authority.targets,
            metadata=authority.metadata13,
            embeddings=embeddings[0.5],
            pca_dimensions=config.pca_dimensions,
            ridge_alpha=config.ridge_alpha,
            tie_tolerance=1.0e-12,
        ).model
        target = authority.targets[indices]
        metadata = authority.metadata13[indices]
        vectors = embeddings[0.5][indices]
        for protocol, model in (("P-A", p_a_model), ("P-B", p_b_model)):
            error = np.abs(target - model.predict(metadata, vectors))
            mae = float(np.mean(error, dtype=np.float64))
            curves[protocol]["uniform"].setdefault(outer_domain, {})[0.5] = mae
            rows.append(
                {
                    "dataset_id": outer_domain,
                    "protocol": protocol,
                    "method": "uniform",
                    "nominal_checkpoint": 0.5,
                    "effective_mean": float(np.mean(effective[0.5][indices])),
                    "effective_min": float(np.min(effective[0.5][indices])),
                    "effective_max": float(np.max(effective[0.5][indices])),
                    "mae": mae,
                }
            )
        full_mae = float(full.domain_mae[config.domain_order.index(outer_domain)])
        for protocol in ("P-A", "P-B"):
            curves[protocol]["full"].setdefault(outer_domain, {})[1.0] = full_mae
            rows.append(
                {
                    "dataset_id": outer_domain,
                    "protocol": protocol,
                    "method": "full",
                    "nominal_checkpoint": 1.0,
                    "effective_mean": 1.0,
                    "effective_min": 1.0,
                    "effective_max": 1.0,
                    "mae": full_mae,
                }
            )
    return rows, curves


def _domain_curve(
    states: pl.DataFrame,
    *,
    method: str,
    error_column: str,
    seed: int | None = None,
) -> dict[str, dict[float, float]]:
    selected = states.filter(pl.col("method") == method)
    if seed is not None:
        selected = selected.filter(pl.col("seed") == seed)
    rows = selected.group_by(["dataset_id", "nominal_checkpoint"]).agg(
        pl.col(error_column).mean().alias("mae")
    )
    output: dict[str, dict[float, float]] = {}
    for row in rows.iter_rows(named=True):
        output.setdefault(str(row["dataset_id"]), {})[
            float(row["nominal_checkpoint"])
        ] = float(row["mae"])
    return output


def _curve_rows(
    states: pl.DataFrame,
    *,
    method: str,
    protocol: str,
    full_mae: float,
) -> list[dict[str, object]]:
    error = "p_b_absolute_error" if protocol == "P-B" else "p_a_absolute_error"
    selected = states.filter(pl.col("method") == method)
    if method == "random":
        per_seed = (
            selected.group_by(["seed", "dataset_id", "nominal_checkpoint"])
            .agg(pl.col(error).mean().alias("domain_mae"))
            .group_by(["seed", "nominal_checkpoint"])
            .agg(pl.col("domain_mae").mean().alias("equal_domain_mae"))
        )
        summary = per_seed.group_by("nominal_checkpoint").agg(
            pl.col("equal_domain_mae").mean().alias("mae_mean"),
            pl.col("equal_domain_mae").median().alias("mae_median"),
            pl.col("equal_domain_mae")
            .quantile(0.05, interpolation="linear")
            .alias("mae_q05"),
            pl.col("equal_domain_mae")
            .quantile(0.95, interpolation="linear")
            .alias("mae_q95"),
        )
        budgets = selected.group_by("nominal_checkpoint").agg(
            pl.col("effective_budget").mean().alias("effective_mean"),
            pl.col("effective_budget").min().alias("effective_min"),
            pl.col("effective_budget").max().alias("effective_max"),
        )
        joined = summary.join(budgets, on="nominal_checkpoint").sort(
            "nominal_checkpoint"
        )
        rows = [dict(row) for row in joined.iter_rows(named=True)]
        for row in rows:
            row.update({"method": method, "protocol": protocol})
        return rows
    domain = selected.group_by(["dataset_id", "nominal_checkpoint"]).agg(
        pl.col(error).mean().alias("domain_mae")
    )
    equal = domain.group_by("nominal_checkpoint").agg(
        pl.col("domain_mae").mean().alias("equal_domain_mae")
    )
    budgets = selected.group_by("nominal_checkpoint").agg(
        pl.col("effective_budget").mean().alias("effective_mean"),
        pl.col("effective_budget").min().alias("effective_min"),
        pl.col("effective_budget").max().alias("effective_max"),
        pl.col("normalized_rgb_mse").mean().alias("reconstruction_mse"),
        pl.col("ssim").mean().alias("ssim"),
    )
    joined = equal.join(budgets, on="nominal_checkpoint").sort("nominal_checkpoint")
    rows = [dict(row) for row in joined.iter_rows(named=True)]
    for row in rows:
        row.update({"method": method, "protocol": protocol})
    return rows


def _rbo(first: list[int], second: list[int], persistence: float = 0.9) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("rankings must be aligned")
    seen_first: set[int] = set()
    seen_second: set[int] = set()
    score = 0.0
    overlap = 0.0
    for depth, (left, right) in enumerate(zip(first, second, strict=True), start=1):
        seen_first.add(left)
        seen_second.add(right)
        overlap = len(seen_first & seen_second) / depth
        score += (1.0 - persistence) * overlap * persistence ** (depth - 1)
    return float(score + overlap * persistence ** len(first))


def _map_similarity(values: pl.DataFrame) -> list[dict[str, object]]:
    initial = values.filter(pl.col("step") == 0)
    pairs = (
        ("mechanical_oracle", "reconstruction_oracle"),
        ("mechanical_oracle", "appearance_oracle"),
        ("reconstruction_oracle", "appearance_oracle"),
    )
    rows: list[dict[str, object]] = []
    specimen_ids = initial.select("specimen_id").unique().to_series().sort().to_list()
    for specimen_id in specimen_ids:
        item = initial.filter(pl.col("specimen_id") == specimen_id)
        domain = str(item["dataset_id"][0])
        maps = {
            method: {
                int(row["cell_index"]): float(row["primary_value"])
                for row in item.filter(pl.col("method") == method).iter_rows(named=True)
            }
            for method in (
                "mechanical_oracle",
                "reconstruction_oracle",
                "appearance_oracle",
            )
        }
        for first, second in pairs:
            keys = sorted(set(maps[first]) & set(maps[second]))
            if len(keys) < 2:
                raise RuntimeError("initial oracle map roster is incomplete")
            left = np.asarray([maps[first][key] for key in keys], dtype=np.float64)
            right = np.asarray([maps[second][key] for key in keys], dtype=np.float64)
            rank_left = sorted(keys, key=lambda key: (-maps[first][key], key))
            rank_right = sorted(keys, key=lambda key: (-maps[second][key], key))
            top = max(1, math.ceil(0.1 * len(keys)))
            rows.append(
                {
                    "specimen_id": specimen_id,
                    "dataset_id": domain,
                    "first_method": first,
                    "second_method": second,
                    "pearson": float(pearsonr(left, right).statistic),
                    "spearman": float(spearmanr(left, right).statistic),
                    "top10_overlap": float(
                        len(set(rank_left[:top]) & set(rank_right[:top])) / top
                    ),
                    "rbo_p0_9": _rbo(rank_left, rank_right),
                    "candidate_count": len(keys),
                }
            )
    return rows


def _specimen_headroom(states: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    deterministic = states.filter(
        pl.col("method").is_in(["uniform", "mechanical_oracle"])
        & pl.col("nominal_checkpoint").is_in(list(PRIMARY_CHECKPOINTS))
    )
    for specimen_id in deterministic["specimen_id"].unique().sort():
        item = deterministic.filter(pl.col("specimen_id") == specimen_id)
        domain = str(item["dataset_id"][0])
        curves: dict[str, np.ndarray] = {}
        for method in ("uniform", "mechanical_oracle"):
            method_rows = item.filter(pl.col("method") == method).sort(
                "nominal_checkpoint"
            )
            if method_rows.height != len(PRIMARY_CHECKPOINTS):
                raise RuntimeError("specimen headroom curve is incomplete")
            curves[method] = method_rows["p_b_absolute_error"].to_numpy()
        uniform_area = auebc(np.asarray(PRIMARY_CHECKPOINTS), curves["uniform"])
        mechanical_area = auebc(
            np.asarray(PRIMARY_CHECKPOINTS), curves["mechanical_oracle"]
        )
        low_index = PRIMARY_CHECKPOINTS.index(0.125)
        rows.append(
            {
                "specimen_id": specimen_id,
                "dataset_id": domain,
                "uniform_auebc": uniform_area,
                "mechanical_auebc": mechanical_area,
                "auebc_reduction": uniform_area - mechanical_area,
                "uniform_mae_0p125": float(curves["uniform"][low_index]),
                "mechanical_mae_0p125": float(curves["mechanical_oracle"][low_index]),
            }
        )
    return sorted(
        rows, key=lambda row: (-float(row["auebc_reduction"]), str(row["specimen_id"]))
    )


def _map_summary(
    rows: list[dict[str, object]], first: str, second: str
) -> dict[str, float]:
    selected = [
        row
        for row in rows
        if row["first_method"] == first and row["second_method"] == second
    ]
    if not selected:
        raise RuntimeError("map-similarity comparison is missing")
    return {
        metric: float(np.mean([float(row[metric]) for row in selected]))
        for metric in ("pearson", "spearman", "top10_overlap", "rbo_p0_9")
    }


def _format_budget(value: float | None) -> str:
    return "not reached before FULL" if value is None else f"{100.0 * value:.3f}%"


def aggregate_a2(config_path: str | Path, *, project_root: str | Path) -> Path:
    """Validate all formal shards and publish the complete A2/A3 package."""

    root = Path(project_root).resolve(strict=True)
    config = load_mva_config(config_path, project_root=root)
    authority = load_mva_authority(config, project_root=root)
    states, trajectories, values = _validate_shards(root, config.domain_order)
    low_states = _validate_low_shards(root, config.domain_order)
    states = pl.concat((states, low_states), how="vertical_relaxed").sort(
        ["dataset_id", "specimen_id", "method", "seed", "nominal_checkpoint"],
        nulls_last=False,
    )
    _expected_roster(states, authority.specimen_count)
    anchor_rows, anchor_curves = _anchor_rows(
        config=config, authority=authority, root=root
    )
    stability = _validate_stability_shards(root, config.domain_order)
    if stability.height != authority.specimen_count * 4:
        raise RuntimeError("stability roster is incomplete")
    output = root / config.output_dir / "a2_oracle_value"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(config_path).resolve(strict=True), output / "config.yaml")
    states.write_parquet(output / "state_metrics.parquet", compression="zstd")
    trajectories.write_parquet(
        output / "oracle_trajectories.parquet", compression="zstd"
    )
    values.write_parquet(output / "oracle_values.parquet", compression="zstd")
    stability.write_csv(output / "stability_diagnostics.csv")

    curve_tables: dict[tuple[str, str], list[dict[str, object]]] = {}
    for protocol in ("P-B", "P-A"):
        for method in (*DETERMINISTIC_METHODS, "random"):
            curve_tables[(method, protocol)] = _curve_rows(
                states, method=method, protocol=protocol, full_mae=config.full_mae
            )
        anchor = [row for row in anchor_rows if row["protocol"] == protocol]
        uniform_50 = [row for row in anchor if row["method"] == "uniform"]
        curve_tables[("uniform", protocol)].append(
            {
                "nominal_checkpoint": 0.5,
                "equal_domain_mae": float(
                    np.mean([float(row["mae"]) for row in uniform_50])
                ),
                "effective_mean": float(
                    np.mean([float(row["effective_mean"]) for row in uniform_50])
                ),
                "effective_min": float(
                    min(float(row["effective_min"]) for row in uniform_50)
                ),
                "effective_max": float(
                    max(float(row["effective_max"]) for row in uniform_50)
                ),
                "reconstruction_mse": None,
                "ssim": None,
                "method": "uniform",
                "protocol": protocol,
            }
        )
    filenames = {
        "uniform": "uniform_curve.csv",
        "random": "random_curve.csv",
        "appearance_oracle": "appearance_curve.csv",
        "reconstruction_oracle": "reconstruction_oracle_curve.csv",
        "mechanical_oracle": "mechanical_oracle_curve.csv",
    }
    for method, filename in filenames.items():
        rows = curve_tables[(method, "P-B")] + curve_tables[(method, "P-A")]
        fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
        _write_csv(output / filename, fieldnames, rows)
    _write_csv(
        output / "anchor_curve.csv",
        (
            "dataset_id",
            "protocol",
            "method",
            "nominal_checkpoint",
            "effective_mean",
            "effective_min",
            "effective_max",
            "mae",
        ),
        anchor_rows,
    )

    error_column = "p_b_absolute_error"
    domain_curves = {
        method: _domain_curve(states, method=method, error_column=error_column)
        for method in DETERMINISTIC_METHODS
    }
    random_seeds = tuple(
        int(value)
        for value in states.filter(pl.col("method") == "random")["seed"].unique().sort()
    )
    random_domain_curves = {
        seed: _domain_curve(
            states, method="random", error_column=error_column, seed=seed
        )
        for seed in random_seeds
    }
    domain_auebc: dict[str, dict[str, float]] = {
        method: {
            domain: auebc(
                np.asarray(PRIMARY_CHECKPOINTS),
                np.asarray(
                    [domain_curves[method][domain][cap] for cap in PRIMARY_CHECKPOINTS]
                ),
            )
            for domain in config.domain_order
        }
        for method in DETERMINISTIC_METHODS
    }
    random_auebc = {
        seed: {
            domain: auebc(
                np.asarray(PRIMARY_CHECKPOINTS),
                np.asarray(
                    [
                        random_domain_curves[seed][domain][cap]
                        for cap in PRIMARY_CHECKPOINTS
                    ]
                ),
            )
            for domain in config.domain_order
        }
        for seed in random_seeds
    }
    random_median_domain_auebc = {
        domain: float(np.median([random_auebc[seed][domain] for seed in random_seeds]))
        for domain in config.domain_order
    }
    indices = synchronized_bootstrap_indices(
        seed=config.bootstrap_seed,
        resamples=config.bootstrap_resamples,
        domains=6,
    )
    reconstruction_effect = paired_domain_bootstrap(
        [
            domain_auebc["reconstruction_oracle"][domain]
            for domain in config.domain_order
        ],
        [domain_auebc["mechanical_oracle"][domain] for domain in config.domain_order],
        indices=indices,
        effect_id="reconstruction_minus_mechanical_auebc",
    )
    appearance_effect = paired_domain_bootstrap(
        [domain_auebc["appearance_oracle"][domain] for domain in config.domain_order],
        [domain_auebc["mechanical_oracle"][domain] for domain in config.domain_order],
        indices=indices,
        effect_id="appearance_minus_mechanical_auebc",
    )

    equal_curves = {
        method: np.asarray(
            [
                np.mean(
                    [
                        domain_curves[method][domain][cap]
                        for domain in config.domain_order
                    ]
                )
                for cap in PRIMARY_CHECKPOINTS
            ]
        )
        for method in DETERMINISTIC_METHODS
    }
    random_median_curve = np.asarray(
        [
            np.median(
                [
                    np.mean(
                        [
                            random_domain_curves[seed][domain][cap]
                            for domain in config.domain_order
                        ]
                    )
                    for seed in random_seeds
                ]
            )
            for cap in PRIMARY_CHECKPOINTS
        ]
    )
    areas = {
        method: auebc(np.asarray(PRIMARY_CHECKPOINTS), curve)
        for method, curve in equal_curves.items()
    }
    areas["random_median"] = auebc(np.asarray(PRIMARY_CHECKPOINTS), random_median_curve)
    full_domain_mae = {
        domain: anchor_curves["P-B"]["full"][domain][1.0]
        for domain in config.domain_order
    }
    metric_domain_curves: dict[str, dict[str, dict[float, float]]] = {}
    for method in DETERMINISTIC_METHODS:
        metric_domain_curves[method] = {}
        for domain in config.domain_order:
            curve = {
                cap: domain_curves[method][domain][cap] for cap in PRIMARY_CHECKPOINTS
            }
            if method == "uniform":
                curve[0.5] = anchor_curves["P-B"]["uniform"][domain][0.5]
            curve[1.0] = full_domain_mae[domain]
            metric_domain_curves[method][domain] = curve
    metric_domain_curves["random_median"] = {
        domain: {
            **{
                cap: float(
                    np.median(
                        [
                            random_domain_curves[seed][domain][cap]
                            for seed in random_seeds
                        ]
                    )
                )
                for cap in PRIMARY_CHECKPOINTS
            },
            1.0: full_domain_mae[domain],
        }
        for domain in config.domain_order
    }
    metric_equal_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, by_domain in metric_domain_curves.items():
        budgets = np.asarray(sorted(next(iter(by_domain.values()))), dtype=np.float64)
        mae = np.asarray(
            [
                np.mean(
                    [by_domain[domain][float(cap)] for domain in config.domain_order]
                )
                for cap in budgets
            ],
            dtype=np.float64,
        )
        metric_equal_curves[method] = (budgets, mae)
    b5 = {
        method: sufficiency_budget(
            budgets, mae, full_mae=config.full_mae, tolerance=0.05
        )
        for method, (budgets, mae) in metric_equal_curves.items()
    }
    low_index = PRIMARY_CHECKPOINTS.index(config.low_budget)
    gate = evaluate_a3_gate(
        A3GateInputs(
            uniform_low_mae=float(equal_curves["uniform"][low_index]),
            mechanical_low_mae=float(equal_curves["mechanical_oracle"][low_index]),
            uniform_domain_low_mae=tuple(
                domain_curves["uniform"][domain][config.low_budget]
                for domain in config.domain_order
            ),
            mechanical_domain_low_mae=tuple(
                domain_curves["mechanical_oracle"][domain][config.low_budget]
                for domain in config.domain_order
            ),
            reconstruction_minus_mechanical_auebc=reconstruction_effect,
            appearance_minus_mechanical_auebc=appearance_effect,
            uniform_auebc=areas["uniform"],
            random_median_auebc=areas["random_median"],
            mechanical_auebc=areas["mechanical_oracle"],
            uniform_b5=b5["uniform"],
            random_median_b5=b5["random_median"],
            mechanical_b5=b5["mechanical_oracle"],
        )
    )

    domain_rows: list[dict[str, object]] = []
    for domain in config.domain_order:
        for method in (*DETERMINISTIC_METHODS, "random_median"):
            budgets, mae = zip(
                *sorted(metric_domain_curves[method][domain].items()), strict=True
            )
            area = (
                random_median_domain_auebc[domain]
                if method == "random_median"
                else domain_auebc[method][domain]
            )
            low_mae = (
                float(
                    np.median(
                        [
                            random_domain_curves[seed][domain][0.125]
                            for seed in random_seeds
                        ]
                    )
                )
                if method == "random_median"
                else domain_curves[method][domain][0.125]
            )
            domain_rows.append(
                {
                    "dataset_id": domain,
                    "method": method,
                    "auebc": area,
                    "mae_0p125": low_mae,
                    "b_5": sufficiency_budget(
                        np.asarray(budgets),
                        np.asarray(mae),
                        full_mae=config.full_mae,
                        tolerance=0.05,
                    ),
                }
            )
    _write_csv(
        output / "domain_metrics.csv",
        ("dataset_id", "method", "auebc", "mae_0p125", "b_5"),
        domain_rows,
    )
    budget_rows = [
        {
            "method": method,
            "auebc": area,
            "b_2p5": sufficiency_budget(
                metric_equal_curves[method][0],
                metric_equal_curves[method][1],
                full_mae=config.full_mae,
                tolerance=0.025,
            ),
            "b_5": b5[method],
            "b_7p5": sufficiency_budget(
                metric_equal_curves[method][0],
                metric_equal_curves[method][1],
                full_mae=config.full_mae,
                tolerance=0.075,
            ),
            "saving_vs_uniform_b5": simulated_saving(b5[method], b5["uniform"]),
        }
        for method, area in areas.items()
    ]
    _write_csv(
        output / "budget_metrics.csv",
        ("method", "auebc", "b_2p5", "b_5", "b_7p5", "saving_vs_uniform_b5"),
        budget_rows,
    )
    map_rows = _map_similarity(values)
    _write_csv(
        output / "map_similarity.csv",
        (
            "specimen_id",
            "dataset_id",
            "first_method",
            "second_method",
            "pearson",
            "spearman",
            "top10_overlap",
            "rbo_p0_9",
            "candidate_count",
        ),
        map_rows,
    )
    specimen_rows = _specimen_headroom(states)
    _write_csv(
        output / "specimen_metrics.csv",
        (
            "specimen_id",
            "dataset_id",
            "uniform_auebc",
            "mechanical_auebc",
            "auebc_reduction",
            "uniform_mae_0p125",
            "mechanical_mae_0p125",
        ),
        specimen_rows,
    )
    bootstrap_rows = [
        {
            "effect_id": effect.effect_id,
            "point_estimate": effect.point_estimate,
            "lower": effect.lower,
            "upper": effect.upper,
            "improved_domains": effect.improved_domains,
            "domain_effects": json.dumps(effect.domain_effects, separators=(",", ":")),
            "seed": config.bootstrap_seed,
            "resamples": config.bootstrap_resamples,
            "indices_sha256": effect.indices_sha256,
        }
        for effect in (reconstruction_effect, appearance_effect)
    ]
    _write_csv(
        output / "bootstrap.csv",
        (
            "effect_id",
            "point_estimate",
            "lower",
            "upper",
            "improved_domains",
            "domain_effects",
            "seed",
            "resamples",
            "indices_sha256",
        ),
        bootstrap_rows,
    )
    stability_summary = {
        str(row["variant"]): {
            "top1_agreement": float(row["top1_agreement"]),
            "top10_overlap": float(row["top10_overlap"]),
            "spearman": float(row["spearman"]),
            "rbo_p0_9": float(row["rbo_p0_9"]),
        }
        for row in stability.group_by("variant")
        .agg(
            pl.col("top1_agreement").cast(pl.Float64).mean(),
            pl.col("top10_overlap").mean(),
            pl.col("spearman").mean(),
            pl.col("rbo_p0_9").mean(),
        )
        .sort("variant")
        .iter_rows(named=True)
    }
    summary = {
        "status": gate.status,
        "gate": gate.__dict__
        if hasattr(gate, "__dict__")
        else {name: getattr(gate, name) for name in gate.__dataclass_fields__},
        "auebc": areas,
        "b5": b5,
        "bootstrap": {
            "indices_sha256": reconstruction_effect.indices_sha256,
            "reconstruction_minus_mechanical": {
                name: getattr(reconstruction_effect, name)
                for name in reconstruction_effect.__dataclass_fields__
            },
            "appearance_minus_mechanical": {
                name: getattr(appearance_effect, name)
                for name in appearance_effect.__dataclass_fields__
            },
        },
        "state_rows": states.height,
        "trajectory_rows": trajectories.height,
        "oracle_value_rows": values.height,
        "specimen_count": authority.specimen_count,
        "random_seed_count": 100,
        "report_checkpoints": REPORT_CHECKPOINTS,
        "anchors": {"uniform": 0.5, "full": 1.0},
        "stability": stability_summary,
        "interpretation": {
            "oracle_role": "retrospective_upper_bound_not_deployable_policy",
            "physical_scope": "normalized_raster_observation_simulation",
            "excluded_claims": [
                "physical_scanner_pitch",
                "inspection_time_reduction",
                "online_adaptive_scanner",
            ],
        },
    }
    _write_json(output / "summary.json", summary)
    advantage_budgets = [
        (
            cap,
            float(
                (
                    equal_curves["uniform"][index]
                    - equal_curves["mechanical_oracle"][index]
                )
                / equal_curves["uniform"][index]
            ),
        )
        for index, cap in enumerate(PRIMARY_CHECKPOINTS)
        if equal_curves["mechanical_oracle"][index] < equal_curves["uniform"][index]
    ]
    advantage_text = (
        ", ".join(
            f"{100.0 * cap:.3f}% ({100.0 * relative:.2f}% relative MAE)"
            for cap, relative in advantage_budgets
        )
        if advantage_budgets
        else "none of the preregistered 6.25%-25% checkpoints"
    )
    domain_headroom = [
        domain
        for domain in config.domain_order
        if domain_auebc["mechanical_oracle"][domain] < domain_auebc["uniform"][domain]
    ]
    map_rec = _map_summary(map_rows, "mechanical_oracle", "reconstruction_oracle")
    map_app = _map_summary(map_rows, "mechanical_oracle", "appearance_oracle")
    best_specimens = ", ".join(
        f"{row['specimen_id']} ({float(row['auebc_reduction']):.8f})"
        for row in specimen_rows[:5]
    )
    saving_uniform = simulated_saving(b5["mechanical_oracle"], b5["uniform"])
    report = [
        "# MVA A0-A3 Oracle Headroom Report",
        "",
        f"Terminal decision: `{gate.status}`.",
        "",
        "This is a retrospective diagnostic upper bound on the normalized raster observation grid. Ground-truth CAI is used only to label the mechanical oracle. The result does not establish physical scanner pitch, inspection-time reduction, or a deployable acquisition policy.",
        "",
        "## Required Questions",
        "",
        f"1. Mechanical CAI oracle versus uniform: {'yes' if gate.h1_pass else 'no'} under H1; 12.5% relative MAE improvement is {100.0 * gate.h1_relative_improvement:.2f}% with {gate.h1_improved_domains}/6 domains in the same direction.",
        f"2. Mechanical CAI oracle versus reconstruction oracle: {'yes' if gate.h2_pass else 'no'} under the synchronized domain bootstrap; reconstruction-minus-mechanical AUEBC is {reconstruction_effect.point_estimate:.8f} (95% interval [{reconstruction_effect.lower:.8f}, {reconstruction_effect.upper:.8f}]).",
        f"3. Mechanical CAI oracle versus the frozen appearance heuristic: {'yes' if gate.h3_pass else 'no'} under the synchronized domain bootstrap; appearance-minus-mechanical AUEBC is {appearance_effect.point_estimate:.8f} (95% interval [{appearance_effect.lower:.8f}, {appearance_effect.upper:.8f}]).",
        f"4. Budgets with lower mechanical-oracle MAE than uniform: {advantage_text}.",
        f"5. Domain-level AUEBC headroom occurs in {len(domain_headroom)}/6 domains: {', '.join(domain_headroom) if domain_headroom else 'none'}.",
        f"6. Largest specimen-level uniform-minus-mechanical AUEBC reductions: {best_specimens}.",
        f"7. Initial mechanical versus reconstruction maps: mean Pearson {map_rec['pearson']:.4f}, Spearman {map_rec['spearman']:.4f}, top-10% overlap {map_rec['top10_overlap']:.4f}, RBO {map_rec['rbo_p0_9']:.4f}.",
        f"8. Initial mechanical versus appearance maps: mean Pearson {map_app['pearson']:.4f}, Spearman {map_app['spearman']:.4f}, top-10% overlap {map_app['top10_overlap']:.4f}, RBO {map_app['rbo_p0_9']:.4f}; this quantifies whether the mechanical map is a copy rather than assuming it is distinct.",
        f"9. B5 is {_format_budget(b5['mechanical_oracle'])} for the mechanical oracle and {_format_budget(b5['uniform'])} for uniform; simulated measurement reduction versus uniform is {'unavailable' if saving_uniform is None else f'{100.0 * saving_uniform:.2f}%'}.",
        f"10. Policy-learning headroom is {'supported' if gate.h4_pass else 'not supported'} by H4; relative AUEBC headroom is {100.0 * gate.h4_relative_auebc_improvement:.2f}% and B5 saving against the stronger fixed/random reference is {'unavailable' if gate.h4_b5_saving is None else f'{100.0 * gate.h4_b5_saving:.2f}%'}.",
        "",
        "## Gate",
        "",
        f"- H1: {gate.h1_pass}; relative improvement {gate.h1_relative_improvement:.6f}; improved domains {gate.h1_improved_domains}/6.",
        f"- H2: {gate.h2_pass}; bootstrap lower {reconstruction_effect.lower:.8f}.",
        f"- H3: {gate.h3_pass}; bootstrap lower {appearance_effect.lower:.8f}.",
        f"- H4: {gate.h4_pass}; relative AUEBC improvement {gate.h4_relative_auebc_improvement:.6f}; B5 saving {gate.h4_b5_saving}.",
        "",
        "## Initial-Ranking Stability",
        "",
        *(
            f"- {variant}: top-1 {metrics['top1_agreement']:.3f}, top-10% overlap {metrics['top10_overlap']:.3f}, Spearman {metrics['spearman']:.3f}, RBO {metrics['rbo_p0_9']:.3f}."
            for variant, metrics in stability_summary.items()
        ),
        "",
        "The 50% uniform and 100% FULL points are report-only anchors and are excluded from AUEBC. A4-A7 were not implemented or executed in this package.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output


__all__ = ["aggregate_a2"]
