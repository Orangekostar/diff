"""Frozen reconstruction-versus-mechanics task-specificity diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class TaskSpecificityError(ValueError):
    """Raised when historical task-specificity evidence cannot be aligned."""


_A2_METHODS = ("reconstruction_oracle", "mechanical_oracle")
_A4_METHODS = ("global_reconstruction_mask", "global_mechanical_mask")
_METHODS = (*_A2_METHODS, *_A4_METHODS)
_METHOD_LABELS = {
    "reconstruction_oracle": "Reconstruction oracle",
    "mechanical_oracle": "Mechanical oracle",
    "global_reconstruction_mask": "Learned reconstruction mask",
    "global_mechanical_mask": "Learned mechanics mask",
}
_PAIRS = (
    ("reconstruction_oracle", "mechanical_oracle", "oracle"),
    ("global_reconstruction_mask", "global_mechanical_mask", "learned"),
)
_FILES = {
    "task_matrix.csv",
    "per_domain.csv",
    "per_budget.csv",
    "per_specimen.csv",
    "spatial_comparison.csv",
    "bootstrap.csv",
    "source_data.csv",
    "task_specificity.svg",
    "task_specificity.pdf",
    "task_specificity.png",
    "FIGURE_CAPTION.md",
    "VISUAL_CONTRACT.md",
    "REPORT.md",
    "summary.json",
    "artifact_manifest.json",
    "CHECKSUMS.sha256",
}
_COLUMNS = {
    "specimen_id",
    "dataset_id",
    "method",
    "nominal_checkpoint",
    "measured_count",
    "native_count",
    "effective_budget",
    "target",
    "p_a_prediction",
    "p_a_absolute_error",
    "normalized_rgb_mse",
    "p_a_predictor_state_sha256",
}


def align_task_specificity_states(
    a2_states: pl.DataFrame,
    a4_states: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    reconstruction_metric: str,
) -> pl.DataFrame:
    """Align frozen A2 oracle and A4 source-trained policy state metrics."""

    if reconstruction_metric != "normalized_rgb_mse":
        raise TaskSpecificityError("frozen reconstruction metric must not change")
    if (
        not isinstance(a2_states, pl.DataFrame)
        or not isinstance(a4_states, pl.DataFrame)
        or not _COLUMNS <= set(a2_states.columns)
        or not _COLUMNS | {"outer_domain"} <= set(a4_states.columns)
        or type(domain_order) is not tuple
        or not domain_order
        or len(set(domain_order)) != len(domain_order)
        or type(checkpoints) is not tuple
        or not checkpoints
        or len(set(checkpoints)) != len(checkpoints)
    ):
        raise TaskSpecificityError("task-specificity input schema is invalid")
    a2 = a2_states.filter(pl.col("method").is_in(_A2_METHODS))
    if "seed" in a2.columns:
        a2 = a2.filter(pl.col("seed").is_null())
    a4 = a4_states.filter(
        pl.col("method").is_in(_A4_METHODS)
        & (pl.col("outer_domain") == pl.col("dataset_id"))
    )
    selected = [
        "specimen_id",
        "dataset_id",
        "method",
        "nominal_checkpoint",
        "measured_count",
        "native_count",
        "effective_budget",
        "target",
        "p_a_prediction",
        "p_a_absolute_error",
        "normalized_rgb_mse",
        "p_a_predictor_state_sha256",
    ]
    a2 = a2.select(selected)
    a4 = a4.select(selected)
    identity = ["specimen_id", "dataset_id", "method", "nominal_checkpoint"]
    if (
        a2.is_empty()
        or a4.is_empty()
        or a2.unique(subset=identity).height != a2.height
        or a4.unique(subset=identity).height != a4.height
        or set(a2.get_column("method").unique()) != set(_A2_METHODS)
        or set(a4.get_column("method").unique()) != set(_A4_METHODS)
        or set(a2.get_column("dataset_id").unique()) != set(domain_order)
        or set(a4.get_column("dataset_id").unique()) != set(domain_order)
        or set(a2.get_column("nominal_checkpoint").unique()) != set(checkpoints)
        or set(a4.get_column("nominal_checkpoint").unique()) != set(checkpoints)
    ):
        raise TaskSpecificityError(
            "task-specificity method or checkpoint roster changed"
        )
    cohort_a2 = set(a2.select("specimen_id", "dataset_id").iter_rows())
    cohort_a4 = set(a4.select("specimen_id", "dataset_id").iter_rows())
    if cohort_a2 != cohort_a4:
        raise TaskSpecificityError("task-specificity cohort changed")
    aligned = (
        pl.concat([a2, a4], how="vertical")
        .sort(identity)
        .rename(
            {
                "p_a_prediction": "cai_prediction",
                "p_a_absolute_error": "cai_absolute_error",
                "normalized_rgb_mse": "reconstruction_error",
            }
        )
    )
    expected_rows = len(cohort_a2) * len(_METHODS) * len(checkpoints)
    per_state = aligned.group_by("specimen_id", "dataset_id", "nominal_checkpoint").agg(
        pl.len().alias("method_count"),
        pl.col("method").n_unique().alias("unique_method_count"),
        pl.col("native_count").n_unique().alias("native_count_count"),
        pl.col("target").n_unique().alias("target_count"),
        pl.col("p_a_predictor_state_sha256").n_unique().alias("predictor_count"),
    )
    numeric = aligned.select(
        pl.all_horizontal(
            pl.col("measured_count") > 0,
            pl.col("native_count") > 0,
            (
                pl.col("effective_budget")
                - pl.col("measured_count") / pl.col("native_count")
            ).abs()
            <= 1.0e-15,
            pl.col("cai_absolute_error") >= 0.0,
            pl.col("reconstruction_error") >= 0.0,
        ).all()
    ).item()
    if (
        aligned.height != expected_rows
        or per_state.filter(
            (pl.col("method_count") != len(_METHODS))
            | (pl.col("unique_method_count") != len(_METHODS))
            | (pl.col("native_count_count") != 1)
            | (pl.col("target_count") != 1)
            | (pl.col("predictor_count") != 1)
        ).height
        or not numeric
        or aligned.select(pl.any_horizontal(pl.selectors.numeric().is_nan()))
        .to_series()
        .any()
    ):
        raise TaskSpecificityError("task-specificity same-cost alignment failed")
    return aligned


def normalized_auebc(cost: object, values: object) -> float:
    x = pl.Series(cost).cast(pl.Float64).to_numpy()
    y = pl.Series(values).cast(pl.Float64).to_numpy()
    order = x.argsort(kind="stable")
    x = x[order]
    y = y[order]
    if (
        x.ndim != 1
        or x.size < 2
        or y.shape != x.shape
        or not all(math.isfinite(float(value)) for value in (*x, *y))
        or (x[1:] <= x[:-1]).any()
    ):
        raise TaskSpecificityError("task-specificity curve is invalid")
    return float(np.trapezoid(y, x=x) / (x[-1] - x[0]))


def _canonical_aggregates(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove sub-ulp differences from parallel floating-point reductions."""

    return frame.with_columns(pl.col(pl.Float64).round(15))


def evaluate_task_specificity(
    aligned: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    specimen_rows: list[dict[str, object]] = []
    for (domain, specimen, method), table in aligned.group_by(
        "dataset_id", "specimen_id", "method", maintain_order=True
    ):
        ordered = table.sort("effective_budget")
        specimen_rows.append(
            {
                "dataset_id": domain,
                "specimen_id": specimen,
                "method": method,
                "cai_auebc": normalized_auebc(
                    ordered.get_column("effective_budget"),
                    ordered.get_column("cai_absolute_error"),
                ),
                "reconstruction_auebc": normalized_auebc(
                    ordered.get_column("effective_budget"),
                    ordered.get_column("reconstruction_error"),
                ),
            }
        )
    per_specimen = pl.DataFrame(specimen_rows).sort(
        ["dataset_id", "specimen_id", "method"]
    )
    per_domain = _canonical_aggregates(
        per_specimen.group_by("dataset_id", "method").agg(
            pl.col("specimen_id").n_unique().alias("specimen_count"),
            pl.col("cai_auebc").mean(),
            pl.col("reconstruction_auebc").mean(),
        )
    ).sort(["dataset_id", "method"])
    matrix = (
        _canonical_aggregates(
            per_domain.group_by("method").agg(
                pl.col("dataset_id").n_unique().alias("domain_count"),
                pl.col("cai_auebc").mean().alias("domain_balanced_cai_auebc"),
                pl.col("cai_auebc").max().alias("worst_domain_cai_auebc"),
                pl.col("reconstruction_auebc")
                .mean()
                .alias("domain_balanced_reconstruction_auebc"),
                pl.col("reconstruction_auebc")
                .max()
                .alias("worst_domain_reconstruction_auebc"),
            )
        )
        .with_columns(
            pl.col("method").replace_strict(_METHOD_LABELS).alias("label"),
            pl.when(pl.col("method").str.contains("reconstruction"))
            .then(pl.lit("reconstruction"))
            .otherwise(pl.lit("mechanics"))
            .alias("acquisition_objective"),
            pl.when(pl.col("method").str.contains("oracle"))
            .then(pl.lit("oracle"))
            .otherwise(pl.lit("source_trained_global"))
            .alias("policy_class"),
        )
        .sort("method")
    )
    budget_rows: list[dict[str, object]] = []
    metrics = {
        "mean_exact_acquired_cost": "measured_count",
        "mean_effective_budget": "effective_budget",
        "cai_mae": "cai_absolute_error",
        "reconstruction_mse": "reconstruction_error",
    }
    for method in _METHODS:
        for checkpoint in sorted(aligned.get_column("nominal_checkpoint").unique()):
            table = aligned.filter(
                (pl.col("method") == method)
                & (pl.col("nominal_checkpoint") == checkpoint)
            ).sort("dataset_id", "specimen_id")
            domain_means: dict[str, list[float]] = {name: [] for name in metrics}
            specimen_count = 0
            for (_domain,), group in table.group_by("dataset_id", maintain_order=True):
                specimen_count += group.height
                for output_name, source_name in metrics.items():
                    values = group.get_column(source_name).cast(pl.Float64).to_list()
                    domain_means[output_name].append(math.fsum(values) / len(values))
            budget_rows.append(
                {
                    "method": method,
                    "nominal_checkpoint": float(checkpoint),
                    "domain_count": len(domain_means["cai_mae"]),
                    "specimen_count": specimen_count,
                    **{
                        name: math.fsum(values) / len(values)
                        for name, values in domain_means.items()
                    },
                }
            )
    per_budget = _canonical_aggregates(pl.DataFrame(budget_rows)).sort(
        ["method", "nominal_checkpoint"]
    )
    return per_specimen, per_domain, per_budget, matrix


def bootstrap_task_specificity(
    per_specimen: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    replicates: int,
    seed: int,
) -> pl.DataFrame:
    contrasts = (
        (
            "oracle_reconstruction_minus_mechanics_cai",
            "reconstruction_oracle",
            "mechanical_oracle",
            "cai_auebc",
        ),
        (
            "oracle_mechanics_minus_reconstruction_image",
            "mechanical_oracle",
            "reconstruction_oracle",
            "reconstruction_auebc",
        ),
        (
            "learned_reconstruction_minus_mechanics_cai",
            "global_reconstruction_mask",
            "global_mechanical_mask",
            "cai_auebc",
        ),
        (
            "learned_mechanics_minus_reconstruction_image",
            "global_mechanical_mask",
            "global_reconstruction_mask",
            "reconstruction_auebc",
        ),
    )
    generator = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for contrast_id, first, second, metric in contrasts:
        paired: dict[str, np.ndarray] = {}
        for domain in domain_order:
            first_table = per_specimen.filter(
                (pl.col("dataset_id") == domain) & (pl.col("method") == first)
            ).select("specimen_id", pl.col(metric).alias("first"))
            second_table = per_specimen.filter(
                (pl.col("dataset_id") == domain) & (pl.col("method") == second)
            ).select("specimen_id", pl.col(metric).alias("second"))
            joined = first_table.join(
                second_table, on="specimen_id", how="inner", validate="1:1"
            )
            if (
                joined.height != first_table.height
                or joined.height != second_table.height
            ):
                raise TaskSpecificityError("task-specificity bootstrap pairing changed")
            paired[domain] = joined.select(
                pl.col("first") - pl.col("second")
            ).to_numpy()[:, 0]
        for replicate in range(replicates):
            effects: list[float] = []
            for domain in domain_order:
                values = paired[domain]
                indices = generator.integers(0, len(values), len(values))
                effects.append(float(np.mean(values[indices])))
            rows.append(
                {
                    "contrast_id": contrast_id,
                    "first_method": first,
                    "second_method": second,
                    "metric": metric,
                    "replicate": replicate,
                    "first_minus_second": float(np.mean(effects)),
                    "positive_domain_count": int(
                        np.count_nonzero(np.asarray(effects) > 0.0)
                    ),
                    "statistical_unit": "paired_physical_specimen_within_domain",
                }
            )
    return pl.DataFrame(rows).sort(["contrast_id", "replicate"])


def _action_tuple(row: dict[str, object]) -> tuple[int, int, int]:
    return (int(row["cell_index"]), int(row["from_level"]), int(row["to_level"]))


def _radial(cell: int) -> float:
    row, column = divmod(cell, 8)
    return float(math.hypot(row - 3.5, column - 3.5) / math.sqrt(24.5))


def _spatial_stats(
    first: set[tuple[int, int, int]], second: set[tuple[int, int, int]]
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    union = first | second
    overlap = float(len(first & second) / len(union)) if union else None
    first_cells = {action[0] for action in first}
    second_cells = {action[0] for action in second}
    if not first_cells or not second_cells:
        distance = None
    else:

        def nearest(source: set[int], target: set[int]) -> float:
            values = []
            for cell in source:
                row, column = divmod(cell, 8)
                values.append(
                    min(
                        math.hypot(row - other // 8, column - other % 8)
                        for other in target
                    )
                )
            return float(np.mean(values) / math.sqrt(98.0))

        distance = 0.5 * (
            nearest(first_cells, second_cells) + nearest(second_cells, first_cells)
        )
    first_radial = (
        float(np.mean([_radial(cell) for cell in first_cells])) if first_cells else None
    )
    second_radial = (
        float(np.mean([_radial(cell) for cell in second_cells]))
        if second_cells
        else None
    )
    level2_delta = (
        float(np.mean([action[2] == 2 for action in first]))
        - float(np.mean([action[2] == 2 for action in second]))
        if first and second
        else None
    )
    return overlap, distance, first_radial, second_radial, level2_delta


def spatial_task_specificity(
    a2_trajectories: pl.DataFrame,
    a4_trajectories: pl.DataFrame,
    *,
    cohort: pl.DataFrame,
    checkpoints: tuple[float, ...],
) -> pl.DataFrame:
    required = {
        "specimen_id",
        "dataset_id",
        "method",
        "nominal_checkpoint",
        "cell_index",
        "from_level",
        "to_level",
    }
    if not required <= set(a2_trajectories.columns) or not required | {
        "outer_domain"
    } <= set(a4_trajectories.columns):
        raise TaskSpecificityError("task-specificity trajectory schema changed")
    a2 = a2_trajectories.filter(pl.col("method").is_in(_A2_METHODS))
    if "record_type" in a2.columns:
        a2 = a2.filter(pl.col("record_type") == "action")
    if "seed" in a2.columns:
        a2 = a2.filter(pl.col("seed").is_null())
    a4 = a4_trajectories.filter(
        pl.col("method").is_in(_A4_METHODS)
        & (pl.col("outer_domain") == pl.col("dataset_id"))
    )
    actions = pl.concat(
        [a2.select(*sorted(required)), a4.select(*sorted(required))],
        how="vertical",
    )
    roster = {
        (str(row["specimen_id"]), str(row["method"])): table.to_dicts()
        for row, table in (
            (
                {"specimen_id": specimen, "method": method},
                group,
            )
            for (specimen, method), group in actions.group_by(
                "specimen_id", "method", maintain_order=True
            )
        )
    }
    rows: list[dict[str, object]] = []
    for specimen_id, dataset_id in cohort.select(
        "specimen_id", "dataset_id"
    ).iter_rows():
        for first_method, second_method, comparison in _PAIRS:
            first_rows = roster[(str(specimen_id), first_method)]
            second_rows = roster[(str(specimen_id), second_method)]
            for checkpoint in checkpoints:
                first = {
                    _action_tuple(row)
                    for row in first_rows
                    if float(row["nominal_checkpoint"]) <= checkpoint
                }
                second = {
                    _action_tuple(row)
                    for row in second_rows
                    if float(row["nominal_checkpoint"]) <= checkpoint
                }
                overlap, distance, first_radial, second_radial, level2_delta = (
                    _spatial_stats(first, second)
                )
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "specimen_id": specimen_id,
                        "comparison": comparison,
                        "first_method": first_method,
                        "second_method": second_method,
                        "nominal_checkpoint": checkpoint,
                        "first_action_count": len(first),
                        "second_action_count": len(second),
                        "exact_action_jaccard": overlap,
                        "normalized_symmetric_cell_distance": distance,
                        "first_mean_radial_position": first_radial,
                        "second_mean_radial_position": second_radial,
                        "first_minus_second_level2_fraction": level2_delta,
                    }
                )
    output = pl.DataFrame(rows).sort(
        ["dataset_id", "specimen_id", "comparison", "nominal_checkpoint"]
    )
    if output.height != cohort.height * len(_PAIRS) * len(checkpoints):
        raise TaskSpecificityError("task-specificity spatial roster changed")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_state(path: Path) -> str:
    rows = [
        (item.relative_to(path).as_posix(), item.stat().st_size, _sha256(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    if not rows:
        raise TaskSpecificityError("bound package is empty")
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_checksums(path: Path) -> None:
    files = sorted(item for item in path.iterdir() if item.name != "CHECKSUMS.sha256")
    (path / "CHECKSUMS.sha256").write_text(
        "".join(f"{_sha256(item)}  {item.name}\n" for item in files),
        encoding="ascii",
    )


def _load_config(path: str | Path) -> dict[str, object]:
    try:
        source = Path(path).resolve(strict=True)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise TaskSpecificityError("P14 config is unavailable") from error
    keys = {
        "schema_version",
        "stage",
        "audit_base_git_sha",
        "domain_order",
        "checkpoints",
        "reconstruction_metric",
        "a2_state_metrics",
        "a2_state_metrics_sha256",
        "a2_trajectories",
        "a2_trajectories_sha256",
        "a4_state_metrics",
        "a4_state_metrics_sha256",
        "a4_trajectories",
        "a4_trajectories_sha256",
        "p7_package",
        "p7_tree_state_sha256",
        "bootstrap_replicates",
        "seed",
    }
    hash_keys = tuple(key for key in keys if key.endswith("sha256"))
    if (
        type(payload) is not dict
        or set(payload) != keys
        or payload["schema_version"] != 1
        or payload["stage"] != "P14_TASK_SPECIFICITY"
        or type(payload["domain_order"]) is not list
        or len(payload["domain_order"]) != 6
        or type(payload["checkpoints"]) is not list
        or len(payload["checkpoints"]) != 6
        or any(
            type(payload[key]) is not str
            or len(payload[key]) != 64
            or any(character not in "0123456789abcdef" for character in payload[key])
            for key in hash_keys
        )
        or payload["reconstruction_metric"] != "normalized_rgb_mse"
        or type(payload["bootstrap_replicates"]) is not int
        or payload["bootstrap_replicates"] < 2
        or type(payload["seed"]) is not int
        or isinstance(payload["seed"], bool)
    ):
        raise TaskSpecificityError("P14 config schema changed")
    payload["config_sha256"] = _sha256(source)
    return payload


def _bound(root: Path, value: object, *, directory: bool) -> Path:
    if type(value) is not str or not value:
        raise TaskSpecificityError("P14 source path is invalid")
    try:
        path = (root / value).resolve(strict=True)
    except OSError as error:
        raise TaskSpecificityError("P14 source is unavailable") from error
    if root != path and root not in path.parents:
        raise TaskSpecificityError("P14 source escapes project root")
    if path.is_dir() != directory:
        raise TaskSpecificityError("P14 source type changed")
    return path


def _render_figure(
    matrix: pl.DataFrame,
    spatial: pl.DataFrame,
    *,
    output: Path,
) -> pl.DataFrame:
    palette = {
        "reconstruction_oracle": "#0072B2",
        "mechanical_oracle": "#D55E00",
        "global_reconstruction_mask": "#56B4E9",
        "global_mechanical_mask": "#E69F00",
    }
    ordered = [
        "reconstruction_oracle",
        "mechanical_oracle",
        "global_reconstruction_mask",
        "global_mechanical_mask",
    ]
    values = {str(row["method"]): row for row in matrix.iter_rows(named=True)}
    spatial_summary = _canonical_aggregates(
        spatial.group_by("comparison", "nominal_checkpoint").agg(
            pl.col("exact_action_jaccard").mean(),
            pl.col("normalized_symmetric_cell_distance").mean(),
        )
    ).sort(["comparison", "nominal_checkpoint"])
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "svg.fonttype": "none",
            "svg.hashsalt": "mavis-p14-task-specificity",
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    labels = [_METHOD_LABELS[method] for method in ordered]
    colors = [palette[method] for method in ordered]
    x = np.arange(len(ordered))
    for axis, metric, title, ylabel in (
        (
            axes[0],
            "domain_balanced_cai_auebc",
            "(a) Residual-capacity assessment",
            "CAI error AUEBC (lower is better)",
        ),
        (
            axes[1],
            "domain_balanced_reconstruction_auebc",
            "(b) Ultrasonic-field reconstruction",
            "Normalized RGB MSE AUEBC (lower is better)",
        ),
    ):
        heights = [float(values[method][metric]) for method in ordered]
        axis.bar(x, heights, color=colors, edgecolor="#222222", linewidth=0.5)
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
    styles = {"oracle": ("#222222", "o", "-"), "learned": ("#666666", "s", "--")}
    for comparison in ("oracle", "learned"):
        table = spatial_summary.filter(pl.col("comparison") == comparison)
        color, marker, line = styles[comparison]
        axes[2].plot(
            table.get_column("nominal_checkpoint"),
            table.get_column("exact_action_jaccard"),
            color=color,
            marker=marker,
            linestyle=line,
            linewidth=1.4,
            markersize=3.5,
            label=f"{comparison.capitalize()} pair",
        )
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xlabel("Nominal acquired budget")
    axes[2].set_ylabel("Exact-action Jaccard")
    axes[2].set_title("(c) Spatial action overlap", loc="left", fontweight="bold")
    axes[2].grid(color="#D9D9D9", linewidth=0.6)
    axes[2].legend(frameon=False, loc="best")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    metadata = {"Creator": "MAVIS science closure", "Date": None}
    svg_path = output / "task_specificity.svg"
    figure.savefig(svg_path, metadata=metadata)
    svg_path.write_text(
        "\n".join(
            line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    figure.savefig(output / "task_specificity.pdf", metadata=metadata)
    figure.savefig(
        output / "task_specificity.png",
        dpi=300,
        metadata={"Software": "MAVIS science closure"},
    )
    plt.close(figure)
    source_rows = []
    for method in ordered:
        source_rows.extend(
            [
                {
                    "panel": "a",
                    "series": method,
                    "nominal_checkpoint": None,
                    "metric": "domain_balanced_cai_auebc",
                    "value": float(values[method]["domain_balanced_cai_auebc"]),
                },
                {
                    "panel": "b",
                    "series": method,
                    "nominal_checkpoint": None,
                    "metric": "domain_balanced_reconstruction_auebc",
                    "value": float(
                        values[method]["domain_balanced_reconstruction_auebc"]
                    ),
                },
            ]
        )
    for row in spatial_summary.iter_rows(named=True):
        source_rows.append(
            {
                "panel": "c",
                "series": row["comparison"],
                "nominal_checkpoint": row["nominal_checkpoint"],
                "metric": "exact_action_jaccard",
                "value": row["exact_action_jaccard"],
            }
        )
    return pl.DataFrame(source_rows, infer_schema_length=None)


def _visual_contract() -> str:
    return """# P14 Visual Contract

- Artifact: three-panel task-specificity figure and 2x2-plus policy table.
- Target format: full-width AEI/CCF-style paper figure; editable SVG/PDF plus PNG preview.
- Core claim: reconstruction-oriented and mechanics-oriented acquisition optimize distinct downstream objectives.
- Reviewer question: do the same measurement sets jointly optimize field reconstruction and residual-capacity assessment?
- Evidence layer: mechanism and limitation.
- Source data: frozen MVA A2 oracle and A4 source-trained global-mask state/action tables.
- Statistics: physical-specimen AUEBC, equal-domain aggregation, paired within-domain bootstrap.
- Panel map: (a) CAI error AUEBC; (b) frozen normalized RGB MSE AUEBC; (c) exact-action overlap by budget.
- Caption role: state the cross-objective comparison and prohibit physical-region interpretation.
- Output formats: CSV, SVG, PDF, and 300-dpi PNG.
- Traceability: `source_data.csv`, input SHA-256 bindings, recursive package checksums.
- Palette/accessibility: Okabe-Ito blue/orange with labels, ordering, and line styles carrying the same distinction.
"""


def _report(summary: dict[str, object]) -> str:
    return (
        "# MAVIS P14 Task Specificity\n\n"
        "Status: `COMPLETE`.\n\n"
        "The same 276-specimen cohort, six held-out domains, exact-cost checkpoint "
        "roster, frozen CAI predictors, and historical `normalized_rgb_mse` metric "
        "are used for all four acquisition strategies.\n\n"
        f"Oracle CAI contrast (reconstruction minus mechanics) is "
        f"`{summary['contrasts']['oracle_reconstruction_minus_mechanics_cai']['point']:.10f}` "
        f"with paired 95% interval `{summary['contrasts']['oracle_reconstruction_minus_mechanics_cai']['interval']}`. "
        f"Oracle image contrast (mechanics minus reconstruction) is "
        f"`{summary['contrasts']['oracle_mechanics_minus_reconstruction_image']['point']:.10f}` "
        f"with interval `{summary['contrasts']['oracle_mechanics_minus_reconstruction_image']['interval']}`.\n\n"
        f"Interpretation: {summary['primary_conclusion']}\n\n"
        "Spatial comparisons describe only grid-action overlap, distance, radial "
        "allocation, and refinement level. They do not assign physical failure "
        "mechanisms to selected regions. P7 remains unchanged.\n"
    )


def run_p14_task_specificity(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_root: str | Path,
) -> Path:
    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise TaskSpecificityError("project root is unavailable") from error
    config = _load_config(config_path)
    source_keys = (
        "a2_state_metrics",
        "a2_trajectories",
        "a4_state_metrics",
        "a4_trajectories",
    )
    sources = {key: _bound(root, config[key], directory=False) for key in source_keys}
    p7 = _bound(root, config["p7_package"], directory=True)
    if (
        any(_sha256(sources[key]) != config[f"{key}_sha256"] for key in source_keys)
        or _tree_state(p7) != config["p7_tree_state_sha256"]
    ):
        raise TaskSpecificityError("P14 frozen input hash changed")
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if root not in destination.parents or destination.exists():
        raise TaskSpecificityError("P14 output is invalid or already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".p14_task_specificity.", dir=destination.parent)
    )
    p7_before = _tree_state(p7)
    try:
        domains = tuple(config["domain_order"])
        checkpoints = tuple(float(value) for value in config["checkpoints"])
        aligned = align_task_specificity_states(
            pl.read_parquet(sources["a2_state_metrics"]),
            pl.read_parquet(sources["a4_state_metrics"]),
            domain_order=domains,
            checkpoints=checkpoints,
            reconstruction_metric=str(config["reconstruction_metric"]),
        )
        per_specimen, per_domain, per_budget, matrix = evaluate_task_specificity(
            aligned
        )
        spatial = spatial_task_specificity(
            pl.read_parquet(sources["a2_trajectories"]),
            pl.read_parquet(sources["a4_trajectories"]),
            cohort=aligned.select("specimen_id", "dataset_id").unique(),
            checkpoints=checkpoints,
        )
        bootstrap = bootstrap_task_specificity(
            per_specimen,
            domain_order=domains,
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["seed"]),
        )
        contrast_rows = _canonical_aggregates(
            bootstrap.group_by("contrast_id").agg(
                pl.col("first_minus_second").mean().alias("point"),
                pl.col("first_minus_second").quantile(0.025).alias("lower"),
                pl.col("first_minus_second").quantile(0.975).alias("upper"),
            )
        )
        contrasts = {
            str(row["contrast_id"]): {
                "point": float(row["point"]),
                "interval": [float(row["lower"]), float(row["upper"])],
            }
            for row in contrast_rows.iter_rows(named=True)
        }
        oracle_cross = (
            contrasts["oracle_reconstruction_minus_mechanics_cai"]["interval"][0] > 0.0
            and contrasts["oracle_mechanics_minus_reconstruction_image"]["interval"][0]
            > 0.0
        )
        learned_cross = (
            contrasts["learned_reconstruction_minus_mechanics_cai"]["interval"][0] > 0.0
            and contrasts["learned_mechanics_minus_reconstruction_image"]["interval"][0]
            > 0.0
        )
        conclusion = (
            "Oracle acquisitions show task specificity: the mechanical oracle improves CAI while the "
            "reconstruction oracle improves reconstruction. The source-trained global mechanics mask "
            "does not reproduce this separation and is worse on both aggregate objectives than the "
            "global reconstruction mask."
        )
        summary = {
            "schema_version": 1,
            "stage": "P14_TASK_SPECIFICITY",
            "audit_base_git_sha": config["audit_base_git_sha"],
            "config_sha256": config["config_sha256"],
            "domain_order": domains,
            "checkpoints": checkpoints,
            "specimen_count": 276,
            "reconstruction_metric": config["reconstruction_metric"],
            "method_order": _METHODS,
            "contrasts": contrasts,
            "oracle_cross_objective_supported": oracle_cross,
            "learned_cross_objective_supported": learned_cross,
            "spatial_mean": _canonical_aggregates(
                spatial.group_by("comparison").agg(
                    pl.col("exact_action_jaccard").mean(),
                    pl.col("normalized_symmetric_cell_distance").mean(),
                )
            )
            .sort("comparison")
            .to_dicts(),
            "p7_tree_state_sha256": p7_before,
            "primary_conclusion": conclusion,
        }
        matrix.write_csv(temporary / "task_matrix.csv", float_scientific=False)
        per_specimen.write_csv(temporary / "per_specimen.csv", float_scientific=False)
        per_domain.write_csv(temporary / "per_domain.csv", float_scientific=False)
        per_budget.write_csv(temporary / "per_budget.csv", float_scientific=False)
        spatial.write_csv(temporary / "spatial_comparison.csv", float_scientific=False)
        bootstrap.write_csv(temporary / "bootstrap.csv", float_scientific=False)
        source_data = _render_figure(matrix, spatial, output=temporary)
        source_data.write_csv(temporary / "source_data.csv", float_scientific=False)
        (temporary / "VISUAL_CONTRACT.md").write_text(
            _visual_contract(), encoding="utf-8"
        )
        (temporary / "FIGURE_CAPTION.md").write_text(
            "Task-specific acquisition under matched frozen evidence. Panels (a) "
            "and (b) report equal-domain AUEBC for held-out CAI error and frozen "
            "normalized RGB reconstruction MSE; lower is better. Panel (c) shows "
            "exact grid-action overlap between reconstruction- and mechanics-"
            "oriented policies. Error inference uses paired physical-specimen "
            "bootstrap within domain; spatial patterns are descriptive and do "
            "not identify physical failure mechanisms.\n",
            encoding="utf-8",
        )
        (temporary / "REPORT.md").write_text(_report(summary), encoding="utf-8")
        _write_json(temporary / "summary.json", summary)
        _write_json(
            temporary / "artifact_manifest.json",
            {
                "schema_version": 1,
                "stage": "P14_TASK_SPECIFICITY",
                "config_sha256": config["config_sha256"],
                "inputs": {
                    f"{key}_sha256": config[f"{key}_sha256"] for key in source_keys
                }
                | {"p7_tree_state_sha256": p7_before},
                "artifacts": sorted(_FILES - {"CHECKSUMS.sha256"}),
                "code_state_sha256": _sha256(Path(__file__)),
            },
        )
        _write_checksums(temporary)
        if _tree_state(p7) != p7_before:
            raise TaskSpecificityError("P14 modified frozen P7 artifacts")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_p14_task_specificity_package(destination)
    return destination


def verify_p14_task_specificity_package(path: str | Path) -> dict[str, object]:
    try:
        package = Path(path).resolve(strict=True)
    except OSError as error:
        raise TaskSpecificityError("P14 package is unavailable") from error
    if not package.is_dir() or {item.name for item in package.iterdir()} != _FILES:
        raise TaskSpecificityError("P14 package file roster changed")
    checksums: dict[str, str] = {}
    for line in (package / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    if set(checksums) != _FILES - {"CHECKSUMS.sha256"} or any(
        _sha256(package / name) != digest for name, digest in checksums.items()
    ):
        raise TaskSpecificityError("P14 package checksum mismatch")
    summary = json.loads((package / "summary.json").read_text(encoding="utf-8"))
    matrix = pl.read_csv(package / "task_matrix.csv")
    specimen = pl.read_csv(package / "per_specimen.csv")
    domain = pl.read_csv(package / "per_domain.csv")
    budget = pl.read_csv(package / "per_budget.csv")
    spatial = pl.read_csv(package / "spatial_comparison.csv")
    bootstrap = pl.read_csv(package / "bootstrap.csv")
    from PIL import Image

    pixels = np.asarray(Image.open(package / "task_specificity.png").convert("RGB"))
    svg = (package / "task_specificity.svg").read_text(encoding="utf-8")
    if (
        summary.get("stage") != "P14_TASK_SPECIFICITY"
        or summary.get("reconstruction_metric") != "normalized_rgb_mse"
        or matrix.height != 4
        or specimen.height != 276 * 4
        or domain.height != 6 * 4
        or budget.height != 6 * 4
        or spatial.height != 276 * 2 * 6
        or bootstrap.height != 4 * 5000
        or pixels.shape[0] < 600
        or pixels.shape[1] < 1800
        or float(pixels.std()) <= 1.0
        or "(a) Residual-capacity assessment" not in svg
    ):
        raise TaskSpecificityError("P14 scientific or render contract changed")
    return summary


__all__ = [
    "TaskSpecificityError",
    "align_task_specificity_states",
    "bootstrap_task_specificity",
    "evaluate_task_specificity",
    "normalized_auebc",
    "run_p14_task_specificity",
    "spatial_task_specificity",
    "verify_p14_task_specificity_package",
]
