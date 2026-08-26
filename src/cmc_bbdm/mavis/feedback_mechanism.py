"""P16 feedback-benefit diagnostics linked to conditional-value evolution."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.stats import spearmanr


class FeedbackMechanismError(ValueError):
    """Raised when the frozen P4/P9 feedback diagnostic contract changes."""


_FEEDBACK = "mavis_full"
_NO_FEEDBACK = "mavis_no_feedback"
_SOURCES = ("real", "teacher")
_STRATIFIERS = ("best_action_turnover", "mean_absolute_value_shift")
_STRATA = ("low", "middle", "high")
_FILES = {
    "CHECKSUMS.sha256",
    "REPORT.md",
    "artifact_manifest.json",
    "association_summary.csv",
    "bootstrap.csv",
    "domain_associations.csv",
    "domain_effects.csv",
    "feedback_effects.csv",
    "quantile_thresholds.csv",
    "specimen_auebc_effects.csv",
    "specimen_diagnostics.csv",
    "state_diagnostics.parquet",
    "stratum_effects.csv",
    "summary.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_state(root: Path) -> str:
    rows = [
        (path.relative_to(root).as_posix(), path.stat().st_size, _sha256(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    if not rows:
        raise FeedbackMechanismError("bound package is empty")
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _canonical(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col(pl.Float64).round(15))


def align_feedback_effect(curves: pl.DataFrame) -> pl.DataFrame:
    """Align frozen feedback/no-feedback curves with a positive-is-helpful sign."""

    identity = ["outer_domain", "specimen_id", "nominal_checkpoint"]
    required = set(identity) | {
        "method",
        "mean_exact_acquired_cost",
        "mean_effective_budget",
        "cai_mae",
        "trajectory_count",
    }
    if curves.is_empty() or not required <= set(curves.columns):
        raise FeedbackMechanismError("feedback curve schema changed")
    selected = curves.filter(pl.col("method").is_in([_FEEDBACK, _NO_FEEDBACK]))
    if (
        set(selected.get_column("method").unique()) != {_FEEDBACK, _NO_FEEDBACK}
        or selected.unique(subset=identity + ["method"]).height != selected.height
    ):
        raise FeedbackMechanismError("feedback method roster changed")

    def side(method: str, prefix: str) -> pl.DataFrame:
        return selected.filter(pl.col("method") == method).select(
            *identity,
            pl.col("mean_exact_acquired_cost").alias(f"{prefix}_exact_cost"),
            pl.col("mean_effective_budget").alias(f"{prefix}_effective_budget"),
            pl.col("cai_mae").alias(f"{prefix}_cai_error"),
            pl.col("trajectory_count").alias(f"{prefix}_trajectory_count"),
        )

    aligned = side(_FEEDBACK, "feedback").join(
        side(_NO_FEEDBACK, "no_feedback"), on=identity, how="inner"
    )
    if aligned.height * 2 != selected.height:
        raise FeedbackMechanismError("feedback curve alignment changed")
    aligned = _canonical(aligned)
    return aligned.with_columns(
        (pl.col("no_feedback_cai_error") - pl.col("feedback_cai_error"))
        .round(15)
        .alias("feedback_benefit"),
        (pl.col("no_feedback_exact_cost") - pl.col("feedback_exact_cost"))
        .round(15)
        .alias("exact_cost_difference"),
    ).sort(identity)


def assign_outcome_blind_tertiles(
    frame: pl.DataFrame,
    *,
    metric: str,
    group_columns: tuple[str, ...] = ("outer_domain",),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Assign stable equal-count tertiles without reading the outcome column."""

    required = set(group_columns) | {"specimen_id", metric}
    if frame.is_empty() or not required <= set(frame.columns):
        raise FeedbackMechanismError("feedback stratification schema changed")
    assigned = []
    thresholds = []
    labels = np.asarray(_STRATA, dtype=object)
    for key, group in frame.group_by(*group_columns, maintain_order=True):
        ordered = group.sort(metric, "specimen_id")
        count = ordered.height
        bins = np.minimum(2, 3 * np.arange(count, dtype=np.int64) // count)
        strata = labels[bins].tolist()
        output = ordered.with_columns(pl.Series("stratum", strata))
        assigned.append(output)
        values = ordered.get_column(metric).cast(pl.Float64).to_numpy()
        low_count = int(np.count_nonzero(bins == 0))
        middle_count = int(np.count_nonzero(bins == 1))
        high_start = low_count + middle_count
        key_values = key if isinstance(key, tuple) else (key,)
        thresholds.append(
            {
                **dict(zip(group_columns, key_values, strict=True)),
                "metric": metric,
                "specimen_count": count,
                "low_count": low_count,
                "middle_count": middle_count,
                "high_count": int(np.count_nonzero(bins == 2)),
                "low_upper": float(values[low_count - 1]),
                "middle_lower": float(values[low_count]),
                "middle_upper": float(values[high_start - 1]),
                "high_lower": float(values[high_start]),
                "low_middle_tie": bool(values[low_count - 1] == values[low_count]),
                "middle_high_tie": bool(values[high_start - 1] == values[high_start]),
                "tie_break": "specimen_id",
            }
        )
    return pl.concat(assigned, how="vertical"), _canonical(pl.DataFrame(thresholds))


def _align_auebc(frame: pl.DataFrame) -> pl.DataFrame:
    identity = ["outer_domain", "specimen_id"]
    selected = frame.filter(pl.col("method").is_in([_FEEDBACK, _NO_FEEDBACK]))
    feedback = selected.filter(pl.col("method") == _FEEDBACK).select(
        *identity, pl.col("cai_auebc").alias("feedback_cai_auebc")
    )
    no_feedback = selected.filter(pl.col("method") == _NO_FEEDBACK).select(
        *identity, pl.col("cai_auebc").alias("no_feedback_cai_auebc")
    )
    aligned = feedback.join(no_feedback, on=identity, how="inner").with_columns(
        (pl.col("no_feedback_cai_auebc") - pl.col("feedback_cai_auebc")).alias(
            "feedback_benefit"
        )
    )
    if aligned.height != 276 or selected.height != 552:
        raise FeedbackMechanismError("feedback AUEBC roster changed")
    return _canonical(aligned).sort(identity)


def _domain_balanced(frame: pl.DataFrame, metric: str) -> float:
    means = []
    for (_domain,), group in frame.sort("outer_domain", "specimen_id").group_by(
        "outer_domain", maintain_order=True
    ):
        values = group.get_column(metric).cast(pl.Float64).to_list()
        means.append(math.fsum(values) / len(values))
    if len(means) != 6:
        raise FeedbackMechanismError("feedback domain roster changed")
    return math.fsum(means) / len(means)


def _domain_effects(
    curve_effects: pl.DataFrame, auebc_effects: pl.DataFrame
) -> pl.DataFrame:
    rows = []
    for checkpoint in sorted(curve_effects.get_column("nominal_checkpoint").unique()):
        table = curve_effects.filter(pl.col("nominal_checkpoint") == checkpoint)
        for (domain,), group in table.group_by("outer_domain", maintain_order=True):
            rows.append(
                {
                    "level": "checkpoint",
                    "outer_domain": domain,
                    "nominal_checkpoint": float(checkpoint),
                    "specimen_count": group.height,
                    "mean_feedback_benefit": float(
                        group.get_column("feedback_benefit").mean()
                    ),
                    "mean_exact_cost_difference": float(
                        group.get_column("exact_cost_difference").mean()
                    ),
                }
            )
        rows.append(
            {
                "level": "checkpoint",
                "outer_domain": "__equal_domain__",
                "nominal_checkpoint": float(checkpoint),
                "specimen_count": 276,
                "mean_feedback_benefit": _domain_balanced(table, "feedback_benefit"),
                "mean_exact_cost_difference": _domain_balanced(
                    table, "exact_cost_difference"
                ),
            }
        )
    for (domain,), group in auebc_effects.group_by("outer_domain", maintain_order=True):
        rows.append(
            {
                "level": "auebc",
                "outer_domain": domain,
                "nominal_checkpoint": None,
                "specimen_count": group.height,
                "mean_feedback_benefit": float(
                    group.get_column("feedback_benefit").mean()
                ),
                "mean_exact_cost_difference": None,
            }
        )
    rows.append(
        {
            "level": "auebc",
            "outer_domain": "__equal_domain__",
            "nominal_checkpoint": None,
            "specimen_count": 276,
            "mean_feedback_benefit": _domain_balanced(
                auebc_effects, "feedback_benefit"
            ),
            "mean_exact_cost_difference": None,
        }
    )
    return _canonical(pl.DataFrame(rows)).sort(
        "level", "nominal_checkpoint", "outer_domain"
    )


def _diagnostics(
    p9: pl.DataFrame,
    curve_effects: pl.DataFrame,
    auebc_effects: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    checkpoints = sorted(curve_effects.get_column("nominal_checkpoint").unique())
    state_base = p9.filter(
        pl.col("value_source").is_in(_SOURCES)
        & pl.col("current_checkpoint").is_in(checkpoints)
    ).join(
        curve_effects.select(
            "outer_domain",
            "specimen_id",
            pl.col("nominal_checkpoint").alias("current_checkpoint"),
            "feedback_benefit",
        ),
        on=["outer_domain", "specimen_id", "current_checkpoint"],
        how="inner",
    )
    if state_base.height != 276 * len(_SOURCES) * len(checkpoints):
        raise FeedbackMechanismError("P9/P4 state alignment changed")
    specimen_base = (
        state_base.group_by("outer_domain", "specimen_id", "value_source")
        .agg(
            pl.col("best_action_turnover").mean(),
            pl.col("mean_absolute_value_shift").mean(),
        )
        .join(auebc_effects, on=["outer_domain", "specimen_id"], how="inner")
    )
    state_outputs = []
    specimen_outputs = []
    threshold_outputs = []
    for source in _SOURCES:
        for stratifier in _STRATIFIERS:
            specimen = specimen_base.filter(pl.col("value_source") == source).rename(
                {stratifier: "mechanism_metric"}
            )
            assigned, thresholds = assign_outcome_blind_tertiles(
                specimen, metric="mechanism_metric"
            )
            specimen_outputs.append(
                assigned.with_columns(
                    pl.lit("specimen_auebc").alias("level"),
                    pl.lit(source).alias("mechanism_source"),
                    pl.lit(stratifier).alias("stratifier"),
                    pl.lit(None, dtype=pl.Float64).alias("current_checkpoint"),
                ).select(
                    "level",
                    "outer_domain",
                    "specimen_id",
                    "current_checkpoint",
                    "mechanism_source",
                    "stratifier",
                    "mechanism_metric",
                    "stratum",
                    "feedback_benefit",
                )
            )
            threshold_outputs.append(
                thresholds.with_columns(
                    pl.lit("specimen_auebc").alias("level"),
                    pl.lit(source).alias("mechanism_source"),
                    pl.lit(stratifier).alias("stratifier"),
                    pl.lit(None, dtype=pl.Float64).alias("current_checkpoint"),
                )
            )
            for checkpoint in checkpoints:
                state = state_base.filter(
                    (pl.col("value_source") == source)
                    & (pl.col("current_checkpoint") == checkpoint)
                ).rename({stratifier: "mechanism_metric"})
                assigned, thresholds = assign_outcome_blind_tertiles(
                    state, metric="mechanism_metric"
                )
                state_outputs.append(
                    assigned.with_columns(
                        pl.lit("state_checkpoint").alias("level"),
                        pl.lit(source).alias("mechanism_source"),
                        pl.lit(stratifier).alias("stratifier"),
                    ).select(
                        "level",
                        "outer_domain",
                        "specimen_id",
                        "current_checkpoint",
                        "mechanism_source",
                        "stratifier",
                        "mechanism_metric",
                        "stratum",
                        "feedback_benefit",
                    )
                )
                threshold_outputs.append(
                    thresholds.with_columns(
                        pl.lit("state_checkpoint").alias("level"),
                        pl.lit(source).alias("mechanism_source"),
                        pl.lit(stratifier).alias("stratifier"),
                        pl.lit(float(checkpoint)).alias("current_checkpoint"),
                    )
                )
    state = _canonical(pl.concat(state_outputs, how="vertical")).sort(
        "mechanism_source",
        "stratifier",
        "current_checkpoint",
        "outer_domain",
        "specimen_id",
    )
    specimen = _canonical(pl.concat(specimen_outputs, how="vertical")).sort(
        "mechanism_source", "stratifier", "outer_domain", "specimen_id"
    )
    thresholds = _canonical(pl.concat(threshold_outputs, how="diagonal")).sort(
        "level",
        "mechanism_source",
        "stratifier",
        "current_checkpoint",
        "outer_domain",
    )
    return state, specimen, thresholds


def _stratum_effects(state: pl.DataFrame, specimen: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for frame in (specimen, state):
        keys = ["level", "mechanism_source", "stratifier", "current_checkpoint"]
        for key, subject in frame.group_by(*keys, maintain_order=True):
            key_values = dict(zip(keys, key, strict=True))
            domain_rows = []
            for (domain, stratum), group in subject.group_by(
                "outer_domain", "stratum", maintain_order=True
            ):
                row = {
                    **key_values,
                    "outer_domain": domain,
                    "stratum": stratum,
                    "specimen_count": group.height,
                    "mean_feedback_benefit": float(
                        group.get_column("feedback_benefit").mean()
                    ),
                }
                rows.append(row)
                domain_rows.append(row)
            for stratum in _STRATA:
                values = [
                    row["mean_feedback_benefit"]
                    for row in domain_rows
                    if row["stratum"] == stratum
                ]
                if len(values) != 6:
                    raise FeedbackMechanismError("feedback stratum domain changed")
                rows.append(
                    {
                        **key_values,
                        "outer_domain": "__equal_domain__",
                        "stratum": stratum,
                        "specimen_count": sum(
                            row["specimen_count"]
                            for row in domain_rows
                            if row["stratum"] == stratum
                        ),
                        "mean_feedback_benefit": math.fsum(values) / 6,
                    }
                )
    return _canonical(pl.DataFrame(rows)).sort(
        "level",
        "mechanism_source",
        "stratifier",
        "current_checkpoint",
        "outer_domain",
        "stratum",
    )


def _associations(
    state: pl.DataFrame, specimen: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    domain_rows = []
    summary_rows = []
    keys = ["level", "mechanism_source", "stratifier", "current_checkpoint"]
    for frame in (specimen, state):
        for key, subject in frame.group_by(*keys, maintain_order=True):
            key_values = dict(zip(keys, key, strict=True))
            correlations = []
            for (domain,), group in subject.group_by(
                "outer_domain", maintain_order=True
            ):
                mechanism = group.get_column("mechanism_metric").to_numpy()
                benefit = group.get_column("feedback_benefit").to_numpy()
                if len(np.unique(mechanism)) < 2 or len(np.unique(benefit)) < 2:
                    correlation = 0.0
                else:
                    correlation = float(spearmanr(mechanism, benefit).statistic)
                    if not math.isfinite(correlation):
                        correlation = 0.0
                correlations.append(correlation)
                domain_rows.append(
                    {
                        **key_values,
                        "outer_domain": domain,
                        "specimen_count": group.height,
                        "spearman": correlation,
                    }
                )
            summary_rows.append(
                {
                    **key_values,
                    "domain_count": len(correlations),
                    "mean_domain_spearman": math.fsum(correlations) / len(correlations),
                    "positive_domain_count": sum(value > 0.0 for value in correlations),
                    "minimum_domain_spearman": min(correlations),
                    "maximum_domain_spearman": max(correlations),
                }
            )
    return _canonical(pl.DataFrame(domain_rows)).sort(
        *keys, "outer_domain"
    ), _canonical(pl.DataFrame(summary_rows)).sort(*keys)


def _paired_bootstrap(frame: pl.DataFrame, *, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.zeros(replicates, dtype=np.float64)
    for (_domain,), group in frame.sort("outer_domain", "specimen_id").group_by(
        "outer_domain", maintain_order=True
    ):
        values = group.get_column("feedback_benefit").to_numpy()
        indices = rng.integers(0, len(values), size=(replicates, len(values)))
        output += values[indices].mean(axis=1)
    return output / 6.0


def _stratum_bootstrap(
    frame: pl.DataFrame, *, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    high = np.zeros(replicates, dtype=np.float64)
    low = np.zeros(replicates, dtype=np.float64)
    for (_domain,), group in frame.sort("outer_domain", "specimen_id").group_by(
        "outer_domain", maintain_order=True
    ):
        for label, accumulator in (("high", high), ("low", low)):
            values = (
                group.filter(pl.col("stratum") == label)
                .get_column("feedback_benefit")
                .to_numpy()
            )
            if not len(values):
                raise FeedbackMechanismError("feedback bootstrap stratum is empty")
            indices = rng.integers(0, len(values), size=(replicates, len(values)))
            accumulator += values[indices].mean(axis=1)
    return high / 6.0, (high - low) / 6.0


def _bootstrap(
    curve_effects: pl.DataFrame,
    auebc_effects: pl.DataFrame,
    state: pl.DataFrame,
    specimen: pl.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pl.DataFrame:
    rows = []
    subjects: list[tuple[dict[str, object], np.ndarray]] = []
    subjects.append(
        (
            {
                "analysis": "overall_feedback",
                "level": "auebc",
                "mechanism_source": None,
                "stratifier": None,
                "current_checkpoint": None,
                "metric": "feedback_benefit",
            },
            _paired_bootstrap(auebc_effects, replicates=replicates, seed=seed),
        )
    )
    for offset, checkpoint in enumerate(
        sorted(curve_effects.get_column("nominal_checkpoint").unique()), start=1
    ):
        subjects.append(
            (
                {
                    "analysis": "overall_feedback",
                    "level": "checkpoint",
                    "mechanism_source": None,
                    "stratifier": None,
                    "current_checkpoint": float(checkpoint),
                    "metric": "feedback_benefit",
                },
                _paired_bootstrap(
                    curve_effects.filter(pl.col("nominal_checkpoint") == checkpoint),
                    replicates=replicates,
                    seed=seed + offset,
                ),
            )
        )
    offset = 10
    keys = ["level", "mechanism_source", "stratifier", "current_checkpoint"]
    for frame in (specimen, state):
        for key, subject in frame.group_by(*keys, maintain_order=True):
            key_values = dict(zip(keys, key, strict=True))
            high, difference = _stratum_bootstrap(
                subject, replicates=replicates, seed=seed + offset
            )
            offset += 1
            for metric, values in (
                ("high_stratum_feedback_benefit", high),
                ("high_minus_low_feedback_benefit", difference),
            ):
                subjects.append(
                    (
                        {
                            "analysis": "conditional_value_strata",
                            **key_values,
                            "metric": metric,
                        },
                        values,
                    )
                )
    for metadata, values in subjects:
        rows.extend(
            {**metadata, "replicate": index, "value": float(value)}
            for index, value in enumerate(values)
        )
    return _canonical(pl.DataFrame(rows, infer_schema_length=None)).sort(
        "analysis",
        "level",
        "mechanism_source",
        "stratifier",
        "current_checkpoint",
        "metric",
        "replicate",
    )


def _load_config(path: str | Path) -> dict[str, object]:
    config_path = Path(path).resolve(strict=True)
    raw = config_path.read_bytes()
    value = yaml.safe_load(raw)
    if (
        not isinstance(value, dict)
        or value.get("stage") != "P16_FEEDBACK_MECHANISM"
        or value.get("feedback_sign") != "no_feedback_error_minus_feedback_error"
        or value.get("value_sources") != list(_SOURCES)
        or value.get("stratifiers") != list(_STRATIFIERS)
        or value.get("strata") != list(_STRATA)
    ):
        raise FeedbackMechanismError("P16 config changed")
    value["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def _bound(root: Path, value: object, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise FeedbackMechanismError("P16 source path is invalid")
    try:
        path = (root / value).resolve(strict=True)
    except OSError as error:
        raise FeedbackMechanismError("P16 source is unavailable") from error
    if root != path and root not in path.parents:
        raise FeedbackMechanismError("P16 source escapes project root")
    if path.is_dir() != directory:
        raise FeedbackMechanismError("P16 source type changed")
    return path


def _metadata(
    output: Path, *, config: dict[str, object], source_hashes: dict[str, str]
) -> None:
    payload = sorted(_FILES - {"CHECKSUMS.sha256", "artifact_manifest.json"})
    manifest = {
        "schema_version": 1,
        "stage": "P16_FEEDBACK_MECHANISM",
        "audit_base_git_sha": config["audit_base_git_sha"],
        "config_sha256": config["config_sha256"],
        "source_sha256": source_hashes,
        "artifacts": [
            {
                "path": name,
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in payload
        ],
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{_sha256(output / name)}  {name}\n"
            for name in sorted(_FILES - {"CHECKSUMS.sha256"})
        ),
        encoding="ascii",
    )


def run_p16_feedback_mechanism(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_root: str | Path,
) -> Path:
    root = Path(project_root).resolve(strict=True)
    config = _load_config(config_path)
    domains = tuple(config["domain_order"])
    checkpoints = tuple(float(value) for value in config["checkpoints"])
    if len(domains) != 6 or len(set(domains)) != 6 or len(checkpoints) != 4:
        raise FeedbackMechanismError("P16 domain or checkpoint roster changed")
    source_names = (
        "per_specimen_auebc",
        "per_specimen_curves",
        "p4_summary",
        "p9_specimen_metrics",
        "p9_summary",
    )
    sources = {name: _bound(root, config[name]) for name in source_names}
    if any(_sha256(path) != config[f"{name}_sha256"] for name, path in sources.items()):
        raise FeedbackMechanismError("P16 frozen source hash changed")
    p7 = _bound(root, config["p7_package"], directory=True)
    if _tree_state(p7) != config["p7_tree_state_sha256"]:
        raise FeedbackMechanismError("P16 P7 state changed")
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    if root not in destination.parents or destination.exists():
        raise FeedbackMechanismError("P16 output is invalid or already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p16_feedback.", dir=destination.parent))
    p7_before = _tree_state(p7)
    try:
        curve_effects = align_feedback_effect(
            pl.read_csv(sources["per_specimen_curves"])
        )
        auebc_effects = _align_auebc(pl.read_csv(sources["per_specimen_auebc"]))
        curve_effects = curve_effects.filter(
            pl.col("nominal_checkpoint").is_in([0.03125, *checkpoints, 0.25])
        )
        if curve_effects.height != 276 * 6:
            raise FeedbackMechanismError("P16 feedback effect roster changed")
        domain_effects = _domain_effects(curve_effects, auebc_effects)
        state, specimen, thresholds = _diagnostics(
            pl.read_parquet(sources["p9_specimen_metrics"]),
            curve_effects.filter(pl.col("nominal_checkpoint").is_in(checkpoints)),
            auebc_effects,
        )
        strata = _stratum_effects(state, specimen)
        domain_associations, association_summary = _associations(state, specimen)
        bootstrap = _bootstrap(
            curve_effects,
            auebc_effects,
            state,
            specimen,
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["seed"]),
        )
        intervals = {}
        keys = [
            "analysis",
            "level",
            "mechanism_source",
            "stratifier",
            "current_checkpoint",
            "metric",
        ]
        for key, group in bootstrap.group_by(*keys, maintain_order=True):
            values = group.get_column("value").to_numpy()
            identifier = "/".join("none" if item is None else str(item) for item in key)
            intervals[identifier] = {
                "bootstrap_mean": round(float(values.mean()), 15),
                "interval": [
                    round(float(np.quantile(values, 0.025)), 15),
                    round(float(np.quantile(values, 0.975)), 15),
                ],
            }
        overall_key = "overall_feedback/auebc/none/none/none/feedback_benefit"
        overall_point = _domain_balanced(auebc_effects, "feedback_benefit")
        real_specimen = {
            stratifier: intervals[
                "conditional_value_strata/specimen_auebc/real/"
                f"{stratifier}/none/high_minus_low_feedback_benefit"
            ]
            for stratifier in _STRATIFIERS
        }
        mechanism_supported = any(
            value["interval"][0] > 0.0
            and intervals[
                "conditional_value_strata/specimen_auebc/real/"
                f"{stratifier}/none/high_stratum_feedback_benefit"
            ]["interval"][0]
            > 0.0
            for stratifier, value in real_specimen.items()
        )
        conclusion = (
            "Feedback benefit is descriptively concentrated in specimens with larger real conditional-value changes, but this post-hoc association is not causal."
            if mechanism_supported
            else "Frozen feedback does not show reliable benefit concentrated in specimens with larger real conditional-value changes; the diagnostic does not support feedback as the operative mechanism."
        )
        summary = {
            "schema_version": 1,
            "stage": "P16_FEEDBACK_MECHANISM",
            "audit_base_git_sha": config["audit_base_git_sha"],
            "config_sha256": config["config_sha256"],
            "specimen_count": 276,
            "domain_order": domains,
            "feedback_sign": config["feedback_sign"],
            "overall_feedback_auebc_benefit": round(overall_point, 15),
            "overall_feedback_auebc_interval": intervals[overall_key]["interval"],
            "feedback_overall_supported": intervals[overall_key]["interval"][0] > 0.0,
            "feedback_value_change_mechanism_supported": mechanism_supported,
            "real_specimen_high_minus_low": real_specimen,
            "bootstrap_intervals": intervals,
            "quantile_protocol": "outcome-blind stable within-held-domain equal-count tertiles with specimen_id tie break",
            "causal_claim_allowed": False,
            "p7_tree_state_sha256": p7_before,
            "primary_conclusion": conclusion,
        }
        curve_effects.write_csv(
            temporary / "feedback_effects.csv", float_scientific=False
        )
        auebc_effects.write_csv(
            temporary / "specimen_auebc_effects.csv", float_scientific=False
        )
        domain_effects.write_csv(
            temporary / "domain_effects.csv", float_scientific=False
        )
        state.write_parquet(
            temporary / "state_diagnostics.parquet",
            compression="zstd",
            statistics=True,
        )
        specimen.write_csv(
            temporary / "specimen_diagnostics.csv", float_scientific=False
        )
        thresholds.write_csv(
            temporary / "quantile_thresholds.csv", float_scientific=False
        )
        strata.write_csv(temporary / "stratum_effects.csv", float_scientific=False)
        domain_associations.write_csv(
            temporary / "domain_associations.csv", float_scientific=False
        )
        association_summary.write_csv(
            temporary / "association_summary.csv", float_scientific=False
        )
        bootstrap.write_csv(temporary / "bootstrap.csv", float_scientific=False)
        (temporary / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "REPORT.md").write_text(
            "# P16 Feedback Mechanism Diagnostic\n\n"
            f"{conclusion}\n\n"
            f"The frozen equal-domain feedback benefit is {overall_point:.8g} "
            "CAI AUEBC, where positive means feedback is better. Its synchronized "
            f"95% interval is {intervals[overall_key]['interval']}. No model was "
            "retrained. P4 feedback/no-feedback outcomes are joined to P9 real-scorer "
            "and teacher value-change diagnostics at the physical-specimen level.\n\n"
            "Tertile probabilities were frozen before outcome analysis and assigned "
            "within held-out domains using only the mechanism variable; specimen ID "
            "breaks exact ties. These are descriptive post-hoc associations over six "
            "historical domains, not causal or external-confirmation evidence. P7 "
            "remains unchanged.\n",
            encoding="utf-8",
        )
        _metadata(
            temporary,
            config=config,
            source_hashes={
                **{name: _sha256(path) for name, path in sources.items()},
                "p7_tree_state": p7_before,
            },
        )
        if _tree_state(p7) != p7_before:
            raise FeedbackMechanismError("P16 modified P7")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    verify_p16_feedback_mechanism_package(destination)
    return destination


def verify_p16_feedback_mechanism_package(
    package_path: str | Path,
) -> dict[str, object]:
    package = Path(package_path)
    if not package.is_dir() or {item.name for item in package.iterdir()} != _FILES:
        raise FeedbackMechanismError("P16 package file roster changed")
    checksums = {}
    for line in (package / "CHECKSUMS.sha256").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    if set(checksums) != _FILES - {"CHECKSUMS.sha256"} or any(
        _sha256(package / name) != digest for name, digest in checksums.items()
    ):
        raise FeedbackMechanismError("P16 package checksum mismatch")
    summary = json.loads((package / "summary.json").read_text(encoding="utf-8"))
    effects = pl.read_csv(package / "feedback_effects.csv")
    specimen = pl.read_csv(package / "specimen_diagnostics.csv")
    state = pl.read_parquet(package / "state_diagnostics.parquet")
    thresholds = pl.read_csv(package / "quantile_thresholds.csv")
    bootstrap = pl.read_csv(package / "bootstrap.csv")
    if (
        summary.get("stage") != "P16_FEEDBACK_MECHANISM"
        or summary.get("causal_claim_allowed") is not False
        or effects.height != 276 * 6
        or specimen.height != 276 * 2 * 2
        or state.height != 276 * 2 * 2 * 4
        or thresholds.height != 6 * 2 * 2 * 5
        or bootstrap.height != (1 + 6 + 2 * (2 * 2 + 2 * 2 * 4)) * 5000
        or effects.select(
            (
                pl.col("feedback_benefit")
                - (
                    pl.col("no_feedback_cai_error") - pl.col("feedback_cai_error")
                ).round(15)
            )
            .abs()
            .max()
        ).item()
        != 0.0
    ):
        raise FeedbackMechanismError("P16 scientific contract changed")
    return summary


__all__ = [
    "FeedbackMechanismError",
    "align_feedback_effect",
    "assign_outcome_blind_tertiles",
    "run_p16_feedback_mechanism",
    "verify_p16_feedback_mechanism_package",
]
