"""Registered full-bank and clean-stratum content attribution for the probe."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..dynamic_metrics import aggregate_dynamic_metrics, bootstrap_dynamic_contrasts
from ..dynamic_package import verify_dynamic_package
from ..mris_data import MRISFeatureBank
from ..mris_metrics import bootstrap_mris_contrasts, evaluate_mris_predictions
from ..mris_package import verify_mris_package
from .artifacts import verify_artifact_integrity, write_artifact_integrity


class NeuralProbeAttributionError(ValueError):
    """Raised when N3 inputs cannot support the registered attribution."""


CLEAN_NONPRIV = frozenset({"uniform", "random"})
_CONTROL_MODES = ("positions_only", "shuffled", "static")


@dataclass(frozen=True, slots=True)
class ContentAttribution:
    domain_metrics: pl.DataFrame
    aggregate_metrics: pl.DataFrame
    bootstrap: pl.DataFrame
    gate: str


def _validate_request(
    *,
    frames: tuple[pl.DataFrame, ...],
    domain_order: tuple[str, ...],
    bootstrap_replicates: int,
    seed: int,
) -> None:
    if (
        any(not isinstance(frame, pl.DataFrame) or frame.height == 0 for frame in frames)
        or type(domain_order) is not tuple
        or len(domain_order) != 6
        or len(set(domain_order)) != len(domain_order)
        or any(type(domain) is not str or not domain for domain in domain_order)
        or type(bootstrap_replicates) is not int
        or bootstrap_replicates <= 0
        or type(seed) is not int
    ):
        raise NeuralProbeAttributionError("N3 attribution request is invalid")


def _strata(table: pl.DataFrame) -> tuple[tuple[str, pl.DataFrame], ...]:
    if "method" not in table.columns:
        raise NeuralProbeAttributionError("N3 method labels are unavailable")
    methods = set(table.get_column("method").unique())
    if not CLEAN_NONPRIV <= methods:
        raise NeuralProbeAttributionError("N3 clean-nonprivileged roster is incomplete")
    return (
        ("full_bank", table),
        ("clean_nonpriv", table.filter(pl.col("method").is_in(CLEAN_NONPRIV))),
    )


def _p2_tables(
    predictions: pl.DataFrame,
    *,
    model: str,
    domain_order: tuple[str, ...],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[pl.DataFrame], list[pl.DataFrame]]:
    domains: list[pl.DataFrame] = []
    bootstraps: list[pl.DataFrame] = []
    for stratum_index, (stratum, selected) in enumerate(_strata(predictions)):
        metrics = evaluate_mris_predictions(selected, domain_order=domain_order)
        bootstrap = bootstrap_mris_contrasts(
            metrics.per_specimen_metrics,
            reference_mode="real",
            control_modes=_CONTROL_MODES,
            domain_order=domain_order,
            replicates=bootstrap_replicates,
            seed=seed + stratum_index,
        ).with_columns(
            pl.lit("p2").alias("stage"),
            pl.lit(model).alias("model"),
            pl.lit(stratum).alias("stratum"),
            pl.col("control_minus_reference_auebc").alias("control_minus_real"),
        )
        real = metrics.domain_auebc.filter(pl.col("mode") == "real").select(
            "outer_domain", pl.col("auebc").alias("real_metric")
        )
        control = (
            metrics.domain_auebc.filter(pl.col("mode").is_in(_CONTROL_MODES))
            .rename({"mode": "control_mode", "auebc": "control_metric"})
            .join(real, on="outer_domain", how="inner")
            .with_columns(
                pl.lit("p2_auebc").alias("metric"),
                pl.lit("p2").alias("stage"),
                pl.lit(model).alias("model"),
                pl.lit(stratum).alias("stratum"),
                (pl.col("control_metric") - pl.col("real_metric")).alias(
                    "control_minus_real"
                ),
            )
            .select(
                "stage",
                "model",
                "stratum",
                "metric",
                "outer_domain",
                "control_mode",
                "real_metric",
                "control_metric",
                "control_minus_real",
            )
        )
        domains.append(control)
        bootstraps.append(
            bootstrap.select(
                "stage",
                "model",
                "stratum",
                "replicate",
                "control_mode",
                "reference_auebc",
                "control_auebc",
                "control_minus_real",
            )
        )
    return domains, bootstraps


def _p3_tables(
    state_metrics: pl.DataFrame,
    *,
    model: str,
    domain_order: tuple[str, ...],
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[pl.DataFrame], list[pl.DataFrame]]:
    domains: list[pl.DataFrame] = []
    bootstraps: list[pl.DataFrame] = []
    for stratum_index, (stratum, selected) in enumerate(_strata(state_metrics)):
        metrics = aggregate_dynamic_metrics(selected)
        bootstrap = bootstrap_dynamic_contrasts(
            metrics.per_specimen,
            reference_mode="real",
            control_modes=_CONTROL_MODES,
            domain_order=domain_order,
            replicates=bootstrap_replicates,
            seed=seed + 300 + stratum_index,
        ).with_columns(
            pl.lit("p3").alias("stage"),
            pl.lit(model).alias("model"),
            pl.lit(stratum).alias("stratum"),
            pl.col("control_minus_reference_regret").alias("control_minus_real"),
        )
        real = metrics.per_domain.filter(pl.col("mode") == "real").select(
            "outer_domain", pl.col("next_action_regret").alias("real_metric")
        )
        control = (
            metrics.per_domain.filter(pl.col("mode").is_in(_CONTROL_MODES))
            .select(
                "outer_domain",
                pl.col("mode").alias("control_mode"),
                pl.col("next_action_regret").alias("control_metric"),
            )
            .join(real, on="outer_domain", how="inner")
            .with_columns(
                pl.lit("p3_next_action_regret").alias("metric"),
                pl.lit("p3").alias("stage"),
                pl.lit(model).alias("model"),
                pl.lit(stratum).alias("stratum"),
                (pl.col("control_metric") - pl.col("real_metric")).alias(
                    "control_minus_real"
                ),
            )
            .select(
                "stage",
                "model",
                "stratum",
                "metric",
                "outer_domain",
                "control_mode",
                "real_metric",
                "control_metric",
                "control_minus_real",
            )
        )
        domains.append(control)
        bootstraps.append(
            bootstrap.select(
                "stage",
                "model",
                "stratum",
                "replicate",
                "control_mode",
                "control_minus_reference_regret",
                "reference_minus_control_utility",
                "control_minus_real",
            )
        )
    return domains, bootstraps


def _aggregate(
    domain_metrics: pl.DataFrame,
    bootstrap: pl.DataFrame,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ("stage", "model", "stratum", "metric", "control_mode")
    for key, table in domain_metrics.group_by(*keys, maintain_order=True):
        stage, model, stratum, metric, control_mode = key
        draws = bootstrap.filter(
            (pl.col("stage") == stage)
            & (pl.col("model") == model)
            & (pl.col("stratum") == stratum)
            & (pl.col("control_mode") == control_mode)
        ).get_column("control_minus_real").to_numpy()
        values = table.get_column("control_minus_real")
        if draws.size == 0 or table.height != 6:
            raise NeuralProbeAttributionError("N3 domain/bootstrap roster is incomplete")
        rows.append(
            {
                "stage": stage,
                "model": model,
                "stratum": stratum,
                "metric": metric,
                "control_mode": control_mode,
                "control_minus_real": float(values.mean()),
                "ci95_lower": float(np.quantile(draws, 0.025)),
                "ci95_upper": float(np.quantile(draws, 0.975)),
                "favorable_domain_count": int((values > 0.0).sum()),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort(list(keys))


def _content_gate(aggregate: pl.DataFrame) -> str:
    headline_controls = ("positions_only", "shuffled")
    spatial = aggregate.filter(
        (pl.col("model") == "spatial")
        & (pl.col("stratum") == "clean_nonpriv")
        & (pl.col("control_mode").is_in(headline_controls))
    )
    if spatial.height != 4:
        raise NeuralProbeAttributionError("N3 headline roster is incomplete")
    strong = spatial.select(
        (
            (pl.col("control_minus_real") > 0.0)
            & (pl.col("ci95_lower") > 0.0)
            & (pl.col("favorable_domain_count") >= 4)
        ).all()
    ).item()
    if strong:
        return "CONTENT_STRONG_GO"
    baseline = aggregate.filter(
        (pl.col("model") == "deepsets")
        & (pl.col("stratum") == "clean_nonpriv")
        & (pl.col("control_mode").is_in(headline_controls))
    ).select(
        "stage",
        "control_mode",
        pl.col("control_minus_real").alias("deepsets_control_minus_real"),
    )
    movement = spatial.join(
        baseline,
        on=["stage", "control_mode"],
        how="inner",
    )
    if movement.height == 4 and movement.select(
        (
            pl.col("control_minus_real")
            > pl.col("deepsets_control_minus_real")
        ).all()
    ).item():
        return "CONTENT_PROMISING"
    return "CONTENT_NO_GO"


def evaluate_content_attribution(
    *,
    spatial_p2_predictions: pl.DataFrame,
    deepsets_p2_predictions: pl.DataFrame,
    spatial_p3_state_metrics: pl.DataFrame,
    deepsets_p3_state_metrics: pl.DataFrame,
    domain_order: tuple[str, ...],
    bootstrap_replicates: int,
    seed: int,
) -> ContentAttribution:
    """Evaluate both registered attribution layers without changing their metrics."""
    frames = (
        spatial_p2_predictions,
        deepsets_p2_predictions,
        spatial_p3_state_metrics,
        deepsets_p3_state_metrics,
    )
    _validate_request(
        frames=frames,
        domain_order=domain_order,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    domain_parts: list[pl.DataFrame] = []
    bootstrap_parts: list[pl.DataFrame] = []
    for model, p2, p3 in (
        ("spatial", spatial_p2_predictions, spatial_p3_state_metrics),
        ("deepsets", deepsets_p2_predictions, deepsets_p3_state_metrics),
    ):
        p2_domains, p2_bootstraps = _p2_tables(
            p2,
            model=model,
            domain_order=domain_order,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        p3_domains, p3_bootstraps = _p3_tables(
            p3,
            model=model,
            domain_order=domain_order,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        domain_parts.extend((*p2_domains, *p3_domains))
        bootstrap_parts.extend((*p2_bootstraps, *p3_bootstraps))
    domains = pl.concat(domain_parts, how="vertical_relaxed").sort(
        ["stage", "model", "stratum", "control_mode", "outer_domain"]
    )
    bootstraps = pl.concat(bootstrap_parts, how="diagonal_relaxed").sort(
        ["stage", "model", "stratum", "control_mode", "replicate"]
    )
    aggregate = _aggregate(domains, bootstraps)
    return ContentAttribution(
        domain_metrics=domains,
        aggregate_metrics=aggregate,
        bootstrap=bootstraps,
        gate=_content_gate(aggregate),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _bind_state_methods(
    state_metrics: pl.DataFrame,
    *,
    bank: MRISFeatureBank,
) -> pl.DataFrame:
    if "method" in state_metrics.columns:
        raise NeuralProbeAttributionError("N3 P3 method labels are already present")
    mapping = pl.DataFrame(
        {"state_id": bank.state_ids, "method": bank.methods},
        schema={"state_id": pl.String, "method": pl.String},
    )
    output = state_metrics.join(mapping, on="state_id", how="inner")
    if output.height != state_metrics.height or output.get_column("method").null_count():
        raise NeuralProbeAttributionError("N3 P3 state/method roster changed")
    return output


def finalize_n3_content_attribution(
    bank: MRISFeatureBank,
    *,
    n1_p2_root: str | Path,
    n2_p3_root: str | Path,
    frozen_p2_root: str | Path,
    frozen_p3_root: str | Path,
    output_root: str | Path,
    source_config_path: str | Path,
    base_commit: str,
    config_sha256: str,
    bootstrap_replicates: int,
    seed: int,
) -> Path:
    """Write the registered N3 analysis from already trained P2/P3 modes."""
    if (
        type(bank) is not MRISFeatureBank
        or len(bank.domain_order) != 6
        or type(base_commit) is not str
        or len(base_commit) != 40
        or type(config_sha256) is not str
        or len(config_sha256) != 64
    ):
        raise NeuralProbeAttributionError("N3 finalization request is invalid")
    n1_root = Path(n1_p2_root)
    n2_root = Path(n2_p3_root)
    frozen_p2 = Path(frozen_p2_root)
    frozen_p3 = Path(frozen_p3_root)
    destination = Path(output_root)
    source_config = Path(source_config_path)
    if _sha256(source_config) != config_sha256 or destination.exists():
        raise NeuralProbeAttributionError("N3 config or output contract changed")
    n1_manifest = verify_artifact_integrity(n1_root)
    n2_manifest = verify_artifact_integrity(n2_root)
    frozen_p2_manifest = verify_mris_package(frozen_p2)
    frozen_p3_manifest = verify_dynamic_package(frozen_p3)
    if (
        n1_manifest.get("artifact") != "mavis_neural_probe_n1_spatial_p2"
        or n2_manifest.get("artifact") != "mavis_neural_probe_n2_dynamic_p3"
        or frozen_p2_manifest.get("artifact") != "mavis_p2_mris"
        or frozen_p3_manifest.get("artifact") != "mavis_p3_dynamic_voi"
    ):
        raise NeuralProbeAttributionError("N3 source artifact identity changed")
    spatial_p2 = pl.read_parquet(n1_root / "state_predictions.parquet")
    deepsets_p2 = pl.read_parquet(frozen_p2 / "state_predictions.parquet")
    spatial_p3 = _bind_state_methods(
        pl.read_parquet(n2_root / "state_metrics.parquet"), bank=bank
    )
    deepsets_p3 = _bind_state_methods(
        pl.read_parquet(frozen_p3 / "state_metrics.parquet"), bank=bank
    )
    result = evaluate_content_attribution(
        spatial_p2_predictions=spatial_p2,
        deepsets_p2_predictions=deepsets_p2,
        spatial_p3_state_metrics=spatial_p3,
        deepsets_p3_state_metrics=deepsets_p3,
        domain_order=bank.domain_order,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    try:
        runtime_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise NeuralProbeAttributionError("N3 runtime Git state is unavailable") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".n3_content.", dir=destination.parent))
    try:
        result.domain_metrics.write_csv(temporary / "domain_metrics.csv")
        result.aggregate_metrics.write_csv(temporary / "aggregate_metrics.csv")
        result.bootstrap.write_csv(temporary / "bootstrap.csv")
        shutil.copyfile(source_config, temporary / "config.json")
        headline = result.aggregate_metrics.filter(
            (pl.col("stratum") == "clean_nonpriv")
            & (pl.col("control_mode").is_in(("positions_only", "shuffled")))
        ).sort(["stage", "model", "control_mode"])
        summary = {
            "base_commit": base_commit,
            "bootstrap_replicates": bootstrap_replicates,
            "clean_nonprivileged_methods": sorted(CLEAN_NONPRIV),
            "config_sha256": config_sha256,
            "frozen_p2_manifest_sha256": _sha256(
                frozen_p2 / "artifact_manifest.json"
            ),
            "frozen_p3_manifest_sha256": _sha256(
                frozen_p3 / "artifact_manifest.json"
            ),
            "gate": result.gate,
            "gate_scope": "both_p2_and_p3_clean_nonprivileged_layers",
            "headline_contrasts": headline.to_dicts(),
            "n1_manifest_sha256": _sha256(n1_root / "artifact_manifest.json"),
            "n2_manifest_sha256": _sha256(n2_root / "artifact_manifest.json"),
            "runtime_head": runtime_head,
            "schema_version": 1,
            "seed": seed,
            "stage": "N3_CONTENT_ATTRIBUTION",
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "unified_contrast": "control_minus_real",
        }
        _write_json(temporary / "summary.json", summary)
        lines = [
            "# N3 Content Attribution",
            "",
            f"Gate: `{result.gate}`.",
            "",
            (
                "Positive `control_minus_real` values mean measured real content is "
                "better. The headline stratum is the pre-registered `CLEAN_NONPRIV = "
                "{uniform, random}`; the complete frozen bank is retained in every "
                "output table."
            ),
            "",
            "| Stage | Model | Control | Point | 95% CI | Favorable domains |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in headline.iter_rows(named=True):
            lines.append(
                f"| {row['stage']} | {row['model']} | {row['control_mode']} | "
                f"{row['control_minus_real']:.10f} | "
                f"[{row['ci95_lower']:.10f}, {row['ci95_upper']:.10f}] | "
                f"{row['favorable_domain_count']}/6 |"
            )
        lines.extend(
            [
                "",
                (
                    "P2 retains AUEBC and P3 retains next-action regret. The unified "
                    "sign is an explicit exploratory report conversion only; the "
                    "registered metric implementations, state rows, controls, and "
                    "bootstrap units are unchanged. The overall gate conservatively "
                    "requires both clean P2 and clean P3 layers to support both headline "
                    "controls."
                ),
                "",
            ]
        )
        (temporary / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        write_artifact_integrity(
            temporary,
            artifact="mavis_neural_probe_n3_content_attribution",
            base_commit=base_commit,
            config_sha256=config_sha256,
        )
        verify_artifact_integrity(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    verify_artifact_integrity(destination)
    return destination


__all__ = [
    "CLEAN_NONPRIV",
    "ContentAttribution",
    "NeuralProbeAttributionError",
    "evaluate_content_attribution",
    "finalize_n3_content_attribution",
]
