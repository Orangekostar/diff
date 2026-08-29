"""Artifact integrity and registered N1 aggregation for the neural probe."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ..dynamic_metrics import (
    DynamicMetricTables,
    aggregate_dynamic_metrics,
    bootstrap_dynamic_contrasts,
)
from ..dynamic_package import (
    _decision_state_roster,
    verify_dynamic_package,
)
from ..dynamic_package import _validate_tables as _validate_dynamic_tables
from ..mris_data import MRISFeatureBank
from ..mris_metrics import (
    MRISMetricTables,
    bootstrap_mris_contrasts,
    evaluate_mris_predictions,
)
from ..mris_package import _validate_scientific_tables, verify_mris_package


class NeuralProbeArtifactError(RuntimeError):
    """Raised when a neural-probe stage artifact violates its contract."""


@dataclass(frozen=True, slots=True)
class N1Comparison:
    point_estimate: float
    ci95_lower: float
    ci95_upper: float
    favorable_domain_count: int
    gate: str
    domain_metrics: pl.DataFrame
    bootstrap: pl.DataFrame
    metric_tables: MRISMetricTables


@dataclass(frozen=True, slots=True)
class N2Comparison:
    point_estimate: float
    ci95_lower: float
    ci95_upper: float
    favorable_domain_count: int
    gate: str
    domain_metrics: pl.DataFrame
    bootstrap: pl.DataFrame
    metric_tables: DynamicMetricTables


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


def assign_directional_gate(
    *,
    prefix: str,
    point_estimate: float,
    ci95_lower: float,
    ci95_upper: float,
    favorable_domain_count: int,
) -> str:
    values = (float(point_estimate), float(ci95_lower), float(ci95_upper))
    if (
        type(prefix) is not str
        or not prefix
        or not all(math.isfinite(value) for value in values)
        or type(favorable_domain_count) is not int
        or not 0 <= favorable_domain_count <= 6
        or ci95_lower > ci95_upper
    ):
        raise NeuralProbeArtifactError("directional gate inputs are invalid")
    if point_estimate > 0.0 and ci95_lower > 0.0 and favorable_domain_count >= 4:
        return f"{prefix}_STRONG_GO"
    if (
        point_estimate > 0.0
        and favorable_domain_count >= 4
        and ci95_lower <= 0.0 <= ci95_upper
    ):
        return f"{prefix}_PROMISING"
    return f"{prefix}_NO_GO"


def evaluate_n1_comparison(
    *,
    spatial_predictions: pl.DataFrame,
    deepsets_predictions: pl.DataFrame,
    domain_order: tuple[str, ...],
    replicates: int,
    seed: int,
) -> N1Comparison:
    if not isinstance(spatial_predictions, pl.DataFrame) or not isinstance(
        deepsets_predictions, pl.DataFrame
    ):
        raise NeuralProbeArtifactError("N1 comparison predictions are invalid")
    spatial = spatial_predictions.filter(pl.col("mode") == "real").with_columns(
        pl.lit("spatial_real").alias("mode")
    )
    deepsets = deepsets_predictions.filter(pl.col("mode") == "real").with_columns(
        pl.lit("deepsets_real").alias("mode")
    )
    if (
        spatial.height == 0
        or deepsets.height != spatial.height
        or set(spatial.get_column("state_id")) != set(deepsets.get_column("state_id"))
    ):
        raise NeuralProbeArtifactError("N1 state rosters do not match")
    metrics = evaluate_mris_predictions(
        pl.concat((spatial, deepsets), how="vertical_relaxed"),
        domain_order=domain_order,
    )
    bootstrap = bootstrap_mris_contrasts(
        metrics.per_specimen_metrics,
        reference_mode="spatial_real",
        control_modes=("deepsets_real",),
        domain_order=domain_order,
        replicates=replicates,
        seed=seed,
    )
    spatial_domains = metrics.domain_auebc.filter(
        pl.col("mode") == "spatial_real"
    ).select("outer_domain", pl.col("auebc").alias("spatial_real_auebc"))
    deepsets_domains = metrics.domain_auebc.filter(
        pl.col("mode") == "deepsets_real"
    ).select("outer_domain", pl.col("auebc").alias("deepsets_real_auebc"))
    domains = (
        deepsets_domains.join(spatial_domains, on="outer_domain", how="inner")
        .with_columns(
            (
                pl.col("deepsets_real_auebc") - pl.col("spatial_real_auebc")
            ).alias("deepsets_minus_spatial")
        )
        .with_columns((pl.col("deepsets_minus_spatial") > 0.0).alias("favorable"))
        .sort("outer_domain")
    )
    if domains.height != 6 or set(domains.get_column("outer_domain")) != set(
        domain_order
    ):
        raise NeuralProbeArtifactError("N1 domain roster is incomplete")
    point = float(domains.get_column("deepsets_minus_spatial").mean())
    draws = bootstrap.get_column("control_minus_reference_auebc").to_numpy()
    lower = float(np.quantile(draws, 0.025))
    upper = float(np.quantile(draws, 0.975))
    favorable = int(domains.get_column("favorable").sum())
    return N1Comparison(
        point_estimate=point,
        ci95_lower=lower,
        ci95_upper=upper,
        favorable_domain_count=favorable,
        gate=assign_directional_gate(
            prefix="REPRESENTATION",
            point_estimate=point,
            ci95_lower=lower,
            ci95_upper=upper,
            favorable_domain_count=favorable,
        ),
        domain_metrics=domains,
        bootstrap=bootstrap,
        metric_tables=metrics,
    )


def evaluate_n2_comparison(
    *,
    spatial_state_metrics: pl.DataFrame,
    deepsets_state_metrics: pl.DataFrame,
    domain_order: tuple[str, ...],
    replicates: int,
    seed: int,
) -> N2Comparison:
    if not isinstance(spatial_state_metrics, pl.DataFrame) or not isinstance(
        deepsets_state_metrics, pl.DataFrame
    ):
        raise NeuralProbeArtifactError("N2 comparison metrics are invalid")
    spatial = spatial_state_metrics.filter(pl.col("mode") == "real").with_columns(
        pl.lit("spatial_real").alias("mode")
    )
    deepsets = deepsets_state_metrics.filter(pl.col("mode") == "real").with_columns(
        pl.lit("deepsets_real").alias("mode")
    )
    if (
        spatial.height == 0
        or deepsets.height != spatial.height
        or set(spatial.get_column("state_id")) != set(deepsets.get_column("state_id"))
    ):
        raise NeuralProbeArtifactError("N2 state rosters do not match")
    metrics = aggregate_dynamic_metrics(
        pl.concat((spatial, deepsets), how="vertical_relaxed")
    )
    bootstrap = bootstrap_dynamic_contrasts(
        metrics.per_specimen,
        reference_mode="spatial_real",
        control_modes=("deepsets_real",),
        domain_order=domain_order,
        replicates=replicates,
        seed=seed,
    )
    spatial_domains = metrics.per_domain.filter(
        pl.col("mode") == "spatial_real"
    ).select(
        "outer_domain",
        pl.col("next_action_regret").alias("spatial_real_regret"),
        pl.col("one_step_cai_utility").alias("spatial_real_utility"),
    )
    deepsets_domains = metrics.per_domain.filter(
        pl.col("mode") == "deepsets_real"
    ).select(
        "outer_domain",
        pl.col("next_action_regret").alias("deepsets_real_regret"),
        pl.col("one_step_cai_utility").alias("deepsets_real_utility"),
    )
    domains = (
        deepsets_domains.join(spatial_domains, on="outer_domain", how="inner")
        .with_columns(
            (
                pl.col("deepsets_real_regret") - pl.col("spatial_real_regret")
            ).alias("deepsets_minus_spatial_regret"),
            (
                pl.col("spatial_real_utility") - pl.col("deepsets_real_utility")
            ).alias("spatial_minus_deepsets_utility"),
        )
        .with_columns(
            (pl.col("deepsets_minus_spatial_regret") > 0.0).alias("favorable")
        )
        .sort("outer_domain")
    )
    if domains.height != 6 or set(domains.get_column("outer_domain")) != set(
        domain_order
    ):
        raise NeuralProbeArtifactError("N2 domain roster is incomplete")
    point = float(domains.get_column("deepsets_minus_spatial_regret").mean())
    draws = bootstrap.get_column("control_minus_reference_regret").to_numpy()
    lower = float(np.quantile(draws, 0.025))
    upper = float(np.quantile(draws, 0.975))
    favorable = int(domains.get_column("favorable").sum())
    return N2Comparison(
        point_estimate=point,
        ci95_lower=lower,
        ci95_upper=upper,
        favorable_domain_count=favorable,
        gate=assign_directional_gate(
            prefix="VALUE",
            point_estimate=point,
            ci95_lower=lower,
            ci95_upper=upper,
            favorable_domain_count=favorable,
        ),
        domain_metrics=domains,
        bootstrap=bootstrap,
        metric_tables=metrics,
    )


def write_artifact_integrity(
    root: str | Path,
    *,
    artifact: str,
    base_commit: str,
    config_sha256: str,
) -> None:
    output = Path(root)
    if (
        not output.is_dir()
        or type(artifact) is not str
        or not artifact
        or type(base_commit) is not str
        or len(base_commit) != 40
        or type(config_sha256) is not str
        or len(config_sha256) != 64
    ):
        raise NeuralProbeArtifactError("artifact integrity request is invalid")
    for name in ("artifact_manifest.json", "CHECKSUMS.sha256"):
        path = output / name
        if path.exists():
            path.unlink()
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "artifact": artifact,
        "base_commit": base_commit,
        "config_sha256": config_sha256,
        "files": {
            path.relative_to(output).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        },
        "schema_version": 1,
    }
    _write_json(output / "artifact_manifest.json", manifest)
    checksum_files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="ascii",
    )


def verify_artifact_integrity(root: str | Path) -> dict[str, object]:
    output = Path(root)
    try:
        manifest = json.loads(
            (output / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        lines = (output / "CHECKSUMS.sha256").read_text(
            encoding="ascii"
        ).splitlines()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NeuralProbeArtifactError("artifact metadata is invalid") from error
    if type(manifest) is not dict or type(manifest.get("files")) is not dict:
        raise NeuralProbeArtifactError("artifact manifest is invalid")
    expected_manifest_files = set(manifest["files"])
    actual_manifest_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "CHECKSUMS.sha256"}
    }
    if expected_manifest_files != actual_manifest_files:
        raise NeuralProbeArtifactError("artifact file roster changed")
    for name, metadata in manifest["files"].items():
        path = output / name
        if (
            type(metadata) is not dict
            or metadata.get("bytes") != path.stat().st_size
            or metadata.get("sha256") != _sha256(path)
        ):
            raise NeuralProbeArtifactError(f"artifact manifest mismatch: {name}")
    expected_checksums = expected_manifest_files | {"artifact_manifest.json"}
    ledger: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise NeuralProbeArtifactError("artifact checksum ledger is invalid") from error
        if len(digest) != 64 or name in ledger:
            raise NeuralProbeArtifactError("artifact checksum ledger is invalid")
        ledger[name] = digest
    if set(ledger) != expected_checksums:
        raise NeuralProbeArtifactError("artifact checksum coverage changed")
    for name, digest in ledger.items():
        if _sha256(output / name) != digest:
            raise NeuralProbeArtifactError(f"artifact checksum mismatch: {name}")
    return manifest


def _load_worker(
    worker: Path,
    *,
    outer_domain: str,
    bank: MRISFeatureBank,
    base_commit: str,
    config_sha256: str,
) -> dict[str, object]:
    try:
        payload = json.loads((worker / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NeuralProbeArtifactError("N1 worker summary is invalid") from error
    target_rows = sum(domain == outer_domain for domain in bank.domain_ids)
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("architecture_name") != "spatial_grid_cnn_v1"
        or payload.get("outer_domain") != outer_domain
        or payload.get("base_commit") != base_commit
        or payload.get("config_sha256") != config_sha256
        or payload.get("feature_bank_input_state_sha256") != bank.input_state_sha256
        or payload.get("feature_bank_target_state_sha256") != bank.target_state_sha256
        or tuple(payload.get("trainable_modes", ()))
        != ("static", "positions_only", "real", "shuffled")
        or payload.get("prediction_count") != target_rows * 5
        or payload.get("inner_fold_count") != 20
        or payload.get("target_data_used_for_selection") is not False
        or len(payload.get("inner_model_state_sha256", {})) != 20
        or set(payload.get("model_state_sha256", {}))
        != {"static", "positions_only", "real", "shuffled", "reconstruction"}
        or type(payload.get("files")) is not dict
    ):
        raise NeuralProbeArtifactError("N1 worker contract changed")
    actual_files = {
        path.relative_to(worker).as_posix()
        for path in worker.rglob("*")
        if path.is_file() and path.name != "complete.json"
    }
    if actual_files != set(payload["files"]):
        raise NeuralProbeArtifactError("N1 worker file roster changed")
    for name, digest in payload["files"].items():
        if type(digest) is not str or _sha256(worker / name) != digest:
            raise NeuralProbeArtifactError(f"N1 worker checksum mismatch: {name}")
    return payload


def _bootstrap_control_summary(bootstrap: pl.DataFrame) -> dict[str, object]:
    output: dict[str, object] = {}
    for control in sorted(bootstrap.get_column("control_mode").unique()):
        values = bootstrap.filter(pl.col("control_mode") == control).get_column(
            "control_minus_reference_auebc"
        ).to_numpy()
        output[control] = {
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
            "mean_control_minus_real": float(np.mean(values, dtype=np.float64)),
        }
    return output


def finalize_n1_spatial_p2(
    bank: MRISFeatureBank,
    *,
    worker_root: str | Path,
    frozen_p2_root: str | Path,
    output_root: str | Path,
    source_config_path: str | Path,
    base_commit: str,
    config_sha256: str,
    bootstrap_replicates: int,
    seed: int,
) -> Path:
    if (
        type(bank) is not MRISFeatureBank
        or len(bank.domain_order) != 6
        or type(bootstrap_replicates) is not int
        or bootstrap_replicates <= 0
        or type(seed) is not int
    ):
        raise NeuralProbeArtifactError("N1 finalization request is invalid")
    workers = Path(worker_root)
    frozen_root = Path(frozen_p2_root)
    destination = Path(output_root)
    source_config = Path(source_config_path)
    if _sha256(source_config) != config_sha256:
        raise NeuralProbeArtifactError("N1 source config hash changed")
    frozen_manifest = verify_mris_package(frozen_root)
    if destination.exists():
        raise NeuralProbeArtifactError("N1 output already exists")
    prediction_parts: list[pl.DataFrame] = []
    audit_parts: list[pl.DataFrame] = []
    donor_parts: list[pl.DataFrame] = []
    worker_payloads: list[dict[str, object]] = []
    for domain in bank.domain_order:
        worker = workers / domain
        worker_payloads.append(
            _load_worker(
                worker,
                outer_domain=domain,
                bank=bank,
                base_commit=base_commit,
                config_sha256=config_sha256,
            )
        )
        prediction_parts.append(pl.read_parquet(worker / "predictions.parquet"))
        audit_parts.append(pl.read_parquet(worker / "model_selection_audit.parquet"))
        donor_parts.append(pl.read_parquet(worker / "donor_mapping.parquet"))
    predictions = pl.concat(prediction_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "mode", "method", "nominal_checkpoint"]
    )
    audits = pl.concat(audit_parts, how="vertical_relaxed").sort(
        ["outer_domain", "mode", "record_type", "validation_domain"],
        nulls_last=True,
    )
    donors = pl.concat(donor_parts, how="vertical_relaxed").sort(
        ["outer_domain", "recipient_id"]
    )
    _validate_scientific_tables(
        predictions,
        audits,
        donors,
        domain_order=bank.domain_order,
        specimen_count=len(set(bank.specimen_ids)),
        state_count=bank.row_count,
    )
    candidate_metrics = evaluate_mris_predictions(
        predictions, domain_order=bank.domain_order
    )
    control_bootstrap = bootstrap_mris_contrasts(
        candidate_metrics.per_specimen_metrics,
        reference_mode="real",
        control_modes=("static", "positions_only", "shuffled", "reconstruction"),
        domain_order=bank.domain_order,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    deepsets_predictions = pl.read_parquet(frozen_root / "state_predictions.parquet")
    comparison = evaluate_n1_comparison(
        spatial_predictions=predictions,
        deepsets_predictions=deepsets_predictions,
        domain_order=bank.domain_order,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    spatial_real = float(
        candidate_metrics.aggregate_auebc.filter(pl.col("mode") == "real").item(
            0, "domain_balanced_auebc"
        )
    )
    deepsets_real = float(
        comparison.metric_tables.aggregate_auebc.filter(
            pl.col("mode") == "deepsets_real"
        ).item(0, "domain_balanced_auebc")
    )
    control_point = {
        row["mode"]: float(row["domain_balanced_auebc"] - spatial_real)
        for row in candidate_metrics.aggregate_auebc.filter(
            pl.col("mode") != "real"
        ).to_dicts()
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".n1_spatial_p2.", dir=destination.parent))
    try:
        predictions.write_parquet(
            temporary / "state_predictions.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        audits.write_parquet(
            temporary / "model_selection_audit.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        donors.write_csv(temporary / "donor_mapping.csv")
        candidate_metrics.per_specimen_metrics.write_csv(
            temporary / "per_specimen_metrics.csv"
        )
        candidate_metrics.domain_metrics.write_csv(
            temporary / "checkpoint_domain_metrics.csv"
        )
        candidate_metrics.aggregate_metrics.write_csv(
            temporary / "aggregate_metrics.csv"
        )
        candidate_metrics.domain_auebc.write_csv(temporary / "domain_auebc.csv")
        candidate_metrics.aggregate_auebc.write_csv(
            temporary / "aggregate_auebc.csv"
        )
        comparison.domain_metrics.write_csv(temporary / "domain_metrics.csv")
        comparison.bootstrap.write_csv(temporary / "bootstrap.csv")
        control_bootstrap.write_csv(temporary / "control_bootstrap.csv")
        shutil.copyfile(source_config, temporary / "config.json")
        checkpoint_root = temporary / "checkpoints"
        inner_root = checkpoint_root / "inner"
        inner_root.mkdir(parents=True)
        for outer_domain in bank.domain_order:
            for mode in ("static", "positions_only", "real", "shuffled"):
                shutil.copyfile(
                    workers / outer_domain / "checkpoints" / f"{mode}.npz",
                    checkpoint_root / f"{outer_domain}__{mode}.npz",
                )
                for validation_domain in bank.domain_order:
                    if validation_domain == outer_domain:
                        continue
                    shutil.copyfile(
                        workers
                        / outer_domain
                        / "checkpoints"
                        / "inner"
                        / f"{validation_domain}__{mode}.npz",
                        inner_root
                        / f"{outer_domain}__{validation_domain}__{mode}.npz",
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
            raise NeuralProbeArtifactError("N1 runtime Git state is unavailable") from error
        summary = {
            "architecture_name": "spatial_grid_cnn_v1",
            "base_commit": base_commit,
            "bootstrap_replicates": bootstrap_replicates,
            "canonical_contrast": "deepsets_real_auebc_minus_spatial_real_auebc",
            "ci95_lower": comparison.ci95_lower,
            "ci95_upper": comparison.ci95_upper,
            "config_sha256": config_sha256,
            "control_minus_real_auebc": control_point,
            "control_bootstrap": _bootstrap_control_summary(control_bootstrap),
            "deepsets_real_domain_balanced_auebc": deepsets_real,
            "favorable_domain_count": comparison.favorable_domain_count,
            "feature_bank_input_state_sha256": bank.input_state_sha256,
            "feature_bank_target_state_sha256": bank.target_state_sha256,
            "frozen_p2_manifest_sha256": _sha256(
                frozen_root / "artifact_manifest.json"
            ),
            "frozen_p2_verified": frozen_manifest.get("artifact") == "mavis_p2_mris",
            "gate": comparison.gate,
            "point_estimate": comparison.point_estimate,
            "runtime_head": runtime_head,
            "schema_version": 1,
            "seed": seed,
            "spatial_real_domain_balanced_auebc": spatial_real,
            "stage": "N1_SPATIAL_P2",
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "worker_model_states": [payload["model_state_sha256"] for payload in worker_payloads],
        }
        _write_json(temporary / "summary.json", summary)
        (temporary / "REPORT.md").write_text(
            "# N1 Spatial P2 Representation Probe\n\n"
            f"Gate: `{comparison.gate}`.\n\n"
            f"DeepSets real AUEBC: `{deepsets_real:.10f}`. Spatial real AUEBC: "
            f"`{spatial_real:.10f}`. The registered DeepSets-minus-Spatial "
            f"contrast is `{comparison.point_estimate:.10f}` with paired 95% CI "
            f"`[{comparison.ci95_lower:.10f}, {comparison.ci95_upper:.10f}]` and "
            f"favorable direction in `{comparison.favorable_domain_count}/6` "
            "held-out domains. Positive values favor Spatial.\n\n"
            "All model selection, normalization, and early stopping use source "
            "domains only. Metrics aggregate state rows within physical specimens "
            "before weighting held-out domains equally. Candidate controls retain "
            "their registered real, positions/history, shuffled, static, and "
            "independent reconstruction semantics. This exploratory stage does not "
            "modify the paper or frozen MAVIS evidence.\n",
            encoding="utf-8",
        )
        write_artifact_integrity(
            temporary,
            artifact="mavis_neural_probe_n1_spatial_p2",
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


def _load_spatial_dynamic_worker(
    worker: Path,
    *,
    outer_domain: str,
    bank: MRISFeatureBank,
    target_group_count: int,
    base_commit: str,
    config_sha256: str,
) -> dict[str, object]:
    try:
        payload = json.loads((worker / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NeuralProbeArtifactError("N2 worker summary is invalid") from error
    modes = {"static", "positions_only", "real", "shuffled"}
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("architecture_name") != "spatial_grid_cnn_v1"
        or payload.get("dynamic_scorer") != "DynamicActionScorer"
        or payload.get("outer_domain") != outer_domain
        or payload.get("base_commit") != base_commit
        or payload.get("config_sha256") != config_sha256
        or payload.get("feature_bank_input_state_sha256") != bank.input_state_sha256
        or payload.get("feature_bank_target_state_sha256") != bank.target_state_sha256
        or tuple(payload.get("modes", ()))
        != ("static", "positions_only", "real", "shuffled")
        or payload.get("target_group_count") != target_group_count
        or payload.get("inner_fold_count") != 20
        or payload.get("target_data_used_for_selection") is not False
        or set(payload.get("p2_model_state_sha256", {})) != modes
        or set(payload.get("dynamic_model_state_sha256", {})) != modes
        or len(payload.get("inner_p2_model_state_sha256", {})) != 20
        or len(payload.get("inner_dynamic_model_state_sha256", {})) != 20
        or type(payload.get("files")) is not dict
    ):
        raise NeuralProbeArtifactError("N2 worker contract changed")
    actual_files = {
        path.relative_to(worker).as_posix()
        for path in worker.rglob("*")
        if path.is_file() and path.name != "complete.json"
    }
    if actual_files != set(payload["files"]):
        raise NeuralProbeArtifactError("N2 worker file roster changed")
    for name, digest in payload["files"].items():
        if type(digest) is not str or _sha256(worker / name) != digest:
            raise NeuralProbeArtifactError(f"N2 worker checksum mismatch: {name}")
    return payload


def _dynamic_bootstrap_summary(bootstrap: pl.DataFrame) -> dict[str, object]:
    output: dict[str, object] = {}
    for control in sorted(bootstrap.get_column("control_mode").unique()):
        table = bootstrap.filter(pl.col("control_mode") == control)
        regret = table.get_column("control_minus_reference_regret").to_numpy()
        utility = table.get_column("reference_minus_control_utility").to_numpy()
        output[control] = {
            "mean_control_minus_real_regret": float(
                np.mean(regret, dtype=np.float64)
            ),
            "mean_real_minus_control_utility": float(
                np.mean(utility, dtype=np.float64)
            ),
            "regret_ci95_lower": float(np.quantile(regret, 0.025)),
            "regret_ci95_upper": float(np.quantile(regret, 0.975)),
            "utility_ci95_lower": float(np.quantile(utility, 0.025)),
            "utility_ci95_upper": float(np.quantile(utility, 0.975)),
        }
    return output


def finalize_n2_dynamic_p3(
    bank: MRISFeatureBank,
    *,
    states: pl.DataFrame,
    worker_root: str | Path,
    n1_p2_root: str | Path,
    frozen_p3_root: str | Path,
    output_root: str | Path,
    source_config_path: str | Path,
    base_commit: str,
    config_sha256: str,
    bootstrap_replicates: int,
    seed: int,
) -> Path:
    if (
        type(bank) is not MRISFeatureBank
        or len(bank.domain_order) != 6
        or not isinstance(states, pl.DataFrame)
        or type(bootstrap_replicates) is not int
        or bootstrap_replicates <= 0
        or type(seed) is not int
    ):
        raise NeuralProbeArtifactError("N2 finalization request is invalid")
    workers = Path(worker_root)
    n1_root = Path(n1_p2_root)
    frozen_root = Path(frozen_p3_root)
    destination = Path(output_root)
    source_config = Path(source_config_path)
    if _sha256(source_config) != config_sha256:
        raise NeuralProbeArtifactError("N2 source config hash changed")
    n1_manifest = verify_artifact_integrity(n1_root)
    if n1_manifest.get("artifact") != "mavis_neural_probe_n1_spatial_p2":
        raise NeuralProbeArtifactError("N2 spatial P2 authority changed")
    frozen_manifest = verify_dynamic_package(frozen_root)
    if destination.exists():
        raise NeuralProbeArtifactError("N2 output already exists")
    decision_state_ids, decision_counts = _decision_state_roster(
        states,
        domain_order=bank.domain_order,
        feature_state_ids=bank.state_ids,
    )
    action_parts: list[pl.DataFrame] = []
    metric_parts: list[pl.DataFrame] = []
    audit_parts: list[pl.DataFrame] = []
    worker_payloads: list[dict[str, object]] = []
    for domain in bank.domain_order:
        worker = workers / domain
        worker_payloads.append(
            _load_spatial_dynamic_worker(
                worker,
                outer_domain=domain,
                bank=bank,
                target_group_count=decision_counts[domain],
                base_commit=base_commit,
                config_sha256=config_sha256,
            )
        )
        action_parts.append(pl.read_parquet(worker / "action_scores.parquet"))
        metric_parts.append(pl.read_parquet(worker / "state_metrics.parquet"))
        audit_parts.append(pl.read_parquet(worker / "model_selection_audit.parquet"))
    action_scores = pl.concat(action_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "state_id", "mode", "candidate_index"]
    )
    state_metrics = pl.concat(metric_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "state_id", "mode"]
    )
    audits = pl.concat(audit_parts, how="vertical_relaxed").sort(
        ["outer_domain", "mode", "record_type", "validation_domain"],
        nulls_last=True,
    )
    _validate_dynamic_tables(
        action_scores,
        state_metrics,
        audits,
        decision_state_ids=decision_state_ids,
        domain_count=len(bank.domain_order),
    )
    candidate_metrics = aggregate_dynamic_metrics(state_metrics)
    bootstrap_seed = seed + 300
    control_bootstrap = bootstrap_dynamic_contrasts(
        candidate_metrics.per_specimen,
        reference_mode="real",
        control_modes=("static", "positions_only", "shuffled"),
        domain_order=bank.domain_order,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    deepsets_metrics = pl.read_parquet(frozen_root / "state_metrics.parquet")
    comparison = evaluate_n2_comparison(
        spatial_state_metrics=state_metrics,
        deepsets_state_metrics=deepsets_metrics,
        domain_order=bank.domain_order,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    spatial_real = candidate_metrics.aggregate.filter(pl.col("mode") == "real").row(
        0, named=True
    )
    deepsets_real = comparison.metric_tables.aggregate.filter(
        pl.col("mode") == "deepsets_real"
    ).row(0, named=True)
    control_points: dict[str, object] = {}
    for row in candidate_metrics.aggregate.filter(pl.col("mode") != "real").to_dicts():
        control_points[row["mode"]] = {
            "control_minus_real_regret": float(
                row["next_action_regret"] - spatial_real["next_action_regret"]
            ),
            "real_minus_control_utility": float(
                spatial_real["one_step_cai_utility"] - row["one_step_cai_utility"]
            ),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".n2_dynamic_p3.", dir=destination.parent))
    try:
        action_scores.write_parquet(
            temporary / "action_scores.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        state_metrics.write_parquet(
            temporary / "state_metrics.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        audits.write_parquet(
            temporary / "model_selection_audit.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        candidate_metrics.per_specimen.write_csv(
            temporary / "per_specimen_metrics.csv"
        )
        candidate_metrics.per_domain.write_csv(
            temporary / "candidate_domain_metrics.csv"
        )
        candidate_metrics.aggregate.write_csv(temporary / "aggregate_metrics.csv")
        comparison.domain_metrics.write_csv(temporary / "domain_metrics.csv")
        comparison.bootstrap.write_csv(temporary / "bootstrap.csv")
        control_bootstrap.write_csv(temporary / "control_bootstrap.csv")
        shutil.copyfile(source_config, temporary / "config.json")
        checkpoint_root = temporary / "checkpoints"
        inner_root = checkpoint_root / "inner"
        inner_root.mkdir(parents=True)
        for outer_domain in bank.domain_order:
            for mode in ("static", "positions_only", "real", "shuffled"):
                shutil.copyfile(
                    workers / outer_domain / "checkpoints" / f"{mode}.npz",
                    checkpoint_root / f"{outer_domain}__{mode}.npz",
                )
                for validation_domain in bank.domain_order:
                    if validation_domain == outer_domain:
                        continue
                    shutil.copyfile(
                        workers
                        / outer_domain
                        / "checkpoints"
                        / "inner"
                        / f"{validation_domain}__{mode}.npz",
                        inner_root
                        / f"{outer_domain}__{validation_domain}__{mode}.npz",
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
            raise NeuralProbeArtifactError("N2 runtime Git state is unavailable") from error
        summary = {
            "architecture_name": "spatial_grid_cnn_v1",
            "base_commit": base_commit,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "canonical_contrast": "deepsets_regret_minus_spatial_regret",
            "ci95_lower": comparison.ci95_lower,
            "ci95_upper": comparison.ci95_upper,
            "config_sha256": config_sha256,
            "control_contrasts": control_points,
            "control_bootstrap": _dynamic_bootstrap_summary(control_bootstrap),
            "deepsets_real_next_action_regret": float(
                deepsets_real["next_action_regret"]
            ),
            "deepsets_real_one_step_cai_utility": float(
                deepsets_real["one_step_cai_utility"]
            ),
            "dynamic_scorer": "DynamicActionScorer",
            "favorable_domain_count": comparison.favorable_domain_count,
            "feature_bank_input_state_sha256": bank.input_state_sha256,
            "feature_bank_target_state_sha256": bank.target_state_sha256,
            "frozen_p3_manifest_sha256": _sha256(
                frozen_root / "artifact_manifest.json"
            ),
            "frozen_p3_verified": frozen_manifest.get("artifact")
            == "mavis_p3_dynamic_voi",
            "gate": comparison.gate,
            "n1_manifest_sha256": _sha256(n1_root / "artifact_manifest.json"),
            "point_estimate": comparison.point_estimate,
            "runtime_head": runtime_head,
            "schema_version": 1,
            "seed": seed,
            "spatial_real_next_action_regret": float(
                spatial_real["next_action_regret"]
            ),
            "spatial_real_one_step_cai_utility": float(
                spatial_real["one_step_cai_utility"]
            ),
            "stage": "N2_DYNAMIC_P3",
            "state_count": len(decision_state_ids),
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "target_data_used_for_selection": False,
            "worker_dynamic_model_states": [
                payload["dynamic_model_state_sha256"] for payload in worker_payloads
            ],
        }
        _write_json(temporary / "summary.json", summary)
        (temporary / "REPORT.md").write_text(
            "# N2 Spatial Dynamic-Value Probe\n\n"
            f"Gate: `{comparison.gate}`.\n\n"
            f"DeepSets real next-action regret: "
            f"`{float(deepsets_real['next_action_regret']):.10f}`. Spatial real "
            f"next-action regret: `{float(spatial_real['next_action_regret']):.10f}`. "
            "The registered DeepSets-minus-Spatial regret contrast is "
            f"`{comparison.point_estimate:.10f}` with paired 95% CI "
            f"`[{comparison.ci95_lower:.10f}, {comparison.ci95_upper:.10f}]` and "
            f"favorable direction in `{comparison.favorable_domain_count}/6` "
            "held-out domains. Positive values favor Spatial.\n\n"
            "The DynamicActionScorer, candidate descriptors, loss weights, teacher, "
            "metrics, and bootstrap are unchanged. Only the legal 64-D P2 embedding "
            "provider is spatial. Selection and fitting remain source-only; target "
            "teacher values are retrospective evaluation data.\n",
            encoding="utf-8",
        )
        write_artifact_integrity(
            temporary,
            artifact="mavis_neural_probe_n2_dynamic_p3",
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
    "N1Comparison",
    "N2Comparison",
    "NeuralProbeArtifactError",
    "assign_directional_gate",
    "evaluate_n1_comparison",
    "evaluate_n2_comparison",
    "finalize_n1_spatial_p2",
    "finalize_n2_dynamic_p3",
    "verify_artifact_integrity",
    "write_artifact_integrity",
]
