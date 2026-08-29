"""Frozen-rollout N4 execution and registered end-to-end comparison."""

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

from ..authority import MAVISAuthority
from ..closed_loop_execution import _rollout_action_rows, evaluate_inspection_curve
from ..closed_loop_metrics import (
    ClosedLoopMetricTables,
    bootstrap_closed_loop_contrasts,
    evaluate_closed_loop_predictions,
)
from ..config import MAVISConfig
from ..dynamic_training import load_fitted_dynamic_checkpoint
from ..final_package import verify_final_package
from ..mris_data import MRISFeatureBank
from ..rollout import rollout_scout_and_focus_curve
from .artifacts import (
    assign_directional_gate,
    verify_artifact_integrity,
    write_artifact_integrity,
)
from .policy import SpatialProbeDeployedScorer
from .training import load_fitted_spatial_mris_checkpoint


class NeuralProbeClosedLoopError(RuntimeError):
    """Raised when N4 changes the registered rollout or evidence contract."""


_CANDIDATE = "spatial_probe"
_STATIC_REFERENCE = "mvd_m1_o2"
_LEARNED_REFERENCE = "mavis_full"
_METHOD_ORDER = (_CANDIDATE, _STATIC_REFERENCE, _LEARNED_REFERENCE)


@dataclass(frozen=True, slots=True)
class N4Comparison:
    point_estimate: float
    ci95_lower: float
    ci95_upper: float
    favorable_domain_count: int
    gate: str
    domain_metrics: pl.DataFrame
    bootstrap: pl.DataFrame
    metric_tables: ClosedLoopMetricTables


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


def evaluate_n4_comparison(
    predictions: pl.DataFrame,
    *,
    domain_order: tuple[str, ...],
    checkpoints: tuple[float, ...],
    bootstrap_replicates: int,
    seed: int,
) -> N4Comparison:
    if (
        not isinstance(predictions, pl.DataFrame)
        or predictions.height == 0
        or type(domain_order) is not tuple
        or len(domain_order) != 6
        or type(checkpoints) is not tuple
        or not checkpoints
        or type(bootstrap_replicates) is not int
        or bootstrap_replicates <= 0
        or type(seed) is not int
    ):
        raise NeuralProbeClosedLoopError("N4 comparison request is invalid")
    metrics = evaluate_closed_loop_predictions(
        predictions,
        domain_order=domain_order,
        method_order=_METHOD_ORDER,
        checkpoints=checkpoints,
    )
    bootstrap = bootstrap_closed_loop_contrasts(
        metrics.specimen_auebc,
        reference_method=_CANDIDATE,
        control_methods=(_STATIC_REFERENCE, _LEARNED_REFERENCE),
        domain_order=domain_order,
        replicates=bootstrap_replicates,
        seed=seed + 400,
    )
    candidate = metrics.domain_auebc.filter(pl.col("method") == _CANDIDATE).select(
        "outer_domain", pl.col("cai_auebc").alias("spatial_probe_auebc")
    )
    static = metrics.domain_auebc.filter(
        pl.col("method") == _STATIC_REFERENCE
    ).select("outer_domain", pl.col("cai_auebc").alias("static_reference_auebc"))
    learned = metrics.domain_auebc.filter(
        pl.col("method") == _LEARNED_REFERENCE
    ).select("outer_domain", pl.col("cai_auebc").alias("learned_reference_auebc"))
    domains = (
        candidate.join(static, on="outer_domain", how="inner")
        .join(learned, on="outer_domain", how="inner")
        .with_columns(
            (
                pl.col("static_reference_auebc")
                - pl.col("spatial_probe_auebc")
            ).alias("static_minus_spatial_probe_auebc"),
            (
                pl.col("learned_reference_auebc")
                - pl.col("spatial_probe_auebc")
            ).alias("learned_minus_spatial_probe_auebc"),
        )
        .with_columns(
            (pl.col("static_minus_spatial_probe_auebc") > 0.0).alias("favorable")
        )
        .sort("outer_domain")
    )
    if domains.height != 6 or set(domains.get_column("outer_domain")) != set(
        domain_order
    ):
        raise NeuralProbeClosedLoopError("N4 domain roster is incomplete")
    primary_draws = bootstrap.filter(
        pl.col("control_method") == _STATIC_REFERENCE
    ).get_column("control_minus_reference_cai_auebc").to_numpy()
    point = float(domains.get_column("static_minus_spatial_probe_auebc").mean())
    lower = float(np.quantile(primary_draws, 0.025))
    upper = float(np.quantile(primary_draws, 0.975))
    favorable = int(domains.get_column("favorable").sum())
    return N4Comparison(
        point_estimate=point,
        ci95_lower=lower,
        ci95_upper=upper,
        favorable_domain_count=favorable,
        gate=assign_directional_gate(
            prefix="END_TO_END",
            point_estimate=point,
            ci95_lower=lower,
            ci95_upper=upper,
            favorable_domain_count=favorable,
        ),
        domain_metrics=domains,
        bootstrap=bootstrap,
        metric_tables=metrics,
    )


def run_spatial_closed_loop_outer_domain(
    authority: MAVISAuthority,
    config: MAVISConfig,
    bank: MRISFeatureBank,
    *,
    outer_domain: str,
    p2_checkpoint_root: str | Path,
    p3_checkpoint_root: str | Path,
    output_root: str | Path,
    device: str,
    base_commit: str,
    config_sha256: str,
) -> Path:
    if (
        type(authority) is not MAVISAuthority
        or type(config) is not MAVISConfig
        or type(bank) is not MRISFeatureBank
        or outer_domain not in config.domain_order
        or bank.domain_order != config.domain_order
        or type(device) is not str
        or not device
        or type(base_commit) is not str
        or len(base_commit) != 40
        or type(config_sha256) is not str
        or len(config_sha256) != 64
    ):
        raise NeuralProbeClosedLoopError("N4 outer worker request is invalid")
    config.require_finalized()
    p2 = load_fitted_spatial_mris_checkpoint(
        Path(p2_checkpoint_root) / f"{outer_domain}__real.npz",
        expected_base_commit=base_commit,
        expected_feature_bank_input_sha256=bank.input_state_sha256,
        expected_feature_bank_target_sha256=bank.target_state_sha256,
        expected_config_sha256=config_sha256,
    )
    p3 = load_fitted_dynamic_checkpoint(
        Path(p3_checkpoint_root) / f"{outer_domain}__real.npz"
    )
    if (
        p2.outer_domain != outer_domain
        or p2.mode != "real"
        or p3.outer_domain != outer_domain
        or p2.mris_dimension != p3.mris_dimension
        or outer_domain in p2.audit.fit_domains
        or outer_domain in p3.audit.fit_domains
    ):
        raise NeuralProbeClosedLoopError("N4 outer-fold model changed")
    scorer = SpatialProbeDeployedScorer(
        mris_model=p2,
        dynamic_model=p3,
        device=device,
    )
    target_ids = tuple(
        sorted(
            specimen_id
            for specimen_id, domain in zip(
                authority.specimen_ids,
                authority.dataset_ids,
                strict=True,
            )
            if domain == outer_domain
        )
    )
    if not target_ids:
        raise NeuralProbeClosedLoopError("N4 target roster is empty")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / outer_domain
    if destination.exists():
        raise NeuralProbeClosedLoopError("N4 outer output already exists")
    temporary = Path(tempfile.mkdtemp(prefix=f".{outer_domain}.", dir=root))
    try:
        prediction_rows: list[dict[str, object]] = []
        trajectory_rows: list[dict[str, object]] = []
        for specimen_id in target_ids:
            curve = rollout_scout_and_focus_curve(
                authority,
                specimen_id=specimen_id,
                initial_budget=float(config.initial_budget_by_domain[outer_domain]),
                checkpoints=config.checkpoints,
                scorer=scorer,
                objective="direct_cost_aware",
                feedback=True,
            )
            prediction_rows.extend(
                evaluate_inspection_curve(
                    authority,
                    outer_domain=outer_domain,
                    method=_CANDIDATE,
                    checkpoints=config.checkpoints,
                    states=curve.checkpoint_states,
                    cai_evaluator=p2,
                    device=device,
                )
            )
            trajectory_rows.extend(
                _rollout_action_rows(
                    curve,
                    outer_domain=outer_domain,
                    method=_CANDIDATE,
                )
            )
        predictions = pl.DataFrame(prediction_rows, infer_schema_length=None).sort(
            ["outer_domain", "specimen_id", "nominal_checkpoint"]
        )
        trajectories = pl.DataFrame(trajectory_rows, infer_schema_length=None).sort(
            ["outer_domain", "specimen_id", "step"]
        )
        if (
            predictions.height != len(target_ids) * len(config.checkpoints)
            or predictions.get_column("specimen_id").n_unique() != len(target_ids)
            or set(predictions.get_column("method")) != {_CANDIDATE}
            or trajectories.height == 0
        ):
            raise NeuralProbeClosedLoopError("N4 outer output roster is incomplete")
        predictions.write_parquet(
            temporary / "predictions.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        trajectories.write_parquet(
            temporary / "trajectories.parquet",
            compression="zstd",
            compression_level=9,
            statistics=True,
        )
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        _write_json(
            temporary / "complete.json",
            {
                "architecture_name": "spatial_grid_cnn_v1",
                "base_commit": base_commit,
                "cai_model_state_sha256": p2.model_state_sha256,
                "config_sha256": config_sha256,
                "dynamic_model_state_sha256": p3.model_state_sha256,
                "dynamic_scorer": "DynamicActionScorer",
                "feature_bank_input_state_sha256": bank.input_state_sha256,
                "feature_bank_target_state_sha256": bank.target_state_sha256,
                "files": {
                    path.relative_to(temporary).as_posix(): _sha256(path)
                    for path in files
                },
                "frozen_mavis_config_sha256": config.config_sha256,
                "future_target_content_used_by_policy": False,
                "method": _CANDIDATE,
                "objective": "direct_cost_aware",
                "outer_domain": outer_domain,
                "prediction_count": predictions.height,
                "rollout": "rollout_scout_and_focus_curve",
                "schema_version": 1,
                "target_specimen_count": len(target_ids),
                "target_true_cai_used_by_policy": False,
                "trajectory_row_count": trajectories.height,
            },
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "complete.json"


def _load_worker(
    root: Path,
    *,
    outer_domain: str,
    target_specimen_count: int,
    checkpoint_count: int,
    base_commit: str,
    config_sha256: str,
    frozen_config_sha256: str,
    bank: MRISFeatureBank,
) -> dict[str, object]:
    try:
        payload = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NeuralProbeClosedLoopError("N4 worker summary is invalid") from error
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("architecture_name") != "spatial_grid_cnn_v1"
        or payload.get("dynamic_scorer") != "DynamicActionScorer"
        or payload.get("rollout") != "rollout_scout_and_focus_curve"
        or payload.get("objective") != "direct_cost_aware"
        or payload.get("method") != _CANDIDATE
        or payload.get("outer_domain") != outer_domain
        or payload.get("base_commit") != base_commit
        or payload.get("config_sha256") != config_sha256
        or payload.get("frozen_mavis_config_sha256") != frozen_config_sha256
        or payload.get("feature_bank_input_state_sha256") != bank.input_state_sha256
        or payload.get("feature_bank_target_state_sha256") != bank.target_state_sha256
        or payload.get("target_specimen_count") != target_specimen_count
        or payload.get("prediction_count")
        != target_specimen_count * checkpoint_count
        or payload.get("target_true_cai_used_by_policy") is not False
        or payload.get("future_target_content_used_by_policy") is not False
        or type(payload.get("files")) is not dict
    ):
        raise NeuralProbeClosedLoopError("N4 worker contract changed")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "complete.json"
    }
    if actual != set(payload["files"]):
        raise NeuralProbeClosedLoopError("N4 worker file roster changed")
    for name, digest in payload["files"].items():
        if type(digest) is not str or _sha256(root / name) != digest:
            raise NeuralProbeClosedLoopError(f"N4 worker checksum mismatch: {name}")
    return payload


def finalize_n4_closed_loop(
    bank: MRISFeatureBank,
    config: MAVISConfig,
    *,
    worker_root: str | Path,
    n1_p2_root: str | Path,
    n2_p3_root: str | Path,
    frozen_p7_root: str | Path,
    output_root: str | Path,
    source_config_path: str | Path,
    base_commit: str,
    config_sha256: str,
    bootstrap_replicates: int,
    seed: int,
) -> Path:
    if type(bank) is not MRISFeatureBank or type(config) is not MAVISConfig:
        raise NeuralProbeClosedLoopError("N4 finalization request is invalid")
    config.require_finalized()
    workers = Path(worker_root)
    n1_root = Path(n1_p2_root)
    n2_root = Path(n2_p3_root)
    p7_root = Path(frozen_p7_root)
    destination = Path(output_root)
    source_config = Path(source_config_path)
    if _sha256(source_config) != config_sha256 or destination.exists():
        raise NeuralProbeClosedLoopError("N4 config or output contract changed")
    n1_manifest = verify_artifact_integrity(n1_root)
    n2_manifest = verify_artifact_integrity(n2_root)
    p7_manifest = verify_final_package(p7_root)
    if (
        n1_manifest.get("artifact") != "mavis_neural_probe_n1_spatial_p2"
        or n2_manifest.get("artifact") != "mavis_neural_probe_n2_dynamic_p3"
        or p7_manifest.get("artifact") != "mavis_p7_final_frozen_eval"
        or p7_manifest.get("config_sha256") != config.config_sha256
    ):
        raise NeuralProbeClosedLoopError("N4 source artifact identity changed")
    prediction_parts: list[pl.DataFrame] = []
    trajectory_parts: list[pl.DataFrame] = []
    worker_payloads: list[dict[str, object]] = []
    for domain in config.domain_order:
        target_count = len(
            {
                specimen
                for specimen, state_domain in zip(
                    bank.specimen_ids,
                    bank.domain_ids,
                    strict=True,
                )
                if state_domain == domain
            }
        )
        worker = workers / domain
        worker_payloads.append(
            _load_worker(
                worker,
                outer_domain=domain,
                target_specimen_count=target_count,
                checkpoint_count=len(config.checkpoints),
                base_commit=base_commit,
                config_sha256=config_sha256,
                frozen_config_sha256=config.config_sha256,
                bank=bank,
            )
        )
        prediction_parts.append(pl.read_parquet(worker / "predictions.parquet"))
        trajectory_parts.append(pl.read_parquet(worker / "trajectories.parquet"))
    candidate = pl.concat(prediction_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "nominal_checkpoint"]
    )
    trajectories = pl.concat(trajectory_parts, how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "step"]
    )
    frozen = pl.read_parquet(p7_root / "closed_loop_predictions.parquet").filter(
        pl.col("method").is_in((_STATIC_REFERENCE, _LEARNED_REFERENCE))
    )
    predictions = pl.concat((candidate, frozen), how="vertical_relaxed").sort(
        ["outer_domain", "specimen_id", "method", "nominal_checkpoint"]
    )
    comparison = evaluate_n4_comparison(
        predictions,
        domain_order=config.domain_order,
        checkpoints=config.checkpoints,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    aggregate = {
        row["method"]: float(row["domain_balanced_cai_auebc"])
        for row in comparison.metric_tables.aggregate_auebc.to_dicts()
    }
    try:
        runtime_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise NeuralProbeClosedLoopError("N4 runtime Git state is unavailable") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".n4_closed_loop.", dir=destination.parent))
    try:
        predictions.write_parquet(
            temporary / "closed_loop_predictions.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        trajectories.write_parquet(
            temporary / "action_trajectories.parquet",
            compression="zstd",
            compression_level=12,
            statistics=True,
        )
        comparison.metric_tables.per_specimen_curve.write_csv(
            temporary / "per_specimen_curves.csv"
        )
        comparison.metric_tables.domain_curve.write_csv(
            temporary / "domain_curves.csv"
        )
        comparison.metric_tables.aggregate_curve.write_csv(
            temporary / "aggregate_curves.csv"
        )
        comparison.metric_tables.specimen_auebc.write_csv(
            temporary / "per_specimen_auebc.csv"
        )
        comparison.domain_metrics.write_csv(temporary / "domain_metrics.csv")
        comparison.metric_tables.aggregate_auebc.write_csv(
            temporary / "aggregate_auebc.csv"
        )
        comparison.bootstrap.write_csv(temporary / "bootstrap.csv")
        shutil.copyfile(source_config, temporary / "config.json")
        summary = {
            "architecture_name": "spatial_grid_cnn_v1",
            "base_commit": base_commit,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed + 400,
            "canonical_contrast": "static_reference_minus_spatial_probe_auebc",
            "ci95_lower": comparison.ci95_lower,
            "ci95_upper": comparison.ci95_upper,
            "config_sha256": config_sha256,
            "favorable_domain_count": comparison.favorable_domain_count,
            "frozen_learned_auebc": aggregate[_LEARNED_REFERENCE],
            "frozen_mavis_config_sha256": config.config_sha256,
            "frozen_p7_manifest_sha256": _sha256(
                p7_root / "artifact_manifest.json"
            ),
            "frozen_static_reference": _STATIC_REFERENCE,
            "frozen_static_reference_auebc": aggregate[_STATIC_REFERENCE],
            "gate": comparison.gate,
            "n1_manifest_sha256": _sha256(n1_root / "artifact_manifest.json"),
            "n2_manifest_sha256": _sha256(n2_root / "artifact_manifest.json"),
            "new_candidate_auebc": aggregate[_CANDIDATE],
            "new_candidate_better_than_frozen_learned": (
                aggregate[_CANDIDATE] < aggregate[_LEARNED_REFERENCE]
            ),
            "point_estimate": comparison.point_estimate,
            "runtime_head": runtime_head,
            "schema_version": 1,
            "seed": seed,
            "stage": "N4_CLOSED_LOOP",
            "statistical_units": ["physical_specimen", "held_out_domain"],
            "target_data_used_for_training_or_selection": False,
            "worker_model_states": [
                {
                    "p2": payload["cai_model_state_sha256"],
                    "p3": payload["dynamic_model_state_sha256"],
                }
                for payload in worker_payloads
            ],
        }
        _write_json(temporary / "summary.json", summary)
        (temporary / "REPORT.md").write_text(
            "# N4 Spatial Closed-Loop Probe\n\n"
            f"Gate: `{comparison.gate}`.\n\n"
            f"Frozen static reference AUEBC: `{aggregate[_STATIC_REFERENCE]:.10f}`. "
            f"Spatial candidate AUEBC: `{aggregate[_CANDIDATE]:.10f}`. The registered "
            f"static-minus-candidate contrast is `{comparison.point_estimate:.10f}` "
            f"with paired 95% CI `[{comparison.ci95_lower:.10f}, "
            f"{comparison.ci95_upper:.10f}]` and favorable direction in "
            f"`{comparison.favorable_domain_count}/6` held-out domains. Positive "
            "values favor the candidate.\n\n"
            f"The frozen current learned implementation AUEBC is "
            f"`{aggregate[_LEARNED_REFERENCE]:.10f}`. The existing uniform scout, "
            "8x8 action grid, acquisition levels, exact native-raster cost, candidate "
            "generation, reveal action, direct-cost-aware objective, rollout, CAI "
            "metrics, and bootstrap are unchanged. No target outcome was used for "
            "training or selection.\n",
            encoding="utf-8",
        )
        write_artifact_integrity(
            temporary,
            artifact="mavis_neural_probe_n4_closed_loop",
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
    "N4Comparison",
    "NeuralProbeClosedLoopError",
    "evaluate_n4_comparison",
    "finalize_n4_closed_loop",
    "run_spatial_closed_loop_outer_domain",
]
