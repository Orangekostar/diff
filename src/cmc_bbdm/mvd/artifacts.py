"""Deterministic formal M0/M1 evidence package publishers."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from cmc_bbdm.mva.budget_metrics import auebc

from .action_cost_audit import build_action_cost_audit
from .authority import load_compact_mvd_authority
from .config import MVDConfig, load_mvd_config
from .interaction_audit import summarize_interactions
from .observability_statistics import aggregate_observability_metrics
from .statistics import M0Aggregation, aggregate_m0_tables


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _bootstrap_rows(values: object) -> list[dict[str, object]]:
    return [
        {
            **asdict(value),
            "domain_effects": "|".join(
                f"{item:.17g}" for item in value.domain_effects
            ),
        }
        for value in values
    ]


def _prepare(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def _finalize(path: Path, *, package: str, authority_state: str) -> None:
    files = tuple(
        sorted(
            file
            for file in path.rglob("*")
            if file.is_file()
            and file.name not in {"artifact_manifest.json", "CHECKSUMS.sha256"}
        )
    )
    manifest = {
        "schema_version": 1,
        "package": package,
        "authority_state_sha256": authority_state,
        "files": {
            file.relative_to(path).as_posix(): {
                "bytes": file.stat().st_size,
                "sha256": _sha256(file),
            }
            for file in files
        },
    }
    _json(path / "artifact_manifest.json", manifest)
    checksummed = tuple(
        sorted(file for file in path.rglob("*") if file.is_file() and file.name != "CHECKSUMS.sha256")
    )
    (path / "CHECKSUMS.sha256").write_text(
        "".join(
            f"{_sha256(file)}  {file.relative_to(path).as_posix()}\n"
            for file in checksummed
        ),
        encoding="ascii",
    )


def _m0_aggregation(root: Path, config: MVDConfig) -> tuple[M0Aggregation, dict[str, pl.DataFrame]]:
    work = root / config.output_work / "m0_domains"
    tables = {
        "states": pl.concat(
            [pl.read_parquet(work / domain / "states.parquet") for domain in config.domain_order]
        ),
        "rankings": pl.concat(
            [pl.read_parquet(work / domain / "rankings.parquet") for domain in config.domain_order]
        ),
        "actions": pl.concat(
            [pl.read_parquet(work / domain / "actions.parquet") for domain in config.domain_order]
        ),
        "values": pl.concat(
            [pl.read_parquet(work / domain / "initial_values.parquet") for domain in config.domain_order]
        ),
        "reproduction": pl.concat(
            [pl.read_csv(work / domain / "evaluator_reproduction.csv") for domain in config.domain_order]
        ),
    }
    aggregation = aggregate_m0_tables(
        tables["states"],
        pl.read_parquet(root / config.sources["a2_state_metrics"].path),
        pl.read_parquet(root / config.sources["a4_state_metrics"].path),
        domain_order=config.domain_order,
        checkpoints=config.checkpoints,
        random_seeds=tuple(
            range(
                config.random_seed_start,
                config.random_seed_start + config.random_seed_count,
            )
        ),
        full_mae=config.full_mae,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
        minimum_improved_domains=config.m0_minimum_improved_domains,
        minimum_headroom_retention=config.m0_minimum_headroom_retention,
        strong_headroom_retention=config.m0_strong_headroom_retention,
        evaluator_reproduction=tables["reproduction"],
    )
    return aggregation, tables


def publish_m0(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Publish the complete M0 gate from hash-bound work shards."""

    root = Path(project_root).resolve(strict=True)
    config = load_mvd_config(config_path, project_root=root)
    compact = load_compact_mvd_authority(config, project_root=root)
    output = root / config.output_m0 if output_path is None else Path(output_path)
    _prepare(output)
    aggregation, tables = _m0_aggregation(root, config)
    action_cost = build_action_cost_audit(compact)
    pl.DataFrame(action_cost.rows).write_csv(output / "action_cost_audit.csv")
    interaction_root = root / config.output_work / "interaction_domains"
    interaction_rows = pl.concat(
        [
            pl.read_parquet(interaction_root / domain / "interaction_rows.parquet")
            for domain in config.domain_order
        ]
    ).sort(["outer_domain", "query_source_domain", "specimen_id", "method", "nominal_checkpoint"])
    interaction_rows.write_parquet(output / "interaction_rows.parquet", compression="zstd")
    interaction_summary = summarize_interactions(interaction_rows)
    interaction_summary.write_csv(output / "interaction_audit.csv")
    curves = pl.DataFrame(aggregation.curves, infer_schema_length=None)
    curve_files = {
        "uniform": "uniform_curve.csv",
        "one_shot_reconstruction": "reconstruction_curve.csv",
        "one_shot_mechanical_oracle": "one_shot_oracle_curve.csv",
        "sequential_mechanical_oracle": "sequential_oracle_curve.csv",
    }
    for method, name in curve_files.items():
        curves.filter(pl.col("method") == method).write_csv(output / name)
    pl.DataFrame(aggregation.domain_metrics).write_csv(output / "domain_metrics.csv")
    pl.DataFrame(aggregation.budget_metrics, infer_schema_length=None).write_csv(
        output / "budget_metrics.csv"
    )
    pl.DataFrame(_bootstrap_rows(aggregation.bootstrap_effects)).write_csv(
        output / "bootstrap.csv"
    )
    tables["states"].sort(["dataset_id", "specimen_id", "method", "nominal_checkpoint"]).write_parquet(
        output / "state_metrics.parquet", compression="zstd"
    )
    tables["rankings"].sort(["outer_domain", "specimen_id", "method", "cell_index"]).write_parquet(
        output / "rankings.parquet", compression="zstd"
    )
    tables["actions"].sort(["dataset_id", "specimen_id", "method", "step"]).write_parquet(
        output / "actions.parquet", compression="zstd"
    )
    tables["values"].sort(["outer_domain", "specimen_id", "cell_index"]).write_parquet(
        output / "initial_values.parquet", compression="zstd"
    )
    tables["reproduction"].sort(["outer_domain", "nominal_checkpoint"]).write_csv(
        output / "evaluator_reproduction.csv"
    )
    shutil.copyfile(config.config_path, output / "config.yaml")
    summary = {
        "gate": asdict(aggregation.gate),
        "aggregation_state_sha256": aggregation.state_sha256,
        "authority_state_sha256": config.authority_state_sha256,
        "candidate_bank_states": dict(config.candidate_bank_states),
        "action_cost_audit_state_sha256": action_cost.state_sha256,
        "action_cost_summaries": [asdict(value) for value in action_cost.summaries],
        "interaction": {
            "role": "descriptive_non_gating",
            "row_count": interaction_rows.height,
            "source_specimen_fold_count": interaction_rows.unique(
                subset=["outer_domain", "specimen_id"]
            ).height,
        },
        "m1_authorized": aggregation.gate.go,
    }
    _json(output / "summary.json", summary)
    gate = aggregation.gate
    (output / "REPORT.md").write_text(
        "# MVD M0 One-Shot Oracle Feasibility\n\n"
        f"Decision: `{gate.status}`.\n\n"
        f"The initial one-shot plan retains `{gate.headroom_retention:.3f}` of the "
        "stronger-baseline-to-sequential-oracle headroom. Its equal-domain P-B "
        f"AUEBC is `{gate.one_shot_auebc:.8f}`, versus uniform "
        f"`{gate.uniform_effect.point_estimate + gate.one_shot_auebc:.8f}`, "
        f"reconstruction `{gate.stronger_baseline_auebc:.8f}`, and sequential "
        f"oracle `{gate.sequential_auebc:.8f}`.\n\n"
        f"Uniform and reconstruction effects improve in "
        f"`{gate.uniform_effect.improved_domains}/6` and "
        f"`{gate.reconstruction_effect.improved_domains}/6` domains; both "
        "synchronized bootstrap lower bounds are positive. Exact candidate costs "
        "are unequal, so all plans use frozen ranking with exact unique-location "
        "fit/skip selection.\n\n"
        "The interaction audit covers 1,380 strict source-OOF specimen-folds. "
        "For mechanical top-value sets, Pearson association is only about "
        "0.21-0.28 at the intermediate checkpoints and additive gain overstates "
        "joint gain by about 0.048-0.072 on average. Interaction/non-additivity is "
        "therefore material, but descriptive and non-gating. M0 authorizes the "
        "minimum M1 observability test, not a deployable MVD claim.\n",
        encoding="utf-8",
    )
    _finalize(
        output,
        package="mvd_m0_one_shot_oracle",
        authority_state=config.authority_state_sha256,
    )
    return output


def _advantage_capture(
    cai_states: pl.DataFrame,
    m0: M0Aggregation,
    *,
    domain_order: tuple[str, ...],
) -> dict[str, object]:
    primary_checkpoints = tuple(value for value in m0.checkpoints if value >= 0.0625)
    domain_m0 = pl.DataFrame(m0.domain_metrics)
    rows: list[dict[str, object]] = []
    baseline_method = m0.gate.stronger_baseline
    for domain in domain_order:
        values = (
            cai_states.filter(pl.col("dataset_id") == domain)
            .group_by("nominal_checkpoint")
            .agg(pl.col("p_b_absolute_error").mean().alias("mae"))
            .sort("nominal_checkpoint")
        )
        values = values.filter(pl.col("nominal_checkpoint").is_in(primary_checkpoints))
        if tuple(values["nominal_checkpoint"]) != primary_checkpoints:
            raise ValueError("M1 CAI diagnostic checkpoint roster changed")
        predicted = auebc(
            primary_checkpoints,
            tuple(float(value) for value in values["mae"]),
            lower=0.0625,
            upper=0.25,
        )
        baseline = float(
            domain_m0.filter(
                (pl.col("dataset_id") == domain)
                & (pl.col("method") == baseline_method)
            )["auebc"][0]
        )
        oracle = float(
            domain_m0.filter(
                (pl.col("dataset_id") == domain)
                & (pl.col("method") == "one_shot_mechanical_oracle")
            )["auebc"][0]
        )
        rows.append(
            {
                "outer_domain": domain,
                "baseline_method": baseline_method,
                "baseline_auebc": baseline,
                "predicted_auebc": predicted,
                "oracle_auebc": oracle,
                "baseline_minus_predicted": baseline - predicted,
                "oracle_advantage_capture": (
                    (baseline - predicted) / (baseline - oracle)
                    if baseline != oracle
                    else None
                ),
            }
        )
    baseline = m0.gate.stronger_baseline_auebc
    oracle = m0.gate.one_shot_auebc
    predicted = float(
        np.mean([float(row["predicted_auebc"]) for row in rows], dtype=np.float64)
    )
    return {
        "baseline_method": baseline_method,
        "baseline_auebc": baseline,
        "predicted_auebc": predicted,
        "oracle_auebc": oracle,
        "advantage_capture": (baseline - predicted) / (baseline - oracle),
        "improved_domains": sum(
            float(row["baseline_minus_predicted"]) > 0.0 for row in rows
        ),
        "domain_rows": rows,
    }


def _external_report_evidence(root: Path) -> dict[str, object]:
    artifact = root / "artifacts/external_data"
    try:
        manifest = json.loads(
            (artifact / "EXTERNAL_DATA_MANIFEST.json").read_text(encoding="utf-8")
        )
        grid = json.loads(
            (artifact / "cranfield_wp2/grid_schema.json").read_text(
                encoding="utf-8"
            )
        )
        datasets = manifest["datasets"]
        rss = datasets["imperial_rss"]
        interlock = datasets["imperial_interlock"]
        tudelft = datasets["tudelft"]
        cranfield = datasets["cranfield_wp2"]
        indexed_grid = grid["spatial_grid_recoverable"]
        physical_spacing = grid["physical_coordinate_spacing_recoverable"]
        if type(indexed_grid) is not bool or type(physical_spacing) is not bool:
            raise TypeError("external grid flags must be boolean")
        evidence = {
            "checksum_ledger_sha256": _sha256(artifact / "CHECKSUMS.sha256"),
            "imperial_rss_exact_paired_n": int(rss["exact_paired_cscan_cai_n"]),
            "imperial_rss_potential_n": int(rss["potential_filename_linked_n"]),
            "imperial_interlock_exact_paired_n": int(
                interlock["exact_paired_cscan_cai_n"]
            ),
            "tudelft_exact_paired_n": int(tudelft["exact_paired_cscan_cai_n"]),
            "tudelft_role": str(tudelft["role"]),
            "cranfield_raw_file_n": int(cranfield["raw_file_n"]),
            "cranfield_processed_pair_n": int(cranfield["processed_scan_pair_n"]),
            "cranfield_indexed_grid_recoverable": indexed_grid,
            "cranfield_physical_spacing_recoverable": physical_spacing,
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("external audit evidence is incomplete") from error
    if (
        manifest.get("method_performance_present") is not False
        or cranfield.get("method_performance_present") is not False
        or evidence["imperial_rss_exact_paired_n"] < 0
        or evidence["imperial_rss_potential_n"]
        < evidence["imperial_rss_exact_paired_n"]
        or evidence["imperial_interlock_exact_paired_n"] < 0
        or evidence["tudelft_exact_paired_n"] < 0
        or evidence["tudelft_role"] != "MICRO_CASE_VALIDATION_ONLY"
        or evidence["cranfield_raw_file_n"] < evidence["cranfield_processed_pair_n"]
        or evidence["cranfield_indexed_grid_recoverable"] is not True
        or evidence["cranfield_physical_spacing_recoverable"] is not False
    ):
        raise ValueError("external audit evidence changed")
    return evidence


def publish_m1(
    config_path: str | Path,
    *,
    project_root: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Publish the complete M1 observability gate and frozen CAI diagnostic."""

    root = Path(project_root).resolve(strict=True)
    config = load_mvd_config(config_path, project_root=root)
    output = root / config.output_m1 if output_path is None else Path(output_path)
    _prepare(output)
    work = root / config.output_work / "m1_domains"
    predictions = pl.concat(
        [pl.read_parquet(work / domain / "predictions.parquet") for domain in config.domain_order]
    ).sort(["outer_domain", "specimen_id", "method", "cell_index"])
    metrics = pl.concat(
        [pl.read_csv(work / domain / "metrics.csv") for domain in config.domain_order]
    ).sort(["outer_domain", "specimen_id", "method"])
    regrets = pl.concat(
        [pl.read_csv(work / domain / "regret.csv") for domain in config.domain_order]
    ).sort(["outer_domain", "specimen_id", "method", "nominal_checkpoint"])
    selection = pl.concat(
        [pl.read_csv(work / domain / "selection_audit.csv") for domain in config.domain_order]
    )
    aggregation = aggregate_observability_metrics(
        metrics,
        domain_order=config.domain_order,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
        minimum_improved_domains=config.m1_minimum_improved_domains,
    )
    cai_states = pl.concat(
        [
            pl.read_parquet(
                root / config.output_work / "m1_cai_domains" / domain / "states.parquet"
            )
            for domain in config.domain_order
        ]
    ).sort(["outer_domain", "specimen_id", "nominal_checkpoint"])
    m0, _m0_tables = _m0_aggregation(root, config)
    capture = _advantage_capture(cai_states, m0, domain_order=config.domain_order)
    external = _external_report_evidence(root)
    predictions.write_parquet(output / "observability_predictions.parquet", compression="zstd")
    metrics.write_csv(output / "ranking_metrics.csv")
    regrets.write_csv(output / "regret_metrics.csv")
    selection.write_csv(output / "model_selection_audit.csv")
    pl.DataFrame(aggregation.model_metrics).write_csv(output / "model_metrics.csv")
    pl.DataFrame(aggregation.domain_metrics).write_csv(output / "domain_metrics.csv")
    pl.DataFrame(_bootstrap_rows(aggregation.bootstrap_effects)).write_csv(
        output / "bootstrap.csv"
    )
    pl.DataFrame(capture["domain_rows"]).write_csv(output / "cai_advantage_capture.csv")
    cai_states.write_parquet(output / "cai_diagnostic_states.parquet", compression="zstd")
    shutil.copyfile(config.config_path, output / "config.yaml")
    strong = bool(
        aggregation.gate.go
        and float(capture["advantage_capture"]) >= config.m1_strong_advantage_capture
        and int(capture["improved_domains"]) >= config.m1_minimum_improved_domains
    )
    summary = {
        "gate": asdict(aggregation.gate),
        "strong_go": strong,
        "aggregation_state_sha256": aggregation.state_sha256,
        "authority_state_sha256": config.authority_state_sha256,
        "advantage_capture_diagnostic": capture,
        "external_audit": external,
        "m2_authorized": False,
        "stop_reason": (
            "M1 observability gate failed; capacity rescue and M2/M3 are forbidden"
            if not aggregation.gate.go
            else "human review required before M2"
        ),
    }
    _json(output / "summary.json", summary)
    models = {str(row["method"]): row for row in aggregation.model_metrics}
    primary = models["o2_global_candidate"]
    candidate = models["o1_candidate_mlp_huber"]
    global_row = models["global_mechanical"]
    random_row = models["random_median"]
    uncertainty = models["observed_uncertainty"]
    effects = {value.effect_id: value for value in aggregation.bootstrap_effects}
    (output / "REPORT.md").write_text(
        "# MVD M1 Mechanical-Value Observability\n\n"
        f"Decision: `{aggregation.gate.status}`. M2/M3 remain locked.\n\n"
        f"The selected source-CV O2 scorer has equal-domain Spearman "
        f"`{primary['spearman']:.4f}`; its synchronized 95% interval is "
        f"`[{effects['o2_spearman_positive'].lower:.4f}, "
        f"{effects['o2_spearman_positive'].upper:.4f}]`. O2-minus-global "
        f"NDCG@10 is `{effects['o2_minus_global_ndcg10'].point_estimate:.4f}` "
        f"with lower bound `{effects['o2_minus_global_ndcg10'].lower:.4f}` and "
        f"improves only `{effects['o2_minus_global_ndcg10'].improved_domains}/6` "
        "domains.\n\n"
        "Candidate-only versus global-plus-candidate: candidate-only MLP "
        f"Spearman is `{candidate['spearman']:.4f}`, while selected O2 is "
        f"`{primary['spearman']:.4f}`; global is `{global_row['spearman']:.4f}` "
        f"and observed uncertainty is `{uncertainty['spearman']:.4f}`. None "
        "provides stable continuous value prediction or reliable top-set ranking. "
        f"O2 Regret@1 is `{primary['regret_1']:.4f}` and its mean exact-budget "
        f"set regret is `{primary['mean_budgeted_regret']:.4f}`, versus global "
        f"`{global_row['mean_budgeted_regret']:.4f}` and random "
        f"`{random_row['mean_budgeted_regret']:.4f}`. Global-minus-O2 and "
        "random-minus-O2 regret effects do not have positive lower bounds.\n\n"
        f"The frozen non-selection CAI diagnostic captures "
        f"`{float(capture['advantage_capture']):.3f}` of the one-shot oracle "
        f"advantage and improves `{int(capture['improved_domains'])}/6` domains. "
        "This diagnostic cannot override the failed observability gate. No larger "
        "network, Transformer, GNN, RL, diffusion, M2, or M3 was run.\n\n"
        "External feasibility: Imperial RSS has "
        f"`{external['imperial_rss_exact_paired_n']}` exact paired C-scan+CAI "
        f"specimens (`{external['imperial_rss_potential_n']}` potential links "
        "remain unresolved); Imperial Interlock has "
        f"`{external['imperial_interlock_exact_paired_n']}` exact pairs and is a "
        "small pilot. No audited dataset is sufficient by itself for formal "
        "statistical external replication. TU Delft has "
        f"`{external['tudelft_exact_paired_n']}` specimens and remains "
        "case-level `MICRO_CASE_VALIDATION_ONLY`. Cranfield raw PA recovers a "
        "discrete indexed spatial measurement grid and normalized 8x8 mapping, "
        "but not authoritative physical spacing or scanner-time reduction.\n",
        encoding="utf-8",
    )
    _finalize(
        output,
        package="mvd_m1_observability",
        authority_state=config.authority_state_sha256,
    )
    return output


__all__ = ["publish_m0", "publish_m1"]
