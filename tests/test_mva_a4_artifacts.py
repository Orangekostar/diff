from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import polars as pl
import pytest

from cmc_bbdm.mva.a4_artifacts import (
    REQUIRED_A4_OUTPUTS,
    A4ArtifactError,
    _write_aggregation_tables,
    finalize_a4_package,
    publish_a4_manifest,
    validate_a4_package,
)
from cmc_bbdm.mva.a4_config import load_a4_config
from cmc_bbdm.mva.a4_evaluation import aggregate_a4_tables
from cmc_bbdm.mva.artifacts import validate_mva_package
from cmc_bbdm.mva.config import load_mva_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mva_a4_global_mask.yaml"


def _package(path: Path) -> None:
    config = load_a4_config(CONFIG, project_root=ROOT)
    base_path = ROOT / config.sources["a0_a3_config"].path
    base = load_mva_config(base_path, project_root=ROOT)
    reference = pl.read_parquet(
        ROOT / base.output_dir / "a2_oracle_value/state_metrics.parquet"
    )
    uniform = reference.filter(
        (pl.col("method") == "uniform")
        & pl.col("nominal_checkpoint").is_in(list(config.checkpoints))
    )
    methods = (
        ("global_appearance_mask", 0.005),
        ("global_reconstruction_mask", 0.01),
        ("global_mechanical_mask", 0.02),
    )
    states = pl.concat(
        [
            uniform.with_columns(
                pl.lit(method).alias("method"),
                (pl.col("p_b_absolute_error") - reduction)
                .clip(lower_bound=0.0)
                .alias("p_b_absolute_error"),
                (pl.col("p_a_absolute_error") - reduction)
                .clip(lower_bound=0.0)
                .alias("p_a_absolute_error"),
            )
            for method, reduction in methods
        ],
        how="vertical_relaxed",
    )
    aggregation = aggregate_a4_tables(
        states,
        reference,
        domain_order=config.domain_order,
        checkpoints=config.checkpoints,
        random_seeds=base.random_seeds,
        full_mae=base.full_mae,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_resamples=config.bootstrap_resamples,
    )
    path.mkdir(parents=True)
    states.write_parquet(path / "state_metrics.parquet", compression="zstd")
    pl.DataFrame({"fixture": [1]}).write_parquet(path / "source_values.parquet")
    pl.DataFrame({"fixture": [1]}).write_parquet(
        path / "fixed_trajectories.parquet"
    )
    for name in ("fit_audits.csv", "rankings.csv", "ranking_stability.csv"):
        pl.DataFrame({"fixture": [1]}).write_csv(path / name)
    _write_aggregation_tables(path, aggregation=aggregation, config=config)
    a2 = validate_mva_package(
        ROOT / base.output_dir / "a2_oracle_value",
        project_root=ROOT,
        config_path=base_path,
    )
    outer_states = {}
    for index, domain in enumerate(config.domain_order):
        count = states.filter(pl.col("dataset_id") == domain)[
            "specimen_id"
        ].n_unique()
        outer_states[domain] = {
            "candidate_bank_state_sha256": f"{index + 1:x}" * 64,
            "source_label_state_sha256": f"{index + 2:x}" * 64,
            "evaluator_model_state_sha256": f"{index + 3:x}" * 64,
            "evaluation_state_sha256": f"{index + 4:x}" * 64,
            "target_specimen_count": count,
        }
    summary = {
        "schema_version": 1,
        "scope": config.scope,
        "global_mask_status": aggregation.gate.global_mask_status,
        "a5_status": aggregation.gate.a5_status,
        "aggregation_state_sha256": aggregation.state_sha256,
        "bootstrap_indices_sha256": aggregation.bootstrap_effects[0].indices_sha256,
        "a2_output_tree_sha256": a2.output_tree_sha256,
        "outer_states": outer_states,
        "gate": json.loads(json.dumps(asdict(aggregation.gate))),
    }
    (path / "summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="ascii"
    )
    (path / "REPORT.md").write_text("# Fixture A4 report\n", encoding="ascii")
    figures = path / "figures"
    figures.mkdir()
    for relative in REQUIRED_A4_OUTPUTS:
        target = path / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"fixture:{relative}\n".encode("ascii"))
    shutil.copyfile(CONFIG, path / "config.yaml")


def test_a4_package_manifest_and_numeric_evidence_are_validated(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    _package(output)

    published = publish_a4_manifest(output, project_root=ROOT, config_path=CONFIG)
    validated = validate_a4_package(output, project_root=ROOT, config_path=CONFIG)

    assert published == validated
    assert validated.global_mask_status in {
        "MVA_A4_GLOBAL_GO",
        "MVA_A4_GLOBAL_NO_GO",
    }
    assert validated.a5_status in {"MVA_A5_AUTHORIZED", "MVA_A5_NOT_AUTHORIZED"}


def test_a4_package_rejects_tampering_and_private_paths(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    _package(output)
    publish_a4_manifest(output, project_root=ROOT, config_path=CONFIG)
    with (output / "budget_metrics.csv").open("ab") as handle:
        handle.write(b"tampered\n")

    with pytest.raises(A4ArtifactError, match="manifest|CHECKSUMS"):
        validate_a4_package(output, project_root=ROOT, config_path=CONFIG)

    private = tmp_path / "private"
    _package(private)
    (private / "REPORT.md").write_text(str(ROOT), encoding="utf-8")
    with pytest.raises(A4ArtifactError, match="absolute path"):
        publish_a4_manifest(private, project_root=ROOT, config_path=CONFIG)


def test_a4_publisher_recomputes_all_derived_tables(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    _package(output)
    curves = pl.read_csv(output / "cai_curves.csv").with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("equal_domain_mae") + 0.01)
        .otherwise(pl.col("equal_domain_mae"))
        .alias("equal_domain_mae")
    )
    curves.write_csv(output / "cai_curves.csv")

    with pytest.raises(A4ArtifactError, match="derived evidence"):
        publish_a4_manifest(output, project_root=ROOT, config_path=CONFIG)


def test_a4_finalization_is_atomic_and_idempotent(tmp_path: Path) -> None:
    work = tmp_path / "work"
    destination = tmp_path / "formal"
    _package(work)

    first = finalize_a4_package(
        work,
        destination,
        project_root=ROOT,
        config_path=CONFIG,
    )
    second = finalize_a4_package(
        work,
        destination,
        project_root=ROOT,
        config_path=CONFIG,
    )

    assert first == second
    assert (destination / "artifact_manifest.json").is_file()
    assert (destination / "CHECKSUMS.sha256").is_file()
