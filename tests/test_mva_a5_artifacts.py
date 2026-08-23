from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from cmc_bbdm.mva.a5_artifacts import A5ArtifactError, _validate_raw_evidence
from cmc_bbdm.mva.a5_config import load_a5_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "paper_v3/configs/mva_a5_imitation_policy.yaml"


def _raw_evidence(tmp_path: Path) -> tuple[object, dict[str, dict[str, object]]]:
    config = load_a5_config(CONFIG_PATH, project_root=PROJECT_ROOT)
    domains = config.domain_order
    specimen_domains = {
        f"s{index:03d}": domains[index % len(domains)]
        for index in range(config.specimen_count)
    }
    teacher_rows = []
    for specimen_id, dataset_id in specimen_domains.items():
        for outer_domain in domains:
            if outer_domain == dataset_id:
                continue
            teacher_rows.append(
                {
                    "outer_domain": outer_domain,
                    "dataset_id": dataset_id,
                    "specimen_id": specimen_id,
                    "state_count": 2,
                    "decision_state_count": 1,
                    "candidate_count": 2,
                    "predictor_state_sha256": "a" * 64,
                    "trajectory_state_sha256": "b" * 64,
                    "cache_sha256": "c" * 64,
                }
            )
    pl.DataFrame(teacher_rows).write_csv(tmp_path / "teacher_index.csv")

    audit_rows = []
    for outer_domain in domains:
        for query_domain in domains:
            if query_domain == outer_domain:
                continue
            allowed_domains = [
                domain
                for domain in domains
                if domain not in {outer_domain, query_domain}
            ]
            for pca_dimension in config.pca_dimensions:
                for inner_domain in allowed_domains:
                    fit_domains = [
                        domain
                        for domain in allowed_domains
                        if domain != inner_domain
                    ]
                    audit_rows.append(
                        {
                            "stage": "inner",
                            "held_out_target_domain": outer_domain,
                            "query_source_domain": query_domain,
                            "query_domains": inner_domain,
                            "fit_domains": "|".join(fit_domains),
                            "pca_dimension": pca_dimension,
                            "predictor_state_sha256": hashlib.sha256(
                                (
                                    f"{outer_domain}:{query_domain}:inner:"
                                    f"{inner_domain}:{pca_dimension}"
                                ).encode("ascii")
                            ).hexdigest(),
                        }
                    )
            audit_rows.append(
                {
                    "stage": "outer",
                    "held_out_target_domain": outer_domain,
                    "query_source_domain": query_domain,
                    "query_domains": query_domain,
                    "fit_domains": "|".join(allowed_domains),
                    "pca_dimension": config.pca_dimensions[0],
                    "predictor_state_sha256": hashlib.sha256(
                        f"{outer_domain}:{query_domain}:outer".encode("ascii")
                    ).hexdigest(),
                }
            )
    pl.DataFrame(audit_rows).write_csv(tmp_path / "teacher_fit_audits.csv")

    outer_states: dict[str, dict[str, object]] = {}
    training_rows = []
    for outer_domain in domains:
        policy_sha256 = hashlib.sha256(outer_domain.encode("ascii")).hexdigest()
        teacher_sha256 = hashlib.sha256(
            f"teacher:{outer_domain}".encode("ascii")
        ).hexdigest()
        source_count = sum(
            dataset_id != outer_domain for dataset_id in specimen_domains.values()
        )
        target_count = config.specimen_count - source_count
        target_rows = target_count * len(config.methods) * len(config.checkpoints)
        outer_states[outer_domain] = {
            "evaluator_model_state_sha256": hashlib.sha256(
                f"evaluator:{outer_domain}".encode("ascii")
            ).hexdigest(),
            "initial_budget": 0.03125,
            "policy_state_sha256": policy_sha256,
            "source_specimen_count": source_count,
            "state_rows": target_rows,
            "target_specimen_count": target_count,
            "teacher_model_state_sha256": teacher_sha256,
            "teacher_state_rows": source_count * 2,
            "trajectory_rows": target_rows,
            "training_state_count": source_count,
        }
        for epoch in range(1, config.epochs + 1):
            training_rows.append(
                {
                    "outer_domain": outer_domain,
                    "epoch": epoch,
                    "weighted_pairwise_loss": 1.0 / epoch,
                    "policy_state_sha256": policy_sha256,
                    "teacher_model_state_sha256": teacher_sha256,
                    "training_state_count": source_count,
                    "source_specimen_count": source_count,
                }
            )
    pl.DataFrame(training_rows).write_csv(tmp_path / "policy_training.csv")

    state_rows = []
    trajectory_rows = []
    for specimen_id, domain in specimen_domains.items():
        policy_sha256 = outer_states[domain]["policy_state_sha256"]
        for method in config.methods:
            trajectory_sha256 = hashlib.sha256(
                f"{specimen_id}:{method}".encode("ascii")
            ).hexdigest()
            issued_policy = policy_sha256 if method == "imitation_policy" else None
            for step, checkpoint in enumerate(config.checkpoints):
                state_rows.append(
                    {
                        "outer_domain": domain,
                        "dataset_id": domain,
                        "specimen_id": specimen_id,
                        "method": method,
                        "nominal_checkpoint": checkpoint,
                        "policy_state_sha256": issued_policy,
                        "trajectory_state_sha256": trajectory_sha256,
                    }
                )
                trajectory_rows.append(
                    {
                        "outer_domain": domain,
                        "dataset_id": domain,
                        "specimen_id": specimen_id,
                        "method": method,
                        "step": step,
                        "nominal_checkpoint": checkpoint,
                        "budget_before": 10 + step,
                        "budget_after": 11 + step,
                        "from_level": step,
                        "to_level": step + 1,
                        "policy_state_sha256": issued_policy,
                        "trajectory_state_sha256": trajectory_sha256,
                    }
                )
    pl.DataFrame(state_rows).write_parquet(tmp_path / "state_metrics.parquet")
    pl.DataFrame(trajectory_rows).write_parquet(
        tmp_path / "target_trajectories.parquet"
    )
    return config, outer_states


def test_a5_raw_evidence_rejects_teacher_fit_leakage(tmp_path: Path) -> None:
    config, outer_states = _raw_evidence(tmp_path)
    _validate_raw_evidence(tmp_path, config=config, outer_states=outer_states)

    audits = pl.read_csv(tmp_path / "teacher_fit_audits.csv")
    leaked = audits.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("fit_domains") + "|" + pl.col("held_out_target_domain"))
        .otherwise(pl.col("fit_domains"))
        .alias("fit_domains")
    )
    leaked.write_csv(tmp_path / "teacher_fit_audits.csv")

    with pytest.raises(A5ArtifactError, match="fit barrier"):
        _validate_raw_evidence(tmp_path, config=config, outer_states=outer_states)


def test_a5_raw_evidence_rejects_missing_target_action(tmp_path: Path) -> None:
    config, outer_states = _raw_evidence(tmp_path)
    trajectories = pl.read_parquet(tmp_path / "target_trajectories.parquet")
    trajectories.head(trajectories.height - 1).write_parquet(
        tmp_path / "target_trajectories.parquet"
    )

    with pytest.raises(A5ArtifactError, match="target (trajectory digest|outer evidence)"):
        _validate_raw_evidence(tmp_path, config=config, outer_states=outer_states)
