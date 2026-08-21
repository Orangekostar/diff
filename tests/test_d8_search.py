from __future__ import annotations

import csv
import hashlib
import inspect
import json
import sqlite3
from pathlib import Path

import numpy as np
import optuna
import pytest

from cmc_bbdm.cpb_diffusion_marginalization.authority import issue_search_view
from cmc_bbdm.cpb_diffusion_marginalization.config import load_d8_config
from cmc_bbdm.cpb_diffusion_marginalization.search import (
    D8Candidate,
    InnerEvaluation,
    robust_inner_objective,
    run_outer_search,
    suggest_candidate,
)
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D8_CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_exploration.yaml"
P1_CONFIG = PROJECT_ROOT / "paper_v3/configs/p1_full_field_oracle.yaml"


@pytest.fixture(scope="module")
def config():
    return load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def search_view(config):
    p1 = load_v3_config(P1_CONFIG, project_root=PROJECT_ROOT)
    data = load_data(p1, PROJECT_ROOT)
    return issue_search_view(data, outer_domain="74t7kcdgkr", config=config)


def test_objective_uses_five_inner_domains_and_registered_formula() -> None:
    values = np.asarray((0.08, 0.09, 0.10, 0.11, 0.12), dtype=np.float64)
    expected = float(np.mean(values) + 0.25 * np.max(values) + 0.10 * np.std(values))
    assert robust_inner_objective(values) == pytest.approx(expected, abs=1.0e-15)


def test_search_runner_has_no_evaluation_view_parameter() -> None:
    assert "evaluation_view" not in inspect.signature(run_outer_search).parameters


def test_suggest_candidate_is_canonical_and_binds_consistency(config) -> None:
    trial = optuna.trial.FixedTrial(
        {
            "control_id": "B6",
            "decomposition_family": "gaussian",
            "band": "high",
            "gaussian_sigma": 2.0,
            "alpha": 0.1,
            "area_relative_deviation": 0.1,
            "width_relative_deviation": 0.1,
            "height_relative_deviation": 0.1,
            "centroid_shift_mm": 2.0,
            "low_frequency_correlation_minimum": 0.99,
            "radial_spearman_minimum": 0.98,
            "K_train": 4,
            "K_test": 8,
            "feature_layer": "global",
            "feature_aggregation": "mean",
            "prediction_aggregation": "mean",
            "consistency": "feature_variance",
            "consistency_weight": 0.1,
            "pca_dimension": 8,
            "regressor": "ridge",
            "ridge_alpha": 10.0,
        }
    )
    candidate = suggest_candidate(trial, config)
    assert candidate.control_id == "B6"
    assert candidate.consistency == "feature_variance"
    assert candidate.consistency_weight == 0.1
    assert candidate.regressor_spec.regressor == "ridge"
    assert candidate.state_sha256 == candidate.canonical_sha256


def test_b6_none_consistency_is_explicitly_pruned(config) -> None:
    trial = optuna.trial.FixedTrial(
        {
            "control_id": "B6",
            "decomposition_family": "gaussian",
            "band": "high",
            "gaussian_sigma": 2.0,
            "alpha": 0.1,
            "area_relative_deviation": 0.1,
            "width_relative_deviation": 0.1,
            "height_relative_deviation": 0.1,
            "centroid_shift_mm": 2.0,
            "low_frequency_correlation_minimum": 0.99,
            "radial_spearman_minimum": 0.98,
            "K_train": 4,
            "feature_layer": "global",
            "feature_aggregation": "mean",
            "consistency": "none",
        }
    )
    with pytest.raises(optuna.TrialPruned, match="B6 requires consistency"):
        suggest_candidate(trial, config)
    assert trial.user_attrs["failure_reason"] == "invalid_combination:B6_requires_consistency"


def test_b1_is_an_actual_low_pass_morphology_control(config) -> None:
    trial = optuna.trial.FixedTrial(
        {
            "control_id": "B1",
            "gaussian_sigma": 2.0,
            "area_relative_deviation": 0.1,
            "width_relative_deviation": 0.1,
            "height_relative_deviation": 0.1,
            "centroid_shift_mm": 2.0,
            "low_frequency_correlation_minimum": 0.95,
            "radial_spearman_minimum": 0.90,
            "K_train": 8,
            "K_test": 16,
            "feature_layer": "global",
            "feature_aggregation": "mean",
            "prediction_aggregation": "mean",
            "pca_dimension": 8,
            "regressor": "ridge",
            "ridge_alpha": 10.0,
        }
    )
    candidate = suggest_candidate(trial, config)
    assert candidate.control_id == "B1"
    assert candidate.decomposition_family == "gaussian"
    assert candidate.band == "low"
    assert candidate.alpha == 1.0
    assert candidate.K_train == candidate.K_test == 1
    assert candidate.marginalization_stage == "feature"


def test_consistency_strategy_must_match_marginalization_stage(config) -> None:
    trial = optuna.trial.FixedTrial(
        {
            "control_id": "B8",
            "decomposition_family": "gaussian",
            "band": "high",
            "gaussian_sigma": 2.0,
            "alpha": 0.1,
            "area_relative_deviation": 0.1,
            "width_relative_deviation": 0.1,
            "height_relative_deviation": 0.1,
            "centroid_shift_mm": 2.0,
            "low_frequency_correlation_minimum": 0.95,
            "radial_spearman_minimum": 0.90,
            "K_train": 4,
            "K_test": 8,
            "feature_layer": "global",
            "marginalization_stage": "feature",
            "feature_aggregation": "mean_variance",
            "consistency": "prediction_variance",
            "consistency_weight": 0.1,
            "pca_dimension": 8,
            "regressor": "ridge",
            "ridge_alpha": 10.0,
        }
    )
    with pytest.raises(optuna.TrialPruned, match="consistency stage"):
        suggest_candidate(trial, config)
    assert trial.user_attrs["failure_reason"] == (
        "invalid_combination:consistency_stage_mismatch"
    )


def test_b7_and_b8_freeze_prediction_vs_feature_marginalization(config) -> None:
    common = {
        "decomposition_family": "gaussian",
        "band": "high",
        "gaussian_sigma": 2.0,
        "alpha": 0.1,
        "area_relative_deviation": 0.1,
        "width_relative_deviation": 0.1,
        "height_relative_deviation": 0.1,
        "centroid_shift_mm": 2.0,
        "low_frequency_correlation_minimum": 0.95,
        "radial_spearman_minimum": 0.90,
        "K_train": 4,
        "K_test": 8,
        "feature_layer": "global",
        "prediction_aggregation": "median",
        "pca_dimension": 8,
        "regressor": "ridge",
        "ridge_alpha": 10.0,
    }
    b7 = suggest_candidate(
        optuna.trial.FixedTrial({"control_id": "B7", **common}), config
    )
    assert b7.marginalization_stage == "prediction"
    assert b7.feature_aggregation == "mean"
    assert b7.prediction_aggregation == "median"

    b8 = suggest_candidate(
        optuna.trial.FixedTrial(
            {
                "control_id": "B8",
                **common,
                "marginalization_stage": "feature",
                "feature_aggregation": "trimmed",
                "consistency": "feature_variance",
                "consistency_weight": 0.1,
            }
        ),
        config,
    )
    assert b8.marginalization_stage == "feature"
    assert b8.feature_aggregation == "trimmed"
    assert b8.prediction_aggregation == "mean"


def test_outer_search_records_72_trials_failures_and_resume_without_duplication(
    tmp_path: Path, config, search_view
) -> None:
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    def evaluator(candidate, inner_fold) -> InnerEvaluation:
        if candidate.control_id == "B0":
            raise ValueError("registered synthetic failure")
        domain_index = config.outer_domains.index(inner_fold.query_domain)
        mae = 0.08 + 0.001 * domain_index + 0.0001 * int(candidate.control_id[1:])
        evidence = hashlib.sha256(
            f"{candidate.state_sha256}:{inner_fold.state_sha256}".encode("ascii")
        ).hexdigest()
        return InnerEvaluation(
            query_domain=inner_fold.query_domain,
            mae=mae,
            accepted_proposals=10,
            proposed_variants=10,
            evidence_sha256=evidence,
        )

    first = run_outer_search(
        search_view,
        config=config,
        output=tmp_path,
        evaluator=evaluator,
    )
    assert first.initial_trial_count == 72
    assert first.trial_count == 72
    assert first.completed_count > 0
    assert first.failed_count > 0
    assert len(first.selected_candidates) == 12
    with (tmp_path / "trial_index.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, strict=True))
    assert len(rows) == 72
    assert any(row["state"] == "FAIL" for row in rows)
    assert all(row["outer_fold"] == search_view.outer_domain for row in rows)
    completed_rows = [row for row in rows if row["state"] == "COMPLETE"]
    assert completed_rows
    expected_domains = tuple(
        domain for domain in config.outer_domains if domain != search_view.outer_domain
    )
    for row in completed_rows:
        assert int(row["accepted_proposals"]) == 50
        assert int(row["proposed_variants"]) == 50
        assert float(row["acceptance_rate"]) == 1.0
        acceptance = json.loads(row["acceptance_by_domain"])
        assert tuple(acceptance) == expected_domains
        assert acceptance == {
            domain: {
                "accepted_proposals": 10,
                "proposed_variants": 10,
                "acceptance_rate": 1.0,
            }
            for domain in expected_domains
        }
    assert (tmp_path / "study.db").is_file()

    resumed = run_outer_search(
        search_view,
        config=config,
        output=tmp_path,
        evaluator=evaluator,
    )
    assert resumed.trial_count == 72
    with (tmp_path / "trial_index.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        resumed_rows = list(csv.DictReader(handle, strict=True))
    assert len(resumed_rows) == 72

    study = optuna.load_study(
        study_name=f"d8::{search_view.outer_domain}",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
    )
    completed = next(
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    )
    payload = dict(completed.user_attrs["candidate"])
    payload.pop("state_sha256")
    payload["alpha"] = 0.123456789
    forged = D8Candidate.from_payload(payload).to_payload()
    with sqlite3.connect(tmp_path / "study.db") as connection:
        connection.execute(
            "UPDATE trial_user_attributes SET value_json = ? "
            "WHERE trial_id = ? AND key = 'candidate'",
            (json.dumps(forged, sort_keys=True), completed._trial_id),
        )
    with pytest.raises(ValueError, match="recorded parameters"):
        run_outer_search(
            search_view,
            config=config,
            output=tmp_path,
            evaluator=evaluator,
        )
