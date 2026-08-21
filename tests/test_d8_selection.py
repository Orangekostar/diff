from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.cpb_diffusion_marginalization.authority import issue_search_view
from cmc_bbdm.cpb_diffusion_marginalization.config import load_d8_config
from cmc_bbdm.cpb_diffusion_marginalization.regression import (
    CandidatePrediction,
    CandidateSpec,
)
from cmc_bbdm.cpb_diffusion_marginalization.search import D8Candidate
from cmc_bbdm.cpb_diffusion_marginalization.selection import (
    evaluate_finalists,
    fit_nonnegative_ensemble,
    freeze_outer_selection,
    rerank_candidates,
    validate_frozen_outer_selection,
)
from cmc_bbdm.cpb_diffusion_marginalization.variants import MorphologyThresholds
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


def _candidate(config, index: int) -> D8Candidate:
    return D8Candidate(
        control_id="B5",
        decomposition_family="gaussian",
        band="high",
        decomposition_parameters={"sigma": 1.0 + 0.1 * index},
        alpha=0.05 + 0.01 * index,
        K_train=1,
        K_test=1,
        thresholds=MorphologyThresholds(
            area_relative_deviation=0.10,
            width_relative_deviation=0.10,
            height_relative_deviation=0.10,
            centroid_shift_mm=2.0,
            low_frequency_correlation_minimum=0.95,
            radial_spearman_minimum=0.90,
        ),
        feature_layer="global",
        feature_aggregation="mean",
        prediction_aggregation="mean",
        morphology_beta=None,
        consistency="none",
        consistency_weight=0.0,
        regressor_spec=CandidateSpec(
            pca_dimension=8,
            regressor="ridge",
            parameters={"alpha": 10.0},
            seed=config.seed + index,
        ),
        seed=config.seed + index,
        config_sha256=config.config_sha256,
    )


def _prediction(candidate, inner_fold) -> CandidatePrediction:
    indices = np.asarray(inner_fold.query_indices, dtype=np.int64)
    targets = np.asarray(
        inner_fold.search_view.data_view.cai_ratio[indices], dtype=np.float64
    )
    predictions = targets + 0.01
    fit_state = hashlib.sha256(
        candidate.state_sha256.encode("ascii") + inner_fold.state_sha256.encode("ascii")
    ).hexdigest()
    state = hashlib.sha256(
        fit_state.encode("ascii") + targets.tobytes() + predictions.tobytes()
    ).hexdigest()
    return CandidatePrediction(
        fit_specimen_ids=inner_fold.fit_specimen_ids,
        query_specimen_ids=inner_fold.query_specimen_ids,
        targets=targets,
        predictions=predictions,
        fit_state_sha256=fit_state,
        state_sha256=state,
    )


def test_rerank_uses_three_fixed_seeds_and_canonical_tie_break(
    config, search_view
) -> None:
    candidates = tuple(_candidate(config, index) for index in range(12))
    result = rerank_candidates(
        candidates,
        view=search_view,
        seeds=config.rerank_seeds,
        evaluator=_prediction,
    )
    assert result.seed_count == 3
    assert len(result.rows) == 12
    assert len(result.finalists) == 4
    assert result.selected.candidate.state_sha256 == min(
        candidate.state_sha256 for candidate in candidates
    )
    assert result.selected.objective == pytest.approx(0.0125, abs=1.0e-15)
    assert all(
        row.oof_predictions.shape == (search_view.specimen_count,)
        for row in result.rows
    )


def test_finalists_are_checked_at_registered_large_k(config, search_view) -> None:
    candidates = tuple(_candidate(config, index) for index in range(4))
    result = evaluate_finalists(
        candidates,
        view=search_view,
        seeds=config.rerank_seeds,
        K_test_values=(8, 16),
        evaluator=_prediction,
    )
    assert len(result.cells) == 8
    assert {cell.candidate.K_test for cell in result.cells} == {8, 16}
    assert len(result.selected) == 4


def test_nonnegative_oof_ensemble_improves_registered_objective() -> None:
    targets = np.asarray([0.0, 1.0] * 5, dtype=np.float64)
    domains = np.repeat(np.asarray([f"d{index}" for index in range(5)]), 2)
    specimen_ids = tuple(f"s{index:02d}" for index in range(10))
    predictions = np.vstack(
        (
            targets + np.asarray([0.10, -0.02] * 5),
            targets + np.asarray([-0.02, 0.10] * 5),
        )
    )
    result = fit_nonnegative_ensemble(
        predictions,
        targets,
        specimen_ids=specimen_ids,
        domain_ids=tuple(domains.tolist()),
        candidate_sha256=("1" * 64, "2" * 64),
        minimum_j_gain=1.0e-4,
    )
    assert result.accepted
    assert np.all(result.weights >= 0.0)
    assert float(np.sum(result.weights)) == pytest.approx(1.0, abs=1.0e-12)
    assert result.crossfit_weights.shape == (5, 2)
    np.testing.assert_allclose(
        np.sum(result.crossfit_weights, axis=1), np.ones(5), atol=1.0e-12
    )
    assert result.objective_gain >= 1.0e-4


def test_ensemble_query_domain_targets_do_not_fit_its_crossfit_weights() -> None:
    targets = np.asarray([0.0, 1.0] * 5, dtype=np.float64)
    domains = tuple(
        str(item)
        for item in np.repeat(np.asarray([f"d{index}" for index in range(5)]), 2)
    )
    predictions = np.vstack(
        (
            targets + np.asarray([0.10, -0.02] * 5),
            targets + np.asarray([-0.02, 0.10] * 5),
        )
    )
    common = {
        "specimen_ids": tuple(f"s{index:02d}" for index in range(10)),
        "domain_ids": domains,
        "candidate_sha256": ("1" * 64, "2" * 64),
        "minimum_j_gain": 1.0e-4,
    }
    original = fit_nonnegative_ensemble(predictions, targets, **common)
    changed_targets = np.array(targets, copy=True)
    changed_targets[:2] += 100.0
    changed = fit_nonnegative_ensemble(predictions, changed_targets, **common)
    np.testing.assert_array_equal(
        original.crossfit_weights[0], changed.crossfit_weights[0]
    )


def test_ensemble_deterministically_falls_back_to_best_member() -> None:
    targets = np.asarray([0.0, 1.0] * 5, dtype=np.float64)
    domains = tuple(
        str(item)
        for item in np.repeat(np.asarray([f"d{index}" for index in range(5)]), 2)
    )
    result = fit_nonnegative_ensemble(
        np.vstack((targets, targets + 0.2)),
        targets,
        specimen_ids=tuple(f"s{index:02d}" for index in range(10)),
        domain_ids=domains,
        candidate_sha256=("1" * 64, "2" * 64),
        minimum_j_gain=1.0e-4,
    )
    assert not result.accepted
    np.testing.assert_array_equal(result.weights, np.asarray((1.0, 0.0)))
    np.testing.assert_array_equal(result.predictions, targets)


def test_rerank_rejects_misaligned_inner_oof_identity(config, search_view) -> None:
    candidates = tuple(_candidate(config, index) for index in range(12))

    def misaligned(candidate, inner_fold) -> CandidatePrediction:
        result = _prediction(candidate, inner_fold)
        return CandidatePrediction(
            fit_specimen_ids=result.fit_specimen_ids,
            query_specimen_ids=tuple(reversed(result.query_specimen_ids)),
            targets=result.targets,
            predictions=result.predictions,
            fit_state_sha256=result.fit_state_sha256,
            state_sha256=result.state_sha256,
        )

    with pytest.raises(ValueError, match="query identities"):
        rerank_candidates(
            candidates,
            view=search_view,
            seeds=config.rerank_seeds,
            evaluator=misaligned,
        )


def test_selection_freezes_before_outer_evaluation(
    tmp_path, config, search_view
) -> None:
    candidates = tuple(_candidate(config, index) for index in range(12))
    reranked = rerank_candidates(
        candidates,
        view=search_view,
        seeds=config.rerank_seeds,
        evaluator=_prediction,
    )
    finalists = evaluate_finalists(
        tuple(row.candidate for row in reranked.finalists),
        view=search_view,
        seeds=config.rerank_seeds,
        K_test_values=(8, 16),
        evaluator=_prediction,
    )
    prediction_matrix = np.vstack([row.oof_predictions for row in finalists.selected])
    ensemble = fit_nonnegative_ensemble(
        prediction_matrix,
        finalists.selected[0].oof_targets,
        specimen_ids=search_view.specimen_ids,
        domain_ids=search_view.dataset_ids,
        candidate_sha256=tuple(
            row.candidate.state_sha256 for row in finalists.selected
        ),
        minimum_j_gain=1.0e-4,
    )
    path = tmp_path / "selection.json"
    frozen = freeze_outer_selection(
        reranked,
        finalists=finalists,
        ensemble=ensemble,
        view=search_view,
        output=path,
    )
    assert frozen.outer_domain == search_view.outer_domain
    assert not frozen.outer_evaluation_started
    assert path.is_file()
    document = path.read_text(encoding="utf-8")
    assert 'outer_evaluation_started":false' in document
    assert '"crossfit_weights"' in document


def test_frozen_selection_recomputes_all_inner_evidence_and_rejects_tampering(
    tmp_path, config, search_view
) -> None:
    candidates = tuple(_candidate(config, index) for index in range(12))
    reranked = rerank_candidates(
        candidates,
        view=search_view,
        seeds=config.rerank_seeds,
        evaluator=_prediction,
    )
    finalists = evaluate_finalists(
        tuple(row.candidate for row in reranked.finalists),
        view=search_view,
        seeds=config.rerank_seeds,
        K_test_values=(8, 16),
        evaluator=_prediction,
    )
    ensemble = fit_nonnegative_ensemble(
        np.vstack([row.oof_predictions for row in finalists.selected]),
        finalists.selected[0].oof_targets,
        specimen_ids=search_view.specimen_ids,
        domain_ids=search_view.dataset_ids,
        candidate_sha256=tuple(
            row.candidate.state_sha256 for row in finalists.selected
        ),
        minimum_j_gain=1.0e-4,
    )
    path = tmp_path / "selection.json"
    frozen = freeze_outer_selection(
        reranked,
        finalists=finalists,
        ensemble=ensemble,
        view=search_view,
        output=path,
    )
    assert (
        validate_frozen_outer_selection(
            path,
            view=search_view,
            search_candidates=candidates,
        )
        == frozen
    )

    document = json.loads(path.read_text(encoding="ascii"))
    document["rerank"]["rows"][0]["seeds"][0]["oof_predictions"][0] += 0.01
    tampered = tmp_path / "tampered.json"
    tampered.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="selection|rerank|prediction|state"):
        validate_frozen_outer_selection(
            tampered,
            view=search_view,
            search_candidates=candidates,
        )
