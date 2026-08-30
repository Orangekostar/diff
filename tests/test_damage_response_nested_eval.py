from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS
from cmc_bbdm.damage_response.feature_views import PRIMARY_TARGET_FIELDS
from cmc_bbdm.damage_response.nested_eval import (
    REDUNDANCY_MODELS,
    EvaluationError,
    P1RedundancyRecord,
    evaluate_p1_redundancy,
)
from cmc_bbdm.damage_response.sources import (
    IMPACTOR_CATEGORIES,
    LAMINATE_CATEGORIES,
    DesignMetadata,
)

DOMAINS = tuple(PRIMARY_COUNTS)


def _records() -> tuple[P1RedundancyRecord, ...]:
    records: list[P1RedundancyRecord] = []
    for domain_index, domain_id in enumerate(DOMAINS):
        for specimen_index in range(4):
            strength = 100.0 + 17.0 * domain_index + 6.0 * specimen_index
            specimen_id = f"{domain_id}-{specimen_index + 1:02d}"
            design = DesignMetadata(
                specimen_id=specimen_id,
                domain_id=domain_id,
                laminate_type=LAMINATE_CATEGORIES[
                    (domain_index + specimen_index) % len(LAMINATE_CATEGORIES)
                ],
                ply_count=8 + 4 * ((domain_index + specimen_index) % 3),
                impactor=IMPACTOR_CATEGORIES[
                    (2 * domain_index + specimen_index) % len(IMPACTOR_CATEGORIES)
                ],
                width_mm=49.8 + 0.1 * specimen_index + 0.02 * domain_index,
                thickness_mm=1.5 + 0.3 * domain_index + 0.05 * specimen_index,
            )
            records.append(
                P1RedundancyRecord(
                    specimen_id=specimen_id,
                    domain_id=domain_id,
                    published_cai_strength_mpa=strength,
                    extension_peak_mm=(
                        0.2
                        + 0.002 * strength
                        + 0.03 * domain_index
                        + 0.004 * specimen_index**2
                    ),
                    slope_u20_u60_mpa_per_mm=(
                        40.0
                        + 1.3 * strength
                        - 2.0 * domain_index
                        + 0.7 * specimen_index**2
                    ),
                    normalized_prepeak_auc=(
                        0.25
                        + 0.0007 * strength
                        + 0.005 * domain_index
                        - 0.001 * specimen_index**2
                    ),
                    design=design,
                )
            )
    return tuple(records)


def test_redundancy_evaluation_has_six_exact_folds_and_complete_oof() -> None:
    records = _records()

    result = evaluate_p1_redundancy(records)

    expected_prediction_count = (
        len(records) * len(PRIMARY_TARGET_FIELDS) * len(REDUNDANCY_MODELS)
    )
    assert len(result.predictions) == expected_prediction_count
    assert len(result.fold_states) == (
        len(DOMAINS) * len(PRIMARY_TARGET_FIELDS) * len(REDUNDANCY_MODELS)
    )
    assert len(result.metrics) == len(PRIMARY_TARGET_FIELDS) * len(REDUNDANCY_MODELS)
    assert {state.held_out_domain for state in result.fold_states} == set(DOMAINS)

    observed = [
        (row.specimen_id, row.endpoint, row.model) for row in result.predictions
    ]
    expected = {
        (record.specimen_id, endpoint, model)
        for record in records
        for endpoint in PRIMARY_TARGET_FIELDS
        for model in REDUNDANCY_MODELS
    }
    assert len(observed) == len(set(observed))
    assert set(observed) == expected
    assert all(row.domain_id == row.held_out_domain for row in result.predictions)


def test_fold_state_uses_only_source_rows_and_source_strength_statistics() -> None:
    records = _records()

    result = evaluate_p1_redundancy(records)

    for state in result.fold_states:
        source = tuple(
            record for record in records if record.domain_id != state.held_out_domain
        )
        source_ids = tuple(sorted(record.specimen_id for record in source))
        strengths = np.asarray(
            [record.published_cai_strength_mpa for record in source], dtype=np.float64
        )
        polynomial = np.column_stack((strengths, strengths**2, strengths**3))
        expected_means = np.mean(polynomial, axis=0)
        expected_scales = np.std(polynomial, axis=0, ddof=0)
        expected_scales[expected_scales == 0.0] = 1.0

        assert state.fit_specimen_ids == source_ids
        assert state.fit_domains == tuple(
            domain for domain in DOMAINS if domain != state.held_out_domain
        )
        assert not set(state.fit_specimen_ids) & {
            record.specimen_id
            for record in records
            if record.domain_id == state.held_out_domain
        }
        np.testing.assert_allclose(state.strength_means, expected_means)
        np.testing.assert_allclose(state.strength_scales, expected_scales)
        assert state.strength_means.flags.writeable is False
        assert state.strength_scales.flags.writeable is False
        assert state.coefficients.flags.writeable is False
        if state.model == "strength_only":
            assert state.design_encoder_sha256 is None
            assert state.coefficients.shape == (3,)
        else:
            assert state.design_encoder_sha256 is not None
            assert state.coefficients.shape == (14,)


def test_metrics_equal_independent_r2_and_per_domain_spearman() -> None:
    result = evaluate_p1_redundancy(_records())

    for metric in result.metrics:
        rows = tuple(
            row
            for row in result.predictions
            if row.endpoint == metric.endpoint and row.model == metric.model
        )
        truth = np.asarray([row.truth for row in rows], dtype=np.float64)
        prediction = np.asarray([row.prediction for row in rows], dtype=np.float64)
        assert metric.pooled_r2 == pytest.approx(float(r2_score(truth, prediction)))
        assert tuple(item.domain_id for item in metric.domain_spearman) == DOMAINS
        for item in metric.domain_spearman:
            domain_rows = tuple(row for row in rows if row.domain_id == item.domain_id)
            expected = float(
                spearmanr(
                    [row.truth for row in domain_rows],
                    [row.prediction for row in domain_rows],
                ).statistic
            )
            assert item.spearman == pytest.approx(expected)
            assert item.specimen_count == len(domain_rows)


def test_evaluation_is_deterministic_and_input_order_invariant() -> None:
    records = _records()

    first = evaluate_p1_redundancy(records)
    second = evaluate_p1_redundancy(tuple(reversed(records)))

    assert first.predictions == second.predictions
    assert first.metrics == second.metrics
    assert [state.state_sha256 for state in first.fold_states] == [
        state.state_sha256 for state in second.fold_states
    ]


def test_target_domain_sentinels_cannot_change_fitted_state_or_outcome_free_predictions() -> None:
    records = _records()
    held_out_domain = DOMAINS[-1]
    reference = evaluate_p1_redundancy(records)
    mutated = tuple(
        replace(
            record,
            published_cai_strength_mpa=record.published_cai_strength_mpa + 10_000.0,
            extension_peak_mm=record.extension_peak_mm + 20_000.0,
            slope_u20_u60_mpa_per_mm=record.slope_u20_u60_mpa_per_mm - 30_000.0,
            normalized_prepeak_auc=record.normalized_prepeak_auc + 40_000.0,
            design=replace(
                record.design,
                ply_count=1_000_000,
                width_mm=2_000_000.0,
                thickness_mm=3_000_000.0,
            ),
        )
        if record.domain_id == held_out_domain
        else record
        for record in records
    )
    sentinel = evaluate_p1_redundancy(mutated)

    reference_states = {
        (state.endpoint, state.model): state
        for state in reference.fold_states
        if state.held_out_domain == held_out_domain
    }
    sentinel_states = {
        (state.endpoint, state.model): state
        for state in sentinel.fold_states
        if state.held_out_domain == held_out_domain
    }
    assert reference_states.keys() == sentinel_states.keys()
    for key, state in reference_states.items():
        changed = sentinel_states[key]
        assert state.state_sha256 == changed.state_sha256
        np.testing.assert_array_equal(state.strength_means, changed.strength_means)
        np.testing.assert_array_equal(state.strength_scales, changed.strength_scales)
        np.testing.assert_array_equal(state.coefficients, changed.coefficients)
        assert state.intercept == changed.intercept

    # Changing held-out outcomes cannot change predictions; changing legitimate
    # held-out reference/design inputs can change their own query predictions.
    outcome_only = tuple(
        replace(
            record,
            extension_peak_mm=record.extension_peak_mm + 20_000.0,
            slope_u20_u60_mpa_per_mm=record.slope_u20_u60_mpa_per_mm - 30_000.0,
            normalized_prepeak_auc=record.normalized_prepeak_auc + 40_000.0,
        )
        if record.domain_id == held_out_domain
        else record
        for record in records
    )
    outcome_result = evaluate_p1_redundancy(outcome_only)
    reference_predictions = tuple(
        row.prediction
        for row in reference.predictions
        if row.held_out_domain == held_out_domain
    )
    outcome_predictions = tuple(
        row.prediction
        for row in outcome_result.predictions
        if row.held_out_domain == held_out_domain
    )
    assert reference_predictions == outcome_predictions


def test_evaluation_rejects_domain_or_identity_contract_drift() -> None:
    records = _records()

    with pytest.raises(EvaluationError, match="six canonical domains"):
        evaluate_p1_redundancy(
            tuple(record for record in records if record.domain_id != DOMAINS[-1])
        )
    with pytest.raises(EvaluationError, match="identity"):
        evaluate_p1_redundancy((records[0], replace(records[1], specimen_id=records[0].specimen_id), *records[2:]))
    with pytest.raises(EvaluationError, match="design identity"):
        evaluate_p1_redundancy(
            (replace(records[0], design=replace(records[0].design, specimen_id="wrong")), *records[1:])
        )
