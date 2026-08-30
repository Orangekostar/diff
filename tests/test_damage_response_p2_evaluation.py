from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS
from cmc_bbdm.damage_response.p2_evaluation import (
    P2_ENDPOINTS,
    P2_VIEWS,
    REGISTERED_P2_PROTOCOL,
    InnerCandidateScore,
    P2EvaluationError,
    P2EvaluationProtocol,
    evaluate_p2_nested_lodo,
    evaluate_p2_outer_fold,
    select_inner_candidate,
)
from cmc_bbdm.damage_response.p2_features import P2FeatureAuthority

DOMAIN_ORDER = tuple(PRIMARY_COUNTS)
TEST_PROTOCOL = P2EvaluationProtocol(
    ridge_alphas=(0.1, 1.0),
    pca_dimensions=(2, 3),
    tie_tolerance=1e-12,
)


def _authority(*, query_sentinel_domain: str | None = None) -> P2FeatureAuthority:
    rng = np.random.default_rng(29)
    specimen_ids = tuple(
        f"{domain[:2]}-{row}" for domain in DOMAIN_ORDER for row in range(4)
    )
    domain_ids = tuple(domain for domain in DOMAIN_ORDER for _ in range(4))
    n = len(specimen_ids)
    laminate = tuple(
        "cross_ply" if index % 2 == 0 else "quasi_isotropic"
        for index in range(n)
    )
    impactors = tuple(
        ("hemia", "hemib", "coni60", "flat")[index % 4] for index in range(n)
    )
    ply = np.asarray([8 + 8 * (index % 3) for index in range(n)])
    width = np.linspace(48.0, 66.0, n)
    thickness = np.linspace(1.6, 3.1, n)
    surface = rng.normal(size=(n, 21))
    scalar = np.abs(rng.normal(size=(n, 3)))
    embedding = rng.normal(size=(n, 512)).astype(np.float32)
    total_energy = np.linspace(4.0, 19.0, n)
    if query_sentinel_domain is not None:
        query = np.asarray([value == query_sentinel_domain for value in domain_ids])
        ply[query] = 1_000_000
        width[query] = 2_000_000.0
        thickness[query] = 3_000_000.0
        surface[query] = 4_000_000.0
        scalar[query] = 5_000_000.0
        embedding[query] = 6_000_000.0
        total_energy[query] = 7_000_000.0
    return P2FeatureAuthority(
        specimen_ids=specimen_ids,
        domain_ids=domain_ids,
        laminate_types=laminate,
        ply_counts=ply,
        widths_mm=width,
        thicknesses_mm=thickness,
        surface_profile_stats=surface,
        scalar_damage=scalar,
        full_cscan_embedding=embedding,
        privileged_total_energy_j=total_energy,
        privileged_impactors=impactors,
        full_embedding_view="FULL",
        encoder_sha256="a" * 64,
        embedding_state_sha256="b" * 64,
        source_sha256={
            "feature_bank": "1" * 64,
            "feature_cache": "2" * 64,
            "physical_descriptors": "3" * 64,
            "provenance_specimens": "4" * 64,
            "lvi_workbook": "5" * 64,
        },
    )


def _targets(authority: P2FeatureAuthority) -> dict[str, np.ndarray]:
    index = np.arange(len(authority.specimen_ids), dtype=np.float64)
    return {
        "extension_peak_mm": 0.25 + 0.01 * index + 0.02 * authority.scalar_damage[:, 0],
        "slope_u20_u60_mpa_per_mm": (
            400.0 + 2.0 * index + 3.0 * authority.surface_profile_stats[:, 0]
        ),
        "normalized_prepeak_auc": (
            0.35 + 0.002 * index + 0.004 * authority.full_cscan_embedding[:, 0]
        ),
    }


@pytest.fixture(scope="module")
def synthetic_evaluation():
    authority = _authority()
    return authority, evaluate_p2_nested_lodo(
        authority, _targets(authority), protocol=TEST_PROTOCOL
    )


def test_registered_protocol_freezes_exact_search_space() -> None:
    assert REGISTERED_P2_PROTOCOL.ridge_alphas == (0.1, 1.0, 10.0, 100.0)
    assert REGISTERED_P2_PROTOCOL.pca_dimensions == (8, 16, 32)
    assert REGISTERED_P2_PROTOCOL.tie_tolerance == 1e-12
    assert P2_ENDPOINTS == (
        "extension_peak_mm",
        "slope_u20_u60_mpa_per_mm",
        "normalized_prepeak_auc",
    )
    assert P2_VIEWS == ("F0", "F1", "F2", "F3", "F4", "F5")


def test_nested_lodo_has_complete_outer_and_inner_coverage(
    synthetic_evaluation,
) -> None:
    authority, evaluation = synthetic_evaluation

    assert len(evaluation.fold_states) == 3 * 6 * 6
    assert len(evaluation.predictions) == len(authority.specimen_ids) * 3 * 6
    assert len(evaluation.metrics) == 3 * 6
    assert len(evaluation.domain_metrics) == 3 * 6 * 6
    assert len(evaluation.inner_scores) == (
        3 * 6 * 3 * 2 + 3 * 6 * 3 * 2 * 2
    )
    keys = {
        (row.specimen_id, row.endpoint, row.view_name)
        for row in evaluation.predictions
    }
    assert len(keys) == len(evaluation.predictions)
    assert {row.held_out_domain for row in evaluation.fold_states} == set(DOMAIN_ORDER)
    for state in evaluation.fold_states:
        assert state.held_out_domain not in state.fit_domains
        assert set(state.fit_domains) == set(DOMAIN_ORDER) - {state.held_out_domain}
        assert not {
            authority.specimen_ids[index]
            for index, domain in enumerate(authority.domain_ids)
            if domain == state.held_out_domain
        } & set(state.fit_specimen_ids)


def test_oof_errors_and_aggregate_metrics_are_exact(synthetic_evaluation) -> None:
    _authority_value, evaluation = synthetic_evaluation
    rows = [
        row
        for row in evaluation.predictions
        if row.endpoint == "extension_peak_mm" and row.view_name == "F2"
    ]
    metric = next(
        row
        for row in evaluation.metrics
        if row.endpoint == "extension_peak_mm" and row.view_name == "F2"
    )
    domain_mae = []
    for domain in DOMAIN_ORDER:
        selected = [row for row in rows if row.domain_id == domain]
        domain_mae.append(np.mean([row.absolute_error for row in selected]))
        assert all(
            row.standardized_absolute_error
            == pytest.approx(row.absolute_error / row.source_target_std)
            for row in selected
        )
    truth = np.asarray([row.truth for row in rows])
    prediction = np.asarray([row.prediction for row in rows])

    assert metric.equal_domain_mae == pytest.approx(np.mean(domain_mae))
    assert metric.pooled_rmse == pytest.approx(
        np.sqrt(np.mean((truth - prediction) ** 2))
    )
    assert metric.pooled_r2 == pytest.approx(
        1.0 - np.sum((truth - prediction) ** 2) / np.sum((truth - truth.mean()) ** 2)
    )


def test_candidate_ties_choose_lower_dimension_then_stronger_alpha() -> None:
    base = InnerCandidateScore(
        held_out_domain=DOMAIN_ORDER[0],
        endpoint=P2_ENDPOINTS[0],
        view_name="F3",
        ridge_alpha=0.1,
        pca_dimension=32,
        inner_domain_mae=tuple((domain, 1.0) for domain in DOMAIN_ORDER[1:]),
        inner_equal_domain_mae=1.0,
        selected=False,
    )
    candidates = (
        base,
        replace(base, ridge_alpha=0.1, pca_dimension=8, inner_equal_domain_mae=1.0 + 5e-13),
        replace(base, ridge_alpha=100.0, pca_dimension=8, inner_equal_domain_mae=1.0 + 8e-13),
    )

    selected = select_inner_candidate(candidates, tie_tolerance=1e-12)

    assert selected.pca_dimension == 8
    assert selected.ridge_alpha == 100.0


def test_outer_query_targets_cannot_change_fit_or_selection_state() -> None:
    authority = _authority()
    targets = _targets(authority)
    held_out = DOMAIN_ORDER[-1]
    mutated = {name: values.copy() for name, values in targets.items()}
    query = np.asarray(authority.domain_ids) == held_out
    for values in mutated.values():
        values[query] += 1_000_000.0

    first = evaluate_p2_outer_fold(
        authority, targets, held_out, protocol=TEST_PROTOCOL
    )
    second = evaluate_p2_outer_fold(
        authority, mutated, held_out, protocol=TEST_PROTOCOL
    )

    assert tuple(row.state_sha256 for row in first.fold_states) == tuple(
        row.state_sha256 for row in second.fold_states
    )
    assert tuple(row.prediction for row in first.predictions) == tuple(
        row.prediction for row in second.predictions
    )
    assert tuple(
        (
            row.endpoint,
            row.view_name,
            row.ridge_alpha,
            row.pca_dimension,
            row.inner_equal_domain_mae,
            row.selected,
        )
        for row in first.inner_scores
    ) == tuple(
        (
            row.endpoint,
            row.view_name,
            row.ridge_alpha,
            row.pca_dimension,
            row.inner_equal_domain_mae,
            row.selected,
        )
        for row in second.inner_scores
    )


def test_outer_query_features_cannot_change_fit_or_selection_state() -> None:
    authority = _authority()
    held_out = DOMAIN_ORDER[-1]
    sentinel = _authority(query_sentinel_domain=held_out)

    first = evaluate_p2_outer_fold(
        authority, _targets(authority), held_out, protocol=TEST_PROTOCOL
    )
    second = evaluate_p2_outer_fold(
        sentinel, _targets(authority), held_out, protocol=TEST_PROTOCOL
    )

    assert tuple(row.state_sha256 for row in first.fold_states) == tuple(
        row.state_sha256 for row in second.fold_states
    )
    assert tuple(
        (row.endpoint, row.view_name, row.ridge_alpha, row.pca_dimension, row.selected)
        for row in first.inner_scores
    ) == tuple(
        (row.endpoint, row.view_name, row.ridge_alpha, row.pca_dimension, row.selected)
        for row in second.inner_scores
    )


@pytest.mark.parametrize(
    "mutation",
    ("missing_endpoint", "extra_endpoint", "nonfinite", "wrong_length"),
)
def test_target_registry_fails_closed(mutation: str) -> None:
    authority = _authority()
    targets = _targets(authority)
    if mutation == "missing_endpoint":
        targets.pop(P2_ENDPOINTS[0])
    elif mutation == "extra_endpoint":
        targets["published_cai_strength_mpa"] = np.ones(len(authority.specimen_ids))
    elif mutation == "nonfinite":
        targets[P2_ENDPOINTS[0]][0] = np.nan
    else:
        targets[P2_ENDPOINTS[0]] = targets[P2_ENDPOINTS[0]][:-1]

    with pytest.raises(P2EvaluationError):
        evaluate_p2_outer_fold(
            authority, targets, DOMAIN_ORDER[0], protocol=TEST_PROTOCOL
        )
