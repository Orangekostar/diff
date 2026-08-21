from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cmc_bbdm.cpb_diffusion_marginalization as d8
import cmc_bbdm.cpb_diffusion_marginalization.pilot as pilot_module
from cmc_bbdm.cpb_diffusion_marginalization.authority import (
    issue_inner_fold,
    issue_search_view,
)
from cmc_bbdm.cpb_diffusion_marginalization.config import load_d8_config
from cmc_bbdm.cpb_diffusion_marginalization.pilot import (
    D8FeatureBundle,
    D8PilotDecision,
    D8PilotEvaluator,
    D8PilotRunResult,
    D8PilotStudyEvidence,
    RegisteredPilotAssets,
    ResidualProposalSet,
    _build_registered_feature_bundle,
    _pilot_json_mapping,
    _selected_diffusion_weight,
    build_candidate_residual_proposals,
    build_pilot_escalation_evidence,
    create_registered_pilot_evaluator,
    decide_pilot_escalation,
    evaluate_feature_bundle,
    load_registered_pilot_assets,
    run_registered_pilot,
)
from cmc_bbdm.cpb_diffusion_marginalization.regression import CandidateSpec
from cmc_bbdm.cpb_diffusion_marginalization.residuals import (
    P6ResidualBank,
    ResidualAuthority,
    ResidualFoldDraws,
    build_residual_bank_from_arrays,
)
from cmc_bbdm.cpb_diffusion_marginalization.search import (
    D8Candidate,
    SearchResult,
    robust_inner_objective,
)
from cmc_bbdm.cpb_diffusion_marginalization.selection import (
    EnsembleResult,
    FinalistResult,
    FrozenOuterSelection,
    RerankResult,
    RerankRow,
)
from cmc_bbdm.cpb_diffusion_marginalization.variants import (
    MorphologyThresholds,
    VariantBatch,
    VariantRecord,
)
from cmc_bbdm.cpb_physical_descriptors import PhysicalCalibration
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D8_CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_exploration.yaml"
P1_CONFIG = PROJECT_ROOT / "paper_v3/configs/p1_full_field_oracle.yaml"


def test_public_api_exposes_frozen_pilot_escalation_contract() -> None:
    assert d8.D8PilotDecision is D8PilotDecision
    assert d8.D8PilotStudyEvidence is D8PilotStudyEvidence
    assert d8.build_pilot_escalation_evidence is build_pilot_escalation_evidence
    assert d8.decide_pilot_escalation is decide_pilot_escalation


@pytest.mark.parametrize("value", (1, "[]"))
def test_pilot_json_mapping_rejects_wrong_types_as_type_errors(value: object) -> None:
    with pytest.raises(TypeError, match="mapping"):
        _pilot_json_mapping(value, label="pilot mapping")


def test_pilot_selection_rejects_missing_structures_as_type_errors() -> None:
    config = load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)
    with pytest.raises(TypeError, match="selection evidence"):
        _selected_diffusion_weight(
            {
                "outer_domain": config.outer_domains[0],
                "state_sha256": "1" * 64,
                "selected_candidates": None,
                "ensemble": {},
            },
            outer_domain=config.outer_domains[0],
            config=config,
        )


def _bank() -> P6ResidualBank:
    specimen_ids = ("a1", "a2", "b1", "b2")
    dataset_ids = ("a", "a", "b", "b")
    measured = np.zeros((4, 3, 64, 64), dtype=np.float32)
    grid = np.linspace(-0.2, 0.2, 64 * 64, dtype=np.float32).reshape(64, 64)
    values = np.stack(
        [
            np.stack((grid + offset, grid.T - offset, grid * (index + 1) / 4))
            for index, offset in enumerate((0.01, 0.02, -0.01, -0.02))
        ]
    ).astype(np.float32)
    draws = np.stack((values, values * 0.5), axis=1)
    authority = ResidualAuthority(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        measured_fields=measured,
        posterior_mean=np.mean(draws, axis=1, dtype=np.float64).astype(np.float32),
        posterior_variance=np.var(draws, axis=1, dtype=np.float64).astype(np.float32),
        source_sha256=tuple(str(index + 1) * 64 for index in range(4)),
    )
    folds = (
        ResidualFoldDraws(
            heldout_domain="a",
            specimen_ids=("a1", "a2"),
            checkpoint_train_ids=("b1", "b2"),
            checkpoint_train_domains=("b", "b"),
            checkpoint_scientific_digest="a" * 64,
            draws=draws[:2],
        ),
        ResidualFoldDraws(
            heldout_domain="b",
            specimen_ids=("b1", "b2"),
            checkpoint_train_ids=("a1", "a2"),
            checkpoint_train_domains=("a", "a"),
            checkpoint_scientific_digest="b" * 64,
            draws=draws[2:],
        ),
    )
    return build_residual_bank_from_arrays(authority, folds, draw_count=2)


def _decision_studies(
    config,
    *,
    trend_count: int,
    mismatch_count: int,
    freeze_count: int,
) -> tuple[D8PilotStudyEvidence, ...]:
    studies = []
    for index, outer in enumerate(config.outer_domains):
        inner = tuple(domain for domain in config.outer_domains if domain != outer)
        trend = index < trend_count
        mismatch = index < mismatch_count
        freeze = index < freeze_count
        studies.append(
            D8PilotStudyEvidence(
                outer_domain=outer,
                baseline_candidate_sha256=f"{index + 1:064x}",
                diffusion_candidate_sha256=f"{index + 7:064x}",
                baseline_objective=0.1000,
                diffusion_objective=0.0998 if freeze else 0.1000,
                improved_inner_domains=inner[:3] if trend else inner[:2],
                low_band_energy_fraction=0.50 if mismatch else 0.49,
                maximum_alpha_point_one_acceptance=0.49 if mismatch else 0.50,
                selected_diffusion_weight=0.05 if freeze else 0.049,
                selection_state_sha256=f"{index + 13:064x}",
                residual_bank_sha256="e" * 64,
            )
        )
    return tuple(studies)


@pytest.mark.parametrize(
    ("trend_count", "mismatch_count", "freeze_count", "expected"),
    (
        (3, 0, 6, "TRAIN_RESIDUAL_DIFFUSION"),
        (0, 3, 6, "TRAIN_RESIDUAL_DIFFUSION"),
        (0, 0, 3, "FREEZE_PILOT_FOR_OUTER_EVALUATION"),
        (0, 0, 2, "CLOSE_DIFFUSION_SPECIFIC_ROUTE"),
    ),
)
def test_pilot_escalation_uses_frozen_priority_and_thresholds(
    trend_count: int,
    mismatch_count: int,
    freeze_count: int,
    expected: str,
) -> None:
    config = load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)
    result = decide_pilot_escalation(
        _decision_studies(
            config,
            trend_count=trend_count,
            mismatch_count=mismatch_count,
            freeze_count=freeze_count,
        ),
        config=config,
    )
    assert type(result) is D8PilotDecision
    assert result.decision == expected
    assert len(result.trend_outer_studies) == trend_count
    assert len(result.mismatch_outer_studies) == mismatch_count
    assert len(result.freeze_outer_studies) == freeze_count
    assert result.residual_bank_sha256 == "e" * 64
    assert result.to_payload()["residual_bank_sha256"] == "e" * 64


def test_pilot_escalation_recomputes_trials_selection_and_residual_energy() -> None:
    config = load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)
    baseline = _candidate("B0", config_sha256=config.config_sha256)
    diffusion = _candidate("B5", config_sha256=config.config_sha256)
    rows: list[dict[str, str]] = []
    selections: list[dict[str, object]] = []
    for outer_index, outer in enumerate(config.outer_domains):
        inner = tuple(domain for domain in config.outer_domains if domain != outer)
        baseline_values = np.full(5, 0.100, dtype=np.float64)
        diffusion_values = np.asarray((0.098, 0.098, 0.098, 0.100, 0.100))
        acceptance = json.dumps(
            {
                domain: {
                    "accepted_proposals": 10,
                    "proposed_variants": 10,
                    "acceptance_rate": 1.0,
                }
                for domain in inner
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for candidate, values in (
            (baseline, baseline_values),
            (diffusion, diffusion_values),
        ):
            row = {
                "outer_fold": outer,
                "state": "COMPLETE",
                "control_id": candidate.control_id,
                "candidate_sha256": candidate.state_sha256,
                "objective": str(robust_inner_objective(values)),
                "alpha": str(candidate.alpha),
                "decomposition_family": candidate.decomposition_family,
                "band": candidate.band,
                "decomposition_parameters": json.dumps(
                    dict(candidate.decomposition_parameters),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "accepted_proposals": "50",
                "proposed_variants": "50",
                "acceptance_rate": "1.0",
                "acceptance_by_domain": acceptance,
            }
            for domain, value in zip(inner, values, strict=True):
                row[f"inner_mae__{domain}"] = str(value)
            rows.append(row)
        selections.append(
            {
                "outer_domain": outer,
                "state_sha256": f"{outer_index + 1:064x}",
                "selected_candidates": [diffusion.to_payload()],
                "ensemble": {
                    "candidate_sha256": [diffusion.state_sha256],
                    "weights": [1.0],
                },
            }
        )
    result = build_pilot_escalation_evidence(
        tuple(rows),
        selections=tuple(selections),
        bank=_bank(),
        config=config,
    )
    assert result.decision == "TRAIN_RESIDUAL_DIFFUSION"
    assert result.trend_outer_studies == config.outer_domains
    for study in result.studies:
        inner = tuple(domain for domain in config.outer_domains if domain != study.outer_domain)
        assert study.improved_inner_domains == inner[:3]
        assert study.baseline_candidate_sha256 == baseline.state_sha256
        assert study.diffusion_candidate_sha256 == diffusion.state_sha256
        assert 0.0 <= study.low_band_energy_fraction <= 1.0
        assert study.maximum_alpha_point_one_acceptance == 1.0
        assert study.selected_diffusion_weight == 1.0
        assert study.residual_bank_sha256 == result.residual_bank_sha256


def _candidate(
    control: str, *, seed: int = 20260820, config_sha256: str = "1" * 64
) -> D8Candidate:
    return D8Candidate(
        control_id=control,
        decomposition_family="gaussian",
        band="low" if control == "B1" else "high",
        decomposition_parameters={"sigma": 2.0},
        alpha=0.1,
        K_train=2,
        K_test=2,
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
            pca_dimension=4,
            regressor="ridge",
            parameters={"alpha": 1.0},
            seed=seed,
        ),
        seed=seed,
        config_sha256=config_sha256,
    )


def test_registered_residual_controls_have_distinct_deterministic_roles() -> None:
    bank = _bank()
    channels = np.meshgrid(
            np.linspace(-0.8, 0.8, 64, dtype=np.float32),
            np.linspace(-0.6, 0.6, 64, dtype=np.float32),
            indexing="ij",
        )
    source = np.stack((*channels, np.zeros((64, 64), dtype=np.float32)))
    raw = build_candidate_residual_proposals(
        _candidate("B0"),
        bank=bank,
        specimen_id="a1",
        dataset_id="a",
        source=source,
        fit_domains=("b",),
    )
    low = build_candidate_residual_proposals(
        _candidate("B1"),
        bank=bank,
        specimen_id="a1",
        dataset_id="a",
        source=source,
        fit_domains=("b",),
    )
    gaussian = build_candidate_residual_proposals(
        _candidate("B2"),
        bank=bank,
        specimen_id="a1",
        dataset_id="a",
        source=source,
        fit_domains=("b",),
    )
    phase = build_candidate_residual_proposals(
        _candidate("B3"),
        bank=bank,
        specimen_id="a1",
        dataset_id="a",
        source=source,
        fit_domains=("b",),
    )
    assert len(raw.residuals) == len(low.residuals) == 1
    np.testing.assert_array_equal(raw.residuals[0], np.zeros_like(source))
    assert np.var(source + low.residuals[0]) < np.var(source)
    assert len(gaussian.residuals) == len(phase.residuals) == 32
    assert gaussian.state_sha256 == build_candidate_residual_proposals(
        _candidate("B2"),
        bank=bank,
        specimen_id="a1",
        dataset_id="a",
        source=source,
        fit_domains=("b",),
    ).state_sha256
    assert gaussian.state_sha256 != phase.state_sha256


def test_empirical_and_diffusion_proposals_respect_inner_domain_authority() -> None:
    bank = _bank()
    source = np.zeros((3, 64, 64), dtype=np.float32)
    empirical = build_candidate_residual_proposals(
        _candidate("B4"),
        bank=bank,
        specimen_id="a1",
        dataset_id="a",
        source=source,
        fit_domains=("b",),
    )
    diffusion = build_candidate_residual_proposals(
        _candidate("B5"),
        bank=bank,
        specimen_id="a1",
        dataset_id="a",
        source=source,
        fit_domains=("b",),
    )
    assert len(empirical.residuals) == 32
    assert set(empirical.origin_dataset_ids) == {"b"}
    assert "a1" not in empirical.origin_specimen_ids
    assert len(diffusion.residuals) == bank.draw_count
    assert set(diffusion.origin_specimen_ids) == {"a1"}
    assert set(diffusion.origin_dataset_ids) == {"a"}
    assert all(not value.flags.writeable for value in diffusion.residuals)


def test_feature_bundle_scores_only_the_issued_inner_query_domain() -> None:
    config = load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)
    data = load_data(load_v3_config(P1_CONFIG, project_root=PROJECT_ROOT), PROJECT_ROOT)
    search = issue_search_view(data, outer_domain="74t7kcdgkr", config=config)
    fold = issue_inner_fold(search, query_domain="cgtnjyggtm")
    candidate = _candidate("B5", config_sha256=config.config_sha256)
    rows = search.specimen_count
    generator = np.random.default_rng(20260820)
    features = generator.normal(size=(rows, 2, 8))
    bundle = D8FeatureBundle(
        candidate_sha256=candidate.state_sha256,
        search_view_sha256=search.state_sha256,
        specimen_ids=search.specimen_ids,
        train_variant_features=features,
        query_variant_features=features,
        morphology_distances=np.zeros((rows, 2), dtype=np.float64),
        accepted_proposals=np.full(rows, 2, dtype=np.int64),
        proposed_variants=np.full(rows, 2, dtype=np.int64),
    )
    result = evaluate_feature_bundle(candidate, fold=fold, bundle=bundle)
    assert result.inner.query_domain == fold.query_domain
    assert result.prediction.query_specimen_ids == fold.query_specimen_ids
    assert result.inner.accepted_proposals == 2 * len(fold.query_indices)
    assert result.inner.proposed_variants == 2 * len(fold.query_indices)
    assert result.inner.mae == pytest.approx(
        np.mean(
            np.abs(result.prediction.predictions - result.prediction.targets),
            dtype=np.float64,
        )
    )
    changed = _candidate(
        "B5", seed=20260821, config_sha256=config.config_sha256
    )
    with pytest.raises(ValueError, match="candidate"):
        evaluate_feature_bundle(changed, fold=fold, bundle=bundle)


def test_feature_bundle_rejects_fractional_proposal_counts() -> None:
    with pytest.raises(ValueError, match="proposal counts must be integers"):
        D8FeatureBundle(
            candidate_sha256="1" * 64,
            search_view_sha256="2" * 64,
            specimen_ids=("specimen-a",),
            train_variant_features=np.zeros((1, 1, 2), dtype=np.float64),
            query_variant_features=np.zeros((1, 1, 2), dtype=np.float64),
            morphology_distances=np.zeros((1, 1), dtype=np.float64),
            accepted_proposals=np.asarray((1.5,), dtype=np.float64),
            proposed_variants=np.asarray((2.0,), dtype=np.float64),
        )


def test_pilot_evaluator_reuses_only_fold_invariant_candidate_bundles() -> None:
    config = load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)
    data = load_data(load_v3_config(P1_CONFIG, project_root=PROJECT_ROOT), PROJECT_ROOT)
    search = issue_search_view(data, outer_domain="74t7kcdgkr", config=config)
    folds = (
        issue_inner_fold(search, query_domain="cgtnjyggtm"),
        issue_inner_fold(search, query_domain="w68dtmpfyf"),
    )
    calls: list[tuple[str, str]] = []

    def builder(candidate: D8Candidate, fold) -> D8FeatureBundle:
        calls.append((candidate.state_sha256, fold.state_sha256))
        rows = search.specimen_count
        generator = np.random.default_rng(candidate.seed)
        features = generator.normal(size=(rows, candidate.K_test, 8))
        train = features[:, : candidate.K_train]
        return D8FeatureBundle(
            candidate_sha256=candidate.state_sha256,
            search_view_sha256=search.state_sha256,
            specimen_ids=search.specimen_ids,
            train_variant_features=train,
            query_variant_features=features,
            morphology_distances=np.zeros(
                (rows, candidate.K_test), dtype=np.float64
            ),
            accepted_proposals=np.full(rows, 2, dtype=np.int64),
            proposed_variants=np.full(rows, 2, dtype=np.int64),
        )

    evaluator = D8PilotEvaluator(builder)
    diffusion = _candidate("B5", config_sha256=config.config_sha256)
    evaluator.evaluate(diffusion, folds[0])
    evaluator.evaluate(diffusion, folds[1])
    evaluator.predict(diffusion, folds[0])
    assert len(calls) == 1

    empirical = _candidate("B4", config_sha256=config.config_sha256)
    evaluator.evaluate(empirical, folds[0])
    evaluator.evaluate(empirical, folds[1])
    assert len(calls) == 3


def test_registered_bundle_builder_encodes_the_maximum_k_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)
    data = load_data(load_v3_config(P1_CONFIG, project_root=PROJECT_ROOT), PROJECT_ROOT)
    search = issue_search_view(data, outer_domain="74t7kcdgkr", config=config)
    fold = issue_inner_fold(search, query_domain="cgtnjyggtm")
    candidate = _candidate("B5", config_sha256=config.config_sha256)
    rows = search.specimen_count
    native = np.zeros((8, 8, 3), dtype=np.uint8)
    native.setflags(write=False)
    calibrations = {
        domain: PhysicalCalibration(
            dataset_id=domain,
            field_width_mm=75.0,
            field_height_mm=75.0,
            calibration_basis="test",
            evidence_path="test",
            evidence_sha256="a" * 64,
        )
        for domain in set(search.dataset_ids)
    }
    assets = RegisteredPilotAssets(
        specimen_ids=search.specimen_ids,
        dataset_ids=search.dataset_ids,
        measured_fields=np.zeros((rows, 3, 64, 64), dtype=np.float32),
        native_images=tuple(native for _ in range(rows)),
        source_sha256=tuple("1" * 64 for _ in range(rows)),
        calibrations=calibrations,
    )
    worker_threads: set[int] = set()
    worker_lock = threading.Lock()
    fit_domain_rosters: set[tuple[str, ...]] = set()

    def fake_proposals(candidate, **kwargs) -> ResidualProposalSet:
        with worker_lock:
            fit_domain_rosters.add(kwargs["fit_domains"])
        residuals = tuple(
            np.zeros((3, 64, 64), dtype=np.float32) for _ in range(2)
        )
        return ResidualProposalSet(
            control_id=candidate.control_id,
            specimen_id=kwargs["specimen_id"],
            dataset_id=kwargs["dataset_id"],
            residuals=residuals,
            origin_specimen_ids=(kwargs["specimen_id"],) * 2,
            origin_dataset_ids=(kwargs["dataset_id"],) * 2,
            state_sha256="2" * 64,
        )

    def fake_batch(source, residuals, **kwargs) -> VariantBatch:
        with worker_lock:
            worker_threads.add(threading.get_ident())
        time.sleep(0.002)
        records = tuple(
            VariantRecord(
                variant=source,
                encoder_image=kwargs["native_source"],
                accepted=True,
                area_deviation=0.0,
                width_deviation=0.0,
                height_deviation=0.0,
                centroid_shift_mm=0.0,
                low_frequency_correlation=1.0,
                radial_profile_correlation=1.0,
                failed_conditions=(),
                state_sha256=str(index + 3) * 64,
            )
            for index in range(kwargs["requested_count"])
        )
        return VariantBatch(
            variants=tuple(record.variant for record in records),
            encoder_images=tuple(record.encoder_image for record in records),
            records=records,
            proposal_count=len(records),
            accepted_count=len(records),
            fallback_count=0,
            acceptance_rate=1.0,
            state_sha256="9" * 64,
        )

    class FakeEncoder:
        calls = 0

        def encode(self, variants, *, layer):
            del layer
            self.calls += 1
            return np.zeros((len(variants), len(variants[0]), 8), dtype=np.float32)

    monkeypatch.setattr(
        pilot_module, "build_candidate_residual_proposals", fake_proposals
    )
    monkeypatch.setattr(pilot_module, "build_variant_batch", fake_batch)
    encoder = FakeEncoder()
    bundle = _build_registered_feature_bundle(
        candidate,
        fold=fold,
        assets=assets,
        bank=_bank(),
        encoder=encoder,
    )
    assert encoder.calls == 1
    assert bundle.train_variant_features.shape == (rows, candidate.K_train, 8)
    assert bundle.query_variant_features.shape == (rows, candidate.K_test, 8)
    assert np.all(bundle.accepted_proposals == 2)
    assert np.all(bundle.proposed_variants == 2)
    assert 1 < len(worker_threads) <= 8
    assert fit_domain_rosters == {
        tuple(
            domain
            for domain in config.outer_domains
            if domain not in {fold.outer_domain, fold.query_domain}
        )
    }


def test_registered_pilot_assets_reject_extra_calibration_domains() -> None:
    domain = "domain-a"
    calibration = PhysicalCalibration(
        dataset_id=domain,
        field_width_mm=75.0,
        field_height_mm=75.0,
        calibration_basis="test",
        evidence_path="test",
        evidence_sha256="a" * 64,
    )
    extra = PhysicalCalibration(
        dataset_id="domain-extra",
        field_width_mm=75.0,
        field_height_mm=75.0,
        calibration_basis="test",
        evidence_path="test",
        evidence_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="calibration mapping does not match"):
        RegisteredPilotAssets(
            specimen_ids=("specimen-a",),
            dataset_ids=(domain,),
            measured_fields=np.zeros((1, 3, 64, 64), dtype=np.float32),
            native_images=(np.zeros((8, 8, 3), dtype=np.uint8),),
            source_sha256=("1" * 64,),
            calibrations={domain: calibration, extra.dataset_id: extra},
        )


def test_registered_pilot_assets_match_the_full_v3_authority() -> None:
    config = load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)
    data = load_data(load_v3_config(P1_CONFIG, project_root=PROJECT_ROOT), PROJECT_ROOT)
    assets = load_registered_pilot_assets(
        data, config=config, project_root=PROJECT_ROOT
    )
    assert assets.specimen_ids == tuple(data.sample_ids.tolist())
    assert assets.dataset_ids == tuple(data.dataset_ids.tolist())
    assert assets.measured_fields.shape == (276, 3, 64, 64)
    assert len(assets.native_images) == len(assets.source_sha256) == 276
    assert set(assets.calibrations) == set(assets.dataset_ids)
    assert assets.measured_fields.flags.writeable is False


def test_registered_pilot_evaluator_factory_binds_assets_bank_and_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)
    data = object()
    bank = object()
    assets = SimpleNamespace(
        specimen_ids=("specimen-a",),
        dataset_ids=("domain-a",),
        source_sha256=("1" * 64,),
    )
    encoder = object()
    marker = object()
    observed: dict[str, object] = {}

    def fake_assets(value, **kwargs):
        observed["data"] = value
        observed["asset_kwargs"] = kwargs
        return assets

    def fake_validate(value, **kwargs):
        observed["bank"] = value
        observed["bank_kwargs"] = kwargs
        return "a" * 64

    def fake_encoder(**kwargs):
        observed["encoder_kwargs"] = kwargs
        return encoder

    def fake_bundle(candidate, **kwargs):
        observed["candidate"] = candidate
        observed["bundle_kwargs"] = kwargs
        return marker

    monkeypatch.setattr(pilot_module, "load_registered_pilot_assets", fake_assets)
    monkeypatch.setattr(pilot_module, "validate_residual_bank", fake_validate)
    monkeypatch.setattr(pilot_module, "create_d8_frozen_encoder", fake_encoder)
    monkeypatch.setattr(pilot_module, "_build_registered_feature_bundle", fake_bundle)
    evaluator = create_registered_pilot_evaluator(
        data,
        config=config,
        bank=bank,
        project_root=PROJECT_ROOT,
        device="cuda:0",
    )
    candidate = object()
    fold = object()
    assert type(evaluator) is D8PilotEvaluator
    assert evaluator._bundle_builder(candidate, fold) is marker
    assert observed["data"] is data
    assert observed["bank"] is bank
    assert observed["candidate"] is candidate
    assert observed["bundle_kwargs"] == {
        "fold": fold,
        "assets": assets,
        "bank": bank,
        "encoder": encoder,
    }


def test_registered_pilot_runs_six_search_rerank_ensemble_freeze_chains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)
    events: list[tuple[str, str]] = []
    candidates = tuple(
        _candidate(
            "B5",
            seed=config.seed + index,
            config_sha256=config.config_sha256,
        )
        for index in range(12)
    )

    def fake_view(data, *, outer_domain, config):
        del data
        events.append((outer_domain, "view"))
        return SimpleNamespace(
            outer_domain=outer_domain,
            config_sha256=config.config_sha256,
            state_sha256=(str(config.outer_domains.index(outer_domain) + 1) * 64),
            specimen_ids=("a", "b", "c", "d", "e"),
            dataset_ids=("d0", "d1", "d2", "d3", "d4"),
        )

    evaluator = SimpleNamespace(evaluate=object(), predict=object())
    monkeypatch.setattr(
        pilot_module,
        "create_registered_pilot_evaluator",
        lambda *args, **kwargs: evaluator,
    )
    monkeypatch.setattr(pilot_module, "issue_search_view", fake_view)

    def fake_search(view, *, config, output, evaluator):
        del config, output
        assert evaluator is not None
        events.append((view.outer_domain, "search"))
        return SearchResult(
            outer_domain=view.outer_domain,
            initial_trial_count=72,
            trial_count=72,
            completed_count=72,
            pruned_count=0,
            failed_count=0,
            selected_candidates=candidates,
            study_database="study.db",
        )

    monkeypatch.setattr(pilot_module, "run_outer_search", fake_search)

    def row(candidate: D8Candidate, value: float) -> RerankRow:
        return RerankRow(
            candidate=candidate,
            seeds=(),
            domain_mae=tuple((f"d{index}", value) for index in range(5)),
            mean_mae=value,
            worst_mae=value,
            domain_sd=0.0,
            objective=value,
            complexity_key=(1,),
            oof_targets=np.zeros(5, dtype=np.float64),
            oof_predictions=np.full(5, value, dtype=np.float64),
            state_sha256=candidate.state_sha256,
        )

    def fake_rerank(values, *, view, seeds, evaluator):
        assert values == candidates
        assert seeds == config.rerank_seeds
        assert evaluator is not None
        events.append((view.outer_domain, "rerank"))
        rows = tuple(row(candidate, 0.08 + index * 0.001) for index, candidate in enumerate(values))
        return RerankResult(
            outer_domain=view.outer_domain,
            config_sha256=config.config_sha256,
            search_view_sha256=view.state_sha256,
            seed_count=3,
            rows=rows,
            finalists=rows[:4],
            state_sha256="a" * 64,
        )

    monkeypatch.setattr(pilot_module, "rerank_candidates", fake_rerank)

    def fake_finalists(values, *, view, seeds, K_test_values, evaluator):
        assert len(values) == 4
        assert seeds == config.rerank_seeds
        assert K_test_values == (8, 16)
        assert evaluator is not None
        events.append((view.outer_domain, "finalists"))
        rows = tuple(row(candidate, 0.075 + index * 0.001) for index, candidate in enumerate(values))
        return FinalistResult(
            outer_domain=view.outer_domain,
            config_sha256=config.config_sha256,
            search_view_sha256=view.state_sha256,
            cells=rows,
            selected=rows,
            state_sha256="b" * 64,
        )

    monkeypatch.setattr(pilot_module, "evaluate_finalists", fake_finalists)

    def fake_ensemble(*args, **kwargs):
        outer = events[-1][0]
        events.append((outer, "ensemble"))
        return EnsembleResult(
            candidate_sha256=tuple(item.state_sha256 for item in candidates[:4]),
            accepted=False,
            weights=np.asarray((1.0, 0.0, 0.0, 0.0)),
            crossfit_weights=np.tile(np.asarray((1.0, 0.0, 0.0, 0.0)), (5, 1)),
            best_member_index=0,
            best_member_objective=0.075,
            objective=0.075,
            objective_gain=0.0,
            predictions=np.full(5, 0.075),
            state_sha256="c" * 64,
        )

    monkeypatch.setattr(pilot_module, "fit_nonnegative_ensemble", fake_ensemble)

    def fake_freeze(reranked, *, finalists, ensemble, view, output):
        del reranked, finalists, ensemble
        events.append((view.outer_domain, "freeze"))
        assert output.name == f"{view.outer_domain}.json"
        return FrozenOuterSelection(
            outer_domain=view.outer_domain,
            config_sha256=config.config_sha256,
            search_view_sha256=view.state_sha256,
            selected_candidate_sha256=tuple(
                item.state_sha256 for item in candidates[:4]
            ),
            ensemble_sha256="c" * 64,
            outer_evaluation_started=False,
            state_sha256="d" * 64,
        )

    monkeypatch.setattr(pilot_module, "freeze_outer_selection", fake_freeze)
    result = run_registered_pilot(
        object(),
        config=config,
        bank=SimpleNamespace(state_sha256="e" * 64),
        project_root=PROJECT_ROOT,
        output=tmp_path,
        device="cuda:0",
    )
    assert type(result) is D8PilotRunResult
    assert result.outer_domains == config.outer_domains
    assert result.outer_evaluation_count == 0
    assert tuple(event for _, event in events) == (
        "view",
        "search",
        "rerank",
        "finalists",
        "ensemble",
        "freeze",
    ) * 6
