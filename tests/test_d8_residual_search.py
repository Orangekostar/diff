from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cmc_bbdm.cpb_diffusion_marginalization.authority import (
    issue_inner_fold,
    issue_search_view,
)
from cmc_bbdm.cpb_diffusion_marginalization.config import (
    DOMAIN_ORDER,
    load_d8_config,
)
from cmc_bbdm.cpb_diffusion_marginalization.pilot import (
    D8FeatureBundle,
    load_registered_pilot_assets,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_config import (
    load_residual_diffusion_config,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_model import (
    ResidualCheckpoint,
    build_residual_unet,
    freeze_residual_checkpoint,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_search import (
    ResidualCellEvaluation,
    ResidualCellRun,
    ResidualFeatureBundle,
    ResidualIncumbentEvidence,
    ResidualOuterSearchRun,
    ResidualSearchCell,
    ResidualSearchError,
    StageAOuterPromotion,
    build_residual_feature_bundle,
    evaluate_residual_feature_bundle,
    load_b0_incumbent_evidence,
    load_pilot_incumbent_evidence,
    load_pilot_scaffold_candidates,
    promote_stage_a,
    promote_stage_a_outer,
    run_residual_outer_search,
    run_residual_search_cell,
    select_stage_b_pipeline,
    stage_a_cell_keys,
    stage_b_cell_keys,
    summarize_candidate_cells,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_targets import (
    build_outer_fit_residual_target_batch,
    load_pilot_diffusion_scaffolds,
    load_search_residual_field_bank,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_training import (
    EpochLossRecord,
    ResidualFinalTrainingResult,
    ResidualTrainingResult,
)
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_residual_diffusion.yaml"


@lru_cache(maxsize=1)
def _config():
    return load_residual_diffusion_config(CONFIG, project_root=PROJECT_ROOT)


@lru_cache(maxsize=1)
def _data_authorities():
    residual = _config()
    exploration = load_d8_config(
        PROJECT_ROOT / residual.sources["exploration_config"].path,
        project_root=PROJECT_ROOT,
    )
    v3_config = load_v3_config(
        PROJECT_ROOT / exploration.sources["p1_config"].path,
        project_root=PROJECT_ROOT,
    )
    data = load_data(v3_config, PROJECT_ROOT)
    return exploration, data


@lru_cache(maxsize=1)
def _inner_fold():
    exploration, data = _data_authorities()
    search = issue_search_view(data, outer_domain=DOMAIN_ORDER[0], config=exploration)
    return issue_inner_fold(search, query_domain=DOMAIN_ORDER[1])


def _evaluation(
    cell,
    *,
    error: float,
    predictions: np.ndarray | None = None,
    accepted: int = 4,
    proposed: int = 4,
) -> ResidualCellEvaluation:
    targets = np.asarray((0.25, 0.75), dtype=np.float64)
    values = (
        np.asarray(predictions, dtype=np.float64)
        if predictions is not None
        else targets + error
    )
    return ResidualCellEvaluation(
        cell=cell,
        specimen_ids=(
            f"{cell.query_domain}-specimen-0",
            f"{cell.query_domain}-specimen-1",
        ),
        targets=targets,
        predictions=values,
        accepted_proposals=accepted,
        proposed_variants=proposed,
        checkpoint_sha256=hashlib.sha256(
            f"checkpoint:{cell.state_sha256}".encode("ascii")
        ).hexdigest(),
        prediction_sha256=hashlib.sha256(
            f"prediction:{cell.state_sha256}".encode("ascii")
        ).hexdigest(),
    )


def test_stage_a_and_stage_b_cartesian_rosters_are_exact() -> None:
    config = _config()
    stage_a = stage_a_cell_keys(config)
    assert len(stage_a) == 6 * 5 * 8
    assert len({cell.state_sha256 for cell in stage_a}) == len(stage_a)
    assert {
        (
            cell.outer_domain,
            cell.query_domain,
            cell.candidate_id,
            cell.training_seed,
        )
        for cell in stage_a
    } == {
        (outer, query, candidate_id, config.screening_seed)
        for outer in DOMAIN_ORDER
        for query in DOMAIN_ORDER
        if query != outer
        for candidate_id in config.candidate_ids
    }
    finalists = {
        outer: ("RD0", "RD1")
        for outer in DOMAIN_ORDER
    }
    stage_b = stage_b_cell_keys(config, finalists=finalists)
    assert len(stage_b) == 6 * 5 * 2 * 3
    assert {
        cell.training_seed for cell in stage_b
    } == set(config.training_seeds)


def test_stage_a_promotes_exactly_two_eligible_candidates_per_outer() -> None:
    config = _config()
    evaluations = tuple(
        _evaluation(
            cell,
            error=0.01 * int(cell.candidate_id[2:]),
        )
        for cell in stage_a_cell_keys(config)
    )

    promotion = promote_stage_a(evaluations, config=config)

    assert set(promotion.finalists) == set(DOMAIN_ORDER)
    assert all(
        candidate_ids == ("RD0", "RD1")
        for candidate_ids in promotion.finalists.values()
    )
    assert len(promotion.summaries) == 6 * 8


def test_stage_a_outer_promotion_is_exactly_one_registered_study() -> None:
    config = _config()
    outer = DOMAIN_ORDER[0]
    evaluations = tuple(
        _evaluation(cell, error=0.01 * int(cell.candidate_id[2:]))
        for cell in stage_a_cell_keys(config)
        if cell.outer_domain == outer
    )

    promotion = promote_stage_a_outer(evaluations, config=config)

    assert type(promotion) is StageAOuterPromotion
    assert promotion.outer_domain == outer
    assert promotion.finalists == ("RD0", "RD1")
    assert len(promotion.summaries) == 8


def test_smoke_stage_a_can_rank_ineligible_candidates_without_weakening_production() -> None:
    config = _config()
    outer = DOMAIN_ORDER[0]
    evaluations = tuple(
        _evaluation(
            cell,
            error=0.01 * int(cell.candidate_id[2:]),
            accepted=0,
            proposed=4,
        )
        for cell in stage_a_cell_keys(config)
        if cell.outer_domain == outer
    )

    with pytest.raises(ResidualSearchError, match="fewer than two"):
        promote_stage_a_outer(evaluations, config=config)

    promotion = promote_stage_a_outer(
        evaluations,
        config=config,
        test_scale_override=True,
    )

    assert promotion.finalists == ("RD0", "RD1")
    assert promotion.test_scale_override is True
    assert all(not summary.eligible for summary in promotion.summaries)


def test_stage_b_averages_seed_predictions_before_domain_mae() -> None:
    config = _config()
    outer = DOMAIN_ORDER[0]
    cells = tuple(
        cell
        for cell in stage_b_cell_keys(
            config,
            finalists={
                domain: ("RD0", "RD1")
                for domain in DOMAIN_ORDER
            },
        )
        if cell.outer_domain == outer and cell.candidate_id == "RD0"
    )
    offsets = dict(zip(config.training_seeds, (-0.3, 0.0, 0.3), strict=True))
    evaluations = tuple(
        _evaluation(
            cell,
            error=0.0,
            predictions=np.asarray((0.25, 0.75)) + offsets[cell.training_seed],
        )
        for cell in cells
    )

    summary = summarize_candidate_cells(
        evaluations,
        config=config,
        stage="B",
    )

    assert summary.domain_mae == tuple((domain, 0.0) for domain in DOMAIN_ORDER[1:])
    assert summary.mean_mae == 0.0
    assert summary.worst_mae == 0.0
    assert summary.domain_sd == 0.0
    assert summary.objective == 0.0
    assert summary.eligible


def test_candidate_is_ineligible_when_one_query_domain_misses_gate() -> None:
    config = _config()
    outer = DOMAIN_ORDER[0]
    cells = tuple(
        cell
        for cell in stage_a_cell_keys(config)
        if cell.outer_domain == outer and cell.candidate_id == "RD0"
    )
    failed_domain = cells[0].query_domain
    evaluations = tuple(
        _evaluation(
            cell,
            error=0.0,
            accepted=0 if cell.query_domain == failed_domain else 4,
            proposed=4,
        )
        for cell in cells
    )

    summary = summarize_candidate_cells(
        evaluations,
        config=config,
        stage="A",
    )

    assert not summary.eligible
    assert summary.failed_domains == (failed_domain,)
    assert dict(summary.domain_acceptance)[failed_domain] == 0.0


def _stage_b_evaluations(
    *,
    outer: str,
    candidate_errors: dict[str, float],
) -> tuple[ResidualCellEvaluation, ...]:
    config = _config()
    finalists = {
        domain: tuple(candidate_errors)
        for domain in DOMAIN_ORDER
    }
    return tuple(
        _evaluation(cell, error=candidate_errors[cell.candidate_id])
        for cell in stage_b_cell_keys(config, finalists=finalists)
        if cell.outer_domain == outer
    )


def _incumbent(
    *,
    pipeline_id: str,
    outer: str,
    error: float,
) -> ResidualIncumbentEvidence:
    config = _config()
    residual = summarize_candidate_cells(
        tuple(
            value
            for value in _stage_b_evaluations(
                outer=outer,
                candidate_errors={"RD0": 0.08, "RD1": 0.09},
            )
            if value.cell.candidate_id == "RD0"
        ),
        config=config,
        stage="B",
    )
    return ResidualIncumbentEvidence(
        pipeline_id=pipeline_id,
        outer_domain=outer,
        specimen_ids=residual.oof_specimen_ids,
        domain_ids=residual.oof_domain_ids,
        targets=residual.oof_targets,
        predictions=residual.oof_targets + error,
        evidence_sha256=hashlib.sha256(
            f"{pipeline_id}:{outer}:{error}".encode("ascii")
        ).hexdigest(),
    )


def test_stage_b_selects_residual_only_after_registered_incumbent_margin() -> None:
    config = _config()
    outer = DOMAIN_ORDER[0]
    evaluations = _stage_b_evaluations(
        outer=outer,
        candidate_errors={"RD0": 0.08, "RD1": 0.10},
    )
    incumbents = (
        _incumbent(pipeline_id="PILOT", outer=outer, error=0.09),
        _incumbent(pipeline_id="B0", outer=outer, error=0.11),
    )

    selected = select_stage_b_pipeline(
        evaluations,
        incumbents=incumbents,
        finalists=("RD0", "RD1"),
        config=config,
    )

    assert selected.best_incumbent.pipeline_id == "PILOT"
    assert selected.best_residual.candidate_id == "RD0"
    assert selected.residual_promoted
    assert selected.selected_pipeline == "RESIDUAL"
    assert selected.selected_components == ("RD0",)
    assert selected.requires_final_residual_checkpoints


def test_stage_b_keeps_incumbent_when_registered_margin_is_not_met() -> None:
    config = _config()
    outer = DOMAIN_ORDER[0]
    evaluations = _stage_b_evaluations(
        outer=outer,
        candidate_errors={"RD0": 0.08995, "RD1": 0.10},
    )
    incumbents = (
        _incumbent(pipeline_id="PILOT", outer=outer, error=0.09),
        _incumbent(pipeline_id="B0", outer=outer, error=0.11),
    )

    selected = select_stage_b_pipeline(
        evaluations,
        incumbents=incumbents,
        finalists=("RD0", "RD1"),
        config=config,
    )

    assert not selected.residual_promoted
    assert selected.best_residual is not None
    assert selected.best_residual.candidate_id == "RD0"
    assert selected.residual_improvement == pytest.approx(0.0000625, abs=1.0e-12)
    assert selected.residual_improvement < config.promotion_margin
    assert selected.selected_pipeline == "INCUMBENT"
    assert selected.selected_components == ("PILOT",)
    assert not selected.requires_final_residual_checkpoints


def test_stage_b_rejects_incumbent_identity_mismatch() -> None:
    config = _config()
    outer = DOMAIN_ORDER[0]
    evaluations = _stage_b_evaluations(
        outer=outer,
        candidate_errors={"RD0": 0.08, "RD1": 0.10},
    )
    pilot = _incumbent(pipeline_id="PILOT", outer=outer, error=0.09)
    b0 = _incumbent(pipeline_id="B0", outer=outer, error=0.11)
    mismatched = ResidualIncumbentEvidence(
        pipeline_id="B0",
        outer_domain=outer,
        specimen_ids=tuple(reversed(b0.specimen_ids)),
        domain_ids=b0.domain_ids,
        targets=b0.targets,
        predictions=b0.predictions,
        evidence_sha256=b0.evidence_sha256,
    )

    with np.testing.assert_raises_regex(ValueError, "identit"):
        select_stage_b_pipeline(
            evaluations,
            incumbents=(pilot, mismatched),
            finalists=("RD0", "RD1"),
            config=config,
        )


def test_stage_b_promotes_crossfit_ensemble_only_for_additional_gain() -> None:
    config = _config()
    outer = DOMAIN_ORDER[0]
    finalists = {domain: ("RD0", "RD1") for domain in DOMAIN_ORDER}
    evaluations = []
    for cell in stage_b_cell_keys(config, finalists=finalists):
        if cell.outer_domain != outer:
            continue
        targets = np.asarray((0.25, 0.75), dtype=np.float64)
        errors = (
            np.asarray((0.08, -0.02), dtype=np.float64)
            if cell.candidate_id == "RD0"
            else np.asarray((0.15, 0.15), dtype=np.float64)
        )
        evaluations.append(
            _evaluation(cell, error=0.0, predictions=targets + errors)
        )
    residual = summarize_candidate_cells(
        tuple(
            value for value in evaluations if value.cell.candidate_id == "RD0"
        ),
        config=config,
        stage="B",
    )
    pilot = ResidualIncumbentEvidence(
        pipeline_id="PILOT",
        outer_domain=outer,
        specimen_ids=residual.oof_specimen_ids,
        domain_ids=residual.oof_domain_ids,
        targets=residual.oof_targets,
        predictions=residual.oof_targets
        + np.tile(np.asarray((-0.02, 0.10)), len(DOMAIN_ORDER) - 1),
        evidence_sha256="a" * 64,
    )
    b0 = ResidualIncumbentEvidence(
        pipeline_id="B0",
        outer_domain=outer,
        specimen_ids=residual.oof_specimen_ids,
        domain_ids=residual.oof_domain_ids,
        targets=residual.oof_targets,
        predictions=residual.oof_targets + 0.20,
        evidence_sha256="b" * 64,
    )

    selected = select_stage_b_pipeline(
        tuple(evaluations),
        incumbents=(pilot, b0),
        finalists=("RD0", "RD1"),
        config=config,
    )

    assert selected.residual_promoted
    assert selected.ensemble_promoted
    assert selected.ensemble is not None
    assert selected.ensemble.objective_gain >= config.ensemble_margin
    assert selected.selected_pipeline == "ENSEMBLE"
    assert selected.selected_components == ("RD0", "PILOT")


def test_full_pilot_scaffold_candidates_are_recovered_from_validated_trials() -> None:
    config = _config()

    candidates = load_pilot_scaffold_candidates(
        config,
        project_root=PROJECT_ROOT,
    )

    assert tuple(candidates) == DOMAIN_ORDER
    assert all(
        candidate.state_sha256 in {
            item.candidate_sha256
            for item in load_pilot_diffusion_scaffolds(
                config,
                project_root=PROJECT_ROOT,
            ).values()
        }
        for candidate in candidates.values()
    )
    assert all(candidate.K_train in {1, 2, 4, 8, 16} for candidate in candidates.values())
    assert all(candidate.K_test in {1, 2, 4, 8, 16} for candidate in candidates.values())


def test_pilot_incumbents_reuse_exact_validated_five_domain_oof_vectors() -> None:
    config = _config()
    exploration, data = _data_authorities()

    incumbents = load_pilot_incumbent_evidence(
        config,
        project_root=PROJECT_ROOT,
    )

    assert tuple(value.outer_domain for value in incumbents) == DOMAIN_ORDER
    assert all(value.pipeline_id == "PILOT" for value in incumbents)
    for evidence in incumbents:
        search = issue_search_view(
            data,
            outer_domain=evidence.outer_domain,
            config=exploration,
        )
        assert evidence.specimen_ids == search.specimen_ids
        assert evidence.domain_ids == search.dataset_ids
        np.testing.assert_array_equal(evidence.targets, search.data_view.cai_ratio)
        assert evidence.predictions.shape == evidence.targets.shape


def test_b0_incumbents_recompute_exact_p1_inner_replay_authority() -> None:
    config = _config()
    exploration, data = _data_authorities()

    incumbents = load_b0_incumbent_evidence(
        data,
        config=config,
        project_root=PROJECT_ROOT,
    )

    assert tuple(value.outer_domain for value in incumbents) == DOMAIN_ORDER
    assert all(value.pipeline_id == "B0" for value in incumbents)
    for evidence in incumbents:
        search = issue_search_view(
            data,
            outer_domain=evidence.outer_domain,
            config=exploration,
        )
        assert evidence.specimen_ids == search.specimen_ids
        assert evidence.domain_ids == search.dataset_ids
        np.testing.assert_array_equal(evidence.targets, search.data_view.cai_ratio)
        assert evidence.predictions.shape == evidence.targets.shape


def test_residual_feature_bundle_evaluates_only_registered_inner_query() -> None:
    config = _config()
    fold = _inner_fold()
    pilot = load_pilot_scaffold_candidates(
        config,
        project_root=PROJECT_ROOT,
    )[fold.outer_domain]
    rows = fold.search_view.specimen_count
    feature_count = max(64, pilot.regressor_spec.pca_dimension)
    generator = np.random.Generator(np.random.PCG64(20260823))
    base = generator.normal(size=(rows, feature_count))
    train = np.stack(
        tuple(base + index * 1.0e-3 for index in range(pilot.K_train)),
        axis=1,
    )
    query = np.stack(
        tuple(base + index * 1.0e-3 for index in range(pilot.K_test)),
        axis=1,
    )
    features = D8FeatureBundle(
        candidate_sha256=pilot.state_sha256,
        search_view_sha256=fold.search_view.state_sha256,
        specimen_ids=fold.search_view.specimen_ids,
        train_variant_features=train,
        query_variant_features=query,
        morphology_distances=np.zeros((rows, pilot.K_test), dtype=np.float64),
        accepted_proposals=np.full(rows, max(pilot.K_train, pilot.K_test)),
        proposed_variants=np.full(rows, 32),
    )
    cell = ResidualSearchCell(
        "A",
        fold.outer_domain,
        fold.query_domain,
        "RD0",
        config.screening_seed,
    )
    bundle = ResidualFeatureBundle(
        cell=cell,
        pilot_candidate_sha256=pilot.state_sha256,
        checkpoint_sha256="a" * 64,
        sampled_target_sha256="b" * 64,
        scaffold_sha256="d" * 64,
        field_bank_sha256="e" * 64,
        target_state_sha256="f" * 64,
        asset_state_sha256="0" * 64,
        feature_bundle=features,
        variant_state_sha256=tuple("c" * 64 for _ in range(rows)),
    )

    result = evaluate_residual_feature_bundle(
        cell,
        fold=fold,
        pilot_candidate=pilot,
        bundle=bundle,
    )

    query_indices = np.asarray(fold.query_indices, dtype=np.int64)
    assert result.specimen_ids == tuple(
        fold.search_view.specimen_ids[int(index)] for index in query_indices
    )
    assert result.predictions.shape == result.targets.shape == (len(query_indices),)
    assert result.accepted_proposals == len(query_indices) * max(
        pilot.K_train, pilot.K_test
    )
    assert result.proposed_variants == len(query_indices) * 32
    assert result.checkpoint_sha256 == bundle.checkpoint_sha256


def test_residual_feature_builder_uses_frozen_checkpoint_before_query_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    exploration, data = _data_authorities()
    fold = _inner_fold()
    scaffolds = load_pilot_diffusion_scaffolds(
        config,
        project_root=PROJECT_ROOT,
    )
    pilot = load_pilot_scaffold_candidates(
        config,
        project_root=PROJECT_ROOT,
    )[fold.outer_domain]
    field_bank = load_search_residual_field_bank(
        fold.search_view,
        project_root=PROJECT_ROOT,
    )
    assets = load_registered_pilot_assets(
        data,
        config=exploration,
        project_root=PROJECT_ROOT,
    )
    all_targets = build_outer_fit_residual_target_batch(
        fold.search_view,
        scaffolds[fold.outer_domain],
        field_bank=field_bank,
    )
    positions = {value: index for index, value in enumerate(assets.specimen_ids)}
    assert tuple(
        assets.source_sha256[positions[value]] for value in field_bank.specimen_ids
    ) == field_bank.native_source_sha256
    candidate = config.candidate("RD0")
    checkpoint = freeze_residual_checkpoint(
        build_residual_unet(candidate),
        candidate=candidate,
        config_sha256=config.config_sha256,
        split_sha256=fold.state_sha256,
        training_seed=config.screening_seed,
    )
    sample_calls = []

    def fake_sample(checkpoint_value, stable_conditions, **kwargs):
        sample_calls.append(
            (
                checkpoint_value.scientific_digest,
                tuple(kwargs["specimen_ids"]),
                kwargs["draws"],
            )
        )
        assert np.array_equal(stable_conditions, all_targets.stable_condition)
        return np.broadcast_to(
            all_targets.training_target[:, None],
            (len(all_targets.specimen_ids), 32, 3, 64, 64),
        )

    class FakeEncoder:
        def encode(self, image_grid, *, layer):
            assert layer == pilot.feature_layer
            rows = len(image_grid)
            variants = len(image_grid[0])
            row = np.arange(rows, dtype=np.float64)[:, None, None]
            draw = np.arange(variants, dtype=np.float64)[None, :, None]
            feature = np.arange(64, dtype=np.float64)[None, None, :]
            return np.sin((row + 1.0) * (feature + 1.0) + draw)

    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_search.sample_residual_targets",
        fake_sample,
    )
    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_targets."
        "build_outer_fit_residual_target_batch",
        lambda *_args, **_kwargs: pytest.fail(
            "the precomputed outer-fit target batch must be reused"
        ),
    )

    def fake_variant_batch(source, residuals, **kwargs):
        del source
        requested = kwargs["requested_count"]
        assert len(residuals) == 32
        assert max(float(np.max(np.abs(value))) for value in residuals) == 0.0
        records = tuple(SimpleNamespace(accepted=True) for _ in range(requested))
        return SimpleNamespace(
            encoder_images=tuple(kwargs["native_source"] for _ in range(requested)),
            records=records,
            accepted_count=requested,
            proposal_count=requested,
            fallback_count=0,
            state_sha256=hashlib.sha256(
                f"variant:{requested}".encode("ascii")
            ).hexdigest(),
        )

    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_search.build_variant_batch",
        fake_variant_batch,
    )
    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_search._morphology_distance",
        lambda _record, _candidate: 0.0,
    )
    cell = ResidualSearchCell(
        "A",
        fold.outer_domain,
        fold.query_domain,
        "RD0",
        config.screening_seed,
    )

    bundle = build_residual_feature_bundle(
        cell,
        fold=fold,
        config=config,
        checkpoint=checkpoint,
        scaffold=scaffolds[fold.outer_domain],
        pilot_candidate=pilot,
        field_bank=field_bank,
        target_batch=all_targets,
        assets=assets,
        encoder=FakeEncoder(),
        device="cuda:0",
    )

    maximum_k = max(pilot.K_train, pilot.K_test)
    assert sample_calls == [
        (checkpoint.scientific_digest, fold.search_view.specimen_ids, 32)
    ]
    assert bundle.feature_bundle.train_variant_features.shape[:2] == (
        fold.search_view.specimen_count,
        pilot.K_train,
    )
    assert bundle.feature_bundle.query_variant_features.shape[:2] == (
        fold.search_view.specimen_count,
        pilot.K_test,
    )
    np.testing.assert_array_equal(
        bundle.feature_bundle.accepted_proposals,
        np.full(fold.search_view.specimen_count, maximum_k),
    )
    np.testing.assert_array_equal(
        bundle.feature_bundle.proposed_variants,
        np.full(fold.search_view.specimen_count, maximum_k),
    )
    assert bundle.checkpoint_sha256 == checkpoint.scientific_digest
    assert bundle.scaffold_sha256 == scaffolds[fold.outer_domain].state_sha256
    assert bundle.field_bank_sha256 == field_bank.state_sha256
    assert bundle.target_state_sha256 == all_targets.state_sha256
    assert bundle.asset_state_sha256 == assets.state_sha256


def test_residual_cell_trains_before_sampling_and_query_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    fold = _inner_fold()
    cell = ResidualSearchCell(
        "A",
        fold.outer_domain,
        fold.query_domain,
        "RD0",
        config.screening_seed,
    )
    checkpoint = freeze_residual_checkpoint(
        build_residual_unet(config.candidate("RD0")),
        candidate=config.candidate("RD0"),
        config_sha256=config.config_sha256,
        split_sha256=fold.state_sha256,
        training_seed=config.screening_seed,
    )
    training = ResidualTrainingResult(
        outer_domain=cell.outer_domain,
        query_domain=cell.query_domain,
        candidate_id=cell.candidate_id,
        seed=cell.training_seed,
        epochs=1,
        fit_specimen_ids=(fold.fit_specimen_ids[0],),
        fit_dataset_ids=(fold.fit_dataset_ids[0],),
        target_state_sha256="1" * 64,
        split_sha256=fold.state_sha256,
        epoch_losses=(EpochLossRecord(1, 1.0, 1.0, 0.0, 0.0, 1, 1),),
        checkpoint=checkpoint,
        sample_count=1,
        batch_count=1,
        response_read_count=0,
        test_scale_override=True,
    )
    feature_bundle = D8FeatureBundle(
        candidate_sha256="2" * 64,
        search_view_sha256=fold.search_view.state_sha256,
        specimen_ids=(fold.search_view.specimen_ids[0],),
        train_variant_features=np.zeros((1, 1, 2), dtype=np.float64),
        query_variant_features=np.zeros((1, 1, 2), dtype=np.float64),
        morphology_distances=np.zeros((1, 1), dtype=np.float64),
        accepted_proposals=np.ones(1, dtype=np.int64),
        proposed_variants=np.ones(1, dtype=np.int64),
    )
    bundle = ResidualFeatureBundle(
        cell=cell,
        pilot_candidate_sha256=feature_bundle.candidate_sha256,
        checkpoint_sha256=checkpoint.scientific_digest,
        sampled_target_sha256="3" * 64,
        scaffold_sha256="4" * 64,
        field_bank_sha256="5" * 64,
        target_state_sha256="6" * 64,
        asset_state_sha256="7" * 64,
        feature_bundle=feature_bundle,
        variant_state_sha256=("8" * 64,),
    )
    evaluation = ResidualCellEvaluation(
        cell=cell,
        specimen_ids=(fold.query_specimen_ids[0],),
        targets=np.asarray((0.5,), dtype=np.float64),
        predictions=np.asarray((0.4,), dtype=np.float64),
        accepted_proposals=1,
        proposed_variants=1,
        checkpoint_sha256=checkpoint.scientific_digest,
        prediction_sha256="9" * 64,
    )
    events: list[str] = []
    fit_target = object()
    outer_target = object()

    def fake_train(*args, **kwargs):
        events.append("train")
        assert args == (fold, fit_target)
        assert kwargs["epochs"] == 1
        return training

    def fake_build(*args, **kwargs):
        events.append("sample")
        assert args == (cell,)
        assert kwargs["checkpoint"] is training.checkpoint
        assert kwargs["target_batch"] is outer_target
        return bundle

    def fake_evaluate(*args, **kwargs):
        events.append("score")
        assert args == (cell,)
        assert kwargs["bundle"] is bundle
        return evaluation

    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_search."
        "train_inner_residual_model",
        fake_train,
    )
    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_search."
        "build_residual_feature_bundle",
        fake_build,
    )
    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_search."
        "evaluate_residual_feature_bundle",
        fake_evaluate,
    )

    result = run_residual_search_cell(
        cell,
        fold=fold,
        config=config,
        fit_target_batch=fit_target,
        outer_target_batch=outer_target,
        scaffold=object(),
        pilot_candidate=object(),
        field_bank=object(),
        assets=object(),
        encoder=object(),
        device="cuda",
        test_scale_override=True,
    )

    assert type(result) is ResidualCellRun
    assert result.training.state_sha256 == training.state_sha256
    assert result.feature_bundle_sha256 == bundle.state_sha256
    assert result.evaluation.state_sha256 == evaluation.state_sha256
    assert events == ["train", "sample", "score"]


def _fake_cell_run(cell: ResidualSearchCell, *, fold) -> ResidualCellRun:
    checkpoint_digest = hashlib.sha256(
        f"checkpoint:{cell.state_sha256}".encode("ascii")
    ).hexdigest()
    checkpoint = ResidualCheckpoint(
        candidate_id=cell.candidate_id,
        config_sha256=_config().config_sha256,
        split_sha256=fold.state_sha256,
        training_seed=cell.training_seed,
        model_config={},
        architecture_sha256="a" * 64,
        state_dict_sha256="b" * 64,
        runtime={},
        state_dict={},
        scientific_digest=checkpoint_digest,
    )
    training = ResidualTrainingResult(
        outer_domain=cell.outer_domain,
        query_domain=cell.query_domain,
        candidate_id=cell.candidate_id,
        seed=cell.training_seed,
        epochs=1,
        fit_specimen_ids=(fold.fit_specimen_ids[0],),
        fit_dataset_ids=(fold.fit_dataset_ids[0],),
        target_state_sha256="c" * 64,
        split_sha256=fold.state_sha256,
        epoch_losses=(EpochLossRecord(1, 1.0, 1.0, 0.0, 0.0, 1, 1),),
        checkpoint=checkpoint,
        sample_count=1,
        batch_count=1,
        response_read_count=0,
        test_scale_override=True,
    )
    error = (
        0.01 * int(cell.candidate_id[2:])
        if cell.stage == "A"
        else {"RD0": 0.08, "RD1": 0.10}[cell.candidate_id]
    )
    evaluation = _evaluation(cell, error=error)
    return ResidualCellRun(
        cell=cell,
        training=training,
        feature_bundle_sha256="d" * 64,
        evaluation=evaluation,
    )


def test_outer_search_runs_exact_stages_and_only_retains_registered_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    exploration, data = _data_authorities()
    outer = DOMAIN_ORDER[0]
    search = issue_search_view(data, outer_domain=outer, config=exploration)
    folds = {
        domain: issue_inner_fold(search, query_domain=domain)
        for domain in DOMAIN_ORDER
        if domain != outer
    }
    scaffold = load_pilot_diffusion_scaffolds(
        config,
        project_root=PROJECT_ROOT,
    )[outer]
    pilot_candidate = load_pilot_scaffold_candidates(
        config,
        project_root=PROJECT_ROOT,
    )[outer]
    incumbents = (
        _incumbent(pipeline_id="PILOT", outer=outer, error=0.09),
        _incumbent(pipeline_id="B0", outer=outer, error=0.11),
    )
    target_events: list[tuple[str, str]] = []
    cell_events: list[tuple[str, str, str, int]] = []
    retained: list[tuple[str, bool]] = []
    final_events: list[int] = []
    final_retained: list[int] = []
    outer_target = object()
    fit_targets = {domain: object() for domain in folds}

    def fake_outer_target(search_value, scaffold_value, **kwargs):
        assert search_value is search
        assert scaffold_value is scaffold
        assert kwargs["field_bank"] is field_bank
        target_events.append(("outer", outer))
        return outer_target

    def fake_fit_target(fold_value, scaffold_value, **kwargs):
        assert scaffold_value is scaffold
        assert kwargs["field_bank"] is field_bank
        target_events.append(("fit", fold_value.query_domain))
        return fit_targets[fold_value.query_domain]

    def fake_cell_runner(cell, **kwargs):
        fold = folds[cell.query_domain]
        assert kwargs["fold"] is fold
        assert kwargs["fit_target_batch"] is fit_targets[cell.query_domain]
        assert kwargs["outer_target_batch"] is outer_target
        cell_events.append(
            (
                cell.stage,
                cell.query_domain,
                cell.candidate_id,
                cell.training_seed,
            )
        )
        return _fake_cell_run(cell, fold=fold)

    def record_cell(run, *, retain_checkpoint):
        retained.append((run.cell.stage, retain_checkpoint))

    def fake_final_training(search_value, target_value, **kwargs):
        assert search_value is search
        assert target_value is outer_target
        assert kwargs["candidate"].candidate_id == "RD0"
        seed = kwargs["seed"]
        final_events.append(seed)
        checkpoint = ResidualCheckpoint(
            candidate_id="RD0",
            config_sha256=config.config_sha256,
            split_sha256=search.state_sha256,
            training_seed=seed,
            model_config={},
            architecture_sha256="e" * 64,
            state_dict_sha256="f" * 64,
            runtime={},
            state_dict={},
            scientific_digest=hashlib.sha256(
                f"final:{outer}:{seed}".encode("ascii")
            ).hexdigest(),
        )
        return ResidualFinalTrainingResult(
            outer_domain=outer,
            candidate_id="RD0",
            seed=seed,
            epochs=1,
            fit_specimen_ids=(search.specimen_ids[0],),
            fit_dataset_ids=(search.dataset_ids[0],),
            target_state_sha256="1" * 64,
            split_sha256=search.state_sha256,
            epoch_losses=(EpochLossRecord(1, 1.0, 1.0, 0.0, 0.0, 1, 1),),
            checkpoint=checkpoint,
            sample_count=1,
            batch_count=1,
            response_read_count=0,
            test_scale_override=True,
        )

    def record_final(result):
        final_retained.append(result.seed)

    field_bank = object()
    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_search."
        "build_outer_fit_residual_target_batch",
        fake_outer_target,
    )
    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_search."
        "build_fit_residual_target_batch",
        fake_fit_target,
    )
    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_search."
        "run_residual_search_cell",
        fake_cell_runner,
    )
    monkeypatch.setattr(
        "cmc_bbdm.cpb_diffusion_marginalization.residual_search."
        "train_outer_fit_residual_model",
        fake_final_training,
    )

    result = run_residual_outer_search(
        search,
        folds=folds,
        config=config,
        scaffold=scaffold,
        pilot_candidate=pilot_candidate,
        field_bank=field_bank,
        incumbents=incumbents,
        assets=object(),
        encoder=object(),
        device="cuda",
        cell_recorder=record_cell,
        final_recorder=record_final,
        test_scale_override=True,
    )

    assert type(result) is ResidualOuterSearchRun
    assert result.outer_domain == outer
    assert result.outer_evaluation_count == 0
    assert result.stage_a.finalists == ("RD0", "RD1")
    assert result.selection.selected_pipeline == "RESIDUAL"
    assert len(result.stage_a_run_sha256) == 5 * 8
    assert len(result.stage_b_run_sha256) == 5 * 2 * 3
    assert len(set(result.stage_a_run_sha256)) == 5 * 8
    assert len(set(result.stage_b_run_sha256)) == 5 * 2 * 3
    assert target_events == [("outer", outer)] + [
        ("fit", domain) for domain in DOMAIN_ORDER if domain != outer
    ]
    assert len(cell_events) == 5 * 8 + 5 * 2 * 3
    assert retained == [("A", False)] * (5 * 8) + [("B", True)] * (
        5 * 2 * 3
    )
    assert final_events == list(config.training_seeds)
    assert final_retained == list(config.training_seeds)
    assert result.final_training_sha256 == tuple(
        hashlib.sha256(f"final:{outer}:{seed}".encode("ascii")).hexdigest()
        for seed in config.training_seeds
    )
