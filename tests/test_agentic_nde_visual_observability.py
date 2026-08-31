from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cmc_bbdm.agentic_nde.p1 import load_p1_config
from cmc_bbdm.agentic_nde.surface_cells import load_surface_cell_authority
from cmc_bbdm.agentic_nde.visual_observability import (
    VisualExamples,
    attach_target_labels,
    center_prior_scores,
    decide_p1,
    evaluate_action_scores,
    fit_mlp_scorer,
    fit_ridge_family,
    freeze_outer_scores,
    fuse_rank_scores,
    load_frozen_c0_scores,
    load_p1_deployable_authority,
    load_p1_source_labels,
    load_p1_target_labels,
    paired_specimen_bootstrap,
    replace_surface_features,
    stable_rank_percentiles,
    subset_visual_examples,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/agentic_nde_p1_visual_observability.yaml"
P0R = ROOT / "results/agentic_task_driven_nde/p0r_author_registration"


def _examples(
    *,
    role: str = "source_train",
    outer_domain: str = "outer",
    include_labels: bool = True,
) -> VisualExamples:
    domains = ("a", "a", "b", "b", "c", "c", "d", "d", "e", "e")
    if role.startswith("outer_"):
        domains = (outer_domain, outer_domain)
    count = len(domains)
    cells = np.arange(64, dtype=np.float64)
    initial = np.zeros((count, 512), dtype=np.float64)
    current = np.linspace(-0.2, 0.2, count, dtype=np.float64)
    candidates = np.zeros((count, 64, 8), dtype=np.float64)
    global_values = np.zeros((count, 512), dtype=np.float32)
    local = np.zeros((count, 64, 512), dtype=np.float32)
    truth = np.empty((count, 64), dtype=np.float64)
    for index in range(count):
        candidates[index, :, 0] = cells / 63.0
        candidates[index, :, 1] = (cells % 8) / 7.0
        global_values[index, 0] = np.float32(index / max(1, count - 1))
        local[index, :, 0] = np.asarray(
            np.sin((cells + 1.0) * (index + 1.0) / 11.0), dtype=np.float32
        )
        local[index, :, 1] = np.asarray(cells / 63.0, dtype=np.float32)
        truth[index] = (
            1.8 * local[index, :, 0]
            + 0.7 * local[index, :, 1]
            + 0.2 * candidates[index, :, 1]
            + 0.1 * current[index]
        )
    return VisualExamples.create(
        outer_domain=outer_domain,
        role=role,
        specimen_ids=tuple(f"s{index:02d}" for index in range(count)),
        dataset_ids=domains,
        initial_embeddings=initial,
        current_predictions=current,
        candidate_features=candidates,
        global_embeddings=global_values,
        local_embeddings=local,
        mechanical_values=truth if include_labels else None,
        feature_control="correct",
    )


def test_visual_examples_enforce_outer_inference_label_barrier() -> None:
    inference = _examples(role="outer_inference", include_labels=False)
    assert inference.mechanical_values is None
    assert inference.specimen_count == 2
    with pytest.raises(ValueError):
        fit_ridge_family(inference, representation="LOCAL", alphas=(1.0,))
    with pytest.raises(ValueError):
        _examples(role="outer_inference", include_labels=True)


def test_source_subset_preserves_outer_exclusion_and_role() -> None:
    source = _examples()
    validation = subset_visual_examples(
        source, included_domains=("c",), role="source_validation"
    )
    assert validation.dataset_ids == ("c", "c")
    assert validation.role == "source_validation"
    assert validation.outer_domain not in validation.dataset_ids
    assert validation.mechanical_values is not None


def test_ridge_family_recovers_registered_local_signal() -> None:
    source = _examples()
    models = fit_ridge_family(
        source, representation="LOCAL", alphas=(0.1, 1.0, 10.0)
    )
    assert tuple(model.alpha for model in models) == (0.1, 1.0, 10.0)
    for model in models:
        scores = model.predict(source)
        assert scores.shape == (10, 64)
        assert model.fit_domains == ("a", "b", "c", "d", "e")
        assert "outer" not in model.fit_domains
        assert model.parameter_count == 1034
        assert len(model.state_sha256) == 64
    assert np.mean(np.abs(models[0].predict(source) - source.mechanical_values)) < 0.03


def test_small_mlp_is_source_only_under_parameter_cap_and_deterministic() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = _examples()
    left = fit_mlp_scorer(
        source,
        representation="LOCAL",
        seed=20260831,
        epochs=50,
        device="cuda:0",
    )
    right = fit_mlp_scorer(
        source,
        representation="LOCAL",
        seed=20260831,
        epochs=50,
        device="cuda:0",
    )
    assert left.parameter_count < 100000
    assert left.state_sha256 == right.state_sha256
    assert np.array_equal(left.predict(source), right.predict(source))
    metrics = [
        evaluate_action_scores(truth, scores)
        for truth, scores in zip(
            source.mechanical_values, left.predict(source), strict=True
        )
    ]
    assert np.mean([metric.ndcg_10 for metric in metrics]) > 0.75


def test_surface_feature_replacement_keeps_old_state_and_changes_control() -> None:
    source = _examples()
    donors = np.asarray([1, 0, 3, 2, 5, 4, 7, 6, 9, 8], dtype=np.int64)
    control = replace_surface_features(
        source,
        global_embeddings=source.global_embeddings[donors],
        local_embeddings=source.local_embeddings[donors],
        feature_control="shuffled",
    )
    assert np.array_equal(control.initial_embeddings, source.initial_embeddings)
    assert np.array_equal(control.current_predictions, source.current_predictions)
    assert np.array_equal(control.candidate_features, source.candidate_features)
    assert not np.array_equal(control.local_embeddings, source.local_embeddings)
    assert control.feature_control == "shuffled"


def test_stable_rank_percentile_and_fusion_tie_rule_are_exact() -> None:
    old = np.arange(64, dtype=np.float64)
    visual = np.zeros(64, dtype=np.float64)
    old_rank = stable_rank_percentiles(old)
    visual_rank = stable_rank_percentiles(visual)
    assert old_rank[63] == 1.0
    assert old_rank[0] == 0.0
    assert visual_rank[0] == 1.0
    assert visual_rank[63] == 0.0
    assert np.array_equal(fuse_rank_scores(old, visual, 0.0), old_rank)
    assert np.array_equal(fuse_rank_scores(old, visual, 1.0), visual_rank)
    assert np.array_equal(
        fuse_rank_scores(old, visual, 0.25), 0.75 * old_rank + 0.25 * visual_rank
    )


def test_action_metrics_and_center_prior_are_specimen_level() -> None:
    truth = np.arange(64, dtype=np.float64)
    perfect = evaluate_action_scores(truth, truth)
    reverse = evaluate_action_scores(truth, -truth)
    assert perfect.next_action_regret == 0.0
    assert perfect.one_step_cai_utility == 63.0
    assert perfect.ndcg_10 == pytest.approx(1.0)
    assert perfect.recall_5 == 1.0
    assert perfect.top_10_percent_overlap == 1.0
    assert perfect.top_1_oracle_match == 1.0
    assert reverse.next_action_regret == 63.0
    center = center_prior_scores()
    assert center.shape == (64,)
    assert int(np.argmax(center)) == 27


def test_formal_a2_inputs_and_labels_obey_score_freeze_barrier() -> None:
    config = load_p1_config(CONFIG, project_root=ROOT)
    surface = load_surface_cell_authority(
        P0R / "surface_manifest.csv",
        P0R / "registration.csv",
        P0R / "grid_mapping_qc.csv",
    )
    deployable = load_p1_deployable_authority(config, surface)
    assert deployable.specimen_ids == surface.specimen_ids
    assert deployable.dataset_ids == surface.dataset_ids
    assert deployable.current_predictions.shape == (276,)
    assert len(deployable.state_sha256) == 64
    outer = "74t7kcdgkr"
    source = load_p1_source_labels(config, deployable, outer_domain=outer)
    assert source.role == "source_train"
    assert source.mechanical_values.shape == (231, 64)
    assert outer not in source.dataset_ids

    target_indices = np.flatnonzero(np.asarray(deployable.dataset_ids) == outer)
    inference = VisualExamples.create(
        outer_domain=outer,
        role="outer_inference",
        specimen_ids=tuple(deployable.specimen_ids[index] for index in target_indices),
        dataset_ids=tuple(deployable.dataset_ids[index] for index in target_indices),
        initial_embeddings=np.zeros((len(target_indices), 512)),
        current_predictions=deployable.current_predictions[target_indices],
        candidate_features=np.zeros((len(target_indices), 64, 8)),
        global_embeddings=np.zeros((len(target_indices), 512), dtype=np.float32),
        local_embeddings=np.zeros(
            (len(target_indices), 64, 512), dtype=np.float32
        ),
        mechanical_values=None,
        feature_control="correct",
    )
    with pytest.raises(TypeError):
        load_p1_target_labels(config, deployable, outer_domain=outer, frozen_scores=None)
    frozen = freeze_outer_scores(
        inference,
        scores={"proposed": np.zeros((len(target_indices), 64))},
        model_state_sha256={"proposed": "a" * 64},
        selection_state_sha256="b" * 64,
    )
    target = load_p1_target_labels(
        config, deployable, outer_domain=outer, frozen_scores=frozen
    )
    evaluation = attach_target_labels(inference, target, frozen_scores=frozen)
    assert target.role == "outer_evaluation"
    assert target.mechanical_values.shape == (45, 64)
    assert evaluation.role == "outer_evaluation"
    assert evaluation.mechanical_values is not None


def test_frozen_c0_is_exact_o2_and_ignores_historical_teacher_column() -> None:
    config = load_p1_config(CONFIG, project_root=ROOT)
    surface = load_surface_cell_authority(
        P0R / "surface_manifest.csv",
        P0R / "registration.csv",
        P0R / "grid_mapping_qc.csv",
    )
    deployable = load_p1_deployable_authority(config, surface)
    c0 = load_frozen_c0_scores(config, deployable)
    assert c0.method == "o2_global_candidate"
    assert c0.scores.shape == (276, 64)
    assert c0.specimen_ids == deployable.specimen_ids
    assert len(c0.state_sha256) == 64


DOMAINS = ("a", "b", "c", "d", "e", "f")


def _auebc_table(values: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "outer_domain": domain,
                "specimen_id": f"{domain}-{index}",
                "method": method,
                "cai_auebc": value + 0.001 * index,
            }
            for domain in DOMAINS
            for index in range(8)
            for method, value in values.items()
        ]
    )


def _ranking_table(*, proposed_better: bool) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "outer_domain": domain,
                "specimen_id": f"{domain}-{index}",
                "method": method,
                "ndcg_10": ndcg,
                "next_action_regret": regret,
            }
            for domain in DOMAINS
            for index in range(8)
            for method, ndcg, regret in (
                ("c0_mvd_m1_o2", 0.5, 0.5),
                (
                    "proposed",
                    0.7 if proposed_better else 0.4,
                    0.3 if proposed_better else 0.6,
                ),
            )
        ]
    )


def test_paired_bootstrap_resamples_specimens_within_equal_weight_domains() -> None:
    table = _auebc_table({"control": 1.0, "proposed": 0.75})
    effect = paired_specimen_bootstrap(
        table,
        domain_order=DOMAINS,
        control="control",
        proposed="proposed",
        value_column="cai_auebc",
        seed=20260831,
        resamples=1000,
        effect_id="control_minus_proposed",
    )
    assert effect.point_estimate == pytest.approx(0.25)
    assert effect.lower == pytest.approx(0.25)
    assert effect.upper == pytest.approx(0.25)
    assert effect.improved_domains == 6
    assert effect.specimen_count == 48


def test_p1_spatial_gate_requires_all_registered_controls() -> None:
    table = _auebc_table(
        {
            "c0_mvd_m1_o2": 1.0,
            "proposed": 0.6,
            "c2_global_context": 0.82,
            "c3_shuffled_surface": 0.9,
            "c4_wrong_orientation": 0.88,
            "c5_spatial_derangement": 0.86,
            "c3_shuffled_global": 0.94,
            "mechanical_oracle_diagnostic": 0.2,
        }
    )
    decision = decide_p1(
        table,
        _ranking_table(proposed_better=True),
        domain_order=DOMAINS,
        bootstrap_seed=20260831,
        bootstrap_resamples=1000,
    )
    assert decision.status == "P1_SPATIAL_VISUAL_OBSERVABILITY_GO"
    assert decision.go
    assert decision.authorized_route == "SPATIAL"
    assert decision.oracle_gap_closure == pytest.approx(0.5)
    assert all(decision.spatial_conditions.values())


def test_p1_global_gate_does_not_authorize_spatial_claim() -> None:
    table = _auebc_table(
        {
            "c0_mvd_m1_o2": 1.0,
            "proposed": 0.72,
            "c2_global_context": 0.78,
            "c3_shuffled_surface": 0.91,
            "c4_wrong_orientation": 0.68,
            "c5_spatial_derangement": 0.85,
            "c3_shuffled_global": 0.93,
            "mechanical_oracle_diagnostic": 0.2,
        }
    )
    decision = decide_p1(
        table,
        _ranking_table(proposed_better=True),
        domain_order=DOMAINS,
        bootstrap_seed=20260831,
        bootstrap_resamples=1000,
    )
    assert decision.status == "P1_GLOBAL_VISUAL_CONTEXT_GO"
    assert decision.go
    assert decision.authorized_route == "GLOBAL_CONTEXT"
    assert not all(decision.spatial_conditions.values())
    assert all(decision.global_conditions.values())


def test_p1_descriptive_and_no_go_statuses_do_not_authorize_downstream() -> None:
    table = _auebc_table(
        {
            "c0_mvd_m1_o2": 1.0,
            "proposed": 1.01,
            "c2_global_context": 1.02,
            "c3_shuffled_surface": 1.0,
            "c4_wrong_orientation": 1.0,
            "c5_spatial_derangement": 1.0,
            "c3_shuffled_global": 1.0,
            "mechanical_oracle_diagnostic": 0.2,
        }
    )
    descriptive = decide_p1(
        table,
        _ranking_table(proposed_better=True),
        domain_order=DOMAINS,
        bootstrap_seed=20260831,
        bootstrap_resamples=1000,
    )
    no_go = decide_p1(
        table,
        _ranking_table(proposed_better=False),
        domain_order=DOMAINS,
        bootstrap_seed=20260831,
        bootstrap_resamples=1000,
    )
    assert descriptive.status == "P1_DESCRIPTIVE_SPATIAL_SIGNAL_ONLY"
    assert no_go.status == "P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO"
    assert not descriptive.go and descriptive.authorized_route is None
    assert not no_go.go and no_go.authorized_route is None
