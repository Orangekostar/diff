from __future__ import annotations

from types import MappingProxyType

import numpy as np

from cmc_bbdm.agentic_nde.p1_execution import (
    P1OuterData,
    evaluate_p1_outer_score_metrics,
    freeze_p1_outer_predictions,
)
from cmc_bbdm.agentic_nde.visual_observability import (
    FrozenOuterScores,
    OuterVisualModelFit,
    P1MechanicalLabels,
    assemble_p1_outer_examples,
)


class _FixedModel:
    representation = "LOCAL"
    config_id = "ridge_alpha_0.1"
    parameter_count = 1034

    def __init__(self, value: float, token: str) -> None:
        self.value = value
        self.state_sha256 = token * 64

    def predict(self, examples: object) -> np.ndarray:
        assert examples.mechanical_values is None
        return np.tile(
            np.arange(64, dtype=np.float64)[None, :] + self.value,
            (examples.specimen_count, 1),
        )


def _examples(feature_control: str):
    domains = ("a", "b", "c", "d", "e", "f")
    specimens = tuple(f"s-{domain}" for domain in domains)
    labels = P1MechanicalLabels(
        outer_domain="f",
        role="source_train",
        specimen_ids=specimens[:-1],
        dataset_ids=domains[:-1],
        mechanical_values=np.tile(np.arange(64, dtype=np.float64), (5, 1)),
        target_source_sha256="a" * 64,
        state_sha256="b" * 64,
    )
    return assemble_p1_outer_examples(
        outer_domain="f",
        specimen_ids=specimens,
        dataset_ids=domains,
        initial_embeddings=np.zeros((6, 512)),
        current_predictions=np.zeros(6),
        candidate_features=np.zeros((6, 64, 8)),
        global_embeddings=np.zeros((6, 512), dtype=np.float32),
        local_embeddings=np.zeros((6, 64, 512), dtype=np.float32),
        source_labels=labels,
        feature_control=feature_control,
    )


def test_target_predictions_freeze_before_mechanical_oracle_is_available() -> None:
    correct = _examples("correct")
    shuffled = _examples("shuffled_surface")
    wrong = _examples("wrong_orientation")
    deranged = _examples("spatial_derangement")
    methods = {
        "old_refit_diagnostic": _FixedModel(0.0, "1"),
        "proposed": _FixedModel(1.0, "2"),
        "c2_global_context": _FixedModel(2.0, "3"),
        "c3_shuffled_surface": _FixedModel(3.0, "4"),
        "c4_wrong_orientation": _FixedModel(4.0, "5"),
        "c5_spatial_derangement": _FixedModel(5.0, "6"),
        "c3_shuffled_global": _FixedModel(6.0, "7"),
    }
    controls = {
        "old_refit_diagnostic": "correct",
        "proposed": "correct",
        "c2_global_context": "correct",
        "c3_shuffled_surface": "shuffled_surface",
        "c4_wrong_orientation": "wrong_orientation",
        "c5_spatial_derangement": "spatial_derangement",
        "c3_shuffled_global": "shuffled_surface",
    }
    fitted = OuterVisualModelFit(
        outer_domain="f",
        correct_representation="LOCAL",
        correct_config_id="ridge_alpha_0.1",
        global_config_id="ridge_alpha_0.1",
        old_config_id="ridge_alpha_0.1",
        correct_lambda=0.75,
        global_lambda=0.5,
        models=MappingProxyType(methods),
        model_feature_controls=MappingProxyType(controls),
        selection_audit=None,
        selection_state_sha256="c" * 64,
    )
    data = P1OuterData(
        outer_domain="f",
        initial_budget=0.015625,
        correct=correct,
        shuffled=shuffled,
        wrong_orientation=wrong,
        spatial_derangement=deranged,
        c0_source_scores=-np.tile(np.arange(64, dtype=np.float64), (5, 1)),
        c0_target_scores=-np.arange(64, dtype=np.float64)[None, :],
        target_candidate_costs=np.ones((1, 64), dtype=np.int64),
        target_native_shapes=((64, 64),),
        target_grid_state_sha256=("d" * 64,),
        candidate_bank_state_sha256="e" * 64,
        observed_feature_state_sha256="f" * 64,
        surface_feature_state_sha256="0" * 64,
        state_sha256="9" * 64,
    )
    frozen = freeze_p1_outer_predictions(data, fitted)
    assert set(frozen.methods) == {
        "c0_mvd_m1_o2",
        "c1_center_prior",
        "old_refit_diagnostic",
        "proposed",
        "c2_global_context",
        "c3_shuffled_surface",
        "c4_wrong_orientation",
        "c5_spatial_derangement",
        "c3_shuffled_global",
    }
    assert "mechanical_oracle_diagnostic" not in frozen.methods
    assert all(value.shape == (1, 64) for value in frozen.scores.values())
    assert frozen.selection_state_sha256 == fitted.selection_state_sha256
    assert len(frozen.state_sha256) == 64


def test_score_metrics_add_oracle_only_after_outer_evaluation() -> None:
    outer = _examples("correct")
    inference = outer.inference
    frozen = FrozenOuterScores(
        outer_domain="f",
        specimen_ids=inference.specimen_ids,
        dataset_ids=inference.dataset_ids,
        methods=("c0_mvd_m1_o2", "proposed"),
        scores=MappingProxyType(
            {
                "c0_mvd_m1_o2": -np.arange(64, dtype=np.float64)[None, :],
                "proposed": np.arange(64, dtype=np.float64)[None, :],
            }
        ),
        model_state_sha256=MappingProxyType(
            {"c0_mvd_m1_o2": "1" * 64, "proposed": "2" * 64}
        ),
        selection_state_sha256="3" * 64,
        inference_state_sha256=inference.state_sha256,
        state_sha256="4" * 64,
    )
    evaluation = inference.create(
        outer_domain="f",
        role="outer_evaluation",
        specimen_ids=inference.specimen_ids,
        dataset_ids=inference.dataset_ids,
        initial_embeddings=inference.initial_embeddings,
        current_predictions=inference.current_predictions,
        candidate_features=inference.candidate_features,
        global_embeddings=inference.global_embeddings,
        local_embeddings=inference.local_embeddings,
        mechanical_values=np.arange(64, dtype=np.float64)[None, :],
        feature_control=inference.feature_control,
    )
    result = evaluate_p1_outer_score_metrics(
        evaluation,
        frozen,
        candidate_costs=np.arange(1, 65, dtype=np.int64)[None, :],
    )
    assert set(result.score_matrices) == {
        "c0_mvd_m1_o2",
        "proposed",
        "mechanical_oracle_diagnostic",
    }
    assert result.per_state_scores.height == 3 * 64
    assert result.per_specimen_metrics.height == 3
    oracle = result.per_state_scores.filter(
        result.per_state_scores["method"] == "mechanical_oracle_diagnostic"
    )
    assert set(oracle["score_role"]) == {"evaluation_only_diagnostic"}
    assert result.per_specimen_metrics.filter(
        result.per_specimen_metrics["method"] == "proposed"
    )["next_action_regret"].item() == 0.0
