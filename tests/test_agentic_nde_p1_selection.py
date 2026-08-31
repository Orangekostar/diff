from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cmc_bbdm.agentic_nde import visual_observability
from cmc_bbdm.agentic_nde.visual_observability import (
    P1MechanicalLabels,
    assemble_p1_outer_examples,
    fit_outer_visual_models,
    select_fusion_lambda,
    select_source_candidate,
)


def test_source_candidate_selection_uses_frozen_lexicographic_order() -> None:
    domains = ("a", "b", "c", "d", "e")
    rows = []
    for domain in domains:
        rows.extend(
            (
                {
                    "candidate_id": "z-large",
                    "validation_domain": domain,
                    "ndcg_10": 0.8,
                    "next_action_regret": 0.2,
                    "parameter_count": 20,
                },
                {
                    "candidate_id": "b-small",
                    "validation_domain": domain,
                    "ndcg_10": 0.8,
                    "next_action_regret": 0.2,
                    "parameter_count": 10,
                },
                {
                    "candidate_id": "a-small",
                    "validation_domain": domain,
                    "ndcg_10": 0.8,
                    "next_action_regret": 0.2,
                    "parameter_count": 10,
                },
                {
                    "candidate_id": "higher-regret",
                    "validation_domain": domain,
                    "ndcg_10": 0.8,
                    "next_action_regret": 0.21,
                    "parameter_count": 1,
                },
            )
        )
    selection = select_source_candidate(pl.DataFrame(rows), domain_order=domains)
    assert selection.candidate_id == "a-small"
    assert selection.ndcg_10 == 0.8
    assert selection.next_action_regret == 0.2
    assert selection.parameter_count == 10
    assert selection.aggregates.height == 4
    assert selection.aggregates.filter(pl.col("selected")).height == 1
    assert len(selection.state_sha256) == 64


def test_fusion_lambda_is_selected_from_source_domains_only() -> None:
    domains = ("a", "b", "c", "d", "e")
    truth = np.tile(np.arange(64, dtype=np.float64), (10, 1))
    dataset_ids = tuple(domain for domain in domains for _ in range(2))
    old = -truth
    visual = truth.copy()
    selection = select_fusion_lambda(
        mechanical_values=truth,
        dataset_ids=dataset_ids,
        old_scores=old,
        visual_scores=visual,
        values=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    assert selection.value == 0.75
    assert selection.audit.height == 5 * len(domains)
    assert set(selection.audit["validation_domain"]) == set(domains)
    assert "outer_domain" not in selection.audit.columns


def test_outer_examples_keep_target_labels_out_of_inference() -> None:
    domains = ("a", "b", "c", "d", "e", "f")
    specimen_ids = tuple(f"s-{domain}" for domain in domains)
    count = len(domains)
    current = np.arange(count, dtype=np.float64)
    source_labels = P1MechanicalLabels(
        outer_domain="f",
        role="source_train",
        specimen_ids=specimen_ids[:-1],
        dataset_ids=domains[:-1],
        mechanical_values=np.arange(5 * 64, dtype=np.float64).reshape(5, 64),
        target_source_sha256="a" * 64,
        state_sha256="b" * 64,
    )
    result = assemble_p1_outer_examples(
        outer_domain="f",
        specimen_ids=specimen_ids,
        dataset_ids=domains,
        initial_embeddings=np.zeros((count, 512)),
        current_predictions=current,
        candidate_features=np.zeros((count, 64, 8)),
        global_embeddings=np.zeros((count, 512), dtype=np.float32),
        local_embeddings=np.zeros((count, 64, 512), dtype=np.float32),
        source_labels=source_labels,
        feature_control="correct",
    )
    assert result.source.role == "source_train"
    assert result.source.mechanical_values is not None
    assert result.source.dataset_ids == domains[:-1]
    assert result.inference.role == "outer_inference"
    assert result.inference.mechanical_values is None
    assert result.inference.dataset_ids == ("f",)
    assert result.inference.current_predictions.tolist() == [5.0]
    assert result.source_indices == (0, 1, 2, 3, 4)
    assert result.target_indices == (5,)
    assert len(result.state_sha256) == 64


class _FakeScorer:
    def __init__(self, representation: str, config_id: str, parameter_count: int) -> None:
        self.representation = representation
        self.config_id = config_id
        self.parameter_count = parameter_count
        self.state_sha256 = (representation + config_id).encode().hex()[:64].ljust(64, "0")

    def predict(self, examples: object) -> np.ndarray:
        if self.representation in {"LOCAL", "LOCAL_GLOBAL"}:
            return np.asarray(examples.local_embeddings[:, :, 0], dtype=np.float64)
        if self.representation == "GLOBAL":
            return np.tile(
                np.asarray(examples.global_embeddings[:, 0], dtype=np.float64)[:, None],
                (1, 64),
            )
        return -np.asarray(examples.candidate_features[:, :, 0], dtype=np.float64)


def test_outer_model_selection_and_controls_are_source_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domains = ("a", "b", "c", "d", "e", "f")
    specimen_ids = tuple(f"s-{domain}" for domain in domains)
    count = len(domains)
    truth = np.tile(np.arange(64, dtype=np.float64), (count, 1))
    local = np.zeros((count, 64, 512), dtype=np.float32)
    local[:, :, 0] = truth
    global_values = np.zeros((count, 512), dtype=np.float32)
    global_values[:, 0] = np.arange(count, dtype=np.float32)
    candidates = np.zeros((count, 64, 8), dtype=np.float64)
    candidates[:, :, 0] = truth
    labels = P1MechanicalLabels(
        outer_domain="f",
        role="source_train",
        specimen_ids=specimen_ids[:-1],
        dataset_ids=domains[:-1],
        mechanical_values=truth[:-1],
        target_source_sha256="a" * 64,
        state_sha256="b" * 64,
    )
    examples = assemble_p1_outer_examples(
        outer_domain="f",
        specimen_ids=specimen_ids,
        dataset_ids=domains,
        initial_embeddings=np.zeros((count, 512)),
        current_predictions=np.zeros(count),
        candidate_features=candidates,
        global_embeddings=global_values,
        local_embeddings=local,
        source_labels=labels,
        feature_control="correct",
    )

    def fake_ridge_family(
        _examples: object, *, representation: str, alphas: tuple[float, ...]
    ) -> tuple[_FakeScorer, ...]:
        return tuple(
            _FakeScorer(
                representation,
                f"ridge_alpha_{alpha:g}",
                {"OLD": 522, "GLOBAL": 1034, "LOCAL": 1034, "LOCAL_GLOBAL": 1546}[
                    representation
                ],
            )
            for alpha in alphas
        )

    def fake_mlp(
        _examples: object,
        *,
        representation: str,
        seed: int,
        epochs: int,
        device: str,
    ) -> _FakeScorer:
        assert (seed, epochs, device) == (20260831, 50, "cuda:0")
        return _FakeScorer(representation, "mlp_smooth_l1_32_16", 50_000)

    monkeypatch.setattr(visual_observability, "fit_ridge_family", fake_ridge_family)
    monkeypatch.setattr(visual_observability, "fit_mlp_scorer", fake_mlp)
    fitted = fit_outer_visual_models(
        correct=examples,
        shuffled=examples,
        wrong_orientation=examples,
        spatial_derangement=examples,
        c0_source_scores=-truth[:-1],
        ridge_alphas=(0.1, 1.0, 10.0, 100.0),
        fusion_values=(0.0, 0.25, 0.5, 0.75, 1.0),
        model_seed=20260831,
        epochs=50,
        device="cuda:0",
    )
    assert fitted.correct_representation == "LOCAL"
    assert fitted.correct_config_id == "ridge_alpha_0.1"
    assert fitted.correct_lambda == 0.75
    assert set(fitted.models) == {
        "old_refit_diagnostic",
        "proposed",
        "c2_global_context",
        "c3_shuffled_surface",
        "c4_wrong_orientation",
        "c5_spatial_derangement",
        "c3_shuffled_global",
    }
    assert set(fitted.model_feature_controls) == set(fitted.models)
    assert fitted.selection_audit.filter(
        pl.col("stage") == "HEAD_INNER"
    ).height == 4 * 5 * 5
    assert "outer_evaluation" not in set(fitted.selection_audit["role"])
    assert len(fitted.selection_state_sha256) == 64
