from __future__ import annotations

import numpy as np
import pytest
import torch

from cmc_bbdm.mva.ranking_policy import (
    RankingExample,
    RankingPolicy,
    equal_hierarchy_weights,
    load_policy_package,
    pairwise_ranking_loss,
    save_policy_package,
    train_ranking_policy,
)


def _examples() -> tuple[RankingExample, ...]:
    rng = np.random.default_rng(17)
    output: list[RankingExample] = []
    for dataset_id, specimen_id, states in (
        ("d1", "s1", 2),
        ("d1", "s2", 1),
        ("d2", "s3", 3),
    ):
        for state in range(states):
            candidates = 3 + state
            output.append(
                RankingExample(
                    dataset_id=dataset_id,
                    specimen_id=specimen_id,
                    global_features=rng.normal(size=579),
                    candidate_features=rng.normal(size=(candidates, 8)),
                    selected_index=state % candidates,
                )
            )
    return tuple(output)


def test_registered_policy_parameter_count_and_candidate_mask() -> None:
    model = RankingPolicy().to(dtype=torch.float64)
    assert sum(parameter.numel() for parameter in model.parameters()) == 41_617
    global_features = torch.zeros((2, 579), dtype=torch.float64)
    candidates = torch.zeros((2, 4, 8), dtype=torch.float64)
    mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]], dtype=torch.bool
    )

    scores = model(global_features, candidates, mask)

    assert scores.shape == (2, 4)
    assert torch.all(torch.isfinite(scores[mask]))
    assert torch.all(torch.isneginf(scores[~mask]))


def test_pairwise_loss_matches_teacher_vs_all_definition() -> None:
    scores = torch.tensor([[2.0, 1.0, -1.0]], dtype=torch.float64)
    mask = torch.ones((1, 3), dtype=torch.bool)
    expected = (
        torch.nn.functional.softplus(torch.tensor(-1.0, dtype=torch.float64))
        + torch.nn.functional.softplus(torch.tensor(-3.0, dtype=torch.float64))
    ) / 2.0

    observed = pairwise_ranking_loss(
        scores,
        selected_indices=torch.tensor([0]),
        candidate_mask=mask,
        state_weights=torch.tensor([1.0], dtype=torch.float64),
    )

    assert observed == pytest.approx(expected)


def test_equal_hierarchy_weights_give_equal_domain_specimen_state_mass() -> None:
    examples = _examples()
    weights = equal_hierarchy_weights(
        tuple(example.dataset_id for example in examples),
        tuple(example.specimen_id for example in examples),
    )

    assert weights.sum() == pytest.approx(1.0)
    assert weights[:3].sum() == pytest.approx(0.5)
    assert weights[3:].sum() == pytest.approx(0.5)
    assert weights[:2].sum() == pytest.approx(0.25)
    assert weights[2] == pytest.approx(0.25)


def test_training_and_model_package_are_deterministic(tmp_path) -> None:
    first = train_ranking_policy(
        _examples(), seed=20260823, epochs=2, batch_states=2
    )
    second = train_ranking_policy(
        _examples(), seed=20260823, epochs=2, batch_states=2
    )
    path = save_policy_package(tmp_path / "policy.npz", first)
    loaded = load_policy_package(path)

    assert first.state_sha256 == second.state_sha256 == loaded.state_sha256
    assert first.loss_trace == second.loss_trace == loaded.loss_trace
    example = _examples()[0]
    with torch.no_grad():
        first_score = first.score(example)
        loaded_score = loaded.score(example)
    assert np.array_equal(first_score, loaded_score)
