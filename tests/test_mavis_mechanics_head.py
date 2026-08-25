from __future__ import annotations

import numpy as np
import torch

from cmc_bbdm.mavis.mechanics_head import MRISMechanicsModel, fit_fold_normalizer
from cmc_bbdm.mavis.state_encoder import MRISStateEncoder


def _arrays() -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    generator = np.random.Generator(np.random.PCG64(20260825))
    domains = ("d0", "d1", "d2", "d3", "d4", "d5")
    specimen_ids = tuple(f"{domain}-{index}" for domain in domains for index in range(3))
    dataset_ids = tuple(domain for domain in domains for _ in range(3))
    contexts = generator.normal(size=(18, 34))
    targets = generator.normal(size=18)
    return contexts, targets, specimen_ids, dataset_ids


def test_mavis_target_domain_not_used_for_normalization() -> None:
    contexts, targets, specimen_ids, dataset_ids = _arrays()
    changed_contexts = contexts.copy()
    changed_targets = targets.copy()
    outer = np.asarray(dataset_ids, dtype=object) == "d0"
    changed_contexts[outer] += 100000.0
    changed_targets[outer] -= 100000.0

    first = fit_fold_normalizer(
        contexts=contexts,
        targets=targets,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        outer_domain="d0",
    )
    second = fit_fold_normalizer(
        contexts=changed_contexts,
        targets=changed_targets,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        outer_domain="d0",
    )

    assert first.state_sha256 == second.state_sha256
    assert set(first.fit_domains) == {"d1", "d2", "d3", "d4", "d5"}
    assert all(not specimen_id.startswith("d0-") for specimen_id in first.fit_specimen_ids)
    np.testing.assert_array_equal(first.context_mean, second.context_mean)
    np.testing.assert_array_equal(first.context_scale, second.context_scale)
    assert first.target_mean == second.target_mean
    assert first.target_scale == second.target_scale


def test_mavis_inner_validation_domain_not_used_for_normalization() -> None:
    contexts, targets, specimen_ids, dataset_ids = _arrays()
    changed_contexts = contexts.copy()
    changed_targets = targets.copy()
    excluded = np.isin(np.asarray(dataset_ids, dtype=object), ["d0", "d1"])
    changed_contexts[excluded] += 100000.0
    changed_targets[excluded] -= 100000.0

    first = fit_fold_normalizer(
        contexts=contexts,
        targets=targets,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        outer_domain="d0",
        additional_excluded_domains=("d1",),
    )
    second = fit_fold_normalizer(
        contexts=changed_contexts,
        targets=changed_targets,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        outer_domain="d0",
        additional_excluded_domains=("d1",),
    )

    assert first.state_sha256 == second.state_sha256
    assert set(first.fit_domains) == {"d2", "d3", "d4", "d5"}


def test_mavis_mechanics_head_supports_batched_training() -> None:
    torch.manual_seed(20260825)
    encoder = MRISStateEncoder(
        context_dimension=34,
        hidden_dimension=64,
        output_dimension=64,
    )
    model = MRISMechanicsModel(encoder)
    contexts = torch.zeros((3, 34), dtype=torch.float32)
    tokens = torch.zeros((3, 64, 6), dtype=torch.float32)
    masks = torch.zeros((3, 64), dtype=torch.bool)
    costs = torch.zeros((3, 3), dtype=torch.float32)

    mris, prediction = model.forward_batch(contexts, tokens, masks, costs)

    assert mris.shape == (3, 64)
    assert prediction.shape == (3,)
