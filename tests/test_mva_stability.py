from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mva.cai_evaluator import fit_sensitivity_cai_predictor
from cmc_bbdm.mva.stability import _stability_embedding_set, ranking_similarity


class _PositionEncoder:
    def __init__(self) -> None:
        self.batch_lengths: list[int] = []

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        self.batch_lengths.append(len(images))
        output = np.zeros((len(images), 512), dtype=np.float64)
        output[:, 0] = np.arange(len(images), dtype=np.float64)
        return output


def test_ranking_similarity_is_exact_for_identical_scores() -> None:
    scores = {index: float(10 - index) for index in range(10)}

    result = ranking_similarity(scores, scores)

    assert result.top1_agreement is True
    assert result.top10_overlap == 1.0
    assert result.spearman == pytest.approx(1.0)
    assert result.rbo_p0_9 == pytest.approx(1.0)


def test_ranking_similarity_detects_reversed_order() -> None:
    first = {index: float(10 - index) for index in range(10)}
    second = {index: float(index) for index in range(10)}

    result = ranking_similarity(first, second)

    assert result.top1_agreement is False
    assert result.top10_overlap == 0.0
    assert result.spearman == pytest.approx(-1.0)
    assert 0.0 <= result.rbo_p0_9 < 1.0


def test_sensitivity_ridge_uses_registered_alphas_and_fit_roster() -> None:
    generator = np.random.default_rng(20260823)
    metadata = generator.normal(size=(30, 3))
    embeddings = generator.normal(size=(30, 8))
    targets = 0.4 + metadata[:, 0] - 0.5 * embeddings[:, 0]
    specimen_ids = tuple(f"s{index}" for index in range(30))
    dataset_ids = tuple(f"d{index // 10}" for index in range(30))
    fit_indices = np.arange(20, dtype=np.int64)

    ridge1 = fit_sensitivity_cai_predictor(
        method="ridge1",
        outer_domain="d2",
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata=metadata,
        embeddings=embeddings,
        dimension=4,
        fit_indices=fit_indices,
        ridge_alpha=1.0,
    )
    ridge100 = fit_sensitivity_cai_predictor(
        method="ridge100",
        outer_domain="d2",
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata=metadata,
        embeddings=embeddings,
        dimension=4,
        fit_indices=fit_indices,
        ridge_alpha=100.0,
    )

    assert ridge1.ridge.alpha == 1.0
    assert ridge100.ridge.alpha == 100.0
    assert ridge1.fit_specimen_ids == specimen_ids[:20]
    assert ridge1.fit_domains == ("d0", "d1")
    first = ridge1.predict(metadata[20:], embeddings[20:])
    second = ridge100.predict(metadata[20:], embeddings[20:])
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))
    assert not np.allclose(first, second)

    with pytest.raises(ValueError, match="registered sensitivity"):
        fit_sensitivity_cai_predictor(
            method="ridge10",
            outer_domain="d2",
            specimen_ids=specimen_ids,
            dataset_ids=dataset_ids,
            targets=targets,
            metadata=metadata,
            embeddings=embeddings,
            dimension=4,
            fit_indices=fit_indices,
            ridge_alpha=10.0,
        )


def test_stability_candidates_keep_the_formal_oracle_batch_positions() -> None:
    encoder = _PositionEncoder()
    current = np.zeros((8, 8, 3), dtype=np.uint8)
    candidates = [np.full_like(current, fill_value=value) for value in (1, 2, 3)]
    issued_current = np.full(512, 7.0, dtype=np.float64)

    embeddings = _stability_embedding_set(
        encoder,
        current,
        candidates,
        issued_current=issued_current,
    )

    assert encoder.batch_lengths == [3]
    assert np.array_equal(embeddings[0], issued_current)
    assert embeddings[1:, 0].tolist() == [0.0, 1.0, 2.0]
