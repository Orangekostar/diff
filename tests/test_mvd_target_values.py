from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.mvd.evaluation import initial_mechanical_values


class _Predictor:
    outer_domain = "target"
    fit_domains = ("d0", "d1", "d2", "d3", "d4")
    state_sha256 = "a" * 64

    def predict(self, metadata: object, embeddings: object) -> np.ndarray:
        meta = np.asarray(metadata, dtype=np.float64)
        values = np.asarray(embeddings, dtype=np.float64)
        assert meta.shape[0] == values.shape[0]
        return values[:, 0]


def test_target_initial_values_use_one_frozen_outer_predictor() -> None:
    candidates = np.zeros((64, 512), dtype=np.float64)
    candidates[:, 0] = np.linspace(0.0, 1.0, 64)
    result = initial_mechanical_values(
        _Predictor(),
        outer_domain="target",
        metadata=np.zeros((1, 13), dtype=np.float64),
        target=0.75,
        initial_embedding=np.zeros(512, dtype=np.float64),
        candidate_embeddings=candidates,
    )

    assert result.current_prediction == 0.0
    assert result.candidate_predictions.shape == (64,)
    assert result.mechanical_values[0] == 0.0
    assert result.mechanical_values[-1] == pytest.approx(0.5)
    assert not result.mechanical_values.flags.writeable
    assert result.predictor_state_sha256 == "a" * 64


def test_target_initial_values_reject_outer_training_leakage() -> None:
    predictor = _Predictor()
    predictor.fit_domains = (*predictor.fit_domains, "target")
    with pytest.raises(ValueError, match="outer-domain"):
        initial_mechanical_values(
            predictor,
            outer_domain="target",
            metadata=np.zeros((1, 13), dtype=np.float64),
            target=0.75,
            initial_embedding=np.zeros(512, dtype=np.float64),
            candidate_embeddings=np.zeros((64, 512), dtype=np.float64),
        )
