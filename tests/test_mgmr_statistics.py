from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cmc_bbdm.mgmr.evaluation import PredictionRecord
from cmc_bbdm.mgmr.statistics import paired_domain_bootstrap, prediction_metrics


def test_prediction_metrics_use_raw_specimens_and_equal_domain_weighting() -> None:
    records = (
        PredictionRecord("M", "a0", "a", 0.0, 0.0, (1,)),
        PredictionRecord("M", "a1", "a", 1.0, 1.2, (1,)),
        PredictionRecord("M", "a2", "a", 2.0, 2.4, (1,)),
        PredictionRecord("M", "b0", "b", 0.0, 1.0, (1,)),
    )

    metric = prediction_metrics(records, domain_order=("a", "b"))

    assert metric.specimen_mae == pytest.approx(0.4)
    assert metric.equal_domain_mae == pytest.approx(0.6)
    assert metric.worst_domain_mae == pytest.approx(1.0)
    assert tuple(row.domain for row in metric.domain_metrics) == ("a", "b")
    assert tuple(row.specimen_count for row in metric.domain_metrics) == (3, 1)


def test_constant_vectors_have_finite_zero_correlations() -> None:
    records = tuple(
        PredictionRecord("M", f"s{i}", "a", 1.0, 2.0, (1,)) for i in range(3)
    )
    metric = prediction_metrics(records, domain_order=("a",))

    assert metric.pearson == 0.0
    assert metric.spearman == 0.0


def test_domain_bootstrap_reuses_one_pcg64_draw_matrix() -> None:
    effects = {
        "x": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        "y": (2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
    }
    result = paired_domain_bootstrap(
        effects,
        domain_order=("a", "b", "c", "d", "e", "f"),
        seed=20260822,
        resamples=100,
        quantiles=(0.025, 0.975),
    )
    draws = np.random.Generator(np.random.PCG64(20260822)).integers(
        0, 6, size=(100, 6), dtype=np.int64
    )
    expected_hash = hashlib.sha256(draws.tobytes(order="C")).hexdigest()

    assert result.draw_sha256 == expected_hash
    assert result.intervals["x"].estimate == pytest.approx(3.5)
    assert result.intervals["y"].estimate == pytest.approx(7.0)
    assert result.intervals["y"].low == pytest.approx(2 * result.intervals["x"].low)
    assert result.intervals["y"].high == pytest.approx(2 * result.intervals["x"].high)
