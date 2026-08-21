from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.cpb_diffusion_marginalization.authority import (
    issue_inner_fold,
    issue_search_view,
)
from cmc_bbdm.cpb_diffusion_marginalization.config import load_d8_config
from cmc_bbdm.cpb_diffusion_marginalization.features import (
    D8FrozenEncoder,
    aggregate_features,
    aggregate_predictions,
    create_d8_frozen_encoder,
    variant_training_weights,
)
from cmc_bbdm.cpb_diffusion_marginalization.regression import (
    CandidateSpec,
    fit_candidate,
    fit_marginalized_candidate,
)
from cmc_bbdm.cpb_v3.config import load_config as load_v3_config
from cmc_bbdm.cpb_v3.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D8_CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_exploration.yaml"
P1_CONFIG = PROJECT_ROOT / "paper_v3/configs/p1_full_field_oracle.yaml"


@pytest.fixture(scope="module")
def inner_fold():
    config = load_d8_config(D8_CONFIG, project_root=PROJECT_ROOT)
    p1 = load_v3_config(P1_CONFIG, project_root=PROJECT_ROOT)
    data = load_data(p1, PROJECT_ROOT)
    search = issue_search_view(data, outer_domain="74t7kcdgkr", config=config)
    return issue_inner_fold(search, query_domain="cgtnjyggtm")


@pytest.mark.slow
def test_raw_global_feature_matches_frozen_p1_encoder() -> None:
    encoder = create_d8_frozen_encoder(project_root=PROJECT_ROOT, device="cuda:0")
    assert isinstance(encoder, D8FrozenEncoder)
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    expected = encoder.base_encoder.encode((image,))
    observed = encoder.encode(((image,),), layer="global")
    np.testing.assert_array_equal(observed[0, 0], expected[0])


@pytest.mark.parametrize(
    ("method", "expected_width"),
    (("mean", 6), ("median", 6), ("trimmed", 6), ("mean_variance", 12)),
)
def test_feature_aggregation_preserves_specimen_axis(
    method: str, expected_width: int
) -> None:
    generator = np.random.Generator(np.random.PCG64(7))
    values = generator.normal(size=(5, 16, 6)).astype(np.float32)
    result = aggregate_features(values, method=method)
    assert result.shape == (5, expected_width)
    assert result.flags.writeable is False
    assert np.isfinite(result).all()


def test_prediction_aggregation_and_specimen_weights_preserve_physical_units() -> None:
    predictions = np.asarray(((0.2, 0.4, 0.6), (0.1, 0.3, 0.8)), dtype=np.float64)
    distances = np.asarray(((0.0, 1.0, 2.0), (2.0, 1.0, 0.0)), dtype=np.float64)
    weighted = aggregate_predictions(
        predictions,
        method="morphology_weighted",
        morphology_distances=distances,
        beta=1.0,
    )
    assert weighted.shape == (2,)
    assert 0.2 <= weighted[0] <= 0.4
    assert 0.3 <= weighted[1] <= 0.8

    ids = ("a", "a", "a", "b", "b")
    weights = variant_training_weights(ids)
    assert np.sum(weights[:3], dtype=np.float64) == 1.0
    assert np.sum(weights[3:], dtype=np.float64) == 1.0


@pytest.mark.parametrize(
    ("name", "parameters"),
    (
        ("ridge", {"alpha": 10.0}),
        ("elastic_net", {"alpha": 0.01, "l1_ratio": 0.5}),
        ("pls", {"n_components": 2}),
        ("huber", {"alpha": 0.0001, "epsilon": 1.35}),
        ("kernel_ridge", {"alpha": 1.0, "gamma": 0.1}),
        ("svr", {"C": 1.0, "epsilon": 0.05, "gamma": 0.1}),
        (
            "hist_gradient_boosting",
            {"l2_regularization": 1.0, "learning_rate": 0.05, "max_leaf_nodes": 15},
        ),
        ("shallow_mlp", {"alpha": 0.01, "hidden_layer_size": 16}),
    ),
)
def test_registered_candidate_fits_only_inner_rows_and_predicts_query_once(
    inner_fold, name: str, parameters: dict[str, object]
) -> None:
    generator = np.random.Generator(np.random.PCG64(19))
    count = inner_fold.search_view.specimen_count
    features = generator.normal(size=(count, 8)).astype(np.float64)
    targets = np.asarray(inner_fold.search_view.data_view.cai_ratio, dtype=np.float64)
    spec = CandidateSpec(
        pca_dimension=4,
        regressor=name,
        parameters=parameters,
        seed=20260820,
    )
    result = fit_candidate(
        spec,
        inner_fold=inner_fold,
        specimen_ids=inner_fold.search_view.specimen_ids,
        features=features,
        targets=targets,
    )
    assert set(result.fit_specimen_ids).isdisjoint(result.query_specimen_ids)
    assert result.fit_specimen_ids == inner_fold.fit_specimen_ids
    assert result.query_specimen_ids == inner_fold.query_specimen_ids
    assert result.predictions.shape == (len(inner_fold.query_indices),)
    assert result.targets.shape == result.predictions.shape
    assert np.isfinite(result.predictions).all()
    assert result.predictions.flags.writeable is False


def test_marginalized_candidate_executes_feature_and_prediction_stages(
    inner_fold,
) -> None:
    rows = inner_fold.search_view.specimen_count
    generator = np.random.default_rng(20260820)
    base = generator.normal(size=(rows, 8))
    train = np.stack((base - 0.1, base + 0.1), axis=1)
    query = np.stack(
        (base - 0.3, base - 0.1, base + 0.1, base + 0.3), axis=1
    )
    targets = np.asarray(inner_fold.search_view.data_view.cai_ratio, dtype=np.float64)
    spec = CandidateSpec(
        pca_dimension=4,
        regressor="ridge",
        parameters={"alpha": 1.0},
        seed=20260820,
    )
    feature = fit_marginalized_candidate(
        spec,
        inner_fold=inner_fold,
        specimen_ids=inner_fold.search_view.specimen_ids,
        train_variant_features=train,
        query_variant_features=query,
        targets=targets,
        marginalization_stage="feature",
        feature_aggregation="mean_variance",
        prediction_aggregation="mean",
        morphology_distances=None,
        morphology_beta=None,
        consistency="feature_variance",
        consistency_weight=0.2,
    )
    prediction = fit_marginalized_candidate(
        spec,
        inner_fold=inner_fold,
        specimen_ids=inner_fold.search_view.specimen_ids,
        train_variant_features=train,
        query_variant_features=query,
        targets=targets,
        marginalization_stage="prediction",
        feature_aggregation="mean",
        prediction_aggregation="median",
        morphology_distances=np.zeros((rows, 4), dtype=np.float64),
        morphology_beta=None,
        consistency="prediction_variance",
        consistency_weight=0.2,
    )
    assert feature.query_specimen_ids == inner_fold.query_specimen_ids
    assert prediction.query_specimen_ids == inner_fold.query_specimen_ids
    assert feature.predictions.shape == prediction.predictions.shape == (
        len(inner_fold.query_indices),
    )
    assert feature.fit_state_sha256 != prediction.fit_state_sha256
    assert feature.predictions.flags.writeable is False
    assert prediction.predictions.flags.writeable is False


def test_prediction_stage_rejects_feature_only_or_misaligned_inputs(
    inner_fold,
) -> None:
    rows = inner_fold.search_view.specimen_count
    features = np.ones((rows, 2, 5), dtype=np.float64)
    targets = np.asarray(inner_fold.search_view.data_view.cai_ratio, dtype=np.float64)
    spec = CandidateSpec(
        pca_dimension=4,
        regressor="ridge",
        parameters={"alpha": 1.0},
        seed=20260820,
    )
    with pytest.raises(ValueError, match="feature aggregation|aligned"):
        fit_marginalized_candidate(
            spec,
            inner_fold=inner_fold,
            specimen_ids=inner_fold.search_view.specimen_ids,
            train_variant_features=features,
            query_variant_features=features,
            targets=targets,
            marginalization_stage="prediction",
            feature_aggregation="mean_variance",
            prediction_aggregation="mean",
            morphology_distances=None,
            morphology_beta=None,
            consistency="none",
            consistency_weight=0.0,
        )
