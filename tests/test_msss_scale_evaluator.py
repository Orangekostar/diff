from __future__ import annotations

import numpy as np

from cmc_bbdm.msss.scale_evaluator import evaluate_axis
from cmc_bbdm.msss.scale_features import ScaleCondition, ScaleFeatureBank


def _inputs(group_sizes: tuple[int, int, int] = (6, 6, 6)) -> tuple[ScaleFeatureBank, np.ndarray, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(29))
    groups = tuple(
        group
        for group, size in zip(("d1", "d2", "d3"), group_sizes, strict=True)
        for _ in range(size)
    )
    conditions = tuple(
        ScaleCondition(
            condition_id=f"gaussian:sigma={value}",
            axis="gaussian",
            value=value,
            coarse_rank=rank,
            primary_eligible=True,
            is_full_identity=rank == 0,
        )
        for rank, value in enumerate((0.0, 1.0, 2.0, 4.0))
    )
    latent = np.linspace(-1.0, 1.0, len(groups))
    targets = 0.7 + 0.1 * latent
    metadata = np.zeros((len(groups), 13), dtype=np.float64)
    metadata[:, 0] = latent
    base = generator.normal(scale=0.05, size=(len(groups), 512))
    base[:, 0] = latent
    bank = ScaleFeatureBank.issue(
        conditions=conditions,
        specimen_ids=tuple(f"s{index}" for index in range(len(groups))),
        dataset_ids=groups,
        features={
            condition.condition_id: base + condition.coarse_rank * 0.0001
            for condition in conditions
        },
        transform_state_sha256={item.condition_id: "1" * 64 for item in conditions},
        encoder_provenance={"encoder": "synthetic", "frozen": True},
    )
    return bank, targets, metadata


def test_axis_evaluation_returns_complete_fixed_and_selected_oof_rosters() -> None:
    bank, targets, metadata = _inputs()
    result = evaluate_axis(
        bank,
        targets=targets,
        metadata13=metadata,
        axis="gaussian",
        pca_dimensions=(2, 3),
        primary_margin=0.05,
    )

    assert len(result.candidate_predictions) == len(bank.conditions) * len(targets)
    assert len(result.selected_predictions) == len(targets)
    assert len(result.candidate_selections) == len(bank.conditions) * 3
    assert len(result.scale_selections) == 3
    assert {
        (row.condition_id, row.specimen_id) for row in result.candidate_predictions
    } == {
        (condition.condition_id, specimen)
        for condition in bank.conditions
        for specimen in bank.specimen_ids
    }
    assert {row.specimen_id for row in result.selected_predictions} == set(
        bank.specimen_ids
    )
    assert all(row.outer_group == row.dataset_id for row in result.selected_predictions)
    assert all(np.isfinite(row.prediction) for row in result.selected_predictions)


def test_axis_evaluation_reports_sensitivity_but_excludes_it_from_selection() -> None:
    bank, targets, metadata = _inputs()
    sensitivity = ScaleCondition(
        condition_id="gaussian:sensitivity",
        axis="gaussian",
        value=8.0,
        coarse_rank=4,
        primary_eligible=False,
        is_full_identity=False,
    )
    extended = ScaleFeatureBank.issue(
        conditions=bank.conditions + (sensitivity,),
        specimen_ids=bank.specimen_ids,
        dataset_ids=bank.dataset_ids,
        features={**bank.features, sensitivity.condition_id: bank.features[bank.conditions[0].condition_id]},
        transform_state_sha256={**bank.transform_state_sha256, sensitivity.condition_id: "2" * 64},
        encoder_provenance=bank.encoder_provenance,
    )
    result = evaluate_axis(
        extended,
        targets=targets,
        metadata13=metadata,
        axis="gaussian",
        pca_dimensions=(2, 3),
        primary_margin=0.05,
    )

    assert sensitivity.condition_id in {
        row.condition_id for row in result.candidate_predictions
    }
    assert sensitivity.condition_id not in {
        row.selected_condition_id for row in result.scale_selections
    }


def test_outer_predictions_use_outer_query_for_unequal_domain_sizes() -> None:
    bank, targets, metadata = _inputs((5, 7, 9))
    result = evaluate_axis(
        bank,
        targets=targets,
        metadata13=metadata,
        axis="gaussian",
        pca_dimensions=(2, 3),
        primary_margin=0.05,
    )

    assert len(result.candidate_predictions) == len(bank.conditions) * len(targets)
    assert all(np.isfinite(row.prediction) for row in result.candidate_predictions)
