from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from cmc_bbdm.msss.protocol import load_protocol
from cmc_bbdm.msss.s1 import summarize_s1
from cmc_bbdm.msss.scale_evaluator import (
    AxisEvaluation,
    CandidatePrediction,
    CandidateSelection,
    ScaleSelection,
    SelectedPrediction,
)
from cmc_bbdm.msss.scale_features import ScaleCondition, ScaleFeatureBank
from cmc_bbdm.msss.spatial_specificity import (
    SpatialPrediction,
    SpatialSpecificityEvaluation,
    specificity_gate,
)

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("74t7kcdgkr", "cgtnjyggtm", "w68dtmpfyf", "xcmzfsbd9t", "yfxyg8jm46", "ykhs7s2dck")


def _conditions(axis: str) -> tuple[ScaleCondition, ...]:
    values = {
        "sampling": (1.0, 0.5, 0.25, 0.125),
        "gaussian": (0.0, 1.0, 2.0, 4.0),
        "wavelet": (0.0, 1.0, 2.0, 3.0),
    }[axis]
    return tuple(
        ScaleCondition(
            condition_id=f"{axis}:{value}",
            axis=axis,
            value=value,
            coarse_rank=rank,
            primary_eligible=True,
            is_full_identity=rank == 0,
            wavelet="db2" if axis == "wavelet" else None,
            level=rank if axis == "wavelet" else None,
            mode="low_only" if axis == "wavelet" else None,
        )
        for rank, value in enumerate(values)
    )


def _axis(axis: str, conditions: tuple[ScaleCondition, ...]) -> AxisEvaluation:
    errors = (0.10, 0.102, 0.104, 0.12)
    predictions = []
    candidate_selections = []
    scale_selections = []
    selected_predictions = []
    for group_index, group in enumerate(GROUPS):
        target = 0.7 + group_index * 0.01
        for condition, error in zip(conditions, errors, strict=True):
            predictions.append(
                CandidatePrediction(
                    condition_id=condition.condition_id,
                    specimen_id=f"s{group_index}",
                    dataset_id=group,
                    outer_group=group,
                    target=target,
                    prediction=target + error,
                    absolute_error=error,
                    selected_pca_dimension=8,
                    fit_state_sha256="1" * 64,
                )
            )
            candidate_selections.append(
                CandidateSelection(
                    outer_group=group,
                    condition_id=condition.condition_id,
                    selected_pca_dimension=8,
                    source_equal_group_mae=error,
                    dimension_scores=((8, error),),
                )
            )
        sufficient = tuple(item.condition_id for item in conditions[:3])
        scale_selections.append(
            ScaleSelection(
                outer_group=group,
                selected_condition_id=conditions[2].condition_id,
                full_condition_id=conditions[0].condition_id,
                over_coarse_condition_id=conditions[3].condition_id,
                boundary_confirmed=True,
                sufficient_sets=((0.025, sufficient[:2]), (0.05, sufficient), (0.075, sufficient)),
                candidate_scores=tuple(
                    (condition.condition_id, error)
                    for condition, error in zip(conditions, errors, strict=True)
                ),
            )
        )
        selected_predictions.append(
            SelectedPrediction(
                axis=axis,
                selected_condition_id=conditions[2].condition_id,
                specimen_id=f"s{group_index}",
                dataset_id=group,
                outer_group=group,
                target=target,
                prediction=target + errors[2],
                absolute_error=errors[2],
                selected_pca_dimension=8,
                fit_state_sha256="1" * 64,
            )
        )
    return AxisEvaluation(
        axis=axis,
        group_order=GROUPS,
        inner_scores=(),
        candidate_selections=tuple(candidate_selections),
        candidate_predictions=tuple(predictions),
        scale_selections=tuple(scale_selections),
        selected_predictions=tuple(selected_predictions),
        state_sha256="2" * 64,
    )


def _specificity(axis: str, selected_condition: str, effect: float = 0.02) -> SpatialSpecificityEvaluation:
    predictions = []
    for group_index, group in enumerate(GROUPS):
        target = 0.7 + group_index * 0.01
        for seed in (20260831, 20260901, 20260902):
            predictions.append(
                SpatialPrediction(
                    axis=axis,
                    base_condition_id=selected_condition,
                    seed=seed,
                    specimen_id=f"s{group_index}",
                    dataset_id=group,
                    target=target,
                    regular_prediction=target + 0.104,
                    shuffled_prediction=target + 0.104 + effect,
                    regular_absolute_error=0.104,
                    shuffled_absolute_error=0.104 + effect,
                    selected_pca_dimension=8,
                )
            )
    regular = (0.104,) * 6
    shuffled = (0.104 + effect,) * 6
    return SpatialSpecificityEvaluation(
        axis=axis,
        predictions=tuple(predictions),
        regular_domain_mae=regular,
        shuffled_domain_mae=shuffled,
        result=specificity_gate((effect,) * 6),
        state_sha256="3" * 64,
    )


def _bank() -> tuple[ScaleFeatureBank, dict[str, tuple[ScaleCondition, ...]]]:
    by_axis = {axis: _conditions(axis) for axis in ("sampling", "gaussian", "wavelet")}
    registry = tuple(item for axis in by_axis.values() for item in axis)
    bank = ScaleFeatureBank.issue(
        conditions=registry,
        specimen_ids=tuple(f"s{index}" for index in range(6)),
        dataset_ids=GROUPS,
        features={item.condition_id: np.zeros((6, 512)) for item in registry},
        transform_state_sha256={item.condition_id: "0" * 64 for item in registry},
        encoder_provenance={"encoder": "synthetic", "frozen": True},
    )
    return bank, by_axis


def synthetic_s1_run():
    protocol = load_protocol(ROOT / "paper_v3/configs/msss.yaml", project_root=ROOT)
    bank, conditions = _bank()
    evaluations = tuple(_axis(axis, conditions[axis]) for axis in conditions)
    specificity = tuple(
        _specificity(axis, conditions[axis][2].condition_id) for axis in conditions
    )
    run = summarize_s1(
        protocol,
        bank=bank,
        evaluations=evaluations,
        specificity=specificity,
        bootstrap_resamples=200,
    )
    return protocol, bank, run


def test_s1_summary_issues_strong_go_only_when_all_axes_pass() -> None:
    protocol = load_protocol(ROOT / "paper_v3/configs/msss.yaml", project_root=ROOT)
    bank, conditions = _bank()
    evaluations = tuple(_axis(axis, conditions[axis]) for axis in conditions)
    specificity = tuple(
        _specificity(axis, conditions[axis][2].condition_id) for axis in conditions
    )
    result = summarize_s1(
        protocol,
        bank=bank,
        evaluations=evaluations,
        specificity=specificity,
        bootstrap_resamples=200,
    )

    assert result.gate.status == "STRONG_GO"
    assert len(result.curves) == 12
    assert len(result.domain_metrics) == 72
    assert len(result.axis_summaries) == 3
    assert all(item.gate.status == "PASS" for item in result.axis_summaries)
    assert all(item.global_noninferiority.plateau for item in result.axis_summaries)
    assert all(item.global_noninferiority.boundary_confirmed for item in result.axis_summaries)


def test_s1_summary_downgrades_one_failed_axis_to_go() -> None:
    protocol = load_protocol(ROOT / "paper_v3/configs/msss.yaml", project_root=ROOT)
    bank, conditions = _bank()
    evaluations = tuple(_axis(axis, conditions[axis]) for axis in conditions)
    specificity = [
        _specificity(axis, conditions[axis][2].condition_id) for axis in conditions
    ]
    specificity[-1] = replace(
        specificity[-1],
        result=specificity_gate((-0.01,) * 6),
        shuffled_domain_mae=(0.094,) * 6,
    )
    result = summarize_s1(
        protocol,
        bank=bank,
        evaluations=evaluations,
        specificity=tuple(specificity),
        bootstrap_resamples=100,
    )

    assert result.gate.status == "GO"
    assert tuple(item.gate.status for item in result.axis_summaries) == (
        "PASS",
        "PASS",
        "FAIL",
    )
