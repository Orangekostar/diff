from __future__ import annotations

import polars as pl
import pytest

from cmc_bbdm.mavis.task_specificity import (
    TaskSpecificityError,
    align_task_specificity_states,
)


def _a2() -> pl.DataFrame:
    rows = []
    for specimen in ("s0", "s1"):
        for method in ("reconstruction_oracle", "mechanical_oracle"):
            for checkpoint, measured in ((0.1, 10), (0.2, 20)):
                rows.append(
                    {
                        "specimen_id": specimen,
                        "dataset_id": "d0",
                        "method": method,
                        "seed": None,
                        "nominal_checkpoint": checkpoint,
                        "measured_count": measured,
                        "native_count": 100,
                        "effective_budget": measured / 100,
                        "target": 0.5,
                        "p_a_prediction": 0.45,
                        "p_a_absolute_error": 0.05,
                        "normalized_rgb_mse": 0.01,
                        "p_a_predictor_state_sha256": "a" * 64,
                    }
                )
    return pl.DataFrame(rows)


def _a4() -> pl.DataFrame:
    rows = []
    for specimen in ("s0", "s1"):
        for method in ("global_reconstruction_mask", "global_mechanical_mask"):
            for checkpoint, measured in ((0.1, 10), (0.2, 20)):
                rows.append(
                    {
                        "specimen_id": specimen,
                        "dataset_id": "d0",
                        "outer_domain": "d0",
                        "method": method,
                        "nominal_checkpoint": checkpoint,
                        "measured_count": measured,
                        "native_count": 100,
                        "effective_budget": measured / 100,
                        "target": 0.5,
                        "p_a_prediction": 0.45,
                        "p_a_absolute_error": 0.05,
                        "normalized_rgb_mse": 0.01,
                        "p_a_predictor_state_sha256": "a" * 64,
                    }
                )
    return pl.DataFrame(rows)


def test_task_specificity_uses_frozen_reconstruction_metric() -> None:
    aligned = align_task_specificity_states(
        _a2(),
        _a4(),
        domain_order=("d0",),
        checkpoints=(0.1, 0.2),
        reconstruction_metric="normalized_rgb_mse",
    )

    assert aligned.get_column("reconstruction_error").to_list() == [0.01] * 16
    with pytest.raises(TaskSpecificityError, match="frozen reconstruction metric"):
        align_task_specificity_states(
            _a2(),
            _a4(),
            domain_order=("d0",),
            checkpoints=(0.1, 0.2),
            reconstruction_metric="ssim",
        )


def test_task_specificity_same_cost_same_cohort() -> None:
    aligned = align_task_specificity_states(
        _a2(),
        _a4(),
        domain_order=("d0",),
        checkpoints=(0.1, 0.2),
        reconstruction_metric="normalized_rgb_mse",
    )

    assert aligned.group_by("specimen_id", "nominal_checkpoint").len().get_column(
        "len"
    ).unique().to_list() == [4]
    assert aligned.get_column("native_count").unique().to_list() == [100]

    incomplete = _a4().filter(pl.col("specimen_id") != "s1")
    with pytest.raises(TaskSpecificityError, match="cohort"):
        align_task_specificity_states(
            _a2(),
            incomplete,
            domain_order=("d0",),
            checkpoints=(0.1, 0.2),
            reconstruction_metric="normalized_rgb_mse",
        )
