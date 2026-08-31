from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cmc_bbdm.inspection_agent.cai_assessor import (
    CAIAssessorError,
    StateCAIAssessor,
    StateFeatureRow,
    fit_state_cai_assessor,
    state_scalars,
)
from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.state import InspectionCellAction
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid


def _rows() -> tuple[StateFeatureRow, ...]:
    generator = np.random.Generator(np.random.PCG64(20260831))
    rows = []
    for domain_index, domain in enumerate(("source-a", "source-b")):
        for specimen_index in range(3):
            specimen = f"{domain}-s{specimen_index}"
            target = 0.2 * domain_index + 0.05 * specimen_index
            for state_index in range(19):
                embedding = generator.normal(size=512)
                embedding[0] += target * 20.0
                effective = state_index / 72.0
                rows.append(
                    StateFeatureRow(
                        sample_id=f"{specimen}|state-{state_index:02d}",
                        specimen_id=specimen,
                        dataset_id=domain,
                        policy="ZERO_ANCHOR" if state_index == 0 else "FIXED",
                        observation_sha256=(f"{len(rows):064x}")[-64:],
                        embedding=embedding,
                        effective_budget=effective,
                        observed_cell_fraction=min(1.0, effective * 4.0),
                        mean_observed_level=0.0 if state_index < 10 else 1.0,
                        true_cai=target,
                    )
                )
    return tuple(rows)


def test_metadata_free_assessor_fits_only_source_domains_with_three_scalars() -> None:
    rows = _rows()
    assessor = fit_state_cai_assessor(
        rows,
        outer_domain="target",
        pca_dimension=32,
        ridge_alpha=10.0,
    )
    assert isinstance(assessor, StateCAIAssessor)
    assert assessor.outer_domain == "target"
    assert assessor.fit_domains == ("source-a", "source-b")
    assert assessor.state_scalar_count == 3
    assert len(assessor.fit_sample_ids) == len(rows)
    assert len(set(assessor.fit_sample_ids)) == len(rows)
    matrix = np.asarray([row.embedding for row in rows[:7]])
    scalars = np.asarray([row.scalars for row in rows[:7]])
    predictions = assessor.predict(matrix, scalars)
    assert predictions.shape == (7,)
    assert np.all(np.isfinite(predictions))
    assert not hasattr(assessor, "metadata13")
    assert not hasattr(assessor, "profile_stats21")


def test_assessor_rejects_outer_target_rows_and_unequal_specimen_weights() -> None:
    rows = _rows()
    with pytest.raises(CAIAssessorError, match="outer target"):
        fit_state_cai_assessor(
            (*rows, replace(rows[0], sample_id="target-row", dataset_id="target")),
            outer_domain="target",
            pca_dimension=32,
            ridge_alpha=10.0,
        )
    with pytest.raises(CAIAssessorError, match="equal state count"):
        fit_state_cai_assessor(
            rows[:-1],
            outer_domain="target",
            pca_dimension=32,
            ridge_alpha=10.0,
        )


def test_state_scalars_are_observable_and_zero_state_is_explicit() -> None:
    image = np.zeros((41, 43, 3), dtype=np.uint8)
    authority = MAVISAuthority.from_arrays(
        specimen_ids=("sample",),
        dataset_ids=("domain",),
        images=(image,),
        targets=np.asarray([7.0]),
        metadata13=np.full((1, 13), 999.0),
        profile_stats21=np.full((1, 21), -999.0),
    )
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    surface = np.zeros((9, 9, 3), dtype=np.uint8)
    world = CausalInspectionWorld(
        authority,
        specimen_id="sample",
        task=InspectionTask.CAI,
        surface_rgb=surface,
        surface_sha256="d" * 64,
        grid=grid,
        endpoint_budget=0.25,
    )
    zero = world.reset()
    np.testing.assert_array_equal(state_scalars(zero), (0.0, 0.0, 0.0))
    observed = world.step(zero, InspectionCellAction(0, -1, 0))
    scalars = state_scalars(observed)
    assert scalars[0] == observed.effective_budget
    assert scalars[1] == pytest.approx(1 / 64)
    assert scalars[2] == 0.0
