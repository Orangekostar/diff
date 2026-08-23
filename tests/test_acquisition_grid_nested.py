from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.cpb_sparse_scan.sampling import sampling_coordinates
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid

SHAPES = ((674, 675), (338, 352), (338, 340))
BUDGETS = (0.015625, 0.03125, 0.0625)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("budget", BUDGETS)
def test_grid_is_nested_and_bound_to_p5(shape: tuple[int, int], budget: float) -> None:
    height, width = shape
    grid = build_acquisition_grid(height, width, initial_budget=budget)
    p5 = sampling_coordinates(height, width, density=0.25)

    assert grid.native_shape == shape
    assert grid.cell_shape == (8, 8)
    assert len(grid.cells) == 64
    assert grid.level1_rows == p5.rows
    assert grid.level1_columns == p5.columns
    assert len(grid.row_boundaries) == 9
    assert len(grid.column_boundaries) == 9
    assert set(grid.row_boundaries) <= set(grid.level0_rows)
    assert set(grid.column_boundaries) <= set(grid.level0_columns)
    assert set(grid.level0_rows) <= set(grid.level1_rows)
    assert set(grid.level0_columns) <= set(grid.level1_columns)
    assert grid.level0_rows[0] == 0
    assert grid.level0_rows[-1] == height - 1
    assert grid.level0_columns[0] == 0
    assert grid.level0_columns[-1] == width - 1

    for index, cell in enumerate(grid.cells):
        assert cell.index == index
        assert cell.row == index // 8
        assert cell.column == index % 8
        for level in (0, 1):
            assert set(cell.rows[level]) <= set(cell.rows[level + 1])
            assert set(cell.columns[level]) <= set(cell.columns[level + 1])
        assert cell.rows[2] == tuple(
            range(grid.row_boundaries[cell.row], grid.row_boundaries[cell.row + 1] + 1)
        )
        assert cell.columns[2] == tuple(
            range(
                grid.column_boundaries[cell.column],
                grid.column_boundaries[cell.column + 1] + 1,
            )
        )
        assert len(cell.rows[0]) >= 2
        assert len(cell.columns[0]) >= 2


@pytest.mark.parametrize("shape", SHAPES)
def test_initial_candidate_grids_are_deterministic_and_nested(
    shape: tuple[int, int],
) -> None:
    grids = [
        build_acquisition_grid(*shape, initial_budget=budget) for budget in BUDGETS
    ]

    assert set(grids[0].level0_rows) <= set(grids[1].level0_rows)
    assert set(grids[1].level0_rows) <= set(grids[2].level0_rows)
    assert set(grids[0].level0_columns) <= set(grids[1].level0_columns)
    assert set(grids[1].level0_columns) <= set(grids[2].level0_columns)
    for first, second in zip(grids, grids, strict=True):
        assert first == second
    assert all(np.isfinite(grid.actual_initial_budget) for grid in grids)
    assert [grid.actual_initial_budget for grid in grids] == sorted(
        grid.actual_initial_budget for grid in grids
    )
