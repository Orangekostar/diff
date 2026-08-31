from __future__ import annotations

import pytest

from cmc_bbdm.agentic_nde.grid import Grid8x8, validate_action_id


@pytest.mark.parametrize("cell_id", [0, 1, 31, 63])
def test_legal_action_ids_are_accepted(cell_id: int) -> None:
    assert validate_action_id(cell_id) == cell_id


@pytest.mark.parametrize("cell_id", [-1, 64, 100])
def test_illegal_action_ids_are_rejected(cell_id: int) -> None:
    with pytest.raises(ValueError, match="0..63"):
        validate_action_id(cell_id)


def test_grid_render_is_deterministic() -> None:
    grid = Grid8x8(width=75.0, height=75.0)
    assert grid.render_records() == Grid8x8(width=75.0, height=75.0).render_records()
