from __future__ import annotations

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import initial_state
from cmc_bbdm.mva.oracle import choose_greedy_oracle_action


def test_oracle_scorer_receives_only_one_step_candidate_states() -> None:
    grid = build_acquisition_grid(91, 93, initial_budget=0.03125)
    state = initial_state(grid)
    observed: list[tuple[int, tuple[int, ...]]] = []

    def score(current: object, action: object, candidate: object) -> float:
        changed = [
            index
            for index, (before, after) in enumerate(
                zip(current.levels, candidate.levels, strict=True)
            )
            if before != after
        ]
        assert changed == [action.cell_index]
        assert (
            candidate.levels[action.cell_index] == current.levels[action.cell_index] + 1
        )
        observed.append((action.cell_index, candidate.levels))
        return float(-action.cell_index)

    selected = choose_greedy_oracle_action(
        grid, state, checkpoint=0.25, score_candidate=score
    )

    assert selected.cell_index == 0
    assert len(observed) == 64
    assert all(sum(levels) == 1 for _index, levels in observed)
