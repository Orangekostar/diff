from __future__ import annotations

from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.measurement_state import initial_state
from cmc_bbdm.mva.oracle_trajectory import run_control_trajectory


def test_trajectory_measurement_count_is_monotonic_and_under_each_cap() -> None:
    grid = build_acquisition_grid(338, 340, initial_budget=0.03125)
    checkpoints = (0.0625, 0.09375, 0.125, 0.1875, 0.25)
    trajectory = run_control_trajectory(
        grid,
        initial_state(grid),
        checkpoints=checkpoints,
        method="uniform",
    )

    counts = [row.measured_count for row in trajectory.snapshots]
    assert counts == sorted(counts)
    assert all(
        row.effective_budget <= row.nominal_checkpoint for row in trajectory.snapshots
    )
    assert all(
        before.levels[index] <= after.levels[index]
        for before, after in zip(
            (initial_state(grid), *(row.state for row in trajectory.snapshots[:-1])),
            (row.state for row in trajectory.snapshots),
            strict=True,
        )
        for index in range(64)
    )
