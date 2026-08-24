"""Per-specimen rank and exact-cost set metrics for MVD observability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.mva.measurement_state import initial_state

from .one_shot_oracle import plan_frozen_ranking, score_initial_ranking


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    spearman: float
    ndcg_5: float
    ndcg_10: float
    recall_5: float
    recall_10: float
    regret_1: float
    checkpoints: tuple[float, ...]
    budgeted_regret: tuple[float, ...]
    selected_value: tuple[float, ...]
    oracle_value: tuple[float, ...]
    value_capture: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class ExactCostContext:
    grid_state_sha256: str
    native_count: int
    initial_measured_count: int
    initial_locations: np.ndarray
    action_locations: tuple[np.ndarray, ...]


def _locations(
    rows: tuple[int, ...], columns: tuple[int, ...], width: int
) -> np.ndarray:
    values = np.asarray(
        [row * width + column for row in rows for column in columns],
        dtype=np.int64,
    )
    output = np.frombuffer(values.tobytes(order="C"), dtype=np.int64)
    output.setflags(write=False)
    return output


def build_exact_cost_context(grid: AcquisitionGrid) -> ExactCostContext:
    """Bind S0 and all 64 level-1 native-location sets to one grid."""

    if type(grid) is not AcquisitionGrid:
        raise ValueError("exact-cost grid changed")
    height, width = grid.native_shape
    initial = _locations(grid.level0_rows, grid.level0_columns, width)
    actions = tuple(
        _locations(cell.rows[1], cell.columns[1], width) for cell in grid.cells
    )
    if (
        len(actions) != 64
        or initial.size != len(grid.level0_rows) * len(grid.level0_columns)
        or any(np.any(value < 0) or np.any(value >= height * width) for value in actions)
    ):
        raise ValueError("exact-cost location roster changed")
    return ExactCostContext(
        grid_state_sha256=grid.state_sha256,
        native_count=height * width,
        initial_measured_count=int(initial.size),
        initial_locations=initial,
        action_locations=actions,
    )


def _order(scores: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(range(64), key=lambda cell: (-scores[cell], cell)))


def _ndcg(truth: np.ndarray, predicted_order: tuple[int, ...], k: int) -> float:
    relevance = truth - float(np.min(truth))
    weights = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    ideal = _order(truth)[:k]
    denominator = float(np.sum(relevance[list(ideal)] * weights))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0
    return float(np.sum(relevance[list(predicted_order[:k])] * weights) / denominator)


def evaluate_ranking(
    mechanical_values: object,
    predicted_scores: object,
    *,
    grid: AcquisitionGrid,
    checkpoints: tuple[float, ...],
    exact_cost_context: ExactCostContext | None = None,
) -> RankingMetrics:
    """Evaluate one 64-cell ranking under the exact M0 selection semantics."""

    truth = np.asarray(mechanical_values, dtype=np.float64)
    scores = np.asarray(predicted_scores, dtype=np.float64)
    if (
        truth.shape != (64,)
        or scores.shape != (64,)
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(scores))
        or type(grid) is not AcquisitionGrid
        or not checkpoints
    ):
        raise ValueError("ranking metric inputs changed")
    predicted_order = _order(scores)
    oracle_order = _order(truth)
    correlation = float(spearmanr(truth, scores).statistic)
    if not np.isfinite(correlation):
        correlation = 0.0
    if exact_cost_context is None:
        predicted_plan = plan_frozen_ranking(
            grid,
            initial_state(grid),
            ranking=score_initial_ranking(lambda: scores, method="predicted"),
            checkpoints=checkpoints,
        )
        oracle_plan = plan_frozen_ranking(
            grid,
            initial_state(grid),
            ranking=score_initial_ranking(lambda: truth, method="oracle"),
            checkpoints=checkpoints,
        )

        def selected_by_checkpoint(
            plan: object, checkpoint: float
        ) -> tuple[int, ...]:
            count = plan.snapshots[
                plan.checkpoints.index(checkpoint)
            ].cumulative_actions
            return tuple(action.cell_index for action in plan.actions[:count])

        predicted_sets = tuple(
            selected_by_checkpoint(predicted_plan, checkpoint)
            for checkpoint in checkpoints
        )
        oracle_sets = tuple(
            selected_by_checkpoint(oracle_plan, checkpoint)
            for checkpoint in checkpoints
        )
    else:
        if (
            type(exact_cost_context) is not ExactCostContext
            or exact_cost_context.grid_state_sha256 != grid.state_sha256
            or exact_cost_context.native_count
            != grid.native_shape[0] * grid.native_shape[1]
        ):
            raise ValueError("exact candidate-cost authority changed")

        def select(order: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
            measured = exact_cost_context.initial_measured_count
            mask = np.zeros(exact_cost_context.native_count, dtype=np.bool_)
            mask[exact_cost_context.initial_locations] = True
            selected: list[int] = []
            selected_set: set[int] = set()
            snapshots: list[tuple[int, ...]] = []
            for checkpoint in checkpoints:
                for cell in order:
                    if cell in selected_set:
                        continue
                    locations = exact_cost_context.action_locations[cell]
                    added = int(np.count_nonzero(~mask[locations]))
                    candidate = measured + added
                    if candidate / exact_cost_context.native_count > checkpoint:
                        continue
                    measured = candidate
                    mask[locations] = True
                    selected.append(cell)
                    selected_set.add(cell)
                snapshots.append(tuple(selected))
            return tuple(snapshots)

        predicted_sets = select(predicted_order)
        oracle_sets = select(oracle_order)

    selected_values: list[float] = []
    oracle_values: list[float] = []
    regrets: list[float] = []
    capture: list[float | None] = []
    for predicted_cells, oracle_cells in zip(
        predicted_sets, oracle_sets, strict=True
    ):
        predicted_value = float(np.sum(truth[list(predicted_cells)], dtype=np.float64))
        oracle_value = float(np.sum(truth[list(oracle_cells)], dtype=np.float64))
        selected_values.append(predicted_value)
        oracle_values.append(oracle_value)
        regrets.append(oracle_value - predicted_value)
        capture.append(
            None
            if abs(oracle_value) <= np.finfo(np.float64).eps
            else float(predicted_value / oracle_value)
        )
    top5 = set(oracle_order[:5])
    top10 = set(oracle_order[:10])
    return RankingMetrics(
        spearman=correlation,
        ndcg_5=_ndcg(truth, predicted_order, 5),
        ndcg_10=_ndcg(truth, predicted_order, 10),
        recall_5=len(top5 & set(predicted_order[:5])) / 5.0,
        recall_10=len(top10 & set(predicted_order[:10])) / 10.0,
        regret_1=float(truth[oracle_order[0]] - truth[predicted_order[0]]),
        checkpoints=checkpoints,
        budgeted_regret=tuple(regrets),
        selected_value=tuple(selected_values),
        oracle_value=tuple(oracle_values),
        value_capture=tuple(capture),
    )


__all__ = [
    "ExactCostContext",
    "RankingMetrics",
    "build_exact_cost_context",
    "evaluate_ranking",
]
