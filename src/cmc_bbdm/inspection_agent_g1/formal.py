"""Common-geometry observable execution used by formal G1 folds."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from cmc_bbdm.inspection_agent.cai_assessor import state_scalars
from cmc_bbdm.inspection_agent.contracts import InspectionObservation, InspectionTask
from cmc_bbdm.inspection_agent.field_task import field_loss
from cmc_bbdm.inspection_agent.generalized_reconstruction import (
    SourceBackgroundPrior,
    reconstruct_observation,
)
from cmc_bbdm.inspection_agent.state import (
    InspectionCellAction,
    action_added_positions_from_mask,
    apply_action,
    fitting_actions,
    zero_state,
)
from cmc_bbdm.inspection_agent.surface_hypothesis import SurfaceHypothesis
from cmc_bbdm.inspection_agent.world import CausalInspectionWorld
from cmc_bbdm.mva.acquisition_grid import AcquisitionGrid
from cmc_bbdm.mva.oracle import uniform_cell_order

from .contracts import CAIContextMode, G1PolicyState, TaskTokenMode
from .features import build_policy_state, canonical_action_from_slot
from .metrics import NOMINAL_CHECKPOINTS, EngineeringCurve, build_engineering_curve
from .warm_start import PRIMARY_WARM_START_CELLS

FIXED_BASELINE_METHODS = (
    "RANDOM",
    "ZERO_UNIFORM",
    "CENTER_FIRST",
    "SURFACE_FOCUS",
    "SURVEY_THEN_REFINE_FIXED",
)


class G1FormalExecutionError(ValueError):
    """Raised when formal observable execution leaves the frozen protocol."""


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _fixed_cell_order(
    method: str,
    surface_hypothesis: SurfaceHypothesis,
    *,
    surface_sha256: str,
    specimen_sha256: str,
    random_seed: int,
) -> tuple[int, ...]:
    if method in {"ZERO_UNIFORM", "SURVEY_THEN_REFINE_FIXED"}:
        return uniform_cell_order()
    if method == "CENTER_FIRST":
        return tuple(
            sorted(
                range(64),
                key=lambda cell: (
                    (cell // 8 - 3.5) ** 2 + (cell % 8 - 3.5) ** 2,
                    cell,
                ),
            )
        )
    if method == "SURFACE_FOCUS":
        return tuple(
            sorted(
                range(64),
                key=lambda cell: (
                    -float(surface_hypothesis.scores[cell]),
                    cell,
                ),
            )
        )
    if method == "RANDOM":
        token = hashlib.sha256(
            (
                f"inspection-agent-g1-fixed-random-v1|{random_seed}|"
                f"{surface_sha256}|{specimen_sha256}"
            ).encode("ascii")
        ).hexdigest()
        generator = np.random.Generator(np.random.PCG64(int(token[:16], 16)))
        return tuple(int(value) for value in generator.permutation(64))
    raise G1FormalExecutionError("fixed baseline method is not registered")


def plan_g1_fixed_actions(
    grid: AcquisitionGrid,
    surface_hypothesis: SurfaceHypothesis,
    *,
    surface_sha256: str,
    specimen_sha256: str,
    method: str,
    random_seed: int,
    endpoint_budget: float,
) -> tuple[InspectionCellAction, ...]:
    endpoint = float(endpoint_budget)
    if (
        type(grid) is not AcquisitionGrid
        or type(surface_hypothesis) is not SurfaceHypothesis
        or not _valid_sha256(surface_sha256)
        or not _valid_sha256(specimen_sha256)
        or method not in FIXED_BASELINE_METHODS
        or type(random_seed) is not int
        or isinstance(endpoint_budget, bool)
        or not math.isfinite(endpoint)
        or endpoint != 0.25
    ):
        raise G1FormalExecutionError("fixed baseline request is invalid")
    order = _fixed_cell_order(
        method,
        surface_hypothesis,
        surface_sha256=surface_sha256,
        specimen_sha256=specimen_sha256,
        random_seed=random_seed,
    )
    state = zero_state(grid)
    mask = np.zeros(grid.native_shape, dtype=np.bool_)
    measured = 0
    output: list[InspectionCellAction] = []
    for cell in PRIMARY_WARM_START_CELLS:
        action = canonical_action_from_slot(cell)
        added = action_added_positions_from_mask(grid, state, action, mask)
        if not len(added):
            raise G1FormalExecutionError("K8 warm start contains a zero-cost action")
        state = apply_action(grid, state, action)
        mask[added[:, 0], added[:, 1]] = True
        measured += len(added)
        output.append(action)
    for source, target in ((-1, 0), (0, 1), (1, 2)):
        for cell in order:
            if state.levels[cell] != source:
                continue
            action = InspectionCellAction(cell, source, target)
            added = action_added_positions_from_mask(grid, state, action, mask)
            if not len(added) or (measured + len(added)) / mask.size > endpoint + 1e-15:
                continue
            state = apply_action(grid, state, action)
            mask[added[:, 0], added[:, 1]] = True
            measured += len(added)
            output.append(action)
    positive = any(
        len(
            action_added_positions_from_mask(
                grid,
                state,
                action,
                mask,
            )
        )
        > 0
        for action in fitting_actions(grid, state, endpoint)
    )
    if positive:
        raise G1FormalExecutionError("fixed baseline stopped before its endpoint")
    return tuple(output)


@dataclass(frozen=True, slots=True)
class G1ObservableStateBuilder:
    grid: AcquisitionGrid
    surface_hypothesis: SurfaceHypothesis
    prior: SourceBackgroundPrior
    assessor: object
    encoder: object
    cai_context_mode: CAIContextMode
    task_token_mode: TaskTokenMode

    def __post_init__(self) -> None:
        if (
            type(self.grid) is not AcquisitionGrid
            or type(self.surface_hypothesis) is not SurfaceHypothesis
            or type(self.prior) is not SourceBackgroundPrior
            or not callable(getattr(self.assessor, "predict", None))
            or not _valid_sha256(getattr(self.assessor, "model_state_sha256", None))
            or not callable(getattr(self.encoder, "encode", None))
            or type(self.cai_context_mode) is not CAIContextMode
            or type(self.task_token_mode) is not TaskTokenMode
        ):
            raise G1FormalExecutionError("observable state builder is invalid")

    def __call__(self, observation: InspectionObservation) -> G1PolicyState:
        if (
            type(observation) is not InspectionObservation
            or observation.grid_sha256 != self.grid.state_sha256
        ):
            raise G1FormalExecutionError("observable state request is invalid")
        reconstruction = reconstruct_observation(observation, self.grid, self.prior)
        embeddings = np.asarray(
            self.encoder.encode((reconstruction.image,)),
            dtype=np.float64,
        )
        scalars = state_scalars(observation)[None]
        estimates = np.asarray(
            self.assessor.predict(embeddings, scalars),
            dtype=np.float64,
        )
        if (
            embeddings.shape != (1, 512)
            or estimates.shape != (1,)
            or not np.all(np.isfinite(embeddings))
            or not np.all(np.isfinite(estimates))
        ):
            raise G1FormalExecutionError("observable state dependencies are invalid")
        return build_policy_state(
            observation,
            self.surface_hypothesis,
            self.grid,
            self.prior,
            reconstruction,
            reconstruction_embedding=embeddings[0],
            cai_estimate=float(estimates[0]),
            cai_context_mode=self.cai_context_mode,
            task_token_mode=self.task_token_mode,
        )


def _checkpoint_observations(
    world: CausalInspectionWorld,
    actions: tuple[InspectionCellAction, ...],
) -> tuple[InspectionObservation, ...]:
    current = world.reset()
    all_states = [current]
    for action in actions:
        current = world.step(current, action)
        all_states.append(current)
    selected: list[InspectionObservation] = []
    for checkpoint in NOMINAL_CHECKPOINTS:
        eligible = tuple(
            value
            for value in all_states
            if value.effective_budget <= checkpoint + 1e-15
        )
        if not eligible:
            raise G1FormalExecutionError("action history misses a nominal checkpoint")
        value = eligible[-1]
        if not selected or selected[-1].state_sha256 != value.state_sha256:
            selected.append(value)
    return tuple(selected)


def evaluate_g1_action_history(
    world: CausalInspectionWorld,
    grid: AcquisitionGrid,
    prior: SourceBackgroundPrior,
    *,
    assessor: object,
    encoder: object,
    method: str,
    target_domain: str,
    specimen_sha256: str,
    actions: tuple[InspectionCellAction, ...],
    full_scan: np.ndarray,
    true_cai: float,
) -> EngineeringCurve:
    if (
        type(world) is not CausalInspectionWorld
        or type(grid) is not AcquisitionGrid
        or type(prior) is not SourceBackgroundPrior
        or not callable(getattr(assessor, "predict", None))
        or not _valid_sha256(getattr(assessor, "model_state_sha256", None))
        or not callable(getattr(encoder, "encode", None))
        or type(method) is not str
        or not method
        or type(target_domain) is not str
        or not target_domain
        or not _valid_sha256(specimen_sha256)
        or type(actions) is not tuple
        or len(actions) < 8
        or any(type(action) is not InspectionCellAction for action in actions)
        or tuple(action.cell_index for action in actions[:8])
        != PRIMARY_WARM_START_CELLS
    ):
        raise G1FormalExecutionError("G1 action-history evaluation is invalid")
    observations = _checkpoint_observations(world, actions)
    reconstructions = tuple(
        reconstruct_observation(observation, grid, prior) for observation in observations
    )
    task = observations[0].task
    if task is InspectionTask.FIELD:
        losses = tuple(field_loss(full_scan, value.image) for value in reconstructions)
    elif task is InspectionTask.CAI and math.isfinite(float(true_cai)):
        embeddings = np.asarray(
            encoder.encode(tuple(value.image for value in reconstructions)),
            dtype=np.float64,
        )
        scalars = np.asarray(
            [state_scalars(observation) for observation in observations],
            dtype=np.float64,
        )
        predictions = np.asarray(assessor.predict(embeddings, scalars), dtype=np.float64)
        if predictions.shape != (len(observations),) or not np.all(
            np.isfinite(predictions)
        ):
            raise G1FormalExecutionError("CAI curve predictions are invalid")
        losses = tuple(abs(float(true_cai) - float(value)) for value in predictions)
    else:
        raise G1FormalExecutionError("G1 engineering task is invalid")
    warm = world.replay(actions[:8])
    evaluator_sha = _json_sha(
        {
            "schema": 1,
            "kind": "g1-engineering-evaluator",
            "task": task.value,
            "prior": prior.state_sha256,
            "assessor": assessor.model_state_sha256,
        }
    )
    return build_engineering_curve(
        method=method,
        target_domain=target_domain,
        specimen_sha256=specimen_sha256,
        task=task,
        grid_sha256=grid.state_sha256,
        evaluator_sha256=evaluator_sha,
        warm_start_sha256=warm.state_sha256,
        state_budgets=tuple(value.effective_budget for value in observations),
        state_losses=losses,
        state_sha256=tuple(value.state_sha256 for value in observations),
    )


__all__ = [
    "FIXED_BASELINE_METHODS",
    "G1FormalExecutionError",
    "G1ObservableStateBuilder",
    "evaluate_g1_action_history",
    "plan_g1_fixed_actions",
]
