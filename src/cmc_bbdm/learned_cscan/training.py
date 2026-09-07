"""Mini-batch BC and queried cost-to-go preference training."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

from .contracts import Task
from .observation import (
    CELL_FEATURE_COUNT,
    GLOBAL_FEATURE_COUNT,
    HISTORY_FEATURE_COUNT,
    SUBBLOCK_FEATURE_COUNT,
)
from .policies import LearnedCellActor
from .stopping import LearnedStopHead


class TrainingRoute(StrEnum):
    BEHAVIOR_CLONING = "BEHAVIOR_CLONING"
    COST_TO_GO = "COST_TO_GO"


@dataclass(frozen=True, slots=True, eq=False)
class PolicyTrainingExample:
    specimen_key: str
    task: Task
    cell_features: np.ndarray
    subblock_features: np.ndarray
    global_features: np.ndarray
    history_features: np.ndarray
    legal_mask: np.ndarray
    queried_cells: tuple[int, ...]
    target_probabilities: np.ndarray

    def __post_init__(self) -> None:
        cells = _readonly(
            self.cell_features, np.float32, (64, CELL_FEATURE_COUNT)
        )
        subblocks = _readonly(
            self.subblock_features,
            np.float32,
            (64, 16, SUBBLOCK_FEATURE_COUNT),
        )
        global_features = _readonly(
            self.global_features, np.float32, (GLOBAL_FEATURE_COUNT,)
        )
        history = _readonly(
            self.history_features, np.float32, (4, HISTORY_FEATURE_COUNT)
        )
        legal = _readonly(self.legal_mask, np.bool_, (64,))
        targets = _readonly(
            self.target_probabilities, np.float32, (len(self.queried_cells),)
        )
        if (
            not self.specimen_key
            or type(self.task) is not Task
            or type(self.queried_cells) is not tuple
            or not self.queried_cells
            or len(set(self.queried_cells)) != len(self.queried_cells)
            or any(
                type(cell) is not int or not 0 <= cell < 64 or not legal[cell]
                for cell in self.queried_cells
            )
            or not np.any(legal)
            or not np.all(np.isfinite(cells))
            or not np.all(np.isfinite(subblocks))
            or not np.all(np.isfinite(global_features))
            or not np.all(np.isfinite(history))
            or not np.all(np.isfinite(targets))
            or np.any(targets < 0.0)
            or not math.isclose(float(targets.sum()), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("policy training example is invalid")
        object.__setattr__(self, "cell_features", cells)
        object.__setattr__(self, "subblock_features", subblocks)
        object.__setattr__(self, "global_features", global_features)
        object.__setattr__(self, "history_features", history)
        object.__setattr__(self, "legal_mask", legal)
        object.__setattr__(self, "target_probabilities", targets)


@dataclass(frozen=True, slots=True)
class TrainingLogRow:
    optimizer_step: int
    train_loss: float
    valid_loss: float | None


@dataclass(frozen=True, slots=True)
class ActorFitResult:
    initial_loss: float
    final_loss: float
    best_valid_loss: float | None
    optimizer_steps: int
    mini_batches: int
    stopped_early: bool
    log: tuple[TrainingLogRow, ...]


@dataclass(frozen=True, slots=True, eq=False)
class StopTrainingExample:
    specimen_key: str
    task: Task
    cell_features: np.ndarray
    subblock_features: np.ndarray
    global_features: np.ndarray
    history_features: np.ndarray
    label: bool

    def __post_init__(self) -> None:
        cells = _readonly(
            self.cell_features, np.float32, (64, CELL_FEATURE_COUNT)
        )
        subblocks = _readonly(
            self.subblock_features,
            np.float32,
            (64, 16, SUBBLOCK_FEATURE_COUNT),
        )
        global_features = _readonly(
            self.global_features, np.float32, (GLOBAL_FEATURE_COUNT,)
        )
        history = _readonly(
            self.history_features, np.float32, (4, HISTORY_FEATURE_COUNT)
        )
        if (
            not self.specimen_key
            or type(self.task) is not Task
            or type(self.label) is not bool
            or not np.all(np.isfinite(cells))
            or not np.all(np.isfinite(subblocks))
            or not np.all(np.isfinite(global_features))
            or not np.all(np.isfinite(history))
        ):
            raise ValueError("stop training example is invalid")
        object.__setattr__(self, "cell_features", cells)
        object.__setattr__(self, "subblock_features", subblocks)
        object.__setattr__(self, "global_features", global_features)
        object.__setattr__(self, "history_features", history)


@dataclass(frozen=True, slots=True)
class StopFitResult:
    initial_loss: float
    final_loss: float
    best_valid_loss: float | None
    optimizer_steps: int
    stopped_early: bool
    log: tuple[TrainingLogRow, ...]


def fit_actor(
    model: LearnedCellActor,
    examples: tuple[PolicyTrainingExample, ...],
    *,
    route: TrainingRoute,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    seed: int,
    device: str,
    validation_examples: tuple[PolicyTrainingExample, ...] = (),
    validation_interval: int = 250,
    patience: int = 4,
) -> ActorFitResult:
    _validate_fit_request(
        model,
        examples,
        route=route,
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        seed=seed,
        validation_examples=validation_examples,
        validation_interval=validation_interval,
        patience=patience,
    )
    target_device = torch.device(device)
    model.to(target_device)
    generator = np.random.default_rng(seed)
    initial_loss = _evaluate_loss(model, examples, target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    groups: dict[tuple[str, Task], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[(example.specimen_key, example.task)].append(index)
    group_keys = tuple(sorted(groups, key=lambda item: (item[0], item[1].value)))
    best_valid = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    validations_without_improvement = 0
    log: list[TrainingLogRow] = []
    stopped_early = False
    for step in range(1, max_steps + 1):
        indices = []
        offset = (step - 1) * batch_size
        for position in range(batch_size):
            key = group_keys[(offset + position) % len(group_keys)]
            indices.append(int(generator.choice(groups[key])))
        batch = tuple(examples[index] for index in indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, target_device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        valid_loss: float | None = None
        if validation_examples and step % validation_interval == 0:
            valid_loss = _evaluate_loss(model, validation_examples, target_device)
            if valid_loss < best_valid - 1e-10:
                best_valid = valid_loss
                best_state = copy.deepcopy(model.state_dict())
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
                if validations_without_improvement >= patience:
                    stopped_early = True
        log.append(
            TrainingLogRow(
                optimizer_step=step,
                train_loss=float(loss.detach().cpu().item()),
                valid_loss=valid_loss,
            )
        )
        if stopped_early:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_loss = _evaluate_loss(model, examples, target_device)
    return ActorFitResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_valid_loss=None if best_state is None else best_valid,
        optimizer_steps=len(log),
        mini_batches=len(log),
        stopped_early=stopped_early,
        log=tuple(log),
    )


def fit_stop_head(
    model: LearnedStopHead,
    examples: tuple[StopTrainingExample, ...],
    *,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    seed: int,
    device: str,
    validation_examples: tuple[StopTrainingExample, ...] = (),
    validation_interval: int = 250,
    patience: int = 4,
) -> StopFitResult:
    if (
        type(model) is not LearnedStopHead
        or type(examples) is not tuple
        or not examples
        or any(type(row) is not StopTrainingExample for row in examples)
        or type(validation_examples) is not tuple
        or any(type(row) is not StopTrainingExample for row in validation_examples)
        or type(max_steps) is not int
        or not 1 <= max_steps <= 4_000
        or type(batch_size) is not int
        or batch_size < 1
        or not 0.0 < float(learning_rate) <= 1e-2
        or not 0.0 <= float(weight_decay) <= 1e-2
        or not 0.0 < float(gradient_clip) <= 10.0
        or type(seed) is not int
        or type(validation_interval) is not int
        or validation_interval < 1
        or type(patience) is not int
        or patience < 1
    ):
        raise ValueError("stop fit request is invalid")
    target_device = torch.device(device)
    model.to(target_device)
    generator = np.random.default_rng(seed)
    initial_loss = _evaluate_stop_loss(model, examples, target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    groups: dict[tuple[str, Task], list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[(example.specimen_key, example.task)].append(index)
    group_keys = tuple(sorted(groups, key=lambda item: (item[0], item[1].value)))
    best_valid = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    log: list[TrainingLogRow] = []
    stopped_early = False
    for step in range(1, max_steps + 1):
        offset = (step - 1) * batch_size
        indices = [
            int(
                generator.choice(
                    groups[group_keys[(offset + position) % len(group_keys)]]
                )
            )
            for position in range(batch_size)
        ]
        batch = tuple(examples[index] for index in indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _stop_batch_loss(model, batch, target_device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        valid_loss: float | None = None
        if validation_examples and step % validation_interval == 0:
            valid_loss = _evaluate_stop_loss(
                model, validation_examples, target_device
            )
            if valid_loss < best_valid - 1e-10:
                best_valid = valid_loss
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                stopped_early = stale >= patience
        log.append(TrainingLogRow(step, float(loss.detach().cpu()), valid_loss))
        if stopped_early:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return StopFitResult(
        initial_loss=initial_loss,
        final_loss=_evaluate_stop_loss(model, examples, target_device),
        best_valid_loss=None if best_state is None else best_valid,
        optimizer_steps=len(log),
        stopped_early=stopped_early,
        log=tuple(log),
    )


def _stop_batch_loss(
    model: LearnedStopHead,
    examples: tuple[StopTrainingExample, ...],
    device: torch.device,
) -> torch.Tensor:
    cell = torch.from_numpy(np.stack([row.cell_features for row in examples])).to(
        device
    )
    subblock = torch.from_numpy(
        np.stack([row.subblock_features for row in examples])
    ).to(device)
    global_features = torch.from_numpy(
        np.stack([row.global_features for row in examples])
    ).to(device)
    history = torch.from_numpy(
        np.stack([row.history_features for row in examples])
    ).to(device)
    labels = torch.tensor(
        [float(row.label) for row in examples], dtype=torch.float32, device=device
    )
    logits = model(cell, subblock, global_features, history)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)


def _evaluate_stop_loss(
    model: LearnedStopHead,
    examples: tuple[StopTrainingExample, ...],
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        return float(_stop_batch_loss(model, examples, device).cpu())


def _batch_loss(
    model: LearnedCellActor,
    examples: tuple[PolicyTrainingExample, ...],
    device: torch.device,
) -> torch.Tensor:
    tensors = _stack(examples, device)
    logits = model(*tensors)
    losses = []
    for row, example in enumerate(examples):
        indices = torch.tensor(example.queried_cells, dtype=torch.long, device=device)
        target = torch.tensor(example.target_probabilities, device=device)
        log_probabilities = torch.log_softmax(logits[row, indices], dim=0)
        losses.append(-(target * log_probabilities).sum())
    return torch.stack(losses).mean()


def _evaluate_loss(
    model: LearnedCellActor,
    examples: tuple[PolicyTrainingExample, ...],
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        return float(_batch_loss(model, examples, device).cpu().item())


def _stack(
    examples: tuple[PolicyTrainingExample, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(np.stack([row.cell_features for row in examples])).to(device),
        torch.from_numpy(
            np.stack([row.subblock_features for row in examples])
        ).to(device),
        torch.from_numpy(np.stack([row.global_features for row in examples])).to(device),
        torch.from_numpy(np.stack([row.history_features for row in examples])).to(device),
        torch.from_numpy(np.stack([row.legal_mask for row in examples])).to(device),
    )


def _validate_fit_request(
    model: LearnedCellActor,
    examples: tuple[PolicyTrainingExample, ...],
    *,
    route: TrainingRoute,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    seed: int,
    validation_examples: tuple[PolicyTrainingExample, ...],
    validation_interval: int,
    patience: int,
) -> None:
    if (
        type(model) is not LearnedCellActor
        or type(examples) is not tuple
        or not examples
        or any(type(row) is not PolicyTrainingExample for row in examples)
        or type(validation_examples) is not tuple
        or any(type(row) is not PolicyTrainingExample for row in validation_examples)
        or type(route) is not TrainingRoute
        or type(max_steps) is not int
        or not 1 <= max_steps <= 4_000
        or type(batch_size) is not int
        or batch_size < 1
        or not 0.0 < float(learning_rate) <= 1e-2
        or not 0.0 <= float(weight_decay) <= 1e-2
        or not 0.0 < float(gradient_clip) <= 10.0
        or type(seed) is not int
        or type(validation_interval) is not int
        or validation_interval < 1
        or type(patience) is not int
        or patience < 1
    ):
        raise ValueError("actor fit request is invalid")
    if route is TrainingRoute.BEHAVIOR_CLONING and any(
        np.count_nonzero(example.target_probabilities > 1e-8) != 1
        for example in examples
    ):
        raise ValueError("behavior-cloning targets must be one-hot")


def _readonly(value: object, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError("policy training array shape is invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


__all__ = [
    "ActorFitResult",
    "PolicyTrainingExample",
    "StopFitResult",
    "StopTrainingExample",
    "TrainingLogRow",
    "TrainingRoute",
    "fit_actor",
    "fit_stop_head",
]
