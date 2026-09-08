"""Immutable actor-input and reviewed-report adapters for the supplement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch

from cmc_bbdm.vlm_cscan.contracts import BenchmarkTask, TaskReport

from .artifacts import _atomic_write
from .observation import (
    CELL_FEATURE_COUNT,
    GLOBAL_FEATURE_COUNT,
    HISTORY_FEATURE_COUNT,
    SUBBLOCK_FEATURE_COUNT,
    ObservationPacket,
)
from .policies import LearnedCellActor
from .readout import TaskReportV2
from .training import PolicyTrainingExample


class ActorInputMode(StrEnum):
    FULL = "FULL"
    NO_VLM = "NO_VLM"
    NO_US_FEEDBACK = "NO_US_FEEDBACK"


@dataclass(frozen=True, slots=True)
class SupplementActorCheckpoint:
    view: SupplementActorView
    method: str
    seed: int
    parent_config_sha256: str
    parameter_count: int
    optimizer_steps: int


def _copied_actor_tensors(
    tensors: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    required = {
        "cell_features": ((64, CELL_FEATURE_COUNT), np.float32),
        "subblock_features": ((64, 16, SUBBLOCK_FEATURE_COUNT), np.float32),
        "global_features": ((GLOBAL_FEATURE_COUNT,), np.float32),
        "history_features": ((4, HISTORY_FEATURE_COUNT), np.float32),
        "legal_mask": ((64,), np.bool_),
    }
    if set(tensors) != set(required):
        raise ValueError("actor tensor fields are invalid")
    copied: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in required.items():
        value = np.asarray(tensors[name])
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"actor tensor {name} is invalid")
        copied[name] = np.array(value, copy=True)
    if (
        not np.all(np.isfinite(copied["cell_features"]))
        or not np.all(np.isfinite(copied["subblock_features"]))
        or not np.all(np.isfinite(copied["global_features"]))
        or not np.all(np.isfinite(copied["history_features"]))
        or not copied["legal_mask"].any()
    ):
        raise ValueError("actor tensors contain invalid values")
    return copied


def mask_actor_tensors(
    tensors: Mapping[str, np.ndarray], mode: ActorInputMode
) -> dict[str, np.ndarray]:
    """Return copied actor arrays with only the contracted channels removed."""

    if type(mode) is not ActorInputMode:
        raise TypeError("typed actor input mode is required")
    copied = _copied_actor_tensors(tensors)
    if mode is ActorInputMode.NO_VLM:
        copied["cell_features"][:, 15:17] = 0.0
        copied["global_features"][6] = 0.0
    elif mode is ActorInputMode.NO_US_FEEDBACK:
        copied["cell_features"][:, 4:15] = 0.0
        copied["subblock_features"][:, :, 2:10] = 0.0
        copied["global_features"][7] = 0.0
    return copied


def transform_policy_example(
    example: PolicyTrainingExample, mode: ActorInputMode
) -> PolicyTrainingExample:
    """Apply the inference-time input view to one immutable training row."""

    if type(example) is not PolicyTrainingExample:
        raise TypeError("typed policy example is required")
    tensors = mask_actor_tensors(
        {
            "cell_features": example.cell_features,
            "subblock_features": example.subblock_features,
            "global_features": example.global_features,
            "history_features": example.history_features,
            "legal_mask": example.legal_mask,
        },
        mode,
    )
    return PolicyTrainingExample(
        specimen_key=example.specimen_key,
        task=example.task,
        cell_features=tensors["cell_features"],
        subblock_features=tensors["subblock_features"],
        global_features=tensors["global_features"],
        history_features=tensors["history_features"],
        legal_mask=tensors["legal_mask"],
        queried_cells=example.queried_cells,
        target_probabilities=np.array(
            example.target_probabilities, dtype=np.float32, copy=True
        ),
    )


@dataclass(frozen=True, slots=True)
class SupplementActorView:
    """Duck-typed planner that applies a fixed actor-input view."""

    actor: LearnedCellActor
    mode: ActorInputMode

    def __post_init__(self) -> None:
        if type(self.actor) is not LearnedCellActor or type(
            self.mode
        ) is not ActorInputMode:
            raise TypeError("supplement actor view is invalid")
        expected_surface = self.mode is not ActorInputMode.NO_VLM
        if self.actor.use_surface_features is not expected_surface:
            raise ValueError("actor architecture and input mode disagree")

    def score_cells(
        self, packet: ObservationPacket, *, device: str = "cpu"
    ) -> np.ndarray:
        if type(packet) is not ObservationPacket:
            raise TypeError("typed observation packet is required")
        arrays = mask_actor_tensors(packet.actor_tensors(), self.mode)
        target = torch.device(device)
        with torch.no_grad():
            logits = self.actor(
                torch.as_tensor(arrays["cell_features"])
                .unsqueeze(0)
                .to(target),
                torch.as_tensor(arrays["subblock_features"])
                .unsqueeze(0)
                .to(target),
                torch.as_tensor(arrays["global_features"])
                .unsqueeze(0)
                .to(target),
                torch.as_tensor(arrays["history_features"])
                .unsqueeze(0)
                .to(target),
                torch.as_tensor(arrays["legal_mask"])
                .unsqueeze(0)
                .to(target),
            )
        scores = logits[0].detach().cpu().numpy().astype(np.float64, copy=True)
        scores.setflags(write=False)
        return scores

    def select_cell(
        self, packet: ObservationPacket, *, device: str = "cpu"
    ) -> int:
        return int(np.argmax(self.score_cells(packet, device=device)))


def adapt_task_report_v2(report: TaskReportV2) -> TaskReport:
    """Convert Reader v2 output without changing its mask or support."""

    if type(report) is not TaskReportV2:
        raise TypeError("typed Reader v2 report is required")
    return TaskReport(
        task=BenchmarkTask(report.task.value),
        predicted_mask=np.array(report.predicted_mask, copy=True),
        support_positions=np.array(report.support_positions, copy=True),
        confidence=float(report.signal_strength),
        public_complete=bool(report.candidate_cells)
        and not bool(report.unverified_boundary_cells),
        reason_code=report.reason_code,
    )


def save_supplement_actor(
    path: str | Path,
    actor: LearnedCellActor,
    *,
    method: str,
    seed: int,
    mode: ActorInputMode,
    parent_config_sha256: str,
    optimizer_steps: int,
) -> None:
    """Save a new supplement checkpoint with explicit non-parent identity."""

    if (
        type(actor) is not LearnedCellActor
        or not method.startswith("BC_")
        or type(seed) is not int
        or seed < 1
        or type(mode) is not ActorInputMode
        or len(parent_config_sha256) != 64
        or type(optimizer_steps) is not int
        or not 0 <= optimizer_steps <= 4000
    ):
        raise ValueError("supplement actor checkpoint request is invalid")
    expected_surface = mode is not ActorInputMode.NO_VLM
    if actor.use_surface_features is not expected_surface:
        raise ValueError("actor architecture and input mode disagree")
    state = {
        name: value.detach().cpu() for name, value in actor.state_dict().items()
    }
    payload = {
        "schema_version": 1,
        "stage": "BC_CSCAN_PATH_B_SUPPLEMENT",
        "method": method,
        "seed": seed,
        "actor_input_mode": mode.value,
        "parent_config_sha256": parent_config_sha256,
        "parameter_count": actor.parameter_count,
        "optimizer_steps": optimizer_steps,
        "state_dict": state,
    }
    _atomic_write(Path(path), lambda temporary: torch.save(payload, temporary))


def load_supplement_actor(
    path: str | Path, *, parent_config_sha256: str, device: str = "cpu"
) -> SupplementActorCheckpoint:
    """Load a strict supplement checkpoint and return its fixed input view."""

    source = Path(path).resolve(strict=True)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if type(payload) is not dict:
        raise ValueError("supplement actor checkpoint is invalid")
    try:
        mode = ActorInputMode(payload["actor_input_mode"])
        method = str(payload["method"])
        seed = int(payload["seed"])
        parameter_count = int(payload["parameter_count"])
        optimizer_steps = int(payload["optimizer_steps"])
        state = payload["state_dict"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("supplement actor checkpoint is invalid") from error
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "BC_CSCAN_PATH_B_SUPPLEMENT"
        or payload.get("parent_config_sha256") != parent_config_sha256
        or not method.startswith("BC_")
        or seed < 1
        or type(state) is not dict
        or not 0 <= optimizer_steps <= 4000
    ):
        raise ValueError("supplement actor checkpoint identity changed")
    actor = LearnedCellActor(
        use_surface_features=mode is not ActorInputMode.NO_VLM
    )
    actor.load_state_dict(state, strict=True)
    if actor.parameter_count != parameter_count:
        raise ValueError("supplement actor parameter count changed")
    actor.to(torch.device(device))
    actor.eval()
    return SupplementActorCheckpoint(
        view=SupplementActorView(actor, mode),
        method=method,
        seed=seed,
        parent_config_sha256=parent_config_sha256,
        parameter_count=parameter_count,
        optimizer_steps=optimizer_steps,
    )


__all__ = [
    "ActorInputMode",
    "SupplementActorCheckpoint",
    "SupplementActorView",
    "adapt_task_report_v2",
    "load_supplement_actor",
    "mask_actor_tensors",
    "save_supplement_actor",
    "transform_policy_example",
]
