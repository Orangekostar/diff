"""Deterministic exact-cost action selection for MAVIS policies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .authority import MAVISAuthority
from .contracts import InspectionState
from .dynamic_training import FittedDynamicVoI
from .dynamic_voi import CandidateDescriptor
from .mris_training import FittedMRISModel
from .state_encoder import (
    build_mris_input,
    build_shuffled_mris_input,
    summarize_mris_input,
)


class MAVISPolicyError(ValueError):
    """Raised when a policy objective or candidate score roster is invalid."""


_OBJECTIVES = frozenset(("raw_score", "value_per_exact_cost", "direct_cost_aware"))


@dataclass(frozen=True, slots=True)
class PolicySelection:
    candidate_index: int
    candidate: CandidateDescriptor
    raw_score: float
    objective_score: float
    objective: str


class DeployedDynamicScorer:
    """Compose frozen P2/P3 models using policy-visible inspection state only."""

    def __init__(
        self,
        *,
        mris_model: FittedMRISModel,
        dynamic_model: FittedDynamicVoI,
        device: str = "cpu",
    ) -> None:
        if (
            type(mris_model) is not FittedMRISModel
            or type(dynamic_model) is not FittedDynamicVoI
            or mris_model.mode not in {"static", "positions_only", "real"}
            or mris_model.outer_domain != dynamic_model.outer_domain
            or mris_model.mris_dimension != dynamic_model.mris_dimension
            or type(device) is not str
            or not device
        ):
            raise MAVISPolicyError("deployed dynamic scorer models are incompatible")
        self._mris = mris_model
        self._dynamic = dynamic_model
        self._device = device

    @property
    def mode(self) -> str:
        return self._mris.mode

    @property
    def outer_domain(self) -> str:
        return self._mris.outer_domain

    def score_actions(
        self,
        state: InspectionState,
        candidates: tuple[CandidateDescriptor, ...],
    ):
        if (
            type(state) is not InspectionState
            or type(candidates) is not tuple
            or not candidates
            or any(type(candidate) is not CandidateDescriptor for candidate in candidates)
            or any(candidate.native_count != state.native_count for candidate in candidates)
        ):
            raise MAVISPolicyError("deployed dynamic scoring request is invalid")
        issued = build_mris_input(state, mode=self._mris.mode)
        summary = summarize_mris_input(issued)
        normalized = self._mris.normalizer.transform_context(
            summary.context_features[None, :]
        )
        mris_model = self._mris.model.to(self._device).eval()
        dynamic_model = self._dynamic.model.to(self._device).eval()
        parameter = next(mris_model.parameters())
        with torch.inference_mode():
            embedding = mris_model.encoder.forward_batch(
                torch.tensor(normalized, dtype=parameter.dtype, device=self._device),
                torch.tensor(
                    summary.token_features[None, :, :],
                    dtype=parameter.dtype,
                    device=self._device,
                ),
                torch.tensor(
                    summary.token_mask[None, :],
                    dtype=torch.bool,
                    device=self._device,
                ),
                torch.tensor(
                    summary.cost_features[None, :],
                    dtype=parameter.dtype,
                    device=self._device,
                ),
            )[0]
            result = dynamic_model.scorer.score_actions(embedding, candidates)
        return type(result)(
            scores=result.scores.detach().cpu(),
            value_predictions=result.value_predictions.detach().cpu(),
        )


class FrozenCellScorer:
    """Issue a frozen post-scout score for each 8x8 acquisition cell."""

    def __init__(self, cell_scores: object) -> None:
        scores = np.ascontiguousarray(cell_scores, dtype="<f8")
        if scores.shape != (64,) or not np.all(np.isfinite(scores)):
            raise MAVISPolicyError("frozen cell scores are invalid")
        self._scores = np.frombuffer(scores.tobytes(order="C"), dtype="<f8")
        self._scores.setflags(write=False)

    def score_actions(
        self,
        _state: object,
        candidates: tuple[CandidateDescriptor, ...],
    ) -> np.ndarray:
        if (
            type(candidates) is not tuple
            or not candidates
            or any(type(candidate) is not CandidateDescriptor for candidate in candidates)
        ):
            raise MAVISPolicyError("frozen cell candidate roster is invalid")
        output = np.ascontiguousarray(
            [self._scores[candidate.cell_index] for candidate in candidates],
            dtype="<f8",
        )
        result = np.frombuffer(output.tobytes(order="C"), dtype="<f8")
        result.setflags(write=False)
        return result


class ShuffledControlScorer:
    """Evaluation-only scorer with recipient positions and registered donor content."""

    def __init__(
        self,
        *,
        mris_model: FittedMRISModel,
        dynamic_model: FittedDynamicVoI,
        authority: MAVISAuthority,
        donor_specimen_id: str,
        device: str = "cpu",
    ) -> None:
        if (
            type(mris_model) is not FittedMRISModel
            or type(dynamic_model) is not FittedDynamicVoI
            or type(authority) is not MAVISAuthority
            or mris_model.mode != "shuffled"
            or mris_model.outer_domain != dynamic_model.outer_domain
            or mris_model.mris_dimension != dynamic_model.mris_dimension
            or type(donor_specimen_id) is not str
            or donor_specimen_id not in authority.specimen_ids
            or type(device) is not str
            or not device
        ):
            raise MAVISPolicyError("shuffled control scorer is invalid")
        self._mris = mris_model
        self._dynamic = dynamic_model
        self._authority = authority
        self._donor_specimen_id = donor_specimen_id
        self._device = device

    def score_actions(
        self,
        state: InspectionState,
        candidates: tuple[CandidateDescriptor, ...],
    ):
        if (
            type(state) is not InspectionState
            or state.specimen_id == self._donor_specimen_id
            or type(candidates) is not tuple
            or not candidates
            or any(type(candidate) is not CandidateDescriptor for candidate in candidates)
            or any(candidate.native_count != state.native_count for candidate in candidates)
        ):
            raise MAVISPolicyError("shuffled control scoring request is invalid")
        summary = summarize_mris_input(
            build_shuffled_mris_input(
                state,
                authority=self._authority,
                donor_specimen_id=self._donor_specimen_id,
            )
        )
        normalized = self._mris.normalizer.transform_context(
            summary.context_features[None, :]
        )
        mris_model = self._mris.model.to(self._device).eval()
        dynamic_model = self._dynamic.model.to(self._device).eval()
        parameter = next(mris_model.parameters())
        with torch.inference_mode():
            embedding = mris_model.encoder.forward_batch(
                torch.tensor(normalized, dtype=parameter.dtype, device=self._device),
                torch.tensor(
                    summary.token_features[None, :, :],
                    dtype=parameter.dtype,
                    device=self._device,
                ),
                torch.tensor(
                    summary.token_mask[None, :],
                    dtype=torch.bool,
                    device=self._device,
                ),
                torch.tensor(
                    summary.cost_features[None, :],
                    dtype=parameter.dtype,
                    device=self._device,
                ),
            )[0]
            result = dynamic_model.scorer.score_actions(embedding, candidates)
        return type(result)(
            scores=result.scores.detach().cpu(),
            value_predictions=result.value_predictions.detach().cpu(),
        )


def select_cost_aware_action(
    candidates: tuple[CandidateDescriptor, ...],
    scores: object,
    *,
    objective: str,
    tie_tolerance: float = 1.0e-12,
) -> PolicySelection:
    values = np.asarray(scores, dtype=np.float64)
    tolerance = float(tie_tolerance)
    if (
        type(candidates) is not tuple
        or not candidates
        or any(type(value) is not CandidateDescriptor for value in candidates)
        or values.shape != (len(candidates),)
        or not np.all(np.isfinite(values))
        or objective not in _OBJECTIVES
        or isinstance(tie_tolerance, bool)
        or not math.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise MAVISPolicyError("policy selection request is invalid")
    if objective == "value_per_exact_cost":
        objective_scores = values / np.asarray(
            [candidate.exact_added_cost for candidate in candidates],
            dtype=np.float64,
        )
    else:
        objective_scores = values
    best = float(np.max(objective_scores))
    tied = np.flatnonzero(objective_scores >= best - tolerance)
    selected = min(
        (int(index) for index in tied),
        key=lambda index: (
            candidates[index].exact_added_cost,
            candidates[index].cell_index,
            candidates[index].from_level,
            candidates[index].to_level,
            index,
        ),
    )
    return PolicySelection(
        candidate_index=selected,
        candidate=candidates[selected],
        raw_score=float(values[selected]),
        objective_score=float(objective_scores[selected]),
        objective=objective,
    )


__all__ = [
    "DeployedDynamicScorer",
    "FrozenCellScorer",
    "MAVISPolicyError",
    "PolicySelection",
    "ShuffledControlScorer",
    "select_cost_aware_action",
]
