"""Rollout adapter for the registered spatial P2 and unchanged P3 scorer."""

from __future__ import annotations

import numpy as np
import torch

from ..contracts import InspectionState
from ..dynamic_training import FittedDynamicVoI
from ..dynamic_voi import CandidateDescriptor
from .training import FittedSpatialMRISModel


class SpatialProbePolicyError(ValueError):
    """Raised when the spatial probe cannot satisfy the frozen policy contract."""


class SpatialProbeDeployedScorer:
    """Compose legal-state spatial P2 with the existing dynamic action scorer."""

    def __init__(
        self,
        *,
        mris_model: FittedSpatialMRISModel,
        dynamic_model: FittedDynamicVoI,
        device: str = "cpu",
    ) -> None:
        if (
            type(mris_model) is not FittedSpatialMRISModel
            or type(dynamic_model) is not FittedDynamicVoI
            or mris_model.mode != "real"
            or mris_model.outer_domain != dynamic_model.outer_domain
            or mris_model.mris_dimension != dynamic_model.mris_dimension
            or type(device) is not str
            or not device
        ):
            raise SpatialProbePolicyError("spatial deployed scorer models are incompatible")
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
            raise SpatialProbePolicyError("spatial policy scoring request is invalid")
        embedding = self._mris.encode_inspection_state(state, device=self._device)
        if embedding.shape != (self._dynamic.mris_dimension,) or not np.all(
            np.isfinite(embedding)
        ):
            raise SpatialProbePolicyError("spatial policy embedding is invalid")
        dynamic_model = self._dynamic.model.to(self._device).eval()
        parameter = next(dynamic_model.parameters())
        with torch.inference_mode():
            result = dynamic_model.scorer.score_actions(
                torch.tensor(
                    embedding,
                    dtype=parameter.dtype,
                    device=self._device,
                ),
                candidates,
            )
        return type(result)(
            scores=result.scores.detach().cpu(),
            value_predictions=result.value_predictions.detach().cpu(),
        )


__all__ = ["SpatialProbeDeployedScorer", "SpatialProbePolicyError"]
