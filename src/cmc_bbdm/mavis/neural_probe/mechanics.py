"""Mechanics prediction head for the registered spatial MRIS encoder."""

from __future__ import annotations

import torch
from torch import nn

from ..mechanics_head import MechanicsHead
from ..state_encoder import MRISInput
from .state_encoder import SpatialGridMRISStateEncoder


class SpatialMRISMechanicsError(ValueError):
    """Raised when the spatial mechanics model contract is violated."""


class SpatialMRISMechanicsModel(nn.Module):
    """Registered spatial encoder followed by the unchanged scalar head shape."""

    def __init__(self, encoder: SpatialGridMRISStateEncoder) -> None:
        super().__init__()
        if type(encoder) is not SpatialGridMRISStateEncoder:
            raise SpatialMRISMechanicsError("issued spatial MRIS encoder is required")
        self.encoder = encoder
        self.head = MechanicsHead(encoder.output_dimension)

    def forward(self, state: MRISInput) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(state)
        return embedding, self.head(embedding)

    def forward_batch(
        self,
        contexts: torch.Tensor,
        token_features: torch.Tensor,
        token_masks: torch.Tensor,
        cost_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder.forward_batch(
            contexts,
            token_features,
            token_masks,
            cost_features,
        )
        return embedding, self.head(embedding)


__all__ = ["SpatialMRISMechanicsError", "SpatialMRISMechanicsModel"]
