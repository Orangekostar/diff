"""Fixed 8x8 spatial encoder for the existing legal MRIS token summary."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..state_encoder import MRISInput, summarize_mris_input


class SpatialGridEncoderError(ValueError):
    """Raised when the registered spatial encoder contract is violated."""


def spatial_grid_from_tokens(
    token_features: torch.Tensor,
    token_masks: torch.Tensor,
) -> torch.Tensor:
    """Convert row-major legal-state tokens into the registered seven-channel grid."""
    if (
        not isinstance(token_features, torch.Tensor)
        or not isinstance(token_masks, torch.Tensor)
        or token_features.ndim != 3
        or token_features.shape[1:] != (64, 6)
        or token_masks.shape != token_features.shape[:2]
        or token_masks.dtype != torch.bool
        or not token_features.is_floating_point()
        or token_features.device != token_masks.device
        or not torch.isfinite(token_features).all()
    ):
        raise SpatialGridEncoderError("spatial token batch is invalid")
    weights = token_masks.unsqueeze(-1).to(dtype=token_features.dtype)
    features = (token_features * weights).transpose(1, 2).reshape(-1, 6, 8, 8)
    mask_channel = token_masks.reshape(-1, 1, 8, 8).to(dtype=token_features.dtype)
    return torch.cat((features, mask_channel), dim=1)


class SpatialGridMRISStateEncoder(nn.Module):
    """Parameter-matched spatial encoder registered as ``spatial_grid_cnn_v1``."""

    architecture_name = "spatial_grid_cnn_v1"

    def __init__(self, *, context_dimension: int, output_dimension: int) -> None:
        super().__init__()
        if context_dimension != 34 or output_dimension != 64:
            raise SpatialGridEncoderError(
                "spatial_grid_cnn_v1 requires 34 context and 64 output dimensions"
            )
        self.context_dimension = context_dimension
        self.output_dimension = output_dimension
        self.spatial_trunk = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dimension + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, output_dimension),
        )

    def forward_batch(
        self,
        contexts: torch.Tensor,
        token_features: torch.Tensor,
        token_masks: torch.Tensor,
        cost_features: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not all(
                isinstance(value, torch.Tensor)
                for value in (contexts, token_features, token_masks, cost_features)
            )
            or contexts.ndim != 2
            or contexts.shape[1] != self.context_dimension
            or token_features.shape != (contexts.shape[0], 64, 6)
            or token_masks.shape != (contexts.shape[0], 64)
            or token_masks.dtype != torch.bool
            or cost_features.shape != (contexts.shape[0], 3)
            or not contexts.is_floating_point()
            or not token_features.is_floating_point()
            or not cost_features.is_floating_point()
            or len(
                {
                    contexts.device,
                    token_features.device,
                    token_masks.device,
                    cost_features.device,
                }
            )
            != 1
            or not torch.isfinite(contexts).all()
            or not torch.isfinite(token_features).all()
            or not torch.isfinite(cost_features).all()
        ):
            raise SpatialGridEncoderError("spatial encoder batch is invalid")
        grid = spatial_grid_from_tokens(token_features, token_masks)
        spatial = self.spatial_trunk(grid)
        pooled = torch.cat(
            (
                torch.mean(spatial, dim=(2, 3)),
                torch.amax(spatial, dim=(2, 3)),
            ),
            dim=1,
        )
        context_state = self.context_mlp(torch.cat((contexts, cost_features), dim=1))
        output = self.fusion(torch.cat((pooled, context_state), dim=1))
        if output.shape != (contexts.shape[0], self.output_dimension) or not torch.isfinite(
            output
        ).all():
            raise SpatialGridEncoderError("spatial encoder output is invalid")
        return output

    def forward(self, state: MRISInput) -> torch.Tensor:
        if type(state) is not MRISInput or state.context_features.shape != (
            self.context_dimension,
        ):
            raise SpatialGridEncoderError("spatial encoder input is invalid")
        parameter = next(self.parameters())
        summary = summarize_mris_input(state)
        output = self.forward_batch(
            torch.tensor(
                np.asarray(summary.context_features)[None],
                dtype=parameter.dtype,
                device=parameter.device,
            ),
            torch.tensor(
                np.asarray(summary.token_features)[None],
                dtype=parameter.dtype,
                device=parameter.device,
            ),
            torch.tensor(
                np.asarray(summary.token_mask)[None],
                dtype=torch.bool,
                device=parameter.device,
            ),
            torch.tensor(
                np.asarray(summary.cost_features)[None],
                dtype=parameter.dtype,
                device=parameter.device,
            ),
        )
        return output[0]


__all__ = [
    "SpatialGridEncoderError",
    "SpatialGridMRISStateEncoder",
    "spatial_grid_from_tokens",
]
