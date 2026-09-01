"""Small shared and structured observable G1 action policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .contracts import (
    ACTION_SLOT_COUNT,
    CANDIDATE_FEATURE_DIMENSION,
    CELL_COUNT,
    CELL_FEATURE_DIMENSION,
    GLOBAL_SCALAR_DIMENSION,
    RECONSTRUCTION_EMBEDDING_DIMENSION,
)

_PARAMETER_CAP = 1_000_000
_ACTION_CELL_INDICES = torch.arange(ACTION_SLOT_COUNT) % CELL_COUNT


class G1PolicyModelError(ValueError):
    """Raised when model inputs or legal masks violate the actor contract."""


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    action_logits: torch.Tensor
    stop_logits: torch.Tensor
    global_context: torch.Tensor


def _validate_inputs(
    reconstruction_embedding: torch.Tensor,
    global_scalars: torch.Tensor,
    task_token: torch.Tensor,
    cell_features: torch.Tensor,
    candidate_features: torch.Tensor,
    legal_action_mask: torch.Tensor,
) -> int:
    values = (
        reconstruction_embedding,
        global_scalars,
        task_token,
        cell_features,
        candidate_features,
        legal_action_mask,
    )
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise G1PolicyModelError("G1 policy inputs must be tensors")
    batch = reconstruction_embedding.shape[0] if reconstruction_embedding.ndim == 2 else -1
    if (
        batch < 1
        or reconstruction_embedding.shape != (batch, RECONSTRUCTION_EMBEDDING_DIMENSION)
        or global_scalars.shape != (batch, GLOBAL_SCALAR_DIMENSION)
        or task_token.shape != (batch, 2)
        or cell_features.shape != (batch, CELL_COUNT, CELL_FEATURE_DIMENSION)
        or candidate_features.shape
        != (batch, ACTION_SLOT_COUNT, CANDIDATE_FEATURE_DIMENSION)
        or legal_action_mask.shape != (batch, ACTION_SLOT_COUNT)
        or legal_action_mask.dtype is not torch.bool
        or any(not value.is_floating_point() for value in values[:-1])
        or any(not torch.isfinite(value).all() for value in values[:-1])
        or len({value.device for value in values}) != 1
    ):
        raise G1PolicyModelError("G1 policy tensor contract is invalid")
    return batch


def _masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -torch.inf)


def masked_action_probabilities(
    action_logits: torch.Tensor,
    legal_action_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        not isinstance(action_logits, torch.Tensor)
        or not isinstance(legal_action_mask, torch.Tensor)
        or action_logits.ndim != 2
        or action_logits.shape != legal_action_mask.shape
        or legal_action_mask.dtype is not torch.bool
        or not torch.all(legal_action_mask.any(dim=1))
    ):
        raise G1PolicyModelError("masked action probability request is invalid")
    return torch.softmax(_masked(action_logits, legal_action_mask), dim=1)


class _ObservablePolicy(nn.Module):
    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _enforce_cap(self) -> None:
        if self.parameter_count >= _PARAMETER_CAP:
            raise G1PolicyModelError("G1 policy exceeds the trainable parameter cap")


class SharedActionMLP(_ObservablePolicy):
    """Score every candidate independently with shared parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.Linear(RECONSTRUCTION_EMBEDDING_DIMENSION + GLOBAL_SCALAR_DIMENSION + 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(128 + CELL_FEATURE_DIMENSION + CANDIDATE_FEATURE_DIMENSION, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.stop_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer("action_cell_indices", _ACTION_CELL_INDICES.clone(), persistent=False)
        self._enforce_cap()

    def forward(
        self,
        reconstruction_embedding: torch.Tensor,
        global_scalars: torch.Tensor,
        task_token: torch.Tensor,
        cell_features: torch.Tensor,
        candidate_features: torch.Tensor,
        legal_action_mask: torch.Tensor,
    ) -> PolicyOutput:
        batch = _validate_inputs(
            reconstruction_embedding,
            global_scalars,
            task_token,
            cell_features,
            candidate_features,
            legal_action_mask,
        )
        context = self.global_encoder(
            torch.cat((reconstruction_embedding, global_scalars, task_token), dim=1)
        )
        cells = cell_features[:, self.action_cell_indices]
        expanded = context.unsqueeze(1).expand(batch, ACTION_SLOT_COUNT, 128)
        logits = self.action_scorer(
            torch.cat((expanded, cells, candidate_features), dim=2)
        ).squeeze(-1)
        return PolicyOutput(
            action_logits=_masked(logits, legal_action_mask),
            stop_logits=self.stop_head(context).squeeze(-1),
            global_context=context,
        )


class StructuredInspectionPolicy(_ObservablePolicy):
    """Contextualize all cells before scoring canonical primitive actions."""

    def __init__(self) -> None:
        super().__init__()
        self.reconstruction_encoder = nn.Sequential(
            nn.Linear(RECONSTRUCTION_EMBEDDING_DIMENSION, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(GLOBAL_SCALAR_DIMENSION, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.task_encoder = nn.Sequential(nn.Linear(2, 16), nn.GELU())
        self.global_fusion = nn.Sequential(
            nn.Linear(176, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(CELL_FEATURE_DIMENSION, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.contextualizer = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(128 + 128 + CANDIDATE_FEATURE_DIMENSION, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.stop_head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer("action_cell_indices", _ACTION_CELL_INDICES.clone(), persistent=False)
        self._enforce_cap()

    def forward(
        self,
        reconstruction_embedding: torch.Tensor,
        global_scalars: torch.Tensor,
        task_token: torch.Tensor,
        cell_features: torch.Tensor,
        candidate_features: torch.Tensor,
        legal_action_mask: torch.Tensor,
    ) -> PolicyOutput:
        batch = _validate_inputs(
            reconstruction_embedding,
            global_scalars,
            task_token,
            cell_features,
            candidate_features,
            legal_action_mask,
        )
        global_token = self.global_fusion(
            torch.cat(
                (
                    self.reconstruction_encoder(reconstruction_embedding),
                    self.task_encoder(task_token),
                    self.scalar_encoder(global_scalars),
                ),
                dim=1,
            )
        )
        cell_tokens = self.cell_encoder(cell_features)
        contextual = self.contextualizer(
            torch.cat((global_token.unsqueeze(1), cell_tokens), dim=1)
        )
        global_context = contextual[:, 0]
        candidate_cells = contextual[:, 1:][:, self.action_cell_indices]
        expanded = global_context.unsqueeze(1).expand(batch, ACTION_SLOT_COUNT, 128)
        logits = self.action_scorer(
            torch.cat((candidate_cells, expanded, candidate_features), dim=2)
        ).squeeze(-1)
        return PolicyOutput(
            action_logits=_masked(logits, legal_action_mask),
            stop_logits=self.stop_head(global_context).squeeze(-1),
            global_context=global_context,
        )


__all__ = [
    "G1PolicyModelError",
    "PolicyOutput",
    "SharedActionMLP",
    "StructuredInspectionPolicy",
    "masked_action_probabilities",
]
