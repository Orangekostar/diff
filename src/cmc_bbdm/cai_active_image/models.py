"""Small visible-state networks for CAI prediction and acquisition."""

from __future__ import annotations

import torch
from torch import nn


def _coordinates() -> torch.Tensor:
    values = [[row / 7.0, column / 7.0] for row in range(8) for column in range(8)]
    return torch.tensor(values, dtype=torch.float32)


def _check_token_inputs(
    surface: torch.Tensor, cscan: torch.Tensor, measured: torch.Tensor
) -> None:
    if (
        surface.ndim != 3
        or cscan.shape != surface.shape
        or surface.shape[1] != 64
        or measured.shape != surface.shape[:2]
        or measured.dtype is not torch.bool
    ):
        raise ValueError("visible-state token shapes are invalid")


class CommonCAIPredictor(nn.Module):
    """One policy-independent predictor using surface and acquired C-scan only."""

    def __init__(
        self,
        *,
        token_dimension: int = 512,
        width: int = 64,
        target_mean: float = 300.0,
        target_scale: float = 100.0,
    ) -> None:
        super().__init__()
        if token_dimension < 1 or width < 4 or target_scale <= 0:
            raise ValueError("predictor dimensions are invalid")
        self.surface = nn.Linear(token_dimension, width)
        self.cscan = nn.Linear(token_dimension, width)
        self.cell = nn.Sequential(
            nn.Linear(2 * width + 3, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width + 1, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.register_buffer("coordinates", _coordinates(), persistent=True)
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_scale", torch.tensor(float(target_scale)))

    def forward(
        self,
        surface: torch.Tensor,
        cscan: torch.Tensor,
        measured: torch.Tensor,
        *,
        cost: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _check_token_inputs(surface, cscan, measured)
        observed = torch.where(measured.unsqueeze(-1), cscan, torch.zeros_like(cscan))
        coords = self.coordinates.to(device=surface.device, dtype=surface.dtype)
        coords = coords.unsqueeze(0).expand(surface.shape[0], -1, -1)
        cell = self.cell(
            torch.cat(
                [
                    self.surface(surface),
                    self.cscan(observed),
                    measured.to(surface.dtype).unsqueeze(-1),
                    coords,
                ],
                dim=-1,
            )
        )
        surface_pool = cell.mean(dim=1)
        weights = measured.to(surface.dtype).unsqueeze(-1)
        measured_pool = (cell * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if cost is None:
            cost_column = measured.to(surface.dtype).mean(dim=1, keepdim=True)
        else:
            if cost.shape != (surface.shape[0],):
                raise ValueError("exact acquisition cost shape is invalid")
            cost_column = cost.to(surface.dtype).unsqueeze(-1)
        normalized = self.head(
            torch.cat([surface_pool, measured_pool, cost_column], dim=-1)
        )
        return self.target_mean + self.target_scale * normalized.squeeze(-1)


class CAIActor(nn.Module):
    """Cell actor with explicit VLM and feedback ablation switches."""

    def __init__(
        self,
        *,
        token_dimension: int = 512,
        width: int = 64,
        use_vlm: bool,
        use_feedback: bool,
    ) -> None:
        super().__init__()
        if token_dimension < 1 or width < 4:
            raise ValueError("actor dimensions are invalid")
        self.use_vlm = bool(use_vlm)
        self.use_feedback = bool(use_feedback)
        self.surface = nn.Linear(token_dimension, width)
        self.cscan = nn.Linear(token_dimension, width)
        local_width = 2 * width + 6
        global_width = 2 * width + 5
        self.score = nn.Sequential(
            nn.Linear(local_width + global_width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(global_width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.register_buffer("coordinates", _coordinates(), persistent=True)

    def forward(
        self,
        surface: torch.Tensor,
        cscan: torch.Tensor,
        measured: torch.Tensor,
        action_history: torch.Tensor,
        vlm_indicator: torch.Tensor,
        vlm_confidence: torch.Tensor,
        vlm_available: torch.Tensor,
        vlm_no_reliable: torch.Tensor,
        current_prediction_mpa: torch.Tensor,
        cost: torch.Tensor,
        remaining_cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_token_inputs(surface, cscan, measured)
        batch = surface.shape[0]
        if (
            action_history.shape != (batch, 64)
            or vlm_indicator.shape != (batch, 64)
            or vlm_confidence.shape != (batch, 64)
            or vlm_available.shape != (batch,)
            or vlm_no_reliable.shape != (batch,)
            or current_prediction_mpa.shape != (batch,)
            or cost.shape != (batch,)
            or remaining_cost.shape != (batch,)
        ):
            raise ValueError("actor side-channel shapes are invalid")

        surface_local = self.surface(surface)
        if self.use_feedback:
            observed = torch.where(
                measured.unsqueeze(-1), cscan, torch.zeros_like(cscan)
            )
            cscan_local = self.cscan(observed)
            observed_weights = measured.to(surface.dtype).unsqueeze(-1)
            cscan_global = (cscan_local * observed_weights).sum(dim=1) / (
                observed_weights.sum(dim=1).clamp_min(1.0)
            )
            prediction = current_prediction_mpa / 500.0
        else:
            cscan_local = torch.zeros(
                batch, 64, self.cscan.out_features, device=surface.device, dtype=surface.dtype
            )
            cscan_global = torch.zeros(
                batch, self.cscan.out_features, device=surface.device, dtype=surface.dtype
            )
            prediction = torch.zeros_like(current_prediction_mpa)

        if self.use_vlm:
            indicator = vlm_indicator.to(surface.dtype)
            confidence = vlm_confidence.to(surface.dtype)
            available = vlm_available.to(surface.dtype)
            no_reliable = vlm_no_reliable.to(surface.dtype)
        else:
            indicator = torch.zeros_like(vlm_indicator, dtype=surface.dtype)
            confidence = torch.zeros_like(vlm_confidence, dtype=surface.dtype)
            available = torch.zeros_like(vlm_available, dtype=surface.dtype)
            no_reliable = torch.zeros_like(vlm_no_reliable, dtype=surface.dtype)

        coords = self.coordinates.to(device=surface.device, dtype=surface.dtype)
        coords = coords.unsqueeze(0).expand(batch, -1, -1)
        local = torch.cat(
            [
                surface_local,
                cscan_local,
                measured.to(surface.dtype).unsqueeze(-1),
                action_history.to(surface.dtype).unsqueeze(-1),
                coords,
                indicator.unsqueeze(-1),
                confidence.unsqueeze(-1),
            ],
            dim=-1,
        )
        global_state = torch.cat(
            [
                surface_local.mean(dim=1),
                cscan_global,
                prediction.unsqueeze(-1),
                cost.unsqueeze(-1),
                remaining_cost.unsqueeze(-1),
                available.unsqueeze(-1),
                no_reliable.unsqueeze(-1),
            ],
            dim=-1,
        )
        expanded = global_state.unsqueeze(1).expand(-1, 64, -1)
        scores = self.score(torch.cat([local, expanded], dim=-1)).squeeze(-1)
        return scores, self.value(global_state).squeeze(-1)


__all__ = ["CAIActor", "CommonCAIPredictor"]
