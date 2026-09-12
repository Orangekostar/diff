"""Visible-state predictors and policy networks for CAI Agent v3."""

from __future__ import annotations

import torch
from torch import nn


def _coordinates() -> torch.Tensor:
    return torch.tensor(
        [[row / 7.0, column / 7.0] for row in range(8) for column in range(8)],
        dtype=torch.float32,
    )


def _validate_image_state(
    surface: torch.Tensor,
    cscan: torch.Tensor,
    measured: torch.Tensor,
    cost: torch.Tensor,
) -> int:
    batch = surface.shape[0] if surface.ndim == 3 else -1
    if (
        batch < 1
        or surface.shape != (batch, 64, 512)
        or cscan.shape != surface.shape
        or measured.shape != (batch, 64)
        or measured.dtype is not torch.bool
        or cost.shape != (batch,)
        or not surface.is_floating_point()
        or not cscan.is_floating_point()
        or not cost.is_floating_point()
        or not torch.isfinite(surface).all()
        or not torch.isfinite(cscan).all()
        or not torch.isfinite(cost).all()
    ):
        raise ValueError("visible image state is invalid")
    return batch


class MeanSCPredictor(nn.Module):
    """V2-style cell MLP and mean aggregation retrained on the v3 split."""

    def __init__(
        self,
        *,
        width: int = 64,
        target_mean: float = 300.0,
        target_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.surface = nn.Linear(512, width)
        self.cscan = nn.Linear(512, width)
        self.cell = nn.Sequential(
            nn.Linear(2 * width + 3, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width + 1, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.register_buffer("coordinates", _coordinates())
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_scale", torch.tensor(float(target_scale)))

    def _forward_weighted(
        self,
        surface: torch.Tensor,
        cscan: torch.Tensor,
        measured_weight: torch.Tensor,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        batch = surface.shape[0]
        observed = measured_weight.unsqueeze(-1) * cscan
        coordinates = self.coordinates.to(surface).unsqueeze(0).expand(batch, -1, -1)
        cells = self.cell(
            torch.cat(
                (
                    self.surface(surface),
                    self.cscan(observed),
                    measured_weight.unsqueeze(-1),
                    coordinates,
                ),
                dim=-1,
            )
        )
        surface_pool = cells.mean(dim=1)
        weights = measured_weight.unsqueeze(-1)
        measured_pool = (cells * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        normalized = self.head(
            torch.cat((surface_pool, measured_pool, cost.unsqueeze(-1)), dim=-1)
        ).squeeze(-1)
        return self.target_mean + self.target_scale * normalized

    def forward(
        self,
        surface: torch.Tensor,
        cscan: torch.Tensor,
        measured: torch.Tensor,
        *,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        _validate_image_state(surface, cscan, measured, cost)
        return self._forward_weighted(surface, cscan, measured.to(surface.dtype), cost)

    def forward_soft(
        self,
        surface: torch.Tensor,
        cscan: torch.Tensor,
        measured_weight: torch.Tensor,
        *,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        """Training-only differentiable grouped-mask input for the GDFS adapter."""

        batch = surface.shape[0] if surface.ndim == 3 else -1
        if (
            batch < 1
            or surface.shape != (batch, 64, 512)
            or cscan.shape != surface.shape
            or measured_weight.shape != (batch, 64)
            or cost.shape != (batch,)
            or torch.any(measured_weight < 0.0)
            or torch.any(measured_weight > 1.0)
            or not torch.isfinite(measured_weight).all()
        ):
            raise ValueError("soft grouped-mask predictor state is invalid")
        return self._forward_weighted(surface, cscan, measured_weight, cost)


class SpatialPredictor(nn.Module):
    """Two-layer spatial Transformer predictor with hard hidden-token masking."""

    token_width = 128
    encoder_layers = 2
    attention_heads = 4

    def __init__(
        self,
        *,
        use_surface: bool,
        target_mean: float = 300.0,
        target_scale: float = 100.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_surface = bool(use_surface)
        self.surface = nn.Linear(512, 48) if self.use_surface else None
        self.cscan = nn.Linear(512, 64)
        self.mask_cscan = nn.Parameter(torch.zeros(512))
        input_width = 48 + 64 + 3 if self.use_surface else 64 + 3
        self.cell = nn.Sequential(
            nn.Linear(input_width, self.token_width),
            nn.GELU(),
            nn.LayerNorm(self.token_width),
        )
        self.query = nn.Sequential(
            nn.Linear(2, self.token_width),
            nn.GELU(),
            nn.LayerNorm(self.token_width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_width,
            nhead=self.attention_heads,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.contextualizer = nn.TransformerEncoder(
            layer, num_layers=self.encoder_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.Linear(self.token_width, self.token_width),
            nn.GELU(),
            nn.Linear(self.token_width, 1),
        )
        self.register_buffer("coordinates", _coordinates())
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_scale", torch.tensor(float(target_scale)))

    def _visible_cscan(
        self, cscan: torch.Tensor, measured_weight: torch.Tensor
    ) -> torch.Tensor:
        mask = self.mask_cscan.to(cscan).view(1, 1, 512)
        return (
            measured_weight.unsqueeze(-1) * cscan
            + (1.0 - measured_weight.unsqueeze(-1)) * mask
        )

    def _forward_weighted(
        self,
        surface: torch.Tensor,
        cscan: torch.Tensor,
        measured_weight: torch.Tensor,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        batch = surface.shape[0]
        visible = self._visible_cscan(cscan, measured_weight)
        coordinates = self.coordinates.to(surface).unsqueeze(0).expand(batch, -1, -1)
        components = [self.cscan(visible)]
        if self.surface is not None:
            components.insert(0, self.surface(surface))
        components.extend((measured_weight.unsqueeze(-1), coordinates))
        cells = self.cell(torch.cat(components, dim=-1))
        query = self.query(torch.stack((cost, measured_weight.mean(dim=1)), dim=-1))
        contextual = self.contextualizer(torch.cat((query.unsqueeze(1), cells), dim=1))
        normalized = self.head(contextual[:, 0]).squeeze(-1)
        return self.target_mean + self.target_scale * normalized

    def forward(
        self,
        surface: torch.Tensor,
        cscan: torch.Tensor,
        measured: torch.Tensor,
        *,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        _validate_image_state(surface, cscan, measured, cost)
        return self._forward_weighted(surface, cscan, measured.to(surface.dtype), cost)

    def forward_soft(
        self,
        surface: torch.Tensor,
        cscan: torch.Tensor,
        measured_weight: torch.Tensor,
        *,
        cost: torch.Tensor,
    ) -> torch.Tensor:
        """Training-only differentiable grouped-mask input for the GDFS adapter."""

        batch = surface.shape[0] if surface.ndim == 3 else -1
        if (
            batch < 1
            or surface.shape != (batch, 64, 512)
            or cscan.shape != surface.shape
            or measured_weight.shape != (batch, 64)
            or cost.shape != (batch,)
            or torch.any(measured_weight < 0.0)
            or torch.any(measured_weight > 1.0)
            or not torch.isfinite(measured_weight).all()
        ):
            raise ValueError("soft grouped-mask predictor state is invalid")
        return self._forward_weighted(surface, cscan, measured_weight, cost)


class SpatialCAIActor(nn.Module):
    """V3 spatial cell Actor adapted from the registered LearnedCellActor shape."""

    token_width = 128
    encoder_layers = 2
    attention_heads = 4

    def __init__(
        self,
        *,
        use_vlm: bool,
        use_feedback: bool,
        target_mean: float = 300.0,
        target_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.use_vlm = bool(use_vlm)
        self.use_feedback = bool(use_feedback)
        self.surface = nn.Linear(512, 48)
        self.cscan = nn.Linear(512, 48)
        self.mask_cscan = nn.Parameter(torch.zeros(512))
        self.cell = nn.Sequential(
            nn.Linear(48 + 48 + 6, self.token_width),
            nn.GELU(),
            nn.LayerNorm(self.token_width),
        )
        self.query = nn.Sequential(
            nn.Linear(6, self.token_width),
            nn.GELU(),
            nn.LayerNorm(self.token_width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_width,
            nhead=self.attention_heads,
            dim_feedforward=256,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.contextualizer = nn.TransformerEncoder(
            layer, num_layers=self.encoder_layers, enable_nested_tensor=False
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(2 * self.token_width, self.token_width),
            nn.GELU(),
            nn.Linear(self.token_width, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.token_width, self.token_width),
            nn.GELU(),
            nn.Linear(self.token_width, 1),
        )
        self.register_buffer("coordinates", _coordinates())
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_scale", torch.tensor(float(target_scale)))

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
        batch = _validate_image_state(surface, cscan, measured, cost)
        if (
            action_history.shape != (batch, 64)
            or vlm_indicator.shape != (batch, 64)
            or vlm_confidence.shape != (batch, 64)
            or vlm_available.shape != (batch,)
            or vlm_no_reliable.shape != (batch,)
            or current_prediction_mpa.shape != (batch,)
            or remaining_cost.shape != (batch,)
        ):
            raise ValueError("Actor side-channel state is invalid")
        surface_local = self.surface(surface)
        if self.use_feedback:
            mask = self.mask_cscan.to(cscan).view(1, 1, 512)
            visible = torch.where(measured.unsqueeze(-1), cscan, mask)
            cscan_local = self.cscan(visible)
            prediction = (current_prediction_mpa - self.target_mean) / self.target_scale
        else:
            cscan_local = torch.zeros(
                batch, 64, 48, dtype=surface.dtype, device=surface.device
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
        coordinates = self.coordinates.to(surface).unsqueeze(0).expand(batch, -1, -1)
        cells = self.cell(
            torch.cat(
                (
                    surface_local,
                    cscan_local,
                    coordinates,
                    measured.to(surface.dtype).unsqueeze(-1),
                    action_history.to(surface.dtype).unsqueeze(-1),
                    indicator.unsqueeze(-1),
                    confidence.unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        query_features = torch.stack(
            (
                cost,
                remaining_cost,
                prediction,
                available,
                no_reliable,
                measured.to(surface.dtype).mean(dim=1),
            ),
            dim=-1,
        )
        query = self.query(query_features)
        contextual = self.contextualizer(torch.cat((query.unsqueeze(1), cells), dim=1))
        global_context = contextual[:, :1].expand(-1, 64, -1)
        scores = self.action_scorer(
            torch.cat((contextual[:, 1:], global_context), dim=-1)
        ).squeeze(-1)
        return scores, self.value_head(contextual[:, 0]).squeeze(-1)


class MeanFeedbackActor(nn.Module):
    """One-seed V2-style mean-feedback structural diagnostic."""

    def __init__(
        self,
        *,
        width: int = 64,
        target_mean: float = 300.0,
        target_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.use_vlm = True
        self.use_feedback = True
        self.surface = nn.Linear(512, width)
        self.cscan = nn.Linear(512, width)
        self.local = nn.Sequential(
            nn.Linear(2 * width + 6, width), nn.GELU(), nn.Linear(width, width)
        )
        self.score = nn.Sequential(
            nn.Linear(2 * width + 5, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(width + 5, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.register_buffer("coordinates", _coordinates())
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_scale", torch.tensor(float(target_scale)))

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
        batch = _validate_image_state(surface, cscan, measured, cost)
        if (
            action_history.shape != (batch, 64)
            or vlm_indicator.shape != (batch, 64)
            or vlm_confidence.shape != (batch, 64)
            or vlm_available.shape != (batch,)
            or vlm_no_reliable.shape != (batch,)
            or current_prediction_mpa.shape != (batch,)
            or remaining_cost.shape != (batch,)
        ):
            raise ValueError("mean Actor side-channel state is invalid")
        surface_local = self.surface(surface)
        observed = torch.where(measured.unsqueeze(-1), cscan, torch.zeros_like(cscan))
        cscan_local = self.cscan(observed)
        weights = measured.to(surface.dtype).unsqueeze(-1)
        feedback_pool = (cscan_local * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        coordinates = self.coordinates.to(surface).unsqueeze(0).expand(batch, -1, -1)
        local = self.local(
            torch.cat(
                (
                    surface_local,
                    cscan_local,
                    coordinates,
                    measured.to(surface.dtype).unsqueeze(-1),
                    action_history.to(surface.dtype).unsqueeze(-1),
                    vlm_indicator.to(surface.dtype).unsqueeze(-1),
                    vlm_confidence.to(surface.dtype).unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        global_state = torch.cat(
            (
                feedback_pool,
                (
                    (current_prediction_mpa - self.target_mean) / self.target_scale
                ).unsqueeze(-1),
                cost.unsqueeze(-1),
                remaining_cost.unsqueeze(-1),
                vlm_available.to(surface.dtype).unsqueeze(-1),
                vlm_no_reliable.to(surface.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        expanded = global_state.unsqueeze(1).expand(-1, 64, -1)
        scores = self.score(torch.cat((local, expanded), dim=-1)).squeeze(-1)
        return scores, self.value(global_state).squeeze(-1)


class TrueStaticActor(nn.Module):
    """One image-independent shared position ranking, sampled without replacement."""

    def __init__(self) -> None:
        super().__init__()
        self.position_logits = nn.Parameter(torch.zeros(64))

    def forward(
        self, *, batch_size: int, device: torch.device | str | None = None
    ) -> torch.Tensor:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("static Actor batch size is invalid")
        logits = (
            self.position_logits if device is None else self.position_logits.to(device)
        )
        return logits.unsqueeze(0).expand(batch_size, -1)


__all__ = [
    "MeanFeedbackActor",
    "MeanSCPredictor",
    "SpatialCAIActor",
    "SpatialPredictor",
    "TrueStaticActor",
]
