from __future__ import annotations

import numpy as np
import pytest
import torch
from mavis_test_support import synthetic_authority

from cmc_bbdm.mavis.neural_probe.state_encoder import (
    SpatialGridEncoderError,
    SpatialGridMRISStateEncoder,
    spatial_grid_from_tokens,
)
from cmc_bbdm.mavis.reveal import reveal_uniform_scout
from cmc_bbdm.mavis.state_encoder import (
    build_mris_input,
    summarize_mris_input,
)


def _real_summary():
    authority = synthetic_authority()
    state = reveal_uniform_scout(
        authority,
        authority.policy_context("sample-001"),
        initial_budget=0.015625,
        checkpoint=0.25,
    )
    return build_mris_input(state, mode="real"), summarize_mris_input(
        build_mris_input(state, mode="real")
    )


def test_spatial_grid_uses_row_major_token_order_and_mask_channel() -> None:
    tokens = torch.zeros((1, 64, 6), dtype=torch.float32)
    tokens[0, :, 0] = torch.arange(64, dtype=torch.float32)
    tokens[0, 9, 1] = 17.0
    masks = torch.ones((1, 64), dtype=torch.bool)
    masks[0, 9] = False

    grid = spatial_grid_from_tokens(tokens, masks)

    assert grid.shape == (1, 7, 8, 8)
    expected = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    expected[1, 1] = 0.0
    torch.testing.assert_close(grid[0, 0], expected, rtol=0.0, atol=0.0)
    assert grid[0, 1, 1, 1].item() == 0.0
    assert grid[0, 6, 1, 1].item() == 0.0
    assert grid[0, 6, 3, 5].item() == 1.0


def test_spatial_grid_rejects_invalid_legal_state_shapes() -> None:
    with pytest.raises(SpatialGridEncoderError, match="batch is invalid"):
        spatial_grid_from_tokens(
            torch.zeros((2, 63, 6)),
            torch.zeros((2, 63), dtype=torch.bool),
        )
    with pytest.raises(SpatialGridEncoderError, match="batch is invalid"):
        spatial_grid_from_tokens(
            torch.zeros((2, 64, 6)),
            torch.zeros((2, 64), dtype=torch.float32),
        )


def test_spatial_encoder_has_registered_shape_and_parameter_count() -> None:
    encoder = SpatialGridMRISStateEncoder(
        context_dimension=34,
        output_dimension=64,
    )

    assert sum(parameter.numel() for parameter in encoder.parameters()) == 27_552
    assert encoder.architecture_name == "spatial_grid_cnn_v1"
    assert encoder.context_dimension == 34
    assert encoder.output_dimension == 64

    contexts = torch.randn((3, 34))
    tokens = torch.randn((3, 64, 6))
    masks = torch.rand((3, 64)) > 0.25
    costs = torch.rand((3, 3))
    output = encoder.forward_batch(contexts, tokens, masks, costs)

    assert output.shape == (3, 64)
    assert torch.isfinite(output).all()


def test_spatial_encoder_single_state_matches_batched_summary() -> None:
    state_input, summary = _real_summary()
    torch.manual_seed(20260825)
    encoder = SpatialGridMRISStateEncoder(
        context_dimension=34,
        output_dimension=64,
    ).eval()

    with torch.inference_mode():
        single = encoder(state_input)
        batched = encoder.forward_batch(
            torch.tensor(summary.context_features[None], dtype=torch.float32),
            torch.tensor(summary.token_features[None], dtype=torch.float32),
            torch.tensor(summary.token_mask[None], dtype=torch.bool),
            torch.tensor(summary.cost_features[None], dtype=torch.float32),
        )[0]

    assert single.shape == (64,)
    torch.testing.assert_close(single, batched, rtol=0.0, atol=1.0e-6)


def test_spatial_encoder_is_fixed_seed_deterministic_and_differentiable() -> None:
    contexts = torch.randn((4, 34))
    tokens = torch.randn((4, 64, 6))
    masks = torch.ones((4, 64), dtype=torch.bool)
    costs = torch.rand((4, 3))

    torch.manual_seed(20260825)
    first = SpatialGridMRISStateEncoder(
        context_dimension=34,
        output_dimension=64,
    )
    torch.manual_seed(20260825)
    second = SpatialGridMRISStateEncoder(
        context_dimension=34,
        output_dimension=64,
    )

    first_output = first.forward_batch(contexts, tokens, masks, costs)
    second_output = second.forward_batch(contexts, tokens, masks, costs)
    torch.testing.assert_close(first_output, second_output, rtol=0.0, atol=0.0)

    first_output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in first.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in first.parameters())


def test_spatial_encoder_rejects_nonfinite_inputs() -> None:
    encoder = SpatialGridMRISStateEncoder(
        context_dimension=34,
        output_dimension=64,
    )
    contexts = torch.zeros((1, 34))
    contexts[0, 0] = np.nan

    with pytest.raises(SpatialGridEncoderError, match="batch is invalid"):
        encoder.forward_batch(
            contexts,
            torch.zeros((1, 64, 6)),
            torch.zeros((1, 64), dtype=torch.bool),
            torch.zeros((1, 3)),
        )
