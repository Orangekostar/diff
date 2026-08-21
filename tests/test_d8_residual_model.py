from __future__ import annotations

import inspect
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from cmc_bbdm.cpb_diffusion_marginalization.residual_config import (
    ResidualCandidate,
    load_residual_diffusion_config,
)
from cmc_bbdm.cpb_diffusion_marginalization.residual_model import (
    build_residual_unet,
    build_train_scheduler,
    fft_magnitude_l1,
    freeze_residual_checkpoint,
    gaussian_low_pass_l1,
    load_residual_checkpoint,
    reconstruct_registered_targets,
    residual_diffusion_loss,
    sample_residual_targets,
    save_residual_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "paper_v3/configs/d8_residual_diffusion.yaml"


@lru_cache(maxsize=1)
def _config():
    return load_residual_diffusion_config(CONFIG, project_root=PROJECT_ROOT)


@pytest.mark.parametrize("candidate_id", [f"RD{index}" for index in range(8)])
def test_registered_model_has_exact_shape_and_no_response_input(
    candidate_id: str,
) -> None:
    candidate = _config().candidate(candidate_id)
    model = build_residual_unet(candidate)
    assert model.config.sample_size == 64
    assert model.config.in_channels == 6
    assert model.config.out_channels == 3
    assert model.config.layers_per_block == 1
    assert tuple(model.config.block_out_channels) == (
        candidate.base_channels,
        2 * candidate.base_channels,
        4 * candidate.base_channels,
    )
    assert model.config.add_attention is candidate.bottleneck_attention
    expected_parameters = 10_128_515 if candidate.bottleneck_attention else 2_471_747
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        expected_parameters
    )
    assert "response" not in inspect.signature(model.forward).parameters
    assert "response" not in inspect.signature(residual_diffusion_loss).parameters


@pytest.mark.parametrize("candidate_id", [f"RD{index}" for index in range(8)])
def test_registered_scheduler_matches_candidate(candidate_id: str) -> None:
    candidate = _config().candidate(candidate_id)
    scheduler = build_train_scheduler(candidate, train_timesteps=1000)
    expected_schedule = (
        "squaredcos_cap_v2"
        if candidate.beta_schedule == "squared_cosine"
        else "linear"
    )
    assert scheduler.config.num_train_timesteps == 1000
    assert scheduler.config.beta_schedule == expected_schedule
    assert scheduler.config.prediction_type == candidate.prediction_type
    assert scheduler.config.clip_sample is False


@pytest.mark.parametrize(
    "candidate_id",
    ("RD0", "RD2", "RD3"),
)
def test_registered_prediction_type_reconstructs_exact_clean_target(
    candidate_id: str,
) -> None:
    candidate: ResidualCandidate = _config().candidate(candidate_id)
    scheduler = build_train_scheduler(candidate, train_timesteps=1000)
    generator = torch.Generator(device="cpu").manual_seed(20260823)
    clean = torch.rand((2, 3, 8, 8), generator=generator) * 2.0 - 1.0
    noise = torch.randn(clean.shape, generator=generator)
    timesteps = torch.tensor((0, 731), dtype=torch.int64)
    noisy = scheduler.add_noise(clean, noise, timesteps)
    if candidate.prediction_type == "epsilon":
        exact_prediction = noise
    elif candidate.prediction_type == "v_prediction":
        exact_prediction = scheduler.get_velocity(clean, noise, timesteps)
    else:
        exact_prediction = clean

    clean_prediction, base_target = reconstruct_registered_targets(
        prediction=exact_prediction,
        clean_target=clean,
        noise=noise,
        noisy=noisy,
        timesteps=timesteps,
        scheduler=scheduler,
        prediction_type=candidate.prediction_type,
    )

    torch.testing.assert_close(clean_prediction, clean, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(base_target, exact_prediction, rtol=0.0, atol=0.0)


def test_registered_auxiliary_losses_are_zero_only_for_matching_fields() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260823)
    target = torch.rand((2, 3, 16, 16), generator=generator) * 2.0 - 1.0
    changed = target.clone()
    changed[:, :, 5:11, 5:11] += 0.2

    assert fft_magnitude_l1(target, target).item() == 0.0
    assert gaussian_low_pass_l1(target, target, sigma=2.0).item() == 0.0
    assert fft_magnitude_l1(changed, target).item() > 0.0
    assert gaussian_low_pass_l1(changed, target, sigma=2.0).item() > 0.0


def test_registered_loss_is_finite_and_composes_exact_weights() -> None:
    candidate = _config().candidate("RD1")
    model = build_residual_unet(candidate)
    scheduler = build_train_scheduler(candidate, train_timesteps=1000)
    generator = torch.Generator(device="cpu").manual_seed(20260823)
    clean = torch.rand((1, 3, 64, 64), generator=generator) * 2.0 - 1.0
    condition = torch.rand((1, 3, 64, 64), generator=generator) * 2.0 - 1.0
    noise = torch.randn(clean.shape, generator=generator)
    timesteps = torch.tensor((137,), dtype=torch.int64)

    loss = residual_diffusion_loss(
        model,
        scheduler,
        clean,
        condition,
        timesteps,
        noise,
        candidate,
    )

    assert loss.total.ndim == 0
    assert all(
        bool(torch.isfinite(value).item())
        for value in (loss.total, loss.diffusion, loss.spectral, loss.low_pass)
    )
    torch.testing.assert_close(
        loss.total,
        loss.diffusion
        + candidate.spectral_weight * loss.spectral
        + candidate.low_pass_weight * loss.low_pass,
        rtol=0.0,
        atol=0.0,
    )


def _checkpoint(tmp_path: Path):
    config = _config()
    candidate = config.candidate("RD0")
    torch.manual_seed(20260823)
    model = build_residual_unet(candidate)
    checkpoint = freeze_residual_checkpoint(
        model,
        candidate=candidate,
        config_sha256=config.config_sha256,
        split_sha256="a" * 64,
        training_seed=20260823,
    )
    weights, metadata = save_residual_checkpoint(
        tmp_path / "rd0",
        checkpoint,
    )
    loaded = load_residual_checkpoint(
        weights,
        metadata,
        candidate=candidate,
        config_sha256=config.config_sha256,
        split_sha256="a" * 64,
    )
    return candidate, checkpoint, loaded, weights, metadata


def test_checkpoint_roundtrip_and_ddim_draws_are_byte_replayable(
    tmp_path: Path,
) -> None:
    candidate, checkpoint, loaded, _weights, _metadata = _checkpoint(tmp_path)
    assert loaded.state_dict_sha256 == checkpoint.state_dict_sha256
    condition = torch.linspace(
        -1.0,
        1.0,
        3 * 64 * 64,
        dtype=torch.float32,
    ).reshape(1, 3, 64, 64).numpy()
    condition.setflags(write=False)
    kwargs = {
        "specimen_ids": ("specimen-001",),
        "draws": 1,
        "steps": 25,
        "eta": 1.0,
        "device": "cpu",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        first = sample_residual_targets(loaded, condition, **kwargs)
    second = sample_residual_targets(loaded, condition, **kwargs)

    assert first.shape == (1, 1, 3, 64, 64)
    assert first.dtype.name == "float32"
    assert not first.flags.writeable
    assert first.tobytes(order="C") == second.tobytes(order="C")
    assert first.min() >= -1.0
    assert first.max() <= 1.0
    assert loaded.candidate_id == candidate.candidate_id


@pytest.mark.skipif(not torch.cuda.is_available(), reason="registered sampling uses CUDA")
def test_ddim_sampling_accepts_an_indexed_visible_cuda_device(
    tmp_path: Path,
) -> None:
    _candidate, _checkpoint_value, loaded, _weights, _metadata = _checkpoint(
        tmp_path
    )
    sampled = sample_residual_targets(
        loaded,
        np.zeros((1, 3, 64, 64), dtype=np.float32),
        specimen_ids=("specimen-001",),
        draws=1,
        steps=25,
        eta=1.0,
        device="cuda:0",
    )

    assert sampled.shape == (1, 1, 3, 64, 64)


def test_checkpoint_rejects_metadata_and_tensor_tampering(tmp_path: Path) -> None:
    candidate, _checkpoint_value, _loaded, weights, metadata = _checkpoint(tmp_path)
    config = _config()
    original = json.loads(metadata.read_text(encoding="ascii"))
    mutations = (
        ("candidate_id", "RD1"),
        ("config_sha256", "b" * 64),
        ("split_sha256", "b" * 64),
        ("state_dict_sha256", "b" * 64),
        ("architecture_sha256", "b" * 64),
        ("runtime", {**original["runtime"], "torch_major": "999"}),
        (
            "model_config",
            {**original["model_config"], "in_channels": 7},
        ),
    )
    for key, replacement in mutations:
        changed = dict(original)
        changed[key] = replacement
        metadata.write_text(
            json.dumps(changed, sort_keys=True) + "\n",
            encoding="ascii",
        )
        with pytest.raises(ValueError):
            load_residual_checkpoint(
                weights,
                metadata,
                candidate=candidate,
                config_sha256=config.config_sha256,
                split_sha256="a" * 64,
            )
    metadata.write_text(
        json.dumps(original, sort_keys=True) + "\n",
        encoding="ascii",
    )
    tensors = load_safetensors(str(weights), device="cpu")
    first_name = min(tensors)
    tensors[first_name] = tensors[first_name].clone()
    tensors[first_name].reshape(-1)[0] += 1.0
    tampered = tmp_path / "tampered.safetensors"
    save_safetensors(tensors, str(tampered))
    with pytest.raises(ValueError):
        load_residual_checkpoint(
            tampered,
            metadata,
            candidate=candidate,
            config_sha256=config.config_sha256,
            split_sha256="a" * 64,
        )
