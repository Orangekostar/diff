"""Registered conditional diffusion model for D8 residual targets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from safetensors import SafetensorError
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from .residual_config import ResidualCandidate

_GRID_SIZE = 64
_CHANNELS = 3
_CANDIDATE_SPECS = {
    "RD0": (32, "epsilon", "squared_cosine", False, 0.00, 0.00),
    "RD1": (32, "epsilon", "squared_cosine", False, 0.05, 0.10),
    "RD2": (32, "v_prediction", "squared_cosine", False, 0.05, 0.10),
    "RD3": (32, "sample", "squared_cosine", False, 0.05, 0.10),
    "RD4": (64, "epsilon", "squared_cosine", True, 0.05, 0.10),
    "RD5": (64, "v_prediction", "squared_cosine", True, 0.05, 0.10),
    "RD6": (32, "epsilon", "linear", False, 0.05, 0.10),
    "RD7": (32, "v_prediction", "linear", False, 0.05, 0.10),
}
_PARAMETER_COUNTS = {False: 2_471_747, True: 10_128_515}
_TRAINING_SEEDS = frozenset({20260823, 20260824, 20260825})
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class ResidualModelError(ValueError):
    """Raised when the registered residual-model contract changes."""


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    """Finite scalar components of one registered training loss."""

    total: torch.Tensor
    diffusion: torch.Tensor
    spectral: torch.Tensor
    low_pass: torch.Tensor

    def __post_init__(self) -> None:
        values = (self.total, self.diffusion, self.spectral, self.low_pass)
        if any(
            type(value) is not torch.Tensor
            or value.ndim != 0
            or not bool(torch.isfinite(value.detach()).item())
            for value in values
        ):
            raise ResidualModelError("loss components must be finite scalar tensors")


@dataclass(frozen=True, slots=True)
class ResidualCheckpoint:
    """One immutable response-free residual-model checkpoint."""

    candidate_id: str
    config_sha256: str
    split_sha256: str
    training_seed: int
    model_config: Mapping[str, object]
    architecture_sha256: str
    state_dict_sha256: str
    runtime: Mapping[str, str]
    state_dict: Mapping[str, torch.Tensor]
    scientific_digest: str


def _validate_candidate(candidate: object) -> ResidualCandidate:
    if type(candidate) is not ResidualCandidate:
        raise TypeError("exact ResidualCandidate is required")
    observed = (
        candidate.base_channels,
        candidate.prediction_type,
        candidate.beta_schedule,
        candidate.bottleneck_attention,
        candidate.spectral_weight,
        candidate.low_pass_weight,
    )
    if _CANDIDATE_SPECS.get(candidate.candidate_id) != observed:
        raise ResidualModelError("residual candidate differs from the frozen roster")
    return candidate


def _candidate_from_id(candidate_id: str) -> ResidualCandidate:
    try:
        values = _CANDIDATE_SPECS[candidate_id]
    except KeyError as error:
        raise ResidualModelError("checkpoint candidate is not registered") from error
    return ResidualCandidate(candidate_id, *values)


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ResidualModelError("checkpoint metadata is not canonical") from error


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and set(value) <= _SHA256_CHARACTERS
    )


def _runtime_major_versions() -> dict[str, str]:
    return {
        "python_major": str(sys.version_info.major),
        "torch_major": torch.__version__.split(".", maxsplit=1)[0],
        "diffusers_major": diffusers.__version__.split(".", maxsplit=1)[0],
    }


def _model_config(candidate: ResidualCandidate) -> dict[str, object]:
    base = candidate.base_channels
    return {
        "sample_size": _GRID_SIZE,
        "in_channels": 2 * _CHANNELS,
        "out_channels": _CHANNELS,
        "layers_per_block": 1,
        "block_out_channels": [base, 2 * base, 4 * base],
        "down_block_types": ["DownBlock2D", "DownBlock2D", "DownBlock2D"],
        "up_block_types": ["UpBlock2D", "UpBlock2D", "UpBlock2D"],
        "norm_num_groups": 8,
        "add_attention": candidate.bottleneck_attention,
    }


def _architecture_sha256(candidate: ResidualCandidate) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_spec": list(_CANDIDATE_SPECS[candidate.candidate_id]),
                "model_config": _model_config(candidate),
                "parameter_count": _PARAMETER_COUNTS[
                    candidate.bottleneck_attention
                ],
            }
        )
    ).hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ResidualModelError("checkpoint state dictionary is invalid")
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if (
            type(name) is not str
            or not name
            or type(tensor) is not torch.Tensor
            or tensor.layout != torch.strided
            or tensor.is_sparse
        ):
            raise ResidualModelError("checkpoint tensor entry is invalid")
        value = tensor.detach().to(device="cpu").contiguous()
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ResidualModelError("checkpoint tensor is not finite")
        array = value.numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(
            _canonical({"dtype": array.dtype.str, "shape": list(array.shape)})
            + b"\0"
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_metadata(checkpoint: ResidualCheckpoint) -> dict[str, object]:
    return {
        "schema_version": 1,
        "format": "cpb_d8_residual_checkpoint_v1",
        "candidate_id": checkpoint.candidate_id,
        "config_sha256": checkpoint.config_sha256,
        "split_sha256": checkpoint.split_sha256,
        "training_seed": checkpoint.training_seed,
        "model_config": dict(checkpoint.model_config),
        "architecture_sha256": checkpoint.architecture_sha256,
        "state_dict_sha256": checkpoint.state_dict_sha256,
        "runtime": dict(checkpoint.runtime),
        "scientific_digest": checkpoint.scientific_digest,
    }


def _checkpoint_scientific_digest(metadata: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _canonical(
            {
                key: value
                for key, value in metadata.items()
                if key != "scientific_digest"
            }
        )
    ).hexdigest()


def _validate_checkpoint(checkpoint: object) -> ResidualCheckpoint:
    if type(checkpoint) is not ResidualCheckpoint:
        raise TypeError("exact ResidualCheckpoint is required")
    candidate = _candidate_from_id(checkpoint.candidate_id)
    metadata = _checkpoint_metadata(checkpoint)
    if (
        not _valid_sha256(checkpoint.config_sha256)
        or not _valid_sha256(checkpoint.split_sha256)
        or type(checkpoint.training_seed) is not int
        or checkpoint.training_seed not in _TRAINING_SEEDS
        or dict(checkpoint.model_config) != _model_config(candidate)
        or checkpoint.architecture_sha256 != _architecture_sha256(candidate)
        or checkpoint.state_dict_sha256
        != _state_dict_sha256(checkpoint.state_dict)
        or dict(checkpoint.runtime) != _runtime_major_versions()
        or checkpoint.scientific_digest
        != _checkpoint_scientific_digest(metadata)
    ):
        raise ResidualModelError("residual checkpoint authority changed")
    return checkpoint


def _beta_schedule(candidate: ResidualCandidate) -> str:
    return (
        "squaredcos_cap_v2"
        if candidate.beta_schedule == "squared_cosine"
        else "linear"
    )


def build_residual_unet(candidate: ResidualCandidate) -> UNet2DModel:
    """Build one exact RD0--RD7 conditional U-Net architecture."""

    clean = _validate_candidate(candidate)
    base = clean.base_channels
    model = UNet2DModel(
        sample_size=_GRID_SIZE,
        in_channels=2 * _CHANNELS,
        out_channels=_CHANNELS,
        layers_per_block=1,
        block_out_channels=(base, 2 * base, 4 * base),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
        add_attention=clean.bottleneck_attention,
    )
    expected = _PARAMETER_COUNTS[clean.bottleneck_attention]
    if sum(parameter.numel() for parameter in model.parameters()) != expected:
        raise ResidualModelError("registered residual U-Net parameter count changed")
    return model


def build_train_scheduler(
    candidate: ResidualCandidate, *, train_timesteps: int
) -> DDPMScheduler:
    """Build the candidate's exact forward diffusion scheduler."""

    clean = _validate_candidate(candidate)
    if type(train_timesteps) is not int or train_timesteps != 1000:
        raise ResidualModelError("training timestep count is frozen at 1000")
    return DDPMScheduler(
        num_train_timesteps=train_timesteps,
        beta_schedule=_beta_schedule(clean),
        prediction_type=clean.prediction_type,
        clip_sample=False,
    )


def _diffusion_tensors(
    *,
    prediction: object,
    clean_target: object,
    noise: object,
    noisy: object,
    timesteps: object,
    scheduler: object,
    prediction_type: object,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    DDPMScheduler,
    str,
]:
    fields = (prediction, clean_target, noise, noisy)
    if any(type(value) is not torch.Tensor for value in fields):
        raise ResidualModelError("diffusion fields must be exact tensors")
    predicted, clean, sampled_noise, noisy_field = fields
    assert isinstance(predicted, torch.Tensor)
    assert isinstance(clean, torch.Tensor)
    assert isinstance(sampled_noise, torch.Tensor)
    assert isinstance(noisy_field, torch.Tensor)
    if (
        clean.ndim != 4
        or clean.shape[1] != _CHANNELS
        or any(value.shape != clean.shape for value in fields)
        or any(value.device != clean.device for value in fields)
        or any(value.dtype != clean.dtype for value in fields)
        or not clean.is_floating_point()
        or any(not bool(torch.isfinite(value).all().item()) for value in fields)
        or type(timesteps) is not torch.Tensor
        or timesteps.dtype != torch.int64
        or timesteps.shape != (len(clean),)
        or timesteps.device != clean.device
        or type(scheduler) is not DDPMScheduler
        or type(prediction_type) is not str
        or prediction_type not in {"epsilon", "v_prediction", "sample"}
        or scheduler.config.prediction_type != prediction_type
        or bool(torch.any(timesteps < 0).item())
        or bool(
            torch.any(timesteps >= scheduler.config.num_train_timesteps).item()
        )
    ):
        raise ResidualModelError("diffusion reconstruction inputs are invalid")
    return (
        predicted,
        clean,
        sampled_noise,
        noisy_field,
        timesteps,
        scheduler,
        prediction_type,
    )


def reconstruct_registered_targets(
    *,
    prediction: torch.Tensor,
    clean_target: torch.Tensor,
    noise: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDPMScheduler,
    prediction_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the clean target estimate and registered base-loss target."""

    predicted, clean, sampled_noise, noisy_field, steps, clean_scheduler, kind = (
        _diffusion_tensors(
            prediction=prediction,
            clean_target=clean_target,
            noise=noise,
            noisy=noisy,
            timesteps=timesteps,
            scheduler=scheduler,
            prediction_type=prediction_type,
        )
    )
    alpha = clean_scheduler.alphas_cumprod.to(
        device=clean.device, dtype=clean.dtype
    )[steps]
    alpha = alpha.reshape(len(clean), *((1,) * (clean.ndim - 1)))
    sqrt_alpha = alpha.sqrt()
    sqrt_one_minus = (1.0 - alpha).sqrt()
    if kind == "epsilon":
        clean_prediction = (noisy_field - sqrt_one_minus * predicted) / sqrt_alpha
        base_target = sampled_noise
    elif kind == "v_prediction":
        clean_prediction = sqrt_alpha * noisy_field - sqrt_one_minus * predicted
        base_target = sqrt_alpha * sampled_noise - sqrt_one_minus * clean
    else:
        clean_prediction = predicted
        base_target = clean
    return clean_prediction, base_target


def fft_magnitude_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean L1 distance between orthonormal channel-wise FFT magnitudes."""

    if (
        type(prediction) is not torch.Tensor
        or type(target) is not torch.Tensor
        or prediction.shape != target.shape
        or prediction.ndim != 4
        or not prediction.is_floating_point()
        or prediction.dtype != target.dtype
        or prediction.device != target.device
        or not bool(torch.isfinite(prediction).all().item())
        or not bool(torch.isfinite(target).all().item())
    ):
        raise ResidualModelError("spectral-loss inputs are invalid")
    predicted_fft = torch.fft.fft2(prediction, dim=(-2, -1), norm="ortho").abs()
    target_fft = torch.fft.fft2(target, dim=(-2, -1), norm="ortho").abs()
    return F.l1_loss(predicted_fft, target_fft)


def _gaussian_low_pass(value: torch.Tensor, *, sigma: float) -> torch.Tensor:
    radius = math.ceil(4.0 * sigma)
    coordinates = torch.arange(
        -radius,
        radius + 1,
        dtype=value.dtype,
        device=value.device,
    )
    kernel = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    channels = value.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.reshape(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    filtered = F.conv2d(
        F.pad(value, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=channels,
    )
    return F.conv2d(
        F.pad(filtered, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=channels,
    )


def gaussian_low_pass_l1(
    prediction: torch.Tensor, target: torch.Tensor, *, sigma: float
) -> torch.Tensor:
    """Mean L1 distance after the registered separable Gaussian low pass."""

    if (
        type(sigma) is not float
        or sigma != 2.0
        or type(prediction) is not torch.Tensor
        or type(target) is not torch.Tensor
        or prediction.shape != target.shape
        or prediction.ndim != 4
        or prediction.shape[1] != _CHANNELS
        or min(prediction.shape[-2:]) <= 8
        or not prediction.is_floating_point()
        or prediction.dtype != target.dtype
        or prediction.device != target.device
        or not bool(torch.isfinite(prediction).all().item())
        or not bool(torch.isfinite(target).all().item())
    ):
        raise ResidualModelError("low-pass-loss inputs are invalid")
    return F.l1_loss(
        _gaussian_low_pass(prediction, sigma=sigma),
        _gaussian_low_pass(target, sigma=sigma),
    )


def residual_diffusion_loss(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    clean_target: torch.Tensor,
    stable_condition: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    candidate: ResidualCandidate,
) -> LossBreakdown:
    """Compute the registered response-free conditional diffusion objective."""

    clean = _validate_candidate(candidate)
    if (
        type(model) is not UNet2DModel
        or type(scheduler) is not DDPMScheduler
        or model.config.in_channels != 6
        or model.config.out_channels != 3
        or scheduler.config.prediction_type != clean.prediction_type
        or clean_target.shape != stable_condition.shape
        or clean_target.shape != noise.shape
        or clean_target.shape[1:] != (_CHANNELS, _GRID_SIZE, _GRID_SIZE)
        or clean_target.dtype != torch.float32
        or stable_condition.dtype != clean_target.dtype
        or noise.dtype != clean_target.dtype
        or stable_condition.device != clean_target.device
        or noise.device != clean_target.device
        or not bool(torch.isfinite(clean_target).all().item())
        or not bool(torch.isfinite(stable_condition).all().item())
        or not bool(torch.isfinite(noise).all().item())
        or float(clean_target.detach().min().item()) < -1.0
        or float(clean_target.detach().max().item()) > 1.0
        or float(stable_condition.detach().min().item()) < -1.0
        or float(stable_condition.detach().max().item()) > 1.0
    ):
        raise ResidualModelError("registered loss inputs are invalid")
    noisy = scheduler.add_noise(clean_target, noise, timesteps)
    prediction = model(
        torch.cat((noisy, stable_condition), dim=1), timesteps
    ).sample
    clean_prediction, base_target = reconstruct_registered_targets(
        prediction=prediction,
        clean_target=clean_target,
        noise=noise,
        noisy=noisy,
        timesteps=timesteps,
        scheduler=scheduler,
        prediction_type=clean.prediction_type,
    )
    diffusion = F.mse_loss(prediction, base_target)
    spectral = fft_magnitude_l1(clean_prediction, clean_target)
    low_pass = gaussian_low_pass_l1(
        clean_prediction,
        clean_target,
        sigma=2.0,
    )
    total = (
        diffusion
        + clean.spectral_weight * spectral
        + clean.low_pass_weight * low_pass
    )
    return LossBreakdown(total, diffusion, spectral, low_pass)


def _observed_model_config(model: UNet2DModel) -> dict[str, object]:
    return {
        "sample_size": model.config.sample_size,
        "in_channels": model.config.in_channels,
        "out_channels": model.config.out_channels,
        "layers_per_block": model.config.layers_per_block,
        "block_out_channels": list(model.config.block_out_channels),
        "down_block_types": list(model.config.down_block_types),
        "up_block_types": list(model.config.up_block_types),
        "norm_num_groups": model.config.norm_num_groups,
        "add_attention": model.config.add_attention,
    }


def freeze_residual_checkpoint(
    model: UNet2DModel,
    *,
    candidate: ResidualCandidate,
    config_sha256: str,
    split_sha256: str,
    training_seed: int,
) -> ResidualCheckpoint:
    """Freeze a final model state under one registered split authority."""

    clean = _validate_candidate(candidate)
    if (
        type(model) is not UNet2DModel
        or _observed_model_config(model) != _model_config(clean)
        or not _valid_sha256(config_sha256)
        or not _valid_sha256(split_sha256)
        or type(training_seed) is not int
        or training_seed not in _TRAINING_SEEDS
    ):
        raise ResidualModelError("checkpoint freeze inputs are invalid")
    state = MappingProxyType(
        {
            name: tensor.detach().to(device="cpu").contiguous().clone()
            for name, tensor in model.state_dict().items()
        }
    )
    model_config = MappingProxyType(_model_config(clean))
    runtime = MappingProxyType(_runtime_major_versions())
    values = {
        "schema_version": 1,
        "format": "cpb_d8_residual_checkpoint_v1",
        "candidate_id": clean.candidate_id,
        "config_sha256": config_sha256,
        "split_sha256": split_sha256,
        "training_seed": training_seed,
        "model_config": dict(model_config),
        "architecture_sha256": _architecture_sha256(clean),
        "state_dict_sha256": _state_dict_sha256(state),
        "runtime": dict(runtime),
    }
    checkpoint = ResidualCheckpoint(
        candidate_id=clean.candidate_id,
        config_sha256=config_sha256,
        split_sha256=split_sha256,
        training_seed=training_seed,
        model_config=model_config,
        architecture_sha256=str(values["architecture_sha256"]),
        state_dict_sha256=str(values["state_dict_sha256"]),
        runtime=runtime,
        state_dict=state,
        scientific_digest=hashlib.sha256(_canonical(values)).hexdigest(),
    )
    return _validate_checkpoint(checkpoint)


def _regular_file(path: Path, *, label: str, maximum_bytes: int) -> None:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ResidualModelError(f"{label} is unavailable") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
        or info.st_size > maximum_bytes
    ):
        raise ResidualModelError(f"{label} is not a bounded regular file")


def save_residual_checkpoint(
    prefix: str | Path,
    checkpoint: ResidualCheckpoint,
) -> tuple[Path, Path]:
    """Write deterministic tensor and metadata files without overwriting."""

    clean = _validate_checkpoint(checkpoint)
    base = Path(prefix)
    weights = base.with_suffix(".safetensors")
    metadata = base.with_suffix(".json")
    weights.parent.mkdir(parents=True, exist_ok=True)
    if weights.exists() or metadata.exists():
        raise ResidualModelError("checkpoint destination already exists")
    try:
        save_safetensors(dict(clean.state_dict), str(weights))
        metadata.write_text(
            json.dumps(
                _checkpoint_metadata(clean),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="ascii",
            newline="\n",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ResidualModelError("checkpoint cannot be written") from error
    return weights, metadata


def load_residual_checkpoint(
    weights: str | Path,
    metadata: str | Path,
    *,
    candidate: ResidualCandidate,
    config_sha256: str,
    split_sha256: str,
) -> ResidualCheckpoint:
    """Load and independently validate one registered residual checkpoint."""

    expected_candidate = _validate_candidate(candidate)
    if not _valid_sha256(config_sha256) or not _valid_sha256(split_sha256):
        raise ResidualModelError("expected checkpoint authority is invalid")
    weight_path = Path(weights)
    metadata_path = Path(metadata)
    _regular_file(weight_path, label="checkpoint tensors", maximum_bytes=128 << 20)
    _regular_file(metadata_path, label="checkpoint metadata", maximum_bytes=64 << 10)
    try:
        item = json.loads(metadata_path.read_text(encoding="ascii"))
        state = load_safetensors(str(weight_path), device="cpu")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RuntimeError,
        SafetensorError,
    ) as error:
        raise ResidualModelError("checkpoint cannot be decoded") from error
    required = {
        "schema_version",
        "format",
        "candidate_id",
        "config_sha256",
        "split_sha256",
        "training_seed",
        "model_config",
        "architecture_sha256",
        "state_dict_sha256",
        "runtime",
        "scientific_digest",
    }
    if (
        type(item) is not dict
        or set(item) != required
        or item["schema_version"] != 1
        or item["format"] != "cpb_d8_residual_checkpoint_v1"
        or item["candidate_id"] != expected_candidate.candidate_id
        or item["config_sha256"] != config_sha256
        or item["split_sha256"] != split_sha256
        or item["model_config"] != _model_config(expected_candidate)
        or item["architecture_sha256"]
        != _architecture_sha256(expected_candidate)
        or item["runtime"] != _runtime_major_versions()
    ):
        raise ResidualModelError("checkpoint metadata authority changed")
    try:
        checkpoint = ResidualCheckpoint(
            candidate_id=item["candidate_id"],
            config_sha256=item["config_sha256"],
            split_sha256=item["split_sha256"],
            training_seed=item["training_seed"],
            model_config=MappingProxyType(dict(item["model_config"])),
            architecture_sha256=item["architecture_sha256"],
            state_dict_sha256=item["state_dict_sha256"],
            runtime=MappingProxyType(dict(item["runtime"])),
            state_dict=MappingProxyType(
                {name: tensor.contiguous() for name, tensor in state.items()}
            ),
            scientific_digest=item["scientific_digest"],
        )
        clean = _validate_checkpoint(checkpoint)
        model = build_residual_unet(expected_candidate)
        model.load_state_dict(dict(clean.state_dict), strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ResidualModelError("checkpoint values or tensors changed") from error
    return clean


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _identity(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "\0" in value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
    ):
        raise ResidualModelError(f"{label} is invalid")
    return value


def _is_registered_runtime_device(device: object, *, allow_cpu: bool) -> bool:
    if type(allow_cpu) is not bool or type(device) is not str:
        return False
    if device == "cpu":
        return allow_cpu
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        return False
    if (
        parsed.type != "cuda"
        or str(parsed) != device
        or not torch.cuda.is_available()
    ):
        return False
    return parsed.index is None or 0 <= parsed.index < torch.cuda.device_count()


def _derived_seed(
    checkpoint: ResidualCheckpoint,
    specimen_id: str,
    draw: int,
    step: int,
) -> int:
    payload = (
        f"{checkpoint.training_seed}\0{checkpoint.candidate_id}\0"
        f"{specimen_id}\0{draw}\0{step}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _seeded_noise(
    checkpoint: ResidualCheckpoint,
    specimen_ids: tuple[str, ...],
    *,
    draw: int,
    step: int,
    device: str,
) -> torch.Tensor:
    rows = []
    for specimen_id in specimen_ids:
        generator = torch.Generator(device="cpu").manual_seed(
            _derived_seed(checkpoint, specimen_id, draw, step)
        )
        rows.append(
            torch.randn(
                (_CHANNELS, _GRID_SIZE, _GRID_SIZE),
                generator=generator,
                dtype=torch.float32,
            )
        )
    return torch.stack(rows).to(device)


def _readonly_numpy(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    output = np.frombuffer(
        contiguous.tobytes(order="C"), dtype=np.float32
    ).reshape(contiguous.shape)
    output.setflags(write=False)
    return output


def sample_residual_targets(
    checkpoint: ResidualCheckpoint,
    stable_conditions: object,
    *,
    specimen_ids: tuple[str, ...],
    draws: int,
    steps: int,
    eta: float,
    device: str,
) -> np.ndarray:
    """Draw deterministic-by-identity residual targets from a frozen model."""

    clean_checkpoint = _validate_checkpoint(checkpoint)
    if np.iscomplexobj(stable_conditions):
        raise ResidualModelError("stable conditions must be real")
    try:
        conditions = np.asarray(stable_conditions, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ResidualModelError("stable conditions must be numeric") from error
    if (
        conditions.ndim != 4
        or conditions.shape[1:] != (_CHANNELS, _GRID_SIZE, _GRID_SIZE)
        or len(conditions) == 0
        or not np.all(np.isfinite(conditions))
        or float(np.min(conditions)) < -1.0
        or float(np.max(conditions)) > 1.0
        or type(specimen_ids) is not tuple
        or len(specimen_ids) != len(conditions)
        or len(set(specimen_ids)) != len(specimen_ids)
        or type(draws) is not int
        or not 1 <= draws <= 32
        or type(steps) is not int
        or steps != 25
        or type(eta) is not float
        or eta != 1.0
        or not _is_registered_runtime_device(device, allow_cpu=True)
    ):
        raise ResidualModelError("residual sampling inputs are invalid")
    ids = tuple(_identity(value, label="specimen_id") for value in specimen_ids)
    candidate = _candidate_from_id(clean_checkpoint.candidate_id)
    _configure_determinism(clean_checkpoint.training_seed)
    model = build_residual_unet(candidate)
    try:
        model.load_state_dict(dict(clean_checkpoint.state_dict), strict=True)
    except RuntimeError as error:
        raise ResidualModelError("checkpoint state is incompatible") from error
    model = model.requires_grad_(False).eval().to(device)
    condition_array = np.array(conditions, dtype=np.float32, copy=True, order="C")
    condition = torch.from_numpy(condition_array).to(device)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule=_beta_schedule(candidate),
        prediction_type=candidate.prediction_type,
        clip_sample=True,
    )
    scheduler.set_timesteps(steps, device=device)
    samples: list[np.ndarray] = []
    with torch.inference_mode():
        for draw in range(draws):
            field = _seeded_noise(
                clean_checkpoint,
                ids,
                draw=draw,
                step=-1,
                device=device,
            )
            for position, timestep in enumerate(scheduler.timesteps):
                prediction = model(
                    torch.cat((field, condition), dim=1), timestep
                ).sample
                variance = _seeded_noise(
                    clean_checkpoint,
                    ids,
                    draw=draw,
                    step=position,
                    device=device,
                )
                field = scheduler.step(
                    prediction,
                    timestep,
                    field,
                    eta=eta,
                    variance_noise=variance,
                ).prev_sample
                if not bool(torch.isfinite(field).all().item()):
                    raise ResidualModelError("residual diffusion produced nonfinite data")
            samples.append(
                field.clamp(-1.0, 1.0).to(device="cpu").numpy()
            )
    result = np.stack(samples, axis=1).astype(np.float32, copy=False)
    return _readonly_numpy(result)


__all__ = [
    "LossBreakdown",
    "ResidualCheckpoint",
    "ResidualModelError",
    "build_residual_unet",
    "build_train_scheduler",
    "fft_magnitude_l1",
    "freeze_residual_checkpoint",
    "gaussian_low_pass_l1",
    "load_residual_checkpoint",
    "reconstruct_registered_targets",
    "residual_diffusion_loss",
    "sample_residual_targets",
    "save_residual_checkpoint",
]
