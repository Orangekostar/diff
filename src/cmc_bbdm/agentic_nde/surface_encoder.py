"""Frozen RGB ResNet18 features and reproducible P1 surface caches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .contracts import Orientation
from .p1 import P1Config
from .surface_cells import (
    SurfaceCellAuthority,
    crop_rgb_patch,
    oriented_surface_boxes,
    wrong_orientation,
)

_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)
_ARRAY_FILES = MappingProxyType(
    {
        "global": "global_surface_embeddings.npy",
        "local_correct": "local_correct_embeddings.npy",
        "local_wrong_orientation": "local_wrong_orientation_embeddings.npy",
    }
)
_EXECUTION_LOCK = threading.RLock()


class _SurfaceEncoder(Protocol):
    weights_sha256: str
    transform_sha256: str
    device: str
    batch_size: int

    def encode(self, images: Iterable[Image.Image]) -> np.ndarray: ...

    def provenance(self) -> dict[str, object]: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


@contextmanager
def _deterministic_runtime(torch: Any):
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = old_cuda_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def preprocess_surface_rgb(image: Image.Image) -> np.ndarray:
    """Apply the preregistered RGB uint8, resize, and ImageNet transform."""

    if not isinstance(image, Image.Image):
        raise TypeError("surface preprocessing requires a Pillow image")
    image.load()
    rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("surface RGB image shape is invalid")
    try:
        import torch
        from torch.nn import functional
    except Exception as error:  # pragma: no cover - dependency failure
        raise RuntimeError("torch is required for P1 surface preprocessing") from error
    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .to(dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )
    resized = functional.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    array = np.asarray(resized.numpy(), dtype=np.float32)
    output = np.ascontiguousarray((array - _MEAN) / _STD, dtype="<f4")
    if output.shape != (3, 224, 224) or not np.all(np.isfinite(output)):
        raise ValueError("surface preprocessing output is invalid")
    return output


def _model_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenSurfaceResNet18:
    model: Any
    torch: Any
    torchvision: Any
    weight_path: Path
    weights_sha256: str
    transform_sha256: str
    device: str
    batch_size: int
    model_state_sha256: str

    def validate(self) -> None:
        if (
            self.weight_path.is_symlink()
            or not self.weight_path.is_file()
            or _sha256_file(self.weight_path) != self.weights_sha256
            or self.model.training
            or any(parameter.requires_grad for parameter in self.model.parameters())
            or _model_state_sha256(self.model) != self.model_state_sha256
        ):
            raise ValueError("frozen P1 surface encoder changed")

    def encode(self, images: Iterable[Image.Image]) -> np.ndarray:
        values = tuple(images)
        if not values:
            empty = np.empty((0, 512), dtype="<f4")
            empty.setflags(write=False)
            return empty
        tensors = tuple(
            self.torch.from_numpy(preprocess_surface_rgb(image)) for image in values
        )
        outputs: list[np.ndarray] = []
        with _EXECUTION_LOCK, _deterministic_runtime(self.torch):
            if self.model.training or any(
                parameter.requires_grad for parameter in self.model.parameters()
            ):
                raise ValueError("P1 surface encoder is not frozen")
            with self.torch.inference_mode():
                for start in range(0, len(tensors), self.batch_size):
                    batch = self.torch.stack(
                        tensors[start : start + self.batch_size]
                    ).to(self.device, non_blocking=False)
                    encoded = self.model(batch).detach().to("cpu").numpy()
                    outputs.append(np.asarray(encoded, dtype="<f4"))
        combined = np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype="<f4")
        if combined.shape != (len(values), 512) or not np.all(np.isfinite(combined)):
            raise ValueError("P1 surface encoder output is invalid")
        snapshot = np.frombuffer(combined.tobytes(order="C"), dtype="<f4").reshape(
            combined.shape
        )
        snapshot.setflags(write=False)
        return snapshot

    def provenance(self) -> dict[str, object]:
        return {
            "architecture": "torchvision_resnet18_post_avgpool_pre_fc",
            "batch_size": self.batch_size,
            "device": self.device,
            "model_state_sha256": self.model_state_sha256,
            "output_dimension": 512,
            "torch": self.torch.__version__,
            "torchvision": self.torchvision.__version__,
            "transform_sha256": self.transform_sha256,
            "weights": "ImageNet1K_V1",
            "weights_sha256": self.weights_sha256,
        }


def build_surface_resnet18(config: P1Config) -> FrozenSurfaceResNet18:
    """Load the sole preregistered frozen RGB encoder from local weights."""

    if type(config) is not P1Config:
        raise TypeError("issued P1Config is required")
    try:
        import torch
        import torchvision
        from torch import nn
        from torchvision.models import resnet18
    except Exception as error:  # pragma: no cover - dependency failure
        raise RuntimeError("torch and torchvision are required for P1") from error
    surface = config.raw["surface_features"]
    runtime = surface["formal_runtime"]
    device = str(runtime["device"])
    batch_size = int(runtime["batch_size"])
    if (
        torch.__version__ != runtime["torch"]
        or torchvision.__version__ != runtime["torchvision"]
        or device != "cuda:0"
        or not torch.cuda.is_available()
        or batch_size != 128
    ):
        raise ValueError("formal P1 surface runtime changed")
    weight_path = config.project_root / config.sources["resnet18_weights"].path
    if _sha256_file(weight_path) != config.encoder_weight_sha256:
        raise ValueError("P1 surface weights changed")
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model = resnet18(weights=None)
        model.load_state_dict(state, strict=True)
        model.fc = nn.Identity()
        model.requires_grad_(False)
        model.eval()
        model.to(device)
    except Exception as error:
        raise ValueError("P1 surface ResNet18 cannot be constructed") from error
    encoder = FrozenSurfaceResNet18(
        model=model,
        torch=torch,
        torchvision=torchvision,
        weight_path=weight_path,
        weights_sha256=config.encoder_weight_sha256,
        transform_sha256=config.surface_transform_sha256,
        device=device,
        batch_size=batch_size,
        model_state_sha256=_model_state_sha256(model),
    )
    encoder.validate()
    return encoder


@dataclass(frozen=True, slots=True)
class SurfaceFeatureBank:
    root: Path
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    wrong_orientations: tuple[str, ...]
    global_embeddings: np.ndarray
    local_correct_embeddings: np.ndarray
    local_wrong_orientation_embeddings: np.ndarray
    array_sha256: Mapping[str, str]
    encoder_provenance: Mapping[str, object]
    authority_state_sha256: str
    transform_sha256: str
    state_sha256: str
    manifest_sha256: str


def _resolve_surface(root: Path, relative: Path, expected_sha256: str) -> Path:
    unresolved = root / relative
    try:
        path = unresolved.resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError("P1 surface source is unavailable") from error
    if (
        unresolved.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != expected_sha256
    ):
        raise ValueError("P1 surface source changed")
    return path


def _manifest_state(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def materialize_surface_feature_bank(
    authority: SurfaceCellAuthority,
    *,
    external_root: str | Path,
    output: str | Path,
    encoder: _SurfaceEncoder,
    wrong_orientation_seed: str,
    notify: Callable[[str], None] | None = None,
) -> SurfaceFeatureBank:
    """Create global/correct/wrong features from P0R surfaces without labels."""

    if (
        type(authority) is not SurfaceCellAuthority
        or authority.specimen_count < 1
        or len(authority.records) != authority.specimen_count
        or not wrong_orientation_seed
        or not hasattr(encoder, "encode")
        or not hasattr(encoder, "provenance")
        or len(encoder.transform_sha256) != 64
        or len(encoder.weights_sha256) != 64
    ):
        raise ValueError("P1 surface feature inputs are invalid")
    external = Path(external_root).resolve(strict=True)
    if not external.is_dir():
        raise ValueError("P1 external surface root is invalid")
    destination = Path(output).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("P1 surface feature output already exists")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    count = authority.specimen_count
    shapes = {
        "global": (count, 512),
        "local_correct": (count, 64, 512),
        "local_wrong_orientation": (count, 64, 512),
    }
    wrong_values: list[str] = []
    try:
        arrays = {
            name: np.lib.format.open_memmap(
                staging / filename, mode="w+", dtype="<f4", shape=shapes[name]
            )
            for name, filename in _ARRAY_FILES.items()
        }
        for index, record in enumerate(authority.records):
            path = _resolve_surface(
                external, record.surface_path, record.surface_sha256
            )
            try:
                with Image.open(path) as opened:
                    opened.load()
                    rgb = opened.convert("RGB")
            except OSError as error:
                raise ValueError("P1 surface image cannot be decoded") from error
            if rgb.size != (record.source.width_px, record.source.height_px):
                raise ValueError("P1 surface frame changed")
            selected_wrong = wrong_orientation(
                record.specimen_id,
                dataset_id=record.dataset_id,
                seed=wrong_orientation_seed,
            )
            wrong_values.append(selected_wrong.value)
            wrong_boxes = oriented_surface_boxes(record, selected_wrong)
            correct_patches = tuple(
                crop_rgb_patch(rgb, tuple(box)) for box in record.cell_boxes
            )
            wrong_patches = tuple(
                crop_rgb_patch(rgb, tuple(box)) for box in wrong_boxes
            )
            encoded = np.asarray(
                encoder.encode((rgb, *correct_patches, *wrong_patches)), dtype="<f4"
            )
            if encoded.shape != (129, 512) or not np.all(np.isfinite(encoded)):
                raise ValueError("P1 surface feature batch changed")
            arrays["global"][index] = encoded[0]
            arrays["local_correct"][index] = encoded[1:65]
            arrays["local_wrong_orientation"][index] = encoded[65:129]
            if notify is not None:
                notify(f"P1 surface features {index + 1}/{count}")
        for array in arrays.values():
            array.flush()
        del arrays
        if hasattr(encoder, "validate"):
            encoder.validate()  # type: ignore[attr-defined]
        array_hashes = {
            name: _sha256_file(staging / filename)
            for name, filename in _ARRAY_FILES.items()
        }
        base: dict[str, object] = {
            "arrays": {
                name: {
                    "dtype": "float32",
                    "filename": _ARRAY_FILES[name],
                    "sha256": array_hashes[name],
                    "shape": list(shapes[name]),
                }
                for name in _ARRAY_FILES
            },
            "authority_state_sha256": authority.state_sha256,
            "dataset_ids": list(authority.dataset_ids),
            "encoder": encoder.provenance(),
            "schema_version": 1,
            "specimen_ids": list(authority.specimen_ids),
            "surface_sha256": list(authority.surface_sha256),
            "transform_sha256": encoder.transform_sha256,
            "wrong_orientation_seed": wrong_orientation_seed,
            "wrong_orientations": wrong_values,
        }
        manifest = {**base, "state_sha256": _manifest_state(base)}
        (staging / "manifest.json").write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_surface_feature_bank(
        destination,
        authority=authority,
        expected_transform_sha256=encoder.transform_sha256,
    )


def load_surface_feature_bank(
    root: str | Path,
    *,
    authority: SurfaceCellAuthority,
    expected_transform_sha256: str,
) -> SurfaceFeatureBank:
    """Load a cache only when its roster, arrays, and state replay exactly."""

    source = Path(root).resolve(strict=True)
    manifest_path = source / "manifest.json"
    if (
        type(authority) is not SurfaceCellAuthority
        or not source.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or len(expected_transform_sha256) != 64
    ):
        raise ValueError("P1 surface feature cache identity is invalid")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("P1 surface feature manifest cannot be read") from error
    if not isinstance(manifest, dict):
        raise TypeError("P1 surface feature manifest is invalid")
    state = manifest.pop("state_sha256", None)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("authority_state_sha256") != authority.state_sha256
        or tuple(manifest.get("specimen_ids", ())) != authority.specimen_ids
        or tuple(manifest.get("dataset_ids", ())) != authority.dataset_ids
        or tuple(manifest.get("surface_sha256", ())) != authority.surface_sha256
        or manifest.get("transform_sha256") != expected_transform_sha256
        or not isinstance(state, str)
        or state != _manifest_state(manifest)
    ):
        raise ValueError("P1 surface feature manifest changed")
    raw_arrays = manifest.get("arrays")
    if not isinstance(raw_arrays, dict) or set(raw_arrays) != set(_ARRAY_FILES):
        raise ValueError("P1 surface feature array registry changed")
    count = authority.specimen_count
    expected_shapes = {
        "global": (count, 512),
        "local_correct": (count, 64, 512),
        "local_wrong_orientation": (count, 64, 512),
    }
    arrays: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for name, filename in _ARRAY_FILES.items():
        entry = raw_arrays.get(name)
        if (
            not isinstance(entry, dict)
            or entry.get("filename") != filename
            or entry.get("dtype") != "float32"
            or tuple(entry.get("shape", ())) != expected_shapes[name]
            or not isinstance(entry.get("sha256"), str)
        ):
            raise ValueError("P1 surface feature array binding changed")
        path = source / filename
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != entry["sha256"]:
            raise ValueError("P1 surface feature array changed")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError("P1 surface feature array cannot be loaded") from error
        if (
            array.shape != expected_shapes[name]
            or array.dtype != np.dtype("<f4")
            or not np.all(np.isfinite(array))
        ):
            raise ValueError("P1 surface feature array values changed")
        arrays[name] = array
        hashes[name] = str(entry["sha256"])
    encoder = manifest.get("encoder")
    wrong = tuple(manifest.get("wrong_orientations", ()))
    if (
        not isinstance(encoder, dict)
        or len(wrong) != count
        or any(value == Orientation.ROT90.value for value in wrong)
    ):
        raise ValueError("P1 surface feature control changed")
    return SurfaceFeatureBank(
        root=source,
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        wrong_orientations=wrong,
        global_embeddings=arrays["global"],
        local_correct_embeddings=arrays["local_correct"],
        local_wrong_orientation_embeddings=arrays["local_wrong_orientation"],
        array_sha256=MappingProxyType(hashes),
        encoder_provenance=MappingProxyType(encoder),
        authority_state_sha256=authority.state_sha256,
        transform_sha256=expected_transform_sha256,
        state_sha256=state,
        manifest_sha256=_sha256_file(manifest_path),
    )


__all__ = [
    "FrozenSurfaceResNet18",
    "SurfaceFeatureBank",
    "build_surface_resnet18",
    "load_surface_feature_bank",
    "materialize_surface_feature_bank",
    "preprocess_surface_rgb",
]
