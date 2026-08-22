"""Hash-bound full-field RGB embeddings and fold-local PCA utilities.

The encoder in this module is deliberately stateless: image preprocessing and the
published ResNet-18 checkpoint are fixed before any fold is fitted.  Labels and
model-fit state are not accepted by the encoder API.
"""

from __future__ import annotations

import dis
import fcntl
import inspect
import json
import os
import secrets
import stat
import tempfile
import threading
import time
import types
import warnings
import weakref
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field
from functools import wraps
from hashlib import sha256
from importlib import metadata as importlib_metadata
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from PIL import Image

from .data import DOMAIN_ORDER as _DATA_DOMAIN_ORDER
from .data import PRIMARY_DATASET_IDS as _DATA_PRIMARY_DATASET_IDS
from .data import PRIMARY_SPECIMEN_IDS as _DATA_PRIMARY_SPECIMEN_IDS
from .data import V3Data as _DATA_V3_DATA
from .data import V3DataView as _DATA_V3_DATA_VIEW
from .data import validate_issued_data_authority as _DATA_AUTHORITY_VALIDATOR


class FeatureValidationError(ValueError):
    """Raised when an embedding, checkpoint, cache, or PCA input is invalid."""


EMBEDDING_DIMENSION = 512
EMBEDDING_BATCH_SIZE = 32
PCA_DIMENSIONS = (8, 16, 32)
_PCA_SAFE_ABS = np.finfo(np.float64).max ** 0.25 / 8.0
_FROZEN_RESNET18_WEIGHTS_FILENAME = "resnet18-f37072fd.pth"
_FROZEN_RESNET18_WEIGHTS_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
_FROZEN_RESNET18_WEIGHTS_URL = (
    "https://download.pytorch.org/models/resnet18-f37072fd.pth"
)
_FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH = Path("paper_v3/assets/resnet18-f37072fd.pth")
RESNET18_WEIGHTS_FILENAME = _FROZEN_RESNET18_WEIGHTS_FILENAME
RESNET18_WEIGHTS_SHA256 = _FROZEN_RESNET18_WEIGHTS_SHA256
RESNET18_WEIGHTS_URL = _FROZEN_RESNET18_WEIGHTS_URL
RESNET18_WEIGHTS_RELATIVE_PATH = _FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH
_MODULE_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPECTED_MODEL_TYPE = "torchvision.models.resnet.ResNet"
_EXPECTED_MODEL_STATE_SHA256 = (
    "cbf2e5609c31f76c589acd162bf0bd55e89d8d297c84d2c3c3334dcea0be8b50"
)
_EXPECTED_ARCHITECTURE_SHA256 = (
    "118174aa09d19fc2d298a6065dc6a20b54b70a2925068377db1759df046d8aef"
)
_EXPECTED_MODEL_CLASS_EXECUTION_SHA256 = (
    "354ba46371afef162c27653ac811b0c7534357ea6e3d59141a674f4fa00fb2fd"
)
_EXPECTED_TORCH_VERSION = "2.12.1+cu130"
_EXPECTED_TORCHVISION_VERSION = "0.27.1+cu130"
_EXPECTED_TORCH_DISTRIBUTION_VERSION = "2.12.1"
_EXPECTED_TORCHVISION_DISTRIBUTION_VERSION = "0.27.1"
_EXPECTED_TORCH_CUDA_VERSION = "13.0"
_NPZ_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_MAX_EMBEDDING_SAMPLES = 276
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_PIXELS = 50_000_000
_MAX_WEIGHT_BYTES = 64 * 1024 * 1024
_MAX_CACHE_BYTES = 16 * 1024 * 1024
_MAX_PROVENANCE_BYTES = 4 * 1024 * 1024
_MAX_NPZ_UNCOMPRESSED_BYTES = 2 * 1024 * 1024
_MAX_NPZ_MEMBER_BYTES = 1024 * 1024
_MAX_NPZ_COMPRESSION_RATIO = 1_000
_TORCH_RUNTIME_LOCK_PATH = Path(tempfile.gettempdir()) / (
    f"cmc-bbdm-cpb-v3-torch-runtime-{os.getuid()}.lock"
)
_TORCH_GLOBAL_STATE_LOCK = threading.RLock()
_ENCODER_EXECUTION_LOCK = threading.RLock()
_PAIR_PUBLISH_THREAD_LOCK = threading.RLock()

_FROZEN_LUMINANCE_COEFFICIENTS = (0.299, 0.587, 0.114)
_FROZEN_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_FROZEN_IMAGENET_STD = (0.229, 0.224, 0.225)
LUMINANCE_COEFFICIENTS = _FROZEN_LUMINANCE_COEFFICIENTS
IMAGENET_MEAN = _FROZEN_IMAGENET_MEAN
IMAGENET_STD = _FROZEN_IMAGENET_STD


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            frozen[str(key)] = _freeze_mapping(item)
        elif isinstance(item, list):
            frozen[str(key)] = tuple(item)
        else:
            frozen[str(key)] = item
    return MappingProxyType(frozen)


_TRANSFORM_SPEC_DATA: Mapping[str, Any] = {
    "align_corners": False,
    "antialias": True,
    "channels": "grayscale_replicated_to_rgb",
    "dtype": "float32",
    "imagenet_mean": _FROZEN_IMAGENET_MEAN,
    "imagenet_std": _FROZEN_IMAGENET_STD,
    "interpolation": "bilinear",
    "luminance_coefficients": _FROZEN_LUMINANCE_COEFFICIENTS,
    "normalization": "imagenet",
    "output_size": [224, 224],
    "replicate_channels": 3,
    "rgb_divisor": 255,
    "version": 1,
}
_FROZEN_TRANSFORM_SPEC = _freeze_mapping(_TRANSFORM_SPEC_DATA)
TRANSFORM_SPEC = _FROZEN_TRANSFORM_SPEC
_EXPECTED_TRANSFORM_SHA = (
    "4cb561ea8d1466a2c98b2b717843c19a3e7888ba6ffc715a503751226c41e3a7"
)
EXPECTED_TRANSFORM_SHA = _EXPECTED_TRANSFORM_SHA
TRANSFORM_SHA256 = _EXPECTED_TRANSFORM_SHA


def validate_transform_spec() -> bool:
    """Validate the immutable transform registry and its independent digest."""

    try:
        if _canonical_json(TRANSFORM_SPEC) != _canonical_json(_FROZEN_TRANSFORM_SPEC):
            raise FeatureValidationError("transform specification drift detected")
        if _sha256_bytes(_canonical_json(TRANSFORM_SPEC)) != _EXPECTED_TRANSFORM_SHA:
            raise FeatureValidationError("transform specification hash mismatch")
        if (
            TRANSFORM_SHA256 != _EXPECTED_TRANSFORM_SHA
            or EXPECTED_TRANSFORM_SHA != _EXPECTED_TRANSFORM_SHA
        ):
            raise FeatureValidationError("transform hash constant drift detected")
        if tuple(LUMINANCE_COEFFICIENTS) != _FROZEN_LUMINANCE_COEFFICIENTS:
            raise FeatureValidationError("transform luminance constants drift detected")
        if (
            tuple(IMAGENET_MEAN) != _FROZEN_IMAGENET_MEAN
            or tuple(IMAGENET_STD) != _FROZEN_IMAGENET_STD
        ):
            raise FeatureValidationError(
                "transform normalization constants drift detected"
            )
    except FeatureValidationError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise FeatureValidationError(
            "transform specification drift detected"
        ) from error
    return True


def _validate_transform_contract() -> None:
    validate_transform_spec()


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a regular local file."""

    digest = sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FeatureValidationError(f"cannot read local file: {path}") from error
    return digest.hexdigest()


def _readonly_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _immutable_array(
    value: Any, *, dtype: Any | None = None
) -> tuple[np.ndarray, bytes]:
    """Build an array backed by immutable bytes, then return its backing bytes."""

    source = np.array(value, dtype=dtype, copy=True, order="C")
    if source.dtype.hasobject:
        raise FeatureValidationError("object arrays cannot be immutable-backed")
    backing = source.tobytes(order="C")
    result = np.frombuffer(backing, dtype=source.dtype).reshape(source.shape)
    return result, backing


def _strict_real_numeric_array(value: Any, name: str) -> np.ndarray:
    """Convert only native real numeric arrays, rejecting lossy coercion."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise FeatureValidationError(f"{name} must be a real numeric array") from error
    if raw.dtype.kind == "c":
        raise FeatureValidationError(f"{name} cannot contain complex values")
    if raw.dtype.kind in {"b", "O", "S", "U"} or not np.issubdtype(
        raw.dtype, np.number
    ):
        raise FeatureValidationError(f"{name} must be a real numeric array")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(over="raise", invalid="raise"):
                result = np.asarray(raw, dtype=np.float64)
    except (FloatingPointError, TypeError, ValueError, OverflowError, Warning) as error:
        raise FeatureValidationError(f"{name} must be a real numeric array") from error
    if not np.all(np.isfinite(result)):
        raise FeatureValidationError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _state_hash(*parts: Any) -> str:
    digest = sha256()
    for index, part in enumerate(parts):
        digest.update(f"part:{index}:".encode("ascii"))
        if isinstance(part, np.ndarray):
            array = np.ascontiguousarray(part)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(_canonical_json(list(array.shape)))
            digest.update(array.tobytes(order="C"))
        else:
            digest.update(_canonical_json(part))
    return digest.hexdigest()


def _as_rgb_float(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        value = np.asarray(image.convert("RGB"))
    else:
        value = np.asarray(image)
    if value.dtype.kind == "c":
        raise FeatureValidationError("image cannot contain complex values")
    if value.dtype.kind in {"f", "i", "u"} and not np.all(np.isfinite(value)):
        raise FeatureValidationError("image must contain only finite values")
    if value.dtype != np.dtype(np.uint8):
        raise FeatureValidationError(
            "image dtype must be uint8 for the registered uint8_div255 scale"
        )
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=-1)
    elif value.ndim == 3 and value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    elif value.ndim != 3 or value.shape[-1] != 3:
        raise FeatureValidationError("image must have shape (height, width, 3)")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise FeatureValidationError("image dimensions must be positive")
    return value.astype(np.float32, copy=False) / np.float32(255.0)


def _strict_sample_ids(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise FeatureValidationError(f"{name} must be a sequence of IDs")
    try:
        result = tuple(value)
    except TypeError as error:
        raise FeatureValidationError(f"{name} must be a sequence of IDs") from error
    if not result or any(type(item) is not str or not item for item in result):
        raise FeatureValidationError(f"{name} must contain nonempty string IDs")
    if len(set(result)) != len(result):
        raise FeatureValidationError(f"{name} must contain unique IDs")
    return result


def to_grayscale_luminance(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a full-field image with the registered luminance coefficients."""

    _validate_transform_contract()
    rgb = _as_rgb_float(image)
    coefficients = np.asarray(_FROZEN_LUMINANCE_COEFFICIENTS, dtype=np.float32)
    grayscale = (
        rgb[..., 0] * coefficients[0]
        + rgb[..., 1] * coefficients[1]
        + rgb[..., 2] * coefficients[2]
    )
    grayscale = np.asarray(grayscale, dtype=np.float32)
    if not np.all(np.isfinite(grayscale)):
        raise FeatureValidationError("grayscale image must contain only finite values")
    return grayscale


def resize_grayscale_bilinear(grayscale: np.ndarray, *, size: int = 224) -> np.ndarray:
    """Resize the entire grayscale field with the frozen torch bilinear rule."""

    _validate_transform_contract()
    value = _strict_real_numeric_array(grayscale, "grayscale image")
    if value.ndim != 2:
        raise FeatureValidationError("grayscale image must be two-dimensional")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise FeatureValidationError("grayscale image dimensions must be positive")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise FeatureValidationError("resize size must be a positive integer")
    if np.any(np.abs(value) > np.finfo(np.float32).max):
        raise FeatureValidationError("grayscale image exceeds float32 range")
    try:
        import torch
        from torch.nn import functional
    except Exception as error:  # pragma: no cover - dependency failure
        raise FeatureValidationError(
            "torch is required for registered bilinear resize"
        ) from error
    tensor = torch.from_numpy(
        np.ascontiguousarray(value.astype(np.float32, copy=False))
    ).reshape(1, 1, value.shape[0], value.shape[1])
    output = (
        functional.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, 0]
        .numpy()
        .astype(np.float32, copy=True)
    )
    if not np.all(np.isfinite(output)):
        raise FeatureValidationError("resized image must contain only finite values")
    return output


def normalize_imagenet_grayscale(grayscale: np.ndarray) -> np.ndarray:
    """Replicate grayscale to three channels and apply ImageNet normalization."""

    _validate_transform_contract()
    value = _strict_real_numeric_array(grayscale, "grayscale image")
    if value.ndim != 2:
        raise FeatureValidationError(
            "grayscale image must be a finite two-dimensional array"
        )
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise FeatureValidationError("grayscale image dimensions must be positive")
    if np.any(np.abs(value) > np.finfo(np.float32).max):
        raise FeatureValidationError("grayscale image exceeds float32 range")
    value = value.astype(np.float32, copy=False)
    output = np.stack(
        [
            (value - np.float32(mean)) / np.float32(std)
            for mean, std in zip(
                _FROZEN_IMAGENET_MEAN, _FROZEN_IMAGENET_STD, strict=True
            )
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(output)):
        raise FeatureValidationError("normalized image must contain only finite values")
    return output


def preprocess_full_field(
    image: Image.Image | np.ndarray,
    *,
    size: int = 224,
) -> np.ndarray:
    """Apply the fixed grayscale, full-field bilinear, and ImageNet transform."""

    return normalize_imagenet_grayscale(
        resize_grayscale_bilinear(to_grayscale_luminance(image), size=size)
    )


# Explicit aliases keep the transform vocabulary stable for downstream fold code.
grayscale_luminance = to_grayscale_luminance
preprocess_image = preprocess_full_field


def _validate_weight_path(
    weight_path: Path | str | None,
    project_root: Path | str | None,
) -> tuple[Path, Path, bytes]:
    if weight_path is None or project_root is None:
        raise FeatureValidationError(
            "explicit project_root and registered relative weight_path are required"
        )
    raw = Path(weight_path)
    root = Path(project_root).resolve()
    if root != _MODULE_REPO_ROOT:
        raise FeatureValidationError(
            "ResNet18 project_root must be the repository containing this module"
        )
    if raw.is_absolute() or raw != _FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH:
        raise FeatureValidationError(
            "ResNet18 weight_path must be the registered repo-relative path"
        )
    candidate = root / _FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH
    if candidate.is_symlink() or candidate.resolve() != candidate:
        raise FeatureValidationError("ResNet18 weight_path must not be a symlink")
    if not candidate.is_file():
        raise FeatureValidationError(
            f"registered ResNet18 weight file is missing: {candidate}"
        )
    snapshot = _read_file_snapshot(
        candidate, "registered ResNet18 weights", max_bytes=_MAX_WEIGHT_BYTES
    )
    actual = _sha256_bytes(snapshot)
    if actual != _FROZEN_RESNET18_WEIGHTS_SHA256:
        raise FeatureValidationError(
            "ResNet18 weight hash mismatch: "
            f"expected {_FROZEN_RESNET18_WEIGHTS_SHA256}, got {actual}"
        )
    return candidate, root, snapshot


def _load_frozen_model(weight_snapshot: bytes, device: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        import torchvision
        from torchvision.models import resnet18
    except Exception as error:  # pragma: no cover - dependency failure
        raise FeatureValidationError(
            "torch and torchvision are required for ResNet18 embeddings"
        ) from error
    _validate_runtime_versions(torch, torchvision)
    try:
        model = resnet18(weights=None)
        _validate_registered_model_class_execution(model)
        state = torch.load(
            BytesIO(weight_snapshot), map_location="cpu", weights_only=True
        )
        if not isinstance(state, Mapping):
            raise FeatureValidationError(
                "ResNet18 checkpoint must contain a state dictionary"
            )
        model.load_state_dict(state, strict=True)
    except FeatureValidationError:
        raise
    except Exception as error:
        raise FeatureValidationError(
            "cannot load the hash-bound ResNet18 checkpoint"
        ) from error
    model.fc = torch.nn.Identity()
    model.requires_grad_(False)
    model.eval()
    try:
        model.to(device)
    except Exception as error:
        raise FeatureValidationError(f"invalid encoder device: {device}") from error
    return model, torch, torchvision


def _load_image_item(
    item: Image.Image | np.ndarray | Path | str,
) -> Image.Image | np.ndarray:
    if isinstance(item, (str, Path)):
        path = Path(item)
        payload = _read_file_snapshot(
            path, "registered image", max_bytes=_MAX_IMAGE_BYTES
        )
        return _decode_image_snapshot(payload, "registered image")
    if isinstance(item, Image.Image):
        if item.width * item.height > _MAX_IMAGE_PIXELS:
            raise FeatureValidationError("registered image exceeds the pixel limit")
        return item
    if isinstance(item, np.ndarray):
        if (
            item.ndim < 2
            or item.shape[0] < 1
            or item.shape[1] < 1
            or item.shape[0] * item.shape[1] > _MAX_IMAGE_PIXELS
        ):
            raise FeatureValidationError("registered image exceeds the pixel limit")
        return item
    raise FeatureValidationError(
        "encoder inputs must be image arrays, PIL images, or local paths"
    )


def _iter_model_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    result: list[tuple[str, Any]] = []
    active: set[int] = set()

    def visit(name: str, module: Any) -> None:
        identity = id(module)
        if identity in active:
            raise FeatureValidationError("encoder module graph contains a cycle")
        active.add(identity)
        result.append((name, module))
        try:
            children = vars(module).get("_modules")
            if not isinstance(children, Mapping):
                raise FeatureValidationError("encoder module registry is invalid")
            for child_name, child in sorted(children.items()):
                if child is not None:
                    qualified = f"{name}.{child_name}" if name else str(child_name)
                    visit(qualified, child)
        finally:
            active.remove(identity)

    visit("", model)
    return tuple(result)


def _model_state_sha256(model: Any) -> str:
    digest = sha256()
    try:
        state: dict[str, Any] = {}
        for module_name, module in _iter_model_modules(model):
            prefix = f"{module_name}." if module_name else ""
            parameters = vars(module).get("_parameters")
            buffers = vars(module).get("_buffers")
            nonpersistent = vars(module).get("_non_persistent_buffers_set", set())
            if not isinstance(parameters, Mapping) or not isinstance(buffers, Mapping):
                raise FeatureValidationError("encoder model state registry is invalid")
            for local_name, tensor in parameters.items():
                if tensor is not None:
                    state[f"{prefix}{local_name}"] = tensor
            for local_name, tensor in buffers.items():
                if tensor is not None and local_name not in nonpersistent:
                    state[f"{prefix}{local_name}"] = tensor
        for name in sorted(state):
            tensor = state[name].detach().to("cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(_canonical_json(list(tensor.shape)))
            digest.update(tensor.numpy().tobytes(order="C"))
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise FeatureValidationError("cannot hash encoder model state") from error
    return digest.hexdigest()


_GRAPH_INTERNAL_ATTRS = frozenset(
    {
        "training",
        "_parameters",
        "_buffers",
        "_non_persistent_buffers_set",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_modules",
    }
)


def _architecture_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return {"type": "path", "value": value.as_posix()}
    if isinstance(value, Mapping):
        return {
            str(key): _architecture_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_architecture_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_architecture_value(item) for item in value), key=repr)
    if isinstance(value, types.CodeType):
        return _code_signature(value)
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _sha256_bytes(np.ascontiguousarray(value).tobytes()),
        }
    if hasattr(value, "detach") and hasattr(value, "dtype"):
        try:
            tensor = value.detach().to("cpu").contiguous()
            return {
                "type": "tensor",
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": _sha256_bytes(tensor.numpy().tobytes(order="C")),
            }
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise FeatureValidationError(
                "cannot hash encoder architecture tensor"
            ) from error
    if callable(value):
        signature = {
            "type": "callable",
            "module": getattr(value, "__module__", "unknown"),
            "qualname": getattr(value, "__qualname__", type(value).__qualname__),
        }
        code = getattr(getattr(value, "__func__", value), "__code__", None)
        if isinstance(code, types.CodeType):
            signature["code"] = _code_signature(code)
        return signature
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _code_signature(code: types.CodeType) -> dict[str, Any]:
    return {
        "argcount": code.co_argcount,
        "cellvars": list(code.co_cellvars),
        "code": code.co_code.hex(),
        "consts": [_architecture_value(value) for value in code.co_consts],
        "exceptiontable": code.co_exceptiontable.hex(),
        "flags": code.co_flags,
        "freevars": list(code.co_freevars),
        "kwonlyargcount": code.co_kwonlyargcount,
        "names": list(code.co_names),
        "posonlyargcount": code.co_posonlyargcount,
        "qualname": code.co_qualname,
        "varnames": list(code.co_varnames),
    }


def _callable_signature(value: Any) -> dict[str, Any]:
    signature = {
        "module": getattr(value, "__module__", "unknown"),
        "qualname": getattr(value, "__qualname__", type(value).__qualname__),
    }
    code = getattr(getattr(value, "__func__", value), "__code__", None)
    if isinstance(code, types.CodeType):
        signature["code"] = _code_signature(code)
        global_namespace = getattr(getattr(value, "__func__", value), "__globals__", {})
        dependencies: dict[str, Any] = {}
        instructions = tuple(dis.get_instructions(code))
        for index, instruction in enumerate(instructions):
            if instruction.opname != "LOAD_GLOBAL":
                continue
            name = instruction.argval
            if name not in global_namespace:
                continue
            current = global_namespace[name]
            attributes: list[str] = []
            cursor = index + 1
            while cursor < len(instructions) and instructions[cursor].opname in {
                "LOAD_ATTR",
                "LOAD_METHOD",
            }:
                attribute = instructions[cursor].argval
                attributes.append(attribute)
                try:
                    current = getattr(current, attribute)
                except AttributeError:
                    break
                cursor += 1
            key = name if not attributes else f"{name}.{'/'.join(attributes)}"
            dependencies[key] = _architecture_value(current)
        signature["global_dependencies"] = dependencies
    return signature


def _module_graph_node(name: str, module: Any) -> dict[str, Any]:
    module_type = type(module)
    forward = getattr(module_type, "forward", None)
    attrs = {
        str(key): _architecture_value(value)
        for key, value in sorted(vars(module).items())
        if key not in _GRAPH_INTERNAL_ATTRS
    }
    parameters = []
    for parameter_name, parameter in sorted(module._parameters.items()):
        if parameter is not None:
            parameters.append(
                {
                    "name": parameter_name,
                    "dtype": str(parameter.dtype),
                    "shape": list(parameter.shape),
                }
            )
    buffers = []
    for buffer_name, buffer in sorted(module._buffers.items()):
        if buffer is not None:
            buffers.append(
                {
                    "name": buffer_name,
                    "dtype": str(buffer.dtype),
                    "shape": list(buffer.shape),
                    "sha256": _sha256_bytes(
                        buffer.detach()
                        .to("cpu")
                        .contiguous()
                        .numpy()
                        .tobytes(order="C")
                    ),
                }
            )
    return {
        "name": name,
        "class": f"{module_type.__module__}.{module_type.__qualname__}",
        "forward": {
            **_callable_signature(forward),
        },
        "reachable_methods": {
            method_name: _callable_signature(method)
            for method_name in sorted(
                set(getattr(forward, "__code__", ()).co_names if forward else ())
            )
            if callable(method := getattr(module_type, method_name, None))
        },
        "attrs": attrs,
        "parameters": parameters,
        "buffers": buffers,
        "children": [
            _module_graph_node(child_name, child)
            for child_name, child in sorted(module._modules.items())
        ],
    }


def _module_graph_sha256(model: Any) -> str:
    try:
        graph = _module_graph_node("", model)
        return _sha256_bytes(_canonical_json(graph))
    except FeatureValidationError:
        raise
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise FeatureValidationError("cannot hash encoder module graph") from error


def _model_type_key(model: Any) -> str:
    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__qualname__}"


def _runtime_snapshot(torch: Any) -> dict[str, Any]:
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
    }


def _canonical_torch_device(torch: Any, device: str) -> Any:
    try:
        requested = torch.device(device)
        if requested.type == "cuda" and requested.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return requested
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise FeatureValidationError("encoder model device is invalid") from error


def _validate_runtime_versions(torch: Any, torchvision: Any) -> None:
    """Require the exact module, build, CUDA, and distribution runtime."""

    try:
        import torch as canonical_torch
        import torchvision as canonical_torchvision

        if torch is not canonical_torch or torchvision is not canonical_torchvision:
            raise FeatureValidationError(
                "torch runtime module proxy is not permitted"
            )
        values = (
            getattr(torch, "__version__", None),
            getattr(torch.version, "__version__", None),
            getattr(torch.version, "cuda", None),
            getattr(torchvision, "__version__", None),
            getattr(torchvision.version, "__version__", None),
            importlib_metadata.version("torch"),
            importlib_metadata.version("torchvision"),
        )
    except (AttributeError, importlib_metadata.PackageNotFoundError) as error:
        raise FeatureValidationError(
            "torch runtime version is not registered"
        ) from error
    expected = (
        _EXPECTED_TORCH_VERSION,
        _EXPECTED_TORCH_VERSION,
        _EXPECTED_TORCH_CUDA_VERSION,
        _EXPECTED_TORCHVISION_VERSION,
        _EXPECTED_TORCHVISION_VERSION,
        _EXPECTED_TORCH_DISTRIBUTION_VERSION,
        _EXPECTED_TORCHVISION_DISTRIBUTION_VERSION,
    )
    if values != expected:
        raise FeatureValidationError("torch runtime version is not registered")


_MODULE_HOOK_REGISTRIES = (
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_always_called",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_load_state_dict_pre_hooks",
    "_is_full_backward_hook",
)
_GLOBAL_HOOK_REGISTRIES = (
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
    "_global_buffer_registration_hooks",
    "_global_forward_hooks",
    "_global_forward_hooks_always_called",
    "_global_forward_hooks_with_kwargs",
    "_global_forward_pre_hooks",
    "_global_is_full_backward_hook",
    "_global_module_registration_hooks",
    "_global_parameter_registration_hooks",
)
_INSTANCE_EXECUTION_OVERRIDES = (
    "__call__",
    "_call_impl",
    "_wrapped_call_impl",
    "buffers",
    "forward",
    "load_state_dict",
    "modules",
    "named_buffers",
    "named_modules",
    "named_parameters",
    "parameters",
    "state_dict",
)


def _torch_class_execution_snapshot(torch: Any) -> tuple[tuple[str, Any], ...]:
    result: list[tuple[str, Any]] = []
    for class_name, runtime_type in (
        ("torch.nn.Module", torch.nn.Module),
        ("torch.Tensor", torch.Tensor),
    ):
        seen: set[str] = set()
        for base in runtime_type.__mro__:
            for attribute_name, value in vars(base).items():
                if attribute_name in seen:
                    continue
                seen.add(attribute_name)
                result.append((f"{class_name}.{attribute_name}", value))
    return tuple(result)


def _execution_literal(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes_sha256": _sha256_bytes(value), "length": len(value)}
    if isinstance(value, types.CodeType):
        return {"code": _code_signature(value)}
    if isinstance(value, type):
        return {
            "class": f"{value.__module__}.{value.__qualname__}",
        }
    if isinstance(value, (tuple, list)):
        return [_execution_literal(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_execution_literal(item) for item in value), key=repr)
    if isinstance(value, Mapping):
        return {
            str(key): _execution_literal(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _execution_descriptor_signature(value: Any) -> Mapping[str, Any]:
    if isinstance(value, (staticmethod, classmethod)):
        return {
            "kind": type(value).__name__,
            "value": _execution_descriptor_signature(value.__func__),
        }
    if isinstance(value, property):
        return {
            "kind": "property",
            "get": (
                _execution_descriptor_signature(value.fget)
                if value.fget is not None
                else None
            ),
            "set": (
                _execution_descriptor_signature(value.fset)
                if value.fset is not None
                else None
            ),
            "delete": (
                _execution_descriptor_signature(value.fdel)
                if value.fdel is not None
                else None
            ),
        }
    target = getattr(value, "__func__", value)
    code = getattr(target, "__code__", None)
    if isinstance(code, types.CodeType):
        closure = getattr(target, "__closure__", None)
        return {
            "kind": "python_callable",
            "module": getattr(target, "__module__", ""),
            "qualname": getattr(target, "__qualname__", ""),
            "code": _code_signature(code),
            "defaults": _execution_literal(getattr(target, "__defaults__", None)),
            "kwdefaults": _execution_literal(
                getattr(target, "__kwdefaults__", None)
            ),
            "closure": (
                [
                    _execution_literal(cell.cell_contents)
                    for cell in closure
                ]
                if closure is not None
                else None
            ),
        }
    if callable(value):
        return {
            "kind": "callable_descriptor",
            "module": getattr(value, "__module__", ""),
            "qualname": getattr(value, "__qualname__", ""),
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    return {"kind": "value", "value": _execution_literal(value)}


def _model_class_execution_snapshot(model: Any) -> tuple[Any, ...]:
    result: list[Any] = []
    seen_types: set[type[Any]] = set()
    for _, module in _iter_model_modules(model):
        runtime_type = type(module)
        if runtime_type in seen_types:
            continue
        seen_types.add(runtime_type)
        mro: list[Any] = []
        for base in runtime_type.__mro__:
            descriptors = tuple(
                (
                    name,
                    value,
                    _sha256_bytes(
                        _canonical_json(_execution_descriptor_signature(value))
                    ),
                )
                for name, value in sorted(vars(base).items())
            )
            mro.append(
                (base, str(base.__module__), str(base.__qualname__), descriptors)
            )
        result.append(
            (
                runtime_type,
                str(runtime_type.__module__),
                str(runtime_type.__qualname__),
                tuple(mro),
            )
        )
    return tuple(result)


def _model_class_execution_digest(snapshot: tuple[Any, ...]) -> str:
    value = [
        {
            "module": module_name,
            "qualname": qualname,
            "mro": [
                {
                    "module": base_module,
                    "qualname": base_qualname,
                    "descriptors": [
                        {"name": name, "sha256": digest}
                        for name, _descriptor, digest in descriptors
                    ],
                }
                for _base, base_module, base_qualname, descriptors in mro
            ],
        }
        for _runtime_type, module_name, qualname, mro in snapshot
    ]
    return _sha256_bytes(_canonical_json(value))


def _validate_model_class_execution(
    model: Any, expected: tuple[Any, ...]
) -> None:
    current = _model_class_execution_snapshot(model)
    expected_by_class = {entry[0]: entry for entry in expected}
    for current_type in current:
        current_class, current_module, current_qualname, current_mro = current_type
        expected_type = expected_by_class.get(current_class)
        if expected_type is None:
            raise FeatureValidationError(
                "encoder module architecture concrete class roster changed"
            )
        expected_class, expected_module, expected_qualname, expected_mro = expected_type
        if (
            current_class is not expected_class
            or current_module != expected_module
            or current_qualname != expected_qualname
            or len(current_mro) != len(expected_mro)
        ):
            raise FeatureValidationError(
                "encoder module architecture concrete class identity changed"
            )
        for current_base, expected_base in zip(
            current_mro, expected_mro, strict=True
        ):
            (
                current_base_class,
                current_base_module,
                current_base_qualname,
                current_descriptors,
            ) = current_base
            (
                expected_base_class,
                expected_base_module,
                expected_base_qualname,
                expected_descriptors,
            ) = expected_base
            if (
                current_base_class is not expected_base_class
                or current_base_module != expected_base_module
                or current_base_qualname != expected_base_qualname
                or len(current_descriptors) != len(expected_descriptors)
            ):
                raise FeatureValidationError(
                    "encoder module architecture concrete class MRO changed"
                )
            for current_descriptor, expected_descriptor in zip(
                current_descriptors, expected_descriptors, strict=True
            ):
                current_name, current_value, current_digest = current_descriptor
                expected_name, expected_value, expected_digest = expected_descriptor
                if (
                    current_name != expected_name
                    or current_value is not expected_value
                    or current_digest != expected_digest
                ):
                    raise FeatureValidationError(
                        "encoder module architecture or state concrete class "
                        "execution descriptor changed"
                    )


def _freeze_model_class_execution() -> tuple[Any, ...]:
    try:
        import torch
        import torchvision
        from torchvision.models import resnet18

        _validate_runtime_versions(torch, torchvision)
        model = resnet18(weights=None)
        before = _model_class_execution_snapshot(model)
        model.fc = torch.nn.Identity()
        after = _model_class_execution_snapshot(model)
        by_class = {entry[0]: entry for entry in before}
        by_class.update({entry[0]: entry for entry in after})
        snapshot = tuple(by_class.values())
    except FeatureValidationError:
        raise
    except Exception as error:
        raise FeatureValidationError(
            "cannot freeze ResNet18 concrete execution classes"
        ) from error
    digest = _model_class_execution_digest(snapshot)
    if digest != _EXPECTED_MODEL_CLASS_EXECUTION_SHA256:
        raise FeatureValidationError(
            "ResNet18 concrete execution classes are not registered"
        )
    return snapshot


def _model_class_execution_runtime(
    snapshot: tuple[Any, ...],
) -> Callable[[Any], None]:
    def validate(model: Any) -> None:
        _validate_model_class_execution(model, snapshot)

    return validate


def _runtime_primitive_snapshot(torch: Any) -> tuple[tuple[str, Any], ...]:
    functional = torch.nn.functional
    module_api = torch.nn.modules.module
    return (
        ("numpy", np),
        ("numpy.abs", np.abs),
        ("numpy.all", np.all),
        ("numpy.any", np.any),
        ("numpy.ascontiguousarray", np.ascontiguousarray),
        ("numpy.asarray", np.asarray),
        ("numpy.concatenate", np.concatenate),
        ("numpy.dtype", np.dtype),
        ("numpy.errstate", np.errstate),
        ("numpy.finfo", np.finfo),
        ("numpy.float32", np.float32),
        ("numpy.isfinite", np.isfinite),
        ("numpy.issubdtype", np.issubdtype),
        ("numpy.number", np.number),
        ("numpy.repeat", np.repeat),
        ("numpy.stack", np.stack),
        ("PIL.Image", Image),
        ("PIL.Image.Image", Image.Image),
        ("PIL.Image.Image.convert", Image.Image.convert),
        ("PIL.Image.Image.copy", Image.Image.copy),
        ("PIL.Image.Image.load", Image.Image.load),
        ("PIL.Image.open", Image.open),
        ("torch.backends", torch.backends),
        ("torch.backends.cuda", torch.backends.cuda),
        ("torch.backends.cuda.matmul", torch.backends.cuda.matmul),
        ("torch.backends.cudnn", torch.backends.cudnn),
        (
            "torch.are_deterministic_algorithms_enabled",
            torch.are_deterministic_algorithms_enabled,
        ),
        ("torch.from_numpy", torch.from_numpy),
        ("torch.get_deterministic_debug_mode", torch.get_deterministic_debug_mode),
        ("torch.get_float32_matmul_precision", torch.get_float32_matmul_precision),
        ("torch.inference_mode", torch.inference_mode),
        ("torch.set_float32_matmul_precision", torch.set_float32_matmul_precision),
        ("torch.set_deterministic_debug_mode", torch.set_deterministic_debug_mode),
        ("torch.stack", torch.stack),
        ("torch.use_deterministic_algorithms", torch.use_deterministic_algorithms),
        ("torch.nn.functional.interpolate", functional.interpolate),
        ("torch.nn.Module.__call__", torch.nn.Module.__call__),
        ("torch.nn.Module._call_impl", torch.nn.Module._call_impl),
        ("torch.nn.Module._wrapped_call_impl", torch.nn.Module._wrapped_call_impl),
        ("torch.Tensor.detach", torch.Tensor.detach),
        ("torch.Tensor.numpy", torch.Tensor.numpy),
        ("torch.Tensor.reshape", torch.Tensor.reshape),
        ("torch.Tensor.to", torch.Tensor.to),
        ("register_module_forward_hook", module_api.register_module_forward_hook),
    ) + _torch_class_execution_snapshot(torch)


import torch as _canonical_torch_runtime

_FROZEN_RUNTIME_PRIMITIVES = _runtime_primitive_snapshot(_canonical_torch_runtime)
del _canonical_torch_runtime


def _module_execution_snapshot(model: Any) -> tuple[tuple[Any, type[Any], Any], ...]:
    return tuple(
        (module, type(module), getattr(type(module), "forward", None))
        for _, module in _iter_model_modules(model)
    )


_validate_registered_model_class_execution = _model_class_execution_runtime(
    _freeze_model_class_execution()
)
del _model_class_execution_runtime


def _local_execution_snapshot() -> tuple[tuple[str, Any, Any], ...]:
    return tuple(
        (name, value, getattr(value, "__code__", None))
        for name, value in (
            ("_as_rgb_float", _as_rgb_float),
            ("_canonical_json", _canonical_json),
            ("_decode_image_snapshot", _decode_image_snapshot),
            ("_deterministic_runtime", _deterministic_runtime),
            ("_execution_descriptor_signature", _execution_descriptor_signature),
            ("_execution_literal", _execution_literal),
            ("_iter_model_modules", _iter_model_modules),
            ("_load_image_item", _load_image_item),
            ("_model_state_sha256", _model_state_sha256),
            ("_model_class_execution_digest", _model_class_execution_digest),
            ("_model_class_execution_snapshot", _model_class_execution_snapshot),
            ("_module_graph_sha256", _module_graph_sha256),
            ("_read_file_snapshot", _read_file_snapshot),
            ("_runtime_snapshot", _runtime_snapshot),
            ("_sha256_bytes", _sha256_bytes),
            ("_strict_real_numeric_array", _strict_real_numeric_array),
            ("_torch_runtime_process_lock", _torch_runtime_process_lock),
            ("_torch_class_execution_snapshot", _torch_class_execution_snapshot),
            ("_validate_input_file_path", _validate_input_file_path),
            (
                "_validate_model_class_execution",
                _validate_model_class_execution,
            ),
            (
                "_validate_registered_model_class_execution",
                _validate_registered_model_class_execution,
            ),
            ("_validate_execution_integrity", _validate_execution_integrity),
            ("_validate_runtime_versions", _validate_runtime_versions),
            ("_validate_transform_contract", _validate_transform_contract),
            ("sha256_file", sha256_file),
            ("normalize_imagenet_grayscale", normalize_imagenet_grayscale),
            ("preprocess_full_field", preprocess_full_field),
            ("resize_grayscale_bilinear", resize_grayscale_bilinear),
            ("to_grayscale_luminance", to_grayscale_luminance),
        )
    )


def _validate_execution_integrity(
    model: Any,
    torch: Any,
    runtime_primitives: tuple[tuple[str, Any], ...],
    module_execution: tuple[tuple[Any, type[Any], Any], ...],
    local_execution: tuple[tuple[str, Any, Any], ...],
) -> None:
    current_modules = _iter_model_modules(model)
    for _, module in current_modules:
        parametrizations = vars(module).get("parametrizations")
        child_modules = vars(module).get("_modules", {})
        if (
            parametrizations is not None
            and len(parametrizations)
            or "parametrizations" in child_modules
        ):
            raise FeatureValidationError("encoder parametrizations are not permitted")
    if len(current_modules) != len(module_execution):
        raise FeatureValidationError("encoder module execution graph changed")
    for (_, module), (expected_module, expected_type, expected_forward) in zip(
        current_modules, module_execution, strict=True
    ):
        if (
            module is not expected_module
            or type(module) is not expected_type
            or getattr(expected_type, "forward", None) is not expected_forward
        ):
            raise FeatureValidationError("encoder module execution graph changed")
        attributes = vars(module)
        if any(attributes.get(name) for name in _MODULE_HOOK_REGISTRIES):
            raise FeatureValidationError("encoder module hooks are not permitted")
        if any(name in attributes for name in _INSTANCE_EXECUTION_OVERRIDES):
            raise FeatureValidationError(
                "encoder instance execution overrides are not permitted"
            )
    module_api = torch.nn.modules.module
    if any(getattr(module_api, name, None) for name in _GLOBAL_HOOK_REGISTRIES):
        raise FeatureValidationError("global torch module hooks are not permitted")
    _validate_registered_model_class_execution(model)
    current_primitives = dict(_runtime_primitive_snapshot(torch))
    if any(current_primitives.get(name) is not value for name, value in runtime_primitives):
        raise FeatureValidationError("encoder state or torch runtime primitive changed")
    namespace = globals()
    if any(
        namespace.get(name) is not value
        or getattr(value, "__code__", None) is not expected_code
        for name, value, expected_code in local_execution
    ):
        raise FeatureValidationError("encoder local execution primitive changed")


_RUNTIME_CONTRACT = MappingProxyType(
    {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "float32_matmul_precision": "highest",
    }
)


@contextmanager
def _torch_runtime_process_lock(
    _lock_path: Path = _TORCH_RUNTIME_LOCK_PATH,
):
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(_lock_path, flags, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise FeatureValidationError(
                "torch runtime lock must be an owned regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    except FeatureValidationError:
        raise
    except OSError as error:
        raise FeatureValidationError("cannot acquire torch runtime process lock") from error
    finally:
        if descriptor >= 0:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


@contextmanager
def _deterministic_runtime(torch: Any):
    with _TORCH_GLOBAL_STATE_LOCK, _torch_runtime_process_lock():
        old = _runtime_snapshot(torch)
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            yield
        finally:
            torch.set_deterministic_debug_mode(old["deterministic_debug_mode"])
            torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
            torch.backends.cudnn.benchmark = old["cudnn_benchmark"]
            torch.backends.cuda.matmul.allow_tf32 = old["cuda_matmul_tf32"]
            torch.backends.cudnn.allow_tf32 = old["cudnn_tf32"]
            torch.set_float32_matmul_precision(old["float32_matmul_precision"])


@dataclass(frozen=True)
class FrozenResNet18Encoder:
    """A frozen, hash-bound ResNet18 feature extractor."""

    _model: Any
    torch: Any
    torchvision: Any
    weight_path: Path
    project_root: Path
    weight_relative_path: Path
    weight_sha256: str
    weight_bytes: int
    device: str
    batch_size: int
    transform_sha256: str = TRANSFORM_SHA256
    model_state_sha256: str = ""
    architecture_sha256: str = ""
    _runtime_primitives: tuple[tuple[str, Any], ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    _module_execution: tuple[tuple[Any, type[Any], Any], ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    _local_execution: tuple[tuple[str, Any, Any], ...] = field(
        default_factory=tuple, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_runtime_versions(self.torch, self.torchvision)
        _validate_registered_model_class_execution(self._model)
        self._model.requires_grad_(False)
        self._model.eval()
        state_sha256 = self.model_state_sha256 or _model_state_sha256(self._model)
        architecture_sha256 = self.architecture_sha256 or _module_graph_sha256(
            self._model
        )
        if self.project_root != _MODULE_REPO_ROOT:
            raise FeatureValidationError("encoder project root is not registered")
        if self.weight_relative_path != _FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH:
            raise FeatureValidationError("encoder weight path provenance is invalid")
        if type(self.weight_bytes) is not int or self.weight_bytes <= 0:
            raise FeatureValidationError("encoder weight byte count is invalid")
        if _model_type_key(self._model) != _EXPECTED_MODEL_TYPE:
            raise FeatureValidationError("encoder model type is not registered")
        if state_sha256 != _EXPECTED_MODEL_STATE_SHA256:
            raise FeatureValidationError("encoder model state is not registered")
        if architecture_sha256 != _EXPECTED_ARCHITECTURE_SHA256:
            raise FeatureValidationError("encoder architecture is not registered")
        object.__setattr__(self, "model_state_sha256", state_sha256)
        object.__setattr__(self, "architecture_sha256", architecture_sha256)
        object.__setattr__(
            self, "_runtime_primitives", _FROZEN_RUNTIME_PRIMITIVES
        )
        object.__setattr__(
            self, "_module_execution", _module_execution_snapshot(self._model)
        )
        object.__setattr__(self, "_local_execution", _FROZEN_LOCAL_EXECUTION)

    def _validate_encoder_state(self) -> None:
        _validate_transform_contract()
        try:
            _validate_runtime_versions(self.torch, self.torchvision)
        except FeatureValidationError as error:
            raise FeatureValidationError("torch runtime version changed") from error
        if self.transform_sha256 != _EXPECTED_TRANSFORM_SHA:
            raise FeatureValidationError("encoder transform hash drift detected")
        if self.batch_size != EMBEDDING_BATCH_SIZE:
            raise FeatureValidationError("encoder batch size is not registered")
        if self.project_root != _MODULE_REPO_ROOT:
            raise FeatureValidationError("encoder project root is not registered")
        if self.weight_relative_path != _FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH:
            raise FeatureValidationError("encoder weight path provenance changed")
        expected_path = _MODULE_REPO_ROOT / _FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH
        if (
            self.weight_path != expected_path
            or expected_path.is_symlink()
            or expected_path.resolve() != expected_path
        ):
            raise FeatureValidationError("encoder weight path is not registered")
        weight_snapshot = _read_file_snapshot(
            self.weight_path,
            "registered ResNet18 weights",
            max_bytes=_MAX_WEIGHT_BYTES,
        )
        if (
            len(weight_snapshot) != self.weight_bytes
            or _sha256_bytes(weight_snapshot) != _FROZEN_RESNET18_WEIGHTS_SHA256
        ):
            raise FeatureValidationError("encoder weight digest changed")
        if self.weight_sha256 != _FROZEN_RESNET18_WEIGHTS_SHA256:
            raise FeatureValidationError("encoder weight provenance changed")
        if _model_type_key(self._model) != _EXPECTED_MODEL_TYPE:
            raise FeatureValidationError("encoder model type changed")
        _validate_execution_integrity(
            self._model,
            self.torch,
            self._runtime_primitives,
            self._module_execution,
            self._local_execution,
        )
        if self._model.training:
            raise FeatureValidationError("encoder model must remain in eval mode")
        if any(parameter.requires_grad for parameter in self._model.parameters()):
            raise FeatureValidationError("encoder model must remain frozen")
        try:
            expected_device = _canonical_torch_device(self.torch, self.device)
            model_devices = {parameter.device for parameter in self._model.parameters()}
            if model_devices and model_devices != {expected_device}:
                raise FeatureValidationError("encoder model device changed")
        except FeatureValidationError:
            raise
        except (AttributeError, RuntimeError, TypeError) as error:
            raise FeatureValidationError("encoder model device is invalid") from error
        if _model_state_sha256(self._model) != _EXPECTED_MODEL_STATE_SHA256:
            raise FeatureValidationError("encoder model state hash changed")
        if _module_graph_sha256(self._model) != _EXPECTED_ARCHITECTURE_SHA256:
            raise FeatureValidationError("encoder module architecture changed")

    def encode_spatial(
        self,
        images: Iterable[Image.Image | np.ndarray | Path | str],
        *,
        layer: str = "layer3",
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode images into frozen pre-pooling ResNet18 feature maps."""

        shapes = {
            "layer1": (64, 56, 56),
            "layer2": (128, 28, 28),
            "layer3": (256, 14, 14),
            "layer4": (512, 7, 7),
        }
        if layer not in shapes:
            raise FeatureValidationError("spatial layer must be layer1, layer2, layer3, or layer4")
        with _ENCODER_EXECUTION_LOCK:
            self._validate_encoder_state()
            try:
                declared_length = len(images)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                declared_length = None
            if declared_length is not None and declared_length > _MAX_EMBEDDING_SAMPLES:
                raise FeatureValidationError(
                    "encoder cohort exceeds the registered 276-row limit"
                )
            values: list[Image.Image | np.ndarray | Path | str] = []
            for item in images:
                values.append(item)
                if len(values) > _MAX_EMBEDDING_SAMPLES:
                    raise FeatureValidationError(
                        "encoder cohort exceeds the registered 276-row limit"
                    )
            if batch_size is not None and batch_size != self.batch_size:
                raise FeatureValidationError(
                    "batch_size is fixed by the encoder contract"
                )
            expected = shapes[layer]
            if not values:
                empty = np.empty((0, *expected), dtype=np.float32)
                empty.setflags(write=False)
                return empty
            tensors = [
                self.torch.from_numpy(
                    preprocess_full_field(_load_image_item(item), size=224)
                )
                for item in values
            ]
            outputs: list[np.ndarray] = []
            try:
                with _deterministic_runtime(self.torch), self.torch.inference_mode():
                    for start in range(0, len(tensors), self.batch_size):
                        batch = self.torch.stack(
                            tensors[start : start + self.batch_size]
                        ).to(self.device, non_blocking=False)
                        model = self._model
                        output = model.maxpool(model.relu(model.bn1(model.conv1(batch))))
                        output = model.layer1(output)
                        if layer != "layer1":
                            output = model.layer2(output)
                        if layer in {"layer3", "layer4"}:
                            output = model.layer3(output)
                        if layer == "layer4":
                            output = model.layer4(output)
                        array = np.asarray(
                            output.detach().to("cpu").numpy(), dtype=np.float32
                        )
                        if array.ndim != 4 or tuple(array.shape[1:]) != expected:
                            raise FeatureValidationError(
                                "ResNet18 spatial output shape is invalid"
                            )
                        outputs.append(array)
            except FeatureValidationError:
                raise
            except Exception as error:
                raise FeatureValidationError(
                    "ResNet18 spatial inference failed"
                ) from error
            self._validate_encoder_state()
            combined = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
            if combined.shape != (len(values), *expected) or not np.all(
                np.isfinite(combined)
            ):
                raise FeatureValidationError(
                    "ResNet18 spatial output is incomplete"
                )
            snapshot = np.frombuffer(
                combined.tobytes(order="C"), dtype=np.float32
            ).reshape(combined.shape)
            snapshot.setflags(write=False)
            return snapshot

    def encode(
        self,
        images: Iterable[Image.Image | np.ndarray | Path | str],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode images in input order without labels or fitted state."""

        with _ENCODER_EXECUTION_LOCK:
            self._validate_encoder_state()
            try:
                declared_length = len(images)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                declared_length = None
            if declared_length is not None and declared_length > _MAX_EMBEDDING_SAMPLES:
                raise FeatureValidationError(
                    "encoder cohort exceeds the registered 276-row limit"
                )
            values: list[Image.Image | np.ndarray | Path | str] = []
            for item in images:
                values.append(item)
                if len(values) > _MAX_EMBEDDING_SAMPLES:
                    raise FeatureValidationError(
                        "encoder cohort exceeds the registered 276-row limit"
                    )
            if batch_size is not None and batch_size != self.batch_size:
                raise FeatureValidationError(
                    "batch_size is fixed by the encoder contract"
                )
            size = self.batch_size
            if not values:
                return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
            tensors = [
                self.torch.from_numpy(
                    preprocess_full_field(_load_image_item(item), size=224)
                )
                for item in values
            ]
            outputs: list[np.ndarray] = []
            try:
                with _deterministic_runtime(self.torch), self.torch.inference_mode():
                    for start in range(0, len(tensors), size):
                        batch = self.torch.stack(tensors[start : start + size]).to(
                            self.device, non_blocking=False
                        )
                        try:
                            output = self._model(batch).detach().to("cpu").numpy()
                        except (RuntimeError, TypeError, ValueError) as error:
                            raise FeatureValidationError(
                                "ResNet18 forward graph is invalid"
                            ) from error
                        output = np.asarray(output, dtype=np.float32)
                        if output.ndim != 2 or output.shape[1] != EMBEDDING_DIMENSION:
                            raise FeatureValidationError(
                                "ResNet18 output is not 512-dimensional"
                            )
                        outputs.append(output)
            except FeatureValidationError:
                raise
            except Exception as error:
                raise FeatureValidationError(
                    "ResNet18 embedding inference failed"
                ) from error
            self._validate_encoder_state()
            embeddings = np.concatenate(outputs, axis=0).astype(
                np.float32, copy=False
            )
            if embeddings.shape != (len(values), EMBEDDING_DIMENSION):
                raise FeatureValidationError(
                    "ResNet18 output rows do not match input rows"
                )
            if not np.all(np.isfinite(embeddings)):
                raise FeatureValidationError(
                    "ResNet18 embeddings must contain only finite values"
                )
            return embeddings

    def encode_paths(
        self,
        image_paths: Iterable[Path | str],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        return self.encode(image_paths, batch_size=batch_size)

    def provenance(self) -> dict[str, Any]:
        self._validate_encoder_state()
        return {
            "encoder": "torchvision_resnet18",
            "embedding_dimension": EMBEDDING_DIMENSION,
            "frozen": True,
            "transform_sha256": _EXPECTED_TRANSFORM_SHA,
            "torch_version": str(getattr(self.torch, "__version__", "unknown")),
            "torchvision_version": str(
                getattr(self.torchvision, "__version__", "unknown")
            ),
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "weights_bytes": self.weight_bytes,
            "weights_filename": _FROZEN_RESNET18_WEIGHTS_FILENAME,
            "weights_sha256": _FROZEN_RESNET18_WEIGHTS_SHA256,
            "weights_url": _FROZEN_RESNET18_WEIGHTS_URL,
            "weight_path": _FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH.as_posix(),
            "device": self.device,
            "batch_size": self.batch_size,
            "runtime": dict(_RUNTIME_CONTRACT),
            "model_state_sha256": _EXPECTED_MODEL_STATE_SHA256,
            "architecture_sha256": _EXPECTED_ARCHITECTURE_SHA256,
        }


def encode_resnet18(
    images: Iterable[Image.Image | np.ndarray | Path | str] | None = None,
    *,
    weight_path: Path | str | None = None,
    project_root: Path | str | None = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> FrozenResNet18Encoder | np.ndarray:
    """Create the frozen encoder, or encode ``images`` when supplied."""

    if not isinstance(device, str) or not device:
        raise FeatureValidationError("device must be a non-empty string")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise FeatureValidationError("batch_size must be a positive integer")
    if batch_size != EMBEDDING_BATCH_SIZE:
        raise FeatureValidationError(
            f"batch_size must equal the registered value {EMBEDDING_BATCH_SIZE}"
        )
    path, root, weight_snapshot = _validate_weight_path(weight_path, project_root)
    model, torch, torchvision = _load_frozen_model(weight_snapshot, device)
    encoder = FrozenResNet18Encoder(
        _model=model,
        torch=torch,
        torchvision=torchvision,
        weight_path=path,
        project_root=root,
        weight_relative_path=_FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH,
        weight_sha256=_FROZEN_RESNET18_WEIGHTS_SHA256,
        weight_bytes=len(weight_snapshot),
        device=device,
        batch_size=batch_size,
    )
    if images is None:
        return encoder
    return encoder.encode(images)


def extract_resnet18_embeddings(
    images: Iterable[Image.Image | np.ndarray | Path | str],
    *,
    weight_path: Path | str | None = None,
    project_root: Path | str | None = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Compatibility wrapper returning only the frozen 512-D array."""

    result = encode_resnet18(
        images,
        weight_path=weight_path,
        project_root=project_root,
        device=device,
        batch_size=batch_size,
    )
    if isinstance(result, FrozenResNet18Encoder):  # pragma: no cover - defensive
        raise TypeError("encoder construction unexpectedly returned without images")
    return result


def _read_file_snapshot(path: Path, name: str, *, max_bytes: int) -> bytes:
    candidate = _validate_input_file_path(Path(path), name)
    if ".." in candidate.parts:
        raise FeatureValidationError(f"{name} path cannot contain parent traversal")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    parts = candidate.parts[1:]
    directory_descriptor = -1
    descriptor = -1
    try:
        directory_descriptor = os.open(candidate.anchor, directory_flags)
        for part in parts[:-1]:
            next_descriptor = os.open(
                part, directory_flags, dir_fd=directory_descriptor
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            parts[-1], file_flags, dir_fd=directory_descriptor
        )
        info_before = os.fstat(descriptor)
        if not stat.S_ISREG(info_before.st_mode):
            raise FeatureValidationError(f"{name} must be a regular file")
        if info_before.st_size > max_bytes:
            raise FeatureValidationError(f"{name} exceeds the byte limit")
        def read_descriptor() -> bytes:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor, min(1024 * 1024, max_bytes + 1 - total)
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    raise FeatureValidationError(f"{name} exceeds the byte limit")
            return b"".join(chunks)

        snapshot = read_descriptor()
        info_after = os.fstat(descriptor)
        before_identity = (
            info_before.st_dev,
            info_before.st_ino,
            info_before.st_mode,
            info_before.st_size,
        )
        after_identity = (
            info_after.st_dev,
            info_after.st_ino,
            info_after.st_mode,
            info_after.st_size,
        )
        if before_identity != after_identity or len(snapshot) != info_before.st_size:
            raise FeatureValidationError(f"{name} changed while reading")
        before_times = (info_before.st_mtime_ns, info_before.st_ctime_ns)
        after_times = (info_after.st_mtime_ns, info_after.st_ctime_ns)
        if before_times != after_times:
            os.lseek(descriptor, 0, os.SEEK_SET)
            verification_before = os.fstat(descriptor)
            verification = read_descriptor()
            verification_after = os.fstat(descriptor)
            verification_identity = (
                verification_before.st_dev,
                verification_before.st_ino,
                verification_before.st_mode,
                verification_before.st_size,
                verification_before.st_mtime_ns,
                verification_before.st_ctime_ns,
            )
            verification_after_identity = (
                verification_after.st_dev,
                verification_after.st_ino,
                verification_after.st_mode,
                verification_after.st_size,
                verification_after.st_mtime_ns,
                verification_after.st_ctime_ns,
            )
            if (
                verification_identity != verification_after_identity
                or len(verification) != verification_before.st_size
                or verification != snapshot
            ):
                raise FeatureValidationError(f"{name} changed while reading")
        return snapshot
    except FeatureValidationError:
        raise
    except (IndexError, OSError) as error:
        raise FeatureValidationError(f"cannot read {name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)


def _decode_image_snapshot(payload: bytes, name: str) -> Image.Image:
    try:
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                raise FeatureValidationError(f"{name} exceeds the pixel limit")
            image.load()
            return image.convert("RGB").copy()
    except FeatureValidationError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise FeatureValidationError(f"cannot decode {name}") from error


def _source_snapshot(
    item: Image.Image | np.ndarray | Path | str,
) -> tuple[str, str, Image.Image | np.ndarray, int, int, int]:
    if isinstance(item, (str, Path)):
        path = Path(item)
        payload = _read_file_snapshot(
            path, "registered image", max_bytes=_MAX_IMAGE_BYTES
        )
        image = _decode_image_snapshot(payload, "registered image")
        return (
            _sha256_bytes(payload),
            str(path.resolve()),
            image,
            image.width,
            image.height,
            len(payload),
        )
    value = np.asarray(item)
    if value.dtype != np.dtype(np.uint8):
        raise FeatureValidationError(
            "in-memory images must use the registered uint8_div255 contract"
        )
    if value.ndim < 2 or value.shape[0] * value.shape[1] > _MAX_IMAGE_PIXELS:
        raise FeatureValidationError("in-memory image exceeds the pixel limit")
    value = _as_rgb_float(item)
    payload = (
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\0"
        + value.tobytes(order="C")
    )
    return (
        _sha256_bytes(payload),
        "test-only:in-memory",
        item,
        int(value.shape[1]),
        int(value.shape[0]),
        len(payload),
    )


def _source_hash(item: Image.Image | np.ndarray | Path | str) -> tuple[str, str]:
    digest, path, _image, _width, _height, _bytes = _source_snapshot(item)
    return digest, path


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, array in sorted(arrays.items()):
                buffer = BytesIO()
                np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=_NPZ_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(
                    info,
                    buffer.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    except OSError as error:
        raise FeatureValidationError(f"cannot write embedding cache: {path}") from error


def _validate_output_path(path: Path, name: str) -> Path:
    if not isinstance(path, Path):
        raise FeatureValidationError(f"{name} must be a Path")
    candidate = path if path.is_absolute() else Path.cwd() / path
    current = Path(candidate.anchor)
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for index, part in enumerate(parts):
        current /= part
        is_final = index == len(parts) - 1
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise FeatureValidationError(f"{name} contains a symlink")
        if not is_final and not stat.S_ISDIR(mode):
            raise FeatureValidationError(f"{name} has a non-directory ancestor")
        if is_final and current.exists() and not stat.S_ISREG(mode):
            raise FeatureValidationError(f"{name} must be a regular file")
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FeatureValidationError(f"cannot create {name} parent") from error
    current = Path(candidate.anchor)
    parent_parts = (
        candidate.parent.parts[1:]
        if candidate.parent.anchor
        else candidate.parent.parts
    )
    for part in parent_parts:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise FeatureValidationError(f"cannot inspect {name} parent") from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise FeatureValidationError(f"{name} parent is not a real directory")
    return candidate


def _validate_input_file_path(path: Path, name: str) -> Path:
    """Validate a read-only file without following symlink components."""

    if not isinstance(path, Path):
        raise FeatureValidationError(f"{name} must be a Path")
    candidate = path if path.is_absolute() else Path.cwd() / path
    current = Path(candidate.anchor)
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for index, part in enumerate(parts):
        current /= part
        is_final = index == len(parts) - 1
        try:
            mode = current.lstat().st_mode
        except (FileNotFoundError, OSError) as error:
            raise FeatureValidationError(f"{name} is missing") from error
        if stat.S_ISLNK(mode):
            raise FeatureValidationError(f"{name} contains a symlink")
        if is_final and not stat.S_ISREG(mode):
            raise FeatureValidationError(f"{name} must be a regular file")
        if not is_final and not stat.S_ISDIR(mode):
            raise FeatureValidationError(f"{name} has a non-directory ancestor")
    return candidate


_FROZEN_LOCAL_EXECUTION = _local_execution_snapshot()


def _write_atomic_cache(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        _write_deterministic_npz(temporary, arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except FeatureValidationError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise FeatureValidationError(
            f"cannot atomically write embedding cache: {path}"
        ) from error


def _stage_cache(
    path: Path, arrays: Mapping[str, np.ndarray]
) -> tuple[Path, bytes, str]:
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        _write_deterministic_npz(temporary, arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        snapshot = _read_file_snapshot(
            temporary, "staged embedding cache", max_bytes=_MAX_CACHE_BYTES
        )
        return temporary, snapshot, _sha256_bytes(snapshot)
    except FeatureValidationError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise FeatureValidationError(f"cannot stage embedding cache: {path}") from error


def _write_atomic_text(path: Path, text: str) -> None:
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise FeatureValidationError(
            f"cannot atomically write provenance: {path}"
        ) from error


def _stage_text(path: Path, text: str) -> Path:
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise FeatureValidationError(f"cannot stage provenance: {path}") from error


def _write_verified_stage_snapshot(
    target: Path, snapshot: bytes, name: str, token: str
) -> tuple[Path, tuple[int, int, int]]:
    temporary = _transaction_stage_path(target, token)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    created = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        created = True
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise FeatureValidationError(
                "embedding transaction stage ownership is invalid"
            )
        remaining = memoryview(snapshot)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        return temporary, (info.st_dev, info.st_ino, info.st_uid)
    except (FeatureValidationError, OSError, TypeError, ValueError) as error:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            temporary.unlink(missing_ok=True)
        raise FeatureValidationError(
            f"cannot bind {name} to the transaction snapshot"
        ) from error


def _pair_transaction_paths(
    cache_path: Path, provenance_path: Path
) -> tuple[Path, tuple[Path, ...]]:
    identity = _sha256_bytes(
        _canonical_json(
            [
                str(cache_path.resolve(strict=False)),
                str(provenance_path.resolve(strict=False)),
            ]
        )
    )[:24]
    journal = cache_path.parent / f".{cache_path.name}.{identity}.transaction.json"
    locks = tuple(
        sorted(
            {
                parent / f".embedding-cache-pair.{identity}.lock"
                for parent in (cache_path.parent, provenance_path.parent)
            },
            key=str,
        )
    )
    return journal, locks


def _regular_or_missing(path: Path, name: str) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    except OSError as error:
        raise FeatureValidationError(f"cannot inspect {name}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise FeatureValidationError(f"{name} must be a real regular file")
    return True


@contextmanager
def _pair_transaction_lock(cache_path: Path, provenance_path: Path):
    _journal, lock_paths = _pair_transaction_paths(cache_path, provenance_path)
    descriptors: list[int] = []
    with _PAIR_PUBLISH_THREAD_LOCK:
        try:
            for lock_path in lock_paths:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                flags = (
                    os.O_RDWR
                    | os.O_CREAT
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(lock_path, flags, 0o600)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    os.close(descriptor)
                    raise FeatureValidationError(
                        "embedding transaction lock must be a regular file"
                    )
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                descriptors.append(descriptor)
            yield
        except FeatureValidationError:
            raise
        except OSError as error:
            raise FeatureValidationError(
                "cannot acquire embedding cache transaction lock"
            ) from error
        finally:
            for descriptor in reversed(descriptors):
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)


def _transaction_backup_path(target: Path, token: str) -> Path:
    return target.with_name(f".{target.name}.{token}.backup")


def _transaction_stage_path(target: Path, token: str) -> Path:
    return target.with_name(f".{target.name}.verified.{token}.stage")


def _write_transaction_journal(path: Path, transaction: Mapping[str, Any]) -> None:
    _write_atomic_text(
        path,
        json.dumps(
            transaction,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def _load_transaction_journal(
    journal: Path, cache_path: Path, provenance_path: Path
) -> dict[str, Any]:
    payload = _read_file_snapshot(
        journal, "embedding transaction journal", max_bytes=64 * 1024
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise FeatureValidationError(
            "embedding transaction journal is malformed"
        ) from error
    keys = {
        "cache_backup",
        "cache_had_old",
        "cache_path",
        "cache_stage_device",
        "cache_stage_inode",
        "cache_stage_owner_uid",
        "cache_stage",
        "cache_target_sha256",
        "owner_pid",
        "owner_started_ns",
        "owner_token",
        "phase",
        "provenance_backup",
        "provenance_had_old",
        "provenance_path",
        "provenance_stage_device",
        "provenance_stage_inode",
        "provenance_stage_owner_uid",
        "provenance_stage",
        "provenance_target_sha256",
        "schema_version",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise FeatureValidationError(
            "embedding transaction journal fields are missing or unknown"
        )
    token = value["owner_token"]
    if (
        value["schema_version"] != 2
        or type(token) is not str
        or len(token) != 32
        or any(character not in "0123456789abcdef" for character in token)
        or type(value["owner_pid"]) is not int
        or value["owner_pid"] <= 0
        or type(value["owner_started_ns"]) is not int
        or value["owner_started_ns"] <= 0
        or type(value["cache_had_old"]) is not bool
        or type(value["provenance_had_old"]) is not bool
        or not _is_lower_sha256(value["cache_target_sha256"])
        or not _is_lower_sha256(value["provenance_target_sha256"])
        or value["phase"]
        not in {
            "prepared",
            "cache_backed_up",
            "pair_backed_up",
            "cache_published",
            "committed",
        }
    ):
        raise FeatureValidationError("embedding transaction owner or phase is invalid")
    expected = {
        "cache_path": str(cache_path.resolve(strict=False)),
        "provenance_path": str(provenance_path.resolve(strict=False)),
        "cache_backup": str(
            _transaction_backup_path(cache_path, token).resolve(strict=False)
        ),
        "provenance_backup": str(
            _transaction_backup_path(provenance_path, token).resolve(strict=False)
        ),
    }
    if any(value[name] != expected_value for name, expected_value in expected.items()):
        raise FeatureValidationError("embedding transaction journal path mismatch")
    for stage_name, target in (
        ("cache_stage", cache_path),
        ("provenance_stage", provenance_path),
    ):
        raw = value[stage_name]
        prefix = stage_name.removesuffix("_stage")
        expected_stage = _transaction_stage_path(target, token).resolve(strict=False)
        identity = (
            value[f"{prefix}_stage_device"],
            value[f"{prefix}_stage_inode"],
            value[f"{prefix}_stage_owner_uid"],
        )
        if (
            type(raw) is not str
            or Path(raw) != expected_stage
            or any(type(item) is not int or item < 0 for item in identity)
            or identity[1] <= 0
            or identity[2] != os.getuid()
        ):
            raise FeatureValidationError("embedding transaction stage path mismatch")
        _validate_transaction_stage(value, prefix)
    return value


def _validate_transaction_stage(
    transaction: Mapping[str, Any], prefix: str
) -> bool:
    path = Path(transaction[f"{prefix}_stage"])
    if not _regular_or_missing(path, "embedding transaction stage"):
        return False
    try:
        info = path.lstat()
    except OSError as error:
        raise FeatureValidationError(
            "cannot inspect embedding transaction stage"
        ) from error
    expected_identity = (
        transaction[f"{prefix}_stage_device"],
        transaction[f"{prefix}_stage_inode"],
        transaction[f"{prefix}_stage_owner_uid"],
    )
    observed_identity = (info.st_dev, info.st_ino, info.st_uid)
    if (
        observed_identity != expected_identity
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise FeatureValidationError(
            "embedding transaction stage ownership changed"
        )
    limit = _MAX_CACHE_BYTES if prefix == "cache" else _MAX_PROVENANCE_BYTES
    snapshot = _read_file_snapshot(
        path, f"{prefix} embedding transaction stage", max_bytes=limit
    )
    if _sha256_bytes(snapshot) != transaction[f"{prefix}_target_sha256"]:
        raise FeatureValidationError(
            "embedding transaction stage digest changed"
        )
    return True


def _cleanup_transaction_artifacts(
    transaction: Mapping[str, Any], journal: Path, *, remove_stages: bool
) -> None:
    paths = [Path(transaction["cache_backup"]), Path(transaction["provenance_backup"])]
    if remove_stages:
        for prefix in ("cache", "provenance"):
            if _validate_transaction_stage(transaction, prefix):
                paths.append(Path(transaction[f"{prefix}_stage"]))
    for path in paths:
        if _regular_or_missing(path, "embedding transaction artifact"):
            path.unlink()
    if _regular_or_missing(journal, "embedding transaction journal"):
        journal.unlink()


def _rollback_transaction(
    transaction: Mapping[str, Any], journal: Path, *, remove_stages: bool
) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []
    for prefix in ("cache", "provenance"):
        target = Path(transaction[f"{prefix}_path"])
        backup = Path(transaction[f"{prefix}_backup"])
        had_old = transaction[f"{prefix}_had_old"]
        try:
            backup_exists = _regular_or_missing(
                backup, "embedding transaction backup"
            )
            target_exists = _regular_or_missing(
                target, "embedding transaction target"
            )
            if backup_exists:
                if target_exists:
                    target.unlink()
                os.replace(backup, target)
            elif not had_old and target_exists:
                target.unlink()
            elif had_old and not target_exists:
                raise FeatureValidationError(
                    "embedding transaction lost its old target and backup"
                )
        except (FeatureValidationError, OSError, TypeError, ValueError) as error:
            errors.append(error)
    if errors:
        return tuple(errors)
    try:
        _cleanup_transaction_artifacts(
            transaction, journal, remove_stages=remove_stages
        )
    except (FeatureValidationError, OSError) as error:
        return (error,)
    return ()


def _recover_embedding_cache_pair_locked(
    cache_path: Path, provenance_path: Path
) -> None:
    journal, _locks = _pair_transaction_paths(cache_path, provenance_path)
    journal_exists = _regular_or_missing(journal, "embedding transaction journal")
    backup_candidates = tuple(cache_path.parent.glob(f".{cache_path.name}.*.backup")) + tuple(
        provenance_path.parent.glob(f".{provenance_path.name}.*.backup")
    )
    for candidate in backup_candidates:
        _regular_or_missing(candidate, "embedding transaction backup")
    if not journal_exists:
        if backup_candidates:
            raise FeatureValidationError(
                "unknown embedding transaction backup requires manual recovery"
            )
        return
    transaction = _load_transaction_journal(journal, cache_path, provenance_path)
    expected_backups = {
        Path(transaction["cache_backup"]),
        Path(transaction["provenance_backup"]),
    }
    if set(backup_candidates) - expected_backups:
        raise FeatureValidationError("unknown embedding transaction backup")
    if transaction["phase"] == "committed":
        if not _regular_or_missing(cache_path, "embedding cache") or not _regular_or_missing(
            provenance_path, "embedding provenance"
        ):
            raise FeatureValidationError(
                "committed embedding transaction is missing a target"
            )
        observed = (
            _sha256_bytes(
                _read_file_snapshot(
                    cache_path, "committed embedding cache", max_bytes=_MAX_CACHE_BYTES
                )
            ),
            _sha256_bytes(
                _read_file_snapshot(
                    provenance_path,
                    "committed embedding provenance",
                    max_bytes=_MAX_PROVENANCE_BYTES,
                )
            ),
        )
        expected = (
            transaction["cache_target_sha256"],
            transaction["provenance_target_sha256"],
        )
        if observed != expected:
            raise FeatureValidationError(
                "committed embedding pair generation digest mismatch"
            )
        _cleanup_transaction_artifacts(transaction, journal, remove_stages=True)
        return
    rollback_errors = _rollback_transaction(
        transaction, journal, remove_stages=True
    )
    if rollback_errors:
        raise FeatureValidationError(
            "embedding transaction recovery could not restore the old pair"
        ) from rollback_errors[0]


def _recover_embedding_cache_pair(cache_path: Path, provenance_path: Path) -> None:
    with _pair_transaction_lock(cache_path, provenance_path):
        _recover_embedding_cache_pair_locked(cache_path, provenance_path)


def _publish_staged_pair_transaction_impl(
    cache_stage: Path,
    provenance_stage: Path,
    cache_path: Path,
    provenance_path: Path,
    *,
    expected_cache_sha256: str | None = None,
    expected_provenance_sha256: str | None = None,
) -> None:
    cache_stage = Path(cache_stage)
    provenance_stage = Path(provenance_stage)
    cache_path = Path(cache_path)
    provenance_path = Path(provenance_path)
    with _pair_transaction_lock(cache_path, provenance_path):
        _recover_embedding_cache_pair_locked(cache_path, provenance_path)
        input_stages = (cache_stage, provenance_stage)
        targets = (cache_path, provenance_path)
        for stage, target in zip(input_stages, targets, strict=True):
            if stage.parent != target.parent or not _regular_or_missing(
                stage, "embedding transaction stage"
            ):
                raise FeatureValidationError(
                    "embedding transaction stage must be a regular sibling file"
                )
        verified: list[tuple[bytes, str]] = []
        for stage, expected, name, limit in (
            (
                cache_stage,
                expected_cache_sha256,
                "staged embedding cache",
                _MAX_CACHE_BYTES,
            ),
            (
                provenance_stage,
                expected_provenance_sha256,
                "staged embedding provenance",
                _MAX_PROVENANCE_BYTES,
            ),
        ):
            if expected is not None:
                snapshot = _read_file_snapshot(stage, name, max_bytes=limit)
                if _sha256_bytes(snapshot) != expected:
                    raise FeatureValidationError(
                        f"{name} digest disagrees with writer authority"
                    )
                verified.append((snapshot, name))
        if len(verified) != 2:
            raise FeatureValidationError(
                "embedding transaction requires writer-bound staged digests"
            )
        token = secrets.token_hex(16)
        owned_stages: list[tuple[Path, tuple[int, int, int]]] = []
        try:
            for target, (snapshot, name) in zip(targets, verified, strict=True):
                owned_stages.append(
                    _write_verified_stage_snapshot(target, snapshot, name, token)
                )
            for stage in input_stages:
                stage.unlink()
        except (FeatureValidationError, OSError) as error:
            for owned_stage, _identity in owned_stages:
                owned_stage.unlink(missing_ok=True)
            raise FeatureValidationError(
                "cannot bind embedding transaction to staged snapshots"
            ) from error
        (cache_stage, cache_stage_identity), (
            provenance_stage,
            provenance_stage_identity,
        ) = owned_stages
        cache_had_old = _regular_or_missing(cache_path, "embedding cache")
        provenance_had_old = _regular_or_missing(
            provenance_path, "embedding provenance"
        )
        journal, _locks = _pair_transaction_paths(cache_path, provenance_path)
        transaction: dict[str, Any] = {
            "cache_backup": str(
                _transaction_backup_path(cache_path, token).resolve(strict=False)
            ),
            "cache_had_old": cache_had_old,
            "cache_path": str(cache_path.resolve(strict=False)),
            "cache_stage": str(cache_stage.resolve(strict=False)),
            "cache_stage_device": cache_stage_identity[0],
            "cache_stage_inode": cache_stage_identity[1],
            "cache_stage_owner_uid": cache_stage_identity[2],
            "cache_target_sha256": expected_cache_sha256,
            "owner_pid": os.getpid(),
            "owner_started_ns": time.time_ns(),
            "owner_token": token,
            "phase": "prepared",
            "provenance_backup": str(
                _transaction_backup_path(provenance_path, token).resolve(strict=False)
            ),
            "provenance_had_old": provenance_had_old,
            "provenance_path": str(provenance_path.resolve(strict=False)),
            "provenance_stage": str(provenance_stage.resolve(strict=False)),
            "provenance_stage_device": provenance_stage_identity[0],
            "provenance_stage_inode": provenance_stage_identity[1],
            "provenance_stage_owner_uid": provenance_stage_identity[2],
            "provenance_target_sha256": expected_provenance_sha256,
            "schema_version": 2,
        }
        for backup in (
            Path(transaction["cache_backup"]),
            Path(transaction["provenance_backup"]),
        ):
            if _regular_or_missing(backup, "embedding transaction backup"):
                raise FeatureValidationError(
                    "embedding transaction backup already exists"
                )
        _write_transaction_journal(journal, transaction)
        try:
            if cache_had_old:
                os.replace(cache_path, Path(transaction["cache_backup"]))
            transaction["phase"] = "cache_backed_up"
            _write_transaction_journal(journal, transaction)
            if provenance_had_old:
                os.replace(provenance_path, Path(transaction["provenance_backup"]))
            transaction["phase"] = "pair_backed_up"
            _write_transaction_journal(journal, transaction)
            os.replace(cache_stage, cache_path)
            transaction["phase"] = "cache_published"
            _write_transaction_journal(journal, transaction)
            os.replace(provenance_stage, provenance_path)
            transaction["phase"] = "committed"
            _write_transaction_journal(journal, transaction)
            for directory in {cache_path.parent, provenance_path.parent}:
                directory_fd = os.open(
                    directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            _cleanup_transaction_artifacts(transaction, journal, remove_stages=True)
        except (FeatureValidationError, OSError, TypeError, ValueError) as error:
            rollback_errors = _rollback_transaction(
                transaction, journal, remove_stages=True
            )
            if rollback_errors:
                raise FeatureValidationError(
                    "cannot publish embedding cache pair; rollback requires recovery"
                ) from error
            raise FeatureValidationError(
                "cannot publish embedding cache pair; old pair was restored"
            ) from error


def _publish_staged_pair_transaction(*_args: object, **_kwargs: object) -> None:
    raise FeatureValidationError(
        "embedding pair transaction requires cache writer authority"
    )


def _embedding_cache_constructor_runtime() -> tuple[
    Callable[[object], bool], Callable[[], object]
]:
    capability = object()

    def validate(candidate: object) -> bool:
        return candidate is capability

    def take_for_writer_initialization() -> object:
        return capability

    return validate, take_for_writer_initialization


_validate_embedding_cache_constructor_capability, _take_cache_writer_capability = (
    _embedding_cache_constructor_runtime()
)
del _embedding_cache_constructor_runtime


@dataclass(frozen=True)
class EmbeddingCache:
    sample_ids: np.ndarray
    embeddings: np.ndarray
    source_sha256: tuple[str, ...]
    weight_sha256: str
    transform_sha256: str
    embedding_cache_sha256: str | None
    provenance: Mapping[str, Any]
    _writer_capability: InitVar[object | None] = None
    _writer_provenance_sha256: InitVar[str | None] = None
    _writer_cache_output_path: InitVar[Path | str | None] = None
    _writer_provenance_output_path: InitVar[Path | str | None] = None
    state_sha256: str = field(init=False)
    _sample_ids_backing: bytes = field(init=False, repr=False)
    _embeddings_backing: bytes = field(init=False, repr=False)
    _provenance_sha256: str | None = field(init=False, repr=False)
    _cache_output_path: str | None = field(init=False, repr=False)
    _provenance_output_path: str | None = field(init=False, repr=False)

    def __post_init__(
        self,
        _writer_capability: object | None,
        _writer_provenance_sha256: str | None,
        _writer_cache_output_path: Path | str | None,
        _writer_provenance_output_path: Path | str | None,
        _capability_validator: Callable[[object], bool] = (
            _validate_embedding_cache_constructor_capability  # noqa: RUF033
        ),
    ) -> None:
        sample_ids = np.asarray(self.sample_ids)
        if sample_ids.ndim != 1 or len(sample_ids) == 0:
            raise FeatureValidationError("embedding cache sample IDs are invalid")
        if len(sample_ids) > _MAX_EMBEDDING_SAMPLES:
            raise FeatureValidationError(
                "embedding cache exceeds the registered 276-row limit"
            )
        if sample_ids.dtype.kind != "U":
            raise FeatureValidationError("embedding cache sample IDs must be strings")
        sample_ids = sample_ids.astype(sample_ids.dtype, copy=True)
        if any(type(value) is not str or not value for value in sample_ids.tolist()):
            raise FeatureValidationError("embedding cache sample IDs must be strings")
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise FeatureValidationError("embedding cache sample IDs must be unique")
        embeddings = np.asarray(self.embeddings)
        if np.iscomplexobj(embeddings) or embeddings.dtype != np.dtype(np.float32):
            raise FeatureValidationError("embedding cache embeddings must be float32")
        if embeddings.shape != (len(sample_ids), EMBEDDING_DIMENSION):
            raise FeatureValidationError(
                "embedding cache embeddings have invalid shape"
            )
        if not np.all(np.isfinite(embeddings)):
            raise FeatureValidationError("embedding cache embeddings must be finite")
        source_hashes = tuple(str(value) for value in self.source_sha256)
        if len(source_hashes) != len(sample_ids) or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in source_hashes
        ):
            raise FeatureValidationError("embedding cache source hashes are invalid")
        if not isinstance(self.provenance, Mapping):
            raise FeatureValidationError("embedding cache provenance must be a mapping")
        _validate_cache_provenance(
            sample_ids,
            embeddings,
            source_hashes,
            self.weight_sha256,
            self.transform_sha256,
            self.embedding_cache_sha256,
            self.provenance,
        )
        if not _capability_validator(_writer_capability):
            raise FeatureValidationError(
                "embedding cache constructor requires the cache writer capability"
            )
        if _writer_provenance_sha256 is not None and not _is_lower_sha256(
            _writer_provenance_sha256
        ):
            raise FeatureValidationError("embedding provenance writer digest is invalid")
        output_values = (_writer_cache_output_path, _writer_provenance_output_path)
        if (output_values[0] is None) != (output_values[1] is None):
            raise FeatureValidationError(
                "embedding writer output destinations must be paired"
            )
        output_paths = tuple(
            None if value is None else str(Path(value).resolve(strict=False))
            for value in output_values
        )
        frozen_provenance = _deep_freeze(self.provenance)
        immutable_ids, ids_backing = _immutable_array(sample_ids)
        immutable_embeddings, embeddings_backing = _immutable_array(
            embeddings, dtype=np.float32
        )
        object.__setattr__(self, "sample_ids", immutable_ids)
        object.__setattr__(self, "embeddings", immutable_embeddings)
        object.__setattr__(self, "source_sha256", source_hashes)
        object.__setattr__(self, "provenance", frozen_provenance)
        object.__setattr__(self, "_sample_ids_backing", ids_backing)
        object.__setattr__(self, "_embeddings_backing", embeddings_backing)
        object.__setattr__(
            self,
            "_provenance_sha256",
            _writer_provenance_sha256,
        )
        object.__setattr__(self, "_cache_output_path", output_paths[0])
        object.__setattr__(self, "_provenance_output_path", output_paths[1])
        object.__setattr__(
            self,
            "state_sha256",
            _state_hash(
                immutable_ids,
                immutable_embeddings,
                source_hashes,
                self.weight_sha256,
                self.transform_sha256,
                self.embedding_cache_sha256,
                frozen_provenance,
            ),
        )

    def __copy__(self) -> EmbeddingCache:
        raise TypeError("writer-issued embedding caches cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> EmbeddingCache:
        raise TypeError("writer-issued embedding caches cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("writer-issued embedding caches cannot be serialized")


def _embedding_cache_content_digest(cache: EmbeddingCache) -> str:
    return _state_hash(
        cache.sample_ids,
        cache.embeddings,
        cache.source_sha256,
        cache.weight_sha256,
        cache.transform_sha256,
        cache.embedding_cache_sha256,
        cache.provenance,
        _sha256_bytes(cache._sample_ids_backing),
        _sha256_bytes(cache._embeddings_backing),
        cache._provenance_sha256,
        cache._cache_output_path,
        cache._provenance_output_path,
    )


def _embedding_cache_writer_runtime(
    issuer_codes: frozenset[types.CodeType],
    capability: object,
) -> tuple[
    Callable[..., EmbeddingCache],
    Callable[[EmbeddingCache], str],
]:
    issued: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}
    lock = threading.RLock()

    def issue(**fields: Any) -> EmbeddingCache:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if caller is None or caller.f_code not in issuer_codes:
            raise FeatureValidationError(
                "embedding cache issuance requires the cache writer"
            )
        provenance_sha256 = fields.pop("provenance_sha256", None)
        cache_output_path = fields.pop("cache_output_path", None)
        provenance_output_path = fields.pop("provenance_output_path", None)
        cache = EmbeddingCache(
            **fields,
            _writer_capability=capability,
            _writer_provenance_sha256=provenance_sha256,
            _writer_cache_output_path=cache_output_path,
            _writer_provenance_output_path=provenance_output_path,
        )
        digest = _embedding_cache_content_digest(cache)
        identity = id(cache)

        def discard(reference: weakref.ReferenceType[Any]) -> None:
            with lock:
                current = issued.get(identity)
                if current is not None and current[0] is reference:
                    issued.pop(identity, None)

        reference = weakref.ref(cache, discard)
        with lock:
            issued[identity] = (reference, digest)
        return cache

    def issued_digest(cache: EmbeddingCache) -> str:
        with lock:
            record = issued.get(id(cache))
        if record is None or record[0]() is not cache:
            raise FeatureValidationError(
                "embedding cache is not issued by the cache writer authority"
            )
        return record[1]

    return issue, issued_digest


def validate_embedding_cache_state(cache: EmbeddingCache) -> bool:
    """Reject array, provenance, or state-hash tampering after construction."""

    if not isinstance(cache, EmbeddingCache):
        raise FeatureValidationError("embedding cache has an invalid type")
    issued_digest = _issued_cache_digest(cache)
    try:
        if cache.sample_ids.flags.writeable or cache.embeddings.flags.writeable:
            raise FeatureValidationError("embedding cache arrays are writable")
        if cache.sample_ids.tobytes(order="C") != cache._sample_ids_backing:
            raise FeatureValidationError("embedding cache sample IDs changed")
        if cache.embeddings.tobytes(order="C") != cache._embeddings_backing:
            raise FeatureValidationError("embedding cache embeddings changed")
        if not (
            cache._provenance_sha256 is None
            or _is_lower_sha256(cache._provenance_sha256)
        ):
            raise FeatureValidationError("embedding cache provenance digest changed")
        current = _state_hash(
            cache.sample_ids,
            cache.embeddings,
            cache.source_sha256,
            cache.weight_sha256,
            cache.transform_sha256,
            cache.embedding_cache_sha256,
            cache.provenance,
        )
        _validate_cache_provenance(
            cache.sample_ids,
            cache.embeddings,
            cache.source_sha256,
            cache.weight_sha256,
            cache.transform_sha256,
            cache.embedding_cache_sha256,
            cache.provenance,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise FeatureValidationError("embedding cache state is malformed") from error
    try:
        if current != cache.state_sha256:
            raise FeatureValidationError("embedding cache state hash changed")
        if _embedding_cache_content_digest(cache) != issued_digest:
            raise FeatureValidationError("embedding cache issued content changed")
    except AttributeError as error:
        raise FeatureValidationError("embedding cache state is malformed") from error
    return True


def _pair_transaction_runtime(
    implementation: Callable[..., None],
    issued_digest: Callable[[EmbeddingCache], str],
    content_digest: Callable[[EmbeddingCache], str],
) -> Callable[..., None]:
    def publish(
        cache_stage: Path,
        provenance_stage: Path,
        cache_path: Path,
        provenance_path: Path,
        *,
        cache_authority: EmbeddingCache | None = None,
    ) -> None:
        if cache_authority is None:
            raise FeatureValidationError(
                "embedding pair publication requires cache writer authority"
            )
        try:
            registered_digest = issued_digest(cache_authority)
            if content_digest(cache_authority) != registered_digest:
                raise FeatureValidationError(
                    "embedding cache content disagrees with writer authority"
                )
            if cache_authority.provenance["artifact_eligible"] is not True:
                raise FeatureValidationError(
                    "embedding pair publication requires artifact writer authority"
                )
            expected_destinations = (
                cache_authority._cache_output_path,
                cache_authority._provenance_output_path,
            )
            observed_destinations = (
                str(Path(cache_path).resolve(strict=False)),
                str(Path(provenance_path).resolve(strict=False)),
            )
            if expected_destinations != observed_destinations:
                raise FeatureValidationError(
                    "embedding publication destination disagrees with writer authority"
                )
            cache_sha256 = cache_authority.embedding_cache_sha256
            provenance_sha256 = cache_authority._provenance_sha256
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise FeatureValidationError(
                "embedding pair publication requires issued writer authority"
            ) from error
        if cache_sha256 is None or provenance_sha256 is None:
            raise FeatureValidationError(
                "embedding writer authority is not bound to artifact digests"
            )
        implementation(
            cache_stage,
            provenance_stage,
            cache_path,
            provenance_path,
            expected_cache_sha256=cache_sha256,
            expected_provenance_sha256=provenance_sha256,
        )

    return publish


_CACHE_PROVENANCE_KEYS = frozenset(
    {
        "artifact_eligible",
        "array_shapes",
        "architecture_sha256",
        "data_authority_state_sha256",
        "data_authority_type",
        "embedding_cache_sha256",
        "embedding_dimension",
        "encoder",
        "frozen",
        "model_state_sha256",
        "preprocessing_sha256",
        "preprocessing_spec",
        "runtime",
        "schema_version",
        "source_hashes",
        "source_sha256",
        "sources",
        "torch_version",
        "torchvision_version",
        "transform_sha256",
        "weight_path",
        "weights",
        "weights_bytes",
        "weights_filename",
        "weights_sha256",
        "weights_url",
        "device",
        "batch_size",
    }
)


def _strict_equal(actual: Any, expected: Any) -> bool:
    return type(actual) is type(expected) and actual == expected


def _is_lower_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_cache_provenance(
    sample_ids: np.ndarray,
    embeddings: np.ndarray,
    source_sha256: tuple[str, ...],
    weight_sha256: str,
    transform_sha256: str,
    embedding_cache_sha256: str | None,
    provenance: Mapping[str, Any],
) -> None:
    if not isinstance(provenance, Mapping) or set(provenance) != _CACHE_PROVENANCE_KEYS:
        raise FeatureValidationError(
            "embedding cache provenance fields are missing or unknown"
        )
    if not _strict_equal(provenance["schema_version"], 1):
        raise FeatureValidationError("embedding cache provenance schema mismatch")
    if not _strict_equal(provenance["embedding_dimension"], EMBEDDING_DIMENSION):
        raise FeatureValidationError("embedding cache dimension provenance mismatch")
    if not _strict_equal(provenance["encoder"], "torchvision_resnet18"):
        raise FeatureValidationError("embedding cache encoder provenance mismatch")
    if provenance["frozen"] is not True or type(provenance["frozen"]) is not bool:
        raise FeatureValidationError("embedding cache frozen provenance mismatch")
    if type(provenance["artifact_eligible"]) is not bool:
        raise FeatureValidationError("embedding cache artifact eligibility is invalid")
    if provenance["data_authority_type"] not in {
        "V3Data",
        "V3DataView",
        "test_only",
    }:
        raise FeatureValidationError("embedding cache data authority is invalid")
    if not _is_lower_sha256(provenance["data_authority_state_sha256"]):
        raise FeatureValidationError("embedding cache data authority hash is invalid")
    if provenance["artifact_eligible"] != (
        provenance["data_authority_type"] in {"V3Data", "V3DataView"}
    ):
        raise FeatureValidationError("embedding cache artifact authority is inconsistent")
    if (
        not _is_lower_sha256(weight_sha256)
        or weight_sha256 != _FROZEN_RESNET18_WEIGHTS_SHA256
    ):
        raise FeatureValidationError("embedding cache weight hash is invalid")
    if provenance["weights_sha256"] != _FROZEN_RESNET18_WEIGHTS_SHA256:
        raise FeatureValidationError("embedding cache weight hash provenance mismatch")
    if (
        transform_sha256 != EXPECTED_TRANSFORM_SHA
        or provenance["transform_sha256"] != EXPECTED_TRANSFORM_SHA
    ):
        raise FeatureValidationError("embedding cache transform provenance mismatch")
    if provenance["preprocessing_sha256"] != EXPECTED_TRANSFORM_SHA:
        raise FeatureValidationError(
            "embedding cache preprocessing provenance mismatch"
        )
    try:
        if _canonical_json(provenance["preprocessing_spec"]) != _canonical_json(
            _jsonable(TRANSFORM_SPEC)
        ):
            raise FeatureValidationError("embedding cache preprocessing spec mismatch")
    except (TypeError, ValueError) as error:
        raise FeatureValidationError(
            "embedding cache preprocessing spec malformed"
        ) from error
    expected_shapes = {
        "embeddings": list(embeddings.shape),
        "sample_ids": list(sample_ids.shape),
        "schema_version": [1],
    }
    if _canonical_json(provenance["array_shapes"]) != _canonical_json(expected_shapes):
        raise FeatureValidationError("embedding cache array shape provenance mismatch")
    if not (
        embedding_cache_sha256 is None or _is_lower_sha256(embedding_cache_sha256)
    ) or not _strict_equal(
        provenance["embedding_cache_sha256"], embedding_cache_sha256
    ):
        raise FeatureValidationError("embedding cache digest provenance mismatch")
    if provenance["weight_path"] != _FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH.as_posix():
        raise FeatureValidationError("embedding cache weight path provenance mismatch")
    if provenance["weights_filename"] != _FROZEN_RESNET18_WEIGHTS_FILENAME:
        raise FeatureValidationError("embedding cache weight filename mismatch")
    if provenance["weights_url"] != _FROZEN_RESNET18_WEIGHTS_URL:
        raise FeatureValidationError("embedding cache weight URL mismatch")
    if type(provenance["weights_bytes"]) is not int or provenance["weights_bytes"] <= 0:
        raise FeatureValidationError("embedding cache weight byte count is invalid")
    if (
        type(provenance["batch_size"]) is not int
        or provenance["batch_size"] != EMBEDDING_BATCH_SIZE
    ):
        raise FeatureValidationError("embedding cache batch provenance mismatch")
    if type(provenance["device"]) is not str or not provenance["device"]:
        raise FeatureValidationError("embedding cache device provenance is invalid")
    if type(provenance["torch_version"]) is not str or not provenance["torch_version"]:
        raise FeatureValidationError("embedding cache torch version is invalid")
    if (
        type(provenance["torchvision_version"]) is not str
        or not provenance["torchvision_version"]
    ):
        raise FeatureValidationError("embedding cache torchvision version is invalid")
    if provenance["torch_version"] != _EXPECTED_TORCH_VERSION:
        raise FeatureValidationError("embedding cache torch version is not registered")
    if provenance["torchvision_version"] != _EXPECTED_TORCHVISION_VERSION:
        raise FeatureValidationError(
            "embedding cache torchvision version is not registered"
        )
    try:
        import torch
        import torchvision

        _validate_runtime_versions(torch, torchvision)
    except FeatureValidationError as error:
        raise FeatureValidationError(
            "embedding cache runtime version is not registered"
        ) from error
    except Exception as error:
        raise FeatureValidationError(
            "embedding cache runtime version is unavailable"
        ) from error
    if provenance["model_state_sha256"] != _EXPECTED_MODEL_STATE_SHA256:
        raise FeatureValidationError("embedding cache model state authority mismatch")
    if provenance["architecture_sha256"] != _EXPECTED_ARCHITECTURE_SHA256:
        raise FeatureValidationError("embedding cache architecture authority mismatch")
    runtime = provenance["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != set(_RUNTIME_CONTRACT):
        raise FeatureValidationError("embedding cache runtime provenance mismatch")
    if any(
        not _strict_equal(runtime[name], expected)
        for name, expected in _RUNTIME_CONTRACT.items()
    ):
        raise FeatureValidationError("embedding cache runtime provenance mismatch")
    records = provenance["sources"]
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or len(records) != len(sample_ids)
    ):
        raise FeatureValidationError("embedding cache source provenance mismatch")
    expected_ids = sample_ids.tolist()
    observed_hashes: list[str] = []
    for expected_id, record in zip(expected_ids, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "dataset_id",
            "height",
            "sample_id",
            "source_bytes",
            "source_path",
            "source_sha256",
            "width",
        }:
            raise FeatureValidationError("embedding cache source fields mismatch")
        if not _strict_equal(record["sample_id"], expected_id):
            raise FeatureValidationError("embedding cache source IDs mismatch")
        if type(record["source_path"]) is not str or not record["source_path"]:
            raise FeatureValidationError("embedding cache source path is invalid")
        if type(record["dataset_id"]) is not str or not record["dataset_id"]:
            raise FeatureValidationError("embedding cache source domain is invalid")
        for dimension in ("width", "height", "source_bytes"):
            if type(record[dimension]) is not int or record[dimension] <= 0:
                raise FeatureValidationError(
                    "embedding cache source shape or byte count is invalid"
                )
        if not _is_lower_sha256(record["source_sha256"]):
            raise FeatureValidationError("embedding cache source hash is invalid")
        observed_hashes.append(record["source_sha256"])
    if tuple(observed_hashes) != source_sha256:
        raise FeatureValidationError("embedding cache source hash mismatch")
    for field_name in ("source_sha256", "source_hashes"):
        if tuple(provenance[field_name]) != tuple(observed_hashes):
            raise FeatureValidationError("embedding cache source hashes disagree")


def _expected_encoder_provenance(
    weight_snapshot: bytes,
    project_root: Path,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if project_root != _MODULE_REPO_ROOT:
        raise FeatureValidationError("encoder project root is not registered")
    model, torch, torchvision = _load_frozen_model(weight_snapshot, device)
    if _model_type_key(model) != _EXPECTED_MODEL_TYPE:
        raise FeatureValidationError("encoder model type is not registered")
    if _model_state_sha256(model) != _EXPECTED_MODEL_STATE_SHA256:
        raise FeatureValidationError("encoder model state is not registered")
    if _module_graph_sha256(model) != _EXPECTED_ARCHITECTURE_SHA256:
        raise FeatureValidationError("encoder architecture is not registered")
    return {
        "encoder": "torchvision_resnet18",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "frozen": True,
        "transform_sha256": _EXPECTED_TRANSFORM_SHA,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "torchvision_version": str(getattr(torchvision, "__version__", "unknown")),
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_bytes": len(weight_snapshot),
        "weights_filename": _FROZEN_RESNET18_WEIGHTS_FILENAME,
        "weights_sha256": _FROZEN_RESNET18_WEIGHTS_SHA256,
        "weights_url": _FROZEN_RESNET18_WEIGHTS_URL,
        "weight_path": _FROZEN_RESNET18_WEIGHTS_RELATIVE_PATH.as_posix(),
        "device": device,
        "batch_size": batch_size,
        "runtime": dict(_RUNTIME_CONTRACT),
        "model_state_sha256": _EXPECTED_MODEL_STATE_SHA256,
        "architecture_sha256": _EXPECTED_ARCHITECTURE_SHA256,
    }


def _validated_data_authority(
    data: object,
    *,
    _data_types: tuple[type[Any], ...] = (_DATA_V3_DATA, _DATA_V3_DATA_VIEW),
    _validator: Callable[[object], str] = _DATA_AUTHORITY_VALIDATOR,
) -> tuple[str, str]:
    try:
        if type(data) not in _data_types:
            raise FeatureValidationError(
                "embedding data authority must be an exact V3Data or V3DataView"
            )
        state = _validator(data)
    except FeatureValidationError:
        raise
    except Exception as error:
        raise FeatureValidationError(
            "embedding data authority is not issued by the loader"
        ) from error
    return type(data).__name__, state


def _authority_sources(
    data: object, project_root: Path
) -> tuple[tuple[str, ...], list[Image.Image], list[dict[str, Any]]]:
    authority_type, _state = _validated_data_authority(data)
    del authority_type
    if project_root != _MODULE_REPO_ROOT:
        raise FeatureValidationError("embedding project root is not registered")
    try:
        ids = _strict_sample_ids(
            tuple(str(value) for value in data.sample_ids.tolist()),
            "embedding cache sample IDs",
        )
        domains = tuple(str(value) for value in data.dataset_ids.tolist())
        records = tuple(data.cscan_records)
    except (AttributeError, TypeError, ValueError) as error:
        raise FeatureValidationError("embedding data authority is malformed") from error
    if len(ids) > _MAX_EMBEDDING_SAMPLES:
        raise FeatureValidationError("embedding cohort exceeds the registered limit")
    if len(records) != len(ids) or len(domains) != len(ids):
        raise FeatureValidationError("embedding data authority rows are misaligned")
    images: list[Image.Image] = []
    source_records: list[dict[str, Any]] = []
    for sample_id, domain_id, record in zip(ids, domains, records, strict=True):
        try:
            relative = Path(record.relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise FeatureValidationError(
                    "embedding manifest source path is not repo-relative"
                )
            if record.specimen_id != sample_id or record.dataset_id != domain_id:
                raise FeatureValidationError(
                    "embedding specimen-image manifest pairing changed"
                )
            digest, _absolute, image, width, height, source_bytes = _source_snapshot(
                project_root / relative
            )
            if (
                digest != record.sha256
                or width != record.width
                or height != record.height
                or getattr(record, "target_mode", None) != "rgb"
            ):
                raise FeatureValidationError(
                    "embedding specimen-image manifest pairing is invalid"
                )
        except FeatureValidationError:
            raise
        except (AttributeError, TypeError, ValueError) as error:
            raise FeatureValidationError(
                "embedding manifest source record is malformed"
            ) from error
        if not isinstance(image, Image.Image):  # pragma: no cover - source is a path
            raise FeatureValidationError("embedding source snapshot is invalid")
        images.append(image)
        source_records.append(
            {
                "dataset_id": domain_id,
                "height": height,
                "sample_id": sample_id,
                "source_bytes": source_bytes,
                "source_path": relative.as_posix(),
                "source_sha256": digest,
                "width": width,
            }
        )
    return ids, images, source_records


def _test_only_sources(
    values: Sequence[Image.Image | np.ndarray | Path | str],
    sample_ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], list[Image.Image | np.ndarray], list[dict[str, Any]]]:
    ids = _strict_sample_ids(
        sample_ids if sample_ids is not None else (),
        "embedding cache sample IDs",
    )
    if len(ids) != len(values):
        raise FeatureValidationError("embedding cache requires one nonempty ID per image")
    if len(ids) > _MAX_EMBEDDING_SAMPLES:
        raise FeatureValidationError("embedding cohort exceeds the registered limit")
    snapshots: list[Image.Image | np.ndarray] = []
    records: list[dict[str, Any]] = []
    for sample_id, item in zip(ids, values, strict=True):
        digest, source_name, image, width, height, source_bytes = _source_snapshot(item)
        snapshots.append(image)
        records.append(
            {
                "dataset_id": "test_only",
                "height": height,
                "sample_id": sample_id,
                "source_bytes": source_bytes,
                "source_path": source_name,
                "source_sha256": digest,
                "width": width,
            }
        )
    return ids, snapshots, records


def build_embedding_cache(
    images: object | None = None,
    *,
    image_paths: Sequence[Path | str] | None = None,
    sample_ids: Sequence[str] | None = None,
    data_authority: object | None = None,
    test_only: bool = False,
    cache_path: Path | str | None = None,
    provenance_path: Path | str | None = None,
    output_path: Path | str | None = None,
    weight_path: Path | str | None = None,
    project_root: Path | str | None = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> EmbeddingCache:
    """Build a deterministic embedding cache with source and transform hashes."""

    if batch_size != EMBEDDING_BATCH_SIZE:
        raise FeatureValidationError(
            f"batch_size must equal the registered value {EMBEDDING_BATCH_SIZE}"
        )
    validate_transform_spec()
    if cache_path is not None and output_path is not None:
        raise FeatureValidationError("specify only one of cache_path and output_path")
    destination = (
        Path(cache_path if cache_path is not None else output_path)
        if (cache_path is not None or output_path is not None)
        else None
    )
    if provenance_path is not None and destination is None:
        raise FeatureValidationError("provenance_path requires cache_path")
    cache_destination: Path | None = None
    provenance_destination: Path | None = None
    if destination is not None:
        if provenance_path is None:
            raise FeatureValidationError(
                "provenance_path is required when writing an embedding cache"
            )
        cache_destination = _validate_output_path(destination, "cache output")
        provenance_destination = _validate_output_path(
            Path(provenance_path), "provenance output"
        )
        if cache_destination.resolve(strict=False) == provenance_destination.resolve(
            strict=False
        ):
            raise FeatureValidationError("cache and provenance paths must be distinct")
    if images is not None and image_paths is not None:
        raise FeatureValidationError("specify only one of images and image_paths")
    embedded_authority = (
        images if type(images) in (_DATA_V3_DATA, _DATA_V3_DATA_VIEW) else None
    )
    authority = data_authority if data_authority is not None else embedded_authority
    if (
        data_authority is not None
        and embedded_authority is not None
        and data_authority is not embedded_authority
    ):
        raise FeatureValidationError("conflicting embedding data authorities")
    if type(test_only) is not bool:
        raise FeatureValidationError("test_only must be an exact boolean")
    if authority is not None:
        if test_only:
            raise FeatureValidationError("production authority cannot be test-only")
        if images is not None and embedded_authority is None or image_paths is not None:
            raise FeatureValidationError(
                "manifest authority forbids caller-supplied specimen-image paths"
            )
        if sample_ids is not None:
            raise FeatureValidationError(
                "manifest authority forbids caller-supplied sample IDs"
            )
        root = Path(project_root).resolve() if project_root is not None else Path()
        authority_type, authority_state = _validated_data_authority(authority)
        ids, values, source_records = _authority_sources(authority, root)
    else:
        if not test_only:
            raise FeatureValidationError(
                "raw embedding inputs require explicit test-only authority"
            )
        if destination is not None or provenance_path is not None:
            raise FeatureValidationError(
                "test-only embedding inputs cannot write cache artifacts"
            )
        raw_source = (
            images
            if images is not None
            else image_paths
            if image_paths is not None
            else ()
        )
        try:
            declared_length = len(raw_source)  # type: ignore[arg-type]
        except (AttributeError, TypeError):
            declared_length = None
        if declared_length is not None and declared_length > _MAX_EMBEDDING_SAMPLES:
            raise FeatureValidationError(
                "embedding cohort exceeds the registered 276-row limit"
            )
        raw_values: list[Image.Image | np.ndarray | Path | str] = []
        for item in raw_source:  # type: ignore[union-attr]
            raw_values.append(item)
            if len(raw_values) > _MAX_EMBEDDING_SAMPLES:
                raise FeatureValidationError(
                    "embedding cohort exceeds the registered 276-row limit"
                )
        ids, values, source_records = _test_only_sources(raw_values, sample_ids)
        authority_type = "test_only"
        authority_state = _state_hash(
            ids, tuple(record["source_sha256"] for record in source_records)
        )

    encoder = encode_resnet18(
        weight_path=weight_path,
        project_root=project_root,
        device=device,
        batch_size=batch_size,
    )
    if not isinstance(encoder, FrozenResNet18Encoder):  # pragma: no cover - defensive
        raise TypeError("encoder construction failed")
    embeddings = encoder.encode(values)
    arrays = {
        "schema_version": np.asarray([1], dtype=np.int64),
        "sample_ids": np.asarray(ids),
        "embeddings": embeddings.astype(np.float32, copy=False),
    }
    if not np.all(np.isfinite(arrays["embeddings"])):
        raise FeatureValidationError("embedding cache contains non-finite values")
    encoder_provenance = encoder.provenance()
    cache_sha256: str | None = None
    provenance_sha256: str | None = None
    if destination is not None:
        if cache_destination is None or provenance_destination is None:
            raise FeatureValidationError("cache output paths are not registered")
        cache_stage, cache_stage_bytes, cache_sha256 = _stage_cache(
            cache_destination, arrays
        )
    provenance: dict[str, Any] = {
        "artifact_eligible": authority_type != "test_only",
        "array_shapes": {
            name: list(value.shape) for name, value in sorted(arrays.items())
        },
        "embedding_cache_sha256": cache_sha256,
        "data_authority_state_sha256": authority_state,
        "data_authority_type": authority_type,
        "schema_version": 1,
        "sources": source_records,
        "source_sha256": [record["source_sha256"] for record in source_records],
        "source_hashes": [record["source_sha256"] for record in source_records],
        "transform_sha256": TRANSFORM_SHA256,
        "preprocessing_sha256": TRANSFORM_SHA256,
        "preprocessing_spec": _jsonable(TRANSFORM_SPEC),
        **encoder_provenance,
    }
    result: EmbeddingCache | None = None
    if provenance_path is not None:
        if cache_destination is None or provenance_destination is None:
            raise FeatureValidationError("provenance output paths are not registered")
        provenance_text = (
            json.dumps(
                provenance,
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        provenance_bytes = provenance_text.encode("utf-8")
        provenance_sha256 = _sha256_bytes(provenance_bytes)
        result = _issue_embedding_cache(
            sample_ids=np.asarray(ids),
            embeddings=embeddings,
            source_sha256=tuple(
                record["source_sha256"] for record in source_records
            ),
            weight_sha256=_FROZEN_RESNET18_WEIGHTS_SHA256,
            transform_sha256=TRANSFORM_SHA256,
            embedding_cache_sha256=cache_sha256,
            provenance=provenance,
            provenance_sha256=provenance_sha256,
            cache_output_path=cache_destination,
            provenance_output_path=provenance_destination,
        )
        try:
            provenance_stage = _stage_text(provenance_destination, provenance_text)
            _load_embedding_cache_from_writer_snapshots(
                cache_destination,
                provenance_destination,
                cache_snapshot=cache_stage_bytes,
                provenance_snapshot=provenance_bytes,
                weight_path=weight_path,
                project_root=project_root,
                data_authority=authority,
                cache_authority=result,
                device=device,
                batch_size=batch_size,
            )
            _publish_staged_pair(
                cache_stage,
                provenance_stage,
                cache_destination,
                provenance_destination,
                cache_authority=result,
            )
        except FeatureValidationError:
            cache_stage.unlink(missing_ok=True)
            if "provenance_stage" in locals():
                provenance_stage.unlink(missing_ok=True)
            raise
    if result is None:
        result = _issue_embedding_cache(
            sample_ids=np.asarray(ids),
            embeddings=embeddings,
            source_sha256=tuple(
                record["source_sha256"] for record in source_records
            ),
            weight_sha256=_FROZEN_RESNET18_WEIGHTS_SHA256,
            transform_sha256=TRANSFORM_SHA256,
            embedding_cache_sha256=cache_sha256,
            provenance=provenance,
            provenance_sha256=provenance_sha256,
            cache_output_path=cache_destination,
            provenance_output_path=provenance_destination,
        )
    return result


def _validate_npz_array_headers(
    archive: zipfile.ZipFile, members: Sequence[zipfile.ZipInfo]
) -> None:
    headers: dict[str, tuple[tuple[int, ...], bool, np.dtype[Any]]] = {}
    for info in members:
        try:
            with archive.open(info, "r") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_1_0(handle)
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_2_0(handle)
                    )
                else:
                    raise FeatureValidationError(
                        "embedding cache NPY version is unsupported"
                    )
                dtype = np.dtype(dtype)
                element_count = 1
                for dimension in shape:
                    if type(dimension) is not int or dimension < 0:
                        raise FeatureValidationError(
                            "embedding cache NPY shape is invalid"
                        )
                    element_count *= dimension
                    if element_count * dtype.itemsize > _MAX_NPZ_MEMBER_BYTES:
                        raise FeatureValidationError(
                            "embedding cache NPY shape exceeds the size limit"
                        )
                if (
                    dtype.hasobject
                    or fortran_order is not False
                    or handle.tell() + element_count * dtype.itemsize
                    != info.file_size
                ):
                    raise FeatureValidationError(
                        "embedding cache NPY header or payload size is invalid"
                    )
                headers[info.filename] = (shape, fortran_order, dtype)
        except FeatureValidationError:
            raise
        except (EOFError, OSError, TypeError, ValueError, zipfile.BadZipFile) as error:
            raise FeatureValidationError(
                "embedding cache NPY header is malformed"
            ) from error
    if set(headers) != {
        "embeddings.npy",
        "sample_ids.npy",
        "schema_version.npy",
    }:
        raise FeatureValidationError("embedding cache NPY schema is invalid")
    embedding_shape, _, embedding_dtype = headers["embeddings.npy"]
    id_shape, _, id_dtype = headers["sample_ids.npy"]
    schema_shape, _, schema_dtype = headers["schema_version.npy"]
    if (
        len(embedding_shape) != 2
        or embedding_shape[1:] != (EMBEDDING_DIMENSION,)
        or not 1 <= embedding_shape[0] <= _MAX_EMBEDDING_SAMPLES
        or id_shape != (embedding_shape[0],)
        or id_dtype.kind != "U"
        or embedding_dtype != np.dtype(np.float32)
        or schema_shape != (1,)
        or schema_dtype != np.dtype(np.int64)
    ):
        raise FeatureValidationError(
            "embedding cache NPY dtype or shape schema is invalid"
        )


def _load_embedding_cache_impl(
    cache_path: Path | str,
    provenance_path: Path | str | None = None,
    *,
    weight_path: Path | str | None = None,
    project_root: Path | str | None = None,
    data_authority: object | None = None,
    cache_authority: EmbeddingCache | None = None,
    source_paths: Sequence[Path | str] | None = None,
    device: str = "cpu",
    batch_size: int = EMBEDDING_BATCH_SIZE,
    _cache_snapshot: bytes | None = None,
    _provenance_snapshot: bytes | None = None,
    _snapshot_capability: object | None = None,
) -> EmbeddingCache:
    """Load and validate a deterministic embedding cache and its provenance."""

    if provenance_path is None:
        raise FeatureValidationError(
            "embedding provenance is required for verified loading"
        )
    if weight_path is None:
        raise FeatureValidationError(
            "local weight path is required for verified loading"
        )
    if data_authority is None:
        raise FeatureValidationError(
            "external manifest data authority is required for verified loading"
        )
    if source_paths is not None:
        raise FeatureValidationError(
            "caller-supplied source paths cannot replace manifest authority"
        )
    if cache_authority is None:
        raise FeatureValidationError(
            "external cache writer authority is required for verified loading"
        )
    validate_embedding_cache_state(cache_authority)
    if cache_authority.embedding_cache_sha256 is None or cache_authority._provenance_sha256 is None:
        raise FeatureValidationError(
            "cache writer authority is not bound to serialized artifacts"
        )
    if not isinstance(device, str) or not device:
        raise FeatureValidationError("device must be a non-empty string")
    if batch_size != EMBEDDING_BATCH_SIZE:
        raise FeatureValidationError(
            f"batch_size must equal the registered value {EMBEDDING_BATCH_SIZE}"
        )
    validate_transform_spec()
    has_snapshots = _cache_snapshot is not None or _provenance_snapshot is not None
    if has_snapshots:
        if (
            not _is_embedding_snapshot_capability(_snapshot_capability)
            or type(_cache_snapshot) is not bytes
            or type(_provenance_snapshot) is not bytes
        ):
            raise FeatureValidationError(
                "embedding writer snapshots require internal writer authority"
            )
        if len(_cache_snapshot) > _MAX_CACHE_BYTES:
            raise FeatureValidationError("embedding cache exceeds the byte limit")
        if len(_provenance_snapshot) > _MAX_PROVENANCE_BYTES:
            raise FeatureValidationError("embedding provenance exceeds the byte limit")
        cache_bytes = _cache_snapshot
        provenance_bytes = _provenance_snapshot
    else:
        if _snapshot_capability is not None:
            raise FeatureValidationError("embedding snapshot capability is invalid")
        cache_bytes = _read_file_snapshot(
            Path(cache_path), "embedding cache", max_bytes=_MAX_CACHE_BYTES
        )
        provenance_bytes = _read_file_snapshot(
            Path(provenance_path),
            "embedding provenance",
            max_bytes=_MAX_PROVENANCE_BYTES,
        )
    try:
        with zipfile.ZipFile(BytesIO(cache_bytes), "r") as archive:
            members = archive.infolist()
            if (
                len(members) != 3
                or any(
                    info.filename not in {
                        "embeddings.npy",
                        "sample_ids.npy",
                        "schema_version.npy",
                    }
                    or info.is_dir()
                    or info.file_size < 0
                    or info.compress_size < 0
                    or info.file_size > _MAX_NPZ_MEMBER_BYTES
                    or (
                        info.file_size > 0
                        and (
                            info.compress_size == 0
                            or info.file_size
                            > info.compress_size * _MAX_NPZ_COMPRESSION_RATIO
                        )
                    )
                    for info in members
                )
                or sum(info.file_size for info in members)
                > _MAX_NPZ_UNCOMPRESSED_BYTES
            ):
                raise FeatureValidationError(
                    "embedding cache ZIP schema or size limit changed"
                )
            _validate_npz_array_headers(archive, members)
        with np.load(BytesIO(cache_bytes), allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
    except FeatureValidationError:
        raise
    except (AttributeError, OSError, ValueError, KeyError, TypeError) as error:
        raise FeatureValidationError("cannot load embedding cache") from error
    if set(arrays) != {"schema_version", "sample_ids", "embeddings"}:
        raise FeatureValidationError("embedding cache array registry changed")
    if arrays["schema_version"].dtype != np.dtype(np.int64) or arrays[
        "schema_version"
    ].tolist() != [1]:
        raise FeatureValidationError("unsupported embedding cache schema")
    ids = arrays["sample_ids"]
    embeddings = arrays["embeddings"]
    if ids.ndim == 1 and len(ids) > _MAX_EMBEDDING_SAMPLES:
        raise FeatureValidationError(
            "embedding cache exceeds the registered 276-row limit"
        )
    if (
        ids.ndim != 1
        or len(ids) == 0
        or ids.dtype.kind != "U"
        or len(set(ids.tolist())) != len(ids)
        or any(type(value) is not str or not value for value in ids.tolist())
    ):
        raise FeatureValidationError(
            "embedding cache sample IDs are empty or duplicated"
        )
    if (
        embeddings.dtype != np.dtype(np.float32)
        or embeddings.shape != (len(ids), EMBEDDING_DIMENSION)
        or not np.all(np.isfinite(embeddings))
    ):
        raise FeatureValidationError(
            "embedding cache must be finite and 512-dimensional"
        )
    cache_sha256 = _sha256_bytes(cache_bytes)
    provenance_sha256 = _sha256_bytes(provenance_bytes)
    try:
        raw = json.loads(provenance_bytes.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, json.JSONDecodeError) as error:
        raise FeatureValidationError("cannot load embedding provenance") from error
    if not isinstance(raw, dict):
        raise FeatureValidationError("embedding provenance must be an object")
    if set(raw) != _CACHE_PROVENANCE_KEYS:
        raise FeatureValidationError(
            "embedding provenance fields are missing or unknown"
        )
    if not _strict_equal(raw["schema_version"], 1):
        raise FeatureValidationError("unsupported embedding provenance schema")
    if not _strict_equal(raw["embedding_cache_sha256"], cache_sha256):
        raise FeatureValidationError("embedding cache hash mismatch")
    if not _strict_equal(raw["transform_sha256"], TRANSFORM_SHA256):
        raise FeatureValidationError("embedding provenance transform hash mismatch")
    if not _strict_equal(raw["preprocessing_sha256"], TRANSFORM_SHA256) or (
        _canonical_json(raw["preprocessing_spec"])
        != _canonical_json(_jsonable(TRANSFORM_SPEC))
    ):
        raise FeatureValidationError("embedding preprocessing provenance mismatch")
    if raw["array_shapes"] != {
        "embeddings": list(embeddings.shape),
        "sample_ids": list(ids.shape),
        "schema_version": [1],
    }:
        raise FeatureValidationError("embedding cache array shape provenance mismatch")
    if not _strict_equal(raw["embedding_dimension"], EMBEDDING_DIMENSION):
        raise FeatureValidationError("embedding dimension provenance mismatch")
    if not _strict_equal(raw["device"], device) or not _strict_equal(
        raw["batch_size"], batch_size
    ):
        raise FeatureValidationError("embedding device or batch provenance mismatch")
    if (
        not isinstance(raw["runtime"], dict)
        or set(raw["runtime"]) != set(_RUNTIME_CONTRACT)
        or any(
            not _strict_equal(raw["runtime"][name], expected)
            for name, expected in _RUNTIME_CONTRACT.items()
        )
    ):
        raise FeatureValidationError("embedding runtime provenance mismatch")
    _local_weight, local_root, weight_snapshot = _validate_weight_path(
        weight_path, project_root
    )
    expected_encoder = _expected_encoder_provenance(
        weight_snapshot, local_root, device=device, batch_size=batch_size
    )
    for field_name, expected in expected_encoder.items():
        if not _strict_equal(raw[field_name], expected):
            if field_name == "weights_sha256":
                raise FeatureValidationError(
                    "embedding provenance weight hash mismatch"
                )
            raise FeatureValidationError(f"embedding provenance {field_name} mismatch")
    authority_type, authority_state = _validated_data_authority(data_authority)
    authority_ids, _images, authority_records = _authority_sources(
        data_authority, local_root
    )
    if not raw["artifact_eligible"]:
        raise FeatureValidationError("test-only cache cannot be loaded as an artifact")
    if raw["data_authority_type"] != authority_type or raw[
        "data_authority_state_sha256"
    ] != authority_state:
        raise FeatureValidationError("embedding manifest authority provenance mismatch")
    if ids.tolist() != list(authority_ids):
        raise FeatureValidationError("embedding cache IDs disagree with manifest authority")
    records = raw["sources"]
    if not isinstance(records, list) or len(records) != len(ids):
        raise FeatureValidationError("embedding provenance source registry mismatch")
    source_hashes: tuple[str, ...]
    try:
        if any(
            not isinstance(record, dict)
            or set(record)
            != {
                "dataset_id",
                "height",
                "sample_id",
                "source_bytes",
                "source_path",
                "source_sha256",
                "width",
            }
            for record in records
        ):
            raise FeatureValidationError("embedding provenance source fields mismatch")
        if [record["sample_id"] for record in records] != ids.tolist():
            raise FeatureValidationError(
                "embedding provenance source registry mismatch"
            )
        source_hashes = tuple(record["source_sha256"] for record in records)
    except (AttributeError, KeyError, TypeError) as error:
        raise FeatureValidationError(
            "embedding provenance source records are malformed"
        ) from error
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in source_hashes
    ):
        raise FeatureValidationError("embedding provenance source hashes are malformed")
    for field_name in ("source_sha256", "source_hashes"):
        declared = raw[field_name]
        if not isinstance(declared, list) or tuple(declared) != source_hashes:
            raise FeatureValidationError("embedding provenance source hashes disagree")
    if records != authority_records:
        raise FeatureValidationError(
            "embedding specimen-image manifest provenance mismatch"
        )
    observed = tuple(record["source_sha256"] for record in authority_records)
    if observed != source_hashes:
        raise FeatureValidationError("embedding source hash mismatch")
    if (
        cache_authority.embedding_cache_sha256 != cache_sha256
        or cache_authority._provenance_sha256 != provenance_sha256
    ):
        raise FeatureValidationError("cache artifacts disagree with writer authority")
    requested_destinations = (
        str(Path(cache_path).resolve(strict=False)),
        str(Path(provenance_path).resolve(strict=False)),
    )
    writer_destinations = (
        cache_authority._cache_output_path,
        cache_authority._provenance_output_path,
    )
    if requested_destinations != writer_destinations:
        raise FeatureValidationError(
            "cache load paths disagree with writer-bound destinations"
        )
    result = _issue_embedding_cache(
        sample_ids=ids,
        embeddings=embeddings.astype(np.float32, copy=False),
        source_sha256=observed,
        weight_sha256=_FROZEN_RESNET18_WEIGHTS_SHA256,
        transform_sha256=TRANSFORM_SHA256,
        embedding_cache_sha256=cache_sha256,
        provenance=raw,
        provenance_sha256=provenance_sha256,
        cache_output_path=Path(cache_path),
        provenance_output_path=Path(provenance_path),
    )
    validate_embedding_cache_state(result)
    return result


def _embedding_snapshot_runtime(
    issuer_code: types.CodeType,
) -> tuple[Callable[..., EmbeddingCache], Callable[[object], bool]]:
    capability = object()

    def is_capability(candidate: object) -> bool:
        return candidate is capability

    def load_writer_snapshots(
        cache_path: Path,
        provenance_path: Path,
        *,
        cache_snapshot: bytes,
        provenance_snapshot: bytes,
        **kwargs: Any,
    ) -> EmbeddingCache:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if caller is None or caller.f_code is not issuer_code:
            raise FeatureValidationError(
                "embedding snapshot validation requires the cache writer"
            )
        return load_embedding_cache(
            cache_path,
            provenance_path,
            _cache_snapshot=cache_snapshot,
            _provenance_snapshot=provenance_snapshot,
            _snapshot_capability=capability,
            **kwargs,
        )

    return load_writer_snapshots, is_capability


def _embedding_cache_loader_runtime(
    implementation: Callable[..., EmbeddingCache],
    issued_digest: Callable[[EmbeddingCache], str],
    content_digest: Callable[[EmbeddingCache], str],
) -> Callable[..., EmbeddingCache]:
    @wraps(implementation)
    def load(*args: Any, **kwargs: Any) -> EmbeddingCache:
        cache_authority = kwargs.get("cache_authority")
        if cache_authority is not None:
            registered_digest = issued_digest(cache_authority)
            try:
                if content_digest(cache_authority) != registered_digest:
                    raise FeatureValidationError(
                        "embedding cache content disagrees with writer authority"
                    )
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise FeatureValidationError(
                    "embedding cache load requires issued writer authority"
                ) from error
        return implementation(*args, **kwargs)

    return load


_load_embedding_cache_from_writer_snapshots, _is_embedding_snapshot_capability = (
    _embedding_snapshot_runtime(build_embedding_cache.__code__)
)
_cache_writer_capability = _take_cache_writer_capability()
_issue_embedding_cache, _issued_cache_digest = _embedding_cache_writer_runtime(
    frozenset({build_embedding_cache.__code__, _load_embedding_cache_impl.__code__}),
    _cache_writer_capability,
)
load_embedding_cache = _embedding_cache_loader_runtime(
    _load_embedding_cache_impl,
    _issued_cache_digest,
    _embedding_cache_content_digest,
)
del _load_embedding_cache_impl
del _embedding_cache_loader_runtime
del _cache_writer_capability
del _take_cache_writer_capability
del _validate_embedding_cache_constructor_capability
_publish_staged_pair = _pair_transaction_runtime(
    _publish_staged_pair_transaction_impl,
    _issued_cache_digest,
    _embedding_cache_content_digest,
)
del _publish_staged_pair_transaction_impl


def validate_embedding_cache(
    cache_or_path: EmbeddingCache | Path | str,
    provenance_path: Path | str | None = None,
    **kwargs: Any,
) -> bool | EmbeddingCache:
    if isinstance(cache_or_path, EmbeddingCache):
        return validate_embedding_cache_state(cache_or_path)
    return load_embedding_cache(cache_or_path, provenance_path, **kwargs)


def _immutable_pca_array(value: Any, name: str) -> tuple[np.ndarray, bytes]:
    converted = _strict_real_numeric_array(value, f"PCA {name}")
    if np.any(np.abs(converted) > _PCA_SAFE_ABS):
        raise FeatureValidationError(f"PCA {name} exceeds safe numeric range")
    return _immutable_array(converted, dtype=np.float64)


def _pca_rank_tolerance(
    singular_values: np.ndarray, n_samples: int, n_features: int
) -> float:
    leading = float(singular_values[0]) if singular_values.size else 0.0
    return (
        max(n_samples, n_features)
        * np.finfo(np.float64).eps
        * leading
        * 64.0
    )


def _pca_numerical_rank(
    singular_values: np.ndarray, n_samples: int, n_features: int
) -> int:
    tolerance = _pca_rank_tolerance(singular_values, n_samples, n_features)
    return int(np.count_nonzero(singular_values > tolerance))


def _validate_pca_contract(
    mean: np.ndarray,
    components: np.ndarray,
    explained_variance: np.ndarray,
    singular_values: np.ndarray,
    n_components: Any,
    n_features_in: Any,
    n_samples_in: Any,
    fit_sample_ids: Any,
    fit_domain_ids: Any,
    authority_mode: Any,
    heldout_domain: Any,
    inner_query_domain: Any,
    fit_authority_state_sha256: Any,
    outer_train_state_sha256: Any,
    fit_embeddings_sha256: Any,
    *,
    _domain_order: tuple[str, ...] = tuple(_DATA_DOMAIN_ORDER),
    _primary_dataset_ids: tuple[str, ...] = tuple(_DATA_PRIMARY_DATASET_IDS),
    _primary_specimen_ids: tuple[str, ...] = tuple(_DATA_PRIMARY_SPECIMEN_IDS),
) -> tuple[
    int,
    int,
    int,
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    str,
    str,
    str,
    str,
]:
    if not isinstance(n_components, (int, np.integer)) or isinstance(
        n_components, (bool, np.bool_)
    ):
        raise FeatureValidationError("PCA components must be an integer")
    if n_components not in PCA_DIMENSIONS:
        raise FeatureValidationError("PCA components must be one of 8, 16, or 32")
    if (
        not isinstance(n_features_in, (int, np.integer))
        or isinstance(n_features_in, (bool, np.bool_))
        or int(n_features_in) <= 0
    ):
        raise FeatureValidationError("PCA feature count must be a positive integer")
    if isinstance(fit_sample_ids, (str, bytes)) or isinstance(
        fit_domain_ids, (str, bytes)
    ):
        raise FeatureValidationError("PCA fit IDs must be sequences")
    try:
        ids = tuple(fit_sample_ids)
        domains = tuple(fit_domain_ids)
    except TypeError as error:
        raise FeatureValidationError("PCA fit IDs must be sequences") from error
    if any(type(value) is not str or not value for value in ids):
        raise FeatureValidationError("PCA fit sample IDs must be non-empty strings")
    if any(type(value) is not str or not value for value in domains):
        raise FeatureValidationError("PCA fit domain IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise FeatureValidationError("PCA fit sample IDs must be unique")
    if authority_mode not in ("outer", "inner") or type(authority_mode) is not str:
        raise FeatureValidationError("PCA authority mode must be outer or inner")
    if type(heldout_domain) is not str or heldout_domain not in _domain_order:
        raise FeatureValidationError(
            "PCA outer-train heldout domain is not registered"
        )
    if authority_mode == "outer":
        if inner_query_domain != "":
            raise FeatureValidationError("outer PCA cannot bind an inner query domain")
        excluded_domains = {heldout_domain}
        expected_domain_count = 5
        if fit_authority_state_sha256 != outer_train_state_sha256:
            raise FeatureValidationError(
                "outer PCA fit and outer authority states must match"
            )
    else:
        if (
            type(inner_query_domain) is not str
            or inner_query_domain not in _domain_order
            or inner_query_domain == heldout_domain
        ):
            raise FeatureValidationError("inner PCA query domain is invalid")
        excluded_domains = {heldout_domain, inner_query_domain}
        expected_domain_count = 4
    if not _is_lower_sha256(fit_authority_state_sha256):
        raise FeatureValidationError("PCA fit authority hash is invalid")
    if not _is_lower_sha256(outer_train_state_sha256):
        raise FeatureValidationError("PCA outer-train authority hash is invalid")
    if not _is_lower_sha256(fit_embeddings_sha256):
        raise FeatureValidationError("PCA fit embedding digest is invalid")
    if not isinstance(n_samples_in, (int, np.integer)) or isinstance(
        n_samples_in, (bool, np.bool_)
    ):
        raise FeatureValidationError("PCA training row count must be an integer")
    n_samples = int(n_samples_in)
    if n_samples < 2:
        raise FeatureValidationError("PCA training row count must be at least two")
    if len(ids) != n_samples:
        raise FeatureValidationError("PCA fit sample IDs must align with training rows")
    if len(domains) != n_samples:
        raise FeatureValidationError("PCA fit domain IDs must align with training rows")
    expected_pairs = tuple(
        (sample_id, domain_id)
        for sample_id, domain_id in zip(
            _primary_specimen_ids, _primary_dataset_ids, strict=True
        )
        if domain_id not in excluded_domains
    )
    if tuple(zip(ids, domains, strict=True)) != expected_pairs:
        raise FeatureValidationError(
            "PCA fit IDs are not the canonical heldout outer-training rows"
        )
    if (
        set(domains) != set(_domain_order) - excluded_domains
        or len(set(domains)) != expected_domain_count
    ):
        raise FeatureValidationError(
            f"PCA fit must contain exactly {expected_domain_count} canonical domains"
        )
    n_components = int(n_components)
    n_features = int(n_features_in)
    if n_components > min(n_samples - 1, n_features):
        raise FeatureValidationError("PCA components exceed the training fold rank")
    if mean.shape != (n_features,):
        raise FeatureValidationError("PCA mean has the wrong feature length")
    if components.shape != (n_components, n_features):
        raise FeatureValidationError("PCA components have the wrong shape")
    if explained_variance.shape != (n_components,):
        raise FeatureValidationError("PCA explained variance has the wrong shape")
    if singular_values.shape != (n_components,):
        raise FeatureValidationError("PCA singular values have the wrong shape")
    if np.any(explained_variance < 0.0) or np.any(singular_values < 0.0):
        raise FeatureValidationError(
            "PCA variance and singular values must be nonnegative"
        )
    orthogonality = components @ components.T
    tolerance = max(n_features, n_components, n_samples) * np.finfo(np.float64).eps * 64
    if not np.allclose(
        orthogonality,
        np.eye(n_components, dtype=np.float64),
        rtol=0.0,
        atol=tolerance,
    ):
        raise FeatureValidationError("PCA components must be orthogonal unit vectors")
    if np.any(np.diff(singular_values) > tolerance * max(1.0, singular_values[0])):
        raise FeatureValidationError("PCA singular values must be sorted descending")
    expected_variance = singular_values**2 / float(n_samples - 1)
    variance_scale = max(
        np.finfo(np.float64).tiny,
        float(np.max(np.abs(expected_variance), initial=0.0)),
    )
    variance_tolerance = (
        max(n_features, n_components, n_samples)
        * np.finfo(np.float64).eps
        * variance_scale
        * 64.0
    )
    if not np.allclose(
        explained_variance,
        expected_variance,
        rtol=np.finfo(np.float64).eps * 64.0,
        atol=variance_tolerance,
    ):
        raise FeatureValidationError(
            "PCA explained variance is inconsistent with singular values"
        )
    if _pca_numerical_rank(singular_values, n_samples, n_features) < n_components:
        raise FeatureValidationError("PCA components exceed the numerical rank")
    return (
        n_components,
        n_features,
        n_samples,
        ids,
        domains,
        authority_mode,
        heldout_domain,
        inner_query_domain,
        fit_authority_state_sha256,
        outer_train_state_sha256,
        fit_embeddings_sha256,
    )


def _pca_constructor_runtime() -> tuple[
    Callable[[object], bool], Callable[[], object]
]:
    capability = object()

    def validate(candidate: object) -> bool:
        return candidate is capability

    def take_for_fit_initialization() -> object:
        return capability

    return validate, take_for_fit_initialization


_validate_pca_constructor_capability, _take_pca_fit_capability = (
    _pca_constructor_runtime()
)
del _pca_constructor_runtime


@dataclass(frozen=True)
class FoldLocalPCA:
    """SVD PCA fitted only on one training fold."""

    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_: np.ndarray
    singular_values_: np.ndarray
    n_components_: int
    n_features_in_: int
    fit_sample_ids: tuple[str, ...]
    fit_domain_ids: tuple[str, ...] = ()
    n_samples_in_: int = 0
    authority_mode: str = "outer"
    heldout_domain: str = ""
    inner_query_domain: str = ""
    fit_authority_state_sha256: str = ""
    outer_train_state_sha256: str = ""
    fit_embeddings_sha256: str = ""
    _fit_capability: InitVar[object | None] = None
    state_sha256: str = field(init=False)
    _mean_backing: bytes = field(init=False, repr=False)
    _components_backing: bytes = field(init=False, repr=False)
    _explained_variance_backing: bytes = field(init=False, repr=False)
    _singular_values_backing: bytes = field(init=False, repr=False)

    def __post_init__(
        self,
        _fit_capability: object | None,
        _capability_validator: Callable[[object], bool] = (
            _validate_pca_constructor_capability  # noqa: RUF033
        ),
    ) -> None:
        mean, mean_backing = _immutable_pca_array(self.mean_, "mean")
        components, components_backing = _immutable_pca_array(
            self.components_, "components"
        )
        explained_variance, explained_backing = _immutable_pca_array(
            self.explained_variance_, "explained variance"
        )
        singular_values, singular_backing = _immutable_pca_array(
            self.singular_values_, "singular values"
        )
        if self.fit_sample_ids is None:
            constructor_sample_ids: Any = ()
        elif isinstance(self.fit_sample_ids, (str, bytes)):
            constructor_sample_ids = self.fit_sample_ids
        else:
            try:
                constructor_sample_ids = tuple(self.fit_sample_ids)
            except TypeError:
                constructor_sample_ids = self.fit_sample_ids
        constructor_n_samples = self.n_samples_in_
        if (
            type(constructor_n_samples) is int
            and constructor_n_samples == 0
            and isinstance(constructor_sample_ids, tuple)
        ):
            constructor_n_samples = len(constructor_sample_ids)
        constructor_fit_authority_state = (
            self.fit_authority_state_sha256 or self.outer_train_state_sha256
        )
        (
            n_components,
            n_features,
            n_samples,
            ids,
            domains,
            authority_mode,
            heldout,
            inner_query,
            fit_authority_state,
            authority_state,
            fit_embeddings_digest,
        ) = _validate_pca_contract(
            mean,
            components,
            explained_variance,
            singular_values,
            self.n_components_,
            self.n_features_in_,
            constructor_n_samples,
            self.fit_sample_ids,
            self.fit_domain_ids,
            self.authority_mode,
            self.heldout_domain,
            self.inner_query_domain,
            constructor_fit_authority_state,
            self.outer_train_state_sha256,
            self.fit_embeddings_sha256,
        )
        if not _capability_validator(_fit_capability):
            raise FeatureValidationError(
                "PCA constructor requires the fit issuer authority"
            )
        object.__setattr__(self, "mean_", mean)
        object.__setattr__(self, "components_", components)
        object.__setattr__(self, "explained_variance_", explained_variance)
        object.__setattr__(self, "singular_values_", singular_values)
        object.__setattr__(self, "n_components_", n_components)
        object.__setattr__(self, "n_features_in_", n_features)
        object.__setattr__(self, "fit_sample_ids", ids)
        object.__setattr__(self, "fit_domain_ids", domains)
        object.__setattr__(self, "n_samples_in_", n_samples)
        object.__setattr__(self, "authority_mode", authority_mode)
        object.__setattr__(self, "heldout_domain", heldout)
        object.__setattr__(self, "inner_query_domain", inner_query)
        object.__setattr__(
            self, "fit_authority_state_sha256", fit_authority_state
        )
        object.__setattr__(self, "outer_train_state_sha256", authority_state)
        object.__setattr__(self, "fit_embeddings_sha256", fit_embeddings_digest)
        object.__setattr__(self, "_mean_backing", mean_backing)
        object.__setattr__(self, "_components_backing", components_backing)
        object.__setattr__(self, "_explained_variance_backing", explained_backing)
        object.__setattr__(self, "_singular_values_backing", singular_backing)
        object.__setattr__(
            self,
            "state_sha256",
            _state_hash(
                mean,
                components,
                explained_variance,
                singular_values,
                n_components,
                n_features,
                n_samples,
                ids,
                domains,
                authority_mode,
                heldout,
                inner_query,
                fit_authority_state,
                authority_state,
                fit_embeddings_digest,
            ),
        )

    @property
    def mean(self) -> np.ndarray:
        return self.mean_

    @property
    def components(self) -> np.ndarray:
        return self.components_

    @property
    def n_components(self) -> int:
        return self.n_components_

    @property
    def fit_domains(self) -> tuple[str, ...]:
        return self.fit_domain_ids

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        return transform_embedding_pca(self, embeddings)

    def __copy__(self) -> FoldLocalPCA:
        raise TypeError("fit-issued PCA models cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> FoldLocalPCA:
        raise TypeError("fit-issued PCA models cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("fit-issued PCA models cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("fit-issued PCA models cannot be serialized")


def _pca_content_digest(model: FoldLocalPCA) -> str:
    return _state_hash(
        model.mean_,
        model.components_,
        model.explained_variance_,
        model.singular_values_,
        model.n_components_,
        model.n_features_in_,
        model.n_samples_in_,
        model.fit_sample_ids,
        model.fit_domain_ids,
        model.authority_mode,
        model.heldout_domain,
        model.inner_query_domain,
        model.fit_authority_state_sha256,
        model.outer_train_state_sha256,
        model.fit_embeddings_sha256,
        _sha256_bytes(model._mean_backing),
        _sha256_bytes(model._components_backing),
        _sha256_bytes(model._explained_variance_backing),
        _sha256_bytes(model._singular_values_backing),
    )


def _pca_issuance_runtime(
    issuer_codes: frozenset[types.CodeType],
    capability: object,
    authority_validator: Callable[[object], str],
    authority_type: type[Any],
    domain_order: tuple[str, ...],
    primary_sample_ids: tuple[str, ...],
    primary_domain_ids: tuple[str, ...],
) -> tuple[
    Callable[..., FoldLocalPCA],
    Callable[[FoldLocalPCA], tuple[str, str, str, str]],
]:
    issued: dict[
        int,
        tuple[
            weakref.ReferenceType[FoldLocalPCA],
            str,
            object,
            str,
            object,
            str,
            str,
        ],
    ] = {}
    lock = threading.RLock()

    def issue(
        *,
        fit_authority: object,
        parent_outer_authority: object,
        **fields: Any,
    ) -> FoldLocalPCA:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if caller is None or caller.f_code not in issuer_codes:
            raise FeatureValidationError("PCA issuance requires the fit implementation")
        try:
            fit_authority_state = authority_validator(fit_authority)
            parent_authority_state = authority_validator(parent_outer_authority)
        except Exception as error:
            raise FeatureValidationError(
                "PCA issuance requires loader-issued fit and outer authorities"
            ) from error
        if (
            fields.get("fit_authority_state_sha256") != fit_authority_state
            or fields.get("outer_train_state_sha256") != parent_authority_state
        ):
            raise FeatureValidationError("PCA fit or outer authority state changed")
        try:
            if (
                type(fit_authority) is not authority_type
                or type(parent_outer_authority) is not authority_type
            ):
                raise FeatureValidationError(
                    "PCA issuance requires exact V3DataView authorities"
                )
            heldout = fields["heldout_domain"]
            mode = fields["authority_mode"]
            inner_query = fields["inner_query_domain"]
            fit_pairs = tuple(
                zip(
                    tuple(fields["fit_sample_ids"]),
                    tuple(fields["fit_domain_ids"]),
                    strict=True,
                )
            )
            authority_pairs = tuple(
                zip(
                    tuple(fit_authority.sample_ids.tolist()),
                    tuple(fit_authority.dataset_ids.tolist()),
                    strict=True,
                )
            )
            parent_pairs = tuple(
                zip(
                    tuple(parent_outer_authority.sample_ids.tolist()),
                    tuple(parent_outer_authority.dataset_ids.tolist()),
                    strict=True,
                )
            )
        except FeatureValidationError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise FeatureValidationError("PCA issuance authority is malformed") from error
        canonical_parent = tuple(
            (sample_id, domain_id)
            for sample_id, domain_id in zip(
                primary_sample_ids, primary_domain_ids, strict=True
            )
            if domain_id != heldout
        )
        if (
            type(heldout) is not str
            or heldout not in domain_order
            or parent_pairs != canonical_parent
            or len({domain for _, domain in parent_pairs}) != 5
        ):
            raise FeatureValidationError(
                "PCA issuance parent authority is not the canonical outer fold"
            )
        if mode == "outer":
            expected_fit = canonical_parent
            if fit_authority is not parent_outer_authority or inner_query != "":
                raise FeatureValidationError(
                    "PCA outer issuance authority identity is invalid"
                )
        elif mode == "inner":
            if (
                type(inner_query) is not str
                or inner_query not in domain_order
                or inner_query == heldout
            ):
                raise FeatureValidationError("PCA inner query authority is invalid")
            expected_fit = tuple(
                pair for pair in canonical_parent if pair[1] != inner_query
            )
            if len({domain for _, domain in expected_fit}) != 4:
                raise FeatureValidationError(
                    "PCA inner issuance requires exactly four domains"
                )
        else:
            raise FeatureValidationError("PCA issuance authority mode is invalid")
        if (
            fit_pairs != expected_fit
            or authority_pairs != expected_fit
            or fields.get("n_samples_in_") != len(expected_fit)
        ):
            raise FeatureValidationError(
                "PCA fit authority identity disagrees with canonical rows"
            )
        fit_digest = fields.get("fit_embeddings_sha256")
        if not _is_lower_sha256(fit_digest):
            raise FeatureValidationError("PCA fit embedding digest is invalid")
        model = FoldLocalPCA(**fields, _fit_capability=capability)
        content_digest = _pca_content_digest(model)
        identity = id(model)

        def discard(reference: weakref.ReferenceType[FoldLocalPCA]) -> None:
            with lock:
                current = issued.get(identity)
                if current is not None and current[0] is reference:
                    issued.pop(identity, None)

        reference = weakref.ref(model, discard)
        with lock:
            issued[identity] = (
                reference,
                content_digest,
                fit_authority,
                fit_authority_state,
                parent_outer_authority,
                parent_authority_state,
                fit_digest,
            )
        return model

    def record(model: FoldLocalPCA) -> tuple[str, str, str, str]:
        with lock:
            entry = issued.get(id(model))
        if entry is None or entry[0]() is not model:
            raise FeatureValidationError("PCA model is not issued by the fit authority")
        try:
            current_fit_state = authority_validator(entry[2])
            current_parent_state = authority_validator(entry[4])
        except Exception as error:
            raise FeatureValidationError(
                "PCA outer-train authority is no longer valid"
            ) from error
        if current_fit_state != entry[3] or current_parent_state != entry[5]:
            raise FeatureValidationError("PCA fit or outer authority state changed")
        return entry[1], entry[3], entry[5], entry[6]

    return issue, record


def validate_fold_local_pca(model: FoldLocalPCA) -> bool:
    """Validate immutable PCA arrays, backing bytes, and fitted-state digest."""

    if not isinstance(model, FoldLocalPCA):
        raise FeatureValidationError("PCA model has an invalid type")
    (
        registered_digest,
        registered_fit_authority_state,
        registered_outer_authority_state,
        registered_fit_digest,
    ) = _issued_pca_record(model)
    try:
        arrays = (
            (model.mean_, model._mean_backing),
            (model.components_, model._components_backing),
            (model.explained_variance_, model._explained_variance_backing),
            (model.singular_values_, model._singular_values_backing),
        )
        if any(array.flags.writeable for array, _ in arrays):
            raise FeatureValidationError("PCA model arrays are writable")
        if any(array.tobytes(order="C") != backing for array, backing in arrays):
            raise FeatureValidationError("PCA model array backing changed")
        if any(array.dtype != np.dtype(np.float64) for array, _ in arrays):
            raise FeatureValidationError("PCA model arrays have a noncanonical dtype")
        if (
            type(model.n_components_) is not int
            or type(model.n_features_in_) is not int
            or type(model.n_samples_in_) is not int
        ):
            raise FeatureValidationError("PCA model dimensions are not canonical")
        if (
            type(model.fit_sample_ids) is not tuple
            or type(model.fit_domain_ids) is not tuple
        ):
            raise FeatureValidationError("PCA fit IDs are mutable")
        mean, _ = _immutable_pca_array(model.mean_, "mean")
        components, _ = _immutable_pca_array(model.components_, "components")
        explained_variance, _ = _immutable_pca_array(
            model.explained_variance_, "explained variance"
        )
        singular_values, _ = _immutable_pca_array(
            model.singular_values_, "singular values"
        )
        (
            n_components,
            n_features,
            n_samples,
            ids,
            domains,
            authority_mode,
            heldout,
            inner_query,
            fit_authority_state,
            authority_state,
            fit_embeddings_digest,
        ) = _validate_pca_contract(
            mean,
            components,
            explained_variance,
            singular_values,
            model.n_components_,
            model.n_features_in_,
            model.n_samples_in_,
            model.fit_sample_ids,
            model.fit_domain_ids,
            model.authority_mode,
            model.heldout_domain,
            model.inner_query_domain,
            model.fit_authority_state_sha256,
            model.outer_train_state_sha256,
            model.fit_embeddings_sha256,
        )
        if ids != model.fit_sample_ids or domains != model.fit_domain_ids:
            raise FeatureValidationError("PCA fit IDs changed")
        current = _state_hash(
            model.mean_,
            model.components_,
            model.explained_variance_,
            model.singular_values_,
            n_components,
            n_features,
            n_samples,
            ids,
            domains,
            authority_mode,
            heldout,
            inner_query,
            fit_authority_state,
            authority_state,
            fit_embeddings_digest,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise FeatureValidationError("PCA model state is malformed") from error
    try:
        if current != model.state_sha256:
            raise FeatureValidationError("PCA model state hash changed")
        if (
            _pca_content_digest(model) != registered_digest
            or model.fit_embeddings_sha256 != registered_fit_digest
            or model.fit_authority_state_sha256
            != registered_fit_authority_state
            or model.outer_train_state_sha256
            != registered_outer_authority_state
        ):
            raise FeatureValidationError("PCA issued fit content or authority changed")
    except AttributeError as error:
        raise FeatureValidationError("PCA model state is malformed") from error
    return True


def _validate_embedding_matrix(embeddings: np.ndarray, name: str) -> np.ndarray:
    value = _strict_real_numeric_array(embeddings, name)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise FeatureValidationError(
            f"{name} must be a nonempty two-dimensional matrix"
        )
    if not np.all(np.isfinite(value)):
        raise FeatureValidationError(f"{name} must contain only finite values")
    if np.any(np.abs(value) > _PCA_SAFE_ABS):
        raise FeatureValidationError(f"{name} exceeds safe numeric range")
    return value


def valid_pca_dimensions(
    train_embeddings: np.ndarray,
    dimensions: Iterable[int] = PCA_DIMENSIONS,
) -> tuple[int, ...]:
    """Return registered PCA dimensions representable by the training fold rank bound."""

    value = _validate_embedding_matrix(train_embeddings, "train embeddings")
    centered = value - np.mean(value, axis=0, dtype=np.float64)
    try:
        singular_values = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError as error:
        raise FeatureValidationError("PCA numerical rank computation failed") from error
    max_rank = _pca_numerical_rank(
        singular_values, value.shape[0], value.shape[1]
    )
    if isinstance(dimensions, (str, bytes)):
        raise FeatureValidationError("PCA dimensions must be an exact sequence")
    try:
        requested_dimensions = tuple(dimensions)
    except TypeError as error:
        raise FeatureValidationError(
            "PCA dimensions must be an exact sequence"
        ) from error
    if requested_dimensions != PCA_DIMENSIONS:
        raise FeatureValidationError(
            "PCA dimensions must be exactly (8, 16, 32) in registered order"
        )
    requested = set(requested_dimensions)
    result: list[int] = []
    for dimension in PCA_DIMENSIONS:
        if dimension in requested and dimension <= max_rank:
            result.append(dimension)
    return tuple(dict.fromkeys(result))


def _validate_outer_train_contract(
    outer_train_authority: object | None,
    heldout_domain: str | None,
    fit_sample_ids: Sequence[str] | None,
    fit_domain_ids: Sequence[str] | None,
    n_samples: int,
    *,
    _authority_type: type[Any] = _DATA_V3_DATA_VIEW,
    _authority_validator: Callable[[object], str] = _DATA_AUTHORITY_VALIDATOR,
    _domain_order: tuple[str, ...] = tuple(_DATA_DOMAIN_ORDER),
    _primary_dataset_ids: tuple[str, ...] = tuple(_DATA_PRIMARY_DATASET_IDS),
    _primary_specimen_ids: tuple[str, ...] = tuple(_DATA_PRIMARY_SPECIMEN_IDS),
) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
    if type(outer_train_authority) is not _authority_type:
        raise FeatureValidationError(
            "PCA requires an issued outer-training V3DataView authority"
        )
    try:
        authority_state = _authority_validator(outer_train_authority)
    except Exception as error:
        raise FeatureValidationError(
            "PCA outer-training authority is not loader-issued"
        ) from error
    if type(heldout_domain) is not str or heldout_domain not in _domain_order:
        raise FeatureValidationError(
            "PCA outer-train heldout domain is not registered"
        )
    if fit_sample_ids is None or fit_domain_ids is None:
        raise FeatureValidationError(
            "PCA requires explicit canonical fit sample and domain IDs"
        )
    if isinstance(fit_sample_ids, (str, bytes)) or isinstance(
        fit_domain_ids, (str, bytes)
    ):
        raise FeatureValidationError("PCA fit IDs must be sequences")
    try:
        ids = tuple(fit_sample_ids)
        domains = tuple(fit_domain_ids)
        authority_ids = tuple(outer_train_authority.sample_ids.tolist())
        authority_domains = tuple(outer_train_authority.dataset_ids.tolist())
    except (AttributeError, TypeError, ValueError) as error:
        raise FeatureValidationError("PCA fit IDs are malformed") from error
    expected_pairs = tuple(
        (sample_id, domain_id)
        for sample_id, domain_id in zip(
            _primary_specimen_ids, _primary_dataset_ids, strict=True
        )
        if domain_id != heldout_domain
    )
    if (
        len(ids) != n_samples
        or tuple(zip(ids, domains, strict=True)) != expected_pairs
        or tuple(zip(authority_ids, authority_domains, strict=True)) != expected_pairs
    ):
        raise FeatureValidationError(
            "PCA fit IDs are not the canonical heldout outer-training rows"
        )
    if set(domains) != set(_domain_order) - {heldout_domain} or len(set(domains)) != 5:
        raise FeatureValidationError(
            "PCA fit must contain exactly five domains and exclude heldout"
        )
    return ids, domains, heldout_domain, authority_state


def _validate_pca_authority_contract(
    outer_train_authority: object | None,
    inner_train_authority: object | None,
    parent_outer_authority: object | None,
    heldout_domain: str | None,
    inner_query_domain: str | None,
    fit_sample_ids: Sequence[str] | None,
    fit_domain_ids: Sequence[str] | None,
    n_samples: int,
    *,
    _authority_type: type[Any] = _DATA_V3_DATA_VIEW,
    _authority_validator: Callable[[object], str] = _DATA_AUTHORITY_VALIDATOR,
    _domain_order: tuple[str, ...] = tuple(_DATA_DOMAIN_ORDER),
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    str,
    str,
    str,
    object,
    object,
]:
    inner_values = (
        inner_train_authority,
        parent_outer_authority,
        inner_query_domain,
    )
    if all(value is None for value in inner_values):
        ids, domains, heldout, authority_state = _validate_outer_train_contract(
            outer_train_authority,
            heldout_domain,
            fit_sample_ids,
            fit_domain_ids,
            n_samples,
        )
        if outer_train_authority is None:  # pragma: no cover - validated above
            raise FeatureValidationError("PCA outer authority is missing")
        return (
            ids,
            domains,
            "outer",
            heldout,
            "",
            authority_state,
            authority_state,
            outer_train_authority,
            outer_train_authority,
        )
    if any(value is None for value in inner_values):
        raise FeatureValidationError(
            "inner PCA requires inner, parent, and query authorities"
        )
    if outer_train_authority is not parent_outer_authority:
        raise FeatureValidationError(
            "inner PCA outer and parent authorities must be the same object"
        )
    if type(parent_outer_authority) is not _authority_type:
        raise FeatureValidationError(
            "inner PCA parent authority must be an issued V3DataView"
        )
    try:
        parent_ids = tuple(parent_outer_authority.sample_ids.tolist())
        parent_domains = tuple(parent_outer_authority.dataset_ids.tolist())
    except (AttributeError, TypeError, ValueError) as error:
        raise FeatureValidationError("inner PCA parent authority is malformed") from error
    _, _, heldout, parent_state = _validate_outer_train_contract(
        parent_outer_authority,
        heldout_domain,
        parent_ids,
        parent_domains,
        len(parent_ids),
    )
    if (
        type(inner_query_domain) is not str
        or inner_query_domain not in _domain_order
        or inner_query_domain == heldout
    ):
        raise FeatureValidationError("inner PCA query domain is invalid")
    if type(inner_train_authority) is not _authority_type:
        raise FeatureValidationError(
            "inner PCA fit authority must be an issued V3DataView"
        )
    try:
        fit_authority_state = _authority_validator(inner_train_authority)
        ids = tuple(fit_sample_ids) if fit_sample_ids is not None else ()
        domains = tuple(fit_domain_ids) if fit_domain_ids is not None else ()
        authority_ids = tuple(inner_train_authority.sample_ids.tolist())
        authority_domains = tuple(inner_train_authority.dataset_ids.tolist())
    except Exception as error:
        raise FeatureValidationError(
            "inner PCA fit authority is not loader-issued or IDs are malformed"
        ) from error
    expected_pairs = tuple(
        (sample_id, domain_id)
        for sample_id, domain_id in zip(parent_ids, parent_domains, strict=True)
        if domain_id != inner_query_domain
    )
    supplied_pairs = tuple(zip(ids, domains, strict=True))
    authority_pairs = tuple(zip(authority_ids, authority_domains, strict=True))
    if (
        len(ids) != n_samples
        or supplied_pairs != expected_pairs
        or authority_pairs != expected_pairs
    ):
        raise FeatureValidationError(
            "inner PCA fit IDs are not the canonical parent-minus-query rows"
        )
    if (
        set(domains) != set(_domain_order) - {heldout, inner_query_domain}
        or len(set(domains)) != 4
    ):
        raise FeatureValidationError(
            "inner PCA fit must contain exactly four canonical domains"
        )
    return (
        ids,
        domains,
        "inner",
        heldout,
        inner_query_domain,
        fit_authority_state,
        parent_state,
        inner_train_authority,
        parent_outer_authority,
    )


def _canonicalize_pca_components(
    components: np.ndarray, singular_values: np.ndarray, n_features: int
) -> np.ndarray:
    result = np.asarray(components, dtype=np.float64).copy()
    if len(result) == 0:
        return result
    spectral_tolerance = (
        max(result.shape)
        * np.finfo(np.float64).eps
        * float(singular_values[0])
        * 128.0
    )
    start = 0
    while start < len(result):
        stop = start + 1
        while stop < len(result) and abs(
            float(singular_values[stop] - singular_values[start])
        ) <= spectral_tolerance:
            stop += 1
        if stop - start > 1:
            basis = result[start:stop]
            projector = np.round(basis.T @ basis, decimals=14)
            canonical: list[np.ndarray] = []
            for axis in range(n_features):
                candidate = projector[:, axis].copy()
                for previous in canonical:
                    candidate -= np.dot(candidate, previous) * previous
                norm = float(np.linalg.norm(candidate))
                if norm > 1e-12:
                    canonical.append(candidate / norm)
                if len(canonical) == stop - start:
                    break
            if len(canonical) != stop - start:
                raise FeatureValidationError(
                    "PCA repeated-spectrum basis cannot be canonicalized"
                )
            result[start:stop] = np.asarray(canonical)
        start = stop
    for row in range(result.shape[0]):
        pivot = int(np.argmax(np.abs(result[row])))
        if result[row, pivot] < 0.0:
            result[row] *= -1.0
    return result


def fit_embedding_pca(
    train_embeddings: np.ndarray,
    *,
    n_components: int,
    fit_sample_ids: Sequence[str] | None = None,
    fit_domain_ids: Sequence[str] | None = None,
    outer_train_authority: object | None = None,
    heldout_domain: str | None = None,
    inner_train_authority: object | None = None,
    parent_outer_authority: object | None = None,
    inner_query_domain: str | None = None,
) -> FoldLocalPCA:
    """Fit PCA from the training fold only; query/test values are not accepted."""

    value = _validate_embedding_matrix(train_embeddings, "train embeddings")
    if not isinstance(n_components, int) or isinstance(n_components, bool):
        raise FeatureValidationError("PCA components must be an integer")
    if n_components not in PCA_DIMENSIONS:
        raise FeatureValidationError("PCA components must be one of 8, 16, or 32")
    if n_components > min(value.shape[0] - 1, value.shape[1]):
        raise FeatureValidationError(
            f"PCA components exceed the training fold rank: {n_components}"
        )
    (
        ids,
        domains,
        authority_mode,
        heldout,
        inner_query,
        fit_authority_state,
        outer_authority_state,
        fit_authority,
        parent_authority,
    ) = _validate_pca_authority_contract(
        outer_train_authority,
        inner_train_authority,
        parent_outer_authority,
        heldout_domain,
        inner_query_domain,
        fit_sample_ids,
        fit_domain_ids,
        value.shape[0],
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                mean = np.mean(value, axis=0, dtype=np.float64)
                if not np.all(np.isfinite(mean)):
                    raise FeatureValidationError("PCA mean must be finite")
                centered = value - mean
                if not np.all(np.isfinite(centered)):
                    raise FeatureValidationError("PCA centered values must be finite")
                u, singular_values, components = np.linalg.svd(
                    centered, full_matrices=False
                )
                if not np.all(np.isfinite(singular_values)) or not np.all(
                    np.isfinite(components)
                ):
                    raise FeatureValidationError("PCA SVD values must be finite")
    except FeatureValidationError:
        raise
    except (
        FloatingPointError,
        np.linalg.LinAlgError,
        RuntimeError,
        ValueError,
        Warning,
    ) as error:
        raise FeatureValidationError("PCA fitting failed safely") from error
    del u
    numerical_rank = _pca_numerical_rank(
        singular_values, value.shape[0], value.shape[1]
    )
    if n_components > numerical_rank:
        raise FeatureValidationError(
            f"PCA components exceed the numerical training rank: {n_components}"
        )
    components = _canonicalize_pca_components(
        components, singular_values, value.shape[1]
    )[:n_components].copy()
    singular_values = singular_values[:n_components].copy()
    denominator = max(value.shape[0] - 1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                explained_variance = (singular_values**2) / denominator
    except (FloatingPointError, RuntimeError, ValueError, Warning) as error:
        raise FeatureValidationError("PCA explained variance is not finite") from error
    if not np.all(np.isfinite(explained_variance)):
        raise FeatureValidationError("PCA explained variance is not finite")
    fit_embeddings_sha256 = _state_hash(value)
    return _issue_fold_local_pca(
        fit_authority=fit_authority,
        parent_outer_authority=parent_authority,
        mean_=mean,
        components_=components,
        explained_variance_=explained_variance,
        singular_values_=singular_values,
        n_components_=n_components,
        n_features_in_=value.shape[1],
        fit_sample_ids=ids,
        fit_domain_ids=domains,
        n_samples_in_=value.shape[0],
        authority_mode=authority_mode,
        heldout_domain=heldout,
        inner_query_domain=inner_query,
        fit_authority_state_sha256=fit_authority_state,
        outer_train_state_sha256=outer_authority_state,
        fit_embeddings_sha256=fit_embeddings_sha256,
    )


_pca_fit_capability = _take_pca_fit_capability()
_issue_fold_local_pca, _issued_pca_record = _pca_issuance_runtime(
    frozenset({fit_embedding_pca.__code__}),
    _pca_fit_capability,
    _DATA_AUTHORITY_VALIDATOR,
    _DATA_V3_DATA_VIEW,
    tuple(_DATA_DOMAIN_ORDER),
    tuple(_DATA_PRIMARY_SPECIMEN_IDS),
    tuple(_DATA_PRIMARY_DATASET_IDS),
)
del _pca_fit_capability
del _pca_issuance_runtime
del _take_pca_fit_capability
del _validate_pca_constructor_capability
del _DATA_AUTHORITY_VALIDATOR


def transform_embedding_pca(model: FoldLocalPCA, embeddings: np.ndarray) -> np.ndarray:
    """Transform arbitrary finite query rows using a previously fitted fold PCA."""

    if not isinstance(model, FoldLocalPCA):
        raise FeatureValidationError("PCA model has an invalid type")
    validate_fold_local_pca(model)
    value = _validate_embedding_matrix(embeddings, "embeddings")
    if value.shape[1] != model.n_features_in_:
        raise FeatureValidationError(
            "embedding feature count does not match fitted PCA"
        )
    output = (value - model.mean_) @ model.components_.T
    if not np.all(np.isfinite(output)):
        raise FeatureValidationError("PCA output must contain only finite values")
    return output


def fit_fold_local_pca(
    train_embeddings: np.ndarray,
    *,
    n_components: int,
    fit_sample_ids: Sequence[str] | None = None,
    fit_domain_ids: Sequence[str] | None = None,
    outer_train_authority: object | None = None,
    heldout_domain: str | None = None,
    inner_train_authority: object | None = None,
    parent_outer_authority: object | None = None,
    inner_query_domain: str | None = None,
) -> FoldLocalPCA:
    return fit_embedding_pca(
        train_embeddings,
        n_components=n_components,
        fit_sample_ids=fit_sample_ids,
        fit_domain_ids=fit_domain_ids,
        outer_train_authority=outer_train_authority,
        heldout_domain=heldout_domain,
        inner_train_authority=inner_train_authority,
        parent_outer_authority=parent_outer_authority,
        inner_query_domain=inner_query_domain,
    )


fit_pca = fit_embedding_pca
transform_pca = transform_embedding_pca


def select_pca_dimension(
    train_embeddings: np.ndarray,
    validation_scores: Mapping[int, float] | None = None,
    *,
    candidate_dimensions: Iterable[int] = PCA_DIMENSIONS,
    fit_sample_ids: Sequence[str] | None = None,
    fit_domain_ids: Sequence[str] | None = None,
    outer_train_authority: object | None = None,
    heldout_domain: str | None = None,
    inner_train_authority: object | None = None,
    parent_outer_authority: object | None = None,
    inner_query_domain: str | None = None,
) -> int:
    """Select a PCA dimension using only supplied inner-fold scores.

    With no scores, the smallest valid dimension is returned.  This fallback is
    deterministic and cannot inspect an outer-test matrix.
    """

    if isinstance(candidate_dimensions, (str, bytes)):
        raise FeatureValidationError("candidate PCA dimensions are not exact")
    try:
        requested_candidates = tuple(candidate_dimensions)
    except TypeError as error:
        raise FeatureValidationError(
            "candidate PCA dimensions are not exact"
        ) from error
    if requested_candidates != PCA_DIMENSIONS:
        raise FeatureValidationError(
            "candidate PCA dimensions must be exactly (8, 16, 32)"
        )
    valid = valid_pca_dimensions(train_embeddings, requested_candidates)
    if not valid:
        raise FeatureValidationError(
            "no registered PCA dimension is valid for the training fold"
        )
    authority_contract = _validate_pca_authority_contract(
        outer_train_authority,
        inner_train_authority,
        parent_outer_authority,
        heldout_domain,
        inner_query_domain,
        fit_sample_ids,
        fit_domain_ids,
        _validate_embedding_matrix(train_embeddings, "train embeddings").shape[0],
    )
    if not isinstance(validation_scores, Mapping):
        raise FeatureValidationError(
            "PCA selection requires an explicit inner-validation score mapping"
        )
    if set(validation_scores) != set(valid):
        raise FeatureValidationError("PCA selection scores are incomplete or unknown")
    scores: dict[int, float] = {}
    for dimension in valid:
        try:
            score = validation_scores[dimension]
        except (KeyError, TypeError) as error:
            raise FeatureValidationError(
                "PCA selection scores are incomplete"
            ) from error
        if (
            isinstance(score, (bool, np.bool_))
            or not isinstance(score, (float, int, np.floating, np.integer))
            or not np.isfinite(score)
        ):
            raise FeatureValidationError("PCA selection scores must be finite numbers")
        scores[dimension] = float(score)
    del authority_contract
    selected = valid[0]
    selected_score = scores[selected]
    for dimension in valid[1:]:
        if scores[dimension] < selected_score - 1e-12:
            selected = dimension
            selected_score = scores[dimension]
    return selected


def _canonical_npz_hash(path: Path) -> str:
    return sha256_file(path)


__all__ = [
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_DIMENSION",
    "EXPECTED_TRANSFORM_SHA",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LUMINANCE_COEFFICIENTS",
    "PCA_DIMENSIONS",
    "RESNET18_WEIGHTS_FILENAME",
    "RESNET18_WEIGHTS_RELATIVE_PATH",
    "RESNET18_WEIGHTS_SHA256",
    "RESNET18_WEIGHTS_URL",
    "TRANSFORM_SHA256",
    "TRANSFORM_SPEC",
    "EmbeddingCache",
    "FeatureValidationError",
    "FoldLocalPCA",
    "FrozenResNet18Encoder",
    "build_embedding_cache",
    "encode_resnet18",
    "extract_resnet18_embeddings",
    "fit_embedding_pca",
    "fit_fold_local_pca",
    "fit_pca",
    "grayscale_luminance",
    "load_embedding_cache",
    "normalize_imagenet_grayscale",
    "preprocess_full_field",
    "preprocess_image",
    "resize_grayscale_bilinear",
    "select_pca_dimension",
    "sha256_file",
    "to_grayscale_luminance",
    "transform_embedding_pca",
    "transform_pca",
    "valid_pca_dimensions",
    "validate_embedding_cache",
    "validate_embedding_cache_state",
    "validate_fold_local_pca",
    "validate_transform_spec",
]
