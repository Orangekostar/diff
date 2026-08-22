"""Immutable, specimen-ordered spatial feature bank for MGMR M0."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .feature_wavelet import directional_gap, dwt2_feature_maps


class MGMRFeatureBankError(ValueError):
    """Raised when the M0 feature bank is incomplete or has drifted."""


_ARRAY_FILES = {
    "full_global": "full_global.npy",
    "full_layer3": "full_layer3.npy",
    "coarse_layer3": "coarse_layer3.npy",
    "coarse_gap": "coarse_gap.npy",
    "full_directional": "full_directional.npy",
}
_MANIFEST_KEYS = {
    "schema_version",
    "specimen_ids",
    "dataset_ids",
    "config_sha256",
    "source_sha256",
    "wavelet",
    "wavelet_mode",
    "arrays",
    "files",
    "state_sha256",
}


def _sha256_text(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise MGMRFeatureBankError(f"{label} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise MGMRFeatureBankError(f"{label} must be a SHA-256") from error
    return value


def _ids(value: Sequence[str], label: str) -> tuple[str, ...]:
    output = tuple(value)
    if (
        not output
        or any(type(item) is not str or not item or item.strip() != item for item in output)
        or len(set(output)) != len(output)
    ):
        raise MGMRFeatureBankError(f"{label} are invalid")
    return output


def _array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(np.float32)
        or array.shape != shape
        or not np.all(np.isfinite(array))
    ):
        raise MGMRFeatureBankError(f"{label} must be finite float32 with shape {shape}")
    if array.flags.c_contiguous and not array.flags.writeable:
        return array
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float32).reshape(
        contiguous.shape
    )
    output.setflags(write=False)
    return output


def _source_hashes(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise MGMRFeatureBankError("source SHA-256 registry is required")
    output: dict[str, str] = {}
    for name, digest in sorted(value.items()):
        if type(name) is not str or not name:
            raise MGMRFeatureBankError("source SHA-256 name is invalid")
        output[name] = _sha256_text(digest, f"source {name}")
    return MappingProxyType(output)


def _bank_state(bank: MGMRFeatureBank) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "config_sha256": bank.config_sha256,
                "dataset_ids": bank.dataset_ids,
                "source_sha256": dict(bank.source_sha256),
                "specimen_ids": bank.specimen_ids,
                "wavelet": bank.wavelet,
                "wavelet_mode": bank.wavelet_mode,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for name, array in zip(_ARRAY_FILES, bank.arrays, strict=True):
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MGMRFeatureBank:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    full_global: np.ndarray
    full_layer3: np.ndarray
    coarse_layer3: np.ndarray
    coarse_gap: np.ndarray
    full_directional: np.ndarray
    config_sha256: str
    source_sha256: Mapping[str, str]
    wavelet: str
    wavelet_mode: str
    state_sha256: str = ""

    def __post_init__(self) -> None:
        specimen_ids = _ids(self.specimen_ids, "specimen IDs")
        dataset_ids = tuple(self.dataset_ids)
        if (
            len(dataset_ids) != len(specimen_ids)
            or any(type(item) is not str or not item for item in dataset_ids)
        ):
            raise MGMRFeatureBankError("dataset IDs do not align")
        rows = len(specimen_ids)
        full_global = _array(self.full_global, (rows, 512), "FULL global features")
        full_layer3 = _array(
            self.full_layer3, (rows, 256, 14, 14), "FULL layer3 maps"
        )
        coarse_layer3 = _array(
            self.coarse_layer3, (rows, 256, 14, 14), "coarse layer3 maps"
        )
        coarse_gap = _array(self.coarse_gap, (rows, 256), "coarse GAP features")
        full_directional = _array(
            self.full_directional, (rows, 768), "FULL directional features"
        )
        config_sha256 = _sha256_text(self.config_sha256, "config SHA-256")
        source_sha256 = _source_hashes(self.source_sha256)
        if self.wavelet != "db2" or self.wavelet_mode != "periodization":
            raise MGMRFeatureBankError("feature bank wavelet registry changed")
        object.__setattr__(self, "specimen_ids", specimen_ids)
        object.__setattr__(self, "dataset_ids", dataset_ids)
        object.__setattr__(self, "full_global", full_global)
        object.__setattr__(self, "full_layer3", full_layer3)
        object.__setattr__(self, "coarse_layer3", coarse_layer3)
        object.__setattr__(self, "coarse_gap", coarse_gap)
        object.__setattr__(self, "full_directional", full_directional)
        object.__setattr__(self, "config_sha256", config_sha256)
        object.__setattr__(self, "source_sha256", source_sha256)
        state = _bank_state(self)
        if self.state_sha256 and self.state_sha256 != state:
            raise MGMRFeatureBankError("feature bank state SHA-256 changed")
        object.__setattr__(self, "state_sha256", state)

    @property
    def arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.full_global,
            self.full_layer3,
            self.coarse_layer3,
            self.coarse_gap,
            self.full_directional,
        )


@dataclass(frozen=True, slots=True)
class FeatureBankPublication:
    manifest_sha256: str
    state_sha256: str
    files: Mapping[str, str]


def make_feature_bank(
    *,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    full_global: object,
    full_layer3: object,
    coarse_layer3: object,
    config_sha256: str,
    source_sha256: Mapping[str, str],
    wavelet: str,
    wavelet_mode: str,
) -> MGMRFeatureBank:
    """Derive the frozen M0 component vectors from ordered spatial maps."""

    full_maps = np.asarray(full_layer3)
    coarse_maps = np.asarray(coarse_layer3)
    if full_maps.dtype != np.dtype(np.float32) or coarse_maps.dtype != np.dtype(
        np.float32
    ):
        raise MGMRFeatureBankError("spatial feature maps must use float32")
    coarse_gap = np.mean(coarse_maps, axis=(-2, -1), dtype=np.float32)
    directional = directional_gap(
        dwt2_feature_maps(full_maps, wavelet=wavelet, mode=wavelet_mode)
    )
    return MGMRFeatureBank(
        specimen_ids=tuple(specimen_ids),
        dataset_ids=tuple(dataset_ids),
        full_global=np.asarray(full_global),
        full_layer3=full_maps,
        coarse_layer3=coarse_maps,
        coarse_gap=coarse_gap,
        full_directional=directional,
        config_sha256=config_sha256,
        source_sha256=source_sha256,
        wavelet=wavelet,
        wavelet_mode=wavelet_mode,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_feature_bank(
    output: str | Path, bank: MGMRFeatureBank
) -> FeatureBankPublication:
    """Atomically persist deterministic NPY arrays and a checksum manifest."""

    if type(bank) is not MGMRFeatureBank:
        raise MGMRFeatureBankError("issued MGMRFeatureBank is required")
    destination = Path(output).resolve()
    if destination.exists():
        raise MGMRFeatureBankError(f"feature bank already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        file_hashes: dict[str, str] = {}
        arrays_meta: dict[str, dict[str, object]] = {}
        for (name, filename), array in zip(
            _ARRAY_FILES.items(), bank.arrays, strict=True
        ):
            path = staging / filename
            np.save(path, array, allow_pickle=False)
            file_hashes[filename] = _file_sha256(path)
            arrays_meta[name] = {"dtype": "float32", "shape": list(array.shape)}
        manifest = {
            "schema_version": 1,
            "specimen_ids": list(bank.specimen_ids),
            "dataset_ids": list(bank.dataset_ids),
            "config_sha256": bank.config_sha256,
            "source_sha256": dict(bank.source_sha256),
            "wavelet": bank.wavelet,
            "wavelet_mode": bank.wavelet_mode,
            "arrays": arrays_meta,
            "files": file_hashes,
            "state_sha256": bank.state_sha256,
        }
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        (staging / "manifest.json").write_bytes(manifest_bytes)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return FeatureBankPublication(
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        state_sha256=bank.state_sha256,
        files=MappingProxyType(file_hashes),
    )


def load_feature_bank(
    path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_specimen_ids: Sequence[str],
    expected_dataset_ids: Sequence[str],
    expected_config_sha256: str,
) -> MGMRFeatureBank:
    """Load and verify every byte of a persisted feature bank."""

    root = Path(path).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise MGMRFeatureBankError("feature bank must be a regular directory")
    expected_files = {"manifest.json", *_ARRAY_FILES.values()}
    if {item.name for item in root.iterdir()} != expected_files:
        raise MGMRFeatureBankError("feature bank file roster changed")
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != _sha256_text(
        expected_manifest_sha256, "manifest SHA-256"
    ):
        raise MGMRFeatureBankError("manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MGMRFeatureBankError("feature bank manifest is invalid") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise MGMRFeatureBankError("feature bank manifest keys changed")
    if manifest["schema_version"] != 1:
        raise MGMRFeatureBankError("feature bank schema changed")
    if tuple(manifest["specimen_ids"]) != tuple(expected_specimen_ids):
        raise MGMRFeatureBankError("feature bank specimen order changed")
    if tuple(manifest["dataset_ids"]) != tuple(expected_dataset_ids):
        raise MGMRFeatureBankError("feature bank dataset order changed")
    if manifest["config_sha256"] != _sha256_text(
        expected_config_sha256, "expected config SHA-256"
    ):
        raise MGMRFeatureBankError("feature bank config SHA-256 changed")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(_ARRAY_FILES.values()):
        raise MGMRFeatureBankError("feature bank file hashes changed")
    for filename, expected in files.items():
        target = root / filename
        if target.is_symlink() or not target.is_file():
            raise MGMRFeatureBankError("feature bank array file is invalid")
        if _file_sha256(target) != _sha256_text(expected, f"file {filename}"):
            raise MGMRFeatureBankError(f"feature bank file SHA-256 mismatch: {filename}")
    arrays = {
        name: np.load(root / filename, mmap_mode="r", allow_pickle=False)
        for name, filename in _ARRAY_FILES.items()
    }
    bank = MGMRFeatureBank(
        specimen_ids=tuple(manifest["specimen_ids"]),
        dataset_ids=tuple(manifest["dataset_ids"]),
        full_global=arrays["full_global"],
        full_layer3=arrays["full_layer3"],
        coarse_layer3=arrays["coarse_layer3"],
        coarse_gap=arrays["coarse_gap"],
        full_directional=arrays["full_directional"],
        config_sha256=manifest["config_sha256"],
        source_sha256=manifest["source_sha256"],
        wavelet=manifest["wavelet"],
        wavelet_mode=manifest["wavelet_mode"],
        state_sha256=manifest["state_sha256"],
    )
    for name, array in zip(_ARRAY_FILES, bank.arrays, strict=True):
        metadata = manifest["arrays"].get(name)
        if metadata != {"dtype": "float32", "shape": list(array.shape)}:
            raise MGMRFeatureBankError("feature bank array metadata changed")
    return bank


__all__ = [
    "FeatureBankPublication",
    "MGMRFeatureBank",
    "MGMRFeatureBankError",
    "load_feature_bank",
    "make_feature_bank",
    "publish_feature_bank",
]
