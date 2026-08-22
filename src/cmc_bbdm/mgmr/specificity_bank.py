"""Checksum-bound directional feature cache for registered P3 controls."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from cmc_bbdm.cpb_v3.embeddings import FrozenResNet18Encoder

from .authority import MGMRM0Authority
from .feature_wavelet import directional_gap, dwt2_feature_maps
from .m0_residual_audit import patch_shuffle_m0_images
from .protocol import MGMRProtocol


class MGMRSpecificityBankError(ValueError):
    """Raised when P3 directional cache bytes or authorities differ."""


_SEEDS = (20260831, 20260901, 20260902)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise MGMRSpecificityBankError(f"{label} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise MGMRSpecificityBankError(f"{label} must be a SHA-256") from error
    return value


def _readonly(value: object, rows: int) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(np.float32)
        or array.shape != (rows, 768)
        or not np.all(np.isfinite(array))
    ):
        raise MGMRSpecificityBankError(
            "P3 directional features must be finite float32 N x 768"
        )
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    output = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float32).reshape(
        contiguous.shape
    )
    output.setflags(write=False)
    return output


def _state(bank: MGMRSpecificityBank) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "config_sha256": bank.config_sha256,
                "control_sha256": dict(bank.control_sha256),
                "dataset_ids": bank.dataset_ids,
                "source_sha256": dict(bank.source_sha256),
                "specimen_ids": bank.specimen_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for seed in bank.seeds:
        digest.update(str(seed).encode("ascii"))
        digest.update(bank.directional[seed].tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MGMRSpecificityBank:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    directional: Mapping[int, np.ndarray]
    config_sha256: str
    source_sha256: Mapping[str, str]
    control_sha256: Mapping[int, str]
    state_sha256: str = ""

    def __post_init__(self) -> None:
        specimens = tuple(self.specimen_ids)
        datasets = tuple(self.dataset_ids)
        if (
            not specimens
            or len(set(specimens)) != len(specimens)
            or len(datasets) != len(specimens)
            or any(type(value) is not str or not value for value in (*specimens, *datasets))
        ):
            raise MGMRSpecificityBankError("P3 cohort IDs are invalid")
        if tuple(self.directional) != _SEEDS or tuple(self.control_sha256) != _SEEDS:
            raise MGMRSpecificityBankError("P3 seed roster changed")
        directional = MappingProxyType(
            {seed: _readonly(self.directional[seed], len(specimens)) for seed in _SEEDS}
        )
        sources = MappingProxyType(
            {
                name: _sha(digest, f"source {name}")
                for name, digest in sorted(self.source_sha256.items())
            }
        )
        controls = MappingProxyType(
            {seed: _sha(self.control_sha256[seed], f"control {seed}") for seed in _SEEDS}
        )
        object.__setattr__(self, "specimen_ids", specimens)
        object.__setattr__(self, "dataset_ids", datasets)
        object.__setattr__(self, "directional", directional)
        object.__setattr__(self, "config_sha256", _sha(self.config_sha256, "config"))
        object.__setattr__(self, "source_sha256", sources)
        object.__setattr__(self, "control_sha256", controls)
        state = _state(self)
        if self.state_sha256 and self.state_sha256 != state:
            raise MGMRSpecificityBankError("P3 feature bank state changed")
        object.__setattr__(self, "state_sha256", state)

    @property
    def seeds(self) -> tuple[int, ...]:
        return _SEEDS


@dataclass(frozen=True, slots=True)
class SpecificityBankPublication:
    manifest_sha256: str
    state_sha256: str
    files: Mapping[str, str]


def make_specificity_bank(
    *,
    specimen_ids: Sequence[str],
    dataset_ids: Sequence[str],
    directional: Mapping[int, object],
    config_sha256: str,
    source_sha256: Mapping[str, str],
    control_sha256: Mapping[int, str],
) -> MGMRSpecificityBank:
    return MGMRSpecificityBank(
        specimen_ids=tuple(specimen_ids),
        dataset_ids=tuple(dataset_ids),
        directional=directional,
        config_sha256=config_sha256,
        source_sha256=source_sha256,
        control_sha256=control_sha256,
    )


def extract_specificity_bank(
    protocol: MGMRProtocol,
    authority: MGMRM0Authority,
    encoder: FrozenResNet18Encoder,
    *,
    feature_bank_state_sha256: str,
    status_hook: Callable[[str], None] | None = None,
) -> MGMRSpecificityBank:
    """Recompute layer3/DWT features after each exact P3 shuffle."""

    if (
        type(protocol) is not MGMRProtocol
        or type(authority) is not MGMRM0Authority
        or type(encoder) is not FrozenResNet18Encoder
    ):
        raise MGMRSpecificityBankError("issued P3 extraction inputs are required")
    notify = status_hook if status_hook is not None else lambda _message: None
    directional: dict[int, np.ndarray] = {}
    control_hashes: dict[int, str] = {}
    for seed in protocol.specificity_seeds:
        notify(f"encoding P3 shuffle seed {seed}")
        images, records = patch_shuffle_m0_images(
            authority.images,
            specimen_ids=authority.specimen_ids,
            dataset_ids=authority.dataset_ids,
            seed=seed,
        )
        maps = encoder.encode_spatial(images, layer=protocol.spatial_layer)
        directional[seed] = directional_gap(
            dwt2_feature_maps(
                maps, wavelet=protocol.wavelet, mode=protocol.wavelet_mode
            )
        )
        control_hashes[seed] = hashlib.sha256(
            json.dumps(
                [asdict(record) for record in records],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
    return make_specificity_bank(
        specimen_ids=authority.specimen_ids,
        dataset_ids=authority.dataset_ids,
        directional=directional,
        config_sha256=protocol.config_sha256,
        source_sha256={
            "authority": authority.state_sha256,
            "feature_bank": _sha(feature_bank_state_sha256, "feature bank state"),
            "resnet_weights": protocol.sources["resnet_weights"].sha256,
        },
        control_sha256=control_hashes,
    )


def publish_specificity_bank(
    output: str | Path, bank: MGMRSpecificityBank
) -> SpecificityBankPublication:
    if type(bank) is not MGMRSpecificityBank:
        raise MGMRSpecificityBankError("issued P3 bank is required")
    destination = Path(output).resolve()
    if destination.exists():
        raise MGMRSpecificityBankError("P3 feature bank already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        files: dict[str, str] = {}
        for seed in bank.seeds:
            name = f"directional_{seed}.npy"
            np.save(staging / name, bank.directional[seed], allow_pickle=False)
            files[name] = hashlib.sha256((staging / name).read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "specimen_ids": list(bank.specimen_ids),
            "dataset_ids": list(bank.dataset_ids),
            "config_sha256": bank.config_sha256,
            "source_sha256": dict(bank.source_sha256),
            "control_sha256": {str(seed): bank.control_sha256[seed] for seed in bank.seeds},
            "arrays": {
                str(seed): {"dtype": "float32", "shape": list(bank.directional[seed].shape)}
                for seed in bank.seeds
            },
            "files": files,
            "state_sha256": bank.state_sha256,
        }
        payload = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("ascii")
        (staging / "manifest.json").write_bytes(payload)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return SpecificityBankPublication(
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        state_sha256=bank.state_sha256,
        files=MappingProxyType(files),
    )


def load_specificity_bank(
    output: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_specimen_ids: Sequence[str],
    expected_dataset_ids: Sequence[str],
    expected_config_sha256: str,
) -> MGMRSpecificityBank:
    root = Path(output).resolve(strict=True)
    names = {"manifest.json", *(f"directional_{seed}.npy" for seed in _SEEDS)}
    if not root.is_dir() or root.is_symlink() or {path.name for path in root.iterdir()} != names:
        raise MGMRSpecificityBankError("P3 feature bank file roster changed")
    payload = (root / "manifest.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != _sha(
        expected_manifest_sha256, "manifest"
    ):
        raise MGMRSpecificityBankError("P3 manifest SHA-256 changed")
    try:
        manifest = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MGMRSpecificityBankError("P3 manifest cannot be decoded") from error
    if (
        manifest.get("schema_version") != 1
        or tuple(manifest.get("specimen_ids", ())) != tuple(expected_specimen_ids)
        or tuple(manifest.get("dataset_ids", ())) != tuple(expected_dataset_ids)
        or manifest.get("config_sha256") != expected_config_sha256
    ):
        raise MGMRSpecificityBankError("P3 manifest authority changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != names - {"manifest.json"}:
        raise MGMRSpecificityBankError("P3 array hash roster changed")
    arrays: dict[int, np.ndarray] = {}
    for seed in _SEEDS:
        name = f"directional_{seed}.npy"
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != files[name]:
            raise MGMRSpecificityBankError("P3 array SHA-256 changed")
        arrays[seed] = np.load(root / name, mmap_mode="r", allow_pickle=False)
    bank = make_specificity_bank(
        specimen_ids=manifest["specimen_ids"],
        dataset_ids=manifest["dataset_ids"],
        directional=arrays,
        config_sha256=manifest["config_sha256"],
        source_sha256=manifest["source_sha256"],
        control_sha256={int(seed): digest for seed, digest in manifest["control_sha256"].items()},
    )
    if bank.state_sha256 != manifest.get("state_sha256"):
        raise MGMRSpecificityBankError("P3 feature bank state SHA-256 changed")
    for seed in _SEEDS:
        if manifest["arrays"].get(str(seed)) != {
            "dtype": "float32",
            "shape": list(bank.directional[seed].shape),
        }:
            raise MGMRSpecificityBankError("P3 array metadata changed")
    return bank


__all__ = [
    "MGMRSpecificityBank",
    "MGMRSpecificityBankError",
    "SpecificityBankPublication",
    "extract_specificity_bank",
    "load_specificity_bank",
    "make_specificity_bank",
    "publish_specificity_bank",
]
