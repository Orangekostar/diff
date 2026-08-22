"""Immutable cohort, image, feature, and structural authorities for MSSS."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmc_bbdm.aei_selective_invariance.feature_bank import PairedFeatureBank
from cmc_bbdm.cpb_physical_descriptors import load_physical_calibrations
from cmc_bbdm.cpb_spatial.pipeline import RegisteredInputs, load_registered_inputs
from cmc_bbdm.cpb_v3.config import load_config
from cmc_bbdm.cpb_v3.data import V3Data, load_data

from .protocol import MSSSProtocol, SourceAuthority


class MSSSAuthorityError(ValueError):
    """Raised when an MSSS input no longer matches its frozen authority."""


@dataclass(frozen=True, slots=True)
class MSSSAuthority:
    data: V3Data
    registered_inputs: RegisteredInputs
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    targets: np.ndarray
    metadata13: np.ndarray
    full_features: np.ndarray
    bilinear50_features: np.ndarray
    bilinear25_features: np.ndarray
    ply_count: np.ndarray
    layup_family: np.ndarray
    feature_bank_state_sha256: str
    state_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)


_CACHE: dict[tuple[str, str], MSSSAuthority] = {}


def _source(protocol: MSSSProtocol, name: str) -> SourceAuthority:
    matches = tuple(item for item in protocol.sources if item.name == name)
    if len(matches) != 1:
        raise MSSSAuthorityError(f"MSSS source is unavailable: {name}")
    return matches[0]


def _readonly(value: object, *, dtype: object) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    output.setflags(write=False)
    return output


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(
        json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_uncached(protocol: MSSSProtocol, root: Path) -> MSSSAuthority:
    config = load_config(_source(protocol, "p1_config").path, project_root=root)
    data = load_data(config, project_root=root)
    data.validate()
    calibration_source = config.sources["physical_calibration"]
    calibrations = load_physical_calibrations(
        root / calibration_source.path,
        project_root=root,
        expected_sha256=calibration_source.sha256,
    )
    inputs = load_registered_inputs(
        data, project_root=root, calibrations=calibrations
    )
    bank_source = _source(protocol, "paired_feature_bank")
    bank = PairedFeatureBank.load(
        bank_source.path, expected_sha256=bank_source.sha256
    )

    specimen_ids = tuple(str(item) for item in data.sample_ids.tolist())
    dataset_ids = tuple(str(item) for item in data.dataset_ids.tolist())
    if (
        len(specimen_ids) != protocol.specimen_count
        or specimen_ids != inputs.specimen_ids
        or dataset_ids != inputs.dataset_ids
        or specimen_ids != bank.specimen_ids
        or dataset_ids != bank.dataset_ids
        or bank.view_names != ("FULL", "BILINEAR_50", "BILINEAR_25")
        or bank.features.shape != (protocol.specimen_count, 3, 512)
        or tuple(dict.fromkeys(dataset_ids)) != protocol.domain_order
    ):
        raise MSSSAuthorityError("MSSS cohort, crops, and feature bank do not align")

    metadata = _readonly(data.metadata13, dtype="<f8")
    targets = _readonly(data.cai_ratio, dtype="<f8")
    features = np.asarray(bank.features, dtype=np.float64)
    full = _readonly(features[:, 0], dtype="<f8")
    bilinear50 = _readonly(features[:, 1], dtype="<f8")
    bilinear25 = _readonly(features[:, 2], dtype="<f8")
    if (
        metadata.shape != (protocol.specimen_count, 13)
        or targets.shape != (protocol.specimen_count,)
        or any(array.shape != (protocol.specimen_count, 512) for array in (full, bilinear50, bilinear25))
        or not all(np.all(np.isfinite(array)) for array in (metadata, targets, full, bilinear50, bilinear25))
    ):
        raise MSSSAuthorityError("MSSS numeric authority is incomplete")

    record_ply = np.asarray(
        [record.ply_count for record in data.cscan_records], dtype=np.int64
    )
    record_layup = tuple(record.laminate for record in data.cscan_records)
    metadata_ply = np.rint(metadata[:, 1] * 24.0).astype(np.int64)
    metadata_layup = tuple(
        "cross_ply" if value == 1.0 else "quasi_isotropic"
        for value in metadata[:, 2]
    )
    if (
        not np.array_equal(record_ply, metadata_ply)
        or record_layup != metadata_layup
        or set(record_ply.tolist()) != set(protocol.ply_counts)
        or set(record_layup) != set(protocol.layup_families)
        or any(
            not math.isclose(float(value), float(round(value)), rel_tol=0.0, abs_tol=1.0e-12)
            for value in metadata[:, 1] * 24.0
        )
    ):
        raise MSSSAuthorityError("ply-count or layup authority changed")
    ply = _readonly(record_ply, dtype="<i8")
    layup_width = max(len(value) for value in record_layup)
    layup = _readonly(record_layup, dtype=f"<U{layup_width}")

    state_payload = {
        "specimen_ids": specimen_ids,
        "dataset_ids": dataset_ids,
        "data_state_sha256": data.state_hash,
        "registered_inputs_state_sha256": inputs.state_sha256,
        "feature_bank_state_sha256": bank.state_sha256,
        "targets_sha256": _array_hash(targets),
        "metadata_sha256": _array_hash(metadata),
        "ply_sha256": _array_hash(ply),
        "layup_sha256": _array_hash(layup),
    }
    state = hashlib.sha256(
        json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    return MSSSAuthority(
        data=data,
        registered_inputs=inputs,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata13=metadata,
        full_features=full,
        bilinear50_features=bilinear50,
        bilinear25_features=bilinear25,
        ply_count=ply,
        layup_family=layup,
        feature_bank_state_sha256=bank.state_sha256,
        state_sha256=state,
    )


def load_authority(
    protocol: MSSSProtocol, *, project_root: str | Path
) -> MSSSAuthority:
    """Load and cross-bind every frozen MSSS data authority."""

    if type(protocol) is not MSSSProtocol:
        raise MSSSAuthorityError("issued MSSSProtocol is required")
    root = Path(project_root).resolve(strict=True)
    key = (protocol.config_sha256, str(root))
    cached = _CACHE.get(key)
    if cached is None:
        cached = _load_uncached(protocol, root)
        _CACHE[key] = cached
    return cached


__all__ = ["MSSSAuthority", "MSSSAuthorityError", "load_authority"]
