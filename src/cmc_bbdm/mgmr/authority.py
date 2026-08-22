"""Cross-bound cohort, crop, target, metadata, and FULL feature authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from cmc_bbdm.aei_selective_invariance.feature_bank import PairedFeatureBank
from cmc_bbdm.cpb_v3.config import load_config
from cmc_bbdm.cpb_v3.data import V3Data, load_data

from .protocol import MGMRProtocol


class MGMRAuthorityError(ValueError):
    """Raised when an M0 input no longer matches its frozen source authority."""


def _readonly(value: object, *, dtype: object) -> np.ndarray:
    try:
        array = np.ascontiguousarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise MGMRAuthorityError("numeric authority cannot be snapshotted") from error
    if not np.all(np.isfinite(array)):
        raise MGMRAuthorityError("numeric authority contains non-finite values")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    output.setflags(write=False)
    return output


def _readonly_image(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 3 or array.shape[2] != 3:
        raise MGMRAuthorityError("C-scan image must be RGB uint8")
    output = np.frombuffer(
        np.ascontiguousarray(array).tobytes(order="C"), dtype=np.uint8
    ).reshape(array.shape)
    output.setflags(write=False)
    return output


def _array_hash(value: np.ndarray) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {"dtype": value.dtype.str, "shape": value.shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MGMRM0Authority:
    data: V3Data
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    images: tuple[np.ndarray, ...]
    image_sha256: tuple[str, ...]
    targets: np.ndarray
    metadata13: np.ndarray
    full_global: np.ndarray
    paired_feature_bank_sha256: str
    state_sha256: str

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)


_CACHE: dict[tuple[str, str], MGMRM0Authority] = {}


def _load_uncached(protocol: MGMRProtocol, root: Path) -> MGMRM0Authority:
    p1_source = protocol.sources["p1_config"]
    p1_config = load_config(root / p1_source.path, project_root=root)
    data = load_data(p1_config, project_root=root)
    data.validate()
    specimen_ids = tuple(str(value) for value in data.sample_ids.tolist())
    dataset_ids = tuple(str(value) for value in data.dataset_ids.tolist())
    if (
        len(specimen_ids) != protocol.specimen_count
        or tuple(dict.fromkeys(dataset_ids)) != protocol.domain_order
    ):
        raise MGMRAuthorityError("V3 cohort changed")

    bank_source = protocol.sources["paired_feature_bank"]
    paired = PairedFeatureBank.load(
        root / bank_source.path, expected_sha256=bank_source.sha256
    )
    if (
        paired.specimen_ids != specimen_ids
        or paired.dataset_ids != dataset_ids
        or paired.view_names != ("FULL", "BILINEAR_50", "BILINEAR_25")
        or paired.features.shape != (protocol.specimen_count, 3, 512)
    ):
        raise MGMRAuthorityError("paired FULL feature authority changed")

    images: list[np.ndarray] = []
    hashes: list[str] = []
    for specimen_id, dataset_id, record in zip(
        specimen_ids, dataset_ids, data.cscan_records, strict=True
    ):
        if record.specimen_id != specimen_id or record.dataset_id != dataset_id:
            raise MGMRAuthorityError("C-scan record order changed")
        payload = record.read_bytes(root)
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise MGMRAuthorityError("C-scan bytes changed")
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                if image.mode != "RGB" or image.size != (record.width, record.height):
                    raise MGMRAuthorityError("C-scan mode or dimensions changed")
                snapshot = _readonly_image(np.asarray(image, dtype=np.uint8))
        except (OSError, ValueError) as error:
            raise MGMRAuthorityError("C-scan image cannot be decoded") from error
        images.append(snapshot)
        hashes.append(record.sha256)

    targets = _readonly(data.cai_ratio, dtype="<f8")
    metadata13 = _readonly(data.metadata13, dtype="<f8")
    full_global = _readonly(paired.features[:, 0], dtype="<f4")
    if (
        targets.shape != (protocol.specimen_count,)
        or metadata13.shape != (protocol.specimen_count, 13)
        or full_global.shape != (protocol.specimen_count, 512)
    ):
        raise MGMRAuthorityError("M0 numeric authority changed")
    state_payload = {
        "config_sha256": protocol.config_sha256,
        "data_state_sha256": data.state_hash,
        "dataset_ids": dataset_ids,
        "full_global_sha256": _array_hash(full_global),
        "image_sha256": hashes,
        "metadata13_sha256": _array_hash(metadata13),
        "paired_feature_bank_sha256": bank_source.sha256,
        "paired_feature_state_sha256": paired.state_sha256,
        "specimen_ids": specimen_ids,
        "targets_sha256": _array_hash(targets),
    }
    state = hashlib.sha256(
        json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    return MGMRM0Authority(
        data=data,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        images=tuple(images),
        image_sha256=tuple(hashes),
        targets=targets,
        metadata13=metadata13,
        full_global=full_global,
        paired_feature_bank_sha256=bank_source.sha256,
        state_sha256=state,
    )


def load_authority(
    protocol: MGMRProtocol, *, project_root: str | Path
) -> MGMRM0Authority:
    """Load all label-independent and evaluation inputs under one state hash."""

    if type(protocol) is not MGMRProtocol:
        raise MGMRAuthorityError("issued MGMRProtocol is required")
    root = Path(project_root).resolve(strict=True)
    key = (protocol.config_sha256, str(root))
    authority = _CACHE.get(key)
    if authority is None:
        authority = _load_uncached(protocol, root)
        _CACHE[key] = authority
    return authority


__all__ = ["MGMRAuthorityError", "MGMRM0Authority", "load_authority"]
