"""Frozen cell-token feature bank for the CAI active-image experiment."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cmc_bbdm.cpb_v3.embeddings import encode_resnet18
from cmc_bbdm.learned_cscan.runtime import (
    load_study_config,
    load_study_context,
    render_registered_surface,
)
from cmc_bbdm.mva.encoder_session import MVAEncoderSession

from .data import load_mpa_targets
from .environment import NativeCellGrid
from .perception import load_frozen_vlm_cache
from .protocol import EXTERNAL_SOURCE_ROOT_BINDING, CAIActiveImageProtocol


def _fixed_strings(values: tuple[str, ...]) -> np.ndarray:
    width = max(len(value) for value in values)
    return np.asarray(values, dtype=f"<U{width}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureBank:
    specimen_keys: tuple[str, ...]
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    splits: tuple[str, ...]
    targets_mpa: np.ndarray
    surface_tokens: np.ndarray
    cscan_tokens: np.ndarray
    native_shapes: np.ndarray
    vlm_indicators: np.ndarray
    vlm_confidences: np.ndarray
    vlm_available: np.ndarray
    vlm_no_reliable: np.ndarray
    vlm_cache_keys: tuple[str, ...]
    surface_sha256: tuple[str, ...]
    cscan_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.specimen_keys)
        fit_mask = np.asarray(
            [split in {"TRAIN", "VALID"} for split in self.splits], dtype=bool
        )
        target_values_valid = bool(
            np.all(np.isfinite(self.targets_mpa[fit_mask]))
            and np.all(
                np.isfinite(self.targets_mpa[~fit_mask])
                | np.isnan(self.targets_mpa[~fit_mask])
            )
        )
        if (
            count < 1
            or len(set(self.specimen_keys)) != count
            or any(
                len(values) != count
                for values in (
                    self.specimen_ids,
                    self.dataset_ids,
                    self.splits,
                    self.vlm_cache_keys,
                    self.surface_sha256,
                    self.cscan_sha256,
                )
            )
            or self.targets_mpa.shape != (count,)
            or self.surface_tokens.ndim != 3
            or self.surface_tokens.shape[:2] != (count, 64)
            or self.cscan_tokens.shape != self.surface_tokens.shape
            or self.native_shapes.shape != (count, 2)
            or self.vlm_indicators.shape != (count, 64)
            or self.vlm_confidences.shape != (count, 64)
            or self.vlm_available.shape != (count,)
            or self.vlm_no_reliable.shape != (count,)
            or not target_values_valid
            or not np.all(np.isfinite(self.surface_tokens))
            or not np.all(np.isfinite(self.cscan_tokens))
            or any(split not in {"TRAIN", "VALID", "TEST"} for split in self.splits)
        ):
            raise ValueError("CAI feature bank is invalid")

    @property
    def token_dimension(self) -> int:
        return int(self.surface_tokens.shape[2])

    def indices(self, split: str) -> np.ndarray:
        if split not in {"TRAIN", "VALID", "TEST"}:
            raise ValueError("feature-bank split is invalid")
        return np.asarray(
            [index for index, value in enumerate(self.splits) if value == split],
            dtype=np.int64,
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            specimen_keys=_fixed_strings(self.specimen_keys),
            specimen_ids=_fixed_strings(self.specimen_ids),
            dataset_ids=_fixed_strings(self.dataset_ids),
            splits=_fixed_strings(self.splits),
            targets_mpa=np.asarray(self.targets_mpa, dtype=np.float32),
            surface_tokens=np.asarray(self.surface_tokens, dtype=np.float32),
            cscan_tokens=np.asarray(self.cscan_tokens, dtype=np.float32),
            native_shapes=np.asarray(self.native_shapes, dtype=np.int64),
            vlm_indicators=np.asarray(self.vlm_indicators, dtype=np.float32),
            vlm_confidences=np.asarray(self.vlm_confidences, dtype=np.float32),
            vlm_available=np.asarray(self.vlm_available, dtype=bool),
            vlm_no_reliable=np.asarray(self.vlm_no_reliable, dtype=bool),
            vlm_cache_keys=_fixed_strings(self.vlm_cache_keys),
            surface_sha256=_fixed_strings(self.surface_sha256),
            cscan_sha256=_fixed_strings(self.cscan_sha256),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> FeatureBank:
        try:
            with np.load(Path(path), allow_pickle=False) as payload:
                return cls(
                    specimen_keys=tuple(str(item) for item in payload["specimen_keys"]),
                    specimen_ids=tuple(str(item) for item in payload["specimen_ids"]),
                    dataset_ids=tuple(str(item) for item in payload["dataset_ids"]),
                    splits=tuple(str(item) for item in payload["splits"]),
                    targets_mpa=np.asarray(payload["targets_mpa"], dtype=np.float32),
                    surface_tokens=np.asarray(payload["surface_tokens"], dtype=np.float32),
                    cscan_tokens=np.asarray(payload["cscan_tokens"], dtype=np.float32),
                    native_shapes=np.asarray(payload["native_shapes"], dtype=np.int64),
                    vlm_indicators=np.asarray(payload["vlm_indicators"], dtype=np.float32),
                    vlm_confidences=np.asarray(payload["vlm_confidences"], dtype=np.float32),
                    vlm_available=np.asarray(payload["vlm_available"], dtype=bool),
                    vlm_no_reliable=np.asarray(payload["vlm_no_reliable"], dtype=bool),
                    vlm_cache_keys=tuple(str(item) for item in payload["vlm_cache_keys"]),
                    surface_sha256=tuple(str(item) for item in payload["surface_sha256"]),
                    cscan_sha256=tuple(str(item) for item in payload["cscan_sha256"]),
                )
        except (KeyError, OSError, ValueError) as error:
            raise ValueError("CAI feature bank cannot be loaded") from error


def _cell_crops(image: np.ndarray) -> list[np.ndarray]:
    grid = NativeCellGrid.from_shape(tuple(int(value) for value in image.shape[:2]))
    return [
        np.ascontiguousarray(
            image[cell.row_start : cell.row_stop, cell.col_start : cell.col_stop]
        )
        for cell in grid.cells
    ]


def build_feature_bank(
    protocol: CAIActiveImageProtocol,
    *,
    project_root: str | Path,
    source_root: str | Path,
    device: str | None = None,
) -> tuple[FeatureBank, dict[str, object]]:
    started = time.perf_counter()
    root = Path(project_root).resolve(strict=True)
    study_config = load_study_config(
        protocol.source("learned_cscan_config"), project_root=root
    )
    context = load_study_context(
        study_config,
        source_root=source_root,
        verify_pilot_hashes=True,
    )
    cache = load_frozen_vlm_cache(
        protocol.source("vlm_cache"), protocol.source("vlm_manifest")
    )
    fit_keys = {
        assignment.record.specimen_key
        for assignment in context.roster.assignments
        if assignment.split.value in {"TRAIN", "VALID"}
    }
    targets = load_mpa_targets(
        protocol.source("cai_mpa_authority"), allowed_keys=fit_keys
    )
    if set(targets) != fit_keys:
        raise ValueError("fit/validation CAI MPa target coverage changed")
    encoder_module = sys.modules[encode_resnet18.__module__]
    encoder_root = Path(encoder_module.__file__).resolve().parents[3]
    encoder_weights = encoder_root / "paper_v3/assets/resnet18-f37072fd.pth"
    if _sha256_file(encoder_weights) != next(
        item.sha256 for item in protocol.sources if item.name == "resnet18_weights"
    ):
        raise ValueError("executed frozen ResNet18 weight copy differs from binding")
    encoder = encode_resnet18(
        weight_path="paper_v3/assets/resnet18-f37072fd.pth",
        project_root=encoder_root,
        device=device or protocol.device,
    )
    session = MVAEncoderSession(encoder)

    specimen_keys: list[str] = []
    specimen_ids: list[str] = []
    dataset_ids: list[str] = []
    splits: list[str] = []
    target_values: list[float] = []
    surface_tokens: list[np.ndarray] = []
    cscan_tokens: list[np.ndarray] = []
    native_shapes: list[tuple[int, int]] = []
    vlm_indicators: list[np.ndarray] = []
    vlm_confidences: list[np.ndarray] = []
    vlm_available: list[bool] = []
    vlm_no_reliable: list[bool] = []
    vlm_cache_keys: list[str] = []
    surface_hashes: list[str] = []
    cscan_hashes: list[str] = []

    for assignment in context.roster.assignments:
        record = assignment.record
        registered = render_registered_surface(
            record,
            max_edge=int(context.config.legacy_config.values["surface"]["max_edge"]),
        )
        cached = cache[record.specimen_key]
        if cached.cache_key == "" or registered.render.clean_sha256 != next(
            entry["clean_image_sha256"]
            for entry in json.loads(
                protocol.source("vlm_manifest").read_text(encoding="utf-8")
            )["records"]
            if entry["specimen_key"] == record.specimen_key
        ):
            raise ValueError("registered surface and frozen VLM cache differ")
        surface = np.asarray(registered.render.clean, dtype=np.uint8)
        full_scan = np.asarray(
            context.authority.source_teacher_view(record.specimen_id).full_scan,
            dtype=np.uint8,
        )
        if full_scan.shape[:2] != record.native_shape:
            raise ValueError("registered C-scan native shape changed")
        encoded = session.encode(_cell_crops(surface) + _cell_crops(full_scan))
        if encoded.shape != (128, 512):
            raise ValueError("frozen cell-token encoding is incomplete")

        specimen_keys.append(record.specimen_key)
        specimen_ids.append(record.specimen_id)
        dataset_ids.append(record.dataset_id)
        splits.append(assignment.split.value)
        target_values.append(
            targets[record.specimen_key]
            if assignment.split.value in {"TRAIN", "VALID"}
            else float("nan")
        )
        surface_tokens.append(encoded[:64])
        cscan_tokens.append(encoded[64:])
        native_shapes.append(record.native_shape)
        vlm_indicators.append(cached.features.region_indicator)
        vlm_confidences.append(cached.features.confidence)
        vlm_available.append(cached.features.available)
        vlm_no_reliable.append(cached.features.no_reliable_cue)
        vlm_cache_keys.append(cached.cache_key)
        surface_hashes.append(registered.render.clean_sha256)
        cscan_hashes.append(_sha256_array(full_scan))

    bank = FeatureBank(
        specimen_keys=tuple(specimen_keys),
        specimen_ids=tuple(specimen_ids),
        dataset_ids=tuple(dataset_ids),
        splits=tuple(splits),
        targets_mpa=np.asarray(target_values, dtype=np.float32),
        surface_tokens=np.asarray(surface_tokens, dtype=np.float32),
        cscan_tokens=np.asarray(cscan_tokens, dtype=np.float32),
        native_shapes=np.asarray(native_shapes, dtype=np.int64),
        vlm_indicators=np.asarray(vlm_indicators, dtype=np.float32),
        vlm_confidences=np.asarray(vlm_confidences, dtype=np.float32),
        vlm_available=np.asarray(vlm_available, dtype=bool),
        vlm_no_reliable=np.asarray(vlm_no_reliable, dtype=bool),
        vlm_cache_keys=tuple(vlm_cache_keys),
        surface_sha256=tuple(surface_hashes),
        cscan_sha256=tuple(cscan_hashes),
    )
    manifest = {
        "schema_version": 2,
        "protocol_sha256": protocol.config_sha256,
        "specimen_count": len(bank.specimen_keys),
        "split_counts": {
            split: int(np.sum(np.asarray(bank.splits) == split))
            for split in ("TRAIN", "VALID", "TEST")
        },
        "token_shape": list(bank.surface_tokens.shape),
        "encoder": encoder.provenance(),
        "encoder_execution_root": EXTERNAL_SOURCE_ROOT_BINDING,
        "encoder_elapsed_seconds": time.perf_counter() - started,
        "vlm_cache_hits": len(cache),
        "vlm_new_calls": 0,
        "vlm_original_calls": sum(item.original_call_count for item in cache.values()),
        "cai_unit": "MPa",
        "target_min_mpa": float(np.nanmin(bank.targets_mpa)),
        "target_max_mpa": float(np.nanmax(bank.targets_mpa)),
        "test_targets_stored": False,
        "test_targets_join_stage": "P5_EVALUATION_ONLY",
        "source_hashes": {item.name: item.sha256 for item in protocol.sources},
    }
    return bank, manifest


def save_feature_bank(
    bank: FeatureBank,
    manifest: dict[str, object],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    bank_path = bank.save(root / "feature_bank.npz")
    payload = dict(manifest)
    payload["feature_bank_sha256"] = _sha256_file(bank_path)
    manifest_path = root / "feature_bank_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bank_path, manifest_path


__all__ = ["FeatureBank", "build_feature_bank", "save_feature_bank"]
