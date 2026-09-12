"""Sharded frozen-image feature bank for the complete v3 reused cohort."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from cmc_bbdm.cai_active_image.environment import NativeCellGrid
from cmc_bbdm.cpb_v3.embeddings import encode_resnet18
from cmc_bbdm.mva.encoder_session import MVAEncoderSession
from cmc_bbdm.vlm_cscan.runtime import render_surface_inputs

from .files import read_csv, sha256_file, write_csv, write_json

_TOKEN_DIMENSION = 512
_SURFACE_MAX_EDGE = 1024
_EXPECTED_WEIGHT_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _cell_crops(image: np.ndarray) -> list[np.ndarray]:
    grid = NativeCellGrid.from_shape(tuple(int(value) for value in image.shape[:2]))
    return [
        np.ascontiguousarray(
            image[cell.row_start : cell.row_stop, cell.col_start : cell.col_stop]
        )
        for cell in grid.cells
    ]


def _load_old_bank(path: Path) -> dict[str, dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        output = {}
        for index, key in enumerate(payload["specimen_keys"]):
            output[str(key)] = {
                "surface_tokens": np.asarray(
                    payload["surface_tokens"][index], dtype=np.float32
                ),
                "cscan_tokens": np.asarray(
                    payload["cscan_tokens"][index], dtype=np.float32
                ),
                "surface_sha256": str(payload["surface_sha256"][index]),
                "cscan_sha256": str(payload["cscan_sha256"][index]),
            }
    return output


@dataclass(frozen=True, slots=True)
class V3FeatureBank:
    specimen_keys: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    specimen_ids: tuple[str, ...]
    capture_group_ids: tuple[str, ...]
    splits: tuple[str, ...]
    targets_mpa: np.ndarray
    surface_tokens: np.ndarray
    cscan_tokens: np.ndarray
    full_surface_tokens: np.ndarray
    full_cscan_tokens: np.ndarray
    native_shapes: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.specimen_keys)
        fit = np.asarray([split != "TEST" for split in self.splits], dtype=bool)
        if (
            count != len(set(self.specimen_keys))
            or any(
                len(values) != count
                for values in (
                    self.dataset_ids,
                    self.specimen_ids,
                    self.capture_group_ids,
                    self.splits,
                )
            )
            or self.targets_mpa.shape != (count,)
            or self.surface_tokens.shape != (count, 64, _TOKEN_DIMENSION)
            or self.cscan_tokens.shape != (count, 64, _TOKEN_DIMENSION)
            or self.full_surface_tokens.shape != (count, _TOKEN_DIMENSION)
            or self.full_cscan_tokens.shape != (count, _TOKEN_DIMENSION)
            or self.native_shapes.shape != (count, 2)
            or not np.all(np.isfinite(self.targets_mpa[fit]))
            or not np.all(np.isnan(self.targets_mpa[~fit]))
            or not all(
                np.all(np.isfinite(values))
                for values in (
                    self.surface_tokens,
                    self.cscan_tokens,
                    self.full_surface_tokens,
                    self.full_cscan_tokens,
                )
            )
            or any(split not in {"TRAIN", "VALID", "TEST"} for split in self.splits)
        ):
            raise ValueError("v3 feature bank is invalid")

    def indices(self, split: str) -> np.ndarray:
        if split not in {"TRAIN", "VALID", "TEST"}:
            raise ValueError("unknown feature-bank split")
        return np.asarray(
            [index for index, value in enumerate(self.splits) if value == split],
            dtype=np.int64,
        )


def load_feature_bank(*, project_root: str | Path) -> V3FeatureBank:
    root = Path(project_root).resolve(strict=True)
    index_rows = read_csv(
        root / "results/cai_agent_v3/new_protocol/feature_bank_index.csv"
    )
    if not index_rows:
        raise ValueError("v3 feature bank index is empty")
    arrays: dict[str, list[np.ndarray]] = {
        "targets_mpa": [],
        "surface_tokens": [],
        "cscan_tokens": [],
        "full_surface_tokens": [],
        "full_cscan_tokens": [],
        "native_shapes": [],
    }
    identities: dict[str, list[str]] = {
        "specimen_keys": [],
        "dataset_ids": [],
        "specimen_ids": [],
        "capture_group_ids": [],
        "splits": [],
    }
    loaded_shards: dict[str, dict[str, np.ndarray]] = {}
    for row in index_rows:
        shard_path = row["shard_path"]
        if shard_path not in loaded_shards:
            with np.load(root / shard_path, allow_pickle=False) as payload:
                loaded_shards[shard_path] = {
                    name: np.asarray(payload[name])
                    for name in (
                        "specimen_keys",
                        "dataset_ids",
                        "specimen_ids",
                        "capture_group_ids",
                        "splits",
                        *arrays,
                    )
                }
        shard = loaded_shards[shard_path]
        offset = int(row["shard_index"])
        if str(shard["specimen_keys"][offset]) != row["specimen_key"]:
            raise ValueError("feature-bank index and shard identity differ")
        for name, values in identities.items():
            values.append(str(shard[name][offset]))
        for name, values in arrays.items():
            values.append(np.asarray(shard[name][offset]))
    return V3FeatureBank(
        specimen_keys=tuple(identities["specimen_keys"]),
        dataset_ids=tuple(identities["dataset_ids"]),
        specimen_ids=tuple(identities["specimen_ids"]),
        capture_group_ids=tuple(identities["capture_group_ids"]),
        splits=tuple(identities["splits"]),
        targets_mpa=np.asarray(arrays["targets_mpa"], dtype=np.float32),
        surface_tokens=np.asarray(arrays["surface_tokens"], dtype=np.float32),
        cscan_tokens=np.asarray(arrays["cscan_tokens"], dtype=np.float32),
        full_surface_tokens=np.asarray(arrays["full_surface_tokens"], dtype=np.float32),
        full_cscan_tokens=np.asarray(arrays["full_cscan_tokens"], dtype=np.float32),
        native_shapes=np.asarray(arrays["native_shapes"], dtype=np.int64),
    )


def build_feature_bank(
    *,
    project_root: str | Path,
    source_root: str | Path,
    device: str,
) -> dict[str, object]:
    started = time.perf_counter()
    root = Path(project_root).resolve(strict=True)
    external = Path(source_root).resolve(strict=True)
    output = root / "results/cai_agent_v3/new_protocol"
    candidates = read_csv(output / "candidate_queue.csv")
    if len(candidates) != 276:
        raise ValueError("feature encoding requires the complete accepted queue")
    old_bank = _load_old_bank(root / "results/cai_active_image/v2/feature_bank.npz")
    encoder_module = sys.modules[encode_resnet18.__module__]
    encoder_root = Path(encoder_module.__file__).resolve().parents[3]
    weight_path = encoder_root / "paper_v3/assets/resnet18-f37072fd.pth"
    if sha256_file(weight_path) != _EXPECTED_WEIGHT_SHA256:
        raise ValueError("executed frozen ResNet18 weights differ from the binding")
    encoder = encode_resnet18(
        weight_path="paper_v3/assets/resnet18-f37072fd.pth",
        project_root=encoder_root,
        device=device,
    )
    session = MVAEncoderSession(encoder)
    records: list[dict[str, object]] = []
    reuse_count = 0
    encoded_count = 0
    for position, row in enumerate(candidates, start=1):
        key = row["specimen_key"]
        surface_path = external / row["impacted_surface_path"]
        cscan_path = external / row["registered_cscan_crop_path"]
        if sha256_file(surface_path) != row["surface_sha256"]:
            raise ValueError(f"surface source hash changed: {key}")
        if sha256_file(cscan_path) != row["registered_cscan_crop_sha256"]:
            raise ValueError(f"registered C-scan source hash changed: {key}")
        with Image.open(surface_path) as image:
            surface_render = render_surface_inputs(image, max_edge=_SURFACE_MAX_EDGE)
            surface = np.asarray(surface_render.clean, dtype=np.uint8)
        with Image.open(cscan_path) as image:
            cscan = np.asarray(image.convert("RGB"), dtype=np.uint8)
        native_shape = (
            int(row["registered_cscan_height_px"]),
            int(row["registered_cscan_width_px"]),
        )
        if cscan.shape[:2] != native_shape:
            raise ValueError(f"registered C-scan dimensions changed: {key}")
        cscan_array_sha256 = _array_sha256(cscan)
        old = old_bank.get(key)
        reusable = bool(
            old is not None
            and old["surface_sha256"] == surface_render.clean_sha256
            and old["cscan_sha256"] == cscan_array_sha256
        )
        if reusable:
            cell_tokens = None
            reuse_count += 1
        else:
            cell_tokens = session.encode(_cell_crops(surface) + _cell_crops(cscan))
            if cell_tokens.shape != (128, _TOKEN_DIMENSION):
                raise ValueError(f"cell feature encoding is incomplete: {key}")
            encoded_count += 1
        full_tokens = session.encode((surface, cscan))
        if full_tokens.shape != (2, _TOKEN_DIMENSION):
            raise ValueError(f"full-image diagnostic encoding is incomplete: {key}")
        records.append(
            {
                "identity": row,
                "surface_tokens": (
                    old["surface_tokens"] if reusable else cell_tokens[:64]
                ),
                "cscan_tokens": old["cscan_tokens"] if reusable else cell_tokens[64:],
                "full_surface_tokens": full_tokens[0],
                "full_cscan_tokens": full_tokens[1],
                "native_shape": native_shape,
                "surface_render_sha256": surface_render.clean_sha256,
                "cscan_array_sha256": cscan_array_sha256,
                "feature_source": "REUSED_V2" if reusable else "ENCODED_V3",
            }
        )
        print(
            f"feature_bank {position:03d}/{len(candidates)} {key} "
            f"{'reuse' if reusable else 'encode'}",
            flush=True,
        )

    by_domain: dict[str, list[dict[str, object]]] = {}
    for record in records:
        identity = record["identity"]
        by_domain.setdefault(str(identity["dataset_id"]), []).append(record)
    index_rows: list[dict[str, object]] = []
    shard_rows: list[dict[str, object]] = []
    for domain, domain_records in sorted(by_domain.items()):
        shard_relative = Path(
            f"results/cai_agent_v3/new_protocol/feature_bank_{domain}.npz"
        )
        shard_path = root / shard_relative
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            shard_path,
            specimen_keys=np.asarray(
                [record["identity"]["specimen_key"] for record in domain_records]
            ),
            dataset_ids=np.asarray([domain] * len(domain_records)),
            specimen_ids=np.asarray(
                [record["identity"]["specimen_id"] for record in domain_records]
            ),
            capture_group_ids=np.asarray(
                [record["identity"]["capture_group_id"] for record in domain_records]
            ),
            splits=np.asarray(
                [record["identity"]["split"] for record in domain_records]
            ),
            targets_mpa=np.asarray(
                [
                    float(record["identity"]["author_cai_mpa"])
                    if record["identity"]["split"] != "TEST"
                    else np.nan
                    for record in domain_records
                ],
                dtype=np.float32,
            ),
            surface_tokens=np.asarray(
                [record["surface_tokens"] for record in domain_records],
                dtype=np.float32,
            ),
            cscan_tokens=np.asarray(
                [record["cscan_tokens"] for record in domain_records], dtype=np.float32
            ),
            full_surface_tokens=np.asarray(
                [record["full_surface_tokens"] for record in domain_records],
                dtype=np.float32,
            ),
            full_cscan_tokens=np.asarray(
                [record["full_cscan_tokens"] for record in domain_records],
                dtype=np.float32,
            ),
            native_shapes=np.asarray(
                [record["native_shape"] for record in domain_records], dtype=np.int64
            ),
        )
        shard_sha = sha256_file(shard_path)
        shard_rows.append(
            {
                "dataset_id": domain,
                "shard_path": shard_relative.as_posix(),
                "specimen_count": len(domain_records),
                "bytes": shard_path.stat().st_size,
                "sha256": shard_sha,
            }
        )
        for index, record in enumerate(domain_records):
            identity = record["identity"]
            index_rows.append(
                {
                    "specimen_key": identity["specimen_key"],
                    "dataset_id": domain,
                    "specimen_id": identity["specimen_id"],
                    "capture_group_id": identity["capture_group_id"],
                    "split": identity["split"],
                    "target_present_in_bank": identity["split"] != "TEST",
                    "feature_source": record["feature_source"],
                    "surface_source_sha256": identity["surface_sha256"],
                    "surface_render_sha256": record["surface_render_sha256"],
                    "cscan_file_sha256": identity["registered_cscan_crop_sha256"],
                    "cscan_array_sha256": record["cscan_array_sha256"],
                    "shard_path": shard_relative.as_posix(),
                    "shard_index": index,
                    "shard_sha256": shard_sha,
                }
            )
    write_csv(output / "feature_bank_index.csv", index_rows)
    write_csv(output / "feature_bank_shards.csv", shard_rows)
    scoring_rows = [
        {
            "specimen_key": row["specimen_key"],
            "dataset_id": row["dataset_id"],
            "split": row["split"],
            "author_cai_mpa": row["author_cai_mpa"],
            "label_source_sha256": row["label_source_sha256"],
            "scoring_only_for_test": row["split"] == "TEST",
        }
        for row in candidates
    ]
    write_csv(output / "scoring_labels.csv", scoring_rows)
    manifest = {
        "schema_version": 3,
        "status": "FEATURE_BANK_READY",
        "specimen_count": len(records),
        "cell_token_shape": [len(records), 64, _TOKEN_DIMENSION],
        "diagnostic_full_token_shape": [len(records), _TOKEN_DIMENSION],
        "surface_registration": "CLOCKWISE_90_ONCE_MAX_EDGE_1024",
        "cscan_registration": "P0R_REGISTERED_CSCAN_CROP_NO_ADDITIONAL_ROTATION",
        "v2_reused_specimen_count": reuse_count,
        "newly_encoded_specimen_count": encoded_count,
        "full_diagnostic_encoded_specimen_count": len(records),
        "encoder_execution_root": encoder_root.as_posix(),
        "encoder": encoder.provenance(),
        "elapsed_seconds": time.perf_counter() - started,
        "shards": shard_rows,
        "test_target_redacted_in_feature_bank": True,
        "diagnostic_full_tokens_for_actor": False,
    }
    write_json(output / "feature_bank_manifest.json", manifest)
    load_feature_bank(project_root=root)
    return manifest


__all__ = ["V3FeatureBank", "build_feature_bank", "load_feature_bank"]
