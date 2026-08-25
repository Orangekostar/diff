"""Hash-bound privileged authority with typed deployable views."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .contracts import EvaluationView, PolicyContext, SourceTeacherView


class MAVISAuthorityError(ValueError):
    """Raised when the MAVIS specimen authority is invalid or unavailable."""


def _snapshot(value: object, *, dtype: object, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.ascontiguousarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise MAVISAuthorityError("authority array cannot be snapshotted") from error
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise MAVISAuthorityError("authority array shape or values are invalid")
    output = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(shape)
    output.setflags(write=False)
    return output


def _snapshot_image(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 3 or array.shape[2] != 3:
        raise MAVISAuthorityError("authority C-scan must be RGB uint8")
    output = np.frombuffer(
        np.ascontiguousarray(array).tobytes(order="C"), dtype=np.uint8
    ).reshape(array.shape)
    output.setflags(write=False)
    return output


def _array_hash(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _PrivilegedSpecimen:
    policy_context: PolicyContext
    dataset_id: str
    full_scan: np.ndarray
    true_cai: float
    source_image_sha256: str
    decoded_image_sha256: str


@dataclass(frozen=True, slots=True)
class MAVISAuthority:
    specimen_ids: tuple[str, ...]
    dataset_ids: tuple[str, ...]
    source_image_sha256: tuple[str, ...]
    decoded_image_sha256: tuple[str, ...]
    source_authority_sha256: str
    state_sha256: str
    _specimens: tuple[_PrivilegedSpecimen, ...]
    _index: MappingProxyType

    @classmethod
    def from_arrays(
        cls,
        *,
        specimen_ids: tuple[str, ...],
        dataset_ids: tuple[str, ...],
        images: tuple[np.ndarray, ...],
        targets: object,
        metadata13: object,
        profile_stats21: object,
        source_image_sha256: tuple[str, ...] | None = None,
        source_authority_sha256: str | None = None,
    ) -> MAVISAuthority:
        if (
            type(specimen_ids) is not tuple
            or type(dataset_ids) is not tuple
            or type(images) is not tuple
            or not specimen_ids
            or len(specimen_ids) != len(dataset_ids) != len(images)
        ):
            raise MAVISAuthorityError("authority roster is invalid")
        count = len(specimen_ids)
        if (
            len(dataset_ids) != count
            or len(images) != count
            or len(set(specimen_ids)) != count
            or any(type(value) is not str or not value for value in specimen_ids)
            or any(type(value) is not str or not value for value in dataset_ids)
        ):
            raise MAVISAuthorityError("authority roster is invalid")
        target_array = _snapshot(targets, dtype="<f8", shape=(count,))
        metadata = _snapshot(metadata13, dtype="<f8", shape=(count, 13))
        profiles = _snapshot(profile_stats21, dtype="<f8", shape=(count, 21))
        if source_image_sha256 is not None and (
            type(source_image_sha256) is not tuple
            or len(source_image_sha256) != count
            or any(not _is_sha256(value) for value in source_image_sha256)
        ):
            raise MAVISAuthorityError("source image hashes are invalid")
        if source_authority_sha256 is not None and not _is_sha256(
            source_authority_sha256
        ):
            raise MAVISAuthorityError("source authority hash is invalid")
        records: list[_PrivilegedSpecimen] = []
        state_rows: list[object] = []
        for index, (specimen_id, dataset_id, raw_image) in enumerate(
            zip(specimen_ids, dataset_ids, images, strict=True)
        ):
            image = _snapshot_image(raw_image)
            context = PolicyContext(
                specimen_id=specimen_id,
                context_features=np.concatenate((metadata[index], profiles[index])),
                native_shape=(int(image.shape[0]), int(image.shape[1])),
                native_count=int(image.shape[0] * image.shape[1]),
            )
            decoded_hash = _array_hash(image)
            source_hash = (
                source_image_sha256[index]
                if source_image_sha256 is not None
                else hashlib.sha256(image.tobytes(order="C")).hexdigest()
            )
            target = float(target_array[index])
            if not math.isfinite(target):
                raise MAVISAuthorityError("authority target is invalid")
            records.append(
                _PrivilegedSpecimen(
                    policy_context=context,
                    dataset_id=dataset_id,
                    full_scan=image,
                    true_cai=target,
                    source_image_sha256=source_hash,
                    decoded_image_sha256=decoded_hash,
                )
            )
            state_rows.append(
                (
                    specimen_id,
                    dataset_id,
                    context.state_sha256,
                    source_hash,
                    decoded_hash,
                    target,
                )
            )
        source_state = source_authority_sha256 or hashlib.sha256(
            json.dumps(
                {"schema": 1, "source_specimens": state_rows},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        state = hashlib.sha256(
            json.dumps(
                {
                    "schema": 1,
                    "source_authority_sha256": source_state,
                    "specimens": state_rows,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            specimen_ids=specimen_ids,
            dataset_ids=dataset_ids,
            source_image_sha256=tuple(
                record.source_image_sha256 for record in records
            ),
            decoded_image_sha256=tuple(
                record.decoded_image_sha256 for record in records
            ),
            source_authority_sha256=source_state,
            state_sha256=state,
            _specimens=tuple(records),
            _index=MappingProxyType(
                {specimen_id: index for index, specimen_id in enumerate(specimen_ids)}
            ),
        )

    @property
    def specimen_count(self) -> int:
        return len(self.specimen_ids)

    def _record(self, specimen_id: str) -> _PrivilegedSpecimen:
        try:
            index = self._index[specimen_id]
        except (KeyError, TypeError) as error:
            raise MAVISAuthorityError("specimen is not in the authority") from error
        return self._specimens[index]

    def policy_context(self, specimen_id: str) -> PolicyContext:
        return self._record(specimen_id).policy_context

    def source_teacher_view(self, specimen_id: str) -> SourceTeacherView:
        record = self._record(specimen_id)
        return SourceTeacherView(
            specimen_id=specimen_id,
            dataset_id=record.dataset_id,
            policy_context=record.policy_context,
            full_scan=record.full_scan,
            true_cai=record.true_cai,
            source_image_sha256=record.source_image_sha256,
        )

    def evaluation_view(self, specimen_id: str) -> EvaluationView:
        record = self._record(specimen_id)
        return EvaluationView(
            specimen_id=specimen_id,
            dataset_id=record.dataset_id,
            full_scan=record.full_scan,
            true_cai=record.true_cai,
            source_image_sha256=record.source_image_sha256,
        )

    def _reveal_values(self, specimen_id: str, positions: np.ndarray) -> np.ndarray:
        record = self._record(specimen_id)
        points = np.asarray(positions)
        if (
            points.dtype.kind not in "iu"
            or points.ndim != 2
            or points.shape[1] != 2
            or np.any(points < 0)
            or np.any(points[:, 0] >= record.full_scan.shape[0])
            or np.any(points[:, 1] >= record.full_scan.shape[1])
        ):
            raise MAVISAuthorityError("reveal positions are invalid")
        values = record.full_scan[points[:, 0], points[:, 1]]
        return _snapshot(values, dtype=np.uint8, shape=(len(points), 3))


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


_CACHE: dict[tuple[str, str], MAVISAuthority] = {}


def load_mavis_authority(
    config: object, *, source_project_root: str | Path
) -> MAVISAuthority:
    from cmc_bbdm.mgmr.authority import load_authority
    from cmc_bbdm.mgmr.protocol import load_protocol

    from .config import MAVISConfig

    if type(config) is not MAVISConfig:
        raise MAVISAuthorityError("issued MAVIS config is required")
    root = Path(source_project_root).resolve(strict=True)
    key = (config.config_sha256, str(root))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    binding = config.sources["mgmr_config"]
    protocol_path = root / binding.path
    try:
        protocol_bytes = protocol_path.read_bytes()
    except OSError as error:
        raise MAVISAuthorityError("registered MGMR config is unavailable") from error
    if hashlib.sha256(protocol_bytes).hexdigest() != binding.sha256:
        raise MAVISAuthorityError("registered MGMR config hash changed")
    protocol = load_protocol(protocol_path, project_root=root)
    upstream = load_authority(protocol, project_root=root)
    if (
        upstream.state_sha256 != config.source_authority_sha256
        or upstream.specimen_count != config.specimen_count
        or tuple(dict.fromkeys(upstream.dataset_ids)) != config.domain_order
    ):
        raise MAVISAuthorityError("registered upstream authority changed")
    result = MAVISAuthority.from_arrays(
        specimen_ids=upstream.specimen_ids,
        dataset_ids=upstream.dataset_ids,
        images=upstream.images,
        targets=upstream.targets,
        metadata13=upstream.metadata13,
        profile_stats21=upstream.data.profile_stats21,
        source_image_sha256=upstream.image_sha256,
        source_authority_sha256=upstream.state_sha256,
    )
    _CACHE[key] = result
    return result


__all__ = [
    "MAVISAuthority",
    "MAVISAuthorityError",
    "load_mavis_authority",
]
