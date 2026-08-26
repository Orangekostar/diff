from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.mavis.authority import MAVISAuthority


def synthetic_inputs(
    *,
    true_cai: float = 0.4,
    native_shape: tuple[int, int] = (41, 43),
) -> dict[str, object]:
    generator = np.random.Generator(np.random.PCG64(20260825))
    image = generator.integers(
        0,
        256,
        size=(*native_shape, 3),
        dtype=np.uint8,
    )
    return {
        "specimen_ids": ("sample-001",),
        "dataset_ids": ("domain-a",),
        "images": (image,),
        "targets": np.asarray([true_cai], dtype=np.float64),
        "metadata13": np.linspace(0.0, 1.2, 13, dtype=np.float64)[None, :],
        "profile_stats21": np.linspace(-1.0, 1.0, 21, dtype=np.float64)[None, :],
    }


def synthetic_authority(
    *,
    true_cai: float = 0.4,
    native_shape: tuple[int, int] = (41, 43),
) -> MAVISAuthority:
    return MAVISAuthority.from_arrays(
        **synthetic_inputs(true_cai=true_cai, native_shape=native_shape)
    )


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
