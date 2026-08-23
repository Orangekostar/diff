from __future__ import annotations

from pathlib import Path

import numpy as np

from cmc_bbdm.mva.encoder_session import MVAEncoderSession
from cmc_bbdm.mva.pipeline import _encoder

ROOT = Path(__file__).resolve().parents[1]


def test_mva_encoder_session_matches_frozen_public_encoder() -> None:
    generator = np.random.Generator(np.random.PCG64(20260823))
    images = [
        generator.integers(0, 256, size=(37, 41, 3), dtype=np.uint8),
        generator.integers(0, 256, size=(43, 39, 3), dtype=np.uint8),
    ]
    encoder = _encoder(ROOT, "cpu")
    expected = encoder.encode(images)

    session = MVAEncoderSession(encoder)
    actual = session.encode(images)
    session.validate()

    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)
