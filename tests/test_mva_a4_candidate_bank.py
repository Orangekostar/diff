from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.mva.a4_candidate_bank import (
    CandidateBankError,
    build_candidate_bank,
    load_candidate_bank,
    save_candidate_bank,
)
from cmc_bbdm.mva.acquisition_grid import build_acquisition_grid
from cmc_bbdm.mva.interpolation import (
    reconstruct_measurement_state,
    refine_reconstruction,
)
from cmc_bbdm.mva.measurement_state import (
    RefinementAction,
    apply_action,
    initial_state,
    measurement_mask,
)
from cmc_bbdm.mva.reconstruction_value import reconstruction_value


class _FixtureEncoder:
    def __init__(self) -> None:
        self.validated = False

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        output = np.empty((len(images), 512), dtype=np.float64)
        for index, image in enumerate(images):
            flat = image.astype(np.float64).reshape(-1, 3)
            channel_means = np.mean(flat, axis=0)
            output[index] = np.resize(channel_means, 512)
        return output

    def validate(self) -> None:
        self.validated = True


def _images() -> tuple[np.ndarray, ...]:
    rows, columns = np.indices((41, 43))
    first = np.stack(
        (
            (rows * 7 + columns * 3) % 256,
            (rows * 2 + columns * 11) % 256,
            (rows * 13 + columns * 5) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    second = np.flip(first, axis=1).copy()
    return first, second


def _build_bank() -> tuple[object, _FixtureEncoder]:
    encoder = _FixtureEncoder()
    bank = build_candidate_bank(
        specimen_ids=("s0", "s1"),
        dataset_ids=("d0", "d1"),
        images=_images(),
        image_sha256=("a" * 64, "b" * 64),
        authority_state_sha256="c" * 64,
        initial_budget=0.015625,
        encoder=encoder,
    )
    return bank, encoder


def test_candidate_bank_contains_exact_initial_actions() -> None:
    bank, encoder = _build_bank()

    assert bank.specimen_ids == ("s0", "s1")
    assert bank.dataset_ids == ("d0", "d1")
    assert bank.embeddings.shape == (2, 64, 512)
    assert bank.initial_embeddings.shape == (2, 512)
    assert bank.reconstruction_values.shape == (2, 64)
    assert bank.appearance_values.shape == (2, 64)
    assert bank.added_measurements.shape == (2, 64)
    assert bank.from_levels == (0,) * 64
    assert bank.to_levels == (1,) * 64
    assert bank.authority_state_sha256 == "c" * 64
    assert all(len(value) == 64 for value in bank.decoded_image_sha256)
    assert encoder.validated
    assert not bank.embeddings.flags.writeable
    assert np.all(np.isfinite(bank.embeddings))
    assert np.all(bank.added_measurements > 0)


def test_candidate_hash_and_value_match_exact_reconstruction() -> None:
    bank, _encoder = _build_bank()
    image = _images()[0]
    grid = build_acquisition_grid(41, 43, initial_budget=0.015625)
    state = initial_state(grid)
    current = reconstruct_measurement_state(
        image,
        grid,
        state,
        interpolation="bilinear",
        specimen_id="s0",
        dataset_id="d0",
    ).image
    action = RefinementAction(cell_index=17, from_level=0, to_level=1)
    candidate = refine_reconstruction(
        image,
        grid,
        state,
        current,
        action,
        interpolation="bilinear",
    )
    candidate_state = apply_action(grid, state, action)
    candidate_mask = measurement_mask(grid, candidate_state)

    assert np.array_equal(candidate[candidate_mask], image[candidate_mask])
    assert bank.candidate_output_sha256[0][17] == hashlib.sha256(
        candidate.tobytes(order="C")
    ).hexdigest()
    assert bank.reconstruction_values[0, 17] == pytest.approx(
        reconstruction_value(image, current, candidate)
    )


def test_candidate_bank_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    bank, _encoder = _build_bank()
    path = tmp_path / "bank.npz"
    save_candidate_bank(path, bank)

    loaded = load_candidate_bank(path)
    assert loaded.state_sha256 == bank.state_sha256
    assert np.array_equal(loaded.embeddings, bank.embeddings)

    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["embeddings"][0, 0, 0] += 1.0
    np.savez_compressed(path, **payload)
    with pytest.raises(CandidateBankError, match="digest"):
        load_candidate_bank(path)


def test_candidate_bank_rejects_invalid_source_hash() -> None:
    with pytest.raises(CandidateBankError, match="image hash"):
        build_candidate_bank(
            specimen_ids=("s0",),
            dataset_ids=("d0",),
            images=(_images()[0],),
            image_sha256=("invalid",),
            authority_state_sha256="c" * 64,
            initial_budget=0.015625,
            encoder=_FixtureEncoder(),
        )
