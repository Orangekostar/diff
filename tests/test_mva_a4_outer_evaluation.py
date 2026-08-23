from __future__ import annotations

import hashlib

import numpy as np

from cmc_bbdm.mva.a4_execution import (
    evaluate_outer_static_masks,
    fit_outer_evaluation_models,
)
from cmc_bbdm.mva.crossfit import fit_outer_source_predictor
from cmc_bbdm.mva.global_mask import GlobalMaskRanking

DOMAINS = tuple(f"d{index}" for index in range(6))
METHODS = (
    "global_appearance_mask",
    "global_reconstruction_mask",
    "global_mechanical_mask",
)
CHECKPOINTS = (0.0625, 0.125)


class _FixtureEncoder:
    def __init__(self) -> None:
        self.validated = False

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        output = np.empty((len(images), 512), dtype=np.float64)
        for index, image in enumerate(images):
            output[index] = np.resize(image.astype(np.float64).ravel(), 512) / 255.0
        return output

    def validate(self) -> None:
        self.validated = True


class _FixturePredictor:
    def __init__(self, token: str) -> None:
        self.fit_domains = DOMAINS[1:]
        self.state_sha256 = hashlib.sha256(token.encode("ascii")).hexdigest()

    def predict(self, metadata: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        return 0.6 + 0.01 * metadata[:, 0] + 0.001 * embeddings[:, 0]


def _rankings() -> tuple[GlobalMaskRanking, ...]:
    output: list[GlobalMaskRanking] = []
    for offset, method in enumerate(METHODS):
        order = tuple(range(offset, 64)) + tuple(range(offset))
        scores = tuple(float(64 - order.index(cell)) / 64.0 for cell in range(64))
        output.append(
            GlobalMaskRanking(
                outer_domain="d0",
                method=method,
                cell_order=order,
                cell_scores=scores,
                mean_raw_values=(0.0,) * 64,
                mean_value_per_measurement=(0.0,) * 64,
                source_domains=DOMAINS[1:],
                source_specimen_count=5,
            )
        )
    return tuple(output)


def _image(offset: int) -> np.ndarray:
    rows, columns = np.indices((41, 43))
    return np.stack(
        (
            (rows * 7 + columns * 3 + offset) % 256,
            (rows * 2 + columns * 11 + offset) % 256,
            (rows * 13 + columns * 5 + offset) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def test_a4_outer_evaluation_uses_source_rankings_and_common_pb_heads() -> None:
    encoder = _FixtureEncoder()
    p_b_models = {
        checkpoint: _FixturePredictor(f"p-b-{checkpoint}")
        for checkpoint in CHECKPOINTS
    }
    result = evaluate_outer_static_masks(
        outer_domain="d0",
        domain_order=DOMAINS,
        specimen_ids=("target-0", "target-1"),
        dataset_ids=("d0", "d0"),
        images=(_image(0), _image(17)),
        targets=np.asarray([0.61, 0.72], dtype=np.float64),
        metadata=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        initial_budget=0.015625,
        checkpoints=CHECKPOINTS,
        rankings=_rankings(),
        source_specimen_ids=tuple(f"source-{index}" for index in range(5)),
        source_label_state_sha256="a" * 64,
        p_a_model=_FixturePredictor("p-a"),
        p_b_models=p_b_models,
        encoder=encoder,
    )

    assert encoder.validated
    assert len(result.states) == 2 * len(METHODS) * len(CHECKPOINTS)
    assert {row["dataset_id"] for row in result.states} == {"d0"}
    assert {row["method"] for row in result.states} == set(METHODS)
    assert all(row["effective_budget"] > 0.0 for row in result.states)
    assert all(np.isfinite(row["normalized_rgb_mse"]) for row in result.states)
    assert all(np.isfinite(row["ssim"]) for row in result.states)
    for checkpoint in CHECKPOINTS:
        hashes = {
            row["p_b_predictor_state_sha256"]
            for row in result.states
            if row["nominal_checkpoint"] == checkpoint
        }
        assert hashes == {p_b_models[checkpoint].state_sha256}


def test_a4_outer_trajectories_are_ranking_prefixes_only() -> None:
    result = evaluate_outer_static_masks(
        outer_domain="d0",
        domain_order=DOMAINS,
        specimen_ids=("target-0",),
        dataset_ids=("d0",),
        images=(_image(0),),
        targets=np.asarray([0.61], dtype=np.float64),
        metadata=np.asarray([[1.0, 2.0]], dtype=np.float64),
        initial_budget=0.015625,
        checkpoints=CHECKPOINTS,
        rankings=_rankings(),
        source_specimen_ids=tuple(f"source-{index}" for index in range(5)),
        source_label_state_sha256="a" * 64,
        p_a_model=_FixturePredictor("p-a"),
        p_b_models={
            checkpoint: _FixturePredictor(f"p-b-{checkpoint}")
            for checkpoint in CHECKPOINTS
        },
        encoder=_FixtureEncoder(),
    )

    for ranking in _rankings():
        cells = tuple(
            row["cell_index"]
            for row in result.trajectories
            if row["method"] == ranking.method
        )
        assert cells == ranking.cell_order[: len(cells)]
        assert all(
            (row["from_level"], row["to_level"]) == (0, 1)
            for row in result.trajectories
            if row["method"] == ranking.method
        )


def test_a4_evaluator_bundle_excludes_outer_and_binds_each_checkpoint() -> None:
    rng = np.random.default_rng(17)
    specimen_ids = tuple(f"{domain}-{row}" for domain in DOMAINS for row in range(9))
    dataset_ids = tuple(domain for domain in DOMAINS for _ in range(9))
    count = len(specimen_ids)
    metadata = rng.normal(size=(count, 3))
    full_embeddings = rng.normal(size=(count, 32))
    targets = 0.6 + 0.03 * metadata[:, 0] + 0.02 * full_embeddings[:, 0]
    uniform_embeddings = {
        checkpoint: full_embeddings + rng.normal(scale=checkpoint, size=(count, 32))
        for checkpoint in CHECKPOINTS
    }

    bundle = fit_outer_evaluation_models(
        outer_domain="d3",
        domain_order=DOMAINS,
        checkpoints=CHECKPOINTS,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata=metadata,
        full_embeddings=full_embeddings,
        uniform_embeddings=uniform_embeddings,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
    )

    assert "d3" not in bundle.p_a_model.fit_domains
    assert set(bundle.p_a_model.fit_domains) == set(DOMAINS) - {"d3"}
    assert set(bundle.p_b_models) == set(CHECKPOINTS)
    assert all("d3" not in model.fit_domains for model in bundle.p_b_models.values())
    assert {audit.evaluator for audit in bundle.fit_audits} == {
        "P-A",
        *(f"P-B:{checkpoint}" for checkpoint in CHECKPOINTS),
    }
    reference_p_a = fit_outer_source_predictor(
        method="MVA_P_A",
        outer_domain="d3",
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        domain_order=DOMAINS,
        targets=targets,
        metadata=metadata,
        embeddings=full_embeddings,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
        tie_tolerance=1.0e-12,
    ).model
    assert bundle.p_a_model.state_sha256 == reference_p_a.state_sha256
    for checkpoint in CHECKPOINTS:
        reference_p_b = fit_outer_source_predictor(
            method=f"MVA_P_B_{checkpoint}",
            outer_domain="d3",
            specimen_ids=specimen_ids,
            dataset_ids=dataset_ids,
            domain_order=DOMAINS,
            targets=targets,
            metadata=metadata,
            embeddings=uniform_embeddings[checkpoint],
            pca_dimensions=(2, 4),
            ridge_alpha=10.0,
            tie_tolerance=1.0e-12,
        ).model
        assert bundle.p_b_models[checkpoint].state_sha256 == reference_p_b.state_sha256
