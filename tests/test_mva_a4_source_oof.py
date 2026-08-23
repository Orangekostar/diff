from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cmc_bbdm.mva.a4_candidate_bank import CandidateBank, _state_sha256
from cmc_bbdm.mva.a4_execution import (
    evaluate_outer_static_masks,
    fit_outer_evaluation_models,
    publish_outer_shard,
)
from cmc_bbdm.mva.a4_source_labels import generate_source_labels

DOMAINS = tuple(f"d{index}" for index in range(6))
METHODS = (
    "global_appearance_mask",
    "global_reconstruction_mask",
    "global_mechanical_mask",
)


def _fixture() -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    CandidateBank,
]:
    rng = np.random.default_rng(20260823)
    specimen_ids = tuple(f"{domain}-{row}" for domain in DOMAINS for row in range(9))
    dataset_ids = tuple(domain for domain in DOMAINS for _ in range(9))
    count = len(specimen_ids)
    metadata = rng.normal(size=(count, 3))
    initial_embeddings = rng.normal(size=(count, 512)).astype("<f8")
    embeddings = (
        initial_embeddings[:, None, :]
        + rng.normal(scale=0.1, size=(count, 64, 512))
    ).astype("<f8")
    targets = (
        0.65 + 0.04 * metadata[:, 0] + 0.03 * initial_embeddings[:, 0]
    )
    reconstruction_values = rng.normal(scale=0.01, size=(count, 64)).astype(
        "<f8"
    )
    appearance_values = rng.uniform(size=(count, 64)).astype("<f8")
    added_measurements = rng.integers(5, 20, size=(count, 64), dtype=np.int64)
    candidate_hashes = tuple(tuple("e" * 64 for _ in range(64)) for _ in range(count))
    bank = CandidateBank(
        schema_version=1,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        image_sha256=tuple("a" * 64 for _ in range(count)),
        decoded_image_sha256=tuple("b" * 64 for _ in range(count)),
        authority_state_sha256="c" * 64,
        initial_budget=0.015625,
        interpolation="bilinear",
        native_shapes=((33, 33),) * count,
        grid_state_sha256=tuple("d" * 64 for _ in range(count)),
        initial_measured_counts=(81,) * count,
        native_counts=(1089,) * count,
        cell_indices=tuple(range(64)),
        from_levels=(0,) * 64,
        to_levels=(1,) * 64,
        initial_output_sha256=tuple("f" * 64 for _ in range(count)),
        candidate_output_sha256=candidate_hashes,
        initial_embeddings=initial_embeddings,
        embeddings=embeddings,
        reconstruction_values=reconstruction_values,
        appearance_values=appearance_values,
        added_measurements=added_measurements,
        state_sha256="",
    )
    bank = replace(bank, state_sha256=_state_sha256(bank))
    return specimen_ids, dataset_ids, targets, metadata, bank


def test_source_label_predictor_excludes_outer_and_query_domains() -> None:
    specimen_ids, dataset_ids, targets, metadata, bank = _fixture()
    result = generate_source_labels(
        outer_domain="d0",
        domain_order=DOMAINS,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata=metadata,
        bank=bank,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
    )

    assert len(result.fit_audits) == 5
    for audit in result.fit_audits:
        assert "d0" not in audit.fit_domains
        assert audit.query_source_domain not in audit.fit_domains
        assert set(audit.query_domains).isdisjoint(audit.fit_domains)
        assert len(audit.fit_domains) == 4
    assert {row["dataset_id"] for row in result.rows} == set(DOMAINS) - {"d0"}
    assert all(row["specimen_id"] not in specimen_ids[:9] for row in result.rows)


def test_source_values_and_rankings_are_complete_and_bound() -> None:
    specimen_ids, dataset_ids, targets, metadata, bank = _fixture()
    result = generate_source_labels(
        outer_domain="d2",
        domain_order=DOMAINS,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata=metadata,
        bank=bank,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
    )

    source_count = sum(domain != "d2" for domain in dataset_ids)
    assert len(result.rows) == source_count * 64 * len(METHODS)
    assert tuple(ranking.method for ranking in result.rankings) == METHODS
    assert all(set(ranking.cell_order) == set(range(64)) for ranking in result.rankings)
    assert all(
        row["candidate_bank_state_sha256"] == bank.state_sha256
        for row in result.rows
    )
    mechanical = next(
        row for row in result.rows if row["method"] == "global_mechanical_mask"
    )
    assert mechanical["primary_value"] == pytest.approx(
        mechanical["absolute_error_before"] - mechanical["absolute_error_after"]
    )
    assert mechanical["predictor_state_sha256"]
    assert all(
        row["predictor_state_sha256"] is None
        for row in result.rows
        if row["method"] != "global_mechanical_mask"
    )


def test_target_cai_and_metadata_cannot_change_source_labels() -> None:
    specimen_ids, dataset_ids, targets, metadata, bank = _fixture()
    changed_targets = targets.copy()
    changed_metadata = metadata.copy()
    target = np.asarray(dataset_ids, dtype=object) == "d4"
    changed_targets[target] += 10_000.0
    changed_metadata[target] -= 20_000.0

    first = generate_source_labels(
        outer_domain="d4",
        domain_order=DOMAINS,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata=metadata,
        bank=bank,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
    )
    second = generate_source_labels(
        outer_domain="d4",
        domain_order=DOMAINS,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=changed_targets,
        metadata=changed_metadata,
        bank=bank,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
    )

    assert first.state_sha256 == second.state_sha256
    assert first.rows == second.rows
    assert first.rankings == second.rankings


class _FixtureEncoder:
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        return np.asarray(
            [np.resize(image.astype(np.float64).ravel(), 512) / 255.0 for image in images]
        )

    def validate(self) -> None:
        return None


def test_outer_shard_is_transactional_and_complete(tmp_path: Path) -> None:
    specimen_ids, dataset_ids, targets, metadata, bank = _fixture()
    outer = "d0"
    checkpoints = (0.0625, 0.125)
    labels = generate_source_labels(
        outer_domain=outer,
        domain_order=DOMAINS,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata=metadata,
        bank=bank,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
    )
    uniform = {
        checkpoint: bank.initial_embeddings + checkpoint * bank.embeddings[:, 0]
        for checkpoint in checkpoints
    }
    models = fit_outer_evaluation_models(
        outer_domain=outer,
        domain_order=DOMAINS,
        checkpoints=checkpoints,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        targets=targets,
        metadata=metadata,
        full_embeddings=bank.initial_embeddings,
        uniform_embeddings=uniform,
        pca_dimensions=(2, 4),
        ridge_alpha=10.0,
    )
    rows, columns = np.indices((41, 43))
    image = np.stack(
        (
            (rows * 7 + columns * 3) % 256,
            (rows * 2 + columns * 11) % 256,
            (rows * 13 + columns * 5) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    target_index = dataset_ids.index(outer)
    evaluation = evaluate_outer_static_masks(
        outer_domain=outer,
        domain_order=DOMAINS,
        specimen_ids=(specimen_ids[target_index],),
        dataset_ids=(outer,),
        images=(image,),
        targets=targets[target_index : target_index + 1],
        metadata=metadata[target_index : target_index + 1],
        initial_budget=bank.initial_budget,
        checkpoints=checkpoints,
        rankings=labels.rankings,
        source_specimen_ids=labels.source_specimen_ids,
        source_label_state_sha256=labels.state_sha256,
        p_a_model=models.p_a_model,
        p_b_models=models.p_b_models,
        encoder=_FixtureEncoder(),
    )

    output = publish_outer_shard(
        tmp_path / outer,
        source_labels=labels,
        evaluator_models=models,
        evaluation=evaluation,
    )

    assert {path.name for path in output.iterdir()} == {
        "complete.json",
        "fit_audits.csv",
        "ranking_stability.csv",
        "rankings.csv",
        "source_values.parquet",
        "states.parquet",
        "trajectories.parquet",
    }
    assert publish_outer_shard(
        output,
        source_labels=labels,
        evaluator_models=models,
        evaluation=evaluation,
    ) == output
