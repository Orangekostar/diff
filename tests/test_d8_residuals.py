from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cmc_bbdm.cpb_diffusion_marginalization.residuals import (
    P6ResidualBank,
    P6ResidualError,
    ResidualAuthority,
    ResidualFoldDraws,
    _decode_p6_normalized_posterior,
    build_residual_bank_from_arrays,
    validate_residual_bank,
)


def test_p6_uncertainty_authority_is_decoded_to_learning_target_scale() -> None:
    normalized_mean = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    normalized_variance = np.asarray(
        [0.0, 0.01, 0.02, 0.03, 0.04], dtype=np.float32
    )

    mean, variance = _decode_p6_normalized_posterior(
        normalized_mean, normalized_variance
    )

    np.testing.assert_array_equal(
        mean, np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        variance,
        np.asarray([0.0, 0.04, 0.08, 0.12, 0.16], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-8,
    )


def _fixture() -> tuple[ResidualAuthority, tuple[ResidualFoldDraws, ...]]:
    specimen_ids = ("s1", "s2", "s3", "s4")
    dataset_ids = (
        "74t7kcdgkr",
        "74t7kcdgkr",
        "cgtnjyggtm",
        "cgtnjyggtm",
    )
    targets = np.zeros((4, 3, 64, 64), dtype=np.float32)
    first = np.stack(
        [
            np.full((2, 3, 64, 64), value, dtype=np.float32)
            for value in (0.1, 0.2, 0.3)
        ],
        axis=1,
    )
    second = np.stack(
        [
            np.full((2, 3, 64, 64), value, dtype=np.float32)
            for value in (-0.1, -0.2, -0.3)
        ],
        axis=1,
    )
    draws = np.empty((4, 3, 3, 64, 64), dtype=np.float32)
    draws[:2] = first
    draws[2:] = second
    authority = ResidualAuthority(
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        measured_fields=targets,
        posterior_mean=np.mean(draws, axis=1, dtype=np.float64).astype(np.float32),
        posterior_variance=np.var(draws, axis=1, dtype=np.float64).astype(np.float32),
        source_sha256=("1" * 64, "2" * 64, "3" * 64, "4" * 64),
    )
    folds = (
        ResidualFoldDraws(
            heldout_domain="74t7kcdgkr",
            specimen_ids=("s1", "s2"),
            checkpoint_train_ids=("s3", "s4"),
            checkpoint_train_domains=("cgtnjyggtm", "cgtnjyggtm"),
            checkpoint_scientific_digest="a" * 64,
            draws=first,
        ),
        ResidualFoldDraws(
            heldout_domain="cgtnjyggtm",
            specimen_ids=("s3", "s4"),
            checkpoint_train_ids=("s1", "s2"),
            checkpoint_train_domains=("74t7kcdgkr", "74t7kcdgkr"),
            checkpoint_scientific_digest="b" * 64,
            draws=second,
        ),
    )
    return authority, folds


def test_residual_bank_is_cross_fitted_complete_and_matches_authority() -> None:
    authority, folds = _fixture()
    bank = build_residual_bank_from_arrays(authority, folds, draw_count=3)
    assert bank.specimen_count == 4
    assert bank.draw_count == 3
    assert len(bank.records) == 12
    assert bank.maximum_mean_error == 0.0
    assert bank.maximum_variance_error == 0.0
    assert {(row.specimen_id, row.draw_index) for row in bank.records} == {
        (specimen_id, draw)
        for specimen_id in authority.specimen_ids
        for draw in range(3)
    }
    for record in bank.records:
        assert record.dataset_id == record.heldout_domain
        assert record.dataset_id not in record.checkpoint_train_domains
        assert record.specimen_id not in record.checkpoint_train_ids
        assert record.residual_64.shape == (3, 64, 64)
        assert record.residual_64.dtype == np.float32
        assert record.residual_64.flags.writeable is False


def test_residual_bank_validation_binds_the_registered_source_roster() -> None:
    authority, folds = _fixture()
    bank = build_residual_bank_from_arrays(authority, folds, draw_count=3)
    assert (
        validate_residual_bank(
            bank,
            specimen_ids=authority.specimen_ids,
            dataset_ids=authority.dataset_ids,
            source_sha256=authority.source_sha256,
            draw_count=3,
        )
        == bank.state_sha256
    )
    changed_record = replace(bank.records[0], source_sha256="f" * 64)
    forged = P6ResidualBank(
        records=(changed_record, *bank.records[1:]),
        specimen_count=bank.specimen_count,
        draw_count=bank.draw_count,
        maximum_mean_error=bank.maximum_mean_error,
        maximum_variance_error=bank.maximum_variance_error,
        state_sha256=bank.state_sha256,
    )
    with pytest.raises(P6ResidualError, match="source roster"):
        validate_residual_bank(
            forged,
            specimen_ids=authority.specimen_ids,
            dataset_ids=authority.dataset_ids,
            source_sha256=authority.source_sha256,
            draw_count=3,
        )


def test_residual_bank_rejects_mean_variance_and_fold_leakage() -> None:
    authority, folds = _fixture()
    changed_mean = np.array(authority.posterior_mean, copy=True)
    changed_mean[0, 0, 0, 0] += np.float32(0.01)
    with pytest.raises(P6ResidualError, match="mean"):
        build_residual_bank_from_arrays(
            replace(authority, posterior_mean=changed_mean),
            folds,
            draw_count=3,
        )

    changed_variance = np.array(authority.posterior_variance, copy=True)
    changed_variance[0, 0, 0, 0] += np.float32(0.01)
    with pytest.raises(P6ResidualError, match="variance"):
        build_residual_bank_from_arrays(
            replace(authority, posterior_variance=changed_variance),
            folds,
            draw_count=3,
        )

    leaked = replace(
        folds[0],
        checkpoint_train_ids=("s1", "s3", "s4"),
        checkpoint_train_domains=(
            "74t7kcdgkr",
            "cgtnjyggtm",
            "cgtnjyggtm",
        ),
    )
    with pytest.raises(P6ResidualError, match="heldout"):
        build_residual_bank_from_arrays(
            authority,
            (leaked, folds[1]),
            draw_count=3,
        )


def test_residual_bank_rejects_missing_duplicate_and_nonfinite_draws() -> None:
    authority, folds = _fixture()
    with pytest.raises(P6ResidualError, match="roster"):
        build_residual_bank_from_arrays(authority, (folds[0],), draw_count=3)
    with pytest.raises(P6ResidualError, match="roster"):
        build_residual_bank_from_arrays(
            authority,
            (folds[0], folds[0], folds[1]),
            draw_count=3,
        )
    invalid = np.array(folds[0].draws, copy=True)
    invalid[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(P6ResidualError, match="finite"):
        build_residual_bank_from_arrays(
            authority,
            (replace(folds[0], draws=invalid), folds[1]),
            draw_count=3,
        )
