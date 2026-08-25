from __future__ import annotations

import numpy as np
import torch

from cmc_bbdm.mavis.dynamic_data import DynamicStateGroup
from cmc_bbdm.mavis.dynamic_training import (
    fit_inner_dynamic_voi,
    load_fitted_dynamic_checkpoint,
    save_fitted_dynamic_checkpoint,
)
from cmc_bbdm.mavis.dynamic_voi import CandidateDescriptor


def _groups() -> tuple[tuple[DynamicStateGroup, ...], np.ndarray]:
    candidates = (
        CandidateDescriptor(
            cell_index=1,
            from_level=0,
            to_level=1,
            exact_added_cost=5,
            native_count=100,
            remaining_cost=20,
        ),
        CandidateDescriptor(
            cell_index=8,
            from_level=1,
            to_level=2,
            exact_added_cost=10,
            native_count=100,
            remaining_cost=20,
        ),
    )
    groups: list[DynamicStateGroup] = []
    embeddings: list[np.ndarray] = []
    for domain_index, domain in enumerate(("d1", "d2", "d3", "d4", "d5")):
        for specimen_index in range(2):
            teacher = np.asarray([0.2, -0.1], dtype=np.float64)
            teacher.setflags(write=False)
            predictions = np.asarray([0.35, 0.65], dtype=np.float64)
            predictions.setflags(write=False)
            groups.append(
                DynamicStateGroup(
                    state_id=f"state-{domain}-{specimen_index}",
                    specimen_id=f"specimen-{domain}-{specimen_index}",
                    domain_id=domain,
                    outer_domain="d0",
                    candidates=candidates,
                    true_cai=0.4,
                    current_prediction=0.6,
                    candidate_predictions=predictions,
                    teacher_values=teacher,
                    teacher_outer_domains=("d0",),
                    teacher_fold_count=1,
                    state_sha256=f"hash-{domain}-{specimen_index}",
                )
            )
            embeddings.append(
                np.asarray(
                    [domain_index / 5.0, specimen_index, 0.25, 0.5],
                    dtype=np.float64,
                )
            )
    return tuple(groups), np.stack(embeddings)


def test_dynamic_training_excludes_outer_and_inner_validation_domains(
    tmp_path,
) -> None:
    groups, embeddings = _groups()
    fitted = fit_inner_dynamic_voi(
        groups,
        embeddings,
        validation_domain="d1",
        hidden_dimension=8,
        learning_rate=0.01,
        max_epochs=4,
        patience=2,
        batch_size=4,
        seed=20260825,
        device="cpu",
        loss_weights={"cai": 1.0, "pair": 1.0, "list": 1.0, "value": 0.25},
    )

    assert fitted.audit.outer_domain == "d0"
    assert fitted.audit.validation_domain == "d1"
    assert set(fitted.audit.fit_domains) == {"d2", "d3", "d4", "d5"}
    assert not set(fitted.audit.fit_specimen_ids) & set(
        fitted.audit.validation_specimen_ids
    )
    result = fitted.score_actions(
        torch.tensor(embeddings[0], dtype=torch.float32),
        groups[0].candidates,
    )
    assert result.scores.shape == (2,)
    assert torch.isfinite(result.scores).all()

    checkpoint = tmp_path / "dynamic-d0.npz"
    save_fitted_dynamic_checkpoint(fitted, checkpoint)
    restored = load_fitted_dynamic_checkpoint(
        checkpoint,
        expected_model_state_sha256=fitted.model_state_sha256,
    )
    restored_result = restored.score_actions(
        torch.tensor(embeddings[0], dtype=torch.float32),
        groups[0].candidates,
    )
    torch.testing.assert_close(restored_result.scores, result.scores)
    torch.testing.assert_close(
        restored_result.value_predictions,
        result.value_predictions,
    )
