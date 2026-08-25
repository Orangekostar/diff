from __future__ import annotations

import numpy as np

from cmc_bbdm.mavis.teacher import (
    fit_strict_oof_teacher,
    label_teacher_candidates,
    predict_teacher_state,
)
from cmc_bbdm.mva.measurement_state import RefinementAction

DOMAINS = ("d0", "d1", "d2", "d3", "d4", "d5")


def _teacher_arrays() -> dict[str, object]:
    generator = np.random.Generator(np.random.PCG64(20260825))
    specimen_ids = tuple(f"{domain}-s{index}" for domain in DOMAINS for index in range(3))
    dataset_ids = tuple(domain for domain in DOMAINS for _ in range(3))
    embeddings = generator.normal(size=(18, 8))
    metadata = generator.normal(size=(18, 4))
    targets = (
        0.3 * metadata[:, 0]
        - 0.2 * metadata[:, 1]
        + 0.1 * embeddings[:, 0]
        + generator.normal(scale=0.01, size=18)
    )
    return {
        "specimen_ids": specimen_ids,
        "dataset_ids": dataset_ids,
        "targets": targets,
        "metadata": metadata,
        "initial_embeddings": embeddings,
    }


def test_mavis_teacher_value_is_strict_oof() -> None:
    arrays = _teacher_arrays()
    teacher = fit_strict_oof_teacher(
        outer_domain="d0",
        query_domain="d1",
        domain_order=DOMAINS,
        pca_dimensions=(1, 2),
        ridge_alpha=10.0,
        **arrays,
    )

    audit = teacher.audit
    assert audit.held_out_target_domain == "d0"
    assert audit.query_source_domain == "d1"
    assert set(audit.fit_domains) == {"d2", "d3", "d4", "d5"}
    assert set(audit.query_domains) == {"d1"}
    assert not set(audit.fit_specimen_ids) & set(audit.query_specimen_ids)
    assert all(not specimen.startswith(("d0-", "d1-")) for specimen in audit.fit_specimen_ids)
    assert teacher.model.fit_domains == audit.fit_domains


def test_mavis_outer_targets_cannot_change_source_teacher() -> None:
    first = _teacher_arrays()
    second = _teacher_arrays()
    outer = np.asarray(first["dataset_ids"], dtype=object) == "d0"
    second["targets"][outer] += 1000.0

    first_teacher = fit_strict_oof_teacher(
        outer_domain="d0",
        query_domain="d1",
        domain_order=DOMAINS,
        pca_dimensions=(1, 2),
        ridge_alpha=10.0,
        **first,
    )
    second_teacher = fit_strict_oof_teacher(
        outer_domain="d0",
        query_domain="d1",
        domain_order=DOMAINS,
        pca_dimensions=(1, 2),
        ridge_alpha=10.0,
        **second,
    )

    assert first_teacher.state_sha256 == second_teacher.state_sha256
    assert first_teacher.model.state_sha256 == second_teacher.model.state_sha256


def test_mavis_dynamic_teacher_value_is_current_state_conditional() -> None:
    arrays = _teacher_arrays()
    teacher = fit_strict_oof_teacher(
        outer_domain="d0",
        query_domain="d1",
        domain_order=DOMAINS,
        pca_dimensions=(1, 2),
        ridge_alpha=10.0,
        **arrays,
    )
    query_index = arrays["dataset_ids"].index("d1")
    metadata = arrays["metadata"][query_index]
    current = arrays["initial_embeddings"][query_index]
    candidates = np.vstack((current + 0.1, current - 0.2))
    actions = (RefinementAction(0, 0, 1), RefinementAction(1, 0, 1))

    labels = label_teacher_candidates(
        teacher,
        specimen_id=arrays["specimen_ids"][query_index],
        dataset_id="d1",
        true_cai=float(arrays["targets"][query_index]),
        metadata=metadata,
        current_embedding=current,
        candidate_embeddings=candidates,
        actions=actions,
        candidate_costs=(11, 17),
    )

    assert tuple(label.action for label in labels) == actions
    assert tuple(label.exact_added_cost for label in labels) == (11, 17)
    for label in labels:
        assert label.primary_value == label.error_before - label.error_after
        assert label.secondary_value == (
            (label.true_cai - label.current_prediction) ** 2
            - (label.true_cai - label.candidate_prediction) ** 2
        )
    assert predict_teacher_state(
        teacher,
        specimen_id=arrays["specimen_ids"][query_index],
        dataset_id="d1",
        metadata=metadata,
        current_embedding=current,
    ) == labels[0].current_prediction


def test_mavis_terminal_state_prediction_does_not_require_candidates() -> None:
    arrays = _teacher_arrays()
    teacher = fit_strict_oof_teacher(
        outer_domain="d0",
        query_domain="d1",
        domain_order=DOMAINS,
        pca_dimensions=(1, 2),
        ridge_alpha=10.0,
        **arrays,
    )
    query_index = arrays["dataset_ids"].index("d1")

    prediction = predict_teacher_state(
        teacher,
        specimen_id=arrays["specimen_ids"][query_index],
        dataset_id="d1",
        metadata=arrays["metadata"][query_index],
        current_embedding=arrays["initial_embeddings"][query_index],
    )
    labels = label_teacher_candidates(
        teacher,
        specimen_id=arrays["specimen_ids"][query_index],
        dataset_id="d1",
        true_cai=float(arrays["targets"][query_index]),
        metadata=arrays["metadata"][query_index],
        current_embedding=arrays["initial_embeddings"][query_index],
        candidate_embeddings=np.empty((0, 8), dtype=np.float64),
        actions=(),
        candidate_costs=(),
    )

    assert np.isfinite(prediction)
    assert labels == ()
