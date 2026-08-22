from __future__ import annotations

import numpy as np

from cmc_bbdm.msss.scale_features import ScaleCondition, ScaleFeatureBank
from cmc_bbdm.msss.transfer_tasks import (
    TransferFitEvent,
    build_task_registry,
    evaluate_transfer_task,
)

GROUPS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _synthetic() -> tuple[ScaleFeatureBank, np.ndarray, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(73))
    rows = 54
    specimen_ids = tuple(f"s{index:03d}" for index in range(rows))
    dataset_ids = tuple(group for group in GROUPS for _ in range(9))
    conditions = tuple(
        ScaleCondition(
            condition_id=f"sampling:density={value}",
            axis="sampling",
            value=value,
            coarse_rank=rank,
            primary_eligible=True,
            is_full_identity=rank == 0,
        )
        for rank, value in enumerate((1.0, 0.25, 0.125))
    )
    base = generator.normal(size=(rows, 512))
    features = {
        condition.condition_id: base + rank * generator.normal(scale=0.01, size=base.shape)
        for rank, condition in enumerate(conditions)
    }
    metadata = generator.normal(size=(rows, 13))
    targets = 0.7 + 0.03 * metadata[:, 0] + 0.01 * base[:, 0]
    bank = ScaleFeatureBank.issue(
        conditions=conditions,
        specimen_ids=specimen_ids,
        dataset_ids=dataset_ids,
        features=features,
        transform_state_sha256={item.condition_id: "0" * 64 for item in conditions},
        encoder_provenance={"encoder": "synthetic", "frozen": True},
    )
    return bank, targets, metadata


def test_transfer_registry_has_six_domain_three_ply_and_two_layup_tasks() -> None:
    specimen_ids = tuple(f"s{index:03d}" for index in range(54))
    datasets = tuple(group for group in GROUPS for _ in range(9))
    ply = tuple((8, 16, 24)[index % 3] for index in range(54))
    layup = tuple(("cross_ply", "quasi_isotropic")[index % 2] for index in range(54))
    tasks = build_task_registry(
        specimen_ids=specimen_ids,
        dataset_ids=datasets,
        ply_count=ply,
        layup_family=layup,
        domain_order=GROUPS,
    )

    assert tuple(sum(item.family == family for item in tasks) for family in ("domain", "ply", "layup")) == (6, 3, 2)
    assert all(set(item.source_indices).isdisjoint(item.target_indices) for item in tasks)


def test_transfer_selection_never_fits_on_target_specimens() -> None:
    bank, targets, metadata = _synthetic()
    task = build_task_registry(
        specimen_ids=bank.specimen_ids,
        dataset_ids=bank.dataset_ids,
        ply_count=tuple((8, 16, 24)[index % 3] for index in range(54)),
        layup_family=tuple(("cross_ply", "quasi_isotropic")[index % 2] for index in range(54)),
        domain_order=GROUPS,
    )[0]
    events: list[TransferFitEvent] = []
    first = evaluate_transfer_task(
        bank,
        targets=targets,
        metadata13=metadata,
        task=task,
        pca_dimensions=(2, 4),
        fit_hook=events.append,
    )
    changed = targets.copy()
    changed[np.asarray(task.target_indices)] = 1.0e9
    second = evaluate_transfer_task(
        bank,
        targets=changed,
        metadata13=metadata,
        task=task,
        pca_dimensions=(2, 4),
    )

    assert first.selection == second.selection
    assert events
    assert all(set(event.fit_ids).isdisjoint(event.target_ids) for event in events)
