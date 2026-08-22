from __future__ import annotations

import numpy as np

from cmc_bbdm.msss.scale_evaluator import evaluate_axis
from cmc_bbdm.msss.scale_features import ScaleCondition, ScaleFeatureBank


def _bank() -> tuple[ScaleFeatureBank, np.ndarray, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(23))
    rows = 18
    conditions = tuple(
        ScaleCondition(
            condition_id=f"sampling:density={value}",
            axis="sampling",
            value=value,
            coarse_rank=rank,
            primary_eligible=True,
            is_full_identity=rank == 0,
        )
        for rank, value in enumerate((1.0, 0.5, 0.25, 0.125))
    )
    latent = np.linspace(-1.0, 1.0, rows)
    metadata = np.zeros((rows, 13), dtype=np.float64)
    metadata[:, 0] = latent
    targets = 0.65 + 0.12 * latent
    base = generator.normal(scale=0.1, size=(rows, 512))
    base[:, 0] = latent
    features = {
        item.condition_id: base + generator.normal(
            scale=0.001 * item.coarse_rank, size=base.shape
        )
        for item in conditions
    }
    bank = ScaleFeatureBank.issue(
        conditions=conditions,
        specimen_ids=tuple(f"s{index:02d}" for index in range(rows)),
        dataset_ids=("d1",) * 6 + ("d2",) * 6 + ("d3",) * 6,
        features=features,
        transform_state_sha256={item.condition_id: "0" * 64 for item in conditions},
        encoder_provenance={"encoder": "synthetic", "frozen": True},
    )
    return bank, targets, metadata


def test_outer_target_values_cannot_change_its_scale_selection() -> None:
    bank, targets, metadata = _bank()
    original = evaluate_axis(
        bank,
        targets=targets,
        metadata13=metadata,
        axis="sampling",
        pca_dimensions=(2, 3),
        primary_margin=0.05,
    )
    changed = targets.copy()
    changed[np.asarray(bank.dataset_ids) == "d3"] = 1.0e9
    modified = evaluate_axis(
        bank,
        targets=changed,
        metadata13=metadata,
        axis="sampling",
        pca_dimensions=(2, 3),
        primary_margin=0.05,
    )

    first = next(item for item in original.scale_selections if item.outer_group == "d3")
    second = next(item for item in modified.scale_selections if item.outer_group == "d3")
    assert first == second


def test_every_fit_event_excludes_outer_query_specimens() -> None:
    bank, targets, metadata = _bank()
    events = []
    evaluate_axis(
        bank,
        targets=targets,
        metadata13=metadata,
        axis="sampling",
        pca_dimensions=(2, 3),
        primary_margin=0.05,
        fit_hook=events.append,
    )

    assert events
    assert all(set(event.fit_ids).isdisjoint(event.query_ids) for event in events)
    assert all(event.outer_group not in event.fit_groups for event in events)
