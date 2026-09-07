from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from cmc_bbdm.learned_cscan.contracts import (
    ReferenceEvidence,
    ReferenceStatus,
    ReviewState,
    Split,
    score_reference_evidence,
)
from cmc_bbdm.learned_cscan.runtime import (
    STUDY_SPLIT_SEED,
    hash_split,
    load_study_config,
    load_study_roster,
    render_registered_surface,
    require_fit_split,
)
from cmc_bbdm.vlm_cscan.runtime import InputSpecimen, render_surface_inputs


def _record(domain: str, index: int) -> InputSpecimen:
    digest = f"{index:064x}"
    return InputSpecimen(
        dataset_id=domain,
        specimen_id=f"sample-{index}",
        surface_path=Path(f"{domain}/sample-{index}.png"),
        surface_sha256=digest,
        cscan_path=Path(f"{domain}/sample-{index}-cscan.png"),
        cscan_sha256="f" * 64,
        native_shape=(16, 18),
        transform_sha256="e" * 64,
    )


def test_hash_split_is_balanced_deterministic_and_fit_is_train_only() -> None:
    records = tuple(
        _record(domain, index)
        for domain in ("d2", "d0", "d1")
        for index in range(10)
    )

    first = hash_split(records, seed=STUDY_SPLIT_SEED)
    second = hash_split(tuple(reversed(records)), seed=STUDY_SPLIT_SEED)

    assert [(row.record.specimen_key, row.split) for row in first] == [
        (row.record.specimen_key, row.split) for row in second
    ]
    assert Counter(row.split for row in first) == {
        Split.TRAIN: 12,
        Split.VALID: 6,
        Split.TEST: 12,
    }
    for domain in ("d0", "d1", "d2"):
        assert Counter(
            row.split for row in first if row.record.dataset_id == domain
        ) == {Split.TRAIN: 4, Split.VALID: 2, Split.TEST: 4}
    assert len({row.record.specimen_key for row in first}) == len(first)
    require_fit_split(Split.TRAIN)
    with pytest.raises(ValueError, match="TRAIN"):
        require_fit_split(Split.VALID)
    with pytest.raises(ValueError, match="TRAIN"):
        require_fit_split(Split.TEST)


def test_algorithm_proxy_never_becomes_formal_success() -> None:
    evidence = ReferenceEvidence(
        status=ReferenceStatus.ALGORITHM_DERIVED_NOT_REVIEWED,
        review_state=ReviewState.PENDING,
        reviewer_alias=None,
    )

    result = score_reference_evidence(evidence, proxy_success=True)

    assert result.reference_eligible is False
    assert result.proxy_success is True
    assert result.proxy_scope == "SAME_READER_SELF_CONSISTENCY"
    assert result.formal_success is None


def test_real_surface_is_nontrivial_and_rotated_exactly_once() -> None:
    source_root = Path("/home/ww/paper3/cmc_damage_inference")
    if not source_root.is_dir():
        pytest.skip("registered Hasebe data root is unavailable")
    project_root = Path(__file__).resolve().parents[1]
    config = load_study_config(
        project_root / "paper_v3/configs/learned_cscan_same_perception.yaml",
        project_root=project_root,
    )
    roster = load_study_roster(
        config,
        source_root=source_root,
        verify_pilot_hashes=False,
    )
    record = roster.assignments[0].record

    actual = render_registered_surface(record, max_edge=128)
    with Image.open(record.surface_path) as image:
        expected = render_surface_inputs(image, max_edge=128)

    assert min(actual.render.clean.size) > 1
    assert actual.clockwise_quarter_turns == 1
    assert actual.render.clean_sha256 == expected.clean_sha256
    assert actual.render.gridded_sha256 == expected.gridded_sha256
