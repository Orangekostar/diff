from __future__ import annotations

import numpy as np
import pytest

from cmc_bbdm.vlm_cscan.contracts import (
    BenchmarkTask,
    CScanReference,
    EvaluationMode,
    ReferenceType,
    ReviewState,
    TaskReport,
)
from cmc_bbdm.vlm_cscan.metrics import (
    CurveSnapshot,
    paired_domain_bootstrap,
    success_curve,
)
from cmc_bbdm.vlm_cscan.references import (
    derive_proxy_reference,
    evaluate_task_report,
    reference_from_payload,
)
from cmc_bbdm.vlm_cscan.reporting import PublicStopTracker


def _reference(
    certain: np.ndarray,
    uncertain: np.ndarray | None = None,
    *,
    reference_type: ReferenceType = ReferenceType.ALGORITHM_DERIVED_NOT_REVIEWED,
    review_state: ReviewState = ReviewState.PENDING,
) -> CScanReference:
    return CScanReference(
        specimen_key="domain:sample",
        source_image_sha256="a" * 64,
        reference_type=reference_type,
        review_state=review_state,
        reviewer_alias="R1" if review_state is ReviewState.REVIEWED else None,
        certain_mask=certain,
        uncertain_mask=(
            np.zeros_like(certain, dtype=np.bool_)
            if uncertain is None
            else uncertain
        ),
    )


def test_unreviewed_proxy_never_becomes_formal_success() -> None:
    """Catches promoting an algorithm-derived reference into damage-task evidence."""

    certain = np.zeros((8, 8), dtype=np.bool_)
    certain[2:4, 3:5] = True
    report = TaskReport(
        task=BenchmarkTask.LOCATE,
        predicted_mask=certain,
        support_positions=np.asarray([[2, 3], [2, 4], [3, 3]], dtype=np.int64),
        confidence=0.9,
        public_complete=True,
        reason_code="LOCATE_STABLE",
    )

    score = evaluate_task_report(report, _reference(certain))

    assert score.proxy_success is True
    assert score.formal_success is None
    assert score.reference_eligible is False


def test_locate_rejects_full_frame_even_with_perfect_reference_overlap() -> None:
    """Catches satisfying LOCATE by returning the whole inspection frame."""

    certain = np.ones((8, 8), dtype=np.bool_)
    report = TaskReport(
        task=BenchmarkTask.LOCATE,
        predicted_mask=np.ones((8, 8), dtype=np.bool_),
        support_positions=np.asarray([[1, 1], [1, 2], [2, 1]], dtype=np.int64),
        confidence=1.0,
        public_complete=True,
        reason_code="FULL_FRAME",
    )
    reference = _reference(
        certain,
        reference_type=ReferenceType.EXPERT_REVIEWED,
        review_state=ReviewState.REVIEWED,
    )

    score = evaluate_task_report(report, reference)

    assert score.iou == 1.0
    assert score.formal_success is False
    assert "FULL_FRAME_PREDICTION" in score.failure_types


def test_characterize_excludes_uncertainty_and_uses_area_interval() -> None:
    """Catches scoring an uncertain reference band as known background."""

    certain = np.zeros((5, 5), dtype=np.bool_)
    certain[2:4, 2:4] = True
    uncertain = np.zeros_like(certain)
    uncertain[1:4, 1:4] = True
    uncertain[certain] = False
    prediction = certain | uncertain
    report = TaskReport(
        task=BenchmarkTask.CHARACTERIZE,
        predicted_mask=prediction,
        support_positions=np.asarray([[2, 2]], dtype=np.int64),
        confidence=0.8,
        public_complete=True,
        reason_code="AREA_STABLE",
    )
    reference = _reference(
        certain,
        uncertain,
        reference_type=ReferenceType.EXPERT_REVIEWED,
        review_state=ReviewState.REVIEWED,
    )

    score = evaluate_task_report(report, reference)

    assert score.iou == 1.0
    assert score.recall == 1.0
    assert score.relative_area_error == 0.0
    assert score.formal_success is True


def test_anytime_curve_uses_latest_report_not_hindsight_max() -> None:
    """Catches replacing the latest report with any earlier lucky success."""

    snapshots = (
        CurveSnapshot(cost=0.1, success=True, stopped=False),
        CurveSnapshot(cost=0.3, success=False, stopped=False),
        CurveSnapshot(cost=0.5, success=True, stopped=True),
    )

    anytime = success_curve(
        snapshots,
        mode=EvaluationMode.ANYTIME_REPORT,
        checkpoints=(0.2, 0.4, 0.6),
    )
    autonomous = success_curve(
        snapshots,
        mode=EvaluationMode.AUTONOMOUS_REPORT,
        checkpoints=(0.2, 0.4, 0.6),
    )

    assert anytime == (1.0, 0.0, 1.0)
    assert autonomous == (0.0, 0.0, 1.0)


def test_paired_bootstrap_weights_domains_equally() -> None:
    """Catches pooling specimens so a large domain dominates the comparison."""

    result = paired_domain_bootstrap(
        {
            "large": np.ones(9, dtype=np.float64),
            "small": -np.ones(1, dtype=np.float64),
        },
        replicates=50,
        seed=7,
    )

    assert result.estimate == 0.0
    assert result.ci_lower == 0.0
    assert result.ci_upper == 0.0


def test_polygon_import_requires_real_human_review_provenance() -> None:
    """Catches relabeling an automatic proposal as a reviewed reference."""

    payload = {
        "specimen_key": "domain:sample",
        "frame": "registered_cscan",
        "source_image_sha256": "a" * 64,
        "reference_type": "EXPERT_REVIEWED",
        "review_state": "reviewed",
        "reviewer_alias": "R1",
        "regions": [
            {
                "id": "r1",
                "polygon": [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]],
                "certainty": "certain",
            }
        ],
        "uncertain_regions": [],
        "notes": "independent review",
    }

    reference = reference_from_payload(payload, native_shape=(10, 10))
    assert reference.formal_eligible is True
    assert reference.certain_mask[4, 4]

    payload["reference_type"] = "ALGORITHM_DERIVED_NOT_REVIEWED"
    with pytest.raises(ValueError, match="review"):
        reference_from_payload(payload, native_shape=(10, 10))


def test_public_locate_stop_requires_two_stable_measured_updates() -> None:
    """Catches authorizing STOP from one report or without new measurements."""

    prediction = np.zeros((8, 8), dtype=np.bool_)
    prediction[2:4, 3:5] = True
    base = TaskReport(
        task=BenchmarkTask.LOCATE,
        predicted_mask=prediction,
        support_positions=np.asarray([[2, 3], [2, 4], [3, 3]], dtype=np.int64),
        confidence=0.9,
        public_complete=False,
        reason_code="EVIDENCE_PRESENT",
    )
    tracker = PublicStopTracker(BenchmarkTask.LOCATE)

    first = tracker.update(base, measured_count=3, cell_levels=(-1,) * 64)
    unchanged = tracker.update(base, measured_count=3, cell_levels=(-1,) * 64)
    second = tracker.update(base, measured_count=4, cell_levels=(-1,) * 64)
    third = tracker.update(base, measured_count=5, cell_levels=(-1,) * 64)

    assert first.public_complete is False
    assert unchanged.public_complete is False
    assert second.public_complete is False
    assert third.public_complete is True


def test_proxy_background_uses_inner_ring_not_rendered_outer_frame() -> None:
    """Catches a bright plotting frame turning the whole C-scan into a proposal."""

    image = np.full((40, 40, 3), (80, 80, 160), dtype=np.uint8)
    image[[0, -1], :, :] = 255
    image[:, [0, -1], :] = 255
    image[16:24, 17:23] = (240, 40, 30)

    reference = derive_proxy_reference(
        specimen_key="domain:sample",
        source_image_sha256="a" * 64,
        full_scan=image,
        distance_threshold=0.18,
        minimum_component_pixels=9,
        uncertainty_band=0.02,
        border_exclusion_fraction=0.10,
    )

    assert reference.certain_mask[20, 20]
    assert not reference.certain_mask[10, 10]
    assert np.count_nonzero(reference.certain_mask) < 100
