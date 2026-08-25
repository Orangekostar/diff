from __future__ import annotations

import dataclasses

import numpy as np
from mavis_test_support import synthetic_inputs

from cmc_bbdm.mavis.authority import MAVISAuthority
from cmc_bbdm.mavis.contracts import EvaluationView, PolicyContext, SourceTeacherView


def test_mavis_target_true_cai_cannot_change_policy() -> None:
    first = MAVISAuthority.from_arrays(**synthetic_inputs(true_cai=0.1))
    second = MAVISAuthority.from_arrays(**synthetic_inputs(true_cai=0.9))

    first_context = first.policy_context("sample-001")
    second_context = second.policy_context("sample-001")

    assert first_context == second_context
    assert isinstance(first_context, PolicyContext)
    assert not hasattr(first_context, "true_cai")
    assert not hasattr(first_context, "full_scan")
    assert tuple(field.name for field in dataclasses.fields(first_context)) == (
        "specimen_id",
        "context_features",
        "native_shape",
        "native_count",
        "state_sha256",
    )


def test_mavis_authority_separates_policy_teacher_and_evaluation_views() -> None:
    inputs = synthetic_inputs(true_cai=0.4)
    authority = MAVISAuthority.from_arrays(**inputs)

    policy = authority.policy_context("sample-001")
    teacher = authority.source_teacher_view("sample-001")
    evaluation = authority.evaluation_view("sample-001")

    assert isinstance(policy, PolicyContext)
    assert isinstance(teacher, SourceTeacherView)
    assert isinstance(evaluation, EvaluationView)
    assert policy.context_features.shape == (34,)
    np.testing.assert_array_equal(policy.context_features[:13], inputs["metadata13"][0])
    np.testing.assert_array_equal(
        policy.context_features[13:], inputs["profile_stats21"][0]
    )
    assert teacher.dataset_id == "domain-a"
    assert teacher.true_cai == 0.4
    np.testing.assert_array_equal(teacher.full_scan, inputs["images"][0])
    assert not teacher.full_scan.flags.writeable
    assert evaluation.dataset_id == "domain-a"
    assert evaluation.true_cai == 0.4
    assert not evaluation.full_scan.flags.writeable


def test_mavis_authority_snapshots_all_numeric_inputs() -> None:
    inputs = synthetic_inputs()
    authority = MAVISAuthority.from_arrays(**inputs)
    policy_before = authority.policy_context("sample-001")
    teacher_before = authority.source_teacher_view("sample-001")

    inputs["images"][0][:] = 0
    inputs["targets"][:] = 0
    inputs["metadata13"][:] = 0
    inputs["profile_stats21"][:] = 0

    np.testing.assert_array_equal(
        authority.policy_context("sample-001").context_features,
        policy_before.context_features,
    )
    np.testing.assert_array_equal(
        authority.source_teacher_view("sample-001").full_scan,
        teacher_before.full_scan,
    )
    assert authority.source_teacher_view("sample-001").true_cai == 0.4
