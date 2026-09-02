from __future__ import annotations

from dataclasses import replace

import pytest
from test_inspection_agent_g1_target_evaluation import _sealed_bank
from test_inspection_agent_g1_target_execution import _Assessor, _Encoder, _sha

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent_g1.formal import FIXED_BASELINE_METHODS
from cmc_bbdm.inspection_agent_g1.target_reference import (
    TARGET_ORACLE_METHODS,
    G1TargetReferenceError,
    build_g1_outer_target_reference_bank,
    evaluate_g1_target_reference_curves,
    read_g1_target_reference_bank,
    target_reference_fragment_path,
    write_g1_target_reference_bank,
)
from cmc_bbdm.mavis.authority import MAVISAuthority


def test_target_reference_curves_share_frozen_geometry_and_truth_view(tmp_path) -> None:
    runtime, prior, bank, trajectories, seal = _sealed_bank(tmp_path)

    records = evaluate_g1_target_reference_curves(
        runtime,
        bank,
        trajectories,
        seal,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        random_seed=2026090101,
    )

    assert len(records) == 12
    for task in (InspectionTask.FIELD, InspectionTask.CAI):
        task_rows = tuple(row for row in records if row.task is task)
        assert {row.method for row in task_rows} == {
            *FIXED_BASELINE_METHODS,
            TARGET_ORACLE_METHODS[task],
        }
        assert len({row.curve.grid_sha256 for row in task_rows}) == 1
        assert len({row.curve.warm_start_sha256 for row in task_rows}) == 1
    assert len({row.truth_view_sha256 for row in records}) == 1
    assert all(row.bank_seal_sha256 == seal.state_sha256 for row in records)


def test_target_reference_rejects_bad_seal_before_truth_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    runtime, prior, bank, trajectories, seal = _sealed_bank(tmp_path)
    opened = 0

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal opened
        opened += 1
        raise AssertionError("reference evaluator opened target truth before seal")

    monkeypatch.setattr(MAVISAuthority, "evaluation_view", forbidden)
    tampered = replace(seal, records_sha256=_sha("tampered-reference-seal"))

    with pytest.raises(G1TargetReferenceError, match="seal"):
        evaluate_g1_target_reference_curves(
            runtime,
            bank,
            trajectories,
            tampered,
            prior=prior,
            assessor=_Assessor(),
            encoder=_Encoder(),
            random_seed=2026090101,
        )
    assert opened == 0


def test_target_reference_bank_roundtrip_is_exact(tmp_path) -> None:
    runtime, prior, trajectory_bank, trajectories, seal = _sealed_bank(tmp_path)
    records = evaluate_g1_target_reference_curves(
        runtime,
        trajectory_bank,
        trajectories,
        seal,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        random_seed=2026090101,
    )
    path = tmp_path / "references.parquet"

    identity = write_g1_target_reference_bank(path, records)
    replay_identity, replay = read_g1_target_reference_bank(path)

    assert replay_identity == identity
    assert tuple(row.state_sha256 for row in replay) == tuple(
        row.state_sha256 for row in records
    )
    assert all(
        actual.curve.task_losses.tobytes() == expected.curve.task_losses.tobytes()
        for actual, expected in zip(replay, records, strict=True)
    )


def test_outer_target_reference_bank_replays_without_truth_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    runtime, prior, trajectory_bank, trajectories, seal = _sealed_bank(tmp_path)
    work_root = tmp_path / "references"
    first = build_g1_outer_target_reference_bank(
        runtime,
        trajectory_bank,
        trajectories,
        seal,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        random_seed=2026090101,
        work_root=work_root,
    )
    assert first.specimen_count == 1
    assert first.record_count == 12

    monkeypatch.setattr(
        MAVISAuthority,
        "evaluation_view",
        lambda *_args, **_kwargs: pytest.fail("complete references reopened truth"),
    )
    replay = build_g1_outer_target_reference_bank(
        runtime,
        trajectory_bank,
        trajectories,
        seal,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        random_seed=2026090101,
        work_root=work_root,
    )
    assert replay.state_sha256 == first.state_sha256


def test_outer_target_reference_bank_rejects_half_written_fragment(tmp_path) -> None:
    runtime, prior, trajectory_bank, trajectories, seal = _sealed_bank(tmp_path)
    work_root = tmp_path / "references"
    fragment = target_reference_fragment_path(work_root, "d6", "target-a")
    fragment.parent.mkdir(parents=True)
    fragment.write_bytes(b"incomplete")

    with pytest.raises(G1TargetReferenceError, match="fragment is incomplete"):
        build_g1_outer_target_reference_bank(
            runtime,
            trajectory_bank,
            trajectories,
            seal,
            prior=prior,
            assessor=_Assessor(),
            encoder=_Encoder(),
            random_seed=2026090101,
            work_root=work_root,
        )
