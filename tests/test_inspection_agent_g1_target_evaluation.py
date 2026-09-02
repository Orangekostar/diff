from __future__ import annotations

from dataclasses import replace

import pytest
from test_inspection_agent_g1_target_execution import (
    _Actor,
    _Assessor,
    _Encoder,
    _sha,
    _target_runtime,
)

from cmc_bbdm.inspection_agent.contracts import InspectionTask
from cmc_bbdm.inspection_agent.generalized_reconstruction import SourceBackgroundPrior
from cmc_bbdm.inspection_agent_g1.contracts import CAIContextMode
from cmc_bbdm.inspection_agent_g1.target_evaluation import (
    G1TargetEvaluationError,
    build_g1_outer_target_curve_bank,
    evaluate_g1_target_trajectory_bank,
    read_g1_target_curve_bank,
    target_curve_fragment_path,
    write_g1_target_curve_bank,
)
from cmc_bbdm.inspection_agent_g1.target_execution import (
    TargetPolicyVariant,
    materialize_g1_target_variant_records,
    seal_g1_target_trajectory_bank,
    write_g1_target_trajectory_bank,
)
from cmc_bbdm.mavis.authority import MAVISAuthority


def _sealed_bank(tmp_path):
    runtime = _target_runtime()
    prior = SourceBackgroundPrior(
        outer_domain="d6",
        source_domains=("d1", "d2", "d3", "d4", "d5"),
        fit_specimen_ids=("a", "b", "c", "d", "e"),
        source_authority_sha256=_sha("source-authority"),
        domain_border_medians=[[0.0, 0.0, 0.0]] * 5,
        background_rgb=[0, 0, 0],
    )
    records = materialize_g1_target_variant_records(
        runtime,
        outer_target="d6",
        specimen_ids=("target-a",),
        task=InspectionTask.CAI,
        variant=TargetPolicyVariant.NO_TASK,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        actor=_Actor(),
        cai_context_mode=CAIContextMode.SHARED_OBSERVABLE_STATE_CONTEXT,
        endpoint_budget=0.25,
        final_dependency_sha256=_sha("dependencies"),
        action_selection_sha256=_sha("selection"),
        action_model_sha256=_sha("action-model"),
        stop_model_sha256=_sha("stop-model"),
        stop_threshold=None,
    )
    bank = write_g1_target_trajectory_bank(tmp_path / "target.parquet", records)
    seal = seal_g1_target_trajectory_bank(runtime, bank, records)
    return runtime, prior, bank, records, seal


def test_sealed_target_bank_opens_truth_and_issues_learned_curve(tmp_path) -> None:
    runtime, prior, bank, records, seal = _sealed_bank(tmp_path)

    output = evaluate_g1_target_trajectory_bank(
        runtime,
        bank,
        records,
        seal,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
    )

    assert len(output) == 1
    result = output[0]
    assert result.variant is TargetPolicyVariant.NO_TASK
    assert result.curve.task is InspectionTask.CAI
    assert result.curve.method == TargetPolicyVariant.NO_TASK.value
    assert result.curve.target_domain == "d6"
    assert result.bank_seal_sha256 == seal.state_sha256
    assert result.truth_view_sha256


def test_tampered_bank_seal_is_rejected_before_target_truth_opens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    runtime, prior, bank, records, seal = _sealed_bank(tmp_path)
    opened = 0

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal opened
        opened += 1
        raise AssertionError("target truth opened for an invalid bank seal")

    monkeypatch.setattr(MAVISAuthority, "evaluation_view", forbidden)
    tampered = replace(seal, records_sha256=_sha("tampered"))

    with pytest.raises(G1TargetEvaluationError, match="seal"):
        evaluate_g1_target_trajectory_bank(
            runtime,
            bank,
            records,
            tampered,
            prior=prior,
            assessor=_Assessor(),
            encoder=_Encoder(),
        )
    assert opened == 0


def test_target_curve_bank_roundtrip_preserves_exact_curve(tmp_path) -> None:
    runtime, prior, trajectory_bank, trajectories, seal = _sealed_bank(tmp_path)
    records = evaluate_g1_target_trajectory_bank(
        runtime,
        trajectory_bank,
        trajectories,
        seal,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
    )
    path = tmp_path / "curves.parquet"

    identity = write_g1_target_curve_bank(path, records)
    replay_identity, replay = read_g1_target_curve_bank(path)

    assert replay_identity == identity
    assert tuple(row.state_sha256 for row in replay) == tuple(
        row.state_sha256 for row in records
    )
    assert replay[0].curve.state_sha256 == records[0].curve.state_sha256
    assert replay[0].curve.exact_budgets.tobytes() == (
        records[0].curve.exact_budgets.tobytes()
    )
    assert replay[0].curve.task_losses.tobytes() == records[0].curve.task_losses.tobytes()


def test_target_curve_bank_rejects_manifest_tampering(tmp_path) -> None:
    runtime, prior, trajectory_bank, trajectories, seal = _sealed_bank(tmp_path)
    records = evaluate_g1_target_trajectory_bank(
        runtime,
        trajectory_bank,
        trajectories,
        seal,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
    )
    path = tmp_path / "curves.parquet"
    write_g1_target_curve_bank(path, records)
    manifest = path.with_suffix(".parquet.manifest.json")
    manifest.write_bytes(manifest.read_bytes().replace(b'"row_count":1', b'"row_count":2'))

    with pytest.raises(G1TargetEvaluationError, match="evidence changed"):
        read_g1_target_curve_bank(path)


def test_outer_target_curve_bank_replays_without_reopening_truth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    runtime, prior, trajectory_bank, trajectories, seal = _sealed_bank(tmp_path)
    work_root = tmp_path / "evaluation"
    first = build_g1_outer_target_curve_bank(
        runtime,
        trajectory_bank,
        trajectories,
        seal,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        work_root=work_root,
    )
    assert first.specimen_count == 1
    assert first.record_count == 1
    assert first.target_outcomes_opened

    monkeypatch.setattr(
        MAVISAuthority,
        "evaluation_view",
        lambda *_args, **_kwargs: pytest.fail("complete target curves reopened truth"),
    )
    replay = build_g1_outer_target_curve_bank(
        runtime,
        trajectory_bank,
        trajectories,
        seal,
        prior=prior,
        assessor=_Assessor(),
        encoder=_Encoder(),
        work_root=work_root,
    )
    assert replay.state_sha256 == first.state_sha256


def test_outer_target_curve_bank_rejects_half_written_specimen_fragment(
    tmp_path,
) -> None:
    runtime, prior, trajectory_bank, trajectories, seal = _sealed_bank(tmp_path)
    work_root = tmp_path / "evaluation"
    fragment = target_curve_fragment_path(work_root, "d6", "target-a")
    fragment.parent.mkdir(parents=True)
    fragment.write_bytes(b"incomplete")

    with pytest.raises(G1TargetEvaluationError, match="fragment is incomplete"):
        build_g1_outer_target_curve_bank(
            runtime,
            trajectory_bank,
            trajectories,
            seal,
            prior=prior,
            assessor=_Assessor(),
            encoder=_Encoder(),
            work_root=work_root,
        )
