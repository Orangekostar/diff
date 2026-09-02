from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from test_inspection_agent_g1_target_analysis import _formal_inputs, _sha
from test_inspection_agent_g1_target_stopping import _outcomes

from cmc_bbdm.inspection_agent.contracts import InspectionDecision, InspectionTask
from cmc_bbdm.inspection_agent_g1.artifacts import (
    REQUIRED_G1_OUTPUTS,
    compare_g1_packages,
)
from cmc_bbdm.inspection_agent_g1.decision_diagnostics import (
    G1SourceDecisionDiagnosticBankFile,
    G1SourceDecisionDiagnosticRecord,
)
from cmc_bbdm.inspection_agent_g1.formal_package import (
    G1TeacherBankManifestRow,
    write_g1_formal_package,
)
from cmc_bbdm.inspection_agent_g1.formal_selection import (
    G1FrozenStopThreshold,
    G1OuterFormalSelection,
)
from cmc_bbdm.inspection_agent_g1.target_analysis import analyze_g1_target_curves
from cmc_bbdm.inspection_agent_g1.target_execution import (
    G1TargetTrajectoryRecord,
    TargetPolicyVariant,
)
from cmc_bbdm.inspection_agent_g1.target_stopping import (
    analyze_g1_target_stopping,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/inspection_agent_g1.yaml"


def _package_evidence():
    learned, references, fixed = _formal_inputs()
    curve_analysis = analyze_g1_target_curves(
        learned,
        references,
        fixed,
        no_target_leakage=True,
        deterministic_replay=True,
        deployment_bridge_valid=True,
    )
    outcomes = _outcomes()
    stopping = analyze_g1_target_stopping(outcomes)
    selections = []
    trajectories = []
    teacher_banks = []
    diagnostic_banks = []
    diagnostics = []
    domains = tuple(sorted({row.outer_target for row in fixed}))
    for domain in domains:
        domain_fixed = tuple(
            row
            for row in fixed
            if row.outer_target == domain
        )
        sources = domain_fixed[0].source_domains
        selections.append(
            G1OuterFormalSelection(
                outer_target=domain,
                source_domains=sources,
                fixed_selections=domain_fixed,
                stop_thresholds=tuple(
                    G1FrozenStopThreshold(
                        outer_target=domain,
                        task=task,
                        status="STOP_NOT_AUTHORIZED",
                        threshold=None,
                        source_evidence_sha256=_sha(
                            f"threshold-{domain}-{task.value}"
                        ),
                    )
                    for task in (InspectionTask.FIELD, InspectionTask.CAI)
                ),
                action_selection_sha256=_sha(f"selection-{domain}"),
                action_model_sha256=_sha(f"action-model-{domain}"),
                stop_model_sha256=_sha(f"stop-model-{domain}"),
                decision_diagnostic_manifest_sha256=_sha(
                    f"decision-diagnostics-{domain}"
                ),
            )
        )
        specimen = f"{domain}-specimen"
        fake = object.__new__(G1TargetTrajectoryRecord)
        object.__setattr__(fake, "outer_target", domain)
        object.__setattr__(fake, "specimen_id", specimen)
        object.__setattr__(fake, "specimen_sha256", _sha(specimen))
        object.__setattr__(fake, "task", InspectionTask.CAI)
        object.__setattr__(fake, "variant", TargetPolicyVariant.PROPOSED)
        object.__setattr__(fake, "final_dependency_sha256", _sha(f"dependency-{domain}"))
        object.__setattr__(fake, "action_selection_sha256", _sha(f"selection-{domain}"))
        object.__setattr__(fake, "action_model_sha256", _sha(f"action-model-{domain}"))
        object.__setattr__(fake, "stop_model_sha256", _sha(f"stop-model-{domain}"))
        object.__setattr__(fake, "state_sha256", _sha(f"target-record-{domain}"))
        object.__setattr__(
            fake,
            "trajectory",
            SimpleNamespace(
                state_sha256=_sha(f"trajectory-{domain}"),
                stop_threshold=None,
                stopped=False,
                termination_reason="NO_LEGAL_ACTION",
                action_history=(),
                steps=(),
                final_observation_sha256=_sha(f"observation-{domain}"),
                acquired_positions_sha256=_sha(f"positions-{domain}"),
                acquired_values_sha256=_sha(f"values-{domain}"),
                native_count=100,
                effective_budget=0.25,
            ),
        )
        trajectories.append(fake)
        for source in sources:
            teacher_banks.append(
                G1TeacherBankManifestRow(
                    outer_target=domain,
                    source_domain=source,
                    row_count=10,
                    parquet_sha256=_sha(f"teacher-parquet-{domain}-{source}"),
                    records_sha256=_sha(f"teacher-records-{domain}-{source}"),
                    manifest_sha256=_sha(f"teacher-manifest-{domain}-{source}"),
                )
            )
            for task in (InspectionTask.FIELD, InspectionTask.CAI):
                diagnostics.append(
                    G1SourceDecisionDiagnosticRecord(
                        outer_target=domain,
                        source_domain=source,
                        specimen_sha256=_sha(f"specimen-{domain}-{source}"),
                        task=task,
                        dagger_iteration=0,
                        state_origin="TEACHER_BANK",
                        training_example_sha256=_sha(
                            f"example-{domain}-{source}-{task.value}"
                        ),
                        policy_state_sha256=_sha(
                            f"policy-state-{domain}-{source}-{task.value}"
                        ),
                        teacher_label_sha256=_sha(
                            f"teacher-label-{domain}-{source}-{task.value}"
                        ),
                        action_model_sha256=_sha(f"action-model-{domain}"),
                        score_sha256=_sha(
                            f"scores-{domain}-{source}-{task.value}"
                        ),
                        candidate_count=2,
                        teacher_selected_slot=0,
                        predicted_selected_slot=0,
                        teacher_decision=InspectionDecision.FOCUS,
                        predicted_decision=InspectionDecision.FOCUS,
                        high_level_decision_accuracy=1.0,
                        primitive_top1_match=1.0,
                        top5_utility_recall=1.0,
                        expected_teacher_regret=0.0,
                        candidate_utility_ndcg=1.0,
                    )
                )
        diagnostic_banks.append(
            G1SourceDecisionDiagnosticBankFile(
                row_count=10,
                outer_target=domain,
                source_domains=sources,
                action_model_sha256=_sha(f"action-model-{domain}"),
                parquet_sha256=_sha(f"diagnostic-parquet-{domain}"),
                records_sha256=_sha(f"diagnostic-records-{domain}"),
                manifest_sha256=_sha(f"decision-diagnostics-{domain}"),
            )
        )
    return (
        tuple(selections),
        tuple(trajectories),
        learned,
        references,
        outcomes,
        curve_analysis,
        stopping,
        tuple(teacher_banks),
        tuple(diagnostic_banks),
        tuple(diagnostics),
    )


def test_formal_package_is_complete_and_byte_replayable(tmp_path) -> None:
    evidence = _package_evidence()
    formal = tmp_path / "formal"
    replay = tmp_path / "replay"

    first = write_g1_formal_package(
        formal,
        *evidence,
        project_root=ROOT,
        config_path=CONFIG,
    )
    second = write_g1_formal_package(
        replay,
        *evidence,
        project_root=ROOT,
        config_path=CONFIG,
    )
    comparison = compare_g1_packages(
        formal,
        replay,
        project_root=ROOT,
        config_path=CONFIG,
    )

    assert first == second
    assert comparison.byte_identical
    assert first.status == "G1_TASK_CONDITIONED_POLICY_GO"
    assert {path.name for path in formal.iterdir()} == {
        *REQUIRED_G1_OUTPUTS,
        "config.yaml",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
    with (formal / "state_level_metrics.csv").open(
        encoding="ascii", newline=""
    ) as handle:
        state_rows = tuple(csv.DictReader(handle))
    assert state_rows
    assert all(None not in row for row in state_rows)
    assert {row["record_type"] for row in state_rows} == {
        "TARGET_ENGINEERING_CHECKPOINT",
        "SOURCE_DECISION_DIAGNOSTIC",
    }
