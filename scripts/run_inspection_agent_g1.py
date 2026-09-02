"""Command-line entry point for the inspection agent G1 audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import cmc_bbdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PACKAGE = str(_PROJECT_ROOT / "src/cmc_bbdm")
if _LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.append(_LOCAL_PACKAGE)

from cmc_bbdm.inspection_agent_g1 import (
    G1ArtifactError,
    G1EngineeringSelectionExecutionError,
    G1ExecutionError,
    G1SourceBridgeError,
    G1StopExecutionError,
    G1TeacherBankManifestRow,
    analyze_g1_target_curves,
    analyze_g1_target_stopping,
    build_g1_all_source_bridge_banks,
    build_g1_all_source_fixed_endpoint_banks,
    build_g1_all_source_stop_banks,
    build_g1_all_source_teacher_banks,
    build_g1_final_dependencies,
    build_g1_outer_target_curve_bank,
    build_g1_outer_target_reference_bank,
    build_g1_outer_target_trajectory_bank,
    build_g1_source_bridge_bank,
    build_g1_source_dependencies,
    build_g1_source_fixed_endpoint_bank,
    build_g1_source_stop_bank,
    build_g1_source_teacher_bank,
    compare_g1_packages,
    freeze_g1_outer_formal_selection,
    load_g1_encoder,
    load_g1_protocol,
    load_g1_runtime,
    materialize_g1_source_decision_diagnostics,
    materialize_g1_target_stop_outcomes,
    outer_formal_selection_path,
    read_g1_outer_formal_selection,
    read_g1_source_decision_diagnostic_bank,
    read_g1_target_curve_bank,
    read_g1_target_reference_bank,
    read_g1_target_trajectory_bank,
    read_source_bridge_bank,
    read_teacher_bank,
    rebind_training_example_modes,
    run_outer_dagger_selection,
    run_outer_engineering_selection,
    run_outer_stop_selection,
    run_outer_supervised_selection,
    seal_g1_target_trajectory_bank,
    source_bridge_bank_path,
    source_decision_diagnostic_bank_path,
    target_curve_bank_path,
    target_reference_bank_path,
    target_trajectory_bank_path,
    validate_g1_package,
    write_g1_formal_package,
    write_g1_source_decision_diagnostic_bank,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_inspection_agent_g1.py")
    commands = parser.add_subparsers(dest="command", required=True)

    build_bank = commands.add_parser("build-bank")
    build_bank.add_argument("--config", required=True)
    build_bank.add_argument("--source-project-root", required=True)
    build_bank.add_argument("--outer-target", required=True)
    build_bank.add_argument("--source-domain", required=True)
    build_bank.add_argument("--project-root", default=str(_PROJECT_ROOT))
    build_bank.add_argument("--device", default=None)
    build_bank.add_argument("--work-root", default=None)

    build_all = commands.add_parser("build-all-banks")
    build_all.add_argument("--config", required=True)
    build_all.add_argument("--source-project-root", required=True)
    build_all.add_argument("--project-root", default=str(_PROJECT_ROOT))
    build_all.add_argument("--device", default=None)
    build_all.add_argument("--work-root", default=None)
    build_all.add_argument("--start-fold", type=int, default=1)

    build_fixed = commands.add_parser("build-fixed-endpoints")
    build_fixed.add_argument("--config", required=True)
    build_fixed.add_argument("--source-project-root", required=True)
    build_fixed.add_argument("--outer-target", required=True)
    build_fixed.add_argument("--source-domain", required=True)
    build_fixed.add_argument("--project-root", default=str(_PROJECT_ROOT))
    build_fixed.add_argument("--device", default=None)
    build_fixed.add_argument("--work-root", default=None)

    build_all_fixed = commands.add_parser("build-all-fixed-endpoints")
    build_all_fixed.add_argument("--config", required=True)
    build_all_fixed.add_argument("--source-project-root", required=True)
    build_all_fixed.add_argument("--project-root", default=str(_PROJECT_ROOT))
    build_all_fixed.add_argument("--device", default=None)
    build_all_fixed.add_argument("--work-root", default=None)
    build_all_fixed.add_argument("--start-fold", type=int, default=1)

    build_stop = commands.add_parser("build-stop-bank")
    build_stop.add_argument("--config", required=True)
    build_stop.add_argument("--source-project-root", required=True)
    build_stop.add_argument("--outer-target", required=True)
    build_stop.add_argument("--source-domain", required=True)
    build_stop.add_argument("--project-root", default=str(_PROJECT_ROOT))
    build_stop.add_argument("--device", default=None)
    build_stop.add_argument("--teacher-bank-root", default=None)
    build_stop.add_argument("--fixed-endpoint-root", default=None)
    build_stop.add_argument("--work-root", default=None)

    build_all_stop = commands.add_parser("build-all-stop-banks")
    build_all_stop.add_argument("--config", required=True)
    build_all_stop.add_argument("--source-project-root", required=True)
    build_all_stop.add_argument("--project-root", default=str(_PROJECT_ROOT))
    build_all_stop.add_argument("--device", default=None)
    build_all_stop.add_argument("--teacher-bank-root", default=None)
    build_all_stop.add_argument("--fixed-endpoint-root", default=None)
    build_all_stop.add_argument("--work-root", default=None)
    build_all_stop.add_argument("--start-fold", type=int, default=1)

    build_bridge = commands.add_parser("build-source-bridge")
    build_bridge.add_argument("--config", required=True)
    build_bridge.add_argument("--source-project-root", required=True)
    build_bridge.add_argument("--outer-target", required=True)
    build_bridge.add_argument("--source-domain", required=True)
    build_bridge.add_argument("--project-root", default=str(_PROJECT_ROOT))
    build_bridge.add_argument("--device", default=None)
    build_bridge.add_argument("--work-root", default=None)

    build_all_bridges = commands.add_parser("build-all-source-bridges")
    build_all_bridges.add_argument("--config", required=True)
    build_all_bridges.add_argument("--source-project-root", required=True)
    build_all_bridges.add_argument("--project-root", default=str(_PROJECT_ROOT))
    build_all_bridges.add_argument("--device", default=None)
    build_all_bridges.add_argument("--work-root", default=None)
    build_all_bridges.add_argument("--start-fold", type=int, default=1)
    build_all_bridges.add_argument("--end-fold", type=int, default=None)

    select_outer = commands.add_parser("select-outer")
    select_outer.add_argument("--config", required=True)
    select_outer.add_argument("--outer-target", required=True)
    select_outer.add_argument("--project-root", default=str(_PROJECT_ROOT))
    select_outer.add_argument("--bank-root", default=None)
    select_outer.add_argument("--work-root", default=None)
    select_outer.add_argument("--device", default=None)

    select_engineering = commands.add_parser("select-outer-engineering")
    select_engineering.add_argument("--config", required=True)
    select_engineering.add_argument("--source-project-root", required=True)
    select_engineering.add_argument("--outer-target", required=True)
    select_engineering.add_argument("--project-root", default=str(_PROJECT_ROOT))
    select_engineering.add_argument("--device", default=None)
    select_engineering.add_argument("--teacher-bank-root", default=None)
    select_engineering.add_argument("--bridge-root", default=None)
    select_engineering.add_argument("--supervised-root", default=None)
    select_engineering.add_argument("--learned-root", default=None)
    select_engineering.add_argument("--work-root", default=None)

    select_dagger = commands.add_parser("select-outer-dagger")
    select_dagger.add_argument("--config", required=True)
    select_dagger.add_argument("--source-project-root", required=True)
    select_dagger.add_argument("--outer-target", required=True)
    select_dagger.add_argument("--project-root", default=str(_PROJECT_ROOT))
    select_dagger.add_argument("--device", default=None)
    select_dagger.add_argument("--teacher-bank-root", default=None)
    select_dagger.add_argument("--bridge-root", default=None)
    select_dagger.add_argument("--supervised-root", default=None)
    select_dagger.add_argument("--learned-root", default=None)
    select_dagger.add_argument("--base-work-root", default=None)
    select_dagger.add_argument("--dagger-bank-root", default=None)
    select_dagger.add_argument("--work-root", default=None)

    select_stop = commands.add_parser("select-outer-stop")
    select_stop.add_argument("--config", required=True)
    select_stop.add_argument("--source-project-root", required=True)
    select_stop.add_argument("--outer-target", required=True)
    select_stop.add_argument("--project-root", default=str(_PROJECT_ROOT))
    select_stop.add_argument("--device", default=None)
    select_stop.add_argument("--teacher-bank-root", default=None)
    select_stop.add_argument("--bridge-root", default=None)
    select_stop.add_argument("--supervised-root", default=None)
    select_stop.add_argument("--learned-root", default=None)
    select_stop.add_argument("--base-work-root", default=None)
    select_stop.add_argument("--dagger-bank-root", default=None)
    select_stop.add_argument("--dagger-work-root", default=None)
    select_stop.add_argument("--fixed-endpoint-root", default=None)
    select_stop.add_argument("--stop-bank-root", default=None)
    select_stop.add_argument("--work-root", default=None)

    build_target = commands.add_parser("build-target-trajectories")
    build_target.add_argument("--config", required=True)
    build_target.add_argument("--source-project-root", required=True)
    build_target.add_argument("--outer-target", required=True)
    build_target.add_argument("--project-root", default=str(_PROJECT_ROOT))
    build_target.add_argument("--device", default=None)
    build_target.add_argument("--teacher-bank-root", default=None)
    build_target.add_argument("--bridge-root", default=None)
    build_target.add_argument("--supervised-root", default=None)
    build_target.add_argument("--learned-root", default=None)
    build_target.add_argument("--base-work-root", default=None)
    build_target.add_argument("--dagger-bank-root", default=None)
    build_target.add_argument("--dagger-work-root", default=None)
    build_target.add_argument("--fixed-endpoint-root", default=None)
    build_target.add_argument("--stop-bank-root", default=None)
    build_target.add_argument("--stop-selection-root", default=None)
    build_target.add_argument("--formal-selection-root", default=None)
    build_target.add_argument("--decision-diagnostic-root", default=None)
    build_target.add_argument("--work-root", default=None)

    evaluate_target = commands.add_parser("evaluate-target")
    evaluate_target.add_argument("--config", required=True)
    evaluate_target.add_argument("--source-project-root", required=True)
    evaluate_target.add_argument("--outer-target", required=True)
    evaluate_target.add_argument("--project-root", default=str(_PROJECT_ROOT))
    evaluate_target.add_argument("--device", default=None)
    evaluate_target.add_argument("--trajectory-root", default=None)
    evaluate_target.add_argument("--formal-selection-root", default=None)
    evaluate_target.add_argument("--curve-root", default=None)
    evaluate_target.add_argument("--reference-root", default=None)

    build_formal = commands.add_parser("build-formal-package")
    build_formal.add_argument("--config", required=True)
    build_formal.add_argument("--source-project-root", required=True)
    build_formal.add_argument("--project-root", default=str(_PROJECT_ROOT))
    build_formal.add_argument("--teacher-bank-root", default=None)
    build_formal.add_argument("--trajectory-root", default=None)
    build_formal.add_argument("--curve-root", default=None)
    build_formal.add_argument("--reference-root", default=None)
    build_formal.add_argument("--formal-selection-root", default=None)
    build_formal.add_argument("--decision-diagnostic-root", default=None)
    build_formal.add_argument("--output", default=None)

    validate = commands.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--path", required=True)
    validate.add_argument("--project-root", default=str(_PROJECT_ROOT))

    compare = commands.add_parser("compare")
    compare.add_argument("--config", required=True)
    compare.add_argument("--formal", required=True)
    compare.add_argument("--replay", required=True)
    compare.add_argument("--project-root", default=str(_PROJECT_ROOT))

    return parser


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-formal-package":
            protocol = load_g1_protocol(
                args.config,
                project_root=args.project_root,
            )
            runtime = load_g1_runtime(
                protocol,
                project_root=args.project_root,
                source_project_root=args.source_project_root,
                progress=_progress,
            )
            work_base = Path(args.project_root) / protocol.work_output
            teacher_root = args.teacher_bank_root or str(
                Path(args.project_root) / protocol.teacher_bank_work_path
            )
            trajectory_root = args.trajectory_root or str(
                work_base / "target_trajectories"
            )
            curve_root = args.curve_root or str(work_base / "target_curves")
            reference_root = args.reference_root or str(
                work_base / "target_references"
            )
            selection_root = args.formal_selection_root or str(
                work_base / "formal_selection"
            )
            diagnostic_root = args.decision_diagnostic_root or str(
                work_base / "source_decision_diagnostics"
            )
            selections = []
            trajectories = []
            learned_curves = []
            reference_curves = []
            teacher_manifests = []
            diagnostic_banks = []
            decision_diagnostics = []
            for outer_target in protocol.domain_order:
                selection = read_g1_outer_formal_selection(
                    outer_formal_selection_path(selection_root, outer_target)
                )
                trajectory_bank, target_rows = read_g1_target_trajectory_bank(
                    target_trajectory_bank_path(trajectory_root, outer_target)
                )
                seal = seal_g1_target_trajectory_bank(
                    runtime,
                    trajectory_bank,
                    target_rows,
                )
                curve_bank, curve_rows = read_g1_target_curve_bank(
                    target_curve_bank_path(curve_root, outer_target)
                )
                reference_bank, reference_rows = read_g1_target_reference_bank(
                    target_reference_bank_path(reference_root, outer_target)
                )
                diagnostic_bank, diagnostic_rows = (
                    read_g1_source_decision_diagnostic_bank(
                        source_decision_diagnostic_bank_path(
                            diagnostic_root,
                            outer_target,
                        )
                    )
                )
                if (
                    selection.outer_target != outer_target
                    or {row.action_selection_sha256 for row in target_rows}
                    != {selection.action_selection_sha256}
                    or {row.action_model_sha256 for row in target_rows}
                    != {selection.action_model_sha256}
                    or {row.stop_model_sha256 for row in target_rows}
                    != {selection.stop_model_sha256}
                    or curve_bank.trajectory_bank_seal_sha256
                    != seal.state_sha256
                    or reference_bank.trajectory_bank_seal_sha256
                    != seal.state_sha256
                    or {row.bank_seal_sha256 for row in curve_rows}
                    != {seal.state_sha256}
                    or {row.bank_seal_sha256 for row in reference_rows}
                    != {seal.state_sha256}
                    or diagnostic_bank.manifest_sha256
                    != selection.decision_diagnostic_manifest_sha256
                    or diagnostic_bank.action_model_sha256
                    != selection.action_model_sha256
                ):
                    raise G1ExecutionError(
                        "formal target evidence differs from its frozen selection"
                    )
                selections.append(selection)
                trajectories.extend(target_rows)
                learned_curves.extend(curve_rows)
                reference_curves.extend(reference_rows)
                diagnostic_banks.append(diagnostic_bank)
                decision_diagnostics.extend(diagnostic_rows)
                for source_domain in selection.source_domains:
                    teacher_bank, _teacher_rows = read_teacher_bank(
                        Path(teacher_root)
                        / outer_target
                        / f"{source_domain}.parquet"
                    )
                    teacher_manifests.append(
                        G1TeacherBankManifestRow(
                            outer_target=outer_target,
                            source_domain=source_domain,
                            row_count=teacher_bank.row_count,
                            parquet_sha256=teacher_bank.parquet_sha256,
                            records_sha256=teacher_bank.records_sha256,
                            manifest_sha256=teacher_bank.manifest_sha256,
                        )
                    )
            fixed_selections = tuple(
                fixed
                for selection in selections
                for fixed in selection.fixed_selections
            )
            stop_outcomes = materialize_g1_target_stop_outcomes(
                tuple(trajectories),
                tuple(learned_curves),
                tuple(reference_curves),
                fixed_selections,
            )
            curve_analysis = analyze_g1_target_curves(
                tuple(learned_curves),
                tuple(reference_curves),
                fixed_selections,
                no_target_leakage=True,
                deterministic_replay=True,
                deployment_bridge_valid=True,
            )
            stopping_analysis = analyze_g1_target_stopping(stop_outcomes)
            result = write_g1_formal_package(
                args.output
                or str(Path(args.project_root) / protocol.formal_output),
                tuple(selections),
                tuple(trajectories),
                tuple(learned_curves),
                tuple(reference_curves),
                stop_outcomes,
                curve_analysis,
                stopping_analysis,
                tuple(teacher_manifests),
                tuple(diagnostic_banks),
                tuple(decision_diagnostics),
                project_root=args.project_root,
                config_path=args.config,
            )
            _print_json(
                {
                    "status": result.status,
                    "output_tree_sha256": result.output_tree_sha256,
                    "manifest_sha256": result.manifest_sha256,
                    "output": args.output
                    or str(Path(args.project_root) / protocol.formal_output),
                }
            )
            return 0

        if args.command == "evaluate-target":
            protocol = load_g1_protocol(
                args.config,
                project_root=args.project_root,
            )
            runtime = load_g1_runtime(
                protocol,
                project_root=args.project_root,
                source_project_root=args.source_project_root,
                progress=_progress,
            )
            encoder = load_g1_encoder(
                args.source_project_root,
                device=args.device or protocol.default_device,
            )
            work_base = Path(args.project_root) / protocol.work_output
            trajectory_root = args.trajectory_root or str(
                work_base / "target_trajectories"
            )
            trajectory_bank, trajectories = read_g1_target_trajectory_bank(
                target_trajectory_bank_path(trajectory_root, args.outer_target)
            )
            formal_selection = read_g1_outer_formal_selection(
                outer_formal_selection_path(
                    args.formal_selection_root
                    or str(work_base / "formal_selection"),
                    args.outer_target,
                )
            )
            if (
                formal_selection.outer_target != args.outer_target
                or {row.action_selection_sha256 for row in trajectories}
                != {formal_selection.action_selection_sha256}
                or {row.action_model_sha256 for row in trajectories}
                != {formal_selection.action_model_sha256}
                or {row.stop_model_sha256 for row in trajectories}
                != {formal_selection.stop_model_sha256}
            ):
                raise G1ExecutionError(
                    "target trajectory differs from frozen formal selection"
                )
            seal = seal_g1_target_trajectory_bank(
                runtime,
                trajectory_bank,
                trajectories,
            )
            dependencies = build_g1_final_dependencies(
                runtime,
                protocol,
                outer_target=args.outer_target,
                encoder=encoder,
                progress=_progress,
            )
            dependency_shas = {row.final_dependency_sha256 for row in trajectories}
            if dependency_shas != {dependencies.state_sha256}:
                raise G1ExecutionError(
                    "target trajectory final dependency changed before evaluation"
                )
            curves = build_g1_outer_target_curve_bank(
                runtime,
                trajectory_bank,
                trajectories,
                seal,
                prior=dependencies.prior,
                assessor=dependencies.assessor,
                encoder=encoder,
                work_root=args.curve_root or str(work_base / "target_curves"),
                progress=_progress,
            )
            references = build_g1_outer_target_reference_bank(
                runtime,
                trajectory_bank,
                trajectories,
                seal,
                prior=dependencies.prior,
                assessor=dependencies.assessor,
                encoder=encoder,
                random_seed=protocol.teacher_bank_seed,
                work_root=args.reference_root
                or str(work_base / "target_references"),
                progress=_progress,
            )
            _print_json(
                {
                    "outer_target": args.outer_target,
                    "trajectory_bank_manifest_sha256": (
                        trajectory_bank.manifest_sha256
                    ),
                    "trajectory_bank_seal_sha256": seal.state_sha256,
                    "formal_selection_sha256": formal_selection.state_sha256,
                    "target_outcomes_opened": True,
                    "curve_bank_path": str(curves.path),
                    "curve_record_count": curves.record_count,
                    "curve_parquet_sha256": curves.bank.parquet_sha256,
                    "curve_manifest_sha256": curves.bank.manifest_sha256,
                    "reference_bank_path": str(references.path),
                    "reference_record_count": references.record_count,
                    "reference_parquet_sha256": references.bank.parquet_sha256,
                    "reference_manifest_sha256": references.bank.manifest_sha256,
                }
            )
            return 0

        if args.command == "build-target-trajectories":
            protocol = load_g1_protocol(
                args.config,
                project_root=args.project_root,
            )
            runtime = load_g1_runtime(
                protocol,
                project_root=args.project_root,
                source_project_root=args.source_project_root,
                progress=_progress,
            )
            encoder = load_g1_encoder(
                args.source_project_root,
                device=args.device or protocol.default_device,
            )
            work_base = Path(args.project_root) / protocol.work_output
            teacher_root = args.teacher_bank_root or str(
                Path(args.project_root) / protocol.teacher_bank_work_path
            )
            bridge_root = args.bridge_root or str(work_base / "source_bridges")
            action_selection = run_outer_dagger_selection(
                runtime,
                protocol,
                outer_target=args.outer_target,
                encoder=encoder,
                teacher_bank_root=teacher_root,
                bridge_root=bridge_root,
                supervised_root=args.supervised_root
                or str(work_base / "model_selection"),
                learned_root=args.learned_root
                or str(work_base / "learned_source_curves"),
                base_work_root=args.base_work_root
                or str(work_base / "engineering_selection"),
                dagger_bank_root=args.dagger_bank_root
                or str(work_base / "dagger_banks"),
                work_root=args.dagger_work_root
                or str(work_base / "dagger_selection"),
                device=args.device or protocol.default_device,
                progress=_progress,
            )
            stop_selection = run_outer_stop_selection(
                runtime,
                protocol,
                outer_target=args.outer_target,
                action_selection=action_selection,
                encoder=encoder,
                teacher_bank_root=teacher_root,
                stop_bank_root=args.stop_bank_root
                or str(work_base / "stop_banks"),
                fixed_endpoint_root=args.fixed_endpoint_root
                or str(work_base / "fixed_endpoints"),
                work_root=args.stop_selection_root
                or str(work_base / "stop_selection"),
                device=args.device or protocol.default_device,
                progress=_progress,
            )
            selected_dagger_iterations = (
                stop_selection.action_policy.hyperparameters.dagger_iterations
            )
            diagnostic_examples = tuple(
                rebind_training_example_modes(
                    row.example,
                    cai_context_mode=(
                        stop_selection.action_policy.hyperparameters.cai_context_mode
                    ),
                    task_token_mode=(
                        stop_selection.action_policy.hyperparameters.task_token_mode
                    ),
                )
                for row in stop_selection.action_selection.dagger_build.records
                if row.example.dagger_iteration <= selected_dagger_iterations
            )
            diagnostic_rows = materialize_g1_source_decision_diagnostics(
                diagnostic_examples,
                stop_selection.action_policy,
            )
            diagnostic_bank = write_g1_source_decision_diagnostic_bank(
                source_decision_diagnostic_bank_path(
                    args.decision_diagnostic_root
                    or str(work_base / "source_decision_diagnostics"),
                    args.outer_target,
                ),
                diagnostic_rows,
            )
            source_domains = tuple(
                domain
                for domain in protocol.domain_order
                if domain != args.outer_target
            )
            bridge_records = []
            for source_domain in source_domains:
                _bridge_bank, source_rows = read_source_bridge_bank(
                    source_bridge_bank_path(
                        bridge_root,
                        args.outer_target,
                        source_domain,
                    )
                )
                bridge_records.extend(source_rows)
            formal_selection = freeze_g1_outer_formal_selection(
                tuple(bridge_records),
                stop_selection.thresholds,
                outer_target=args.outer_target,
                action_selection_sha256=stop_selection.state_sha256,
                action_model_sha256=(
                    stop_selection.action_policy.model_state_sha256
                ),
                stop_model_sha256=stop_selection.stop_policy.model_state_sha256,
                decision_diagnostic_manifest_sha256=(
                    diagnostic_bank.manifest_sha256
                ),
                path=outer_formal_selection_path(
                    args.formal_selection_root
                    or str(work_base / "formal_selection"),
                    args.outer_target,
                ),
            )
            dependencies = build_g1_final_dependencies(
                runtime,
                protocol,
                outer_target=args.outer_target,
                encoder=encoder,
                progress=_progress,
            )
            result = build_g1_outer_target_trajectory_bank(
                runtime,
                protocol,
                dependencies,
                stop_selection,
                encoder=encoder,
                work_root=args.work_root
                or str(work_base / "target_trajectories"),
                progress=_progress,
            )
            _print_json(
                {
                    "outer_target": result.outer_target,
                    "specimen_count": result.specimen_count,
                    "record_count": result.record_count,
                    "target_outcomes_opened": result.target_outcomes_opened,
                    "final_dependency_sha256": result.final_dependency_sha256,
                    "outer_selection_sha256": result.outer_selection_sha256,
                    "formal_selection_sha256": formal_selection.state_sha256,
                    "decision_diagnostic_manifest_sha256": (
                        diagnostic_bank.manifest_sha256
                    ),
                    "bank_path": str(result.path),
                    "parquet_sha256": result.bank.parquet_sha256,
                    "records_sha256": result.bank.records_sha256,
                    "manifest_sha256": result.bank.manifest_sha256,
                    "state_sha256": result.state_sha256,
                }
            )
            return 0

        if args.command == "select-outer-stop":
            protocol = load_g1_protocol(
                args.config,
                project_root=args.project_root,
            )
            runtime = load_g1_runtime(
                protocol,
                project_root=args.project_root,
                source_project_root=args.source_project_root,
                progress=_progress,
            )
            encoder = load_g1_encoder(
                args.source_project_root,
                device=args.device or protocol.default_device,
            )
            work_base = Path(args.project_root) / protocol.work_output
            teacher_root = args.teacher_bank_root or str(
                Path(args.project_root) / protocol.teacher_bank_work_path
            )
            action_selection = run_outer_dagger_selection(
                runtime,
                protocol,
                outer_target=args.outer_target,
                encoder=encoder,
                teacher_bank_root=teacher_root,
                bridge_root=args.bridge_root
                or str(work_base / "source_bridges"),
                supervised_root=args.supervised_root
                or str(work_base / "model_selection"),
                learned_root=args.learned_root
                or str(work_base / "learned_source_curves"),
                base_work_root=args.base_work_root
                or str(work_base / "engineering_selection"),
                dagger_bank_root=args.dagger_bank_root
                or str(work_base / "dagger_banks"),
                work_root=args.dagger_work_root
                or str(work_base / "dagger_selection"),
                device=args.device or protocol.default_device,
                progress=_progress,
            )
            result = run_outer_stop_selection(
                runtime,
                protocol,
                outer_target=args.outer_target,
                action_selection=action_selection,
                encoder=encoder,
                teacher_bank_root=teacher_root,
                stop_bank_root=args.stop_bank_root
                or str(work_base / "stop_banks"),
                fixed_endpoint_root=args.fixed_endpoint_root
                or str(work_base / "fixed_endpoints"),
                work_root=args.work_root or str(work_base / "stop_selection"),
                device=args.device or protocol.default_device,
                progress=_progress,
            )
            _print_json(
                {
                    "outer_target": result.outer_target,
                    "action_model_sha256": result.action_policy.model_state_sha256,
                    "stop_model_sha256": result.stop_policy.model_state_sha256,
                    "selected_stop_epochs": result.selected_stop_epochs,
                    "thresholds": {
                        row.task.value: {
                            "status": row.status,
                            "threshold": row.threshold,
                        }
                        for row in result.thresholds
                    },
                    "target_outcomes_opened": result.target_outcomes_opened,
                    "selection_path": str(result.path),
                    "state_sha256": result.state_sha256,
                }
            )
            return 0

        if args.command == "select-outer-dagger":
            protocol = load_g1_protocol(
                args.config,
                project_root=args.project_root,
            )
            runtime = load_g1_runtime(
                protocol,
                project_root=args.project_root,
                source_project_root=args.source_project_root,
                progress=_progress,
            )
            encoder = load_g1_encoder(
                args.source_project_root,
                device=args.device or protocol.default_device,
            )
            work_base = Path(args.project_root) / protocol.work_output
            result = run_outer_dagger_selection(
                runtime,
                protocol,
                outer_target=args.outer_target,
                encoder=encoder,
                teacher_bank_root=args.teacher_bank_root
                or str(Path(args.project_root) / protocol.teacher_bank_work_path),
                bridge_root=args.bridge_root or str(work_base / "source_bridges"),
                supervised_root=args.supervised_root
                or str(work_base / "model_selection"),
                learned_root=args.learned_root
                or str(work_base / "learned_source_curves"),
                base_work_root=args.base_work_root
                or str(work_base / "engineering_selection"),
                dagger_bank_root=args.dagger_bank_root
                or str(work_base / "dagger_banks"),
                work_root=args.work_root or str(work_base / "dagger_selection"),
                device=args.device or protocol.default_device,
                progress=_progress,
            )
            selected = next(
                row
                for row in result.candidates
                if row.candidate.hyperparameters.state_sha256
                == result.selection.selected_hyperparameters_sha256
            )
            _print_json(
                {
                    "outer_target": result.outer_target,
                    "selected_hyperparameters_sha256": (
                        result.selection.selected_hyperparameters_sha256
                    ),
                    "selected_dagger_iterations": (
                        selected.candidate.hyperparameters.dagger_iterations
                    ),
                    "aawr_status": result.aawr_authorization.status,
                    "aawr_authorized_tasks": [
                        task.value
                        for task in result.aawr_authorization.authorized_tasks
                    ],
                    "target_outcomes_opened": result.target_outcomes_opened,
                    "selection_path": str(result.path),
                }
            )
            return 0

        if args.command == "select-outer-engineering":
            protocol = load_g1_protocol(
                args.config,
                project_root=args.project_root,
            )
            runtime = load_g1_runtime(
                protocol,
                project_root=args.project_root,
                source_project_root=args.source_project_root,
                progress=_progress,
            )
            encoder = load_g1_encoder(
                args.source_project_root,
                device=args.device or protocol.default_device,
            )
            work_base = Path(args.project_root) / protocol.work_output
            result = run_outer_engineering_selection(
                runtime,
                protocol,
                outer_target=args.outer_target,
                encoder=encoder,
                teacher_bank_root=args.teacher_bank_root
                or str(Path(args.project_root) / protocol.teacher_bank_work_path),
                bridge_root=args.bridge_root
                or str(work_base / "source_bridges"),
                supervised_root=args.supervised_root
                or str(work_base / "model_selection"),
                learned_root=args.learned_root
                or str(work_base / "learned_source_curves"),
                work_root=args.work_root
                or str(work_base / "engineering_selection"),
                device=args.device or protocol.default_device,
                progress=_progress,
            )
            _print_json(
                {
                    "outer_target": result.outer_target,
                    "example_count": result.example_count,
                    "candidate_count": len(result.candidates),
                    "selected_hyperparameters_sha256": (
                        result.selection.selected_hyperparameters_sha256
                    ),
                    "final_refit_epochs": result.selection.final_refit_epochs,
                    "target_outcomes_opened": (
                        result.selection.target_outcomes_opened
                    ),
                    "selection_path": str(result.path),
                }
            )
            return 0

        if args.command in {"build-stop-bank", "build-all-stop-banks"}:
            protocol = load_g1_protocol(
                args.config,
                project_root=args.project_root,
            )
            runtime = load_g1_runtime(
                protocol,
                project_root=args.project_root,
                source_project_root=args.source_project_root,
                progress=_progress,
            )
            encoder = load_g1_encoder(
                args.source_project_root,
                device=args.device or protocol.default_device,
            )
            teacher_root = args.teacher_bank_root or str(
                Path(args.project_root) / protocol.teacher_bank_work_path
            )
            fixed_root = args.fixed_endpoint_root or str(
                Path(args.project_root)
                / protocol.work_output
                / "fixed_endpoints"
            )
            work_root = args.work_root or str(
                Path(args.project_root) / protocol.work_output / "stop_banks"
            )
            if args.command == "build-all-stop-banks":
                results = build_g1_all_source_stop_banks(
                    runtime,
                    protocol,
                    encoder=encoder,
                    teacher_bank_root=teacher_root,
                    fixed_endpoint_root=fixed_root,
                    work_root=work_root,
                    start_fold=args.start_fold,
                    progress=_progress,
                )
                _print_json(
                    {
                        "bank_count": len(results),
                        "banks": [
                            {
                                "bank_path": str(result.path),
                                "outer_target": result.outer_target,
                                "source_domain": result.source_domain,
                                "specimen_count": result.specimen_count,
                                "dependency_sha256": result.dependency_sha256,
                                "row_count": result.bank.row_count,
                                "parquet_sha256": result.bank.parquet_sha256,
                                "records_sha256": result.bank.records_sha256,
                                "manifest_sha256": result.bank.manifest_sha256,
                            }
                            for result in results
                        ],
                    }
                )
                return 0
            dependencies = build_g1_source_dependencies(
                runtime,
                protocol,
                outer_target=args.outer_target,
                labeled_domain=args.source_domain,
                encoder=encoder,
                progress=_progress,
            )
            result = build_g1_source_stop_bank(
                runtime,
                protocol,
                dependencies,
                encoder=encoder,
                teacher_bank_root=teacher_root,
                fixed_endpoint_root=fixed_root,
                work_root=work_root,
                progress=_progress,
            )
            _print_json(
                {
                    "bank_path": str(result.path),
                    "outer_target": result.outer_target,
                    "source_domain": result.source_domain,
                    "specimen_count": result.specimen_count,
                    "dependency_sha256": result.dependency_sha256,
                    "row_count": result.bank.row_count,
                    "parquet_sha256": result.bank.parquet_sha256,
                    "records_sha256": result.bank.records_sha256,
                    "manifest_sha256": result.bank.manifest_sha256,
                }
            )
            return 0

        if args.command == "select-outer":
            protocol = load_g1_protocol(
                args.config,
                project_root=args.project_root,
            )
            result = run_outer_supervised_selection(
                protocol,
                outer_target=args.outer_target,
                bank_root=args.bank_root
                or str(Path(args.project_root) / protocol.teacher_bank_work_path),
                work_root=args.work_root
                or str(
                    Path(args.project_root)
                    / protocol.work_output
                    / "model_selection"
                ),
                device=args.device or protocol.default_device,
                progress=_progress,
            )
            _print_json(
                {
                    "outer_target": result.outer_target,
                    "example_count": result.example_count,
                    "candidate_count": len(result.core_results)
                    + len(result.tuning_results),
                    "selected_hyperparameters_sha256": (
                        result.selection.selected_hyperparameters_sha256
                    ),
                    "equal_domain_mean_regret": (
                        result.selection.equal_domain_mean_regret
                    ),
                    "final_refit_epochs": result.selection.final_refit_epochs,
                    "selection_path": str(result.path),
                }
            )
            return 0

        if args.command in {
            "build-bank",
            "build-all-banks",
            "build-fixed-endpoints",
            "build-all-fixed-endpoints",
            "build-source-bridge",
            "build-all-source-bridges",
        }:
            protocol = load_g1_protocol(
                args.config,
                project_root=args.project_root,
            )
            runtime = load_g1_runtime(
                protocol,
                project_root=args.project_root,
                source_project_root=args.source_project_root,
                progress=_progress,
            )
            encoder = load_g1_encoder(
                args.source_project_root,
                device=args.device or protocol.default_device,
            )
            fixed_endpoints = args.command in {
                "build-fixed-endpoints",
                "build-all-fixed-endpoints",
            }
            source_bridges = args.command in {
                "build-source-bridge",
                "build-all-source-bridges",
            }
            work_root = args.work_root or str(
                Path(args.project_root)
                / (
                    Path(protocol.work_output) / "fixed_endpoints"
                    if fixed_endpoints
                    else (
                        Path(protocol.work_output) / "source_bridges"
                        if source_bridges
                        else Path(protocol.teacher_bank_work_path)
                    )
                )
            )
            if args.command == "build-all-source-bridges":
                results = build_g1_all_source_bridge_banks(
                    runtime,
                    protocol,
                    encoder=encoder,
                    work_root=work_root,
                    start_fold=args.start_fold,
                    end_fold=args.end_fold,
                    progress=_progress,
                )
                _print_json(
                    {
                        "bank_count": len(results),
                        "banks": [
                            {
                                "bank_path": str(result.path),
                                "outer_target": result.outer_target,
                                "source_domain": result.source_domain,
                                "specimen_count": result.specimen_count,
                                "dependency_sha256": result.dependency_sha256,
                                "row_count": result.bank.row_count,
                                "parquet_sha256": result.bank.parquet_sha256,
                                "records_sha256": result.bank.records_sha256,
                                "manifest_sha256": result.bank.manifest_sha256,
                            }
                            for result in results
                        ],
                    }
                )
                return 0
            if args.command == "build-all-fixed-endpoints":
                results = build_g1_all_source_fixed_endpoint_banks(
                    runtime,
                    protocol,
                    encoder=encoder,
                    work_root=work_root,
                    start_fold=args.start_fold,
                    progress=_progress,
                )
                _print_json(
                    {
                        "bank_count": len(results),
                        "banks": [
                            {
                                "bank_path": str(result.path),
                                "outer_target": result.outer_target,
                                "source_domain": result.source_domain,
                                "specimen_count": result.specimen_count,
                                "dependency_sha256": result.dependency_sha256,
                                "row_count": result.bank.row_count,
                                "parquet_sha256": result.bank.parquet_sha256,
                                "records_sha256": result.bank.records_sha256,
                                "manifest_sha256": result.bank.manifest_sha256,
                            }
                            for result in results
                        ],
                    }
                )
                return 0
            if args.command == "build-all-banks":
                results = build_g1_all_source_teacher_banks(
                    runtime,
                    protocol,
                    encoder=encoder,
                    work_root=work_root,
                    start_fold=args.start_fold,
                    progress=_progress,
                )
                _print_json(
                    {
                        "bank_count": len(results),
                        "banks": [
                            {
                                "bank_path": str(result.path),
                                "outer_target": result.outer_target,
                                "source_domain": result.source_domain,
                                "specimen_count": result.specimen_count,
                                "dependency_sha256": result.dependency_sha256,
                                "row_count": result.bank.row_count,
                                "parquet_sha256": result.bank.parquet_sha256,
                                "records_sha256": result.bank.records_sha256,
                                "manifest_sha256": result.bank.manifest_sha256,
                            }
                            for result in results
                        ],
                    }
                )
                return 0
            dependencies = build_g1_source_dependencies(
                runtime,
                protocol,
                outer_target=args.outer_target,
                labeled_domain=args.source_domain,
                encoder=encoder,
                progress=_progress,
            )
            if args.command == "build-fixed-endpoints":
                result = build_g1_source_fixed_endpoint_bank(
                    runtime,
                    protocol,
                    dependencies,
                    encoder=encoder,
                    work_root=work_root,
                    progress=_progress,
                )
                _print_json(
                    {
                        "bank_path": str(result.path),
                        "outer_target": result.outer_target,
                        "source_domain": result.source_domain,
                        "specimen_count": result.specimen_count,
                        "dependency_sha256": result.dependency_sha256,
                        "row_count": result.bank.row_count,
                        "parquet_sha256": result.bank.parquet_sha256,
                        "records_sha256": result.bank.records_sha256,
                        "manifest_sha256": result.bank.manifest_sha256,
                    }
                )
                return 0
            if args.command == "build-source-bridge":
                result = build_g1_source_bridge_bank(
                    runtime,
                    protocol,
                    dependencies,
                    encoder=encoder,
                    work_root=work_root,
                    progress=_progress,
                )
                _print_json(
                    {
                        "bank_path": str(result.path),
                        "outer_target": result.outer_target,
                        "source_domain": result.source_domain,
                        "specimen_count": result.specimen_count,
                        "dependency_sha256": result.dependency_sha256,
                        "row_count": result.bank.row_count,
                        "parquet_sha256": result.bank.parquet_sha256,
                        "records_sha256": result.bank.records_sha256,
                        "manifest_sha256": result.bank.manifest_sha256,
                    }
                )
                return 0
            result = build_g1_source_teacher_bank(
                runtime,
                protocol,
                dependencies,
                encoder=encoder,
                work_root=work_root,
                progress=_progress,
            )
            _print_json(
                {
                    "bank_path": str(result.path),
                    "outer_target": result.outer_target,
                    "source_domain": result.source_domain,
                    "specimen_count": result.specimen_count,
                    "dependency_sha256": result.dependency_sha256,
                    "row_count": result.bank.row_count,
                    "parquet_sha256": result.bank.parquet_sha256,
                    "records_sha256": result.bank.records_sha256,
                    "manifest_sha256": result.bank.manifest_sha256,
                }
            )
            return 0

        if args.command == "validate":
            result = validate_g1_package(
                args.path,
                project_root=args.project_root,
                config_path=args.config,
            )
            _print_json(
                {
                    "status": result.status,
                    "output_tree_sha256": result.output_tree_sha256,
                    "manifest_sha256": result.manifest_sha256,
                }
            )
            return 0

        result = compare_g1_packages(
            args.formal,
            args.replay,
            project_root=args.project_root,
            config_path=args.config,
        )
        _print_json(
            {
                "byte_identical": result.byte_identical,
                "package_sha256": result.package_sha256,
                "replay_sha256": result.replay_sha256,
            }
        )
        return 0
    except (
        G1ExecutionError,
        G1EngineeringSelectionExecutionError,
        G1SourceBridgeError,
        G1StopExecutionError,
        G1ArtifactError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
