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
    build_g1_all_source_bridge_banks,
    build_g1_all_source_fixed_endpoint_banks,
    build_g1_all_source_stop_banks,
    build_g1_all_source_teacher_banks,
    build_g1_source_bridge_bank,
    build_g1_source_dependencies,
    build_g1_source_fixed_endpoint_bank,
    build_g1_source_stop_bank,
    build_g1_source_teacher_bank,
    compare_g1_packages,
    load_g1_encoder,
    load_g1_protocol,
    load_g1_runtime,
    run_outer_dagger_selection,
    run_outer_engineering_selection,
    run_outer_supervised_selection,
    validate_g1_package,
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
