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
    G1ExecutionError,
    build_g1_all_source_teacher_banks,
    build_g1_source_dependencies,
    build_g1_source_teacher_bank,
    compare_g1_packages,
    load_g1_encoder,
    load_g1_protocol,
    load_g1_runtime,
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
        if args.command in {"build-bank", "build-all-banks"}:
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
            work_root = args.work_root or str(
                Path(args.project_root) / protocol.teacher_bank_work_path
            )
            if args.command == "build-all-banks":
                results = build_g1_all_source_teacher_banks(
                    runtime,
                    protocol,
                    encoder=encoder,
                    work_root=work_root,
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
        G1ArtifactError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
