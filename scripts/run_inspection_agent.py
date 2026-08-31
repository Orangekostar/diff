"""Command-line entry point for the inspection agent G0 audit."""

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

from cmc_bbdm.inspection_agent.artifacts import (
    InspectionArtifactError,
    compare_g0_packages,
    validate_g0_package,
)
from cmc_bbdm.inspection_agent.g0 import (
    G0ExecutionError,
    run_g0,
    smoke_g0_assessor,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_inspection_agent.py")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--source-project-root", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--project-root", default=str(_PROJECT_ROOT))
    run.add_argument("--device", default=None)

    smoke = commands.add_parser("smoke-assessor")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--source-project-root", required=True)
    smoke.add_argument("--project-root", default=str(_PROJECT_ROOT))
    smoke.add_argument("--device", default=None)

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
        if args.command == "run":
            result = run_g0(
                args.config,
                project_root=args.project_root,
                source_project_root=args.source_project_root,
                output_dir=args.output,
                device=args.device,
                progress=_progress,
            )
            _print_json(
                {
                    "output_dir": str(result.output_dir),
                    "status": result.status,
                    "cai_assessor_authorized": result.cai_assessor_authorized,
                    "specimen_count": result.specimen_count,
                    "output_tree_sha256": result.package.output_tree_sha256,
                    "manifest_sha256": result.package.manifest_sha256,
                }
            )
            return 0

        if args.command == "smoke-assessor":
            result = smoke_g0_assessor(
                args.config,
                project_root=args.project_root,
                source_project_root=args.source_project_root,
                device=args.device,
                progress=_progress,
            )
            _print_json(result)
            return 0

        if args.command == "validate":
            result = validate_g0_package(
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

        result = compare_g0_packages(
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
        G0ExecutionError,
        InspectionArtifactError,
        OSError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
