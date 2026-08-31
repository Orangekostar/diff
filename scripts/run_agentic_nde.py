"""Command-line entry point for stage-gated agentic NDE research."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cmc_bbdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PACKAGE = str(_PROJECT_ROOT / "src/cmc_bbdm")
if _LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.append(_LOCAL_PACKAGE)

from cmc_bbdm.agentic_nde.artifacts import ArtifactError, replay_p0
from cmc_bbdm.agentic_nde.p0 import PipelineError, audit_p0
from cmc_bbdm.agentic_nde.p0r import P0RPipelineError, audit_p0r
from cmc_bbdm.agentic_nde.p0r_artifacts import (
    P0RArtifactError,
    replay_p0r_package,
)
from cmc_bbdm.agentic_nde.p0r_qc import P0RQCError, render_p0r_qc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_agentic_nde.py")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit-p0")
    audit.add_argument("--config", required=True)
    audit.add_argument("--surface-root", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--project-root", default=str(_PROJECT_ROOT))
    replay = commands.add_parser("replay-p0")
    replay.add_argument("--path", required=True)
    replay.add_argument("--surface-root")
    audit_p0r_parser = commands.add_parser("audit-p0r")
    audit_p0r_parser.add_argument("--config", required=True)
    audit_p0r_parser.add_argument("--surface-root", required=True)
    audit_p0r_parser.add_argument("--output", required=True)
    audit_p0r_parser.add_argument("--project-root", default=str(_PROJECT_ROOT))
    audit_p0r_parser.add_argument("--qc-output")
    replay_p0r_parser = commands.add_parser("replay-p0r")
    replay_p0r_parser.add_argument("--path", required=True)
    replay_p0r_parser.add_argument("--surface-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "audit-p0":
            result = audit_p0(
                config_path=args.config,
                surface_root=args.surface_root,
                output=args.output,
                project_root=args.project_root,
            )
            print(result)
            return 0
        if args.command == "replay-p0":
            summary = replay_p0(
                args.path,
                surface_root=args.surface_root,
                project_root=_PROJECT_ROOT if args.surface_root else None,
            )
        elif args.command == "audit-p0r":
            result = audit_p0r(
                config_path=args.config,
                surface_root=args.surface_root,
                output=args.output,
                project_root=args.project_root,
            )
            if args.qc_output:
                render_p0r_qc(
                    package=result,
                    surface_root=args.surface_root,
                    output=args.qc_output,
                )
            summary = replay_p0r_package(result)
        else:
            replay_kwargs = {}
            if args.surface_root:
                replay_kwargs = {
                    "surface_root": args.surface_root,
                    "project_root": _PROJECT_ROOT,
                }
            summary = replay_p0r_package(args.path, **replay_kwargs)
        print(summary["status"])
        return 0
    except (
        ArtifactError,
        PipelineError,
        P0RArtifactError,
        P0RPipelineError,
        P0RQCError,
        OSError,
        ValueError,
    ) as error:
        print(f"agentic NDE integrity error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
