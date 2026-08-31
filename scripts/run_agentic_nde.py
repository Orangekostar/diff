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
        summary = replay_p0(
            args.path,
            surface_root=args.surface_root,
            project_root=_PROJECT_ROOT if args.surface_root else None,
        )
        print(summary["status"])
        return 0
    except (ArtifactError, PipelineError, OSError, ValueError) as error:
        print(f"agentic NDE integrity error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
