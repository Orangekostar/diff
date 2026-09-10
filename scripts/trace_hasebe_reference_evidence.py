#!/usr/bin/env python3
"""Trace bounded Hasebe reference evidence into versioned artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cmc_bbdm

LOCAL_PACKAGE = str(ROOT / "src/cmc_bbdm")
if LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.append(LOCAL_PACKAGE)

from cmc_bbdm.learned_cscan.hasebe_reference_evidence import (
    run_compare,
    run_extract,
    run_inventory,
    write_summary_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "extract", "compare"):
        command = commands.add_parser(name)
        command.add_argument("--project-root", type=Path, default=ROOT)
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "compare":
            command.add_argument("--artifact-root", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--output-root", type=Path, required=True)
    summary.add_argument("--artifact-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summarize":
        write_summary_artifacts(args.output_root, args.artifact_root)
        return 0
    if args.command == "inventory":
        run_inventory(args.project_root, args.source_root, args.output_root)
    elif args.command == "extract":
        run_extract(args.project_root, args.source_root, args.output_root)
    elif args.command == "compare":
        run_compare(
            args.project_root,
            args.source_root,
            args.output_root,
            args.artifact_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
