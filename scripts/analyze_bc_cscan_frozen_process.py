#!/usr/bin/env python3
"""Analyze frozen BC C-scan trajectories and finalize external evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import cmc_bbdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE = str(PROJECT_ROOT / "src/cmc_bbdm")
if LOCAL_PACKAGE not in cmc_bbdm.__path__:
    cmc_bbdm.__path__.append(LOCAL_PACKAGE)

DEFAULT_CONFIG = PROJECT_ROOT / "paper_v3/configs/bc_cscan_frozen_process_analysis.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    analyze.add_argument("--source-root", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    finalize.add_argument("--source-root", type=Path, required=True)
    finalize.add_argument("--references", type=Path)
    finalize.add_argument("--blind-reviews", type=Path)
    finalize.add_argument("--human-sessions", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    from cmc_bbdm.learned_cscan.frozen_process_analysis import (
        execute_frozen_analysis,
        execute_frozen_finalize,
    )

    if arguments.command == "analyze":
        result = execute_frozen_analysis(
            config_path=arguments.config,
            project_root=PROJECT_ROOT,
            source_root=arguments.source_root,
        )
    else:
        result = execute_frozen_finalize(
            config_path=arguments.config,
            project_root=PROJECT_ROOT,
            source_root=arguments.source_root,
            references_path=arguments.references,
            blind_reviews_path=arguments.blind_reviews,
            human_sessions_path=arguments.human_sessions,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
