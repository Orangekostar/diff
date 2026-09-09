#!/usr/bin/env python3
"""Prepare, run, and summarize the pooled expert C-scan rescore."""

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

DEFAULT_CONFIG = (
    PROJECT_ROOT / "paper_v3/configs/bc_cscan_expert_pooled_rescore_v1.yaml"
)
SOURCE_CONFIG = PROJECT_ROOT / "paper_v3/configs/bc_cscan_frozen_process_analysis.yaml"
LEGACY_OUTPUT = PROJECT_ROOT / "results/bc_cscan_frozen_process_analysis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--export-root", type=Path, required=True)
    prepare.add_argument("--legacy-output-root", type=Path, default=LEGACY_OUTPUT)
    prepare.add_argument("--original-session-root", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument(
        "--references",
        type=Path,
        default=PROJECT_ROOT
        / "results/bc_cscan_expert_pooled_rescore/v1/inputs/references",
    )
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    summarize.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    summarize.add_argument("--legacy-output-root", type=Path, default=LEGACY_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    from cmc_bbdm.learned_cscan.expert_pooled_rescore import (
        prepare_expert_pool,
        run_expert_rescore,
        summarize_expert_rescore,
    )

    if arguments.command == "prepare":
        result = prepare_expert_pool(
            config_path=arguments.config,
            source_config_path=arguments.source_config,
            project_root=PROJECT_ROOT,
            source_root=arguments.source_root,
            export_root=arguments.export_root,
            legacy_output_root=arguments.legacy_output_root,
            original_session_root=arguments.original_session_root,
        )
    elif arguments.command == "run":
        result = run_expert_rescore(
            config_path=arguments.config,
            source_config_path=arguments.source_config,
            project_root=PROJECT_ROOT,
            source_root=arguments.source_root,
            references_path=arguments.references,
        )
    else:
        result = summarize_expert_rescore(
            config_path=arguments.config,
            source_config_path=arguments.source_config,
            project_root=PROJECT_ROOT,
            legacy_output_root=arguments.legacy_output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
