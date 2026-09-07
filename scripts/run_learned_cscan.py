#!/usr/bin/env python3
"""Run the same-perception learned C-scan study stages."""

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

from cmc_bbdm.learned_cscan.benchmark import (
    build_training_bank,
    evaluate_study,
    prepare_study,
    run_perception,
    run_representative_replay_audit,
    summarize_study,
    train_models,
    validate_models,
)

DEFAULT_CONFIG = PROJECT_ROOT / "paper_v3/configs/learned_cscan_same_perception.yaml"
DEFAULT_SOURCE_ROOT = Path("/home/ww/paper3/cmc_damage_inference")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare",
        "perception",
        "build-train-bank",
        "train",
        "validate",
        "replay-audit",
        "summarize",
    ):
        command = commands.add_parser(name)
        _add_common(command)
    evaluate = commands.add_parser("evaluate")
    _add_common(evaluate)
    evaluate.add_argument("--split", choices=("test",), required=True)
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    common = {
        "config_path": arguments.config,
        "project_root": PROJECT_ROOT,
        "source_root": arguments.source_root,
    }
    if arguments.command == "prepare":
        result = prepare_study(**common)
    elif arguments.command == "perception":
        result = run_perception(**common)
    elif arguments.command == "build-train-bank":
        result = build_training_bank(**common)
    elif arguments.command == "train":
        result = train_models(**common)
    elif arguments.command == "validate":
        result = validate_models(**common)
    elif arguments.command == "evaluate":
        result = evaluate_study(**common, split=arguments.split)
    elif arguments.command == "replay-audit":
        result = run_representative_replay_audit(**common)
    else:
        result = summarize_study(**common)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
