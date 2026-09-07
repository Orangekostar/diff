#!/usr/bin/env python3
"""Run the VLM C-scan success-efficiency benchmark stages."""

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

from cmc_bbdm.vlm_cscan.artifacts import (
    evaluate_benchmark,
    export_annotation_queue,
    infer_surface_plans,
    prepare_benchmark,
    run_benchmark,
    verify_benchmark,
)
from cmc_bbdm.vlm_cscan.runtime import load_benchmark_config

DEFAULT_CONFIG = PROJECT_ROOT / "paper_v3/configs/vlm_cscan_efficiency.yaml"
DEFAULT_SOURCE_ROOT = Path("/home/ww/paper3/cmc_damage_inference")
COHORT_CHOICES = ("smoke", "pilot", "all")


def _path_from_project_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def _add_source_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)


def _add_cohort(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cohort", choices=COHORT_CHOICES, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    _add_config(prepare)
    _add_source_root(prepare)

    export_annotations = commands.add_parser("export-annotations")
    _add_config(export_annotations)
    _add_source_root(export_annotations)

    infer_surface = commands.add_parser("infer-surface")
    _add_config(infer_surface)
    _add_source_root(infer_surface)
    _add_cohort(infer_surface)

    run = commands.add_parser("run")
    _add_config(run)
    _add_source_root(run)
    _add_cohort(run)
    run.add_argument("--workers", type=int, default=4)

    evaluate = commands.add_parser("evaluate")
    _add_config(evaluate)
    _add_source_root(evaluate)
    _add_cohort(evaluate)

    verify = commands.add_parser("verify")
    _add_config(verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = _path_from_project_root(arguments.config)

    if arguments.command == "prepare":
        result = prepare_benchmark(
            config,
            project_root=PROJECT_ROOT,
            source_root=_path_from_project_root(arguments.source_root),
        )
    elif arguments.command == "export-annotations":
        result = export_annotation_queue(
            config,
            project_root=PROJECT_ROOT,
            source_root=_path_from_project_root(arguments.source_root),
        )
    elif arguments.command == "infer-surface":
        result = infer_surface_plans(
            config,
            project_root=PROJECT_ROOT,
            source_root=_path_from_project_root(arguments.source_root),
            cohort=arguments.cohort,
        )
    elif arguments.command == "run":
        result = run_benchmark(
            config,
            project_root=PROJECT_ROOT,
            source_root=_path_from_project_root(arguments.source_root),
            cohort=arguments.cohort,
            workers=arguments.workers,
        )
    elif arguments.command == "evaluate":
        result = evaluate_benchmark(
            config,
            project_root=PROJECT_ROOT,
            source_root=_path_from_project_root(arguments.source_root),
            cohort=arguments.cohort,
        )
    else:
        benchmark_config = load_benchmark_config(config, project_root=PROJECT_ROOT)
        result = verify_benchmark(benchmark_config)

    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
