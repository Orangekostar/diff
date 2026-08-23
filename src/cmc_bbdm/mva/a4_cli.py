"""Command-line orchestration for the registered MVA A4 workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .a4_artifacts import (
    aggregate_a4_shards,
    finalize_a4_package,
    validate_a4_package,
)
from .a4_config import load_a4_config
from .a4_figures import render_a4_figures
from .a4_replay import replay_a4_package
from .a4_report import render_a4_report


def prepare_a4_candidate_bank(*args, **kwargs):
    """Load the CUDA execution stack only for candidate-bank commands."""

    from .a4_execution import prepare_a4_candidate_bank as implementation

    return implementation(*args, **kwargs)


def run_a4_outer_worker(*args, **kwargs):
    """Load the CUDA execution stack only for outer-domain commands."""

    from .a4_execution import run_a4_outer_worker as implementation

    return implementation(*args, **kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen MVA A4 stages")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    candidate = commands.add_parser("candidate-bank")
    candidate.add_argument(
        "--initial-budget", required=True, type=float, choices=(0.015625, 0.03125)
    )
    candidate.add_argument("--device", required=True)

    domain = commands.add_parser("domain")
    domain.add_argument("--outer-domain", required=True)
    domain.add_argument("--device", required=True)

    commands.add_parser("aggregate")
    commands.add_parser("render")
    commands.add_parser("finalize")
    commands.add_parser("validate")
    commands.add_parser("replay")
    return parser


def _print(command: str, **values: object) -> None:
    print(json.dumps({"command": command, **values}, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one explicit A4 stage and emit one machine-readable result."""

    args = _parser().parse_args(argv)
    root = args.project_root.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    config = load_a4_config(config_path, project_root=root)
    work = root / "results/mva/.work/a4_aggregate"
    formal = root / config.output_dir
    replay = root / config.replay_dir

    if args.command == "candidate-bank":
        output = prepare_a4_candidate_bank(
            config_path,
            project_root=root,
            initial_budget=args.initial_budget,
            device=args.device,
        )
        _print(args.command, output=str(output))
    elif args.command == "domain":
        output = run_a4_outer_worker(
            config_path,
            project_root=root,
            outer_domain=args.outer_domain,
            device=args.device,
        )
        _print(args.command, output=str(output), outer_domain=args.outer_domain)
    elif args.command == "aggregate":
        output = aggregate_a4_shards(config_path, project_root=root)
        _print(args.command, output=str(output))
    elif args.command == "render":
        figures = render_a4_figures(
            work, config_path=config_path, project_root=root
        )
        report = render_a4_report(work, config_path=config_path, project_root=root)
        _print(args.command, figures=str(figures), report=str(report))
    elif args.command == "finalize":
        validation = finalize_a4_package(
            work,
            formal,
            project_root=root,
            config_path=config_path,
        )
        _print(
            args.command,
            output=str(formal),
            global_mask_status=validation.global_mask_status,
            a5_status=validation.a5_status,
            output_tree_sha256=validation.output_tree_sha256,
        )
    elif args.command == "validate":
        validation = validate_a4_package(
            formal, project_root=root, config_path=config_path
        )
        _print(
            args.command,
            output=str(formal),
            global_mask_status=validation.global_mask_status,
            a5_status=validation.a5_status,
            output_tree_sha256=validation.output_tree_sha256,
        )
    elif args.command == "replay":
        validation = replay_a4_package(
            formal,
            replay,
            project_root=root,
            config_path=config_path,
        )
        _print(
            args.command,
            output=str(replay),
            global_mask_status=validation.global_mask_status,
            a5_status=validation.a5_status,
            output_tree_sha256=validation.output_tree_sha256,
        )
    else:  # pragma: no cover - argparse owns command validation
        raise AssertionError("unreachable A4 command")
    return 0


__all__ = ["main"]
