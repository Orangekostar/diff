"""Command-line entry point for MVA A0-A3."""

from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate import aggregate_a2
from .artifacts import publish_mva_manifest, validate_mva_package
from .config import load_mva_config
from .figures import render_mva_figures
from .oracle_execution import (
    prepare_uniform_bank,
    run_domain_worker,
    run_low_checkpoint_worker,
)
from .pipeline import run_a0_a1
from .replay import replay_mva_package
from .stability import run_stability_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mva-a0-a3")
    parser.add_argument(
        "command",
        choices=(
            "a0-a1",
            "prepare-a2",
            "domain",
            "low-domain",
            "stability-domain",
            "aggregate",
            "figures",
            "finalize",
            "validate",
            "replay",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initial-budget", type=float)
    parser.add_argument("--outer-domain")
    parser.add_argument("--max-specimens", type=int)
    parser.add_argument("--random-seed-count", type=int, default=100)
    parser.add_argument("--source")
    parser.add_argument("--destination")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "a0-a1":
        run_a0_a1(
            Path(arguments.config),
            project_root=Path(arguments.project_root),
            device=arguments.device,
        )
    elif arguments.command == "prepare-a2":
        if arguments.initial_budget is None:
            raise ValueError("--initial-budget is required")
        prepare_uniform_bank(
            Path(arguments.config),
            project_root=Path(arguments.project_root),
            initial_budget=arguments.initial_budget,
            device=arguments.device,
        )
    elif arguments.command == "domain":
        if arguments.outer_domain is None:
            raise ValueError("--outer-domain is required")
        run_domain_worker(
            Path(arguments.config),
            project_root=Path(arguments.project_root),
            outer_domain=arguments.outer_domain,
            device=arguments.device,
            max_specimens=arguments.max_specimens,
            random_seed_count=arguments.random_seed_count,
        )
    elif arguments.command == "low-domain":
        if arguments.outer_domain is None:
            raise ValueError("--outer-domain is required")
        run_low_checkpoint_worker(
            Path(arguments.config),
            project_root=Path(arguments.project_root),
            outer_domain=arguments.outer_domain,
            device=arguments.device,
            max_specimens=arguments.max_specimens,
            random_seed_count=arguments.random_seed_count,
        )
    elif arguments.command == "stability-domain":
        if arguments.outer_domain is None:
            raise ValueError("--outer-domain is required")
        run_stability_domain(
            Path(arguments.config),
            project_root=Path(arguments.project_root),
            outer_domain=arguments.outer_domain,
            device=arguments.device,
            max_specimens=arguments.max_specimens,
        )
    elif arguments.command == "aggregate":
        aggregate_a2(Path(arguments.config), project_root=Path(arguments.project_root))
    elif arguments.command == "figures":
        render_mva_figures(
            Path(arguments.config), project_root=Path(arguments.project_root)
        )
    elif arguments.command == "finalize":
        root = Path(arguments.project_root).resolve(strict=True)
        config = load_mva_config(Path(arguments.config), project_root=root)
        publish_mva_manifest(
            root / config.output_dir / "a2_oracle_value",
            project_root=root,
            config_path=Path(arguments.config),
        )
    elif arguments.command == "validate":
        root = Path(arguments.project_root).resolve(strict=True)
        config = load_mva_config(Path(arguments.config), project_root=root)
        validate_mva_package(
            root / config.output_dir / "a2_oracle_value",
            project_root=root,
            config_path=Path(arguments.config),
        )
    elif arguments.command == "replay":
        if arguments.source is None or arguments.destination is None:
            raise ValueError("--source and --destination are required")
        replay_mva_package(
            Path(arguments.source),
            Path(arguments.destination),
            project_root=Path(arguments.project_root),
            config_path=Path(arguments.config),
        )
    return 0


__all__ = ["build_parser", "main"]
