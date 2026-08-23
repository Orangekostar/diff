#!/usr/bin/env python3
"""Run one or all formal MVA A5 outer-domain workers."""

from __future__ import annotations

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import json
from pathlib import Path

from cmc_bbdm.mva.a5_artifacts import (
    aggregate_a5_shards,
    finalize_a5_package,
    validate_a5_package,
)
from cmc_bbdm.mva.a5_config import load_a5_config
from cmc_bbdm.mva.a5_execution import run_a5_outer_worker
from cmc_bbdm.mva.a5_figures import render_a5_figures
from cmc_bbdm.mva.a5_replay import replay_a5_package
from cmc_bbdm.mva.a5_report import render_a5_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="paper_v3/configs/mva_a5_imitation_policy.yaml"
    )
    parser.add_argument("--project-root", default=".")
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--outer-domain")
    commands.add_argument("--all", action="store_true")
    commands.add_argument("--aggregate", action="store_true")
    commands.add_argument("--render", action="store_true")
    commands.add_argument("--finalize", action="store_true")
    commands.add_argument("--validate", action="store_true")
    commands.add_argument("--replay", action="store_true")
    parser.add_argument("--device", default="cuda:2")
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve(strict=True)
    config_path = (root / arguments.config).resolve(strict=True)
    config = load_a5_config(config_path, project_root=root)
    work = root / config.work_dir / "aggregate"
    formal = root / config.output_dir
    replay = root / config.replay_dir
    if arguments.all or arguments.outer_domain:
        domains = config.domain_order if arguments.all else (arguments.outer_domain,)
        for domain in domains:
            path = run_a5_outer_worker(
                config_path,
                project_root=root,
                outer_domain=domain,
                device=arguments.device,
            )
            print(path, flush=True)
    elif arguments.aggregate:
        print(aggregate_a5_shards(config_path, project_root=root), flush=True)
    elif arguments.render:
        figures = render_a5_figures(
            work, config_path=config_path, project_root=root
        )
        report = render_a5_report(
            work, config_path=config_path, project_root=root
        )
        print(json.dumps({"figures": str(figures), "report": str(report)}))
    elif arguments.finalize:
        result = finalize_a5_package(
            work, formal, project_root=root, config_path=config_path
        )
        print(json.dumps({"output": str(formal), "tree": result.output_tree_sha256}))
    elif arguments.validate:
        result = validate_a5_package(
            formal, project_root=root, config_path=config_path
        )
        print(json.dumps({"output": str(formal), "tree": result.output_tree_sha256}))
    elif arguments.replay:
        result = replay_a5_package(
            formal,
            replay,
            project_root=root,
            config_path=config_path,
        )
        print(json.dumps({"output": str(replay), "tree": result.output_tree_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
