#!/usr/bin/env python3
"""Run the bounded BC C-scan Path-B supplementary study."""

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

from cmc_bbdm.learned_cscan.bc_supplement import audit_supplement

DEFAULT_CONFIG = PROJECT_ROOT / "paper_v3/configs/bc_cscan_path_b_supplement.yaml"
DEFAULT_SOURCE_ROOT = Path("/home/ww/paper3/cmc_damage_inference")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    audit.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = audit_supplement(
        config_path=arguments.config,
        project_root=PROJECT_ROOT,
        source_root=arguments.source_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
