#!/usr/bin/env python3
"""Export one frozen reviewed TEST case as paper-ready mask overlays."""

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

DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/bc_cscan_single_case_overlay_diagnostic/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Read-only cmc_damage_inference source root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for PNG panels and the machine-readable manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    from cmc_bbdm.learned_cscan.single_case_overlay import (
        export_single_case_overlay_diagnostic,
    )

    result = export_single_case_overlay_diagnostic(
        project_root=PROJECT_ROOT,
        source_root=arguments.source_root,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
