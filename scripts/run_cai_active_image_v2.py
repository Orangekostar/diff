#!/usr/bin/env python3
"""Run the frozen VLM-guided CAI active-image v2 stages."""

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

from cmc_bbdm.cai_active_image.pipeline import (
    encode_features,
    evaluate_study,
    load_registered_protocol,
    prepare_study,
    train_actors,
    train_predictors,
)
from cmc_bbdm.cai_active_image.reporting import finalize_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare",
        "encode",
        "train-predictors",
        "train-actors",
        "evaluate",
        "finalize",
        "all",
    ):
        command = commands.add_parser(name)
        command.add_argument("--device", default=None)
        command.add_argument("--source-root", type=Path, required=True)
    return parser


def _run(command: str, *, source_root: Path, device: str | None) -> dict[str, object]:
    common = {"project_root": PROJECT_ROOT}
    if command == "prepare":
        return prepare_study(**common, source_root=source_root)
    if command == "encode":
        return encode_features(**common, source_root=source_root, device=device)
    if command == "train-predictors":
        return train_predictors(**common, device=device)
    if command == "train-actors":
        return train_actors(**common, device=device)
    if command == "evaluate":
        return evaluate_study(**common, device=device)
    if command == "finalize":
        protocol = load_registered_protocol(PROJECT_ROOT)
        return finalize_study(
            protocol, project_root=PROJECT_ROOT, source_root=source_root
        )
    result: dict[str, object] = {}
    result["prepare"] = prepare_study(**common, source_root=source_root)
    result["encode"] = encode_features(
        **common, source_root=source_root, device=device
    )
    result["train_predictors"] = train_predictors(**common, device=device)
    result["train_actors"] = train_actors(**common, device=device)
    if result["train_actors"].get("status") == "WALL_CLOCK_LIMIT_REACHED":  # type: ignore[union-attr]
        return result
    result["evaluate"] = evaluate_study(**common, device=device)
    protocol = load_registered_protocol(PROJECT_ROOT)
    result["finalize"] = finalize_study(
        protocol, project_root=PROJECT_ROOT, source_root=source_root
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = _run(
        arguments.command,
        source_root=arguments.source_root,
        device=arguments.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
