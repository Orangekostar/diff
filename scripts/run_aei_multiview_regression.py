#!/usr/bin/env python3
"""Audit, run, and replay the registered AEI multi-view experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cmc_bbdm.aei_multiview_regression.artifacts import (
    ArtifactError,
    publish_formal_chain,
    replay_stage,
)
from cmc_bbdm.aei_multiview_regression.formal_outer import run_formal_chain
from cmc_bbdm.aei_multiview_regression.oof_predictions import load_authoritative_inputs
from cmc_bbdm.aei_multiview_regression.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "paper_v3/configs/aei_multiview_regression.yaml"


def _source(protocol, name: str) -> Path:
    matches = [item.path for item in protocol.sources if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"registered source is missing: {name}")
    return matches[0]


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"summary is not a mapping: {path}")
    return value


def command_audit(config: Path) -> int:
    protocol = load_protocol(config, project_root=ROOT)
    a0 = _load_json(_source(protocol, "a0_summary"))
    a2 = _load_json(_source(protocol, "a2_summary"))
    a5 = _load_json(_source(protocol, "a5_summary"))
    statuses = (a0.get("status"), a2.get("status"), a5.get("status"))
    expected = (
        "A0_BASELINE_PASS",
        "A2_PAIRED_FEATURES_PASS",
        "FACTORISATION_NO_GO",
    )
    if statuses != expected:
        raise ValueError("frozen AEI stage statuses changed")
    print(
        json.dumps(
            {
                "a0": statuses[0],
                "a2": statuses[1],
                "a5": statuses[2],
                "sources_verified": len(protocol.sources),
                "status": "MULTIVIEW_AUDIT_PASS",
            },
            sort_keys=True,
        )
    )
    return 0


def command_run(config: Path) -> int:
    protocol = load_protocol(config, project_root=ROOT)
    inputs = load_authoritative_inputs(protocol, project_root=ROOT)
    result = run_formal_chain(inputs, protocol=protocol)
    if (
        result.e4_status == "AUTHORIZED_NOT_RUN"
        or result.e5_status == "AUTHORIZED_NOT_RUN"
    ):
        raise RuntimeError(
            "an evidence gate authorized an unimplemented conditional branch"
        )
    paths = publish_formal_chain(result, protocol=protocol)
    for path in paths:
        replay_stage(path)
    print(
        json.dumps(
            {
                "e1": result.e1.gate_status,
                "e2": None if result.e2 is None else result.e2.gate_status,
                "e3": None if result.e3 is None else result.e3.gate_status,
                "e4": result.e4_status,
                "e5": result.e5_status,
                "published": [str(path.relative_to(ROOT)) for path in paths],
                "status": "MULTIVIEW_RUN_PASS",
            },
            sort_keys=True,
        )
    )
    return 0


def command_replay(config: Path, stage: str) -> int:
    protocol = load_protocol(config, project_root=ROOT)
    allowed = {
        "e1_audit": "E1",
        "e2_cooperative": "E2",
        "e3_complementarity": "E3",
        "e4_moe": "E4",
        "e5_transport": "E5",
    }
    replayed = replay_stage(protocol.output_root / stage)
    if replayed.stage != allowed[stage]:
        raise ArtifactError("replayed stage identity changed")
    print(
        json.dumps(
            {
                "stage": replayed.stage,
                "manifest_sha256": replayed.manifest_sha256,
                "status": "REPLAY_PASS",
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "run"):
        child = commands.add_parser(name)
        child.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    replay = commands.add_parser("replay")
    replay.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    replay.add_argument(
        "--stage",
        required=True,
        choices=(
            "e1_audit",
            "e2_cooperative",
            "e3_complementarity",
            "e4_moe",
            "e5_transport",
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "audit":
            return command_audit(arguments.config.resolve())
        if arguments.command == "run":
            return command_run(arguments.config.resolve())
        return command_replay(arguments.config.resolve(), arguments.stage)
    except (ArtifactError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
