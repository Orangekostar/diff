#!/usr/bin/env python3
"""Execute, publish, or replay the conditionally authorized MSSS S2 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cmc_bbdm.msss.authority import load_authority
from cmc_bbdm.msss.protocol import load_protocol
from cmc_bbdm.msss.transfer_artifacts import (
    publish_s2_package,
    replay_s2_package,
    validate_s2_package,
)
from cmc_bbdm.msss.transfer_pipeline import run_s2_experiment

REGISTERED_CONFIG = PROJECT_ROOT / "paper_v3/configs/msss.yaml"


def execute_s2(config_path: Path, *, device: str, replay: bool):
    resolved = config_path.resolve(strict=True)
    if resolved != REGISTERED_CONFIG.resolve(strict=True):
        raise ValueError("MSSS S2 CLI requires the exact registered config")
    if device != "cuda":
        raise ValueError("MSSS S2 requires the registered cuda device")
    protocol = load_protocol(resolved, project_root=PROJECT_ROOT)
    if protocol.device != device:
        raise ValueError("CLI device differs from the registered MSSS config")
    s1_package = protocol.output_paths["s1_formal"]
    output = protocol.output_paths["s2_replay" if replay else "s2_formal"]
    if replay:
        return output, replay_s2_package(
            protocol.output_paths["s2_formal"],
            output,
            project_root=PROJECT_ROOT,
            config_path=resolved,
            s1_package=s1_package,
        )
    print("[MSSS S2] loading frozen cohort authority", file=sys.stderr, flush=True)
    authority = load_authority(protocol, project_root=PROJECT_ROOT)
    _authorized, run = run_s2_experiment(
        protocol,
        authority,
        s1_package=s1_package,
        project_root=PROJECT_ROOT,
        config_path=resolved,
        status_hook=lambda message: print(
            f"[MSSS S2] {message}", file=sys.stderr, flush=True
        ),
    )
    published = publish_s2_package(
        output,
        protocol=protocol,
        run=run,
        config_path=resolved,
        s1_package=s1_package,
    )
    validated = validate_s2_package(
        output,
        project_root=PROJECT_ROOT,
        config_path=resolved,
        s1_package=s1_package,
    )
    if published != validated:
        raise RuntimeError("post-publication S2 validation changed")
    return output, validated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the conditionally authorized MSSS S2 transfer experiment."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cuda",))
    parser.add_argument("--replay", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if sum(value == "--config" or value.startswith("--config=") for value in raw) != 1:
        parser.error("exactly one explicit --config is required")
    args = parser.parse_args(raw)
    config = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output, result = execute_s2(config, device=args.device, replay=args.replay)
    print(
        json.dumps(
            {
                "package_dir": str(output.relative_to(PROJECT_ROOT)),
                "gate_status": result.gate_status,
                "s1_scientific_digest": result.s1_scientific_digest,
                "scientific_digest": result.scientific_digest,
                "output_tree_sha256": result.output_tree_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
