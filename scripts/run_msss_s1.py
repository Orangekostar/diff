#!/usr/bin/env python3
"""Execute, publish, or replay the registered MSSS S1 experiment."""

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

from cmc_bbdm.msss.artifacts import publish_s1_package, validate_s1_package
from cmc_bbdm.msss.authority import load_authority
from cmc_bbdm.msss.pipeline import run_s1_experiment
from cmc_bbdm.msss.protocol import load_protocol
from cmc_bbdm.msss.replay import replay_s1_package

REGISTERED_CONFIG = PROJECT_ROOT / "paper_v3/configs/msss.yaml"


def resolve_output_path(*, mode: str, replay: bool) -> Path:
    if mode == "smoke":
        if replay:
            raise ValueError("smoke replay is not registered")
        return PROJECT_ROOT / "results/msss/smoke/s1_scale_discovery"
    if mode != "full":
        raise ValueError("MSSS S1 mode must be full or smoke")
    if replay:
        return PROJECT_ROOT / "results/msss/replay/s1_scale_discovery"
    return PROJECT_ROOT / "results/msss/s1_scale_discovery"


def execute_s1(
    config_path: Path, *, mode: str, device: str, replay: bool
):
    resolved = config_path.resolve(strict=True)
    if resolved != REGISTERED_CONFIG.resolve(strict=True):
        raise ValueError("MSSS S1 CLI requires the exact registered config")
    if device != "cuda":
        raise ValueError("MSSS S1 requires the registered cuda device")
    protocol = load_protocol(resolved, project_root=PROJECT_ROOT)
    if protocol.device != device:
        raise ValueError("CLI device differs from the registered MSSS config")
    output = resolve_output_path(mode=mode, replay=replay)
    if replay:
        return output, replay_s1_package(
            protocol.output_paths["s1_formal"],
            output,
            project_root=PROJECT_ROOT,
            config_path=resolved,
        )
    print("[MSSS S1] loading frozen cohort authority", file=sys.stderr, flush=True)
    authority = load_authority(protocol, project_root=PROJECT_ROOT)
    execution = run_s1_experiment(
        protocol,
        authority,
        project_root=PROJECT_ROOT,
        mode=mode,
        status_hook=lambda message: print(
            f"[MSSS S1] {message}", file=sys.stderr, flush=True
        ),
    )
    validation = publish_s1_package(
        output,
        protocol=protocol,
        bank=execution.bank,
        run=execution.run,
        config_path=resolved,
        mode="smoke" if execution.test_only else "formal",
        test_only=execution.test_only,
    )
    post = validate_s1_package(
        output, project_root=PROJECT_ROOT, config_path=resolved
    )
    if validation != post:
        raise RuntimeError("post-publication S1 validation changed")
    return output, validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the registered MSSS S1 scale-discovery experiment."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("full", "smoke"))
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
    if args.mode == "smoke" and args.replay:
        parser.error("--replay is only registered for --mode full")
    output, result = execute_s1(
        config, mode=args.mode, device=args.device, replay=args.replay
    )
    print(
        json.dumps(
            {
                "package_dir": str(output.relative_to(PROJECT_ROOT)),
                "mode": args.mode,
                "test_only": result.test_only,
                "gate_status": result.gate_status,
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
