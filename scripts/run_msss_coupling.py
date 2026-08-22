#!/usr/bin/env python3
"""Execute or replay the registered MSSS S1 NO-GO coupling diagnostic."""

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

from cmc_bbdm.msss.coupling_artifacts import (
    build_coupling_diagnostic,
    load_coupling_protocol,
    publish_coupling_package,
    replay_coupling_package,
    validate_coupling_package,
)

REGISTERED_CONFIG = Path("paper_v3/configs/msss_no_go_coupling.yaml")


def execute_coupling(
    config_path: Path, *, project_root: Path, replay: bool
):
    root = project_root.resolve(strict=True)
    config = config_path if config_path.is_absolute() else root / config_path
    config = config.resolve(strict=True)
    registered = (root / REGISTERED_CONFIG).resolve(strict=True)
    if config != registered:
        raise ValueError("coupling CLI requires the exact registered config")
    protocol = load_coupling_protocol(config, project_root=root)
    if replay:
        output = protocol.output_replay
        result = replay_coupling_package(
            protocol.output_formal,
            output,
            protocol=protocol,
            project_root=root,
            config_path=config,
        )
        return output, result
    print("[MSSS coupling] loading frozen parent and authorities", file=sys.stderr, flush=True)
    diagnostic = build_coupling_diagnostic(protocol, project_root=root)
    output = protocol.output_formal
    published = publish_coupling_package(
        output,
        protocol=protocol,
        diagnostic=diagnostic,
        config_path=config,
    )
    validated = validate_coupling_package(
        output, protocol=protocol, config_path=config
    )
    if published != validated:
        raise RuntimeError("post-publication coupling validation changed")
    return output, validated


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the registered post-S1-NO-GO coupling diagnostic."
    )
    parser.add_argument("--config", type=Path, default=REGISTERED_CONFIG)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--replay", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    output, result = execute_coupling(
        args.config, project_root=args.project_root, replay=args.replay
    )
    root = args.project_root.resolve(strict=True)
    print(
        json.dumps(
            {
                "package_dir": str(output.relative_to(root)),
                "coupling_status": result.coupling_status,
                "validation_status": result.validation_status,
                "s2_status": result.s2_status,
                "diagnostic_state_sha256": result.diagnostic_state_sha256,
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
