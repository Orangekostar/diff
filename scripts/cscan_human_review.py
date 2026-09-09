#!/usr/bin/env python3
"""Build and validate the local C-scan human-review tool."""

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

DEFAULT_CONFIG = PROJECT_ROOT / "paper_v3/configs/cscan_human_review_tool.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_ui = commands.add_parser("build-ui")
    build_ui.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build_ui.add_argument("--output", type=Path)

    references = commands.add_parser("prepare-references")
    references.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    references.add_argument("--source-root", type=Path, required=True)
    references.add_argument("--packet-root", type=Path)
    references.add_argument("--result-root", type=Path)
    references.add_argument("--dist-root", type=Path)

    blind = commands.add_parser("prepare-blind")
    blind.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    blind.add_argument("--source-root", type=Path, required=True)
    blind.add_argument("--packet-root", type=Path)
    blind.add_argument("--private-index-root", type=Path)
    blind.add_argument("--result-root", type=Path)
    blind.add_argument("--cache-root", type=Path)

    validate = commands.add_parser("validate-return")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate.add_argument("--session", type=Path, required=True)
    validate.add_argument("--packet", type=Path, required=True)
    validate.add_argument("--private-index", type=Path)

    export_references = commands.add_parser("export-references")
    export_references.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    export_references.add_argument("--session", type=Path, required=True)
    export_references.add_argument("--packet", type=Path, required=True)
    export_references.add_argument("--output-root", type=Path, required=True)

    export_blind = commands.add_parser("export-blind-reviews")
    export_blind.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    export_blind.add_argument("--session", type=Path, required=True)
    export_blind.add_argument("--packet", type=Path, required=True)
    export_blind.add_argument("--private-index", type=Path, required=True)
    export_blind.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    from cmc_bbdm.learned_cscan.human_review_tool import (
        build_delivery_ui,
        export_blind_review_session,
        export_reference_session,
        load_human_review_config,
        prepare_blind_packets,
        prepare_reference_packets,
        validate_return_session,
    )

    config = load_human_review_config(arguments.config, project_root=PROJECT_ROOT)
    if arguments.command == "build-ui":
        result = build_delivery_ui(config, output_path=arguments.output)
    elif arguments.command == "prepare-references":
        result = prepare_reference_packets(
            config,
            source_root=arguments.source_root,
            packet_root=arguments.packet_root,
            result_root=arguments.result_root,
            dist_root=arguments.dist_root,
        )
    elif arguments.command == "prepare-blind":
        result = prepare_blind_packets(
            config,
            source_root=arguments.source_root,
            packet_root=arguments.packet_root,
            private_index_root=arguments.private_index_root,
            result_root=arguments.result_root,
            cache_root=arguments.cache_root,
        )
    elif arguments.command == "validate-return":
        session = json.loads(arguments.session.read_text(encoding="utf-8"))
        packet = json.loads(arguments.packet.read_text(encoding="utf-8"))
        private_index = (
            None
            if arguments.private_index is None
            else json.loads(arguments.private_index.read_text(encoding="utf-8"))
        )
        result = validate_return_session(
            session=session, packet=packet, private_index=private_index
        )
    elif arguments.command == "export-references":
        result = export_reference_session(
            session=json.loads(arguments.session.read_text(encoding="utf-8")),
            packet=json.loads(arguments.packet.read_text(encoding="utf-8")),
            output_root=arguments.output_root,
        )
    elif arguments.command == "export-blind-reviews":
        result = export_blind_review_session(
            session=json.loads(arguments.session.read_text(encoding="utf-8")),
            packet=json.loads(arguments.packet.read_text(encoding="utf-8")),
            private_index=json.loads(
                arguments.private_index.read_text(encoding="utf-8")
            ),
            output_root=arguments.output_root,
        )
    else:
        raise AssertionError("unreachable command")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
