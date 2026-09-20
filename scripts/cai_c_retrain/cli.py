"""Unified entry point for the C=P0+R1 retraining release."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cai_c_retrain.context import METHOD_SEEDS, PHASES, TaskContext


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (*PHASES, "all"):
        child = subparsers.add_parser(name)
        _add_config(child)
        if name == "train":
            child.add_argument("--resume", action="store_true")
            child.add_argument("--method", choices=tuple(METHOD_SEEDS))
    export = subparsers.add_parser("export-inputs")
    _add_config(export)
    export.add_argument("--specimen-key", required=True)
    return parser


def _call(module_name: str, function_name: str, *args, **kwargs):
    module = importlib.import_module(f"scripts.cai_c_retrain.{module_name}")
    return getattr(module, function_name)(*args, **kwargs)


def _run_all(context: TaskContext) -> dict[str, object]:
    lock_path = context.path("output") / "runtime_lock.json"
    results: dict[str, object] = {}
    state_path = getattr(
        context, "task_state_path", context.path("output") / "task_state.json"
    )
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"phases": {}}
    )
    prepare_complete = (
        state.get("phases", {}).get("prepare", {}).get("status") == "COMPLETE"
    )
    if not prepare_complete:
        _call("prepare", "prepare_stage", context)
        results["prepare"] = 0
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    mapping = {
        "prepare": "report_python",
        "vlm": "vlm_python",
        "train": "actor_python",
        "assemble": "report_python",
        "analyze": "report_python",
        "paper": "report_python",
        "verify": "report_python",
        "publish": "report_python",
    }
    cli = Path(__file__).resolve()
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"phases": {}}
    )
    for phase in PHASES:
        if phase == "prepare":
            results.setdefault(phase, "already complete")
            continue
        if state.get("phases", {}).get(phase, {}).get("status") == "COMPLETE":
            results[phase] = "already complete"
            continue
        interpreter = lock[mapping[phase]]
        command = [
            interpreter,
            str(cli),
            phase,
            "--config",
            str(context.config_path),
        ]
        try:
            completed = subprocess.run(command, cwd=context.root, check=True)
        except subprocess.CalledProcessError:
            if phase != "train":
                raise
            completed = subprocess.run(
                [*command, "--resume"], cwd=context.root, check=True
            )
        results[phase] = completed.returncode
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = TaskContext.load(args.config)
    if args.command == "prepare":
        result = _call("prepare", "prepare_stage", context)
    elif args.command == "vlm":
        result = _call("vlm", "vlm_stage", context)
    elif args.command == "train":
        result = _call(
            "train",
            "train_stage",
            context,
            resume=args.resume,
            method=args.method,
        )
    elif args.command == "assemble":
        result = _call("assemble", "assemble_stage", context)
    elif args.command == "analyze":
        result = _call("evidence", "analyze_stage", context)
    elif args.command == "paper":
        result = _call("paper", "paper_stage", context)
    elif args.command == "verify":
        result = _call("validate", "verify_stage", context)
    elif args.command == "publish":
        result = _call("validate", "publish_stage", context)
    elif args.command == "export-inputs":
        result = _call("vlm", "export_inputs", context, args.specimen_key)
    else:
        result = _run_all(context)
    print(json.dumps(result, indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
