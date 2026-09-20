"""Unified entry point for frozen Actor C0 mechanism diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cai_actor_c0_diagnostic.context import PHASES, TaskContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (*PHASES, "status", "all"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, type=Path)
    return parser


def _prepare(context: TaskContext):
    from scripts.cai_actor_c0_diagnostic.prepare import prepare_stage
    return prepare_stage(context)


def _replay(context: TaskContext):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from scripts.cai_actor_c0_diagnostic.replay import replay_stage
    return replay_stage(context, device="cuda:0")


def _diagnose(context: TaskContext):
    from scripts.cai_actor_c0_diagnostic.diagnose import diagnose_stage
    return diagnose_stage(context, device="cpu")


def _render(context: TaskContext):
    from scripts.cai_actor_c0_diagnostic.render import render_stage
    return render_stage(context)


def _verify(context: TaskContext):
    from scripts.cai_actor_c0_diagnostic.validate import verify_stage
    return verify_stage(context)


def _publish(context: TaskContext):
    from scripts.cai_actor_c0_diagnostic.validate import publish_stage
    return publish_stage(context)


def _status(context: TaskContext):
    state = json.loads(context.state_path.read_text(encoding="utf-8")) if context.state_path.exists() else {"phases": {}}
    usage_path = context.path("output") / "resource_usage.json"
    usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else {}
    return {"task_id": context.scope["task_id"], "phases": state.get("phases", {}), "resources": usage}


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    arguments = build_parser().parse_args(argv)
    context = TaskContext.load(arguments.config)
    actions = {
        "prepare": _prepare,
        "replay": _replay,
        "diagnose": _diagnose,
        "render": _render,
        "verify": _verify,
        "publish": _publish,
        "status": _status,
    }
    if arguments.command == "all":
        result = {}
        for phase in PHASES:
            result[phase] = actions[phase](context)
    else:
        result = actions[arguments.command](context)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
