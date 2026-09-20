from __future__ import annotations

import json
from pathlib import Path

import pytest


def _scope() -> dict[str, object]:
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/cai/c_render_retrain/C_RETRAIN_RELEASE_SCOPE.json"
        ).read_text(encoding="utf-8")
    )


def test_scope_rejects_test_access_and_wrong_branch(tmp_path):
    from scripts.cai_c_retrain.context import TaskContext

    scope = _scope()
    scope["cohort"]["vlm_authorized_splits"] = ["TRAIN", "VALID", "TEST"]
    with pytest.raises(ValueError, match="TEST"):
        TaskContext.from_mapping(
            tmp_path,
            scope,
            tmp_path / "scope.json",
            branch="research/cai-vlm-agent-v3-controlled-reuse",
            head=scope["source_commit"],
        )

    scope = _scope()
    with pytest.raises(ValueError, match="branch"):
        TaskContext.from_mapping(
            tmp_path,
            scope,
            tmp_path / "scope.json",
            branch="main",
            head=scope["source_commit"],
        )


def test_scope_rejects_changed_scientific_values(tmp_path):
    from scripts.cai_c_retrain.context import TaskContext

    scope = _scope()
    scope["models"][0]["max_logical_updates"] = 1500
    with pytest.raises(ValueError, match="logical updates"):
        TaskContext.from_mapping(
            tmp_path,
            scope,
            tmp_path / "scope.json",
            branch="research/cai-vlm-agent-v3-controlled-reuse",
            head=scope["source_commit"],
        )


def test_phase_signature_detects_changed_inputs(tmp_path):
    from scripts.cai_c_retrain.context import TaskContext

    config = tmp_path / "scope.json"
    config.write_text(json.dumps(_scope()), encoding="utf-8")
    context = TaskContext.from_mapping(
        tmp_path,
        _scope(),
        config,
        branch="research/cai-vlm-agent-v3-controlled-reuse",
        head=_scope()["source_commit"],
    )
    source = tmp_path / "source.bin"
    source.write_bytes(b"first")
    signature = context.phase_signature("prepare", [source])
    context.require_phase_signature("prepare", signature, [source])

    source.write_bytes(b"second")
    with pytest.raises(ValueError, match="signature"):
        context.require_phase_signature("prepare", signature, [source])


def test_state_transition_is_atomic_and_does_not_erase_other_phases(tmp_path):
    from scripts.cai_c_retrain.context import TaskContext

    config = tmp_path / "scope.json"
    config.write_text(json.dumps(_scope()), encoding="utf-8")
    context = TaskContext.from_mapping(
        tmp_path,
        _scope(),
        config,
        branch="research/cai-vlm-agent-v3-controlled-reuse",
        head=_scope()["source_commit"],
    )
    context.transition("prepare", "COMPLETE", specimen_count=211)
    context.transition("vlm", "RUNNING", completed=6)

    state = json.loads(context.task_state_path.read_text(encoding="utf-8"))
    assert state["task_id"] == "CAI_V3_C_RENDER_RETRAIN_RELEASE_R1_331f5295"
    assert state["phases"]["prepare"]["specimen_count"] == 211
    assert state["phases"]["vlm"]["completed"] == 6
    assert not list(context.task_state_path.parent.glob("*.tmp"))


def test_cli_exposes_all_authorized_commands():
    from scripts.cai_c_retrain.cli import build_parser

    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {
        "prepare",
        "vlm",
        "train",
        "assemble",
        "analyze",
        "paper",
        "verify",
        "publish",
        "export-inputs",
        "all",
    }
    args = parser.parse_args(
        [
            "train",
            "--config",
            "scope.json",
            "--resume",
            "--method",
            "VLM_SPATIAL_FEEDBACK",
        ]
    )
    assert args.resume is True
    assert args.method == "VLM_SPATIAL_FEEDBACK"


def test_all_does_not_repeat_prepare_after_creating_runtime_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.cai_c_retrain import cli

    output = tmp_path / "out"
    context = SimpleNamespace(
        root=tmp_path,
        config_path=tmp_path / "scope.json",
        path=lambda name: output,
    )
    calls = []

    def prepare(module, function, passed_context):
        calls.append((module, function, passed_context))
        output.mkdir()
        (output / "runtime_lock.json").write_text(
            json.dumps(
                {
                    "vlm_python": "/usr/bin/python3",
                    "actor_python": "/usr/bin/python3",
                    "report_python": "/usr/bin/python3",
                }
            ),
            encoding="utf-8",
        )

    phases = []
    monkeypatch.setattr(cli, "_call", prepare)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kwargs: (
            phases.append(command[2]) or SimpleNamespace(returncode=0)
        ),
    )

    cli._run_all(context)

    assert [(module, function) for module, function, _ in calls] == [
        ("prepare", "prepare_stage")
    ]
    assert phases == list(cli.PHASES[1:])


def test_all_skips_complete_phases_and_retries_train_once(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.cai_c_retrain import cli

    output = tmp_path / "out"
    output.mkdir()
    (output / "runtime_lock.json").write_text(
        json.dumps(
            {
                "vlm_python": "/usr/bin/python3",
                "actor_python": "/usr/bin/python3",
                "report_python": "/usr/bin/python3",
            }
        ),
        encoding="utf-8",
    )
    (output / "task_state.json").write_text(
        json.dumps(
            {
                "phases": {
                    "prepare": {"status": "COMPLETE"},
                    "vlm": {"status": "COMPLETE"},
                }
            }
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        root=tmp_path,
        config_path=tmp_path / "scope.json",
        task_state_path=output / "task_state.json",
        path=lambda name: output,
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[2] == "train" and "--resume" not in command:
            raise cli.subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    result = cli._run_all(context)

    assert [command[2] for command in calls] == [
        "train",
        "train",
        "assemble",
        "analyze",
        "paper",
        "verify",
        "publish",
    ]
    assert "--resume" in calls[1]
    assert result["prepare"] == "already complete"
    assert result["vlm"] == "already complete"
