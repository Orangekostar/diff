from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_release_index_links_to_paper_outside_results_tree(tmp_path):
    from scripts.cai_c_retrain.validate import _release_index

    output = tmp_path / "results/cai_agent_v3/c_render_retrain/r1_331f5295"
    paper = tmp_path / "paper_cai_aei/r2_c_331f5295"
    context = SimpleNamespace(
        path=lambda name: output if name == "output" else paper,
    )

    _release_index(context)

    index = (output / "index.html").read_text(encoding="utf-8")
    relative = Path("../../../../paper_cai_aei/r2_c_331f5295")
    assert f"{relative.as_posix()}/manuscript.html" in index
    assert f"{relative.as_posix()}/build/main.pdf" in index


def test_result_path_allowlist_is_exact():
    from scripts.cai_c_retrain.validate import _allowed_result_path

    assert _allowed_result_path(
        "results/cai_agent_v3/c_render_retrain/r1_331f5295/index.html"
    )
    assert _allowed_result_path("results/cai_agent_v3/compute_ledger.jsonl")
    assert _allowed_result_path("paper_cai_aei/r2_c_331f5295/build/main.pdf")
    assert _allowed_result_path("scripts/cai_c_retrain/validate.py")
    assert _allowed_result_path("tests/test_cai_c_retrain_validate.py")
    assert _allowed_result_path("tests/test_cai_c_retrain_vlm.py")
    assert _allowed_result_path(
        "docs/superpowers/plans/2026-09-20-c-render-retrain-release.md"
    )
    assert not _allowed_result_path("scripts/unrelated.py")
    assert not _allowed_result_path(
        "results/cai_agent_v3/c_render_retrain/r1_331f5295-other/file"
    )


def test_mark_plan_publish_complete_is_exact_and_idempotent(tmp_path):
    from scripts.cai_c_retrain.validate import (
        PLAN_PATH,
        _mark_plan_publish_complete,
    )

    plan = tmp_path / PLAN_PATH
    plan.parent.mkdir(parents=True)
    plan.write_text(
        "- [ ] **Step 6: Commit results/handoff and publish**\n",
        encoding="utf-8",
    )

    _mark_plan_publish_complete(tmp_path)
    _mark_plan_publish_complete(tmp_path)

    assert plan.read_text(encoding="utf-8") == (
        "- [x] **Step 6: Commit results/handoff and publish**\n"
    )


def test_task_ledger_requires_fifteen_unique_closed_segments():
    from scripts.cai_c_retrain.context import METHOD_SEEDS
    from scripts.cai_c_retrain.validate import _task_ledger_summary

    task_id = "task"
    rows = []
    for method in METHOD_SEEDS:
        for start in range(0, 1250, 250):
            rows.append(
                {
                    "task_id": task_id,
                    "method": method,
                    "status": "COMPLETED",
                    "start_update": start,
                    "end_update": start + 250,
                    "actual_optimizer_updates": 250,
                    "actual_optimizer_updates_upper_bound": 0,
                    "event_id": f"{method}:{start}",
                    "reservation_id": f"reservation:{method}:{start}",
                }
            )

    summary = _task_ledger_summary(rows, task_id)

    assert summary["segment_events"] == 15
    assert summary["actual_optimizer_updates"] == 3750
    with pytest.raises(ValueError, match="unique"):
        _task_ledger_summary([*rows, rows[0]], task_id)


def test_pending_paths_preserves_first_porcelain_path(tmp_path):
    from scripts.cai_c_retrain.validate import _pending_paths

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "docs/result.md"
    tracked.parent.mkdir()
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/result.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    tracked.write_text("after\n", encoding="utf-8")

    assert _pending_paths(tmp_path) == ["docs/result.md"]


def test_commit_force_add_excludes_python_bytecode(tmp_path):
    from scripts.cai_c_retrain.validate import _commit

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    script = tmp_path / "scripts/tool.py"
    cache = tmp_path / "scripts/__pycache__/tool.pyc"
    script.parent.mkdir()
    cache.parent.mkdir()
    script.write_text("value = 1\n", encoding="utf-8")
    cache.write_bytes(b"bytecode")

    commit = _commit(tmp_path, "result", ("scripts",))
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=tmp_path, text=True
    ).splitlines()

    assert paths == ["scripts/tool.py"]


def test_interrupted_result_commit_is_resumable(tmp_path):
    from scripts.cai_c_retrain.validate import (
        RESULTS_COMMIT_MESSAGE,
        _interrupted_results_commit,
    )

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "result.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "result.json"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", RESULTS_COMMIT_MESSAGE], cwd=tmp_path, check=True
    )

    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    assert _interrupted_results_commit(tmp_path) == expected


def test_verify_commit_paths_requires_each_file_in_commit(tmp_path):
    from scripts.cai_c_retrain.validate import _verify_commit_paths

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.bin"
    tracked.write_bytes(b"payload")
    subprocess.run(["git", "add", "tracked.bin"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()

    assert _verify_commit_paths(tmp_path, commit, ["tracked.bin"]) == 1
    with pytest.raises(ValueError, match="missing.bin"):
        _verify_commit_paths(tmp_path, commit, ["missing.bin"])
