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
    assert _allowed_result_path("paper_cai_aei/r2_c_331f5295/build/main.pdf")
    assert not _allowed_result_path("scripts/cai_c_retrain/validate.py")
    assert not _allowed_result_path(
        "results/cai_agent_v3/c_render_retrain/r1_331f5295-other/file"
    )


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
