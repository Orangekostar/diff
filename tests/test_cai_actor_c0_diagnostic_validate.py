import subprocess
from pathlib import Path

from scripts.cai_actor_c0_diagnostic.validate import (
    _commit,
    _porcelain_paths,
    required_delivery_paths,
)


def test_required_delivery_inventory_contains_prompt_minimum(tmp_path: Path):
    paths = set(required_delivery_paths(tmp_path, tmp_path / "artifacts"))
    assert {
        tmp_path / "index.html",
        tmp_path / "diagnostic_lock.json",
        tmp_path / "selected_cases.csv",
        tmp_path / "model_bindings.json",
        tmp_path / "state_manifest.csv",
        tmp_path / "first_action_c0_audit.csv",
        tmp_path / "c0_intervention_results.csv",
        tmp_path / "fixed_state_prior_sensitivity.csv",
        tmp_path / "surface_probe_results.csv",
        tmp_path / "attribution_checks.json",
        tmp_path / "attention_checks.json",
        tmp_path / "resource_usage.json",
        tmp_path / "diagnostic_manifest.json",
        tmp_path / "artifacts" / "SOURCE_AND_TASK_BINDINGS.md",
        tmp_path / "artifacts" / "FINDINGS_ZH.md",
        tmp_path / "artifacts" / "VERIFICATION.md",
    } <= paths


def test_commit_force_adds_manifest_artifact_ignored_by_repository(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("*.npz\n", encoding="utf-8")
    output = tmp_path / "results"
    output.mkdir()
    ignored = output / "attention.npz"
    ignored.write_bytes(b"diagnostic-array")

    commit = _commit(
        tmp_path,
        "test forced diagnostic artifact",
        [tmp_path / ".gitignore", output],
        force_paths=[ignored],
    )

    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=tmp_path, text=True
    ).splitlines()
    assert "results/attention.npz" in tracked


def test_porcelain_parser_preserves_first_path_when_index_column_is_blank():
    payload = b" M artifacts/report.md\0?? results/new.csv\0"
    assert _porcelain_paths(payload) == ["artifacts/report.md", "results/new.csv"]
