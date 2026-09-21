import subprocess
from pathlib import Path

from scripts.cai_actor_c0_diagnostic.validate import (
    _commit,
    _porcelain_paths,
    _session_timing,
    _vlm_candidate_overlay_expected_empty,
    required_delivery_paths,
    validate_html_contract,
    validate_state_provenance_rows,
)


def test_empty_candidate_overlay_requires_complete_all_zero_vlm_evidence():
    empty = [{"vlm_indicator": "0.0"} for _ in range(64)]
    assert _vlm_candidate_overlay_expected_empty(empty)

    active = [dict(row) for row in empty]
    active[17]["vlm_indicator"] = "0.25"
    assert not _vlm_candidate_overlay_expected_empty(active)
    assert not _vlm_candidate_overlay_expected_empty(empty[:-1])


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
        tmp_path / "t0_equal_weight_summary.csv",
        tmp_path / "panels" / "summary" / "group_G_t0_equal_weight_summary.png",
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


def test_state_provenance_contract_requires_explicit_origins_prefixes_and_hashes():
    valid = {
        "specimen_key": "domain:q1",
        "state_id": "s00_shared",
        "sources": "['C_NATIVE_PREFIX', 'N_NATIVE_PREFIX']",
        "source_models": "['C', 'N']",
        "prefix_lengths": "[0, 0]",
        "action_count": "0",
        "t": "0",
        "prefix_cells_in_order": "[]",
        "state_origins": (
            '[{"label":"C_NATIVE_PREFIX","model":"C","t":0,'
            '"prefix_cells_in_order":[]},{"label":"N_NATIVE_PREFIX",'
            '"model":"N","t":0,"prefix_cells_in_order":[]}]'
        ),
        "physical_state_sha256": "a" * 64,
        "policy_input_sha256": '{"C":"' + "b" * 64 + '","N":"' + "c" * 64 + '"}',
        "c_prior_sha256": "d" * 64,
    }
    assert validate_state_provenance_rows([valid]) == {"state_count": 1}
    for missing in (
        "t",
        "prefix_cells_in_order",
        "state_origins",
        "policy_input_sha256",
        "c_prior_sha256",
    ):
        invalid = valid.copy()
        invalid.pop(missing)
        try:
            validate_state_provenance_rows([invalid])
        except ValueError as error:
            assert missing in str(error)
        else:
            raise AssertionError(f"missing {missing} should fail")


def test_html_contract_rejects_missing_required_views_and_source_links():
    required = (
        "原生轨迹对照",
        "相同状态对照",
        "C来源状态",
        "N来源状态",
        "VLM全部候选",
        "动作顺序及差值表",
        "metadata.json",
        "physical_state.npz",
        "zero_prior_sensitivity.json",
        "query_checks.json",
        "t0_equal_weight_summary.csv",
    )
    document = "\n".join(required)
    assert validate_html_contract(document) == {"required_marker_count": len(required)}
    for marker in required:
        try:
            validate_html_contract(document.replace(marker, "", 1))
        except ValueError as error:
            assert marker in str(error)
        else:
            raise AssertionError(f"missing {marker} should fail")


def test_session_timing_adds_cache_render_once_without_counting_idle_gap():
    prior = {
        "measurement": "CONSERVATIVE_ARTIFACT_MTIME_WALL_ENVELOPES",
        "total_task_seconds_upper_bound": 2750.0,
        "gpu_session_seconds_upper_bound": 502.0,
        "cpu_compute_seconds_upper_bound": 2246.0,
    }
    limits = {
        "gpu_session_seconds_max": 2700,
        "cpu_compute_seconds_max": 5400,
    }
    updated = _session_timing(prior, cache_render_seconds=274.0, limits=limits)
    assert updated["historical_diagnostic_cpu_seconds_upper_bound"] == 2246.0
    assert updated["cache_report_render_seconds_upper_bound"] == 274.0
    assert updated["cpu_compute_seconds_upper_bound"] == 2520.0
    assert updated["total_task_seconds_upper_bound"] == 3024.0

    repeated = _session_timing(updated, cache_render_seconds=274.0, limits=limits)
    assert repeated == updated
