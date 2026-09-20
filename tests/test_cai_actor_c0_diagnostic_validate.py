from pathlib import Path

from scripts.cai_actor_c0_diagnostic.validate import required_delivery_paths


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
