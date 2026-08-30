from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cmc_bbdm.damage_response import pipeline
from cmc_bbdm.damage_response.contracts import PRIMARY_COUNTS, StageStatus

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/damage_to_failure_response.yaml"
P0_SUMMARY = (
    ROOT / "results/damage_to_failure_response/p0_data_audit/summary.json"
)


def test_recommended_p0_config_preserves_frozen_contract() -> None:
    config = pipeline.load_p0_config(CONFIG)

    assert config.base_sha == "3951f71f28b6efdf8c74eea0fe274b2a78a9cd57"
    assert config.dataset_id == "8scdmfdcfb"
    assert config.dataset_version == 3
    assert dict(config.primary_counts) == dict(PRIMARY_COUNTS)
    assert config.load_kn_per_volt == 25.0
    assert config.displacement_mm_per_volt == 1.0
    assert config.minimum_exact_pairs_per_domain == 20
    assert config.maximum_missing_primary_channel_fraction == 0.2


def test_p0_pipeline_requires_explicit_external_roots(tmp_path: Path) -> None:
    with pytest.raises(pipeline.PipelineError, match="explicit external root"):
        pipeline.run_p0_audit(
            config_path=CONFIG,
            repo_root=ROOT,
            legacy_root=tmp_path / "missing-legacy",
            hasebe_v3_root=tmp_path / "missing-v3",
            output=tmp_path / "output",
        )


def test_p0_pipeline_refuses_existing_output_before_execution(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(pipeline.PipelineError, match="already exists"):
        pipeline.run_p0_audit(
            config_path=CONFIG,
            repo_root=ROOT,
            legacy_root=tmp_path / "missing-legacy",
            hasebe_v3_root=tmp_path / "missing-v3",
            output=output,
        )


def test_completed_no_go_audit_is_cli_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(**_kwargs: object) -> pipeline.P0RunResult:
        return pipeline.P0RunResult(
            status=StageStatus.P0_NO_GO,
            output=Path("audit-output"),
        )

    monkeypatch.setattr(pipeline, "run_p0_audit", fake_run)
    exit_code = pipeline.main(
        [
            "audit-p0",
            "--config",
            "config.yaml",
            "--legacy-root",
            "legacy",
            "--hasebe-v3-root",
            "v3",
            "--output",
            "output",
        ]
    )
    assert exit_code == 0


def test_integrity_error_is_cli_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_run(**_kwargs: object) -> pipeline.P0RunResult:
        raise pipeline.PipelineError("integrity failure")

    monkeypatch.setattr(pipeline, "run_p0_audit", fail_run)
    exit_code = pipeline.main(
        [
            "audit-p0",
            "--config",
            "config.yaml",
            "--legacy-root",
            "legacy",
            "--hasebe-v3-root",
            "v3",
            "--output",
            "output",
        ]
    )
    assert exit_code == 1


def test_p0_orchestration_imports_no_model_framework() -> None:
    source = inspect.getsource(pipeline)
    assert "scikit-learn" not in source
    assert "sklearn" not in source
    assert "torch" not in source


def test_cli_bootstraps_compact_local_package() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_damage_response.py"), "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_committed_p0_summary_closes_all_ten_authority_questions() -> None:
    summary = json.loads(P0_SUMMARY.read_text(encoding="ascii"))
    questions = summary["pre_model_authority_questions"]

    assert set(questions) == {f"q{index}" for index in range(1, 11)}
    assert all(record["status"].startswith("CLOSED") for record in questions.values())
    assert all(record["evidence"] for record in questions.values())
