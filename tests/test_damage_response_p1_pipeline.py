from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest

from cmc_bbdm.damage_response import pipeline
from cmc_bbdm.damage_response.contracts import StageStatus

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/damage_to_failure_response_p1.yaml"
P0_PACKAGE = ROOT / "results/damage_to_failure_response/p0_data_audit"


def test_recommended_p1_config_preserves_frozen_contract() -> None:
    config = pipeline.load_p1_config(CONFIG)

    assert config.base_sha == "3951f71f28b6efdf8c74eea0fe274b2a78a9cd57"
    assert config.p0_summary_relative_path == (
        "results/damage_to_failure_response/p0_data_audit/summary.json"
    )
    assert config.p0_summary_sha256 == (
        "9d44ead975119db2181a91efbf14b74165671a9d25b7b576d90f6e104757a633"
    )
    assert config.baseline_samples == 50
    assert config.grid_points == 101
    assert config.minimum_unique_extension_positions == 50
    assert config.ridge_alpha == 1e-6
    assert config.minimum_coverage_fraction == 0.90
    assert config.near_deterministic_pooled_r2 == 0.90
    assert config.maximum_representative_pairs == 12


def test_committed_p0_package_authorizes_p1() -> None:
    config = pipeline.load_p1_config(CONFIG)

    package = pipeline.validate_p1_p0_authority(config, ROOT)

    assert package == P0_PACKAGE


def test_p1_rejects_p0_summary_hash_drift_before_external_roots(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    copied_package = (
        repository / "results/damage_to_failure_response/p0_data_audit"
    )
    copied_package.parent.mkdir(parents=True)
    shutil.copytree(P0_PACKAGE, copied_package)
    summary = copied_package / "summary.json"
    summary.write_bytes(summary.read_bytes() + b" ")

    with pytest.raises(pipeline.PipelineError, match="P0 summary SHA-256"):
        pipeline.run_p1_audit(
            config_path=CONFIG,
            repo_root=repository,
            legacy_root=tmp_path / "missing-legacy",
            hasebe_v3_root=tmp_path / "missing-v3",
            output=tmp_path / "p1",
            decision_output=tmp_path / "decision.md",
        )


@pytest.mark.parametrize("existing", ("package", "decision"))
def test_p1_refuses_existing_outputs_before_execution(
    tmp_path: Path, existing: str
) -> None:
    output = tmp_path / "p1"
    decision = tmp_path / "decision.md"
    target = output if existing == "package" else decision
    if existing == "package":
        target.mkdir()
    else:
        target.write_text("existing\n", encoding="ascii")

    with pytest.raises(pipeline.PipelineError, match="already exists"):
        pipeline.run_p1_audit(
            config_path=CONFIG,
            repo_root=ROOT,
            legacy_root=tmp_path / "missing-legacy",
            hasebe_v3_root=tmp_path / "missing-v3",
            output=output,
            decision_output=decision,
        )


def test_completed_p1_no_go_is_cli_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(**_kwargs: object) -> pipeline.P1RunResult:
        return pipeline.P1RunResult(
            status=StageStatus.RESPONSE_BEYOND_STRENGTH_NO_GO,
            output=Path("p1-output"),
            decision_output=Path("decision.md"),
            passing_endpoints=(),
        )

    monkeypatch.setattr(pipeline, "run_p1_audit", fake_run)

    exit_code = pipeline.main(
        [
            "audit-p1",
            "--config",
            "config.yaml",
            "--legacy-root",
            "legacy",
            "--hasebe-v3-root",
            "v3",
            "--output",
            "output",
            "--decision-output",
            "decision.md",
        ]
    )

    assert exit_code == 0


def test_p1_code_is_lazy_and_does_not_change_p0_import_boundary() -> None:
    source = inspect.getsource(pipeline)
    assert "from sklearn" not in source
    assert "import sklearn" not in source
