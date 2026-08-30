from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest

from cmc_bbdm.damage_response import pipeline
from cmc_bbdm.damage_response.contracts import StageStatus

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/damage_to_failure_response_p2.yaml"
P0_PACKAGE = ROOT / "results/damage_to_failure_response/p0_data_audit"
P1_PACKAGE = ROOT / "results/damage_to_failure_response/p1_response_richness"


def test_recommended_p2_config_preserves_frozen_protocol() -> None:
    config = pipeline.load_p2_config(CONFIG)

    assert config.base_sha == "3951f71f28b6efdf8c74eea0fe274b2a78a9cd57"
    assert config.config_sha256 == (
        "c04a206be7fc6847dbcb43b1eb9252733dce173901276ee4e62dcfc5f3494d92"
    )
    assert config.p1_summary_sha256 == (
        "37da95962395a0915f586820ab03f06d8d859856e8637d975bc302b1d555ebc7"
    )
    assert config.endpoints == (
        "extension_peak_mm",
        "slope_u20_u60_mpa_per_mm",
        "normalized_prepeak_auc",
    )
    assert config.ridge_alphas == (0.1, 1.0, 10.0, 100.0)
    assert config.pca_dimensions == (8, 16, 32)
    assert config.bootstrap_seed == 20260830
    assert config.bootstrap_replicates == 100000
    assert config.minimum_relative_improvement == 0.10
    assert config.minimum_improved_domains == 4


def test_committed_p0_and_p1_packages_authorize_p2() -> None:
    config = pipeline.load_p2_config(CONFIG)

    authority = pipeline.validate_p2_upstream_authority(config, ROOT)

    assert authority.p0_package == P0_PACKAGE
    assert authority.p1_package == P1_PACKAGE
    assert authority.descriptor_table == P1_PACKAGE / "descriptor_table.csv"


def test_p2_rejects_p1_drift_before_external_roots(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    copied_p0 = repository / "results/damage_to_failure_response/p0_data_audit"
    copied_p1 = repository / "results/damage_to_failure_response/p1_response_richness"
    copied_p0.parent.mkdir(parents=True)
    shutil.copytree(P0_PACKAGE, copied_p0)
    shutil.copytree(P1_PACKAGE, copied_p1)
    summary = copied_p1 / "summary.json"
    summary.write_bytes(summary.read_bytes() + b" ")

    with pytest.raises(pipeline.PipelineError, match="P1 summary|P1 package"):
        pipeline.run_p2_audit(
            config_path=CONFIG,
            repo_root=repository,
            legacy_root=tmp_path / "missing-legacy",
            hasebe_v3_root=tmp_path / "missing-v3",
            output=tmp_path / "p2",
            decision_output=tmp_path / "decision.md",
        )


def test_p2_rejects_repository_symlink_before_external_roots(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.symlink_to(ROOT, target_is_directory=True)

    with pytest.raises(pipeline.PipelineError, match="repository"):
        pipeline.run_p2_audit(
            config_path=CONFIG,
            repo_root=repository,
            legacy_root=tmp_path / "missing-legacy",
            hasebe_v3_root=tmp_path / "missing-v3",
            output=tmp_path / "p2",
            decision_output=tmp_path / "decision.md",
        )


@pytest.mark.parametrize("existing", ("package", "decision"))
def test_p2_refuses_existing_outputs_before_execution(
    tmp_path: Path, existing: str
) -> None:
    output = tmp_path / "p2"
    decision = tmp_path / "decision.md"
    target = output if existing == "package" else decision
    if existing == "package":
        target.mkdir()
    else:
        target.write_text("existing\n", encoding="ascii")

    with pytest.raises(pipeline.PipelineError, match="already exists"):
        pipeline.run_p2_audit(
            config_path=CONFIG,
            repo_root=ROOT,
            legacy_root=tmp_path / "missing-legacy",
            hasebe_v3_root=tmp_path / "missing-v3",
            output=output,
            decision_output=decision,
        )


def test_completed_p2_no_go_is_cli_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(**_kwargs: object) -> pipeline.P2RunResult:
        return pipeline.P2RunResult(
            status=StageStatus.MACK_EXTENSION_NO_GO,
            output=Path("p2-output"),
            decision_output=Path("decision.md"),
            passing_contrasts=(),
        )

    monkeypatch.setattr(pipeline, "run_p2_audit", fake_run)

    exit_code = pipeline.main(
        [
            "audit-p2",
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


def test_p2_execution_import_remains_lazy() -> None:
    source = inspect.getsource(pipeline)
    assert "from sklearn" not in source
    assert "import sklearn" not in source
