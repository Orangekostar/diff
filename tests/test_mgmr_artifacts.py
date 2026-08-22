from __future__ import annotations

from pathlib import Path

import pytest
from mgmr_test_support import synthetic_formal

from cmc_bbdm.mgmr.artifacts import (
    MGMRArtifactError,
    publish_m0_package,
    validate_m0_package,
)
from cmc_bbdm.mgmr.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "paper_v3/configs/mgmr_m0.yaml"


def test_m0_package_is_complete_and_recalculates_metrics(tmp_path: Path) -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)
    formal = synthetic_formal(protocol)
    output = tmp_path / "formal"

    published = publish_m0_package(
        output,
        protocol=protocol,
        formal=formal,
        feature_manifest_sha256="3" * 64,
    )
    validated = validate_m0_package(
        output, project_root=ROOT, config_path=CONFIG
    )

    assert published == validated
    assert validated.status == "MGMR_GO"
    assert {path.name for path in output.iterdir()} == {
        "config.yaml",
        "predictions.csv",
        "aggregate_metrics.csv",
        "domain_metrics.csv",
        "bootstrap.csv",
        "source_residuals.csv",
        "summary.json",
        "REPORT.md",
        "artifact_manifest.json",
        "CHECKSUMS.sha256",
    }
    assert str(ROOT) not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.iterdir()
        if path.suffix in {".csv", ".json", ".md", ".yaml"}
        or path.name == "CHECKSUMS.sha256"
    )


def test_m0_package_rejects_prediction_tampering(tmp_path: Path) -> None:
    protocol = load_protocol(CONFIG, project_root=ROOT)
    output = tmp_path / "formal"
    publish_m0_package(
        output,
        protocol=protocol,
        formal=synthetic_formal(protocol),
        feature_manifest_sha256="3" * 64,
    )
    with (output / "predictions.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(MGMRArtifactError, match="checksum|SHA-256"):
        validate_m0_package(output, project_root=ROOT, config_path=CONFIG)
