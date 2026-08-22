from __future__ import annotations

from pathlib import Path

import pytest
from test_msss_s1 import ROOT, synthetic_s1_run

from cmc_bbdm.msss.artifacts import (
    MANDATORY_S1_FILES,
    S1ArtifactError,
    publish_s1_package,
    validate_s1_package,
)


def test_s1_package_contains_mandatory_tables_figures_and_hashes(tmp_path: Path) -> None:
    protocol, bank, run = synthetic_s1_run()
    output = tmp_path / "s1"
    published = publish_s1_package(
        output,
        protocol=protocol,
        bank=bank,
        run=run,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
        mode="smoke",
        test_only=True,
    )
    validated = validate_s1_package(
        output,
        project_root=ROOT,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
    )

    assert published == validated
    assert MANDATORY_S1_FILES <= {item.name for item in output.iterdir()}
    assert validated.gate_status == "STRONG_GO"
    assert validated.test_only
    assert len(validated.scientific_digest) == 64
    assert len(validated.output_tree_sha256) == 64


def test_s1_publication_refuses_overwrite(tmp_path: Path) -> None:
    protocol, bank, run = synthetic_s1_run()
    output = tmp_path / "s1"
    publish_s1_package(
        output,
        protocol=protocol,
        bank=bank,
        run=run,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
        mode="smoke",
        test_only=True,
    )

    with pytest.raises(S1ArtifactError, match="already exists"):
        publish_s1_package(
            output,
            protocol=protocol,
            bank=bank,
            run=run,
            config_path=ROOT / "paper_v3/configs/msss.yaml",
            mode="smoke",
            test_only=True,
        )


def test_s1_validation_detects_scientific_file_corruption(tmp_path: Path) -> None:
    protocol, bank, run = synthetic_s1_run()
    output = tmp_path / "s1"
    publish_s1_package(
        output,
        protocol=protocol,
        bank=bank,
        run=run,
        config_path=ROOT / "paper_v3/configs/msss.yaml",
        mode="smoke",
        test_only=True,
    )
    path = output / "sampling_curve.csv"
    path.write_bytes(path.read_bytes() + b"corruption\n")

    with pytest.raises(S1ArtifactError, match="checksum"):
        validate_s1_package(
            output,
            project_root=ROOT,
            config_path=ROOT / "paper_v3/configs/msss.yaml",
        )
