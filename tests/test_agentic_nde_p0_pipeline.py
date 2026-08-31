from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cmc_bbdm.agentic_nde.p0 import PipelineError, audit_p0

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_requires_explicit_existing_surface_root(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="ascii")
    with pytest.raises(PipelineError, match="surface root"):
        audit_p0(config_path=config, surface_root=tmp_path / "missing", output=tmp_path / "out", project_root=ROOT)


def test_pipeline_refuses_existing_output_before_work(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("schema_version: 1\n", encoding="ascii")
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(PipelineError, match="output"):
        audit_p0(config_path=config, surface_root=tmp_path, output=output, project_root=ROOT)


def test_importing_p0_does_not_import_model_frameworks() -> None:
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "sklearn" not in sys.modules
