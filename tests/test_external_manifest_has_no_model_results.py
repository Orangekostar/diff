from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_external_manifest_has_no_model_results() -> None:
    manifest = json.loads(
        (ROOT / "artifacts/external_data/EXTERNAL_DATA_MANIFEST.json").read_text()
    )
    assert manifest["method_performance_present"] is False
    assert manifest["discipline"].endswith("document_only")
    assert all(
        dataset.get("method_performance_present", False) is False
        for dataset in manifest["datasets"].values()
    )


def test_external_csv_manifests_use_lf_line_endings() -> None:
    artifact = ROOT / "artifacts/external_data"
    for path in artifact.rglob("*.csv"):
        assert b"\r" not in path.read_bytes(), path
