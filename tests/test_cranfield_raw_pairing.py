from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cranfield_raw_pairing_and_grid_schema() -> None:
    artifact = ROOT / "artifacts/external_data/cranfield_wp2"
    with (artifact / "raw_file_manifest.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    with (artifact / "scan_pair_manifest.csv").open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    schema = json.loads((artifact / "grid_schema.json").read_text())

    assert len(raw) == 29
    assert len(pairs) == 26
    assert sum(int(row["raw_file_count"]) for row in pairs) == 29
    assert {int(row["waveform_buckets"]) for row in raw} == {504, 836}
    assert {int(row["focal_law_count"]) for row in raw} == {57}
    assert schema["spatial_grid_recoverable"] is True
    assert schema["physical_coordinate_spacing_recoverable"] is False
    assert schema["scanner_time_claim_authorized"] is False
