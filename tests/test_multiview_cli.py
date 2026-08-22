from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_aei_multiview_regression.py"
CONFIG = ROOT / "paper_v3/configs/aei_multiview_regression.yaml"


def test_audit_cli_reports_all_frozen_authorities() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "audit", "--config", str(CONFIG)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "A0_BASELINE_PASS" in result.stdout
    assert "A2_PAIRED_FEATURES_PASS" in result.stdout
    assert "FACTORISATION_NO_GO" in result.stdout


def test_cli_rejects_unknown_command() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "unknown"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
