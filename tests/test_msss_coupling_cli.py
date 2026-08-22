from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_coupling_cli_exposes_only_registered_config_root_and_replay_options() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_msss_coupling.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert "--project-root" in completed.stdout
    assert "--replay" in completed.stdout
    assert "--output" not in completed.stdout
