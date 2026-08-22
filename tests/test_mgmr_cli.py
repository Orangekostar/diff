from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_mgmr_m0.py"
CONFIG = ROOT / "paper_v3/configs/mgmr_m0.yaml"


def test_cli_accepts_only_registered_device() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--device", "cpu"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr
    assert completed.stdout == ""


def test_cli_parser_has_only_config_device_and_replay() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from run_mgmr_m0 import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["--config", str(CONFIG), "--device", "cuda", "--replay"]
        )
        assert args.config == CONFIG
        assert args.device == "cuda"
        assert args.replay is True
        with pytest.raises(SystemExit):
            parser.parse_args(["--config", str(CONFIG), "--unknown"])
    finally:
        sys.path.remove(str(ROOT / "scripts"))
