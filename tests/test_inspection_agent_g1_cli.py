from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_inspection_agent_g1.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_g1_cli_help_lists_inspection_commands() -> None:
    completed = _run("--help")

    assert completed.returncode == 0
    for command in ("build-bank", "build-all-banks", "validate", "compare"):
        assert command in completed.stdout


def test_g1_cli_subcommand_help_lists_required_flags() -> None:
    expected = {
        "build-bank": (
            "--config",
            "--source-project-root",
            "--outer-target",
            "--source-domain",
        ),
        "build-all-banks": ("--config", "--source-project-root"),
        "validate": ("--config", "--path"),
        "compare": ("--config", "--formal", "--replay"),
    }

    for command, flags in expected.items():
        completed = _run(command, "--help")
        assert completed.returncode == 0, completed.stderr
        assert command in completed.stdout
        for flag in flags:
            assert flag in completed.stdout


def test_g1_cli_missing_required_arguments_returns_code_two() -> None:
    for command in ("build-bank", "build-all-banks", "validate", "compare"):
        completed = _run(command)

        assert completed.returncode == 2
        assert completed.stdout == ""
