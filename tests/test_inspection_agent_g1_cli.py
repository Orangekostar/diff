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
    for command in (
        "build-bank",
        "build-all-banks",
        "build-fixed-endpoints",
        "build-all-fixed-endpoints",
        "build-stop-bank",
        "build-all-stop-banks",
        "build-source-bridge",
        "build-all-source-bridges",
        "select-outer",
        "select-outer-engineering",
        "select-outer-dagger",
        "select-outer-stop",
        "build-target-trajectories",
        "evaluate-target",
        "build-formal-package",
        "validate",
        "compare",
    ):
        assert command in completed.stdout


def test_g1_cli_subcommand_help_lists_required_flags() -> None:
    expected = {
        "build-bank": (
            "--config",
            "--source-project-root",
            "--outer-target",
            "--source-domain",
        ),
        "build-all-banks": (
            "--config",
            "--source-project-root",
            "--start-fold",
        ),
        "build-fixed-endpoints": (
            "--config",
            "--source-project-root",
            "--outer-target",
            "--source-domain",
        ),
        "build-all-fixed-endpoints": (
            "--config",
            "--source-project-root",
            "--start-fold",
        ),
        "build-stop-bank": (
            "--config",
            "--source-project-root",
            "--outer-target",
            "--source-domain",
            "--teacher-bank-root",
            "--fixed-endpoint-root",
        ),
        "build-all-stop-banks": (
            "--config",
            "--source-project-root",
            "--start-fold",
            "--teacher-bank-root",
            "--fixed-endpoint-root",
        ),
        "build-source-bridge": (
            "--config",
            "--source-project-root",
            "--outer-target",
            "--source-domain",
        ),
        "build-all-source-bridges": (
            "--config",
            "--source-project-root",
            "--start-fold",
            "--end-fold",
        ),
        "select-outer": ("--config", "--outer-target"),
        "select-outer-engineering": (
            "--config",
            "--source-project-root",
            "--outer-target",
            "--teacher-bank-root",
            "--bridge-root",
            "--supervised-root",
            "--learned-root",
        ),
        "select-outer-dagger": (
            "--config",
            "--source-project-root",
            "--outer-target",
            "--teacher-bank-root",
            "--bridge-root",
            "--supervised-root",
            "--learned-root",
            "--base-work-root",
            "--dagger-bank-root",
            "--work-root",
        ),
        "select-outer-stop": (
            "--config",
            "--source-project-root",
            "--outer-target",
            "--teacher-bank-root",
            "--bridge-root",
            "--fixed-endpoint-root",
            "--stop-bank-root",
            "--dagger-work-root",
            "--work-root",
        ),
        "build-target-trajectories": (
            "--config",
            "--source-project-root",
            "--outer-target",
            "--teacher-bank-root",
            "--bridge-root",
            "--fixed-endpoint-root",
            "--stop-bank-root",
            "--dagger-work-root",
            "--stop-selection-root",
            "--formal-selection-root",
            "--decision-diagnostic-root",
            "--work-root",
        ),
        "evaluate-target": (
            "--config",
            "--source-project-root",
            "--outer-target",
            "--trajectory-root",
            "--formal-selection-root",
            "--curve-root",
            "--reference-root",
        ),
        "build-formal-package": (
            "--config",
            "--source-project-root",
            "--teacher-bank-root",
            "--trajectory-root",
            "--curve-root",
            "--reference-root",
            "--formal-selection-root",
            "--decision-diagnostic-root",
            "--output",
        ),
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
    for command in (
        "build-bank",
        "build-all-banks",
        "build-fixed-endpoints",
        "build-all-fixed-endpoints",
        "build-stop-bank",
        "build-all-stop-banks",
        "build-source-bridge",
        "build-all-source-bridges",
        "select-outer",
        "select-outer-engineering",
        "select-outer-dagger",
        "select-outer-stop",
        "build-target-trajectories",
        "evaluate-target",
        "build-formal-package",
        "validate",
        "compare",
    ):
        completed = _run(command)

        assert completed.returncode == 2
        assert completed.stdout == ""
