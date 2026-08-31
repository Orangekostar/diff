from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cmc_bbdm.agentic_nde.artifacts import write_p0_package
from cmc_bbdm.agentic_nde.contracts import P0GateFacts, decide_p0
from scripts.run_agentic_nde import main

ROOT = Path(__file__).resolve().parents[1]


def _no_go_summary() -> dict[str, object]:
    facts = P0GateFacts(
        authorized_by_domain={},
        exact_identity_hashes=True,
        orientation_resolved=False,
        deterministic_transform=False,
        deployable_evidence_only=True,
        replay_verified=False,
    )
    decision = decide_p0(facts)
    return {**decision.as_dict(), "gate_facts": facts.as_dict()}


def _package(output: Path) -> Path:
    return write_p0_package(
        output,
        config_text="schema_version: 1\n",
        surface_manifest=[{"specimen_id": "s1"}],
        surface_qc=[{"specimen_id": "s1"}],
        registration=[{"specimen_id": "s1", "status": "UNRESOLVED"}],
        registration_qc=[{"specimen_id": "s1", "authorized": "false"}],
        source_hashes=[{"logical_path": "x", "sha256": "a" * 64}],
        summary=_no_go_summary(),
        report="# Report\n",
    )


def test_replay_cli_returns_zero_for_completed_no_go(tmp_path: Path) -> None:
    package = _package(tmp_path / "p0")
    assert main(["replay-p0", "--path", str(package)]) == 0


def test_replay_cli_returns_one_for_integrity_error(tmp_path: Path) -> None:
    assert main(["replay-p0", "--path", str(tmp_path / "missing")]) == 1


def test_script_bootstraps_the_local_namespace_package() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "scripts/run_agentic_nde.py", "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
