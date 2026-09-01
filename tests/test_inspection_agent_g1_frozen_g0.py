from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

from cmc_bbdm.inspection_agent.artifacts import (
    compare_g0_packages,
    validate_g0_package,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "7a10cd425de582fa158bf6639285731ccd8ff7a7"
G1_CONFIG = ROOT / "paper_v3/configs/inspection_agent_g1.yaml"
G0_CONFIG = ROOT / "paper_v3/configs/inspection_agent_g0.yaml"
FROZEN_PATHS = (
    "results/inspection_agent/g0/",
    "results/inspection_agent/replay/g0/",
    "artifacts/inspection_agent_g0/",
    "docs/INSPECTION_AGENT_G0_PROTOCOL.md",
    "paper_v3/configs/inspection_agent_g0.yaml",
    "src/cmc_bbdm/inspection_agent/state.py",
    "src/cmc_bbdm/inspection_agent/world.py",
    "src/cmc_bbdm/inspection_agent/contracts.py",
    "src/cmc_bbdm/inspection_agent/surface_hypothesis.py",
    "src/cmc_bbdm/inspection_agent/generalized_reconstruction.py",
    "src/cmc_bbdm/inspection_agent/cai_assessor.py",
    "src/cmc_bbdm/inspection_agent/state_bank.py",
    "src/cmc_bbdm/inspection_agent/oracle.py",
    "src/cmc_bbdm/inspection_agent/stopping.py",
    "src/cmc_bbdm/inspection_agent/evaluation.py",
    "src/cmc_bbdm/inspection_agent/statistics.py",
    "src/cmc_bbdm/inspection_agent/g0.py",
    "results/agentic_task_driven_nde/",
    "artifacts/agentic_task_driven_nde/",
    "results/mva/",
    "results/mvd/",
    "results/mavis/",
    "results/mavis_science_closure/",
    "src/cmc_bbdm/mva/",
    "src/cmc_bbdm/mvd/",
    "src/cmc_bbdm/mavis/",
)
EXPECTED_STATUS = "G0_ACTIVE_INSPECTION_OPPORTUNITY_GO"
EXPECTED_MANIFEST_SHA256 = "a85a62f14bd05d69c684deab1673e01a1a84d7ebf9c3e7805760c2898eacc179"
EXPECTED_TREE_SHA256 = "e1441d847eaf187eb98de7eb84e93b708225924b5263d99560354120a7f30b0a"
EXPECTED_PACKAGE_SHA256 = "429f829b60bc9f520a41814ae2b6d34d05ef07cdfa188f39e2d9dbac93c45eca"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_g1_frozen_paths_are_unchanged_from_base() -> None:
    diff = _git("diff", "--name-only", BASE_SHA, "--", *FROZEN_PATHS)
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout == ""

    status = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *FROZEN_PATHS,
    )
    assert status.returncode == 0, status.stderr
    assert status.stdout == ""


def test_g1_declared_g0_sources_match_sha256() -> None:
    config = yaml.safe_load(G1_CONFIG.read_text(encoding="utf-8"))

    for name, source in config["sources"].items():
        relative_path = source["path"]
        path = ROOT / relative_path
        assert path.is_file(), f"{name}: missing source file {relative_path}"
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha256 == source["sha256"], (
            f"{name}: SHA256 mismatch for {relative_path}"
        )


def test_g1_formal_and_replay_g0_packages_match_frozen_identity() -> None:
    formal = ROOT / "results/inspection_agent/g0"
    replay = ROOT / "results/inspection_agent/replay/g0"

    formal_validation = validate_g0_package(
        formal,
        project_root=ROOT,
        config_path=G0_CONFIG,
    )
    replay_validation = validate_g0_package(
        replay,
        project_root=ROOT,
        config_path=G0_CONFIG,
    )
    comparison = compare_g0_packages(
        formal,
        replay,
        project_root=ROOT,
        config_path=G0_CONFIG,
    )

    for validation in (formal_validation, replay_validation):
        assert validation.status == EXPECTED_STATUS
        assert validation.manifest_sha256 == EXPECTED_MANIFEST_SHA256
        assert validation.output_tree_sha256 == EXPECTED_TREE_SHA256
    assert comparison.byte_identical is True
    assert comparison.package_sha256 == EXPECTED_PACKAGE_SHA256
    assert comparison.replay_sha256 == EXPECTED_PACKAGE_SHA256
