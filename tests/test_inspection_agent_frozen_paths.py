from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "892d92ea4979d9ca8ceeafef3348cd43266ed1b8"
CONFIG = ROOT / "paper_v3/configs/inspection_agent_g0.yaml"
FROZEN_PATHS = (
    "results/mva/",
    "results/mvd/",
    "results/mavis/",
    "results/mavis_science_closure/",
    "src/cmc_bbdm/mva/",
    "src/cmc_bbdm/mvd/",
    "src/cmc_bbdm/mavis/",
    "results/agentic_task_driven_nde/p0_registration/",
    "results/agentic_task_driven_nde/p0r_author_registration/",
    "results/agentic_task_driven_nde/p1_visual_observability/",
    "results/agentic_task_driven_nde/replay/p1_visual_observability/",
    "artifacts/agentic_task_driven_nde/P0_REGISTRATION_DECISION.md",
    "artifacts/agentic_task_driven_nde/P0R_AUTHOR_REGISTRATION_DECISION.md",
    "artifacts/agentic_task_driven_nde/P1_VISUAL_OBSERVABILITY_DECISION.md",
    "artifacts/agentic_task_driven_nde/P1_SPATIAL_VS_GLOBAL_CONTEXT_ANALYSIS.md",
)


def test_frozen_paths_unchanged_from_base() -> None:
    diff = subprocess.run(
        ["git", "diff", "--name-only", BASE_SHA, "--", *FROZEN_PATHS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert diff.returncode == 0, diff.stderr
    assert diff.stdout == ""

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *FROZEN_PATHS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    assert status.stdout == ""


def test_g0_source_files_match_declared_sha256() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    for name, source in config["sources"].items():
        relative_path = source["path"]
        path = ROOT / relative_path
        assert path.is_file(), f"{name}: missing source file {relative_path}"
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_sha256 == source["sha256"], (
            f"{name}: SHA256 mismatch for {relative_path}"
        )
