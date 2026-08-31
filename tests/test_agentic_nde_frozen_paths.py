from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "15db6edad14ef36364fbda17945ccc924f600e47"
FROZEN_PATHS = (
    "results/mva/",
    "results/mvd/",
    "results/mavis/",
    "results/mavis_science_closure/",
    "results/p1_full_field_oracle/",
    "results/p3_spatial_specificity/",
    "results/p5_sparse_scan/",
    "artifacts/mavis/",
    "artifacts/mavis_science_closure/",
    "artifacts/mvd_authority/",
    "artifacts/mavis_authority/",
    "artifacts/aei_information_hierarchy/",
    "results/damage_to_failure_response/",
    "artifacts/damage_to_failure_response/",
    "paper_aei_information_hierarchy/",
    "src/cmc_bbdm/mva/",
    "src/cmc_bbdm/mvd/",
    "src/cmc_bbdm/mavis/",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_agentic_nde_work_does_not_modify_frozen_paths() -> None:
    tracked_diff = _git("diff", "--name-only", BASE_SHA, "--", *FROZEN_PATHS)
    worktree_status = _git(
        "status",
        "--short",
        "--untracked-files=all",
        "--",
        *FROZEN_PATHS,
    )

    assert tracked_diff == ""
    assert worktree_status == ""
