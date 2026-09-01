from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "7a10cd425de582fa158bf6639285731ccd8ff7a7"
CONFIG = ROOT / "paper_v3/configs/inspection_agent_g1.yaml"
CONTROLLING_PROMPT_SHA256 = "37b7e4b9860dd338589cf2d9dd8f2cc8d202451a84eeafbc4961b2c5a28fcda5"
DOMAIN_ORDER = (
    "74t7kcdgkr",
    "cgtnjyggtm",
    "w68dtmpfyf",
    "xcmzfsbd9t",
    "yfxyg8jm46",
    "ykhs7s2dck",
)
DOMAIN_COUNTS = {
    "74t7kcdgkr": 45,
    "cgtnjyggtm": 49,
    "w68dtmpfyf": 43,
    "xcmzfsbd9t": 59,
    "yfxyg8jm46": 42,
    "ykhs7s2dck": 38,
}


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_g1_base_is_a_local_ancestor_and_config_binds_prompt() -> None:
    object_type = _git("cat-file", "-t", BASE_SHA)
    assert object_type.returncode == 0, object_type.stderr
    assert object_type.stdout.strip() == "commit"

    ancestor = _git("merge-base", "--is-ancestor", BASE_SHA, "HEAD")
    assert ancestor.returncode == 0, ancestor.stderr

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["repository_base_sha"] == BASE_SHA
    assert config["controlling_prompt_sha256"] == CONTROLLING_PROMPT_SHA256


def test_g1_cohort_and_nested_lodo_contract_are_exact() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cohort = config["cohort"]

    assert cohort["specimen_count"] == 276
    assert tuple(cohort["domain_order"]) == DOMAIN_ORDER
    assert cohort["domain_counts"] == DOMAIN_COUNTS
    assert cohort["outer_split"] == "leave_one_domain_out"
    assert cohort["inner_split"] == "leave_one_source_domain_out"
