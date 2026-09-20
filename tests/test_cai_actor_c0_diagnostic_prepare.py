import csv
import hashlib
from pathlib import Path

import pytest

from scripts.cai_actor_c0_diagnostic.prepare import select_cases

FIXED = (
    "cgtnjyggtm:q24-48",
    "74t7kcdgkr:c8-16",
    "w68dtmpfyf:q16-29",
)
DOMAINS = ("xcmzfsbd9t", "yfxyg8jm46", "ykhs7s2dck")


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("specimen_key", "dataset_id", "split"))
        writer.writeheader()
        writer.writerows(rows)


def score(key: str) -> tuple[str, str]:
    return hashlib.sha256(f"ACTOR_C0_VIS_V2|{key}".encode()).hexdigest(), key


def test_select_cases_is_fixed_plus_hash_min_valid_per_domain(tmp_path):
    rows = [
        {"specimen_key": key, "dataset_id": key.split(":", 1)[0], "split": "VALID"}
        for key in FIXED
    ]
    for domain in DOMAINS:
        rows.extend(
            {"specimen_key": f"{domain}:case-{number}", "dataset_id": domain, "split": "VALID"}
            for number in range(3)
        )
        rows.append(
            {"specimen_key": f"{domain}:test", "dataset_id": domain, "split": "TEST"}
        )
    path = tmp_path / "index.csv"
    write_index(path, rows)
    selected = select_cases(path, fixed=FIXED, remaining_domains=DOMAINS)
    assert tuple(row["specimen_key"] for row in selected[:3]) == FIXED
    expected = tuple(
        min((f"{domain}:case-{number}" for number in range(3)), key=score)
        for domain in DOMAINS
    )
    assert tuple(row["specimen_key"] for row in selected[3:]) == expected
    assert all(row["split"] == "VALID" for row in selected)
    assert [row["selection_rule"] for row in selected] == [
        "FIXED_CASE",
        "FIXED_CASE",
        "FIXED_CASE",
        "HASH_MIN_VALID",
        "HASH_MIN_VALID",
        "HASH_MIN_VALID",
    ]


def test_select_cases_never_substitutes_missing_fixed_case(tmp_path):
    path = tmp_path / "index.csv"
    write_index(
        path,
        [
            {"specimen_key": FIXED[0], "dataset_id": "cgtnjyggtm", "split": "VALID"},
            {"specimen_key": FIXED[1], "dataset_id": "74t7kcdgkr", "split": "VALID"},
        ],
    )
    with pytest.raises(ValueError, match="fixed VALID case"):
        select_cases(path, fixed=FIXED, remaining_domains=DOMAINS)
