from __future__ import annotations

import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cmc_bbdm.mavis import aei_paper_evidence

_BASE_COMMIT = "c2eab6eac79dd3fbb9ecb0d19f98923e515e762b"
_FROZEN_PATHS = (
    "results/p1_full_field_oracle",
    "analysis_tables",
    "results/mva",
    "results/mvd",
    "results/mavis",
    "results/mavis_science_closure",
    "artifacts/mavis",
    "artifacts/mavis_science_closure",
    "artifacts/mvd_authority",
    "artifacts/mavis_authority",
)
_NUMERIC_FIELDS = (
    "reference_value",
    "candidate_value",
    "estimate",
    "relative_effect",
    "ci95_lower",
    "ci95_upper",
)
_SCIENTIFIC = re.compile(
    r"(?P<mantissa>[+-]?\d+(?:\.\d+)?)\\times10\^\{(?P<exponent>[+-]?\d+)\}"
)
_DECIMAL = re.compile(r"(?<![A-Za-z0-9_.])(?P<value>[+-]?\d+\.\d+)")
_ALLOWED_PROTOCOL_VALUES = (18.75,)


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    canonical_claim_count: int
    mapped_claim_count: int
    figure_count: int
    table_count: int
    section_count: int
    unmatched_numbers: tuple[str, ...]
    changed_frozen_files: tuple[str, ...]


def _visible_text(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _canonical_values(root: Path) -> list[float]:
    path = root / "artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv"
    values: list[float] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            for field in _NUMERIC_FIELDS:
                if row[field]:
                    value = float(row[field])
                    values.extend(
                        (value, abs(value), 100.0 * value, 100.0 * abs(value))
                    )
    values.extend(_ALLOWED_PROTOCOL_VALUES)
    return values


def _matches_rounding(value: float, decimals: int, candidates: list[float]) -> bool:
    tolerance = 0.500001 * 10.0 ** (-decimals)
    return any(math.isclose(value, candidate, abs_tol=tolerance) for candidate in candidates)


def unmatched_results_numbers(root: Path) -> list[str]:
    manuscript = (root / "paper_aei_information_hierarchy/main.tex").read_text(
        encoding="utf-8"
    )
    results = manuscript.split(
        r"\section{Experimental Results and Discussion}", maxsplit=1
    )[1].split(r"\section{Conclusions}", maxsplit=1)[0]
    results = _visible_text(results).replace("--", " ")
    candidates = _canonical_values(root)
    unmatched: list[str] = []

    scientific_spans: list[tuple[int, int]] = []
    for match in _SCIENTIFIC.finditer(results):
        scientific_spans.append(match.span())
        mantissa_text = match.group("mantissa")
        exponent = int(match.group("exponent"))
        value = float(mantissa_text) * 10.0**exponent
        decimals = len(mantissa_text.partition(".")[2]) - exponent
        if not _matches_rounding(value, decimals, candidates):
            unmatched.append(match.group(0))

    ordinary_text = list(results)
    for start, end in scientific_spans:
        ordinary_text[start:end] = " " * (end - start)
    for match in _DECIMAL.finditer("".join(ordinary_text)):
        token = match.group("value")
        decimals = len(token.partition(".")[2])
        if not _matches_rounding(float(token), decimals, candidates):
            unmatched.append(token)
    return sorted(set(unmatched))


def changed_frozen_paths(root: Path) -> list[str]:
    command = [
        "git",
        "diff",
        "--name-only",
        _BASE_COMMIT,
        "--",
        *_FROZEN_PATHS,
    ]
    result = subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def validate_paper(root: Path) -> ValidationReport:
    manuscript = (root / "paper_aei_information_hierarchy/main.tex").read_text(
        encoding="utf-8"
    )
    canonical = {
        row.claim_id for row in aei_paper_evidence.build_canonical_metrics(root)
    }
    mapped = set(re.findall(r"\b[UOA][1-5]_[A-Z0-9_]+\b", manuscript))
    figures = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", manuscript)
    tables = re.findall(r"\\input\{tables/[^}]+\}", manuscript)
    sections = re.findall(r"^\\section\{[^}]+\}", manuscript, flags=re.MULTILINE)
    unmatched = tuple(unmatched_results_numbers(root))
    changed = tuple(changed_frozen_paths(root))
    passed = (
        mapped == canonical
        and len(figures) == 4
        and len(tables) == 2
        and len(sections) == 6
        and not unmatched
        and not changed
    )
    return ValidationReport(
        passed=passed,
        canonical_claim_count=len(canonical),
        mapped_claim_count=len(mapped),
        figure_count=len(figures),
        table_count=len(tables),
        section_count=len(sections),
        unmatched_numbers=unmatched,
        changed_frozen_files=changed,
    )


__all__ = [
    "ValidationReport",
    "changed_frozen_paths",
    "unmatched_results_numbers",
    "validate_paper",
]
