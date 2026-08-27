from __future__ import annotations

import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cmc_bbdm.mavis import aei_paper_evidence

_BASE_COMMIT = "ba9709545e3ade21424540547e6ab277279345de"
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
    "artifacts/external_data",
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
_CHRONOLOGY_COLUMNS = (
    "claim_id",
    "paper_layer",
    "source_stage",
    "chronology_class",
    "evidence_frozen_before_p7",
    "analysis_created_after_p7",
    "used_to_modify_p7",
    "source_path",
    "manuscript_role",
    "allowed_description",
    "forbidden_description",
)
_PRE_P7_PREFIXES = ("U1_", "U2_", "U3_", "O1_")
_SOURCE_STAGES = {
    "U1_": "P1_FULL_FIELD",
    "U2_": "P5_SPARSE_SCAN",
    "U3_": "MVD_M0",
    "U4_": "POST_FREEZE_TASK_SPECIFICITY",
    "U5_": "POST_FREEZE_VALUE_STABILITY",
    "O1_": "MVD_M1",
    "O2_": "POST_FREEZE_VALUE_EVOLUTION",
    "O3_": "POST_FREEZE_STATE_CONTROLS",
    "O4_": "POST_FREEZE_DYNAMIC_VALUATION",
    "A1_": "POST_FREEZE_COMPONENT_ATTRIBUTION",
    "A2_": "POST_FREEZE_SET_PLANNING",
    "A3_": "POST_FREEZE_FEEDBACK",
    "A4_": "FROZEN_OUTER_EVALUATION",
}


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
    semantic_errors: tuple[str, ...]


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
    p15 = root / "results/mavis_science_closure/p15_value_stability/model_metrics.csv"
    with p15.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for learner in ("ridge", "huber", "shallow_mlp"):
        learner_values = [
            float(row["full_state_mae"]) for row in rows if row["learner"] == learner
        ]
        if len(learner_values) != 6:
            raise ValueError(f"P15 must contain six domain MAEs for {learner}")
        values.append(sum(learner_values) / len(learner_values))
    return values


def evidence_chronology_rows(root: Path) -> list[dict[str, str]]:
    """Classify each canonical claim by its role relative to the P7 freeze."""
    rows: list[dict[str, str]] = []
    for metric in aei_paper_evidence.build_canonical_metrics(root):
        prefix = next(
            (
                candidate
                for candidate in _SOURCE_STAGES
                if metric.claim_id.startswith(candidate)
            ),
            None,
        )
        if prefix is None:
            raise ValueError(f"unclassified paper claim: {metric.claim_id}")
        if metric.claim_id.startswith(_PRE_P7_PREFIXES):
            chronology_class = "PRE_P7_FROZEN_EVIDENCE"
            frozen_before = "true"
            created_after = "false"
            used_to_modify = "true"
            allowed = (
                "frozen evidence available before the outer endpoint was finalized"
            )
            forbidden = (
                "post-freeze diagnostic or proof that the outer endpoint was superior"
            )
        elif metric.claim_id == "A4_BASELINE_MINUS_MAVIS":
            chronology_class = "FROZEN_OUTER_ENDPOINT"
            frozen_before = "false"
            created_after = "false"
            used_to_modify = "false"
            allowed = (
                "frozen outer policy endpoint evaluated once under the fixed protocol"
            )
            forbidden = "endpoint selected or modified using post-freeze diagnostics"
        else:
            chronology_class = "POST_P7_DIAGNOSTIC"
            frozen_before = "false"
            created_after = "true"
            used_to_modify = "false"
            allowed = (
                "post-freeze diagnostic using hash-bound frozen states and outcomes; "
                "not used to modify the frozen outer endpoint"
            )
            forbidden = "preregistered confirmatory evidence or input to outer-endpoint selection"
        rows.append(
            {
                "claim_id": metric.claim_id,
                "paper_layer": metric.layer,
                "source_stage": _SOURCE_STAGES[prefix],
                "chronology_class": chronology_class,
                "evidence_frozen_before_p7": frozen_before,
                "analysis_created_after_p7": created_after,
                "used_to_modify_p7": used_to_modify,
                "source_path": metric.source_artifact,
                "manuscript_role": metric.paper_role,
                "allowed_description": allowed,
                "forbidden_description": forbidden,
            }
        )
    return rows


def write_evidence_chronology(root: Path, output_dir: Path) -> None:
    """Write deterministic CSV and human-readable chronology authorities."""
    rows = evidence_chronology_rows(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "PAPER_EVIDENCE_CHRONOLOGY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_CHRONOLOGY_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        value: sum(row["chronology_class"] == value for row in rows)
        for value in (
            "PRE_P7_FROZEN_EVIDENCE",
            "FROZEN_OUTER_ENDPOINT",
            "POST_P7_DIAGNOSTIC",
        )
    }
    lines = [
        "# Paper Evidence Chronology",
        "",
        "The frozen outer endpoint predates all post-freeze diagnostics. The later analyses reuse hash-bound frozen states and outcomes and were not used to re-select or modify that endpoint.",
        "",
        "| Chronology class | Claims | Endpoint-selection role |",
        "|---|---:|---|",
        f"| PRE_P7_FROZEN_EVIDENCE | {counts['PRE_P7_FROZEN_EVIDENCE']} | Available before endpoint freeze |",
        f"| FROZEN_OUTER_ENDPOINT | {counts['FROZEN_OUTER_ENDPOINT']} | Fixed evaluation endpoint |",
        f"| POST_P7_DIAGNOSTIC | {counts['POST_P7_DIAGNOSTIC']} | Not used to modify P7 |",
        "",
        "Post-freeze diagnostics are diagnostic evidence, not preregistered confirmatory evidence.",
        "",
        "| Claim | Layer | Source stage | Class | Source |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {row['claim_id']} | {row['paper_layer']} | {row['source_stage']} | {row['chronology_class']} | `{row['source_path']}` |"
        for row in rows
    )
    (output_dir / "PAPER_EVIDENCE_CHRONOLOGY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def semantic_validation_errors(root: Path) -> list[str]:
    """Return manuscript-review contract violations with stable identifiers."""
    manuscript = (root / "paper_aei_information_hierarchy/main.tex").read_text(
        encoding="utf-8"
    )
    flat = re.sub(r"\s+", " ", manuscript)
    errors: list[str] = []

    required_main = {
        "auebc_normalization": r"\frac{1}{x_{i,K}-x_{i,1}}",
        "effective_budget": r"x_{i,1}<\cdots<x_{i,K}",
        "chronology": "distinct chronological roles",
        "operational_novelty": "under one causal acquisition contract",
        "part_i": r"\subsection{Part I --- From Spatial Morphology to State-Conditioned Task Value}",
        "part_ii": r"\subsection{Part II --- From State-Conditioned Value to Evidence-Calibrated Decisions}",
        "history_control": "acquired-position/history control",
        "reconstruction_control": "registered normalized-RGB-MSE reconstruction objective",
        "deployment_boundary": "not performance-superior",
        "predictor_accuracy_boundary": "substantially less accurate shallow MLP",
        "transfer_heading": "Transfer conditions beyond the present case study",
        "bounded_domains": "across the six held-out experimental domains in the present data program",
    }
    for label, phrase in required_main.items():
        if phrase not in flat:
            errors.append(label)

    chronology_path = (
        root / "artifacts/aei_information_hierarchy/PAPER_EVIDENCE_CHRONOLOGY.csv"
    )
    if not chronology_path.is_file():
        errors.append("chronology_missing")
    else:
        with chronology_path.open(encoding="utf-8", newline="") as stream:
            actual = list(csv.DictReader(stream))
        expected = evidence_chronology_rows(root)
        if actual != expected:
            errors.append("chronology_mismatch")
        if any(
            row["used_to_modify_p7"] != "false"
            for row in actual
            if row["chronology_class"] == "POST_P7_DIAGNOSTIC"
        ):
            errors.append("post_freeze_modified_endpoint")

    narrative_path = (
        root / "artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.csv"
    )
    if not narrative_path.is_file():
        errors.append("narrative_map_missing")
    else:
        with narrative_path.open(encoding="utf-8", newline="") as stream:
            narrative = list(csv.DictReader(stream))
        canonical = [
            metric.claim_id
            for metric in aei_paper_evidence.build_canonical_metrics(root)
        ]
        if [row.get("claim_id") for row in narrative] != canonical:
            errors.append("narrative_map_claim_order")
        if len(narrative) != 39 or any(
            row.get("new_part") not in {"PART_I", "PART_II"} for row in narrative
        ):
            errors.append("narrative_map_stage_coverage")

    closest = root / "results/aei_information_hierarchy/tables/table1_closest_work.csv"
    if not closest.is_file():
        errors.append("closest_work_missing")
    else:
        with closest.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 6 or any(
            row.get("source_status") != "VERIFIED_PRIMARY" for row in rows
        ):
            errors.append("closest_work_source_coverage")

    forbidden = (
        "first adaptive ultrasonic",
        "first ultrasound voi",
        "first task-driven",
        "externally validates the hierarchy",
        "external replication",
        r"\subsection{rq1",
        r"\subsection{rq2",
        r"\subsection{rq3",
        "figure2_usefulness.pdf",
        "figure3_observability.pdf",
        "figure4_actionability.pdf",
        "table3_hierarchy_evidence.tex",
    )
    lower = manuscript.lower()
    errors.extend(f"forbidden:{phrase}" for phrase in forbidden if phrase in lower)
    return errors


def _matches_rounding(value: float, decimals: int, candidates: list[float]) -> bool:
    tolerance = 0.500001 * 10.0 ** (-decimals)
    return any(
        math.isclose(value, candidate, abs_tol=tolerance) for candidate in candidates
    )


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
        and len(tables) == 3
        and len(sections) == 6
        and not unmatched
        and not changed
        and not semantic_validation_errors(root)
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
        semantic_errors=tuple(semantic_validation_errors(root)),
    )


__all__ = [
    "ValidationReport",
    "changed_frozen_paths",
    "evidence_chronology_rows",
    "semantic_validation_errors",
    "unmatched_results_numbers",
    "validate_paper",
    "write_evidence_chronology",
]
