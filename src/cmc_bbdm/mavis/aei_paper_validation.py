from __future__ import annotations

import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cmc_bbdm.mavis import aei_paper_evidence

_BASE_COMMIT = "ff4730b3fcf368d6ac43f0f72f034703e1556f7d"
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
_ALLOWED_PROTOCOL_VALUES = (6.25, 18.75)
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
    main_visible_claim_count: int
    main_mapped_claim_count: int
    combined_mapped_claim_count: int
    figure_count: int
    table_count: int
    section_count: int
    unmatched_numbers: tuple[str, ...]
    changed_frozen_files: tuple[str, ...]
    semantic_errors: tuple[str, ...]


def _visible_text(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _claim_ids(text: str) -> set[str]:
    return set(re.findall(r"\b[UOA][1-5]_[A-Z0-9_]+\b", text))


def _visibility_rows(root: Path) -> list[dict[str, str]]:
    path = root / "artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


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
    errors: list[str] = []

    visible = _visible_text(manuscript)
    visible_flat = re.sub(r"\s+", " ", visible)
    required_main = {
        "title": (
            "Task-Relevant Ultrasonic Information Acquisition for Impacted "
            "Composites: From Spatial Information to State-Conditioned Sensing"
        ),
        "auebc_normalization": r"\frac{1}{x_{i,K}-x_{i,1}}",
        "effective_budget": r"x_{i,1}<\cdots<x_{i,K}",
        "operational_novelty": "under one causal acquisition contract",
        "part_i": r"\subsection{Task-Relevant Information Characterization}",
        "part_ii": r"\subsection{State-Conditioned Task-Oriented Acquisition}",
        "history_control": "acquired-position/history control",
        "reconstruction_control": "registered normalized-RGB-MSE reconstruction objective",
        "deployment_boundary": "this endpoint is an implementation boundary",
        "predictor_accuracy_boundary": "substantially less accurate shallow MLP",
        "scope_cost": "exact native-raster acquisition cost",
        "scope_domains": "six CFRP experimental domains",
        "scope_conditioning": "downstream CAI predictor and the registered action space",
        "scope_validation": "prospective validation under the corresponding measurement process",
        "figure4": "figure4_valuation_planning_realization.pdf",
        "table1": r"\input{tables/table1_case_protocol.tex}",
        "table2": r"\input{tables/table2_task_relevant_results.tex}",
    }
    for label, phrase in required_main.items():
        if phrase not in visible_flat:
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

    visibility = _visibility_rows(root)
    canonical_claims = {
        metric.claim_id for metric in aei_paper_evidence.build_canonical_metrics(root)
    }
    if (
        len(visibility) != 39
        or {row.get("claim_id") for row in visibility} != canonical_claims
    ):
        errors.append("visibility_claim_coverage")
    visibility_counts = {
        value: sum(row.get("visibility") == value for row in visibility)
        for value in (
            "MAIN_HEADLINE",
            "MAIN_SUPPORT",
            "MAIN_SYSTEM_DIAGNOSTIC",
            "SUPPLEMENT_ONLY",
        )
    }
    if visibility_counts != {
        "MAIN_HEADLINE": 12,
        "MAIN_SUPPORT": 15,
        "MAIN_SYSTEM_DIAGNOSTIC": 1,
        "SUPPLEMENT_ONLY": 11,
    }:
        errors.append("visibility_partition")

    literature = root / "artifacts/mavis_science_closure/LITERATURE_LEDGER.md"
    if not literature.is_file():
        errors.append("literature_ledger_missing")
    else:
        literature_text = literature.read_text(encoding="utf-8")
        if literature_text.count("- Primary source") != 6:
            errors.append("literature_primary_source_coverage")

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
        "figure4_decision_calibration.pdf",
        "table1_closest_work.tex",
        "table3_progressive_evidence_chain.tex",
        "distinct chronological roles",
        "were not used to re-select or modify the frozen outer endpoint",
        "transfer conditions beyond the present case study",
        "not proof of universal empirical transfer",
        "travel, coupling, settling",
        "selected image regions do not identify causal failure mechanisms",
        "these boundaries specify what must be revalidated",
    )
    lower = visible.lower()
    errors.extend(f"forbidden:{phrase}" for phrase in forbidden if phrase in lower)
    introduction = lower.split(r"\section{introduction}", maxsplit=1)[1].split(
        r"\section{related work}", maxsplit=1
    )[0]
    errors.extend(
        f"introduction_audit_residue:{phrase}"
        for phrase in ("frozen", "post-freeze", "hash-bound")
        if phrase in introduction
    )
    for phrase in ("mavis", "mvd_m1_o2"):
        if phrase in lower:
            errors.append(f"internal_identity:{phrase}")
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
    supplement = (
        root / "paper_aei_information_hierarchy/supplementary/supplementary.tex"
    ).read_text(encoding="utf-8")
    canonical = {
        row.claim_id for row in aei_paper_evidence.build_canonical_metrics(root)
    }
    visibility = _visibility_rows(root)
    main_visible = {
        row["claim_id"]
        for row in visibility
        if row["visibility"] in {"MAIN_HEADLINE", "MAIN_SUPPORT"}
    }
    main_mapped = _claim_ids(manuscript)
    combined_mapped = _claim_ids(manuscript + "\n" + supplement)
    figures = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", manuscript)
    tables = re.findall(r"\\input\{tables/[^}]+\}", manuscript)
    sections = re.findall(r"^\\section\{[^}]+\}", manuscript, flags=re.MULTILINE)
    unmatched = tuple(unmatched_results_numbers(root))
    changed = tuple(changed_frozen_paths(root))
    passed = (
        main_mapped == main_visible
        and combined_mapped == canonical
        and len(figures) == 4
        and len(tables) == 2
        and len(sections) == 6
        and not unmatched
        and not changed
        and not semantic_validation_errors(root)
    )
    return ValidationReport(
        passed=passed,
        canonical_claim_count=len(canonical),
        main_visible_claim_count=len(main_visible),
        main_mapped_claim_count=len(main_mapped),
        combined_mapped_claim_count=len(combined_mapped),
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
