"""Deterministic tables for the AEI task-relevant acquisition paper."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_CANONICAL = "artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv"
_VISIBILITY = "artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.csv"
_AUTHORITY_MANIFEST = "artifacts/mavis_authority/artifact_manifest.json"
_SCAN_MANIFEST = "artifacts/mavis_authority/scan_manifest.csv"

_PROTOCOL_COLUMNS = ("item", "value", "source_artifact", "source_hash")
_RESULT_COLUMNS = (
    "stage",
    "scientific_question",
    "headline_evidence",
    "scope_boundary",
    "source_claim_ids",
    "source_artifact",
    "source_hash",
)


@dataclass(frozen=True)
class TableArtifact:
    """Files belonging to one paper table."""

    csv: Path
    tex: Path
    caption: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if base not in path.parents or not path.is_file():
        raise ValueError(f"missing or invalid table source: {relative}")
    return path


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(
    path: Path, rows: Iterable[dict[str, str]], columns: tuple[str, ...]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_rows(root: Path) -> dict[str, dict[str, str]]:
    rows = _csv_rows(_source(root, _CANONICAL))
    by_claim = {row["claim_id"]: row for row in rows}
    if len(rows) != len(by_claim):
        raise ValueError("canonical paper claim IDs are not unique")
    return by_claim


def _protocol_rows(root: Path) -> list[dict[str, str]]:
    authority_path = _source(root, _AUTHORITY_MANIFEST)
    scan_path = _source(root, _SCAN_MANIFEST)
    canonical_path = _source(root, _CANONICAL)
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if not isinstance(authority, dict):
        raise TypeError("authority manifest must be a JSON object")
    if authority["files"]["scan_manifest.csv"]["sha256"] != _sha256(scan_path):
        raise ValueError("scan roster no longer matches its authority manifest")

    scans = _csv_rows(scan_path)
    domain_order = authority["domain_order"]
    counts = Counter(row["dataset_id"] for row in scans)
    ordered_counts = [counts[domain] for domain in domain_order]
    if len(scans) != authority["specimen_count"] or len(domain_order) != 6:
        raise ValueError("case-study cohort no longer matches the frozen authority")

    sources = {
        _AUTHORITY_MANIFEST: _sha256(authority_path),
        _SCAN_MANIFEST: _sha256(scan_path),
        _CANONICAL: _sha256(canonical_path),
    }

    def row(item: str, value: str, source_artifact: str) -> dict[str, str]:
        return {
            "item": item,
            "value": value,
            "source_artifact": source_artifact,
            "source_hash": sources[source_artifact],
        }

    count_text = ", ".join(str(count) for count in ordered_counts)
    return [
        row(
            "Cohort",
            f"{len(scans)} physical specimens across 6 experimental domains; "
            f"domain counts {count_text}",
            _SCAN_MANIFEST,
        ),
        row(
            "Information representations",
            "Specimen metadata and surface observables; full ultrasonic C-scan "
            "field; 25% normalized-raster sparse field; current partial state",
            _CANONICAL,
        ),
        row(
            "Actions and cost",
            "8 x 8 cells with three acquisition levels; cost is exact unique "
            "native-raster locations divided by native count, with no scanner-time "
            "equivalence",
            _CANONICAL,
        ),
        row(
            "Information boundary",
            "Deployable state contains metadata, acquired positions and content, "
            "legal actions, and exact cost; teacher and oracle information is "
            "retrospective and non-deployable",
            _CANONICAL,
        ),
        row(
            "Evaluation",
            "Strict nested leave-one-domain-out (LODO); outer-domain outcomes are "
            "excluded from training, selection, and calibration",
            _AUTHORITY_MANIFEST,
        ),
        row(
            "Statistics",
            "physical specimen first and held-out domain second; equal-domain "
            "aggregation with synchronized within-domain bootstrap; repeated "
            "computational records are not independent samples",
            _CANONICAL,
        ),
    ]


def _format(value: str, digits: int = 4) -> str:
    number = float(value)
    if 0 < abs(number) < 0.001:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def _visible_stage_claims(root: Path) -> dict[str, list[str]]:
    stages: dict[str, list[str]] = {}
    for row in _csv_rows(_source(root, _VISIBILITY)):
        if row["visibility"] not in {"MAIN_HEADLINE", "MAIN_SUPPORT"}:
            continue
        stages.setdefault(row["compressed_stage"], []).append(row["claim_id"])
    return stages


def _result_rows(root: Path) -> list[dict[str, str]]:
    metrics = _canonical_rows(root)
    stage_claims = _visible_stage_claims(root)

    def metric(claim_id: str) -> dict[str, str]:
        return metrics[claim_id]

    u1 = metric("U1_MATCHED_FIELD")
    u2 = metric("U2_SPARSE_RETENTION")
    u3 = metric("U3_UNIFORM_ORACLE")
    u4_cai = metric("U4_ORACLE_CAI_SPECIFICITY")
    u4_image = metric("U4_ORACLE_IMAGE_SPECIFICITY")
    o2 = metric("O2_TEACHER_TURNOVER")
    u5_huber = metric("U5_RIDGE_HUBER_SPEARMAN")
    u5_mlp = metric("U5_RIDGE_MLP_SPEARMAN")
    o4 = metric("O4_DYNAMIC_MINUS_STATIC")
    o1 = metric("O1_STATIC_SPEARMAN")
    o3_pos = metric("O3_REAL_MINUS_POSITIONS")
    o3_rec = metric("O3_REAL_MINUS_RECONSTRUCTION")
    a1 = metric("A1_VALUATION_SUBSTITUTION")
    a2_greedy = metric("A2_GREEDY_PLANNING_REGRET")
    a2_beam = metric("A2_BEAM4_PLANNING_REGRET")

    definitions = (
        (
            "I-A_SPATIAL_AND_SPARSE",
            "Spatial information and sparse recoverability",
            (
                "Does spatial ultrasonic information improve CAI estimation, and how "
                "much of the gain survives sparse observation?"
            ),
            (
                f"{float(u1['relative_effect']) * 100:.1f}% MAE reduction; "
                f"{float(u2['estimate']) * 100:.1f}% of full-field gain retained"
            ),
            (
                "Matched confirmatory field and retrospective 25% normalized-raster "
                "sampling; normalized cost is not scanner time"
            ),
        ),
        (
            "I-B_TASK_CONDITIONED_SPATIAL_VALUE",
            "Task-conditioned spatial measurement value",
            (
                "Is measurement opportunity spatially heterogeneous and conditioned "
                "on the downstream task?"
            ),
            (
                f"Mechanical vs uniform {_format(u3['estimate'])}; CAI task contrast "
                f"{_format(u4_cai['estimate'])}; image task contrast "
                f"{_format(u4_image['estimate'])}"
            ),
            (
                "Oracle rows and cross-task priorities are retrospective; learned-policy "
                "specificity is outside this test"
            ),
        ),
        (
            "I-C_STATE_AND_PREDICTOR_CONDITIONED_VALUE",
            "State- and predictor-conditioned measurement value",
            (
                "Does measurement value change with accumulated evidence and the "
                "downstream predictor?"
            ),
            (
                f"Best-action turnover {float(o2['estimate']) * 100:.1f}%; value-rank "
                f"agreement {_format(u5_huber['estimate'], 3)} vs "
                f"{_format(u5_mlp['estimate'], 3)}"
            ),
            (
                "Teacher targets are retrospective; the shallow MLP is less accurate, "
                "and equal-accuracy structurally distinct predictors remain unresolved"
            ),
        ),
        (
            "II-A_STATE_CONDITIONED_VALUATION",
            "State-conditioned valuation",
            (
                "Can legal-state conditioning improve next-action valuation over a "
                "static scorer?"
            ),
            (
                f"Dynamic - static regret {_format(o4['estimate'])}; static Spearman "
                f"{_format(o1['estimate'])}"
            ),
            (
                "Same legal actions and exact cost at 18.75%; this does not uniquely "
                "attribute the gain to accumulated measured content"
            ),
        ),
        (
            "II-B_SOURCE_AND_COMPONENT_DECOMPOSITION",
            "Information-source and component decomposition",
            (
                "Which legal-state sources and bounded components account for the "
                "remaining acquisition headroom?"
            ),
            (
                f"Real - history {_format(o3_pos['estimate'])}; real - reconstruction "
                f"{_format(o3_rec['estimate'])}; valuation substitution "
                f"{_format(a1['estimate'])}"
            ),
            (
                "Matched controls preserve position history and exact cost; component "
                "substitutions are retrospective and do not identify a representation "
                "bottleneck"
            ),
        ),
        (
            "II-C_COST_CONSTRAINED_REALIZATION",
            "Cost-constrained set realization",
            (
                "How closely does bounded set realization approach a retrospective "
                "joint near-oracle set?"
            ),
            (
                f"Greedy regret {_format(a2_greedy['estimate'])}; beam-4 regret "
                f"{_format(a2_beam['estimate'])}"
            ),
            (
                "Two-action reachable-pool diagnostic at 6.25%; the near-oracle is "
                "non-deployable and does not establish arbitrary-horizon regret"
            ),
        ),
    )

    result: list[dict[str, str]] = []
    for stage_id, stage, question, evidence, boundary in definitions:
        claims = stage_claims[stage_id]
        claim_rows = [metrics[claim_id] for claim_id in claims]
        sources = list(dict.fromkeys(row["source_artifact"] for row in claim_rows))
        hashes = [_sha256(_source(root, source)) for source in sources]
        result.append(
            {
                "stage": stage,
                "scientific_question": question,
                "headline_evidence": evidence,
                "scope_boundary": boundary,
                "source_claim_ids": ";".join(claims),
                "source_artifact": ";".join(sources),
                "source_hash": ";".join(hashes),
            }
        )
    return result


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex(value: str) -> str:
    return "".join(_LATEX_ESCAPES.get(character, character) for character in value)


def _protocol_latex(rows: list[dict[str, str]]) -> str:
    body = "\n".join(
        f"{_latex(row['item'])} & {_latex(row['value'])} \\\\" for row in rows
    )
    return f"""\\begin{{table*}}[t]
\\centering
\\caption{{Case study and evaluation protocol.}}
\\label{{tab:case-protocol}}
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabularx}}{{\\textwidth}}{{@{{}}>{{\\RaggedRight\\arraybackslash}}p{{0.20\\textwidth}}X@{{}}}}
\\toprule
Item & Value \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabularx}}
\\end{{table*}}
"""


def _results_latex(rows: list[dict[str, str]]) -> str:
    body = "\n".join(
        " & ".join(
            _latex(row[column])
            for column in (
                "stage",
                "scientific_question",
                "headline_evidence",
                "scope_boundary",
            )
        )
        + r" \\"
        for row in rows
    )
    return f"""\\begin{{table*}}[t]
\\centering
\\caption{{Task-relevant acquisition results across the six-stage evidence chain.}}
\\label{{tab:task-relevant-results}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabularx}}{{\\textwidth}}{{@{{}}>{{\\RaggedRight\\arraybackslash}}p{{0.19\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.22\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.25\\textwidth}}X@{{}}}}
\\toprule
Stage & Question & Headline evidence & Scope boundary \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabularx}}
\\end{{table*}}
"""


_CAPTIONS = {
    "table1": (
        "**Table 1. Case study and protocol.** The cohort, information forms, "
        "action-cost definition, deployability boundary, strict nested "
        "leave-one-domain-out evaluation, and physical-specimen-first statistical "
        "contract are fixed before the main scientific comparisons.\n"
    ),
    "table2": (
        "**Table 2. Six-stage task-relevant acquisition results.** The table "
        "compresses the main-visible evidence from spatial information and sparse "
        "recoverability through state-conditioned valuation, matched source "
        "controls, component substitutions, and cost-constrained set realization; "
        "each row retains its governing scope boundary.\n"
    ),
}


def _write_manifest(output_root: Path, artifacts: dict[str, TableArtifact]) -> Path:
    path = output_root / "TABLE_CHECKSUMS.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("table_id", "artifact_type", "path", "sha256", "bytes"),
            lineterminator="\n",
        )
        writer.writeheader()
        for table_id in sorted(artifacts):
            artifact = artifacts[table_id]
            for artifact_type, deliverable in (
                ("csv", artifact.csv),
                ("tex", artifact.tex),
                ("caption", artifact.caption),
            ):
                writer.writerow(
                    {
                        "table_id": table_id,
                        "artifact_type": artifact_type,
                        "path": deliverable.name,
                        "sha256": _sha256(deliverable),
                        "bytes": deliverable.stat().st_size,
                    }
                )
    return path


def build_paper_tables(root: Path, output_root: Path) -> dict[str, TableArtifact]:
    """Generate the two main-table CSV, booktabs LaTeX, and caption files."""
    root = root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    table1_rows = _protocol_rows(root)
    table2_rows = _result_rows(root)

    table1 = TableArtifact(
        csv=output_root / "table1_case_protocol.csv",
        tex=output_root / "table1_case_protocol.tex",
        caption=output_root / "table1_case_protocol_caption.md",
    )
    table2 = TableArtifact(
        csv=output_root / "table2_task_relevant_results.csv",
        tex=output_root / "table2_task_relevant_results.tex",
        caption=output_root / "table2_task_relevant_results_caption.md",
    )
    _write_csv(table1.csv, table1_rows, _PROTOCOL_COLUMNS)
    _write_csv(table2.csv, table2_rows, _RESULT_COLUMNS)
    table1.tex.write_text(_protocol_latex(table1_rows), encoding="utf-8", newline="\n")
    table2.tex.write_text(_results_latex(table2_rows), encoding="utf-8", newline="\n")
    table1.caption.write_text(_CAPTIONS["table1"], encoding="utf-8", newline="\n")
    table2.caption.write_text(_CAPTIONS["table2"], encoding="utf-8", newline="\n")
    artifacts = {"table1": table1, "table2": table2}
    _write_manifest(output_root, artifacts)
    return artifacts
