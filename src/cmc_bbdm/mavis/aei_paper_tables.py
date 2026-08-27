"""Deterministic, evidence-bound tables for the AEI information hierarchy."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_CANONICAL = "artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv"
_AUTHORITY_MANIFEST = "artifacts/mavis_authority/artifact_manifest.json"
_SCAN_MANIFEST = "artifacts/mavis_authority/scan_manifest.csv"
_P10_CONTRASTS = "results/mavis_science_closure/p10_mris_causal/contrasts.csv"
_LITERATURE_LEDGER = "artifacts/mavis_science_closure/LITERATURE_LEDGER.md"
_CLOSEST_WORK_COLUMNS = (
    "work",
    "citation_key",
    "measurement_setting",
    "acquisition_objective",
    "state_dependence",
    "downstream_endpoint",
    "outside_scope",
    "primary_source",
    "source_status",
    "source_artifact",
    "source_hash",
)
_HIERARCHY_COLUMNS = (
    "layer",
    "question",
    "key_comparison",
    "effect",
    "ci95",
    "domains",
    "evidence_type",
    "conclusion",
    "source_claim_ids",
    "source_artifacts",
    "source_hashes",
    "canonical_authority_hash",
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


def _canonical_rows(root: Path) -> tuple[dict[str, dict[str, str]], str]:
    path = _source(root, _CANONICAL)
    rows = _csv_rows(path)
    by_claim = {row["claim_id"]: row for row in rows}
    if len(rows) != len(by_claim):
        raise ValueError("canonical paper claim IDs are not unique")
    return by_claim, _sha256(path)


def _float(row: dict[str, str], key: str) -> float:
    value = row[key]
    if not value:
        raise ValueError(f"canonical field is empty for {row['claim_id']}: {key}")
    return float(value)


def _fmt(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 0.1:
        return f"{value:.3f}"
    if magnitude >= 0.01:
        return f"{value:.4f}"
    if magnitude >= 0.001:
        return f"{value:.5f}"
    return f"{value:.3e}"


def _ci(row: dict[str, str]) -> str:
    if not row["ci95_lower"] or not row["ci95_upper"]:
        return "descriptive; no interval"
    return f"[{_fmt(float(row['ci95_lower']))}, {_fmt(float(row['ci95_upper']))}]"


def _provenance(rows: list[dict[str, str]]) -> tuple[str, str, str]:
    return (
        ";".join(row["claim_id"] for row in rows),
        ";".join(dict.fromkeys(row["source_artifact"] for row in rows)),
        ";".join(dict.fromkeys(row["source_hash"] for row in rows)),
    )


def _hierarchy_row(
    canonical_hash: str,
    *,
    layer: str,
    question: str,
    comparison: str,
    effect: str,
    ci95: str,
    domains: str,
    evidence_type: str,
    conclusion: str,
    claims: list[dict[str, str]],
) -> dict[str, str]:
    claim_ids, sources, hashes = _provenance(claims)
    return {
        "layer": layer,
        "question": question,
        "key_comparison": comparison,
        "effect": effect,
        "ci95": ci95,
        "domains": domains,
        "evidence_type": evidence_type,
        "conclusion": conclusion,
        "source_claim_ids": claim_ids,
        "source_artifacts": sources,
        "source_hashes": hashes,
        "canonical_authority_hash": canonical_hash,
    }


def _closest_work_rows(root: Path) -> list[dict[str, str]]:
    ledger = _source(root, _LITERATURE_LEDGER)
    source_hash = _sha256(ledger)

    def row(
        work: str,
        citation_key: str,
        measurement_setting: str,
        acquisition_objective: str,
        state_dependence: str,
        downstream_endpoint: str,
        outside_scope: str,
        primary_source: str,
    ) -> dict[str, str]:
        return {
            "work": work,
            "citation_key": citation_key,
            "measurement_setting": measurement_setting,
            "acquisition_objective": acquisition_objective,
            "state_dependence": state_dependence,
            "downstream_endpoint": downstream_endpoint,
            "outside_scope": outside_scope,
            "primary_source": primary_source,
            "source_status": "VERIFIED_PRIMARY",
            "source_artifact": _LITERATURE_LEDGER,
            "source_hash": source_hash,
        }

    return [
        row(
            "Fuentes et al. (2020)",
            "fuentes2020autonomous",
            "robotic ultrasonic A-scan features sampled over a C-scan surface",
            "Bayesian-optimization search over novelty and damage probability",
            "sequential posterior update and next-location selection",
            "damage and novelty localization",
            "CAI-conditioned value and held-out-domain closed-loop utility were not study objectives",
            "https://strathprints.strath.ac.uk/72351/",
        ),
        row(
            "Cantero-Chinchilla et al. (2020)",
            "cantero2020optimal",
            "ultrasonic guided-wave structural-monitoring sensors",
            "expected value of information with sensor-number and placement cost",
            "static sensor configuration",
            "Bayesian damage diagnosis and localization",
            "within-specimen partial C-scan reveal and CAI prediction were not study objectives",
            "https://www.sciencedirect.com/science/article/abs/pii/S0888327019305989",
        ),
        row(
            "Memarzadeh and Pozzi (2016)",
            "memarzadeh2016value",
            "probabilistic component inspection and permanent monitoring",
            "value of information for inspection and system scheduling",
            "sequential belief and decision updates",
            "maintenance and inspection scheduling",
            "ultrasonic imaging and CAI mechanics were not study objectives",
            "https://www.sciencedirect.com/science/article/abs/pii/S0951832016300771",
        ),
        row(
            "Blumberg et al. (2024; TADRED)",
            "blumberg2024experimental",
            "densely sampled multi-channel imaging followed by sparse channel design",
            "task-driven feature and channel selection",
            "static global subset design",
            "user-specified downstream image-analysis task",
            "state-dependent within-specimen causal reveal and exact-cost sequential planning were outside scope",
            "https://openreview.net/forum?id=MloaGA6WwX",
        ),
        row(
            "Ji et al. (2026)",
            "ji2026adaptive",
            "sparse SLDV Lamb wavefield measurements on composite structures",
            "Bayesian-optimization-guided sampling for STMAE reconstruction",
            "adaptive spatial sampling-pattern construction",
            "full wavefield reconstruction and damage-area fidelity",
            "CAI mechanics and predictor-conditioned acquisition value were not study objectives",
            "https://www.sciencedirect.com/science/article/abs/pii/S0041624X26000247",
        ),
        row(
            "Mack et al. (2026)",
            "mack2026cscan",
            "complete ultrasonic C-scan images of impacted composites",
            "no acquisition objective; full scans are inputs",
            "static full-field prediction",
            "impact energy and CAI-strength regression",
            "sparse or adaptive acquisition and conditional next-measurement value were outside scope",
            "https://www.sciencedirect.com/science/article/abs/pii/S1474034626002107",
        ),
    ]


def _protocol_rows(root: Path, canonical_hash: str) -> list[dict[str, str]]:
    authority_path = _source(root, _AUTHORITY_MANIFEST)
    scan_path = _source(root, _SCAN_MANIFEST)
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if type(authority) is not dict:
        raise ValueError("MAVIS authority manifest must be a JSON object")
    if authority["files"]["scan_manifest.csv"]["sha256"] != _sha256(scan_path):
        raise ValueError("MAVIS scan roster no longer matches its authority manifest")
    scans = _csv_rows(scan_path)
    order = authority["domain_order"]
    counts = {domain: 0 for domain in order}
    for row in scans:
        counts[row["dataset_id"]] += 1
    if sum(counts.values()) != authority["specimen_count"] or len(scans) != 276:
        raise ValueError("MAVIS table cohort is no longer 276 physical specimens")
    roster = "; ".join(f"{domain}: {counts[domain]}" for domain in order)

    checkpoints = sorted(
        {
            float(row["nominal_checkpoint"])
            for row in _csv_rows(_source(root, _P10_CONTRASTS))
        }
    )
    checkpoint_text = ", ".join(f"{value * 100:g}%" for value in checkpoints)
    source_info = {
        _AUTHORITY_MANIFEST: _sha256(authority_path),
        _SCAN_MANIFEST: _sha256(scan_path),
        _CANONICAL: canonical_hash,
        _P10_CONTRASTS: _sha256(_source(root, _P10_CONTRASTS)),
    }

    def item(name: str, value: str, source_artifact: str) -> dict[str, str]:
        return {
            "item": name,
            "value": value,
            "source_artifact": source_artifact,
            "source_hash": source_info[source_artifact],
        }

    return [
        item(
            "Physical cohort",
            "276 CAI-complete physical specimens",
            _AUTHORITY_MANIFEST,
        ),
        item("Held-out domains", "6 experimental domains", _AUTHORITY_MANIFEST),
        item("Domain specimen counts", roster, _SCAN_MANIFEST),
        item(
            "Available modalities",
            "specimen metadata; surface observables; ultrasonic RGB C-scan internal field; CAI-ratio outcome",
            _CANONICAL,
        ),
        item(
            "Validation protocol",
            "strict nested leave-one-domain-out (LODO); outer-domain outcomes excluded from training and selection",
            _CANONICAL,
        ),
        item("Registered checkpoints", checkpoint_text, _P10_CONTRASTS),
        item(
            "Acquisition cost",
            "exact unique newly observed native-raster locations divided by native count; no scanner-time equivalence",
            _CANONICAL,
        ),
        item(
            "Teacher/oracle information",
            "future outcomes, unmeasured fields, or counterfactual utilities; retrospective and non-deployable",
            _CANONICAL,
        ),
        item(
            "Deployable information",
            "specimen metadata, acquired positions, measured content, legal actions, and current exact cost",
            _CANONICAL,
        ),
        item(
            "Statistical units",
            "physical specimen first; held-out domain second; six domains receive equal weight",
            _CANONICAL,
        ),
        item(
            "Bootstrap aggregation",
            "specimens resampled within domain with synchronized contrasts; domain means then equally weighted",
            _CANONICAL,
        ),
        item(
            "Computational rows",
            "state-action and checkpoint rows are repeated computational records, not independent statistical samples",
            _CANONICAL,
        ),
    ]


def _hierarchy_rows(root: Path) -> list[dict[str, str]]:
    metrics, canonical_hash = _canonical_rows(root)

    def claims(*ids: str) -> list[dict[str, str]]:
        return [metrics[claim_id] for claim_id in ids]

    u1 = metrics["U1_MATCHED_FIELD"]
    u2_retention = metrics["U2_SPARSE_RETENTION"]
    u2_gap = metrics["U2_SPARSE_FULL_GAP"]
    u3_uniform = metrics["U3_UNIFORM_ORACLE"]
    u3_reconstruction = metrics["U3_RECONSTRUCTION_ORACLE"]
    u4_cai = metrics["U4_ORACLE_CAI_SPECIFICITY"]
    u4_image = metrics["U4_ORACLE_IMAGE_SPECIFICITY"]
    u5_huber = metrics["U5_RIDGE_HUBER_SPEARMAN"]
    u5_mlp = metrics["U5_RIDGE_MLP_SPEARMAN"]
    o1 = metrics["O1_STATIC_SPEARMAN"]
    o2_turnover = metrics["O2_TEACHER_TURNOVER"]
    o2_rank = metrics["O2_TEACHER_RANK"]
    o2_topk = metrics["O2_TEACHER_TOPK"]
    o3_positions = metrics["O3_REAL_MINUS_POSITIONS"]
    o3_reconstruction = metrics["O3_REAL_MINUS_RECONSTRUCTION"]
    o4_static = metrics["O4_DYNAMIC_MINUS_STATIC"]
    o4_shuffled = metrics["O4_DYNAMIC_MINUS_SHUFFLED"]
    a1_value = metrics["A1_VALUATION_SUBSTITUTION"]
    a1_learned = metrics["A1_LEARNED_PLANNING_SUBSTITUTION"]
    a1_true = metrics["A1_TRUE_VALUE_PLANNING_SUBSTITUTION"]
    a2 = metrics["A2_GREEDY_PLANNING_REGRET"]
    a3 = metrics["A3_FEEDBACK_BENEFIT"]
    a4 = metrics["A4_BASELINE_MINUS_MAVIS"]

    useful_question = "Does it improve the task?"
    observable_question = "Is value observable from legal state?"
    actionable_question = "Does it improve a bounded decision?"
    return [
        _hierarchy_row(
            canonical_hash,
            layer="Useful",
            question=useful_question,
            comparison="matched scalar vs selected B-family spatial field",
            effect=f"MAE reduction {_fmt(_float(u1, 'estimate'))} ({_float(u1, 'relative_effect') * 100:.1f}%)",
            ci95=_ci(u1),
            domains=u1["domains_improved"],
            evidence_type="registered confirmatory",
            conclusion="Spatial internal morphology preserves CAI-relevant information absent from the matched scalar representation.",
            claims=claims("U1_MATCHED_FIELD"),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Useful",
            question=useful_question,
            comparison="surface vs sparse field; sparse vs full field",
            effect=f"{_float(u2_retention, 'estimate') * 100:.1f}% gain retained; full-field gap {_fmt(_float(u2_gap, 'estimate'))}",
            ci95=f"gap {_ci(u2_gap)}",
            domains="5/6 vs surface; 0/6 vs full",
            evidence_type="registered retrospective sparse protocol",
            conclusion="Sparse measurements retain most, but not all, registered full-field gain; no scanner-time claim is made.",
            claims=claims(
                "U2_SPARSE_RETENTION", "U2_SPARSE_GAIN", "U2_SPARSE_FULL_GAP"
            ),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Useful",
            question=useful_question,
            comparison="mechanical oracle vs uniform and reconstruction oracles",
            effect=f"CAI-AUEBC improvements {_fmt(_float(u3_uniform, 'estimate'))} and {_fmt(_float(u3_reconstruction, 'estimate'))}",
            ci95=f"{_ci(u3_uniform)}; {_ci(u3_reconstruction)}",
            domains="6/6 for both",
            evidence_type="retrospective one-shot oracle",
            conclusion="Specimen-specific mechanical acquisition headroom exists, but the oracle is non-deployable.",
            claims=claims(
                "U3_UNIFORM_ORACLE", "U3_RECONSTRUCTION_ORACLE", "U3_HEADROOM_RETENTION"
            ),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Useful",
            question=useful_question,
            comparison="mechanics and reconstruction task-specific oracles; learned global masks",
            effect=f"CAI {_fmt(_float(u4_cai, 'estimate'))}; image error {_fmt(_float(u4_image, 'estimate'))}; learned indicator 0",
            ci95=f"{_ci(u4_cai)}; {_ci(u4_image)}",
            domains="6-domain contrasts; learned separation unsupported",
            evidence_type="retrospective cross-objective oracle + learned adverse control",
            conclusion="Oracle task specificity holds; learned global masks do not reproduce the separation.",
            claims=claims(
                "U4_ORACLE_CAI_SPECIFICITY",
                "U4_ORACLE_IMAGE_SPECIFICITY",
                "U4_LEARNED_SPECIFICITY_BOUNDARY",
            ),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Useful",
            question=useful_question,
            comparison="Ridge-Huber vs Ridge-MLP action-value rankings",
            effect=f"Spearman rho {_fmt(_float(u5_huber, 'estimate'))} vs {_fmt(_float(u5_mlp, 'estimate'))}",
            ci95=f"{_ci(u5_huber)}; {_ci(u5_mlp)}",
            domains="6 held-out domains",
            evidence_type="strict-OOF learner sensitivity",
            conclusion="Value is relatively stable for similarly accurate Ridge and Huber predictors but not for the less accurate shallow MLP; equally accurate structurally distinct predictors remain unresolved.",
            claims=claims("U5_RIDGE_HUBER_SPEARMAN", "U5_RIDGE_MLP_SPEARMAN"),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Observable",
            question=observable_question,
            comparison="static legal-state scorer vs strict-OOF teacher values",
            effect=f"Spearman rho {_fmt(_float(o1, 'estimate'))}",
            ci95=_ci(o1),
            domains=o1["domains_improved"],
            evidence_type="strict-OOF static scorer",
            conclusion="Static value observability is not supported.",
            claims=claims(
                "O1_STATIC_SPEARMAN",
                "O1_STATIC_SET_REGRET",
                "O1_GLOBAL_SET_REGRET",
                "O1_RANDOM_SET_REGRET",
            ),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Observable",
            question=observable_question,
            comparison="conditional teacher values from initial to final tested checkpoint",
            effect=f"turnover {_fmt(_float(o2_turnover, 'estimate'))}; rank {_fmt(_float(o2_rank, 'estimate'))}; top-5 {_fmt(_float(o2_topk, 'estimate'))}",
            ci95="not estimated",
            domains="276 specimens; 6 domains",
            evidence_type="strict-OOF retrospective teacher",
            conclusion="True conditional measurement value evolves with acquisition state.",
            claims=claims(
                "O2_TEACHER_TURNOVER",
                "O2_TEACHER_RANK",
                "O2_TEACHER_TOPK",
                "O2_TEACHER_OPPORTUNITY",
            ),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Observable",
            question=observable_question,
            comparison="measured content vs matched positions and reconstruction controls",
            effect=f"real minus position MAE {_fmt(_float(o3_positions, 'estimate'))}; real minus reconstruction {_fmt(_float(o3_reconstruction, 'estimate'))}",
            ci95=f"{_ci(o3_positions)}; {_ci(o3_reconstruction)}",
            domains="1/6 favors measured content for each control",
            evidence_type="matched representation adverse controls",
            conclusion="Measured content is worse than positions and reconstruction controls; beyond-geometry value is not established.",
            claims=claims("O3_REAL_MINUS_POSITIONS", "O3_REAL_MINUS_RECONSTRUCTION"),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Observable",
            question=observable_question,
            comparison="conditional real-state regret vs static and shuffled controls",
            effect=f"vs static {_fmt(_float(o4_static, 'estimate'))}; vs shuffled {_fmt(_float(o4_shuffled, 'estimate'))}",
            ci95=f"{_ci(o4_static)}; {_ci(o4_shuffled)}",
            domains="5/6 vs static; 1/6 vs shuffled",
            evidence_type="frozen dynamic valuation controls",
            conclusion="Conditional scoring narrowly beats static, but shuffled content remains better.",
            claims=claims("O4_DYNAMIC_MINUS_STATIC", "O4_DYNAMIC_MINUS_SHUFFLED"),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Actionable",
            question=actionable_question,
            comparison="retrospective valuation and planning substitutions",
            effect=f"valuation {_fmt(_float(a1_value, 'estimate'))}; learned planning {_fmt(_float(a1_learned, 'estimate'))}; true-value planning {_fmt(_float(a1_true, 'estimate'))}",
            ci95=f"{_ci(a1_value)}; {_ci(a1_learned)}; {_ci(a1_true)}",
            domains="6-domain equal weighting",
            evidence_type="retrospective component substitution",
            conclusion="Valuation and bounded planning gaps exist; a representation bottleneck is not established.",
            claims=claims(
                "A1_VALUATION_SUBSTITUTION",
                "A1_LEARNED_PLANNING_SUBSTITUTION",
                "A1_TRUE_VALUE_PLANNING_SUBSTITUTION",
            ),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Actionable",
            question=actionable_question,
            comparison="current greedy vs retrospective joint near-oracle reachable pool",
            effect=f"planning regret {_fmt(_float(a2, 'estimate'))}",
            ci95=_ci(a2),
            domains="6-domain equal weighting",
            evidence_type="retrospective bounded two-action planning",
            conclusion="A bounded set-planning gap remains; the near-oracle is non-deployable.",
            claims=claims("A2_GREEDY_PLANNING_REGRET", "A2_BEAM4_PLANNING_REGRET"),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Actionable",
            question=actionable_question,
            comparison="no-feedback minus feedback under the frozen policy",
            effect=f"feedback benefit {_fmt(_float(a3, 'estimate'))}",
            ci95=_ci(a3),
            domains=a3["domains_improved"],
            evidence_type="frozen feedback control",
            conclusion="Feedback is adverse under the frozen cross-domain protocol.",
            claims=claims("A3_FEEDBACK_BENEFIT"),
        ),
        _hierarchy_row(
            canonical_hash,
            layer="Actionable",
            question=actionable_question,
            comparison="strongest deployable baseline minus frozen learned policy",
            effect=f"CAI-AUEBC effect {_fmt(_float(a4, 'estimate'))}",
            ci95=_ci(a4),
            domains=a4["domains_improved"],
            evidence_type="frozen deployable endpoint",
            conclusion="The frozen learned policy did not outperform the strongest deployable baseline.",
            claims=claims("A4_BASELINE_MINUS_MAVIS"),
        ),
    ]


_LATEX_ESCAPES = {
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


def _closest_work_latex(rows: list[dict[str, str]]) -> str:
    body = "\n".join(
        " & ".join(
            (
                f"{_latex(row['work'])} \\cite{{{row['citation_key']}}}",
                _latex(row["measurement_setting"]),
                _latex(row["acquisition_objective"]),
                _latex(row["state_dependence"]),
                _latex(row["downstream_endpoint"]),
                _latex(row["outside_scope"]),
            )
        )
        + r" \\"
        for row in rows
    )
    return f"""\\begin{{table*}}[!tp]
\\centering
\\caption{{Closest-work positioning by measurement setting, acquisition objective, state dependence, downstream endpoint, and scope. The present study jointly tests three non-equivalent information claims under one causal acquisition contract; it does not claim the first adaptive ultrasonic, ultrasound-VoI, or task-driven design.}}
\\label{{tab:closest-work}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.8pt}}
\\begin{{tabularx}}{{\\textwidth}}{{@{{}}>{{\\RaggedRight\\arraybackslash}}p{{0.13\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.15\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.15\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.13\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.14\\textwidth}}Y@{{}}}}
\\toprule
Work & Measurement & Objective & State dependence & Endpoint & Outside scope \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabularx}}
\\end{{table*}}
"""


def _protocol_latex(rows: list[dict[str, str]]) -> str:
    body = "\n".join(
        f"{_latex(row['item'])} & {_latex(row['value'])} \\\\" for row in rows
    )
    return f"""\\begin{{table*}}[t]
\\centering
\\caption{{Multi-domain CFRP case study and evaluation protocol.}}
\\label{{tab:case-protocol}}
\\small
\\begin{{tabularx}}{{\\textwidth}}{{@{{}}>{{\\RaggedRight\\arraybackslash}}p{{0.22\\textwidth}}Y@{{}}}}
\\toprule
Item & Protocol value \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabularx}}
\\end{{table*}}
"""


def _hierarchy_latex(rows: list[dict[str, str]]) -> str:
    body_lines: list[str] = []
    previous_layer: str | None = None
    for row in rows:
        new_layer = row["layer"] != previous_layer
        if previous_layer is not None and new_layer:
            body_lines.append(r"\addlinespace[2pt]")
        values = (
            row["layer"] if new_layer else "",
            row["question"] if new_layer else "",
            row["key_comparison"],
            row["effect"],
            row["ci95"],
            row["domains"],
            row["evidence_type"],
            row["conclusion"],
        )
        body_lines.append(" & ".join(_latex(value) for value in values) + r" \\")
        previous_layer = row["layer"]
    body = "\n".join(body_lines)
    return f"""\\begingroup
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{longtable}}{{@{{}}>{{\\RaggedRight\\arraybackslash}}p{{0.105\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.085\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.158\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.12\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.10\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.082\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.12\\textwidth}}>{{\\RaggedRight\\arraybackslash}}p{{0.137\\textwidth}}@{{}}}}
\\caption{{Evidence summary for the task-relevant information hierarchy. Positive and negative directions are stated in the comparison text.}}
\\label{{tab:hierarchy-evidence}}\\\\
\\toprule
Layer & Question & Key comparison & Effect & 95\\% CI & Domains & Evidence type & Conclusion \\\\
\\midrule
\\endfirsthead
\\multicolumn{{8}}{{@{{}}l}}{{\\tablename~\\thetable{{}} (continued)}} \\\\
\\toprule
Layer & Question & Key comparison & Effect & 95\\% CI & Domains & Evidence type & Conclusion \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{{8}}{{r@{{}}}}{{Continued on next page}} \\\\
\\endfoot
\\bottomrule
\\endlastfoot
{body}
\\end{{longtable}}
\\endgroup
"""


_CAPTIONS = {
    "table1": (
        "**Table 1. Closest-work positioning.** The comparison separates measurement "
        "setting, acquisition objective, state dependence, downstream endpoint, and "
        "scope. The contribution is an operational joint test of three non-equivalent "
        "information claims, not a first adaptive-ultrasound or generic-VoI claim.\n"
    ),
    "table2": (
        "**Table 2. Multi-domain CFRP case study and evaluation protocol.** "
        "The physical specimen and held-out domain are the statistical units; "
        "state-action rows are repeated computational records. Acquisition cost "
        "uses exact native-raster locations and is not scanner-time equivalence.\n"
    ),
    "table3": (
        "**Table 3. Evidence summary for the task-relevant information hierarchy.** "
        "Registered and retrospective usefulness evidence is separated from legal-state "
        "observability and deployable actionability. Central adverse controls remain in "
        "the main table; retrospective teachers, oracles, and substitutions are not "
        "deployable policies.\n"
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
    """Generate the three main-table CSV, booktabs LaTeX, and caption files."""
    root = root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _, canonical_hash = _canonical_rows(root)
    table1_rows = _closest_work_rows(root)
    table2_rows = _protocol_rows(root, canonical_hash)
    table3_rows = _hierarchy_rows(root)

    table1 = TableArtifact(
        csv=output_root / "table1_closest_work.csv",
        tex=output_root / "table1_closest_work.tex",
        caption=output_root / "table1_closest_work_caption.md",
    )
    table2 = TableArtifact(
        csv=output_root / "table2_case_protocol.csv",
        tex=output_root / "table2_case_protocol.tex",
        caption=output_root / "table2_case_protocol_caption.md",
    )
    table3 = TableArtifact(
        csv=output_root / "table3_hierarchy_evidence.csv",
        tex=output_root / "table3_hierarchy_evidence.tex",
        caption=output_root / "table3_hierarchy_evidence_caption.md",
    )
    _write_csv(table1.csv, table1_rows, _CLOSEST_WORK_COLUMNS)
    _write_csv(
        table2.csv, table2_rows, ("item", "value", "source_artifact", "source_hash")
    )
    _write_csv(table3.csv, table3_rows, _HIERARCHY_COLUMNS)
    table1.tex.write_text(
        _closest_work_latex(table1_rows), encoding="utf-8", newline="\n"
    )
    table2.tex.write_text(_protocol_latex(table2_rows), encoding="utf-8", newline="\n")
    table3.tex.write_text(
        _hierarchy_latex(table3_rows), encoding="utf-8", newline="\n"
    )
    table1.caption.write_text(_CAPTIONS["table1"], encoding="utf-8", newline="\n")
    table2.caption.write_text(_CAPTIONS["table2"], encoding="utf-8", newline="\n")
    table3.caption.write_text(_CAPTIONS["table3"], encoding="utf-8", newline="\n")
    artifacts = {"table1": table1, "table2": table2, "table3": table3}
    _write_manifest(output_root, artifacts)
    return artifacts
