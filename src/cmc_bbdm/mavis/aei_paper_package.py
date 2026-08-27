from __future__ import annotations

import csv
import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

_PAPER = Path("paper_aei_information_hierarchy")
_RESULTS = Path("results/aei_information_hierarchy")

_MAIN_FILES = (
    "README.md",
    "MANUSCRIPT_OUTLINE.md",
    "CLAIM_SENTENCE_BANK.md",
    "main.tex",
    "references.bib",
    "elsarticle.cls",
    "elsarticle-num.bst",
)

_FIGURES = (
    "figure1_task_relevant_acquisition_framework.pdf",
    "figure2_information_characterization.pdf",
    "figure3_state_conditioned_value.pdf",
    "figure4_decision_calibration.pdf",
)

_TABLES = (
    "table1_closest_work.tex",
    "table2_case_protocol.tex",
    "table3_progressive_evidence_chain.tex",
)

_LEGACY_FIGURES = (
    "figure1_information_hierarchy.pdf",
    "figure2_usefulness.pdf",
    "figure3_observability.pdf",
    "figure4_actionability.pdf",
)

_LEGACY_TABLES = ("table3_hierarchy_evidence.tex",)

_SUPPLEMENTARY_SOURCES = (
    (
        "S01_p1_full_field_domain_metrics.csv",
        "results/p1_full_field_oracle/domain_metrics.csv",
    ),
    (
        "S02_mvd_static_observability_domain_metrics.csv",
        "results/mvd/m1_observability/domain_metrics.csv",
    ),
    (
        "S03_p9_checkpoint_metrics.csv",
        "results/mavis_science_closure/p9_value_evolution/aggregate_metrics.csv",
    ),
    (
        "S04_p9_domain_metrics.csv",
        "results/mavis_science_closure/p9_value_evolution/domain_metrics.csv",
    ),
    (
        "S05_p10_control_matrix.csv",
        "results/mavis_science_closure/p10_mris_causal/state_cost_curve.csv",
    ),
    (
        "S06_p10_control_contrasts.csv",
        "results/mavis_science_closure/p10_mris_causal/contrasts.csv",
    ),
    (
        "S07_p10_domain_metrics.csv",
        "results/mavis_science_closure/p10_mris_causal/domain_metrics.csv",
    ),
    (
        "S08_p11_secondary_metrics.csv",
        "results/mavis_science_closure/p11_dynamic_valuation/regret_by_cost.csv",
    ),
    (
        "S09_p11_domain_metrics.csv",
        "results/mavis_science_closure/p11_dynamic_valuation/domain_metrics.csv",
    ),
    (
        "S10_p12_substitution_matrix.csv",
        "results/mavis_science_closure/p12_rvp_attribution/substitution_matrix.csv",
    ),
    (
        "S11_p12_substitution_per_domain.csv",
        "results/mavis_science_closure/p12_rvp_attribution/per_domain.csv",
    ),
    (
        "S12_p13_planning_per_domain.csv",
        "results/mavis_science_closure/p13_set_planning/per_domain.csv",
    ),
    (
        "S13_p15_learner_metrics.csv",
        "results/mavis_science_closure/p15_value_stability/model_metrics.csv",
    ),
    (
        "S14_p15_learner_pair_metrics.csv",
        "results/mavis_science_closure/p15_value_stability/rank_agreement.csv",
    ),
    (
        "S15_p16_feedback_strata.csv",
        "results/mavis_science_closure/p16_feedback_mechanism/stratum_effects.csv",
    ),
    (
        "S16_p16_feedback_associations.csv",
        "results/mavis_science_closure/p16_feedback_mechanism/association_summary.csv",
    ),
    (
        "S17_provenance_hashes.csv",
        "artifacts/aei_information_hierarchy/PAPER_SOURCE_HASHES.csv",
    ),
    (
        "S18_external_dataset_audit.json",
        "results/mavis/p7_final_frozen_eval/external_data_audit.json",
    ),
)


@dataclass(frozen=True)
class PaperPackage:
    root: Path
    figures: Path
    tables: Path
    supplementary_data: Path
    submission_source: Path
    manifest: Path
    archive: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_manifest(directory: Path, manifest: Path) -> None:
    rows = []
    for path in sorted(item for item in directory.iterdir() if item != manifest):
        if not path.is_file():
            raise ValueError(f"submission source is not flat: {path}")
        rows.append(
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "bytes"))
        writer.writeheader()
        writer.writerows(rows)


def _write_supplementary_manifest(root: Path, data_dir: Path) -> None:
    manifest = data_dir / "SUPPLEMENTARY_DATA_MANIFEST.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        fields = ("file", "source_artifact", "source_sha256", "bytes")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, source_name in _SUPPLEMENTARY_SOURCES:
            source = root / source_name
            writer.writerow(
                {
                    "file": name,
                    "source_artifact": source_name,
                    "source_sha256": _sha256(source),
                    "bytes": source.stat().st_size,
                }
            )


def _populate_working_package(root: Path, target: Path) -> tuple[Path, Path, Path]:
    paper = root / _PAPER
    figures = target / "figures"
    tables = target / "tables"
    supplementary = target / "supplementary"
    data_dir = supplementary / "data"

    for name in _MAIN_FILES:
        _copy(paper / name, target / name)
    for name in _FIGURES:
        _copy(root / _RESULTS / "figures" / name, figures / name)
    for name in _TABLES:
        _copy(root / _RESULTS / "tables" / name, tables / name)
    for name in ("README.md", "supplementary.tex"):
        _copy(paper / "supplementary" / name, supplementary / name)
    for name, source in _SUPPLEMENTARY_SOURCES:
        _copy(root / source, data_dir / name)
    _write_supplementary_manifest(root, data_dir)
    return figures, tables, data_dir


def materialize_paper_assets(root: Path) -> None:
    paper = root / _PAPER
    for name in _LEGACY_FIGURES:
        (paper / "figures" / name).unlink(missing_ok=True)
    for name in _LEGACY_TABLES:
        (paper / "tables" / name).unlink(missing_ok=True)
    for name in _FIGURES:
        _copy(root / _RESULTS / "figures" / name, paper / "figures" / name)
    for name in _TABLES:
        _copy(root / _RESULTS / "tables" / name, paper / "tables" / name)
    data_dir = paper / "supplementary" / "data"
    for name, source in _SUPPLEMENTARY_SOURCES:
        _copy(root / source, data_dir / name)
    _write_supplementary_manifest(root, data_dir)


def _flat_manuscript(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    return (
        text.replace(r"\graphicspath{{figures/}}", r"\graphicspath{{./}}")
        .replace(
            r"\input{tables/table1_closest_work.tex}",
            r"\input{table1_closest_work.tex}",
        )
        .replace(
            r"\input{tables/table2_case_protocol.tex}",
            r"\input{table2_case_protocol.tex}",
        )
        .replace(
            r"\input{tables/table3_progressive_evidence_chain.tex}",
            r"\input{table3_progressive_evidence_chain.tex}",
        )
    )


def _write_deterministic_zip(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for path in sorted(source.iterdir()):
            info = zipfile.ZipInfo(path.name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes(), compresslevel=9)


def build_paper_package(root: Path, output_root: Path) -> PaperPackage:
    working = output_root / "working"
    _reset_directory(working)
    figures, tables, supplementary_data = _populate_working_package(root, working)

    submission = output_root / "submission_source"
    _reset_directory(submission)
    paper = root / _PAPER
    (submission / "main.tex").write_text(
        _flat_manuscript(paper / "main.tex"), encoding="utf-8", newline="\n"
    )
    for name in ("references.bib", "elsarticle.cls", "elsarticle-num.bst"):
        _copy(paper / name, submission / name)
    for name in _FIGURES:
        _copy(root / _RESULTS / "figures" / name, submission / name)
    for name in _TABLES:
        _copy(root / _RESULTS / "tables" / name, submission / name)
    (submission / "BUILD.txt").write_text(
        "Build with: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest = submission / "SUBMISSION_MANIFEST.csv"
    _write_manifest(submission, manifest)
    archive = output_root / "AEI_PAPER_SUBMISSION_SOURCE.zip"
    _write_deterministic_zip(submission, archive)
    (output_root / "AEI_PAPER_SUBMISSION_SOURCE.sha256").write_text(
        f"{_sha256(archive)}  {archive.name}\n", encoding="ascii", newline="\n"
    )
    return PaperPackage(
        root=working,
        figures=figures,
        tables=tables,
        supplementary_data=supplementary_data,
        submission_source=submission,
        manifest=manifest,
        archive=archive,
    )


__all__ = ["PaperPackage", "build_paper_package", "materialize_paper_assets"]
