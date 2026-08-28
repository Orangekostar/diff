from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_aei_information_hierarchy"
ARTIFACTS = ROOT / "artifacts/aei_information_hierarchy"
TABLES = ROOT / "results/aei_information_hierarchy/tables"
MAIN = PAPER / "main.tex"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_aei_paper_auebc_equation_matches_closed_loop_implementation() -> None:
    manuscript = _text(MAIN)
    implementation = _text(ROOT / "src/cmc_bbdm/mavis/closed_loop_metrics.py")
    assert "np.trapezoid(y, x=x) / (x[-1] - x[0])" in implementation
    assert r"\frac{1}{x_{i,K}-x_{i,1}}" in manuscript
    assert r"(x_{i,k+1}-x_{i,k})" in manuscript


def test_aei_paper_auebc_uses_effective_budget() -> None:
    manuscript = _text(MAIN)
    assert r"x_{i,1}<\cdots<x_{i,K}" in manuscript
    assert "effective specimen budgets" in manuscript
    assert "actual/effective specimen budget" in _text(
        ARTIFACTS / "AUEBC_DEFINITION_AUDIT.md"
    )


def test_aei_paper_auebc_states_budget_span_normalization() -> None:
    required = (
        "budget-span-normalized trapezoidal mean error over the observed "
        "effective-budget range"
    )
    assert required in re.sub(r"\s+", " ", _text(MAIN))
    audit = _text(ARTIFACTS / "AUEBC_DEFINITION_AUDIT.md")
    assert "historical numbers do not require recomputation" in audit
    assert "task_specificity.normalized_auebc" in audit


def test_aei_paper_every_canonical_claim_has_chronology_class() -> None:
    canonical = {
        row["claim_id"] for row in _rows(ARTIFACTS / "PAPER_CANONICAL_METRICS.csv")
    }
    chronology = _rows(ARTIFACTS / "PAPER_EVIDENCE_CHRONOLOGY.csv")
    assert len(chronology) == 39
    assert {row["claim_id"] for row in chronology} == canonical
    assert {row["chronology_class"] for row in chronology} == {
        "PRE_P7_FROZEN_EVIDENCE",
        "FROZEN_OUTER_ENDPOINT",
        "POST_P7_DIAGNOSTIC",
    }


def test_aei_paper_post_p7_diagnostics_not_used_to_modify_p7() -> None:
    rows = _rows(ARTIFACTS / "PAPER_EVIDENCE_CHRONOLOGY.csv")
    post = [row for row in rows if row["chronology_class"] == "POST_P7_DIAGNOSTIC"]
    assert len(post) == 25
    assert all(row["evidence_frozen_before_p7"] == "false" for row in post)
    assert all(row["analysis_created_after_p7"] == "true" for row in post)
    assert all(row["used_to_modify_p7"] == "false" for row in post)


def test_aei_paper_does_not_call_post_freeze_diagnostics_preregistered() -> None:
    text = (_text(MAIN) + _text(ARTIFACTS / "PAPER_EVIDENCE_CHRONOLOGY.md")).lower()
    assert "post-freeze diagnostics were preregistered" not in text
    assert "post-freeze diagnostics were registered confirmatory" not in text


def test_aei_paper_chronology_statement_is_present() -> None:
    manuscript = _text(MAIN)
    assert "distinct chronological roles" in manuscript
    assert (
        "were not used to re-select or modify the frozen outer endpoint" in manuscript
    )


def test_aei_paper_closest_work_matrix_has_verified_sources() -> None:
    rows = _rows(TABLES / "table1_closest_work.csv")
    assert all(row["primary_source"].startswith("https://") for row in rows)
    assert all(row["source_status"] == "VERIFIED_PRIMARY" for row in rows)


def test_aei_paper_closest_work_matrix_contains_required_prior_works() -> None:
    works = "\n".join(row["work"] for row in _rows(TABLES / "table1_closest_work.csv"))
    for name in (
        "Fuentes",
        "Cantero-Chinchilla",
        "Memarzadeh and Pozzi",
        "Blumberg",
        "Ji",
        "Mack",
    ):
        assert name in works


def test_aei_paper_no_first_adaptive_ultrasound_claim() -> None:
    assert "first adaptive ultrasonic" not in _text(MAIN).lower()


def test_aei_paper_no_first_ultrasound_voi_claim() -> None:
    assert "first ultrasound voi" not in _text(MAIN).lower()


def test_aei_paper_no_first_task_driven_design_claim() -> None:
    assert "first task-driven" not in _text(MAIN).lower()


def test_aei_paper_novelty_is_operational_not_taxonomy_only() -> None:
    manuscript = re.sub(r"\s+", " ", _text(MAIN))
    assert "not a new generic definition of value of information" in manuscript
    assert "Information Characterization" in manuscript
    assert "Evidence-Calibrated Decision Realization" in manuscript
    assert "one causal acquisition contract" in manuscript


def test_all_39_canonical_claims_have_one_primary_narrative_stage() -> None:
    canonical = _rows(ARTIFACTS / "PAPER_CANONICAL_METRICS.csv")
    narrative = _rows(ARTIFACTS / "PAPER_POSITIVE_NARRATIVE_MAP.csv")
    required = {
        "claim_id",
        "canonical_layer",
        "chronology_class",
        "new_part",
        "new_stage",
        "narrative_role",
        "main_or_supporting",
        "positive_headline",
        "required_boundary",
        "source_artifact",
        "figure_assignment",
        "table_assignment",
        "manuscript_assignment",
        "narrative_visibility",
    }
    assert len(canonical) == len(narrative) == 39
    assert set(narrative[0]) == required
    assert [row["claim_id"] for row in narrative] == [
        row["claim_id"] for row in canonical
    ]
    assert len({row["claim_id"] for row in narrative}) == 39
    assert all(row["new_part"] in {"PART_I", "PART_II"} for row in narrative)
    assert all(row["new_stage"] for row in narrative)


def test_positive_narrative_map_uses_visibility_aware_writing_contract() -> None:
    narrative = _rows(ARTIFACTS / "PAPER_POSITIVE_NARRATIVE_MAP.csv")
    visibility = {
        row["claim_id"]: row["visibility"]
        for row in _rows(ARTIFACTS / "PAPER_CLAIM_VISIBILITY_MAP.csv")
    }
    expected = {
        "MAIN_HEADLINE": "MAIN_REQUIRED",
        "MAIN_SUPPORT": "MAIN_OPTIONAL",
        "MAIN_SYSTEM_DIAGNOSTIC": "MAIN_REQUIRED",
        "SUPPLEMENT_ONLY": "SUPPLEMENT_REQUIRED",
    }
    assert {
        row["narrative_visibility"] for row in narrative
    } == {"MAIN_REQUIRED", "MAIN_OPTIONAL", "SUPPLEMENT_REQUIRED"}
    for row in narrative:
        assert row["narrative_visibility"] == expected[visibility[row["claim_id"]]]

    markdown = _text(ARTIFACTS / "PAPER_POSITIVE_NARRATIVE_MAP.md")
    for value in (
        "MAIN_REQUIRED",
        "MAIN_OPTIONAL",
        "SUPPLEMENT_REQUIRED",
        "INTERNAL_ONLY",
    ):
        assert f"`{value}`" in markdown


def test_positive_narrative_map_matches_six_stage_visibility_authority() -> None:
    narrative = {
        row["claim_id"]: row
        for row in _rows(ARTIFACTS / "PAPER_POSITIVE_NARRATIVE_MAP.csv")
    }
    visibility = _rows(ARTIFACTS / "PAPER_CLAIM_VISIBILITY_MAP.csv")
    role_by_visibility = {
        "MAIN_HEADLINE": "main",
        "MAIN_SUPPORT": "supporting",
        "MAIN_SYSTEM_DIAGNOSTIC": "system_diagnostic",
        "SUPPLEMENT_ONLY": "supplement",
    }
    for expected in visibility:
        actual = narrative[expected["claim_id"]]
        assert actual["new_part"] == (
            "PART_I"
            if expected["paper_module"] == "PART_I_CHARACTERIZATION"
            else "PART_II"
        )
        assert actual["new_stage"] == expected["compressed_stage"]
        assert actual["main_or_supporting"] == role_by_visibility[
            expected["visibility"]
        ]
        assert actual["figure_assignment"] == expected["main_figure"]
        assert actual["table_assignment"] == expected["main_table"]
        assert actual["manuscript_assignment"] == expected["main_section"]

    markdown = _text(ARTIFACTS / "PAPER_POSITIVE_NARRATIVE_MAP.md")
    for stage in {
        row["compressed_stage"] for row in visibility
    }:
        assert f"`{stage}`" in markdown
    for obsolete in (
        "I1 Spatial enrichment",
        "I6 Predictor conditioning",
        "II1 Static reference",
        "II6 Deployment calibration",
    ):
        assert obsolete not in markdown


def test_narrative_map_preserves_canonical_layer_chronology_and_source() -> None:
    canonical = {
        row["claim_id"]: row for row in _rows(ARTIFACTS / "PAPER_CANONICAL_METRICS.csv")
    }
    chronology = {
        row["claim_id"]: row
        for row in _rows(ARTIFACTS / "PAPER_EVIDENCE_CHRONOLOGY.csv")
    }
    for row in _rows(ARTIFACTS / "PAPER_POSITIVE_NARRATIVE_MAP.csv"):
        claim = canonical[row["claim_id"]]
        timing = chronology[row["claim_id"]]
        assert row["canonical_layer"] == claim["layer"]
        assert row["chronology_class"] == timing["chronology_class"]
        assert row["source_artifact"] == claim["source_artifact"]


def test_canonical_metrics_file_is_unchanged_by_narrative_refactor() -> None:
    import hashlib

    payload = (ARTIFACTS / "PAPER_CANONICAL_METRICS.csv").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6"
    )


def test_aei_paper_predictor_conditioning_discloses_accuracy_boundary() -> None:
    manuscript = re.sub(r"\s+", " ", _text(MAIN))
    assert "comparably performing low-complexity predictors" in manuscript
    assert "substantially less accurate shallow MLP" in manuscript
    assert all(value in manuscript for value in ("0.08964", "0.08618", "0.15067"))


def test_aei_paper_does_not_claim_predictor_dependence_independent_of_model_quality() -> (
    None
):
    manuscript = re.sub(r"\s+", " ", _text(MAIN))
    assert (
        "does not determine how much value-map variation remains among equally accurate"
        in manuscript
    )
    assert "predictor dependence independent of model quality" not in manuscript.lower()


def test_aei_paper_keeps_predictor_index_f() -> None:
    manuscript = _text(MAIN)
    assert r"U_f(X\mid\legal_{i,t})" in manuscript
    assert "retain the predictor index $f$" in manuscript


def test_aei_paper_has_explicit_transfer_conditions() -> None:
    manuscript = re.sub(r"\s+", " ", _text(MAIN))
    assert "Transfer conditions beyond the present case study" in manuscript
    for phrase in (
        "defined downstream engineering endpoint and loss",
        "legal partial-observation state excluding future measurements and outcomes",
        "retrospective or counterfactual target for marginal information value",
        "matched controls separating measurement content from acquisition geometry and history",
        "end-to-end cost-constrained decision metric distinct from local value prediction",
    ):
        assert phrase in manuscript


def test_aei_paper_transfer_conditions_do_not_claim_external_validation() -> None:
    manuscript = re.sub(r"\s+", " ", _text(MAIN)).lower()
    assert (
        "methodological conditions, not proof of universal empirical transfer"
        in manuscript
    )
    assert "externally validates the hierarchy" not in manuscript


def test_aei_paper_empirical_cross_domain_wording_is_bounded() -> None:
    manuscript = re.sub(r"\s+", " ", _text(MAIN))
    assert (
        "across the six held-out experimental domains in the present data program"
        in manuscript
    )
    assert "external empirical generalization" not in manuscript.lower()


def test_aei_paper_external_literature_is_not_called_external_replication() -> None:
    manuscript = _text(MAIN).lower()
    references = _text(PAPER / "references.bib")
    assert "10.1016/j.compscitech.2019.107681" in references
    assert "10.5281/zenodo.1476887" in references
    assert "external replication" not in manuscript


def test_aei_paper_external_data_feasibility_records_no_go() -> None:
    report = _text(ARTIFACTS / "EXTERNAL_DATA_FEASIBILITY_2026.md")
    for identifier in (
        "10.17632/8scdmfdcfb.3",
        "10.5281/zenodo.1476887",
        "10.17632/wg4dmwddjy.2",
        "10.4121/21621381",
        "10.5281/zenodo.4405277",
        "10.15125/BATH-00103",
    ):
        assert identifier in report
    assert "EXTERNAL_MICRO_PILOT_NO_GO" in report
    assert "MANUSCRIPT_ONLY_PRIMARY" in report


def test_aei_paper_external_data_roles_preserve_pairing_and_scale_boundaries() -> None:
    report = _text(ARTIFACTS / "EXTERNAL_DATA_FEASIBILITY_2026.md")
    assert "UNRESOLVED_CSCAN_SPECIMEN_ROI" in report
    assert "paired_cscan_cai = false" in report
    assert "10 exact pairs" in report
    assert "N=3" in report
    assert "cannot be described as benchmark-scale validation" in report


def test_aei_paper_review_fix_audits_exist() -> None:
    required = {
        "REVIEW_FIX_P0_AUDIT.md",
        "AUEBC_DEFINITION_AUDIT.md",
        "PAPER_EVIDENCE_CHRONOLOGY.csv",
        "PAPER_EVIDENCE_CHRONOLOGY.md",
        "CLOSEST_WORK_POSITIONING.md",
        "TRANSFER_CONDITIONS.md",
        "EXTERNAL_DATA_FEASIBILITY_2026.md",
        "REVIEW_FIX_COMPLETION_AUDIT.md",
    }
    assert required <= {path.name for path in ARTIFACTS.iterdir()}
