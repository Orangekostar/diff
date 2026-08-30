from __future__ import annotations

import csv
import importlib.util
from hashlib import sha256
from pathlib import Path

import pytest

from cmc_bbdm.mavis import aei_paper_evidence

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def metrics() -> dict[str, object]:
    rows = aei_paper_evidence.build_canonical_metrics(ROOT)
    return {row.claim_id: row for row in rows}


def test_aei_paper_evidence_module_exists() -> None:
    assert importlib.util.find_spec("cmc_bbdm.mavis.aei_paper_evidence") is not None


def test_aei_paper_registered_p1_uses_b_field_selected_not_i_field_selected(
    metrics: dict[str, object],
) -> None:
    row = metrics["U1_MATCHED_FIELD"]
    assert row.reference_method == "B_scalar"
    assert row.candidate_method == "B_field_selected"
    assert "I_field_selected" not in row.contrast


def test_aei_paper_p1_scalar_spatial_contrast_matches_registered_report(
    metrics: dict[str, object],
) -> None:
    row = metrics["U1_MATCHED_FIELD"]
    assert row.reference_value == pytest.approx(0.1892044106549976)
    assert row.candidate_value == pytest.approx(0.12848935652346993)
    assert row.estimate == pytest.approx(0.06071505413152767)
    assert row.relative_effect == pytest.approx(0.32089661082075815)
    assert row.domains_improved == "5/6"


def test_aei_paper_p1_simultaneous_interval_matches_registered_report(
    metrics: dict[str, object],
) -> None:
    row = metrics["U1_MATCHED_FIELD"]
    assert row.ci95_lower == pytest.approx(0.006638732487036322)
    assert row.ci95_upper == pytest.approx(0.15364295185378532)


def test_aei_paper_sparse_retention_matches_frozen_ledger(
    metrics: dict[str, object],
) -> None:
    retention = metrics["U2_SPARSE_RETENTION"]
    gap = metrics["U2_SPARSE_FULL_GAP"]
    assert retention.estimate == pytest.approx(0.8989734769519824)
    assert retention.reference_value == pytest.approx(0.12848935652346993)
    assert retention.candidate_value == pytest.approx(0.13451371200896975)
    assert retention.domains_improved == "5/6"
    assert gap.estimate == pytest.approx(0.006024355485499817)
    assert gap.ci95_lower == pytest.approx(0.0017300689873626627)
    assert gap.ci95_upper == pytest.approx(0.010833148828451869)
    assert gap.domains_improved == "0/6"


def test_aei_paper_m0_oracle_metrics_match_frozen_summary(
    metrics: dict[str, object],
) -> None:
    uniform = metrics["U3_UNIFORM_ORACLE"]
    reconstruction = metrics["U3_RECONSTRUCTION_ORACLE"]
    assert uniform.estimate == pytest.approx(0.003905513917662457)
    assert uniform.ci95_lower == pytest.approx(0.0028007400590012803)
    assert uniform.ci95_upper == pytest.approx(0.005017861416957538)
    assert uniform.domains_improved == "6/6"
    assert reconstruction.estimate == pytest.approx(0.0037297162801028347)
    assert reconstruction.domains_improved == "6/6"
    assert uniform.deployable_status == "retrospective_non_deployable"


def test_aei_paper_appearance_saliency_auebc_matches_frozen_a2_bootstrap(
    metrics: dict[str, object],
) -> None:
    row = metrics["U3_CAI_VS_APPEARANCE_SALIENCY_AUEBC"]
    assert row.metric == "cai_auebc"
    assert row.reference_method == "appearance_oracle"
    assert row.candidate_method == "mechanical_oracle"
    assert row.estimate == pytest.approx(0.007080059382261465)
    assert row.ci95_lower == pytest.approx(0.004799356600193281)
    assert row.ci95_upper == pytest.approx(0.00974029297002471)
    assert row.domains_improved == "6/6"
    assert row.source_artifact == "results/mva/a2_oracle_value/bootstrap.csv"
    assert row.deployable_status == "retrospective_non_deployable"
    assert "scanner time" in row.forbidden_wording


def test_aei_paper_cai_saliency_map_similarity_matches_all_frozen_initial_maps(
    metrics: dict[str, object],
) -> None:
    spearman = metrics["U4_CAI_SALIENCY_MAP_SPEARMAN"]
    overlap = metrics["U4_CAI_SALIENCY_TOP10_OVERLAP"]
    assert spearman.estimate == pytest.approx(0.02221200907923673)
    assert overlap.estimate == pytest.approx(0.20031055900621111)
    assert (
        spearman.source_artifact
        == overlap.source_artifact
        == ("results/mva/a2_oracle_value/map_similarity.csv")
    )
    assert (
        spearman.cohort
        == overlap.cohort
        == ("276 physical specimens; 6 held-out experimental domains")
    )
    assert spearman.status == overlap.status == "DESCRIPTIVE_BOUNDARY"
    assert "independence" in spearman.forbidden_wording
    assert "no overlap" in overlap.forbidden_wording


def test_aei_paper_task_specificity_uses_frozen_reconstruction_metric(
    metrics: dict[str, object],
) -> None:
    row = metrics["U4_ORACLE_IMAGE_SPECIFICITY"]
    assert row.metric == "normalized_rgb_mse"
    assert row.estimate == pytest.approx(0.000550254888632)


def test_aei_paper_task_specificity_oracle_rows_marked_non_deployable(
    metrics: dict[str, object],
) -> None:
    rows = [
        metrics["U4_ORACLE_CAI_SPECIFICITY"],
        metrics["U4_ORACLE_IMAGE_SPECIFICITY"],
    ]
    assert all(row.deployable_status == "retrospective_non_deployable" for row in rows)
    assert metrics["U4_LEARNED_SPECIFICITY_BOUNDARY"].status == "ADVERSE_CONTROL"


def test_aei_paper_value_term_is_predictor_conditioned(
    metrics: dict[str, object],
) -> None:
    ridge_mlp = metrics["U5_RIDGE_MLP_SPEARMAN"]
    assert ridge_mlp.estimate == pytest.approx(0.116218028111857)
    assert "downstream-predictor-conditioned task value" in ridge_mlp.allowed_wording
    assert "intrinsic mechanical value" in ridge_mlp.forbidden_wording


def test_aei_paper_predictor_dependence_preserves_action_agreement_controls(
    metrics: dict[str, object],
) -> None:
    assert metrics["U5_RIDGE_HUBER_BEST_ACTION"].estimate == pytest.approx(
        0.67498834491049
    )
    assert metrics["U5_RIDGE_HUBER_TOPK"].estimate == pytest.approx(0.685361389821799)
    assert metrics["U5_RIDGE_MLP_BEST_ACTION"].estimate == pytest.approx(
        0.232645072680073
    )
    assert metrics["U5_RIDGE_MLP_TOPK"].estimate == pytest.approx(0.213252902913861)


def test_aei_paper_static_observability_preserves_null_result(
    metrics: dict[str, object],
) -> None:
    row = metrics["O1_STATIC_SPEARMAN"]
    assert row.estimate == pytest.approx(-0.01958620493212195)
    assert row.ci95_lower == pytest.approx(-0.05906029453672887)
    assert row.ci95_upper == pytest.approx(0.01947782215198501)
    assert row.domains_improved == "3/6"


def test_aei_paper_teacher_value_evolution_uses_strict_oof_endpoint(
    metrics: dict[str, object],
) -> None:
    turnover = metrics["O2_TEACHER_TURNOVER"]
    opportunity = metrics["O2_TEACHER_OPPORTUNITY"]
    assert turnover.estimate == pytest.approx(0.7037382293631609)
    assert opportunity.estimate == pytest.approx(0.005308664773700671)
    assert turnover.evidence_type == "strict_oof_retrospective_teacher"


def test_aei_paper_p10_does_not_claim_real_beyond_geometry(
    metrics: dict[str, object],
) -> None:
    positions = metrics["O3_REAL_MINUS_POSITIONS"]
    reconstruction = metrics["O3_REAL_MINUS_RECONSTRUCTION"]
    assert positions.estimate == pytest.approx(0.01740295127696979)
    assert positions.ci95_lower > 0
    assert reconstruction.estimate == pytest.approx(0.03419171130439013)
    assert reconstruction.ci95_lower > 0
    assert positions.status == reconstruction.status == "ADVERSE_CONTROL"
    assert "does not establish" in positions.allowed_wording


def test_aei_paper_p11_preserves_shuffled_adverse_control(
    metrics: dict[str, object],
) -> None:
    static = metrics["O4_DYNAMIC_MINUS_STATIC"]
    shuffled = metrics["O4_DYNAMIC_MINUS_SHUFFLED"]
    assert static.estimate == pytest.approx(-0.0012601005460586463)
    assert static.ci95_upper < 0
    assert shuffled.estimate == pytest.approx(0.00023277699588920581)
    assert shuffled.ci95_lower > 0
    assert shuffled.status == "ADVERSE_CONTROL"


def test_aei_paper_p12_does_not_claim_representation_bottleneck(
    metrics: dict[str, object],
) -> None:
    row = metrics["A1_VALUATION_SUBSTITUTION"]
    assert row.estimate == pytest.approx(0.000049786742278373763)
    assert "representation bottleneck" in row.forbidden_wording
    assert all(item.layer != "Actionable: representation" for item in metrics.values())


def test_aei_paper_p13_planning_gap_is_non_deployable(
    metrics: dict[str, object],
) -> None:
    row = metrics["A2_GREEDY_PLANNING_REGRET"]
    assert row.estimate == pytest.approx(0.00012065415511003872)
    assert row.ci95_lower == pytest.approx(0.00010330836663465408)
    assert row.deployable_status == "retrospective_non_deployable"


def test_aei_paper_p16_preserves_adverse_feedback(
    metrics: dict[str, object],
) -> None:
    row = metrics["A3_FEEDBACK_BENEFIT"]
    assert row.estimate == pytest.approx(-0.000014962761715)
    assert row.ci95_upper < 0
    assert row.domains_improved == "2/6"
    assert row.status == "ADVERSE_CONTROL"


def test_aei_paper_p7_preserves_non_superiority(
    metrics: dict[str, object],
) -> None:
    row = metrics["A4_BASELINE_MINUS_MAVIS"]
    assert row.reference_value == pytest.approx(0.12499204011570479)
    assert row.candidate_value == pytest.approx(0.12505318220938968)
    assert row.estimate == pytest.approx(-0.00006114209368489)
    assert row.ci95_upper < 0
    assert row.domains_improved == "2/6"
    assert row.status == "BOUNDARY_SUPPORTED"


def test_aei_paper_source_hashes_bind_existing_files(
    metrics: dict[str, object],
) -> None:
    for row in metrics.values():
        source = ROOT / row.source_artifact
        assert source.is_file()
        assert sha256(source.read_bytes()).hexdigest() == row.source_hash


def test_aei_paper_claim_ids_are_unique() -> None:
    rows = aei_paper_evidence.build_canonical_metrics(ROOT)
    assert len(rows) == len({row.claim_id for row in rows}) == 42


def test_aei_paper_authority_files_regenerate_deterministically(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    aei_paper_evidence.write_paper_authority(ROOT, first)
    aei_paper_evidence.write_paper_authority(ROOT, second)
    expected = {
        "EVIDENCE_AUTHORITY_RECONCILIATION.md",
        "PAPER_CANONICAL_METRICS.csv",
        "PAPER_CLAIM_MAP.md",
        "PAPER_SOURCE_HASHES.csv",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert {path.name for path in second.iterdir()} == expected
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_aei_paper_source_hashes_bind_domain_roster(tmp_path: Path) -> None:
    output = tmp_path / "authority"
    aei_paper_evidence.write_paper_authority(ROOT, output)
    with (output / "PAPER_SOURCE_HASHES.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        sources = {row["source_artifact"] for row in csv.DictReader(stream)}
    assert {
        "artifacts/mavis_authority/artifact_manifest.json",
        "artifacts/mavis_authority/scan_manifest.csv",
        "docs/MVA_A0_A3_PROTOCOL.md",
        "src/cmc_bbdm/mva/appearance_value.py",
        "src/cmc_bbdm/mva/oracle_execution.py",
        "results/mva/a2_oracle_value/bootstrap.csv",
        "results/mva/a2_oracle_value/domain_metrics.csv",
        "results/mva/a2_oracle_value/map_similarity.csv",
        "results/mva/a2_oracle_value/oracle_values.parquet",
    } <= sources
