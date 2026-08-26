"""Paper-specific evidence aggregation for the AEI information hierarchy."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PaperMetric:
    claim_id: str
    layer: str
    paper_role: str
    metric: str
    contrast: str
    direction: str
    reference_method: str
    candidate_method: str
    reference_value: float | None
    candidate_value: float | None
    estimate: float
    relative_effect: float | None
    ci95_lower: float | None
    ci95_upper: float | None
    domains_improved: str
    cohort: str
    protocol: str
    source_artifact: str
    source_hash: str
    evidence_type: str
    deployable_status: str
    status: str
    manuscript_location: str
    allowed_wording: str
    forbidden_wording: str


_COHORT = "276 physical specimens; 6 held-out experimental domains"
_P7_TREE = "931dc86c26caf1c7246709c4706a7cd0428e3a1533b6ff1ad3c2ad8f9517d1e4"
_SOURCE_PATHS = (
    "results/p1_full_field_oracle/metrics.json",
    "results/p1_full_field_oracle/domain_metrics.csv",
    "results/p5_sparse_scan/retention.csv",
    "results/p5_sparse_scan/bootstrap.csv",
    "results/mvd/m0_one_shot_oracle/summary.json",
    "results/mvd/m1_observability/bootstrap.csv",
    "results/mvd/m1_observability/model_metrics.csv",
    "results/mavis_science_closure/p9_value_evolution/summary.json",
    "results/mavis_science_closure/p10_mris_causal/summary.json",
    "results/mavis_science_closure/p10_mris_causal/contrasts.csv",
    "results/mavis_science_closure/p11_dynamic_valuation/summary.json",
    "results/mavis_science_closure/p12_rvp_attribution/summary.json",
    "results/mavis_science_closure/p13_set_planning/summary.json",
    "results/mavis_science_closure/p14_task_specificity/summary.json",
    "results/mavis_science_closure/p15_value_stability/summary.json",
    "results/mavis_science_closure/p16_feedback_mechanism/summary.json",
    "results/mavis_science_closure/p16_feedback_mechanism/domain_effects.csv",
    "results/mavis/p7_final_frozen_eval/claim_evidence.csv",
)


class PaperEvidenceError(ValueError):
    """Raised when frozen evidence violates the paper authority contract."""


def _source(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if base not in path.parents or not path.is_file():
        raise PaperEvidenceError(f"missing or invalid paper evidence source: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(root: Path, relative: str) -> dict[str, Any]:
    value = json.loads(_source(root, relative).read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise PaperEvidenceError(f"paper evidence JSON must be an object: {relative}")
    return value


def _csv(root: Path, relative: str) -> list[dict[str, str]]:
    with _source(root, relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _one(rows: list[dict[str, str]], source: str, **matches: str) -> dict[str, str]:
    selected = [
        row for row in rows if all(row.get(column) == value for column, value in matches.items())
    ]
    if len(selected) != 1:
        raise PaperEvidenceError(
            f"expected one row in {source} for {matches}, found {len(selected)}"
        )
    return selected[0]


def _float(value: Any, label: str) -> float:
    if type(value) not in (int, float, str):
        raise PaperEvidenceError(f"{label} is not numeric")
    try:
        number = float(value)
    except ValueError as error:
        raise PaperEvidenceError(f"{label} is not numeric") from error
    if not (-float("inf") < number < float("inf")):
        raise PaperEvidenceError(f"{label} is not finite")
    return number


def _mean_method(rows: list[dict[str, str]], method: str) -> float:
    values = [_float(row["mae"], f"{method} MAE") for row in rows if row["method"] == method]
    if len(values) != 6:
        raise PaperEvidenceError(f"{method} must have exactly six domain MAEs")
    return sum(values) / len(values)


def _metric(
    root: Path,
    *,
    claim_id: str,
    layer: str,
    paper_role: str,
    metric: str,
    contrast: str,
    direction: str,
    reference_method: str,
    candidate_method: str,
    reference_value: float | None,
    candidate_value: float | None,
    estimate: float,
    relative_effect: float | None,
    ci95_lower: float | None,
    ci95_upper: float | None,
    domains_improved: str,
    protocol: str,
    source_artifact: str,
    evidence_type: str,
    deployable_status: str,
    status: str,
    manuscript_location: str,
    allowed_wording: str,
    forbidden_wording: str,
) -> PaperMetric:
    if ci95_lower is not None and ci95_upper is not None and ci95_lower > ci95_upper:
        raise PaperEvidenceError(f"reversed interval for {claim_id}")
    return PaperMetric(
        claim_id=claim_id,
        layer=layer,
        paper_role=paper_role,
        metric=metric,
        contrast=contrast,
        direction=direction,
        reference_method=reference_method,
        candidate_method=candidate_method,
        reference_value=reference_value,
        candidate_value=candidate_value,
        estimate=estimate,
        relative_effect=relative_effect,
        ci95_lower=ci95_lower,
        ci95_upper=ci95_upper,
        domains_improved=domains_improved,
        cohort=_COHORT,
        protocol=protocol,
        source_artifact=source_artifact,
        source_hash=_sha256(_source(root, source_artifact)),
        evidence_type=evidence_type,
        deployable_status=deployable_status,
        status=status,
        manuscript_location=manuscript_location,
        allowed_wording=allowed_wording,
        forbidden_wording=forbidden_wording,
    )


def build_canonical_metrics(root: Path) -> tuple[PaperMetric, ...]:
    """Build ordered, hash-bound paper metrics from frozen evidence."""
    for relative in _SOURCE_PATHS:
        _source(root, relative)

    p1_source = "results/p1_full_field_oracle/metrics.json"
    p1 = _json(root, p1_source)
    p1_domains = _csv(root, "results/p1_full_field_oracle/domain_metrics.csv")
    p1_effects = p1["effects"]
    p1_bootstrap = p1["bootstrap"]
    scalar_effect = p1_effects["scalar_vs_field"]
    scalar_interval = p1_bootstrap["scalar_vs_field"]
    surface_effect = p1_effects["A_vs_field"]
    surface_interval = p1_bootstrap["A_vs_field"]
    scalar_mae = _mean_method(p1_domains, "B_scalar")
    field_mae = _mean_method(p1_domains, "B_field_selected")
    surface_mae = _mean_method(p1_domains, "A_surface")
    independent_mae = _mean_method(p1_domains, "I_field_selected")
    if scalar_effect["reference"] != "B_scalar" or scalar_effect["candidate"] != "B_field_selected":
        raise PaperEvidenceError("registered P1 scalar-field methods changed")

    p5_retention_source = "results/p5_sparse_scan/retention.csv"
    p5_retention_rows = _csv(root, p5_retention_source)
    p5_selected = _one(
        p5_retention_rows, p5_retention_source, condition_id="bilinear_0.25"
    )
    p5_boot_source = "results/p5_sparse_scan/bootstrap.csv"
    p5_boot = _csv(root, p5_boot_source)
    p5_gain = _one(p5_boot, p5_boot_source, effect_id="surface_minus_bilinear_0.25")
    p5_gap = _one(p5_boot, p5_boot_source, effect_id="bilinear_0.25_minus_full")

    m0_source = "results/mvd/m0_one_shot_oracle/summary.json"
    m0 = _json(root, m0_source)["gate"]
    m1_boot_source = "results/mvd/m1_observability/bootstrap.csv"
    m1_boot = _csv(root, m1_boot_source)
    m1_spearman = _one(m1_boot, m1_boot_source, effect_id="o2_spearman_positive")
    m1_models_source = "results/mvd/m1_observability/model_metrics.csv"
    m1_models = _csv(root, m1_models_source)
    m1_o2 = _one(m1_models, m1_models_source, method="o2_global_candidate")
    m1_global = _one(m1_models, m1_models_source, method="global_mechanical")
    m1_random = _one(m1_models, m1_models_source, method="random_median")

    p9_source = "results/mavis_science_closure/p9_value_evolution/summary.json"
    p9 = _json(root, p9_source)
    p9_endpoint = p9["teacher_by_checkpoint"][-1]
    if _float(p9_endpoint["current_checkpoint"], "P9 endpoint") != 0.1875:
        raise PaperEvidenceError("P9 teacher endpoint changed")

    p10_source = "results/mavis_science_closure/p10_mris_causal/summary.json"
    p10 = _json(root, p10_source)
    if p10["full_field_method"] != "I_field_selected":
        raise PaperEvidenceError("P10 full-field sensitivity endpoint changed")
    p10_contrast_source = "results/mavis_science_closure/p10_mris_causal/contrasts.csv"
    p10_contrasts = _csv(root, p10_contrast_source)
    p10_positions = _one(
        p10_contrasts,
        p10_contrast_source,
        nominal_checkpoint="0.25",
        control_mode="positions_only",
    )
    p10_reconstruction = _one(
        p10_contrasts,
        p10_contrast_source,
        nominal_checkpoint="0.25",
        control_mode="reconstruction",
    )

    p11_source = "results/mavis_science_closure/p11_dynamic_valuation/summary.json"
    p11 = _json(root, p11_source)
    p12_source = "results/mavis_science_closure/p12_rvp_attribution/summary.json"
    p12 = _json(root, p12_source)
    if p12["optional_row_f"] != "OMITTED_P10_DID_NOT_SUPPORT_REPRESENTATION_BOTTLENECK":
        raise PaperEvidenceError("P12 representation-boundary status changed")
    p13_source = "results/mavis_science_closure/p13_set_planning/summary.json"
    p13 = _json(root, p13_source)
    p14_source = "results/mavis_science_closure/p14_task_specificity/summary.json"
    p14 = _json(root, p14_source)
    if p14["reconstruction_metric"] != "normalized_rgb_mse":
        raise PaperEvidenceError("P14 reconstruction metric changed")
    p15_source = "results/mavis_science_closure/p15_value_stability/summary.json"
    p15 = _json(root, p15_source)
    p16_source = "results/mavis_science_closure/p16_feedback_mechanism/summary.json"
    p16 = _json(root, p16_source)
    p16_interval = p16["bootstrap_intervals"][
        "overall_feedback/auebc/none/none/none/feedback_benefit"
    ]["interval"]
    p16_domains = _csv(
        root, "results/mavis_science_closure/p16_feedback_mechanism/domain_effects.csv"
    )
    feedback_improved = sum(
        _float(row["mean_feedback_benefit"], "P16 domain feedback") > 0
        for row in p16_domains
        if row["level"] == "auebc" and row["outer_domain"] != "__equal_domain__"
    )

    p7_source = "results/mavis/p7_final_frozen_eval/claim_evidence.csv"
    p7 = _one(_csv(root, p7_source), p7_source, claim_tier="B")

    rows: list[PaperMetric] = []
    add = rows.append
    add(
        _metric(
            root,
            claim_id="U1_MATCHED_FIELD",
            layer="Useful",
            paper_role="headline",
            metric="equal_domain_cai_ratio_mae",
            contrast="B_scalar minus B_field_selected",
            direction="reference_minus_candidate; positive_favors_spatial_field",
            reference_method="B_scalar",
            candidate_method="B_field_selected",
            reference_value=scalar_mae,
            candidate_value=field_mae,
            estimate=_float(scalar_effect["delta"], "P1 scalar effect"),
            relative_effect=_float(
                scalar_effect["relative_improvement"], "P1 relative effect"
            ),
            ci95_lower=_float(scalar_interval["simultaneous_low"], "P1 lower"),
            ci95_upper=_float(scalar_interval["simultaneous_high"], "P1 upper"),
            domains_improved=f"{scalar_effect['improved_domains']}/6",
            protocol="strict nested LODO; specimen-first equal-domain MAE; familywise simultaneous interval",
            source_artifact=p1_source,
            evidence_type="registered_confirmatory",
            deployable_status="full_field_evaluation",
            status="SUPPORTED",
            manuscript_location="main",
            allowed_wording="full spatial internal morphology preserves CAI-relevant information absent from the matched scalar representation",
            forbidden_wording="I_field_selected is the registered matched estimator; spatial information is universally sufficient",
        )
    )
    add(
        _metric(
            root,
            claim_id="U1_SURFACE_FIELD",
            layer="Useful",
            paper_role="related",
            metric="equal_domain_cai_ratio_mae",
            contrast="A_surface minus B_field_selected",
            direction="reference_minus_candidate; positive_favors_spatial_field",
            reference_method="A_surface",
            candidate_method="B_field_selected",
            reference_value=surface_mae,
            candidate_value=field_mae,
            estimate=_float(surface_effect["delta"], "P1 surface effect"),
            relative_effect=_float(
                surface_effect["relative_improvement"], "P1 surface relative effect"
            ),
            ci95_lower=_float(surface_interval["simultaneous_low"], "P1 surface lower"),
            ci95_upper=_float(surface_interval["simultaneous_high"], "P1 surface upper"),
            domains_improved=f"{surface_effect['improved_domains']}/6",
            protocol="strict nested LODO; specimen-first equal-domain MAE; familywise simultaneous interval",
            source_artifact=p1_source,
            evidence_type="registered_related_contrast",
            deployable_status="full_field_evaluation",
            status="SUPPORTED",
            manuscript_location="main",
            allowed_wording="measured spatial internal information adds CAI-relevant signal beyond metadata and surface statistics",
            forbidden_wording="surface measurements can reconstruct the full internal field",
        )
    )
    add(
        _metric(
            root,
            claim_id="U1_INDEPENDENT_FIELD_SENSITIVITY",
            layer="Useful",
            paper_role="sensitivity",
            metric="equal_domain_cai_ratio_mae",
            contrast="independent metadata-only-prefix field estimator",
            direction="lower_is_better",
            reference_method="independent metadata-only prefix",
            candidate_method="I_field_selected",
            reference_value=None,
            candidate_value=independent_mae,
            estimate=independent_mae,
            relative_effect=None,
            ci95_lower=None,
            ci95_upper=None,
            domains_improved="not_applicable",
            protocol="strict nested LODO; independent internal-only candidate family",
            source_artifact="results/p1_full_field_oracle/domain_metrics.csv",
            evidence_type="sensitivity_estimator",
            deployable_status="full_field_evaluation",
            status="SENSITIVITY_ONLY",
            manuscript_location="appendix",
            allowed_wording="a distinct independent estimator yields a lower point MAE and is not the matched confirmatory path",
            forbidden_wording="0.099568606 is the registered B-family effect",
        )
    )
    sparse_mae = _float(p5_selected["sparse_equal_domain_mae"], "P5 sparse MAE")
    full_mae = _float(p5_selected["full_equal_domain_mae"], "P5 full MAE")
    add(
        _metric(
            root,
            claim_id="U2_SPARSE_RETENTION",
            layer="Useful",
            paper_role="headline",
            metric="registered_full_field_gain_retention",
            contrast="25% nominal bilinear sparse gain divided by full-field gain",
            direction="higher_is_more_registered_gain_retained",
            reference_method="B_field_selected",
            candidate_method="bilinear_0.25 sparse field",
            reference_value=full_mae,
            candidate_value=sparse_mae,
            estimate=_float(p5_selected["retention"], "P5 retention"),
            relative_effect=None,
            ci95_lower=None,
            ci95_upper=None,
            domains_improved=f"{p5_selected['improved_domains']}/6",
            protocol="retrospective normalized-raster nominal-density protocol; strict nested LODO",
            source_artifact=p5_retention_source,
            evidence_type="registered_retrospective",
            deployable_status="retrospective_non_deployable",
            status="SUPPORTED",
            manuscript_location="main",
            allowed_wording="sparse real C-scan measurements retain most registered full-field CAI gain under the normalized-raster protocol",
            forbidden_wording="25% scan time is sufficient; 75% scanner-time reduction",
        )
    )
    add(
        _metric(
            root,
            claim_id="U2_SPARSE_GAIN",
            layer="Useful",
            paper_role="supporting",
            metric="equal_domain_cai_ratio_mae",
            contrast="A_surface minus bilinear_0.25 sparse field",
            direction="reference_minus_candidate; positive_favors_sparse_field",
            reference_method="A_surface",
            candidate_method="bilinear_0.25 sparse field",
            reference_value=_float(p5_selected["surface_equal_domain_mae"], "P5 surface MAE"),
            candidate_value=sparse_mae,
            estimate=_float(p5_gain["point_estimate"], "P5 sparse gain"),
            relative_effect=_float(p5_selected["retention"], "P5 retention"),
            ci95_lower=_float(p5_gain["simultaneous_lower"], "P5 gain lower"),
            ci95_upper=_float(p5_gain["simultaneous_upper"], "P5 gain upper"),
            domains_improved="5/6",
            protocol="retrospective normalized-raster nominal-density protocol; synchronized familywise bootstrap",
            source_artifact=p5_boot_source,
            evidence_type="registered_retrospective",
            deployable_status="retrospective_non_deployable",
            status="SUPPORTED",
            manuscript_location="main",
            allowed_wording="the selected sparse condition improves CAI prediction over the surface reference in five held-out domains",
            forbidden_wording="sparse measurement equals the full field",
        )
    )
    add(
        _metric(
            root,
            claim_id="U2_SPARSE_FULL_GAP",
            layer="Useful",
            paper_role="boundary",
            metric="equal_domain_cai_ratio_mae",
            contrast="bilinear_0.25 sparse field minus B_field_selected",
            direction="candidate_minus_reference; positive_means_sparse_is_worse",
            reference_method="B_field_selected",
            candidate_method="bilinear_0.25 sparse field",
            reference_value=full_mae,
            candidate_value=sparse_mae,
            estimate=_float(p5_gap["point_estimate"], "P5 sparse-full gap"),
            relative_effect=None,
            ci95_lower=_float(p5_gap["simultaneous_lower"], "P5 gap lower"),
            ci95_upper=_float(p5_gap["simultaneous_upper"], "P5 gap upper"),
            domains_improved="0/6",
            protocol="retrospective normalized-raster nominal-density protocol; synchronized familywise bootstrap",
            source_artifact=p5_boot_source,
            evidence_type="registered_retrospective_boundary",
            deployable_status="retrospective_non_deployable",
            status="BOUNDARY_SUPPORTED",
            manuscript_location="main",
            allowed_wording="the selected sparse condition retains most, but not all, full-field gain",
            forbidden_wording="0.001730069 is the lower interval for surface-to-sparse improvement",
        )
    )

    for claim_id, label, effect_key, baseline_key in (
        ("U3_UNIFORM_ORACLE", "uniform", "uniform_effect", "uniform"),
        (
            "U3_RECONSTRUCTION_ORACLE",
            "one-shot reconstruction",
            "reconstruction_effect",
            "stronger_baseline",
        ),
    ):
        effect = m0[effect_key]
        baseline_value = (
            _float(m0["one_shot_auebc"], "M0 oracle AUEBC")
            + _float(effect["point_estimate"], "M0 uniform effect")
            if baseline_key == "uniform"
            else _float(m0["stronger_baseline_auebc"], "M0 reconstruction AUEBC")
        )
        add(
            _metric(
                root,
                claim_id=claim_id,
                layer="Useful",
                paper_role="headline" if label == "uniform" else "supporting",
                metric="cai_auebc",
                contrast=f"{label} minus one-shot mechanical oracle",
                direction="reference_minus_candidate; positive_favors_mechanical_oracle",
                reference_method=label,
                candidate_method="one-shot mechanical oracle",
                reference_value=baseline_value,
                candidate_value=_float(m0["one_shot_auebc"], "M0 oracle AUEBC"),
                estimate=_float(effect["point_estimate"], f"{claim_id} estimate"),
                relative_effect=None,
                ci95_lower=_float(effect["lower"], f"{claim_id} lower"),
                ci95_upper=_float(effect["upper"], f"{claim_id} upper"),
                domains_improved=f"{effect['improved_domains']}/6",
                protocol="retrospective one-shot exact-cost oracle; source-only fit; held-out-domain evaluation",
                source_artifact=m0_source,
                evidence_type="retrospective_oracle",
                deployable_status="retrospective_non_deployable",
                status="SUPPORTED_ORACLE_ONLY",
                manuscript_location="main",
                allowed_wording="retrospective one-shot mechanical acquisition headroom exists",
                forbidden_wording="the oracle is a deployable scanner policy; physical inspection-time reduction",
            )
        )
    add(
        _metric(
            root,
            claim_id="U3_HEADROOM_RETENTION",
            layer="Useful",
            paper_role="supporting",
            metric="one_shot_sequential_headroom_retention",
            contrast="one-shot oracle gain relative to sequential oracle headroom",
            direction="higher_is_more_sequential_headroom_retained",
            reference_method="sequential mechanical oracle",
            candidate_method="one-shot mechanical oracle",
            reference_value=_float(m0["sequential_auebc"], "M0 sequential oracle"),
            candidate_value=_float(m0["one_shot_auebc"], "M0 one-shot oracle"),
            estimate=_float(m0["headroom_retention"], "M0 headroom retention"),
            relative_effect=None,
            ci95_lower=None,
            ci95_upper=None,
            domains_improved="6/6 versus uniform and reconstruction",
            protocol="retrospective one-shot and sequential exact-cost oracle comparison",
            source_artifact=m0_source,
            evidence_type="retrospective_oracle",
            deployable_status="retrospective_non_deployable",
            status="SUPPORTED_ORACLE_ONLY",
            manuscript_location="main",
            allowed_wording="a specimen-specific initial plan retains part of the sequential oracle headroom",
            forbidden_wording="one-shot acquisition is deployable or additive",
        )
    )

    p14_cai = p14["contrasts"]["oracle_reconstruction_minus_mechanics_cai"]
    p14_image = p14["contrasts"]["oracle_mechanics_minus_reconstruction_image"]
    for claim_id, metric_name, contrast, values, allowed in (
        (
            "U4_ORACLE_CAI_SPECIFICITY",
            "cai_auebc",
            "reconstruction oracle minus mechanical oracle on CAI",
            p14_cai,
            "mechanics-optimal oracle acquisition improves the CAI objective",
        ),
        (
            "U4_ORACLE_IMAGE_SPECIFICITY",
            p14["reconstruction_metric"],
            "mechanical oracle minus reconstruction oracle on image error",
            p14_image,
            "reconstruction-optimal oracle acquisition improves the reconstruction objective",
        ),
    ):
        add(
            _metric(
                root,
                claim_id=claim_id,
                layer="Useful",
                paper_role="headline",
                metric=metric_name,
                contrast=contrast,
                direction="positive_supports_oracle_task_specificity",
                reference_method="reconstruction oracle",
                candidate_method="mechanical oracle",
                reference_value=None,
                candidate_value=None,
                estimate=_float(values["point"], f"{claim_id} point"),
                relative_effect=None,
                ci95_lower=_float(values["interval"][0], f"{claim_id} lower"),
                ci95_upper=_float(values["interval"][1], f"{claim_id} upper"),
                domains_improved="6 held-out domains; equal-domain contrast",
                protocol="same cohort and exact-cost oracle trajectories; frozen normalized_rgb_mse reconstruction metric",
                source_artifact=p14_source,
                evidence_type="retrospective_oracle_cross_objective",
                deployable_status="retrospective_non_deployable",
                status="SUPPORTED_ORACLE_ONLY",
                manuscript_location="main",
                allowed_wording=allowed,
                forbidden_wording="oracle task specificity proves learned-policy task specificity or deployability",
            )
        )
    add(
        _metric(
            root,
            claim_id="U4_LEARNED_SPECIFICITY_BOUNDARY",
            layer="Useful",
            paper_role="adverse_control",
            metric="cross_objective_support_indicator",
            contrast="global mechanical mask versus global reconstruction mask",
            direction="one_would_support_learned_separation",
            reference_method="global reconstruction mask",
            candidate_method="global mechanical mask",
            reference_value=None,
            candidate_value=None,
            estimate=float(bool(p14["learned_cross_objective_supported"])),
            relative_effect=None,
            ci95_lower=None,
            ci95_upper=None,
            domains_improved="not_supported",
            protocol="source-trained global masks; same cohort and exact-cost alignment",
            source_artifact=p14_source,
            evidence_type="learned_adverse_control",
            deployable_status="source_trained_static",
            status="ADVERSE_CONTROL",
            manuscript_location="main",
            allowed_wording="the source-trained global mechanics mask does not reproduce oracle task separation",
            forbidden_wording="the learned mechanics policy outperforms reconstruction-driven acquisition",
        )
    )

    for claim_id, key, metric_name, role in (
        (
            "U5_RIDGE_HUBER_SPEARMAN",
            "rank_agreement/ridge__huber/spearman",
            "action_value_rank_spearman",
            "agreement",
        ),
        (
            "U5_RIDGE_HUBER_BEST_ACTION",
            "rank_agreement/ridge__huber/best_action_agreement",
            "best_action_agreement",
            "agreement",
        ),
        (
            "U5_RIDGE_HUBER_TOPK",
            "rank_agreement/ridge__huber/topk_jaccard",
            "top_k_jaccard",
            "agreement",
        ),
        (
            "U5_RIDGE_MLP_SPEARMAN",
            "rank_agreement/ridge__shallow_mlp/spearman",
            "action_value_rank_spearman",
            "dependence_boundary",
        ),
        (
            "U5_RIDGE_MLP_BEST_ACTION",
            "rank_agreement/ridge__shallow_mlp/best_action_agreement",
            "best_action_agreement",
            "dependence_boundary",
        ),
        (
            "U5_RIDGE_MLP_TOPK",
            "rank_agreement/ridge__shallow_mlp/topk_jaccard",
            "top_k_jaccard",
            "dependence_boundary",
        ),
    ):
        values = p15["bootstrap_intervals"][key]
        add(
            _metric(
                root,
                claim_id=claim_id,
                layer="Useful",
                paper_role=role,
                metric=metric_name,
                contrast=key.split("/")[1],
                direction="higher_is_more_value_rank_agreement",
                reference_method=key.split("/")[1].split("__")[0],
                candidate_method=key.split("/")[1].split("__")[1],
                reference_value=None,
                candidate_value=None,
                estimate=_float(values["point"], f"{claim_id} point"),
                relative_effect=None,
                ci95_lower=_float(values["interval"][0], f"{claim_id} lower"),
                ci95_upper=_float(values["interval"][1], f"{claim_id} upper"),
                domains_improved="6 held-out domains; equal-domain agreement",
                protocol="identical strict-OOF splits and frozen state bank across downstream learners",
                source_artifact=p15_source,
                evidence_type="strict_oof_learner_sensitivity",
                deployable_status="retrospective_non_deployable",
                status=(
                    "PARTIAL_STABILITY"
                    if "ridge__huber" in key
                    else "LEARNER_DEPENDENT"
                ),
                manuscript_location="main",
                allowed_wording="measurement value is downstream-predictor-conditioned task value",
                forbidden_wording="intrinsic mechanical value of a location; universal mechanical-value map",
            )
        )

    add(
        _metric(
            root,
            claim_id="O1_STATIC_SPEARMAN",
            layer="Observable",
            paper_role="headline",
            metric="strict_oof_action_value_spearman",
            contrast="O2 predicted score versus strict-OOF teacher value",
            direction="higher_is_better_observability",
            reference_method="strict-OOF teacher value",
            candidate_method="static O2 scorer",
            reference_value=None,
            candidate_value=None,
            estimate=_float(m1_spearman["point_estimate"], "M1 Spearman"),
            relative_effect=None,
            ci95_lower=_float(m1_spearman["lower"], "M1 Spearman lower"),
            ci95_upper=_float(m1_spearman["upper"], "M1 Spearman upper"),
            domains_improved=f"{m1_spearman['improved_domains']}/6",
            protocol="registered low-complexity static pre-acquisition scorer; strict source-only LODO",
            source_artifact=m1_boot_source,
            evidence_type="registered_static_observability",
            deployable_status="deployable_static_scorer",
            status="NOT_SUPPORTED",
            manuscript_location="main",
            allowed_wording="the registered static pre-acquisition representation did not establish transferable specimen-specific task-value observability",
            forbidden_wording="task value is information-theoretically unobservable",
        )
    )
    for claim_id, row, method in (
        ("O1_STATIC_SET_REGRET", m1_o2, "static O2 scorer"),
        ("O1_GLOBAL_SET_REGRET", m1_global, "global mechanical score"),
        ("O1_RANDOM_SET_REGRET", m1_random, "random median"),
    ):
        add(
            _metric(
                root,
                claim_id=claim_id,
                layer="Observable",
                paper_role="control",
                metric="exact_budget_set_regret",
                contrast=f"{method} exact-budget regret",
                direction="lower_is_better",
                reference_method="retrospective best legal set",
                candidate_method=method,
                reference_value=None,
                candidate_value=_float(row["mean_budgeted_regret"], f"{claim_id} regret"),
                estimate=_float(row["mean_budgeted_regret"], f"{claim_id} regret"),
                relative_effect=None,
                ci95_lower=None,
                ci95_upper=None,
                domains_improved="6-domain equal weighting",
                protocol="registered exact-budget action-set evaluation",
                source_artifact=m1_models_source,
                evidence_type="registered_static_observability_control",
                deployable_status="deployable_or_control",
                status="NOT_SUPPORTED" if claim_id == "O1_STATIC_SET_REGRET" else "REFERENCE",
                manuscript_location="main",
                allowed_wording="the static O2 scorer has higher exact-budget regret than global and random controls",
                forbidden_wording="larger model capacity would solve static observability",
            )
        )

    for claim_id, metric_name, field in (
        ("O2_TEACHER_TURNOVER", "best_action_turnover", "best_action_turnover"),
        ("O2_TEACHER_RANK", "rank_spearman_to_initial", "rank_spearman"),
        ("O2_TEACHER_TOPK", "top_k_jaccard_to_initial", "top_k_jaccard"),
        (
            "O2_TEACHER_OPPORTUNITY",
            "dynamic_vs_initial_opportunity",
            "dynamic_vs_initial_opportunity",
        ),
    ):
        add(
            _metric(
                root,
                claim_id=claim_id,
                layer="Observable",
                paper_role="headline" if claim_id in {"O2_TEACHER_TURNOVER", "O2_TEACHER_OPPORTUNITY"} else "supporting",
                metric=metric_name,
                contrast="strict-OOF teacher at 18.75% versus initial state",
                direction="descriptive_value_evolution",
                reference_method="initial strict-OOF teacher value",
                candidate_method="conditional strict-OOF teacher value at 18.75%",
                reference_value=None,
                candidate_value=None,
                estimate=_float(p9_endpoint[field], f"{claim_id} endpoint"),
                relative_effect=None,
                ci95_lower=None,
                ci95_upper=None,
                domains_improved="276 specimens across 6 held-out domains",
                protocol="causal measurement histories; strict-OOF retrospective teacher; specimen-first aggregation",
                source_artifact=p9_source,
                evidence_type="strict_oof_retrospective_teacher",
                deployable_status="retrospective_non_deployable",
                status="SUPPORTED",
                manuscript_location="main",
                allowed_wording="true strict-OOF conditional measurement value evolves with acquired evidence",
                forbidden_wording="the frozen real scorer reliably captures all dynamic opportunity",
            )
        )

    add(
        _metric(
            root,
            claim_id="O3_REAL_CHANGE",
            layer="Observable",
            paper_role="supporting",
            metric="cai_mae_change_from_initial",
            contrast="real state at 25% minus real state at initial checkpoint",
            direction="negative_is_improvement",
            reference_method="real state at initial checkpoint",
            candidate_method="real state at 25%",
            reference_value=None,
            candidate_value=None,
            estimate=_float(p10["endpoint_real_change_from_initial_mae"], "P10 real change"),
            relative_effect=None,
            ci95_lower=_float(p10["endpoint_real_change_ci95"][0], "P10 change lower"),
            ci95_upper=_float(p10["endpoint_real_change_ci95"][1], "P10 change upper"),
            domains_improved="6-domain equal weighting",
            protocol="matched frozen P2 state predictions along causal trajectories",
            source_artifact=p10_source,
            evidence_type="matched_state_diagnostic",
            deployable_status="diagnostic_frozen_state",
            status="MIXED",
            manuscript_location="main",
            allowed_wording="real partial measurements add signal relative to the static initial state",
            forbidden_wording="real content is more informative than positions or reconstruction",
        )
    )
    add(
        _metric(
            root,
            claim_id="O3_FULL_FIELD_RECOVERY",
            layer="Observable",
            paper_role="sensitivity",
            metric="static_to_independent_full_field_utility_recovery_fraction",
            contrast="P10 real-state recovery relative to I_field_selected endpoint",
            direction="higher_is_more_diagnostic_recovery",
            reference_method="static state and I_field_selected endpoint",
            candidate_method="real state at 25%",
            reference_value=None,
            candidate_value=None,
            estimate=_float(p10["endpoint_full_field_utility_recovery_fraction"], "P10 recovery"),
            relative_effect=None,
            ci95_lower=_float(p10["endpoint_recovery_ci95"][0], "P10 recovery lower"),
            ci95_upper=_float(p10["endpoint_recovery_ci95"][1], "P10 recovery upper"),
            domains_improved="6-domain equal weighting",
            protocol="P10 closure diagnostic with explicit I_field_selected full-field endpoint",
            source_artifact=p10_source,
            evidence_type="independent_endpoint_sensitivity",
            deployable_status="diagnostic_frozen_state",
            status="MIXED",
            manuscript_location="appendix",
            allowed_wording="P10 recovers a fraction of the static-to-independent-full-field error ratio",
            forbidden_wording="this fraction uses the registered B-family confirmatory endpoint",
        )
    )
    for claim_id, control, values in (
        ("O3_REAL_MINUS_POSITIONS", "positions_only", p10_positions),
        ("O3_REAL_MINUS_RECONSTRUCTION", "reconstruction", p10_reconstruction),
    ):
        add(
            _metric(
                root,
                claim_id=claim_id,
                layer="Observable",
                paper_role="central_adverse_control",
                metric="equal_domain_cai_mae",
                contrast=f"real minus {control} at 25%",
                direction="candidate_minus_reference; negative_would_favor_real",
                reference_method=control,
                candidate_method="real partial state",
                reference_value=_float(values["control_equal_domain_mae"], f"{control} MAE"),
                candidate_value=_float(values["real_equal_domain_mae"], "P10 real MAE"),
                estimate=_float(values["equal_domain_real_minus_control_mae"], f"{control} contrast"),
                relative_effect=None,
                ci95_lower=_float(values["ci95_lower"], f"{control} lower"),
                ci95_upper=_float(values["ci95_upper"], f"{control} upper"),
                domains_improved=f"{values['improved_domain_count']}/6",
                protocol="matched causal trajectory, exact acquired cost, action roster, cohort, and checkpoint",
                source_artifact=p10_contrast_source,
                evidence_type="matched_causal_control",
                deployable_status="diagnostic_frozen_state",
                status="ADVERSE_CONTROL",
                manuscript_location="main",
                allowed_wording="the registered representation does not establish specimen-specific measurement-content value beyond matched acquisition-geometry/reconstruction controls",
                forbidden_wording="geometry alone is universally sufficient; MRIS captures specimen mechanics beyond controls",
            )
        )

    for claim_id, control, point_key, interval_key, status in (
        (
            "O4_DYNAMIC_MINUS_STATIC",
            "static M1/O2",
            "endpoint_real_minus_static_regret",
            "endpoint_real_minus_static_regret_ci95",
            "NARROWLY_SUPPORTED",
        ),
        (
            "O4_DYNAMIC_MINUS_SHUFFLED",
            "dynamic shuffled content",
            "endpoint_real_minus_shuffled_regret",
            "endpoint_real_minus_shuffled_regret_ci95",
            "ADVERSE_CONTROL",
        ),
    ):
        interval = p11[interval_key]
        add(
            _metric(
                root,
                claim_id=claim_id,
                layer="Observable",
                paper_role="headline" if status == "NARROWLY_SUPPORTED" else "central_adverse_control",
                metric="one_step_value_regret_at_18_75_percent",
                contrast=f"dynamic real minus {control}",
                direction="negative_favors_dynamic_real",
                reference_method=control,
                candidate_method="dynamic real-state scorer",
                reference_value=None,
                candidate_value=None,
                estimate=_float(p11[point_key], f"{claim_id} point"),
                relative_effect=None,
                ci95_lower=_float(interval[0], f"{claim_id} lower"),
                ci95_upper=_float(interval[1], f"{claim_id} upper"),
                domains_improved="5/6" if status == "NARROWLY_SUPPORTED" else "1/6",
                protocol="same legal actions and exact costs at the final tested decision checkpoint",
                source_artifact=p11_source,
                evidence_type="registered_dynamic_valuation_control",
                deployable_status="frozen_learned_scorer",
                status=status,
                manuscript_location="main",
                allowed_wording="the dynamic real-state scorer narrowly lowers endpoint regret versus static, while shuffled content remains better",
                forbidden_wording="accumulated real ultrasonic content caused the gain; dynamic MRIS is uniformly superior",
            )
        )

    p12_rows = (
        (
            "A1_VALUATION_SUBSTITUTION",
            "retrospective valuation substitution",
            p12["valuation_improvement"],
            [-value for value in reversed(p12["bootstrap_intervals"]["B_minus_A"])],
            "retrospective_non_deployable",
        ),
        (
            "A1_LEARNED_PLANNING_SUBSTITUTION",
            "bounded learned planning substitution",
            p12["learned_planning_improvement"],
            [-value for value in reversed(p12["bootstrap_intervals"]["C_minus_A"])],
            "diagnostic_learned_planner",
        ),
        (
            "A1_TRUE_VALUE_PLANNING_SUBSTITUTION",
            "true-value stronger-planning substitution",
            p12["true_planning_improvement"],
            [-value for value in reversed(p12["bootstrap_intervals"]["E_minus_D"])],
            "retrospective_non_deployable",
        ),
    )
    for claim_id, contrast, estimate, interval, deployable in p12_rows:
        add(
            _metric(
                root,
                claim_id=claim_id,
                layer="Actionable",
                paper_role="headline" if claim_id == "A1_VALUATION_SUBSTITUTION" else "supporting",
                metric="cai_auebc_improvement",
                contrast=contrast,
                direction="positive_is_improvement",
                reference_method="frozen current component",
                candidate_method=contrast,
                reference_value=None,
                candidate_value=None,
                estimate=_float(estimate, f"{claim_id} estimate"),
                relative_effect=None,
                ci95_lower=_float(interval[0], f"{claim_id} lower"),
                ci95_upper=_float(interval[1], f"{claim_id} upper"),
                domains_improved="6-domain equal weighting",
                protocol="one-component-at-a-time retrospective substitution under exact acquisition budgets",
                source_artifact=p12_source,
                evidence_type="retrospective_component_substitution",
                deployable_status=deployable,
                status="SUPPORTED_RETROSPECTIVE",
                manuscript_location="main",
                allowed_wording="retrospective substitutions distinguish valuation and bounded set-planning gaps",
                forbidden_wording="a representation bottleneck is established; all layers are proven bottlenecks; oracle substitutions are deployable",
            )
        )

    for claim_id, method in (
        ("A2_GREEDY_PLANNING_REGRET", "current_greedy"),
        ("A2_BEAM4_PLANNING_REGRET", "beam_width_4"),
    ):
        values = p13["aggregate"][method]
        interval = p13["planning_regret_intervals"][method]
        add(
            _metric(
                root,
                claim_id=claim_id,
                layer="Actionable",
                paper_role="headline" if method == "current_greedy" else "control",
                metric="bounded_set_planning_regret",
                contrast=f"{method} versus retrospective joint near-oracle reachable pool",
                direction="lower_is_better; positive_regret_is_planning_gap",
                reference_method="retrospective joint near-oracle reachable pool",
                candidate_method=method,
                reference_value=0.0,
                candidate_value=_float(values["planning_regret"], f"{claim_id} regret"),
                estimate=_float(values["planning_regret"], f"{claim_id} regret"),
                relative_effect=None,
                ci95_lower=_float(interval[0], f"{claim_id} lower"),
                ci95_upper=_float(interval[1], f"{claim_id} upper"),
                domains_improved="6-domain equal weighting",
                protocol="bounded two-action reachable-pool planning diagnostic at 6.25%",
                source_artifact=p13_source,
                evidence_type="retrospective_bounded_set_planning",
                deployable_status="retrospective_non_deployable",
                status="SUPPORTED_GAP",
                manuscript_location="main",
                allowed_wording="a bounded set-planning gap remains under the registered reachable-pool diagnostic",
                forbidden_wording="near-oracle planning is deployable; the result establishes arbitrary-horizon planning regret",
            )
        )

    add(
        _metric(
            root,
            claim_id="A3_FEEDBACK_BENEFIT",
            layer="Actionable",
            paper_role="central_adverse_control",
            metric="cai_auebc_feedback_benefit",
            contrast="no-feedback minus feedback",
            direction="positive_favors_feedback",
            reference_method="frozen no-feedback policy",
            candidate_method="frozen feedback policy",
            reference_value=None,
            candidate_value=None,
            estimate=_float(p16["overall_feedback_auebc_benefit"], "P16 feedback benefit"),
            relative_effect=None,
            ci95_lower=_float(p16_interval[0], "P16 lower"),
            ci95_upper=_float(p16_interval[1], "P16 upper"),
            domains_improved=f"{feedback_improved}/6",
            protocol="frozen feedback/no-feedback trajectories; specimen-first synchronized bootstrap",
            source_artifact=p16_source,
            evidence_type="frozen_feedback_control",
            deployable_status="frozen_learned_policy",
            status="ADVERSE_CONTROL",
            manuscript_location="main",
            allowed_wording="feedback is adverse under the frozen cross-domain protocol",
            forbidden_wording="feedback improves acquisition; value evolution explains the feedback mechanism",
        )
    )
    baseline = _float(p7["baseline_cai_auebc"], "P7 baseline AUEBC")
    mavis = _float(p7["mavis_cai_auebc"], "P7 MAVIS AUEBC")
    add(
        _metric(
            root,
            claim_id="A4_BASELINE_MINUS_MAVIS",
            layer="Actionable",
            paper_role="headline_boundary",
            metric="cai_auebc",
            contrast="strongest deployable baseline minus frozen MAVIS",
            direction="positive_favors_MAVIS",
            reference_method=p7["strongest_deployable_baseline"],
            candidate_method="frozen MAVIS",
            reference_value=baseline,
            candidate_value=mavis,
            estimate=baseline - mavis,
            relative_effect=None,
            ci95_lower=_float(p7["mavis_control_minus_reference_ci95_lower"], "P7 lower"),
            ci95_upper=_float(p7["mavis_control_minus_reference_ci95_upper"], "P7 upper"),
            domains_improved=f"{p7['mavis_improved_domain_count']}/{p7['domain_count']}",
            protocol="frozen source-selected cross-domain policy; exact cost; specimen-first equal-domain AUEBC",
            source_artifact=p7_source,
            evidence_type="frozen_deployable_endpoint",
            deployable_status="frozen_learned_policy",
            status="BOUNDARY_SUPPORTED",
            manuscript_location="main",
            allowed_wording="the frozen learned acquisition policy did not outperform the strongest deployable baseline",
            forbidden_wording="MAVIS outperforms the strongest deployable baseline; MAVIS improves most held-out domains",
        )
    )

    claim_ids = [row.claim_id for row in rows]
    if len(claim_ids) != len(set(claim_ids)):
        raise PaperEvidenceError("paper claim IDs are not unique")
    return tuple(rows)


def _csv_text(rows: list[dict[str, object]], columns: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: "" if value is None else format(value, ".17g") if type(value) is float else value
                for key, value in row.items()
            }
        )
    return stream.getvalue()


def _reconciliation_text(rows: tuple[PaperMetric, ...]) -> str:
    matched = next(row for row in rows if row.claim_id == "U1_MATCHED_FIELD")
    independent = next(
        row for row in rows if row.claim_id == "U1_INDEPENDENT_FIELD_SENSITIVITY"
    )
    sparse = next(row for row in rows if row.claim_id == "U2_SPARSE_RETENTION")
    sparse_gap = next(row for row in rows if row.claim_id == "U2_SPARSE_FULL_GAP")
    return f"""# Evidence Authority Reconciliation

Generated deterministically from frozen machine-readable evidence.

## Registered scalar/spatial authority

Paper 1 uses `{matched.contrast}` as its confirmatory contrast. The
reference/candidate MAEs are `{matched.reference_value:.10f}` and
`{matched.candidate_value:.10f}`; the effect is `{matched.estimate:.10f}` with
simultaneous interval `[{matched.ci95_lower:.10f}, {matched.ci95_upper:.10f}]`
and `{matched.domains_improved}` held-out domains improved.

`I_field_selected` is a different independent metadata-only-prefix estimator
with equal-domain MAE `{independent.estimate:.10f}`. It is sensitivity evidence,
not the registered B-family endpoint. The historical `0.099568606` comparison
must not replace the matched confirmatory effect.

## Sparse-retention authority

At 25% nominal normalized-raster density, full and sparse MAEs are
`{sparse.reference_value:.10f}` and `{sparse.candidate_value:.10f}` and gain
retention is `{sparse.estimate:.10f}`. The distinct sparse-minus-full gap is
`{sparse_gap.estimate:.10f}` with simultaneous interval
`[{sparse_gap.ci95_lower:.10f}, {sparse_gap.ci95_upper:.10f}]`; sparse is worse
in all six domains. The gap interval is not the interval for surface-to-sparse
improvement.

## Paper rule

`PAPER_CANONICAL_METRICS.csv` is the only numeric source for manuscript prose,
captions, and main tables. Historical artifacts remain unchanged. P10's
explicit `I_field_selected` recovery endpoint remains a separately labeled
diagnostic.
"""


def _claim_map_text(rows: tuple[PaperMetric, ...]) -> str:
    lines = [
        "# AEI Paper Claim Map",
        "",
        "All rows use physical specimens and held-out domains as the statistical units.",
        "",
        "| ID | Layer | Status | Evidence type | Source | Allowed wording | Forbidden wording |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        allowed = row.allowed_wording.replace("|", "\\|")
        forbidden = row.forbidden_wording.replace("|", "\\|")
        lines.append(
            f"| {row.claim_id} | {row.layer} | {row.status} | {row.evidence_type} | "
            f"`{row.source_artifact}` | {allowed} | {forbidden} |"
        )
    lines.extend(
        [
            "",
            "Required term: `downstream-predictor-conditioned task value`.",
            "Retrospective teachers, oracles, and substitutions are explicitly non-deployable.",
            "",
        ]
    )
    return "\n".join(lines)


def write_paper_authority(root: Path, output_dir: Path) -> None:
    """Write the canonical metric, claim, reconciliation, and source-hash files."""
    rows = build_canonical_metrics(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_columns = [field.name for field in fields(PaperMetric)]
    metric_payload = _csv_text([asdict(row) for row in rows], metric_columns)
    (output_dir / "PAPER_CANONICAL_METRICS.csv").write_text(
        metric_payload, encoding="utf-8", newline=""
    )
    (output_dir / "EVIDENCE_AUTHORITY_RECONCILIATION.md").write_text(
        _reconciliation_text(rows), encoding="utf-8"
    )
    (output_dir / "PAPER_CLAIM_MAP.md").write_text(
        _claim_map_text(rows), encoding="utf-8"
    )
    hash_rows = [
        {
            "source_artifact": relative,
            "source_hash": _sha256(_source(root, relative)),
            "bytes": _source(root, relative).stat().st_size,
            "role": "frozen_input_evidence",
        }
        for relative in _SOURCE_PATHS
    ]
    hash_rows.append(
        {
            "source_artifact": "results/mavis/p7_final_frozen_eval/",
            "source_hash": _P7_TREE,
            "bytes": "",
            "role": "frozen_tree_state",
        }
    )
    hash_payload = _csv_text(
        hash_rows, ["source_artifact", "source_hash", "bytes", "role"]
    )
    (output_dir / "PAPER_SOURCE_HASHES.csv").write_text(
        hash_payload, encoding="utf-8", newline=""
    )
