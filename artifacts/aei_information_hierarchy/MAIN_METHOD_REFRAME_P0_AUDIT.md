# AEI Main-Method Reframe P0 Audit

Audit date: 2026-08-28  
Repository: `git@github.com:Orangekostar/diff.git`  
Status: `READ_ONLY_AUDIT_COMPLETE`

## Repository identity

| Item | Audited value |
|---|---|
| Authoritative base branch | `aei-positive-progressive-narrative` |
| Required base SHA | `ff4730b3fcf368d6ac43f0f72f034703e1556f7d` |
| Starting local SHA | `ff4730b3fcf368d6ac43f0f72f034703e1556f7d` |
| Starting remote SHA | `ff4730b3fcf368d6ac43f0f72f034703e1556f7d` |
| Working branch | `aei-main-method-reframe` |
| Starting worktree | clean |
| Starting `git diff --check` | clean |
| Baseline paper suite | `106 passed in 23.32s` |
| Baseline command | `PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_*.py` |

The existing linked worktree was already isolated and clean at the required
base SHA. The new branch was created in that worktree; no user work was reset,
discarded, or merged.

## Current manuscript presentation locks

| Surface | Current state at the base SHA | Required reframe |
|---|---|---|
| Top-level sections | 6 | retain 6 |
| Related Work subsections | 4 | reduce to 3 |
| Section 3 subsections | 5 | reduce to 3 |
| Section 4 subsections | 6 | reduce to 3 |
| Part-I result stages | 6 | reduce to 3 |
| Part-II result stages | 6 | reduce to 3 |
| Main figures | 4 | retain 4 with revised roles |
| Main tables | 3 | reduce to 2 |
| Canonical claims required in main | 39/39 | replace with visibility-aware combined coverage |
| Closest-work table | main Table 1 | internal artifact only |
| Deployment-calibration stage | dedicated Part-II stage | remove as a headline stage |

The current Part-II identity is `Evidence-Calibrated Decision Realization`.
The required Part-II identity is `State-Conditioned Task-Oriented Acquisition`.

## Current terminology exposure in `main.tex`

Exact case-insensitive occurrence counts at the base SHA:

| Term | Count | Current locations/role |
|---|---:|---|
| `MAVIS` | 2 | one method-section use plus the A4 claim-ID comment |
| `residual deployable gap` | 2 | abstract and Results |
| `not performance-superior` | 2 | abstract/Results and Conclusions surface |
| `frozen` | 25 | abstract, Introduction, method, protocol, Results, interpretation, Conclusions |
| `post-freeze` | 2 | Introduction and Section 4 chronology |
| `hash-bound` | 2 | Introduction and Section 4 chronology |
| `not a new` | 1 | contribution novelty disclaimer |
| `not the novelty` | 0 | equivalent defensive wording appears in Related Work |

The P7 endpoint is currently promoted in the abstract, the method/deployment
framing, Results 5.2.6, Figure 4, the 12-row main evidence table, and the
Conclusions. The internal comparator name `mvd_m1_o2` is absent from main prose,
but its result is presented as the `strongest deployable baseline`, which
creates a method-versus-competitor reading. The paper-level framework is
defined in the title, abstract, Introduction, and Section 3 but is not kept
distinct from the concrete closed-loop implementation throughout those
surfaces.

## Code and test locks

| Old lock | Enforcing source/test |
|---|---|
| P7 boundary in abstract | `tests/test_mavis_aei_paper_manuscript.py::test_aei_paper_abstract_states_progressive_evidence_and_boundary` |
| P7 boundary in conclusion | `tests/test_mavis_aei_paper_manuscript.py::test_aei_paper_introduction_and_conclusion_close_both_progressive_parts` |
| A3/A4 boundary in main Results | `tests/test_mavis_aei_paper_manuscript.py::test_aei_paper_keeps_central_adverse_controls_in_results` |
| Six Part-I stages | `PART1_STAGES` and `test_part1_stage_order_is_spatial_sparse_heterogeneous_objective_state_predictor` |
| Six Part-II stages | `PART2_STAGES` and `test_part2_stage_order_is_static_dynamic_attribution_decomposition_planning_calibration` |
| Five Section-3 subsections | `paper_aei_information_hierarchy/MANUSCRIPT_OUTLINE.md` and `main.tex` |
| Six Section-4 subsections | `paper_aei_information_hierarchy/MANUSCRIPT_OUTLINE.md` and `main.tex` |
| Three main tables | `aei_paper_package._TABLES`, `test_aei_paper_uses_exactly_four_main_figures_and_three_main_tables`, `test_aei_paper_builds_exactly_three_main_tables`, and `ValidationReport.table_count == 3` |
| Closest-work main-table packaging | `aei_paper_package._TABLES`, `_flat_manuscript()`, and `tests/test_mavis_aei_paper_package.py` |
| A3/A4 in Figure 4 | `aei_paper_figures._figure4_rows()` and `test_aei_paper_figure4_preserves_feedback_and_final_boundary` |
| Audit-heavy 12-row evidence table | `aei_paper_tables._progressive_rows()` and the Table-3 tests |
| All 39 claims in main | `test_aei_paper_results_map_every_canonical_claim` and `aei_paper_validation.validate_paper()` |

The current validation report exposes `mapped_claim_count` and `table_count`.
It must be replaced by explicit main-visible and combined-evidence coverage,
while retaining 39 canonical claims as the immutable authority.

## Canonical evidence and frozen-path gate

| Item | Audited value |
|---|---|
| Canonical row count | 39 |
| Canonical CSV SHA-256 | `f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6` |
| Frozen-path diff from required base | empty |
| New training authorized | no |
| Numerical recomputation authorized | no |

Frozen roots:

```text
results/p1_full_field_oracle/
results/p5_sparse_scan/
results/mvd/
results/mavis/
results/mavis_science_closure/
artifacts/mavis/
artifacts/mavis_science_closure/
artifacts/mvd_authority/
artifacts/mavis_authority/
```

`artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv` is also
immutable. The governing execution flag is `NO_NEW_TRAINING`.

## Approved reframe contract

```text
PRIMARY METHOD
Task-Relevant Information Acquisition

PART I
Task-Relevant Information Characterization

PART II
State-Conditioned Task-Oriented Acquisition

MAVIS
codebase identity for one supervised closed-loop implementation; supplement only

mvd_m1_o2
static reference; not a published competing method
```

Claim visibility will be split into 12 `MAIN_HEADLINE`, 15 `MAIN_SUPPORT`, one
`MAIN_SYSTEM_DIAGNOSTIC`, and 11 `SUPPLEMENT_ONLY` claims. A4 will appear once
in main Results as a concise system diagnostic and in full in the supplement.
A3 will be supplement-only. Main plus supplement plus the visibility map must
cover all 39 canonical claims without changing any numerical direction.

## P0 decision

`GO`: the authoritative branch and SHA match, the worktree was clean, the
canonical hash matches, the frozen-path diff is empty, and all obsolete
presentation locks have been located. Narrative, generator, test, and package
implementation may begin on `aei-main-method-reframe`.
