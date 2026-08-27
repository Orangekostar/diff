# Positive Progressive Narrative P0 Audit

Audit date: 2026-08-27

## 1. Repository state

- Branch at audit: `main`; implementation branch:
  `aei-positive-progressive-narrative`.
- HEAD: `9c2d0f1c97a96358c5e697f488373254a099d0fe`.
- Relation to required base `9c2d0f1`: exact, with left/right count `0/0`.
- Worktree: clean; `git diff --check` produced no output.
- Baseline paper suite: 93 tests passed under
  `python -m pytest` with `PYTHONPATH=src` in the paper build environment.

## 2. Old narrative locks

The old RQ1/RQ2/RQ3 structure is hard-coded in `main.tex`,
`MANUSCRIPT_OUTLINE.md`, and `test_mavis_aei_paper_manuscript.py`. The U/O/A
result blocks also appear in `CLAIM_SENTENCE_BANK.md`, the paper README and
supplement, `AEI_SCOPE_AND_STRUCTURE_LEDGER.md`, the figure/table generators,
the validation contract, and manuscript/table/validation tests. The old title
appears in the manuscript, README, supplement, and manuscript test. The old
negative abstract and conclusion contract is asserted in manuscript and
validation tests.

The filenames `figure1_information_hierarchy`, `figure2_usefulness`,
`figure3_observability`, and `figure4_actionability` are fixed in the
manuscript, figure generator, package builder, manuscript/package tests, and
generated manifests. `table3_hierarchy_evidence` is fixed in the manuscript,
table generator, package builder, manuscript/table/package tests, and generated
manifests. Files below `results/aei_information_hierarchy/deterministic_package`
are derived and must be regenerated rather than edited.

Canonical U/O/A layers in `PAPER_CANONICAL_METRICS.csv`, `PAPER_CLAIM_MAP.md`,
and `aei_paper_evidence.py` are provenance, not narrative locks. Their
scientific directions remain unchanged.

## 3. Canonical evidence immutability

- Claim count: 39.
- SHA-256:
  `f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`.
- No canonical numerical rewrite or historical result regeneration is needed.

## 4. Part-I evidence map

| Stage | Claims | Source | Headline evidence | Required boundary |
|---|---|---|---|---|
| I1 Spatial enrichment | `U1_*` | P1 metrics/domain metrics | Matched MAE 0.1892044107 to 0.1284893565; 32.0897%; CI [0.0066387, 0.1536430]; 5/6 | `I_field_selected` is sensitivity only |
| I2 Sparse recoverability | `U2_*` | P5 retention/bootstrap | 89.8973% retention; sparse/full MAE 0.1345137120/0.1284893565; gap 0.0060243555, CI [0.0017301, 0.0108331] | No scanner-time conversion; recovery is incomplete |
| I3 Spatial heterogeneity | `U3_*` | MVD M0 summary | Mechanical AUEBC 0.0134579441 vs uniform 0.0173634580 and reconstruction 0.0171876604; effects 0.0039055139/0.0037297163; both 6/6 | Oracle is retrospective and non-deployable |
| I4 Objective conditioning | `U4_*` | P14 summary | CAI effect 0.0486228431, CI [0.0452732645, 0.0520453505]; reconstruction effect 0.0005502549, CI [0.0005006233, 0.0006062920] | Say `registered normalized-RGB-MSE reconstruction objective`; learned global masks do not reproduce separation |
| I5 State conditioning | `O2_*` | P9 summary | Turnover 70.4%, rank 0.405, top-five Jaccard 0.307, opportunity 0.00531 | Retrospective teacher-target characterization, not scorer success |
| I6 Predictor conditioning | `U5_*` | P15 summary | Ridge/Huber/MLP OOF MAE 0.08964/0.08618/0.15067; Spearman 0.762/0.116 | Retain predictor index `f`; equal-accuracy structural variation unresolved |

## 5. Part-II evidence map

| Stage | Claims | Source | Framework evidence | Required boundary |
|---|---|---|---|---|
| II1 Static reference | `O1_*` | MVD M1 bootstrap/model metrics | Spearman -0.0196, CI [-0.0591, 0.0195] | Motivation/reference, not an information-theoretic failure claim |
| II2 Dynamic valuation | `O4_DYNAMIC_MINUS_STATIC` | P11 summary | Regret -0.001260, CI [-0.002123, -0.000444], 5/6 | Does not attribute improvement to measured content |
| II3 Information-source attribution | `O3_*`, `O4_DYNAMIC_MINUS_SHUFFLED` | P10 summary/contrasts; P11 | Real change -0.000731; 25% MAE 0.12463/0.10722/0.09043; shuffled contrast +0.0002328 | Use `acquired-position/history control`; retain every adverse direction |
| II4 Valuation/planning decomposition | `A1_*` | P12 summary | Effects 4.979e-5, 3.164e-6, 1.117e-4 | Retrospective substitutions are non-deployable |
| II5 Bounded set realization | `A2_*` | P13 summary | Greedy regret 1.207e-4, CI [1.033e-4, 1.386e-4] | Bounded two-action reachable pool only |
| II6 Deployment calibration | `A3_*`, `A4_*` | P16 summary; P7 claim evidence | No-feedback advantage 1.496e-5; learned/reference AUEBC 0.1250531822/0.1249920401; residual gap 6.114e-5; favorable 2/6 | Learned policy is not performance-superior |

## 6. Positions/history control audit

`build_mris_input(..., mode="positions_only")` inherits the exact
`state.acquired_positions`, context, and budget while removing measurement
values. Frozen source plans include `random`, `uniform`,
`reconstruction_driven`, `sequential_mechanical_oracle`, and
`one_shot_mechanical_oracle`; pure-geometry wording is therefore not licensed.

`per_specimen_predictions.parquet` has no trajectory-family or method field.
It cannot support a no-training family stratification:
`POSITION_HISTORY_STRATIFICATION_NOT_IDENTIFIABLE`.

## 7. P14 reconstruction metric audit

The exact field is `normalized_rgb_mse`. The allowed manuscript phrase is
`registered normalized-RGB-MSE reconstruction objective`.

## 8. Six-section and Section 5 contracts

The top-level sections are Introduction; Related Research and Problem
Formulation; Task-Relevant Information Acquisition Framework; Multi-Domain
CFRP Case Study and Experimental Design; Experimental Results and Discussion;
and Conclusions.

Part I stages are spatial morphology, sparse observation, spatial
heterogeneity, objective conditioning, state evolution, and predictor
conditioning. Part II stages are static reference, dynamic valuation,
information-source attribution, valuation/planning decomposition, bounded set
realization, and deployment calibration. Section 5.3 is synthesis only.

## 9. Figure redesign

- Figure 1: two-part acquisition framework; new stem
  `figure1_task_relevant_acquisition_framework`.
- Figure 2: matched morphology, sparse retention, spatial heterogeneity, and
  objective conditioning; new stem `figure2_information_characterization`.
- Figure 3: P9 evolution, static/dynamic valuation, and matched attribution
  controls; new stem `figure3_state_conditioned_value`.
- Figure 4: component substitutions, bounded planning, no-feedback reference,
  and final residual gap; new stem `figure4_decision_calibration`.

All source rows remain claim-, source-, and hash-bound.

## 10. Table redesign

Tables 1 and 2 retain closest-work and protocol roles. Table 3 becomes
`table3_progressive_evidence_chain` with exactly 12 rows ordered I1-I6 and
II1-II6, explicit conclusion/boundary columns, claim provenance, source
artifacts/hashes, and canonical authority hash.

## 11. Files to modify

Authored paper sources, the scope ledger, figure/table/validation/package code,
and the six existing paper narrative/figure/table/validation/package tests are
in scope. Paper assets and `results/aei_information_hierarchy` outputs will be
regenerated. Frozen result trees and canonical metrics are excluded.

## 12. Files to create

This audit, `PAPER_POSITIVE_NARRATIVE_MAP.csv/.md`, the completion audit, the
design/implementation plan, and a fresh review record will be created.

## 13. Test replacement

Tests will enforce two RQs, two experimental parts, both six-stage orders,
U/O/A as validation criteria, O2 in Part I, O1 as a reference, bounded
position/history and P14 wording, 39 unique stage assignments, canonical hash
immutability, all adverse directions, new names, deterministic rendering, and
deterministic packaging.

## 14. Training decision

No new model training is required or authorized.

## 15. Scientific integrity

The refactor changes narrative role only. Claim IDs, canonical layers,
chronology, values, intervals, domain directions, statistical units, and
deployability boundaries remain unchanged. Sparse/full, position/history,
reconstruction, shuffled, feedback, and final policy/reference directions stay
visible and are never reversed.
