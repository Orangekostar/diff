# Paper Positive Progressive Narrative Map

This map changes narrative placement only. Numerical authority remains
`PAPER_CANONICAL_METRICS.csv`, and timing authority remains
`PAPER_EVIDENCE_CHRONOLOGY.csv`.

## Writing visibility contract

The CSV field `narrative_visibility` is mapped mechanically from the
`visibility` field in `PAPER_CLAIM_VISIBILITY_MAP.csv`. The allowed writing
visibility enums are:

| Enum | Definition |
|---|---|
| `MAIN_REQUIRED` | The claim is required in the main paper narrative. |
| `MAIN_OPTIONAL` | The claim may support the main paper narrative when needed. |
| `SUPPLEMENT_REQUIRED` | The claim is required in the supplement and is not required in the main narrative. |
| `INTERNAL_ONLY` | The claim remains available for internal traceability and is omitted from the main paper and supplement. |

The current map has 42 claims and no `INTERNAL_ONLY` claims. The complete
claim-level classification is:

| Claim ID | `narrative_visibility` |
|---|---|
| `U1_MATCHED_FIELD` | `MAIN_REQUIRED` |
| `U1_SURFACE_FIELD` | `MAIN_OPTIONAL` |
| `U1_INDEPENDENT_FIELD_SENSITIVITY` | `SUPPLEMENT_REQUIRED` |
| `U2_SPARSE_RETENTION` | `MAIN_REQUIRED` |
| `U2_SPARSE_GAIN` | `MAIN_OPTIONAL` |
| `U2_SPARSE_FULL_GAP` | `MAIN_OPTIONAL` |
| `U3_UNIFORM_ORACLE` | `MAIN_REQUIRED` |
| `U3_RECONSTRUCTION_ORACLE` | `SUPPLEMENT_REQUIRED` |
| `U3_HEADROOM_RETENTION` | `MAIN_OPTIONAL` |
| `U3_CAI_VS_APPEARANCE_SALIENCY_AUEBC` | `MAIN_REQUIRED` |
| `U4_CAI_SALIENCY_MAP_SPEARMAN` | `MAIN_OPTIONAL` |
| `U4_CAI_SALIENCY_TOP10_OVERLAP` | `MAIN_OPTIONAL` |
| `U4_ORACLE_CAI_SPECIFICITY` | `SUPPLEMENT_REQUIRED` |
| `U4_ORACLE_IMAGE_SPECIFICITY` | `SUPPLEMENT_REQUIRED` |
| `U4_LEARNED_SPECIFICITY_BOUNDARY` | `SUPPLEMENT_REQUIRED` |
| `U5_RIDGE_HUBER_SPEARMAN` | `MAIN_REQUIRED` |
| `U5_RIDGE_HUBER_BEST_ACTION` | `SUPPLEMENT_REQUIRED` |
| `U5_RIDGE_HUBER_TOPK` | `SUPPLEMENT_REQUIRED` |
| `U5_RIDGE_MLP_SPEARMAN` | `MAIN_OPTIONAL` |
| `U5_RIDGE_MLP_BEST_ACTION` | `SUPPLEMENT_REQUIRED` |
| `U5_RIDGE_MLP_TOPK` | `SUPPLEMENT_REQUIRED` |
| `O1_STATIC_SPEARMAN` | `MAIN_OPTIONAL` |
| `O1_STATIC_SET_REGRET` | `SUPPLEMENT_REQUIRED` |
| `O1_GLOBAL_SET_REGRET` | `SUPPLEMENT_REQUIRED` |
| `O1_RANDOM_SET_REGRET` | `SUPPLEMENT_REQUIRED` |
| `O2_TEACHER_TURNOVER` | `MAIN_REQUIRED` |
| `O2_TEACHER_RANK` | `MAIN_OPTIONAL` |
| `O2_TEACHER_TOPK` | `MAIN_OPTIONAL` |
| `O2_TEACHER_OPPORTUNITY` | `MAIN_OPTIONAL` |
| `O3_REAL_CHANGE` | `MAIN_OPTIONAL` |
| `O3_FULL_FIELD_RECOVERY` | `SUPPLEMENT_REQUIRED` |
| `O3_REAL_MINUS_POSITIONS` | `MAIN_OPTIONAL` |
| `O3_REAL_MINUS_RECONSTRUCTION` | `SUPPLEMENT_REQUIRED` |
| `O4_DYNAMIC_MINUS_STATIC` | `MAIN_REQUIRED` |
| `O4_DYNAMIC_MINUS_SHUFFLED` | `MAIN_OPTIONAL` |
| `A1_VALUATION_SUBSTITUTION` | `MAIN_REQUIRED` |
| `A1_LEARNED_PLANNING_SUBSTITUTION` | `MAIN_OPTIONAL` |
| `A1_TRUE_VALUE_PLANNING_SUBSTITUTION` | `MAIN_REQUIRED` |
| `A2_GREEDY_PLANNING_REGRET` | `MAIN_REQUIRED` |
| `A2_BEAM4_PLANNING_REGRET` | `MAIN_OPTIONAL` |
| `A3_FEEDBACK_BENEFIT` | `SUPPLEMENT_REQUIRED` |
| `A4_BASELINE_MINUS_MAVIS` | `MAIN_REQUIRED` |

## Part I: Information Characterization

| Stage | Main required claims | Main optional claims | Supplement required claims | Scientific role | Boundary |
|---|---|---|---|---|---|
| `I-A_SPATIAL_AND_SPARSE` | `U1_MATCHED_FIELD`; `U2_SPARSE_RETENTION` | `U1_SURFACE_FIELD`; `U2_SPARSE_GAIN`; `U2_SPARSE_FULL_GAP` | `U1_INDEPENDENT_FIELD_SENSITIVITY` | Spatial information matters, and most of its gain survives sparse observation. | The matched estimator remains confirmatory; sparse retention is not full-field recovery, and normalized-raster fraction is not scanner time. |
| `I-B_TASK_CONDITIONED_SPATIAL_VALUE` | `U3_UNIFORM_ORACLE`; `U3_CAI_VS_APPEARANCE_SALIENCY_AUEBC` | `U3_HEADROOM_RETENTION`; `U4_CAI_SALIENCY_MAP_SPEARMAN`; `U4_CAI_SALIENCY_TOP10_OVERLAP` | `U3_RECONSTRUCTION_ORACLE`; `U4_ORACLE_CAI_SPECIFICITY`; `U4_ORACLE_IMAGE_SPECIFICITY`; `U4_LEARNED_SPECIFICITY_BOUNDARY` | CAI-oriented measurement value extends beyond the preregistered task-agnostic C-scan saliency comparator. | The appearance oracle is retrospective/nondeployable and does not imply scanner time; legacy reconstruction evidence is supplement-only and distinct from appearance saliency. |
| `I-C_STATE_AND_PREDICTOR_CONDITIONED_VALUE` | `U5_RIDGE_HUBER_SPEARMAN`; `O2_TEACHER_TURNOVER` | `U5_RIDGE_MLP_SPEARMAN`; `O2_TEACHER_RANK`; `O2_TEACHER_TOPK`; `O2_TEACHER_OPPORTUNITY` | `U5_RIDGE_HUBER_BEST_ACTION`; `U5_RIDGE_HUBER_TOPK`; `U5_RIDGE_MLP_BEST_ACTION`; `U5_RIDGE_MLP_TOPK` | Measurement value changes with accumulated evidence and the downstream predictor. | The teacher target is strict-OOF and retrospective; variation among equally accurate structurally distinct predictors remains unresolved. |

## Part II: State-Conditioned Task-Oriented Acquisition

| Stage | Main required claims | Main optional claims | Supplement required claims | Scientific role | Boundary |
|---|---|---|---|---|---|
| `II-A_STATE_CONDITIONED_VALUATION` | `O4_DYNAMIC_MINUS_STATIC` | `O1_STATIC_SPEARMAN` | `O1_STATIC_SET_REGRET`; `O1_GLOBAL_SET_REGRET`; `O1_RANDOM_SET_REGRET` | State-conditioned valuation improves next-action estimation over the static reference. | The static comparison is a registered reference, not an information-theoretic impossibility claim. |
| `II-B_SOURCE_AND_COMPONENT_DECOMPOSITION` | `A1_VALUATION_SUBSTITUTION`; `A1_TRUE_VALUE_PLANNING_SUBSTITUTION` | `O3_REAL_CHANGE`; `O3_REAL_MINUS_POSITIONS`; `O4_DYNAMIC_MINUS_SHUFFLED`; `A1_LEARNED_PLANNING_SUBSTITUTION` | `O3_FULL_FIELD_RECOVERY`; `O3_REAL_MINUS_RECONSTRUCTION` | Matched controls identify state signals, and component substitutions localize headroom. | Real change alone does not identify measured-content value; acquired-position/history controls retain their frozen semantics, while the legacy reconstruction control is supplement-only and distinct from appearance saliency. |
| `II-C_COST_CONSTRAINED_REALIZATION` | `A2_GREEDY_PLANNING_REGRET` | `A2_BEAM4_PLANNING_REGRET` | `A3_FEEDBACK_BENEFIT` | Cost-constrained set realization converts estimated value into budgeted measurements. | Scope is the registered two-action reachable pool; feedback remains a frozen diagnostic and the observed direction is retained. |

## A4 System Diagnostic

`A4_BASELINE_MINUS_MAVIS` is a single `system_diagnostic` recorded under
`II-C_COST_CONSTRAINED_REALIZATION` in the CSV authority. It is main-required
for traceability, but it is not a framework headline, table, or abstract
conclusion. Its CSV assignment is `figure4d` and `none` for the main table,
with manuscript section `5.2.3`; it reports one frozen implementation endpoint
and does not redefine the framework.

Every canonical claim has exactly one primary stage. Figure, table, and
manuscript assignments are machine-readable in the CSV companion.
