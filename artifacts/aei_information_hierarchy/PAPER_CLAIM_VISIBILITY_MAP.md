# Paper Claim Visibility Map

This authority distributes all **39 canonical claims** across the main paper,
supplement, figures, and tables without changing numerical values, directions,
sources, or chronological roles. The machine-readable authority is
`PAPER_CLAIM_VISIBILITY_MAP.csv`.

## Visibility counts

| Visibility | Count | Role |
|---|---:|---|
| `MAIN_HEADLINE` | 12 | Primary evidence statements that organize the six compressed stages |
| `MAIN_SUPPORT` | 15 | Main-text context or interpretation supporting a headline stage |
| `MAIN_SYSTEM_DIAGNOSTIC` | 1 | One concise A4 endpoint diagnostic in Section 5.2.3 |
| `SUPPLEMENT_ONLY` | 11 | Full controls, sensitivities, and implementation-boundary details |

## Six-stage paper contract

| Stage | Main scientific role | Main headline claims | Main support claims | Supplement-only claims |
|---|---|---|---|---|
| I-A Spatial information and sparse recoverability | Spatial morphology matters and most gain survives sparse observation | `U1_MATCHED_FIELD`; `U2_SPARSE_RETENTION` | `U1_SURFACE_FIELD`; `U2_SPARSE_GAIN`; `U2_SPARSE_FULL_GAP` | `U1_INDEPENDENT_FIELD_SENSITIVITY` |
| I-B Task-conditioned spatial measurement value | Acquisition opportunity is heterogeneous and objective-dependent | `U3_UNIFORM_ORACLE`; `U3_RECONSTRUCTION_ORACLE`; `U4_ORACLE_CAI_SPECIFICITY`; `U4_ORACLE_IMAGE_SPECIFICITY` | `U3_HEADROOM_RETENTION` | `U4_LEARNED_SPECIFICITY_BOUNDARY` |
| I-C State- and predictor-conditioned value | Value evolves with evidence and depends on the downstream predictor | `O2_TEACHER_TURNOVER`; `U5_RIDGE_HUBER_SPEARMAN` | `O2_TEACHER_RANK`; `O2_TEACHER_TOPK`; `O2_TEACHER_OPPORTUNITY`; `U5_RIDGE_MLP_SPEARMAN` | `U5_RIDGE_HUBER_BEST_ACTION`; `U5_RIDGE_HUBER_TOPK`; `U5_RIDGE_MLP_BEST_ACTION`; `U5_RIDGE_MLP_TOPK` |
| II-A State-conditioned valuation | Dynamic valuation improves next-action estimation over the static reference | `O4_DYNAMIC_MINUS_STATIC` | `O1_STATIC_SPEARMAN` | `O1_STATIC_SET_REGRET`; `O1_GLOBAL_SET_REGRET`; `O1_RANDOM_SET_REGRET` |
| II-B Information-source / component decomposition | Controls identify contributing state signals and substitutions localize headroom | `A1_VALUATION_SUBSTITUTION`; `A1_TRUE_VALUE_PLANNING_SUBSTITUTION` | `O3_REAL_CHANGE`; `O3_REAL_MINUS_POSITIONS`; `O3_REAL_MINUS_RECONSTRUCTION`; `O4_DYNAMIC_MINUS_SHUFFLED`; `A1_LEARNED_PLANNING_SUBSTITUTION` | `O3_FULL_FIELD_RECOVERY` |
| II-C Cost-constrained set realization | Estimated value is converted into budgeted measurement sets | `A2_GREEDY_PLANNING_REGRET` | `A2_BEAM4_PLANNING_REGRET` | `A3_FEEDBACK_BENEFIT` |

`A4_BASELINE_MINUS_MAVIS` is the only `MAIN_SYSTEM_DIAGNOSTIC`. It is not a
headline or summary-table result. The main manuscript gives it one concise
paragraph in Section 5.2.3; the supplement retains its exact AUEBC values,
confidence interval, and two-of-six domain direction.

## Integrity rules

- Every claim retains the layer and source from
  `PAPER_CANONICAL_METRICS.csv` and the chronology class from
  `PAPER_EVIDENCE_CHRONOLOGY.csv`.
- `SUPPLEMENT_ONLY` does not mean hidden or deleted: each claim remains in the
  supplement and machine-readable evidence.
- Figure 4 contains only A1/A2 evidence. A3 and A4 are not plotted in main.
- The main summary table excludes A3/A4 and all audit-only hash/source columns.
- Main plus supplement plus this authority covers all 39 canonical claims.
