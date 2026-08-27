# Paper Evidence Chronology

The frozen outer endpoint predates all post-freeze diagnostics. The later analyses reuse hash-bound frozen states and outcomes and were not used to re-select or modify that endpoint.

| Chronology class | Claims | Endpoint-selection role |
|---|---:|---|
| PRE_P7_FROZEN_EVIDENCE | 13 | Available before endpoint freeze |
| FROZEN_OUTER_ENDPOINT | 1 | Fixed evaluation endpoint |
| POST_P7_DIAGNOSTIC | 25 | Not used to modify P7 |

Post-freeze diagnostics are diagnostic evidence, not preregistered confirmatory evidence.

| Claim | Layer | Source stage | Class | Source |
|---|---|---|---|---|
| U1_MATCHED_FIELD | Useful | P1_FULL_FIELD | PRE_P7_FROZEN_EVIDENCE | `results/p1_full_field_oracle/metrics.json` |
| U1_SURFACE_FIELD | Useful | P1_FULL_FIELD | PRE_P7_FROZEN_EVIDENCE | `results/p1_full_field_oracle/metrics.json` |
| U1_INDEPENDENT_FIELD_SENSITIVITY | Useful | P1_FULL_FIELD | PRE_P7_FROZEN_EVIDENCE | `results/p1_full_field_oracle/domain_metrics.csv` |
| U2_SPARSE_RETENTION | Useful | P5_SPARSE_SCAN | PRE_P7_FROZEN_EVIDENCE | `results/p5_sparse_scan/retention.csv` |
| U2_SPARSE_GAIN | Useful | P5_SPARSE_SCAN | PRE_P7_FROZEN_EVIDENCE | `results/p5_sparse_scan/bootstrap.csv` |
| U2_SPARSE_FULL_GAP | Useful | P5_SPARSE_SCAN | PRE_P7_FROZEN_EVIDENCE | `results/p5_sparse_scan/bootstrap.csv` |
| U3_UNIFORM_ORACLE | Useful | MVD_M0 | PRE_P7_FROZEN_EVIDENCE | `results/mvd/m0_one_shot_oracle/summary.json` |
| U3_RECONSTRUCTION_ORACLE | Useful | MVD_M0 | PRE_P7_FROZEN_EVIDENCE | `results/mvd/m0_one_shot_oracle/summary.json` |
| U3_HEADROOM_RETENTION | Useful | MVD_M0 | PRE_P7_FROZEN_EVIDENCE | `results/mvd/m0_one_shot_oracle/summary.json` |
| U4_ORACLE_CAI_SPECIFICITY | Useful | POST_FREEZE_TASK_SPECIFICITY | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p14_task_specificity/summary.json` |
| U4_ORACLE_IMAGE_SPECIFICITY | Useful | POST_FREEZE_TASK_SPECIFICITY | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p14_task_specificity/summary.json` |
| U4_LEARNED_SPECIFICITY_BOUNDARY | Useful | POST_FREEZE_TASK_SPECIFICITY | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p14_task_specificity/summary.json` |
| U5_RIDGE_HUBER_SPEARMAN | Useful | POST_FREEZE_VALUE_STABILITY | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p15_value_stability/summary.json` |
| U5_RIDGE_HUBER_BEST_ACTION | Useful | POST_FREEZE_VALUE_STABILITY | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p15_value_stability/summary.json` |
| U5_RIDGE_HUBER_TOPK | Useful | POST_FREEZE_VALUE_STABILITY | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p15_value_stability/summary.json` |
| U5_RIDGE_MLP_SPEARMAN | Useful | POST_FREEZE_VALUE_STABILITY | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p15_value_stability/summary.json` |
| U5_RIDGE_MLP_BEST_ACTION | Useful | POST_FREEZE_VALUE_STABILITY | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p15_value_stability/summary.json` |
| U5_RIDGE_MLP_TOPK | Useful | POST_FREEZE_VALUE_STABILITY | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p15_value_stability/summary.json` |
| O1_STATIC_SPEARMAN | Observable | MVD_M1 | PRE_P7_FROZEN_EVIDENCE | `results/mvd/m1_observability/bootstrap.csv` |
| O1_STATIC_SET_REGRET | Observable | MVD_M1 | PRE_P7_FROZEN_EVIDENCE | `results/mvd/m1_observability/model_metrics.csv` |
| O1_GLOBAL_SET_REGRET | Observable | MVD_M1 | PRE_P7_FROZEN_EVIDENCE | `results/mvd/m1_observability/model_metrics.csv` |
| O1_RANDOM_SET_REGRET | Observable | MVD_M1 | PRE_P7_FROZEN_EVIDENCE | `results/mvd/m1_observability/model_metrics.csv` |
| O2_TEACHER_TURNOVER | Observable | POST_FREEZE_VALUE_EVOLUTION | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p9_value_evolution/summary.json` |
| O2_TEACHER_RANK | Observable | POST_FREEZE_VALUE_EVOLUTION | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p9_value_evolution/summary.json` |
| O2_TEACHER_TOPK | Observable | POST_FREEZE_VALUE_EVOLUTION | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p9_value_evolution/summary.json` |
| O2_TEACHER_OPPORTUNITY | Observable | POST_FREEZE_VALUE_EVOLUTION | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p9_value_evolution/summary.json` |
| O3_REAL_CHANGE | Observable | POST_FREEZE_STATE_CONTROLS | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p10_mris_causal/summary.json` |
| O3_FULL_FIELD_RECOVERY | Observable | POST_FREEZE_STATE_CONTROLS | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p10_mris_causal/summary.json` |
| O3_REAL_MINUS_POSITIONS | Observable | POST_FREEZE_STATE_CONTROLS | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p10_mris_causal/contrasts.csv` |
| O3_REAL_MINUS_RECONSTRUCTION | Observable | POST_FREEZE_STATE_CONTROLS | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p10_mris_causal/contrasts.csv` |
| O4_DYNAMIC_MINUS_STATIC | Observable | POST_FREEZE_DYNAMIC_VALUATION | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p11_dynamic_valuation/summary.json` |
| O4_DYNAMIC_MINUS_SHUFFLED | Observable | POST_FREEZE_DYNAMIC_VALUATION | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p11_dynamic_valuation/summary.json` |
| A1_VALUATION_SUBSTITUTION | Actionable | POST_FREEZE_COMPONENT_ATTRIBUTION | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p12_rvp_attribution/summary.json` |
| A1_LEARNED_PLANNING_SUBSTITUTION | Actionable | POST_FREEZE_COMPONENT_ATTRIBUTION | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p12_rvp_attribution/summary.json` |
| A1_TRUE_VALUE_PLANNING_SUBSTITUTION | Actionable | POST_FREEZE_COMPONENT_ATTRIBUTION | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p12_rvp_attribution/summary.json` |
| A2_GREEDY_PLANNING_REGRET | Actionable | POST_FREEZE_SET_PLANNING | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p13_set_planning/summary.json` |
| A2_BEAM4_PLANNING_REGRET | Actionable | POST_FREEZE_SET_PLANNING | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p13_set_planning/summary.json` |
| A3_FEEDBACK_BENEFIT | Actionable | POST_FREEZE_FEEDBACK | POST_P7_DIAGNOSTIC | `results/mavis_science_closure/p16_feedback_mechanism/summary.json` |
| A4_BASELINE_MINUS_MAVIS | Actionable | FROZEN_OUTER_EVALUATION | FROZEN_OUTER_ENDPOINT | `results/mavis/p7_final_frozen_eval/claim_evidence.csv` |
