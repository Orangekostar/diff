# Submission Evidence Matrix

| Claim | Task | Reference | N | Status | Source | Permitted wording | Prohibited overstatement |
|---|---|---|---:|---|---|---|---|
| C1_PLANNER_VS_P8 | BOTH | PROXY_LEGACY | 24 | SUPPORTED | results/bc_cscan_path_b_supplement/paired_effects.csv | Positive planning signal under the shared proxy Reader. | Independently validated diagnostic superiority. |
| C2_SURFACE_INPUT | BOTH | PROXY_LEGACY | 24 | SUPPORTED | results/bc_cscan_path_b_supplement/ablation_effects.csv | Seed-1 actor input increment under proxy scoring. | Whole-system causal or multi-seed proof. |
| C3_US_FEEDBACK | BOTH | PROXY_LEGACY | 24 | SUPPORTED | results/bc_cscan_path_b_supplement/ablation_effects.csv | Seed-1 actor input increment under proxy scoring. | Natural-language reasoning or physical-causal proof. |
| C4_PATH_B_LOCATE | LOCATE | REVIEWED_PENDING | 0 | PENDING_INPUT | reviewed/task_decisions.json | Report the task-specific decision and reference coverage. | Transfer one task's decision to the other or choose a 95% interval. |
| C4_PATH_B_CHARACTERIZE | CHARACTERIZE | REVIEWED_PENDING | 0 | PENDING_INPUT | reviewed/task_decisions.json | Report the task-specific decision and reference coverage. | Transfer one task's decision to the other or choose a 95% interval. |
| C5_BLIND_DELIVERABILITY | BOTH | REVIEWED_PENDING | 0 | PENDING_INPUT | human_review/paired_specimen_differences.csv | Only the observed reviewer distribution and coverage. | Invented consensus or independent physical sample inflation. |
| C6_HUMAN_COMPARISON | BOTH | REVIEWED_PENDING | 0 | PENDING_INPUT | human_planning/matched_method_comparison.csv | Only matched-session descriptive results. | Superhuman performance or unmatched cost conversion. |
| C7_PROCESS_ASSOCIATION | BOTH | PROXY_LEGACY | 24 | SUPPORTED | analysis_summary.json | Descriptive frozen-path associations. | Causal mechanism or guaranteed performance gain. |
| C8_SCOPE_BOUNDARIES | BOTH | NOT_APPLICABLE | 0 | NOT_EVALUATED | FINAL_RESULTS_AND_CLAIM_BOUNDARIES.md | Explicitly state these boundaries. | Generalization, hardware efficiency, or language reasoning claims. |

## Complete Claim Records

### C1_PLANNER_VS_P8

- Task: BOTH
- Exact claim: Under the shared proxy Reader, BC planning has higher AUsC than P8 in both tasks.
- Planned evidence: Paired physical-specimen planner AUsC
- Existing evidence: paired_effects.csv
- New evidence: paired_diagnostic_effects.csv
- Reference version: PROXY_LEGACY
- Physical n: 24
- Seed count: 3
- Human coverage: 0
- Estimate: LOCATE +0.097059; CHARACTERIZE +0.040324
- Interval: 97.5% lower bounds +0.012096 and +0.015609
- Source table: results/bc_cscan_path_b_supplement/paired_effects.csv
- Status: SUPPORTED
- Permitted wording: Positive planning signal under the shared proxy Reader.
- Prohibited overstatement: Independently validated diagnostic superiority.

### C2_SURFACE_INPUT

- Task: BOTH
- Exact claim: Surface cues add planner AUsC for the frozen seed-1 actor under proxy scoring.
- Planned evidence: Seed-1 frozen input ablation
- Existing evidence: ablation_effects.csv
- New evidence: reviewed/actor_input_ablations.csv when references arrive
- Reference version: PROXY_LEGACY
- Physical n: 24
- Seed count: 1
- Human coverage: 0
- Estimate: LOCATE +0.080795; CHARACTERIZE +0.058830
- Interval: 97.5% intervals positive
- Source table: results/bc_cscan_path_b_supplement/ablation_effects.csv
- Status: SUPPORTED
- Permitted wording: Seed-1 actor input increment under proxy scoring.
- Prohibited overstatement: Whole-system causal or multi-seed proof.

### C3_US_FEEDBACK

- Task: BOTH
- Exact claim: Dynamic ultrasound content adds planner AUsC for the frozen seed-1 actor under proxy scoring.
- Planned evidence: Seed-1 frozen input ablation
- Existing evidence: ablation_effects.csv
- New evidence: reviewed/actor_input_ablations.csv when references arrive
- Reference version: PROXY_LEGACY
- Physical n: 24
- Seed count: 1
- Human coverage: 0
- Estimate: LOCATE +0.459773; CHARACTERIZE +0.441817
- Interval: 97.5% intervals positive
- Source table: results/bc_cscan_path_b_supplement/ablation_effects.csv
- Status: SUPPORTED
- Permitted wording: Seed-1 actor input increment under proxy scoring.
- Prohibited overstatement: Natural-language reasoning or physical-causal proof.

### C4_PATH_B_LOCATE

- Task: LOCATE
- Exact claim: Frozen BC satisfies the completion-noninferiority and cost-improvement Path B criterion for LOCATE.
- Planned evidence: Reviewed first-STOP paired 97.5% joint criterion
- Existing evidence: Proxy criterion not supported
- New evidence: reviewed/autonomous_effects.csv
- Reference version: REVIEWED_PENDING
- Physical n: 0
- Seed count: 3
- Human coverage: 0
- Estimate:
- Interval: 97.5% paired domain bootstrap
- Source table: reviewed/task_decisions.json
- Status: PENDING_INPUT
- Permitted wording: Report the task-specific decision and reference coverage.
- Prohibited overstatement: Transfer one task's decision to the other or choose a 95% interval.

### C4_PATH_B_CHARACTERIZE

- Task: CHARACTERIZE
- Exact claim: Frozen BC satisfies the completion-noninferiority and cost-improvement Path B criterion for CHARACTERIZE.
- Planned evidence: Reviewed first-STOP paired 97.5% joint criterion
- Existing evidence: Proxy criterion not supported
- New evidence: reviewed/autonomous_effects.csv
- Reference version: REVIEWED_PENDING
- Physical n: 0
- Seed count: 3
- Human coverage: 0
- Estimate:
- Interval: 97.5% paired domain bootstrap
- Source table: reviewed/task_decisions.json
- Status: PENDING_INPUT
- Permitted wording: Report the task-specific decision and reference coverage.
- Prohibited overstatement: Transfer one task's decision to the other or choose a 95% interval.

### C5_BLIND_DELIVERABILITY

- Task: BOTH
- Exact claim: Blind reviewers judge frozen BC reports more deliverable than P8 reports.
- Planned evidence: User-supplied anonymous report reviews
- Existing evidence: None
- New evidence: human_review/method_summary.csv
- Reference version: REVIEWED_PENDING
- Physical n: 0
- Seed count: 3
- Human coverage: 0
- Estimate:
- Interval:
- Source table: human_review/paired_specimen_differences.csv
- Status: PENDING_INPUT
- Permitted wording: Only the observed reviewer distribution and coverage.
- Prohibited overstatement: Invented consensus or independent physical sample inflation.

### C6_HUMAN_COMPARISON

- Task: BOTH
- Exact claim: BC outperforms human planning in comparable replay sessions.
- Planned evidence: Matched user-supplied human sessions
- Existing evidence: None
- New evidence: human_planning/matched_method_comparison.csv
- Reference version: REVIEWED_PENDING
- Physical n: 0
- Seed count: 3
- Human coverage: 0
- Estimate:
- Interval:
- Source table: human_planning/matched_method_comparison.csv
- Status: PENDING_INPUT
- Permitted wording: Only matched-session descriptive results.
- Prohibited overstatement: Superhuman performance or unmatched cost conversion.

### C7_PROCESS_ASSOCIATION

- Task: BOTH
- Exact claim: Frozen process diagnostics quantify report instability, STOP delay, cost allocation, and surface-proxy conflict.
- Planned evidence: W2-W5 frozen cohort diagnostics
- Existing evidence: Frozen trajectories
- New evidence: W2-W5 result tables
- Reference version: PROXY_LEGACY
- Physical n: 24
- Seed count: 3
- Human coverage: 0
- Estimate: See analysis_summary.json
- Interval: 95% exploratory intervals for selected paired effects
- Source table: analysis_summary.json
- Status: SUPPORTED
- Permitted wording: Descriptive frozen-path associations.
- Prohibited overstatement: Causal mechanism or guaranteed performance gain.

### C8_SCOPE_BOUNDARIES

- Task: BOTH
- Exact claim: Current evidence does not establish material generalization, hardware benefit, or natural-language reasoning.
- Planned evidence: External material/hardware/causal studies
- Existing evidence: None
- New evidence: None in this frozen analysis
- Reference version: NOT_APPLICABLE
- Physical n: 0
- Seed count: 0
- Human coverage: 0
- Estimate:
- Interval:
- Source table: FINAL_RESULTS_AND_CLAIM_BOUNDARIES.md
- Status: NOT_EVALUATED
- Permitted wording: Explicitly state these boundaries.
- Prohibited overstatement: Generalization, hardware efficiency, or language reasoning claims.
