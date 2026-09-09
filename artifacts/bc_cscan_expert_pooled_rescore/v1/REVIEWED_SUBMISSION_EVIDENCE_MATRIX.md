# Reviewed Submission Evidence Matrix

| Claim | Task | Status | Evidence | Boundary |
|---|---|---|---|---|
| R1_LOCATE | LOCATE | NOT_SUPPORTED | `reviewed/planner_effects.csv` | Positive support requires complete coverage and 97.5% CI lower > 0. |
| R1_CHARACTERIZE | CHARACTERIZE | NOT_SUPPORTED | `reviewed/planner_effects.csv` | Positive support requires complete coverage and 97.5% CI lower > 0. |
| R2_LOCATE | LOCATE | NOT_SUPPORTED | `reviewed/actor_input_ablations.csv` | Positive support requires complete coverage and 97.5% CI lower > 0. |
| R2_CHARACTERIZE | CHARACTERIZE | NOT_SUPPORTED | `reviewed/actor_input_ablations.csv` | Positive support requires complete coverage and 97.5% CI lower > 0. |
| R3_LOCATE | LOCATE | SUPPORTED | `reviewed/actor_input_ablations.csv` | Positive support requires complete coverage and 97.5% CI lower > 0. |
| R3_CHARACTERIZE | CHARACTERIZE | NOT_SUPPORTED | `reviewed/actor_input_ablations.csv` | Positive support requires complete coverage and 97.5% CI lower > 0. |
| R4_CHARACTERIZE | CHARACTERIZE | NOT_SUPPORTED | `reviewed/task_decisions.json` | Uses frozen Path B joint rule; tasks are not pooled. |
| R4_LOCATE | LOCATE | NOT_SUPPORTED | `reviewed/task_decisions.json` | Uses frozen Path B joint rule; tasks are not pooled. |
| R5_CHARACTERIZE | CHARACTERIZE | DESCRIPTIVE_RESULT | `tables/full_input_reader_summary.csv` | Descriptive Reader agreement; no new safety threshold. |
| R5_LOCATE | LOCATE | DESCRIPTIVE_RESULT | `tables/full_input_reader_summary.csv` | Descriptive Reader agreement; no new safety threshold. |
| R6_PROCESS | BOTH | DESCRIPTIVE_RESULT | `tables/reviewed_process_summary.csv` | Post hoc process description; no new STOP policy. |
| R7_BLIND_REVIEW | BOTH | PENDING_INPUT | `human_review/input_coverage.csv` | Pending real human input; no proxy substitution. |
| R8_HUMAN_PLANNING | BOTH | PENDING_INPUT | `human_planning/comparability_manifest.csv` | Pending real human input; no proxy substitution. |
