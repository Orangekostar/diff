# Reviewed Results and Claim Boundaries

Reference version: `REVIEWED_02339eda86d486c3`. The 24 references are pooled across two source experts and are not grouped by reviewer.

## Full-input Reader

| Task | Success | Mean IoU | Mean recall | Mean area error |
|---|---:|---:|---:|---:|
| CHARACTERIZE | 0/24 | 0.2173 | 0.4734 | 2.7002 |
| LOCATE | 6/24 | 0.2610 | 0.2500 | 0.0000 |

LOCATE recall and area-error fields retain their task-specific bbox/support semantics; CHARACTERIZE uses mask IoU, recall, and the registered area tolerance.

## Reviewed planner AUSC

| Task | P8 | BC S1 | BC S2 | BC S3 | BC 3-seed | BC-P8 (97.5% CI) |
|---|---:|---:|---:|---:|---:|---:|
| LOCATE | 0.1912 | 0.2292 | 0.2265 | 0.1720 | 0.2092 | 0.0180 [-0.0083, 0.0457] |
| CHARACTERIZE | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |

BC 3-seed is a statistical seed average, not an inference-time ensemble.

## Original calibrated STOP and Path B

| Task | Method | Completion | False stop | Exhausted | Failure cost | Prefix cost | Path B |
|---|---|---:|---:|---:|---:|---:|---|
| LOCATE | R_BALANCED_P8 | 0.2500 | 0.7500 | 0.0000 | 0.9066 | 0.6726 | NOT_SUPPORTED |
| LOCATE | BC_3SEED | 0.2361 | 0.7639 | 0.0000 | 0.9125 | 0.6712 | NOT_SUPPORTED |
| CHARACTERIZE | R_BALANCED_P8 | 0.0000 | 0.8750 | 0.1250 | 1.0000 | 0.8135 | NOT_SUPPORTED |
| CHARACTERIZE | BC_3SEED | 0.0000 | 0.9722 | 0.0278 | 1.0000 | 0.7797 | NOT_SUPPORTED |

Path B is reported separately by task and uses the frozen 97.5% joint completion/cost rule. S_RULE remains an auxiliary diagnostic in the machine-readable tables.

## Existing seed-1 Actor input ablations

| Task | Input | Full | Ablated | Delta (97.5% CI) |
|---|---|---:|---:|---:|
| LOCATE | ACTOR_SURFACE_CUES | 0.2292 | 0.2185 | 0.0107 [-0.0071, 0.0283] |
| CHARACTERIZE | ACTOR_SURFACE_CUES | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |
| LOCATE | ACTOR_US_FEEDBACK_CONTENT | 0.2292 | 0.0279 | 0.2013 [0.0732, 0.3372] |
| CHARACTERIZE | ACTOR_US_FEEDBACK_CONTENT | 0.0000 | 0.0000 | 0.0000 [0.0000, 0.0000] |

These ablations isolate the two registered seed-1 Actor inputs; they are not whole-system no-VLM/no-ultrasound experiments.

## Claim status

| Claim | Task | Status | Evidence |
|---|---|---|---|
| R1_LOCATE | LOCATE | NOT_SUPPORTED | `reviewed/planner_effects.csv` |
| R1_CHARACTERIZE | CHARACTERIZE | NOT_SUPPORTED | `reviewed/planner_effects.csv` |
| R2_LOCATE | LOCATE | NOT_SUPPORTED | `reviewed/actor_input_ablations.csv` |
| R2_CHARACTERIZE | CHARACTERIZE | NOT_SUPPORTED | `reviewed/actor_input_ablations.csv` |
| R3_LOCATE | LOCATE | SUPPORTED | `reviewed/actor_input_ablations.csv` |
| R3_CHARACTERIZE | CHARACTERIZE | NOT_SUPPORTED | `reviewed/actor_input_ablations.csv` |
| R4_CHARACTERIZE | CHARACTERIZE | NOT_SUPPORTED | `reviewed/task_decisions.json` |
| R4_LOCATE | LOCATE | NOT_SUPPORTED | `reviewed/task_decisions.json` |
| R5_CHARACTERIZE | CHARACTERIZE | DESCRIPTIVE_RESULT | `tables/full_input_reader_summary.csv` |
| R5_LOCATE | LOCATE | DESCRIPTIVE_RESULT | `tables/full_input_reader_summary.csv` |
| R6_PROCESS | BOTH | DESCRIPTIVE_RESULT | `tables/reviewed_process_summary.csv` |
| R7_BLIND_REVIEW | BOTH | PENDING_INPUT | `human_review/input_coverage.csv` |
| R8_HUMAN_PLANNING | BOTH | PENDING_INPUT | `human_planning/comparability_manifest.csv` |

Historical C1-C3 remain `PROXY_LEGACY`; this document and `reviewed_claims.json` are the reviewed claim authority. Blind report review and human planning comparison remain pending real input.
