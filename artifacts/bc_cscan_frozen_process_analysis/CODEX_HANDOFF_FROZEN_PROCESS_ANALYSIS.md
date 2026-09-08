# Codex Handoff: Frozen Process Analysis

## Repository

- Base: `fe58de39298c412d0829580cfda40b7c7c4c53e9`
- Branch: `research/bc-cscan-frozen-process-analysis`
- Final commit: resolve with `git rev-parse HEAD`; the document does not self-reference its commit.
- Frozen source root: `/home/ww/paper3/cmc_damage_inference`
- Config: `paper_v3/configs/bc_cscan_frozen_process_analysis.yaml`; SHA-256 `6fc7d88e72bec9a37f3527a0a137b3407fb11803d5f90547183f56c82db4f5d7`
- Frozen source identities: `results/bc_cscan_frozen_process_analysis/input_manifest.json`

## External Input Identities

- reviewed_references: status `PENDING_USER_INPUT`; path `NOT_PROVIDED`; SHA-256 `NOT_AVAILABLE`.
- blind_reviews: status `PENDING_USER_INPUT`; path `NOT_PROVIDED`; SHA-256 `NOT_AVAILABLE`.
- human_sessions: status `PENDING_USER_INPUT`; path `NOT_PROVIDED`; SHA-256 `NOT_AVAILABLE`.

Reference coverage: 0 matched TEST references, 0 reviewed references without a certain region, 0 missing TEST references, 24 pending TEST references, and 0 unmatched supplied references. Missing external inputs are not synthesized.

## W0-W8

- W0-W5: executed from immutable source tables and stored-action Reader recovery.
- W6: implemented; reference status `PENDING_USER_INPUT`.
- W7A: `PENDING_INPUT`.
- W7B: `PENDING_INPUT`.
- W8: claim matrix generated with per-claim status and wording boundaries.

## Main Frozen Recovery

- Physical cohort: 24 specimens across 6 domains.
- Scalar analysis: 336 episodes and 64,848 recorded states.
- Main stored-action recovery: 192 episodes, 37,056 report states, and 36,864 action transitions.
- Main recovery cap: 40,000; cache reuse: `SURFACE_PERCEPT_CACHE_STORED_ACTIONS_AND_CELL_INTERPOLATION`.
- Report digest mismatches: 0; proxy score mismatches: 0.

## Reviewed Recovery

- Status: `PENDING_USER_INPUT`.
- Episodes: 0; report states: 0; source action transitions represented: 0.
- Transitions recovered in this run: 0; transitions reused from identity-bound cache: 0.
- Cache-hit specimens: 0; cache-hit episodes: 0; recovery cap: 60,000.

## Planning-to-Stop Findings

- LOCATE planner AUsC, BC minus P8: +0.097059 (95% CI [+0.021459, +0.175436]).
- LOCATE autonomous AUsC, BC minus P8: +0.000979 (95% CI [-0.032420, +0.036724]).
- LOCATE S_BC_CAL completion 0.958333; false-stop 0.041667; exhaustion 0.000000 across 96 main-method episodes.
- CHARACTERIZE planner AUsC, BC minus P8: +0.040324 (95% CI [+0.018870, +0.061449]).
- CHARACTERIZE autonomous AUsC, BC minus P8: +0.020795 (95% CI [+0.007150, +0.034829]).
- CHARACTERIZE S_BC_CAL completion 0.895833; false-stop 0.052083; exhaustion 0.052083 across 96 main-method episodes.

These are exploratory frozen-cohort diagnostics under `PROXY_LEGACY`. Historical 97.5% Path B results remain the authoritative proxy decision; reviewed Path B is reported separately below.

## Report Stability and STOP Delay

- LOCATE: post-sustained STOP delay is defined in 92/96 main-method episodes, with conditional mean native-raster cost 0.276662; 63 episodes regress after first success.
- CHARACTERIZE: post-sustained STOP delay is defined in 86/96 main-method episodes, with conditional mean native-raster cost 0.255965; 57 episodes regress after first success.

Delay means are conditional on a sustained-success point existing before the frozen stop. They are descriptive and do not replace completion or failure-penalized cost.

## Action-Cost Allocation

- LOCATE R_BALANCED_P8: survey 0.015480, medium 0.193096, dense 0.463975, total 0.672551.
- LOCATE BC_3SEED: survey 0.014968, medium 0.191803, dense 0.464411, total 0.671181.
- CHARACTERIZE R_BALANCED_P8: survey 0.015489, medium 0.221161, dense 0.576872, total 0.813523.
- CHARACTERIZE BC_3SEED: survey 0.015241, medium 0.215265, dense 0.549204, total 0.779710.

Values are autonomous-prefix native-raster measurement fractions under `S_BC_CAL`; they are not wall-clock or hardware travel costs.

## Process Associations and Boundaries

- Observed fact: Frozen trajectories permit exact signed planning-to-stop and action-cost accounting.
- Supported association: Report stability, STOP waiting, and spatial proxy groups are descriptive frozen-path associations.
- Boundary: No causal mechanism, hardware-time benefit, independent correctness, or human superiority is established.
- Surface/proxy groups: `{"CUE_DISJOINT": 9, "CUE_OVERLAP": 39}`.

## Reviewed Path B and Claim Status

- Reviewed LOCATE Path B: `PENDING_INPUT`.
- Reviewed CHARACTERIZE Path B: `PENDING_INPUT`.
- C1_PLANNER_VS_P8 (BOTH): `SUPPORTED`; reference `PROXY_LEGACY`; N=24; source `results/bc_cscan_path_b_supplement/paired_effects.csv`.
- C2_SURFACE_INPUT (BOTH): `SUPPORTED`; reference `PROXY_LEGACY`; N=24; source `results/bc_cscan_path_b_supplement/ablation_effects.csv`.
- C3_US_FEEDBACK (BOTH): `SUPPORTED`; reference `PROXY_LEGACY`; N=24; source `results/bc_cscan_path_b_supplement/ablation_effects.csv`.
- C4_PATH_B_LOCATE (LOCATE): `PENDING_INPUT`; reference `REVIEWED_PENDING`; N=0; source `reviewed/task_decisions.json`.
- C4_PATH_B_CHARACTERIZE (CHARACTERIZE): `PENDING_INPUT`; reference `REVIEWED_PENDING`; N=0; source `reviewed/task_decisions.json`.
- C5_BLIND_DELIVERABILITY (BOTH): `PENDING_INPUT`; reference `REVIEWED_PENDING`; N=0; source `human_review/paired_specimen_differences.csv`.
- C6_HUMAN_COMPARISON (BOTH): `PENDING_INPUT`; reference `REVIEWED_PENDING`; N=0; source `human_planning/matched_method_comparison.csv`.
- C7_PROCESS_ASSOCIATION (BOTH): `SUPPORTED`; reference `PROXY_LEGACY`; N=24; source `analysis_summary.json`.
- C8_SCOPE_BOUNDARIES (BOTH): `NOT_EVALUATED`; reference `NOT_APPLICABLE`; N=0; source `FINAL_RESULTS_AND_CLAIM_BOUNDARIES.md`.

## Human Evidence and Comparability

- Blind-review status: `PENDING_INPUT`; raw decisions: 0; paired specimen/reviewer contrasts: 0.
- Human-session status: `PENDING_INPUT`; supplied sessions: 0; matched model comparisons: 0.
- Measurement cost is comparable only for `NATIVE_RASTER_FRACTION`; route cost is comparable only for `FROZEN_GRID_ROUTE_COST`.

## Resource Integrity

- Training updates: 0
- New VLM calls: 0
- Actor forward calls: 0
- STOP forward calls: 0
- Reviewed recovery world steps: `0`
- Main report cache hits: `0`

## Scientific Status

Historical proxy status remains `BC_PLANNING_SUPPORTED_STOP_NOT_SUPPORTED_PROXY_ONLY`. Computation completion and scientific support are reported separately in `final_evidence_manifest.json` and `SUBMISSION_EVIDENCE_MATRIX.md`.

## Output Inventory

Result root: `results/bc_cscan_frozen_process_analysis/`

- `input_manifest.json`
- `first_stop_decomposition.csv`
- `report_stability_and_delay.csv`
- `stop_wait_components.csv`
- `planning_to_stop_accounting.csv`
- `action_events.parquet`
- `action_cost_allocation.csv`
- `prefix_divergence.csv`
- `surface_proxy_agreement.csv`
- `spatial_action_statistics.parquet`
- `spatial_episode_summary.csv`
- `recovery_manifest.json`
- `paired_diagnostic_effects.csv`
- `case_manifest.csv`
- `analysis_summary.json`
- `reviewed/`
- `human_review/`
- `human_planning/`
- `final_evidence_manifest.json`
- `figures/`
- `reproduce.md`
- `CHECKSUMS.sha256`

Artifact root: `artifacts/bc_cscan_frozen_process_analysis/`

- `INPUT_AND_CODE_BINDINGS.md`
- `RESULTS_AND_LIMITATIONS.md`
- `INPUTS_FROM_USER.md`
- `SUBMISSION_EVIDENCE_MATRIX.md`
- `FINAL_RESULTS_AND_CLAIM_BOUNDARIES.md`
- `CODEX_HANDOFF_FROZEN_PROCESS_ANALYSIS.md`

## Later Finalize Command

```bash
PYTHONPATH=src python scripts/analyze_bc_cscan_frozen_process.py finalize \
  --config paper_v3/configs/bc_cscan_frozen_process_analysis.yaml \
  --source-root /home/ww/paper3/cmc_damage_inference \
  --references <reference-directory> \
  --blind-reviews <blind-review.csv> \
  --human-sessions <human-sessions.csv-or-json>
```

The same command imports any subset of the three external tracks; omitted tracks remain `PENDING_INPUT`. It performs no training and no Actor, STOP, or VLM inference.
