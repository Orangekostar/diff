# BC C-scan Path-B Supplement Visual Contract

Target format: AEI supplementary figures; vector PDF plus 300 dpi PNG preview.

## Statistical Figure 1: Planner evidence

- Core claim: show current-report success as exact acquisition cost grows.
- Reviewer question: does three-seed BC acquire useful information earlier than the registered P4/P8 rules?
- Evidence layer: main planner-only result.
- Source: `results/bc_cscan_path_b_supplement/trajectories.parquet`.
- Map: LOCATE and CHARACTERIZE panels; BC seed mean with seed range, P4, and P8.
- Constraint: success remains non-monotone; no cumulative maximum or interpolation.

## Statistical Figure 2: Autonomous evidence

- Core claim: jointly show completion and failure-penalized cost.
- Reviewer question: does BC plus the locked STOP satisfy the Path-B completion/cost criterion relative to P8 plus the same STOP?
- Evidence layer: main autonomous result and risk boundary.
- Source: `per_episode_metrics.csv`, `paired_effects.csv`, and `risk_coverage.csv`.
- Map: two tasks by completion and failure cost; S_RULE and S_BC_CAL remain distinct.
- Constraint: completion and cost are displayed together; no safety or reviewed-reference claim.

## Statistical Figure 3: Paired stability

- Core claim: expose domain and finite-seed variation behind the mean.
- Reviewer question: is the BC-minus-rule direction concentrated in one domain or one seed?
- Evidence layer: robustness.
- Source: `paired_effects.csv` and `per_domain_effects.csv`.
- Map: BC versus P4/P8, task-specific estimates, 95% intervals, seed-specific points, and six-domain effects.
- Constraint: seeds are not counted as additional physical specimens.

## Statistical Figure 4: Actor-input ablations

- Core claim: isolate surface-cue and new-ultrasound-content increments for the Actor.
- Reviewer question: do the inputs improve planner-only AUSC at seed 1?
- Evidence layer: mechanism.
- Source: `ablation_effects.csv`.
- Map: two tasks and two independently retrained ablations with 95% intervals.
- Constraint: this is an Actor-input result, not a whole-system no-VLM/no-US claim.

## Representative Figure: Six-domain trajectories

- Core claim: make the predetermined true-break prefixes inspectable.
- Reviewer question: do actual terminal states match cached full-trajectory prefixes across all domains?
- Evidence layer: qualitative execution audit.
- Source: `report_manifest.json` and `trajectories.parquet`.
- Map: one hash-selected TEST specimen per domain; LOCATE and CHARACTERIZE traces for BC_S1 and P8.
- Constraint: examples are selected without outcomes and are not a quantitative substitute.

## Style and traceability

- BC uses Okabe-Ito blue `#0072B2`; P8 uses neutral `#666666`; P4 uses orange `#E69F00`; ablations use `#CC79A7` and `#009E73`.
- Markers and line styles carry method identity in grayscale.
- Output: PDF and PNG; source data paths are named above and in each caption metadata.
- All values remain `PROXY_LEGACY`; `REVIEWED_V1` effects remain null until attributable review exists.
