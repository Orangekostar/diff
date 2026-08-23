# MVA A5 Preregistered Oracle-Imitation Protocol

Date: 2026-08-23
Status: frozen after `MVA_A5_AUTHORIZED` and before A5 result generation

## 1. Authority

- Cohort, targets, grids, interpolation, encoder, metadata, P-A/P-B estimator,
  checkpoints, random seeds, FULL MAE, and simulator are inherited unchanged
  from the validated A0-A4 packages.
- A4 formal status is `MVA_A4_GLOBAL_NO_GO`; A5 authorization status is
  `MVA_A5_AUTHORIZED`.
- A5 may train one supervised imitation policy per outer domain. A6/A7, RL,
  Transformer, GNN, and post-hoc architecture search remain forbidden.

## 2. Outer-safe supervision

For outer domain `d` and query source domain `q`, teacher model fitting excludes
both `d` and `q`. Nested PCA selection is confined to the remaining four
domains. Every serialized training audit must expose query domains, fit
domains, specimen IDs, selected dimension, and predictor digest.

A2 oracle values are reference evidence only and cannot train A5. A5 teacher
states are regenerated from the bound simulator and source-only predictors.
Eight source specimens are advanced in encoder lockstep; incomplete final
batches retain source-roster order and are not padded.

## 3. State, candidates, and action space

The policy state has 579 values: current reconstructed-image embedding (512),
normalized 64-cell levels, current P-A prediction, used budget, and remaining
budget to 25%. Candidate features have the eight fields frozen in the design.
All features are derived from current observed state and geometry.

The action set is the exact registered feasible one-level refinements at the
current checkpoint, including both `0 -> 1` and `1 -> 2`. Tie breaks use lower
cell index and then lower target level.

## 4. Model and training

The fixed shared scorer is `579-64-32`, `8-32-16`, then `48-32-1`, with ReLU.
Float64 CPU training uses fixed seeds, one thread, Adam (lr 1e-3, weight decay
1e-4), gradient cap 5, 50 complete epochs, 128 states per compute batch, and
equal-domain/equal-specimen/equal-state weights. Primary loss is pairwise
logistic ranking against the teacher-selected action. No early stopping,
validation sweep, or target-informed checkpoint choice is allowed.
The deterministic training seed for an outer domain is `20260823` plus its
zero-based position in the registered six-domain order.

## 5. Target execution

At each target step the policy receives only current-state tensors, scores the
feasible actions, chooses one, and then the simulator reveals the registered
measurements. The policy never receives target CAI, full-image features,
unmeasured true pixels, teacher values, or oracle actions.

P-A supplies the current deployable CAI-prediction feature. P-B evaluates all
methods at checkpoints and must match the bound A2 predictor digest exactly.

## 6. Comparators

Required methods are uniform, random median with the 5%-95% seed band,
center-first, observed-gradient appearance, observed-uncertainty
reconstruction, global mechanical mask, imitation policy, and mechanical
oracle. The latter is a retrospective upper bound, not a deployable method.

The three deployable heuristic scores are frozen before A5 execution. With the
eight candidate features indexed as registered in Section 3, center-first uses
the negative squared distance from `(row, column)` to `(0.5, 0.5)`;
observed-gradient uses `added_fraction * local_gradient`; and
observed-uncertainty uses
`added_fraction * (local_variance + nearest_measured_distance)`. Higher scores
are selected, with lower cell index and then lower target level as tie breaks.

## 7. Statistics and gates

Metrics are equal-domain P-B MAE, AUEBC on 6.25%-25%, B2.5/B5/B7.5, per-domain
effects, and oracle gap closure. All paired effects reuse one synchronized
100000 x 6 PCG64 bootstrap matrix with seed 20260823.

`MVA_A5_POLICY_GO` requires statistically positive improvement over both global
mechanical and uniform, at least 4/6 improved domains for each, and at least 20%
point oracle-gap closure. All other outcomes issue `MVA_A5_POLICY_NO_GO`.
A6 is authorized only by A5 GO.

## 8. Artifacts and replay

The formal directory is `results/mva/a5_imitation_policy`. It contains teacher
fit audits and digests, policy training records and models, target decisions
and trajectories, state metrics, curves, domain/budget/specimen/bootstrap
tables, summary, report, figures with source data, config, manifest, and
checksums. Replay must validate and reproduce an identical byte tree.

## 9. Stop rule

If A5 fails either primary baseline comparison or closes less than 20% of the
oracle gap, stop. Do not add RL, Transformer, GNN, backbone search, or
target-informed tuning. A6 and A7 remain locked.
