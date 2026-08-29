# Spatial Neural Probe Protocol

## Registration

This protocol was fixed before N1 training or result inspection. Its only
hypothesis is that explicit local 2-D interaction in the legal 8x8 partial-state
grid improves Task-Relevant Information Acquisition relative to the current
DeepSets encoder.

## Authorities

- base commit: `9794d53a9549f2e3501fe482e8db8735f468ba20`
- seed: `20260825`
- canonical metrics SHA-256:
  `f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`
- frozen feature-bank SHA-256:
  `280c608d43be164cce8617aea1cc24bf3152d537c94a48d40b61ea15085d6467`
- frozen config SHA-256:
  `e99b47e161663fdaefe28719d16321a010f95b4ad8cf8f506a6e18d1d7f57b9d`

Runtime artifacts, not values quoted in the task specification, are the
authority for baseline comparisons.

## Inputs And Controls

The candidate receives only context `(34,)`, legal-state tokens `(64, 6)`, mask
`(64,)`, and exact-cost features `(3,)`. The 64 tokens have fixed row-major 8x8
ordering. Modes are trained separately:

- `real`: acquired position/history/cost and measured content;
- `positions_only`: identical position/history/cost with RGB zeroed;
- `shuffled`: recipient position/history/context/cost with content from another
  specimen under the existing donor assignment;
- `static`: context with empty acquisition and zero budget.

The independent reconstruction predictor retains its existing meaning. N3 uses
both the complete frozen bank and the pre-registered clean subset
`CLEAN_NONPRIV = {"uniform", "random"}`. The subset will not be changed after
results are viewed.

## Model And Optimization

The registered candidate is `spatial_grid_cnn_v1`:

```text
grid: B x 64 x 6 + mask -> B x 7 x 8 x 8
Conv(7,16,3,pad=1), ReLU
Conv(16,32,3,pad=1), ReLU
Conv(32,32,3,pad=1), ReLU
global average + maximum -> 64
context/cost: Linear(37,32), ReLU, Linear(32,32), ReLU
fusion: Linear(96,64), ReLU, Linear(64,64)
```

Parameter count is 27,552 for the encoder and 27,617 with the P2 head. Output
dimension is 64. The existing 21,442-parameter `DynamicActionScorer`, its loss,
8-D candidate descriptor, and teacher remain unchanged.

Optimization is fixed at learning rate 0.001. P2 uses batch 256, at most 80
epochs, and patience 10. P3 uses batch 64, at most 40 epochs, and patience 5.
The current six-domain nested LODO normalizer, inner validation, selected-epoch
rule, and final source-only refit are preserved.

## Evaluation And Gates

### N1 Representation

Primary contrast: `DeepSets_real_AUEBC - Spatial_real_AUEBC`.

Strong GO requires point estimate above zero, paired 95% CI lower bound above
zero, and favorable direction in at least four of six domains. Promising
requires a positive point estimate and at least four favorable domains with a
CI containing zero. Otherwise N1 is NO_GO.

### N2 Dynamic Value

Primary contrast:
`DeepSets_next_action_regret - Spatial_next_action_regret`.
Use the same strong/promising domain and CI rules as N1. Existing rank/top-k and
utility metrics are reported only where already exposed.

### N3 Content Attribution

Report `positions/history - real` and `shuffled - real`, where positive means
real content is better. Report full-bank and `CLEAN_NONPRIV`, with the latter
primary. Strong GO requires both clean contrasts to have positive point and CI
lower bounds and at least four favorable domains. Promising requires both gaps
to move favorably relative to DeepSets without jointly crossing the strong
gate. Stable unfavorable controls without gap shrinkage are NO_GO.

### N4 End To End

Primary contrast: `static_reference_AUEBC - new_candidate_AUEBC`; positive means
the candidate is better. Strong GO additionally requires new AUEBC below the
frozen static reference, a positive paired CI lower bound, and at least four
favorable domains. Promising requires a better point estimate and four favorable
domains with a CI containing zero. Also report comparison with the frozen learned
implementation, per-checkpoint curves, and six domain directions.

## Statistics And Runtime

Use existing equal-domain aggregation and physical-specimen-first paired
bootstrap functions. Do not create a second AUEBC, alternate bootstrap, or
post-hoc equivalence margin. Reuse the same uniform scout, 8x8 grid, acquisition
levels, checkpoints, exact unique-native-raster cost, candidates, reveal action,
and rollout objective.

## Stage Artifacts

Each of `n0`, `n1_spatial_p2`, `n2_dynamic_p3`,
`n3_content_attribution`, and `n4_closed_loop` must contain at least:

- `REPORT.md`
- `summary.json`
- `domain_metrics.csv`
- `bootstrap.csv`
- `artifact_manifest.json`
- `CHECKSUMS.sha256`

Checkpoint provenance binds schema version, architecture name, base commit,
fold and specimen rosters, hashes, normalizer, feature-bank identity,
hyperparameters, seed, selected epoch, state dictionary hash, configuration
SHA-256, and audit information.

## Stop Conditions And Claims

Any legal-state leakage, target-domain selection, action/cost mismatch, frozen
hash change, nondeterminism, or old DeepSets regression failure is
`INTEGRATION_NO_GO`. N5 is not authorized. The paper is not modified. Only N1--N4
all reaching strong GO may produce a separate paper-integration recommendation;
it still cannot edit the manuscript automatically.
