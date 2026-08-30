# AEI Signal-Saliency Reframe Design

## Objective and scientific authority

Replace the main paper's reconstruction/field-recovery comparator with the
already frozen, preregistered `appearance_oracle`, presented as
task-agnostic C-scan appearance saliency. The exact base is
`35248f17f603e94962dc19e939162e9ef4eee5f2`; all result trees and scientific
runtime inputs remain immutable. This is a controlled paper-authority and
narrative migration, not a new experiment or a relabeling of reconstruction
evidence.

The saliency score for one legal candidate is the mean element-wise absolute
RGB deviation of newly revealed native-raster values from the
specimen-specific full-image border median, divided by 255. It reads no CAI
outcome. Because it uses full counterfactual candidate values, it is a
retrospective, non-deployable oracle and is not damage severity, a causal map,
an information-theoretic quantity, or scanner time.

## Evidence decision

Route B is supported by the frozen MVA A2 package:

- appearance-minus-mechanical CAI AUEBC is
  `0.007080059382261465`, with synchronized 100000-resample held-out-domain
  bootstrap interval `[0.004799356600193281, 0.00974029297002471]`;
- all six domain effects are positive;
- across 276 registered initial maps, mean mechanical-versus-appearance
  Spearman is `0.02221200907923673` and mean top-decile overlap is
  `0.20031055900621111`;
- the `c8-2` initial comparison contains exactly 64 unique
  `mechanical_oracle` rows and 64 unique `appearance_oracle` rows on the same
  domain and 8-by-8 grid.

No science file needs to be regenerated or changed.

## Canonical authority migration

Preserve all 39 existing canonical rows so legacy technical evidence remains
traceable. Add three new IDs without reusing old semantics:

- `U3_CAI_VS_APPEARANCE_SALIENCY_AUEBC` from the frozen A2 bootstrap;
- `U4_CAI_SALIENCY_MAP_SPEARMAN` from the frozen A2 map-similarity table;
- `U4_CAI_SALIENCY_TOP10_OVERLAP` from the same table.

The resulting authority has 42 unique rows. Demote
`U3_RECONSTRUCTION_ORACLE`, `U4_ORACLE_CAI_SPECIFICITY`,
`U4_ORACLE_IMAGE_SPECIFICITY`, `U4_LEARNED_SPECIFICITY_BOUNDARY`, and
`O3_REAL_MINUS_RECONSTRUCTION` to supplement/legacy visibility. The first
three and the O3 row leave the main paper; the learned boundary was already
supplement-only. `U3_UNIFORM_ORACLE` and `U3_HEADROOM_RETENTION` remain bounded
main support but are not mixed into Figure 2(c), because their one-shot MVD
protocol differs from the sequential A2 saliency comparison.

The new A2 claims are `PRE_P7_FROZEN_EVIDENCE`: the preregistration and A2
result commit predate the frozen P7 outer endpoint. Chronology generation will
classify these exact IDs as `MVA_A2_ORACLE_VALUE` without changing the timing
of legacy U4 diagnostics.

## Figure contract

### Figure 2: WHAT

Core claim: spatial C-scan information helps CAI prediction, and retrospective
CAI-oriented acquisition is not reproduced by the preregistered task-agnostic
appearance-saliency heuristic.

Panel sequence:

- (a) unchanged U1/U2 spatial gain and sparse retention;
- (b) hash-verified initial legal C-scan state, with bilinear interpolation
  used only as a partial-state visualization;
- (c) appearance-minus-mechanical CAI AUEBC forest point, interval, and 6/6
  direction under the common A2 protocol;
- (d) `mechanical_oracle` within-map percentiles and top five cells;
- (e) `appearance_oracle` within-map percentiles and top five cells;
- (f) paired CAI-minus-saliency percentile difference, with the 276-map mean
  Spearman and top-decile overlap as aggregate context.

Panels (d) and (e) share the same 0--1 percentile scale. Panel (f) uses a
zero-centered diverging scale. Absolute appearance and CAI utilities are never
placed on a common scale.

### Figure 4: HOW

Remove the reconstruction-derived endpoint and contrast from panel (a). Keep
measured state, acquired-position/history, real-state change, and the adverse
shuffled-content contrast. Keep panels (b)--(d), including A4's unchanged
negative direction. The frozen O3 reconstruction control remains explicit in
the supplement and machine-readable authority and is never renamed saliency.

## Manuscript contract

Rewrite Abstract through Conclusion around task-oriented value versus a simple
task-agnostic saliency heuristic. Methods state the exact registered formula,
CAI-label exclusion, retrospective access, and non-deployability. Main-visible
text and generated figure captions must not contain a scientific
reconstruction, field-content, field-recovery, or image-recovery storyline.
Internal code names and source paths may retain historical identifiers.

The supplement keeps the complete reconstruction-derived U3/U4/O3 evidence,
P14 normalized-RGB-MSE scope, adverse controls, chronology, and implementation
identity. It labels these rows as legacy technical reconstruction evidence,
distinct from appearance saliency.

The Chinese comparison draft mirrors the final English source section by
section and remains outside the submission package.

## Failure behavior and verification

Loaders fail closed on missing methods, duplicate or missing cells, mismatched
domain/grid/state, non-finite values, or writable arrays. No synthetic
fallback exists. Tests first establish RED for the saliency adapter, new
canonical claims, Figure 2 source bindings and visible text, Figure 4 removal,
main/supp terminology partition, chronology, and package determinism.

Completion requires the focused/full paper suites, Ruff, deterministic figure
and authority replay, rendered figure QA, clean main/supplement/flat LaTeX,
deterministic ZIP replay, empty frozen-path diff, two commits, remote/local SHA
equality, and a clean worktree.
