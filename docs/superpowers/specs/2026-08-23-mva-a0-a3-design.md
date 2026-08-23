# MVA A0-A3 Design

Date: 2026-08-23
Status: frozen from the user-approved controlling prompt before implementation

## Scope and selected approach

This package tests whether true CAI reveals useful spatial acquisition
headroom. It implements A0 through A3 and stops after the oracle gate. It does
not implement a global mask, imitation policy, laminate conditioning, RL,
attention, diffusion, a new backbone, or a modified historical gate.

Three implementation approaches were considered:

1. A fixed FULL-trained evaluator for every candidate (P-A). This is simple,
   stable, and makes the retrospective value definition unambiguous, but its
   head is not trained on sparse states.
2. A method-specific predictor retrained on each adaptive trajectory. This
   matches each policy distribution but is circular during greedy construction
   and makes controls use different evaluators.
3. **Selected:** use strict OOF P-A to construct oracle trajectories, then use
   one budget-specific P-B predictor trained only on source-domain uniform
   states to evaluate every method at that checkpoint. This avoids circularity,
   uses the same evaluator for all acquisition methods, exactly recovers the
   registered FULL predictor and the 25% sampling, reconstruction, and
   embedding endpoint, and isolates acquisition effects.

P-A remains a serialized sensitivity. P-B is the primary A3 curve.

## Authority and baseline stop

The exact P1 cohort, target, domain order, RGB crop bytes, ResNet18 weights,
transform, PCA candidates `(8,16,32)`, Ridge alpha `10`, and equal-domain MAE
definition are immutable inputs. Before A1, fresh nested LODO must reproduce
`I_frozen = 0.08963580465761432` within `1e-12`. Failure stops execution.

P7 supplies the image-only 25% bilinear geometry endpoint. A uniform all-cell
level-1 state must reproduce its measurement mask, reconstruction, and
embedding within registered tolerances. Its prediction value
`0.09011592076510834` omitted `metadata13` and is context rather than an MVA
P-B prediction authority.

## Nested acquisition grid

Each native crop is partitioned into 8 x 8 normalized cells. The exact P5 25%
endpoint-preserving row and column coordinates form cell level 1. Cell
boundaries are selected from those coordinates, and each legal initial lattice
is an endpoint-preserving subset of level 1 that contains every boundary.
Level 2 contains every native raster coordinate in the cell. Therefore every
cell obeys `level0 subset level1 subset level2`.

Initial nominal candidates are 1.5625%, 3.125%, and 6.25%. For each outer
domain, only its five source domains select the candidate using inner LODO. The
rule selects the lowest-budget candidate whose source mean MAE is at most 1.5
times the matched FULL source MAE and at least 1.025 times that FULL MAE. If no
candidate satisfies both, it selects the lowest candidate below the upper bound
and marks weak headroom; if none retains the upper bound, execution stops.

An action advances one cell by one level and never changes existing values.
Budget is the count of unique measured locations divided by native pixel count.
For a mixed-resolution mask, reconstruction is cell-wise bilinear on the local
rectilinear lattice, stitched in normalized cell order, followed by exact global
measured-value restoration. A globally rectilinear state uses the exact P5
tensor interpolation. This makes all-level-1 exactly P5-equivalent.

## Policies and value definitions

- `uniform`: deterministic farthest-spread cell order, level 0 to level 1.
- `random`: 100 preregistered PCG64 seeds; candidate cells are sampled uniformly
  from legal one-level refinements.
- `appearance_oracle`: full-image RGB distance from border-median appearance;
  diagnostic, nondeployable, and not called damage severity.
- `reconstruction_oracle`: greedy normalized RGB MSE reduction.
- `mechanical_oracle`: greedy CAI absolute-error reduction under the strict OOF
  P-A evaluator; squared-error reduction is secondary.

Center-first is omitted because impact-center authority is absent. Every oracle
action is limited to candidates whose actual measurement count fits the current
specimen's registered checkpoint cap. Ties use lower cell index, then lower
resulting level.

The requested checkpoints are 3.125%, 6.25%, 9.375%, 12.5%, 18.75%, and 25%
within the primary AUEBC range; lower checkpoints are skipped when the selected
survey exceeds them. The 50% and 100% points are reported as uniform/full
anchors but do not drive A3. Because native rounding differs, every row records
nominal cap and actual fraction; AUEBC uses the common nominal interval
`[0.0625,0.25]`.

## Leakage boundary

For specimen `i` in outer domain `d`, its P-A predictor is fitted only on the
other five domains. PCA dimension is selected only through their inner LODO.
Thus neither `i` nor any specimen in `d` trains its oracle predictor. Candidate
generation reads the current state plus the candidate's newly revealed RGB
values only. True CAI is available solely to the retrospective mechanical value
calculation.

P-B at each checkpoint is fitted on uniform states from the five source domains
and applied unchanged to all target-domain methods. The target domain never
fits PCA, Ridge, survey selection, or a gate threshold. Global cross-fitted A2
source trajectories may not train a future A4/A5 method; outer-specific source
trajectories must be regenerated if A3 passes.

## Metrics and statistics

Primary prediction metric is specimen absolute error followed by equal-domain
MAE. Reconstruction uses normalized RGB MSE; SSIM is secondary and nonblocking.
Each curve reports domain MAE, equal-domain MAE, mean/min/max effective budget,
and method role. Random reports mean, median, 5th, and 95th percentiles over 100
seeds.

`AUEBC` is trapezoidal integration over nominal budgets 6.25% through 25%.
`B_5%` is the lowest checkpoint with MAE no greater than
`1.05 * 0.08963580465761432`; 2.5% and 7.5% are secondary. Missing sufficiency
within 25% is serialized as unavailable, never imputed.

Initial mechanical, reconstruction, and appearance value maps are compared by
Pearson, Spearman, top-10% overlap, and rank-biased overlap with `p=0.9`.
Stability diagnostics compare bilinear with nearest/bicubic interpolation and
Ridge alpha 10 with alpha 1/100 on initial candidate rankings. They cannot
select or rescue the primary oracle.

A shared 100000 x 6 PCG64 domain-bootstrap matrix evaluates all curve/AUEBC
effects. Every adverse domain and random seed remains serialized.

## A3 gate

The preregistered low budget is 12.5%.

- H1: mechanical P-B improves over uniform by at least 5% at 12.5% and in at
  least 4/6 domains.
- H2: reconstruction-oracle AUEBC minus mechanical-oracle AUEBC is positive and
  its ordinary synchronized-bootstrap lower bound is above zero.
- H3: appearance-oracle AUEBC minus mechanical-oracle AUEBC is positive and its
  ordinary synchronized-bootstrap lower bound is above zero.
- H4: relative AUEBC headroom over the stronger of uniform and random-median is
  at least 10%, or the mechanical `B_5%` is at least 25% lower.

All H1-H4 must pass for `MVA_ORACLE_GO`. Otherwise the result is
`MVA_ORACLE_NO_GO`, A4-A7 remain absent, and no rescue search is allowed.

## Artifacts and replay

`results/mva/a0_acquisition_audit` records source and geometry authority.
`results/mva/a1_simulator` records grids, invariants, pilot selection, and P5
reproduction. `results/mva/a2_oracle_value` contains Parquet oracle values and
trajectories, every requested curve, domain/budget metrics, stability and map
similarity tables, summary JSON, report, figures, checksums, and a manifest.

Formal publication is transactional. Validation recalculates budgets,
prediction errors, curves, AUEBC, `B_5%`, bootstrap effects, and gates from raw
tables. Replay independently validates the formal package and republishes an
identical byte tree. Figures read only validated result tables and never feed
the gate.
