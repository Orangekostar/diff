# MVA A4 Preregistered Global Static Mask Protocol

Date: 2026-08-23
Status: frozen after `MVA_ORACLE_GO` and before A4 result generation
Authority: MVA controlling prompt and the approved A4 design

## 1. Scope and authorization

A3 passed all four registered hypotheses and issued `MVA_ORACLE_GO`; therefore
A4 is authorized. This stage evaluates source-learned fixed acquisition masks
and decides whether a sequential A5 policy is warranted. It does not train a
differentiable mask, imitation policy, reinforcement learner, laminate-aware
model, or transfer model.

A4 cannot alter the A0-A3 cohort, baseline, acquisition grid, predictor,
interpolation, checkpoints, metrics, bootstrap matrix, result, or claim scope.

## 2. Frozen authority

- Cohort: the same 276 specimens and six registered domains as A0-A3.
- Target: continuous damaged-to-intact CAI strength ratio.
- Geometry: normalized 8 x 8 cells on the native registered raster crop.
- Initial survey: the source-selected A1 budget for the current outer domain.
- Static action: advance one cell from level 0 to level 1.
- Reconstruction: deterministic bilinear interpolation with exact restoration
  of every measured RGB value.
- Encoder: frozen ImageNet ResNet18 final 512-dimensional embedding.
- Estimator: metadata13 plus embedding, fold-local mean imputation and scaling,
  PCA candidates 8/16/32, and Ridge alpha 10.
- Primary evaluator: the A2 P-B head trained on source-domain uniform states at
  the same checkpoint.

The A3 full-domain equal-domain MAE remains `0.08963580465761432` and defines
the registered sufficiency thresholds.

## 3. Outer-specific source OOF supervision

For outer target domain `d`, only the other five domains may construct a mask.
For every query source domain `q`, the mechanical-value P-A predictor is fitted
on the four domains excluding both `d` and `q`. Its PCA dimension is selected
by nested leave-one-domain-out evaluation within those four fit domains.

Every query specimen receives one label for each of the 64 initial level-0 to
level-1 candidates:

- mechanical: absolute CAI-error reduction under the strict OOF P-A predictor;
- reconstruction: normalized RGB MSE reduction;
- appearance: mean absolute newly revealed RGB deviation from the full-image
  border median, retained as offline source supervision rather than physical
  damage severity.

The target domain contributes no CAI, image, embedding, value, rank, model fit,
hyperparameter, or threshold to mask construction. Published A2 source rows
are not reused as A4 labels because their predictors can include the current
A4 target domain.

## 4. Candidate bank

Initial candidate images and embeddings are deterministic functions of a
specimen and its initial budget. A checksum-bound work cache may store all 64
candidate embeddings, reconstruction values, appearance values, and exact
added-measurement counts for each specimen at 1.5625% and 3.125% initial
budgets. Cache validation binds specimen order, authority state, input hashes,
budget, grid state, shapes, dtypes, finite values, and content digests.

The cache is a runtime optimization, not an artifact or evidence source. A
cache miss or invalid cache causes deterministic regeneration.

## 5. Global rankings

The registered methods are:

- `global_appearance_mask`;
- `global_reconstruction_mask`;
- `global_mechanical_mask`.

For each specimen and method, candidate values are sorted descending with lower
cell index as the tie break. Rank 1 maps to 1 and rank 64 maps to 0. Normalized
ranks are averaged first within each source domain and then equally across the
five source domains. The global cell order sorts these equal-domain scores
descending with lower cell index as the tie break.

Mean raw value and mean value per newly measured location are diagnostic only.
They cannot select or modify the primary ranking.

## 6. Target acquisition and evaluation

Each target specimen uses the exact same cell order learned for its outer fold.
At each checkpoint the policy takes the next ordered level-1 refinement only
when its exact unique-measurement count fits the specimen cap. It never reads
the target image, current RGB content, CAI prediction, or true CAI to choose a
cell. Revealed RGB values are used only to update the registered reconstruction
after the fixed action is chosen.

Registered checkpoints are 3.125%, 6.25%, 9.375%, 12.5%, 18.75%, and 25%.
Checkpoints below a selected initial survey are absent. Every state records the
nominal cap, measured count, native count, and actual effective fraction.

One P-B model per outer domain and checkpoint is trained only on source uniform
states and applied unchanged to all global masks. P-A predictions are retained
as sensitivity. A2 uniform, all 100 random seeds, and the specimen-specific
mechanical oracle are checksum-bound reference curves.

## 7. Metrics

Primary CAI evidence is specimen absolute error, equal-domain MAE, AUEBC on
`[0.0625, 0.25]`, and `B_5%`. `B_2.5%` and `B_7.5%` are secondary. Normalized
RGB MSE and SSIM are reported for each global mask at every checkpoint.

The key image-versus-task comparison is global reconstruction mask versus
global mechanical mask under the same target specimens, budgets, encoder, and
P-B evaluator. Better image fidelity cannot substitute for lower CAI error.

## 8. Statistics

All paired AUEBC effects reuse one 100000 x 6 PCG64 domain-bootstrap index
matrix with seed 20260823 and ordinary percentile 95% intervals. Domain order
is frozen to the A0-A3 order. All six domain effects remain visible.

## 9. A4 global-mask gate

The global-mask gate passes only when all conditions hold:

1. uniform-minus-global-mechanical AUEBC is positive, its synchronized 95%
   lower bound is above zero, and global mechanical improves in at least four
   of six domains;
2. global-reconstruction-minus-global-mechanical AUEBC is positive and its
   synchronized 95% lower bound is above zero;
3. global-appearance-minus-global-mechanical AUEBC is positive and its
   synchronized 95% lower bound is above zero.

The terminal A4 status is exactly one of:

- `MVA_A4_GLOBAL_GO`;
- `MVA_A4_GLOBAL_NO_GO`.

## 10. A5 authorization gate

A5 is authorized only when all conditions hold:

1. global-mechanical-minus-mechanical-oracle AUEBC is positive;
2. that point difference divided by global-mechanical AUEBC is at least 3%;
3. its synchronized 95% lower bound is above zero;
4. the oracle improves in at least four of six domains.

The A5 status is exactly one of:

- `MVA_A5_AUTHORIZED`;
- `MVA_A5_NOT_AUTHORIZED`.

The A4 and A5 decisions are independent. A failed global mask does not change
A3. A failed A5 authorization stops sequential-policy development without a
rescue search.

## 11. Nonselecting diagnostics

For each outer fold and method, remove one of the five source domains and
recompute the ranking. Compare it with the primary ranking using Spearman,
top-10% overlap, and rank-biased overlap with persistence 0.9. These diagnostics
cannot alter a ranking, threshold, or decision.

## 12. Artifacts and replay

The formal directory is `results/mva/a4_global_task_mask`. It must contain raw
source values, fit audits, rankings, ranking stability, fixed trajectories,
target state metrics, CAI curves, image curves, domain/budget metrics,
bootstrap effects, `summary.json`, `REPORT.md`, figures with source data,
checksums, and an artifact manifest.

Validation independently recomputes source/target rosters, rank aggregation,
action order, budgets, errors, curves, AUEBC, sufficiency, bootstrap effects,
and both decisions. Replay must reproduce an identical validated byte tree.

## 13. Claim boundary

A4 is a retrospective normalized-raster simulation. A positive result supports
a fixed source-learned observation pattern only. It does not establish physical
scanner coordinates, inspection-time reduction, an online adaptive scanner,
or a deployable specimen-specific policy.
