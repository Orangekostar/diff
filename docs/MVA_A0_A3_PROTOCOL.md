# MVA A0-A3 Preregistered Protocol

Date: 2026-08-23
Status: frozen before formal A1/A2 execution
Authority: the user-approved Mechanical-Value Acquisition controlling prompt

## 1. Scope and stop boundary

This protocol implements only acquisition audit (A0), refinement simulation
(A1), retrospective oracle construction (A2), and the oracle-headroom gate
(A3). It must emit exactly one terminal decision:

- `MVA_ORACLE_GO`: all four preregistered hypotheses pass; or
- `MVA_ORACLE_NO_GO`: one or more hypotheses fail.

No A4-A7 model, global mask, imitation policy, reinforcement learner, laminate
conditioning, rescue search, or threshold change is permitted in this run.

## 2. Frozen cohort and prediction authority

- Cohort: the existing 276-specimen, six-domain P1 authority cohort.
- Target: continuous CAI.
- Domain order: the existing registered P1 domain order.
- Input: frozen `metadata13` plus the complete registered RGB crop, resized to
  224 x 224 only by the frozen ResNet18 transform.
- Encoder: frozen ImageNet ResNet18 final 512-dimensional pooled feature.
- Predictor: fold-local standardized PCA with candidate dimensions 8, 16, and
  32, followed by Ridge with alpha 10.
- Selection: nested leave-one-domain-out on the five source domains only.
- Primary error: specimen absolute error, aggregated as equal-domain MAE.

Before any formal oracle output, a fresh baseline reproduction must equal
`0.08963580465761432` within absolute tolerance `1e-12`. Failure is terminal.
P7's image-only 25% reconstruction and embedding are the registered sparse
geometry endpoint. Its prediction value `0.09011592076510834` is reported only
as prior context because P7 omitted `metadata13`; it is not an MVA P-B
prediction authority.

## 3. Acquisition representation

The experiment is a retrospective raster acquisition simulation. It does not
claim native scanner pitch, scan time, or one physical ultrasonic measurement
per RGB pixel.

Each native crop is divided into a normalized 8 x 8 cell grid. Cell boundaries
are aligned to the P5 25% endpoint-preserving sampling coordinates. A cell has
three nested levels:

1. level 0: the selected initial endpoint-preserving lattice;
2. level 1: the exact P5 25% lattice restricted to that cell;
3. level 2: all native raster coordinates in that cell.

Every action advances one cell by exactly one level. Measurements are never
removed. Budget is

`number of unique measured native-raster locations / native pixel count`.

Mixed states are reconstructed by deterministic cell-wise bilinear
interpolation on local rectilinear lattices, stitched in row-major normalized
cell order. All measured RGB values are restored exactly afterward. A global
rectilinear state uses the exact P5 tensor interpolation so that all-level-1 is
byte-identical to the P5 25% reconstruction.

## 4. Initial survey selection

Candidate nominal initial budgets are 1.5625%, 3.125%, and 6.25%. Selection is
performed independently inside each outer fold using only its five source
domains and inner LODO evaluation.

Choose the lowest candidate whose source equal-domain sparse MAE is:

- no more than 1.5 times the matched FULL source MAE; and
- at least 1.025 times the matched FULL source MAE.

If no candidate satisfies both, choose the lowest candidate under the upper
bound and label the fold `weak_headroom`. If none satisfies the upper bound,
stop without A2.

## 5. Oracle construction and controls

For outer domain `d`, every target specimen is evaluated by a P-A predictor
trained only on the other five domains. PCA selection uses inner LODO inside
those source domains. True target CAI and newly revealed candidate RGB values
are used only for retrospective oracle labeling.

The registered methods are:

- `uniform`: deterministic farthest-spread cell refinement;
- `random`: legal refinements sampled with 100 fixed PCG64 seeds;
- `appearance_oracle`: greedy full-image RGB distance from border-median
  appearance, explicitly not a damage-severity method;
- `reconstruction_oracle`: greedy normalized RGB MSE reduction;
- `mechanical_oracle`: greedy CAI absolute-error reduction under P-A;
  squared-error reduction is retained as a secondary label.

Ties are broken by lower cell index and then lower resulting level. An action
is legal only when its unique newly measured locations fit the specimen's
checkpoint cap. Center-first is excluded because impact-center authority is
absent.

## 6. Curve evaluator

P-B is the primary curve evaluator. At each checkpoint it is trained only on
uniform states from the five source domains and then applied unchanged to every
method in the outer target domain. Thus all methods share one checkpoint-level
predictor and the target domain never fits preprocessing, PCA, Ridge, survey
selection, or thresholds. P-A values are published as sensitivity only.

Global cross-fitted A2 trajectories are diagnostic artifacts. They are not
eligible training data for a future A4/A5 model; outer-specific source
trajectories would have to be regenerated after a GO decision.

## 7. Checkpoints and metrics

Registered nominal checkpoints are 3.125%, 6.25%, 9.375%, 12.5%, 18.75%, 25%,
50%, and 100%. Checkpoints below a fold's selected initial survey are absent.
The 50% uniform and 100% FULL values are report-only anchors.

Every row contains nominal cap, measured count, native count, and actual
effective fraction. Primary AUEBC is trapezoidal integration on the common
nominal interval `[0.0625, 0.25]`. `B_5%` is the first checkpoint with MAE no
greater than `1.05 * 0.08963580465761432`; 2.5% and 7.5% are secondary.
Unreached thresholds remain null.

Random reports all seeds plus mean, median, 5th percentile, and 95th percentile.
Map comparison uses Pearson, Spearman, top-10% overlap, and rank-biased overlap
with persistence 0.9. Nonselecting stability diagnostics compare bilinear to
nearest/bicubic and Ridge alpha 10 to 1/100.

## 8. Statistics

One preregistered 100000 x 6 PCG64 domain-bootstrap index matrix is shared by
all paired effects. Ordinary percentile 95% confidence intervals are primary.
All six domain results and all 100 random seeds remain visible.

## 9. A3 hypotheses and decision

The fixed low-budget checkpoint is 12.5%.

- H1 passes when mechanical P-B improves on uniform by at least 5% at 12.5%
  and has lower domain MAE in at least four of six domains.
- H2 passes when reconstruction-oracle AUEBC minus mechanical-oracle AUEBC is
  positive and its synchronized-bootstrap 95% lower bound is above zero.
- H3 passes when appearance-oracle AUEBC minus mechanical-oracle AUEBC is
  positive and its synchronized-bootstrap 95% lower bound is above zero.
- H4 passes when mechanical improves relative AUEBC by at least 10% over the
  stronger of uniform and random median, or reduces `B_5%` by at least 25%.

All H1-H4 must pass for `MVA_ORACLE_GO`; otherwise the decision is
`MVA_ORACLE_NO_GO`. Missing metrics fail their associated hypothesis.

## 10. Required artifacts and replay

A0, A1, and A2 are published under `results/mva/`. A2 must include raw oracle
values and trajectories, all method curves, budget/domain metrics, map and
stability diagnostics, `summary.json`, `REPORT.md`, O1-O5/error-budget figures,
checksums, and a manifest. The report must state adverse domains, failed gates,
and simulation-only limitations.

Formal publication is transactional. Validation recomputes masks, budgets,
errors, curves, AUEBC, sufficiency budgets, bootstrap effects, and gates from
raw tables. Replay must reproduce an identical validated byte tree before the
terminal decision is accepted.
