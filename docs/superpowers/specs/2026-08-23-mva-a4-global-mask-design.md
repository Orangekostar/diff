# MVA A4 Global Task-Aware Static Acquisition Design

Date: 2026-08-23
Status: approved through the user's no-question, recommended-configuration authorization

## Scope

A3 issued `MVA_ORACLE_GO`, so A4 is authorized. A4 tests whether a single
source-learned spatial refinement ranking can improve CAI assessment under the
same normalized-raster budget. It does not implement a differentiable mask,
specimen-adaptive policy, reinforcement learning, laminate conditioning, or
structured transfer.

The stage emits two independent decisions:

- whether the global mechanical-value mask improves fixed acquisition; and
- whether enough oracle-versus-static headroom remains to authorize A5.

## Considered approaches

1. Aggregate the published A2 oracle table. Rejected: a source specimen's A2
   predictor can include the current A4 held-out domain, so the resulting mask
   would leak outer-domain information.
2. Aggregate outer-specific, strictly OOF initial candidate ranks. Selected:
   it produces one fixed 8 x 8 ranking per outer fold, keeps the target domain
   absent from every fit and label, and remains simple enough for 276 specimens.
3. Aggregate complete specimen-specific greedy trajectories. Rejected: later
   values are conditioned on different specimen states and do not define a
   comparable static score for each cell.

## Leakage-safe source labels

For held-out domain `d`, the candidate training pool is the other five domains.
For each query source domain `q`, the P-A label predictor is fitted on the four
domains excluding both `d` and `q`. PCA selection is nested inside those four
domains. Every query specimen in `q` receives values for all 64 legal initial
level-0 to level-1 cell refinements.

Mechanical labels use true query CAI only after the OOF predictor is fixed.
Reconstruction and appearance labels use the query full image only as offline
source supervision. No target CAI, target full-image value, target embedding,
or target-selected threshold contributes to a ranking.

Initial candidate reconstructions and ResNet18 embeddings depend only on the
specimen and the A1-selected initial budget. A checksum-bound work cache may
reuse them across outer folds. Cache reuse never changes model fitting or
source/target rosters.

## Static ranking

The primary global score is an equal-source-domain mean of specimen-normalized
candidate ranks:

1. sort each specimen's 64 values descending, breaking ties by lower cell index;
2. map rank 1 to 1 and rank 64 to 0;
3. average ranks within each source domain;
4. average the five domain means equally;
5. sort the 64 cells descending, again breaking ties by lower cell index.

This prevents domains with more specimens or larger raw value scales from
dominating the pattern. Mean raw value and value per newly measured location
are serialized as diagnostics but cannot select the primary ranking.

The three source-learned masks are:

- `global_mechanical_mask` from absolute CAI-error reduction;
- `global_reconstruction_mask` from normalized RGB MSE reduction;
- `global_appearance_mask` from full-image border-median appearance intensity.

For every target specimen, a mask applies the same cell order until no next
legal action fits the nominal checkpoint. Actions advance only from level 0 to
level 1, so the static mask spans the registered range through 25% without
introducing specimen-specific decisions.

## Evaluation

The outer target domain is evaluated at 3.125%, 6.25%, 9.375%, 12.5%, 18.75%,
and 25%. A checkpoint below the selected survey is absent. P-B is exactly the
A2 budget-specific predictor trained on uniform states from the five source
domains and applied unchanged to every global method.

Primary comparisons are:

- uniform;
- random median with all 100 A2 seeds retained;
- global appearance mask;
- global reconstruction mask;
- global mechanical-value mask;
- A2 specimen-specific mechanical oracle as a diagnostic upper bound.

Primary prediction evidence is equal-domain CAI MAE, AUEBC on `[0.0625, 0.25]`,
and `B_5%`. Normalized RGB MSE and SSIM are reported for the three global masks
to test whether image fidelity and mechanical utility diverge. P-A remains a
sensitivity only.

## Statistics and gates

All paired AUEBC effects reuse one preregistered 100000 x 6 PCG64 synchronized
domain-bootstrap index matrix with ordinary percentile 95% intervals.

The A4 global-mask decision passes only when all conditions hold:

- uniform-minus-global-mechanical AUEBC is positive, its 95% lower bound is
  above zero, and at least 4/6 domains improve;
- global-reconstruction-minus-global-mechanical AUEBC is positive and its 95%
  lower bound is above zero;
- global-appearance-minus-global-mechanical AUEBC is positive and its 95%
  lower bound is above zero.

The independent A5 authorization passes only when:

- global-mechanical-minus-mechanical-oracle AUEBC is positive;
- the relative gap is at least 3% of global-mechanical AUEBC;
- the synchronized-bootstrap lower bound is above zero; and
- the oracle improves in at least 4/6 domains.

Failure of the A4 mask gate does not rescue or alter A3. Failure of the A5
authorization gate stops sequential-policy development. No threshold may be
changed after formal results are inspected.

## Robustness and artifacts

Nonselecting ranking diagnostics remove one source domain at a time and report
top-10% overlap, Spearman correlation, and rank-biased overlap against the
five-domain ranking. They may reveal instability but cannot alter the primary
mask or gate.

`results/mva/a4_global_task_mask/` contains source OOF candidate values, global
rankings, target trajectories and state metrics, CAI and image-quality curves,
domain effects, bootstrap intervals, stability diagnostics, figures,
`summary.json`, `REPORT.md`, checksums, and an artifact manifest. An independent
replay must reproduce the validated byte tree.

## Claim boundary

A positive A4 result supports a source-learned fixed normalized-raster sampling
pattern. It does not establish a physical scanner mask, inspection-time saving,
online adaptation, or a deployable specimen-specific policy. The A2 oracle
remains retrospective and nondeployable.
