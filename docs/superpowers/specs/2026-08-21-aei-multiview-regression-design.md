# AEI Mechanics-Consistent Multi-View Regression Design

Date: 2026-08-21
Status: approved by the controlling prompt and the user's recommended-configuration authorization
Target venue: Advanced Engineering Informatics

## Objective

Test whether `FULL`, `BILINEAR_50`, and `BILINEAR_25` C-scan views encode the
same CAI-relevant mechanical state through distinct representations, and whether
prediction-level cooperation can improve strict cross-domain CAI assessment
without returning to feature invariance.

The frozen primary comparator is equal-domain CAI-ratio MAE
`0.08963580465761432`. Existing A0--A5 artifacts remain read-only, including the
`FACTORISATION_NO_GO` decision.

## Authoritative Inputs

- Cohort: 276 unique specimens in six registered domains.
- Independent unit: specimen for every split, fit, bootstrap, stacking row, and
  statistical comparison.
- Views: exactly `FULL`, `BILINEAR_50`, and `BILINEAR_25` from the immutable
  `(276, 3, 512)` paired feature bank.
- Encoder: frozen ImageNet ResNet18 bound to the registered checkpoint digest.
- Response: damaged-to-intact CAI strength ratio, unit `1`.
- Metadata: the same 13 mechanics fields used by the frozen `I_frozen` baseline.
- Primary evaluation: outer six-domain LODO with inner source-domain LODO.

## Stage Design

### E1: Predictive Audit

Each view receives an independent fold-local SVD-PCA and Ridge predictor. PCA
dimension is selected independently from `8`, `16`, and `32` on source-only
inner LODO MAE. Fold-local mean imputation, scaling, and `Ridge(alpha=10)` match
the registered P1 estimator. This construction must reproduce all frozen FULL
predictions to numerical tolerance before the other views are interpreted.

Strict outer OOF predictions drive individual metrics, prediction and residual
correlations, pairwise disagreement, oracle MAE, best-view frequencies, and
cross-view reliability. The oracle remains diagnostic and is never listed as a
deployable method.

### E2: Cooperative Regression

The three experts retain separate PCA bases and coefficient vectors. A joint
continuous-regression objective combines per-view target loss with all three
pairwise prediction-agreement penalties. The registered consistency grid is
`[0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]`; MSE and robust Huber target losses
are source-only inner candidates. Selection uses the mean prediction of the
three experts, while output variance and residual correlation are recorded to
detect collapse.

The implementation uses the augmented-system formulation from Cooperative
Learning for quadratic fits and a deterministic robust optimizer for Huber
fits. No feature-alignment term is permitted.

### E3: Consistency and Complementarity

Required comparators are equal late fusion, non-negative simplex weights fitted
on strict source OOF predictions, leakage-safe stacking, and a lightweight GMvR
variant. Stacking candidates are Ridge, non-negative Ridge, and Huber; their
selection uses domain-held-out meta-validation over source OOF base predictions.

The GMvR variant searches cooperative consistency strength jointly with
non-negative view-specific fusion weights. A weight-concentration penalty
prevents a nominal multi-view model from silently collapsing to one view. This
implements complementarity through view-specific predictive contributions, not
through forced feature differences.

### E4 and E5 Gates

E4 is implemented and evaluated only if an E3 deployable fusion beats the best
single view and the frozen FULL baseline while improving at least four of six
domains. The gate is restricted to a small linear or tree model trained from
source OOF expert predictions and disagreement features.

E5 remains disabled unless E1 is nontrivial, E3 or E4 confirms complementarity,
and a material oracle gap remains after deterministic models. Image generation,
feature invariance, and pixel residual diffusion are forbidden in every branch.

## Metrics and Statistics

Primary reporting uses equal-domain MAE. Secondary reporting includes per-domain
MAE, worst-domain MAE, domain-MAE SD, pooled RMSE, and pooled R2. Multi-view
diagnostics include prediction correlation, residual correlation, prediction
disagreement, oracle MAE, best-view frequency, fusion gain, dispersion, and
dispersion-error correlation.

Confirmatory method effects reuse the P1 common six-domain PCG64 bootstrap:
100,000 shared resamples, seed `20260811`, ordinary 95% intervals, and
family-wise intervals for four registered comparisons. Leave-ply-count-out and
leave-layup-family-out are secondary engineering stress tests.

## Components

- `view_experts.py`: exact fold-local PCA/Ridge experts and baseline replay.
- `agreement_audit.py`: E1 metrics, oracle, and grouped win/loss analysis.
- `cooperative_regression.py`: joint peer regressors and collapse diagnostics.
- `late_fusion.py`: equal and non-negative validation-weighted fusion.
- `stacking.py`: strict-OOF meta-regression.
- `gmvr_regression.py`: consistency plus view-specific weighted contribution.
- `reliability.py`: dispersion, error correlation, and reliability strata.
- `formal_outer.py`: ordered E1--E3 outer evaluation and conditional gates.
- `artifacts.py` and `replay.py`: atomic, hash-bound outputs and replay checks.
- `scripts/run_aei_multiview_regression.py`: audit, run, and replay entry point.

## Failure Handling

- Any roster, source digest, view order, target, or frozen-baseline mismatch
  aborts before fitting.
- Every data-dependent transform is fitted from the current training rows only.
- A non-finite solve, failed optimizer, or incomplete prediction vector is a hard
  error; it is not replaced with a favorable score.
- Ties use the registered order, preferring simpler and weaker regularization.
- Stage outputs are written atomically and cannot overwrite an existing formal
  package.

## Verification Contract

Unit tests must prove baseline equivalence, fold isolation, specimen grouping,
agreement-objective behavior, collapse detection, simplex weights, strict OOF
stacking, gate ordering, metric correctness, and artifact replay. The production
run must then generate all required E1--E3 artifacts and pass their checksum
replay before any scientific conclusion is promoted.
