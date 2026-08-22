# AEI Mechanics-Consistent Multi-View Scientific Protocol

Protocol date: 2026-08-21
Registration state: frozen before E1--E3 formal evaluation
Target venue: Advanced Engineering Informatics

## Scientific Question

The experiment tests whether three mechanics-validated C-scan sensing views can
encode the same CAI-relevant state through different feature geometries, and
whether prediction-level cooperation improves cross-domain assessment while
preserving useful view-specific information.

The experiment does not optimize feature invariance, stable subspaces, image
reconstruction, pixel residual diffusion, or full-image diffusion.

## Authorities

- The cohort is 276 unique specimens in six frozen domains.
- The independent unit is the specimen for fitting, splitting, stacking,
  weighting, bootstrap, and inference.
- The response is damaged-to-intact CAI strength ratio, unit `1`.
- The registered workbook also supplies damaged CAI strength in MPa. For
  evaluation only, specimen-specific intact strength is recovered as damaged
  strength divided by the source-defined ratio; ratio predictions are then
  multiplied by that frozen scale. MPa values never enter fitting, selection,
  weighting, or gates.
- Primary views are exactly `FULL`, `BILINEAR_50`, and `BILINEAR_25`.
- Features come from the immutable `(276, 3, 512)` frozen ResNet18 bank.
- The primary comparator is equal-domain MAE `0.08963580465761432`.
- Existing A0--A5 outputs are read-only; A5 remains `FACTORISATION_NO_GO`.

## Estimation

Each view has its own source-fitted PCA basis and regression coefficients. PCA
dimension is selected from `8`, `16`, and `32` by inner source-domain LODO. The
estimator uses the 13 registered mechanics metadata fields, training-only mean
imputation and scaling, and Ridge regularization `alpha=10`. FULL must reproduce
the frozen baseline predictions to absolute tolerance `1e-12`.

Cooperative regression minimizes per-view continuous target loss plus all three
pairwise prediction-agreement penalties. It searches MSE and Huber target loss
and consistency strengths `0`, `0.001`, `0.003`, `0.01`, `0.03`, `0.1`, `0.3`,
and `1.0` using only source-domain inner predictions.
Every candidate records all three prediction variances and all three pairwise
residual correlations so that agreement without accuracy gain cannot be
misclassified as improvement.

## Evaluation and Gates

E1 reports strict outer OOF individual accuracy, agreement, residual
correlation, oracle complementarity, best-view maps, and reliability. Predictive
equivalence or useful complementarity authorizes E2.

E2 compares independent mean, validation-weighted mean, and selected cooperative
regression. It is positive only below the frozen FULL MAE with at least four of
six domains improved. A selected zero consistency strength is evidence against
agreement regularization, not permission to increase model complexity.

E3 compares equal fusion, source-OOF non-negative weighting, leakage-safe Ridge,
non-negative Ridge and Huber stacking, and a lightweight GMvR-style objective.
Complementarity requires a deployable fusion below both the best single view and
the frozen FULL comparator, with at least four improved domains.

E4 is unauthorized until E3 confirms complementarity. E5 is unauthorized until
E1 is nontrivial, E3 or E4 confirms complementarity, and deterministic methods
leave a material oracle gap.

The remaining-oracle-gap threshold for E5 authorization is `0.05` relative to
the best deterministic deployable MAE. E2 and E3 require at least four improved
domains; E3 additionally requires performance below both FULL and the best
individual view.

## Statistics

Primary performance is equal-domain CAI-ratio MAE. Secondary metrics are
per-domain MAE, worst-domain MAE, domain-MAE SD, pooled RMSE, and pooled R2.
Because the source workbook preserves both the ratio and damaged strength,
equal-domain MAE and pooled RMSE are additionally reported in MPa on the
specimen-specific recovered intact-strength scale.
Four confirmatory effects share one six-domain `PCG64(20260811)` bootstrap with
100,000 resamples, ordinary 95% intervals, and family-wise intervals using
quantiles `0.00625` and `0.99375`.

Leave-ply-count-out and leave-layup-family-out are secondary engineering stress
tests. Their preprocessing, selection, and fusion weights remain source-only.

## Claim Rule

Only hash-bound formal artifacts may promote a claim. Oracle selection is an
upper-bound diagnostic. Expected trends, unrun branches, and source-inner scores
are never reported as deployable outer performance.
