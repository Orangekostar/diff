# E1 Cross-View Audit Protocol

Protocol date: 2026-08-21
Formal status: registered before evaluation; executed 2026-08-21

## Inputs and Split

E1 uses the exact 276-specimen roster and the three immutable views `FULL`,
`BILINEAR_50`, and `BILINEAR_25`. Outer evaluation is six-domain LODO. For each
outer domain and view, PCA dimension is selected independently through LODO over
the five source domains. No view copy becomes a new specimen.

## Required OOF Table

`results/multiview/e1_audit/oof_predictions.csv` contains one row per specimen
and at least:

```text
specimen_id,domain_id,y_true,pred_full,pred_50,pred_25,err_full,err_50,err_25
```

The formal table additionally records source CAI strength, recovered intact
strength, MPa predictions, and MPa absolute errors. Those columns are
evaluation-only and cannot be accessed by model fitting or selection.

All three predictions on a row are made by models that excluded that specimen's
outer domain from every fit and selection step.

## Individual Metrics

For each view, report equal-domain MAE, each domain MAE, worst-domain MAE,
domain-MAE SD, pooled RMSE, and pooled R2. FULL must match all frozen P1
predictions within `1e-12` before other E1 results are valid.

## Agreement and Complementarity

Report all three prediction Pearson correlations, residual Pearson
correlations, and mean absolute pairwise prediction disagreements. The oracle
uses the smallest absolute error among the three views for each specimen and is
labelled non-deployable.

Best-view frequency is reported globally and by domain, ply count, layup family,
and available damage descriptors. Registered view order resolves exact ties.

## Reliability

Cross-view dispersion is the population standard deviation of the three outer
OOF predictions. Correlate it with absolute error of each named deployable fusion
using Pearson and Spearman statistics. Error is also reported for deterministic
rank strata: lowest 25%, middle 50%, and highest 25% dispersion.
Both coefficients and two-sided p-values are stored, together with a one-row-per-
specimen dispersion/error/stratum table.

## Gate

E1 returns GO when either prediction correlation is high with competitive
individual performance, or at least two useful views have non-redundant residuals
and a material oracle gain. An oracle improvement of 10% is the preferred
complementarity signal, not a deployable success threshold.

The frozen numerical interpretation is: predictive equivalence requires every
pairwise prediction correlation to be at least `0.95` and every individual MAE
to be at most `1.10` times FULL. Complementarity requires at least two such
useful views, at least one residual-correlation pair at or below `0.90`, and an
oracle improvement of at least `0.10` relative to the frozen FULL comparator.

Extremely high residual correlation together with negligible oracle improvement
blocks E3/E4 complementarity claims. E2 may still run under predictive
equivalence to test whether soft prediction cooperation is useful.
