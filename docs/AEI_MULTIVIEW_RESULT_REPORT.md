# AEI Multi-View Regression Result Report

Report date: 2026-08-21
Protocol: strict specimen-grouped six-domain LODO with source-only selection

## Outcome

E1 passed only through predictive equivalence. E2 cooperative regression and E3
complementarity both failed their registered primary gates. Consequently E4
dynamic gating and E5 distributional transport were not authorized or run.

## Primary Results

| Method | Ratio MAE | Worst-domain MAE | MAE MPa | RMSE MPa | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| FULL | 0.0896358 | 0.127431 | 37.2609 | 54.1971 | Frozen comparator |
| BILINEAR_50 | 0.0920902 | 0.130406 | 38.3345 | 55.7618 | Predictively equivalent |
| BILINEAR_25 | 0.0911802 | 0.132260 | 38.0400 | 54.7305 | Predictively equivalent |
| Equal fusion | 0.0906272 | 0.129242 | 37.7432 | 54.7325 | E3 best, but worse than FULL |
| Validation-weighted fusion | 0.0913031 | 0.132260 | 38.0622 | 55.1140 | No gain |
| Selected cooperative | 0.0939054 | 0.138467 | 38.9542 | 55.5731 | E2 NO-GO; 0/6 domains improved |
| Selected stacking | 0.0922109 | 0.146556 | 38.4892 | 56.4004 | No gain |
| Selected GMvR | 0.0940065 | 0.139376 | 39.0320 | 55.8447 | No gain |

FULL reproduced the frozen P1 predictions with maximum absolute difference
`4.44e-16`. Pairwise prediction correlations were `0.9935`--`0.9977`, while
residual correlations remained `0.9870`--`0.9957`. The non-deployable oracle MAE
was `0.0821809`, an `8.32%` upper-bound improvement, below the registered `10%`
complementarity signal.

The MPa evaluation uses each specimen's workbook CAI strength divided by its
source-defined ratio as the intact-strength scale. It is a post-hoc unit
conversion only and does not affect training, selection, or gates.

All E2 inner candidates now expose prediction variance and pairwise residual
correlation in `search_diagnostics.csv`; no candidate was classified as
collapsed. Cross-view dispersion had weak, nonsignificant association with
equal-fusion absolute error (Pearson `r=0.0734`, `p=0.2241`; Spearman
`rho=0.0681`, `p=0.2594`). The per-specimen evidence is stored in
`e1_audit/reliability_oof.csv`.

## Inference

All four confirmatory effects use the same 100,000-resample six-domain
`PCG64(20260811)` bootstrap. FULL-minus-candidate point effects were negative:
cooperative `-0.004270`, validation-weighted `-0.001667`, selected stacking
`-0.002575`, and GMvR `-0.004371`. The family-wise intervals for cooperative and
GMvR were entirely below zero; stacking intervals crossed zero.

## Engineering Stress Tests

Leave-ply-count-out best equal-group MAE was `0.091209` for GMvR versus FULL
`0.097959`. Leave-layup-family-out best equal-group MAE was `0.154123` for GMvR
versus FULL `0.187293`. These are secondary group-extrapolation diagnostics with
substantially larger absolute errors, not evidence that overturns the primary
six-domain complementarity NO-GO.

## Claim Boundary

The supported conclusion is that different mechanics-valid sensing views yield
highly similar specimen-level CAI judgments without feature-level invariance.
The data do not support a claim that prediction agreement, static fusion,
stacking, GMvR-style weighting, disagreement-based reliability, dynamic gating,
or distributional transport improves primary cross-domain CAI prediction.
