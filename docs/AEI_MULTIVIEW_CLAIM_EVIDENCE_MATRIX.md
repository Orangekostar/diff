# AEI Multi-View Claim-Evidence Matrix

Matrix date: 2026-08-21
State: formal E1--E3 complete and replay-bound

| Claim | Formal evidence | Promotion rule | Outcome |
| --- | --- | --- | --- |
| FULL baseline is reproduced | `results/multiview/e1_audit/summary.json` | Maximum prediction difference at most `1e-12` | SUPPORTED: maximum difference `4.44e-16`; equal-domain MAE `0.08963580465761434` |
| Views are predictively equivalent | `e1_audit/aggregate_metrics.csv`, `agreement.csv` | Pairwise prediction correlation at least `0.95`; each MAE at most `1.10` times FULL | SUPPORTED: correlations `0.9935`--`0.9977`; MAEs `0.08964`--`0.09209` |
| Views have complementary errors | `e1_audit/agreement.csv`, `summary.json` | Residual correlation at most `0.90` and oracle improvement at least `10%` | NOT SUPPORTED: residual correlations `0.9870`--`0.9957`; diagnostic oracle gain `8.32%` |
| Prediction consistency is useful | `e2_cooperative/aggregate_metrics.csv`, `summary.json` | MAE below FULL with at least four improved domains | REJECTED: cooperative MAE `0.0939054`; `0/6` domains improved |
| Consistency avoids collapse | `e2_cooperative/search_diagnostics.csv` | Finite peer variance, monitored residual correlation, and no collapsed candidate | DIAGNOSTIC PASS: all 96 candidates retained finite variance and nonzero disagreement; all three residual correlations are serialized for every candidate; this did not improve accuracy |
| View complementarity improves CAI | `e3_complementarity/aggregate_metrics.csv`, `summary.json` | Best fusion below best single and FULL with at least four improved domains | REJECTED: equal fusion was best at `0.0906272`, worse than FULL, with `2/6` domains improved |
| Disagreement is a reliability signal | `e1_audit/reliability.csv`, `reliability_oof.csv` | Material significant positive dispersion-error association | NOT PROMOTED: Pearson `0.0734` (`p=0.2241`) and Spearman `0.0681` (`p=0.2594`); descriptive strata differ but association is weak and nonsignificant |
| Ratio errors are recoverable in MPa | E1--E3 `aggregate_metrics.csv`, `domain_metrics.csv`, and `oof_predictions.csv` | Workbook strength and ratio remain specimen-aligned; conversion is evaluation-only | SUPPORTED: FULL equal-domain MAE `37.2609 MPa`, pooled RMSE `54.1971 MPa`; conversion does not enter selection or gates |
| Dynamic gating is warranted | `e3_complementarity/summary.json` | E3 complementarity must pass first | NO-GO: E4 was not run and no `e4_moe` artifact exists |
| Distributional transport is warranted | `e3_complementarity/summary.json` | Nontrivial E1, confirmed complementarity, and material remaining oracle gap | NO-GO: E5 was not run and no `e5_transport` artifact exists |

Oracle selection is a non-deployable diagnostic. Engineering stress tests are
secondary evidence and do not reverse the primary six-domain NO-GO decisions.
