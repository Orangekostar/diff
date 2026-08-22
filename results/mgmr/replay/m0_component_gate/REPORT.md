# MGMR M0 Component Gate

Status: `MGMR_NO_GO`

## Direct models

| Method | Equal-domain MAE | Worst-domain MAE |
|---|---:|---:|
| B0 | 0.089635804658 | 0.127430743874 |
| B1 | 0.120742737488 | 0.171712010446 |
| B2 | 0.123837657140 | 0.166288493389 |
| B3 | 0.123962237568 | 0.167445788931 |
| B4 | 0.092414614249 | 0.135316841286 |

## Residual audit

| Method | Equal-domain MAE | Residual Pearson | Residual Spearman |
|---|---:|---:|---:|
| R_coarse | 0.121767985577 | 0.141189046899 | 0.116176995191 |
| R_full | 0.092419622617 | 0.051704941821 | 0.074283757366 |
| P3_20260831 | 0.123358084424 | 0.009933242721 | 0.027023586992 |
| P3_20260901 | 0.121119172785 | 0.141325706699 | 0.143003667185 |
| P3_20260902 | 0.121975483437 | 0.086227754512 | 0.070636549136 |

## Gate

- Gate A: FAIL
- Gate B: FAIL
- Gate C: FAIL
- Gate D: FAIL

All six domains were exposed in earlier project phases; this is a registered post-hoc follow-up, not untouched external confirmation.

M1 remains blocked by the frozen stop rule.
