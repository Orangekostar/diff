# MAVIS P10 MRIS Causal Informativeness Closure

Status: `COMPLETE`.

All rows reuse frozen nested-LODO predictions. No representation or CAI
model is retrained. Effects are `real - control` CAI MAE, so negative
values favor specimen-specific real ultrasonic content.

| Checkpoint | Real MAE | Positions MAE | Shuffled MAE | Static MAE | Reconstruction MAE |
|---:|---:|---:|---:|---:|---:|
| 0.03125 | 0.125357 | 0.107456 | 0.128800 | 0.137276 | 0.089058 |
| 0.06250 | 0.125266 | 0.107265 | 0.129025 | 0.137276 | 0.086829 |
| 0.09375 | 0.125201 | 0.107200 | 0.129090 | 0.137276 | 0.087141 |
| 0.12500 | 0.125114 | 0.107170 | 0.129143 | 0.137276 | 0.086925 |
| 0.18750 | 0.124874 | 0.107169 | 0.129217 | 0.137276 | 0.087181 |
| 0.25000 | 0.124626 | 0.107223 | 0.129259 | 0.137276 | 0.090435 |

Specimen-specific content beyond geometry is **not supported** under the
predeclared paired contrasts. Real MRIS beats the static initial state,
but it does not beat positions-only at the registered checkpoints.

Real-state CAI error decreases from the first to final checkpoint.
The final partial state recovers `0.265521` of the static-to-full-field mechanical utility under the registered
ratio; its paired interval is reported in `bootstrap.csv`.

Metrics first average trajectories within each physical specimen, then
weight the six held-out domains equally. The full-field reference is the
hash-bound source-only `I_field_selected` prediction. This state-utility
recovery is not policy oracle-gap recovery and does not alter P7 Tier B.
