# MAVIS Manuscript Evidence Map

This map is established before model development. Rows move from `PLANNED` to a
machine-readable result reference; unsupported claims remain absent.

| Manuscript move | Evidence | Current status | Authoritative artifact |
|---|---|---|---|
| Scalar summaries are insufficient | measured/predicted scalar CAI comparisons | FROZEN NEGATIVE | `results/g2_scalar_utility/metrics.json` |
| Full spatial C-scan carries mechanical information | registered full-field nested-LODO gain | FROZEN POSITIVE | `analysis_tables/extended_stage_gate_summary.csv` |
| Sparse measurements retain information | P5 25% exact-raster sparse result | FROZEN POSITIVE | `analysis_tables/extended_stage_gate_summary.csv` |
| Measurement locations have unequal task value | sequential mechanical oracle vs baselines | FROZEN POSITIVE | `results/mva/a2_oracle_value/` |
| Initial one-shot planning has headroom | MVD M0 | FROZEN POSITIVE | `results/mvd/m0_one_shot_oracle/` |
| Static value observability is inadequate | MVD M1 O2 | FROZEN NEGATIVE | `results/mvd/m1_observability/` |
| Acquired content enriches a mechanics state | real vs static/positions/shuffled/reconstruction CAI curves | PLANNED E2/E7 | `results/mavis/p2_mris/` |
| Conditional dynamic value beats static value | next-action regret and one-step utility | PLANNED E3 | `results/mavis/p3_dynamic_voi/` |
| Feedback changes useful acquisition decisions | feedback vs post-scout frozen ranking | PLANNED E4/E6 | `results/mavis/p4_closed_loop/` |
| Mechanics-driven acquisition differs from reconstruction | paired CAI and reconstruction curves | PLANNED E5 | `results/mavis/p4_closed_loop/` |
| A safe system controls heterogeneous risk | risk-coverage and fallback frequency | PLANNED E9 | `results/mavis/p5_safe_policy/` |
| Final claim tier S/A/B | one frozen nested-LODO evaluation and bootstrap | LOCKED UNTIL FINAL | `results/mavis/p6_final_frozen_eval/` |

All final tables and figures must be regenerated from the referenced
machine-readable package. No development result may be promoted to final frozen
evidence.
