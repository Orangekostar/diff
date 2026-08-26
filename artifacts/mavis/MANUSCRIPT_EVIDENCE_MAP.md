# MAVIS Manuscript Evidence Map

This map was established before model development and is now frozen to the
machine-readable P7 evaluation. Unsupported positive claims are marked rather
than promoted from development results.

| Manuscript move | Evidence | Current status | Authoritative artifact |
|---|---|---|---|
| Scalar summaries are insufficient | measured/predicted scalar CAI comparisons | FROZEN NEGATIVE | `results/g2_scalar_utility/metrics.json` |
| Full spatial C-scan carries mechanical information | registered full-field nested-LODO gain | FROZEN POSITIVE | `analysis_tables/extended_stage_gate_summary.csv` |
| Sparse measurements retain information | P5 25% exact-raster sparse result | FROZEN POSITIVE | `analysis_tables/extended_stage_gate_summary.csv` |
| Measurement locations have unequal task value | sequential mechanical oracle vs baselines | FROZEN POSITIVE | `results/mva/a2_oracle_value/` |
| Initial one-shot planning has headroom | MVD M0 | FROZEN POSITIVE | `results/mvd/m0_one_shot_oracle/` |
| Static value observability is inadequate | MVD M1 O2 | FROZEN NEGATIVE | `results/mvd/m1_observability/` |
| Acquired content enriches a mechanics state | real vs static/positions/shuffled/reconstruction CAI curves | FROZEN MIXED; broad enrichment claim unsupported because positions-only and reconstruction have lower error | `results/mavis/p2_mris/` |
| Conditional dynamic value beats static value | next-action regret and one-step utility | FROZEN NOT SUPPORTED; static is statistically indistinguishable and shuffled content has lower regret | `results/mavis/p3_dynamic_voi/` |
| Feedback improves useful acquisition decisions | feedback vs post-scout frozen ranking | FROZEN NEGATIVE; no-feedback has lower CAI AUEBC with a CI excluding zero | `results/mavis/p4_closed_loop/` |
| Mechanics-driven acquisition improves over reconstruction | paired CAI and reconstruction curves | FROZEN INCONCLUSIVE; paired CI crosses zero | `results/mavis/p4_closed_loop/` |
| Source-only aggregation improves closed-loop MAVIS | before vs after aggregation | FROZEN PARTIAL; aggregation lowers MAVIS AUEBC but remains worse than the strongest deployable baseline | `results/mavis/p7_final_frozen_eval/claim_evidence.csv` |
| A safe system controls heterogeneous risk | risk-coverage, routing, fallback, and paired bootstrap | FROZEN NOT ESTABLISHED; safe and source-selected fallback are statistically indistinguishable | `results/mavis/p7_final_frozen_eval/` |
| Final claim tier S/A/B | one frozen outer evaluation and paired bootstrap | FROZEN TIER B; conservative ceiling, with missing positive subclaims left unclaimed | `results/mavis/p7_final_frozen_eval/` |
| Final package is deterministically reproducible | independent package regeneration and byte comparison | FROZEN POSITIVE; 31/31 files byte-identical | `results/mavis/replay/` |
| External generalization or scanner-time reduction | external method-performance evaluation | NOT PERFORMED AND NOT CLAIMED | `results/mavis/p7_final_frozen_eval/external_data_audit.json` |

All final tables and figures are generated from the referenced machine-readable
package. Development results are retained for diagnosis and are not promoted
when the frozen outer evaluation does not support them.
