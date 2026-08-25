# MAVIS P0 Frozen Evidence Ledger

All entries were re-read from current machine-readable artifacts. Historical
packages are evidence inputs and are not subject to MAVIS model selection.

| Evidence | Current artifact | Re-verified result | Interpretation |
|---|---|---|---|
| Scalar bottleneck, measured internal | `results/g2_scalar_utility/metrics.json` | surface MAE `0.1881207811`; measured internal `0.1892044107`; relative effect `-0.0057603`; improved `4/6`; simultaneous lower `-0.00506965` | A scalar internal summary did not improve the registered endpoint. |
| Scalar bottleneck, predicted internal | same | MAE `0.1855983291`; relative effect `0.0134087`; improved `3/6`; simultaneous lower `-0.0137375` | Predicted scalar evidence is not stable across domains. |
| Full-field mechanical information | `analysis_tables/extended_stage_gate_summary.csv` | reference `0.1881207811`; registered full-field `0.1284893565`; relative gain `0.316985`; improved `5/6`; lower `0.007218` | Spatial internal NDE contains transferable mechanical information. |
| Historical dense FULL estimator | `docs/MVA_CLAIM_EVIDENCE_MATRIX.md` | MAE `0.0896358047` | Different estimator from the registered P1 comparison; do not conflate the values. |
| P5 sparse retention | `analysis_tables/extended_stage_gate_summary.csv` | 25% nominal budget; MAE `0.1345137120`; retention `0.898973`; improved `5/6`; lower `0.001730069` | Sparse real C-scan measurements retain most registered full-field gain. |
| P5 cost/cohort | `docs/MVA_ACQUISITION_SEMANTICS_AUDIT.md` | unique observed native-raster locations / native count; 276 specimens, 6 domains, nested LODO | Physical scanner-time equivalence is not established. |
| MVA oracle headroom | `results/mva/a2_oracle_value/budget_metrics.csv` | sequential mechanical oracle AUEBC `0.0106245258`; uniform `0.0173634580`; relative improvement `0.3881100` | Measurement locations have unequal task value. |
| Static/adaptive limitation | `results/mva/a4_global_task_mask/budget_metrics.csv`, `results/mva/a5_imitation_policy/budget_metrics.csv` | global mechanical `0.0176388913`; imitation `0.0170922299`; A5 oracle-gap closure `0.07793455` | Existing deployable static/adaptive formulations recover little oracle headroom. |
| MVD M0 | `results/mvd/m0_one_shot_oracle/summary.json` | one-shot mechanical `0.0134579441`; uniform `0.0173634580`; reconstruction `0.0171876604`; sequential oracle `0.0106245258`; both baselines improved `6/6`; headroom retention `0.5682828` | A specimen-specific initial plan has substantial privileged headroom. |
| MVD interaction | `results/mvd/m0_one_shot_oracle/interaction_audit.csv` | 27,600 rows, 1,380 source specimen-folds; checkpoint Pearson values are not consistently additive | Individual action value cannot be assumed additive. |
| MVD M1 static observability | `results/mvd/m1_observability/summary.json` | O2 Spearman `-0.0195862`, CI `[-0.0590603, 0.0194778]`; exact-budget regret `0.0817054` vs global `0.0799269` and random `0.0797831` | Registered static pre-acquisition state did not establish transferable value observability. |
| MVD M1 downstream capture | same | predicted AUEBC `0.0172545220` vs reconstruction `0.0171876604`; advantage capture `-0.0179267`; improved `2/6` | Motivates state-dependent feedback; it does not prove impossibility. |

## Integrity verification

- MVA A2/A4/A5 and MVD M0/M1 formal checksums: PASS.
- MVD M0 and M1 formal/replay directory comparisons: byte-identical.
- Current focused MVD/external suite: `32 passed`.
- Historical MVA baseline/replay/A4 suite in the complete project: `5 passed`.
- Ruff and `git diff --check`: PASS.
