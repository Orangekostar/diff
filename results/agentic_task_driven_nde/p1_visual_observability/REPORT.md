# P1 Visual Observability

Status: P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO
Authorized route: NONE
Specimens: 276
Outer domains: 6
Bootstrap resamples: 100000

## Equal-domain metrics

| method | CAI AUEBC | next-action regret | NDCG@10 |
|---|---:|---:|---:|
| c0_mvd_m1_o2 | 0.092024117 | 0.0150831566 | 0.687688454 |
| c1_center_prior | 0.092977501 | 0.0158492209 | 0.706585261 |
| c2_global_context | 0.0921868626 | 0.0154951808 | 0.69671421 |
| c3_shuffled_global | 0.0921522662 | 0.0154951808 | 0.696697124 |
| c3_shuffled_surface | 0.092024117 | 0.0150831566 | 0.687688454 |
| c4_wrong_orientation | 0.092024117 | 0.0150831566 | 0.687688454 |
| c5_spatial_derangement | 0.0920241156 | 0.0150831566 | 0.687688454 |
| mechanical_oracle_diagnostic | 0.0800481106 | 0 | 1 |
| old_refit_diagnostic | 0.0922049319 | 0.0146005706 | 0.705858043 |
| proposed | 0.0920241159 | 0.0150831566 | 0.687688454 |

Target-domain scores were frozen before target mechanical labels were read.
All selection used source domains only. CAI uses exact native-raster cost.
