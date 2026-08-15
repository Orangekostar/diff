# P2 Mechanical Utility

- Specimens: 276; held-out datasets: 6.
- Primary response: published damaged-to-intact CAI strength ratio (unit 1).
- Secondary response: CAI strength (unit MPa).
- Primary metric: raw equal-domain MAE on the source-defined dimensionless ratio.
- Secondary sensitivity: outer-train standardized equal-domain MAE; it does not determine G2.
- Z0: registered mechanics metadata plus surface profile statistics.
- Internal primary block: area, height, width; secondary block: six physical descriptors.
- All pathways use fold-local StandardScaler and Ridge(alpha=10); C uses strict double-OOF predictions; D shuffles outer-training rows only.
- C predictor contract: `p1_nested_selected_surface_v1`; P1 selection is `frozen_p1_nested_selected_surface`.
- C predictor candidates: `profile_ridge, profile_extra_trees, frozen_rgb, combined`; nested model selection and provenance are recorded per fit call.
- Bootstrap intervals use Bonferroni familywise size three across B>A, C>A, and C>D.
- Internal descriptor units are recorded separately from response metric units in predictions.csv.

## Four pathways

| Pathway | Raw ratio equal-domain MAE | Standardized sensitivity |
|---|---:|---:|
| A_surface_only | 0.18812078 | 0.87087739 |
| B_measured_internal | 0.18920441 | 0.87556529 |
| C_predicted_internal | 0.18559833 | 0.86065627 |
| D_train_deranged_internal | 0.18808914 | 0.87143179 |

## G2

- B>A: FAIL
- C>A: FAIL
- C>D paired interval: FAIL
- D same surface-only gate: false
- Overall G2: FAIL
- Oracle gap recovery: undefined (undefined; MAE_A-MAE_B <= 0)

## Secondary MPa response

- Overall gate status is reported descriptively as FAIL; it does not replace the ratio gate.
