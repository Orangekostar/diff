# 5 Results and discussion

## 5.1 Prediction quality under matched acquisition caps

Under the common 25% cap, C spatial feedback achieved a pooled MAE of 43.711 MPa, RMSE of 57.756 MPa and R2 of 0.668 on 50 selected validation specimens. The best non-adaptive strategy by six-domain-equal area was Geometry-spread, with endpoint MAE 46.910 MPa. The main policy's endpoint MAE was lower by 3.199 MPa, with a 95% exploratory paired interval of [-0.359, 6.656] MPa. This fixed-cohort interval quantifies uncertainty but does not undo validation-based checkpoint selection.

Across the full observed range, C spatial feedback had A=45.847 MPa and early A=50.339 MPa. Geometry-spread had A=47.311 MPa. Area and endpoint error need not rank strategies identically because A weights each improvement by how long it remains available. Figure 2 preserves all nine held-error paths, including reversals, and the source tables report actual acquired fractions and ranges at each cap.

| Acquisition policy | A (MPa) | Early A (MPa) | MAE at cap (MPa) | RMSE at cap (MPa) | R2 at cap |
|---|---:|---:|---:|---:|---:|
| Center-first | 48.475 | 51.821 | 48.027 | 65.223 | 0.577 |
| Geometry-spread | 47.311 | 51.880 | 46.910 | 62.278 | 0.614 |
| Serpentine | 48.911 | 52.170 | 48.013 | 63.041 | 0.605 |
| Random | 49.594 | 54.117 | 46.886 | 61.941 | 0.618 |
| Learned-static | 47.977 | 52.946 | 46.993 | 62.168 | 0.616 |
| No-VLM spatial feedback | 43.597 | 48.771 | 42.385 | 56.534 | 0.682 |
| C mean feedback | 44.926 | 48.657 | 44.066 | 60.593 | 0.635 |
| C spatial feedback (main) | 45.847 | 50.339 | 43.711 | 57.756 | 0.668 |
| C spatial open-loop | 44.907 | 49.154 | 43.917 | 58.523 | 0.659 |
| Complete input, fraction 1.0 | NA | NA | 41.690 | 53.944 | 0.711 |

**Table 3. Current C common-predictor comparison.** Partial policies use cap 0.25. Complete input is a separate observed point at cost 1.0 and has no partial-range area. Areas weight domains equally; endpoint metrics pool physical specimens after within-specimen repeat-loss averaging.

![Current C MAE and RMSE along the five frozen acquisition caps. Curves retain observed reversals and use no smoothing.](figures/F1_cost_error_curves.pdf){width=100%}

## 5.2 Feedback, VLM prior and spatial interaction

Against Geometry-spread, the main C policy had lower error by 1.464 MPa for area_mpa; the 95% exploratory interval was [-2.088, 4.691] MPa and included zero. Against C spatial open-loop, the main C policy had higher error by 0.940 MPa for area_mpa; the 95% exploratory interval was [-2.868, 0.890] MPa and included zero. Against no-VLM spatial feedback, the main C policy had higher error by 1.568 MPa for early_area_mpa; the 95% exploratory interval was [-5.397, 2.554] MPa and included zero. Against C mean feedback, the main C policy had higher error by 0.921 MPa for area_mpa; the 95% exploratory interval was [-2.484, 0.636] MPa and included zero.

| Contrast | Comparator | Measure | Control - C main (MPa) | 95% exploratory interval |
|---|---:|---:|---:|---:|
| best_nonadaptive | Geometry-spread | area_mpa | 1.464 | [-2.088, 4.691] |
| feedback | C spatial open-loop | area_mpa | -0.940 | [-2.868, 0.890] |
| vlm_early | No-VLM spatial feedback | early_area_mpa | -1.568 | [-5.397, 2.554] |
| spatial_vs_mean | C mean feedback | area_mpa | -0.921 | [-2.484, 0.636] |

**Table 4. Prespecified C contrasts.** Positive differences favour C spatial feedback for the stated measure. Intervals use 5000 paired capture-group bootstrap draws within domains and remain conditional on the selected checkpoints.

The component comparisons are descriptive rather than causal allocations. The policies were trained separately and can visit different states. A negative estimate is retained as evidence against a uniform benefit from the added component. Figure 3 reports the endpoint and area contrasts without filtering by direction.

![Paired endpoint and trajectory-area contrasts. Positive values favour current C spatial feedback; all signs and intervals are retained.](figures/F2_paired_effects.pdf){width=100%}

## 5.3 What changed from historical A to current C

Current C changes only the numbered rendering and the three retrained VLM policies; P0, predictors, feature bank, cost definition and six primary controls remain fixed. For C spatial feedback (main), historical A minus current C was -0.736 MPa ([-2.946, 1.455]). For C spatial open-loop, historical A minus current C was 1.594 MPa ([-0.500, 3.720]). For C mean feedback, historical A minus current C was 0.337 MPa ([-2.208, 2.988]). These comparisons isolate the observed version change within each policy name but remain selected-validation comparisons rather than a second independent experiment.

Domain-level A-minus-C values varied across the six source domains. This heterogeneity is retained in Figure 4 and the source table. The data do not identify a material-specific cause because domain, image source and predictor quality are not experimentally separated.

![Endpoint performance by domain and historical-A-minus-current-C trajectory area for the three retrained policies.](figures/F5_domain_results.pdf){width=100%}

## 5.4 Timing and executed acquisition cases

The largest signed main-minus-open-loop timing difference occurred in completion stage 2, where it was 1.200 MPa. The four stage differences were -2.109, +1.200, -0.031, +0.000 MPa. These terms satisfy A=e0-sum(g) and preserve negative contributions. They describe when saved prediction changes affected the remaining budget interval, not a causal percentage attributable to feedback.

Individual acquisitions did not guarantee monotonic improvement. 369 of 794 current main-policy events increased absolute prediction error. This adverse-event count is compatible with a favourable overall area when earlier or larger reductions dominate. Figure 5 shows all signed stage contributions and the identity components.

![Signed timing-weighted contributions for all nine methods. Negative values and unequal initial-error terms are retained.](figures/F4_timing_contributions.pdf){width=100%}

The three prespecified cases were regenerated from the selected current C main-policy trajectories and current C priors. Surface candidates, the real first action, measured states after 1, 4 and 8 acquisitions, the endpoint and prediction process are new outputs. The measured set grows monotonically on the strict 8x8 grid, while predictions may reverse.

![Current C acquisition progression for the three fixed validation cases. Panels show the C prior, acquired states and saved prediction paths.](figures/F6_case_progression.pdf){width=100%}

## 5.5 Equal-quality and full-input boundaries

Equal-quality comparisons use the earliest observed population-curve crossing on both the five-cap grid and the union of saved event costs. Unreached targets, negative savings, zero denominators and later recrossings remain in the source tables. Figure 7 displays the complete integer target range rather than selecting a favourable threshold. These population first passages do not define a label-free stopping policy for individual specimens.

![Earliest observed costs and current-main savings across the complete empirical MAE target grid.](figures/F3_equal_quality.pdf){width=100%}

The complete-input predictor achieved MAE 41.690 MPa, RMSE 53.944 MPa and R2 0.711 at cost 1.0. Current C spatial feedback retained an endpoint MAE gap of 2.021 MPa at the 25% cap. No unobserved segment between 0.25 and 1.0 is interpolated. The comparison therefore bounds the observed image-acquisition trade-off without converting native-pixel fraction into physical inspection time.

Taken together, the current evidence supports an offline task-driven acquisition analysis on a selected validation cohort. It does not establish independent generalization, causal component shares or deployment-time savings. Those questions require new specimens, repeated policy seeds and a physical acquisition-cost model.
