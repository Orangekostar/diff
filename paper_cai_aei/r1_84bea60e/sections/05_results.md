# 5 Results and discussion

## 5.1 Prediction quality under matched acquisition caps

Task-driven feedback produced lower observed errors than several fixed acquisition rules under the common 25% cap. The main policy achieved a pooled MAE of 44.286 MPa, RMSE of 58.517 MPa and R² of 0.659 on the 50 validation specimens. Geometry-spread achieved 46.910 MPa MAE, giving a paired improvement of 2.624 MPa with a 95% exploratory interval of [−0.488,5.925] MPa. The actual acquired fraction for the main policy averaged 0.247437 and ranged from 0.234492 to 0.249999. The comparison is therefore at a matched cap, with small specimen-dependent differences in the completed native-pixel fractions.

Across the entire acquisition range, the main policy had a six-domain-equal A of 45.110 MPa, compared with 47.311 MPa for geometry-spread. The lower area indicates that its advantage was not confined to the terminal estimate. However, area and terminal quality capture different properties: Random had a slightly lower endpoint MAE than geometry-spread but a higher A. Inspecting only the final state would miss this difference in the quality supplied during acquisition. Figure 2 retains the observed stepwise MAE paths, including reversals, and Supplementary Table S2 reports all five common caps with their actual acquisition ranges.

| Acquisition policy | A (MPa) | Early A (MPa) | MAE at cap (MPa) | RMSE at cap (MPa) | R² at cap |
|------------------------|-------:|-------:|-------:|-------:|-------:|
| Center-first | 48.475 | 51.821 | 48.027 | 65.223 | 0.577 |
| Geometry-spread | 47.311 | 51.880 | 46.910 | 62.278 | 0.614 |
| Serpentine | 48.911 | 52.170 | 48.013 | 63.041 | 0.605 |
| Random | 49.594 | 54.117 | 46.886 | 61.941 | 0.618 |
| Learned-static | 47.977 | 52.946 | 46.993 | 62.168 | 0.616 |
| No-VLM spatial feedback | 43.597 | 48.771 | 42.385 | 56.534 | 0.682 |
| VLM mean feedback | 45.263 | 50.390 | 43.611 | 59.012 | 0.654 |
| VLM spatial feedback (main) | 45.110 | 49.642 | 44.286 | 58.517 | 0.659 |
| VLM spatial open-loop | 46.501 | 51.048 | 44.822 | 59.663 | 0.646 |
| Complete input, fraction 1.0 | — | — | 41.690 | 53.944 | 0.711 |

**Table 3. Common-predictor comparison.** Partial policies use cap 0.25; the complete-input row is a separate cost-1.0 reference. A integrates over 0–0.25 and early A over 0–0.0625, each normalized by its own interval. Areas weight domains equally; endpoint measures pool physical specimens after repeat-loss averaging. All values are selected-validation estimates, with n=50 physical specimens. Actual fraction ranges and full paired results are provided in the supplementary source tables.

![MAE along the recorded acquisition range. Curves are held current-state values on the shared event grid, with no smoothing. The complete-input horizontal reference represents quality measured only at cost 1.0.](figures/Fig2_mae.pdf){width=100%}

## 5.2 From a shared order to evidence-dependent acquisition

The three policies spanning a shared ranking, specimen-conditioned open-loop choice and internal feedback had progressively lower area estimates: 47.977, 46.501 and 45.110 MPa. This sequence connects the experiment to the intended engineering distinction. Learned-static offers one order for every specimen. Open-loop can use surface variation to allocate observations differently, while feedback can revise its allocation after seeing internal content. Their performance pattern is consistent with useful specimen dependence, followed by further improvement when current acquired evidence participates in the next decision.

The main policy improved on open-loop by 1.391 MPa in A and 0.536 MPa in endpoint MAE. The area direction favoured feedback in five of the six domains. The comparison therefore links feedback to a lower observed trajectory loss over much of this cohort, while the interval in Table 4 retains uncertainty about the difference. These adjacent contrasts do not identify percentages of total improvement caused by individual modules: policies were separately trained, used different initializations and visited different states. Their value is to show the consequences of distinct decision-information arrangements under the same assessment model.

| Contrast, control minus main | Measure | Difference (MPa) | 95% exploratory interval (MPa) |
|---|---|---:|---|
| Geometry-spread versus main | A | 2.200 | [−0.787,5.033] |
| Open-loop versus feedback | A | 1.391 | [−0.683,3.406] |
| No-VLM versus VLM | Early A | −0.872 | [−4.870,3.255] |
| Mean versus spatial feedback | A | 0.152 | [−2.186,2.425] |

**Table 4. Component-related comparisons.** Positive values favour the main policy for the stated measure. All four intervals contain zero. Intervals use the existing paired within-domain capture-group bootstrap and are conditional on selected validation checkpoints. Figure 3 displays these same signed contrasts; the VLM contrast uses early A rather than full-range A.

The no-VLM feedback policy obtained lower full-range A and endpoint MAE than the main policy, at 43.597 and 42.385 MPa, respectively. Its surface descriptors were still present, so this result concerns the additional VLM prior rather than the utility of surface information as a whole. Spatial interaction likewise provided a mixed comparison with mean feedback: the main actor's area was lower by 0.152 MPa and its RMSE by 0.495 MPa, but its endpoint MAE was higher by 0.675 MPa. These results support keeping the components conceptually distinct instead of treating architectural complexity as evidence of a uniform performance gain.

![Signed component-related differences and existing 95% pointwise exploratory intervals. Positive differences favour the main policy. The VLM comparison concerns early area; the other contrasts concern full-range area. All four intervals contain zero.](figures/Fig3_contrasts.pdf){width=100%}

## 5.3 Where in the trajectory are gains accumulated?

The saved-event decomposition makes the timing of observed error changes explicit. All methods shared an initial domain-equal error of 57.049 MPa. For the main policy, the timing-weighted contributions grouped by the four acquisition-completion stages were 11.031, 0.960, −0.091 and 0.038 MPa. Their sum, 11.938 MPa, subtracts from the common initial error to recover A=45.110 MPa. Across all 650 episodes, the largest discrepancy between direct integration and this identity was below 3×$10^{-14}$ MPa. Aggregating the contributions reproduced every method's saved area and every contrast with the main policy to floating-point precision.

Although the main policy's largest absolute contribution was assigned to the first stage, its difference from open-loop was concentrated in the second stage. Open-loop contributions were 10.616, −0.336, 0.132 and 0.135 MPa. Thus, the main-minus-open-loop stage differences were approximately 0.415, 1.296, −0.223 and −0.097 MPa, summing to the 1.391 MPa area improvement. The first-stage contribution describes early observations that influence much of the remaining trajectory; it is not an estimate of improvement confined to the first 6.25% interval. The second-stage contrast shows why a blanket explanation based only on a better starting region would be incomplete.

Geometry-spread displayed a different timing pattern, with contributions of 7.143, 2.672, −0.124 and 0.048 MPa. Relative to that rule, the main policy gained more from first-stage completions but less from the second stage. Learned-static also accumulated a substantial first-stage contribution, 9.729 MPa, followed by a negative second-stage contribution of −1.058 MPa. Figure 4 displays all nine policies and retains negative stage totals. These patterns express how their actual prediction changes enter the loss; they do not isolate the mechanical importance of particular cells or prove that a counterfactual swap in acquisition order would have the same effect.

![Timing-weighted error contributions grouped by acquisition completion stage. All nine methods and negative values are retained. Totals use episode sums, specimen repeat means, within-domain means and equal domain weighting, with 50 physical specimens. No new intervals were calculated; these are descriptive algebraic contributions, not causal effects.](figures/Fig4_timing.pdf){width=100%}

The event records also show why useful sequential acquisition should not be equated with improvement at every step. Among 793 main-policy observations, 381 increased absolute error. A newly observed descriptor changes the input of a finite learned regressor and can move an individual estimate away from its label even when the overall acquisition sequence performs well. The signed accounting retains these reversals instead of replacing current predictions with the best historical estimate. This distinction is relevant to interpreting both the area curves and empirical quality crossings.

Three previously fixed specimens, c8-16, q16-29 and q24-48, provide the process illustrations. Their source-domain identifiers and exact image paths are retained in the supplementary case index. The illustrations show the available surface prior, actual first action, acquired regions at selected steps and the corresponding prediction trajectory. The higher-error q24-48 case is included alongside the other two. The main policy's first proposal was genuinely narrower than the environment legal set for 45 of 50 specimens; subsequent actions were selected after the hard first-step restriction had been released. The diagrams document executed choices and visible-state updates, without assigning language reasoning, attention maps or damage-ground-truth explanations to those choices.

![An executed partial-observation state for the preselected c8-16 case after four acquisitions. Grey regions remain unobserved; the outlined next cell and acquired-state overlay are copied unchanged from the frozen case figure. All three fixed cases, including q24-48, are retained in the supplement. Source imagery: Hasebe et al., version-1 dataset, CC BY 4.0; local registered overlay reused unchanged.](figures/74t7kcdgkr_c8-16_measured_4.png){width=100%}

## 5.4 Acquisition requirements at matched empirical quality

An acquisition advantage depends on the target quality and the comparator. At q=46.909917 MPa, the geometry-spread endpoint MAE used as one of the complete anchor set, the main policy first met the target at cap 0.0625 on the five-point grid. Geometry-spread first met it at 0.125, giving a 50% reduction in the earliest supported cap. Learned-static also first met it at 0.0625, giving zero reduction relative to that comparator. Using geometry-spread's nominal 0.25 endpoint as its required cost would overstate the reduction, because that method had already reached the same empirical quality earlier.

The finer event grid resolves earlier crossings but also reveals their instability. For the same target, the earliest main, geometry and static caps were 0.031388, 0.093426 and 0.062407, respectively. These correspond to reductions of 66.40% and 49.70% against geometry and static on that grid. At the main policy's crossing, its actual mean fraction was 0.030861 and MAE was 46.507051 MPa; its curve subsequently rose above the target. We retain this event-grid result in the supplementary analysis, separately from the primary 50% and 0% comparisons, because grid resolution and first-passage reversals materially change its interpretation.

Figure 6 presents the full main-grid target range rather than a single favourable threshold. Supplementary tables retain all integer targets and source-labelled anchors on both grids, including unreached targets and negative or undefined ratios. This representation answers a practical descriptive question: at what supported cap did each population curve first achieve a specified quality? It does not supply an individual stopping certificate. An operational stopping policy would need information available without knowing the specimen's true CAI strength, and the present first-passage calculation does not provide that capability.

![Earliest supported cap at each empirical MAE target on the primary five-cap grid. Missing points denote unreached quality; the finer event-grid results remain separate. The curves summarize group-level first passages, not a specimen stopping policy.](figures/Fig6_quality.pdf){width=100%}

## 5.5 Full-information trade-off and engineering interpretation

The common predictor with complete internal and surface input achieved 41.690 MPa MAE, 53.944 MPa RMSE and R²=0.711. The main policy at the 25% cap therefore retained a 2.596 MPa MAE gap to complete input. None of the nine partial-acquisition policies reached the complete-input MAE anywhere in either observed grid. The full-input point helps locate the quality cost of partial observation under this predictor, but does not define an optimal industrial reference or a missing trajectory between fractions 0.25 and 1.0.

The engineering contribution is the explicit coupling between an assessment objective and the organization of acquired information. A policy can be compared at a shared cap, through its complete observed error trajectory, and at a shared empirical quality target. The agreement or disagreement among these views is informative: a lower area does not ensure a lower terminal MAE, and one threshold can show a reduction against geometry while showing none against a learned shared order. The framework makes those distinctions measurable rather than expressing inspection efficiency as a single percentage detached from a quality requirement.

Domain-level results further bound the comparison. Against geometry-spread, the main policy's area difference favoured it in three domains and favoured the control in three. Against open-loop, five domain differences favoured feedback. This heterogeneity is compatible with acquisition decisions depending on specimen evidence, but it also leaves room for differences in predictor quality, source imaging and the distribution of specimens across domains. The reported data do not determine a material-specific causal explanation. The supplementary domain table preserves these directions without treating the six domains as six independent replications of the entire learning experiment.

The current evidence supports an offline image-acquisition method and a quality–acquisition analysis on a selected validation cohort. Physical deployment remains a separate question because native image pixels omit probe travel, coupling, setup and measurement latency; a 25% image fraction cannot be translated directly into a 75% time saving. A single policy-seed panel and validation-based selection also limit inference about reproducibility and transport to new specimens. Independent specimen evaluation and instrument-linked acquisition costs would resolve those specific uncertainties. Within the present scope, the method supplies a reproducible way to examine which information is acquired, when it affects CAI prediction and where partial observation leaves a remaining quality gap.
