# Evidence-Bounded Claim Sentence Bank

Every numerical sentence below is bound to
`artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv`. Wording may be
compressed in the manuscript, but its direction, evidence type, uncertainty,
domain consistency, and deployability qualifier must be preserved.

## Global definitions

- Use **downstream-predictor-conditioned task value**, not intrinsic or
  universal mechanical value.
- Use **retrospective non-deployable teacher/oracle** whenever future outcomes,
  unrevealed measurements, or counterfactual utilities are used.
- Acquisition percentage denotes the exact or registered native-raster sampling
  fraction specified by the experiment; it does not denote scanner time.
- Statistical inference is specimen-first and then equal-domain aggregated;
  state-action and checkpoint rows are repeated computational records.
- AUEBC is the budget-span-normalized trapezoidal mean error over the observed
  actual/effective specimen-budget range; lower is better.

## Review-fix boundaries

- Chronology: historical comparisons and the frozen outer endpoint predate the
  post-freeze diagnostics; later diagnostics reused hash-bound frozen states
  and outcomes and did not modify the endpoint.
- Novelty: the contribution connects Information Characterization and
  Evidence-Calibrated Decision Realization under one causal acquisition
  contract. Usefulness, task-value observability, and actionability remain
  validation criteria. Do not claim the first adaptive ultrasonic inspection,
  ultrasound VoI, or task-driven design.
- Transfer: framework reuse requires a defined endpoint/loss, a legal partial
  state, a defensible marginal-value target, matched content/geometry/history
  controls, and an end-to-end cost-constrained metric distinct from local value
  prediction. These conditions are not external validation.
- External feasibility: the current route is `MANUSCRIPT_ONLY_PRIMARY` and the
  optional external micro-pilot is `EXTERNAL_MICRO_PILOT_NO_GO`.

## Part I --- Information Characterization

### U1_MATCHED_FIELD

Registered sentence: Under strict nested LODO evaluation, the matched B-family
spatial field reduced equal-domain CAI-ratio MAE from 0.18920 to 0.12849, a
reduction of 0.06072 (32.1%; familywise 95% CI 0.00664--0.15364), with lower
error in five of six held-out domains.

Interpretation: Full spatial internal morphology preserves CAI-relevant
information absent from the matched scalar representation.

Boundary: `B_field_selected`, not `I_field_selected`, is the registered matched
confirmatory estimator.

### U1_SURFACE_FIELD

Registered sentence: Relative to metadata and surface statistics, the selected
B-family spatial field reduced equal-domain CAI-ratio MAE by 0.05963 (31.7%;
familywise 95% CI 0.00722--0.14842) and improved five of six held-out domains.

Interpretation: Measured spatial internal information adds CAI-relevant signal
beyond the surface representation; this does not imply surface-to-field
reconstruction.

### U1_INDEPENDENT_FIELD_SENSITIVITY

Appendix-only sentence: A distinct independent internal-field estimator had a
point MAE of 0.08964, but it is a sensitivity path rather than the matched
B-family confirmatory estimator.

### U2_SPARSE_RETENTION / U2_SPARSE_GAIN / U2_SPARSE_FULL_GAP

Registered sentence: The selected 25% bilinear sparse condition reduced MAE
relative to the surface reference by 0.05361 (95% CI 0.00561--0.13544), improved
five of six domains, and retained 89.9% of the registered full-field gain.

Boundary sentence: Its MAE remained 0.00602 above the selected full-field
reference (95% CI 0.00173--0.01083), with the sparse condition above the
full-field error in all six domains.

Interpretation: Sparse measurements retain most, but not all, of the registered
full-field gain under the normalized-raster protocol; no scanner-time reduction
is established.

### U3_UNIFORM_ORACLE / U3_RECONSTRUCTION_ORACLE / U3_HEADROOM_RETENTION

Registered sentence: A retrospective non-deployable one-shot mechanical oracle
reduced CAI AUEBC by 0.00391 relative to uniform acquisition (95% CI
0.00280--0.00502) and by 0.00373 relative to a reconstruction oracle (95% CI
0.00284--0.00469), with both contrasts favorable in all six held-out domains.

Supporting sentence: The one-shot oracle retained 56.8% of the registered
sequential-oracle headroom.

Interpretation: Specimen-specific acquisition headroom exists retrospectively;
the result does not identify a deployable policy.

### U4_ORACLE_CAI_SPECIFICITY / U4_ORACLE_IMAGE_SPECIFICITY

Registered sentence: Cross-objective retrospective oracles were task-specific:
the mechanics oracle improved CAI AUEBC relative to the reconstruction oracle
by 0.04862 (95% CI 0.04527--0.05205), whereas the reconstruction oracle improved
normalized RGB reconstruction MSE by 5.503e-4 (95% CI 5.006e-4--6.063e-4).

### U4_LEARNED_SPECIFICITY_BOUNDARY

Boundary sentence: Source-trained global mechanics and reconstruction masks
produced a support indicator of 0; the learned global mechanics mask does not
reproduce the oracle task separation.

Interpretation: Oracle task specificity does not establish learned-policy task
specificity or deployment value.

### O2_TEACHER_TURNOVER / O2_TEACHER_RANK / O2_TEACHER_TOPK / O2_TEACHER_OPPORTUNITY

Registered sentence: Between the initial state and the 18.75% checkpoint, the
strict-OOF retrospective teacher changed its best action for 70.4% of specimens;
rank agreement with initial action values was 0.405, top-five overlap was 0.307,
and descriptive opportunity was 0.00531.

Interpretation: Conditional measurement value evolves materially with acquired
evidence, although this does not show that a deployable scorer tracks that
evolution.

### U5_RIDGE_HUBER_SPEARMAN / U5_RIDGE_HUBER_BEST_ACTION / U5_RIDGE_HUBER_TOPK / U5_RIDGE_MLP_SPEARMAN / U5_RIDGE_MLP_BEST_ACTION / U5_RIDGE_MLP_TOPK

Registered sentence: Strict-OOF action-value rankings agreed substantially
between ridge and Huber predictors (Spearman rho 0.762; 95% CI 0.699--0.821) but
weakly between ridge and a shallow MLP (rho 0.116; 95% CI 0.069--0.164).

Accuracy boundary: Full-state equal-domain strict-OOF MAE was 0.08964 for Ridge,
0.08618 for Huber, and 0.15067 for the shallow MLP.

Interpretation: Measurement value was relatively stable between the comparably
performing low-complexity Ridge and Huber predictors but did not persist under
the substantially less accurate shallow MLP. Retain the predictor index $f$.
The experiment does not determine variation among equally accurate but
structurally distinct predictors.

## Part II --- Evidence-Calibrated Decision Realization

### O1_STATIC_SPEARMAN

Registered sentence: Static pre-acquisition scores were effectively
uncorrelated with strict-OOF teacher values across held-out domains (Spearman
rho -0.0196; 95% CI -0.0591--0.0195), with a favorable direction in three of six
domains.

Control sentence: Exact-budget set regret was 0.08171 for the static scorer,
compared with 0.07993 for the global score and 0.07978 for the random median.

Interpretation: The registered static representation did not establish
transferable specimen-specific task-value observability; this is not a claim
of information-theoretic impossibility.

### O3_REAL_CHANGE / O3_FULL_FIELD_RECOVERY

Supporting sentence: Real partial measurements changed the prediction relative
to the static initial state, and an endpoint diagnostic recovered part of the
static-to-independent-full-field error ratio.

Boundary: The recovery endpoint is the independent `I_field_selected`
sensitivity estimator, not the registered matched B-family full-field path.

### O3_REAL_MINUS_POSITIONS / O3_REAL_MINUS_RECONSTRUCTION

Boundary sentence: The real-content representation had MAE 0.0174 above the
acquired-position/history control (95% CI 0.00890--0.0258) and 0.0342 above the
registered normalized-RGB-MSE reconstruction control (95% CI
0.0245--0.0438); measured content was favorable in only one of six domains for
each contrast. The positions-only input inherits acquisition history and is not
a pure geometry-only control.

Interpretation: Real measurements change prediction from the initial state,
while the registered representation does not establish specimen-specific value
from measured content beyond the matched controls.

### O4_DYNAMIC_MINUS_STATIC

Registered sentence: At the registered endpoint, the conditional real-state
scorer reduced regret relative to the static scorer by 0.00126 (95% CI
0.000444--0.00212), favorable in five of six domains.

### O4_DYNAMIC_MINUS_SHUFFLED

Boundary sentence: Dynamic real minus shuffled-content regret was 2.328e-4
(95% CI 6.060e-5--4.180e-4), favorable for real content in one of six domains.

Interpretation: Dynamic scoring captures incremental state dependence relative
to the static reference; the shuffled control prevents attribution of that
advantage to specimen-specific accumulated ultrasonic content.

### Decision realization evidence

### A1_VALUATION_SUBSTITUTION / A1_LEARNED_PLANNING_SUBSTITUTION / A1_TRUE_VALUE_PLANNING_SUBSTITUTION

Registered sentence: Retrospective component substitutions identified a
valuation effect of 4.979e-5 (95% CI 1.845e-5--8.105e-5), a learned-planning
effect of 3.164e-6 (95% CI 1.411e-7--6.250e-6), and a true-value planning effect
of 1.117e-4 (95% CI 9.898e-5--1.248e-4).

Interpretation: Valuation and bounded planning gaps are distinguishable under
retrospective substitutions; these tests neither establish a representation
bottleneck nor define deployable components.

### A2_GREEDY_PLANNING_REGRET / A2_BEAM4_PLANNING_REGRET

Registered sentence: Within the registered reachable-pool diagnostic, current
greedy selection had regret of 1.207e-4 relative to a retrospective joint
near-oracle set (95% CI 1.033e-4--1.386e-4).

Interpretation: A bounded two-action set-planning gap remains; the near-oracle
is non-deployable and does not establish arbitrary-horizon regret.

### A3_FEEDBACK_BENEFIT

Boundary sentence: The no-feedback reference retained an AUEBC advantage of
1.496e-5 (95% CI 1.064e-5--1.944e-5). Equivalently, feedback benefit was
-1.496e-5 (95% CI -1.944e-5 to -1.064e-5), with feedback favorable in two of
six held-out domains.

Interpretation: The frozen stress test retains the simpler no-feedback
reference; the direction is not generalized to other implementations.

### A4_BASELINE_MINUS_MAVIS

Registered sentence: The frozen learned policy had CAI AUEBC 0.125053, compared
with 0.124992 for the strongest deployable baseline; baseline minus learned
policy was -6.114e-5 (95% CI -8.461e-5 to -3.777e-5), and the learned policy
improved two of six held-out domains.

Interpretation: A residual deployable gap of 6.114e-5 remains in favor of the
reference; the learned endpoint is not performance-superior.

## Integrated claim

Main synthesis: Part I characterizes structured task-relevant information from
spatial morphology through state evolution. Part II tests its realization with
static and dynamic valuation, matched information-source controls, component
substitutions, bounded planning, feedback, and the frozen outer comparison.

## Prohibited manuscript wording

Do not claim superior learned acquisition, most-domain improvement,
specimen-specific mechanical-state capture, real-content superiority over
matched controls, beneficial feedback, a universal mechanical-value map,
scanner-time reduction, industrial deployment, external empirical
generalization, or any unmeasured causal failure mechanism.
