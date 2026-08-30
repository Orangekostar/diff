# Evidence-Bounded Claim Sentence Bank

Numerical authority is
`artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv`; placement
authority is `PAPER_CLAIM_VISIBILITY_MAP.csv`. Wording may be compressed, but
direction, uncertainty, scope, chronology, and deployability qualifiers must
remain unchanged.

## Global wording rules

- Use downstream-predictor-conditioned task value, not an intrinsic location
  value.
- Oracles and strict-OOF teachers are retrospective and non-deployable.
- Acquisition percentage is registered native-raster fraction, not scanner
  time.
- Physical specimens are the first statistical unit; six domain estimates are
  equally weighted.
- AUEBC is the budget-span-normalized trapezoidal mean error over observed
  effective specimen budgets; lower is better.
- The preregistered task-agnostic C-scan appearance-saliency reference is the
  mean absolute RGB deviation of newly revealed samples from the
  specimen-specific border median, normalized by 255; it uses no CAI outcome
  and is retrospective/non-deployable.
- Legacy reconstruction claim IDs and control modes remain supplement-only and
  must not be relabeled as appearance saliency.
- Main text contains headline/support claims, one concise system diagnostic,
  and no implementation identity.

## I-A: Spatial information and sparse retention

### U1_MATCHED_FIELD

Main: The matched B-family spatial field reduced equal-domain CAI-ratio MAE
from 0.18920 to 0.12849, a reduction of 0.06072 (32.1%; familywise 95% CI
0.00664--0.15364), favorable in five of six held-out domains.

### U1_SURFACE_FIELD

Main support: Relative to metadata and surface statistics, the selected field
reduced MAE by 0.05963 (31.7%; familywise 95% CI 0.00722--0.14842), again
favorable in five domains.

### U1_INDEPENDENT_FIELD_SENSITIVITY

Supplement: The independent internal-field estimator had point MAE 0.08964;
it is a sensitivity path, not the matched confirmatory estimator.

### U2_SPARSE_RETENTION

Main: The registered 25% bilinear sparse condition retained 89.9% of the
matched full-field gain.

### U2_SPARSE_GAIN

Main support: Sparse observation reduced MAE relative to the surface reference
by 0.05361 (95% CI 0.00561--0.13544), favorable in five domains.

### U2_SPARSE_FULL_GAP

Main boundary: Sparse MAE remained 0.00602 above the selected full field (95%
CI 0.00173--0.01083), with the sparse condition higher in all six domains.

## I-B: CAI-task measurement value beyond task-agnostic C-scan saliency

### U3_UNIFORM_ORACLE

Main: The one-shot mechanical oracle improved CAI AUEBC over uniform
acquisition by 0.00391 (95% CI 0.00280--0.00502), favorable in all domains.

### U3_RECONSTRUCTION_ORACLE

Supplement legacy: The one-shot mechanical oracle improved CAI AUEBC over the
registered reconstruction oracle by 0.00373 (95% CI 0.00284--0.00469),
favorable in all domains. This is not the appearance-saliency comparator.

### U3_HEADROOM_RETENTION

Main support: The one-shot oracle retained 56.8% of the registered sequential-
oracle headroom. The comparison is retrospective and non-deployable.

### U3_CAI_VS_APPEARANCE_SALIENCY_AUEBC

Main: Under the common registered sequential protocol, appearance-saliency
minus CAI-oracle AUEBC was 0.007080 (95% CI 0.004799--0.009740), positive in
all six held-out domains. Lower AUEBC favors the CAI-oriented retrospective
oracle; neither oracle is deployable.

### U4_CAI_SALIENCY_MAP_SPEARMAN

Main support: Across 276 paired initial-state maps, mean mechanical-versus-
appearance Spearman rank agreement was 0.0222. This is weak agreement under
the registered comparator, not statistical independence.

### U4_CAI_SALIENCY_TOP10_OVERLAP

Main support: Mean top-decile overlap between paired CAI-task and appearance-
saliency maps was 0.2003. This is limited overlap under the registered
comparator, not universal disjointness.

### U4_ORACLE_CAI_SPECIFICITY

Supplement legacy: In the registered reconstruction cross-objective
diagnostic, the CAI-task oracle improved CAI AUEBC by 0.04862 (95% CI
0.04527--0.05205). This claim is not appearance saliency.

### U4_ORACLE_IMAGE_SPECIFICITY

Supplement legacy: The reconstruction oracle improved normalized RGB MSE by
5.503e-4 (95% CI 5.006e-4--6.063e-4) under the registered objective.

### U4_LEARNED_SPECIFICITY_BOUNDARY

Supplement legacy: Source-trained global mechanics and reconstruction masks
produced a support indicator of 0; the learned global mechanics mask did not
reproduce the legacy oracle separation.

## I-C: State- and predictor-conditioned measurement value

### O2_TEACHER_TURNOVER

Main: From initial state to 18.75%, the strict-OOF teacher changed its best
action for 70.4% of specimens.

### O2_TEACHER_RANK

Main support: Rank agreement with initial values was 0.405.

### O2_TEACHER_TOPK

Main support: Initial-to-18.75% top-five overlap was 0.307.

### O2_TEACHER_OPPORTUNITY

Main support: The descriptive state-conditioned opportunity was 0.00531.

### U5_RIDGE_HUBER_SPEARMAN

Main: Ridge--Huber value ranks agreed at Spearman 0.762 (95% CI
0.699--0.821) under identical strict-OOF states and splits.

### U5_RIDGE_HUBER_BEST_ACTION

Supplement: Ridge--Huber best-action agreement was 0.675 (95% CI
0.622--0.727).

### U5_RIDGE_HUBER_TOPK

Supplement: Ridge--Huber top-k agreement was 0.685 (95% CI 0.650--0.719).

### U5_RIDGE_MLP_SPEARMAN

Main support: Ridge--MLP value-rank agreement was 0.116 (95% CI
0.069--0.164). Full-state MAE was 0.08964 for Ridge, 0.08618 for Huber, and
0.15067 for the shallow MLP, so unequal accuracy bounds interpretation.

### U5_RIDGE_MLP_BEST_ACTION

Supplement: Ridge--MLP best-action agreement was 0.233 (95% CI
0.182--0.283).

### U5_RIDGE_MLP_TOPK

Supplement: Ridge--MLP top-k agreement was 0.213 (95% CI 0.192--0.235).

## II-A: State-conditioned valuation

### O4_DYNAMIC_MINUS_STATIC

Main opening: At 18.75%, dynamic real minus static regret was -0.001260 (95% CI
-0.002123 to -0.000444), favorable in five of six domains.

### O1_STATIC_SPEARMAN

Main support: The static score had Spearman -0.0196 with strict-OOF values
(95% CI -0.0591--0.0195), favorable in three domains.

### O1_STATIC_SET_REGRET

Supplement: Static exact-budget set regret was 0.08171.

### O1_GLOBAL_SET_REGRET

Supplement: Global exact-budget set regret was 0.07993.

### O1_RANDOM_SET_REGRET

Supplement: Random-median exact-budget set regret was 0.07978.

## II-B: Information-source and component decomposition

### O3_REAL_CHANGE

Main support: At 25%, real-state CAI MAE changed by -0.000731 from the initial
state (95% CI -0.001184 to -0.000250).

### O3_FULL_FIELD_RECOVERY

Supplement: The endpoint diagnostic recovered 26.6% (95% CI 10.6%--41.0%) of
the static-to-independent-full-field error ratio. The denominator is the
independent sensitivity estimator.

### O3_REAL_MINUS_POSITIONS

Main boundary: Real minus acquired-position/history MAE was 0.01740 (95% CI
0.00890--0.02582), favorable for real content in one domain.

### O3_REAL_MINUS_RECONSTRUCTION

Supplement legacy: Real minus the reconstruction-derived control was 0.03419
in CAI MAE (95% CI 0.02450--0.04384), favorable for real content in one
domain. The technical control is not an appearance-saliency control.

### O4_DYNAMIC_MINUS_SHUFFLED

Main boundary: Dynamic real minus shuffled-content regret was 2.328e-4 (95% CI
6.060e-5--4.180e-4), favorable for real content in one domain.

### A1_VALUATION_SUBSTITUTION

Main: Privileged valuation substitution improved CAI AUEBC by 4.979e-5 (95% CI
1.845e-5--8.105e-5).

### A1_LEARNED_PLANNING_SUBSTITUTION

Main support: Bounded learned-planning substitution improved AUEBC by 3.164e-6
(95% CI 1.411e-7--6.250e-6).

### A1_TRUE_VALUE_PLANNING_SUBSTITUTION

Main: True-value planning substitution improved AUEBC by 1.117e-4 (95% CI
9.898e-5--1.248e-4). It is retrospective.

## II-C: Cost-constrained set realization

### A2_GREEDY_PLANNING_REGRET

Main: In the registered two-action reachable pool, greedy regret relative to a
joint near-oracle set was 1.207e-4 (95% CI 1.033e-4--1.386e-4).

### A2_BEAM4_PLANNING_REGRET

Main support: Beam-4 regret was 1.127e-4 (95% CI 9.485e-5--1.308e-4).

### A3_FEEDBACK_BENEFIT

Supplement: Feedback benefit was -1.496e-5 (95% CI -1.944e-5 to -1.064e-5),
favorable in two of six domains. The direction is not generalized beyond the
frozen implementation.

### A4_BASELINE_MINUS_MAVIS

Main diagnostic: The supervised state-conditioned implementation had CAI
AUEBC 0.125053 versus 0.124992 for the static reference; the direction favors
the reference. Full supplement: reference minus learned was -6.114e-5 (95% CI
-8.461e-5 to -3.777e-5), and the learned implementation improved two of six
domains. This is one implementation endpoint, not the framework definition.

## Prohibited main-manuscript wording

Do not claim superior learned acquisition, real-content superiority over
matched controls, beneficial feedback, a universal location-value map,
scanner-time reduction, industrial deployment, external empirical
generalization, an unmeasured causal failure mechanism, saliency independence,
universal saliency irrelevance, or a causal material interpretation of the
CAI-minus-saliency percentile map.
