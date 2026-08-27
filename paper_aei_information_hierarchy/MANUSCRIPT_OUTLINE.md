# Fixed Six-Section Manuscript Outline

## 1. Introduction

### Paragraph sequence

1. Establish ultrasonic inspection as spatial engineering information for
   residual-capacity assessment, without assuming every measured location has
   equal task value.
2. Place the work after full C-scan-to-CAI prediction, sparse/adaptive
   ultrasound, task-driven sensing, and value-of-information research.
3. Identify the positive gap between characterizing structured task-relevant
   information and realizing it as an evidence-calibrated sensing decision.
4. State RQ-A on information characterization and RQ-B on decision realization.
5. Summarize the 276-specimen, six-domain, strict nested LODO evidence chain.
6. State three contributions: a two-stage acquisition framework, leakage-safe
   state-conditioned valuation, and calibrated end-to-end decision evidence.

### Assigned evidence

- Figure 1: two-part acquisition framework and information boundary.
- Literature boundary: direct C-scan CAI prediction, adaptive ultrasound,
  ultrasonic value of information, and task-driven multi-channel imaging.
- No result numbers beyond the minimum needed to preview the evidence chain.

## 2. Related Research and Problem Formulation

### 2.1 Ultrasonic information for post-impact performance assessment

- Surface observables, C-scan morphology, and residual CAI assessment.
- Closest AEI full C-scan-to-CAI study as prior work, not as a matched baseline.
- Distinguish predictive feasibility from task-specific information value.

### 2.2 Sparse and adaptive ultrasonic inspection

- Sparse ultrasound reconstruction and adaptive wavefield acquisition.
- Distinguish native-raster acquisition fraction from scanner time.
- Separate reconstruction quality from residual-capacity utility.

### 2.3 Task-driven sensing and value of information

- Task-driven imaging, Bayesian/sequential value of information, and
  cost-constrained sensing.
- Establish that retrospective oracle utility is a diagnostic target, not a
  deployable policy.

### 2.4 Problem formulation and research questions

- Define a downstream predictor `f`, legal current information `I_t`, candidate
  measurement `X`, and predictor-conditioned value
  `U_f(X | I_t) = R_f(I_t) - R_f(I_t union X)`.
- RQ-A: what spatial, sparse, objective-conditioned, and state-dependent
  structure defines task-relevant information?
- RQ-B: how far can that structure be realized through legal-state valuation,
  matched controls, bounded planning, feedback, and frozen policy evaluation?

### Assigned evidence

- Conceptual definition only; numerical answers are reserved for Section 5.
- Required term: downstream-predictor-conditioned task value.
- Table 1: source-backed closest-work positioning across six primary sources.
- Novelty statement: operational joint test under one causal acquisition
  contract, not a new generic taxonomy or first adaptive/VoI design.

## 3. Task-Relevant Information Acquisition Framework

### 3.1 Framework overview

- Define Part I, Information Characterization, and Part II,
  Evidence-Calibrated Decision Realization.
- Explain Figure 1, the causal state transition, and the retrospective-teacher
  versus deployable-policy boundary.

### 3.2 Characterizing task-relevant information

- Compare risk with and without a measurement under a fixed predictor and
  validation protocol.
- Explain why a location has no universal intrinsic mechanical value.

### 3.3 Causal partial-state valuation

- Define the score available from legal state information and compare it with
  strict-OOF retrospective teacher values.
- Separate value evolution from the ability of a learned representation to
  track that evolution.

### 3.4 Evidence calibration through matched controls

- Separate real measured content from acquired-position/history,
  reconstruction, and shuffled-content signals.
- Treat usefulness, task-value observability, and actionability as validation
  criteria inside the two stages.

### 3.5 Decision realization and deployment calibration

- Present MAVIS only as the frozen operational testbed used to evaluate
  state-conditioned valuation and decision calibration.
- Define exact-cost sequential choice and budget-span-normalized AUEBC on the
  actual/effective specimen-budget coordinate.
- Describe causal reveal, current-state construction, strict-OOF teachers,
  dynamic scoring, legal rollouts, and frozen policy evaluation at the level
  required to interpret the evidence.
- State that no future C-scan content or held-out-domain CAI outcome enters a
  deployable decision.

### Assigned implementation

- `src/cmc_bbdm/mavis/reveal.py`
- `src/cmc_bbdm/mavis/state_bank.py`
- `src/cmc_bbdm/mavis/state_encoder.py`
- `src/cmc_bbdm/mavis/mechanics_head.py`
- `src/cmc_bbdm/mavis/teacher.py`
- `src/cmc_bbdm/mavis/dynamic_voi.py`
- `src/cmc_bbdm/mavis/dynamic_training.py`
- `src/cmc_bbdm/mavis/rollout.py`
- `src/cmc_bbdm/mavis/policy.py`
- `src/cmc_bbdm/mavis/dynamic_metrics.py`
- `src/cmc_bbdm/mavis/closed_loop_metrics.py`

## 4. Multi-Domain CFRP Case Study and Experimental Design

### 4.1 Specimens and ultrasonic/CAI measurements

- 276 CAI-complete CFRP specimens across six experimental domains.
- Domain counts: 45, 49, 43, 59, 42, and 38.
- Modalities: specimen metadata, surface observables, ultrasonic RGB C-scan,
  and damaged-to-intact CAI-strength ratio.

### 4.2 Information representations and candidate measurements

- Scalar surface/internal descriptors, spatial full field, registered 25%
  bilinear sparse field, hierarchical native-raster actions, and current-state
  representations.
- Distinguish mechanics-targeted and reconstruction-targeted utility.

### 4.3 Held-out-domain evaluation protocol

- Strict nested leave-one-domain-out evaluation.
- Outer-domain specimens and outcomes excluded from fitting and selection.
- Physical specimen is the first statistical unit; six domains receive equal
  weight.

### 4.4 Privileged, deployable, and forbidden information

- Deployable: metadata, acquired positions, measured current content, legal
  actions, current exact cost.
- Retrospective only: future measurements, counterfactual values, true outcomes,
  and oracle action sets.
- Forbidden at decision time: unrevealed C-scan content and held-out outcomes.

### 4.5 Exact acquisition cost, metrics, and statistical inference

- Cost: unique newly observed native-raster locations divided by native count.
- Checkpoints: 3.125%, 6.25%, 9.375%, 12.5%, 18.75%, and 25%.
- CAI MAE and AUEBC, reconstruction MSE, value rank/agreement, and exact-budget
  set regret.
- Synchronized within-domain specimen bootstrap followed by equal-domain
  aggregation; computational state-action rows are not independent samples.

### 4.6 Validation criteria and evidence chronology

- Table 2: cohort, modalities, protocol, information boundary, cost, and
  inference.
- Map RQ-A and RQ-B to registered comparisons and matched controls.
- Distinguish pre-freeze evidence, the frozen outer endpoint, and post-freeze
  diagnostics; state that later diagnostics did not modify the endpoint.

## 5. Experimental Results and Discussion

### 5.1 Part I --- From Spatial Morphology to State-Conditioned Task Value

#### 5.1.1 Spatial morphology enriches residual-capacity information

- Claims U1_MATCHED_FIELD and U1_SURFACE_FIELD.

#### 5.1.2 Sparse observations preserve most task-relevant information

- Claims U2_SPARSE_RETENTION, U2_SPARSE_GAIN, and U2_SPARSE_FULL_GAP.

#### 5.1.3 Measurement utility is spatially heterogeneous

- Claims U3_UNIFORM_ORACLE, U3_RECONSTRUCTION_ORACLE, and
  U3_HEADROOM_RETENTION.

#### 5.1.4 Downstream objectives induce distinct measurement priorities

- Claims U4_ORACLE_CAI_SPECIFICITY, U4_ORACLE_IMAGE_SPECIFICITY, and
  U4_LEARNED_SPECIFICITY_BOUNDARY.

#### 5.1.5 Measurement value evolves with inspection state

- Claims O2_TEACHER_TURNOVER, O2_TEACHER_RANK, O2_TEACHER_TOPK, and
  O2_TEACHER_OPPORTUNITY.

#### 5.1.6 Predictor conditioning bounds the task-value definition

- Claims U5_RIDGE_HUBER_SPEARMAN and U5_RIDGE_MLP_SPEARMAN, with best-action
  and top-k agreement in the supplement.
- Report full-state OOF accuracy and state that the experiment does not resolve
  value-map variation among equally accurate structurally distinct predictors.

#### Part-I synthesis

Spatial morphology, sparse retention, heterogeneous opportunity, objective
specificity, state evolution, and predictor conditioning jointly characterize
the task-relevant information target.

### 5.2 Part II --- From State-Conditioned Value to Evidence-Calibrated Decisions

#### 5.2.1 A static reference motivates state-conditioned valuation

- Claims O1_STATIC_SPEARMAN and its exact-budget controls.

#### 5.2.2 Dynamic valuation captures incremental state dependence

- Claim O4_DYNAMIC_MINUS_STATIC.

#### 5.2.3 Control decomposition identifies the source of decision signal

- Claims O3_REAL_CHANGE, O3_FULL_FIELD_RECOVERY,
  O3_REAL_MINUS_POSITIONS, O3_REAL_MINUS_RECONSTRUCTION, and
  O4_DYNAMIC_MINUS_SHUFFLED.

#### 5.2.4 Component attribution separates valuation and planning headroom

- Claims A1_VALUATION_SUBSTITUTION, A1_LEARNED_PLANNING_SUBSTITUTION, and
  A1_TRUE_VALUE_PLANNING_SUBSTITUTION.

#### 5.2.5 Bounded set realization exposes decision-level headroom

- Claims A2_GREEDY_PLANNING_REGRET and A2_BEAM4_PLANNING_REGRET.

#### 5.2.6 End-to-end stress testing calibrates deployment readiness

- Claims A3_FEEDBACK_BENEFIT and A4_BASELINE_MINUS_MAVIS.

#### Part-II synthesis

State conditioning improves over the static reference; matched controls bound
source attribution; controlled substitutions expose valuation and planning
headroom; frozen stress tests quantify the residual deployable gap.

### 5.3 Integrated interpretation

- Synthesize the progressive contract from information structure through legal
  state attribution to frozen decision outcomes.
- Explain why reconstruction fidelity is task-specific and why measurement
  value changes with both state and downstream predictor.
- State the current deployment boundary and the need to validate each layer
  independently before increasing policy complexity.
- State five transfer conditions and label the Pascoe/Rhead examples as
  literature-only motivation, not external validation.
- Table 3 provides the complete progressive evidence chain; Figures 2--4 show
  information characterization, state-conditioned value, and decision
  calibration.

## 6. Conclusions

1. Close Part I on structured task-relevant information and its boundaries.
2. Close Part II on evidence-calibrated realization and the residual deployable
   gap.
3. State the engineering implication and limitations without presenting MAVIS
   as performance-superior.
