# Fixed Six-Section Manuscript Outline

## 1. Introduction

### Paragraph sequence

1. Establish ultrasonic inspection as spatial engineering information for
   residual-capacity assessment, without assuming every measured location has
   equal task value.
2. Place the work after full C-scan-to-CAI prediction, sparse/adaptive
   ultrasound, task-driven sensing, and value-of-information research.
3. Identify the unresolved distinction: useful information may not be
   observable from the legal current state, and observable value may not be
   actionable under a cost constraint.
4. Introduce the three-layer Task-Relevant Information Hierarchy and state the
   three research questions.
5. Summarize the 276-specimen, six-domain, strict nested LODO evidence chain.
6. State three contributions: the hierarchy, a leakage-safe operationalization,
   and an experimental separation of the three layers including adverse
   controls and non-superiority.

### Assigned evidence

- Figure 1: conceptual hierarchy and information boundary.
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
- RQ1: what information is useful, and does the acquisition objective depend on
  the downstream task?
- RQ2: can future-measurement value be inferred from legal current information?
- RQ3: can conditionally valued information improve a cost-constrained sensing
  decision?

### Assigned evidence

- Conceptual definition only; numerical answers are reserved for Section 5.
- Required term: downstream-predictor-conditioned task value.

## 3. Task-Relevant Information Hierarchy and Operational Framework

### 3.1 Framework overview

- Define Useful, Observable, and Actionable as three nested tests rather than
  assumed implications.
- Explain Figure 1 and the teacher/deployable-policy boundary.

### 3.2 Downstream-predictor-conditioned task usefulness

- Compare risk with and without a measurement under a fixed predictor and
  validation protocol.
- Explain why a location has no universal intrinsic mechanical value.

### 3.3 Conditional observability

- Define the score available from legal state information and compare it with
  strict-OOF retrospective teacher values.
- Separate value evolution from the ability of a learned representation to
  track that evolution.

### 3.4 Decision actionability

- Define exact-cost sequential choice and AUEBC.
- Separate valuation error, bounded set-planning error, and feedback effects.

### 3.5 Operationalization with causal partial-state measurement valuation

- Present MAVIS only as the frozen operational testbed used to evaluate
  state-conditioned valuation and actionability.
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

### 4.3 Whole-dataset generalization protocol

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

### 4.6 Validation matrix for RQ1-RQ3

- Table 1: cohort, modalities, protocol, information boundary, cost, and
  inference.
- Map each research question to its registered comparison and adverse controls.

## 5. Experimental Results and Discussion

### 5.1 RQ1: Task usefulness of spatial and sparse ultrasonic information

#### 5.1.1 Registered scalar-to-spatial information contrast

- Claims U1_MATCHED_FIELD and U1_SURFACE_FIELD.

#### 5.1.2 Sparse retention and oracle headroom

- Claims U2_SPARSE_RETENTION, U2_SPARSE_GAIN, U2_SPARSE_FULL_GAP,
  U3_UNIFORM_ORACLE, U3_RECONSTRUCTION_ORACLE, and U3_HEADROOM_RETENTION.

#### 5.1.3 Mechanics versus reconstruction task specificity

- Claims U4_ORACLE_CAI_SPECIFICITY, U4_ORACLE_IMAGE_SPECIFICITY, and
  U4_LEARNED_SPECIFICITY_BOUNDARY.

#### 5.1.4 Predictor-conditioned task value

- Claims U5_RIDGE_HUBER_SPEARMAN and U5_RIDGE_MLP_SPEARMAN, with best-action
  and top-k agreement in the supplement.

#### Direct RQ1 answer

Spatial and sparse ultrasonic information is useful for the CAI task, and the
oracle-optimal acquisition objective changes with the downstream task; the
exact value map remains predictor-conditioned.

### 5.2 RQ2: Conditional observability during partial inspection

#### 5.2.1 Static observability

- Claims O1_STATIC_SPEARMAN and its exact-budget controls.

#### 5.2.2 Conditional value evolution

- Claims O2_TEACHER_TURNOVER, O2_TEACHER_RANK, O2_TEACHER_TOPK, and
  O2_TEACHER_OPPORTUNITY.

#### 5.2.3 Real-content, position, and reconstruction controls

- Claims O3_REAL_CHANGE, O3_FULL_FIELD_RECOVERY,
  O3_REAL_MINUS_POSITIONS, and O3_REAL_MINUS_RECONSTRUCTION.

#### 5.2.4 Narrow dynamic-valuation gain and shuffled adverse control

- Claims O4_DYNAMIC_MINUS_STATIC and O4_DYNAMIC_MINUS_SHUFFLED.

#### Direct RQ2 answer

Conditional value evolves materially, but it is only partially observable:
the frozen real-content representation does not establish value beyond matched
geometry/reconstruction controls, and shuffled content remains an adverse
control.

### 5.3 RQ3: From valuation to cost-constrained sensing decisions

#### 5.3.1 Valuation and planning attribution

- Claims A1_VALUATION_SUBSTITUTION, A1_LEARNED_PLANNING_SUBSTITUTION, and
  A1_TRUE_VALUE_PLANNING_SUBSTITUTION.

#### 5.3.2 Set-level planning gap

- Claims A2_GREEDY_PLANNING_REGRET and A2_BEAM4_PLANNING_REGRET.

#### 5.3.3 Feedback stress test

- Claim A3_FEEDBACK_BENEFIT.

#### 5.3.4 Frozen cross-domain acquisition boundary

- Claim A4_BASELINE_MINUS_MAVIS.

#### Direct RQ3 answer

Not under the current frozen learned implementation: retrospective valuation
and planning headroom exists, but feedback is adverse and the learned policy
does not outperform the strongest deployable baseline.

### 5.4 Integrated engineering-informatics interpretation

- Synthesize, without implying success: Useful does not imply Observable, and
  Observable does not imply Actionable.
- Explain why reconstruction fidelity is task-specific and why measurement
  value changes with both state and downstream predictor.
- State the current deployment boundary and the need to validate each layer
  independently before increasing policy complexity.
- Table 2 provides the complete main-text evidence ladder; Figures 2--4 provide
  the three direct experimental answers.

## 6. Conclusions

1. Answer RQ1: useful spatial/sparse information exists and oracle acquisition
   is task-specific.
2. Answer RQ2: true conditional value evolves, but robust specimen-specific
   observability is not established.
3. Answer RQ3: retrospective headroom does not become a superior frozen learned
   acquisition policy.
4. Close on the Task-Relevant Information Hierarchy as an evidence discipline
   for engineering sensing, not on MAVIS performance.
