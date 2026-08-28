# AEI Manuscript Outline: Main-Method Identity Reframe

## Scientific identity

- Primary method: **Task-Relevant Information Acquisition**.
- Part I: **Task-Relevant Information Characterization**.
- Part II: **State-Conditioned Task-Oriented Acquisition**.
- The supervised closed-loop implementation is evidence for one system
  diagnostic, not the paper-level method.
- The static source-only scorer is a reference; oracles and source controls are
  evidence instruments.

## 1. Introduction

Six paragraphs: engineering motivation; prior capability; integrated gap;
RQ-A/RQ-B and two-part method; 276-specimen six-domain protocol; three
contributions. Headline identity stays at the framework level.

## 2. Related Work

### 2.1 Post-impact ultrasonic information and residual-capacity assessment

Full-field and surface predictors establish predictive feasibility and motivate
spatial internal information.

### 2.2 Sparse and adaptive ultrasonic acquisition

Sparse reconstruction, adaptive paths, and robotic inspection establish the
acquisition setting; task loss is distinguished from reconstruction loss.

### 2.3 Task-relevant information acquisition formulation

Task-driven design, value of information, and partial observability motivate
state-conditioned value under one causal acquisition contract.

## 3. Task-Relevant Information Acquisition Framework

### 3.1 Task-Relevant Information Characterization

Define legal information, downstream-predictor-conditioned task value
`U_f(X | I_{i,t})`, and the retrospective opportunity boundary.

### 3.2 State-Conditioned Measurement Valuation

Define `U_hat`, strict source-only fitting, and static,
acquired-position/history, reconstruction, and shuffled-content controls.

### 3.3 Cost-Constrained Task-Oriented Acquisition

Define the legal policy, causal reveal, exact incremental cost, and
budget-span-normalized AUEBC.

## 4. Multi-Domain CFRP Experimental Design

### 4.1 Dataset and Information Representations

Describe 276 specimens, six indivisible domains, scalar/full/sparse/partial
state representations, 25% sparse observation, and the 8-by-8 action grid.

### 4.2 Causal Acquisition Protocol

Separate legal, privileged, and forbidden information; preserve exact reveal
and native-raster cost semantics.

### 4.3 Held-Out-Domain Evaluation and Statistical Analysis

Use nested LODO, specimen-first synchronized bootstrap contrasts, equal-domain
aggregation, and explicit evidence chronology.

## 5. Experimental Results and Discussion

### 5.1 From Spatial Information to State-Conditioned Task Value

#### 5.1.1 Spatial information and sparse recoverability

Main: U1 matched/surface evidence and U2 retention/gain/full-gap evidence.
Supplement: independent-field sensitivity.

#### 5.1.2 Task-conditioned spatial measurement value

Main: U3 oracle opportunity/headroom and U4 cross-objective specificity.
Supplement: learned global-mask boundary.

#### 5.1.3 State- and predictor-conditioned measurement value

Main: O2 value evolution and U5 rank agreement/accuracy boundary. Supplement:
best-action and top-k learner-pair details.

### 5.2 State-Conditioned Task-Oriented Acquisition

#### 5.2.1 State-conditioned valuation improves next-action estimation

Open with dynamic-minus-static regret, then annotate the static rank reference.

#### 5.2.2 Information-source and component decomposition

Report real-state change, acquired-position/history, reconstruction and
shuffled controls, followed by valuation/planning substitutions.

#### 5.2.3 Cost-constrained set realization

Report greedy and beam-4 reachable-set regret. Retain one concise system-level
diagnostic against the static reference; keep its full interval and domain
detail in the supplement.

### 5.3 Engineering Interpretation

Synthesize characterization, valuation, attribution, and realization. State
five transfer conditions and scope the evidence to the current digital-cost,
six-domain CFRP program.

## 6. Conclusions

Three paragraphs: Part-I evidence; Part-II mechanism and boundaries;
engineering implication from measuring everything to measuring task-relevant
information.

## Main visual contract

- Figure 1: framework identity and causal information flow.
- Figure 2: three Part-I stages plus registered sparse specimen states.
- Figure 3: state-conditioned valuation, priority evolution, and source controls.
- Figure 4: A1/A2 valuation, planning, and set realization only.
- Figure 5: paired task-specific priority overlays on one registered CFRP state.
- Table 1: compact case and protocol contract.
- Table 2: six-stage task-relevant result summary.
- Supplementary Figure S1: deterministic six-domain state-priority gallery.
