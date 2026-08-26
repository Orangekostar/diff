# MAVIS Science-Closure Literature Ledger

Verification date: 2026-08-26. Sources are publisher, conference, preprint, or
institutional-repository records. The ledger bounds novelty; it does not treat
the literature as evidence that the MAVIS implementation is correct.

## Fuentes et al. (2020)

- Paper: R. Fuentes et al., "Autonomous ultrasonic inspection using Bayesian
  optimisation and robust outlier analysis."
- DOI / venue: `10.1016/j.ymssp.2020.106897`, *Mechanical Systems and Signal
  Processing* 145, 106897.
- Primary source: [Strathprints record and final published PDF](https://strathprints.strath.ac.uk/72351/).
- Problem: autonomous robotic NDT that locates damage with few observations.
- Observation type: sequential ultrasonic A-scan features (time of flight and
  attenuation) sampled over a C-scan surface.
- Acquisition objective: Bayesian-optimization acquisition over a robust
  novelty-index/damage-probability field.
- Static vs sequential: sequential; the posterior/objective field is updated
  after observations and the next physical location is chosen adaptively.
- Downstream task: damage/novelty detection, not CAI prediction.
- What it supports for MAVIS: adaptive sequential ultrasound and evolving
  inspection state are established; the experiment includes a robotically
  scanned carbon-fibre wing panel with delamination.
- What it does not support: mechanics-conditioned CAI value, strict-OOF
  cross-domain valuation, or the MAVIS representation/valuation/planning
  decomposition.
- Novelty boundary: do not claim first adaptive or first sequential ultrasonic
  inspection.

## Cantero-Chinchilla et al. (2020)

- Paper: S. Cantero-Chinchilla et al., "Optimal sensor configuration for
  ultrasonic guided-wave inspection based on value of information."
- DOI / venue: `10.1016/j.ymssp.2019.106377`, *Mechanical Systems and Signal
  Processing* 135, 106377.
- Primary source: [Elsevier article record](https://www.sciencedirect.com/science/article/abs/pii/S0888327019305989).
- Problem: choose the number and positions of guided-wave sensors for SHM.
- Observation type: ultrasonic guided-wave sensor responses in a structural
  monitoring configuration.
- Acquisition objective: expected value of information with an explicit
  information/cost trade-off.
- Static vs sequential: static system/sensor configuration, not a
  specimen-specific post-measurement policy.
- Downstream task: damage diagnosis/localization under a Bayesian model.
- What it supports for MAVIS: ultrasound VoI and cost-aware measurement design
  are established motivations.
- What it does not support: an evolving partial C-scan state, per-specimen
  conditional CAI utility, or feedback acquisition.
- Novelty boundary: do not claim first ultrasound VoI or first cost-aware
  ultrasonic sensor design.

## Memarzadeh and Pozzi (2016)

- Paper: M. Memarzadeh and M. Pozzi, "Value of information in sequential
  decision making: Component inspection, permanent monitoring and system-level
  scheduling."
- DOI / venue: `10.1016/j.ress.2016.05.014`, *Reliability Engineering & System
  Safety* 154, 137-151.
- Primary source: [Elsevier article record](https://www.sciencedirect.com/science/article/abs/pii/S0951832016300771).
- Problem: value information and schedule inspection/monitoring in sequential
  infrastructure-management decisions.
- Observation type: probabilistic inspection or monitoring observations in
  POMDPs.
- Acquisition objective: VoI under stochastic-availability and fee-based
  information models, with system-level scheduling heuristics.
- Static vs sequential: sequential; current beliefs and observations affect
  future decisions.
- Downstream task: component/system maintenance and inspection scheduling.
- What it supports for MAVIS: representation/belief, information valuation, and
  system scheduling are conceptually distinct layers.
- What it does not support: the correctness of MAVIS labels, neural MRIS, or
  ultrasound/CAI-specific empirical claims.
- Novelty boundary: cite as decision-theoretic motivation, not as validation of
  MAVIS.

## Blumberg, Slator, and Alexander (2024)

- Paper: S. B. Blumberg, P. J. Slator, and D. C. Alexander, "Experimental Design
  for Multi-Channel Imaging via Task-Driven Feature Selection."
- DOI / venue: ICLR 2024; arXiv `2210.06891`.
- Primary sources: [ICLR paper PDF](https://openreview.net/pdf/40758e44a35ee658f81eec070e7556108e298bb1.pdf),
  [arXiv record](https://arxiv.org/abs/2210.06891).
- Problem: select a compact image-channel design for a user-specified task.
- Observation type: densely sampled multi-channel imaging for a small training
  cohort, then a sparse global channel subset.
- Acquisition objective: downstream task loss rather than only parameter
  estimation or reconstruction.
- Static vs sequential: primarily static subset design of prescribed size.
- Downstream task: user-specified MRI, hyperspectral, remote-sensing, or
  physiological image-analysis task.
- What it supports for MAVIS: task-driven acquisition and dense-to-sparse
  supervision are established.
- What it does not support: state-dependent within-specimen valuation, causal
  reveal, or a sequential exact-cost planner.
- Novelty boundary: MAVIS is not described as TADRED applied to ultrasound.

## Ji et al. (2026)

- Paper: D. Ji et al., "Adaptive sampling for efficient Lamb wavefield
  reconstruction in composite laminates with Spatial-Temporal Masked
  AutoEncoder."
- DOI / venue: `10.1016/j.ultras.2026.107972`, *Ultrasonics* 162, 107972.
- Primary source: [Elsevier article record](https://www.sciencedirect.com/science/article/abs/pii/S0041624X26000247).
- Problem: reconstruct full composite Lamb wavefields from highly sparse SLDV
  measurements.
- Observation type: time-series Lamb wavefield samples over CFRP plates/blades.
- Acquisition objective: BO-guided damage-region sampling that minimizes
  wavefield reconstruction error for STMAE.
- Static vs sequential: adaptive spatial sampling-pattern construction; the
  reported learning endpoint is full wavefield reconstruction.
- Downstream task: reconstruction and damage-area fidelity, not CAI strength.
- What it supports for MAVIS: reconstruction-oriented adaptive composite
  ultrasound is a strong direct comparator and motivates P14.
- What it does not support: mechanics-optimal acquisition, CAI utility, or
  predictor-stable mechanical value.
- Novelty boundary: do not frame reconstruction-driven adaptive composite
  ultrasound as absent prior work.

## Mack et al. (2026)

- Paper: J. P. Mack et al., "Deep learning for predicting impact energy and
  compression after impact strength of composite materials using C-scan images."
- DOI / venue: `10.1016/j.aei.2026.104518`, *Advanced Engineering Informatics*
  72, 104518.
- Primary sources: [Elsevier article record](https://www.sciencedirect.com/science/article/abs/pii/S1474034626002107),
  [University of Akron repository record](https://ideaexchange.uakron.edu/university_research/55/).
- Problem: predict impact energy and residual CAI strength directly from damage
  morphology.
- Observation type: full ultrasonic C-scan images.
- Acquisition objective: none; full scans are already available.
- Static vs sequential: static full-image prediction.
- Downstream task: impact energy and CAI-strength regression with a ResNet18.
- What it supports for MAVIS: full experimental C-scan morphology can contain
  CAI-relevant information beyond hand-engineered scalar descriptors.
- What it does not support: sparse acquisition, adaptive inspection, conditional
  measurement value, or cross-domain closed-loop benefit.
- Metadata note: the publisher record lists Mack first and seven authors; the
  institutional repository displays a shorter Tan-first four-author record.
  The publisher author order is used here.
- Novelty boundary: MAVIS must not reduce its contribution to full C-scan to CNN
  to CAI prediction.

## Synthesis

The defensible distinction is not "adaptive ultrasound," "ultrasound VoI," or
"C-scan CAI prediction" in isolation. MAVIS studies predictor-conditioned
mechanical measurement value under an evolving, causally revealed inspection
state and separately diagnoses whether representation, valuation, and set-level
planning make that information actionable across held-out domains.
