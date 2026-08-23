# Literature Search: Mechanical-Value Acquisition

Date: 2026-08-23
Purpose: ground A0-A3 baselines and novelty boundaries
Source policy: primary papers and official proceedings/publisher pages; policy
exclusions applied

## Closest-work clusters

| Cluster | Closest work | What is covered | MVA-specific gap |
|---|---|---|---|
| Ultrasonic adaptive sampling | AdaSTMAE | Adaptive sparse allocation for wavefield reconstruction | No CAI task value, strict cross-domain oracle, or screenshot-grid caveat |
| Task-driven imaging | TACKLE | End-to-end sampler/retriever/predictor for downstream MRI tasks | No ultrasonic CAI value maps or reconstruction-vs-mechanical oracle gate |
| Learnable masks | LOUPE | Budgeted population mask optimized for reconstruction | Static A4 baseline only; not an A2 oracle |
| Value of information | EDDI, ACO | Target-oriented and oracle-based feature acquisition | Not spatial C-scan acquisition under a structural response |
| Sequential acquisition | Active MRI, imitation coaching | POMDP/RL and oracle imitation templates | Current cohort first requires oracle headroom, not policy complexity |
| Task-based sensing theory | Task-based ADC | Optimize acquisition for target loss rather than signal recovery | Current data cannot validate hardware or physical acquisition speed |

## Opportunity and risk

The direct novelty risk is TACKLE: task-driven acquisition is not new in
general. The defensible opportunity is a domain-specific, falsifiable result
that separates reconstruction value from CAI mechanical value under strict
cross-configuration evaluation. AdaSTMAE makes a reconstruction-only ultrasonic
baseline mandatory. ACO and imitation-learning work make it mandatory to label
the true-target oracle as cheating/nondeployable and to prevent its unavailable
information from entering a later policy state.

The minimum convincing A0-A3 evidence is therefore uniform/random controls,
reconstruction and appearance oracles, a strict OOF CAI oracle, matched-budget
CAI and reconstruction metrics, map-rank similarity, domain-level headroom,
synchronized uncertainty, and an explicit no-go gate.
