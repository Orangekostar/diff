# MVA Claim-Evidence Matrix

The evidence requirements were frozen before formal A2 execution on
2026-08-23. The status column was filled only after validation and aggregation.

| Claim | Required evidence | Formal status |
|---|---|---|
| Dense C-scan predicts CAI | Frozen P1 `I_frozen` equal-domain MAE | PROVEN: 0.08963580465761432 |
| Spatial organization matters | Frozen P3 shuffle gate | PROVEN |
| Uniform sparse sensing retains CAI value | Frozen P7 image-only 25% curve | PROVEN: 0.09011592076510834 |
| Current data represent physical point acquisition | Raw scanner point registry | NOT SUPPORTED |
| Reconstruction fidelity is a sufficient acquisition objective | Matched-budget CAI and reconstruction curves | REJECTED AS SUFFICIENT: mechanical AUEBC 0.01062453 versus reconstruction 0.01719904 |
| Mechanical value differs from reconstruction value | AUEBC, map rank correlation, top-k overlap | SUPPORTED: mean Spearman 0.0419, top-10% overlap 0.3302; paired AUEBC difference 0.00657452 |
| Mechanical oracle beats uniform | 12.5% relative MAE and >=4/6 domains | SUPPORTED: 35.43% relative improvement; 6/6 domains |
| Mechanical oracle beats reconstruction oracle | Synchronized AUEBC comparison | SUPPORTED: difference 0.00657452; 95% CI [0.00477228, 0.00863115] |
| Mechanical oracle beats appearance value | Synchronized AUEBC comparison | SUPPORTED: difference 0.00708006; 95% CI [0.00479936, 0.00974029] |
| Oracle headroom warrants testing policy learning | >=10% relative AUEBC or >=25% `B_5%` reduction | SUPPORTED: 38.81% AUEBC improvement and 66.67% `B_5%` saving; no deployable policy claim |
| Global task-aware mask improves acquisition | A4 evidence | REJECTED: global mechanical AUEBC 0.017639 versus uniform 0.017363; paired interval crosses zero |
| Deployable policy imitates the oracle | A5 evidence | NOT ESTABLISHED: policy AUEBC 0.017092, only 3/6 improvements over each primary baseline, 7.793% oracle-gap closure |
| Laminate context improves acquisition | A6/A7 structured transfer | NOT TESTED: A5 issued `MVA_A6_NOT_AUTHORIZED` |

The CAI oracle uses true CAI and unobserved candidate RGB values. It is an
upper-bound diagnostic and cannot be reported as deployable method performance.
The A5 policy is deployable only within the registered normalized-raster
simulator; no result establishes physical scanner control or inspection-time
savings.
