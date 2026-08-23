# MVA A5 Oracle-Imitation Policy Report

- A5 policy status: `MVA_A5_POLICY_NO_GO`
- A6 authorization: `MVA_A6_NOT_AUTHORIZED`
- Oracle gap closure: 7.79%

## Synchronized held-out-domain effects

| Contrast | Point | 95% interval | Improved domains |
|---|---:|---:|---:|
| Global mechanical - policy | 0.000547 | [-0.000793, 0.002097] | 3/6 |
| Policy - oracle | 0.006468 | [0.004210, 0.008768] | 6/6 |
| Uniform - policy | 0.000271 | [-0.001924, 0.002606] | 3/6 |

Positive values favor the second method in each contrast.

## Equal-domain P-B budget metrics

| Method | AUEBC | B2.5 | B5 | B7.5 |
|---|---:|---:|---:|---:|
| Center-first | 0.017289 | 18.750% | 9.375% | 9.375% |
| Observed gradient | 0.017193 | 12.500% | 9.375% | 9.375% |
| Observed uncertainty | 0.017097 | 12.500% | 9.375% | 6.250% |
| Imitation policy | 0.017092 | 6.250% | 6.250% | 6.250% |
| Uniform | 0.017363 | 18.750% | 18.750% | 6.250% |
| Random median | 0.017619 | NA | 18.750% | 6.250% |
| Global mechanical | 0.017639 | 18.750% | 18.750% | 6.250% |
| Mechanical oracle | 0.010625 | 6.250% | 3.125% | 3.125% |

## Decision

The imitation policy improves AUEBC over global mechanical in 3/6 held-out domains. The preregistered decision above also requires improvement over uniform and at least 20% point oracle-gap closure. No B5 comparison changes the gate.

## Interpretation boundary

This is retrospective normalized-raster acquisition simulation. The deployable selectors never receive true CAI, unmeasured RGB values, full-image features, teacher values, or oracle actions. Mechanical oracle remains a diagnostic upper bound. These results do not establish physical scan-time or inspection-time savings.
