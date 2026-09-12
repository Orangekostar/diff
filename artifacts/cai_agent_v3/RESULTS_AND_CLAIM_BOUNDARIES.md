# CAI Agent v3 Results and Claim Boundaries

## Status

- Implementation: complete through the authorized resource gate.
- Protocol: conformant after the native-cost precision issue was detected, fixed and all affected W3 checkpoints invalidated.
- Scientific evidence: positive seed-1 internal VALID pilot; no three-seed or TEST claim.
- Engineering status: `ENGINEERING_ACCURACY_CRITERION_NOT_SPECIFIED`.

## Executed results

| Stage | Result | Evidence boundary |
|---|---|---|
| Corrected v2 rescore | Early VLM -0.4161 MPa, CI [-1.6645, 0.8974]; feedback +0.1468, CI [-0.3343, 0.6140]; fixed baseline +0.8584, CI [-1.4455, 3.0866] | All three 98.3333% intervals cross zero: `NOT_SUPPORTED` |
| Cohort | 276 physical specimens, 259 capture groups; split 161/50/65 over six domains | Reused internal cohort, not untouched external confirmation |
| Common predictor | `MEAN_SC`; VALID full-input MAE 41.1409 MPa, RMSE 53.1760, R2 0.7187; zero-input MAE 58.2584; prefix area 48.3126 | Internal VALID selection only |
| Frozen VLM | 211 TRAIN/VALID specimens requested; 205 valid, 6 unavailable; 167 cumulative v3 calls including 12 failed calls | Surface priority cue, not CAI information-gain ground truth |
| W3 main seed 1 | Main area 44.6276 vs `GEOMETRY_SPREAD` 46.9671; difference +2.3395 MPa (4.98% lower) | Passes the registered VALID continuation gate; one seed, no CI |
| Feedback direction | Open-loop 44.8529 minus main 44.6276 = +0.2253 MPa | Positive internal VALID direction only |
| Early VLM direction | no-VLM early 49.8969 minus main early 48.9080 = +0.9889 MPa | Positive internal VALID direction only |
| Structure diagnostic | `VLM_MEAN_FEEDBACK` 44.2955, lower than spatial main 44.6276 | Does not support superiority of the spatial Actor structure |
| GDFS adapter | hard VALID area 46.3356; frozen predictor hashes unchanged | Adapter integrity supported; performance superiority not supported |
| Seed expansion / TEST | `RESOURCE_LIMITED`; TEST VLM calls 0; TEST labels not accessed | No formal three-seed effect, bootstrap interval or TEST performance claim |

## Allowed claims

1. The left-step metric correction changes neither of the three old v2 conclusions: all remain unsupported.
2. A frozen-image `MEAN_SC` predictor passes the registered internal VALID readiness gate on the 276-candidate protocol.
3. With seed 1 on internal VALID, the VLM spatial-feedback policy passes the predeclared continuation threshold against the best nonadaptive route and is directionally better than its open-loop ablation.
4. The frozen-predictor grouped-Concrete GDFS adaptation runs with hard acquisition at validation and leaves predictor parameters unchanged.

## Prohibited claims

- Do not call the W3 result a three-seed, statistically confirmed or TEST result.
- Do not claim external generalization, deployment readiness or engineering accuracy.
- Do not claim that the spatial Actor is superior to the mean-feedback structure.
- Do not claim GDFS is an exact reproduction of Covert et al. or superior to the main policy.
- Do not treat VLM regions as CAI ground truth or the reused TEST role as untouched confirmation.
