# P1 Visual Observability Decision

- Status: `P1_SURFACE_VISUAL_OBSERVABILITY_NO_GO`
- Authorized route: `NONE`
- Specimens: `276`
- Bootstrap resamples: `100000`
- Oracle-gap closure: `9.798758058215481e-08`
- Decision state SHA-256: `e563a42cea940e2ebb9b82a7f2757d46ac54c688416556e3e6a351607601bd95`

## Gate Effects

| effect | point | 95% CI | improved domains |
|---|---:|---:|---:|
| c0_minus_global | -0.000162745576 | [-0.0011903367, 0.000863469304] | 3/6 |
| c0_minus_proposed | 1.17349987e-09 | [-1.48003689e-09, 3.83560312e-09] | 4/6 |
| deranged_minus_proposed | -2.81494855e-10 | [-1.34226204e-09, 7.75537891e-10] | 2/6 |
| global_minus_proposed | 0.00016274675 | [-0.000863467145, 0.00119033801] | 3/6 |
| shuffled_global_minus_global | -3.45964577e-05 | [-9.61636017e-05, 6.23938723e-08] | 0/6 |
| shuffled_minus_proposed | 1.17349987e-09 | [-1.48003689e-09, 3.83560312e-09] | 4/6 |
| wrong_minus_proposed | 1.17349987e-09 | [-1.48003689e-09, 3.83560312e-09] | 4/6 |

## Downstream Authorization

- P2: `NOT_RUN_NOT_AUTHORIZED`
- P3: `NOT_RUN_NOT_AUTHORIZED`
- P4: `NOT_RUN_NOT_AUTHORIZED`

The preregistered existing-data route stops at P1; no result rescue was run.
