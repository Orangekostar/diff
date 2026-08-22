# MSSS S1 Scale Discovery Report

Decision: **NO_GO** (0/3 axes passed).

| Axis | Gate | Selected scale | FULL MAE | Selected MAE | SSG | Positive domains |
|---|---:|---|---:|---:|---:|---:|
| sampling | FAIL | `sampling:density=0.0625` | 0.089636 | 0.091940 | 0.025722 | 6/6 |
| gaussian | FAIL | `gaussian:sigma=1.5` | 0.089636 | 0.105692 | 0.015566 | 5/6 |
| wavelet | FAIL | `wavelet:db2:low_only:level=2` | 0.089636 | 0.097679 | 0.023278 | 6/6 |

Scale selection used source domains only inside each outer fold. Global curves are descriptive.
Fourier sensitivity: `NOT_RUN_NONBLOCKING`.
This package is the registered formal S1 result.
