# MSSS Existing Evidence Freeze

Date: 2026-08-22
Status: frozen before formal S1 execution

This ledger records only results that predate MSSS. It does not reinterpret any
prior experiment as scale discovery or transfer validation.

| ID | Frozen evidence | Status | Authority |
|---|---|---|---|
| E1 | A/W/H-only CAI prediction | `FAIL` | Frozen scalar baseline |
| E2 | Full C-scan contains CAI-relevant information | `mechanically informative` | P1 full-field oracle |
| E3 | Registered spatial destruction increases CAI error | `spatial organization necessary` | P3: 8x8 patch shuffle minus original equal-domain MAE = `0.05545046998688848`, positive in 6/6 domains, simultaneous lower = `0.009047576688234813` |
| E4 | Reduced digital observations retain substantial CAI value | `reduced observations retain substantial CAI value` | P5: bilinear 25% retention = `0.8989734769519824`; P5 gate `PASS` |
| E5 | Full-image generative reconstruction produced no registered mechanical gain | `full-image generative reconstruction unnecessary` | P6 gate `NO_MECHANICAL_GAIN` |
| E6 | MASI A5 did not retain CAI while satisfying the full factorization gate | `feature-level invariance insufficient` | A5 decision `FACTORISATION_NO_GO` |
| E7 | FULL, BILINEAR_50, and BILINEAR_25 are predictively redundant under ordinary LODO | `FULL / 50 / 25 predictively redundant under ordinary LODO` | Multi-view E1 predictive equivalence `true`; best-view counts 87/80/109 |
| E8 | Reduced-resolution representations have exploratory positive transfer signals | `exploratory only` | Leave-ply 25% MAE `0.09969490878581515` vs FULL `0.09795914047592547`; leave-layup 25% MAE `0.16976499193647399` vs FULL `0.1872925033731498` |

The evidence does not establish a mechanically sufficient scale, a common
boundary across scale operators, or target-independent transferability.

```text
MSSS = TO TEST
TRANSFERABILITY = TO TEST
```
