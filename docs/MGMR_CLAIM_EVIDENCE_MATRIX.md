# MGMR Claim-Evidence Matrix

Date initialized: 2026-08-22

| Claim | Status | Authority / next test |
|---|---|---|
| Global area, width, and height are insufficient | PROVEN | Frozen G1/G2 evidence |
| Full C-scan contains CAI-relevant information | PROVEN | P1 |
| Spatial organization matters | PROVEN | P3 8x8 patch-shuffle control |
| Reduced sampling retains substantial CAI value | PROVEN | P5 |
| Pure isotropic smoothing is insufficient | PROVEN | MSSS S1 Gaussian axis |
| Wavelet low-only loses information | PROVEN | MSSS S1 primary wavelet axis |
| Low plus directional detail is promising | EXPLORATORY | MSSS sensitivity only |
| Feature invariance improves transfer | NOT SUPPORTED | MASI and multi-view evidence |
| A transferable MSSS exists | NOT SUPPORTED | MSSS S1 `NO_GO` |
| Coarse and directional components are complementary | TO TEST M0 | B3 vs B1/B2, Gate A |
| Directional component predicts coarse residual | TO TEST M0 | Strict OOF residual audit, Gate B |
| Directional component predicts FULL residual | TO TEST M0 | Strict OOF residual audit, Gate C |
| Directional benefit is spatially specific | TO TEST M0 | Real vs P3 shuffle, Gate D |
| Explicit spatial graph adds value | BLOCKED ON M0 | M1 only after M0 GO |
| Laminate orientation context adds value | BLOCKED ON M0 | M2 only after M0 GO |
| MGMR improves ordinary LODO | TO TEST | B4 in M0; graph claim conditional |
| MGMR improves leave-ply transfer | BLOCKED ON M0 | Structured transfer phase |
| MGMR improves leave-layup transfer | BLOCKED ON M0 | Structured transfer phase |

No `TO TEST` or `BLOCKED` row may be promoted without its named artifact and
gate. Historical use of the six domains must accompany any positive M0 claim.
