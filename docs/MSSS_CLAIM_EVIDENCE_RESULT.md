# MSSS Claim-Evidence Result

Date: 2026-08-22
Formal S1 decision: `NO_GO`
S2 status: `NOT_RUN_NOT_AUTHORIZED`

This is the post-run result ledger. The initial
`MSSS_CLAIM_EVIDENCE_MATRIX.md` remains unchanged because it is a hash-bound
pre-registration source in the executed protocol.

| Claim | Final status | Evidence |
|---|---|---|
| spatial organization matters | PROVEN | Frozen P3 8x8 patch-shuffle result |
| reduced sampling retains CAI value | PROVEN, protocol-specific | Frozen P5 bilinear sampling study |
| mechanically sufficient scale exists | NOT ESTABLISHED | Formal S1 `NO_GO`; 0/3 complete axis gates passed |
| same scale appears across independent scale axes | NOT ESTABLISHED | Sampling lacked an over-coarse boundary; Gaussian was unstable and cross-fitted non-inferiority failed; wavelet cross-fitted non-inferiority failed |
| MSSS remains spatially specific | COMPONENT SUPPORTED, CLAIM NOT PROMOTED | SSG passed on all three axes, but no axis satisfied the complete MSSS gate |
| MSSS is transferable across ply count | NOT RUN | S1 did not authorize S2 |
| MSSS is transferable across layup | NOT RUN | S1 did not authorize S2 |
| MSSS improves severe structural extrapolation | NOT RUN | S1 did not authorize S2 |
| scale depends systematically on laminate architecture | NOT EVALUATED | Exploratory analysis was not authorized by S1 |

## Formal S1 Axis Audit

| Axis | Gate | Selected descriptive scale | Cross-fitted sufficient | Stable | Boundary | Spatially specific |
|---|---|---|---:|---:|---:|---:|
| sampling | FAIL | density `0.0625` | PASS | PASS | FAIL | PASS |
| Gaussian | FAIL | sigma `1.5 px` | FAIL | FAIL | PASS | PASS |
| wavelet | FAIL | db2 low-only level `2` | FAIL | PASS | PASS | PASS |

The formal and replay packages share scientific digest
`6ac389b0a4e09487202f5a8a9273dfdf5b338ef40de705661c5877e3e9bd0152`
and output-tree digest
`e41c42bdd8cb022b2d7d3c286685ae2530f2c302a236cf2e0a77d76ecf6a365b`.
Optional Fourier analysis was `NOT_RUN_NONBLOCKING`; impact-condition transfer
was not run because S2 was not authorized.
