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
| scale depends systematically on laminate architecture | EXPLORATORY SIGNAL, NOT VALIDATED | Post-`NO_GO` diagnostic found 3/3-axis coarser directions for quasi-isotropic layup, higher damage area, and higher damage width; ply count was not aligned and damage height was non-monotonic |

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

## Post-NO-GO Coupling Diagnostic

The controlling prompt's configuration-dependent branch was executed as a
separate immutable package at `results/msss/s1_no_go_coupling`. It consumed only
the formal cross-fitted S1 candidate predictions and the registered structural
and physical authorities.

| Factor | Cross-axis result | Scope restriction |
|---|---|---|
| ply count | no alignment; sampling `SAME`, Gaussian `NON_MONOTONIC`, wavelet `COARSER` | no ply-count coupling claim |
| layup family | 3/3 axes coarser from cross-ply to quasi-isotropic | post-hoc exploratory only |
| damage area | 3/3 axes coarser from low to high tertile | post-hoc exploratory only |
| damage height | 3/3 axes non-monotonic | no height coupling claim |
| damage width | 3/3 axes coarser from low to high tertile | post-hoc exploratory only |

The diagnostic and replay share scientific digest
`e59dff40aa1d588c6654795c8130b22fa1be66950824d6224afc673418897203`
and output-tree digest
`9a2c972055b364815cacfc0c31e4cf29f928645832f07918ddd4f8055ac03318`.
Several aligned groups terminate at the coarsest registered candidate without
an over-coarse boundary. The status therefore remains
`NOT_VALIDATED_POST_HOC`. Independent preregistered evidence is required before
promoting Scale-Laminate Coupling; the same inspected cohort cannot supply that
independence.
