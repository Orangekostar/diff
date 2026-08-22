# MGMR M0 Completion Audit

Date completed: 2026-08-22
Decision: `MGMR_NO_GO`

## Scope completed

The registered 276-specimen, six-domain M0 experiment was executed exactly at
the frozen boundary. It included the B0--B4 direct roster, strict source-domain
OOF residual correction for coarse and FULL baselines, and all three registered
P3 8x8 patch-shuffle controls. The formal package and byte-identical replay both
passed independent checksum, prediction-to-metric, bootstrap, gate, and leakage
validation.

No M1 graph, laminate-aware model, diffusion model, or M1 result directory was
created. This is required by the frozen stop rule.

## Direct evidence

| Method | Equal-domain MAE | Worst-domain MAE |
|---|---:|---:|
| B0 exact P1 FULL | 0.089635804658 | 0.127430743874 |
| B1 coarse | 0.120742737488 | 0.171712010446 |
| B2 directional | 0.123837657140 | 0.166288493389 |
| B3 coarse + directional | 0.123962237568 | 0.167445788931 |
| B4 FULL + directional | 0.092414614249 | 0.135316841286 |

B3 was worse than both B1 and B2 in aggregate and improved over B1 in only
3/6 domains. B4 was worse than the exact B0. Gate A therefore failed.

## Residual evidence

| Audit | Corrected MAE | Benefit | Improved domains | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| Real directional -> coarse residual | 0.121767985577 | -0.001025248089 | 2/6 | 0.141189046899 | 0.116176995191 |
| Real directional -> FULL residual | 0.092419622617 | -0.002783817959 | 2/6 | 0.051704941821 | 0.074283757366 |
| P3 seed 20260831 -> coarse residual | 0.123358084424 | -0.002615346936 | n/a | 0.009933242721 | 0.027023586992 |
| P3 seed 20260901 -> coarse residual | 0.121119172785 | -0.000376435297 | n/a | 0.141325706699 | 0.143003667185 |
| P3 seed 20260902 -> coarse residual | 0.121975483437 | -0.001232745949 | n/a | 0.086227754512 | 0.070636549136 |

The real correction increased coarse and FULL error and improved only 2/6
domains in each audit. Gates B and C failed. The real coarse benefit did not
strictly exceed every shuffled benefit; Gate D failed even though its per-domain
comparison count was 4/6.

## Gate decision

| Gate | Result | Registered test |
|---|---|---|
| A | FAIL | B3 below B1/B2 and at least 4/6 B1-domain improvements |
| B | FAIL | Positive coarse residual benefit and at least 4/6 improvements |
| C | FAIL | Positive FULL residual benefit and at least 4/6 improvements |
| D | FAIL | Real benefit above every seed control and at least 4/6 domain comparisons |

Required gates A, B, and D did not all pass. The mechanics-guided two-component
story is not supported on this frozen cohort, so the only permitted decision is
`MGMR_NO_GO`.

## Claim outcomes

The pre-result `MGMR_CLAIM_EVIDENCE_MATRIX.md` remains byte-identical because it
is a checksum-bound registration input. This append-only outcome ledger resolves
only its M0 rows:

| Registered claim | Final status | Evidence |
|---|---|---|
| Coarse and directional components are complementary | NOT SUPPORTED | Gate A failed |
| Directional component predicts coarse residual | NOT SUPPORTED | Gate B failed |
| Directional component predicts FULL residual | NOT SUPPORTED | Gate C failed |
| Directional benefit is spatially specific | NOT SUPPORTED | Gate D failed |
| MGMR improves ordinary LODO | NOT SUPPORTED AT M0 | B4 was worse than B0 |
| Explicit graph and laminate claims | BLOCKED | M0 stop rule |

All six domains had prior project exposure. These results are a registered
post-hoc follow-up and are not untouched external confirmation.

## Reproducibility authority

- Config SHA-256: `77fa63fe58a17d426346db50fbb5ca0eb26efb2e25c2454c990552c2983607ff`
- Primary feature manifest: `f8961b9c8cee8a37fb5f54b4c3a62298999f3a11bfcd5d7e588671389ddc0d2e`
- P3 feature manifest: `749d2acb5c9703eb7dea4f58ddac10316d93e8f41b4f7de7fcf64aefc0534b39`
- Combined feature authority: `49c30ae4b8113c20e83cecb3fe1aebb34ae59526d1e322425b74e8b0b0a6ccc0`
- Formal scientific digest: `5f03f58f57543f18c06dc29ae6c7f9abe4534ddcce37096ddaa1b66856283c54`
- Formal/replay tree SHA-256: `8294c4c108c74bd13d34c49078c1873da8b76037a5de8b9f720a2c1d2ff45aaf`
- Formal state SHA-256: `e48eaf9c4d2029022987ea88d77aed549ad059485f3f5336ef2a86cecadba12e`

Authoritative outputs are under `results/mgmr/feature_bank`,
`results/mgmr/feature_bank_p3`, `results/mgmr/m0_component_gate`, and
`results/mgmr/replay/m0_component_gate`.
