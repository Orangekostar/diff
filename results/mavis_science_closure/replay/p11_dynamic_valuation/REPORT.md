# MAVIS P11 Dynamic Valuation Closure

Status: `COMPLETE`.

All five scorers are evaluated on the same frozen current-state legal
actions and strict-OOF teacher values. MVD O2 and candidate-only scores
are specimen-cell static diagnostics; their historical costs never replace
the current exact marginal action cost.

| Checkpoint | Dynamic real regret | Static M1/O2 | Candidate-only | Positions | Shuffled |
|---:|---:|---:|---:|---:|---:|
| 0.03125 | 0.016057 | 0.016079 | 0.016316 | 0.016460 | 0.015958 |
| 0.06250 | 0.012231 | 0.013862 | 0.012992 | 0.012539 | 0.011970 |
| 0.09375 | 0.011539 | 0.013037 | 0.012324 | 0.011864 | 0.011367 |
| 0.12500 | 0.011257 | 0.012686 | 0.011977 | 0.011586 | 0.011094 |
| 0.18750 | 0.010012 | 0.011272 | 0.010387 | 0.010109 | 0.009780 |

At the final decision checkpoint, dynamic real-state valuation has lower regret than frozen static M1/O2, but growth of that advantage from the initial checkpoint is uncertain and shuffled-content valuation remains better; the gain is not attributable to accumulated specimen-specific measured content.

Primary inference uses next-action regret and one-step CAI utility.
Spearman, NDCG, and Recall@K are secondary diagnostics. Metrics first
average states within physical specimens at each cost, then weight held-
out domains equally; paired bootstrap resamples specimens within domain.

This is a retrospective valuation diagnostic. It does not retune target
domains, replace a P7 checkpoint, or change the frozen Tier-B claim.
