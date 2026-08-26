# MVD M1 GO/NO-GO

Decision: `MVD_OBSERVABILITY_NO_GO`.

M1 fails the predeclared observability gate. M2/M3 and capacity-rescue work
remain unauthorized.

| Gate / diagnostic | Result | Decision |
|---|---:|---|
| Selected O2 equal-domain Spearman | `-0.0195862049`, 95% CI `[-0.0590602945, 0.0194778222]` | FAIL |
| O2-minus-global NDCG@10 | `0.0210225609`, 95% CI `[-0.0240199088, 0.0719177659]`, 3/6 domains | FAIL |
| O2 Regret@1 | `0.0199900082` | No reliable top-action identification |
| O2 exact-budget set regret | `0.0817053737` | Worse than global `0.0799269237` and random `0.0797830698` |
| Candidate-only MLP Spearman | `0.0169130339` | Weak diagnostic, not a rescue |
| Predicted CAI AUEBC | `0.0172545220` | Worse than reconstruction `0.0171876604` |
| Oracle-advantage capture | `-0.0179267235`, 2/6 domains improved | FAIL |

Authority: `results/mvd/m1_observability/summary.json`,
`results/mvd/m1_observability/model_metrics.csv`, and
`results/mvd/m1_observability/bootstrap.csv`.

Allowed conclusion: static pre-inspection mechanical value is not reliably
observable with the registered low-complexity scorers. Forbidden conclusion:
increasing network capacity would solve observability, or M1 supports external
generalization.
