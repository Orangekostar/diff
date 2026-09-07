# G1 Stopping Result

Status: `G1_FIELD_STOPPING_NO_GO` and `G1_CAI_STOPPING_NO_GO`

The observable STOP head was trained independently from the action policy on
source-only labels. Each source selection considered thresholds 0.50, 0.70,
0.80, 0.90, 0.95, 0.975, and 0.99 subject to premature-stop rate <= 0.05 and
task-loss ratio <= 1.05. No threshold was authorized for either task in any of
the six outer folds. The preregistered fallback therefore retained the fixed
0.25 endpoint for every target trajectory.

| Task | Authorized domains | Saving | 95% CI | Improved domains | Task-loss ratio | Premature stop | Never stopped |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIELD | 0/6 | 0.0 | [0.0, 0.0] | 0/6 | 1.4281895236048958 | 0.0 | 1.0 |
| CAI | 0/6 | 0.0 | [0.0, 0.0] | 0/6 | 2.538297473468669 | 0.0 | 1.0 |

No specimen-specific measurement saving is supported. The zero premature-stop
rate is a consequence of never authorizing STOP, not evidence that an observable
early-termination policy succeeded.
