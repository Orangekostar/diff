# G0-C Task-Conditioning Audit

Status: `TASK_CONDITIONING_HEADROOM_GO` as a component gate.

At common exact-budget checkpoints, the FIELD oracle is evaluated on CAI and the
CAI oracle on FIELD. Positive values below mean wrong-task AUEBC minus
correct-task AUEBC.

| Objective | Correct-task advantage | 95% CI | Improving domains |
|---|---:|---|---:|
| FIELD | 0.009724283073 | [0.009285352589, 0.010169558857] | 6/6 |
| CAI | 0.020280014444 | [0.018388785185, 0.022221862075] | 6/6 |

Equal-domain descriptive trajectory overlap is low: action Jaccard 0.3444, cell
Jaccard 0.5027, high-level-action overlap 0.2248, and normalized edit distance
0.9405. The decisive evidence is the wrong-task performance loss, not trajectory
difference alone.

This supports preserving task identity in later learning. It does not establish
a learned task-conditioned policy, and the final G0 label remains
`G0_ACTIVE_INSPECTION_OPPORTUNITY_GO` because the separate initialization gate
failed.

Authority: `task_swap.csv` SHA-256
`5469e0185f4abc411487bd24a1f0343bc4f59cc52c5ae86ba4ae5cc9afbfdcaa`.
