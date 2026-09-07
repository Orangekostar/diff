# G1 Task-Conditioning Result

Status: `G1_TASK_CONDITIONING_NO_GO`

Each effect is comparator AUEBC minus correctly task-conditioned AUEBC, so a
positive value favors the correct task token.

| Task and control | Effect | 95% CI | Improved domains |
|---|---:|---:|---:|
| FIELD WRONG_TASK | 0.0000030030636740644057 | [0.0000012334469453508708, 0.000005033437921055967] | 4/6 |
| FIELD NO_TASK | 0.0000017949810500134293 | [0.0000006563384341036675, 0.0000031700703012368995] | 4/6 |
| CAI WRONG_TASK | 0.00006731435252771757 | [-0.00007233853622756369, 0.0002084059247615635] | 4/6 |
| CAI NO_TASK | 0.00003107987501816518 | [-0.00011143823251693787, 0.0001704391507672262] | 4/6 |

FIELD satisfies the preregistered positive-CI and 4/6-domain conditions against
both controls. Both CAI intervals cross zero. Because task-conditioning evidence
is required for both engineering tasks, the combined component is NO_GO; the
data do not establish that task identity improves CAI acquisition on held-out
domains.
