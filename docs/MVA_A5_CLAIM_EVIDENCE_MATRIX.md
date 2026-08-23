# MVA A5 Claim-Evidence Matrix

Frozen before formal A5 result generation on 2026-08-23.

| Claim | Required evidence | Status after formal A5 |
|---|---|---|
| A4 leaves meaningful specimen-adaptive headroom | Positive oracle gap lower bound, >=4/6 domains, relative gap >=3% | PROVEN: `MVA_A5_AUTHORIZED` |
| A2 oracle rows can directly train outer-safe A5 | Predictor-roster audit | REJECTED: current outer domain may enter the source teacher |
| Policy inputs use only current observations | API and perturbation tests, target decision audits | SUPPORTED in the registered simulator: input-isolation tests and 33049 audited target actions passed |
| Imitation improves over global mechanical mask | Paired AUEBC, positive lower bound, >=4/6 domains | REJECTED: point effect 0.000547, 95% interval [-0.000793, 0.002097], 3/6 domains |
| Imitation improves over uniform | Paired AUEBC, positive lower bound, >=4/6 domains | REJECTED: point effect 0.000271, 95% interval [-0.001924, 0.002606], 3/6 domains |
| Imitation closes at least 20% oracle gap | Registered AUEBC ratio | REJECTED: 7.793% closure |
| Imitation reduces B5 | Equal-domain B5 comparison | SECONDARY SUPPORTED: 6.25% versus 18.75% for both global mechanical and uniform; does not change the primary gate |
| Policy is deployable on a physical scanner | Prospective scanner integration | NOT SUPPORTED |
| Laminate conditioning improves policy | A6/A7 structured-transfer evidence | NOT TESTED: `MVA_A6_NOT_AUTHORIZED` |

Terminal status: `MVA_A5_POLICY_NO_GO`. Per the frozen stop rule, A6 and A7
were not implemented or executed.
