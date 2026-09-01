# G0 Metadata-Free CAI Assessor Audit

Status: `CAI_PLANNING_AUTHORIZED`.

The assessor uses only a frozen 512-D reconstruction embedding, effective
budget, observed-cell fraction, and mean observed level. It receives no domain
identity, specimen identity, impact history, laminate metadata, ply count,
`metadata13`, or `profile_stats21`.

Each leave-one-domain-out fold fits PCA-32 and Ridge alpha 10 on the five source
domains. Every source specimen contributes one zero-state anchor and 18
label-independent acquired states, for 19 equally weighted states. Depending on
the held-out domain, fitting uses 217-238 physical specimens and 4,123-4,522
states. All six folds record target exclusion and deterministic prediction
replay as true.

| Metric | Equal-domain result |
|---|---:|
| Zero-state CAI MAE | 0.182643120536 |
| Fixed 25% CAI MAE | 0.100215374569 |
| MAE improvement | 0.0824277459671 |
| Synchronized 95% CI | [0.0696398759474, 0.0951896340019] |
| Improving domains | 6/6 |

The authorization gate passes because endpoint MAE is lower, the paired lower
confidence bound is positive, every domain improves, replay is valid, and every
outer target is excluded from fitting, normalization, and selection. True CAI
remains evaluation and oracle-teacher privilege; it is not a policy input.

Authority: `cai_assessor_metrics.csv` SHA-256
`356729c7e298cabbe477db37d6aa3286ec67bf64afb65dba27e8f611e186f87e`.
