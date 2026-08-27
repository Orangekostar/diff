# AEI Paper 1 Review-Fix P0 Audit

Audit date: 2026-08-27 UTC.

## Repository state

- Branch: `aei-information-hierarchy`
- Starting HEAD: `ba9709545e3ade21424540547e6ab277279345de`
- Relation to required base: exact match
- Starting worktree: clean
- Baseline paper suite: 70 passed

## Review issues

1. The manuscript AUEBC equation omitted the implementation's budget-span normalization.
2. The manuscript did not distinguish pre-freeze evidence, the frozen outer endpoint, and post-freeze diagnostics.
3. Closest-work differences were described in prose but not made operational in a source-backed comparison.
4. Predictor-conditioned wording did not disclose the shallow MLP's substantially worse OOF accuracy.
5. Framework reuse was discussed without five explicit transfer conditions.

## Revision scope

The revision may modify the AEI paper manuscript, paper-only code, tests, and
the `artifacts/aei_information_hierarchy/` and
`results/aei_information_hierarchy/` namespaces. The historical P1, MVA, MVD,
MAVIS, science-closure, and `artifacts/external_data/` namespaces remain
read-only.

## External-data cache audit

No local `rss.zip`, `interlock.zip`, `cranfield_wp2.zip`, or corresponding raw
cache directory was available under `/home/ww`. Public metadata and frozen
repository manifests are available, but no external performance experiment is
authorized by the current feasibility state.

Decision: `MANUSCRIPT_ONLY_PRIMARY`.
