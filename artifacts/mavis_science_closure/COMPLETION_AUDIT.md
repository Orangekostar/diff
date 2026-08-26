# MAVIS Science-Closure Completion Audit

## Scope and decision

P8-P16 are complete. Final decision: `CLAIM_NARROWING_GO` and
`METHOD_EXTENSION_NO_GO`. The closure preserves the frozen P7 Tier-B boundary
and does not retune or replace a P7 checkpoint.

## Stage ledger

| Stage | Package | Scientific result |
|---|---|---|
| P8 | `artifacts/mavis_science_closure/P8_CLAIM_EVIDENCE_AUDIT.md` | Reverse-mapped P1-P7 evidence and claim boundaries |
| P9 | `results/mavis_science_closure/p9_value_evolution/` | True conditional action value evolves; the real scorer does not capture the opportunity reliably |
| P10 | `results/mavis_science_closure/p10_mris_causal/` | Real partial content does not beat positions-only/reconstruction controls |
| P11 | `results/mavis_science_closure/p11_dynamic_valuation/` | Endpoint advantage over static is narrow; shuffled remains better |
| P12 | `results/mavis_science_closure/p12_rvp_attribution/` | Retrospective valuation and planning substitutions improve AUEBC |
| P13 | `results/mavis_science_closure/p13_set_planning/` | Bounded set-planning regret is positive |
| P14 | `results/mavis_science_closure/p14_task_specificity/` | Oracle task specificity is supported; learned separation is not |
| P15 | `results/mavis_science_closure/p15_value_stability/` | Value structure is downstream-predictor-conditioned |
| P16 | `results/mavis_science_closure/p16_feedback_mechanism/` | Feedback is adverse and the proposed value-change mechanism is unsupported |

## Integrity barriers

- Frozen P7 tree SHA-256:
  `931dc86c26caf1c7246709c4706a7cd0428e3a1533b6ff1ad3c2ad8f9517d1e4`.
- Every P9-P16 formal summary binds the same P7 tree and records no P7 change.
- P9-P16 replay contains 87 files and 72,062,424 bytes per formal/replay side.
- Every replay file is byte-identical to its formal counterpart.
- Every formal and replay `CHECKSUMS.sha256` ledger verifies.
- P9/P10 were replayed at their originating commits because their summaries
  intentionally bind the then-current runtime code state; P11-P16 replay under
  their final stage code is byte-identical.
- Oracle rows are marked retrospective and non-deployable.
- Target outcomes do not select models, policies, thresholds, or beam widths.

Replay details: `results/mavis_science_closure/replay/REPLAY_AUDIT.csv`.
Claim limits: `artifacts/mavis_science_closure/MANUSCRIPT_CLAIM_MAP.md`.

## Verification

| Check | Result |
|---|---|
| `PYTHONPATH=src python -m pytest -q tests/test_mavis_*.py` | 155 passed |
| `PYTHONPATH=src python -m pytest -q tests/test_mvd_*.py` | 29 passed |
| Historical MVA regression in the complete authority repository | 126 passed |
| Completion/replay/P7 integrity selection | 4 passed |
| Formal/replay directory comparison | 8/8 packages byte-identical |
| Formal and replay checksum ledgers | 16/16 passed |
| `ruff check .` | passed |
| `git diff --check` | passed |

The complete authority-repository suite was also run as a non-gating diagnostic:
`2955 passed, 68 failed, 25 errors` in 1:43:01. Those pre-existing failures are
outside this change and concentrate in historical CPB/D8 frozen-code
provenance, stale D8 selection evidence, and generated-package authority
validation. The reduced Git worktree likewise cannot run authority-dependent
legacy tests without the untracked data and sibling prompt files. These broader
results are reported explicitly and are not represented as a green full-suite
gate.

Remote commit and path verification is performed after the completion commit is
pushed.

## Final boundary

The defensible paper is a mechanism-and-boundary account of representation,
conditional valuation, and set planning. It is not a claim that MAVIS beats the
strongest deployable baseline, improves most domains, reduces real scanner time,
or generalizes to an external cohort.
