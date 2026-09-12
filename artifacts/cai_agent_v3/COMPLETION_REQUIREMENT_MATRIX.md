# CAI Agent v3 Completion Requirement Matrix

This audit maps the full execution specification to the delivered state. `PASS` means the executed requirement is verified; `STOPPED` is an authorized gate/resource stop; `FAIL` is a discovered hard-contract violation whose affected results are invalidated.

| Source | Requirement | Evidence | Status |
|---|---|---|---|
| 0.1-0.2 | No forbidden legacy reruns; isolated branch and output roots | Git diff and `compute_ledger.jsonl`; all new work is under v3 roots | PASS |
| W0 | Reconstruct stored v2 states and independently price runs | 7,283 states, 432 runs; five required legacy outputs | PASS |
| 3.1-3.3 | Left-step metric, tail, terminal 0.25, independent seed pricing | External numeric oracle 7/7 and focused tests | PASS |
| W1.1-1.2 | Authorized 276 cohort, author MPa labels, source-only groups, TEST redaction | 276/276 labels matched; 259 groups; TEST targets NaN | PASS |
| W1.3 | Per-cell frozen features and real frozen VLM without TEST calls | Six NPZ shards; 60 reused/216 encoded; 205 valid fit VLM records | PASS |
| W2.1-2.2 | Fixed Ridge and three declared neural candidates | Four Ridge fits and three architecture manifests | PASS |
| W2.3 | Select every checkpoint on fixed exact-cost VALID prefixes | Three prefix masks changed after float32-based selection; intermediate checkpoints absent | FAIL |
| W2.4-2.5 | Lock `P_all` and three OOF reward models only after valid selection | Gate corrected to `PREDICTOR_NOT_READY` / `REWARD_MODELS_NOT_READY` | STOPPED |
| W3 | Train/evaluate policy matrix only after W2 ready | Historically executed; all results now `DIAGNOSTIC_ONLY_INVALIDATED` | FAIL |
| W4 | One GDFS pilot after ready predictor and completed W3 | Historically executed; mechanism artifacts retained, performance invalidated | FAIL |
| W5 | Expand seeds and open TEST only after valid gates and sufficient resources | Zero expansion updates, zero TEST VLM calls, TEST labels not accessed | STOPPED |
| W6 | A01-A12 review with separate implementation/protocol/science/engineering status | Corrected JSON has all required fields; A01/A03/A11 are blockers | PASS |
| Resource | Count all failed/invalid work under 28,100 updates | 21,514 known + <=750 unknown = <=22,264; remaining >=5,836 | PASS |
| Recovery | Do not lower requirements when budget cannot fund recovery | W2 replay cap 12,000 exceeds remaining; no retraining performed | STOPPED |
| W7 | Code, CLI, configs, source/license, artifacts, review, ledger and Git delivery | Focused tests/Ruff/help/diff checks; final SHA verified after push | PASS |

Final interpretation: software and audit delivery are complete, but the scientific execution is not protocol-conformant beyond W1. W2-W4 metrics remain available only as invalidated diagnostic residue. No formal TEST claim exists.
