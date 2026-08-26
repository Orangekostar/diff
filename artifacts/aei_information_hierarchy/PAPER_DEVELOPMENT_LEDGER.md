# AEI Paper Development Ledger

Base commit: `c2eab6eac79dd3fbb9ecb0d19f98923e515e762b`

| Sequence | Commit | Stage | Verification | Frozen evidence changed? |
|---:|---|---|---|---|
| 0 | `c2eab6eac79dd3fbb9ecb0d19f98923e515e762b` | registered starting state | MAVIS 155 passed; MVD 29 passed; authority MVA 126 passed | no |
| 1 | `541f6c4` | P0 repository, evidence, manuscript-source, AEI-scope, and structure audit | `git diff --check`; Ruff; P0 evidence baselines | no |
| 2 | `2807384` | B/I full-field and sparse-retention semantic reconciliation; tested evidence generator | 21 paper-evidence tests; Ruff; `git diff --check` | no |
| 3 | `c069221` | canonical paper metrics, claim map, and source-hash authority | deterministic regeneration; 21 paper-evidence tests; Ruff; `git diff --check` | no |

Each subsequent paper commit will be appended after its SHA exists. The final
entry must bind the pushed remote branch and verified artifact paths.
