# AEI Paper Development Ledger

Base commit: `c2eab6eac79dd3fbb9ecb0d19f98923e515e762b`

| Sequence | Commit | Stage | Verification | Frozen evidence changed? |
|---:|---|---|---|---|
| 0 | `c2eab6eac79dd3fbb9ecb0d19f98923e515e762b` | registered starting state | MAVIS 155 passed; MVD 29 passed; authority MVA 126 passed | no |
| 1 | `541f6c4` | P0 repository, evidence, manuscript-source, AEI-scope, and structure audit | `git diff --check`; Ruff; P0 evidence baselines | no |
| 2 | `2807384` | B/I full-field and sparse-retention semantic reconciliation; tested evidence generator | 21 paper-evidence tests; Ruff; `git diff --check` | no |
| 3 | `c069221` | canonical paper metrics, claim map, and source-hash authority | deterministic regeneration; 21 paper-evidence tests; Ruff; `git diff --check` | no |
| 4 | `6f4ff0c` | four traceable paper figures, captions, source CSVs, vector/raster exports, and visual QA | 10 figure tests; embedded PDF fonts; editable SVG text; checksum verification | no |
| 5 | `d808a5f` | protocol and information-hierarchy evidence tables with source and caption files | 11 table tests; standalone LaTeX compile and visual QA; checksum verification | no |
| 6 | `9da2fbe` | fixed six-section outline, evidence-bounded claim sentence bank, and paper package contract | 43 paper evidence/figure/table tests; claim-ID closure; Ruff; `git diff --check` | no |
| 7 | `c2c98e9` | related research, hierarchy framework, and multi-domain case/protocol draft | 51 paper tests; isolated journal-class compile; 15-entry citation closure; Ruff; `git diff --check` | no |
| 8 | `b158398` | Section 5 results and discussion drafted from canonical metrics; readable multipage hierarchy table integration | 57 paper tests; 23-page isolated compile with zero warnings/overfull boxes; table render QA; Ruff | no |

Each subsequent paper commit will be appended after its SHA exists. The final
entry must bind the pushed remote branch and verified artifact paths.
