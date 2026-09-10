# Hasebe Reference Evidence Trace Implementation Plan

> **For Codex:** Execute this plan under the user-provided HASEBE reference-evidence execution contract v1.0. Preserve all frozen reviewed evidence and perform no training or model inference.

**Goal:** Trace author-supplied Hasebe evidence to exact files, workbook cells, specimens, stages, and units; crosswalk the 24 reviewed TEST specimens; and state what scan-evaluation claims the evidence does and does not support.

**Architecture:** Add one pure evidence module and one thin CLI with four stages: `inventory`, `extract`, `compare`, and `summarize`. Extraction reads only the explicitly supplied external source root and current frozen artifacts. Summarization reads only extracted CSV/JSON, so rebuilding documents cannot trigger workbook reads, Reader recovery, reviewed rescoring, or training.

**Tech Stack:** Python 3.13, stdlib CSV/JSON/zip/XML, openpyxl, Pillow, NumPy, Polars, matplotlib, pytest, Ruff.

---

### Task 1: Define evidence contracts with failing tests

**Files:**
- Create: `tests/test_hasebe_reference_evidence.py`
- Create later: `src/cmc_bbdm/learned_cscan/hasebe_reference_evidence.py`

Write and run focused tests for workbook merged headers/formulas, explicit provenance classification, strict specimen/stage joins, null reasons and 24-specimen coverage, calibration-dependent comparison, and summary isolation.

### Task 2: Implement inventory and workbook extraction

**Files:**
- Create: `src/cmc_bbdm/learned_cscan/hasebe_reference_evidence.py`
- Create: `scripts/trace_hasebe_reference_evidence.py`

Implement bound-source validation, hash checks, all-sheet workbook cell extraction, exact source-cell lineage, and author measurement records. Never recalculate formulas, infer missing measurements, or promote derived fields to author evidence.

### Task 3: Build the TEST-24 crosswalk and fixed-case review

**Files:**
- Modify: `src/cmc_bbdm/learned_cscan/hasebe_reference_evidence.py`
- Create outputs under: `results/hasebe_reference_evidence/v1/`

Join by exact `(dataset_id, specimen_id)` identities, retain stage and ambiguity, review only `ykhs7s2dck:q8-4`, `74t7kcdgkr:c8-24`, and `cgtnjyggtm:q24-40`, and emit null physical comparisons when scale or quantity semantics are unresolved.

### Task 4: Verify primary literature and source descriptions

**Files:**
- Create: `artifacts/hasebe_reference_evidence/v1/SOURCE_EVIDENCE_LEDGER.md`

Record DOI/data-repository evidence at section/page/record level. Label abstract-only and incomplete-index access explicitly. Do not infer absence of author masks from an incomplete source index.

### Task 5: Generate bounded artifacts and decisions

**Files:**
- Create: `artifacts/hasebe_reference_evidence/v1/SOURCE_BINDINGS_AND_FIELD_LINEAGE.md`
- Create: `artifacts/hasebe_reference_evidence/v1/REFERENCE_FEASIBILITY_AND_NEXT_STEP.md`
- Create: `artifacts/hasebe_reference_evidence/v1/AUTHOR_DATA_REQUEST_DRAFT.md`
- Create: `artifacts/hasebe_reference_evidence/v1/CODEX_HANDOFF_HASEBE_REFERENCE_EVIDENCE.md`
- Create: at most three evidence PNGs under `artifacts/hasebe_reference_evidence/v1/figures/`

Use the CLI outputs as the only source for summary counts. Keep raw screenshot aspect ratios and distinguish author scalar evidence, author ultrasound measurement, expert reviewed references, and local derived proxies.

### Task 6: Verify, commit, and push

Run the new focused test, the small existing reference/overlay test set, Ruff on changed Python, `git diff --check`, old-path diff checks, and CLI reconstruction checks. Inspect all generated outputs, commit only intended files, push `research/hasebe-reference-evidence-trace`, and verify local/upstream/remote SHA equality with a clean worktree.
