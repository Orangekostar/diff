# AEI Positive Progressive Narrative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a two-part, 12-stage AEI narrative and deterministic paper package without changing any canonical result.

**Architecture:** A new 39-row narrative map joins the immutable canonical metrics and chronology authorities. Manuscript, visuals, Table 3, validation, and packaging adopt the same stage contract while historical evidence remains read-only.

**Tech Stack:** Python 3.13, pytest, matplotlib, CSV/JSON, LaTeX/latexmk, Git.

---

### Task 1: Freeze audit and narrative authority

**Files:**
- Create: `artifacts/aei_information_hierarchy/POSITIVE_NARRATIVE_P0_AUDIT.md`
- Create: `artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.csv`
- Create: `artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.md`
- Test: `tests/test_mavis_aei_paper_review_fixes.py`

- [ ] **Step 1: Write failing narrative-map tests**

Add tests that load the canonical and narrative CSVs, require all 39 claim IDs
exactly once, require the 13 specified columns, and require stage order
`I1..I6, II1..II6`.

- [ ] **Step 2: Verify RED**

Run:
```bash
PYTHONPATH=src python -m pytest -q tests/test_mavis_aei_paper_review_fixes.py
```
Expected: failure because `PAPER_POSITIVE_NARRATIVE_MAP.csv` is absent.

- [ ] **Step 3: Add the 39-row map**

Populate the exact prompt-defined stage assignment, canonical layer,
chronology, role, boundary, source, figure, table, and manuscript location for
each claim. Generate the Markdown table from the same row order.

- [ ] **Step 4: Verify GREEN and immutability**

Run the scoped test and:
```bash
sha256sum artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv
```
Expected hash:
`f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`.

- [ ] **Step 5: Commit**

```bash
git add artifacts/aei_information_hierarchy docs/superpowers tests/test_mavis_aei_paper_review_fixes.py
git commit -m "audit: map canonical evidence to progressive narrative"
```

### Task 2: Replace manuscript structure contracts

**Files:**
- Modify: `tests/test_mavis_aei_paper_manuscript.py`
- Modify: `tests/test_mavis_aei_paper_validation.py`
- Modify: `paper_aei_information_hierarchy/main.tex`
- Modify: `paper_aei_information_hierarchy/MANUSCRIPT_OUTLINE.md`
- Modify: `paper_aei_information_hierarchy/CLAIM_SENTENCE_BANK.md`
- Modify: `paper_aei_information_hierarchy/README.md`
- Modify: `paper_aei_information_hierarchy/supplementary/supplementary.tex`
- Modify: `artifacts/aei_information_hierarchy/AEI_SCOPE_AND_STRUCTURE_LEDGER.md`

- [ ] **Step 1: Write failing structure tests**

Require the preferred title, exactly two RQs, exactly two primary experimental
parts, both six-stage orders, O2 in Part I, O1 as a static reference, U/O/A as
validation criteria, position/history wording, P14 metric wording, three
conclusion paragraphs, and no failure-oriented heading.

- [ ] **Step 2: Verify RED**

Run the manuscript and validation tests. Expected failures must point to the
old title/RQ/headings and not to import or fixture errors.

- [ ] **Step 3: Rewrite manuscript sources**

Keep all six numbered sections, equations, citations, cohort/protocol details,
claim comments, uncertainty, directions, transfer conditions, and limitations.
Reorder evidence into the two six-stage parts and replace Figure/Table names.

- [ ] **Step 4: Verify GREEN**

Run manuscript, review-fix, and validation tests; scan all headings for the
prompt's risky terms.

- [ ] **Step 5: Commit**

```bash
git add paper_aei_information_hierarchy artifacts/aei_information_hierarchy/AEI_SCOPE_AND_STRUCTURE_LEDGER.md tests/test_mavis_aei_paper_manuscript.py tests/test_mavis_aei_paper_validation.py
git commit -m "paper: restructure evidence into two progressive parts"
```

### Task 3: Redesign four evidence figures

**Files:**
- Modify: `tests/test_mavis_aei_paper_figures.py`
- Modify: `src/cmc_bbdm/mavis/aei_paper_figures.py`

- [ ] **Step 1: Write failing figure tests**

Require the four new stems, Part I/II claim allocations, no performance values
in Figure 1, explicit acquired-position/history source labeling, 300-DPI
nonblank PNGs, vector outputs, traceable source rows, and two-run byte identity.

- [ ] **Step 2: Verify RED**

Run the figure test file. Expected failures must show old stems and old claim
allocation.

- [ ] **Step 3: Implement the visual contracts**

Build four source CSVs from canonical evidence and the narrative map; redraw
the framework and three evidence figures with accessible semantic colors,
markers/hatches, stable dimensions, and evidence-bounded captions.

- [ ] **Step 4: Verify GREEN and inspect renders**

Run the figure tests and inspect all four 300-DPI previews at final paper size
for clipping, overlaps, contrast, label size, and nonblank pixels.

- [ ] **Step 5: Commit**

```bash
git add src/cmc_bbdm/mavis/aei_paper_figures.py tests/test_mavis_aei_paper_figures.py
git commit -m "figures: render progressive acquisition evidence"
```

### Task 4: Replace Table 3 and semantic validation

**Files:**
- Modify: `tests/test_mavis_aei_paper_tables.py`
- Modify: `tests/test_mavis_aei_paper_validation.py`
- Modify: `src/cmc_bbdm/mavis/aei_paper_tables.py`
- Modify: `src/cmc_bbdm/mavis/aei_paper_validation.py`

- [ ] **Step 1: Write failing table and semantic tests**

Require the 12-row stage order, new filename/columns, complete 39-claim
coverage, explicit boundaries, numeric-direction checks, canonical SHA-256,
and the new manuscript semantic contract.

- [ ] **Step 2: Verify RED**

Run table and validation tests. Expected failures must identify the old
13-row U/O/A table and old semantic phrases.

- [ ] **Step 3: Implement Table 3 and validators**

Generate rows I1-I6 and II1-II6 from canonical claims; preserve source hashes
and authority hash. Update semantic validation to inspect RQs, stages,
positions/history, P14 metric, headings, mapping uniqueness, and adverse signs.

- [ ] **Step 4: Verify GREEN**

Run table/validation tests and `git diff --check`.

- [ ] **Step 5: Commit**

```bash
git add src/cmc_bbdm/mavis/aei_paper_tables.py src/cmc_bbdm/mavis/aei_paper_validation.py tests/test_mavis_aei_paper_tables.py tests/test_mavis_aei_paper_validation.py
git commit -m "tables: encode the twelve-stage evidence chain"
```

### Task 5: Update deterministic packaging

**Files:**
- Modify: `tests/test_mavis_aei_paper_package.py`
- Modify: `src/cmc_bbdm/mavis/aei_paper_package.py`

- [ ] **Step 1: Write failing package tests**

Require new figure/table names in working and flat trees, local-input rewrites,
complete manifests, and byte-identical ZIPs.

- [ ] **Step 2: Verify RED**

Run the package test file. Expected failures must show old packaged names.

- [ ] **Step 3: Update package constants and flattening**

Replace the four figure stems and Table 3 input; retain flat source behavior,
fixed ZIP timestamps, sorted entries, and supplementary data roster.

- [ ] **Step 4: Verify GREEN**

Run package tests twice through pytest and compare generated archive hashes.

- [ ] **Step 5: Commit**

```bash
git add src/cmc_bbdm/mavis/aei_paper_package.py tests/test_mavis_aei_paper_package.py
git commit -m "package: publish renamed progressive paper assets"
```

### Task 6: Regenerate, compile, audit, and review

**Files:**
- Regenerate: `results/aei_information_hierarchy/`
- Regenerate: `paper_aei_information_hierarchy/figures/`
- Regenerate: `paper_aei_information_hierarchy/tables/`
- Create: `artifacts/aei_information_hierarchy/POSITIVE_NARRATIVE_COMPLETION_AUDIT.md`
- Create: `ccfa-review-reports/2026-08-27-aei-positive-progressive-fresh-review.md`

- [ ] **Step 1: Regenerate evidence assets and package**

Run the paper figure/table builders, materialize assets, build manuscript and
supplement PDFs, then build the deterministic package twice.

- [ ] **Step 2: Compile twice and inspect**

Use `latexmk -pdf -interaction=nonstopmode -halt-on-error`; verify stable page
count, resolved references, no overfull/underfull boxes, and readable figure and
table pages.

- [ ] **Step 3: Run complete integrity checks**

Verify the canonical hash, 39 mappings, all adverse signs, frozen-path diff,
source hashes, citations, headings, risky-word placement, PDF metadata, and ZIP
byte identity. Record every completion checkbox with command evidence.

- [ ] **Step 4: Perform a fresh review**

Review only the revised manuscript/supplement, canonical metrics, chronology,
new narrative map, and closest-work materials. Do not consult old review scores
or reports.

- [ ] **Step 5: Run full test and lint suites**

```bash
PYTHONPATH=src python -m pytest -q tests/test_mavis_aei_paper_evidence.py tests/test_mavis_aei_paper_manuscript.py tests/test_mavis_aei_paper_review_fixes.py tests/test_mavis_aei_paper_figures.py tests/test_mavis_aei_paper_tables.py tests/test_mavis_aei_paper_validation.py tests/test_mavis_aei_paper_package.py
python -m ruff check src/cmc_bbdm/mavis/aei_paper_figures.py src/cmc_bbdm/mavis/aei_paper_tables.py src/cmc_bbdm/mavis/aei_paper_validation.py src/cmc_bbdm/mavis/aei_paper_package.py tests/test_mavis_aei_paper_*.py
git diff --check
```
Expected: all tests and lint checks pass with no diff-check output.

- [ ] **Step 6: Final commit and branch publication**

```bash
git add paper_aei_information_hierarchy results/aei_information_hierarchy artifacts/aei_information_hierarchy ccfa-review-reports
git commit -m "audit: finalize positive narrative paper package"
git push -u origin aei-positive-progressive-narrative
```
