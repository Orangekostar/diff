# AEI Main-Method Identity Reframe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Task-Relevant Information Acquisition the unambiguous paper-level method, compress the manuscript to the required 3+3 evidence structure, and publish a deterministic AEI package without changing any frozen scientific result.

**Architecture:** The immutable 39-row canonical metrics remain the numerical authority. A new method-identity ledger and visibility map govern how canonical claims are distributed across main text, supplement, figures, and tables; manuscript validators and package builders consume that contract instead of requiring all claims in main. The concrete MAVIS implementation and full A3/A4 diagnostics remain evidence, but no longer define the paper-level method.

**Tech Stack:** Python 3.13, pytest, Ruff, CSV/JSON, matplotlib, LaTeX/latexmk, poppler/font tools, Git/GitHub.

---

### Task 1: Freeze P0 and create identity/visibility authorities

**Files:**
- Create: `artifacts/aei_information_hierarchy/MAIN_METHOD_REFRAME_P0_AUDIT.md`
- Create: `artifacts/aei_information_hierarchy/PAPER_METHOD_IDENTITY_LEDGER.md`
- Create: `artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.csv`
- Create: `artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.md`
- Modify: `artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.csv`
- Modify: `artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.md`
- Modify: `tests/test_mavis_aei_paper_review_fixes.py`
- Create: `tests/test_aei_paper_main_method_identity.py`

- [ ] **Step 1: Preserve and verify the P0 audit**

Run:

```bash
git diff --check
sha256sum artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv
git diff --name-only ff4730b3fcf368d6ac43f0f72f034703e1556f7d -- results/p1_full_field_oracle results/p5_sparse_scan results/mvd results/mavis results/mavis_science_closure artifacts/mavis artifacts/mavis_science_closure artifacts/mvd_authority artifacts/mavis_authority
```

Expected: no diff-check output, canonical hash
`f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`,
and an empty frozen-root diff.

- [ ] **Step 2: Write RED identity and visibility tests**

Add assertions that require the 10-row method-identity table, the 14-column
visibility CSV, exact enum values, exact 12/15/1/11 category counts, all 39
claim IDs once in canonical order, A3 supplement-only, and A4 as the single
main system diagnostic.

- [ ] **Step 3: Verify RED**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_review_fixes.py tests/test_aei_paper_main_method_identity.py
```

Expected: failures identify missing identity/visibility files and old map
semantics, with no collection errors.

- [ ] **Step 4: Add the authorities**

Use the prompt-defined 39-row mapping. Preserve `canonical_layer`,
`chronology_class`, and `source_artifact` from canonical/chronology sources;
assign `MAIN_HEADLINE`, `MAIN_SUPPORT`, `MAIN_SYSTEM_DIAGNOSTIC`, or
`SUPPLEMENT_ONLY` exactly once per claim.

- [ ] **Step 5: Verify GREEN and commit**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_review_fixes.py tests/test_aei_paper_main_method_identity.py
git diff --check
git add artifacts/aei_information_hierarchy docs/superpowers/plans/2026-08-28-aei-main-method-identity-reframe.md tests/test_mavis_aei_paper_review_fixes.py tests/test_aei_paper_main_method_identity.py
git commit -m "narrative: add method identity and claim visibility authorities"
```

### Task 2: Replace manuscript structure and headline contracts

**Files:**
- Modify: `tests/test_mavis_aei_paper_manuscript.py`
- Modify: `paper_aei_information_hierarchy/main.tex`
- Modify: `paper_aei_information_hierarchy/MANUSCRIPT_OUTLINE.md`
- Modify: `paper_aei_information_hierarchy/CLAIM_SENTENCE_BANK.md`

- [ ] **Step 1: Write RED manuscript tests**

Require the exact title and six top-level sections; 3 Related Work
subsections; 3 Section-3 subsections; 3 Section-4 subsections; Part I and Part
II with 3 subsubsections each; Engineering Interpretation; the required
abstract terms and exclusions; exactly four figures/two tables; no closest-work
input; A4 once; and A3 absent from main.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_manuscript.py tests/test_aei_paper_main_method_identity.py
```

Expected: failures point to the old title, 4/5/6/6+6 structure, three tables,
and old headline terminology.

- [ ] **Step 3: Rewrite title, abstract, Introduction, and Related Work**

Use the exact title. Keep Introduction at six paragraphs, present RQ-A/RQ-B,
and use three forward-positioned contributions. Remove the closest-work main
table and defensive novelty disclaimers while preserving source-backed
citations and anti-overclaim boundaries.

- [ ] **Step 4: Compress Sections 3 and 4**

Section 3 becomes Characterization, Valuation, and Cost-Constrained
Acquisition. Section 4 becomes Dataset/Representations, Causal Protocol, and
Held-Out-Domain Evaluation/Statistics. Preserve the predictor-conditioned
utility, causal reveal, effective-budget AUEBC, LODO, cost, and statistical
contracts.

- [ ] **Step 5: Verify GREEN and commit in two manuscript stages**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_manuscript.py tests/test_mavis_aei_paper_review_fixes.py tests/test_aei_paper_main_method_identity.py
git diff --check
git add paper_aei_information_hierarchy/main.tex paper_aei_information_hierarchy/MANUSCRIPT_OUTLINE.md paper_aei_information_hierarchy/CLAIM_SENTENCE_BANK.md tests/test_mavis_aei_paper_manuscript.py
git commit -m "manuscript: establish primary method and compressed framework"
```

### Task 3: Compress Results and move full boundaries to supplement

**Files:**
- Modify: `paper_aei_information_hierarchy/main.tex`
- Modify: `paper_aei_information_hierarchy/MANUSCRIPT_OUTLINE.md`
- Modify: `paper_aei_information_hierarchy/CLAIM_SENTENCE_BANK.md`
- Modify: `paper_aei_information_hierarchy/supplementary/supplementary.tex`
- Modify: `tests/test_mavis_aei_paper_manuscript.py`
- Modify: `tests/test_mavis_aei_paper_review_fixes.py`
- Modify: `tests/test_aei_paper_main_method_identity.py`

- [ ] **Step 1: Add RED visibility-placement tests**

Assert Part II opens with O4 dynamic-minus-static; A3 numerical detail is
absent from main and present in supplement; A4 occurs exactly once in main and
its full CI/domain result is in supplement; MAVIS is absent from main headline
surfaces and defined in supplement; all 11 supplement-only claims are covered.

- [ ] **Step 2: Verify RED**

Run the three manuscript/identity test files and confirm old P7/A3 promotion
causes the failures.

- [ ] **Step 3: Rewrite Part I and Part II**

Combine U1/U2, U3/U4, and O2/U5 in Part I. Order Part II as O4/O1, O3/O4 plus
A1, then A2 with one concise A4 diagnostic. Keep every required direction and
scope; do not place P7 in a heading, abstract, table, figure, or conclusion.

- [ ] **Step 4: Rewrite Engineering Interpretation and Conclusions**

Use three to four interpretation paragraphs and exactly three conclusion
paragraphs, ending with the prompt-specified measuring-what-matters sentence.

- [ ] **Step 5: Expand supplement to S1-S6 and verify GREEN**

Retain U1 independent sensitivity, U4 learned boundary, U5 pair details, O1
set regrets, O3 recovery, A3 full feedback result, A4 full static-reference
comparison, positions/history semantics, P14 scope, and chronology.

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_manuscript.py tests/test_mavis_aei_paper_review_fixes.py tests/test_aei_paper_main_method_identity.py
git diff --check
git add paper_aei_information_hierarchy tests/test_mavis_aei_paper_manuscript.py tests/test_mavis_aei_paper_review_fixes.py tests/test_aei_paper_main_method_identity.py
git commit -m "supplement: separate implementation diagnostics from main method"
```

### Task 4: Refocus the four main figures

**Files:**
- Modify: `src/cmc_bbdm/mavis/aei_paper_figures.py`
- Modify: `tests/test_mavis_aei_paper_figures.py`
- Regenerate: `results/aei_information_hierarchy/figures/`
- Regenerate: `paper_aei_information_hierarchy/figures/`

- [ ] **Step 1: Write RED figure tests**

Require Figure 2 to exclude `U4_LEARNED_SPECIFICITY_BOUNDARY`, Figure 3 to
lead with `O4_DYNAMIC_MINUS_STATIC` and contain source controls, Figure 4 to
contain exactly the five A1/A2 rows, and all visible outputs to exclude MAVIS
and `not performance-superior`.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_figures.py
```

Expected: Figure 2 learned-boundary and Figure 4 A3/A4 assertions fail.

- [ ] **Step 3: Update source-row builders and renderers**

Keep the four existing stable stems. Refocus Figure 1 on method identity,
Figure 2 on Part-I visible claims, Figure 3 on dynamic valuation/source
decomposition, and Figure 4 on A1/A2 realization only.

- [ ] **Step 4: Verify deterministic renders and visual integrity**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_figures.py
```

Inspect all generated 300-DPI PNGs for nonblank pixels, clipping, overlap,
panel balance, readable labels, and correspondence with their source CSVs.

- [ ] **Step 5: Commit**

```bash
git add src/cmc_bbdm/mavis/aei_paper_figures.py tests/test_mavis_aei_paper_figures.py results/aei_information_hierarchy/figures paper_aei_information_hierarchy/figures
git commit -m "figures: center task-relevant information acquisition"
```

### Task 5: Reduce main tables and implement visibility-aware validation

**Files:**
- Modify: `src/cmc_bbdm/mavis/aei_paper_tables.py`
- Modify: `src/cmc_bbdm/mavis/aei_paper_validation.py`
- Modify: `tests/test_mavis_aei_paper_tables.py`
- Modify: `tests/test_mavis_aei_paper_validation.py`
- Regenerate: `results/aei_information_hierarchy/tables/`
- Regenerate: `paper_aei_information_hierarchy/tables/`

- [ ] **Step 1: Write RED table/validation tests**

Require exactly two main tables; a compact protocol table; a six-row result
table with the four approved display columns; no A3/A4 or audit columns; and a
`ValidationReport` with canonical/main-visible/combined counts, four figures,
two tables, six sections, no unmatched numbers, no frozen changes, and no
semantic errors.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_tables.py tests/test_mavis_aei_paper_validation.py
```

Expected: old three-table, 12-row, and all-claims-in-main contracts fail.

- [ ] **Step 3: Implement two-table generation**

Retain provenance hashes in CSV artifacts where useful, but omit them from
visible TeX. Generate the six compressed stages and keep closest-work source
verification as an internal artifact test only.

- [ ] **Step 4: Implement combined-evidence validation**

Validate the 39 canonical IDs against the union of main, supplement, and the
visibility map. Add semantic checks for title, method identity, A3/A4 placement,
two tables, 3+3 results, directions, and forbidden headline terms.

- [ ] **Step 5: Verify GREEN and commit**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_tables.py tests/test_mavis_aei_paper_validation.py tests/test_aei_paper_main_method_identity.py
git diff --check
git add src/cmc_bbdm/mavis/aei_paper_tables.py src/cmc_bbdm/mavis/aei_paper_validation.py tests/test_mavis_aei_paper_tables.py tests/test_mavis_aei_paper_validation.py results/aei_information_hierarchy/tables paper_aei_information_hierarchy/tables
git commit -m "tables: compress main evidence into two task-oriented summaries"
```

### Task 6: Update deterministic packaging

**Files:**
- Modify: `src/cmc_bbdm/mavis/aei_paper_package.py`
- Modify: `tests/test_mavis_aei_paper_package.py`
- Regenerate: `results/aei_information_hierarchy/deterministic_package/`
- Regenerate: `results/aei_information_hierarchy/submission/`

- [ ] **Step 1: Write RED package tests**

Require four PDF figures, two TeX tables, no closest-work TeX in the main
source, only the two actual table-input rewrites, complete flat-source
manifests, required A3/A4 supplementary data, and byte-identical two-run ZIPs.

- [ ] **Step 2: Verify RED**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_package.py
```

- [ ] **Step 3: Update constants and flattening**

Remove closest-work from `_TABLES`; package the protocol and task-relevant
result tables; preserve fixed ZIP timestamps, sorted entries, `BUILD.txt`, and
supplementary source records.

- [ ] **Step 4: Verify GREEN and commit**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_package.py
git diff --check
git add src/cmc_bbdm/mavis/aei_paper_package.py tests/test_mavis_aei_paper_package.py results/aei_information_hierarchy/deterministic_package results/aei_information_hierarchy/submission
git commit -m "package: publish two-table AEI submission source"
```

### Task 7: Compile, review, and complete evidence audits

**Files:**
- Create: `artifacts/aei_information_hierarchy/MAIN_METHOD_REFRAME_COMPLETION_AUDIT.md`
- Create: `artifacts/aei_information_hierarchy/CODEX_HANDOFF_MAIN_METHOD_REFRAME.md`
- Create: `ccfa-review-reports/2026-08-28-aei-main-method-reframe-review.md`
- Regenerate: `results/aei_information_hierarchy/submission/AEI_PAPER1_MANUSCRIPT.pdf`
- Regenerate: `results/aei_information_hierarchy/submission/AEI_PAPER1_SUPPLEMENTARY.pdf`

- [ ] **Step 1: Build main and supplement**

Run `latexmk -pdf -interaction=nonstopmode -halt-on-error` from the main paper
and supplement directories, then compile the flat submission source. Build
twice so cross-references and bibliography are stable.

- [ ] **Step 2: Inspect PDF and log quality**

Require embedded fonts, resolved references/citations, zero overfull/underfull
warnings, nonblank rendered pages, correct table/figure placement, and readable
labels at manuscript scale.

- [ ] **Step 3: Perform scientific review and integrity audit**

Review the revised paper against the identity ledger, visibility map, canonical
metrics, chronology, closest-work evidence, and supplement. Record only
evidence-backed strengths, risks, and factual unresolved issues.

- [ ] **Step 4: Run complete regression suites**

```bash
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_aei_paper_*.py tests/test_aei_paper_main_method_identity.py
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mavis_*.py
PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q tests/test_mvd_*.py
/home/ww/miniconda3/bin/python -m ruff check src/cmc_bbdm/mavis/aei_paper_figures.py src/cmc_bbdm/mavis/aei_paper_tables.py src/cmc_bbdm/mavis/aei_paper_validation.py src/cmc_bbdm/mavis/aei_paper_package.py tests/test_mavis_aei_paper_*.py tests/test_aei_paper_main_method_identity.py
/home/ww/miniconda3/bin/python -m ruff format --check src/cmc_bbdm/mavis/aei_paper_figures.py src/cmc_bbdm/mavis/aei_paper_tables.py src/cmc_bbdm/mavis/aei_paper_validation.py src/cmc_bbdm/mavis/aei_paper_package.py tests/test_mavis_aei_paper_*.py tests/test_aei_paper_main_method_identity.py
git diff --check
```

Run the complete MVA suite only in its registered adjacent-Prompt layout and
record the actual command/count; do not infer its result from the worktree.

- [ ] **Step 5: Complete the audit and handoff**

Record page counts and SHA-256 values for main PDF, supplement PDF, source ZIP,
and canonical CSV; list every changed file and exact verification result; prove
the frozen-root diff is empty and no training occurred.

### Task 8: Commit, push, and verify the GitHub handoff

**Files:**
- Modify: `artifacts/aei_information_hierarchy/CODEX_HANDOFF_MAIN_METHOD_REFRAME.md`
- Modify: `artifacts/aei_information_hierarchy/MAIN_METHOD_REFRAME_COMPLETION_AUDIT.md`

- [ ] **Step 1: Final verification before the completion commit**

Run the paper validation, all scoped tests, Ruff, LaTeX/log/font checks,
deterministic two-run replay, canonical hash check, frozen-root diff, and
`git diff --check`. Update the two audit files only with observed results.

- [ ] **Step 2: Commit all final generated artifacts**

```bash
git add paper_aei_information_hierarchy results/aei_information_hierarchy artifacts/aei_information_hierarchy ccfa-review-reports src/cmc_bbdm/mavis tests docs/superpowers/plans/2026-08-28-aei-main-method-identity-reframe.md
git commit -m "audit: finalize AEI main-method identity reframe"
```

- [ ] **Step 3: Push without merging or force-pushing**

```bash
git push -u origin aei-main-method-reframe
```

- [ ] **Step 4: Verify remote identity and required tree paths**

```bash
git rev-parse HEAD
git ls-remote origin refs/heads/aei-main-method-reframe
git status --short --branch
git ls-tree -r --name-only origin/aei-main-method-reframe
```

Expected: local and remote SHA match, worktree is clean, and every prompt-listed
manuscript, authority, code, test, audit, and handoff file exists remotely.
