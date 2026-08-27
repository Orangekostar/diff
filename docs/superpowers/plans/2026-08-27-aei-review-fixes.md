# AEI Nature-Style Review Fixes Implementation Plan

**Goal:** Deliver a source-backed, hash-bound no-new-model revision from the
frozen AEI Paper 1 evidence.

**Runtime:** `PYTHONPATH=src /home/ww/miniconda3/bin/python`.

## Task 1: Lock review contracts with RED tests

- Add semantic tests for normalized AUEBC notation, chronology coverage,
  closest-work coverage, predictor-quality boundaries, transfer conditions,
  external-data roles, three-table packaging, and frozen paths.
- Run the focused review-fix tests and record expected failures.

## Task 2: Add audit and chronology authorities

- Create the P0 audit, AUEBC audit, 39-row chronology CSV/Markdown, and external
  feasibility report.
- Extend validation without changing historical metric code or results.

## Task 3: Generate closest-work Table 1

- Add six verified primary-source rows and a bounded positioning report.
- Generate CSV, booktabs LaTeX, and caption; renumber protocol/evidence tables.

## Task 4: Revise manuscript and paper wording

- Correct Section 3.4 AUEBC notation.
- Add Section 4.6 chronology prose.
- Sharpen Section 2 novelty, narrow Section 5.1.4 predictor wording, and add
  five transfer conditions plus literature-only convergent evidence in 5.4.
- Update outline, sentence bank, references, and paper-specific wording.

## Task 5: Rebuild and validate the package

- Regenerate canonical paper authorities, all three tables, working source,
  supplement, flat source, PDFs, ZIP, manifests, and checksums.
- Verify deterministic replay and frozen historical paths.

## Task 6: Final scientific and submission review

- Run paper, MAVIS, MVD, and authority-MVA tests; Ruff and diff checks.
- Inspect rendered table placement, fonts, warnings, overfull boxes, chronology,
  claim calibration, citation support, and package completeness.
- Create completion audit, commit, push feature branch, integrate into `main`,
  push, and verify remote commit and artifacts.
