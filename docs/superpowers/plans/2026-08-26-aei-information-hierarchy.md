# AEI Information Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:executing-plans` and stop on any semantic ambiguity.

**Goal:** Build, verify, and publish a hash-bound six-section AEI Paper 1
package from frozen repository evidence without new model training.

**Architecture:** A paper-only Python package reads immutable registered
artifacts, creates canonical evidence rows, renders four figures and two tables,
validates the manuscript, and produces a deterministic checksummed package.

**Tech Stack:** Python 3.11, Polars, NumPy, Matplotlib, Pytest, Ruff, LaTeX.

---

## Task 1: P0 authority and structure audit

**Files:**

- Create `artifacts/aei_information_hierarchy/P0_REPO_AND_EVIDENCE_AUDIT.md`
- Create `artifacts/aei_information_hierarchy/P0_MANUSCRIPT_SOURCE_AUDIT.md`
- Create `artifacts/aei_information_hierarchy/P0_EVIDENCE_CONFLICTS.md`
- Create `artifacts/aei_information_hierarchy/AEI_SCOPE_AND_STRUCTURE_LEDGER.md`
- Create `docs/literature-search-20260826-aei-paper1/*`
- Create `artifacts/aei_information_hierarchy/PAPER_DEVELOPMENT_LEDGER.md`

- [x] Record exact base, inventory, baseline tests, immutable namespaces, and
  missing manuscript source.
- [x] Reconcile B/I full-field semantics, P5 interval semantics, and P10's
  separate I-field endpoint.
- [x] Record official AEI scope and at least ten recent research-paper structure
  checks without inferring inaccessible headings.
- [x] Run `git diff --check`, Ruff, and the paper-audit tests once added.
- [x] Commit `audit: map AEI paper evidence and manuscript starting state`.

## Task 2: Canonical authority with RED tests

**Files:**

- Create `src/cmc_bbdm/mavis/aei_paper_evidence.py`
- Create `tests/test_mavis_aei_paper_evidence.py`
- Create `artifacts/aei_information_hierarchy/EVIDENCE_AUTHORITY_RECONCILIATION.md`
- Create `artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv`
- Create `artifacts/aei_information_hierarchy/PAPER_CLAIM_MAP.md`
- Create `artifacts/aei_information_hierarchy/PAPER_SOURCE_HASHES.csv`

- [x] Write failing tests for exact required source files, SHA-256 binding,
  unique claim IDs, contrast directions, interval ordering, six-domain counts,
  B/I method semantics, P5 dual contrasts, and oracle/deployable labels.
- [x] Implement strict readers and canonical rows for all RQ1-RQ3 claims.
- [x] Recompute only registered deterministic aggregations from frozen rows and
  reject any mismatch with serialized authority beyond tolerance.
- [x] Commit the semantic reconciliation and canonical authority in separate
  evidence commits.

## Task 3: Figure data and rendering with RED tests

**Files:**

- Create `src/cmc_bbdm/mavis/aei_paper_figures.py`
- Create `tests/test_mavis_aei_paper_figures.py`
- Create `results/aei_information_hierarchy/figures/*`
- Create `paper_aei_information_hierarchy/figures/*`
- Create `docs/superpowers/visual-composer/aei-information-hierarchy-*`

- [ ] Read the visual-composer contracts before implementation.
- [ ] Write failing tests for four figure IDs, required panels, source-data
  schemas, deterministic ordering, adverse-control visibility, no internal
  stage labels, SVG/PDF/PNG presence, 300 dpi PNG metadata, and nonblank renders.
- [ ] Render Figure 1 hierarchy, Figure 2 usefulness, Figure 3 observability,
  and Figure 4 actionability from canonical data.
- [ ] Inspect desktop-scale PNGs and record visual QA.
- [ ] Commit `figures: add useful-observable-actionable paper figure package`.

## Task 4: Two generated tables with RED tests

**Files:**

- Create `src/cmc_bbdm/mavis/aei_paper_tables.py`
- Create `tests/test_mavis_aei_paper_tables.py`
- Create `results/aei_information_hierarchy/tables/*`
- Create `paper_aei_information_hierarchy/tables/*`

- [ ] Write failing tests for exactly two main tables, stable row order,
  canonical-number equality, cohort/protocol fields, CI direction, domain
  consistency, and oracle/deployable visibility.
- [ ] Generate Table 1 case/protocol and Table 2 hierarchy evidence in CSV and
  LaTeX from canonical authority.
- [ ] Commit `tables: add protocol and evidence hierarchy tables`.

## Task 5: Six-section outline and claim bank

**Files:**

- Create `paper_aei_information_hierarchy/README.md`
- Create `paper_aei_information_hierarchy/MANUSCRIPT_OUTLINE.md`
- Create `paper_aei_information_hierarchy/CLAIM_SENTENCE_BANK.md`
- Create `tests/test_mavis_aei_paper_manuscript.py`

- [ ] Write failing tests for exact section names/order, claim-to-source links,
  forbidden claims, and internal-label exclusion.
- [ ] Map every subsection, figure, table, and claim to canonical evidence.
- [ ] Commit `manuscript: add fixed six-section AEI outline and claim sentence bank`.

## Task 6: Draft Sections 2-4

**Files:**

- Create `paper_aei_information_hierarchy/main.tex`
- Create `paper_aei_information_hierarchy/references.bib`

- [ ] Add only verified primary citations and explicit prior-art boundaries.
- [ ] Draft Related Research and Problem Formulation, the hierarchy framework,
  and the multi-domain case/protocol without results leakage.
- [ ] Keep statistical units, LODO, cost semantics, and non-deployable oracle
  status explicit.
- [ ] Commit `manuscript: draft sections 2-4 from verified evidence and literature`.

## Task 7: Draft Results and Discussion

**Files:**

- Modify `paper_aei_information_hierarchy/main.tex`

- [ ] Draft one integrated Useful-Observable-Actionable evidence argument from
  canonical rows only.
- [ ] Include all central adverse controls and answer each RQ directly.
- [ ] Run numeric provenance lint before commit.
- [ ] Commit `manuscript: draft section 5 from canonical metrics`.

## Task 8: Complete title-facing sections

**Files:**

- Modify `paper_aei_information_hierarchy/main.tex`

- [ ] Draft Introduction, Conclusions, abstract, keywords, and title last.
- [ ] State conceptual generality without claiming external empirical
  generalization.
- [ ] Keep the paper as a mechanism-and-boundary contribution.
- [ ] Commit `manuscript: draft introduction conclusion abstract and title`.

## Task 9: Supplement and deterministic package

**Files:**

- Create `paper_aei_information_hierarchy/supplementary/main.tex`
- Create `paper_aei_information_hierarchy/supplementary/README.md`
- Create `src/cmc_bbdm/mavis/aei_paper_validation.py`
- Create `src/cmc_bbdm/mavis/aei_paper_package.py`
- Create `tests/test_mavis_aei_paper_validation.py`
- Create `results/aei_information_hierarchy/CHECKSUMS.sha256`
- Create `results/aei_information_hierarchy/artifact_manifest.json`

- [ ] Write failing replay, manifest, numeric-lint, frozen-tree, section-count,
  forbidden-claim, and package-completeness tests.
- [ ] Generate a byte-identical replay tree and verify every checksum.
- [ ] Add detailed protocols, domain rows, sensitivity evidence, and complete
  provenance to the supplement.

## Task 10: Submission QA and integration

**Files:**

- Create `artifacts/aei_information_hierarchy/PRE_SUBMISSION_AUDIT.md`
- Update `artifacts/aei_information_hierarchy/PAPER_DEVELOPMENT_LEDGER.md`

- [ ] Read the submission-checker and paper-reviewer instructions.
- [ ] Audit originality, importance, technical soundness, engineering relevance,
  readability, anonymity/template state, references, figures, tables, and
  supplementary package.
- [ ] Run all paper tests, MAVIS tests, MVD tests, full authority MVA tests,
  Ruff, `git diff --check`, LaTeX compile, render QA, numeric provenance lint,
  checksum verification, replay comparison, and P7 frozen-tree verification.
- [ ] Commit `evidence: add deterministic paper package and pre-submission audit`.
- [ ] Push the branch, merge/integrate only after full review, push `main`, and
  verify remote commit/path presence with `git ls-remote` and `git ls-tree`.
