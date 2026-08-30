# AEI Visual Narrative Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Mechanical
> workers may execute only explicitly bounded leaf transformations; the primary
> agent owns semantics, integration, review, and completion.

**Goal:** Consolidate the AEI paper from five figures/two main tables to four
figures/one main table, migrate reviewer-visible reconstruction roles to the
bounded field-content terminology, and publish deterministic paper/package and
independent-panel PNG artifacts without changing science.

**Architecture:** Immutable canonical evidence feeds redistributed figure
source-row builders and four live Matplotlib renderers. Governance maps define
claim placement before rendering. Packaging includes only Figure 1-4 and Table
1, while the internal Table 2 generator remains provenance-only. Panel PNGs are
lossless crops of final 300-dpi figure rasters using alignment geometry.

**Tech Stack:** Python 3.13, pytest, Ruff, Matplotlib, Pillow, CSV/JSON, LaTeX,
Poppler/font QA tools, Git/GitHub.

---

### Task 1: Freeze preflight and governance placement

**Files:**
- Modify: `artifacts/aei_information_hierarchy/PAPER_METHOD_IDENTITY_LEDGER.md`
- Modify: `artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.csv`
- Modify: `artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.md`
- Modify: `artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.csv`
- Modify: `artifacts/aei_information_hierarchy/PAPER_CLAIM_MAP.md`
- Modify: `artifacts/aei_information_hierarchy/MAIN_METHOD_REFRAME_VISUAL_CONTRACT.md`
- Modify: `artifacts/aei_information_hierarchy/FIGURE_SOURCE_MAP.csv`
- Modify: `tests/test_aei_paper_main_method_identity.py`

- [ ] Verify exact base, canonical SHA, empty frozen diff, and clean worktree.
- [ ] Write RED identity tests for field-content visible names, unchanged frozen
      entities, A4=`figure4d`, A3=`none`, no main Table 2, and 12/15/1/11 counts.
- [ ] Run the identity test and confirm failures describe old placement/naming.
- [ ] Update all maps in canonical order using the prompt-defined panel mapping;
      change no claim ID, chronology, direction, scope, source, or visibility.
- [ ] Run identity tests and `git diff --check` GREEN.

### Task 2: Define Figure 2 WHAT contract and implementation

**Files:**
- Modify: `tests/test_mavis_aei_paper_figures.py`
- Modify: `src/cmc_bbdm/mavis/aei_paper_figures.py`

- [ ] Write RED tests requiring only Figure 1-4 sources and Figure 2 bindings
      for U1/U2/U3/U4, state manifest, oracle parquet, and all six panels.
- [ ] Confirm RED against the standalone Figure 5 and old O2/U5 Figure 2 rows.
- [ ] Refactor `_figure2_rows` to merge Figure 5 bindings and remove O2/U5.
- [ ] Render the specified 2-by-3 WHAT panels using only existing loaders and
      canonical rows; retain same-state top-five and paired percentile semantics.
- [ ] Remove `_figure5_rows`, its live renderer/registry entries, stem, source
      name, and unreachable legacy main renderers.
- [ ] Run Figure 2 source, render, alignment, nonblank, editable-SVG, and
      deterministic tests GREEN; inspect the standalone PNG at final size.

### Task 3: Define Figure 3 WHEN contract and implementation

**Files:**
- Modify: `tests/test_mavis_aei_paper_figures.py`
- Modify: `src/cmc_bbdm/mavis/aei_paper_figures.py`

- [ ] Extend RED tests to require O2/U5/O4-dynamic/O1 and initial/later/action
      sources in Figure 3, and to exclude O3 source controls.
- [ ] Move O2 checkpoint rows and U5 rows from Figure 2 to `_figure3_rows`.
- [ ] Render initial/later priority maps, stored path, O2 curves,
      dynamic-static forest, and predictor dependence in a 2-by-3 grid.
- [ ] Preserve one percentile scale, the negative-favors-dynamic sign note, and
      unequal-MLP-accuracy boundary.
- [ ] Run focused Figure 3 tests and standalone visual/alignment QA GREEN.

### Task 4: Define Figure 4 HOW contract and implementation

**Files:**
- Modify: `tests/test_mavis_aei_paper_figures.py`
- Modify: `src/cmc_bbdm/mavis/aei_paper_figures.py`

- [ ] Extend RED tests to require O3/O4-shuffled/A1/A2/A4 in Figure 4 and A3
      absent, with A4 direction/CI unchanged.
- [ ] Change `_figure4_rows` to accept `root`, move matched control rows from
      Figure 3, retain A1/A2, and append A4.
- [ ] Render 2-by-2 panels for controls, substitutions, set realization, and
      subordinate deployment calibration.
- [ ] Use `Field-content control` visibly while retaining the source row's
      frozen `control_mode="reconstruction"` semantics.
- [ ] Run focused Figure 4 tests and inspect adverse directions/zero lines.

### Task 5: Finalize Figure 1 and independent panel PNGs

**Files:**
- Modify: `src/cmc_bbdm/mavis/aei_paper_figures.py`
- Modify: `tests/test_mavis_aei_paper_figures.py`
- Generate: `paper_aei_information_hierarchy/panel_pngs/`

- [ ] Keep Figure 1 conceptual and free of performance values; make WHY and the
      Part I/Part II/legal-state flow explicit.
- [ ] Add deterministic alignment-geometry panel cropping with no resizing.
- [ ] Export Figure 1 full panel, Figure 2 a-f, Figure 3 a-f, Figure 4 a-d, and
      Supplementary S1 a-l under organized subdirectories.
- [ ] Write a panel manifest containing figure, panel, source PNG, dimensions,
      aspect ratio, SHA-256, and bytes; test exact roster and nonblank images.
- [ ] Inspect the S1 panel exports to confirm the previous horizontal stretching
      is absent.

### Task 6: Synchronize manuscript, Chinese draft, supplement, and ledgers

**Files:**
- Modify: `paper_aei_information_hierarchy/main.tex`
- Modify: `paper_aei_information_hierarchy/MANUSCRIPT_OUTLINE.md`
- Modify: `paper_aei_information_hierarchy/MANUSCRIPT_CHINESE_DRAFT.md`
- Modify: `paper_aei_information_hierarchy/CLAIM_SENTENCE_BANK.md`
- Modify: `paper_aei_information_hierarchy/README.md`
- Modify: `paper_aei_information_hierarchy/supplementary/supplementary.tex`
- Modify: `tests/test_mavis_aei_paper_manuscript.py`

- [ ] Write RED manuscript tests for Figure 1-4, Table 1 only, explicit absence
      of Figure 5/Table 2, new 5.1.2 heading, and bounded field-content terms.
- [ ] Update abstract/Introduction/Related Work/Framework/Design/Results and
      captions while preserving every number, formula, direction, and section
      structure.
- [ ] Remove the standalone Figure 5 block and Table 2 input; map 5.1.3/5.2.1/
      5.2.2/5.2.3 references to Figure 3/3e/4a-b/4c-d.
- [ ] Synchronize the complete Chinese comparison draft and supporting docs.
- [ ] Keep supplement provenance and internal reconstruction identifiers; add
      one explicit visible-role-to-frozen-identifier note only if needed.
- [ ] Run manuscript/identity tests GREEN and audit all 39 claims.

### Task 7: Enforce 4/1 validation and package cleanup

**Files:**
- Modify: `src/cmc_bbdm/mavis/aei_paper_package.py`
- Modify: `src/cmc_bbdm/mavis/aei_paper_validation.py`
- Modify: `tests/test_mavis_aei_paper_package.py`
- Modify: `tests/test_mavis_aei_paper_validation.py`
- Regenerate: `results/aei_information_hierarchy/figures/`
- Regenerate: `results/aei_information_hierarchy/deterministic_package/`
- Regenerate: `paper_aei_information_hierarchy/figures/`
- Regenerate: `paper_aei_information_hierarchy/tables/`

- [ ] Write RED validation/package tests for four figures, one table, absence of
      stale Figure 5/Table 2, flat Table 1 rewrite only, and manifest completeness.
- [ ] Set package registries to Figure 1-4/Table 1 and add old current Figure 5
      and Table 2 to enumerated stale cleanup.
- [ ] Update semantic validation without weakening canonical, chronology,
      visibility, adverse-sign, or frozen-path gates.
- [ ] Keep `aei_paper_tables.py` and its tests unchanged unless a real internal
      provenance failure requires a minimal adjustment.
- [ ] Regenerate current assets and verify materialized and flat trees contain
      no Figure 5 or Table 2.

### Task 8: Full verification and deterministic publication

**Files:**
- Update: `docs/superpowers/visual-composer/aei-information-hierarchy-visual-contract.md`
- Update: `docs/superpowers/visual-composer/aei-information-hierarchy-qa-ledger.md`
- Update: `docs/superpowers/visual-composer/aei-information-hierarchy-iteration-log.md`
- Generate: `results/aei_information_hierarchy/submission/`

- [ ] Run the required six-file paper/figure pytest suite and table tests only
      if the table generator changed; record the exact pass count.
- [ ] Run Ruff and format checks on every changed Python file.
- [ ] Render into two independent temporary directories and byte-compare PDF,
      SVG, PNG, source CSV, caption, alignment, and checksum files.
- [ ] Run the existing source validator, alignment auditor, minimum-font/text
      audit, and collision audit; inspect any warning overlays manually.
- [ ] Build main, supplement, and flat source with `latexmk`; require zero errors,
      undefined refs/citations, overfull boxes, and underfull boxes.
- [ ] Build the deterministic ZIP twice and verify byte-identical archive and
      manifest; record PDF pages and all required SHA-256 values.
- [ ] Recheck canonical SHA, `changed_frozen_paths == []`, adverse directions,
      exact 4/1 counts, and `git diff --check`.

### Task 9: Commit, handoff, push, and remote verification

**Files:**
- Create: `artifacts/aei_information_hierarchy/CODEX_HANDOFF_VISUAL_NARRATIVE_CONSOLIDATION.md`

- [ ] Review full diff, generated artifacts, and frozen-root name-only diff.
- [ ] Commit all implementation and current paper results as
      `figures: consolidate AEI visual narrative`; record SHA.
- [ ] Write the handoff with exact changed paths, panel sources, terminology
      boundary, Table 2 disposition, tests/QA/build evidence, hashes, boundaries,
      and implementation SHA.
- [ ] Commit handoff as `docs: record AEI visual narrative consolidation`.
- [ ] Push `aei-visual-narrative-consolidation` without force, PR, or merge.
- [ ] Verify local HEAD equals `git ls-remote` remote HEAD, required files exist
      on the remote branch, and the worktree is clean.
