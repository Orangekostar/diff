# AEI Signal-Saliency Reframe Implementation Plan

> The primary agent owns evidence semantics, manuscript logic, integration,
> debugging, review, and completion. Mechanical workers may execute only
> explicitly specified, independently verifiable leaf tasks.

**Goal:** Replace the main reconstruction/field-content comparator with frozen
task-agnostic C-scan appearance-saliency evidence, preserve all legacy science
outside the main story, and publish a deterministic bilingual paper package.

**Architecture:** Frozen MVA A2 CSV/Parquet evidence feeds three new canonical
claims and a read-only saliency-map adapter. Canonical visibility drives source
rows, Figure 2/4, English/Chinese prose, supplement provenance, validation, and
the deterministic package. Frozen result trees are read-only.

### Task 1: P0 authority and baseline

- [ ] Record exact base, old canonical SHA/IDs, source hashes, evidence values,
      clean status, frozen diff, and baseline paper test count.
- [ ] Complete the pre-migration section of
      `SIGNAL_SALIENCY_AUTHORITY_MIGRATION_AUDIT.md`.
- [ ] Confirm Route B GO and no new training.

### Task 2: TDD saliency evidence adapter

- [ ] Add RED tests for exact mechanical/appearance method rows, 64 unique
      cells per method, paired domain/grid, independent rank percentiles,
      read-only arrays, and no synthetic fallback.
- [ ] Implement `TaskSaliencyMaps` and `load_task_saliency_maps` without
      changing the legacy reconstruction loader.
- [ ] Run focused asset tests GREEN.

### Task 3: TDD canonical and chronology migration

- [ ] Add RED tests for the three exact new metrics, source hashes, 42 unique
      IDs, and preserved adverse O3/O4/A3/A4 signs.
- [ ] Add frozen MVA A2 sources, extract the registered bootstrap and 276-map
      means, and append new IDs without mutating old rows.
- [ ] Classify new A2 IDs as pre-P7 while preserving legacy U4 chronology.
- [ ] Regenerate canonical, claim-map, source-hash, and chronology authorities.

### Task 4: TDD Figure 2 and Figure 4 migration

- [ ] Add RED source/render tests requiring appearance rather than
      reconstruction for Figure 2 and excluding field-content visible text.
- [ ] Bind Figure 2(c) to the A2 AUEBC claim and Figure 2(d--f) to mechanical
      plus appearance oracle values and A2 map similarity.
- [ ] Remove the O3 reconstruction row and visible comparator from Figure 4(a)
      while retaining history and shuffled adverse directions.
- [ ] Render twice, byte-compare, and inspect final-size panels.

### Task 5: Governance migration

- [ ] Update the 42-row visibility, narrative, chronology, identity, claim,
      figure-source, and visual-contract maps in canonical order.
- [ ] Demote legacy reconstruction rows to supplement-only; do not rename them.
- [ ] Re-run identity, source-binding, and visibility tests.

### Task 6: English, Chinese, supplement, and support docs

- [ ] Rewrite the English main manuscript systematically from Abstract through
      Conclusion while preserving structure, formulas, numbers, and directions.
- [ ] Mirror the English paper in `MANUSCRIPT_CHINESE_DRAFT.md`.
- [ ] Update outline, sentence bank, README, and supplement provenance.
- [ ] Enforce main-visible forbidden terminology and supplement legacy
      traceability tests.

### Task 7: Regeneration and complete verification

- [ ] Regenerate authority, figures, panel PNGs, tables, materialized paper,
      main/supp/flat PDFs, ZIP, and manifests through existing pipelines.
- [ ] Run focused and full paper tests, Ruff, `git diff --check`, and semantic
      validation; record exact counts only.
- [ ] Run source, alignment, font, collision, and final-size visual QA.
- [ ] Build main/supp/flat LaTeX with zero errors, undefined refs/citations,
      overfull boxes, and underfull boxes.
- [ ] Rebuild ZIP twice and require byte identity.
- [ ] Verify frozen science diff is empty and all source hashes remain fixed.

### Task 8: Two commits, handoff, and remote sync

- [ ] Review the complete diff and commit implementation/package changes.
- [ ] Create `CODEX_HANDOFF_SIGNAL_SALIENCY_REFRAME.md` with all 23 required
      fields and final hashes/counts.
- [ ] Commit the handoff, push without force, and verify remote HEAD equals
      local HEAD and the worktree is clean.
