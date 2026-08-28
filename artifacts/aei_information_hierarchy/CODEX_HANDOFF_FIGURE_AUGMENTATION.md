# Codex Handoff: AEI Figure Augmentation

## 1. Repository identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Branch: `aei-main-method-reframe`
- User-referenced scientific base: `a21f84f583a3767f727aeace4c38ae7be3f880ee`
- Operational task baseline: `b0aeba3f5c2eedda25e5b1c64cfd44d6ae0f4f2c`
- Figure implementation commit: `65c2987d2ceaca132cbffeed71aea6953bf6ccb0`
- Verified remote content commit: `65c2987d2ceaca132cbffeed71aea6953bf6ccb0`

## 2. Figure plan summary

The final package contains five main figures and one supplementary figure. The
visual sequence is now framework, spatial/sparse characterization, state evolution,
valuation/planning realization, and paired task-specific priorities. Figure 6 was
not created because its proposed stage rows combine MAE, AUEBC, rank, turnover,
one-step regret, and set regret without a registered common normalization.

The governing design workflow was the locally installed `nature-figure` skill.
It determined the vector-first export contract, restrained palette, strict panel
alignment, minimum-font audit, collision audit, and final-size visual inspection.

## 3. Exact figures modified or added

- Figure 1: positive two-part framework restyled; internal evidence/audit labels removed.
- Figure 2: replaced by a 2x3 Nature-style composition with aggregate performance,
  real initial/25% sparse states, objective conditioning, state evolution, and
  predictor dependence.
- Figure 3: replaced by a 2x3 composition with dynamic-static evidence, initial and
  later teacher-priority overlays, stored acquisition history, matched controls,
  and source contrasts.
- Figure 4: re-rendered with cleaner component and reachable-set forest plots.
- Figure 5: added paired CAI and normalized-RGB task-priority overlays and difference map.
- Supplementary Figure S1: added a 6x2 cross-domain initial/later priority gallery.
- Chinese comparison draft: Figure 2/3/5 captions synchronized with the revised main text.

## 4. Exact source files

| Figure | Source files | Reader/generator |
| --- | --- | --- |
| Figure 1 | `PAPER_POSITIVE_NARRATIVE_MAP.csv` | `_figure1_rows`, `_render_figure1` |
| Figure 2 | canonical metrics; P1 state manifest/payloads; P9 summary | `_figure2_rows`, `load_reconstructed_state`, `_render_figure2_nature` |
| Figure 3 | canonical metrics; P1 states; P3 action scores; P10/P11 evidence | `_figure3_rows`, `load_priority_state`, `_render_figure3_nature` |
| Figure 4 | canonical A1/A2 metrics and bound P12/P13 artifacts | `_figure4_rows`, `_render_figure4_reframed` |
| Figure 5 | P1 initial state; MVA `oracle_values.parquet`; P14 context | `load_task_priority_maps`, `_render_figure5_nature` |
| Supplementary S1 | P1 states; P3 action scores; scan manifest | `gallery_specimen_roster`, `load_gallery_states`, `_render_supplementary_gallery` |

Exact paths and panel-level roles are recorded in
`artifacts/aei_information_hierarchy/FIGURE_SOURCE_MAP.csv`.

## 5. Generation code

- `src/cmc_bbdm/mavis/aei_paper_figures.py`: all figure source rows, rendering,
  captions, vector/raster export, checksum manifests, and alignment gates.
- `src/cmc_bbdm/mavis/aei_paper_visual_assets.py`: fail-closed state restoration,
  teacher/oracle maps, and deterministic gallery selection.
- `src/cmc_bbdm/mavis/nature_figure_alignment.py`: vendored local copy of the
  `nature-figure` Matplotlib alignment auditor used at render time.
- `src/cmc_bbdm/mavis/aei_paper_package.py`: five main figures and one supplement
  figure in working/submission packages.
- `src/cmc_bbdm/mavis/aei_paper_validation.py`: exact five-figure manuscript contract.

All figures are reproducibly generated as editable SVG, PDF, and 300 dpi PNG.
No PDF was hand edited.

## 6. GO/NO-GO rationale

- GO: Figures 1-5 and Supplementary S1 have real, versioned, traceable sources.
- NO_GO: Figure 6 has no frozen common cross-stage effect scale. Creating one would
  add a new analytical choice and violate the evidence-freeze boundary.
- NO_GO: an imagined raw/full-raster qualitative panel was not substituted for the
  absent original raster. Hash-verified compact reconstructions are used instead.

## 7. Scientific integrity

No scientific result changed. No model was retrained. No canonical metric changed.
No frozen scientific result path changed. A4 numerical direction is unchanged.
The canonical metrics SHA-256 remains
`f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`.
The frozen science diff from `a21f84f...` is empty.

MAVIS/MVD/MVA scientific suites: Not rerun; no scientific/shared training or
evaluation runtime changed in this figure-only paper task.

## 8. Test and build evidence

```text
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_mavis_aei_paper_figures.py \
  tests/test_mavis_aei_paper_visual_assets.py \
  tests/test_mavis_aei_paper_manuscript.py \
  tests/test_mavis_aei_paper_package.py \
  tests/test_mavis_aei_paper_validation.py \
  tests/test_aei_paper_main_method_identity.py
62 passed in 77.52s
```

- Ruff: `python -m ruff check src/cmc_bbdm/mavis tests` passed; all 10
  task-relevant Python files passed `ruff format --check`.
- Figure source validator: ready, 19 PASS and 0 FAIL.
- Alignment: five multi-panel PASS reports at strict 1.5 pt tolerance.
- PDF text: all six outputs auditable, zero runs below 5 pt.
- Collision audit: zero FAIL; four reviewed fill-edge warnings are intentional label/mark adjacency.
- Main PDF: 17 pages, clean LaTeX log, all fonts embedded/subsetted.
- Supplement PDF: 4 pages, clean LaTeX log, all fonts embedded/subsetted.
- Flat source: 17 pages, clean LaTeX log.
- Deterministic ZIP/manifest/SHA replay: exact byte match.

## 9. Generated artifacts

| Path | SHA-256 |
| --- | --- |
| `results/aei_information_hierarchy/submission/AEI_PAPER1_MANUSCRIPT.pdf` | `40f83a2041a16dba27caeebac8a2c79afbb8918bd0b7144d0b1668598c83ccba` |
| `results/aei_information_hierarchy/submission/AEI_PAPER1_SUPPLEMENTARY.pdf` | `153310a80c7e6c64e696e04568f21f24bc8e737262b5c14e616704557ed9fe0f` |
| `results/aei_information_hierarchy/deterministic_package/AEI_PAPER_SUBMISSION_SOURCE.zip` | `18e9e703c7a30d32f8b4bfa2355a3ab6052b95b15f2981a60834e7ff4c08147d` |
| `results/aei_information_hierarchy/deterministic_package/submission_source/SUBMISSION_MANIFEST.csv` | `fedde533a690d2a3dfcdaff994298f8916f87e7abf4364c6827fa22c6d833c2b` |
| `artifacts/aei_information_hierarchy/FIGURE_AUGMENTATION_COMPLETION_AUDIT.md` | Generated in this task; final digest is commit-bound |

## 10. GitHub verification

- Commit command: `git commit -m "figures: strengthen AEI visual evidence and overlay support"`
- Push command: `git push origin aei-main-method-reframe`
- Remote branch: `refs/heads/aei-main-method-reframe`
- Local content SHA: `65c2987d2ceaca132cbffeed71aea6953bf6ccb0`
- Remote content SHA: `65c2987d2ceaca132cbffeed71aea6953bf6ccb0`
- Remote verification: PASS; `git rev-parse HEAD` matched
  `git ls-remote origin refs/heads/aei-main-method-reframe` after the content push.
