# AEI visual narrative consolidation handoff

## 1. Repository identity

- Repository: `Orangekostar/diff`
- Source branch: `aei-main-method-reframe`
- Exact base commit: `9794d53a9549f2e3501fe482e8db8735f468ba20`
- Task branch: `aei-visual-narrative-consolidation`
- Implementation commit: `e49026a389c6d4d255dde2cdf3d8891440317e58`

## 2. Scientific objective

The main-paper visual narrative is consolidated from five figures and two tables to four figures and one table. The former Figure 5 task-priority evidence is now integrated into Figure 2d-f, and the former main Table 2 is retired. Reader-facing terminology uses `field-content reference/control`, while frozen reconstruction identifiers and control modes remain unchanged. No model training, experiment rerun, metric recomputation, or scientific-value modification was performed.

## 3. Exact files changed

The implementation commit is the authoritative inventory. Status codes are `A` (added), `M` (modified), and `D` (deleted).

```text
M artifacts/aei_information_hierarchy/FIGURE_SOURCE_MAP.csv
M artifacts/aei_information_hierarchy/MAIN_METHOD_REFRAME_VISUAL_CONTRACT.md
M artifacts/aei_information_hierarchy/PAPER_CLAIM_MAP.md
M artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.csv
M artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.md
M artifacts/aei_information_hierarchy/PAPER_METHOD_IDENTITY_LEDGER.md
M artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.csv
M artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.md
A docs/superpowers/plans/2026-08-30-aei-visual-narrative-consolidation.md
A docs/superpowers/specs/2026-08-30-aei-visual-narrative-consolidation-design.md
M docs/superpowers/visual-composer/aei-information-hierarchy-iteration-log.md
M docs/superpowers/visual-composer/aei-information-hierarchy-qa-ledger.md
M docs/superpowers/visual-composer/aei-information-hierarchy-visual-contract.md
M paper_aei_information_hierarchy/CLAIM_SENTENCE_BANK.md
M paper_aei_information_hierarchy/MANUSCRIPT_CHINESE_DRAFT.md
M paper_aei_information_hierarchy/MANUSCRIPT_OUTLINE.md
M paper_aei_information_hierarchy/README.md
M paper_aei_information_hierarchy/figures/figure1_task_relevant_acquisition_framework.pdf
M paper_aei_information_hierarchy/figures/figure2_information_characterization.pdf
M paper_aei_information_hierarchy/figures/figure3_state_conditioned_value.pdf
M paper_aei_information_hierarchy/figures/figure4_valuation_planning_realization.pdf
D paper_aei_information_hierarchy/figures/figure5_task_specific_measurement_priorities.pdf
M paper_aei_information_hierarchy/main.tex
A paper_aei_information_hierarchy/panel_pngs/PANEL_PNG_MANIFEST.csv
A paper_aei_information_hierarchy/panel_pngs/README.md
A paper_aei_information_hierarchy/panel_pngs/figure1/figure1_panel_full.png
A paper_aei_information_hierarchy/panel_pngs/figure2/figure2_panel_a.png
A paper_aei_information_hierarchy/panel_pngs/figure2/figure2_panel_b.png
A paper_aei_information_hierarchy/panel_pngs/figure2/figure2_panel_c.png
A paper_aei_information_hierarchy/panel_pngs/figure2/figure2_panel_d.png
A paper_aei_information_hierarchy/panel_pngs/figure2/figure2_panel_e.png
A paper_aei_information_hierarchy/panel_pngs/figure2/figure2_panel_f.png
A paper_aei_information_hierarchy/panel_pngs/figure3/figure3_panel_a.png
A paper_aei_information_hierarchy/panel_pngs/figure3/figure3_panel_b.png
A paper_aei_information_hierarchy/panel_pngs/figure3/figure3_panel_c.png
A paper_aei_information_hierarchy/panel_pngs/figure3/figure3_panel_d.png
A paper_aei_information_hierarchy/panel_pngs/figure3/figure3_panel_e.png
A paper_aei_information_hierarchy/panel_pngs/figure3/figure3_panel_f.png
A paper_aei_information_hierarchy/panel_pngs/figure4/figure4_panel_a.png
A paper_aei_information_hierarchy/panel_pngs/figure4/figure4_panel_b.png
A paper_aei_information_hierarchy/panel_pngs/figure4/figure4_panel_c.png
A paper_aei_information_hierarchy/panel_pngs/figure4/figure4_panel_d.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_a.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_b.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_c.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_d.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_e.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_f.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_g.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_h.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_i.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_j.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_k.png
A paper_aei_information_hierarchy/panel_pngs/supplementary_figure_s1/supplementary_figure_s1_panel_l.png
M paper_aei_information_hierarchy/supplementary/figures/supplementary_figure_s1_cross_domain_state_priority_gallery.pdf
M paper_aei_information_hierarchy/supplementary/supplementary.tex
D paper_aei_information_hierarchy/tables/table2_task_relevant_results.tex
M results/aei_information_hierarchy/deterministic_package/AEI_PAPER_SUBMISSION_SOURCE.sha256
M results/aei_information_hierarchy/deterministic_package/AEI_PAPER_SUBMISSION_SOURCE.zip
M results/aei_information_hierarchy/deterministic_package/submission_source/SUBMISSION_MANIFEST.csv
M results/aei_information_hierarchy/deterministic_package/submission_source/figure1_task_relevant_acquisition_framework.pdf
M results/aei_information_hierarchy/deterministic_package/submission_source/figure2_information_characterization.pdf
M results/aei_information_hierarchy/deterministic_package/submission_source/figure3_state_conditioned_value.pdf
M results/aei_information_hierarchy/deterministic_package/submission_source/figure4_valuation_planning_realization.pdf
D results/aei_information_hierarchy/deterministic_package/submission_source/figure5_task_specific_measurement_priorities.pdf
M results/aei_information_hierarchy/deterministic_package/submission_source/main.tex
D results/aei_information_hierarchy/deterministic_package/submission_source/table2_task_relevant_results.tex
M results/aei_information_hierarchy/deterministic_package/working/CLAIM_SENTENCE_BANK.md
M results/aei_information_hierarchy/deterministic_package/working/MANUSCRIPT_OUTLINE.md
M results/aei_information_hierarchy/deterministic_package/working/README.md
M results/aei_information_hierarchy/deterministic_package/working/figures/figure1_task_relevant_acquisition_framework.pdf
M results/aei_information_hierarchy/deterministic_package/working/figures/figure2_information_characterization.pdf
M results/aei_information_hierarchy/deterministic_package/working/figures/figure3_state_conditioned_value.pdf
M results/aei_information_hierarchy/deterministic_package/working/figures/figure4_valuation_planning_realization.pdf
D results/aei_information_hierarchy/deterministic_package/working/figures/figure5_task_specific_measurement_priorities.pdf
M results/aei_information_hierarchy/deterministic_package/working/main.tex
M results/aei_information_hierarchy/deterministic_package/working/supplementary/figures/supplementary_figure_s1_cross_domain_state_priority_gallery.pdf
M results/aei_information_hierarchy/deterministic_package/working/supplementary/supplementary.tex
D results/aei_information_hierarchy/deterministic_package/working/tables/table2_task_relevant_results.tex
M results/aei_information_hierarchy/figure_qa/figure1_collisions.json
M results/aei_information_hierarchy/figure_qa/figure1_collisions_overlay.pdf
M results/aei_information_hierarchy/figure_qa/figure2_collisions.json
M results/aei_information_hierarchy/figure_qa/figure2_collisions_overlay.pdf
M results/aei_information_hierarchy/figure_qa/figure3_collisions.json
M results/aei_information_hierarchy/figure_qa/figure3_collisions_overlay.pdf
M results/aei_information_hierarchy/figure_qa/figure4_collisions.json
M results/aei_information_hierarchy/figure_qa/figure4_collisions_overlay.pdf
D results/aei_information_hierarchy/figure_qa/figure5_collisions.json
M results/aei_information_hierarchy/figures/FIGURE_CHECKSUMS.csv
M results/aei_information_hierarchy/figures/figure1_task_relevant_acquisition_framework.csv
M results/aei_information_hierarchy/figures/figure1_task_relevant_acquisition_framework.pdf
M results/aei_information_hierarchy/figures/figure1_task_relevant_acquisition_framework.png
M results/aei_information_hierarchy/figures/figure1_task_relevant_acquisition_framework.svg
M results/aei_information_hierarchy/figures/figure2_information_characterization.alignment.json
M results/aei_information_hierarchy/figures/figure2_information_characterization.alignment.svg
M results/aei_information_hierarchy/figures/figure2_information_characterization.csv
M results/aei_information_hierarchy/figures/figure2_information_characterization.pdf
M results/aei_information_hierarchy/figures/figure2_information_characterization.png
M results/aei_information_hierarchy/figures/figure2_information_characterization.svg
M results/aei_information_hierarchy/figures/figure2_information_characterization_caption.md
M results/aei_information_hierarchy/figures/figure3_state_conditioned_value.csv
M results/aei_information_hierarchy/figures/figure3_state_conditioned_value.pdf
M results/aei_information_hierarchy/figures/figure3_state_conditioned_value.png
M results/aei_information_hierarchy/figures/figure3_state_conditioned_value.svg
M results/aei_information_hierarchy/figures/figure3_state_conditioned_value_caption.md
M results/aei_information_hierarchy/figures/figure4_valuation_planning_realization.alignment.json
M results/aei_information_hierarchy/figures/figure4_valuation_planning_realization.alignment.svg
M results/aei_information_hierarchy/figures/figure4_valuation_planning_realization.csv
M results/aei_information_hierarchy/figures/figure4_valuation_planning_realization.pdf
M results/aei_information_hierarchy/figures/figure4_valuation_planning_realization.png
M results/aei_information_hierarchy/figures/figure4_valuation_planning_realization.svg
M results/aei_information_hierarchy/figures/figure4_valuation_planning_realization_caption.md
D results/aei_information_hierarchy/figures/figure5_task_specific_measurement_priorities.alignment.json
D results/aei_information_hierarchy/figures/figure5_task_specific_measurement_priorities.alignment.svg
D results/aei_information_hierarchy/figures/figure5_task_specific_measurement_priorities.csv
D results/aei_information_hierarchy/figures/figure5_task_specific_measurement_priorities.pdf
D results/aei_information_hierarchy/figures/figure5_task_specific_measurement_priorities.png
D results/aei_information_hierarchy/figures/figure5_task_specific_measurement_priorities.svg
D results/aei_information_hierarchy/figures/figure5_task_specific_measurement_priorities_caption.md
M results/aei_information_hierarchy/submission/AEI_PAPER1_MANUSCRIPT.pdf
M results/aei_information_hierarchy/submission/AEI_PAPER1_SUPPLEMENTARY.pdf
M results/aei_information_hierarchy/supplementary_figures/FIGURE_CHECKSUMS.csv
M results/aei_information_hierarchy/supplementary_figures/supplementary_figure_s1_cross_domain_state_priority_gallery.pdf
M results/aei_information_hierarchy/supplementary_figures/supplementary_figure_s1_cross_domain_state_priority_gallery.png
M results/aei_information_hierarchy/supplementary_figures/supplementary_figure_s1_cross_domain_state_priority_gallery.svg
M src/cmc_bbdm/mavis/aei_paper_figures.py
M src/cmc_bbdm/mavis/aei_paper_package.py
M src/cmc_bbdm/mavis/aei_paper_validation.py
M tests/test_aei_paper_main_method_identity.py
M tests/test_mavis_aei_paper_figures.py
M tests/test_mavis_aei_paper_manuscript.py
M tests/test_mavis_aei_paper_package.py
M tests/test_mavis_aei_paper_validation.py
```

## 4. Final visual contract

- Figure 1, WHY: one unframed framework panel connecting task objective, state-conditioned value, sequential acquisition, and task outcome. Source: registered canonical evidence represented in `figure1_task_relevant_acquisition_framework.csv`.
- Figure 2, WHAT: a 2x3 evidence figure. (a) canonical spatial/sparse characterization; (b) registered state `c8-2`; (c) U3 opportunity; (d) CAI task priority; (e) field-content priority; (f) CAI-versus-field-content difference. Sources: canonical table, registered state manifest/parquet, and frozen task-priority evidence formerly displayed in Figure 5.
- Figure 3, WHEN: a 2x3 state-conditioned-value figure. (a) initial priority; (b) updated priority; (c) acquisition history; (d) turnover/rank/top-5 behavior; (e) dynamic-versus-static outcome; (f) predictor dependence. Sources: P9/state/action records and O2/U5/O4/O1 evidence.
- Figure 4, HOW: a 2x2 realization figure. (a) matched controls O3/O4; (b) substitution A1; (c) planning A2; (d) A4 system-boundary diagnostic.
- Supplementary Figure S1 retains the cross-domain state-priority gallery. Twenty-nine aspect-preserving panel PNGs and a stable manifest are provided for downstream editing and inspection.

## 5. Terminology migration

Reader-visible manuscript, caption, table, and figure language now uses `field-content reference` and `field-content control`. Frozen internal identifiers, reconstruction IDs, schemas, and executable control-mode values were preserved where changing them would alter provenance or compatibility. The operational field-content comparison remains the registered normalized-RGB-MSE surrogate; it is not relabeled as a perceptual or task loss.

## 6. Table 2 disposition

The former main Table 2 is removed from the manuscript, working tree, flat submission source, and deterministic package. Its evidence is represented in the consolidated Figure 2d-f narrative and remains traceable through the evidence/claim maps. The internal table-generation path is retained as provenance-only infrastructure and is no longer a main-paper output.

## 7. Scientific integrity

- Canonical source checksum: `f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`.
- Frozen-path comparison: `changed_frozen_paths == []`.
- Claim validation: 39 registered claims, 27 main-visible claims, 39 combined claims, 4 main figures, 1 main table, 6 sections, 0 unmatched claims, and 0 semantic errors.
- Visibility partition: 12 `MAIN_HEADLINE`, 15 `MAIN_SUPPORT`, 1 `MAIN_SYSTEM_DIAGNOSTIC`, and 11 `SUPPLEMENT_ONLY`.
- Adverse O3/O4/A4 signs remain visible. A4 remains a system diagnostic favoring the static reference.
- No training, dataset replacement, experiment rerun, scientific-value recomputation, or frozen-result mutation occurred.

## 8. Tests

Environment note: the explicit `PYTHONPATH` and `python -S` prevent an unrelated site `.pth` entry from shadowing this repository's namespace.

```bash
PYTHONPATH="$PWD/src:$CONDA_PREFIX/lib/python3.13/site-packages" python -S -m pytest -q -p no:cacheprovider \
  tests/test_aei_paper_main_method_identity.py \
  tests/test_mavis_aei_paper_evidence.py \
  tests/test_mavis_aei_paper_figures.py \
  tests/test_mavis_aei_paper_manuscript.py \
  tests/test_mavis_aei_paper_package.py \
  tests/test_mavis_aei_paper_review_fixes.py \
  tests/test_mavis_aei_paper_tables.py \
  tests/test_mavis_aei_paper_validation.py \
  tests/test_mavis_aei_paper_visual_assets.py
```

Result: `126 passed in 112.16s`.

```bash
python -m ruff check src/cmc_bbdm/mavis tests
python -m ruff format --check src/cmc_bbdm/mavis/aei_paper_figures.py src/cmc_bbdm/mavis/aei_paper_package.py src/cmc_bbdm/mavis/aei_paper_validation.py tests/test_aei_paper_main_method_identity.py tests/test_mavis_aei_paper_figures.py tests/test_mavis_aei_paper_manuscript.py tests/test_mavis_aei_paper_package.py tests/test_mavis_aei_paper_validation.py
```

Results: Ruff checks passed; all eight changed Python files are formatted.

## 9. Figure QA

- Physical-size review: manuscript Figures 1-4 and Supplementary Figure S1 were rendered and visually inspected; no clipping or incoherent overlap was found.
- Alignment: Figure 2 passed 7 comparisons across 6 panels; Figure 3 passed 7 across 6; Figure 4 passed 4 across 4; S1 passed 8 across 12. Total: 26 comparisons, 0 warnings, 0 exemptions.
- Collision QA: Figures 1, 3, 4, and S1 passed. Figure 2 has three reviewed warnings caused by direct numeric labels intentionally touching their own point markers; all collision reports have 0 failures.
- Source preflight: 18 pass, 3 reviewed warnings, 0 failures. The warnings reflect the repository contract of PDF/SVG plus 300-dpi PNG rather than TIFF/600-dpi output, and a width constant used by the generator.
- PDF text audit: 5 PDFs checked, minimum glyph size at least 5 pt, 0 glyphs below threshold.
- Panel manifest: 29 rows; every SHA-256 verified; source paths are relative and deterministic.

## 10. Builds and artifacts

- Main manuscript: `results/aei_information_hierarchy/submission/AEI_PAPER1_MANUSCRIPT.pdf`, 18 pages, SHA-256 `6c5df7e6ef09c77b42d2c1466c875876dfcfab38c0f34edaef3da88f84a44085`.
- Supplement: `results/aei_information_hierarchy/submission/AEI_PAPER1_SUPPLEMENTARY.pdf`, 4 pages, SHA-256 `28785e4688e43f763294937c62031b04071f5d8228a8594da4eba8dc9cd1e62f`.
- Deterministic submission ZIP: SHA-256 `93b77a91b9c0c9287902f262bdd62704c1fa076fe320d12fa21cc18c4205203a`; independent two-directory replay compared 113 files successfully.
- Submission manifest: SHA-256 `0b003e0b0b8ba57ee1b63763cd381b0546846aa70d7dc909742686d1243e7d29`.
- Main figure checksum manifest: SHA-256 `f36dde54eeaa752fdda0ca9746e2180ea8f4c8e43a57c37139fb7e919a2c9bd6`.
- Panel PNG manifest: SHA-256 `6e66da4201d75c330772affa0a51250461b83be88ce4b2453ae1002b4b0b464d`.
- Working, supplementary, and clean flat-source LaTeX builds completed with 0 undefined references/citations, 0 overfull/underfull boxes, 0 package errors, and embedded/subset fonts.

## 11. Known boundaries

- The clean-clone package does not include the full raw raster dataset.
- The 25% raster-coverage budget is not scanner time or an acquisition-duration claim.
- The oracle teacher is retrospective and is not a deployable online policy.
- Field-content is one registered normalized-RGB-MSE surrogate, not a universal content metric.
- The learned endpoint does not outperform the strongest static reference; this negative boundary remains explicit.

## 12. GitHub verification

The implementation commit was pushed before this handoff was authored.

```text
git ls-remote origin refs/heads/aei-visual-narrative-consolidation
e49026a389c6d4d255dde2cdf3d8891440317e58  refs/heads/aei-visual-narrative-consolidation
```

Result: PASS. The remote branch exactly matched the implementation commit at verification time. The subsequent documentation-only handoff commit is verified separately after its push.
