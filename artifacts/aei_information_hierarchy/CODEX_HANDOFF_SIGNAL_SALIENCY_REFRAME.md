# Codex Handoff: AEI Signal-Saliency Reframe

## 1. Repository identity

- Repository: `git@github.com:Orangekostar/diff.git`
- Branch: `aei-signal-saliency-reframe`
- Worktree: `/home/ww/diff/.worktrees/aei-signal-saliency-reframe`
- Upstream: `origin/aei-signal-saliency-reframe`

## 2. Starting base SHA

`35248f17f603e94962dc19e939162e9ef4eee5f2`

The branch and worktree were created directly from this exact commit.

## 3. Final implementation SHA

`a87d39be027fc26ae5a90f1993782547433e9bd8`

Commit subject: `paper: replace reconstruction story with saliency evidence`

## 4. Final documentation/handoff SHA

This field denotes the commit containing this file (`SELF`). Its exact local and
remote SHA is reported by the final `git rev-parse HEAD` and `git ls-remote`
verification after the commit is created and pushed. A commit cannot embed its
own exact SHA in its content because changing the content changes that SHA.

## 5. Exact changed-file inventory

The implementation commit changes exactly the following 83 files:

```text
artifacts/aei_information_hierarchy/EVIDENCE_AUTHORITY_RECONCILIATION.md
artifacts/aei_information_hierarchy/FIGURE_SOURCE_MAP.csv
artifacts/aei_information_hierarchy/MAIN_METHOD_REFRAME_VISUAL_CONTRACT.md
artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv
artifacts/aei_information_hierarchy/PAPER_CLAIM_MAP.md
artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.csv
artifacts/aei_information_hierarchy/PAPER_CLAIM_VISIBILITY_MAP.md
artifacts/aei_information_hierarchy/PAPER_EVIDENCE_CHRONOLOGY.csv
artifacts/aei_information_hierarchy/PAPER_EVIDENCE_CHRONOLOGY.md
artifacts/aei_information_hierarchy/PAPER_METHOD_IDENTITY_LEDGER.md
artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.csv
artifacts/aei_information_hierarchy/PAPER_POSITIVE_NARRATIVE_MAP.md
artifacts/aei_information_hierarchy/PAPER_SOURCE_HASHES.csv
artifacts/aei_information_hierarchy/SIGNAL_SALIENCY_AUTHORITY_MIGRATION_AUDIT.md
docs/superpowers/plans/2026-08-30-aei-signal-saliency-reframe.md
docs/superpowers/specs/2026-08-30-aei-signal-saliency-reframe-design.md
docs/superpowers/visual-composer/aei-information-hierarchy-iteration-log.md
docs/superpowers/visual-composer/aei-information-hierarchy-visual-contract.md
paper_aei_information_hierarchy/CLAIM_SENTENCE_BANK.md
paper_aei_information_hierarchy/MANUSCRIPT_CHINESE_DRAFT.md
paper_aei_information_hierarchy/MANUSCRIPT_OUTLINE.md
paper_aei_information_hierarchy/README.md
paper_aei_information_hierarchy/figures/figure2_information_characterization.pdf
paper_aei_information_hierarchy/figures/figure4_valuation_planning_realization.pdf
paper_aei_information_hierarchy/main.tex
paper_aei_information_hierarchy/panel_pngs/PANEL_PNG_MANIFEST.csv
paper_aei_information_hierarchy/panel_pngs/figure2/figure2_panel_b.png
paper_aei_information_hierarchy/panel_pngs/figure2/figure2_panel_c.png
paper_aei_information_hierarchy/panel_pngs/figure2/figure2_panel_e.png
paper_aei_information_hierarchy/panel_pngs/figure2/figure2_panel_f.png
paper_aei_information_hierarchy/panel_pngs/figure4/figure4_panel_a.png
paper_aei_information_hierarchy/supplementary/data/S17_provenance_hashes.csv
paper_aei_information_hierarchy/supplementary/data/SUPPLEMENTARY_DATA_MANIFEST.csv
paper_aei_information_hierarchy/supplementary/supplementary.tex
results/aei_information_hierarchy/deterministic_package/AEI_PAPER_SUBMISSION_SOURCE.sha256
results/aei_information_hierarchy/deterministic_package/AEI_PAPER_SUBMISSION_SOURCE.zip
results/aei_information_hierarchy/deterministic_package/submission_source/SUBMISSION_MANIFEST.csv
results/aei_information_hierarchy/deterministic_package/submission_source/figure2_information_characterization.pdf
results/aei_information_hierarchy/deterministic_package/submission_source/figure4_valuation_planning_realization.pdf
results/aei_information_hierarchy/deterministic_package/submission_source/main.tex
results/aei_information_hierarchy/deterministic_package/working/CLAIM_SENTENCE_BANK.md
results/aei_information_hierarchy/deterministic_package/working/MANUSCRIPT_OUTLINE.md
results/aei_information_hierarchy/deterministic_package/working/README.md
results/aei_information_hierarchy/deterministic_package/working/figures/figure2_information_characterization.pdf
results/aei_information_hierarchy/deterministic_package/working/figures/figure4_valuation_planning_realization.pdf
results/aei_information_hierarchy/deterministic_package/working/main.tex
results/aei_information_hierarchy/deterministic_package/working/supplementary/data/S17_provenance_hashes.csv
results/aei_information_hierarchy/deterministic_package/working/supplementary/data/SUPPLEMENTARY_DATA_MANIFEST.csv
results/aei_information_hierarchy/deterministic_package/working/supplementary/supplementary.tex
results/aei_information_hierarchy/figure_qa/figure2_collisions.json
results/aei_information_hierarchy/figure_qa/figure2_collisions_overlay.pdf
results/aei_information_hierarchy/figure_qa/figure4_collisions.json
results/aei_information_hierarchy/figures/FIGURE_CHECKSUMS.csv
results/aei_information_hierarchy/figures/figure1_task_relevant_acquisition_framework.csv
results/aei_information_hierarchy/figures/figure2_information_characterization.csv
results/aei_information_hierarchy/figures/figure2_information_characterization.pdf
results/aei_information_hierarchy/figures/figure2_information_characterization.png
results/aei_information_hierarchy/figures/figure2_information_characterization.svg
results/aei_information_hierarchy/figures/figure2_information_characterization_caption.md
results/aei_information_hierarchy/figures/figure4_valuation_planning_realization.csv
results/aei_information_hierarchy/figures/figure4_valuation_planning_realization.pdf
results/aei_information_hierarchy/figures/figure4_valuation_planning_realization.png
results/aei_information_hierarchy/figures/figure4_valuation_planning_realization.svg
results/aei_information_hierarchy/figures/figure4_valuation_planning_realization_caption.md
results/aei_information_hierarchy/submission/AEI_PAPER1_MANUSCRIPT.pdf
results/aei_information_hierarchy/submission/AEI_PAPER1_SUPPLEMENTARY.pdf
results/aei_information_hierarchy/tables/TABLE_CHECKSUMS.csv
results/aei_information_hierarchy/tables/table1_case_protocol.csv
results/aei_information_hierarchy/tables/table2_task_relevant_results.csv
results/aei_information_hierarchy/tables/table2_task_relevant_results.tex
src/cmc_bbdm/mavis/aei_paper_evidence.py
src/cmc_bbdm/mavis/aei_paper_figures.py
src/cmc_bbdm/mavis/aei_paper_tables.py
src/cmc_bbdm/mavis/aei_paper_validation.py
src/cmc_bbdm/mavis/aei_paper_visual_assets.py
tests/test_aei_paper_main_method_identity.py
tests/test_mavis_aei_paper_evidence.py
tests/test_mavis_aei_paper_figures.py
tests/test_mavis_aei_paper_manuscript.py
tests/test_mavis_aei_paper_review_fixes.py
tests/test_mavis_aei_paper_tables.py
tests/test_mavis_aei_paper_validation.py
tests/test_mavis_aei_paper_visual_assets.py
```

This handoff file is the only file added by the documentation/handoff commit.

## 6. Scientific rationale

The main paper previously contrasted CAI acquisition with reconstruction or
field-content recovery. The replacement answers the intended bounded question:
whether CAI-priority regions are merely regions ranked highly by the already
registered task-agnostic C-scan appearance heuristic. The migration uses frozen
MVA A2 evidence and does not introduce a new saliency metric or experiment.

## 7. Precise saliency definition

For newly revealed native-raster locations `N`, RGB C-scan channel values
`C_k(p)`, and the specimen-specific RGB border median `m_border,k`, the
implementation computes

```text
S_app = [1 / (255 * 3 * |N|)]
        * sum_{p in N} sum_{k=1}^3 |C_k(p) - m_border,k|.
```

This is the mean element-wise absolute RGB deviation from the border median,
divided by 255. It uses no CAI outcome. Access to unrevealed candidate C-scan
values makes the oracle retrospective and non-deployable. It is not damage
severity, a causal material map, or a universal saliency definition.

## 8. Frozen evidence values and hashes

- Appearance minus mechanical CAI AUEBC:
  `0.007080059382261465`
- Synchronized 95% domain-bootstrap CI:
  `[0.004799356600193281, 0.00974029297002471]`
- Direction: CAI/mechanical oracle favored in `6/6` held-out domains.
- Mean initial-map Spearman across 276 specimens:
  `0.022212009079236737`
- Mean top-decile overlap: `0.20031055900621117`

| Frozen authority | SHA-256 |
|---|---|
| `src/cmc_bbdm/mva/appearance_value.py` | `a10ee487e0b33d28599166dbe200f0c1cf2cd0211dbe7b146a2f0aadc877d9cc` |
| `src/cmc_bbdm/mva/oracle_execution.py` | `472edf5a6d9ad1cf4e44eb0d1380a043d54ce8ab743bb60612c6c0aa7dc8f931` |
| `docs/MVA_A0_A3_PROTOCOL.md` | `6b1e6f91329d80196ac67cef78c596da9ca2ad9cfc83700b500809e79ab77ec0` |
| `results/mva/a2_oracle_value/REPORT.md` | `6c92fcff56c893a30c1e6c6a763a85562bb10e6044b2403b712a4070410f6b65` |
| `results/mva/a2_oracle_value/bootstrap.csv` | `a6abce9d9e3647d0668854f2772614bb5b940d5c2bf6355e12de293e487d765d` |
| `results/mva/a2_oracle_value/domain_metrics.csv` | `de31087b353f71dd42855e62a97b783c093ff0f197ac5eaf979b1f970b44127f` |
| `results/mva/a2_oracle_value/map_similarity.csv` | `e161c2043269456aa2e7321bf676ac534cc9fa01c7e0e22213781f32b2980998` |
| `results/mva/a2_oracle_value/oracle_values.parquet` | `6b289f2f6f74ac75dde47ea7cbfefcda1c49f025e74227bfb34ef269182ff963` |

## 9. Old reconstruction concept disposition

- Main Figure 2 and main prose no longer use reconstruction, image recovery,
  field recovery, or field-content as the scientific comparator.
- Main Figure 4 no longer shows the reconstruction-derived O3 control and does
  not relabel it as saliency.
- `U3_RECONSTRUCTION_ORACLE`, `U4_ORACLE_CAI_SPECIFICITY`,
  `U4_ORACLE_IMAGE_SPECIFICITY`, `U4_LEARNED_SPECIFICITY_BOUNDARY`, and
  `O3_REAL_MINUS_RECONSTRUCTION` remain canonical, chronology-bound, and
  explicitly reported as legacy reconstruction evidence in the Supplement.

## 10. Old-to-new canonical claim migration

| Removed main-story authority | Added main-story authority |
|---|---|
| `U3_RECONSTRUCTION_ORACLE` | `U3_CAI_VS_APPEARANCE_SALIENCY_AUEBC` |
| `U4_ORACLE_CAI_SPECIFICITY` | `U4_CAI_SALIENCY_MAP_SPEARMAN` |
| `U4_ORACLE_IMAGE_SPECIFICITY` | `U4_CAI_SALIENCY_TOP10_OVERLAP` |
| `O3_REAL_MINUS_RECONSTRUCTION` | No saliency replacement; history and shuffled adverse controls remain main-visible. |

Old claim count: 39. New claim count: 42. Final visibility is
`10 MAIN_HEADLINE / 16 MAIN_SUPPORT / 1 MAIN_SYSTEM_DIAGNOSTIC /
15 SUPPLEMENT_ONLY`.

## 11. New canonical SHA

- Old: `f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`
- New: `59ce986b56961370dcee5772e199f2d897bc1bcdc04bacae4f5b772af31a5408`

This is a controlled paper-authority migration using pre-existing frozen
evidence, not a change to any frozen scientific result.

## 12. Frozen-path diff result

The following command returned no paths:

```bash
git diff --name-only 35248f17f603e94962dc19e939162e9ef4eee5f2 -- \
  results/p1_full_field_oracle results/p5_sparse_scan results/mvd \
  results/mavis results/mavis_science_closure results/mva \
  artifacts/mavis artifacts/mavis_science_closure \
  artifacts/mvd_authority artifacts/mavis_authority
```

No model was trained, tuned, reselected, or reevaluated. No frozen bootstrap
or endpoint was recomputed.

## 13. Figure 1-4 final contract

- Figure 1, WHY: unchanged method identity and Part I to Part II flow.
- Figure 2, WHAT: spatial/sparse evidence; initial legal state; frozen CAI versus
  appearance-saliency AUEBC; paired mechanical and true `appearance_oracle`
  priority maps; percentile difference.
- Figure 3, WHEN: unchanged state evolution, acquisition history, dynamic versus
  static valuation, and predictor dependence.
- Figure 4, HOW: measured-state, position/history, and shuffled adverse evidence;
  component substitutions; bounded set realization; unchanged deployment
  calibration direction. The reconstruction-derived row is absent.

The main manuscript contains exactly four figures and one table.

## 14. Main/Supp terminology audit

- Main manuscript, visible figure SVG text, and captions contain none of:
  `reconstruction oracle`, `field-content reference`, `field-content control`,
  `C-scan-content priority`, `image recovery`, `field recovery`, or
  `recovering the C-scan field`.
- Main Methods contains the exact element-wise appearance-saliency definition,
  no-CAI-label condition, and retrospective/non-deployable boundary.
- Main Results reports `0.007080`, `0.0222`, `0.2003`, and the `6/6` direction.
- The Chinese draft mirrors the saliency story, values, scope, and four-figure
  narrative.
- The Supplement retains all five legacy reconstruction claim IDs, chronology,
  full A3/A4 evidence, MAVIS implementation identity, and scope boundaries.

## 15. Tests and counts

Fresh final paper command covered the method-identity file and every
`tests/test_mavis_aei_paper*.py` file:

```text
134 passed in 106.61s (0:01:46)
```

The semantic validator reported:

```text
passed=True
canonical_claim_count=42
main_visible_claim_count=26
main_mapped_claim_count=26
combined_mapped_claim_count=42
figure_count=4
table_count=1
section_count=6
unmatched_numbers=()
changed_frozen_files=()
semantic_errors=()
```

MAVIS/MVD/MVA scientific suites were not rerun; no scientific or shared
training/evaluation runtime code changed in this paper-only evidence migration.

## 16. Ruff results

```text
python -m ruff check src/cmc_bbdm/mavis tests
All checks passed!

python -m ruff format --check <13 modified Python files>
13 files already formatted

git diff --check
clean
```

## 17. Figure QA results

- Source preflight: 18 PASS, 3 reviewed WARN, 0 FAIL. Warnings are the package's
  intentional PDF/SVG plus 300-dpi PNG contract rather than TIFF/600 dpi, and a
  final-width constant not statically inferred by the validator.
- Alignment: Figure 2, Figure 3, Figure 4, and S1 all PASS; 26 comparisons,
  0 warnings, 0 failures, 0 exemptions.
- PDF text: all five figure PDFs auditable; no glyph run below 5 pt.
- Collision audit: 0 failures. Figures 1, 3, 4, and S1 PASS. Figure 2 has three
  reviewed fill-edge warnings caused by numeric labels beside their own markers.
- Visual inspection: all standalone figures and final manuscript pages were
  inspected; no clipping, distortion, blank panel, incoherent overlap, or stale
  reconstruction/field-content label remains.
- Deterministic replay: figures (27 files), supplementary figures (8), tables
  (7), panel PNG package (31), and deterministic package (47) were byte-identical
  between the formal output and an independent temporary-directory replay.
- Panel PNG manifest: 29 panels; every SHA-256 verified.

## 18. Manuscript and Supplement pages

- Main manuscript: 18 pages.
- Supplement: 4 pages.
- Flat submission-source build: 18 pages.

All three final logs contain zero LaTeX/package errors, undefined references,
undefined citations, overfull boxes, and underfull boxes. All listed PDF fonts
are embedded and subsetted.

## 19. PDF, ZIP, and manifest SHA-256

| Artifact | SHA-256 |
|---|---|
| `results/aei_information_hierarchy/submission/AEI_PAPER1_MANUSCRIPT.pdf` | `7428378f3c05ea7a891e5f4ca4244ef613e002954e1367a0870967df6e59a913` |
| `results/aei_information_hierarchy/submission/AEI_PAPER1_SUPPLEMENTARY.pdf` | `ed666927080620271cbeb38e84dc9780489a0ffe609efc1dafdcce2c97978bbb` |
| `results/aei_information_hierarchy/deterministic_package/AEI_PAPER_SUBMISSION_SOURCE.zip` | `46f376defc1092ebe2d64c373e211d2ba7524c2987dc0fc51f3fd35e96ce76da` |
| `results/aei_information_hierarchy/deterministic_package/submission_source/SUBMISSION_MANIFEST.csv` | `7968a3b388f44f56fb761717883f08edcc23be75ed46c0a4421e831ec6dd85f1` |
| `results/aei_information_hierarchy/figures/FIGURE_CHECKSUMS.csv` | `3c1758da4fb19ddefc3ae3f67e857498560336aab331898db0d4747547d5952d` |
| `results/aei_information_hierarchy/supplementary_figures/FIGURE_CHECKSUMS.csv` | `edc0de0ddbbf397e6120e41f539b41adfef545e78b7c698427ab720` |
| `paper_aei_information_hierarchy/panel_pngs/PANEL_PNG_MANIFEST.csv` | `06656c0f5b4e2a2ff7e18f9bd30ae38be9654b86254f42bcf963b70c0b3ae892` |

The flat source manifest has 10 payload rows; the Supplement data manifest has
18 rows. The ZIP's `.sha256` sidecar matches the archive, and an independent
package replay produced the same bytes.

## 20. Known boundaries

- Evidence is limited to 276 specimens and six held-out CFRP domains.
- The registered appearance metric is one deliberately simple task-agnostic
  heuristic; weak rank agreement is not statistical independence and does not
  generalize to every possible non-CAI objective.
- Both mechanical and appearance maps in Figure 2 are retrospective oracles,
  not deployable acquisition policies.
- Exact normalized native-raster cost is not scanner time and excludes travel,
  coupling, settling, and path-planning cost.
- Priority maps are predictor- and state-conditioned; they are not causal
  damage maps or intrinsic coordinate importance.
- The learned implementation remains slightly worse than the static reference
  at A4; its numerical direction and values are unchanged.
- Prospective scanner-level validation remains future work.

## 21. Local HEAD

At implementation push verification:
`a87d39be027fc26ae5a90f1993782547433e9bd8`.

After the handoff commit, local HEAD is the `SELF` commit described in Section 4
and is reported by the final completion response.

## 22. Remote HEAD

After the implementation push:
`a87d39be027fc26ae5a90f1993782547433e9bd8`.

The final ordinary (non-force) push updates
`refs/heads/aei-signal-saliency-reframe` to the `SELF` handoff commit. Final
local/remote equality is verified and reported after that push.

## 23. Clean worktree status

The worktree was clean after the implementation commit and first push. After
this file is committed and pushed, `git status --short` must again be empty;
that final check is part of the external GitHub verification reported in the
completion response.

No pull request was created, no merge was performed, and no force push was used.
