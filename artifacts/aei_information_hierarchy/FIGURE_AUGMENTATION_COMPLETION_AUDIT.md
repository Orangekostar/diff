# AEI Figure Augmentation Completion Audit

## Repository identity

| Field | Value |
| --- | --- |
| Repository | `git@github.com:Orangekostar/diff.git` |
| Branch | `aei-main-method-reframe` |
| User-referenced scientific base | `a21f84f583a3767f727aeace4c38ae7be3f880ee` |
| Operational branch baseline | `b0aeba3f5c2eedda25e5b1c64cfd44d6ae0f4f2c` |
| Figure implementation commit | `PENDING_CONTENT_COMMIT` |
| Verified remote content commit | `PENDING_REMOTE_PUSH` |

## Completion checklist

- [x] Inspected the current figure, manuscript, evidence, and package pipeline.
- [x] Mapped every added or modified figure to authoritative source files.
- [x] No qualitative figure uses fabricated data.
- [x] Figure 2 includes quantitative evidence and hash-verified sparse states.
- [x] Figure 3 includes state evolution, priority maps, and acquisition history.
- [x] Figure 5 shows paired task-specific priorities on one registered state.
- [x] Figure 6 is explicitly `NO_GO` because no frozen cross-endpoint normalization exists.
- [x] Supplementary Figure S1 provides deterministic six-domain breadth.
- [x] The paper narrative remains positive and correctly ordered.
- [x] No canonical metric changed.
- [x] No frozen science path changed.
- [x] Reproducible figure generation passes.
- [x] Paper and figure tests pass.
- [x] Main PDF builds.
- [x] Supplement PDF builds.
- [x] Flat source and deterministic ZIP build.

## Implemented visual contract

| Figure | Decision | Result |
| --- | --- | --- |
| Figure 1 | GO | Nature-style framework cleanup; method identity and two-part flow preserved. |
| Figure 2 | GO | Six panels join U1-U5/O2 evidence to real initial and 25% sparse states. |
| Figure 3 | GO | Six panels join O1/O3/O4 evidence to initial/later priority maps and stored history. |
| Figure 4 | GO | Nature-style re-render of A1/A2 valuation, planning, and realization evidence. |
| Figure 5 | GO | Real `c8-2` CAI-versus-RGB oracle priority overlays and paired difference map. |
| Figure 6 | NO_GO | Stage endpoints are not commensurate and have no frozen normalization contract. |
| Supplementary Figure S1 | GO | Initial/later overlays for a result-independent specimen from each of six domains. |

The raw original full raster is not present in the versioned clean clone. Figure 2
therefore retains the full-field result quantitatively and uses only hash-verified
compact state reconstructions for qualitative panels.

## Source grounding

- Canonical values: `artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv`.
- Sparse and sequential states: `results/mavis/p1_state_bank/state_manifest.parquet`
  and `results/mavis/p1_state_bank/revealed_measurements/`.
- State-conditioned teacher values: `results/mavis/p3_dynamic_voi/action_scores.parquet`.
- Paired task oracle values: `results/mva/a2_oracle_value/oracle_values.parquet`.
- Domain/specimen dimensions: `artifacts/mavis_authority/scan_manifest.csv`.
- Aggregate P9/P10/P11/P14 evidence remains bound through the canonical figure rows.
- Complete per-panel mapping: `artifacts/aei_information_hierarchy/FIGURE_SOURCE_MAP.csv`.

`src/cmc_bbdm/mavis/aei_paper_visual_assets.py` restores compact measurements,
checks the legal mask and grid SHA, reconstructs with the registered bilinear
implementation, and rejects any reconstruction whose output SHA differs from the
stored manifest. It has no synthetic fallback.

## Nature figure QA

| Check | Result |
| --- | --- |
| Static source validator | Ready: 19 PASS, 0 FAIL; two expected raster warnings (no TIFF, 300 dpi rather than optional 600 dpi) |
| Panel alignment | Five multi-panel reports PASS at strict 1.5 pt tolerance |
| PDF text audit | Six PDFs auditable; zero glyph runs below 5 pt |
| Collision audit | Zero FAIL across all six PDFs |
| Visual inspection | Standalone exports and integrated manuscript pages inspected at final size |

Figure 2 retains three reviewed `text-fill-edge` warnings because numeric labels
are intentionally adjacent to their plotted markers. Figure 3 retains one reviewed
fill-edge warning because the `History` tick is centered below its bar. Neither is a
text-text, text-stroke, clipping, or visible-overlap failure. Reports and QA overlays
are under `results/aei_information_hierarchy/figure_qa/`; alignment JSON/SVG files
are stored beside their figures.

## Tests and builds

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

Main, supplement, and flat-source `latexmk` builds produced 17, 4, and 17 pages.
Final logs contain zero LaTeX/package warnings, undefined references/citations,
overfull or underfull boxes, and errors. The deterministic ZIP, manifest, and SHA
file matched an independent replay byte for byte.

MAVIS/MVD/MVA scientific suites were not rerun; no scientific or shared
training/evaluation runtime changed in this paper-rendering task.

## Scientific integrity

| Gate | Result |
| --- | --- |
| Canonical SHA-256 | `f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6` |
| Frozen science diff from `a21f84f...` | Empty |
| New training | No |
| Scientific endpoint recomputation | No |
| Canonical metric change | No |
| A4 direction/value change | No |
| Method identity change | No |

## Generated artifacts

| Artifact | Pages | SHA-256 |
| --- | ---: | --- |
| `results/aei_information_hierarchy/submission/AEI_PAPER1_MANUSCRIPT.pdf` | 17 | `40f83a2041a16dba27caeebac8a2c79afbb8918bd0b7144d0b1668598c83ccba` |
| `results/aei_information_hierarchy/submission/AEI_PAPER1_SUPPLEMENTARY.pdf` | 4 | `153310a80c7e6c64e696e04568f21f24bc8e737262b5c14e616704557ed9fe0f` |
| `results/aei_information_hierarchy/deterministic_package/AEI_PAPER_SUBMISSION_SOURCE.zip` | n/a | `18e9e703c7a30d32f8b4bfa2355a3ab6052b95b15f2981a60834e7ff4c08147d` |
| `results/aei_information_hierarchy/deterministic_package/submission_source/SUBMISSION_MANIFEST.csv` | n/a | `fedde533a690d2a3dfcdaff994298f8916f87e7abf4364c6827fa22c6d833c2b` |
| `results/aei_information_hierarchy/figures/FIGURE_CHECKSUMS.csv` | n/a | `7137dc065b15568f5a7ae56dae782d67d4d3a426c6b2c6ffbb3abb9ffe375c10` |
| `results/aei_information_hierarchy/supplementary_figures/FIGURE_CHECKSUMS.csv` | n/a | `b2698d69c74463ccbd06c2cb2cb58857c9179b1c57a74d0019768b856fc10c04` |
