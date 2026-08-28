# AEI Main-Method Reframe Completion Audit

Audit date: 2026-08-28 UTC
Target: *Advanced Engineering Informatics*, original research article
Branch: `aei-main-method-reframe`
Base: `ff4730b3fcf368d6ac43f0f72f034703e1556f7d`

## Completion Status

- [x] Proposed method is Task-Relevant Information Acquisition.
- [x] `MAVIS` is absent from title, abstract, Introduction, and Conclusions.
- [x] The supplement records MAVIS as a codebase implementation only.
- [x] `mvd_m1_o2` is described only as the static reference.
- [x] The abstract contains no P7 endpoint.
- [x] Related Work has three subsections.
- [x] The closest-work table is absent from the main manuscript.
- [x] Sections 3 and 4 each have three subsections.
- [x] Part-I and Part-II Results each have three subsubsections.
- [x] Part II opens with the favorable dynamic-versus-static result.
- [x] Figure 4 contains A1/A2 only and excludes A3/A4.
- [x] The main manuscript has four figures and two tables.
- [x] A3 is supplement-only.
- [x] A4 appears once as a concise main system diagnostic; full detail remains
  in the supplement.
- [x] All 39 canonical claims are assigned by the visibility authority.
- [x] The canonical metrics SHA-256 is unchanged.
- [x] The frozen scientific-root diff from the base is empty.
- [x] No model training, tuning, re-selection, or result recomputation occurred.
- [x] Main, supplement, and flat submission source compile.
- [x] All main and supplement fonts are embedded and subsetted.
- [x] Final logs contain no undefined references/citations, overfull/underfull
  boxes, multiply-defined labels, or LaTeX errors.
- [x] Figure, table, and source-ZIP deterministic replay passes.
- [x] `git diff --check` is clean.

## Scientific Integrity

`validate_paper()` returned:

```text
passed=True
canonical_claim_count=39
main_visible_claim_count=27
main_mapped_claim_count=27
combined_mapped_claim_count=39
figure_count=4
table_count=2
section_count=6
unmatched_numbers=()
changed_frozen_files=()
semantic_errors=()
```

Visibility is 12 `MAIN_HEADLINE`, 15 `MAIN_SUPPORT`, one
`MAIN_SYSTEM_DIAGNOSTIC`, and 11 `SUPPLEMENT_ONLY` claims. A3 is
supplement-only. A4 is the single main system diagnostic and retains its full
interval and domain directions in the supplement.

The Nguyen et al. citation context was checked against the publisher record
and narrowed to limited-data segmentation with augmentation/synthetic examples;
the manuscript does not misdescribe that work as incomplete spatial sampling.
BibTeX compilation passes for 18 unique entries. Sixteen DOI records were
resolved; fourteen returned HTTP 200 directly, and the two SAGE DOI endpoints
that returned HTTP 403 were independently found on their publisher/PubMed
records. No missing citation key or unsupported citation context remains.

No canonical numerical result changed. No scientific model retraining was
performed. No frozen result path changed.

Canonical metrics:

```text
f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6
artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv
```

## Regression Evidence

| Scope | Command | Result |
| --- | --- | --- |
| Paper | `PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_aei_paper_main_method_identity.py tests/test_mavis_aei_paper*.py` | 110 passed |
| MAVIS | `PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_mavis_*.py` | 260 passed |
| MVD | `PYTHONPATH=src /home/ww/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_mvd_*.py` | 29 passed |
| MVA authority | `cd /home/ww/paper3/cmc_damage_inference && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/ww/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_mva_*.py` | 126 passed |
| Ruff lint | `python -m ruff check src/cmc_bbdm/mavis tests` | pass |
| Ruff format | focused `python -m ruff format --check ...` command from the execution plan | 12 files already formatted |
| Whitespace | `git diff --check` | pass |

The MVA count is an actual run in the registered adjacent-Prompt source-tree
layout, not an inference from historical documentation.

## Build And Package Evidence

| Artifact | Result |
| --- | --- |
| Main PDF | 17 pages; `aa714debdbfdd6180fd695ba4e9c2b1091c22125f342356c6f22aeb4e354c58a` |
| Supplement PDF | 3 pages; `40ab5645fc93aa938732452292db3f1bca92ce90333fd4314976e4c6fa546c4b` |
| Flat source ZIP | `e42e9b059759ff777772513c6c3174c36c408762c54c3e8ee1b0633fada412b8` |
| Figure manifest | `results/aei_information_hierarchy/figures/FIGURE_CHECKSUMS.csv` |
| Table manifest | `results/aei_information_hierarchy/tables/TABLE_CHECKSUMS.csv` |
| Flat-source manifest | `results/aei_information_hierarchy/deterministic_package/submission_source/SUBMISSION_MANIFEST.csv` |
| Supplement-data manifest | `paper_aei_information_hierarchy/supplementary/data/SUPPLEMENTARY_DATA_MANIFEST.csv` |

The flat ZIP contains 12 files: build instructions, Elsevier class/style,
four figures, two tables, `main.tex`, `references.bib`, and its manifest. Its
entries have a fixed 2026-01-01 timestamp and deterministic order.

## Submission Readiness

Technical status: **conditional pass**. The manuscript uses `elsarticle` 3.4
in anonymous preprint mode, the review PDF metadata contains only the withheld
author placeholder, and the source package is flat. The current official AEI
scope was checked on 2026-08-28 and requires engineering relevance,
knowledge-intensive task support, and vigorous validation; the manuscript is
in scope on those axes. The journal-specific ScienceDirect Guide for Authors
returned HTTP 403, so live Editorial Manager requirements must be confirmed at
upload time.

Official sources checked:

- <https://shop.elsevier.com/journals/advanced-engineering-informatics/1474-0346>
- <https://www.sciencedirect.com/journal/advanced-engineering-informatics/publish/guide-for-authors>

## Frozen Evidence Included In Delivery

The delivery branch contains, without modification in this reframe:

- `artifacts/mavis_science_closure/M0_GO_NOGO.md`;
- `artifacts/mavis_science_closure/M1_GO_NOGO.md`;
- `artifacts/mavis_science_closure/GO_NOGO.md`;
- `artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv`;
- the supplementary machine-readable CSV/JSON evidence package.

## Final Git State

Working branch: `aei-main-method-reframe`
Final local and remote SHA: the commit containing this audit and handoff is
resolved after commit with `git rev-parse HEAD` and verified against
`git ls-remote origin refs/heads/aei-main-method-reframe`. The exact immutable
value is reported in the final delivery response because a Git commit cannot
embed its own hash in its tracked contents.

## Remaining Factual Issues

Before journal submission, authors must supply names, affiliations,
corresponding-author details, funding, competing-interest and CRediT statements,
and the final repository/archive identifier and license. The live AEI upload
form, review model, declarations, and optional highlights must be confirmed in
Editorial Manager. The evidence does not establish external empirical transfer,
physical scanner-time savings, or learned end-to-end superiority over the
static reference.
