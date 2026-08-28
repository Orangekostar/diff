# AEI Manuscript Audit-Residue Cleanup Completion

Audit date: 2026-08-28 UTC
Target: *Advanced Engineering Informatics*, original research article
Repository: `git@github.com:Orangekostar/diff.git`
Base branch: `aei-main-method-reframe`
Base SHA: `a21f84f583a3767f727aeace4c38ae7be3f880ee`
Working branch: `aei-main-method-reframe`
Final local SHA: `PENDING_PAYLOAD_COMMIT`
Final remote SHA: `PENDING_REMOTE_VERIFICATION`

## Exact Main-Text Changes

1. **Introduction.** Replaced `frozen held-out-domain outcomes` with
   `held-out-domain evaluation`, while retaining exact native-raster cost and
   synchronized bootstrap contrasts.
2. **Section 4.3.** Deleted the complete evidence-chronology paragraph. The
   statistical-analysis paragraph now transitions directly into Results.
3. **Section 5.3.** Removed the transfer heading, five-condition checklist, and
   extended defense-oriented limitations. They were replaced by one 63-word
   scope paragraph covering exact native-raster acquisition cost, six CFRP
   domains, predictor/action-space conditioning, and prospective validation.

The supplementary manuscript was not changed. Chronology, full A3/A4 evidence,
MAVIS implementation identity, static-reference identity, and detailed scope
boundaries remain there and in the internal authorities.

## Test Contract Changes

- `tests/test_mavis_aei_paper_manuscript.py`
  - Introduction now forbids every `frozen` occurrence.
  - The Chinese comparison draft is required to preserve the method identity,
    core headline results, A4 direction, cost definition, and future-validation
    scope.
- `tests/test_mavis_aei_paper_review_fixes.py`
  - Chronology language is forbidden in `main.tex` and required in the
    supplement/chronology authority.
  - The old transfer heading and defense phrases are forbidden.
  - Section 5.3 must end in one 60--100-word compact scope paragraph.
  - The supplement must retain chronology, A3, A4, MAVIS implementation
    identity, and detailed scope boundaries.
- `src/cmc_bbdm/mavis/aei_paper_validation.py`
  - Removed chronology/transfer-defense phrases from `required_main`.
  - Added compact-scope requirements and forbidden audit-residue checks.

## Scientific Integrity

No scientific result changed. No model was retrained. No canonical metric
changed. No frozen scientific result path changed. A4 numerical direction is
unchanged.

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

Canonical metrics SHA-256:

```text
f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6
```

Frozen science diff from the task base SHA across the specified result and
authority roots: empty.

## Build And Test Evidence

| Check | Command | Result |
| --- | --- | --- |
| Paper suite | `PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_aei_paper_main_method_identity.py tests/test_mavis_aei_paper*.py` | 112 passed |
| Ruff lint | `python -m ruff check src/cmc_bbdm/mavis tests` | pass |
| Ruff format | `python -m ruff format --check src/cmc_bbdm/mavis/aei_paper_validation.py tests/test_mavis_aei_paper_manuscript.py tests/test_mavis_aei_paper_review_fixes.py` | 3 files already formatted |
| Main LaTeX | `latexmk -pdf -interaction=nonstopmode -halt-on-error` | 16 pages; clean |
| Supplement LaTeX | same command on `supplementary.tex` | 3 pages; clean |
| Flat-source LaTeX | same command on flat `main.tex` | 16 pages; clean |
| Deterministic replay | rebuild ZIP and manifest in an independent temporary directory | exact match |
| Whitespace | `git diff --check` | pass |

For all three LaTeX targets, final logs contain zero undefined references, zero
undefined citations, zero overfull boxes, zero underfull boxes, and zero LaTeX
errors. Main and supplementary fonts are embedded and subsetted; every PDF page
contains extractable text.

MAVIS, MVD, and MVA scientific suites were **not rerun; no scientific/shared
runtime code changed in this manuscript-only cleanup**.

## Generated Artifacts

| Artifact | Pages | SHA-256 |
| --- | ---: | --- |
| `results/aei_information_hierarchy/submission/AEI_PAPER1_MANUSCRIPT.pdf` | 16 | `e457bcae1a8ab346edab3005b0267e2553755b8530ec4f97777263ae2f4f008f` |
| `results/aei_information_hierarchy/submission/AEI_PAPER1_SUPPLEMENTARY.pdf` | 3 | `167b4bb3be001c7750397b0f9fec6d68f617b7310f6665d13d72c26cd4db8a53` |
| `results/aei_information_hierarchy/deterministic_package/AEI_PAPER_SUBMISSION_SOURCE.zip` | n/a | `8bd5e9c956037a2867e9fcbfe94db6be3d560e39fb21ae8946f1611591a2b900` |
| `results/aei_information_hierarchy/deterministic_package/submission_source/SUBMISSION_MANIFEST.csv` | n/a | `5fa7602fcb072a4d58ce6a02bbc55c6ab291b7c953d57797ffea0113d90a60b8` |

The author-facing Chinese comparison draft is
`paper_aei_information_hierarchy/MANUSCRIPT_CHINESE_DRAFT.md`. It is not copied
into the deterministic submission package.

## Completion Checklist

- [x] Introduction no longer contains audit-style frozen wording.
- [x] Section 4.3 chronology paragraph was removed from the main manuscript.
- [x] Chronology remains in the supplement and authority.
- [x] Section 5.3 transfer checklist was removed from the main manuscript.
- [x] Section 5.3 limitations were compressed into one short scope paragraph.
- [x] Primary method identity is unchanged.
- [x] A4 numerical values and direction are unchanged.
- [x] Main Results retain the 3+3 structure.
- [x] The main manuscript retains four figures and two tables.
- [x] All canonical values are unchanged.
- [x] Canonical SHA-256 is unchanged.
- [x] Frozen science diff is empty.
- [x] No new training occurred.
- [x] Paper tests pass.
- [x] LaTeX builds are clean.
- [x] Deterministic package was rebuilt and independently replayed.
- [x] `git diff --check` is clean.

## Remaining Factual Issues

Author identity, affiliations, correspondence, funding, competing-interest and
CRediT statements, repository/archive identifier, and license remain
author-supplied. Live AEI upload requirements still require confirmation.
External empirical transfer, physical scanner-time savings, and learned
end-to-end superiority over the static reference are not established.
