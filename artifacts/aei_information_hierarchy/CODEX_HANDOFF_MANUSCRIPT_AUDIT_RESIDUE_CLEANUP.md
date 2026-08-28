# Codex Handoff: AEI Manuscript Audit-Residue Cleanup

Handoff date: 2026-08-28 UTC

## Repository Identity

| Field | Value |
| --- | --- |
| Repository | `git@github.com:Orangekostar/diff.git` |
| Branch | `aei-main-method-reframe` |
| Base SHA | `a21f84f583a3767f727aeace4c38ae7be3f880ee` |
| Final local SHA (verified manuscript payload) | `9d7effaf18ecb5aa62f84b8380d8fbf701b9ef6b` |
| Final remote SHA (verified manuscript payload) | `9d7effaf18ecb5aa62f84b8380d8fbf701b9ef6b` |
| Remote branch | `refs/heads/aei-main-method-reframe` |

The exact verified manuscript payload SHA is filled after its first push. A
tracked file cannot contain the hash of the commit that contains that same
file, so the final verification-record HEAD is reported by the post-push command
and final delivery response.

## Exact Manuscript Changes

### Introduction

The contribution paragraph now ends with `held-out-domain evaluation` instead
of `frozen held-out-domain outcomes`. Exact native-raster cost and synchronized
bootstrap contrasts remain unchanged.

### Section 4.3

The complete evidence-chronology paragraph was deleted. Statistical inference
now transitions directly to Experimental Results and Discussion. Chronology
facts remain unchanged in the supplement and `PAPER_EVIDENCE_CHRONOLOGY.*`.

### Section 5.3

The transfer heading, five-item methodological checklist, and extended
defense-oriented limitations were replaced by one 63-word scope paragraph. It
retains exact native-raster acquisition cost, six CFRP domains,
predictor/action-space conditioning, and prospective scanner-level validation.

`paper_aei_information_hierarchy/MANUSCRIPT_CHINESE_DRAFT.md` provides a full
Chinese comparison draft aligned to the cleaned English source and is excluded
from the submission package.

## Tests Changed

- `tests/test_mavis_aei_paper_manuscript.py`
  - forbids `frozen`, `post-freeze`, and `hash-bound` in Introduction;
  - validates the Chinese draft's method identity, core values, A4 direction,
    cost definition, and prospective-validation scope.
- `tests/test_mavis_aei_paper_review_fixes.py`
  - removes the old requirement that chronology appear in the main manuscript;
  - requires chronology in the supplement/authority;
  - forbids the old transfer heading and defense phrases;
  - enforces one 60--100-word final scope paragraph;
  - locks supplement chronology, A3/A4, MAVIS identity, and scope boundaries.
- `src/cmc_bbdm/mavis/aei_paper_validation.py`
  - replaces old main-text chronology/transfer requirements with compact-scope
    and audit-residue absence checks.

## Scientific Integrity

No scientific result changed.
No model was retrained.
No canonical metric changed.
No frozen scientific result path changed.
A4 numerical direction is unchanged.

Canonical SHA-256:
`f0d2615637a6470744f275a2ac6e1c5e7aff110ca7e31cb323793c29405be4e6`.
Frozen science diff from `a21f84f583a3767f727aeace4c38ae7be3f880ee`:
empty.

## Build And Test Evidence

```text
PYTHONPATH=src python -m pytest -q -p no:cacheprovider \
  tests/test_aei_paper_main_method_identity.py tests/test_mavis_aei_paper*.py
112 passed in 25.62s

python -m ruff check src/cmc_bbdm/mavis tests
All checks passed!

python -m ruff format --check \
  src/cmc_bbdm/mavis/aei_paper_validation.py \
  tests/test_mavis_aei_paper_manuscript.py \
  tests/test_mavis_aei_paper_review_fixes.py
3 files already formatted

git diff --check
pass
```

Main, supplement, and flat-source `latexmk` builds completed with 16, 3, and
16 pages. Final logs contained zero undefined references/citations, overfull or
underfull boxes, and LaTeX errors. Fonts are embedded/subsetted, every page is
nonblank, and independent deterministic ZIP/manifest replay passed.

MAVIS, MVD, and MVA scientific suites: **Not rerun; no scientific/shared runtime
code changed in this manuscript-only cleanup.**

## Generated Artifacts

| Artifact | SHA-256 |
| --- | --- |
| `results/aei_information_hierarchy/submission/AEI_PAPER1_MANUSCRIPT.pdf` | `e457bcae1a8ab346edab3005b0267e2553755b8530ec4f97777263ae2f4f008f` |
| `results/aei_information_hierarchy/submission/AEI_PAPER1_SUPPLEMENTARY.pdf` | `167b4bb3be001c7750397b0f9fec6d68f617b7310f6665d13d72c26cd4db8a53` |
| `results/aei_information_hierarchy/deterministic_package/AEI_PAPER_SUBMISSION_SOURCE.zip` | `8bd5e9c956037a2867e9fcbfe94db6be3d560e39fb21ae8946f1611591a2b900` |
| `results/aei_information_hierarchy/deterministic_package/submission_source/SUBMISSION_MANIFEST.csv` | `5fa7602fcb072a4d58ce6a02bbc55c6ab291b7c953d57797ffea0113d90a60b8` |
| `artifacts/aei_information_hierarchy/MANUSCRIPT_AUDIT_RESIDUE_CLEANUP_COMPLETION.md` | `11305a4905dadee2923aaac479b3e8a7103deb4b590b08320470ebebe71fcb78` |

## GitHub Verification

Push command: `git push origin aei-main-method-reframe`
Remote branch: `refs/heads/aei-main-method-reframe`
Local payload SHA: `9d7effaf18ecb5aa62f84b8380d8fbf701b9ef6b`
Remote payload SHA: `9d7effaf18ecb5aa62f84b8380d8fbf701b9ef6b`
Remote verification: passed; eight required remote paths were readable, the
main-text residue scan passed, and remote PDF/ZIP/canonical hashes matched.
PR: not created; no merge to `main`; no force push.

## Remaining Factual Issues

Author metadata and declarations, archive identifier/license, and live AEI
upload-form requirements remain author-supplied or portal-dependent. The study
does not establish external empirical transfer, physical scanner-time savings,
or learned end-to-end superiority over the static reference.
