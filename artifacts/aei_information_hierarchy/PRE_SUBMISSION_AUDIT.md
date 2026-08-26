# AEI Paper 1 Pre-Submission Audit

Date: 2026-08-26
Target journal: *Advanced Engineering Informatics* (AEI)
Status: **conditional pass**

The scientific manuscript, evidence package, supplementary material, and
deterministic source archive are technically complete. Submission remains
conditional on author-supplied identity, disclosure, contribution, licensing,
and archival metadata listed below.

## Submission-policy check

- The manuscript uses Elsevier's supported `elsarticle` class and includes the
  bibliography style, figures, tables, and build instructions required for a
  source submission.
- The Editorial Manager source archive is flat: no source dependency is stored
  in a subdirectory. This follows Elsevier's LaTeX submission instructions:
  <https://www.elsevier.com/en-gb/researcher/author/policies-and-guidelines/latex-instructions>.
- A single review PDF and a separate supplementary PDF are provided, consistent
  with Elsevier's Your Paper Your Way guidance:
  <https://www.elsevier.com/en-in/subject/next/guide-for-authors>.
- The package is identity-neutral. AEI's current journal-specific review model
  could not be conclusively retrieved from the ScienceDirect guide during this
  audit; the corresponding author must confirm the live Editorial Manager
  requirements before upload.

## Scientific audit

| Criterion | Result | Evidence |
|---|---|---|
| Originality | Pass | The paper contributes a task-relevant hierarchy separating useful, observable, and actionable information and tests each transition with matched adverse controls. |
| Importance | Pass | The hierarchy directly addresses when richer ultrasonic information changes cross-domain mechanical inference or acquisition decisions. |
| Soundness | Pass | Claims are limited to the registered six-domain CFRP program; uncertainty intervals, domain signs, matched comparisons, and negative controls are reported. |
| Engineering relevance | Pass | The study links inspection representation, sparse acquisition, observability, recovery, supervision, and planning to explicit engineering tasks. |
| Readability | Pass | The paper has exactly six numbered sections, four main figures, two main tables, and a visible evidence chain from question to implication. |

The independent review report records an overall simulated score of 7/10 with
5/5 confidence. Its principal scientific limitation is external empirical
breadth, not a fatal soundness or evidence-integrity defect.

## Evidence and integrity check

- All 39 canonical claim IDs are mapped into the manuscript.
- Headline P1 uses the registered matched `B_scalar -> B_field_selected`
  comparison; `I_field_selected` is labeled as sensitivity evidence only.
- Sparse-retention, observability, recovery, supervision, and planning claims
  retain their registered estimands and uncertainty language.
- Adverse controls remain in the main paper rather than being moved solely to
  supplementary material.
- Results-section numeric provenance lint passes against the canonical claim
  authority.
- No tracked file under the historical MAVIS/MVD result and audit paths differs
  from base commit `c2eab6eac79dd3fbb9ecb0d19f98923e515e762b`.
- The frozen P7 Git tree object is unchanged (`b7fb24ff2d808db6fd8ec4f6571daef55016b96c`),
  and its registered replay-tree SHA-256
  `931dc86c26caf1c7246709c4706a7cd0428e3a1533b6ff1ad3c2ad8f9517d1e4`
  passes in the MAVIS regression suite.

## Build and validation check

| Check | Result |
|---|---|
| Paper-specific tests | 70 passed |
| Historical MAVIS suite | 225 passed |
| Historical MVD suite | 29 passed |
| Historical MVA suite in complete authority repository | 126 passed |
| Repository Ruff check | Pass |
| `git diff --check` | Pass |
| Main manuscript isolated LaTeX/BibTeX build | Pass; 27 pages; no real warnings or overfull boxes |
| Supplement isolated LaTeX build | Pass; 2 pages; no real warnings or overfull boxes |
| Flat Editorial Manager source build | Pass; no real warnings or overfull boxes |
| PDF font embedding | Pass; all listed fonts embedded and subsetted |
| PDF identity metadata | Pass; review-safe placeholder only |

## Delivered package

- Review manuscript:
  `results/aei_information_hierarchy/submission/AEI_PAPER1_MANUSCRIPT.pdf`
  (`dfe6b69e3c6bacef400ebf35f5a4df50792f49a429b479485925eb8438b1565b`).
- Supplement:
  `results/aei_information_hierarchy/submission/AEI_PAPER1_SUPPLEMENTARY.pdf`
  (`555264e7c158c8ee0603d05ddcdc6f83bd0a0f9aee643126f61e3f1f92f7af12`).
- Deterministic flat source archive:
  `results/aei_information_hierarchy/deterministic_package/AEI_PAPER_SUBMISSION_SOURCE.zip`
  (`034fc19196b3251dd708f2e0c824e08832815632c849033421ec7463b7ec9680`).
- The source manifest contains 11 payload files; the supplementary-data
  manifest contains 18 machine-readable evidence files.

## Author-completion boundary

Before submission, the authors must provide or confirm:

1. author names, affiliations, corresponding-author address, and contact data;
2. competing-interest declaration;
3. CRediT contribution statement;
4. final code/data/model release license;
5. archival repository URL and DOI, when available;
6. AEI's live review-model, file-type, and declaration requirements in the
   submission portal.

These are author-governance items. No scientific result, frozen artifact,
headline number, or evidentiary conclusion remains unresolved in the delivered
technical package.
