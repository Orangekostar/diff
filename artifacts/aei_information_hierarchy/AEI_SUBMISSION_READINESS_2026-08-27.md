# AEI Submission Readiness Check

Check date: 2026-08-27
Target: Advanced Engineering Informatics, original research article
Mode: full source/package check

## Verdict

The manuscript, supplement, and flat LaTeX source package are technically ready
for editorial upload. Final submission remains conditional on author metadata,
the repository/archive identifier, and confirmation of journal-specific fields
and forms in Editorial Manager.

## Technical Checks

| Check | Status | Evidence |
| --- | --- | --- |
| Journal class | pass | `elsarticle` 3.4, 12 pt preprint mode |
| Main compilation | pass | `latexmk` exit 0; 28 pages; no final warnings |
| Supplement compilation | pass | `latexmk` exit 0; 2 pages; no final warnings |
| Flat source compilation | pass | deterministic `submission_source/main.tex` compiles to 28 pages |
| References and labels | pass | no undefined citation/reference in final logs |
| Fonts | pass | every main and supplement font is embedded |
| Figure/table package | pass | four PDF figures and three generated LaTeX tables present |
| Editorial Manager layout | pass | deterministic ZIP is flat; no source subdirectories |
| Anonymity | pass for review | author and affiliation are withheld; repository identity deferred |
| Data/code statement | pass with finalization | public datasets stated; archive identifier deferred until review policy permits |
| AI disclosure | pass | Codex use, author validation, and author responsibility are declared |
| Source reproducibility | pass | two package builds are byte-identical |

## Official-Policy Freshness

The current Elsevier journal page was checked on 2026-08-27. It describes AEI as
supporting knowledge-intensive engineering activities and expects engineering
relevance plus vigorous qualitative and quantitative validation. The manuscript
addresses an artifacts-centered mechanical-engineering decision and states its
causal information contract, held-out-domain validation, and deployment limits.

Elsevier's current LaTeX instructions require a flat source layout for Editorial
Manager and allow a manuscript PDF plus a single source archive; the generated
package follows that layout. The journal-specific ScienceDirect Guide for
Authors returned HTTP 403 during this check, so a journal-specific word/page
limit and required upload forms could not be independently verified.

Official sources:

- https://shop.elsevier.com/journals/advanced-engineering-informatics/1474-0346
- https://www.elsevier.com/en-gb/researcher/author/policies-and-guidelines/latex-instructions
- https://www.elsevier.com/publishing/publish-in-a-journal/manuscript-preparation

## Remaining Submission Actions

1. Restore author names, affiliations, corresponding-author details, funding,
   conflict-of-interest, and contributor-role metadata in the submission system.
2. Add the public code repository and archival identifier when compatible with
   the selected review mode.
3. Confirm the current AEI Guide for Authors and Editorial Manager checklist at
   upload time, especially article type, declarations, highlights, and any
   journal-specific length requirement.
