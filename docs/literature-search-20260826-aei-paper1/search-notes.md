# Search Notes

## Sources and date

Search performed 2026-08-26 against the official Elsevier AEI scope page and
primary ScienceDirect article pages. Ten recent AEI research articles were
retained. Publisher pages were used for titles, venue/year metadata, abstracts,
and exposed section snippets.

## Query families

- official Advanced Engineering Informatics aims and scope knowledge-intensive
- 2026 Advanced Engineering Informatics full length article
- exact-title searches for each retained paper
- exact-title plus `This paper is organized`, `Introduction`, and `Conclusion`
- C-scan CAI Advanced Engineering Informatics 2026

## Access limitation

Some ScienceDirect full-text requests returned access restrictions while their
publisher-indexed abstract and section snippets remained visible. For those
papers, only publisher-visible headings were recorded. Complete structure was
marked `STRUCTURE_NOT_VERIFIED` when no top-level headings were exposed.

## Exclusions

- editorials and issue introductions were not counted among the ten research
  papers;
- non-AEI papers were not used for the journal structure benchmark;
- secondary aggregators and non-primary summaries were not used as evidence;
- section headings were not inferred from abstract content.

## Evidence quality

Scores in `papers.csv` are triage scores from 1 (weak) to 5 (strong) for
conceptual insight, accessible completeness, and numerical evidence. They are
not paper-quality or acceptance scores.
