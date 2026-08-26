# P0 Manuscript Source Audit

Audit date: 2026-08-26

## Tracked source status

No tracked `*.tex` or `*.bib` file exists at the audit base `c2eab6e`. The
tracked manuscript-like evidence consists of:

- `artifacts/mavis/MANUSCRIPT_EVIDENCE_MAP.md`
- `artifacts/mavis_science_closure/MANUSCRIPT_CLAIM_MAP.md`
- stage reports under `results/mavis/`, `results/mvd/`, `results/mva/`, and
  `results/mavis_science_closure/`
- paper experiment configurations and P1 frozen predictions under `paper_v3/`
- project protocols and result reports under `docs/`

These files are evidence and protocol sources. None is a complete manuscript.

## New paper namespace

The manuscript will be created under `paper_aei_information_hierarchy/` with:

- `README.md`
- `MANUSCRIPT_OUTLINE.md`
- `CLAIM_SENTENCE_BANK.md`
- `main.tex`
- `references.bib`
- `figures/`
- `tables/`
- `supplementary/`

An AEI-compatible generic article source will be used unless a licensed,
tracked journal template becomes available during the convention audit. No
template provenance will be invented.

## Fixed top-level structure

The manuscript must contain exactly these six numbered top-level sections:

1. Introduction
2. Related Research and Problem Formulation
3. Task-Relevant Information Hierarchy and Operational Framework
4. Multi-Domain CFRP Case Study and Experimental Design
5. Experimental Results and Discussion
6. Conclusions

The title is fixed as:

> From Useful to Actionable Information: A Task-Relevant Information Hierarchy
> for Ultrasonic Inspection of Impacted Composites

## Drafting gate

Abstract and Results drafting are blocked until the paper-specific canonical
metric table, claim map, and source-hash ledger are complete. Internal stage
labels such as M0/M1 decisions, Tier labels, and P-stage pass/fail language may
not appear in manuscript prose.
