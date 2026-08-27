# AEI Information Hierarchy Paper Package

This directory contains the paper-only package for the first Advanced
Engineering Informatics manuscript derived from the frozen evidence at base
commit `c2eab6eac79dd3fbb9ecb0d19f98923e515e762b`.

Working title:

> From Useful to Actionable Information: A Task-Relevant Information
> Hierarchy for Ultrasonic Inspection of Impacted Composites

The manuscript is a mechanism-and-boundary engineering-informatics paper. It
does not claim a superior adaptive scanner or introduce new model training.
Its fixed argument is that task usefulness, conditional observability, and
decision actionability are distinct propositions that require separate
validation.

## Authority

- Numeric authority: `../artifacts/aei_information_hierarchy/PAPER_CANONICAL_METRICS.csv`
- Claim authority: `../artifacts/aei_information_hierarchy/PAPER_CLAIM_MAP.md`
- Source hashes: `../artifacts/aei_information_hierarchy/PAPER_SOURCE_HASHES.csv`
- Main figure sources: `../results/aei_information_hierarchy/figures/`
- Main table sources: `../results/aei_information_hierarchy/tables/`

Historical result and artifact directories are read-only inputs. The paper
package must not rewrite them.

## Fixed Manuscript Contract

The paper has exactly six top-level sections:

1. Introduction
2. Related Research and Problem Formulation
3. Task-Relevant Information Hierarchy and Operational Framework
4. Multi-Domain CFRP Case Study and Experimental Design
5. Experimental Results and Discussion
6. Conclusions

The four main figures and three main tables are generated from machine-readable
paper evidence. Retrospective teachers, oracles, and component substitutions
are always identified as non-deployable. Acquisition cost is the exact fraction
of unique native-raster locations revealed, not scanner time.

## Build

Build the working manuscript from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Build the reader-facing supplement from `supplementary/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

The working tree includes the Elsevier class/style, four manuscript-width PDF
figures, three generated LaTeX tables, and the machine-readable supplement. The
deterministic flat Editorial Manager source bundle is generated under
`../results/aei_information_hierarchy/deterministic_package/` by
`cmc_bbdm.mavis.aei_paper_package`.

## Submission Boundary

The manuscript and supplement are identity-neutral. Author names,
corresponding-author details, competing-interest declarations, CRediT roles,
the final code license, and the public archival identifier require author
completion before external submission; they are not inferred in this package.
