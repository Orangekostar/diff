# AEI Task-Relevant Information Acquisition Paper Package

This directory contains the paper-only package for the first Advanced
Engineering Informatics manuscript derived from the frozen evidence at base
commit `9794d53a9549f2e3501fe482e8db8735f468ba20`.

Working title:

> Task-Relevant Ultrasonic Information Acquisition for Impacted Composites:
> From Spatial Information to State-Conditioned Sensing

The manuscript follows a positive progressive argument. Part I, Task-Relevant
Information Characterization, characterizes structured task-relevant
information. Part II, State-Conditioned Task-Oriented Acquisition, tests its
realization in sensing decisions. It introduces no new model training and does
not claim a performance-superior adaptive scanner. Usefulness, task-value
observability, and actionability remain validation criteria within the two
parts.

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
3. Task-Relevant Information Acquisition Framework
4. Multi-Domain CFRP Case Study and Experimental Design
5. Experimental Results and Discussion
6. Conclusions

The four main figures and one main table are generated from machine-readable
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
figures, one main LaTeX table, and the machine-readable supplement. Organized,
unscaled component PNGs and their SHA-256 manifest are written to `panel_pngs/`
for manual composition; they are not part of the submission package. The
deterministic flat Editorial Manager source bundle is generated under
`../results/aei_information_hierarchy/deterministic_package/` by
`cmc_bbdm.mavis.aei_paper_package`.

## Submission Boundary

The manuscript and supplement are identity-neutral. Author names,
corresponding-author details, competing-interest declarations, CRediT roles,
the final code license, and the public archival identifier require author
completion before external submission; they are not inferred in this package.
