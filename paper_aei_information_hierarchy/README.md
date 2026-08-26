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

The four main figures and two main tables are generated from machine-readable
paper evidence. Retrospective teachers, oracles, and component substitutions
are always identified as non-deployable. Acquisition cost is the exact fraction
of unique native-raster locations revealed, not scanner time.

## Package Status

`MANUSCRIPT_OUTLINE.md` and `CLAIM_SENTENCE_BANK.md` freeze the paper structure
and evidence-bounded wording before drafting. The LaTeX manuscript,
bibliography, copied figure/table deliverables, supplementary material, and
build instructions are added in later evidence-preserving commits.
