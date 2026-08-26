# Supplementary Evidence Package

This directory contains audit-heavy evidence for the manuscript. Central
negative controls remain in the main text; these files provide the complete
checkpoint, domain, learner-pair, component-substitution, planning, feedback,
provenance, and external-data records behind those conclusions.

`supplementary.tex` is the reader-facing supplement. The `data/` directory is
materialized deterministically from frozen repository artifacts by
`cmc_bbdm.mavis.aei_paper_package.materialize_paper_assets`.

The machine-readable files cover:

- `S01`--`S02`: full-field and static-observability domain metrics;
- `S03`--`S04`: complete conditional-value checkpoint and domain metrics;
- `S05`--`S07`: partial-state control matrix, contrasts, and domain metrics;
- `S08`--`S09`: secondary dynamic-valuation metrics, including rank metrics;
- `S10`--`S11`: component substitutions and domain-level effects;
- `S12`: bounded set-planning domain results;
- `S13`--`S14`: learner metrics and all learner-pair value comparisons;
- `S15`--`S16`: feedback strata and association diagnostics;
- `S17`: paper evidence provenance hashes;
- `S18`: external-dataset feasibility audit without performance claims.

`data/SUPPLEMENTARY_DATA_MANIFEST.csv` binds every copied file to its source
artifact, SHA-256 digest, and byte count.
