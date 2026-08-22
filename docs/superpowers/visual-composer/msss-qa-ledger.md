# MSSS Figure QA Ledger

| Issue | Artifact | Page/section | Severity | Fix | Owner | Status |
|---|---|---|---|---|---|---|
| Formal render inspection | Figures A--D | S1 results | Medium | Inspected all four PNGs at rendered resolution; curves, intervals, markers, legends, and labels are visible without overlap | primary agent | RESOLVED |
| Vector font embedding | Figures A--D | S1 results | Low | `pdffonts` confirms embedded CID TrueType DejaVu Sans fonts | primary agent | RESOLVED |
| Source/output binding | Figures A--D | S1 results | High | `figure_manifest.json`, package checksums, and byte-identical replay bind every figure to the published CSV sources | primary agent | RESOLVED |
